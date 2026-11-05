# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llamafactory.v1.core.model_engine import FORCED_ATTN_IMPLEMENTATIONS, ModelEngine
from llamafactory.v1.plugins.trainer_plugins.distributed.expert_adapters import bailing_hybrid
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import FSDPTurboEPModelSpec
from llamafactory.v1.utils.types import AttentionFunction


HIDDEN_SIZE = 16
INTERMEDIATE_SIZE = 32
NUM_EXPERTS = 4
TOP_K = 2


class _Mlp(torch.nn.Module):
    """Minimal copy of ``BailingMoeV3MLP``."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, intermediate_size: int = INTERMEDIATE_SIZE) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _Gate(torch.nn.Module):
    """Minimal copy of ``BailingMoeV3Gate``."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_experts: int = NUM_EXPERTS, top_k: int = TOP_K) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = torch.nn.Parameter(torch.randn(num_experts, hidden_size) * 0.1)

    def forward(self, hidden_states: torch.Tensor):
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(flat.float(), self.weight.float())
        scores = torch.sigmoid(logits)
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight, logits


class _SparseMoeBlock(torch.nn.Module):
    """Minimal copy of ``BailingMoeV3SparseMoeBlock`` (list-backed experts and shared experts)."""

    def __init__(self) -> None:
        super().__init__()
        self.num_experts_per_tok = TOP_K
        self.experts = torch.nn.ModuleList([_Mlp() for _ in range(NUM_EXPERTS)])
        self.gate = _Gate()
        self.shared_experts = _Mlp()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        identity = hidden_states
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        flat = hidden_states.view(-1, hidden_size)
        flat_topk_idx = topk_idx.view(-1)
        repeated = flat.repeat_interleave(self.num_experts_per_tok, dim=0)
        routed = torch.zeros_like(repeated)
        for expert_id, expert in enumerate(self.experts):
            selected = flat_topk_idx == expert_id
            if selected.any():
                routed[selected] = expert(repeated[selected])

        routed = (routed.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        routed = routed.to(hidden_states.dtype).view(batch_size, seq_len, hidden_size)
        routed = routed + self.shared_experts(identity)
        return routed, (router_logits.view(batch_size, seq_len, -1), topk_idx.view(batch_size, seq_len, -1))


class _DecoderLayer(torch.nn.Module):
    """Minimal copy of ``BailingMoeV3DecoderLayer`` keeping the ``layers.N.mlp`` module path."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = _SparseMoeBlock()


class _BailingModel(torch.nn.Module):
    """Minimal copy of the ``BailingMoeV3ForCausalLM`` module tree used by FSDPTurbo EP."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="bailing_hybrid")
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_DecoderLayer() for _ in range(num_layers)])


def _make_model() -> _BailingModel:
    torch.manual_seed(0)
    return _BailingModel()


def test_bailing_hybrid_ep_model_spec():
    spec = FSDPTurboEPModelSpec.get(_make_model())

    assert spec is not None
    assert spec.ep_modules == ["model.layers.{*}.mlp.experts"]
    assert spec.ep_fsdp_modules == ["model.layers.{*}.mlp"]


def test_bailing_hybrid_fuses_routed_experts():
    model = _make_model()
    hidden_states = torch.randn(2, 5, HIDDEN_SIZE)
    expected, (expected_logits, expected_idx) = model.model.layers[0].mlp(hidden_states)

    prepared = bailing_hybrid.prepare_for_ep(model)

    fused = prepared.model.layers[0].mlp.experts
    assert isinstance(fused, bailing_hybrid.FusedRoutedExperts)
    assert fused.gate_up_proj.shape == (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert fused.down_proj.shape == (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    assert (fused.hidden_dim, fused.num_experts) == (HIDDEN_SIZE, NUM_EXPERTS)

    actual, (actual_logits, actual_idx) = prepared.model.layers[0].mlp(hidden_states)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_idx, expected_idx)


def test_bailing_hybrid_fused_weights_keep_the_linear_layout():
    model = _make_model()
    experts = model.model.layers[0].mlp.experts
    expected_gate = torch.stack([expert.gate_proj.weight for expert in experts])
    expected_up = torch.stack([expert.up_proj.weight for expert in experts])
    expected_down = torch.stack([expert.down_proj.weight for expert in experts])

    bailing_hybrid.prepare_for_ep(model)
    fused = model.model.layers[0].mlp.experts

    torch.testing.assert_close(fused.gate_up_proj[:, :INTERMEDIATE_SIZE], expected_gate)
    torch.testing.assert_close(fused.gate_up_proj[:, INTERMEDIATE_SIZE:], expected_up)
    torch.testing.assert_close(fused.down_proj, expected_down)


def test_bailing_hybrid_fuses_only_once():
    model = _make_model()
    bailing_hybrid.prepare_for_ep(model)
    fused = model.model.layers[0].mlp.experts

    bailing_hybrid.prepare_for_ep(model)

    assert model.model.layers[0].mlp.experts is fused


def test_bailing_hybrid_rejects_unrecognized_expert_lora_layout():
    model = _make_model()
    model.model.layers[0].mlp.experts[0].lora_A = torch.nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="LoRA adapters were found"):
        bailing_hybrid.prepare_for_ep(model)


def _add_expert_lora(model):
    from peft.tuners.lora.layer import Linear

    for param in model.parameters():
        param.requires_grad_(False)
    for layer in model.model.layers:
        for expert in layer.mlp.experts:
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = Linear(getattr(expert, name), "default", r=2, lora_alpha=4, lora_dropout=0.0)
                with torch.no_grad():
                    projection.lora_B["default"].weight.normal_(std=0.02)
                setattr(expert, name, projection)
    return model


def test_expert_lora_fusion_preserves_outputs_gradients_and_checkpoint_keys():
    from llamafactory.v1.plugins.trainer_plugins.distributed.expert_adapters.bailing_lora import (
        unpack_expert_lora_state_dict,
    )

    reference = _add_expert_lora(_make_model())
    model = deepcopy(reference)
    x = torch.randn(2, 3, HIDDEN_SIZE, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    bailing_hybrid.prepare_for_ep(model)
    expected = reference.model.layers[0].mlp(x)[0]
    actual = model.model.layers[0].mlp(y)[0]
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(x.grad, y.grad)
    original = dict(reference.named_parameters())
    unpacked = unpack_expert_lora_state_dict(model, dict(model.state_dict()))
    for name, param in original.items():
        if "lora_" in name:
            torch.testing.assert_close(unpacked[name], param)
    fused = model.model.layers[0].mlp.experts
    for projection, spec in fused._expert_lora_specs.items():
        for factor in ("A", "B"):
            packed_grad = getattr(fused, spec[factor]).grad
            for expert in range(NUM_EXPERTS):
                key = f"model.layers.0.mlp.experts.{expert}.{projection}.lora_{factor}.default.weight"
                expected_grad = original[key].grad
                if expected_grad is not None:
                    torch.testing.assert_close(packed_grad[expert], expected_grad)


def test_bailing_hybrid_rejects_meta_init():
    model = _make_model()
    model._init_mode = "init_on_meta"

    with pytest.raises(RuntimeError, match="init_mode=init_on_meta"):
        bailing_hybrid.prepare_for_ep(model)


def test_bailing_hybrid_allows_patched_flash_attention():
    assert FORCED_ATTN_IMPLEMENTATIONS["bailing_hybrid"] == AttentionFunction.SDPA

    engine = object.__new__(ModelEngine)
    engine.model_config = SimpleNamespace(model_type="bailing_hybrid")
    engine.args = SimpleNamespace(flash_attn=AttentionFunction.FLASH_ATTENTION_2)
    assert engine._resolve_attn_implementation() == "flash_attention_2"

    engine.args = SimpleNamespace(flash_attn=AttentionFunction.SDPA)
    assert engine._resolve_attn_implementation() == "sdpa"


def test_other_models_keep_the_requested_attention():
    engine = object.__new__(ModelEngine)
    engine.model_config = SimpleNamespace(model_type="qwen3_moe")
    engine.args = SimpleNamespace(flash_attn=AttentionFunction.FLASH_ATTENTION_2)

    assert engine._resolve_attn_implementation() == "flash_attention_2"


def test_grouped_lora_dropout_keeps_fp32_mask_rounding():
    from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import _FP32LoraDropout

    x = torch.linspace(-2, 2, 128, dtype=torch.bfloat16, requires_grad=True)
    dropout = _FP32LoraDropout(0.05)
    torch.manual_seed(17)
    expected = F.dropout(x.float(), p=0.05, training=True).to(x.dtype)
    torch.manual_seed(17)
    actual = dropout(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert torch.isfinite(x.grad).all()
    dropout.eval()
    torch.testing.assert_close(dropout(x), x, rtol=0, atol=0)


@pytest.mark.parametrize("mismatch", ["rank", "dropout", "scaling", "dora", "active_adapter"])
def test_expert_lora_rejects_inconsistent_or_unsupported_layouts(mismatch):
    model = _add_expert_lora(_make_model())
    layer = model.model.layers[0].mlp.experts[1].gate_proj
    if mismatch == "rank":
        layer.lora_A["default"] = torch.nn.Linear(HIDDEN_SIZE, 3, bias=False)
    elif mismatch == "dropout":
        layer.lora_dropout["default"] = torch.nn.Dropout(0.1)
    elif mismatch == "scaling":
        layer.scaling["default"] *= 2
    elif mismatch == "dora":
        layer.use_dora["default"] = True
    else:
        layer.set_adapter(["default", "other"])
    with pytest.raises(ValueError, match="Expert LoRA|Ling EP"):
        bailing_hybrid.prepare_for_ep(model)


def test_ling_flash_attention_rejects_unsupported_kda_context_parallelism():
    engine = object.__new__(ModelEngine)
    engine.model_config = SimpleNamespace(model_type="bailing_hybrid")
    engine.args = SimpleNamespace(flash_attn=AttentionFunction.FLASH_ATTENTION_2, cp_size=2)
    with pytest.raises(ValueError, match="cp_size=1"):
        engine._resolve_attn_implementation()
