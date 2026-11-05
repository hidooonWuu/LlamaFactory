# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from copy import deepcopy
from types import ModuleType

import pytest
import torch
from einops import rearrange
from transformers import AutoModelForCausalLM, PretrainedConfig

from llamafactory.v1.plugins.model_plugins import bailing_hybrid


class LingConfig(PretrainedConfig):
    model_type = "bailing_hybrid"


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(bailing_hybrid, "is_transformers_version_greater_than", lambda _: True)
    config = LingConfig()
    config.rope_theta = 6000000
    config.qk_rope_head_dim = 64
    config.max_position_embeddings = 131072
    config.rope_scaling = {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5}
    return config


def test_config_preserves_checkpoint_theta(config):
    bailing_hybrid.patch_bailing_hybrid_config(config)
    assert config.rope_scaling["rope_theta"] == 6000000
    assert config.rope_scaling["partial_rotary_factor"] == 0.5
    assert "factor" not in config.rope_scaling


def test_rope_and_mla_preserve_original_config(config):
    class Rotary(torch.nn.Module):
        pass

    class Attention(torch.nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.config = config
            self.scaling = config.rope_scaling["factor"]

    module = ModuleType("ling_compat_test")
    module.BailingMoeV3RotaryEmbedding = Rotary
    module.BailingMoeV3MultiLatentAttention = Attention
    module.ROPE_INIT_FUNCTIONS = {}
    bailing_hybrid.patch_bailing_hybrid_config(config)
    original = deepcopy(config.to_dict())
    bailing_hybrid._patch_model_module(module)
    patched_init = Attention.__init__
    bailing_hybrid._patch_model_module(module)
    assert Attention.__init__ is patched_init
    rotary = Rotary(config)
    expected = 1.0 / (6000000 ** (torch.arange(0, 64, 2).float() / 64))
    torch.testing.assert_close(rotary.inv_freq, expected)
    assert rotary.config.rope_parameters["partial_rotary_factor"] == 1.0
    assert Attention(config, 3).scaling == 1.0
    assert config.to_dict() == original

    # Scaled RoPE keeps its actual factor and uses the existing initializer.
    config.rope_scaling = {"rope_type": "yarn", "factor": 8.0, "rope_theta": 6000000}
    assert Attention(config, 3).scaling == 8.0
    seen = []

    def yarn_init(config, device):
        seen.append(config.rope_parameters["partial_rotary_factor"])
        return torch.ones(32), 2.0

    module.ROPE_INIT_FUNCTIONS["yarn"] = yarn_init
    assert Rotary(config).attention_scaling == 2.0
    assert seen == [1.0]


def test_import_shim_restored_after_import_failure(config, monkeypatch):
    config.auto_map = {"AutoModelForCausalLM": "modeling_ling.Model"}
    monkeypatch.delattr(bailing_hybrid.import_utils, "is_torch_fx_available", raising=False)

    def fail(*args):
        assert bailing_hybrid.import_utils.is_torch_fx_available is bailing_hybrid.import_utils.is_torch_available
        raise ImportError("missing fla")

    monkeypatch.setattr(bailing_hybrid, "get_class_from_dynamic_module", fail)
    with pytest.raises(ImportError, match="missing fla"):
        bailing_hybrid.prepare_bailing_hybrid_model(config, "unused", AutoModelForCausalLM, True)
    assert not hasattr(bailing_hybrid.import_utils, "is_torch_fx_available")


@pytest.mark.parametrize(
    "trusted,model_type,tf5",
    [(False, "bailing_hybrid", True), (True, "qwen3", True), (True, "bailing_hybrid", False)],
)
def test_untrusted_other_models_and_transformers4_are_untouched(config, monkeypatch, trusted, model_type, tf5):
    config.model_type = model_type
    monkeypatch.setattr(bailing_hybrid, "is_transformers_version_greater_than", lambda _: tf5)

    def unexpected_import(*args):
        pytest.fail("must not import remote code")

    monkeypatch.setattr(bailing_hybrid, "get_class_from_dynamic_module", unexpected_import)
    before = deepcopy(config.to_dict())
    bailing_hybrid.prepare_bailing_hybrid_model(config, "unused", AutoModelForCausalLM, trusted)
    if model_type != "bailing_hybrid" or not tf5:
        bailing_hybrid.patch_bailing_hybrid_config(config)
    assert config.to_dict() == before


def _kda_module(no_kda_lora):
    class Conv(torch.nn.Module):
        def forward(self, x, **kwargs):
            return x, None

    class Norm(torch.nn.Module):
        def forward(self, x, gate):
            return x * gate.sigmoid()

    class KDA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.no_kda_lora = no_kda_lora
            self.head_dim = self.head_k_dim = 4
            self.safe_gate = True
            self.lower_bound = -5
            self.A_log = torch.nn.Parameter(torch.zeros(2))
            self.dt_bias = torch.nn.Parameter(torch.zeros(8))
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "f_proj", "g_proj"):
                setattr(self, name, torch.nn.Linear(8, 8, bias=False))
            for name in ("f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"):
                setattr(self, name, torch.nn.Linear(8, 8, bias=False))
            self.b_proj = torch.nn.Linear(8, 2, bias=False)
            self.q_conv1d, self.k_conv1d, self.v_conv1d = Conv(), Conv(), Conv()
            self.o_norm = Norm()

        def forward(self, hidden_states, **kwargs):
            if self.training and hidden_states.shape[1] <= 64:
                raise AssertionError("Only chunk mode is supported in training.")
            return hidden_states, "original", kwargs.get("past_key_value")

    module = ModuleType("ling_kda_test")
    module.BailingMoeV3KimiDeltaAttention = KDA
    module.rearrange = rearrange
    return module


@pytest.mark.parametrize("length", [1, 64])
@pytest.mark.parametrize("no_kda_lora", [True, False])
def test_short_kda_training_uses_chunk_and_backpropagates(length, no_kda_lora):
    module = _kda_module(no_kda_lora)
    attention = module.BailingMoeV3KimiDeltaAttention()
    hidden = torch.randn(1, length, 8, requires_grad=True)
    with pytest.raises(AssertionError, match="Only chunk"):
        attention(hidden)
    calls = []
    cu_seqlens = torch.tensor([0, length])

    def chunk(**kwargs):
        calls.append(kwargs)
        return (kwargs["q"] + kwargs["k"] + kwargs["v"] + kwargs["g"]) * kwargs["beta"][..., None], None

    module.chunk_kda = chunk
    bailing_hybrid._patch_kda_training(module)
    output, weights, cache = attention(hidden, cu_seqlens=cu_seqlens)
    assert output.shape == hidden.shape
    assert weights is cache is None
    assert calls[0]["cu_seqlens"] is cu_seqlens
    assert calls[0]["initial_state"] is None
    output.square().mean().backward()
    assert torch.isfinite(hidden.grad).all()
    assert attention.q_proj.weight.grad is not None
    gate_proj = attention.g_proj if no_kda_lora else attention.g_b_proj
    assert gate_proj.weight.grad is not None


@pytest.mark.parametrize("training,length", [(False, 1), (True, 65)])
def test_kda_inference_and_long_inputs_keep_original_path(training, length):
    module = _kda_module(True)
    bailing_hybrid._patch_kda_training(module)
    attention = module.BailingMoeV3KimiDeltaAttention().train(training)
    hidden, cache = torch.randn(1, length, 8), object()
    output = attention(hidden, past_key_value=cache)
    assert output[0] is hidden
    assert output[1:] == ("original", cache)
