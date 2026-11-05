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

"""Expert-parallel preparation for BailingMoeV3 (``model_type: bailing_hybrid``).

The remote-code Bailing MoE keeps the routed experts in an ``nn.ModuleList`` and executes them
with a Python loop over experts. FSDPTurbo's EP dispatcher instead expects one fused expert
block that exposes ``gate_up_proj`` / ``down_proj`` / ``hidden_dim`` and the
``forward(hidden_states, top_k_index, top_k_weights)`` entry point, so this module rewrites the
routed experts into that fused layout before ``expert_parallelize_modules`` runs.

The fused weights keep the linear-layer convention used by FSDPTurbo's grouped matmul:
``gate_up_proj`` is ``[num_experts, 2 * moe_intermediate_size, hidden_size]`` with the gate half
first, and ``down_proj`` is ``[num_experts, hidden_size, moe_intermediate_size]``.
"""

from collections.abc import Iterator

import torch
import torch.nn as nn

from .....utils.logging import get_logger
from .....utils.types import HFModel
from .bailing_lora import expert_mlp, pack_expert_lora


logger = get_logger(__name__)


#: Module names of the routed experts, resolved by ``fsdp_turbo.utils.str_match.module_name_match``.
EP_MODULES = ["model.layers.{*}.mlp.experts"]
#: Expert fully-sharded (EFSDP) targets, i.e. the sparse MoE blocks owning the routed experts.
EP_FSDP_MODULES = ["model.layers.{*}.mlp"]

_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SUPPORTED_INIT_MODE = "init_on_default"


class FusedRoutedExperts(nn.Module):
    """Routed experts stored in the fused layout expected by the FSDPTurbo EP dispatcher.

    ``forward`` accepts the same ``(hidden_states, top_k_index, top_k_weights)`` signature that
    ``expert_parallelize_modules`` installs, so the module also runs when it is fused without an EP
    dispatcher, for example in unit tests.
    """

    def __init__(
        self,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        act_fn: nn.Module,
        num_experts: int,
        hidden_dim: int,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.act_fn = act_fn
        self.gate_up_proj = nn.Parameter(gate_up_proj, requires_grad=requires_grad)
        self.down_proj = nn.Parameter(down_proj, requires_grad=requires_grad)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Reference (non-EP) forward that assembles the weighted expert outputs."""
        hidden_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, self.hidden_dim)
        flat_weights = top_k_weights.to(hidden_states.dtype)
        output = torch.zeros_like(flat_hidden)
        for expert_id in range(self.num_experts):
            token_ids, slot_ids = torch.where(top_k_index == expert_id)
            if token_ids.numel() == 0:
                continue

            expert_output = expert_mlp(self, expert_id, flat_hidden[token_ids])
            output.index_add_(0, token_ids, expert_output * flat_weights[token_ids, slot_ids].unsqueeze(-1))

        return output.view(hidden_shape)


def _is_sparse_moe_block(module: nn.Module) -> bool:
    """Return whether the module is a list-backed sparse MoE block with a router and a shared MLP."""
    experts = getattr(module, "experts", None)
    if not isinstance(experts, nn.ModuleList) or len(experts) == 0:
        return False
    if getattr(module, "gate", None) is None:
        return False

    return all(hasattr(experts[0], name) for name in _EXPERT_PROJECTIONS)


def _iter_sparse_moe_blocks(model: HFModel) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if _is_sparse_moe_block(module):
            yield name, module


def _has_lora_layers(module: nn.Module) -> bool:
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:
        LoraLayer = None

    for name, child in module.named_modules():
        if LoraLayer is not None and isinstance(child, LoraLayer):
            return True
        if "lora_" in name:
            return True

    return False


@torch.no_grad()
def _fuse_experts(experts: nn.ModuleList) -> FusedRoutedExperts:
    """Stack the per-expert MLPs into grouped-matmul weights."""
    first_expert = experts[0]
    act_fn = getattr(first_expert, "act_fn", None)
    if act_fn is None:
        act_fn = nn.SiLU()
    # ``torch.stack`` runs under ``no_grad``, so the trainability flag must be carried over explicitly.
    requires_grad = first_expert.gate_proj.weight.requires_grad
    gate_up_proj = torch.stack(
        [torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0).detach() for expert in experts],
        dim=0,
    )
    down_proj = torch.stack([expert.down_proj.weight.detach() for expert in experts], dim=0)
    fused = FusedRoutedExperts(
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        act_fn=act_fn,
        num_experts=len(experts),
        hidden_dim=first_expert.gate_proj.in_features,
        requires_grad=requires_grad,
    )
    if _has_lora_layers(experts):
        pack_expert_lora(experts, fused)
    return fused


def _sparse_moe_block_forward(module: nn.Module, hidden_states: torch.Tensor):
    """Run the router, the fused expert block and the shared expert of one sparse MoE block."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    topk_idx, topk_weight, router_logits = module.gate(hidden_states)
    hidden_states_out = module.experts(hidden_states, topk_idx, topk_weight)
    hidden_states_out = hidden_states_out.to(hidden_states.dtype)
    if getattr(module, "shared_experts", None) is not None:
        hidden_states_out = hidden_states_out + module.shared_experts(hidden_states)

    # The decoder layer expects ``(hidden_states, (router_logits, topk_idx))`` for this model.
    return hidden_states_out.view(batch_size, seq_len, hidden_dim), (
        router_logits.view(batch_size, seq_len, -1),
        topk_idx.view(batch_size, seq_len, -1),
    )


def prepare_for_ep(model: HFModel) -> HFModel:
    """Fuse the routed experts of every sparse MoE block so that FSDPTurbo EP can take over.

    Args:
        model: The (optionally LoRA-wrapped) base model resolved by the FSDPTurbo backend.

    Returns:
        The same model with ``mlp.experts`` replaced by :class:`FusedRoutedExperts` and the
        sparse MoE block ``forward`` rewritten to call the fused block.
    """
    init_mode = getattr(model, "_init_mode", _SUPPORTED_INIT_MODE)
    if init_mode != _SUPPORTED_INIT_MODE:
        raise RuntimeError(
            "FSDPTurbo expert parallelism fuses the BailingMoE routed experts at sharding time, so the "
            "fused parameter names no longer match the Hugging Face checkpoint and the sharded weight "
            f"loader cannot fill them (init_mode={init_mode}). Remove `init_config` (use the default "
            "init) or disable expert parallelism with `ep_size: 1`."
        )

    fused_blocks = 0
    for block_name, block in list(_iter_sparse_moe_blocks(model)):
        if getattr(block, "_ep_fused", False):
            continue
        block.experts = _fuse_experts(block.experts)
        block.forward = _sparse_moe_block_forward.__get__(block, type(block))
        block._ep_fused = True
        fused_blocks += 1

    if fused_blocks:
        logger.info_rank0(
            f"FSDPTurbo EP adapter: fused the routed experts of {fused_blocks} BailingMoE sparse MoE blocks."
        )
    else:
        logger.info_rank0("FSDPTurbo EP adapter did not find a BailingMoE sparse MoE block to fuse.")

    return model
