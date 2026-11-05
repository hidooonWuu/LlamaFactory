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

"""Packed Ling expert LoRA, dispatched on the same EP mesh as the frozen experts."""

from types import MethodType

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def pack_expert_lora(experts, fused):
    """Keep A/B separate and trainable; do not merge adapters into frozen weights."""
    from peft.tuners.lora import LoraLayer

    specs = {}
    trainable = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        layers = [getattr(expert, projection) for expert in experts]
        if not any(isinstance(layer, LoraLayer) for layer in layers):
            continue
        if not all(isinstance(layer, LoraLayer) for layer in layers):
            raise ValueError("Expert LoRA must target the same projection in every routed expert.")
        first = layers[0]
        adapters = first.active_adapters
        if len(adapters) != 1:
            raise ValueError("Ling EP expert LoRA currently supports one active adapter.")
        adapter = adapters[0]
        for layer in layers:
            if (
                layer.active_adapters != adapters
                or set(layer.lora_A) != {adapter}
                or set(layer.lora_B) != {adapter}
                or layer.merged
                or layer.disable_adapters
                or layer.use_dora.get(adapter, False)
                or layer.lora_B[adapter].bias is not None
            ):
                raise ValueError("Ling EP requires a single unmerged additive LoRA without DoRA or adapter bias.")
            if layer.scaling[adapter] != first.scaling[adapter] or getattr(
                layer.lora_dropout[adapter], "p", 0.0
            ) != getattr(first.lora_dropout[adapter], "p", 0.0):
                raise ValueError("Expert LoRA scaling and dropout must be uniform across experts.")
        spec = {
            "adapter": adapter,
            "scale": first.scaling[adapter],
            "dropout": getattr(first.lora_dropout[adapter], "p", 0.0),
        }
        for factor in ("A", "B"):
            name = f"lora_{projection}_{factor}_{adapter}"
            weights = [getattr(layer, f"lora_{factor}")[adapter].weight for layer in layers]
            if any(weight.shape != weights[0].shape for weight in weights):
                raise ValueError("Expert LoRA ranks must be uniform across experts.")
            if any(weight.requires_grad != weights[0].requires_grad for weight in weights):
                raise ValueError("Expert LoRA trainability must be uniform across experts.")
            requires_grad = weights[0].requires_grad
            fused.register_parameter(
                name, torch.nn.Parameter(torch.stack([w.detach() for w in weights]), requires_grad)
            )
            trainable[name] = requires_grad
            spec[factor] = name
        specs[projection] = spec
    if not specs:
        raise RuntimeError("LoRA adapters were found in experts but their layout is unsupported.")
    fused._expert_lora_specs = specs
    fused._expert_lora_trainable = trainable


def add_expert_lora(module, projection, expert, inputs, result):
    spec = getattr(module, "_expert_lora_specs", {}).get(projection)
    if spec is None:
        return result
    a = local_tensor(getattr(module, spec["A"]))[expert]
    b = local_tensor(getattr(module, spec["B"]))[expert]
    # Match PEFT dropout before FSDP's compute-dtype cast, while storing A/B in FP32.
    x = F.dropout(inputs.to(a.dtype), p=spec["dropout"], training=module.training)
    dtype = getattr(module, "_expert_lora_compute_dtype", None) or a.dtype
    delta = F.linear(F.linear(x.to(dtype), a.to(dtype)), b.to(dtype)) * spec["scale"]
    return (result + delta).to(result.dtype)


def expert_mlp(module, expert, inputs):
    gate_up = F.linear(inputs, local_tensor(module.gate_up_proj)[expert])
    gate, up = gate_up.chunk(2, dim=-1)
    gate = add_expert_lora(module, "gate_proj", expert, inputs, gate)
    up = add_expert_lora(module, "up_proj", expert, inputs, up)
    hidden = module.act_fn(gate) * up
    result = F.linear(hidden, local_tensor(module.down_proj)[expert])
    return add_expert_lora(module, "down_proj", expert, hidden, result)


def enable_expert_lora_ep(model, ep_group, compute_dtype):
    """Reuse FSDPTurbo token routing, with a LoRA-aware local expert computation."""
    from fsdp_turbo.distributed.expert_parallel.dispatcher import (
        alltoall_combine,
        alltoall_dispatch,
        dispatch_preprocess,
    )
    from fsdp_turbo.distributed.expert_parallel.utils import normalize_expert_args

    ep_size = torch.distributed.get_world_size(ep_group)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        shape = hidden_states.shape
        indices, weights = normalize_expert_args(top_k_index, top_k_weights)
        flat = hidden_states.reshape(-1, self.hidden_dim)
        counts, splits = dispatch_preprocess(ep_group, indices, self.num_global_experts, self.expert_ids_per_ep_rank)
        routed, permutations = alltoall_dispatch(ep_group, flat, indices, counts, splits, False)
        outputs = []
        offset = 0
        for expert, count in enumerate(counts[0].tolist()):
            if count:
                outputs.append(expert_mlp(self, expert, routed[offset : offset + count]))
            offset += count
        output = torch.cat(outputs, 0) if outputs else routed
        # Ensure inactive packed adapters get zero gradients (and normal AdamW
        # step/weight-decay semantics), including ranks receiving no tokens.
        zero = sum(local_tensor(getattr(self, name)).reshape(-1)[0] * 0 for name in self._expert_lora_trainable)
        output = output + zero.to(output.dtype)
        output = alltoall_combine(ep_group, output, weights, permutations, splits, False)
        return output.view(shape)

    for module in model.modules():
        if not getattr(module, "_expert_lora_specs", None):
            continue
        module._expert_lora_compute_dtype = compute_dtype
        module.forward = MethodType(forward, module)
        for name, trainable in module._expert_lora_trainable.items():
            parameter = getattr(module, name)
            parameter.requires_grad_(trainable)
            if trainable:
                # These EP-sharded parameters see the sum of losses from all EP
                # ranks. Outer FSDP averages the other gradients over DP already.
                parameter.register_hook(lambda grad, divisor=ep_size: grad / divisor)


def unpack_expert_lora_state_dict(model, state_dict):
    """Export full gathered tensors under standard per-expert PEFT adapter keys."""
    for prefix, module in model.named_modules():
        if not getattr(module, "_expert_lora_specs", None):
            continue
        for projection, spec in module._expert_lora_specs.items():
            for factor in ("A", "B"):
                key = f"{prefix}.{spec[factor]}"
                if key not in state_dict:
                    continue
                packed = state_dict.pop(key)
                for expert, tensor in enumerate(packed.unbind(0)):
                    state_dict[f"{prefix}.{expert}.{projection}.lora_{factor}.{spec['adapter']}.weight"] = (
                        tensor.contiguous()
                    )
    return state_dict
