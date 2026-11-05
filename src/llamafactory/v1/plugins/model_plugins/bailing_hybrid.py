# Copyright 2026 the LlamaFactory team.
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

"""Adapt Ling's attention and Transformers 5 compatibility without editing checkpoint files."""

import sys
from copy import deepcopy
from functools import wraps
from threading import RLock

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils

from ...utils.packages import is_transformers_version_greater_than


_IMPORT_LOCK = RLock()


def patch_bailing_hybrid_config(config) -> None:
    """Preserve the checkpoint's RoPE theta during Transformers 5 normalization."""
    if config.model_type != "bailing_hybrid" or not is_transformers_version_greater_than("5.0.0"):
        return

    rope = dict(config.rope_scaling or {})
    # The remote config consumes rope_theta before PretrainedConfig sees it. In 5.0,
    # normalization consequently inserts 10000 instead of the checkpoint's value.
    rope["rope_theta"] = config.rope_theta
    rope.setdefault("rope_type", rope.get("type", "default"))
    config.rope_scaling = rope


def _default_rope_parameters(config, device=None, seq_len=None):
    dim = config.qk_rope_head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, 1.0


def _patch_kda_training(module) -> None:
    """Use differentiable chunk KDA for short training sequences as well."""
    attention = getattr(module, "BailingMoeV3KimiDeltaAttention", None)
    if attention is None or getattr(attention, "_llamafactory_chunk_training", False):
        return
    original_forward = attention.forward

    @wraps(original_forward)
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, **kwargs):
        if not self.training or hidden_states.shape[1] > 64:
            return original_forward(
                self, hidden_states, attention_mask=attention_mask, past_key_value=past_key_value, **kwargs
            )
        if past_key_value is not None:
            raise ValueError("Ling KDA short-sequence training requires use_cache=False.")

        # The original forward selects fused_recurrent for <=64 tokens before
        # asserting that training uses chunk. Keep its training math and existing
        # projections/convolutions, but dispatch these short inputs to chunk_kda.
        cu_seqlens = kwargs.get("cu_seqlens")
        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        g = self.f_proj(hidden_states) if self.no_kda_lora else self.f_b_proj(self.f_a_proj(hidden_states))
        beta = self.b_proj(hidden_states).float().sigmoid()
        q, k = (module.rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = module.rearrange(v, "... (h d) -> ... h d", d=self.head_dim)
        g = module.rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        output, _ = module.chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens,
        )
        gate = self.g_proj(hidden_states) if self.no_kda_lora else self.g_b_proj(self.g_a_proj(hidden_states))
        gate = module.rearrange(gate, "... (h d) -> ... h d", d=self.head_dim)
        output = self.o_norm(output, gate)
        output = module.rearrange(output, "b t h d -> b t (h d)")
        return self.o_proj(output), None, None

    attention.forward = forward
    attention._llamafactory_chunk_training = True


def _patch_mla_flash_attention(module) -> None:
    """Dispatch Ling MLA to HF FA2 (Ascend uses torch_npu fusion attention).

    Keep the remote forward's Q/K/V projections, value padding, RoPE, cache,
    scaling and output gating. Only replace its hard-coded eager kernel. The
    enclosing model already supplies FA2's 2D padding mask in this mode.
    """
    if getattr(module, "_llamafactory_mla_flash_attention", False):
        return
    eager = getattr(module, "eager_attention_forward", None)
    attention_class = getattr(module, "BailingMoeV3MultiLatentAttention", None)
    if eager is None or attention_class is None:
        return

    @wraps(eager)
    def attention_forward(attention, query, key, value, attention_mask, **kwargs):
        is_mla = isinstance(attention, attention_class)
        if is_mla and attention.config._attn_implementation == "flash_attention_2":
            from transformers.integrations.flash_attention import flash_attention_forward

            if attention_mask is not None and attention_mask.ndim != 2:
                raise ValueError("Ling FlashAttention 2 requires a 2D padding mask or None.")
            if kwargs.get("output_attentions", False):
                raise ValueError("Ling FlashAttention 2 cannot return attention weights; use sdpa/eager instead.")
            return flash_attention_forward(attention, query, key, value, attention_mask, **kwargs)
        if is_mla and (attention_mask is None or attention_mask.ndim == 2):
            # The SDPA mask helper may return None for all-valid inputs because
            # real SDPA applies is_causal itself. Ling's eager kernel does not.
            # Restore an explicit bottom-right-aligned causal mask, including
            # chunked cache decoding where query and key lengths differ.
            q_len, kv_len = query.shape[-2], key.shape[-2]
            causal_mask = torch.full(
                (q_len, kv_len), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            ).triu(kv_len - q_len + 1)[None, None]
            if attention_mask is not None:
                causal_mask = causal_mask.expand(query.shape[0], 1, q_len, kv_len).masked_fill(
                    ~attention_mask[:, None, None, :kv_len].bool(), torch.finfo(query.dtype).min
                )
            attention_mask = causal_mask
        return eager(attention, query, key, value, attention_mask, **kwargs)

    module.eager_attention_forward = attention_forward
    module._llamafactory_mla_flash_attention = True


def _patch_model_module(module) -> None:
    _patch_mla_flash_attention(module)
    if getattr(module, "_llamafactory_tf5_compat", False):
        return

    def rotary_init(self, config, device=None):
        torch.nn.Module.__init__(self)
        self.config = deepcopy(config)
        self.config.head_dim = config.qk_rope_head_dim
        self.config.partial_rotary_factor = 1.0
        self.config.rope_parameters["partial_rotary_factor"] = 1.0
        self.rope_type = self.config.rope_scaling.get("rope_type", "default")
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_init_fn = (
            _default_rope_parameters if self.rope_type == "default" else module.ROPE_INIT_FUNCTIONS[self.rope_type]
        )
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    attention_init = module.BailingMoeV3MultiLatentAttention.__init__

    @wraps(attention_init)
    def mla_init(self, config, *args, **kwargs):
        rope = config.rope_scaling or {}
        if rope.get("rope_type", "default") == "default" and "factor" not in rope:
            # Legacy MLA reads factor unconditionally. Default RoPE has no scaling;
            # supply the identity only on this attention instance's private config.
            config = deepcopy(config)
            config.rope_scaling["factor"] = 1.0
        attention_init(self, config, *args, **kwargs)

    module.BailingMoeV3RotaryEmbedding.__init__ = rotary_init
    module.BailingMoeV3MultiLatentAttention.__init__ = mla_init
    _patch_kda_training(module)
    module._llamafactory_tf5_compat = True


def prepare_bailing_hybrid_model(config, model_path: str, auto_class, trust_remote_code: bool) -> None:
    """Patch the imported model classes before AutoModel constructs the model."""
    if config.model_type != "bailing_hybrid" or not trust_remote_code:
        return

    reference = getattr(config, "auto_map", {}).get(auto_class.__name__)
    if reference is None:
        return

    # 4.57 implemented this helper as is_torch_available(). Provide it only while
    # importing Ling's trusted code, then restore the Transformers namespace.
    with _IMPORT_LOCK:
        missing_fx_helper = not hasattr(import_utils, "is_torch_fx_available")
        if missing_fx_helper:
            import_utils.is_torch_fx_available = import_utils.is_torch_available
        try:
            model_class = get_class_from_dynamic_module(reference, model_path)
        finally:
            if missing_fx_helper:
                del import_utils.is_torch_fx_available

        module = sys.modules[model_class.__module__]
        if is_transformers_version_greater_than("5.0.0"):
            _patch_model_module(module)
        else:
            _patch_mla_flash_attention(module)
