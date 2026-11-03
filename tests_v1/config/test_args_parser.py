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

import sys
from pathlib import Path
from unittest.mock import patch

from llamafactory.v1.config.arg_parser import get_args
from llamafactory.v1.utils.types import AttentionFunction


def test_get_args_from_yaml(tmp_path: Path):
    config_yaml = """
        ### model
        model: llamafactory/tiny-random-qwen3
        trust_remote_code: true
        model_class: llm
        kernel_config:
            name: auto, flash-linear-attention
            include_kernels: chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
            chunk_size: 32
        peft_config:
            name: lora
            r: 8
        quant_config: null

        ### data
        train_dataset: llamafactory/v1-sft-demo

        ### training
        output_dir: outputs/test_run
        micro_batch_size: 1
        global_batch_size: 1
        cutoff_len: 2048
        learning_rate: 1.0e-4
        bf16: false
        dist_config: null

        ### sample
        sample_backend: hf
        max_new_tokens: 128
    """

    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml, encoding="utf-8")

    test_argv = ["test_args_parser.py", str(config_file)]

    with patch.object(sys, "argv", test_argv):
        model_args, data_args, training_args, sample_args = get_args()
        assert data_args.train_dataset == "llamafactory/v1-sft-demo"
        assert model_args.model == "llamafactory/tiny-random-qwen3"
        assert model_args.kernel_config.name == "auto, flash-linear-attention"
        assert model_args.kernel_config.get("include_kernels") == (
            "chunk_gated_delta_rule, fused_recurrent_gated_delta_rule"
        )
        assert model_args.kernel_config.get("chunk_size") == 32
        assert model_args.peft_config.name == "lora"
        assert model_args.peft_config.get("r") == 8
        assert training_args.output_dir == "outputs/test_run"
        assert training_args.micro_batch_size == 1
        assert training_args.global_batch_size == 1
        assert training_args.learning_rate == 1.0e-4
        assert training_args.bf16 is False
        assert training_args.dist_config is None
        assert sample_args.sample_backend == "hf"


def test_qwen35_fsdpturbo_example_uses_v1_arguments():
    config_file = (
        Path(__file__).parents[2] / "examples" / "v1" / "train_full" / "train_full_qwen3_moe_fsdpturbo_ep_fsdp.yaml"
    )

    with patch.object(sys, "argv", ["test_args_parser.py", str(config_file)]):
        model_args, _, training_args, _ = get_args()

    assert model_args.model == "Qwen/Qwen3.5-35B-A3B"
    assert model_args.custom_chat_template is None
    assert training_args.dist_config.name == "fsdpturbo"


def test_ling_v3_tiny_lora_fsdpturbo_example_uses_v1_arguments():
    config_file = (
        Path(__file__).parents[2] / "examples" / "v1" / "train_lora" / "train_lora_ling_v3_tiny_fsdpturbo.yaml"
    )

    with patch.object(sys, "argv", ["test_args_parser.py", str(config_file)]):
        model_args, _, training_args, _ = get_args()

    assert model_args.model == "/home/models/ling-3.0-tiny"
    assert model_args.trust_remote_code is True
    assert model_args.flash_attn == AttentionFunction.SDPA
    assert model_args.peft_config.name == "lora"
    assert model_args.peft_config.get("target_modules") == "all"
    assert training_args.dist_config.name == "fsdpturbo"
    assert training_args.dist_config.get("ep_size") is None
    assert training_args.enable_activation_checkpointing is True


def test_ling_v3_tiny_lora_fsdpturbo_ep_example_uses_v1_arguments():
    config_file = (
        Path(__file__).parents[2] / "examples" / "v1" / "train_lora" / "train_lora_ling_v3_tiny_fsdpturbo_ep.yaml"
    )

    with patch.object(sys, "argv", ["test_args_parser.py", str(config_file)]):
        model_args, _, training_args, _ = get_args()

    target_modules = model_args.peft_config.get("target_modules")
    assert training_args.dist_config.name == "fsdpturbo"
    assert training_args.dist_config.get("ep_size") == 8
    assert training_args.global_batch_size == 8
    assert isinstance(target_modules, list)
    assert "q_a_proj" in target_modules
    # routed experts are fused by FSDPTurbo EP, so their projections must not be adapted
    for expert_projection in ("gate_proj", "up_proj", "down_proj"):
        assert expert_projection not in target_modules
