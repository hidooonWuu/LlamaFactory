# 分布式训练

训练命令检测到多设备后会自动通过 `torchrun` 启动。拓扑字段属于 `TrainingArguments`，后端专属字段放在 `dist_config`。

本页列出后端和拓扑配置，完整任务配置见 [SFT](sft.md)、[DPO](dpo.md)和 [RM](rm.md)。设备安装与支持范围见 [NPU 说明](../multi-backend/npu/index.md)。

## 数据并行

未设置 `dist_config` 时，多个 DP 进程使用 DDP，每个进程持有完整模型；单设备直接训练。

## FSDP2

FSDP2 通过分片降低每个设备上的模型状态内存开销，配置入口为 `dist_config.name: fsdp2`。

```yaml
dist_config:
  name: fsdp2
  reshard_after_forward: true
  offload_params: false
  pin_memory: true
  dcp_path: null
```

## FSDPTurbo

FSDPTurbo 在 FSDP2 基础上提供 MoE 专家并行和专家参数分片。先安装 FSDPTurbo 依赖：

```bash
python -m pip install -r requirements/fsdpturbo.txt
```

FSDPTurbo 的配置入口为 `dist_config.name: fsdpturbo`：

```yaml
dist_config:
  name: fsdpturbo
  ep_size: 16
  ep_dispatcher: eager
```

`ep_size` 必须能够整除 data parallel size。完整示例见 `examples/v1/train_full/train_full_qwen3_moe_fsdpturbo_ep_fsdp.yaml`。

### 专家并行模型支持

`ep_size > 1` 时，模型的专家模块结构由 `FSDPTurboEPModelSpec` 注册表维护，目前覆盖 `qwen3_moe`、`qwen3_5_moe` 和 `bailing_hybrid`，未注册的模型会直接报错。

Ling 3.0（`model_type: bailing_hybrid`）的 remote code 把路由专家保存在 `nn.ModuleList` 中并用 Python 循环执行，FSDPTurbo 的专家并行需要融合后的 `gate_up_proj` / `down_proj` 布局，因此插件在切分模型前会把每个稀疏 MoE Block 的路由专家融合成一个专家模块，并重写该 Block 的 `forward`。由此带来两个限制：

- 融合发生在切分阶段，融合后的参数名与 HF checkpoint 不再对应。`init_config: init_on_meta` 和 `init_on_rank0` 会跳过专家权重，因此被明确拒绝；不要配置 `init_config`，让 Transformers 先完成权重加载再切分。
- Ling 路由专家支持 `peft_config.target_modules: all`：先由 PEFT 注入 LoRA，再将 A/B 按专家轴打包并随基座分片；基座冻结，A/B 保持 FP32 存储。当前要求 `ep_dispatcher: eager`、`ep_size == dp_size`、单个普通加性 LoRA，专家间 rank/scaling/dropout 一致，不支持 DoRA 和适配器 bias。梯度按 EP 组平均，checkpoint 导出时还原标准 PEFT 参数名。其他不支持的专家 LoRA 布局会显式报错。

`ep_size` 还需要整除路由专家数（Ling 3.0 tiny 为 128）。建议令 `ep_size` 等于 data parallel size，以避免产生专家分片维度：`first_k_dense_replace` 之前的层是 dense MLP，没有路由专家，而 EFSDP 计划匹配的是 `model.layers.{*}.mlp`，会把这些层也当作专家模块分片。

LoRA 示例见 `examples/v1/train_lora/train_lora_ling_v3_tiny_fsdpturbo.yaml`（`ep_size: 1`）和 `examples/v1/train_lora/train_lora_ling_v3_tiny_fsdpturbo_ep.yaml`（专家并行）。

## DeepSpeed

DeepSpeed 后端从 `config_file` 读取 ZeRO 等配置，该字段必填。

```yaml
dist_config:
  name: deepspeed
  config_file: examples/deepspeed/ds_z3_config.json
```

## Ulysses Context Parallel

Ulysses CP 跨设备切分序列计算，由顶层 `cp_mode` 和 `cp_size` 启用：

```yaml
flash_attn: flash_attention_2
cp_mode: ulysses
cp_size: 2

dist_config:
  name: fsdp2
```

设置 `cp_size > 1` 后，训练使用 Ulysses 通信和 Sequence Parallel loss 完成跨 CP 进程的损失聚合。Ulysses 需要 `flash_attention_2` 和 FSDP2，不要求特定的 `batching_strategy`，支持 `normal` 和符合[批处理约束](batching.md)的 padding-free 策略。

`cp_size` 需要能够整除 world size。模型的 attention head 数必须能被 `cp_size` 整除，即 `num_attention_heads % cp_size == 0`；例如 32 个 attention head 可以使用 `cp_size: 2`。KV head 数与 `cp_size` 则要求其中一个能被另一个整除，即 `num_key_value_heads % cp_size == 0` 或 `cp_size % num_key_value_heads == 0`。当前只有 SFT 支持 `cp_size > 1`；DPO 和 RM 要求 `cp_size: 1`。

当前训练器不支持 `model_type: qwen3_5` 的 CP 路径。

## 配置并行拓扑

```yaml
dp_size: 4
cp_size: 2
cp_mode: ulysses
mp_replicate_size: 2
mp_shard_size: 4
dist_timeout: 18000
```

上例使用 8 个进程，并同时构造两套 DeviceMesh：

- Data Mesh 的形状为 `dp_size × cp_size = 4 × 2`，分别用于 Data Parallel 和 Context Parallel。
- Model Mesh 的形状为 `mp_replicate_size × mp_shard_size = 2 × 4`。FSDP 在 4 个进程间分片参数，并在 2 个分片组间复制参数。

`mp_replicate_size` 和 `mp_shard_size` 描述 FSDP 的二维参数 Mesh，不是额外的 Tensor Parallel 配置。未显式指定时，`dp_size` 默认为 `world_size / cp_size`，`mp_shard_size` 默认为 `world_size / mp_replicate_size`。后端完整配置见[训练参数](../configuration/training.md#dist_config)。

## 配置多机启动

CLI 读取 `NNODES`、`NODE_RANK`、`NPROC_PER_NODE`、`MASTER_ADDR` 和 `MASTER_PORT`。例如使用 4 台机器、每台机器 8 个设备时，在每台机器上执行：

```bash
NNODES=4 \
NODE_RANK=<0到3，各节点不同> \
NPROC_PER_NODE=8 \
MASTER_ADDR=<rank 0 节点的 IP> \
MASTER_PORT=29500 \
llamafactory-cli sft config.yaml
```

4 个节点需要使用相同的 `NNODES`、`NPROC_PER_NODE`、`MASTER_ADDR` 和 `MASTER_PORT`，并分别设置 `NODE_RANK=0`、`1`、`2`、`3`。
