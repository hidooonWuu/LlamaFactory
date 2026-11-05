# Ascend NPU

本页介绍 Ascend NPU 的手动安装、功能范围和融合算子加速能力。

## 手动安装

手动安装需要依次准备 HDK 驱动与固件、CANN 和 `torch-npu`。安装包需要与 NPU 型号、CPU 架构和操作系统匹配。本分支使用以下软件组合：

| 组件 | 版本 |
|------|------|
| CANN | `9.1.0` |
| PyTorch | `2.10.0` |
| torch-npu | `2.10.0.post2` |
| torchvision | `0.25.0` |
| torchaudio | `2.10.0` |
| Triton Ascend | `3.2.1` |

安装前使用[Ascend 兼容性查询助手](https://www.hiascend.com/hardware/compatibility)确认硬件与操作系统组合，并从[CANN 9.1.0 社区版资源中心](https://www.hiascend.com/developer/download/community/result?cann=9.1.0&module=cann)选择对应 CPU 架构的软件包。

### 安装驱动与固件

从 Ascend 下载与设备匹配的 HDK 驱动和固件安装包。以下命令中的文件名需要替换为实际下载的包名：

```bash
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run

sudo ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run \
  --full --install-for-all
sudo ./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

根据安装程序提示重启系统，然后验证驱动：

```bash
npu-smi info
```

### 安装 CANN

先安装 Toolkit，再安装与设备匹配的 ops 算子包：

```bash
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
sudo ./Ascend-cann-toolkit_<version>_linux-<arch>.run --install

chmod +x Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run
sudo ./Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run --install
```

默认以 root 安装时，加载以下环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

普通用户安装时，环境脚本位于用户选择的 CANN 安装目录。需要在每个运行 LlamaFactory 的 shell 中加载该脚本，也可以将命令加入 shell 配置文件。

### 安装 LlamaFactory

在项目根目录创建 Python 环境，并安装 NPU 依赖和 LlamaFactory：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements/npu.txt
python -m pip install -r requirements/triton_ascend.txt
```

`requirements/npu.txt` 固定了相互匹配的 PyTorch、torchvision、torchaudio 和 `torch-npu` 版本；`requirements/triton_ascend.txt` 安装 NPU 对应的 Triton 实现。如果使用已有 Python 环境，应确认安装完成后的 PyTorch 与 `torch-npu` 版本仍与上述版本一致。

### 验证 PyTorch NPU

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

输出 `True` 表示 PyTorch 已识别 NPU。启用 v1 后即可运行 SFT：

```bash
export USE_V1=1
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
```

## 功能范围

| 功能 | 状态 | 说明 |
|------|:----:|------|
| [SFT 全参训练](../../feature-guide/sft.md) | 支持 | 支持 FSDP2 |
| [LoRA / Freeze](../../feature-guide/sft.md) | 支持 | 使用通用 PEFT 路径 |
| [QLoRA](../../feature-guide/sft.md#qlora) | 不支持 | 当前量化路径依赖 bitsandbytes |
| [DPO](../../feature-guide/dpo.md) | 支持 | 使用偏好对数据 |
| [RM](../../feature-guide/rm.md) | 支持 | `cp_size` 需要为 `1` |
| [FSDP2](../../feature-guide/distributed_training.md) | 支持 | 使用 NPU 设备和通信后端 |
| [FSDPTurbo](../../feature-guide/distributed_training.md) | 支持 | 提供 MoE 专家并行和专家参数分片 |
| [Ulysses CP](../../feature-guide/distributed_training.md#ulysses-context-parallel) | 支持 | 依赖适配的 attention 实现 |
| [DeepSpeed](../../feature-guide/distributed_training.md#deepspeed) | 依赖环境 | 由 NPU DeepSpeed 发行版和配置决定 |
| [HF CLI 推理](../../feature-guide/inference.md) | 支持 | `sample_backend: hf` |

## 量化支持

当前 v1 QLoRA 依赖 bitsandbytes，因此不适用于 NPU。

## 融合算子加速

`kernel_config.name: auto` 在 NPU 上依次尝试：

- `npu_fused_moe`
- `npu_fused_rmsnorm`
- `npu_fused_rope`
- `npu_fused_swiglu`

也可以显式选择一个或多个名称：

```yaml
kernel_config:
  name: npu_fused_rmsnorm,npu_fused_rope
```

每个实现会检查当前设备及 `torch_npu` 依赖。模型结构不匹配时，具体 Kernel 可能跳过替换或抛出明确错误。

## Liger Kernel

`liger_kernel` 接受 CUDA 或 NPU，但仍要求模型类型在当前 Liger 映射中，并且已安装兼容版本的 `liger_kernel`。

## Ling 3.0（bailing_hybrid）

Ling 3.0 使用 remote code，`model_type` 为 `bailing_hybrid`，需要 `trust_remote_code: true`。

在 Transformers 5.x 下，LlamaFactory 会在加载模型时适配旧版 FX 导入接口和 RoPE 初始化，保留检查点的 `rope_theta` 与 MLA 旋转维度，无需修改模型目录的 Python 文件。该适配已在 Transformers 5.0.0 下验证模型结构初始化和缩小 MLA 模型的前向、反向计算；完整 NPU 训练仍依赖匹配的 FLA 与 Triton 环境。

适配同时处理短序列 KDA 训练：remote code 会对不超过 64 token 的输入选择 `fused_recurrent`，但训练仅支持 `chunk`，因此 LlamaFactory 对这类训练输入调用 `chunk_kda`。推理和超过 64 token 的输入保留原有执行路径。`cutoff_len` 是截断上限，不能保证实际输入长度超过 64。

在 NPU 上运行该模型前需要额外满足：

- 安装 `fla-core`（`python -m pip install -U fla-core`）。remote code 在 import 阶段就会加载 `fla` 的 KDA 算子与 `ShortConvolution`、`FusedRMSNormGated`，缺少该依赖会直接失败；KDA 的 triton 算子需要在 NPU 上可用的 Triton 实现。
- `flash_attn: flash_attention_2` 已接入 Ling MLA 的实际计算；Ascend 使用 Transformers 的 NPU FA2 接口，调用 `torch_npu.npu_fusion_attention`。保留原始 Q/K/V 投影、RoPE、缩放、KV cache、门控和 value 维度填充；支持因果注意力与 2D padding mask，不返回 attention weights。KDA 层保持 FLA 实现，当前要求 `cp_size: 1`；KDA context parallelism 尚未适配，padding-free 尚未验证。
- `flash_attn: sdpa` 保留 legacy eager MLA 路径。SDPA 辅助函数可能省略全 1 mask，因此适配器会在 eager 入口补齐显式因果遮罩，避免未来 token 泄漏。旧版训练日志即使两后端 loss 接近，也不能代替因果性验证；切换 FA2 的精度对照必须使用修正后的基线。
- LoRA 与 FSDPTurbo 的配置见 `examples/v1/train_lora/train_lora_ling_v3_tiny_fsdpturbo.yaml`（`ep_size: 1`）和 `examples/v1/train_lora/train_lora_ling_v3_tiny_fsdpturbo_ep.yaml`（专家并行）。Ling 专家并行已支持路由专家 LoRA：A/B 按专家轴打包分片，冻结的专家基座不参与训练，保存时恢复标准 PEFT 参数名。当前要求 `ep_dispatcher: eager`、`ep_size == dp_size`、单个活动的普通加性 LoRA、专家间 rank/scaling/dropout 一致，不支持 DoRA 和适配器 bias。

对比 EP 与原生 FSDP2 的 loss 时，`train_lora_ling_v3_tiny_fsdpturbo_ep.yaml` 与 `train_lora_ling_v3_tiny_fsdp.yaml` 均使用 `target_modules: all`，并保持数据、种子、batch、数据加载进程数和 `flash_attn` 一致。EP 示例当前选用 FA2，原生 FSDP2 示例保留 sdpa 作为因果 eager 参考；若只对比后端，需要将两边 attention 选项设为相同值。`all` 包括 attention、路由专家、共享专家和普通 MLP，不能与仅训练 attention 的结果直接比较。原生 FSDP2 将 Ling 专家适配器按 MoE 层分组分片，减少小通信；该路径的冻结专家基座保持复制，因此显存占用会增加。FSDPTurbo 在 LoRA 模式下保留适配器的 FP32 存储精度，`bf16` 控制混合精度计算，避免整模型转换把优化器更新一起舍入到 BF16。

训练日志为追加写入，比较时需要分离重复启动的运行并按 step 对齐。可使用 `python scripts/stat_utils/compare_backend_losses.py <fsdp2_log> <ep_log> --expected-steps 250 --threshold-percent 2 --output-dir <report_dir> --plot` 生成完整运行的逐步误差、汇总和曲线图。平均相对误差定义为 `mean(abs(loss_ep - loss_fsdp2) / abs(loss_fsdp2))`，工具只在完整步数齐全且误差低于阈值时成功退出。

## 分布式训练

FSDP2、FSDPTurbo 与 Ulysses 使用 accelerator 抽象选择 NPU 通信设备。DeepSpeed 的可用性由 NPU DeepSpeed 发行版、依赖版本和配置共同决定。
