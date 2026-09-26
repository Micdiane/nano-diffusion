# RTX 4090：SANA-600M 真实推理对比

比较的是 **nano 的完整进程内推理路径** 与 **SGLang 原生 SanaPipeline 的对齐配置**。
两者均从文本生成真实图像。SGLang 原始 `DiffGenerator` 入口也已完成出图检查；
主表不使用原始默认配置，因为它的 Gemma2 softcapping、噪声和 CFG 精度不同。

## 固定实验条件

| 项目 | 设置 |
| --- | --- |
| GPU / 驱动 | NVIDIA RTX 4090 24 GB / 580.142 |
| Python / PyTorch | 3.12.13 / 2.13.0 + CUDA 13.0 |
| Diffusers / Transformers | 0.37.0 / 5.12.1 |
| 模型 | `Efficient-Large-Model/Sana_600M_512px_diffusers` |
| 权重版本 | `83d7a190bfd1fd070570a793d2dab5c7a3231b9d` |
| SGLang 源码 | `0e2aac500bbd7010503bff67bd221bfbd7f04389` |
| 输入 | 512 × 512、20 steps、CFG 4.5、空 negative prompt |
| 负载 | batch=1；3 个提示词，seed 42/43/44；每个提示词重复 4 次 |
| 预热 | 每个引擎 3 次，不计入结果 |
| 精度 | DiT BF16、Gemma2 BF16、VAE FP32；噪声/latent/CFG FP32 |
| 文本 | HF Gemma2 eager，保留 softcapping，右侧 pad 到 300 token，无 prompt enhancement |
| 其他 | 8 个 CPU 线程；无 offload、缓存、编译、CUDA Graph；TF32 与 cuDNN SDPA 关闭 |

[完整环境锁定](requirements-rtx4090.txt) · [每个权重文件的 SHA-256](model-manifest.json)

## 为什么配置 SGLang 的组件

该版本原生 Gemma2 明确跳过 attention-logit softcapping；默认初始噪声与 CFG
还使用 BF16。直接把默认结果与官方 FP32 latent 路径放进同一张表，会混入算法差异。

[sglang_backend.py](sglang_backend.py)通过现有 `loaded_modules` 和 pipeline config
接口，传入相同的 HF 文本编码器、tokenizer、DC-AE，并指定 FP32 latent 和 CFG。
**DiT、权重映射、执行阶段、并行组、输入检查和 DPM scheduler wrapper 均为
SGLang 原生实现**。上游 checkout 保持干净，脚本启动时验证其 commit 与 tracked 文件。

这是一项用于学习系统开销的受控实验，不是对 SGLang 开箱即用配置的排名。
SGLang 的 fused projections 等 BF16 计算路径与 nano 不完全相同，因此还要检查
最终 latent 和图像误差。

## 重跑

建议在 Linux x86_64、Python 3.12 的独立环境中执行。以下锁定依赖针对本次 CUDA
机器，下载较大；只使用 nano 时不需要安装整套 SGLang 依赖。

```bash
python3.12 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install -r benchmarks/requirements-rtx4090.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/download_model.py

# 可使用已有源码目录；以下路径只是示例。
git clone --filter=blob:none https://github.com/sgl-project/sglang.git ../sglang
git -C ../sglang checkout 0e2aac500bbd7010503bff67bd221bfbd7f04389

python scripts/validate_checkpoint.py --checkpoint models/Sana_600M_512px_diffusers
OMP_NUM_THREADS=8 python benchmarks/run.py \
  --engine nano --checkpoint models/Sana_600M_512px_diffusers
OMP_NUM_THREADS=8 PYTHONPATH=../sglang/python python benchmarks/run.py \
  --engine sglang --checkpoint models/Sana_600M_512px_diffusers
python benchmarks/compare.py
```

所有命令均从 nano-diffusion 根目录执行。两个引擎应依次运行；脚本逐个校验本地
检查点文件的哈希，拒绝把不同权重标成相同版本。`run.py` 保存源码哈希和逐次耗时，
并要求每次请求实际完成 40 次 DiT forward。

## 指标口径

- **吞吐**：测量图像总数 / 总 wall time，不能把每次吞吐直接平均。
- **延迟**：同步 GPU 后开始计时，完整生成并将 float32 图像与最终 latent 传到
  CPU 后再次同步。包含文本编码、噪声、去噪、采样、VAE 和 CPU 输出；排除模型加载、
  下载、预热、哈希校验和 PNG/JSON/PT 写盘。不包含 HTTP/RPC 或多用户排队。
- **去噪 GPU 时间**：CUDA events 包围 nano 去噪循环或 SGLang DenoisingStage，
  包含其中 GPU 等待 CPU 发射的空档；不是所有 kernel 时长的简单求和。SGLang
  stage 还包含其条件/运行上下文准备，这属于两引擎执行结构的差异。
- **显存**：请求开始时重置 PyTorch peak 计数，包括常驻组件，分别记录 allocated
  和 reserved。不是 `nvidia-smi` 的整卡占用，不能用它估计任何负载都能运行的最低显存。
- **P95**：12 次请求的 nearest-rank 统计，此样本量下等于最大值。不是稳定的线上尾延迟估计。
- **一致性**：3 个提示词对应的 FP32 CPU 图像计算 PSNR；另记录 latent 相对 L2、
  cosine 与最大误差。它们反映输出差异，不衡量语义质量或 FID。

## 实测记录

| 引擎 | images/s | 平均延迟 | P95 | 去噪 GPU 时间 | peak allocated / reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| nano | 0.699 | 1.430 s | 1.629 s | 1.281 s | 7.967 / 8.576 GiB |
| SGLang 对齐路径 | 0.612 | 1.635 s | 1.680 s | 1.513 s | 7.970 / 8.625 GiB |

吞吐比为 **1.143×**。固定三个提示词的 PSNR 分别为 36.29、43.23、33.91 dB。

最终结果保存在 [rtx4090/](rtx4090/)，原始行包含 warmup，汇总自动排除 warmup。
三组提示词在 [run.py](run.py) 中逐字固定。实际输出可以直接并排查看：

| case | nano | SGLang 对齐路径 |
| --- | --- | --- |
| 小熊猫，seed 42 | ![nano panda](rtx4090/nano-0.png) | ![SGLang panda](rtx4090/sglang-0.png) |
| 木船，seed 43 | ![nano boat](rtx4090/nano-1.png) | ![SGLang boat](rtx4090/sglang-1.png) |
| 茶具，seed 44 | ![nano teapot](rtx4090/nano-2.png) | ![SGLang teapot](rtx4090/sglang-2.png) |

[checkpoint-validation.json](checkpoint-validation.json)记录真实权重与 Diffusers 的
六组单步对照及完整采样对照。CPU 测试只补充生命周期和小尺寸数学检查。

[首轮数据](rtx4090-initial/)也予以保留：当时 SGLang 进度条仍开启，尚未加 forward
调用计数与源码哈希；环境存在未使用的 CUTLASS cu13/base 版本冲突。最终轮已关闭
进度条、统一 CUTLASS 4.6.2，且 `uv pip check` 通过。首轮不是 README 主表的数据来源。

本次只覆盖一张 4090、三个提示词、固定 eager batch=1。没有测 FID、长负载并发、
动态图批处理、torch.compile、CUDA Graph 或更大模型；不能从这个表推出 SGLang 的性能上限。
