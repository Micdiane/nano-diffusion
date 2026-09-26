# nano-diffusion

参考 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 分层思路实现的小型 flow-matching 推理引擎，用于学习扩散模型的请求、执行、采样与 Attention 调用链。

引擎自己实现 FIFO 调度、初始噪声、CFG 和 Flow Euler；模型通过适配接口接入。默认随机小 DiT 可在 CPU 运行，**输出是未训练模型的诊断图案**。可选 Wan2.1 T2V 适配器复用 Diffusers 的预训练组件，真实权重及 GPU 路径尚未验证。

## 快速开始

需要 Python 3.10+ 和 PyTorch 2.4+。在仓库根目录运行：

```bash
python -m pip install -e .
python -m nanodiffusion --frames 3 --steps 4 --output outputs/toy.pt
python -m examples.trace_requests
```

默认不下载模型。输出 `.pt` 保存 CPU `frames`、`latents` 和 `metadata`，同名 `.json` 保存可读元数据；帧布局是 `[B,F,H,W,3]`，FP32，范围 `[0,1]`。

```python
from nanodiffusion import Diffusion, SamplingParams
from nanodiffusion.models.toy import ToyAdapter

with Diffusion(ToyAdapter()) as engine:
    result = engine.generate("A boat on a lake", SamplingParams(num_steps=4))[0]
    print(result.frames.shape, result.num_model_calls)
```

需要逐步跟踪多个请求时，使用 `add_request()` 和 `step()`，示例见 [trace_requests.py](examples/trace_requests.py)。

## 源码阅读顺序

| 文件 | 职责 |
|---|---|
| [config.py](nanodiffusion/config.py)、[request.py](nanodiffusion/request.py) | 采样参数、请求中的 latent/条件/步数、返回结果 |
| [engine.py](nanodiffusion/engine.py) | `generate → add_request → step`，异常清理与结果收集 |
| [scheduler.py](nanodiffusion/scheduler.py) | FIFO waiting 队列与单个 active 请求 |
| [runner.py](nanodiffusion/runner.py) | 编码、噪声、模型输入 dtype、CFG、采样和解码 |
| [sampler.py](nanodiffusion/sampler.py) | 静态 sigma 时间表与 Flow Euler 积分 |
| [models/base.py](nanodiffusion/models/base.py) | 模型适配接口 |
| [models/toy.py](nanodiffusion/models/toy.py) | 随机小 DiT：文本/时间/空间条件及视频张量变换 |
| [models/wan.py](nanodiffusion/models/wan.py) | Wan2.1 组件加载、T5 padding、设备驻留、VAE 归一化 |
| [attention.py](nanodiffusion/attention.py) | 非 causal SDPA / 可选 SageAttention 后端 |

```text
prompt → Request → Scheduler → ModelRunner.prepare
                                  ↓
                         ModelAdapter.predict_velocity
                                  ↓
                            CFG → FlowEuler.step
                                  ↓
                         更新 latent，直到步骤完成
                                  ↓
                           ModelAdapter.decode → frames
```

请求是否排队/运行由 scheduler 统一管理。runner 保持 FP32 latent 状态，并准备设备上的模型 compute-dtype 输入和 FP32 timestep；适配器负责实际组件调用，decode 返回 CPU FP32 帧。

`step()` 推进一个去噪步骤，首步包含文本编码，末步包含解码。`generate()` 要求队列空闲；请求或进度回调异常会释放当前请求，保留 waiting 队列，可继续 `step()` 或 `close()`。使用 `with` 可确保调用适配器清理接口。当前是单请求顺序执行，没有 continuous batching、KV/TeaCache、训练或分布式逻辑。

## 采样公式

采用 `x_sigma = (1-sigma)*x_data + sigma*noise`，模型预测 `dx/dsigma`，从 sigma=1 积分到 0：

```text
v_cfg  = v_negative + guidance_scale * (v_positive - v_negative)
x_next = x_current + (sigma_next - sigma_current) * v_cfg
```

CFG=1 仅计算正条件分支；CFG>1 每步顺序计算两个分支。时间表为 `linspace(1, 1/1000, steps)`，经过 `shift*sigma / ((1-sigma)+shift*sigma)` 后追加终点 0，模型时间为 `1000*sigma`。这是明确选定的一阶积分基线，不等同于 Wan 官方 UniPC 的采样结果。

每个请求通过独立 CPU generator 产生噪声，同 seed 不受排队顺序影响；不同硬件、dtype 或 Attention 后端的结果不保证逐位一致。小 DiT 的文本条件是 UTF-8 字节 embedding 均值，RGB 解码器是随机卷积与插值，用于观察接口和张量，不代表 Wan 的完整网络。

## 可选 Wan2.1 T2V

在适配 CUDA 的独立环境中安装可选依赖，准备 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 格式的本地模型目录：

```bash
python -m pip install -e '.[wan,video]'
python -m nanodiffusion --model wan \
  --checkpoint /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --device cuda --dtype bfloat16 \
  --height 256 --width 256 --frames 17 --steps 30 --cfg 5 --shift 3 \
  --prompt 'A small boat drifting on a calm lake' \
  --output outputs/wan.pt --mp4 outputs/wan.mp4
```

只有显式传入 `--allow-download` 才允许远程下载。去掉 `--mp4` 可只保存张量，无需视频导出依赖。示例尺寸用于检查接口，未验证画质或 4090 显存占用。

- 只支持单 transformer、16 通道的 Wan2.1 T2V 布局；高宽为 16 的倍数，帧数为 `4*n+1`。目标是 1.3B，不提供 14B 部署保证。
- 默认整组件顺序卸载：文本编码器 → DiT → VAE；`--no-offload` 保持常驻。尚无逐层卸载或 VAE tiling。
- VAE 使用 FP32；解码前恢复 `latent * latents_std + latents_mean`。DiT 保留加载器设定的混合精度例外。
- `WanPipeline` 仅用于加载组件，不调用其推理入口或 scheduler；CFG 和积分始终在本项目中。

## Attention 与验证

默认使用 PyTorch SDPA。小 DiT 可通过 `--device cuda --dtype float16 --attention sage` 接入已安装的 SageAttention，接口限 CUDA FP16/BF16、head_dim 64/128、相同 head 数。**Wan 仍使用 Diffusers 原生 SDPA**，Sage 选项只接入小 DiT；GPU/Sage 扩展未在本项目中实测。

```bash
python -m unittest discover -s tests -v
```

测试使用 CPU 和 fake Wan 组件，覆盖解析积分、CFG、随机种子隔离、请求与异常生命周期、Attention 数学参考、模型张量契约、CLI 输出及组件卸载。测试通过不代表真实预训练模型的数值、生成质量或性能已经验证。

设计参考：[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)、[Diffusers Wan](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py)、[SGLang Diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)。LLM 调度进阶可继续阅读 [Mini-SGLang](https://github.com/sgl-project/mini-sglang)。本仓库不包含这些项目的权重。
