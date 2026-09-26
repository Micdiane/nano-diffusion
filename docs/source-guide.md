# 从 nano-vLLM 读到真实 SANA

这条主线固定为 **SANA-600M → RTX 4090 → SGLang Diffusion 对照**。
模型小，文本编码器、DiT、VAE 可同时常驻 24 GB 显存；每一处优化都能在真实
预训练权重和生成图像上验证。先读清当前 eager 基线，再研究 CUDA Graph、
投影融合与 Triton。

## 一条请求怎样执行

1. `SamplingParams` 指定尺寸、步数、CFG 和 seed；`Request` 保存当前 latent、
   两份文本条件和一个独立 DPM-Solver 状态。
2. `Diffusion.add_request()` 检查 latent 尺寸并排入 FIFO。一次只推进一个请求。
   `step()` 推进一步；请求完成后释放条件和采样器历史。
3. `ModelRunner.prepare()` 用真实 Gemma2 编码正负提示词。每条请求使用独立 CPU
   generator 生成 FP32 噪声，再传到 GPU。固定 seed 还不够，RNG 设备和噪声 dtype
   也必须一致。
4. `denoise_step()` 把输入转 BF16，顺序执行正负两个 DiT 分支，在 FP32 合成
   `v = v_neg + cfg * (v_pos - v_neg)`，交给检查点配置的 DPM-Solver++ 更新 latent。
5. DC-AE 将 `latent / 0.41407` 解码为真实像素。输出归一化至 `[0,1]` 并传回 CPU。

DPM-Solver 是二阶、多步求解器，保存前一步预测；不能让两条请求共享可变历史。
这里复用 Diffusers 的数值实现，SGLang 的 SANA scheduler 同样委托该实现。
模型实际时间表来自 `use_flow_sigmas=true, flow_shift=3.0`，不能换成任意 Euler
时间表后仍声称是相同模型基线。

## DiT 的真实结构

512px 输入经 DC-AE 的 32 倍空间压缩，得到 `[B,32,16,16]` latent，即 256 个
空间 token。600M 检查点使用 28 个 block、1152 维隐藏层、36 个线性注意力头，
文本 cross-attention 为 16 个头。实际加载仍以配置文件为准，权重键严格匹配。

每个 block：

```text
AdaLN → ReLU linear self-attention → gated residual
      → softmax text cross-attention → residual
      → AdaLN → spatial GLUMBConv → gated residual
```

线性注意力计算 `φ(Q)(φ(K)^T V)`，其中 `φ=ReLU`，分母用相同的 K/Q 累积归一化。
实现把一行常数 1 拼到 V 中，使输出和分母共用两次 matmul。FP32 累积减少分母
过小时的误差。它没有构造空间 `N×N` 注意力矩阵，因此不能直接把该部分替换为
FlashAttention/SageAttention；两者解决的是 softmax attention 的不同实现问题。

GLUMBConv 的路线是 `1×1 expand → SiLU → depthwise 3×3 → GLU → 1×1 project`。
这里必须恢复二维空间布局；把它写成普通 Transformer MLP 会改变模型本身。

## 一个真实的数值陷阱

BF16 GPU 对照中，`x + attention(x)` 与 `attention(x) + x` 的首层数值可以相同，
但输出 stride 不同。后续矩阵乘法可能选择不同执行方式，偏差会沿 28 层传播。
本实现保留官方 cross-attention 残差加法顺序；真实权重的六组单步检查达到
逐元素一致。不要仅凭 CPU 小矩阵测试就认定 CUDA 低精度路径完全一致。

完整 Diffusers pipeline 把 CFG 两个分支合成 batch=2，本实现和 SGLang 对齐为
顺序 batch=1；GEMM 形状不同会产生可见于数值指标的舍入差异。前向一致和整条
采样轨迹一致是两件需要分别测量的事。

## SGLang 的对应入口

上游固定到 `0e2aac500bbd7010503bff67bd221bfbd7f04389`，从以下位置阅读：

| nano | SGLang 中的对应位置 |
| --- | --- |
| `engine.py` / `scheduler.py` | `runtime/entrypoints/diffusion_generator.py`、`runtime/managers/scheduler.py` |
| `runner.py` | `runtime/pipelines_core/stages/{text_encoding,latent_preparation,denoising,decoding}.py` |
| `models/sana_transformer.py` | `runtime/models/dits/sana.py` |
| `models/sana.py::make_sampler` | `runtime/models/schedulers/scheduling_dpm_solver_multistep.py` |
| 公平对照配置 | 本仓库 `benchmarks/sglang_backend.py` |

路径均相对于上游 `python/sglang/multimodal_gen/`。

## 接下来的单卡深挖顺序

- **先 profile**：区分 GPU kernel 时间与 CPU 发射间隙，分别看 Gemma2、40 次
  DiT forward 和 DC-AE；不要由 FLOPs 直接推断吞吐。
- **CUDA Graph**：固定 512px、300 文本 token，给正负分支分别准备静态输入，
  先捕获 DiT forward。请求调度与 DPM 历史仍由 CPU 管理，检查重放时输入更新。
- **投影融合**：合并 self-attention 的 Q/K/V 和 cross-attention 的 K/V。
  同一检查点先做权重映射，再测单步误差、完整图像误差与真实吞吐。
- **Triton AdaLN**：只在 profile 证明小算子发射占比高时尝试；对照 BF16
  舍入顺序，尤其注意 norm、scale、shift 与 residual 的融合误差。
- **扩大工作负载**：完成 batch=1 后再测 batch=2/4、不同尺寸和多请求调度，
  保留原始行记录，分别报告吞吐与延迟。

每一步只改变一个机制，同时记录数值差异、images/s、延迟和显存。当前项目的
吞吐表只代表固定 eager 小负载，不代表 SGLang 开启图捕获、编译或动态批处理后的上限。
