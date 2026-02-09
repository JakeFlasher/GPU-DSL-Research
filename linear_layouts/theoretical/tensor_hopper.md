2025 年就要结束了，感觉大家都在冲业绩，我也来知乎冲一下 KPI 吧（

### 理论分析

算子仙人在施法时，时常会先查看法器的一些属性，比如法器的威力、施法速度等，以评估法术施展的实际效果。鉴于现在 Hopper 系列的 GPU 相对来说使用比较广泛，我们先来看看 Hopper 系列不同 GPU 的一些数据：

| GPU | Peak FP16 Tensor TFLOPS | Memory Bandwidth |
| --- | --- | --- |
| HGX H20 | 148 | 4.0 TB/s |
| H100 PCIe | 756 | 2039 GB/s |
| H100 SXM5 | 989.4 | 3352 GB/s |

一般来说，我们会直接用 TFLOPS / Bandwidth 来算出一个硬件的峰值计算强度 (PAI, Peak Arithmetic Intensity) FLOPS/Byte，用于评估某个算子在该硬件上是计算受限还是访存受限。例如，对于 HGX H20 来说：

PAI\=1484\=37 FLOPS/Byte  \\text{PAI} = \\dfrac{148 }{4} = 37 \\text{ FLOPS/Byte}

对于相当一部分算子开发者来说，故事到这里就结束了。然而，较为成熟的算子开发者来说通常还会进一步地了解标称算力的组成部分。他们知道配有 Tensor Core (TCU) 的 GPU 的标称算力可以通过以下公式计算得出：

TFLOPS\=Number of TCUs×FLOPs/TCU/cycle×Clock Speed  \\text{TFLOPS} = \\text{Number of TCUs} \\times \\text{FLOPs/TCU/cycle} \\times \\text{Clock Speed}

以 H100 PCIe 为例，从白皮书中我们可以获得以下信息：

| GPU Features | Value |
| --- | --- |
| SMs | 114 |
| Tensor Cores/SM | 4   |
| Tensor Cores/GPU | 456 |
| GPU Boost Clock2 for FP8, FP16, BF16, TF32 Tensor Core Ops | 1620 MHz |

于是我们有 456×1620MHz×FLOPs/TCU/cycle\=756TFLOPS456 \\times 1620 \\text{MHz} \\times \\text{FLOPs/TCU/cycle} = 756 \\text{TFLOPS}，掏出计算器可以得到 FLOPs/TCU/cycle\=1023.39≈1024\\text{FLOPs/TCU/cycle} = 1023.39 \\approx 1024。同样地，我们也可以计算出 H100 SXM5 的 FLOPs/TCU/cycle\\text{FLOPs/TCU/cycle} 也是约等于 1024，而 HGX H20 的 FLOPs/TCU/cycle\\text{FLOPs/TCU/cycle} 则是约等于 256。

对于相当一部分较为成熟的算子开发者来说，故事到这里又结束了。然而，对微架构有更深入了解的算子开发者来说，他们还会进一步地了解不同的 Tensor Core 指令的延迟和吞吐量 (Latency & Throughput) 并对 Tensor Core 的计算方式进行相关资料的查阅或进行实验猜测。除了白皮书外，他们还会参考下面的一些资料：

1.  针对架构的微基准测试论文：[Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.13499)
2.  相关专利：[Efficient Matrix Multiply and Add with a Group of Warps](https://link.zhihu.com/?target=https%3A//patents.google.com/patent/US20230289398A1/en%3Foq%3DUS20230289398A1%25EF%25BC%25880091%25EF%25BC%2589)
3.  看大佬知乎吹水

笔者借助 [NVIDIA Hopper Benchmark](https://link.zhihu.com/?target=https%3A//github.com/HPMLL/NVIDIA-Hopper-Benchmark) 对 A/B 矩阵均位于 Shared Memory 的 Tensor Core 指令 `wgmma.mma_async.shape.f32.f16.f16` 在不同的 `shape` 下进行了延迟测试，于下表中与 [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.13499) 中 H800 的测试结果一并列出：

| Instruction Shape | H800 | H20 |
| --- | --- | --- |
| m64n8k16 | 18  | 18  |
| m64n16k16 | 20  | 32  |
| m64n32k16 | 24  | 64  |
| m64n64k16 | 32  | 128 |
| m64n128k16 | 64  | 256 |
| m64n256k16 | 128 | 512 |

需要指出的是，该延迟的测试方式为持续不断地发射指令以填满 Tensor Pipe，而非对单次指令进行测量。

我们有如下发现：

1.  N\=8N = 8 时，H20 和 H800 的指令延迟相同
2.  在 H800 上，当 N≥64N ≥ 64 时，指令延迟倍增
3.  在 H20 上，当 N≥16N ≥ 16 时，指令延迟倍增

我们先着眼于第一点，两者的指令延迟均为 18 个时钟周期，读者就要问了：为什么是 18 个时钟周期？因为这个数比较吉利吗？

对数字较为敏锐的读者可能已经注意到了，18 个时钟周期正好是读取 A 矩阵和 B 矩阵的数据所需的时间。我们知道，Shared Memory 的带宽为 128 Bytes/cycle，A 矩阵和 B 矩阵的大小分别为 64×16×2\=204864 \\times 16 \\times 2 = 2048 Bytes 和 8×16×2\=2568 \\times 16 \\times 2 = 256 Bytes，读取 A 需要 2048⁄128 = 16 个周期，读取 B 需要 256⁄128 = 2 个周期，总共需要 18 个周期。

同理，对于 H800 来说，在 N 变化时，A 矩阵的大小不变，而 B 矩阵的大小会随着 N 的变化而变化。当 N = 16 时，读取 B 矩阵需要 4 个周期，总共 20 个周期；同理可计算出 N = 32、64 时的延迟分别为 24、32 个周期。

按这个思路，读者会进一步计算 N=128 时，读取 B 矩阵需要 32 个周期，总共 48 个周期，然后发现实际延迟为 64 个周期。或者看到 H20 上 N=16 时，延迟为 32 个周期，而非 20 个周期。

So…why?

我们换另一种算法，在前面我们已经得知 H800 的 Tensor Core 每个周期可以完成 1024 FLOPs，而 `wgmma` 指令需要由一个 `Warp Group = 4 Warps` 来完成计算，实际上是同时使用了单个 SM 里的 4 个 Tensor Cores 来完成计算。因此，每个周期可以完成 1024×4\=40961024 \\times 4 = 4096 FLOPs。

对于 `m64n128k16` 指令来说，从计算角度讲，其需要 64×128×16×24096\=64 \\frac{64 \\times 128 \\times 16 \\times 2}{4096} = 64 个周期来完成计算。读者按同样地计算可以得到 H20 上 N≥16N ≥ 16 的延迟也是计算所需的时间。

由此我们可以对 Tensor Core 也建立一个粗糙的模型，当 N 大于某个阈值时，Tensor Core 会由访存受限转为计算受限。那么，这个阈值是怎么计算得到的呢？我们不妨简单猜测，当计算时间大于加载 A 和 B 矩阵的时间时，指令就会变为计算受限。于是对于 `wgmma.mma_async.shape.f32.f16.f16` 指令，以 H20 为例，我们可以根据下式计算该阈值：

64×N×16×2256×4≥64×16×2128+N×16×2128  \\dfrac{64 \\times N \\times 16 \\times 2}{256 \\times 4} ≥ \\dfrac{64 \\times 16 \\times 2}{128} + \\dfrac{N \\times 16 \\times 2}{128}

可得

7N≥64  7N ≥ 64

因此，在 H20 上，只有 N = 8 时，指令才是访存受限的。同时，基于上述分析，我们也可以将 Tensor Pipe 简单地以双缓冲模型进行建模，此处不多赘述。

### 实践指导

> 叽里咕噜说啥呢？实践中能用到吗？我直接往大了使用 N 不就行了？

由于本专栏为实用教程，怎么用确实值得说道说道。上述的理论分析最直接的应用是 GEMM 的 SwapAB 优化，实际应用中可见 [DeepGEMM PR #192](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepGEMM/pull/192)、 [TensorRT-LLM PR #4430](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/TensorRT-LLM/pull/4430)等。简单来说，当在 Hopper 系列 GPU 上使用 `wgmma` 计算 C\=ABT,A(m,k),B(n,k)C = AB^T, A(m,k), B(n,k) 时，如果 m<64m < 64，考虑到 `wgmma` 指令的 M 维度限死为 64，为避免浪费算力，一个很自然的优化思路是将 AA 和 BB 进行交换计算 C\=(BAT)TC = (B A^T)^T。

然而，上述的 PR 均只在 H20 上效果拔群，在 H100 上效果并不明显，甚至有时会变差。上述 PR 的作者感觉也不是很清楚为什么。下面我们以 `m=16, n=64, k=16` 的矩阵在一个 SM 上的计算进行分析。

由于 Tensor Core 访存受限时指令延迟变化较小，假定 N 取某个值 C 时 `wgmma` 指令由访存受限变为计算受限且延迟为 T，对 N<CN < C 的指令，我们同样认为其延迟为 T，对于 N\>CN > C 的指令，我们认为其延迟为 kTkT。例如，在 H20 上，`m64n16k16` 的指令延迟为 T，而 `m64n64k16` 的指令延迟为 4T4T。而在 H100 上，`m64n64k16` 的指令延迟为 T，`m64n16k16` 的指令延迟也视为 T。

在 H20 上计算 `m=16, n=64, k=16` 的 C\=ABTC = AB^T 有如下几种方式：

1.  使用 `wgmma.m64n64k16` 指令计算 C\=ABTC = AB^T，延迟为 4T4T
2.  使用 `wgmma.m64n16k16` 指令计算 C\=ABTC = AB^T，分别计算 `C[:,0:16], C[:,16:32], C[:,32:48], C[:,48:64]`，总延迟为 4T4T
3.  使用 `wgmma.m64n16k16` 指令计算 C\=(BAT)TC = (B A^T)^T，延迟为 TT

而在 H100 上计算 `m=16, n=64, k=16` 的 C\=ABTC = AB^T 同样有如下几种方式：

1.  使用 `wgmma.m64n64k16` 指令计算 C\=ABTC = AB^T，延迟为 TT
2.  使用 `wgmma.m64n16k16` 指令计算 C\=ABTC = AB^T，分别计算 `C[:,0:16], C[:,16:32], C[:,32:48], C[:,48:64]`，总延迟为 4T4T
3.  使用 `wgmma.m64n16k16` 指令计算 C\=(BAT)TC = (B A^T)^T，延迟为 TT

可以看到，在 H20 上，进行 SwapAB 优化后，延迟从 4T4T 降低到了 TT，而在 H100 上，即使不进行 SwapAB 优化，选取较大的 N 也能使得延迟为 T，因此 SwapAB 优化的收益并不明显。

### 总结

相信读者已经从本文了解到了如何通过指令延迟分析某种优化在特定硬件上可能带来的收益。笔者认为，能够针对具体硬件进行优化是算子开发者的核心竞争力之一，希望本文能为读者提供一些思路。