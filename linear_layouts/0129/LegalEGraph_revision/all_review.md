# Formal Literature Review Report — *LegalEGraph: Legality‑Aware Equality Saturation for Layout + Async Schedule Co‑Optimization*

## 1) Executive Summary (Verdict)

The proposal’s core premise is **factually grounded**: on Hopper‑class GPUs, peak performance paths (TMA + `cp.async.bulk.tensor`, `mbarrier` with tx‑count, and WGMMA) are gated by **hard, discrete legality and protocol constraints** that are *not* naturally expressed in conventional “layout algebra only” systems, nor reliably navigated by “generate schedules + cost model” alone. NVIDIA’s own specifications explicitly encode binary cliffs (alignment, rank bounds, finite swizzle enums, stride congruences, tx‑count/phase completion rules, and warpgroup uniformity requirements). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
**Novelty is plausible**: while equality saturation is established for tensor *graphs* (e.g., Tensat) and e‑class analyses exist (egg), applying EqSat to the *joint* space of (layout rewrites × async schedule rewrites) **under GPU‑specific legality/protocol analyses** looks meaningfully different from prior art and fills a real gap. ([ar5iv.org](https://ar5iv.org/abs/2101.01332))  
Feasibility is **medium‑to‑high if aggressively scoped** (e.g., 2D/3D TMA for GEMM/attention first, bounded rewrite sets, and fast lattice analyses); risk is mostly engineering and search‑space explosion, not “does the problem exist.”

---

## 2) Problem Verification — The “Hardware Reality” Audit (with doc‑level validation)

### 2.1 TMA legality cliffs (*cuTensorMap* / tensor map encoding)

Your proposal cites a cluster of TMA constraints as “binary legality cliffs.” The CUDA Driver API confirms these are **real and explicit** constraints for `cuTensorMapEncodeTiled` (and related encode calls):

**Verified constraints (CUDA Driver API v13.0.1):**

- **`tensorMap` address alignment:** must be **64‑byte aligned**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **`tensorRank` bounds:** must be **non‑zero and ≤ 5**; and if interleave is enabled, rank must be **≥ 3**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - Note: some specific APIs (e.g., im2col variants) further constrain rank (e.g., “must be at least 3”), but the *tiled* encoder’s general upper bound is clearly 5. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **`globalAddress` alignment:** must be **16‑byte aligned** (with stronger 32B constraints under some interleave/data type modes). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **`globalStrides` constraints:** strides (bytes) must be **multiple of 16** and **< 2^40** (with stronger multiple‑of‑32 constraints in some modes). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **`boxDim[i]` bounds:** each dimension’s traversal box must be **non‑zero and ≤ 256**; additionally, when `interleave` is none, `boxDim[0] * elementSize` must be a **multiple of 16 bytes**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **`elementStrides[i]` bounds:** each stride must be **non‑zero and ≤ 8** (and interacts with how many elements are loaded). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Finite swizzle mode set (and non‑classic variants):** swizzle is an enum including `128B_ATOM_32B`, `128B_ATOM_32B_FLIP_8B`, `128B_ATOM_64B`, etc.—i.e., *discrete modes*, not a continuous parameter. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Swizzle legality depends on inner dimension:** when `interleave` is none and swizzle is enabled, the **bounding box inner dimension (in bytes)** must be ≤ the swizzle span (32/64/128). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

**Critique: are these “hard cliffs,” or “just pad it”?**

- **Some are often “pad‑able”** (e.g., choosing tile sizes so that `boxDim[0]*elementSize` is a multiple of 16, or using leading dimensions that are multiples of alignment constraints). For contiguous matrices in DL frameworks, these constraints are frequently satisfiable without changing user‑visible layout.
- **But they are still cliffs** because:
  - They are **binary gating conditions** for the fast path: if violated, the TMA path is invalid or unavailable → you must **fallback** to non‑TMA code paths. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - Padding is not always free: it can require **extra memory traffic**, **extra allocation**, or **input copying**, and may be unacceptable for general compiler‑generated kernels where input tensors can be arbitrary views/strides.
  - Some constraints are **structural**, not cosmetic (e.g., `elementStrides ≤ 8`, rank limits, interleave‑swizzle coupling). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

### 2.2 TMA swizzle + alignment constraints (Programming Guide)

The CUDA Programming Guide’s async‑copy/TMA swizzle section corroborates that TMA swizzle legality is **not just “choose a swizzle”**; it has explicit validity conditions:

- When using TMA swizzle patterns, **global memory must be 128‑byte aligned**; shared memory alignment and inner‑dimension requirements are also specified; and swizzle mapping granularity is fixed at **16 bytes**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

This matters because it makes the legality space **discrete and alignment‑heavy**, and therefore well‑suited to “analysis lattice + pruning,” not “continuous tuning.”

### 2.3 `mbarrier` tx‑count / phase protocol (“temporal cliff”)

Your claim that `mbarrier` phase completion depends on both arrivals and tx‑count is **exactly stated** in PTX ISA 9.1:

- An `mbarrier` phase completes only when **(pending arrivals == 0) AND (tx‑count == 0)**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- PTX also imposes additional protocol structure, e.g., you must observe completion (via `test_wait` / `try_wait`) before proceeding to arrive in the subsequent phase. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- The existence of `mbarrier.expect_tx` and its role in managing tx‑count is explicitly defined. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

**Critique:** this is not “pad‑able.” It is a **temporal correctness protocol**, and violations can manifest as hangs, UB, or silent under‑synchronization. So the proposal is on strong ground in calling this a modern dominant failure mode. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### 2.4 Bulk async non‑ordering inside groups (ordering cliff)

PTX ISA 9.1 explicitly states:

- There is **no memory ordering guarantee** between any two `cp.async` operations within the same cp.async‑group; similarly for `cp.async.bulk` within a bulk async‑group. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source=openai))  
- In particular, two `cp.async` operations in the same group writing to the same location is **undefined behavior**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source=openai))  

This validates the proposal’s “SSA dependencies aren’t enough; schedule structure matters” framing.

### 2.5 WGMMA descriptor encodability + warpgroup uniformity

Two core claims are validated by PTX ISA 9.1:

- **Descriptor quantization:** `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4` (i.e., effectively **16‑byte granularity** plus masking). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Warpgroup uniformity requirement:** “The contents of a matrix descriptor must be same across all the warps in the warpgroup.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

This is an archetypal **binary legality cliff**: you cannot “mostly satisfy” warpgroup uniformity.

---

## 3) State‑of‑the‑Art Analysis (Competitors + gaps LegalEGraph would fill)

### 3.1 Triton (and the Triton ecosystem): what it does today

**What Triton demonstrably supports**
- Triton now exposes a **first‑class TMA descriptor API**: `triton.language.make_tensor_descriptor`. The docs state constraints that mirror the underlying legality cliffs:
  - base pointer must be **16‑byte aligned**,
  - leading dimensions must be **multiples of 16‑byte strides**,
  - last dimension must be **contiguous**,
  - currently supports **2–5D tensors**, and
  - on NVIDIA GPUs with TMA support, loads/stores are backed by **TMA hardware**. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
- The ecosystem documentation acknowledges that **descriptor construction is host‑side** and references CUDA’s `cuTensorMapEncode` flow. The PyTorch Hopper TMA deep dive explicitly describes Triton filling descriptors via `cuTensorMapEncode`‑based helpers and passing them to the device. ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai))  

**Where Triton appears to fall short relative to LegalEGraph (audit)**
- Triton’s model is still largely “**you write the kernel + choose parameters** (tile sizes, `num_stages`, warps), and the system compiles (or autotunes across a small discrete set).” The official API docs describe the legality conditions, but they do not describe an equality‑saturation‑style *global* search over layout equivalences × schedule equivalences under hardware legality. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
  - This is not a criticism—Triton is intentionally a programmable compiler—but it supports your claim that “legality‑aware search” is not the default story.
- Triton’s approach to “hard constraints” is primarily: **encode them at the API boundary** (descriptor construction) and/or rely on **kernel author discipline**. This matches your proposal’s gap thesis: legality is acknowledged, but not *systematically explored* as a search constraint.

**Bottom line:** Triton is a strong baseline for *implementing TMA* and proving it matters, but not yet a baseline for *legality‑aware EqSat co‑optimization.* ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  

### 3.2 TVM / Ansor / MetaSchedule: search + legality is “generate‑and‑test,” and Hopper‑specific features lag

**Evidence of gaps / limitations**
- TVM explicitly tracked Hopper TMA support; the tracking issue was ultimately **closed as “not planned.”** ([github.com](https://github.com/apache/tvm/issues/15956?utm_source=openai))  
  - That is a strong indicator that, as of the last documented status, TVM is not actively delivering TMA‑level legality + scheduling as a first‑class optimization target.
- MetaSchedule’s architecture (evolutionary search + postprocessing) is visibly “generate candidates then reject illegal ones”: a real‑world log shows `meta_schedule.VerifyGPUCode` producing **dozens of failures** per iteration, i.e., legality filtering is a measurable part of the pipeline. ([discuss.tvm.apache.org](https://discuss.tvm.apache.org/t/tvm-not-generating-legal-code-for-vulkan-on-windows/18591?utm_source=openai))  

**Interpretation vs your proposal**
- This supports your thesis that “cost model search alone” is insufficient: if legality constraints are tight, the search spends a lot of budget in the invalid region unless legality is represented as a **hard constraint early in the search** (exactly what e‑class analyses / pruning would do).
- However, MetaSchedule already has the *concept* of legality checks; what it lacks is the **domain‑specific legality lattice** for Hopper/Blackwell fast paths (TMA descriptor admissibility, WGMMA descriptor uniformity/encoding, `mbarrier` tx‑protocol), and a representation where **layout and async schedule are rewritten in one space**.

### 3.3 CUTLASS/CuTe: powerful abstractions, but primarily expert‑directed (not a verified optimizer)

**What CuTe provides (verified)**
- CuTe’s “TMA Tensors” documentation is explicit that TMA tensors are about mapping a logical coordinate to a **TMA coordinate space** (not a linear index), and it builds an algebraic layout story around that. ([docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  

**What it does *not* provide (gap)**
- CuTe (and CUTLASS more broadly) provides **building blocks** for experts and template authors; it is not an equality‑saturation optimizer that explores a large equivalence class of layouts/schedules and extracts the best legal one.
- In your proposal’s framing: CUTLASS/CuTe is “**manual correctness + manual schedule design**,” with strong abstractions, not automatic protocol verification or legality‑pruned search.

This validates the “expert‑only” positioning in a narrow sense: the tooling is designed for experts, even if many users consume the end results indirectly.

### 3.4 Equality Saturation: prior work exists, but mostly at graph level or without GPU legality

**egg (core infrastructure)**
- The egg paper explicitly introduces **e‑class analyses** as a general mechanism to integrate domain‑specific analyses into e‑graphs. This is the core conceptual lever your LegalEGraph proposal depends on. ([arxiv.org](https://arxiv.org/abs/2004.03082?utm_source=openai))  

**Tensat (EqSat for tensor graphs)**
- Tensat applies equality saturation to **tensor computation graph** rewriting and highlights two relevant ideas:
  1. EqSat avoids rewrite phase‑ordering traps, and  
  2. it includes mechanisms to handle **invalid subgraphs** and uses ILP‑style extraction. ([ar5iv.org](https://ar5iv.org/abs/2101.01332))  
- But it is **not** tackling low‑level GPU codegen legality (TMA descriptors, WGMMA descriptors, `mbarrier` tx‑protocol). So it is a *conceptual predecessor*, not a direct solution.

**Recent EqSat theory relevant to feasibility**
- “Optimism in Equality Saturation” (2025) directly targets limitations of pessimistic e‑class analysis on cyclic/SSA‑like programs—highly relevant if LegalEGraph tries to represent async scheduling constraints in SSA‑shaped IR. ([arxiv.org](https://arxiv.org/abs/2511.20782?utm_source=openai))  

**Other rewriting‑centric tensor IRs**
- Glenside shows layout‑aware rewriting via access patterns (term rewriting) for mapping to accelerators, but it is not Hopper‑specific legality/protocol reasoning. ([arxiv.org](https://arxiv.org/abs/2105.09377?utm_source=openai))  

**Gap LegalEGraph fills**
- The literature supports the following precise gap statement:

> We have EqSat infrastructure (egg) and tensor‑graph EqSat (Tensat), and we have layout formalisms (Linear Layouts, ISL relations) and library abstractions (CuTe), but we do not yet have an EqSat system that *jointly rewrites layouts and async schedules* while treating **GPU fast‑path legality/protocol constraints as first‑class pruning analyses.** ([arxiv.org](https://arxiv.org/abs/2004.03082?utm_source=openai))  

### 3.5 Layout formalisms (Linear Layouts / ISL relations): strong on spatial equivalence, weak on hardware legality + temporal protocol

**Linear Layouts (Zhou et al., arXiv 2025)**
- The paper is explicitly about a unified *spatial* representation (linear algebra over $$\mathbb{F}_2$$) integrated into Triton.  
- It also explicitly notes that many Triton/GPU layout parameters are **restricted to powers of two** (“dimensions … are restricted to powers of two”). ([ar5iv.org](https://ar5iv.org/html/2505.23819v3))  
- Importantly for your proposal: nothing in the core pitch is about encoding **cuTensorMap admissibility** or **`mbarrier` tx‑count protocols** as first‑class constraints—those live “below” the algebra, in the ISA/API contract. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

**ISL relations for layouts (Bhaskaracharya et al., arXiv 2025)**
- The abstract positions the contribution as a unified mathematical representation enabling “rigorous formal analysis, correctness verification, and the foundation for future cross‑system optimization strategies.” ([arxiv.org](https://arxiv.org/abs/2511.10374))  
- That’s compatible with your claim that it’s more “foundational/canonicalization” than “GPU legality optimizer,” and it does not claim to encode Hopper‑specific descriptor legality or async protocol constraints.

**Conclusion on novelty vs seed papers**
- Your proposal’s novelty claim that these papers do not solve “descriptor legality + async schedule protocol co‑optimization” is **consistent with what the papers claim to do** (spatial formalization and/or cross‑system modeling), and with what NVIDIA’s legality constraints actually look like (finite enums + alignment + protocol automata). ([arxiv.org](https://arxiv.org/abs/2511.10374))  

---

## 4) Bibliography (10–15 key references with links)

Below are the most load‑bearing references for this proposal, prioritized toward **primary sources** (NVIDIA docs + major papers):

1. **NVIDIA — PTX ISA 9.1 (Parallel Thread Execution)** (mbarrier tx‑count/phase completion, bulk async ordering, WGMMA descriptor format & rules).  
   `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html` ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

2. **NVIDIA — CUDA Driver API v13.0.1, Tensor Map Object Management (`cuTensorMapEncodeTiled`)** (rank/alignment/stride/boxDim/elementStrides/swizzle legality).  
   `https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html` ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

3. **NVIDIA — CUDA C++ Programming Guide v13.1.0, Asynchronous Data Copies (TMA swizzle requirements)**.  
   `https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html` ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

4. **Triton Documentation — `triton.language.make_tensor_descriptor`** (documented legality constraints and TMA lowering behavior).  
   `https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html` ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  

5. **PyTorch Blog — “Deep Dive on the Hopper TMA Unit for FP8 GEMMs”** (practical description of Triton TMA descriptor creation + generated PTX).  
   `https://pytorch.org/blog/hopper-tma-unit/` ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai))  

6. **MLIR — NVGPU Dialect Documentation** (models `tma.create.descriptor`, `tma.async.load`, `warpgroup.mma`, mbarrier ops/types).  
   `https://mlir.llvm.org/docs/Dialects/NVGPU/` ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

7. **Willsey et al. — “egg: Fast and Extensible Equality Saturation”** (e‑class analyses as the enabling mechanism).  
   `https://arxiv.org/abs/2004.03082` ([arxiv.org](https://arxiv.org/abs/2004.03082?utm_source=openai))  

8. **POPL 2021 artifact page — “egg: Fast and Extensible Equality Saturation”** (canonical venue link + DOI).  
   `https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation` ([popl21.sigplan.org](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai))  

9. **Yang et al. — “Equality Saturation for Tensor Graph Superoptimization (Tensat)”** (EqSat precedent + invalid‑subgraph filtering + extraction).  
   `https://ar5iv.org/abs/2101.01332` ([ar5iv.org](https://ar5iv.org/abs/2101.01332))  

10. **Shah et al. — “FlashAttention‑3: Fast and Accurate Attention with Asynchrony and Low‑precision”** (motivating evidence that peak Hopper kernels are protocol‑heavy TMA pipelines).  
   `https://arxiv.org/abs/2407.08608` ([arxiv.org](https://arxiv.org/abs/2407.08608?utm_source=openai))  

11. **Zhou et al. — “Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $$\mathbb{F}_2$$”** (spatial layout formalism; power‑of‑two constraint reality).  
   `https://ar5iv.labs.arxiv.org/html/2505.23819v3` ([ar5iv.org](https://ar5iv.org/html/2505.23819v3))  

12. **Bhaskaracharya et al. — “Modeling Layout Abstractions Using Integer Set Relations”** (ISL canonicalization / modeling foundation).  
   `https://arxiv.org/abs/2511.10374` ([arxiv.org](https://arxiv.org/abs/2511.10374))  

13. **NVIDIA CUTLASS / CuTe docs — “CuTe TMA Tensors”** (manual abstraction baseline).  
   `https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html` ([docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  

14. **Apache TVM — Hopper TMA tracking issue (status + scope)** (evidence of ecosystem gap).  
   `https://github.com/apache/tvm/issues/15956` ([github.com](https://github.com/apache/tvm/issues/15956?utm_source=openai))  

15. **Arbore, Cheung, Willsey — “Optimism in Equality Saturation” (2025)** (relevance to SSA/cyclic analyses if schedules/tokens create cycles).  
   `https://arxiv.org/abs/2511.20782` ([arxiv.org](https://arxiv.org/abs/2511.20782?utm_source=openai))  

---

### Final feasibility note (actionable audit insight)

If you want this proposal to read as maximally credible to compiler + GPU reviewers, anchor the scope explicitly in what the docs force you to do:

- Start with **`cuTensorMapEncodeTiled` legality lattice** (rank/align/stride/boxDim/elementStrides/swizzle) as an e‑class analysis. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- Then add **temporal legality** as a second analysis layer: `mbarrier` phase completion + tx‑count conservation, and bulk‑group non‑ordering constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- Treat WGMMA descriptor uniformity and encoding quantization as a third legality layer (warpgroup‑uniform descriptors, 16B quantization). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

That staged story matches the true hardware contracts and makes “legality‑aware equality saturation” look like the *right tool*, not just a clever hammer.

---
Learn more:
1. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
2. [\[2101.01332\] 1 Introduction](https://ar5iv.org/abs/2101.01332)
3. [4.11. Asynchronous Data Copies — CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai)
4. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
5. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source=openai)
6. [triton.language.make\_tensor\_descriptor — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai)
7. [Deep Dive on the Hopper TMA Unit for FP8 GEMMs – PyTorch](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai)
8. [\[Tracking Issue\] Hopper TMA support for bulk asynchronous loads · Issue #15956 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/15956?utm_source=openai)
9. [TVM not generating legal code for Vulkan on Windows - Questions - Apache TVM Discuss](https://discuss.tvm.apache.org/t/tvm-not-generating-legal-code-for-vulkan-on-windows/18591?utm_source=openai)
10. [CuTe TMA Tensors — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai)
11. [egg: Fast and Extensible Equality Saturation](https://arxiv.org/abs/2004.03082?utm_source=openai)
12. [Optimism in Equality Saturation](https://arxiv.org/abs/2511.20782?utm_source=openai)
13. [Pure Tensor Program Rewriting via Access Patterns (Representation Pearl)](https://arxiv.org/abs/2105.09377?utm_source=openai)
14. [\[2505.23819\] Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://ar5iv.org/html/2505.23819v3)
15. [\[2511.10374\] Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/abs/2511.10374)
16. ['nvgpu' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)
17. [egg: Fast and Extensible Equality Saturation (POPL 2021 - POPL Research Papers) - POPL 2021](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai)
18. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608?utm_source=openai)
