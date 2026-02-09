## 1) Executive Summary (Viability + Novelty Verdict)

The proposal’s core diagnosis—**“legal” is now the dominant optimization constraint on Hopper/Blackwell, not merely “fast”**—is strongly supported by current NVIDIA documentation: TMA tensor maps have *hard admissibility bounds* (rank/stride/box/alignment), bulk async groups have *explicit non-ordering*, and `mbarrier` phase completion is *tx-count sensitive* in a way that can deadlock if mishandled. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) In the competitive landscape, today’s production systems largely handle this via (a) **expert libraries (CUTLASS/CuTe)** that expose abstractions but do not globally verify/synthesize legality, ([docs.nvidia.com](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai)) (b) **Triton** which increasingly exposes TMA/barrier primitives but still exhibits real “protocol cliff” failure modes (e.g., hangs) and relies on tuning rather than legality-by-construction across arbitrary rewrites, ([github.com](https://github.com/triton-lang/triton/issues/6354?utm_source=openai)) and (c) **TVM/Ansor/MetaSchedule** which excel at cost-model + search over *legal schedule primitives*, but do not natively model TMA tensor-map admissibility or `mbarrier` tx-count protocols as first-class, target-versioned *hard constraints* inside the search itself. ([usenix.org](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai)) Against prior equality-saturation work (Tensat/Glenside/MLIR eqsat/DialEgg), LegalEGraph’s novelty is credible: those systems apply rewriting mostly at **graph-level** or in **dialect-agnostic** settings, not at the level where **PTX legality predicates + async protocols** (tx-count/phase discipline) must be preserved. ([arxiv.org](https://arxiv.org/abs/2101.01332?utm_source=openai)) The feasibility risk is not correctness (you can always post-verify), but **search blow-up and dynamic-shape legality**; the proposal is implementable if it narrows scope initially (e.g., SM90 TMA-to-SMEM pipelines + a restricted schedule language) and treats legality as “hard pruning + exact post-check,” as written. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))

---

## 2) Problem Verification (“Hardware Reality” Audit)

### 2.1 TMA / `CUtensorMap` admissibility: the constraints are real and hard

The proposal’s key driver-API constraints for **tiled** tensor maps are accurate in current CUDA Driver docs (CUDA Toolkit **13.1.1**, last updated **Jan 12, 2026**). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) Concretely:

- **Rank bound:** `tensorRank` must be non-zero and **≤ 5** (and if `interleave != NONE`, rank must be **≥ 3**). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Descriptor alignment:** `tensorMap` address must be **64B aligned**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Base address alignment:** `globalAddress` must be **16B aligned** (and **32B** under certain `interleave`/packed-type conditions). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Stride modular/bound constraints:** `globalStrides` must be a **multiple of 16** and **< 2^40** (plus extra 32B-multiple conditions in some modes/types). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Box bounds:** `boxDim[i]` must be non-zero and **≤ 256**, with extra constraints (e.g., `boxDim[0]*elementSize` multiple of 16 bytes under `interleave==NONE`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Element-stride bounds:** `elementStrides[i]` must be non-zero and **≤ 8**; and critically, when `interleave==NONE`, **dimension-0 stride is ignored** (“TMA doesn’t support the stride for dimension zero”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

**Why this is a “binary cliff” in practice:** these are not “performance suggestions”; they are explicit *requirements* for descriptor encoding. Violations do not merely slow down—they make the fast path unavailable (or potentially invalid/undefined depending on where the violation occurs). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

**Can they be padded away?** Some, sometimes:
- Alignment and stride multiples can often be achieved by padding leading dimensions or choosing aligned allocations—but padding is not free: it changes memory footprint, changes vectorization, and interacts with boundary tiles/masking. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- Hard bounds like **rank ≤ 5**, **boxDim ≤ 256**, and **elementStrides ≤ 8** are structural and can’t be “padded away” without changing the algorithmic decomposition (multiple TMA ops, different tiling, or fallback). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

This supports the proposal’s premise that *layout equivalence* (indexing semantics) is insufficient: legality is a separate, discrete predicate.

---

### 2.2 TMA swizzle legality (CUDA Programming Guide): also real, and sometimes “instruction invalid”

The proposal’s claims about swizzle being finite and gated by alignment/inner-dimension rules are confirmed in the CUDA Programming Guide section on TMA swizzling:

- **Only four swizzle modes** (none/32B/64B/128B). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- **Global memory alignment requirement: 128 bytes.** ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- **Inner dimension requirements; otherwise “the instruction is considered invalid.”** ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- **Granularity fixed at 16 bytes.** ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- Additionally: “When using TMA, the shared memory is required to be aligned to **128 bytes**.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

This directly validates the proposal’s “binary legality cliff” framing for swizzled TMA paths. If the compiler extracts a “logically equivalent” layout whose inner dimension violates the table constraints, the PTX instruction is invalid—there is no smooth degradation. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))

---

### 2.3 Bulk async non-ordering: explicitly stated in PTX, and not optional

Your proposal’s statement that **bulk async groups provide no ordering guarantee within the group** is explicitly in PTX ISA 9.1:

> “There is no memory ordering guarantee provided between any two `cp{.reduce}.async.bulk.{.prefetch}{.tensor}` operations within the same bulk async-group.” ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

This is exactly the kind of fact that breaks “SSA-dependence-only” reasoning: a compiler rewrite that reorders two bulk copies within a group (or assumes their order) can introduce silent correctness bugs unless explicit waits/barriers are maintained. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

---

### 2.4 `mbarrier` tx-count and phase completion: deadlock cliffs are real

The proposal’s most important “temporal cliff” claim is also explicitly stated in PTX ISA 9.1:

- Starting with Hopper (`sm_9x`), `mbarrier` tracks **tx-count** for async transactions. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- `expect-tx` increments tx-count; `complete-tx` decrements it. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- **Phase completion requires BOTH:**
  1) pending arrivals reach zero, **and**  
  2) tx-count reaches zero. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

This validates the deadlock/bubble risk: if the compiler’s tx accounting is wrong (under/over), you can block phase completion or introduce unnecessary waits. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

Also, the PTX 9.1 doc is very explicit about **lifecycle UB**:

- Doing any `mbarrier` operation except `init` on an uninitialized mbarrier is UB.  
- Doing any non-`mbarrier` op (or re-`init`) on initialized storage is UB. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

That makes “protocol verification” (as your proposal emphasizes) a legitimate compiler correctness problem, not just a performance tweak. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))

Finally, CUDA’s higher-level programming guide confirms the semantics in more “CUDA programmer” terms: a bulk async copy updates the barrier’s transaction count; `cuda::memcpy_async` handles `mbarrier.expect_tx`; and the barrier flips only when **all threads arrived and all bytes arrived**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html?utm_source=openai)) This reinforces the correctness stakes and shows the ecosystem already treats tx-count as a first-class protocol, not an incidental detail. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html?utm_source=openai))  

---

### 2.5 WGMMA descriptor encodability: quantization is real; “invalid bit patterns” needs a nuance

The proposal is correct that PTX defines a **64-bit descriptor** with quantized offsets:

- PTX provides `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4`, i.e., **16B granularity**. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- The swizzle mode in the **WGMMA matrix descriptor format** is a small enumerated field (No/128B/64B/32B). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- PTX states: **“The contents of a matrix descriptor must be same across all the warps in the warpgroup.”** ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

So the “quantization cliff” and warpgroup-uniformity claims are verified. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

**Important corrective nuance:** your proposal text claims the WGMMA matrix descriptor has “invalid bit patterns.” In PTX 9.1, the *WGMMA matrix descriptor* swizzle field shown in “Matrix Descriptor Format” is **2 bits with four defined modes**, so *that field alone* is not presented as having invalid patterns. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
However, PTX 9.1 also describes **tcgen05-family matrix descriptors** (separate section) whose swizzle encoding has **explicitly invalid values** (“Values 3, 5 and 7 are invalid”). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

So: **the “invalid bit patterns” claim is correct for tcgen05 descriptors** and should be *scoped accordingly*, but it is slightly sloppy if attributed to WGMMA’s matrix descriptor format. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

---

## 3) State-of-the-Art Analysis (Competitors / Baselines)

### 3.1 Triton (compiler reality in 2024–2026): increasingly powerful, but not legality-saturating

**Evidence of real TMA/`mbarrier` complexity in practice:**
- Triton users have hit *hangs* when “loading from TMA descriptor,” consistent with tx-count / barrier protocol cliffs. ([github.com](https://github.com/triton-lang/triton/issues/6354?utm_source=openai))  
- A PyTorch blog post (“Deep Dive on the Hopper TMA Unit for FP8 GEMMs”) shows Triton generating PTX containing `cp.async.bulk.tensor.*.mbarrier::complete_tx::bytes` and emphasizes extensive tuning (e.g., “TMA_SIZE from 128 to 512” materially affects throughput). ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai))  

This suggests Triton’s current mode is: **expose the primitives and let experts (or autotuning) manage the protocol**, rather than “compiler proves protocol invariant under rewrites.” ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai))  

**Does Triton search layouts/schedules automatically?**  
Triton’s public-facing mechanism is still primarily **parameter search/autotune** over user-specified configurations and kernel variants (as seen in the PyTorch blog’s “tuned extensively” narrative), rather than an internal equality-saturation engine that rewrites layouts/schedules under hard legality analyses. ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai))  

**Triton has NVIDIA-specific scheduling/async constructs—but not the proposal’s global guarantee.**  
The Triton NVIDIA GPU dialect includes ops like `ttng.tc_gen5_commit` that make an `mbarrier` track completion of async tcgen5 ops, and it states ordering guarantees about completion mechanisms “in the order the commit operations were issued.” ([triton-lang.org](https://triton-lang.org/main/dialects/TritonNvidiaGPUOps.html?utm_source=openai)) This is *useful infrastructure*, but it is not the same as **e-graph-based exploration + extraction under legality constraints**.

**Bottom line:** Triton is a strong baseline for code generation and has real-world evidence of the protocol minefield, but it does not yet look like it is solving the *global legality + schedule co-optimization* problem in the way LegalEGraph proposes. ([github.com](https://github.com/triton-lang/triton/issues/6354?utm_source=openai))  

---

### 3.2 TVM / Ansor / MetaSchedule: strong at search + async modeling, weaker at *descriptor-driven legality* (TMA/WGMMA)

**Ansor (and the TVM lineage) is cost-model + search driven.**  
Ansor explicitly explores a large optimization space via hierarchical sampling + evolutionary search + learned cost model. ([usenix.org](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai)) This is orthogonal to descriptor admissibility protocols: Ansor assumes the “schedule space” is meaningful and then ranks candidates. ([usenix.org](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai))  

**TVM has explicit async queue semantics (commit/wait analogs).**  
TVM’s TIR attributes define an “async commit queue” model explicitly analogized to PTX `commit_group` and `wait_group` and FIFO completion semantics. ([tvm.apache.org](https://tvm.apache.org/docs/reference/api/doxygen/namespacetvm_1_1tir_1_1attr.html?utm_source=openai)) This is a meaningful competitor to the *schedule-side representation* in your proposal.

**TVM also exposes PTX async intrinsics**, including `ptx_cp_async_bulk`. ([tvm.apache.org](https://tvm.apache.org/docs/v0.16.0/reference/api/python/tir.html?utm_source=openai))  

**But: TVM’s surfaced intrinsics are not (yet) the same as Hopper TMA’s `cp.async.bulk.tensor` path.**  
The TVM `ptx_cp_async_bulk` signature is pointer-based (global_ptr/shared_ptr) and does not incorporate **tensor-map descriptors** (`CUtensorMap`), which are required for TMA tensor copies. ([tvm.apache.org](https://tvm.apache.org/docs/v0.16.0/reference/api/python/tir.html?utm_source=openai)) In PTX, the TMA tensor form consumes `[tensorMap, tensorCoords]`, and has distinct completion semantics (including `complete_tx::bytes` behavior). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  

**Interpretation:** TVM/Ansor/MetaSchedule have real infrastructure for exploring schedules and representing async operations, but they do not currently appear to have a first-class legality domain for **(TMA tensor-map admissibility + WGMMA descriptor encodability + `mbarrier` tx-count protocol)** comparable to what LegalEGraph proposes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

---

### 3.3 CUTLASS/CuTe: expert abstractions, not global synthesis/verification

CuTe’s TMA documentation is explicit about the core reality your proposal highlights:

- A TMA instruction consumes a **TMA descriptor** representing a tensor in global memory with **1–5 dimensions**, including size/stride/smem box/swizzle/OOB behavior. ([docs.nvidia.com](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  
- The descriptor **must be created on the host** before kernel execution and is then shared across CTAs; the instruction consumes descriptor pointer + SMEM pointer + coordinates. ([docs.nvidia.com](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  

This supports your “descriptor-driven legality” framing: the hardware is not taking raw pointers and doing arbitrary address arithmetic; it is consuming a structured, constrained descriptor. ([docs.nvidia.com](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  

**However:** CuTe is primarily an *abstraction layer for experts* to express these descriptors and coordinate transformations; it is not (as documented) a legality-saturating compiler pass that searches across alternative layouts/schedules and proves tx-count/phase correctness under arbitrary rewrites. ([docs.nvidia.com](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  

That leaves room for your proposed contribution.

---

### 3.4 Equality saturation ecosystem: strong infrastructure, but mostly not at PTX legality/protocol level

**Tensat (MLSys 2021)** uses equality saturation for **tensor graph** superoptimization (choosing rewrite sequences for computational graphs). ([arxiv.org](https://arxiv.org/abs/2101.01332?utm_source=openai)) That is meaningfully different from LegalEGraph’s target, which is the **low-level kernel layout + async schedule** space where PTX legality constraints can invalidate an implementation. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

**Glenside (2021)** proposes a rewriting-friendly IR (“access patterns”) to enable low-level, layout-aware rewrites and accelerator mapping. ([arxiv.org](https://arxiv.org/abs/2105.09377?utm_source=openai)) It is close in spirit (rewrite-based tensor program optimization), but it predates Hopper TMA/`mbarrier::expect_tx` realities and does not, as described in the abstract, target *PTX 9.x protocol legality* (tx-count/phase completion). ([arxiv.org](https://arxiv.org/abs/2105.09377?utm_source=openai))  

**MLIR eqsat dialect (2025)** and **DialEgg (CGO 2025)** are enabling infrastructure: they integrate equality saturation into MLIR, dialect-agnostically. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai)) They do not claim GPU-specific legality modeling; LegalEGraph can be framed as a *domain instantiation* of this infrastructure with a new analysis lattice keyed to TMA/WGMMA/mbarrier semantics. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  

---

## 4) Theoretical & Novelty Assessment (Does the proposal add real theory?)

### 4.1 Seed layout papers do not “already solve” descriptor legality or async protocol safety

**Linear Layouts (ASPLOS 2026)**:  
It provides a strong backend representation and conversion story for many layouts, but it explicitly states limitations (power-of-two shapes; slicing/flipping not expressible without affine extension). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) Nothing in that limitation framing suggests it is encoding *driver/PTX descriptor admissibility constraints* like `tensorRank ≤ 5`, `boxDim ≤ 256`, tx-count protocols, etc.—those live in CUDA Driver and PTX specs, not in the layout algebra itself. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**ISL relations paper (2025)**:  
It explicitly says its goal is **not performance optimization** but formal groundwork for reasoning about layout abstractions. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) It also notes representability constraints (if an index mapping isn’t strictly affine, “no straightforward layout representation exists for the given shape…”). ([arxiv.org](https://arxiv.org/html/2511.10374v1)) Again, the paper’s own positioning indicates it is not solving the *hardware legality* layer.

So your proposal’s “seed papers lack legality awareness” claim is **substantiated** as a matter of scope and stated goals, not as a criticism of those works. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

### 4.2 Is “TxGraph-style temporal verification inside e-graphs” novel?

What is *not novel* (in the abstract):
- Using an analysis lattice (or typestate-like constraints) to filter candidates.
- Using equality saturation plus extraction under costs.
- Using IR tokens to represent async dependencies.

What *does* look novel in your concrete domain:
- The PTX 9.x `mbarrier` semantics introduce **tx-count** as a correctness-critical state component, and phase completion depends on tx-count reaching zero. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)) This is not covered by traditional instruction scheduling legality checks.  
- Bulk async-group non-ordering is explicitly stated for `cp.async.bulk.*` and must be respected by rewrites. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- The combination of (a) **descriptor admissibility** (driver API) and (b) **temporal protocol legality** (PTX) as *hard constraints* inside a rewrite search is not something I found already “productized” in the existing eqsat-for-tensors literature (Tensat/Glenside/MLIR eqsat/DialEgg). ([arxiv.org](https://arxiv.org/abs/2101.01332?utm_source=openai))  

So the novelty claim is plausible: **LegalEGraph is best viewed as a new domain instantiation of equality saturation where e-class analyses encode GPU ISA legality, not only algebraic equivalences.** ([azizzayed.com](https://azizzayed.com/publications/dialegg/?utm_source=openai))  

### 4.3 Feasibility audit (where this can fail, concretely)

1. **E-graph blow-up is a first-order risk**  
   Combining layout algebra rewrites with schedule rewrites can explode. Your mitigation (“bounded saturation,” legality pruning, ISL canonicalization) is directionally correct, but the success condition is: legality analyses must be *cheap enough* to run per-iteration, and *strong enough* to prune early. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

2. **Dynamic shapes and pointer alignment**  
   Many legality constraints involve runtime values (pointer alignment after slicing, dynamic strides). The CUDA PG swizzle section and driver constraints are strict; if your IR allows arbitrary subviews, your legality lattice must become symbolic (congruences) or you need runtime guards/fallbacks. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

3. **Tx-count computation under masking/OOB is genuinely hard**  
   PTX defines tx-count in “units specified by the async memory operation.” ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)) If your transformation changes which bytes are actually transferred (masking, OOB fill, partial tiles), conservative accounting may be required, and too much conservatism destroys performance (over-waiting). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

4. **Version drift (PTX/CUDA)**  
   Your plan to version legality by (PTX version, SM target) is necessary: these semantics do evolve (e.g., PTX 9.1 adds new bulk/multimem ops; tx-count is “starting with Hopper”). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

---

## 5) Bibliography (10–15 key references with links)

Below are the most “load-bearing” references for your proposal; I’m listing the link in inline code (per your formatting constraints) and also citing the crawled source.

1. **PTX ISA 9.1 (mbarrier tx-count + bulk async semantics + WGMMA descriptors)**  
   `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html` ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

2. **CUDA Driver API Tensor Map Object Management (CUtensorMap / cuTensorMapEncodeTiled constraints)** (CUDA Toolkit 13.1.1, updated Jan 12, 2026)  
   `https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html` ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

3. **CUDA Programming Guide: TMA swizzle modes + 128B alignment + “instruction invalid”**  
   `https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html` ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

4. **CUDA Programming Guide (barrier tx-count narrative; `cuda::memcpy_async` / barrier flips when bytes arrive)**  
   `https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html` ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html?utm_source=openai))  

5. **MLIR NVGPU dialect (tokenized mbarrier ops; expect_tx; bridge to PTX)**  
   `https://mlir.llvm.org/docs/Dialects/NVGPU/` ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

6. **CUTLASS/CuTe TMA tensors documentation (host-created descriptor, coordinate-driven semantics)**  
   `https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html` ([docs.nvidia.com](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  

7. **Triton + TMA in practice (PyTorch blog deep dive; shows Triton emits `cp.async.bulk.tensor` and needs tuning)**  
   `https://pytorch.org/blog/hopper-tma-unit/` ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai))  

8. **Triton evidence of protocol cliffs (hang in TMA descriptor load issue)**  
   `https://github.com/triton-lang/triton/issues/6354` ([github.com](https://github.com/triton-lang/triton/issues/6354?utm_source=openai))  

9. **Triton NVIDIA GPU dialect docs (tcgen5 commit/mbarrier integration)**  
   `https://triton-lang.org/main/dialects/TritonNvidiaGPUOps.html` ([triton-lang.org](https://triton-lang.org/main/dialects/TritonNvidiaGPUOps.html?utm_source=openai))  

10. **Ansor (OSDI 2020): cost-model + evolutionary search baseline framing**  
   `https://www.usenix.org/conference/osdi20/presentation/zheng` ([usenix.org](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai))  

11. **TVM TIR async queue semantics (commit/wait modeling) + PTX async intrinsics**  
   `https://tvm.apache.org/docs/reference/api/doxygen/namespacetvm_1_1tir_1_1attr.html` ([tvm.apache.org](https://tvm.apache.org/docs/reference/api/doxygen/namespacetvm_1_1tir_1_1attr.html?utm_source=openai))  
   `https://tvm.apache.org/docs/v0.16.0/reference/api/python/tir.html` ([tvm.apache.org](https://tvm.apache.org/docs/v0.16.0/reference/api/python/tir.html?utm_source=openai))  

12. **FlashAttention-3 (Hopper speedups framed around TMA + asynchrony)**  
   `https://arxiv.org/abs/2407.08608` ([arxiv.org](https://arxiv.org/abs/2407.08608?utm_source=openai))  

13. **Linear Layouts (ASPLOS 2026): seed layout formalism + explicit limitation**  
   `https://arxiv.org/html/2505.23819v3` ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

14. **ISL relations for layouts (2025): explicit “not performance optimizations” positioning**  
   `https://arxiv.org/html/2511.10374v1` ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

15. **Equality saturation infrastructure & adjacent work**  
   - Tensat (MLSys 2021): `https://arxiv.org/abs/2101.01332` ([arxiv.org](https://arxrg/abs/2101.01332?utm_source=openai))  
   - MLIR eqsat dialect (2025): `https://arxiv.org/abs/2505.09363` ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
   - DialEgg (CGO 2025): `https://azizzayed.com/publications/dialegg/` ([azizzayed.com](https://azizzayed.com/publications/dialegg/?utm_source=openai))  
   - Glenside (2021): `https://arxiv.org/abs/2105.09377` ([arxiv.org](https://arxiv.org/abs/2105.09377?utm_source=openai))  

---

### “If I were reviewing this proposal” (one bluentence)

You have a real, well-documented problem and a plausible novel angle; the proposal will become significantly stronger if you (1) precisely scope which descriptor formats have invalid swizzle bit patterns (tcgen05 vs WGMMA), ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)) and (2) specify an explicit plan for dynamic-shape legality (symbolic congruences + runtime guards) so that “never emit illegal fast-path code” remains defensible outside fully static kerne([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))

---
Learn more:
1. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
2. [CuTe TMA Tensors — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai)
3. [Loading from TMA descriptor hangs · Issue #6354 · triton-lang/triton · GitHub](https://github.com/triton-laiton/issues/6354?utm_source=openai)
4. [Ansor: Generating High-Performance Tensor Programs for Deep Learning | USENIX](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai)
5. [Equality Saturation for Tensor Graph Superoptimization](https://arxiv.org/abs/2101.01332?utm_source=openai)
6. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)
7. [4.11. Asynchronous Data Copies — CUDA Programming Guide](https://docs.nvidia.coda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai)
8. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html?utm_source=openai)
9. [Deep Dive on the Hopper TMA Unit for FP8 GEMMs – PyTorch](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai)
10. [TritonNvidiaGPUOps — Triton documentation](https://triton-lang.org/main/dialects/TritonNvidiaGPUOps.html?utm_source=openai)
11. [tvm: tvm::tir::attr Namee Reference](https://tvm.apache.org/docs/reference/api/doxygen/namespacetvm_1_1tir_1_1attr.html?utm_source=openai)
12. [tvm.tir — tvm 0.17.dev0 documentation](https://tvm.apache.org/docs/v0.16.0/reference/api/python/tir.html?utm_source=openai)
13. [PTX ISA :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai)
14. [Pure Tensor Program Rewriting via Access Patterns (Representation Pearl)](https://arxiv.org/abs/2105.09377?utm_sourcepenai)
15. [eqsat: An Equality Saturation Dialect for Non-destructive Rewriting](https://arxiv.org/abs/2505.09363?utm_source=openai)
16. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
17. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
18. [DialEgg: Dialect-Agnostic MLIR Optimizer using Equality Saturation with Egglog - CGO 2025 | Aziz Zayed](https://azizzayed.com/publications/dialegg/?utsource=openai)
19. ['nvgpu' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)
20. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608?utm_source=openai)
