## Direction 1 — **TxGraph**: Typestate‑Verified Async Pipelines for Hopper TMA + `mbarrier`

### 1) Gap (validated + re‑verified)
**What the seed layout formalisms don’t cover is now the dominant failure mode:** on Hopper, getting peak kernels is less “express a clever swizzle” and more “satisfy a *protocol* with UB + deadlock cliffs.”

- **Temporal cliff (hard):** PTX `mbarrier` phases only complete when **(pending arrivals == 0) AND (tx‑count == 0)**, and Hopper introduces tx‑count tracking explicitly. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  This means “my layout is correct” is insufficient: you can have correct indexing but hang (bad tx accounting) or underfill (conservative waits).

- **Bulk async ordering cliff:** PTX states **no ordering guarantee** among ops inside the same `cp.async.bulk` group. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  So even if SSA dependencies look “fine,” you can still create incorrect overlaps without explicit wait/barrier structure.

- **Legality cliff (binary):** TMA requires host‑created tensor maps with strict bounds (e.g., `tensorMap` 64B aligned, `globalAddress` 16B aligned, `globalStrides` multiple of 16 and < 2^40, `boxDim[i] <= 256`, `elementStrides[i] <= 8`, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  Swizzle is *finite* and includes additional modes beyond the classic 32/64/128 (e.g., `*_ATOM_*` variants). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
  Separately, the Programming Guide’s TMA swizzle section is explicit about **128B alignment**, **inner‑dimension validity**, and **16B granularity**. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))

- **Compute consumption cliff:** WGMMA shared‑memory descriptors quantize offsets via `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4` and expose a small enumerated swizzle field; descriptors must be identical across warps in the warpgroup. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**Why this is timely (recent literature):** FlashAttention‑3 explicitly frames Hopper speedups around **TMA + asynchrony** (warp specialization, overlap, interleaving compute/softmax). ([arxiv.org](https://arxiv.org/abs/2407.08608?utm_source=openai))  
That is: the best kernels already look like protocol‑driven async pipelines, not just static layouts.

**Novelty check (3 closest systems/papers and what we do differently):**
1. **FlashAttention‑3**: demonstrates Hopper performance via hand‑built TMA+async pipelines, but does *not* provide compiler‑level protocol verification or schedule synthesis. ([arxiv.org](https://arxiv.org/abs/2407.08608?utm_source=openai))  
2. **CUTLASS/CuTe TMA + pipeline abstractions**: exposes TMA descriptor semantics and barrier/pipeline helpers, but correctness is still largely “by construction + discipline,” not verified across arbitrary compiler rewrites. ([docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  
3. **MLIR NVGPU dialect**: already models `mbarrier` and async copy ops with SSA token types, but it stops at representation; it doesn’t enforce the full tx‑count/phase protocol nor synthesize safe schedules. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

---

### 2) Theory (from Stage‑1.5 toolbox)
**Named theory:** *effect/typestate for async transactions* + *linear capabilities (tokens)*.

- Treat each “in‑flight async transfer” as a **linear resource** that must be consumed exactly once to advance barrier phase/visibility.  
- Optional second lens for debugging subtle order/visibility: axiomatic semantics + bounded checking, leveraging PTX‑model work as inspiration. ([research.nvidia.com](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai))  

---

### 3) Artifact (new compiler/runtime mechanism)
**TxGraph = a verified “temporal + legal layout” layer in Triton→MLIR→PTX.**

#### IR design
Add an MLIR extension layer (either a small new dialect or a disciplined subset on top of NVGPU) with:

- **Transaction tokens:** `!tx.bytes<N>` (static) or `!tx.bytes` (dynamic) returned by `tma.async_load` / `cp_async_bulk_tensor`.  
- **Barrier phase tokens:** `!mbar.phase<P>` for parity/phase tracking and reuse control.
- **Proof‑carrying descriptors:** `!tma.desc<rank, swizzle, align, stride_mod, box_bounds>` as a *type refinement* on a tensormap handle.

This builds directly on NVGPU’s existing token modeling (`!nvgpu.device.async.token`, `!nvgpu.mbarrier.token`, tensormap descriptor types/attrs). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

#### Algorithm (compiler pass pipeline)
1. **Legality inference pass (spatial):** propagate refinements (alignment, stride mod, box bounds) and check them against driver/PTX constraints; emit either (a) a certified TMA path, or (b) a diagnosed fallback. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
2. **Tx accounting pass (temporal):** compute the expected tx units (bytes) per stage and insert `mbarrier.expect_tx` / arrive‑expect sequences so that phase completion is provably reachable. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
3. **Schedule synthesis pass:** choose pipeline depth and “wait distance” to maximize overlap while satisfying the typestate constraints (no early reuse of smem tiles; no use-before-visible). This is the compiler’s version of what FA‑3 does manually. ([arxiv.org](https://arxiv.org/abs/2407.08608?utm_source=openai))  
4. **Verifier:** a fast typestate checker over the SSA token graph that rejects any kernel violating:
   - init‑before‑use / invalidate rules for `mbarrier`, and  
   - phase discipline (must observe completion before next phase arrive), and  
   - tx‑count conservation (expected bytes = completed bytes), and  
   - completion visibility before consumer ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

**Legality story (explicit):** tensormap construction + swizzle legality + descriptor encodability are surfaced as type/proof obligations. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
**Temporal story (explicit):** `mbarrier` phase completion + bulk‑group non‑ordering are modeled as typestate transitions enforced by the verifier. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

---

### 4) Lowering strategy (to GPU primitives)
Lower TxGraph IR to **NVGPU → NVVM → PTX**:

- **TMA load path (Hopper):**
  - Host: create `CUtensorMap` (opaque) with verified params. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - Device: emit `cp.async.bulk.tensor.*.mbarrier::complete_tx::bytes` and matching `mbarrier.expect_tx`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

- **Bulk‑group path (optional for some kernels):**
  - Emit `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group`; do *not* assume ordering inside group. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

- **WGMMA operand legality:**
  - Ensure shared‑memory descriptor fields are encodable (16B granularity, swizzle enums) and warpgroup‑consistent before generating `wgmma.mma_async`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

- **Fallback path (when legality fails):**
  - Lower to classic `cp.async`/LDGSTS or `ld.global`→`st.shared` and keep TxGraph tokens but with weaker/cheaper obligations.

---

### 5) Evaluation plan (benchmarks + metrics)
**Bench suites**
- **TritonBench** (Meta PyTorch): use its curated operator suite as the primary regression/benchmark harness. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **LLM kernel family:**  
  - Attention: FlashAttention‑3‑style kernels (forward + possibly backward), because they stress TMA + asynchrony. ([arxiv.org](https://arxiv.org/abs/2407.08608?utm_source=openai))  
  - Optional second family: long‑context MLA variants (e.g., transpose pipelines leveraging WGMMA), to stress descriptor legality + schedule. ([arxiv.org](https://arxiv.org/abs/2506.01969?utm_source=openai))  

**Hardware**
- Hopper H100 / SM90 (or H20 class if that’s the lab’s availability), since `cp.async.bulk.tensor` and `mbarrier.expect_tx` require SM90+. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

**Baselines**
- Triton backend w/ Linear Layouts but without temporal verification/synthesis. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- CUTLASS/CuTe implementations where available (best‑effort reference). ([docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai))  

**Key metrics (must report)**
- Runtime throughput (TFLOP/s for GEMM/attention), latency.  
- **Barrier stall fraction / pipeline bubbles** (Nsight Compute stall metrics).  
- **TMA fast‑path hit rate**: percent kernels using `cp.async.bulk.tensor` vs fallback. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Tx correctness diagnostics**: #bugs caught / #kernels rejected (and why), verifier time per kernel (ms).  
- **Bandwidth + bank conflicts**: achieved HBM GB/s; shared bank conflict indicators; plus swizzle validity outcomes. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- Code size and register pressure deltas.

---

### 6) Risks (and mitigation)
- **Risk: tx‑unit accounting mismatch** (masked tiles, OOB fill, mixed shapes) leads to conservative waits or false rejects. Mitigation: support symbolic “upper bounds” tokens and prove safety; optionally auto‑instrument to calibrate on micro‑instances. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Risk: protocol drift across PTX versions** (new completion mechanisms / new restrictions). Mitigation: tie the verifier to PTX versioned semantics and gate by target features. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Risk: engineering scope (3–4 months)**. Mitigation: start with SM90 TMA→SMEM pipelines only, then expand to cluster and multicast.

---

## Direction 2 — **LegalEGraph**: Legality‑Aware Equality Saturation for **Layout + Schedule Co‑Optimization**

### 1) Gap (validated + re‑verified)
Today, compilers face a combinatorial explosion where **(layout equivalences)** × **(pipeline schedules)** × **(hardware legality predicates)** interact.

- **Spatial equivalence is abundant** (Linear Layouts, CuTe algebra, ISL relations), but **hardware legality is sparse and discrete**:
  - CUtensorMap is opaque and comes with strict admissibility bounds (alignment, stride multiples, box dims, element strides, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - Swizzle is a finite mode set with extra `_ATOM_*` variants. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
  - WGMMA matrix descriptors quantize and have enumerated swizzle fields (plus invalid bit patterns in related descriptor formats). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

- **Temporal equivalence is also abundant** (many ways to place commit/wait, or to structure barrier phases), but **PTX explicitly forbids assuming ordering within bulk groups**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

- **Seed papers’ limitation alignment:**  
  - Linear Layouts gives a strong backend representation + conversions, but explicitly says its core limitation is power‑of‑two shapes and it doesn’t claim to model these PTX legality/schedule contracts. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
  - ISL work is explicitly foundational (“goal is not performance optimizations”), yet it provides exactly the canonicalization machinery compilers want for de‑duplication. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**Novelty check (3 closest papers/systems and what we do differently):**
1. **Linear Layouts**: provides backend layout representation and conversion/codegen, but doesn’t make TMA/WGMMA legality + async schedule constraints first‑class in the algebra/search. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
2. **ISL integer‑set relations for layouts**: unifies CuTe + linear layouts for formal reasoning, but explicitly not a performance optimizer and doesn’t integrate PTX legality/schedule as hard constraints in extraction. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
3. **egg (e‑graphs / equality saturation)**: provides the infrastructure for rewrite‑driven search with analyses, but not the GPU‑specific legality+temporal analyses we need. ([popl21.sigplan.org](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai))  

---

### 2) Theory (from Stage‑1.5 toolbox)
**Named theory:** *equality saturation (e‑graphs) + legality‑aware analyses*.

- Use **e‑class analyses** to track invariants (alignment, stride congruences, descriptor encodability, tx‑count obligations). ([popl21.sigplan.org](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai))  
- For cost: hybrid of analytic features + learned ranking (Ansor‑style cost model) for selecting among many legal candidates. ([usenix.org](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai))  

---

### 3) Artifact (new compiler/runtime mechanism)
**LegalEGraph = a “layout‑and‑schedule superoptimizer” for GPU kernels that never emits illegal fast‑path code.**

#### IR design
Introduce a dedicated MLIR `layout` + `async_schedule` dialect:

- `layout.*` ops: split/merge/permute/swizzle/pad/reshape, with semantics convertible to:
  - Linear Layouts matrices (when in \( \mathbb{F}_2 \) domain), ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
  - ISL relations (general canonical form). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

- `async_schedule.*` ops: represent pipeline structure explicitly:  
  `schedule.stage(i)`, `schedule.commit_group`, `schedule.wait_group(k)`, `schedule.mbarrier_phase(next)`.

#### Algorithm
1. **Lower candidate kernels to e‑graph terms**: treat layout expressions and schedule expressions as rewriteable IR terms.
2. **Saturate with bounded rewrites**:
   - Spatial rewrites (layout equivalences from Linear Layouts/CuTe/ISL).
   - Temporal rewrites (commute independent copies, fuse commits, adjust wait distance) **but guarded** by legality analyses.
3. **Legality analysis as an e‑class analysis**:
   - CUtensorMap admissibility predicates (alignment, bounds, mods). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
   - TMA swizzle validity (inner‑dimension bounds, 128B alignment, 16B granularity). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
   - Bulk async non‑ordering constraints (disallow rewrites that assume intra‑group order). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
   - WGMMA descriptor encodability constraints (16B quantization, swizzle enum). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
4. **Extraction**: choose a *legal* representative minimizing a multi‑objective cost:  
   instruction count + predicted BW + predicted stalls + padding overhead + register pressure proxy.

**Legality story (explicit):** legality is an analysis lattice in the e‑graph; illegal states are pruned early, not late. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
**Temporal story (explicit):** schedule rewrites are part of the search space, but constrained by bulk‑group semantics and mbarrier phase rules. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

---

### 4) Lowering strategy (to GPU primitives)
After extraction, lower to concrete codegen:

- **Descriptor synthesis boundary:** generate (or require) a host‑side `CUtensorMap` creation call with parameters extracted from the chosen layout. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **PTX emission:** 
  - use `cp.async.bulk.tensor` variants plus chosen completion mechanism; ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - enforce explicit `commit_group/wait_group` or `mbarrier` usage depending on schedule plan. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **WGMMA readiness:** only emit `wgmma.mma_async` when descriptors are encodable + uniform. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

---

### 5) Evaluation plan (benchmarks + metrics)
**Bench suites**
- **TritonBench** as the core operator set (compile‑time explosion is realistic here). ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **LLM kernel family:** attention (FlashAttention‑3 baseline) and at least one fused MLP/GEMM family from TritonBench’s submodules (flash‑attention/xformers, etc.). ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

**Baselines**
- Linear Layouts backend selection without equality saturation. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Optional: Ansor‑style tuning on a reduced schedule space (to compare search quality vs time). ([usenix.org](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai))  

**Metrics (must report)**
- **Compile time / memory**: e‑graph size, saturation iterations, extraction time.  
- **#candidates pruned by legality** and top rejection reasons (alignment, boxDim, swizzle invalidity, descriptor encode failure). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- Runtime: throughput + achieved BW + bank conflict indicators + barrier stall. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- Code size: instruction count deltas; register count (occupancy impact).

---

### 6) Risks (and mitigation)
- **Risk: e‑graph blow‑up**. Mitigation: strong legality pruning + bounded saturation + ISL canonical forms for aggressive merging. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **Risk: cost model misses temporal hazards** (predicts BW but not stalls). Mitigation: include explicit temporal features (tx‑count, stage depth) and validate against Nsight stall metrics; fall back to small autotuning loops when close. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Risk: rule correctness**. Mitigation: build a rewrite test harness over micro‑instances + differential testing vs baseline kernels.

---

## Direction 3 — **TMEM‑IR**: Blackwell Tensor Memory as a First‑Class Resource (Allocation + Lane Partition + Temporal Safety)

### 1) Gap (validated + re‑verified)
Blackwell changes the meaning of “layout”: it is now **layout + allocation + access partitioning + lifetime.**

Key PTX‑level realities:

- **TMEM is dynamically allocated** and must be allocated by a **single warp**; allocation is in **columns**, unit is **32 columns**, `nCols` must be **power‑of‑two** and within **[32, 512]**; **all allocations must be deallocated before kernel exit**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Access restrictions are structural**: TMEM lanes are partitioned across warps in a warpgroup (warp 0 gets lanes 0–31, warp 1 lanes 32–63, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Collective access UB:** `tcgen05.ld` / `tcgen05.st` are undefined if threads in the warp do not use the same `taddr` (or diverge). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Temporal / memory‑model wrinkle:** tcgen05 shared‑memory accesses occur in an **async proxy**, and PTX requires explicit cross‑proxy fencing (`fence.proxy.async`) when mixing proxies. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

These are exactly the kinds of constraints that are not expressed in layout algebras (Linear Layouts even cites TMEM as a “special memory unit” example, but the allocation/lifetime rules are extra‑algebraic). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**Novelty check (3 closest papers/systems and what we do differently):**
1. **PTX ISA Tensor Memory + tcgen05**: defines the allocation/lane/access/UB contracts, but it’s a spec—not a compiler artifact that can enforce or optimize them. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
2. **CUTLASS/CuTe Blackwell tutorials + tooling**: show how experts hand‑craft cluster+UMMA/TMA pipelines, but this is library‑level, pattern‑specific, and not generally synthesized/verified for arbitrary kernels. ([research.colfax-intl.com](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/?utm_source=openai))  
3. **Linear Layouts**: provides the spatial algebra and conversion infrastructure, but not dynamic TMEM allocation/lifetime management or warpgroup‑partition semantics. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

---

### 2) Theory (from Stage‑1.5 toolbox)
**Named theory:** *resource typestate / linear capabilities* applied to **on‑chip allocations**.

- TMEM allocations become linear resources: `alloc` produces a capability; `dealloc` consumes it; use is scoped by warpgroup/warp‑uniformity constraints.
- Pair with constraint solving (small SMT/ILP) for choosing `nCols` (power‑of‑two, <= 512) and mapping tiles to lane partitions.

---

### 3) Artifact (new compiler/runtime mechanism)
**TMEM‑IR = an MLIR dialect + analyses that make TMEM a safe, optimizable tier alongside SMEM.**

#### IR design
Add a `tmem` dialect (or NVGPU/NVVM extensions) with:

- `tmem.alloc nCols -> !tmem.handle<cols=n, lanes=128, scope=cta_group>`  
- `tmem.dealloc !tmem.handle` (must dominate all exits)
- `tmem.subview(handle, warp_id)` producing a **lane‑restricted view** consistent with PTX’s per‑warp lane access (0–31, 32–63, …). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- `tmem.ld/st` ops that require:
  - warp‑uniform `taddr`, and
  - `.aligned` execution proof (no divergence). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

#### Algorithm
1. **Lifetime & uniformity verifier (temporal):**
   - statically prove alloc/dealloc pairing on all control paths;
   - prove warp‑uniform `taddr` and aligned execution (or conservatively reject). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
2. **Tiering pass (spatial + resource):**
   - decide which tiles live in SMEM vs TMEM based on reuse + capacity (`nCols`) + lane partition constraints.
3. **Proxy‑aware fence insertion (temporal/memory model):**
   - insert `fence.proxy.async` when required to ensure visibility between generic and async proxy operations. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
4. **Integration with TMA:** optional prefetch/loads to SMEM via TMA, then staged movement into TMEM via tcgen05 data movement instructions where appropriate (future expansion).

**Legality story (explicit):** TMEM legality = power‑of‑two `nCols`, range bounds, single‑warp alloc semantics, dealloc‑before‑exit, lane partition legality, and warp‑uniform access. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
**Temporal story (explicit):** TMEM lifetime (alloc/dealloc), aligned execution, and proxy fencing requirements are verified/inserted. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

---

### 4) Lowering strategy (to GPU primitives)
Lower `tmem` dialect to **tcgen05** PTX:

- `tmem.alloc` → `tcgen05.alloc.*` (writes taddr to shared). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- `tmem.dealloc` → `tcgen05.dealloc.*` with matching `nCols`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- `tmem.ld/st` → `tcgen05.ld` / `tcgen05.st`, enforcing warp‑uniform `taddr` and `.aligned`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- Insert required proxy fences around SMEM interactions if tcgen05 ops access SMEM via async proxy. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

Optionally integrate with Hopper‑style TMA in mixed pipelines (HBM→SMEM via `cp.async.bulk.tensor`, then SMEM→TMEM). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

---

### 5) Evaluation plan (benchmarks + metrics)
**Bench suites**
- **TritonBench** (for broad operator coverage and to expose tiering opportunities). ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **LLM kernel family:**  
  - attention forward for Blackwell (even if kernel variants differ), and  
  - a GEMM/MLP family where TMEM residency plausibly improves compute utilization.

**Hardware**
- A Blackwell‑class GPU that supports `tcgen05` (PTX lists SM100* targets). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  If hardware access is limited, scope the implementation to compile+functional correctness first, then performance once hardware is available.

**Baselines**
- CUTLASS/CuTe Blackwell examples (where available) as “expert kernel” reference points. ([research.colfax-intl.com](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/?utm_source=openai))  
- Triton/MLIR codegen without TMEM (SMEM‑only staging).

**Metrics**
- Runtime throughput + utilization (tensor core issue rate proxy).  
- **Allocation overhead**: cycles spent in `tcgen05.alloc/dealloc`, and whether allocations block. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Correctness/verification**: #IR programs rejected for violating uniformity/lifetime, and verifier time. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- Proxy fence impact (added instructions) and measured stall/visibility issues. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- Code size and register pressure changes (occupancy effects).

---

### 6) Risks (and mitigation)
- **Risk: tooling maturity / fast‑moving ISA** (tcgen05 ecosystem is newer, plus arch variants like `sm_100a`, `sm_110f` naming changes). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  Mitigation: gate features by target and keep the dialect minimal and versioned.
- **Risk: performance regressions from alloc/dealloc if done too frequently. Mitigation: hoist allocations, reuse buffers, and restrict to kernels with enough compute to amortize. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Risk: verification too conservative** (rejects useful kernels). Mitigation: add “escape hatches” with runtime assertions for uniformity when static proof fails, then iterate.

---

## Decision Matrix

| Direction | Novelty(1-5) | Hardware_Relevance(1-5) | Impl_Risk(1-5) | Why_it_wins | unknowns |
|---|---:|---:|---:|---|---|
| **TxGraph (Dir 1)** | 4 | 5 | 3 | Directly targets the biggest Hopper cliff: **temporal correctness + latency hiding** with `mbarrier` tx‑count and bulk async semantics; should turn “expert‑only pipelines” into a compiler feature. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | How to compute/approximate tx bytes robustly under masking/OOB; how much schedule synthesis helps beyond heuristics. ([docs.nvidia.com](https:vidia.com/cuda/parallel-thread-execution/index.html)) |
| **LegalEGraph (Dir 2)** | 5 | 4 | 4 | Unifies **spatial + temporal** search with legality pruning; could become the “layout/schedule selection engine” for Triton/MLIR and reduce conversion/schedule explosions. ([popl21.sigplan.org](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai)) | Can legality analyses stay fast enough inside e‑graphs? Will cost models predict barrills well without heavy autotuning? ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) |
| **TMEM‑IR (Dir 3)** | 4 | 5 | 5 | Unlocks the **new Blackwell tier (TMEM)** by making allocation/lane partition/lifetime compiler‑managed and safe—something current layout theories don’t model. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | Hardware/toolchain access and stability; whether TMEM residency beats optimized SMEM for the taLM kernels in practice. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) |

---
Learn more:
1. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
2. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
3. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CA__TENSOR__MEMORY.html?utm_source=openai)
4. [4.11. Asynchronous Data Copies — CUDA Programming Guide](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai)
5. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608?utm_source=openai)
6. [CuTe TMA Tensors — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html?utm_source=openai)
7. ['nvgpu' DialecMLIR](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)
8. [A Formal Analysis of the NVIDIA PTX Memory Consistency Model. | Research](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai)
9. [GitHub - meta-pytorch/tritonbench: Tritonbench is a collection of PyTorch custom operators with example inputs to measure their performance.](https://github.com/meta-pytorch/tritonbench?utm_source=openai)
10. [FlashMLA-ETAP: Efficient Transpose Attention Pipeline for Accelerating MLA Inference on NVIDIA H20 GPUs](https://arxiv.org/abs/2506.01969?utm_source=openai)
11. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
12. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
13. [egg: Fast and Extensible Equality Saturation (POPL 2021 - POPL Research Papers) - POPL 2021](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egFast-and-Extensible-Equality-Saturation?utm_source=openai)
14. [Ansor: Generating High-Performance Tensor Programs for Deep Learning | USENIX](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai)
15. [CUTLASS Tutorial: GEMM with Thread Block Clusters on NVIDIA® Blackwell GPUs – Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/?utm_source=openai)
