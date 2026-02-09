## Step 1 — Re-verified bottlenecks (manuals + recent literature)

The Stage‑1 bottleneck ranking holds up under current (late‑2025 / early‑2026) manuals + toolchain docs:

1. **Descriptor legality cliffs are real and enumerated.**  
   `CUtensorMap` encoding has hard constraints (e.g., `tensorRank ≤ 5`, `boxDim[i] ≤ 256`, `elementStrides[i] ≤ 8`, strict alignment, and swizzle/interleave coupling such as `interleave==32B ⇒ swizzle==32B`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
   The CUDA Programming Guide further tightens “swizzle usability” with **128B global-memory alignment** and explicit **inner-dimension caps** per swizzle width; violating these makes the instruction invalid. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?utm_source=openai))  

2. **Temporal correctness is a contract, not a hint.**  
   PTX makes `cp.async.bulk.tensor` a *true async operation* with two completion mechanisms: `.mbarrier::complete_tx::bytes` vs `.bulk_group`, and the bulk‑group path has its own `commit_group / wait_group` semantics (including no ordering guarantee between ops inside a group). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
   For `mbarrier`, phase completion *requires* **pending arrivals == 0 AND tx‑count == 0**, with `expect-tx` and `complete-tx` updating the tx‑count. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

3. **Blackwell TMEM adds new legality + collective constraints.**  
   TMEM allocation is **column‑granular** with **min 32 columns** and **power‑of‑two**, and allocating a column allocates **all 128 lanes**. TMEM access is lane‑partitioned per warp within a warpgroup (warp 0: lanes 0–31, …, warp 3: 96–127). `tcgen05.ld` is warp‑collective and requires all threads to provide the **same `taddr`**, else behavior is undefined. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
   Toolchains now expose this: LLVM NVPTX documents `tmem` as addrspace(6) and provides `tcgen05.alloc/dealloc` intrinsics; MLIR NVVM dialect includes `nvvm.tcgen05.*` ops. ([llvm.org](https://llvm.org/docs/NVPTXUsage.html?utm_source=openai))  

4. **Evaluation instrumentation is mature and metric-aligned.**  
   Nsight Compute explicitly defines “Eligible Warps,” “Issued Warp,” and “skipped issue slots” as latency-hiding indicators, plus warp stall sampling for barrier/membar/scoreboard stalls. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html?utm_source=openai))  
   It also clarifies Achieved Occupancy as `sm__warps_active.avg.pct_of_peak_sustained_active` (distinct from “SM %” throughput). ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750?utm_source=openai))  

5. **Benchmark substrate exists.**  
   TritonBench is a ready-made operator suite with real kernels and submodules (including flash-attention/xformers-like stacks) suitable for ASPLOS-style evaluation. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
   For an LLM kernel family, FlashAttention remains the canonical IO-aware attention baseline. ([arxiv.org](https://arxiv.org/abs/2205.14135?utm_source=openai))  

---

## Direction 1 — **AsyncContractIR: typestate-verified TMA pipelines (deadlock-free + tx-correct by construction)**

### 1) Gap
**Validated gap:** seed layout formalisms are *spatially strong* but *temporally weak*: they don’t ensure a kernel’s async pipeline is **mbarrier‑phase correct**, **tx‑count correct**, and **completion‑mechanism consistent**. ([arxiv.org](https://arxiv.org/abs/2505.23819?utm_source=openai))  

- **Legality story (required):** “layout equivalence” doesn’t imply **TMA encodability**. Even if a layout map is correct, `CUtensorMap` has discrete constraints (rank/boxDim/alignment/swizzle/interleave) that gate whether we can emit TMA at all. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Temporal story (required):** even with a legal tensormap, the kernel can hang/stall unless `mbarrier`’s phase discipline and tx‑count contract are satisfied, and unless `.mbarrier` vs `.bulk_group` completion modes are scheduled coherently. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

### 2) Theory
We apply **Typestate / effect systems** (Stage‑1.5 Theory A) to encode the async pipeline as a finite‑state protocol, with optional **Petri‑net/SDF** (Stage‑1.5 Theory B) as a *schedule synthesizer*.

**Novelty check (3 closest + difference):**
1) **PTX ISA + CUDA PG** define the contract (tx‑count, completion mechanisms), but provide no compile-time proof system; we make these rules *type-checkable invariants* in IR. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
2) **MLIR NVGPU dialect** exposes `mbarrier.*` and `tma.*` ops, but does not enforce *global* correctness properties like “every phase has a wait that returns true before next phase’s arrive,” or “txcount equals the sum of bytes that will complete.” We add a verifier + synthesis pass that makes those properties explicit. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
3) **Triton** (and the newer **Linear Layouts** work) provides strong spatial/layout codegen, but does not elevate async orchestration to a first-class, checkable contract; we add that missing temporal layer as an IR+pass artifact. ([pldi19.sigplan.org](https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations?utm_source=openai))  

### 3) Artifact
**New artifact:** an MLIR-level “AsyncContractIR” layer that sits *above* NVGPU/NVVM, and makes the pipeline protocol explicit.

**IR design (concrete):**
- Types:
  - `!ac.phase<barrier_id, parity>`: permission token for a specific `mbarrier` phase.
  - `!ac.tx<bytes>`: linear capability representing “bytes outstanding” that must be discharged via `.complete_tx::bytes`.
  - `!ac.tma_desc<rank, swizzle, interleave, align>`: a *refinement-carrying handle* to a tensormap (can be symbolic).
- Ops (sketch):
  - `ac.tma.issue %desc, %coords, %dst_smem, %barrier : (...) -> (!ac.tx<k>, !ac.phase<id,p>)`
  - `ac.mbarrier.arrive_expect_tx %barrier, %tx : !ac.phase<id,p> -> !ac.phase<id,p>`
  - `ac.mbarrier.wait %barrier, %phase : !ac.phase<id,p> -> !ac.phase<id,p+1>`
  - `ac.bulk.commit_group` / `ac.bulk.wait_group` for `.bulk_group` completion (separate typestate track).
- **Legality integration:** a verifier that either:
  1) proves `%desc` satisfies `CUtensorMap` constraints (rank/alignment/swizzle caps), or  
  2) emits a *repair plan* (“pad dim0 to multiple of 16B”, “switch swizzle to NONE”, “reduce boxDim”) and re-types `%desc`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

**Core algorithm:**
- **Typestate checking as SSA dataflow**:
  - Ensure every `ac.tma.issue` in `.mbarrier::complete_tx::bytes` mode is covered by a matching `arrive_expect_tx` for the same phase and correct byte total.
  - Enforce PTX’s requirement: *a phase must be observed complete* (via wait/test returning true) before issuing arrive in the subsequent phase. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- Optional Petri-net/SDF mode: auto-synthesize stage depth + placement of commit/wait/arrive to maximize steady-state throughput subject to bounded buffering.

### 4) Lowering
**Lowering strategy (down to GPU primitives):**
1) `AsyncContractIR` → **NVGPU dialect**:
   - `ac.mbarrier.*` → `nvgpu.mbarrier.create/init/arrive.expect_tx/try_wait.parity` ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
   - `ac.tma.issue` → `nvgpu.tma.async.load` + the appropriate `mbarrier` operand.
2) NVGPU → NVVM → PTX:
   - Emit `cp.async.bulk.tensor...mbarrier::complete_tx::bytes` when using barrier completion. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
   - Emit `.bulk_group` plus `cp.async.bulk.commit_group / wait_group` when chosen. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
3) Descriptor creation:
   - Static shapes: use `cuTensorMapEncodeTiled` (or the NVGPU descriptor op if using device-side creation) and enforce all constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

### 5) Eval
**Benchmarks (required):**
- **TritonBench**: start with `gemm`, `softmax`, `layernorm`, and any op that already has a TMA-capable variant. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **LLM kernel family:** FlashAttention-like attention forward (prefill) and causal attention (decode). ([arxiv.org](https://arxiv.org/abs/2205.14135?utm_source=openai))  

**Baselines:**
- Stock Triton (current) without contract typing.
- “Best effort” hand-inserted barriers (if available) as an upper bound.

**Metrics (must include these categories):**
- **Speed:** end-to-end kernel time; Nsight “SpeedOfLight” compute/memory throughput breakdown. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html?utm_source=openai))  
- **Barrier / pipeline health:**  
  - SchedulerStats: skipped issue slots / eligible warps. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html?utm_source=openai))  
  - Warp stall sampling: `smsp__pcsamp_warps_issue_stalled_barrier`, `..._membar`, `..._long_scoreboard`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html?utm_source=openai))  
- **Achieved occupancy:** `sm__warps_active.avg.pct_of_peak_sustained_active`. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750?utm_source=openai))  
- **Bank conflicts:** track shared-memory access patterns; validate against the 32-bank model and use Nsight’s shared-memory analysis tables. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html?utm_source=openai))  
- **Compile-time + code size:** type-check wall time, number of inserted ops, binary size deltas.

### 6) Risks
- **Semantic coverage risk:** multi-producer/multi-consumer pipelines, cluster multicast, and mixed completion mechanisms can create false positives/negatives if the typestate model is incomplete. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- **Adoption risk:** if the verifier rejects too many kernels, we need an auto-repair story (e.g., downgrade to `.bulk_group` or non-TMA copy) to avoid usability cliff. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Engineering risk:** integrating into Triton’s lowering pipeline without destabilizing existing codegen.

---

## Direction 2 — **LegalSAT: legality-aware equality saturation + SMT to synthesize TMA descriptors *and* async plans**

### 1) Gap
**Validated gap:** the feasible region for “fast path” is **discrete and brittle**: `CUtensorMap` legality + swizzle tables + alignment rules decide if TMA is even possible. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
Current layout algebras (Linear Layouts / ISL relations / categorical CuTe foundations) can prove spatial properties but don’t emit **descriptor field assignments** that satisfy the driver/PTX constraints. ([arxiv.org](https://arxiv.org/abs/2505.23819?utm_source=openai))  

- **Legality story (required):** synthesize (or reject) concrete `CUtensorMap` fields—`globalStrides`, `boxDim`, `elementStrides`, `swizzle`, `interleave`—under the driver’s constraints, including packed-type special cases. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Temporal story (required):** couple the chosen layout/descriptor with a valid async completion plan (either `.mbarrier::complete_tx::bytes` with correct tx accounting, or `.bulk_group` with `commit_group/wait_group` sequencing). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

### 2) Theory
We compose Stage‑1.5 **Equality saturation (e-graphs)** with **Refinement types / SMT(ILP)**:
- E-graphs explore layout rewrites without exponential blowup from duplicated subexpressions.
- SMT/ILP solves the *field-level* integer constraints (padding, alignment, enum coupling) and produces a witness assignment.

**Novelty check (3 closest + difference):**
1) **egg** provides fast equality saturation + e-class analyses; we specialize analyses to track **GPU descriptor legality** (rank caps, swizzle/inner-dim, alignment) and extract a *descriptor witness*, not just an equivalent expression. ([arxiv.org](https://arxiv.org/abs/2004.03082?utm_source=openai))  
2) **eqsat (MLIR dialect)** proposes keeping e-graphs in IR to avoid translation overhead; we use that idea but add a legality+async synthesis objective tied to concrete PTX/CUDA contracts. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
3) **Ansor** explores schedule spaces with a learned cost model; we differ by centering the search on **legality cliffs** (descriptor encodability + completion-mechanism correctness) and returning a *proof/witness* that enables emitting TMA/bulk-tensor paths. ([arxiv.org](https://arxiv.org/abs/2006.06762?utm_source=openai))  

### 3) Artifact
**New artifact:** “LegalSAT” pass pipeline = `(Layout E-Graph) + (Legality Solver) + (AsyncPlan Synthesizer)`.

**IR design:**
- A minimal layout term IR (either:
  - dedicated `layout.*` ops, or
  - embed into `eqsat` dialect nodes) with attributes for shape/stride/swizzle candidates.
- A first-class descriptor object:
  - `!legal.tma_desc<rank, dtype, interleave?, swizzle?>` with symbolic fields.
- A schedule object:
  - `!legal.async_plan<mode = mbarrier | bulk_group, stage_count, tx_bytes_per_stage>`.

**Algorithm (end-to-end):**
1) **Saturate** layout terms with rewrites:
   - pad/unpad, split/join, reorder dims, introduce legal swizzle candidates, change `boxDim` factoring, etc.
2) Run **e-class legality analyses**:
   - track alignment lower bounds, candidate inner dimension sizes, and “maybe legal” predicates.
3) For top‑K candidates, run **SMT/ILP** to:
   - pick `boxDim[i]`, `elementStrides[i]`, padding, and enum choices satisfying driver constraints (including `interleave==32B ⇒ swizzle==32B`, `boxDim[i]≤256`, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
4) **AsyncPlan Synthesizer**:
   - decide `.mbarrier::complete_tx::bytes` vs `.bulk_group` based on directionality and constraints, then compute required tx-count bytes and stage schedule. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

### 4) Lowering
**Lowering strategy:**
- Descriptor creation:
  - Host-side: emit `cuTensorMapEncodeTiled(...)` calls with synthesized fields. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - Dynamic shapes: use PTX `tensormap.replace` to patch selected fields (e.g., `global_dim`, `global_stride`, `box_dim`) when safe, preserving legality. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html?utm_source=openai))  
- Copy/compute:
  - If plan selects mbarrier completion: emit `cp.async.bulk.tensor...mbarrier::complete_tx::bytes` + `mbarrier.expect_tx`/`arrive.expect_tx` schedule. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
  - Else bulk-group: emit `cp.async.bulk.tensor...bulk_group` + `cp.async.bulk.commit_group/wait_group`. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- Integration point:
  - Implement as a Triton/MLIR pass before NVGPU lowering, so we can still rewrite layout terms and specialize codegen before emitting target ops.

### 5) Eval
**Benchmarks (required):**
- **TritonBench:** measure “% ops that become TMA-fastpath legal” and end-to-end speedups for relevant kernels (`addmm`, `gemm`, conv-like ops). ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **LLM family:** FlashAttention forward + backward (or FlashAttention-like Triton variants if in TritonBench submodules). ([arxiv.org](https://arxiv.org/abs/2205.14135?utm_source=openai))  

**Metrics:**
- **Speed / utilization:** Nsight “SpeedOfLight” throughput sections; compare compute vs DRAM saturation. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html?utm_source=openai))  
- **Legality hit rate:** % kernels that successfully emit TMA (`cp.async.bulk.tensor`) vs fallback.
- **Bank conflicts:** validate swizzle choices reduce conflicts per the 32-bank model. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html?utm_source=openai))  
- **Temporal health:** eligible warps + skipped issue slots + barrier stall reasons. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html?utm_source=openai))  
- **Compile-time / memory:** e-graph size, rewrite iterations, SMT solve time, and code-size blowup from specialization.

### 6) Risks
- **Compile-time blowup:** e-graphs + SMT can explode unless we bound rewrite sets and enforce strong pruning using legality analyses. ([arxiv.org](https://arxiv.org/abs/2004.03082?utm_source=openai))  
- **Versioning risk:** CUDA/driver/PTX legality tables evolve (new packed types, new swizzle modes). The solver must be versioned against toolkit/PTX targets. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Extraction risk:** best legal solution might require a rewrite not in the rule set; we need a repair-rule discovery loop (possibly learned).

---

## Direction 3 — **TierGrainIR: TMEM-aware tiering + warpgroup-grain typing (H100↔Blackwell portable kernel synthesis)**

### 1) Gap
**Validated gap:** Blackwell introduces a new on-chip tier (TMEM) with strict collective + allocation semantics; ignoring it yields register-pressure/occupancy failures, but using it incorrectly is UB (lane partitions, same-`taddr` rule). ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
Seed layout math does not encode “where values live” (Reg vs SMEM vs TMEM) nor the grain constraints needed for correctness across warps/warpgroups. ([arxiv.org](https://arxiv.org/abs/2505.23819?utm_source=openai))  

- **Legality story (required):**
  - TMEM: `tcgen05.alloc` column rules (≥32, power-of-two), lane access restrictions per warp, `tcgen05.ld` requires same `taddr`. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
  - TMA: tensormap legality + swizzle alignment/inner-dim rules still gate feeding TMEM pipelines. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Temporal story (required):**
  - Orchestrate a multi-tier pipeline: `cp.async.bulk.tensor` (mbarrier completion) → SMEM tiles → `tcgen05.cp/mma/ld/st` with proper fences/ordering and explicit alloc/dealloc lifetime. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

### 2) Theory
We apply Stage‑1.5 **Memory-tier type systems** + **Roofline/operational-intensity guidance**:
- Make tier placement a checked, explicit decision (not an accidental register spill).
- Use a lightweight roofline-style classifier to decide whether to invest in deeper staging / TMEM accumulation vs simpler paths.

**Novelty check (3 closest + difference):**
1) **PTX TMEM / tcgen05 spec** defines the rules but offers no compiler-level synthesis; we provide an IR that *types* those rules and a pass that generates legal alloc/dealloc + collective addressing automatically. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
2) **LLVM NVPTXUsage + MLIR NVVM dialect** now expose tcgen05 intrinsics/ops (plumbing), but do not solve the policy problem: *which* tensors go to TMEM, and *how* to schedule fences/async ops to benefit performance. We add placement + scheduling algorithms. ([llvm.org](https://llvm.org/docs/NVPTXUsage.html?utm_source=openai))  
3) **TileLang-style TMEM APIs** surface TMEM allocation to programmers; we go further by integrating TMEM into compiler legality + cost semantics so the compiler can choose TMEM placement automatically and safely. ([tilelang.com](https://www.tilelang.com/autoapi/tilelang/language/allocate/index.html?utm_source=openai))  

### 3) Artifact
**New artifact:** “TierGrainIR” = a typed IR extension that makes **tier** and **execution grain** explicit.

**IR design (concrete):**
- Types:
  - `!tier.mem<Reg | SMEM | TMEM, bytes, collective = warp | warpgroup>`
  - `!grain.exec<warp32 | warpgroup128>`
  - `!tmem.alloc_handle<ncols>` (affine/linear: must be deallocated)
- Ops:
  - `tier.alloc.tmem %ncols : !tmem.alloc_handle<ncols>`
  - `tier.dealloc.tmem %handle`
  - `tier.copy.g2s.tma %tma_desc, %coords, %smem, %mbar` (returns tx token)
  - `tier.mma.hopper` vs `tier.mma.blackwell` (selects wgmma vs tcgen05 family based on target)
- Invariants enforced:
  - `tmem` alloc/dealloc in same CTA lifetime, dealloc dominates all uses, and no “post-relinquish” allocation. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
  - Any `tcgen05.ld/st` use is warp-collective with uniform `taddr`. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  

**Algorithm: placement + scheduling**
1) **Placement solver**: choose where accumulators live (Reg vs TMEM) under capacity + grain constraints.
2) **Pipeline scheduler**:
   - Stage count selection
   - `mbarrier` phase plan (bytes per stage)
   - tcgen05 fence placement (before/after) when using asynchronous tcgen05 ops. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVVMDialect/?utm_source=openai))  

### 4) Lowering
**Lowering strategy (H100 + Blackwell):**
- **H100 (sm90)** path:
  - Use NVGPU warpgroup MMA ops (wgmma-style) where applicable, and keep accumulators in registers; still use TMA for gmem→smem when legal. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
- **Blackwell (sm100)** path:
  - `tier.alloc.tmem` → `nvvm.tcgen05.alloc` (tmem addrspace(6)), store `taddr` in shared. ([llvm.org](https://llvm.org/docs/NVPTXUsage.html?utm_source=openai))  
  - `tier.copy.*`:
    - gmem→smem via `cp.async.bulk.tensor...mbarrier::complete_tx::bytes` ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
    - smem→tmem via `nvvm.tcgen05.cp` when used (plus required tcgen05 fences). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVVMDialect/?utm_source=openai))  
  - MMA: lower to `nvvm.tcgen05.*` ops for compute; enforce uniform `taddr`. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
  - `tier.dealloc.tmem` → `nvvm.tcgen05.dealloc`. ([llvm.org](https://llvm.org/docs/NVPTXUsage.html?utm_source=openai))  

### 5) Eval
**Benchmarks (required):**
- **TritonBench:** focus on GEMM/MLP-like ops where accumulator pressure is high and could benefit from TMEM on Blackwell. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **LLM family:** FlashAttention kernels (attention is IO-heavy and highlights tiering/staging tradeoffs). ([arxiv.org](https://arxiv.org/abs/2205.14135?utm_source=openai))  

**Metrics:**
- **Performance:** Nsight compute/memory throughput (“SpeedOfLight”), plus compare H100 vs Blackwell scaling. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html?utm_source=openai))  
- **Occupancy & reg pressure:** Achieved Occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`) plus launch stats on registers. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750?utm_source=openai))  
- **Warpgroup / collective stalls:** include warp stall reasons, including warpgroup-related stalls where present. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html?utm_source=openai))  
- **Correctness guards:** runtime asserts (debug mode) ensuring uniform `taddr` and balanced alloc/dealloc; static verifier rejects UB patterns. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- **Compile-time + code size:** tier/grain inference time, number of specialized variants.

### 6) Risks
- **Hardware access risk:** meaningful evaluation needs SM100 (or equivalent) access; otherwise we can validate legality via PTXAS + microbenchmarks but not full performance. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- **Surface area risk:** TMEM introduces many new constraints (allocation blocking, fences, lane alignment tables), making a 3–4 month prototype ambitious. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html))  
- **Portability risk:** a tiering IR must keep a clean Hopper fallback (wgmma/regs) to avoid becoming “Blackwell-only.”

---

## Decision Matrix

| Direction | Novelty(1-5) | Hardware_Relevance(1-5) | Impl_Risk(1-5) | Why_it_wins | Key_unknowns |
|---|---:|---:|---:|---|---|
| **1) AsyncContractIR** | 4 | 5 | 3 | Fastest path to “real correctness wins”: provable elimination of tx/p+ measurable latency-hiding improvements, directly tied to PTX/Nsight semantics. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)) | Can the typestate model cover real kernels (multi-stage, predication, mixed completion mechanisms) without too many false rejects? ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)) |
| **2) LegalSAT (E-graphs + SMT)** | 5 | 4 | 4 | Highest research “hook”: turns discrete legality tables into a synthesis lem; can raise TMA hit-rate and expose new layout+descriptor solutions not found by hand. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | Will compile-time stay bounded? Can we build a rewrite set that reaches legal solutions reliably across operator families? ([arxiv.org](https://arxiv.org/abs/2004.03082?utm_source=openai)) |
| **3) TierGrainIR (TMEM)** | 4 | 5 | 5 | Strong Blackwell story: makes TMEM exploit **systematic** (not “expert-ly”), with legality guarantees for alloc/dealloc + collective addressing. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)) | How much speedup is available for real kernels once constraints are respected? How sensitive are gains to tile sizes / stage counts / tcgen05 fence semantics? ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)) |

---
Learn more:
1. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archiv13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
2. [CUDA C++ Programming Guide (Legacy) — CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?utm_source=openai)
3. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html)
4. [User Guide for NVPTX Back-end — LLVM 22.0.0git documentation](https://llvm.org/docs/NVPTXUsage.html?utm_source=openai)
5. [2. Profiling Guide — NsightCompute 12.9 documentatis://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html?utm_source=openai)
6. [Metric references and description - Nsight Compute - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/metric-references-and-description/111750?utm_source=openai)
7. [GitHub - meta-pytorch/tritonbench: Tritonbench is a collection of PyTorch custom operators with example inputs to measure their performance.](https://github.com/meta-pytorch/tritonbench?utm_source=openai)
8. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135?utm_source=openai)
9. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $\\mathbb{F}\_2$](https://arxiv.org/abs/2505.23819?utm_source=openai)
10. ['nvgpu' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)
11. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (MAPL 2019) - PLDI 2019](https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations?utm_source=openai)
12. [2. Profiling Guide — NsightCompute 13.0 documentation](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html?utm_source=openai)
13. [2.2. Writing CUDA SIMT Kernels — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html?utm_source=openai)
14. [egg: Fast and Extensible Equality Saturation](https://arxiv.org/abs/2004.03082?utm_souopenai)
15. [eqsat: An Equality Saturation Dialect for Non-destructive Rewriting](https://arxiv.org/abs/2505.09363?utm_source=openai)
16. [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762?utm_source=openai)
17. [PTX ISA 8.5](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html?utm_source=openai)
18. [tilelang.language.allocate - TileLang 0.1.7.post3 documentation](https://www.tilelang.com/autoapi/tilelang/language/allocate/index.html?utm_source=openai)
19. ['nvvm' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVVMDialect/?utm_source=openai)
