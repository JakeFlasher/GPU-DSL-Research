### Gap 1: **Masked + Piecewise Linear Layouts (Correct + Fast on Ragged / Non-\(2^k\) Shapes)**
**1. The Problem:**
- The seed paper’s core formalism is layouts as linear maps over \(\mathbb{F}_2\), so **modeled extents are inherently \(2^k\)**. Non-\(2^k\) shapes are handled by **padding + masks** (explicitly acknowledged in the Conclusions).
- In real Triton workloads, padding is not just wasted work: it can **break the assumptions behind vectorization, bank-conflict reasoning, and shuffle-based layout conversions** unless “validity” is tracked through the same machinery.
- Stage-1 highlighted a correctness landmine: the paper’s shuffle-based optimizations (e.g., gather / convert) use **layout-only legality** (“axis fits within warp”), but **predication/lane-activity** determines whether shuffles are well-defined. If some lanes are inactive, reading from them is ISA-dependent/undefined.
- **Manual baseline to beat:** hand-written CUDA/Triton+PTX kernels:
  - split ragged tails into separate paths,
  - use explicit predication and/or subgroup-uniform masks for shuffles,
  - fall back to shared memory when uniformity isn’t provable.

**2. The Proposed Solution (Software/Math):**
- **Algebraic Extension: “Layout + Domain” (partial layouts).**  
  Represent a layout as a pair:
  \[
  \mathcal{L} = (A,\;D)
  \]
  where:
  - \(A\) is the existing \(\mathbb{F}_2\)-linear map (matrix) on a *padded* hypercube,
  - \(D\) is a **domain predicate** describing which logical coordinates are valid (and/or which lanes produce defined values).
- **Key design choice:** keep the fast \(\mathbb{F}_2\) machinery, but make validity first-class.
  - For common cases, \(D\) is a conjunction of per-dimension inequalities (e.g., \(0 \le i < N\)), plus optional ragged constraints from runtime metadata.
  - For performance, introduce **piecewise-linear decomposition** so many non-\(2^k\) extents avoid masking entirely:
    - Example: \(96 = 64 + 32\). Treat as union of two power-of-two tiles, each with its own \((A,D)\) where \(D\) is “always true” inside the tile.
- **Compiler Pass: Mask-/Uniformity-Aware Conversion Lowering.**
  - Add a legality check for shuffle-based exchange:
    - **Uniform participation** over the shuffle group, or
    - use explicit subgroup masks and safe materialization of inactive lanes (neutral value + select), or
    - split into piecewise tiles with uniform masks, else fallback to shared.

**3. Implementation Plan:**
- **(A) IR/Analysis plumbing**
  - Extend TritonGPU’s layout analysis to attach a `Domain` object to each tensor value.
  - Concretely:
    - Modify the layout lattice in something like `triton/lib/Analysis/Layout.cpp` to carry:
      - `LinearLayout A`
      - `Domain D` (symbolic bounds + boolean constraints)
    - Define `Domain` operations needed by the layout engine:
      - `intersect(D1, D2)` at merges,
      - `transform(D, op)` for `tt.reshape`, `tt.trans`, `tt.broadcast`, `tt.split/join`.
- **(B) New legality analysis for shuffles**
  - Add a “lane mask uniformity” analysis:
    - Compute whether all lanes in the intended shuffle subgroup are active on the same control-flow predicate.
    - Require either:
      - `uniform_active_mask == true`, or
      - rewrite to a safe sequence.
  - Practical encoding:
    - attach `ttg.uniform_mask` / `ttg.active_mask` metadata to ops that introduce divergence (`tt.load`/`tt.store` masks, boundary checks).
- **(C) Conversion lowering changes**
  - In `ttg.convert_layout` lowering:
    1. compute the candidate shuffle plan from the linear algebra (as in seed),
    2. query `Domain` + uniformity,
    3. choose:
       - shuffle (fast),
       - shuffle+select (safe),
       - piecewise split (fast + safe),
       - shared fallback.
- **(D) Piecewise split transform**
  - Add a pre-pass that rewrites non-\(2^k\) extents into `tt.split` along the problematic axis, runs the existing linear-layout engine per slice, then `tt.join`.
  - Keep the slice count bounded (e.g., binary decomposition with max 2–3 slices) to avoid code explosion.

**Sketch (MLIR-ish pseudocode):**
```mlir
// Before: single masked tile
%tile = tt.load %ptr, %mask : tensor<128xf16>

// After: piecewise split (64 + 32), last remainder masked
%tile0 = tt.load %ptr0 : tensor<64xf16>
%tile1 = tt.load %ptr1 : tensor<32xf16>
%tile2 = tt.load %ptr2, %mask_tail : tensor<32xf16>
%tile  = tt.join %tile0, %tile1, %tile2
```

**4. Evaluation Plan (Real Hardware):**
- **Baselines**
  - Legacy Triton (pre-linear-layouts behavior)
  - Seed-style Triton-Linear (linear layouts but no domain/uniformity correctness framework)
  - Optional: hand-tuned CUDA/Triton+asm baseline for one stress case (masked gather / masked convert)
- **Microbenchmarks**
  1. **Masked shuffle correctness:** boundary tile with divergent lanes; validate numerical correctness vs reference.
  2. **Ragged attention-like mask:** per-row valid length varies; compare shuffle vs shared fallback.
  3. **Non-\(2^k\) head dims:** head_dim in \(\{80, 96, 112\}\) with tile sizes chosen by compiler.
- **TritonBench / E2E**
  - `flex_attention`, `template_attention`, `rope`, `layer_norm`, `embedding` with non-\(2^k\) dims + ragged seq lengths.
- **Metrics (must-haves)**
  - Latency (ms), throughput (TFLOPS / GB/s), correctness pass rate
  - Register pressure (ptxas/ROCm reg counts), spills
  - Compilation time (s), binary size
- **Hardware**
  - NVIDIA H100 (Hopper), NVIDIA A100 (Ampere), AMD MI300 (or MI250 if that’s what’s available in your lab)

---

### Gap 2: **Pipeline/Descriptor-Aware Layouts (Async Copy + Multi-Stage Shared with Stage-Skew)**
**1. The Problem:**
- Stage-1 flagged a major gap: high-end kernels win via **async global\(\rightarrow\)shared pipelines** and **producer/consumer staging**, often with **stage-dependent skew/padding** to avoid bank conflicts across stages.
- The seed paper’s linear layouts:
  - give strong *legality* results (e.g., tile divisibility for `ldmatrix`-like ops),
  - but do not address **coupling layout selection with async-copy scheduling**, barriers, or stage allocation.
- The hard part is that staged addressing often looks like:
  \[
  off = base + stage \cdot stride + f(row, col)
  \]
  where `stage*stride` and skewing involve **integer add/mul (carry)**, which is not \(\mathbb{F}_2\)-linear (and often not XOR-affine).
- **Manual baseline to beat:** FlashAttention-style / CUTLASS-like kernels:
  - explicit cp.async/TMA-style bulk transfers where supported,
  - multi-stage shared buffers,
  - careful barrier placement (`mbarrier`/equivalents),
  - stage-aware skew to avoid systematic bank conflicts.

**2. The Proposed Solution (Software/Math):**
- **Algebraic Extension: “Staged Memory Layouts” with a hybrid model.**
  - Keep the seed’s swizzle/bank reasoning in \(\mathbb{F}_2\) for the *intra-stage* mapping.
  - Add a *stage dimension* and allow one of two representations for stage addressing:
    1. **Carry-free staging (stay in \(\mathbb{F}_2\))**  
       Constrain stride so `stage` occupies disjoint high bits (power-of-two stride in elements). Then:
       - stage selection becomes bit concatenation / block product (still \(\mathbb{F}_2\)-linear).
    2. **\(\mathbb{Z}_{2^k}\)-affine overlay for the stage term**  
       Model:
       \[
       off = (a \cdot stage + b) \bmod 2^k
       \]
       over \(\mathbb{Z}_{2^k}\), which *does* capture carry-based add/mul mod \(2^k\).  
       Intra-stage swizzle remains \(\mathbb{F}_2\)-linear on the low bits.
- **Compiler Pass: Pipeline-Aware Layout + Schedule Co-Design**
  - Jointly pick:
    - stage count \(S\),
    - shared strides/padding/skew parameters,
    - (optional) swizzle parameters,
    - async-copy primitive choice (vectorized ld/st vs cp.async vs descriptor-based bulk copy),
  - subject to:
    - bank-conflict minimization,
    - vectorization maximization,
    - async-copy alignment/size constraints,
    - register-pressure constraints (avoid occupancy cliffs).

**3. Implementation Plan:**
- **(A) Make staging explicit in TritonGPU IR**
  - Extend shared-memory alloc to include a stage dimension:
    - `ttg.local_alloc` (or equivalent) gains `(stages, stride, swizzle)` attributes.
  - Introduce/standardize async-copy ops in TritonGPU IR:
    - `ttg.async_copy_global_to_shared`
    - `ttg.async_commit`, `ttg.async_wait`
    - barrier ops (where supported) as first-class IR to enable scheduling.
- **(B) Extend swizzle synthesis to be stage-aware**
  - Modify the seed’s “optimal swizzling” construction to include stage as another contributor to bank conflicts:
    - avoid repeated bank hits not just within a stage, but across \(stage \rightarrow stage+1\) accesses in steady state.
  - Add a stride selection routine:
    - try carry-free power-of-two stride first,
    - fall back to \(\mathbb{Z}_{2^k}\)-affine stage offsets if shared memory budget would explode.
- **(C) Scheduling pass**
  - Add a `TritonGPU/Transforms/PipelineSchedule` pass that:
    - assigns ops to stages,
    - places barriers/waits,
    - reorders compute vs async copies to overlap latency,
    - optionally unrolls by a small factor to expose ILP.
- **(D) Backend lowering**
  - NVIDIA:
    - emit async-copy instructions when the IR pattern matches,
    - otherwise degrade to vectorized `ld.global` + `st.shared` but still benefit from stage-aware layout/swizzle.
  - AMD:
    - target available async DS/VMEM idioms when possible; otherwise preserve correctness and focus on bank-conflict reduction + vectorization.

**4. Evaluation Plan (Real Hardware):**
- **Baselines**
  - Legacy Triton
  - Seed Triton-Linear (no explicit pipeline/layout co-design)
  - Optional: a reference tuned CUDA kernel for GEMM/attention where you have one (do not require it for all tests)
- **Microbenchmarks**
  1. **Copy+MMA pipeline kernel** with controllable stages \(S=2..5\):
     - measure overlap (latency reduction) and sensitivity to stride/skew.
  2. **Bank-conflict stress** across stages:
     - fixed access pattern, vary stride/skew; measure throughput cliffs.
- **TritonBench / E2E**
  - `gemm`, `int4_gemm`, `fp8_gemm`, `template_attention`, `flex_attention`
- **Metrics**
  - Latency, throughput
  - Register count + spills (to validate the schedule isn’t “winning on paper” but losing occupancy)
  - Compilation time and binary size (pipeline unrolling can blow both up)
- **Hardware**
  - NVIDIA H100 (exercise descriptor/bulk-copy paths if present)
  - NVIDIA A100 (exercise cp.async-like paths)
  - AMD MI300 (or closest available ROCm platform)

---

### Gap 3: **Bitpacking-Aware Layout Algebra + Pressure-Aware Conversion Planning (Register-Only Permute Networks)**
**1. The Problem:**
- Stage-1 identified a missing SOTA technique: **within-register subword permutes** (byte/nibble rearrangements) used heavily in low-precision kernels to avoid shared memory.
- The seed paper’s conversion machinery is strong for:
  - register permutations at *element granularity*,
  - warp shuffles for cross-lane exchange,
  - shared-memory swizzle for bank conflicts.
- But it doesn’t explicitly model **bit/byte packing** inside registers, so the compiler can’t reliably synthesize:
  - register-only repacking for INT4/FP8 fragments,
  - minimal permute networks that match MMA fragment requirements.
- Also, Stage-1 called out a real performance risk: swap shared-memory conversions for shuffles/permutes and you may hit **register pressure cliffs** (occupancy drop, spills).
- **Manual baseline to beat:** PTX/ISA-tuned kernels that:
  - keep lowp repacking in registers (permute/shifts/boolean ops),
  - minimize cross-lane traffic,
  - pick strategies based on pressure and schedule.

**2. The Proposed Solution (Software/Math):**
- **Algebraic Extension: Bit-Layouts as \(\mathbb{F}_2\)-linear maps on word bits.**
  - Represent each 32-bit (or 64-bit) register value as a vector in \(\mathbb{F}_2^{32}\) (or \(\mathbb{F}_2^{64}\)).
  - Packing/unpacking/reordering of subwords becomes a linear transform \(P\) (often a permutation matrix) on bit vectors.
  - Cross-register bit moves are captured by operating on concatenated bitspaces.
- **Compiler Pass: Two-level conversion synthesis**
  - **Level 1:** decide the *data movement topology* (intra-thread permute vs cross-lane shuffle vs shared).
  - **Level 2:** for intra-thread permutes, synthesize a **target-specific permute network** from \(P\).
- **Pressure-aware planner**
  - Use a static estimator (SSA liveness + temp count) to choose among:
    - shared-memory conversion (low regs, high mem),
    - warp shuffles (medium regs),
    - bit-permute network (low mem, may increase ALU/temps).

**3. Implementation Plan:**
- **(A) IR additions**
  - Add `ttg.bit_permute` (or `ttg.subword_permute`) op with:
    - input values (one or more regs),
    - a compile-time permutation descriptor (e.g., byte-lane mapping, nibble mapping).
  - Add canonicalization patterns to rewrite common `bitcast`/`reshape`/`arith` sequences into `bit_permute`.
- **(B) Analysis extensions**
  - Extend layout metadata to include **packing state**:
    - “this tensor’s element \(e\) lives in bits \([b:b+w)\) of register \(r\)”
  - When converting between two packed layouts, compute the required bit permutation matrix \(P\).
- **(C) Backend lowering**
  - NVIDIA path:
    - lower `bit_permute` to a small instruction basis (byte permutes + shifts + boolean ops) via LLVM intrinsics or inline PTX.
    - keep a correctness-first fallback: if \(P\) too complex, revert to shared memory.
  - AMD path:
    - lower to AMDGPU permute/shuffle intrinsics where available; otherwise use vector ops.
- **(D) Pressure-aware selection**
  - Implement a heuristic cost model at `ttg.convert_layout`:
    - estimate extra live temps for each strategy,
    - estimate instruction count,
    - estimate occupancy impact from reg count (simple model is sufficient for a first paper),
    - choose strategy minimizing predicted time.

**4. Evaluation Plan (Real Hardware):**
- **Baselines**
  - Legacy Triton
  - Seed Triton-Linear (no bitpacking permute synthesis; conversions may go through shared/shuffles)
  - Optional: CUTLASS-based or other tuned CUDA kernel for int4/fp8 GEMM if you already have it
- **Microbenchmarks**
  1. **INT4/FP8 repack microkernel**: load packed lowp, repack into MMA-friendly fragments, measure throughput.
  2. **Conversion strategy sweep**: force shared vs shuffle vs permute; measure regs/spills and runtime.
- **TritonBench / E2E**
  - `int4_gemm`, `fp8_gemm`, `bf16xint16_gemm`, and attention kernels that include lowp paths.
- **Metrics**
  - Latency, TFLOPS
  - Register count + spills, achieved occupancy
  - Compilation time + binary size (permute synthesis should not explode code size)
- **Hardware**
  - NVIDIA H100 + RTX4090 (cross-check SM90 vs consumer behavior)
  - AMD MI300 (or closest ROCm target available)

---

#### Quick “where it lands in Triton” map (so the proposals stay compiler-real)
| Gap | Primary ops impacted | Primary backend target |
|---|---|---|
| Gap 1 (Domain/mask) | `tt.load/store(mask=)`, `ttg.convert_layout`, `ttg.local_load/store`, `tt.gather` | correctness + fast-path shuffles |
| Gap 2 (Pipeline) | shared alloc, async-copy ops, `tt.dot` scheduling, `convert_layout` placement | cp.async/TMA/async DS + scheduling |
| Gap 3 (Bitpacking) | `bitcast`/packing patterns, `convert_layout` for lowp, MMA operand prep | register-only permute networks |

If you want, I can turn these into a concrete **12-week implementation plan** with a patch breakdown (analysis plumbing → legality checks → lowering → benchmarks), plus a minimal CI suite focused on the exact failure modes Stage-1 called out (masked shuffles, ragged tails, lowp repacking).