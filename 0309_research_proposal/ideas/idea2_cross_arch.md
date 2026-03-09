## 1. Title

**ATLAS: Proof-Carrying Cross-GPU Layout Translation via a Universal Relational-Linear IR**

## 2. Abstract

Tensor-core kernels are not portable because their layouts entangle algorithmic semantics with vendor-specific facts: warp or wave size, fragment register maps, bank swizzles, and async staging. I propose **ATLAS**, a retargetability layer that separates these concerns. ATLAS defines a vendor-neutral layout IR whose objects are hierarchical named-axis spaces and whose morphisms are piecewise Presburger relations plus F2-linear bit transforms. CuTe nested layouts, Triton linear layouts, and vendor tensor-core fragment maps all embed into this IR. Translation is performed by first lowering a source kernel to a semantic contract, then synthesizing a target layout in the fiber of all legal layouts that realize the same contract. Coarse tiling, inversion, and atom decomposition are solved with ISL relation composition/division; low-bit swizzles are solved exactly with F2 linear algebra. ATLAS emits a compact proof certificate showing that the translated kernel preserves logical read/write, replication, and reduction interfaces, while a cost model chooses among semantically equivalent target realizations. The result is not another per-backend rewrite of Triton, but a proof-carrying transport layer that can move kernels optimized for Hopper/Blackwell CuTe into AMD MFMA or Intel DPAS backends, including FlashAttention-class kernels, with minimal manual layout redesign.

## 3. Key Insight / Thesis Statement

**Portability fails because current systems treat layouts as backend syntax.** If layouts are lifted into an architecture-independent semantic category, then cross-architecture translation becomes selecting a new physical realization of the same semantic object. Correctness is therefore a theorem of functoriality plus relation equality; performance becomes a constrained optimization problem over the target architecture’s legal realization space.

## 4. Technical Approach

ATLAS has one core idea: **separate “what logical tile/fragments a kernel stage means” from “how a given GPU realizes that tile in lanes, registers, banks, and memory.”** Today, those are fused together inside CuTe templates, Triton backends, or handwritten kernels. ATLAS inserts a new layer in between.

### 4.1 Universal layout IR

ATLAS defines a category `U` of vendor-neutral layout objects.

- **Objects** are hierarchical spaces of named axes: logical axes (`m,n,k,batch,head,seq`), physical axes (`cta, warpgroup, warp/wave/subgroup, lane, reg, bank, mem`), and stage axes (`pipe, stage, copy-slot`).
- **Morphisms** are normalized as  
  `L = (D, R, O, X, G)`  
  where:
  - `D`: sharding/distribution of logical axes onto hardware axes,
  - `R`: replication/broadcast,
  - `O`: affine offset/tile/stride structure,
  - `X`: low-bit swizzle/permutation as an F2-linear map,
  - `G`: guards/domain predicates for tails, padding, legality.

`D/R/O/G` are represented as ISL relations. `X` is a binary matrix (or affine bit map) over selected low-order bits. This gives a strict split between:
- **coarse integer structure**: tiling, partitioning, complements, inversion,
- **fine bit structure**: XOR swizzles, bank permutations, lane shuffles.

This IR directly subsumes the cited systems:
- **CuTe** lowers into the hierarchical object language plus ISL relations.
- **Triton linear layouts** become a special case where much of the action is in `X`.
- **Axe D/R/O** becomes the normal form for the physical mapping component.
- **Power-of-2 swizzles** are exact in F2; non-power-of-2 tails remain in guarded Presburger pieces.

The key theoretical move is to define **architecture-independent semantic observation**. Each concrete physical layout has an observation morphism that answers: *which logical tensor elements does this lane/register/bank instance represent?* Two layouts are equivalent if they induce the same observation up to permitted replication/permutation.

### 4.2 Architecture descriptors as functors, not handwritten backends

For each architecture `A` (Hopper, Blackwell, AMD MFMA generation, Intel DPAS generation), ATLAS defines a small declarative descriptor:

- hardware axis topology,
- legal subgroup sizes,
- tensor-core atom library,
- shared-memory/LDS/SLM bank structure,
- async copy/barrier primitives,
- resource limits and legality predicates.

Each tensor-core instruction is modeled as a typed atom:

`Atom_A : LogicalTile -> (subgroup, lane, reg, issue-slot)`

So WGMMA, MFMA, and DPAS are all represented in the same language; they differ only in the descriptor.

This is crucial: **retargeting is not “rewrite the Triton backend.”** Instead, ATLAS lowers a tuned kernel into `U`, then re-realizes it using a different descriptor. In other words, it retargets optimized kernels **sideways**, not generic loops **downward**.

### 4.3 Translation as search in a target fiber

Given a source kernel stage with layout `L_s` on architecture `S`, ATLAS computes its semantic contract:

`C = Obs_S ∘ L_s`

where `Obs_S` maps source physical coordinates back to logical meaning.

For a target architecture `T`, ATLAS searches the fiber:

`F_T(C) = { L | Obs_T ∘ L = C and Legal_T(L) }`

This is the mathematically clean part: all objects in `F_T(C)` are semantically correct target realizations of the same source stage.

The search is structured, not brute force:

1. **Normalize source layout algebraically.**  
   Use categorical equalities and equality saturation to canonicalize compositions, products, divisions, complements, and inverses before synthesis.

2. **Factor by target atoms.**  
   Use categorical product/division to ask whether the source semantic tile can be realized as a tiled product of target atoms.  
   Example: a Blackwell warpgroup MMA tile may divide into multiple AMD MFMA wave tiles or Intel DPAS subgroup tiles.

3. **Solve coarse structure with ISL.**  
   ISL handles the integer part: subgroup partitioning, tile nesting, CTA/workgroup shapes, memory offsets, staging layouts, inversion, complements, and tail guards.

4. **Solve fine swizzles with F2.**  
   Once the coarse tile shape is fixed, low-bit bank/lane/register permutations become a linear system over F2.  
   This is where NVIDIA SMEM XOR swizzles, AMD LDS bank remaps, and Intel subgroup permutations unify cleanly.

5. **Choose among equivalent realizations with a cost model.**  
   Since every candidate in `F_T(C)` is already correct, optimization is safe. The cost model can use bank conflicts, coalescing, occupancy, register pressure, tensor-core throughput, and async overlap without threatening correctness.

6. **Return proof or impossibility.**  
   If the fiber is empty under the current resource budget, ATLAS returns an **impossibility certificate** rather than silently generating poor code. That is important for cross-generation portability: the compiler can prove that a Blackwell-specific fragment layout cannot be realized on a target without an additional transpose, extra staging, or relaxed schedule constraint.

### 4.4 Proof-carrying translation

ATLAS should generate a compact certificate per stage:

- semantic equivalence: `Obs_T ∘ L_t = Obs_S ∘ L_s`,
- legality: alignment, bank constraints, barrier domain correctness, no out-of-bounds,
- optional reduction-order invariant if exact FP reproducibility is requested.

Because layout composition is functorial, stage-level certificates compose into whole-kernel certificates. This enables a **small checker** in the compiler or even external validation. A PLDI-strength version would mechanize the core theorems in Lean/Coq and ship proof terms or checkable certificates.

### 4.5 FlashAttention-4 retargeting case study

This is the killer demo.

A Blackwell FA-4 kernel written in CuTe-DSL uses Blackwell-specific ingredients: 2-CTA MMA mode, TMEM staging, asynchronous pipelines, and highly tuned fragment layouts. ATLAS lowers that kernel into semantic contracts for the Q/K/V tiles, accumulator fragments, stage ordering, and sequence-block traversal.

Then:
- WGMMA fragment contracts are divided into target MFMA or DPAS atom products.
- TMEM-backed layouts are re-realized as LDS/SLM/register staging contracts.
- Blackwell low-bit swizzles are replaced by target-valid F2 solutions.
- Sawtooth wavefront reordering is represented as a permutation on the sequence-block axis and can transport unchanged because it is above hardware realization.

The result is not “FA-4 everywhere for free”; target-specific pipeline primitives still matter. But the hardest part—the fragment layout and memory-layout redesign—becomes automatic and provably correct.

## 5. Expected Contributions

- **A universal relational-linear layout IR** that subsumes CuTe hierarchical layouts, Triton linear layouts, and hardware-aware named-axis mappings.
- **A categorical formulation of cross-architecture layout transport** as semantic observation plus target-fiber realization.
- **A synthesis algorithm combining ISL and F2 algebra** for exact translation of coarse tilings and fine swizzles.
- **Proof-carrying kernel retargeting**, including impossibility certificates when no legal target realization exists.
- **The first compiler demonstration of sideways retargeting of hand-tuned tensor-core kernels** (e.g., FA-class kernels) across NVIDIA, AMD, and Intel.

## 6. Evaluation Plan

1. **Formal validation**
   - Prove faithful lowering from CuTe/Triton-style layouts into ATLAS.
   - Prove soundness of target synthesis.
   - Measure certificate size and checking time.

2. **Microbenchmarks**
   - Translate standalone fragment layouts and swizzles across NVIDIA/AMD/Intel.
   - Validate exact equivalence by exhaustive enumeration on small tiles and by symbolic equality on large ones.

3. **Kernel suite**
   - GEMM, grouped GEMM, FP8/BF16/INT8 kernels, FlashAttention-2/4-style kernels, MoE expert kernels, quantized epilogues.
   - Sources: CuTe/CUTLASS, Triton, and at least one vendor-specific hand-tuned kernel family.

4. **Platforms**
   - NVIDIA Hopper + Blackwell,
   - AMD MI300/MI400-class MFMA GPUs,
   - one Intel DPAS-capable Xe platform.

5. **Baselines**
   - Handwritten per-vendor kernels,
   - native Triton backends,
   - vendor libraries (CUTLASS/cuDNN, CK/Tensile, oneDNN/XeTLA-like stacks).

6. **Metrics**
   - Translation success rate,
   - performance vs hand-tuned target code,
   - compile-time overhead,
   - lines of backend-specific layout code eliminated,
   - effort to add a new architecture descriptor.

A strong success criterion would be: most kernels retarget automatically, translated kernels reach near-handwritten performance on the majority of cases, and new architecture support requires only new descriptors/atom specs rather than kernel rewrites.

## 7. Target Venue and Why

**PLDI**.

The core novelty is a new formal IR, a categorical correctness story, and a proof-carrying compilation algorithm—not just a faster kernel or a new hardware optimization. The cross-vendor performance study is essential, but the paper’s central claim is that optimized GPU layouts can be transported with machine-checkable correctness. That is squarely PLDI-shaped, with enough systems depth to be compelling.

## 8. Potential Weaknesses and Mitigations

- **Weakness: correctness is only as good as the hardware descriptor.**  
  **Mitigation:** make the result explicitly *spec-relative*, and derive descriptors from vendor docs plus black-box microbench validation.

- **Weakness: some performance-critical differences are about schedule/pipeline, not layout.**  
  **Mitigation:** keep ATLAS centered on layout correctness, but pair it with a smaller dependence-preserving schedule search layer. Even if schedule tuning remains target-specific, the hardest manual redesign work disappears.

- **Weakness: search space may explode.**  
  **Mitigation:** normalize first, factor by target atoms, separate ISL and F2 solving, memoize common contracts, and use impossibility certificates to prune early.

- **Weakness: exact semantic preservation may conflict with FP reassociation changes.**  
  **Mitigation:** offer two modes: strict reduction-order preservation, or numerically relaxed transport with explicit error bounds.

- **Weakness: some source kernels may rely on features with no target analogue (e.g., TMEM-like behavior).**  
  **Mitigation:** let ATLAS either synthesize a semantically equivalent staged realization using extra buffers/transposes, or prove that no realization exists within the requested resource budget. That negative result is itself valuable.
