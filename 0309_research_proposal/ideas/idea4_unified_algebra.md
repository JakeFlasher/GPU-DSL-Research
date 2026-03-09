## 1. Title

**ATLAS: Algebraic Tensor Layouts Across Scales — One Layout Algebra from Threads to Clusters**

## 2. Abstract

Current ML compilers split layout optimization into two disconnected problems: intra-kernel mapping of tensor coordinates to registers/SMEM/TMEM, and inter-kernel or distributed mapping of tensor axes to GPUs, collectives, and pipeline stages. **ATLAS** proposes that these are the same mathematical object: a multiscale mapping from logical tensor coordinates to a stratified hardware space-time. ATLAS extends CuTe’s hierarchical layout algebra and Axe’s named-axis shard/replica/offset decomposition into a single executable formalism whose morphisms are mixed affine/bitwise relations over named nested axes indexed by hardware scope (lane, warp, CTA, cluster, GPU, memory, time). A tensor layout is no longer just a function but an **annotated relation**, allowing replication, caching, sparse dispatch, and reductions to be expressed uniformly. This yields a normal form that factors layouts across hierarchy cuts and lets a compiler jointly choose swizzles, tiling, TMEM/SMEM/L2/HBM residency, GPU sharding/replication, and pipeline placement. Lowered through ISL-backed reasoning to CuTe/CUTLASS/Triton kernels plus NCCL/NVSHMEM communication, ATLAS would unify kernel generation and distributed planning and expose new cross-level optimizations, e.g. trading inter-GPU replication for larger TMEM tiles or co-optimizing sequence partitioning with L2-aware CTA wavefront order.

## 3. Key Insight / Thesis Statement

**Kernel layout synthesis, memory placement, distributed sharding, and pipeline scheduling should not be separate compiler passes. They are all factorization choices of one multiscale layout relation from logical tensor coordinates to physical resources.**

The non-incremental step is to redefine “layout” from a local address map into a **named, hierarchical, space-time relation** that can express ownership, replication, reduction, and residency at every level: register, warp, CTA, cluster, GPU, and multi-GPU.

## 4. Technical Approach

### 4.1 Formal core: extend CuTe/Axe into one indexed algebra

ATLAS starts from the observation that CuTe and Axe differ more in **scope** than in **mathematical structure**. Both describe mappings from logical axes to physical resources. The missing piece is a common formal object that spans all levels.

Formally, let \(H\) be a hardware hierarchy graph with execution scopes  
`lane < warp < CTA < cluster < GPU < node`,  
storage scopes  
`reg, TMEM, SMEM, DSMEM, L2, HBM, peer-HBM`,  
and a first-class time scope  
`stage/time`.

We extend the categorical `Tuple/Nest` view into an indexed category **Nest\_H**:

- **Objects**: named nested coordinate spaces tagged by scope in \(H\).  
  Example: `(seq, head, expert, hidden)` on the logical side; `(gpu, cluster, cta, warp, lane, tmem_col, smem_bank, time)` on the physical side.
- **Morphisms**: mixed affine/bitwise relations, not only functions.  
  This is crucial: a logical element may map to multiple physical images because of caching, replication, multicast, or reduction.
- **Executable representation**: each morphism is materialized as an ISL-style relation with affine, modular, and bit-level constraints, so CuTe stride layouts and Triton-style swizzles live in the same backend.

This embeds prior systems as strict subcases:

- **CuTe** = single-device, mostly affine, hierarchical layout morphisms.
- **Axe** = named-axis shard/replica/offset morphisms at device/memory scope.
- **Linear layouts/swizzles** = bit-linear offset maps.
- **Hexcute** = local synthesis of legal morphisms from type constraints.

### 4.2 The key abstraction: ownership, residency, and address are one factorization

Every ATLAS layout \(L\) is factored as:

\[
L = O \circ R \circ D
\]

where:

- **D (Distribute/Own)** chooses the canonical owner of each logical coordinate along hierarchy cuts.  
  This covers tensor parallelism, sequence parallelism, expert parallelism, CTA assignment, warp partitioning.
- **R (Replicate/Reside)** creates additional physical images.  
  This is the critical unification step: **caching in SMEM/L2 and replication across GPUs are the same operator applied at different cuts**.
- **O (Offset/Order)** maps owned or replicated coordinates into concrete local addresses, swizzles, bank mappings, TMEM fragments, wavefront order, etc.

This gives a new theoretical lens:

- Replicating a hot MoE expert across 4 GPUs and caching a K/V tile in SMEM are both `R`.
- An all-gather is the inverse of a shard `D`; a reduce-scatter is its monoidal adjoint.
- Sawtooth wavefront reordering is an `O` transform on `(cta, time)` axes.
- Pipeline parallelism is `D` over `layer -> stage` plus time shifts in `O`.

So the paper’s central theorem would be a **Replica–Caching Duality**: locality management and distributed replication are instances of the same algebraic operation over different hierarchy cuts.

### 4.3 Normal form and tractability

To make this compilable, ATLAS introduces an **adjacent-cut normal form**:

\[
L = \prod_{e \in E(H)} (O_e \circ R_e \circ D_e)
\]

where each factor acts only across one adjacent hierarchy edge, e.g.:

- lane ↔ warp
- warp ↔ CTA
- CTA ↔ cluster
- cluster ↔ GPU
- GPU ↔ node
- reg/TMEM/SMEM ↔ L2 ↔ HBM ↔ peer-HBM

This matters because it turns a massive global search into a composition of small frontier problems. The associated tractability result would extend the categorical “tractable layouts” story: if each cut exposes bounded frontier rank, then composition, inversion, and complement are exponential only in frontier size, not in total tensor rank or cluster size.

That gives a principled reason why end-to-end optimization is feasible for ML workloads: logical tensors are large, but the number of axes crossing any one cut is typically small.

### 4.4 Compiler architecture: one IR, two solvers, many backends

ATLAS would be implemented as an IR where every tensor carries an **ATLAS type**:

- logical named axes
- scope-tagged physical axes
- D/R/O factorization
- legality constraints
- access semantics: read-only, reduction, mutable
- optional reduction monoid: sum/max/etc.

The compiler then performs:

1. **E-graph rewriting over layout algebra**  
   Saturate using CuTe laws (compose/divide/coalesce/complement), Axe laws (shard/replica/offset), and collective adjunction laws. This enumerates equivalent distributed/kernel plans in one space.

2. **Global coarse solve for D/R across cluster/GPU/time**  
   Use ILP/CP-SAT or dynamic programming to choose:
   - shard vs replicate
   - pipeline stage boundaries
   - collective forms
   - cluster placement
   - soft-residency targets (L2/peer copies)

3. **Local fine solve for O within GPU**  
   Given cut frontiers, use Hexcute-style constraint solving to synthesize:
   - thread-value layouts
   - SMEM/TMEM layouts
   - bank swizzles
   - MMA fragment mappings
   - CTA traversal order

4. **Iterative refinement with a compositional cost model**  
   Cost is accumulated per hierarchy edge:
   - bytes moved
   - messages/synchronizations
   - reuse distance
   - occupancy
   - bank conflicts
   - spills
   - NVLink/cluster contention

The important systems insight is that these are no longer separate cost models. Traffic across `warp→SMEM` and `GPU→NVLink` are computed from the same boundary-volume logic over the same layout object.

### 4.5 Why this enables qualitatively new optimizations

**MoE:** Today, expert placement and expert kernel layout are chosen by separate systems. ATLAS can ask: should I replicate the hottest experts across GPUs so each local expert GEMM can use a larger TMEM tile and avoid token exchange? That is a single D/R/O tradeoff.

**Long-context attention:** Sequence partitioning across GPUs, CTA wavefront order, and L2/TMEM residency are usually tuned independently. ATLAS represents all three as one layout over `seq × gpu × cta × time`, so it can jointly search inter-GPU sequence shards and intra-GPU sawtooth schedules.

**Pipeline + tensor parallel:** `hidden -> gpu`, `layer -> stage`, and `microbatch -> time` become the same layout factorization problem. The compiler can decide whether to resolve reductions at cluster scope, GPU scope, or pipeline boundaries.

## 5. Expected Contributions

- **A new formalism:** the first executable algebra that unifies CuTe-style intra-kernel layouts, Axe-style distributed layouts, and ISL/linear-layout reasoning in one framework.
- **A new theorem:** adjacent-cut normal form plus an extended tractability characterization for multiscale layouts.
- **A new compiler abstraction:** tensor types that carry ownership, replication/residency, and local address/order together.
- **A new optimization capability:** joint search over swizzling, tiling, cache residency, sharding, replication, collectives, and pipeline placement.
- **A new empirical result class:** cross-level plans that existing split compilers cannot even express, especially for attention and MoE.

## 6. Evaluation Plan

**Implementation**
- Build an `AtlasIR` dialect in MLIR or a CuTe-DSL-adjacent compiler.
- Lower intra-GPU factors to CuTe/CUTLASS/Triton-like codegen.
- Lower inter-GPU factors to NCCL/NVSHMEM/P2P schedules.

**Benchmarks**
- Microbenchmarks: GEMM, grouped GEMM, FlashAttention, MoE dispatch + expert GEMM, all-gather/reduce-scatter/all-to-all.
- End-to-end: long-context transformer inference, tensor-parallel MLP blocks, MoE inference/training, pipeline-parallel transformer stacks.

**Platforms**
- H100 for baseline portability.
- B200/GB200-class systems for TMEM, cluster, and multi-GPU hierarchy.

**Baselines**
- CUTLASS/CuTe-DSL
- Triton / linear-layout backends
- Hexcute-style local synthesis
- Megatron-style parallelism planning
- hand-tuned NCCL + kernel pipelines

**Metrics**
- Throughput / latency
- HBM, L2, SMEM/TMEM traffic
- NVLink bytes
- occupancy, spills, bank conflicts
- compile/search time
- cost-model accuracy

**Ablations**
- no D/R/O unification
- no time axis
- no cross-GPU optimization
- no intra-GPU synthesis
- no rewrite saturation

## 7. Target Venue and Why

**ASPLOS**.

This idea is strongest as a cross-layer paper: new formal abstraction, new compiler, and new hardware-aware optimization results on modern GPUs and multi-GPU systems. It is broader than a pure PL paper and deeper than a pure systems paper, which is exactly ASPLOS territory.

## 8. Potential Weaknesses and Mitigations

- **Weakness: search space explosion.**  
  **Mitigation:** adjacent-cut normal form, frontier-bounded DP, and staged D/R then O solving.

- **Weakness: L2 and network behavior are not fully software-controlled.**  
  **Mitigation:** model these as soft-residency scopes with profile-guided calibration rather than exact address spaces.

- **Weakness: too general for first implementation.**  
  **Mitigation:** restrict v1 to read-only + reduction-heavy tensor programs, where replication and collectives dominate.

- **Weakness: backend complexity.**  
  **Mitigation:** lower to existing proven pieces (CuTe/CUTLASS/NCCL) instead of building a new codegen stack from scratch.

- **Weakness: reviewers may see this as “just adding gpu axes to CuTe.”**  
  **Mitigation:** emphasize the real novelty: layouts become annotated space-time relations, making caching, replication, collectives, and pipeline placement algebraically identical operations.

If you want, I can next turn this into a **1-page paper pitch**, a **PLDI-style theorem + system outline**, or an **ASPLOS submission structure with figures**.
