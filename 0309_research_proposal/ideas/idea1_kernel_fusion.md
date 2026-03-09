## 1. Title

**CuTeFuse: Algebraic Kernel-Boundary Elimination for Automatic Fused GPU Pipelines**

## 2. Abstract

Current GPU fusion systems decide fusibility mostly from operator syntax (e.g., pointwise-after-GEMM) rather than from whether the producer’s tile layout can be consumed directly on-chip by the next kernel. This leaves major opportunities unrealized for fusion across “hard” boundaries such as GEMM→softmax→GEMM, routing→packing→grouped GEMM, and other Transformer/MoE patterns. I propose **CuTeFuse**, a compiler that treats every kernel as a **layout transducer** from logical tensor coordinates to hardware-time coordinates: CTA, warpgroup, warp, lane, register/shared/TMEM location, and pipeline stage. For each intermediate tensor, CuTeFuse uses **CuTe composition, division, and complement** to derive a **fusibility certificate**: if producer and consumer layouts factor through a common on-chip carrier layout, and the residual complement is realizable with local transforms (register permutation, warp shuffle, shared-memory transpose/swizzle, TMEM relay, barriers), then the boundary can be eliminated with no HBM round-trip. A **Hexcute-style constraint solver** then synthesizes the fused thread-value layouts, storage layouts, and asynchronous pipeline schedule under occupancy and resource constraints. This reframes kernel fusion as **algebraic elimination of intermediates**, not pattern matching, and could automatically recover FlashAttention-4-like pipelines while discovering new fused kernels in Transformer and MoE workloads.

## 3. Key Insight / Thesis Statement

**A kernel boundary is eliminable iff the producer output layout and consumer input layout admit a factorization through a common on-chip carrier layout, with residual layout complement realizable by local on-chip transforms under a valid temporal pipeline.**  
Composition proves connectivity, division extracts the shared tile basis, complement quantifies exactly what local movement remains, and Hexcute-style synthesis constructs the fused implementation. In this view, **FlashAttention-4 is a hand-written fusibility certificate** for GEMM→online softmax→GEMM on Blackwell; the research goal is to make such certificates automatic.

## 4. Technical Approach (detailed)

### 4.1 Kernel IR: from operator semantics to layout transducers

The core shift is to stop representing a kernel only as “an op” and instead represent it as a **layout transducer**. For every tensor boundary, a kernel exposes:

- a logical index domain,
- a producer/consumer layout over hardware axes,
- a dependence relation for reductions,
- a temporal availability relation across pipeline stages.

Formally, for a tensor \(T\), a lowered kernel variant defines a mapping

\[
L_T : I_T \rightarrow H \times S
\]

where \(I_T\) is the logical index space, \(H\) is a product of hardware axes (CTA, warpgroup, warp, lane, register/shared/TMEM address, optionally gpuid), and \(S\) is pipeline stage/time. This makes layout a first-class type, not a post-hoc codegen detail.

This is crucial because many “unfusible” boundaries are only unfusible in one layout. The same computation can become fusible if the producer emits tiles in a basis the consumer can directly reuse.

### 4.2 Fusibility certificates from CuTe algebra

For an intermediate tensor \(X\) between producer \(K_p\) and consumer \(K_c\), define:

- \(P_X\): producer output layout of \(X\),
- \(C_X\): consumer input layout of \(X\),
- transfer relation \(R_X = C_X \circ P_X^{-1}\).

Today’s compilers would ask whether \(K_p\) and \(K_c\) match a known fusion pattern. CuTeFuse instead asks whether \(R_X\) can be factorized through an on-chip carrier.

The key algebraic object is a **fusibility certificate**:

\[
\mathcal{F}(X) = (B, U_p, U_c, M, \sigma)
\]

such that:

1. **Division extracts a common basis** \(B\): both layouts factor through a shared tile basis over the intermediate domain.
2. **Complement yields the residual transforms**:
   - \(U_p = \mathrm{comp}(P_X, B)\)
   - \(U_c = \mathrm{comp}(C_X, B)\)
3. The residual transfer \(U_c \circ U_p^{-1}\) lies in the closure of a target-specific local transform set
   \[
   \mathcal{L} = \{\text{identity, reg permute, warp shuffle, smem transpose/swizzle, TMEM relay, barrier, multicast, 2-CTA exchange}\}
   \]
4. \(\sigma\) is a valid stage assignment satisfying dependence, lifetime, register, shared-memory, TMEM, and occupancy constraints.

If such a certificate exists, the boundary is eliminable with no HBM materialization. If it does not, the intermediate must be materialized or partially materialized.

This is the formal “iff” criterion, with one caveat: it is exact for tractable/static layout classes and conservative for highly dynamic irregular layouts.

### 4.3 Global discovery: a layout-compatibility hypergraph

Pairwise fusibility is not enough. A fusion that is locally good can be globally bad because it increases live state, reduces occupancy, or blocks a better downstream fusion. So CuTeFuse builds a **layout-compatibility hypergraph**:

- **Nodes**: operator instances with candidate lowerings/layouts.
- **Edges**: materialized boundaries.
- **Fusion hyperedges**: candidate fused regions whose internal boundaries all admit composable fusibility certificates.

Each edge/hyperedge gets a cost model:

- HBM bytes if materialized,
- on-chip complement cost (shuffle count, smem traffic, barriers),
- register pressure and occupancy penalty,
- async pipeline balance,
- tensor-core issue efficiency,
- launch overhead.

For chain-like blocks (common in Transformer subgraphs), optimal fusion can be solved with dynamic programming. For general DAGs with fan-in/fan-out, use MILP or A*-guided hypergraph partitioning. The result is a **global fusion plan**, not greedy local fusion.

This is where the proposal becomes non-incremental: fusion is no longer an epilogue heuristic but a graph optimization over algebraically certified boundary eliminations.

### 4.4 Hexcute-style synthesis of the fused kernel

Once a candidate fused region is selected, CuTeFuse invokes a **Hexcute-like solver** to synthesize the actual implementation. The solver jointly chooses:

- thread-value layouts,
- warpgroup/CTA tilings,
- shared-memory and TMEM layouts,
- swizzles/transposes,
- stage count and double/triple buffering,
- warp specialization,
- reduction decomposition,
- barrier placement,
- async copy / TMA / async MMA ordering.

The constraints come from three places:

1. **algebraic constraints** from the fusibility certificates,
2. **hardware constraints** from instruction shapes, bank conflicts, register file size, TMEM capacity, and occupancy,
3. **semantic constraints** from numerics and dependence legality (e.g., stable online softmax, reduction associativity, predicate masks).

The objective is not merely legality; it is to minimize a predicted runtime proxy:
\[
T \approx \max(T_{TC}, T_{HBM}, T_{SMEM}, T_{SFU/FMA}) + T_{sync} + T_{launch}
\]
with learned residual correction.

### 4.5 Automating FlashAttention-4-style pipelines

The flagship demonstration is attention. For GEMM\(_1\) \(QK^T\) → online softmax → GEMM\(_2\) \(PV\), current systems usually materialize the score matrix or rely on hand-written kernels. In CuTeFuse:

- **division** identifies a row-stationary common basis across score production and softmax consumption,
- **complement** reveals that the remaining work is a local row reduction plus normalization state,
- the solver discovers that score tiles can live in TMEM/register space while max/sum statistics live in registers,
- the second GEMM can consume normalized probability fragments without HBM traffic.

The same framework can choose:
- fully async MMA pipelines,
- software-emulated exponential on FMA units when SFUs bottleneck,
- 2-CTA cooperative modes when reuse favors it.

In other words, manual FA-4 pipeline design becomes an automatically discovered consequence of the algebra + solver. A further extension is to treat traversal order itself as a temporal layout axis, allowing search over wavefront orders that improve cache reuse.

### 4.6 Beyond attention

The same machinery applies to:

- **MLP blocks**: GEMM → bias/activation → GEMM,
- **MoE**: routing scores → top-k/packing → grouped GEMM,
- **norm-heavy blocks**: GEMM → RMSNorm/LayerNorm → residual add,
- **distributed kernels**: if gpuid is modeled as a layout axis, sharding/replication can be fused with on-device layouts.

The unifying principle is always the same: **intermediate tensors are eliminated when their boundary layout has a certificate**.

## 5. Expected Contributions

- **A formal fusibility theory** based on CuTe composition, division, and complement, extended with temporal/on-chip carrier constraints.
- **Fusibility certificates** as a new compiler abstraction for deciding when an intermediate tensor can be eliminated.
- **A global layout-compatibility search algorithm** that selects multi-op fused regions beyond simple epilogues.
- **A Hexcute-style synthesis engine** that generates fused layouts and async pipelines automatically, potentially rediscovering FlashAttention-4-class kernels.
- **The first automatic fusion framework for GEMM→reduction→GEMM patterns** driven by layout compatibility rather than operator templates.

## 6. Evaluation Plan

### Workloads
- Transformer blocks: attention, MLP, norm/residual.
- Long-context attention and decode/prefill.
- MoE models: routing + packing + grouped GEMM.
- Selected training kernels and inference kernels.

### Hardware
- H100 and B200/Blackwell-class GPUs.

### Baselines
- TorchInductor/Triton fusion.
- CUTLASS/cuBLASLt epilogue fusion.
- TensorRT/XLA-style graph fusion.
- Hand-written kernels: FlashAttention-3/4, expert kernels where available.

### Metrics
- End-to-end throughput and latency.
- Kernel count and launch overhead.
- HBM bytes written/read per block.
- Tensor core utilization, occupancy, register/smem/TMEM usage.
- Compile-time overhead and search time.
- % of hand-designed fused kernels automatically rediscovered.

### Ablations
- No complement reasoning.
- No temporal/stage axis.
- No global hypergraph search.
- No Hexcute synthesis (template-only).
- No Blackwell-specific carrier choices.

### Case studies
- Whether the system rediscovers FA-4-style attention.
- Whether it finds new fusions in MoE routing/packing not encoded by hand today.

## 7. Target Venue and Why

**Primary target: PLDI.**  
The central novelty is a new formal compiler abstraction—kernel fusion as algebraic elimination of layout boundaries—plus a synthesis system and strong GPU evaluation. The work has the kind of semantic/formal core PLDI values, while still delivering systems impact.

## 8. Potential Weaknesses and Mitigations

- **Weakness: search-space explosion.**  
  **Mitigation:** canonicalize layouts, memoize certificates, use hierarchical search (pairwise prune → region synthesis), and learn residual cost-model corrections.

- **Weakness: algebra captures layout, not all numerical legality.**  
  **Mitigation:** add a semantic effect system for reductions/nonlinear ops, with conservative guards and stable online formulations.

- **Weakness: long fused regions may lose occupancy due to register pressure.**  
  **Mitigation:** allow partial fusion/materialization and optimize globally, not greedily.

- **Weakness: irregular MoE/top-k permutations may break exact certificates.**  
  **Mitigation:** support conservative segmented/block-local fusion and guarded multiversioning for dynamic sparsity.

- **Weakness: some wins may be hardware-specific (e.g., TMEM, 2-CTA MMA).**  
  **Mitigation:** parameterize the carrier transform set by target; degrade gracefully to register/smem-only fusion on older GPUs.

If you want, I can turn this into a **1-page extended abstract**, a **PLDI-style intro**, or a **full paper outline with theorem statements**.
