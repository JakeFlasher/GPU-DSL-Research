## 1. Title

**Cache-as-Layout: A Complementary Reuse Calculus for Synthesizing Hierarchy-Optimal GPU Kernels**

## 2. Abstract

Modern GPU kernels are increasingly bottlenecked not by arithmetic tiling alone, but by whether their traversal order matches a deep memory hierarchy spanning L2, L1/shared-memory banks, registers, and Blackwell TMEM. Today, cache-friendly schedules such as Sawtooth wavefronts are discovered manually, while layout systems such as CuTe and Axe describe where data lives but not how execution should traverse it. We propose **Cache-as-Layout**, a formal compiler framework that treats each hierarchy level as a named layout and derives tiling, traversal, and swizzling directly from layout specifications. For an access layout \(L_{\text{acc}} = L_{\text{thread}} \circ L_{\text{data}}\), CuTe division by a cache-object tile yields a resident-object quotient; CuTe complement yields the eviction dimensions that determine reuse distance. This factorization produces symbolic miss, conflict, and residency formulas, enabling synthesis of loop orders, parity reversals, wavefront schedules, and low-bit swizzles under capacity and associativity constraints. Sawtooth emerges automatically as the 2D corollary when the complement of the L2-resident KV layout is a path over Q-block wavefronts. Using Axe named axes, the same calculus spans SMs, clusters, cache lines, banks, and TMEM tiles. The result is a compiler that generates cache-aware schedules with proof obligations, replacing large autotuning spaces with algebraic derivation.

## 3. Key Insight / Thesis Statement

**Thesis:** a GPU cache level is itself a layout object. Once addresses are projected onto named hierarchy axes, **CuTe division isolates what stays resident**, and **CuTe complement isolates what causes eviction**. Therefore, cache optimization is not fundamentally a search problem; it is a **layout factorization problem**.

**Non-incremental leap:** Sawtooth, bank swizzles, TMEM staging, and even cluster/SM ordering are all instances of the same theorem:  
> *Optimal traversal minimizes growth of the complementary layout of the resident-object quotient at each hierarchy level.*

## 4. Technical Approach

### A. Represent the cache hierarchy as a first-class layout

Let \(I\) be the dynamic iteration space of a kernel: CTA id, warp id, lane id, pipeline stage, loop tiles, wavefront ids, etc. Let:

- \(L_{\text{alg}} : I \rightarrow E\) map each dynamic execution point to logical tensor elements,
- \(L_{\text{data}} : E \rightarrow A\) be the CuTe/Axe data layout from elements to addresses.

For each hierarchy level \(h\), define a **hardware layout**:

\[
H_h : A \rightarrow \mathcal{R}_h \times \mathcal{O}_h
\]

where:

- \(\mathcal{R}_h\) is the resident-object identity at level \(h\),
- \(\mathcal{O}_h\) is the intra-object offset.

Examples:

- L2: \(\mathcal{R}_{L2} = (\text{slice}, \text{set}, \text{line})\)
- Shared memory: \(\mathcal{R}_{SMEM} = (\text{bank}, \text{row})\)
- TMEM: \(\mathcal{R}_{TMEM} = (\text{slot}, \text{phase}, \text{tile})\)

Then the access layout seen by hierarchy level \(h\) is:

\[
A_h = H_h \circ L_{\text{data}} \circ L_{\text{alg}}
\]

This is the core move: **cache state becomes an image of layout composition**. With Axe, these hierarchy coordinates become named axes, so the same representation spans lane/warp/SM/cluster/gpuid/memory uniformly.

---

### B. Use CuTe division and complement to expose reuse and eviction

For a cache-object tile \(T_h\) (e.g. cache line, bank row, TMEM slot), factor \(A_h\) into:

- **Resident quotient**  
  \[
  Q_h = A_h \div T_h
  \]
  which identifies which resident object an access touches,

- **Intra-object offset**  
  \[
  O_h = A_h \bmod T_h
  \]

Only \(Q_h\) matters for residency and eviction. Then compute:

\[
K_h = \operatorname{comp}(Q_h)
\]

where \(K_h\) is the **complementary layout**: the iteration directions not absorbed into the resident object. This is the key formal object. Intuitively:

- \(Q_h\) tells us **what stays the same** for a hit,
- \(K_h\) tells us **what changes and can evict it**.

This makes the user’s intuition precise: the reuse distance of a cache line is governed by the dimensions *not* being consumed inside that line/tile.

For tractable CuTe layouts, these operations remain symbolically computable. The optimizer then derives closed-form expressions for:

- live footprint over a prefix: \(|\mathrm{Im}(Q_h)|\),
- stack distance between reuses,
- set occupancy for set-associative caches,
- bank multiplicity for shared memory,
- stage occupancy for TMEM.

So cache analysis becomes a layout image-cardinality problem, not a trace-only heuristic.

---

### C. State the central theorem: Complementary Reuse Theorem

**Informal theorem.**  
For a tractable access layout \(A_h\) and fixed inner-tile execution order, the miss/conflict behavior at hierarchy level \(h\) is fully determined by the schedule over the complement \(K_h\). Among dependence-respecting schedules in the same legality class, the optimal schedule minimizes boundary growth of prefixes in \(K_h\), subject to capacity and associativity constraints on \(Q_h\).

This is powerful because it converts cache optimization into geometry over layout quotients.

#### Sawtooth as a corollary
FlashAttention’s KV reuse pattern becomes the canonical example:

- \(Q_{L2}\): identity of L2-resident KV tiles/lines,
- \(K_{L2}\): the Q-wavefront dimension that revisits those KV tiles.

A monotone KV sweep causes the next Q block to begin reuse from the far end of the prior sweep, maximizing stack distance. The frontier-minimizing schedule instead reverses the KV direction every other Q wavefront. That is exactly **Sawtooth**.

Under this proposal, Sawtooth is no longer a manually discovered pattern; it is the **2D serpentine solution** of the Complementary Reuse Theorem.

More importantly, the theorem predicts more than Sawtooth:

- 1D complements → monotone scans
- 2D complements → serpentine/boustrophedon
- higher-dimensional complements → Morton/Hilbert/Gray-like traversals depending on set/bank anisotropy

So the framework can derive patterns humans have not yet named.

---

### D. Jointly synthesize tiling, traversal, and swizzling

The synthesis problem has three coupled decisions.

#### 1. Capacity-constrained tiling
Choose tiles \(T_{L2}, T_{SMEM}, T_{TMEM}, T_{REG}\) such that symbolic live-footprint constraints fit each level’s capacity and pipeline budget. This turns CuTe tiling into **residency-aware tiling** rather than only throughput-aware tiling.

#### 2. Traversal synthesis
Choose a schedule transformation \(\sigma\) over iteration axes:

- axis permutations,
- strip-mining,
- wavefront skewing,
- parity reversals,
- CTA-cluster order,
- space-filling traversals on quotient axes.

Rather than exhaustive search, the compiler searches only a small algebraically complete family induced by quotient/complement structure.

#### 3. Swizzle synthesis
For low-bit address behavior, represent swizzles as linear maps over \(F_2\) (à la Linear Layouts). Then solve for an XOR/permutation transform \(S\) that:

- balances shared-memory banks,
- reduces L2 set hot spots,
- preserves high-level locality already chosen by traversal synthesis.

This unifies “cache order” and “bank swizzle” as the same optimization at different address granularities.

---

### E. Use Axe’s D/R/O decomposition to make optimization compositional

Axe gives a natural factorization:

- **D (shard):** where data is placed across SM/cluster/gpuid/cache slice
- **R (replica):** what is intentionally duplicated for persistence/reuse
- **O (offset):** low-level placement inside a line/bank/tile

This is crucial because many transformations act on disjoint named axes and therefore commute:

- L2 wavefront reversal acts on outer schedule axes,
- SMEM swizzle acts on low bank bits,
- TMEM staging acts on pipeline-slot axes.

When orthogonal, the optimizer solves them independently and composes them. When not orthogonal, it solves a small Pareto DP over the conflicting axes. This is what makes the full hierarchy tractable.

---

### F. Compiler realization: ISL + CuTe + Hexcute + Linear Layouts

The implementation would combine four ideas into one pass:

- **CuTe algebra** for shape/stride/nesting and quotient/complement
- **ISL-style relations** for exact symbolic image/cardinality reasoning
- **Linear Layouts** for low-bit swizzle synthesis
- **Hexcute-style type inference** for solving legal layouts under constraints

Concretely, we extend Hexcute’s constraint system with cache constraints:

- quotient image size must fit level capacity,
- set occupancy must fit associativity,
- bank conflict degree must stay below threshold,
- TMEM slot occupancy must fit stage budget.

The compiler takes as input:

- tensor layouts,
- thread/value layouts,
- hardware hierarchy descriptor,
- legal dependence/pipeline constraints.

It emits:

- transformed CuTe/Axe layouts,
- new loop/wavefront order,
- swizzle maps,
- optional proof certificates: “optimal within this schedule class under this cache model.”

---

### G. TMEM and Blackwell-specific opportunity

Blackwell makes the proposal stronger, not weaker. TMEM adds a new, fast, capacity-limited level with explicit pipeline semantics. Instead of treating TMEM as a special case, we model it as another named layout level. Then the same quotient/complement logic chooses:

- TMEM tile residency,
- stage order,
- 2-CTA sharing policy,
- async MMA grouping,
- spill-avoidance schedule.

This would let the compiler co-optimize FA4-style asynchronous pipelines with cache hierarchy behavior in one formal system.

## 5. Expected Contributions

- **A new formal abstraction:** modeling GPU cache hierarchy levels as layout objects with named axes.
- **A new theorem:** the **Complementary Reuse Theorem**, which derives miss-optimal traversal from quotient/complement structure and generalizes Sawtooth.
- **A unified optimizer:** joint synthesis of tiling, traversal, and swizzling across L2, shared memory/L1, and TMEM.
- **A compiler implementation:** extending CuTe/Hexcute/Triton-style systems with proof-carrying cache optimization instead of large autotuning spaces.
- **New empirical schedules:** automatic recovery of known patterns (Sawtooth) and discovery of new cache-optimal orders for attention, GEMM, MoE, and sparse kernels.

## 6. Evaluation Plan

### Platforms
- H100 / H200
- B200 / GB200 if available
- At least one Blackwell platform with TMEM exposure

### Workloads
- FlashAttention / FlashAttention-4 style kernels
- Paged attention
- Grouped GEMM / split-K GEMM
- MoE dispatch + expert GEMM + combiner
- Block-sparse attention / sparse GEMM
- Convolution/im2col-style tensor layouts

### Baselines
- Hand-tuned CUTLASS/CuTe kernels
- Triton autotuned kernels
- Hexcute-generated kernels
- Manual Sawtooth-style schedules where available
- cuBLAS / cuDNN / FlashAttention baselines

### Metrics
- Throughput / TFLOPS
- L2 hit rate, DRAM bytes, replay stalls
- Shared-memory bank conflicts
- TMEM occupancy / spills / pipeline bubbles
- Compile time and search-space reduction
- Prediction error between symbolic cost model and measured counters

### Critical experiments
1. **Recover Sawtooth automatically** from layout + hardware description only.
2. **Discover a new traversal** for at least two non-attention kernels.
3. **Show hierarchy compositionality:** L2 traversal gains plus SMEM swizzle plus TMEM staging are additive.
4. **Ablate the theorem:** remove complement reasoning, swizzle reasoning, or TMEM modeling and quantify loss.
5. **Robustness study:** vary hardware descriptor accuracy and show graceful degradation.

## 7. Target Venue and Why

**ASPLOS**.

Why: this idea sits exactly at the ASPLOS boundary of architecture, compilers, and systems. The paper would contribute:

- a new formal abstraction,
- a theorem about cache behavior,
- a compiler realization,
- and strong end-to-end performance results on modern GPUs.

It is broader than a pure PLDI paper and more formal/compiler-driven than a pure MICRO paper, making ASPLOS the best fit.

## 8. Potential Weaknesses and Mitigations

- **Weakness: GPU cache policies are partially undocumented.**  
  **Mitigation:** use a parameterized hardware descriptor inferred from microbenchmarks; prove optimality within a policy family, not one hidden implementation.

- **Weakness: exact global miss minimization may be intractable for irregular programs.**  
  **Mitigation:** target the tractable CuTe layout class first; give exact synthesis there and bounded heuristics/profile-guided refinement elsewhere.

- **Weakness: cache-optimal schedules may hurt occupancy or tensor-core utilization.**  
  **Mitigation:** make the solver multi-objective with hard legality constraints on occupancy, register pressure, MMA fragment shapes, and pipeline depth.

- **Weakness: dynamic sparsity or runtime-dependent routing may break static assumptions.**  
  **Mitigation:** support piecewise layouts plus a small runtime schedule switch among precompiled variants.

- **Weakness: TMEM semantics may evolve across GPU generations.**  
  **Mitigation:** keep TMEM as a pluggable hierarchy descriptor so the framework degrades gracefully to L2/SMEM optimization.

If you want, I can next turn this into a **1-page extended abstract**, a **paper outline with theorem statements**, or a **mock related-work/positioning section**.
