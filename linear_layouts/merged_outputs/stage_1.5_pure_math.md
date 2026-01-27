# Unified Research Note: The "Math Stack" for Next-Gen Tensor Compilers
**Synthesizing Linear Layouts & LEGO into a General Theoretical Toolbox**

## 1. Orientation: The Theoretical "Ladder"
Both research notes critique the current limitations of the "Linear Layouts" (Triton-Linear) and "LEGO" approaches, proposing a generalized mathematical hierarchy to transcend them.

*   **The Status Quo:**
    *   **Linear Layouts** rely on **Linear Maps on bit-vectors over \( \mathbb{F}_2 \)** (homomorphisms between finite 2-groups). This explicitly assumes power-of-two dimensions and relies on padding/masking for exceptions. It fails to express flips, slices, or negative strides natively [1], [2].
    *   **LEGO** operates as a **Layout-Expression Frontend** based on **bijective permutations** of index spaces (subgroups of \( S_N \)) and symbolic integer expressions [1], [2].

*   **The Proposed "Ladder" of Abstractions [2]:**
    To solve the bottlenecks of ragged shapes, asynchrony, and deep hierarchies, the compiler stack must ascend this algebraic ladder:
    1.  **(Seed) Bit-linear layouts:** \( \mathrm{GL}(n,2) \) actions on bit-vectors.
    2.  **2-adic / word-level modular:** Modules over \( \mathbb{Z}_{2^k} \) (bit-friendly but arithmetic).
    3.  **General mixed-radix:** Modules over \( \mathbb{Z}_m \) / products of cyclic groups (ragged shapes).
    4.  **Affine / Polyhedral:** \( \mathbb{Z}^n \to \mathbb{Z}^k \) affine maps + Presburger sets (slices/partial tiles).
    5.  **Relations / Indirection:** Boolean semiring relations (sparse 0-1 matrices).
    6.  **Asynchrony / Concurrency:** Trace monoids, Petri nets, Event structures.

---

## 2. The Theoretical Toolbox (Solutions by Bottleneck)

### Bottleneck 1: Ragged Extents & Dynamic Shapes (The Padding Cliff)
Current systems restrict shapes to powers of two, which is inefficient for LLM decode or MoE routing [1].

#### Theory A: Finite Abelian Groups / \( \mathbb{Z} \)-Modules + Smith Normal Form (SNF) [1], [2]
*   **Concept:** Replace bit-vectors with **mixed-radix index groups**. An index space is a product of cyclic groups \( G \cong \mathbb{Z}_{n_0} \times \mathbb{Z}_{n_1} \dots \). Layouts become group/module homomorphisms [1].
*   **Key Mechanism:** **Smith Normal Form (SNF)** or Hermite Normal Form (HNF). SNF provides a canonical decomposition of integer-linear maps, serving as the "Gaussian elimination" for rings like \( \mathbb{Z} \) [1], [2].
*   **Compiler Payoff:**
    *   Decide contiguity/vectorization for **prime/non-power-of-two strides** (e.g., stride 160) via modular constraints [1].
    *   Synthesize conversion plans by solving linear congruences rather than finding inverses in \( \mathbb{F}_2 \) [1].
    *   Use SNF "multipliers" to generate explicit code for non-bijective mappings [2].

#### Theory B: Presburger Arithmetic / Polyhedral Model [1], [2]
*   **Concept:** Treat ragged tensors and partial tiles as **integer points in polyhedra** defined by affine inequalities (Presburger arithmetic) [1]. This upgrades "masking" from a heuristic to a first-class domain constraint [2].
*   **Compiler Payoff:**
    *   Generate **piecewise affine** layouts: one fast path for the full interior (vectorized) and specialized paths for edges [1].
    *   Precise reasoning about `floor-div` and `mod` constraints, which LEGO currently handles via Z3 but does not fully exploit for tiling [2].

---

### Bottleneck 2: Flips, Slices, Negative Strides & Partial Validity
Linear Layouts cannot express flips or slices without extending to affine maps; LEGO relies on global bijections [1].

#### Theory A: Affine Groups \( \mathrm{AGL}(n, R) \) [1], [2]
*   **Concept:** Extend linear maps \( x \mapsto Ax \) to affine maps \( x \mapsto Ax + b \).
    *   Offsets \( b \) capture **slices/sub-tiles** [1].
    *   Working over \( \mathbb{Z}_m \) (instead of \( \mathbb{F}_2 \)) allows negative strides (multiplication by -1) [1].
*   **Compiler Payoff:** Represents views, slices, and reinterprets without fallback to "pad + mask" [1].

#### Theory B: Groupoids / Inverse Semigroups (Partial Bijections) [1]
*   **Concept:** Formalize "partial symmetries." Inverse semigroups model maps that are not total functions.
*   **Compiler Payoff:** Principled logic for **validity domains**. The compiler allows an inverse only within the "interior" region, avoiding global overhead for boundary handling.

#### Theory C: Piecewise-Affine Transformations [2]
*   **Concept:** Define layouts as affine maps under a case split (Presburger-definable).
*   **Compiler Payoff:** Enables an IR where a layout is `(map expression + domain constraints + cost model)`.

---

### Bottleneck 3: Asynchrony (TMA, Proxies, Barriers)
Hopper/Blackwell architectures require ordering events (bulk copies, mbarrier arrivals), not just mapping indices.

#### Theory A: Petri Nets & Max-Plus Algebra [1], [2]
*   **Concept:**
    *   **Petri Nets:** Model correctness, resource constraints (tokens), and event dependencies (TMA issue -> Arrive -> Wait) [1].
    *   **Max-Plus (Tropical) Algebra:** Models steady-state throughput and critical paths (latency hiding) [2].
*   **Compiler Payoff:** Solves for the optimal stage count (double vs. triple buffering) and identifies bottleneck cycles (HBM vs. Compute) [2].

#### Theory B: Trace Theory (Mazurkiewicz Traces) [2]
*   **Concept:** Asynchrony is a **partial order**. Trace theory defines equivalence classes of execution where independent operations commute.
*   **Compiler Payoff:** A "schedule" becomes a word in a trace monoid. Validity checks become rewriting rules on traces (e.g., reordering a copy earlier if it commutes with intervening ops).

#### Theory C: Separation Logic / Effect Systems [1]
*   **Concept:** CUDA requires explicit proxy fences between different proxy types (async vs. generic). Effect systems track properties like `(proxy=async, space=shared, read/write)`.
*   **Compiler Payoff:** Safe construction of aggressive reordering; prevents data races caused by missing fences.

---

### Bottleneck 4: Clusters & Hierarchical Locality
The introduction of Thread Block Clusters creates a hierarchy (Cluster → CTA → Warp) that breaks simple "CTA-contained" assumptions.

#### Theory A: Wreath Products (Hierarchical Groups) [1], [2]
*   **Concept:** The GPU hierarchy is a group acting on a group. The **Wreath Product** \( G \wr H \) captures "H acts on copies of G" (e.g., Cluster acts on blocks, Block acts on warps) [2].
*   **Compiler Payoff:** Canonical factoring of layouts into "cluster-level" vs. "warp-level" components, enabling systematic multicast discovery [1].

#### Theory B: Quotients / Cosets [2]
*   **Concept:** Locality is an equivalence relation. The "cosets" of a subgroup define which elements share a specific hardware resource (e.g., "elements within the same CTA").
*   **Compiler Payoff:** Conversions become "choose coset representative" problems—minimizing cross-domain movement (DSMEM vs. Shared).

#### Theory C: Scoped Resource Semantics [1]
*   **Concept:** Use **Monoidal Categories** ("typed wiring diagrams") to enforce scope correctness.
*   **Compiler Payoff:** A "Cluster DSMEM pointer" is a distinct type from a "Shared Memory pointer," preventing illegal composition at the IR level.

---

### Bottleneck 5: Indirection (Gather/Scatter, MoE, Paging)
Gathers break "warp-contained" assumptions; they are relations, not functions.

#### Theory A: Relations & Boolean Semirings [2]
*   **Concept:** A gather is a **relation** \( R \subseteq X \times Y \) (sparse 0-1 matrix), not a bijection. Relations form a Semiring.
*   **Compiler Payoff:** Optimization becomes **factorizing relations**: detecting when a sparse matrix \( P \) is "nearly a permutation" or block-diagonal to maximize dense compute.

#### Theory B: Permutation Group Orbits / Cosets [1]
*   **Concept:** A layout is a group action on indices. "Warp-contained" means the **orbit** of an axis under the "warp-movement subgroup" remains within one warp.
*   **Compiler Payoff:** Systematically select between warp shuffles, shared-mem transpose, or cluster exchange based on orbit partitions.

#### Theory C: Inspector-Executor / Relation Factorization [2]
*   **Concept:** Factor a relation \( P \) into \( \Pi_2 \circ \text{BlockDiag} \circ \Pi_1 \).
*   **Compiler Payoff:** The mathematical basis for "sorting tokens by expert" (MoE) or paging: permute data to make the inner computation dense.

#### Theory D: Synthesis (SMT/ILP) [1]
*   **Concept:** Treat gather lowering as a search problem with constraints (register budget, latency overlap).
*   **Compiler Payoff:** Instead of heuristics, solve: "Find movement plan with cost < X".

---

### Bottleneck 6: Shuffle-Round Explosion & Instruction Cliffs
Linear Layouts heuristics for warp shuffles fail when rounds explode; instruction count is non-linear [1].

#### Theory A: Permutation Group Word Problem (Cayley Graphs) [2]
*   **Concept:** Warp hardware provides a generating set \( G = \{ \text{xor\_1}, \text{xor\_2}, \dots \} \). Finding the optimal shuffle sequence is a **shortest path problem on the Cayley graph** of \( S_{32} \).
*   **Compiler Payoff:** Optimal synthesis of shuffle sequences rather than heuristic generation.

#### Theory B: Routing Networks & Lower Bounds [2]
*   **Concept:** Modeled as **hypercube interconnects**. Use Bisection Bandwidth arguments to prove lower bounds.
*   **Compiler Payoff:** Immediate prediction of "Instruction Cliffs"—knowing *when* to fall back to shared memory without generating the code first.

---

### Bottleneck 7: Bank Conflicts (AMD Wave64 vs. NVIDIA)
Bank conflict rules vary by vendor and instruction phase, breaking simple XOR-swizzle models.

#### Theory A: Affine Algebra over Rings \( \mathbb{Z}/B\mathbb{Z} \) [1], [2]
*   **Concept:** Bank indices are congruences modulo \( B \). Swizzles are affine transforms in this modular ring [1].
*   **Compiler Payoff:** A parametric framework that emits XOR-based swizzles for NVIDIA and phase-aware swizzles for AMD by changing the modulus ring [1].

#### Theory B: Constraint Solving (SMT/SAT) [1], [2]
*   **Concept:** Bank conflict avoidance is a constraint satisfaction problem (all lanes in a phase group must map to distinct banks) [2].
*   **Compiler Payoff:** Provably conflict-free swizzles for specific instruction widths (e.g., `ds_write_b128`) [1].

---

### Bottleneck 8: Register Pressure & Occupancy [2]
(Unique to Source 2)

#### Theory A: Resource-Constrained Project Scheduling (RCPSP)
*   **Concept:** Registers are a renewable resource with a capacity cap. Use **Max-Plus** algebra to model "latest start times" given capacity.
*   **Compiler Payoff:** Explains occupancy cliffs before profiling.

#### Theory B: Affine Type Systems (Linear Types)
*   **Concept:** Values have "uniqueness" or "affine" lifetimes (cannot be freely duplicated).
*   **Compiler Payoff:** Prevents "accidental broadcasting" of large accumulator fragments that explodes register usage.

---

### Bottleneck 9: The Global Search Problem [1]
(Unique to Source 1)

#### Theory A: Rewrite Systems / Normal Forms
*   **Concept:** Layout expressions form an algebra. A **confluent rewrite system** allows canonicalization (finding a "normal form" plan).
*   **Compiler Payoff:** Deterministic canonicalization shrinks the autotuning search space significantly.

#### Theory B: Equality Saturation (E-graphs)
*   **Concept:** Encode algebraic, hardware, and schedule identities into an E-graph to explore equivalent programs.
*   **Compiler Payoff:** Unifies layout optimization, conversion placement, and scheduling into a single "best cost extraction" pass.

---

## 3. Literature Scan: Math Breakthroughs for Compilers

**Group Theory & Algebra:**
1.  **McKay Conjecture (2025):** Proved by Cabanes & Späth. Demonstrates deriving global invariants from local structures (relevant to deriving global layout guarantees from warp/block constraints) [1].
2.  **Brauer's Height Zero Conjecture (2024/2025):** Completed for odd primes (Malle et al.) and prime 2 (Ruhstorfer). Relevance: "Hard corner case completion" mirrors handling specific backend architecture quirks (e.g., warp size 32 vs 64) [1].
3.  **Artin Group Word Problems (2024):** Quadratic time solutions using length-preserving rewrite rules. Relevance: Algorithmic rewriting for layout canonicalization [1].
4.  **Finite Group Isomorphism (2023):** Faster algorithms for p-groups (Sun). Relevance: Symmetry detection to prune the layout search space [2].
5.  **2-closure for Rank-3 Groups (2023):** Polynomial time computation. Relevance: Modeling "pairwise" communication structures [2].
6.  **Tensor/Group Isomorphism (ITCS 2024):** Isomorphism under classical group actions. Relevance: Principled reasoning about "equivalence classes" of layouts [2].

**Systems & Complexity:**
7.  **Krohn-Rhodes Complexity (2024):** Decidability for finite semigroups/automata. Relevance: Decomposing async pipeline control flow into minimal "group complexity" (automata cascades) [2].
8.  **Fast Smith Normal Form (2023):** New algorithms for SNF *with multipliers*. Relevance: Efficient synthesis of layout conversion code for non-power-of-two shapes [2].
9.  **PTX/CUDA Formalization:** Explicit distinction between generic and async proxies [1].

---

## 4. Blueprint: The "Successor Abstraction"
(Combining proposals from both sources)

**The Proposal:**
A unified IR layer **"Affine/Modular Layouts + Transport Schedules"** consisting of:

1.  **Constraint Extraction:** Shapes, layouts, bank models, and resource caps fed into an engine.
2.  **Solver / Synthesizer:**
    *   **SNF/HNF** for \( \mathbb{Z} \)-module layouts (ragged/dynamic).
    *   **SMT/ILP** for swizzles and packings.
    *   **Trace Rewriting** for async transport pipelines.
    *   **Cayley Search** for optimal shuffle sequences.
3.  **Plan Extraction:** Outputting piecewise layouts, Petri-net schedules, and optimal conversion paths.

**Evaluation Plan:**

Implement in Triton backend; evaluate on H100/MI300 using TritonBench + Nsight Compute. This replaces the heuristic "padding + masking" and "warp-only" assumptions with a rigorous algebraic solver [1], [2].

### Appendix: Visualizations of the "Math Stack" for Tensor Compilers

#### Figure A. The Algebraic Ladder of Tensor Layouts
This diagram illustrates the progression from current state-of-the-art abstractions (Linear Layouts and LEGO) to the generalized "Ladder" proposed for next-generation compilers [2]. It highlights how shifting from $\mathbb{F}_2$ to $\mathbb{Z}$-modules and Relations resolves specific hardware bottlenecks.

```text
      COMPLEXITY LEVEL          MATHEMATICAL STRUCTURE               CAPABILITY / HARDWARE FEATURE
    ==================================================================================================

          (Highest)
              ^            +-----------------------------+
              |            |  Relations / Boolean Semiring|   <-- Indirection, MoE, Sparse Gathering
              |            |  ( P ⊆ X × Y, 0-1 Matrices )|       (Non-bijective, 1-to-many)
              |            +-----------------------------+
              |                         ^
              |                         | Factorization (P ≈ Π₂ D Π₁)
              |                         |
              |            +-----------------------------+
              |            |  Affine / Polyhedral Maps   |   <-- Slices, Pad-free Tiles, Partial Domains
              |            | ( Z^n -> Z^k + Constraints )|       (Presburger Arithmetic, Boundaries)
     PROPOSED |            +-----------------------------+
     "LADDER" |                         ^
       [2]    |                         | SNF Decomposition
              |                         |
              |            +-----------------------------+
              |            |  Generalized Mixed-Radix    |   <-- Ragged Tensors, Prime Strides
              |            | ( Z_n0 × Z_n1 ... Modules ) |       (Avoids 2^k padding waste)
              |            +-----------------------------+
              |                         ^
              |                         | Generalize Ring
              |                         |
    ----------+-------------------------+-------------------------------------------------------------
              |
     CURRENT  |            +-----------------------------+
      STATE   |            |  Bit-Linear Layouts [1,3]   |   <-- Power-of-2 Tiling, XOR Swizzling
              |            |  ( GL(n, 2) / F_2 vector )  |       (Limited to 2-groups/bijections)
              |            +-----------------------------+
              |
              |            +-----------------------------+
              |            |  LEGO Bijections [2,4]      |   <-- Hierarchical Tiling, Permutations
              |            | ( Subgroups of S_N, Exprs ) |       (Bijective-only, Strided-only)
              |            +-----------------------------+

    ==================================================================================================
```
*Caption: The theoretical evolution from bit-linear/bijective maps (bottom) to algebraic relations (top). Each step up the ladder enables the compiler to natively express hardware behaviors that currently require fragile heuristics [2].*

***

#### Figure B. Hierarchical Locality via Wreath Products
Visualizing the proposal to model the GPU hierarchy (Cluster → CTA → Warp) using Wreath Products ($G \wr H$), enabling systematic factoring of layouts across distributed memory tiers [2][3].

```text
    LOGICAL TENSOR             GROUP ACTION HIERARCHY             HARDWARE MAPPING
    (Global Index)            (Wreath Product G ≀ H)              (Distributed View)
   +--------------+           +--------------------+            +------------------+
   |              |           |  Cluster Action    |            |  Cluster Shared  |
   |  i, j, k ... |   <-----> | (Permute Blocks)   | ---------> |  Memory (DSMEM)  |
   |              |           +---------+----------+            +------------------+
   +------+-------+                     |                                |
          |                     (Acting on copies of)                    | Multicast/
          |                             |                                | Copy
          v                             v                                v
   +--------------+           +--------------------+            +------------------+
   |  Local Tile  |           |     CTA Action     |            |   CTA Shared     |
   | (Partition)  |   <-----> |  (Permute Warps)   | ---------> |     Memory       |
   |              |           +---------+----------+            +------------------+
   +------+-------+                     |                                |
          |                     (Acting on copies of)                    | Load/
          |                             |                                | Store
          v                             v                                v
   +--------------+           +--------------------+            +------------------+
   |  Micro Tile  |           |    Warp Action     |            |     Register     |
   | (Elements)   |   <-----> |  (Permute Lanes)   | ---------> |       File       |
   +--------------+           +--------------------+            +------------------+

   MATHEMATICAL FORMALISM:
   Layout L ≅ L_cluster  ⋊  (L_CTA  ⋊  (L_warp))
   Valid lowerings are discovered by decomposing the wreath product.
```
*Caption: Modeling hardware hierarchy as nested group actions. This allows the compiler to distinguish between "warp-contained" moves (shuffles) and "cluster-contained" moves (DSMEM) algebraically rather than via heuristics [1][2].*

***

#### Figure C. Asynchronous Transport Pipeline (Petri Net Model)
A visualization of the "Transport Schedule" layer, where asynchrony is modeled using Petri Nets or Trace Monoids to optimize TMA (Tensor Memory Accelerator) pipelines [1][2].

```text
    TIME ->
    
    [ Global Memory ]
           |
           | (A) Issue TMA Copy (Async Proxy) [3]
           | Token: Empty_Buffer -> Full_Buffer
           v
    ( )--[ Buffer 0 ]--( )               ( )--[ Buffer 1 ]--( )
     |                    \             /                    |
     | (Dependency)        \           /                     |
     v                      \         /                      v
 [ Barrier Arrive ]          \       /               [ Barrier Arrive ]
     |                        \     /                        |
     | (Latency Hiding)        \   /                         |
     v                          \ /                          v
 [ Wait (mbarrier) ] <-------(Overlap)---------->    [ Wait (mbarrier) ]
     |                                                       |
     | (B) Consumer Op                                       |
     v                                                       v
 [ WGMMA (Matrix Mult) ]                             [ WGMMA (Matrix Mult) ]
     |                                                       |
     | (C) Register Write                                    |
     v                                                       v
 [ Accumulator ]                                     [ Accumulator ]
 
    ALGEBRAIC CONSTRAINT:
    Schedule S is valid iff S is a linearization of the Partial Order (Trace)
    defined by the Petri Net reachability graph.
```
*Caption: Asynchrony modeled as a dependency graph (Petri Net). The compiler uses Max-Plus algebra to solve for the optimal buffer depth (stage count) and verify correctness via proxy fences [1][3].*

***

#### Figure D. The Unified "Solver-Based" Compiler Flow
A blueprint for the proposed "successor" architecture that merges the constraints from Linear Layouts and LEGO into a single solver pass [1][2].

```text
    +-------------------------+       +--------------------------+
    |   Input Program (IR)    |       |   Hardware Spec (ISA)    |
    | (Triton/Python/PyTorch) |       | (Banks, Units, Latency)  |
    +-----------+-------------+       +-------------+------------+
                |                                   |
                v                                   v
    +------------------------------------------------------------+
    |                 CONSTRAINT EXTRACTION ENGINE               |
    |                                                            |
    | 1. Domain:  Presburger Sets (Ragged/Partial shapes) [2]    |
    | 2. Layout:  Z-Module Homomorphisms (Strides/Paddings) [1]  |
    | 3. Bank:    Modular Congruence ( i % B != j % B ) [3]      |
    | 4. Async:   Event Dependencies (Proxy Fences) [3]          |
    +---------------------------+--------------------------------+
                                |
                                v
                   +---------------------------+
                   |   SYNTHESIS / SOLVER      |
                   |                           |
                   |  (A) Smith Normal Form    | --> Synthesize Layouts
                   |  (B) Max-Plus Algebra     | --> Synthesize Pipeline
                   |  (C) Cayley Graph Search  | --> Synthesize Swizzles/Shuffles
                   |  (D) E-Graph Rewriting    | --> Canonicalize Plan
                   +------------+--------------+
                                |
                                v
    +------------------------------------------------------------+
    |                    GENERATED PLAN                          |
    |                                                            |
    |  Layout:   Piecewise Affine Maps (No padding needed)       |
    |  Schedule: Asynchronous Trace (Optimal overlapping)        |
    |  Code:     Intrinsic Lowering (TMA/WGMMA/Permute)          |
    +------------------------------------------------------------+
```
*Caption: The unified "Math Stack" flow. Instead of heuristics, the compiler extracts algebraic constraints and solves them using group-theoretic and combinatorial tools (SNF, Cayley Graphs, SMT) to generate correct-by-construction kernels [1][2].*
