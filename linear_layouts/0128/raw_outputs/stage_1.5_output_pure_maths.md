## Orientation: what the seed *really* assumes (in group-theoretic terms)

The Linear Layouts seed (“Triton‑Linear”) is not just “linear algebra over bits”; it is (implicitly) working inside the **linear group** \(GL(n,2)\) (invertible \(n\times n\) matrices over \(\mathbb{F}_2\)), plus some **surjective maps** (distributed layouts allow “zero columns”) and **bitwise affine structure** (“XOR/AND on index bits”). The paper explicitly leans on the fact that **GPU parameters and Triton layout parameters are powers of two**, making index arithmetic naturally 2‑adic/bit‑sliced. ([arxiv.org](https://arxiv.org/html/2505.23819v3))

LEGO, by contrast, lives in the world of **bijections/permutations** (i.e., subgroups of \(S_N\)) with explicit `apply`/`inv`, and it builds **symbolic integer expressions** (mod / floor‑div) with range-aware simplification and SMT validation (Z3). ([arxiv.org](https://arxiv.org/html/2505.08091))

Your Stage‑1 cliffs are exactly the places where \(GL(n,2)\) is the wrong ambient algebra:

- **non‑power‑of‑two / ragged** ⇒ you need \(\mathbb{Z}\)-arithmetic and mixed moduli, not only \(\mathbb{F}_2\)
- **indirection / paging / MoE** ⇒ not a bijection (not a group action); it’s a **relation**
- **TMA + async proxy** ⇒ not a pure function; it’s a **partial order of events**
- **clusters / DSMEM** ⇒ hierarchy needs **composed group actions**, not “warp-only” reasoning
- **shuffle rounds exploding** ⇒ you’re doing word problems in a generating set of a permutation group

A useful “ladder” (this is the *theoretical arsenal map*):

```
(Seed) bit-linear layouts:         GL(n,2) actions on bit-vectors
          |
          v
2-adic / word-level modular:       modules over Z_(2^k)  (still bit-friendly)
          |
          v
general mixed-radix modular:       modules over Z_m / product of cyclic groups
          |
          v
integer-affine / polyhedral:       Z^n → Z^k affine maps + Presburger sets
          |
          v
relations / sparsity / indirection: Boolean semiring relations (sparse 0-1 matrices)
          |
          v
asynchrony / concurrency:          trace monoids, Petri nets, event structures
```

---

# 1) The Theoretical Toolbox (2 frameworks per Stage‑1 bottleneck)

Below I treat each bottleneck as “what structure is missing,” then give **two distinct theories** that *naturally* model it and suggest what compiler/runtime object you’d build.

---

## Bottleneck: **Dynamic shapes + ragged tensors** (padding/masking explosion, decode + MoE)

Linear Layouts acknowledges the core limitation: **restriction to power-of-two shapes**, with “define larger tensors + mask OOB” as mitigation. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
LEGO’s evaluation explicitly uses power-of-two matmuls and avoids partial tiles/masking to keep comparisons “fair,” which dodges the ragged regime. ([arxiv.org](https://arxiv.org/html/2505.08091))

### Theory A: **Finitely generated abelian groups + \(\mathbb{Z}\)-modules (SNF/HNF)**  
**Why it works:**  
A dense tensor index space with arbitrary extents is naturally a product of cyclic groups:
\[
\mathbb{Z}_{d_0}\times \mathbb{Z}_{d_1}\times \cdots \times \mathbb{Z}_{d_{k-1}}
\]
A layout (as a bijection on the finite index set) is an **automorphism** (or affine automorphism) of that product when it’s invertible; a non-bijective mapping is a homomorphism into another module plus a kernel.

The canonical tool here is **Smith Normal Form (SNF)** / **Hermite Normal Form (HNF)** of integer matrices: it classifies sublattices and module homomorphisms. Conceptually:

- contiguity / vectorization = properties of an integer linear form (strides)
- “invertibility” = unimodular transforms (det \(\pm 1\)) on \(\mathbb{Z}^n\) or invertibility mod \(d\)
- conversion synthesis = solve \(A x = b\) in \(\mathbb{Z}\) or \(\mathbb{Z}_d\)

**Why this generalizes \(\mathbb{F}_2\):** \(\mathbb{F}_2\) is the special case where every dimension is 2‑adic and only the LSB structure matters. SNF/HNF keeps *all* prime factors, so **prime-sized** or awkward dimensions stop being “second-class.”

**Implementation hook (compiler):**  
“Layout matrices” become integer matrices; you use SNF/HNF to:
- decide when a tile is contiguous modulo cacheline/transaction size
- synthesize “best-effort inverses” (right inverses) under non-invertible maps
- generate piecewise masks only where necessary (rather than padding to next power-of-two)

**Recent algorithmic signal:** fast SNF-with-multipliers algorithms exist (symbolic computation literature), meaning “canonicalization + multipliers” is no longer purely theoretical. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0747717122000955?utm_source=openai))

---

### Theory B: **Presburger arithmetic + semilinear sets (polyhedral-with-parameters, piecewise affine)**  
**Why it works:**  
Raggedness and masking are not “layout problems” so much as **domain problems**: your iteration space is not a full box; it is a **set of integer points** with constraints:
\[
0 \le i < B,\quad 0 \le j < L_i \quad (\text{ragged}), \quad \text{etc.}
\]
These domains are definable in **Presburger arithmetic** (addition, inequalities, modular constraints), and their sets are **semilinear**. That’s exactly what polyhedral compilation builds on, except you keep parameters symbolically (batch size, page size, per-request lengths).

**What you gain vs. \(\mathbb{F}_2\):**
- masks are not “afterthoughts”; they are first-class domain constraints
- you can generate **piecewise affine** layouts/schedules specialized on shape buckets
- you can reason about floor-div/mod precisely (critical for non-power-of-two)

**Bridge to LEGO:** LEGO already has to reason about modulo/floor-div and uses **range constraints + Z3** to justify simplifications. ([arxiv.org](https://arxiv.org/html/2505.08091))  
The theoretical leap is: don’t just simplify expressions—treat shapes/ranges as the *semantic object* and drive tiling, masking, and specialization from it.

---

## Bottleneck: **Non-power-of-two strides / flips / slices** (not expressible as \(\mathbb{F}_2\)-linear)

Linear Layouts explicitly says flipping/slicing are not expressible as linear layouts but could be captured by “affine layouts.” ([arxiv.org](https://arxiv.org/html/2505.23819v3))

### Theory A: **Affine groups \(AGL(n,\mathbb{Z})\) and mixed-mod affine groups \(AGL(n,\mathbb{Z}_m)\)**  
**Why it works:**  
A huge amount of “real layout” is affine:
\[
x \mapsto A x + b
\]
where \(b\) captures offsets (slices) and \(A\) captures permutation/strides. Over finite domains, you do this modulo extents. Affine groups naturally encode:
- **slicing**: add offset \(b\)
- **flip**: multiply axis by \(-1\) and add bias
- **non-power-of-two**: work over \(\mathbb{Z}_m\) rather than only \(\mathbb{F}_2\)

You can still preserve bit-friendly behavior in the common case by using \(\mathbb{Z}_{2^k}\) first (word-level modular arithmetic), then generalize.

### Theory B: **Piecewise-affine transformations (Presburger-definable maps)**  
**Why it works:**  
Many practical layouts are affine *except at boundaries* (partial tiles) or are “affine under a case split.” Piecewise affine maps are exactly what Presburger arithmetic can describe. This is the right mathematical home for:
- partial tiles
- ragged tail handling
- modulo-based swizzles with non-uniform ranges

Compiler-wise, this suggests an IR where a “layout” is:
- a map expression
- plus its domain constraints
- plus a cost model for guarded execution (predication/masks)

---

## Bottleneck: **TMA / async pipelines / async-proxy correctness** (layout engine is “conversion-centric”, not “transport-centric”)

This is the biggest Hopper/Blackwell mismatch: correctness and performance depend on *event ordering* (copies, barriers, waits), not only index mapping.

### Theory A: **Trace theory / Mazurkiewicz traces (partial commutation monoids)**  
**Why it works:**  
Asynchrony is naturally modeled as a **partial order** of events, not a total order. Trace theory builds equivalence classes of executions where **independent** actions commute.

- Events = “issue TMA copy”, “arrive barrier”, “wgmma commit”, “wgmma wait”, “consumer load”, etc.
- Independence relation \(I\) = when two events can reorder without changing correctness

A schedule is then a word in a **trace monoid** (a quotient of the free monoid by commutation rules). This gives you a clean formal handle on:
- when you can reorder copies earlier/later
- when a fence is required (not independent)
- how to prove a rewrite preserves correctness (a compiler pass, not just heuristics)

**Compiler payoff:** you can make “TMA pipeline planning” an **e-graph / rewriting problem** over traces, validated by a dependency model.

---

### Theory B: **Petri nets + max-plus algebra (tropical semiring) for throughput/latency**  
**Why it works:**  
TMA pipelines are essentially *token flow* with capacity constraints:
- a copy “in flight” occupies resources
- barriers are synchronization places
- stages form a pipeline

Petri nets model reachability and resource constraints; max-plus algebra models **steady-state throughput** and **critical cycles** in such systems.

**Why this matters for GPUs:**  
Register pressure and occupancy cliffs are often symptoms of “pipeline not hiding latency.” A max-plus view lets you ask:
- what stage count is needed to saturate tensor pipes?
- where is the bottleneck cycle (HBM→SMEM copy vs wgmma vs epilogue)?
- can we retime (shift) operations without violating causality?

This is the “math of asynchrony” that your current \(\mathbb{F}_2\) layout engine doesn’t provide.

---

## Bottleneck: **Thread Block Clusters / DSMEM** (new locality tier beyond CTA)

### Theory A: **Wreath products (hierarchical permutation group actions)**  
**Why it works:**  
Hardware hierarchy naturally composes actions:
- cluster permutes blocks
- within each block, warps permute lanes
- within each lane, registers permute elements

The group-theoretic object that captures “an action of groups on groups hierarchically” is the **wreath product**. Intuitively:

\[
G \wr H \quad \text{= “H acts on copies of G”}
\]

This matches the semantics of “per-CTA layout + cluster-level distribution of CTAs.” It is the *right* algebra if you want a unified representation that spans:
\[
\{\text{cluster}, \text{CTA}, \text{warp}, \text{lane}, \text{reg}\}
\]

**Compiler object:** “distributed layout” becomes a hierarchy of actions, not a single matrix.

---

### Theory B: **Quotients / cosets (reasoning about locality as equivalence classes)**  
**Why it works:**  
Cluster locality is fundamentally “which elements share a locality domain.” That’s an **equivalence relation**, i.e., a quotient.

- define a subgroup/submodule corresponding to “within-CTA” movement
- cosets correspond to “which CTA/which warp/which lane”
- conversions become “choose coset representatives” (minimize cross-domain movement)

This gives a principled way to ask: “is this conversion intra-CTA, intra-cluster, or cross-cluster?” and attach different costs (e.g., DSMEM vs shared vs registers).

---

## Bottleneck: **Indirection (KV paging, MoE routing)** destroys contiguity; not a permutation, not linear

Linear Layouts’ gather fast path only triggers when the gather axis is warp-contained; otherwise it degenerates into more expensive movement. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
That’s exactly what indirection breaks: the access is not a structured affine map.

### Theory A: **Relations as algebra (Boolean semiring; sparse 0–1 matrices)**  
**Why it works:**  
A gather/scatter with duplication is not a bijection. Mathematically it is a **relation** \(R \subseteq X \times Y\), not a function. Relations compose, but they form a **semiring** (Boolean matrix multiplication), not a group.

Represent an indexing operation as a sparse 0–1 matrix \(P\):
- gather = \(y = P x\) (possibly duplicating or dropping)
- scatter-add = \(x \mathrel{+}= P^\top y\)

Now your optimization problem becomes:
- reorder / block / factorize sparse operators
- detect when \(P\) is “nearly a permutation” (few collisions) and exploit that
- choose data reordering to make \(P\) block-diagonal (dense inner loops)

**Key conceptual upgrade:** you stop pretending indirection is “a layout.” It’s a **sparsity pattern**.

---

### Theory B: **Inspector–executor as factorization of relations into (permute ∘ block ∘ permute)**  
**Why it works:**  
Many relations used in systems workloads are “structured enough” that a runtime can compute a permutation that densifies them. LEGO even cites inspector–executor techniques for non-affine programs as a runtime approach. ([arxiv.org](https://arxiv.org/html/2505.08091))

Mathematically, you’re trying to factor a relation \(P\) into:
\[
P \approx \Pi_2^\top \; \mathrm{BlkDiag}(\text{dense}) \; \Pi_1
\]
where \(\Pi_1,\Pi_2\) are permutations (group elements) and the middle is dense compute.

That’s exactly the “pack tokens by expert/page, then run dense kernels” strategy—now stated as a **relation factorization problem**.

---

## Bottleneck: **Shuffle-round explosion** (instruction-bound gathers/conversions)

Linear Layouts says “use warp shuffles when warp-contained,” but then the cost grows with more rounds. ([arxiv.org](https://arxiv.org/html/2505.23819v3))

### Theory A: **Permutation group word problem in a hardware generator set (Cayley graph search)**  
**Why it works:**  
Treat each warp-lane data exchange as a permutation \(\sigma \in S_{32}\) (or \(S_{64}\)). Hardware gives you a generating set \(G=\{\text{shfl.xor(1)},\text{shfl.xor(2)},...\}\) (plus maybe permute/byte-permute).

You want a short “program”:
\[
\sigma = g_1 g_2 \cdots g_k,\quad g_i \in G
\]
Minimize \(k\) (or weighted cost). This is a shortest path problem on the **Cayley graph** of the subgroup generated by \(G\).

**Compiler payoff:** you replace heuristic shuffle generation with an optimal/near-optimal synthesis method under a real cost model (latency, issue pressure, register pressure).

---

### Theory B: **Routing / sorting networks + communication lower bounds (hypercube model)**  
**Why it works:**  
A warp with XOR shuffles is basically a **hypercube** interconnect. There are known lower bounds on the number of rounds needed for certain permutations (bisection bandwidth arguments, congestion).

**Why this matters:**  
It gives you a *mathematically justified* decision boundary:
- “this permutation is too expensive to realize with shuffles → stage via shared/TMA”
- “this one is cheap → stay in-register and avoid shared”

That directly addresses the “instruction cliff”: you can predict it before codegen.

---

## Bottleneck: **Target-specific bank conflicts** (AMD wave64/LDS rules differ; \(\mathbb{F}_2\) bank model isn’t portable)

### Theory A: **Linear algebra over \(\mathbb{Z}_B\) (banks) + combinatorial designs**  
**Why it works:**  
Bank index is ultimately computed modulo number of banks \(B\) (commonly 32), but the mapping from addresses/lane groups to banks is not always pure \(\mathbb{F}_2\). Modeling swizzles as transforms in \(\mathbb{Z}_B\) (or mixed \(\mathbb{Z}_{2^k}\)) lets you express:
- “distinct banks per lane group” as modular distinctness constraints
- instruction-width effects as different projections of the address bits

Design theory tools (orthogonal arrays / Latin squares style constraints) can be used to systematically build conflict-free mappings under multiple projections.

---

### Theory B: **SMT / ILP synthesis over modular constraints**  
**Why it works:**  
Bank conflict avoidance is naturally a constraint satisfaction problem:
- variables = swizzle parameters, bit-mix choices
- constraints = “lanes in a conflict domain map to distinct banks”
- objectives = maximize vectorization, minimize address-gen cost, respect alignment

This is *exactly* the style of reasoning LEGO already uses: it proves side conditions of modulo/floor-div simplifications with Z3 SMT. ([arxiv.org](https://arxiv.org/html/2505.08091))  
The difference is you move from “prove simplification legality” to “synthesize the layout/swizzle.”

---

## Bottleneck: **Register pressure / occupancy cliffs** (layout conversions + wgmma fragments + epilogue fusion)

### Theory A: **Resource-constrained scheduling + max-plus (treat registers as capacity, not a scalar cost)**  
**Why it works:**  
Register pressure is a *peak* property (max live values), not a sum. Max-plus algebra models “latest start times” and critical paths; RCPSP models limited resources.

You can model:
- each conversion step as a task that increases/decreases live values
- wgmma fragments as long-lived resources until commit/wait points
- objective = minimize peak registers while maintaining pipeline overlap

This yields an optimizer that can explain occupancy cliffs *before* Nsight does.

---

### Theory B: **Linear/affine type systems for values with constrained lifetime (regions, uniqueness)**  
**Why it works:**  
If “fragment registers” and “TMA staging buffers” are treated as affine resources:
- you can enforce at the IR level that certain temporaries cannot overlap
- you can force early-release / fence placement to shrink live ranges

Mathematically, this is the move from “everything is freely duplicable” (classical) to “resources live in an affine logic” (no implicit copying). That lines up with “avoid accidental broadcasting of large fragments” that triggers register explosions.

---

# 2) Literature Scan — recent math/group-theory breakthroughs that are surprisingly relevant

These are not “GPU papers,” but they provide **fresh algebraic machinery** that maps to your compiler needs (canonicalization, decomposition, concurrency-as-algebra, synthesis).

1) **Decidability of Krohn–Rhodes complexity (finite semigroups/automata)**  
Margolis–Rhodes–Schilling prove decidability of Krohn–Rhodes complexity for all finite semigroups/automata. ([arxiv.org](https://arxiv.org/abs/2406.18477?utm_source=openai))  
**Relevance:** Krohn–Rhodes is about decomposing finite-state behavior into cascades (wreath products of groups + aperiodic parts). That is a deep match to “pipeline decomposition”: you can model async schedules / barrier protocols as automata and reason about minimal “group complexity” of the control structure.

2) **Fast Smith Normal Form with multipliers (integer matrices)**  
A 2023 Journal of Symbolic Computation paper gives fast algorithms for SNF *with unimodular multipliers*. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0747717122000955?utm_source=openai))  
**Relevance:** If you move beyond \(\mathbb{F}_2\) to \(\mathbb{Z}\)-module layouts, SNF/HNF becomes your canonicalizer and conversion synthesizer—*and having multipliers matters*, because multipliers correspond to explicit conversion code.

3) **Substantial progress on the finite Group Isomorphism bottleneck case (p-groups class 2, exponent p)**  
Sun (2023) gives a faster isomorphism algorithm for a hard p-group family. ([arxiv.org](https://arxiv.org/abs/2303.15412?utm_source=openai))  
**Relevance:** Layout search and schedule search often need **symmetry detection** to prune equivalent candidates. Efficient canonical labeling/isomorphism tools from group theory translate into “don’t re-autotune equivalent layouts.”

4) **Polynomial-time computation of 2-closure for rank-3 permutation groups (computational group theory)**  
There’s a 2023 Journal of Algebra result giving a polynomial-time algorithm to compute the 2-closure of rank-3 groups. ([sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0021869323002260?utm_source=openai))  
**Relevance:** “Closure under pair relations” is a natural model for “what communications are induced by this mapping?” (pairs of lanes/addresses). 2-closure is exactly “what structure is determined by pairwise orbits.”

5) **Isomorphism problems for tensors/groups/polynomials under classical group actions (ITCS 2024)**  
Chen–Grochow–Qiao–Tang–Zhang (ITCS 2024) study isomorphism under classical group actions. ([drops.dagstuhl.de](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITCS.2024.31?utm_source=openai))  
**Relevance:** Your layout + schedule search space is full of “same computation under a group action.” This line of work provides principled ways to reason about equivalence classes under group actions—useful for e-graphs and canonicalization in codegen.

---

## (Optional) A compiler-facing “math-to-IR” blueprint (to keep the theory grounded)

If you want a single mental model for how these theories become software:

```
             +-------------------------------+
tt/ttg IR -->|  Constraint Extraction        |--> constraints:
shapes,      |  - domains (Presburger)       |    - bounds/masks
layouts      |  - transport events (PO)      |    - modular/alignment
             |  - bank model (target)        |    - resource caps (regs/smem)
             +-------------------------------+
                             |
                             v
             +-------------------------------+
             |  Solver / Synthesizer         |
             |  A) SNF/HNF (Z-mod layouts)   |
             |  B) SMT/ILP (swizzles, packs) |
             |  C) Trace rewriting (async)   |
             |  D) Cayley shortest-path      |
             +-------------------------------+
                             |
                             v
             +-------------------------------+
             |  Plan                         |
             |  - layout (possibly piecewise)|
             |  - transport schedule (TMA)   |
             |  - conversion strategy        |
             +-------------------------------+
                             |
                             v
                     PTX/SASS (real metal)
```

This is the “render the seed obsolete” direction: **layouts are no longer only \(\mathbb{F}_2\) matrices**, but the *output* of a solver that jointly reasons about:
- \(\mathbb{Z}\)/mod arithmetic (non-power-of-two),
- partial orders (async),
- relations (indirection),
- and target-parametric bank/occupancy constraints.

If you want, I can take the strongest 2–3 theories above and turn them into a *single unified* research proposal (“Triton: from \(\mathbb{F}_2\) layouts to \(\mathbb{Z}\)+trace+relation layouts”) with concrete IR extensions + an evaluation plan on TritonBench/Nsight.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping](https://arxiv.org/html/2505.08091)
3. [A fast algorithm for computing the Smith normal form with multipliers for a nonsingular integer matrix - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0747717122000955?utm_source=openai)
4. [Decidability of Krohn-Rhodes complexity for all finite semigroups and automata](https://arxiv.org/abs/2406.18477?utm_source=openai)
5. [Faster Isomorphism for $p$-Groups of Class 2 and Exponent $p$](https://arxiv.org/abs/2303.15412?utm_source=openai)
6. [Two-closure of rank 3 groups in polynomial time - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021869323002260?utm_source=openai)
7. [On the Complexity of Isomorphism Problems for Tensors, Groups, and Polynomials III: Actions by Classical Groups](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITCS.2024.31?utm_source=openai)