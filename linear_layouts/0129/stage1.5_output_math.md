## 0) Access verification (S3)

I independently verified that all three seed artifacts are accessible and parseable:

1. **Linear Layouts (\(\mathbb{F}_2\))** — arXiv HTML v3 loads and contains the full paper structure. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
2. **ISL / Integer Set Relations for Layouts** — arXiv HTML v1 loads. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
3. **Categorical Foundations (CuTe)** — arXiv PDF v1 loads (174 pages). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

---

# 1) The Theoretical Toolbox (Stage‑1 bottlenecks → 2 theories each)

Below, each bottleneck is phrased as an **axiom that fails on silicon**, then two frameworks that **strictly generalize** (or properly *refine*) the seed’s \(\mathbb{F}_2\) isomorphism view into something that can survive: **non‑\(2^n\) shapes**, **descriptor legality**, and **async/barrier reality**.

I’ll use the same “hardware‑grounded theory” lens from Stage 1: every abstraction must land on something real like **CuTensorMap constraints**, **`cp.async.bulk.tensor` weak ordering**, **`mbarrier` acquire semantics**, or **LDS lane‑group bank rules**.

---

## Bottleneck A — “Power‑of‑2 Tyranny” \(\Rightarrow\) ragged/prime shapes collapse into mask‑heavy tails

### Failure mode (seed‑level axiom that breaks)
Seed A explicitly admits a primary limitation: **restriction to power‑of‑two shapes**, mitigated by “define larger tensors and mask out‑of‑boundary elements.” ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
That’s semantically correct, but on GPUs it becomes the **tail effect** (wasted bandwidth, divergence, and broken bulk‑copy regularity).

### Theory A: **Finite Abelian Groups + Mixed‑Radix Modules (Structure Theorem lens)**  
**Core idea:** Replace “bits in \((\mathbb{Z}/2)^k\)” with **digits in \(\prod_i \mathbb{Z}_{n_i}\)** (mixed radix). Your index space is no longer a vector space over \(\mathbb{F}_2\); it’s a **finite abelian group / \(\mathbb{Z}\)-module with bounds**.

**Why it works (beyond \(\mathbb{F}_2\)):**
- \(\mathbb{F}_2\) linearity is exactly “carry‑free arithmetic” (XOR).  
- Non‑power‑of‑two shapes introduce **carries** and **mixed moduli**. Those are naturally handled by **\(\mathbb{Z}\)** / **\(\mathbb{Z}_m\)** algebra, not \(\mathbb{F}_2\).

**Concrete compiler optimization it enables (S2):**
- **Core+tail layout as a *group decomposition***:  
  - core region uses a regular subgroup-like structure (tileable, vectorizable, descriptor‑friendly),  
  - tail region is a coset remainder (handled by a second kernel or predicated epilogue).  
This directly attacks the tail effect without forcing “lift to \(2^n\) + mask”.

**Hardware mapping (what it buys):**
- Lets you keep the “core” inside **TMA‑friendly boxed traversals** (when possible) while isolating irregular tails away from TMA constraints.

**Implementation sketch (C1‑feasible):**
- Add an MLIR **LayoutGroup** attribute carrying a mixed‑radix factorization (per dimension) and a **piecewise** mapping:
  - `layout.core : (∏ Z_{n_i_core}) → offsets`
  - `layout.tail : (domain remainder) → offsets`
- Lowering strategy:
  - `core` → candidate for `cuTensorMapEncodeTiled`
  - `tail` → vector `ld/st` or `cp.async` scalar path.

---

### Theory B: **Semilinear Sets / Presburger Arithmetic (Polyhedral generalization, but “layout‑first”)**  
Seed B explicitly shows integer set relations encode much richer mappings and even non‑rectangular domains (e.g., triangular domains). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**Why it works:**
- Raggedness is a **domain** problem (iteration space is not a single hyper‑rectangle). Semilinear sets / Presburger constraints naturally express “union of rectangles,” “holes,” and parametric bounds.

**Concrete optimization it enables (S2):**
- **Domain‑split codegen**: compile a small set of “region kernels”:
  - main rectangular region: uses vectorization/TMA
  - ragged remainder: compact scalar/predicated path  
This is the mathematical version of “separate the tail,” but *proved* from set decomposition.

**Hardware mapping:**
- Keeps the bulk path regular enough to match TMA’s boxed model (rank \(\le 5\), etc.), while the ragged part avoids poisoning descriptor legality.

**Implementation sketch (C1):**
- Use ISL‑style relations as a **verification/canonicalization** layer (not the generator), consistent with Stage‑1 concerns about compile‑time blowups. Seed B already positions ISL as a unifier with rich operations (composition/inverse/domain/range). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- In MLIR: represent regions as `affine.if` / `scf.if` guards derived from Presburger splits; cache region decisions.

---

#### Math‑to‑Hardware diagram (core/tail as piecewise algebra)

```text
Ragged domain D  (e.g., per-seq lengths)
   |
   |  Presburger / mixed-radix decomposition
   v
D = D_core  ⊎  D_tail
   |              |
   |              +--> predicated / scalar fallback (minimize wasted lanes)
   v
Regular tileable domain
   |
   +--> attempt TMA descriptor synthesis (rank<=5, boxDim, alignment)
```

---

## Bottleneck B — “Descriptor Admissibility Gap”: valid \(\mathbb{F}_2\) swizzles \(\not\Rightarrow\) valid CuTensorMap

### Failure mode (seed‑level axiom that breaks)
Seed A defines broad families of layouts:
- **Distributed layout**: surjective linear layout with at most one \(1\) per column (permutation-ish with zero columns). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **Memory layout**: invertible linear layout with 1–2 non‑zero bits per column. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

But Hopper TMA is not “any invertible bit mixing”; it is a **descriptor‑driven** engine. Hopper’s TMA is explicitly pitched as: single thread can issue large moves; supports 1D–5D tensors; enables warp specialization. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/hopper-tuning-guide/index.html))  
And the driver API gives **hard constraints**:  
- `tensorRank` bounded (tiled encoding supports up to max 5), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- `globalAddress` alignment (16B, sometimes 32B), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- `globalStrides` multiples of 16 (sometimes 32 depending on interleave/type), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- swizzle and interleave are **enums with additional coupling constraints** (e.g., inner dimension \(\le\) swizzle size when `interleave==NONE`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

So: the seed’s layout language strictly contains many layouts that **cannot** be encoded as `cuTensorMapEncodeTiled`.

### Theory A: **“Hardware‑Refined Subgroup” viewpoint (Affine group + membership / constructive recognition)**  
**Core idea:** Treat “TMA‑admissible layouts” as a *mathematically defined sublanguage* of the seed’s layouts: a **subgroup / sub-semigroup** of allowed transformations, parameterized by descriptor fields and congruences.

- Seed A lives in something like \(GL(k, \mathbb{F}_2)\) (plus restrictions) acting on bit‑vectors. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- TMA legality lives in “affine‑ish traversal + enumerated swizzles + congruence constraints” (alignment/stride multiples). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

**Why it works:**
- It replaces the vague “admissible?” question with a crisp **membership test**:
  \[
  L \in \mathcal{L}_{\text{TMA}} \subset \mathcal{L}_{\text{seed}}
  \]
- Once you have a generating set (swizzle enums + affine stride templates), you can do **constructive membership**: recover descriptor parameters as a witness.

**Concrete optimization it enables (S2):**
- A **swizzle normal form / admissibility proof** pass:
  1. attempt to rewrite an \(\mathbb{F}_2\) layout into a descriptor‑witnessable form,  
  2. if success: emit `cp.async.bulk.tensor.*` path,  
  3. else: surface “TMA path lost” early and plan fallback + cost.

**Relevant “recent pure math / algorithmic group theory” hook (optional):**
- Constructive recognition algorithms for matrix groups (e.g., recognizing when a generated matrix group is isomorphic to \(\mathrm{SL}(d,q)\)) are an existence proof that “membership + witness extraction” is a real algorithmic discipline. ([arxiv.org](https://arxiv.org/abs/2404.18860?utm_source=openai))  
We won’t literally need \(\mathrm{SL}(d,q)\), but the *methodology* transfers.

**Implementation sketch (C1):**
- In MLIR/Triton backend: introduce an attribute/type `tma.admissible_layout` with fields mirroring descriptor degrees of freedom (rank, strides, swizzle, interleave, boxDim).
- Pass `Layout → TensorMapDescriptor?`:
  - compute candidate `globalStrides` & check multiples (16/32), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - check `globalAddress` alignment constraints, ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - check coupling constraints with `interleave` and swizzle size. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

---

### Theory B: **Template‑based synthesis as “restricted program synthesis over congruences” (SMT/ILP + modular linear algebra)**  
**Core idea:** Don’t try to encode *all* layouts; instead, define a **restricted template family** that matches the hardware descriptor, and solve for parameters.

Template:
\[
\text{addr}(\vec{i}) = \text{base} + \sum_d s_d \, i_d \quad \text{with constraints} \quad s_d \equiv 0 \pmod{16}, \dots
\]
plus “swizzle choice” from a finite set.

**Why it works:**
- TMA’s legality constraints are mostly:
  - **congruence constraints** (multiples of 16/32; alignment), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - **small enums** (`interleave`, `swizzle`), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - **bounds** on sizes and boxes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

That’s exactly the kind of structure modern solvers like: mixed integer + finite choices.

**Concrete optimization it enables (S2):**
- “Synthesize me a TMA descriptor or prove impossible” becomes a decidable, bounded search:
  - maximize vectorization
  - minimize expected bank conflicts
  - subject to: descriptor legality

**Hardware mapping:**
- If you succeed, you get the real TMA property: **one thread issues bulk copy** and other warps compute while data is in flight (warp specialization). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/hopper-tuning-guide/index.html))  

**Implementation sketch (C1):**
- Use an OMT solver to pick:
  - swizzle enum
  - `boxDim`, `elementStrides`
  - padding to make strides multiples of 16/32 ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- Output is a *witness object* that directly feeds `cuTensorMapEncodeTiled`.

---

#### Math‑to‑Hardware diagram (descriptor synthesis as witness production)

```text
Seed layout L (huge, F2-linear, XOR mixing allowed)
        |
        |  "Admissibility = witnessable by descriptor params"
        v
Solve for (tensorRank, globalStrides, boxDim, elementStrides, interleave, swizzle)
        |
        +-- success --> emit CuTensorMap + cp.async.bulk.tensor.*  (TMA path)
        |
        +-- fail ----> emit SM copy path (cp.async / ld/st) + surface cost cliff
```

---

## Bottleneck C — Spatial vs Temporal: layout says “where”, Hopper requires “when” (async + barriers + weak ordering)

### Failure mode (seed‑level axiom that breaks)
Seed A/B/C reason about mappings/composition, but Hopper’s fast path is *intrinsically temporal*:

- PTX `cp.async.bulk.tensor.*` exists and is **bulk async** with tensor maps and completion mechanisms. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  
- `cp.async.bulk.commit_group` explicitly states there is **no memory ordering guarantee between bulk ops within the same group**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  
- `mbarrier.test_wait` / `try_wait` define the visibility/ordering you must rely on; notably there is **no ordering/visibility guarantee for accesses after arrive and before test_wait**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

So the “layout” problem cannot be separated from a “schedule/effects” problem.

### Theory A: **Linear Capabilities / Separation‑Logic‑style effect system for async tokens**  
**Core idea:** Make “data movement completion” a *typed resource*.

- `tma.load` produces a token \(t\) representing “these bytes will become visible after the right wait.”  
- Using the data requires consuming \(t\) via `await`/`mbarrier.test_wait`.

**Why it works:**
- It matches PTX’s true semantics:
  - bulk ops are weakly ordered (so you can’t infer visibility from program order), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  
  - the acquire point is `mbarrier.test_wait` returning true. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

**Concrete optimization it enables (S2):**
- Safe **automatic pipelining**:
  - compiler can reorder independent copies/computes as long as it respects token dependencies
  - supports warp specialization correctly (copy warp produces tokens; compute warps consume)

**Implementation sketch (C1):**
- MLIR dialect idea:
  - `tma.async_load ... -> !async.token<smem_region>`
  - `async.await %tok`
  - memory effects: `tma.async_load` writes “future‑smem”, `await` commits visibility.
- Verification pass checks:
  - every use of smem tile is dominated by an `await`
  - tokens are not duplicated (linear) unless explicitly “broadcast/multicast”.

---

### Theory B: **Trace theory / Pomsets / Partial‑Order Semantics for GPU pipelines**  
**Core idea:** The semantics of async pipelines is a **partial order** of events, not a total order.

Model:
- events: issue‑copy, commit‑group, barrier arrive, barrier wait, compute consume  
- independence relation: operations commute if they touch disjoint resources or are ordered only by performance, not correctness  
- schedules correspond to traces in a partially commutative monoid (trace monoid).

**Why it works:**
- It mathematically expresses “weak ordering”: you can’t reason by linear program order; you reason by partial order constraints.
- It directly supports transformations like:
  - move prefetch earlier
  - overlap compute with in‑flight copies
  - reorder independent groups

**Concrete optimization it enables (S2):**
- A compiler pass that *synthesizes* a high‑overlap pipeline under:
  - resource constraints (TMA engine, SM issue slots)
  - correctness constraints (token/barrier partial order)

**Hardware mapping:**
- Precisely matches the `cp.async.bulk.tensor` + `mbarrier` protocol in PTX. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

**Implementation sketch (C1):**
- Build a small DAG IR for each pipeline region (per CTA / per warpgroup).
- Use list scheduling with explicit “happens‑before” edges derived from tokens.
- Lower to:
  - `cp.async.bulk.tensor.*` forms ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  
  - `mbarrier.*` waits with the documented acquire semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

---

#### Math‑to‑Hardware diagram (tokens make time explicit)

```text
[Layout L]  (spatial mapping)
    |
    +--> choose movement primitive:
         - TMA bulk tensor copy
         - cp.async
         - ld/st
    |
    v
[Temporal IR]
  issue_copy  -> token t
  compute     (cannot read tile yet)
  await t     (mbarrier.test_wait acquire)
  compute     (tile visible)
```

---

## Bottleneck D — Bank conflict “proofs” break when the predicate is opcode‑dependent (MI300 LDS lane‑groups)

### Failure mode (seed‑level axiom that breaks)
On AMD MI‑series GPUs, LDS is banked and conflicts serialize throughput. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  
But the ROCm guidance highlights a key microarchitectural fact:

- `ds_write_b128` is conflict‑free if no conflict within 8 contiguous lane groups (0–7, 8–15, …). ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  
- `ds_read_b128` uses *different* lane groupings (paired, non‑contiguous sets like \(0{:}3 + 20{:}23\), etc.). ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  

So “bank conflict free” is a property of \((\text{layout} \times \text{opcode} \times \text{lane‑partition})\), not layout alone.

### Theory A: **Group action + kernel/coset analysis of bank mappings (opcode selects the subgroup)**  
**Core idea:** Formalize bank mapping as a homomorphism into a finite group and analyze collisions via its kernel.

Let:
- lanes in a lane group \(G\) be a finite set (often modeled as \(\mathbb{Z}_{32}\) or \(\mathbb{Z}_{64}\) depending on wave)
- bank mapping be:
  \[
  \beta(addr) = \left(\left\lfloor \frac{addr}{w}\right\rfloor \bmod B\right)
  \]
  where \(B=32\) banks and \(w=4\) bytes per bank on MI‑series per ROCm guidance ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  

An opcode induces a partition of lanes into subgroups \(G_1,\dots,G_k\) (the lane‑groups in the ROCm blog). ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  
Conflict freedom requires injectivity of \(\beta \circ addr(l)\) restricted to each \(G_j\).

**Why it works:**
- Makes the “instruction‑indexed predicate” first‑class:
  - opcode \(I\) selects the lane subgrouping \(P(I)\)
  - objective is “injective on each subgroup”

**Concrete optimization it enables (S2):**
- Synthesize an XOR reindexing (seed‑A style) that is simultaneously conflict‑free for both read and write lane partitions. The ROCm blog explicitly demonstrates XOR reindexing to fix a read conflict while preserving write behavior. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  

**Implementation sketch (C1):**
- Add an MLIR analysis pass `bank_conflict(layout, opcode, vecwidth)`:
  - compute bank IDs per lane group
  - emit metric: max conflict degree / expected serialization
- Feed that metric into the layout optimizer.

---

### Theory B: **Transformation semigroups / Krohn–Rhodes decomposition as an “ISA semantics algebra”**  
**Core idea:** Treat the opcode’s lane grouping/permutation behavior as a **finite transformation semigroup** acting on lanes. Then reason compositionally.

This is not abstract indulgence; it maps to a concrete need:
- Different LDS instructions define different lane partitions → different transformations.
- You need a *systematic* way to model and compose these transformations.

**Why it works:**
- Krohn–Rhodes theory decomposes finite automata/semigroups into wreath products of groups and aperiodic components (exactly the algebra of “hierarchical composition”).  
- A very recent result: **decidability of Krohn–Rhodes complexity for all finite semigroups and automata** was proved (open problem >50 years). ([arxiv.org](https://arxiv.org/abs/2406.18477?utm_source=openai))  

**Concrete optimization it enables (S2):**
- Build an **opcode‑semantics library** where each instruction’s lane behavior has a canonical decomposition.
- Use that to automatically derive:
  - the “conflict predicate” partitions
  - safe commuting transformations
  - which swizzles must be applied before/after a given opcode to preserve conflict freedom

**Implementation sketch (C1):**
- Offline: encode each relevant LDS opcode semantics as a transformation on lane indices; compute decomposition/canonical form.
- Online (compile time): choose layout transformations that make bank mapping injective under the opcode’s induced partition.

---

#### Math‑to‑Hardware diagram (opcode chooses the equivalence relation)

```text
Layout L  : lane -> address
Opcode I  : induces lane partition P(I)

BankConflictFree(L, I) :=
  for each lane-group G in P(I),
    bank(addr_L(l)) are all distinct for l in G
```

---

## Bottleneck E — “Don’t handwrite layouts; solve for them” under hardware constraints + compile-time budgets

### Failure mode (what breaks in practice)
Seed A gives a huge layout space and powerful linear reasoning. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
Seed B gives a unifying relation language and shows XOR/swizzle can be expressed via mod‑2 constraints. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
But neither yields an end‑to‑end *hardware‑typed synthesizer* that:
- respects TMA descriptor legality (alignment/stride/swizzle coupling), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- respects async protocol constraints (weak ordering + acquire waits), ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  
- respects opcode‑indexed bank conflict predicates. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  

### Theory A: **SMT/OMT (with finite-field / mod‑arithmetic support) as the layout backend**  
**Core idea:** Cast “layout + descriptor + schedule” as a constraint system with an objective:
- constraints:
  - descriptor legality (congruences + bounds)
  - bank conflict freedom predicates
  - schedule correctness (token partial order)
- objective:
  - minimize conflicts, maximize vector width, minimize register pressure, etc.

**Why it works:**
- The “XOR world” is naturally finite‑field/mod arithmetic.
- Recent solver work has improved finite‑field reasoning by using *multiple simpler Gröbner bases rather than one full basis* (Split Gröbner Bases, integrated in cvc5). ([eprint.iacr.org](https://eprint.iacr.org/2024/572?utm_source=openai))  

**Concrete optimization it enables (S2):**
- **Swizzle synthesis under legality constraints**:
  - “Find me a swizzle making bank conflicts zero for `ds_read_b128` lane groups *and* legal for TMA swizzle enums.”
  - If impossible, solver returns UNSAT → compiler chooses fallback early.

**Implementation sketch (C1):**
- Use a two-level strategy:
  1. small finite enumeration over enums (swizzle/interleave)
  2. SMT for the remaining modular constraints and objective
- Emit a **certificate** (model) stored in IR, so later passes only verify.

---

### Theory B: **Equality Saturation (e-graphs) + cost-guided extraction for layout expressions**  
**Core idea:** Layouts are expressions in a rewrite system (tile, reshape, transpose, XOR-swizzle, etc). Use equality saturation to explore equivalent forms and then extract the best under a cost model.

Two relevant recent pieces:
- **MLIR-native equality saturation dialect (`eqsat`)**: keep the e-graph inside the compiler IR rather than treating it as an external “one-shot” optimizer. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
- Equality saturation + **Monte Carlo Tree Search** applied to tensor graphs to mitigate phase-ordering / memory blowups during exploration. ([arxiv.org](https://arxiv.org/abs/2410.05534?utm_source=openai))  

**Why it works:**
- Layout algebra is rife with equivalences (commuting reshapes, reassociating tilings, factoring swizzles).
- You want a canonical form that exposes descriptor admissibility and vectorization opportunities.

**Concrete optimization it enables (S2):**
- **Swizzle normal form** and “descriptor‑friendliness rewriting”:
  - Push XOR reindexing into low bits only (where TMA swizzle might simulate it)
  - Factor layout conversions to reduce shuffle instructions
  - Canonicalize compositions to reduce code size

**Implementation sketch (C1):**
- Add a Triton/MLIR pass:
  - build e-graph of layout expressions
  - apply rewrite rules until saturation/timeout
  - extract best candidate using a cost model that includes:
    - “TMA admissible?” (huge bonus)
    - predicted bank conflicts
    - predicted register pressure / shuffle count

---

#### Math‑to‑Hardware diagram (e-graphs + SMT as complementary)

```text
E-graph exploration (cheap algebraic equivalences)
      |
      v
Small set of candidate normal forms
      |
      v
SMT/OMT: choose descriptor params + prove legality + optimize bank conflicts
      |
      v
Emit: (layout, descriptor, schedule tokens) with certificates
```

---

# 2) Literature Scan — 3–5 “recent breakthroughs” (pure-math / group-theoretic or adjacent) that matter here

These are not “layout papers”; they’re *math/theory tools* that directly support the arsenal above.

1. **Decidability of Krohn–Rhodes complexity (finite semigroups & automata)**  
   Settles a long-open problem: deciding the complexity \(k\) for finite semigroups/automata is decidable. ([arxiv.org](https://arxiv.org/abs/2406.18477?utm_source=openai))  
   **Why relevant:** opcode semantics (lane partitions/permutations) are naturally modeled astransformations; a decidable decomposition theory supports building canonical “ISA semantics algebras” for bank-conflict predicates.

2. **Constructive recognition algorithms for special linear groups (algorithmic group theory)**  
   Provides new constructive recognition for \(\mathrm{SL}(d,q)\) given generators, enabling efficient computations like word problems (implemented in GAP). ([arxiv.org](https://arxiv.org/abs/2404.18860?utm_source=openai))  
   **Why relevant:** TMA admissibility can be frame membership/witness extraction in a small “allowed swizzle group”; constructive recognition is the blueprint for turning “membership” into “descriptor parameters”.

3. **Split Gröbner bases for satisfiability modulo finite fields (solver breakthrough; CAV’24)**  
   Uses multiple simpler Gröbner bases instead of one full basis; implemented in cvc5; improves scaling on bitsum-heavy constraints. ([eprint.iacr.org](https://eprint.iacr.org/2024/572?utm_source=openai))  
   **Why relevant:** XOR-heavy swizzle constraints are naturally finite-field; this supports an SMT backend for “layout legality + optimization” where \(\mathbb{F}_2\) structure is first-class.

4. **`eqsat`: MLIR-native equality saturation dialect (compiler-theory breakthrough)**  
   Proposes keeping e-graphs as an MLIR dialect to integrate equality saturation across passes and avoid translation overhead. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
   **Why relevant:** layout canonicalization and “swnormal forms” are rewrite problems; this is a direct path to implementability (C1) and compile-time control (C2 metrics).

5. **Equality saturation + Monte Carlo Tree Search for tensor graph optimization**  
   Uses MCTS to guide which rewrites to apply so the e-graph construction doesn’t reintroduce phase-ordering problems via memory limits. ([arxiv.org](https://arxiv.org/abs/2410.05534?utm_source=openai))  
   **Why relevant:** your layout/schedule search space is enormous; this is a principled way tooritize exploration under compile-time budgets.

---

## What this sets up for Stage 2 (preview)

Stage 2 can now be a clean synthesis:

- **Replace** “layouts are \(\mathbb{F}_2\) matrices” with “layouts live in a *tower*”:
  1. \(\mathbb{F}_2\) (XOR swizzles) for low-bit structure,
  2. \(\mathbb{Z}\)/mixed-radix modules for non-\(2^n\) and carries,
  3. Presburger/semilinear sets for ragged domains,
  4. A *refinement type* that carves out the **TMA‑admissible** sublanguage. ([docs.nvidia.com](cs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

- **Add** a temporal/effect layer that matches PTX’s actual async + barrier semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

If you want, I can turn the above into a single Stage‑2 design with:
- a **typed Layout IR** (spatial + temporal tokens),
- a **TMA descriptor synthesis pass** (template + solver),
- an **opcode-indexed bank model** (MI300 lane itions),
- and an evaluation matrix that includes **compile time**, **code size**, and **measured bank conflict rate**.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
3. [Categorical Foundations for CuTe Layouts](https://arxiv.org/pdf/2601.05972v1)
4. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.0/hopper-tuning-guide/index.html)
5. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
6. [Constructive Recognition of Special Linear Groups](https://arxiv.org/abs/2404.18860?utm_source=openai)
7. [PTX ISA :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html)
8. [Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile Framework — ROCm Blogs](https://rocm.blo.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
9. [Decidability of Krohn-Rhodes complexity for all finite semigroups and automata](https://arxiv.org/abs/2406.18477?utm_source=openai)
10. [Split Gröbner Bases for Satisfiability Modulo Finite Fields](https://eprint.iacr.org/2024/572?utm_source=openai)
11. [eqsat: An Equality Saturation Dialect for Non-destructive Rewriting](https://arxiv.org/abs/2505.09363?utm_source=openai)
12. [Optimizing Tensor Computation Graphs with Equality Saturaton and Monte Carlo Tree Search](https://arxiv.org/abs/2410.05534?utm_source=openai)
