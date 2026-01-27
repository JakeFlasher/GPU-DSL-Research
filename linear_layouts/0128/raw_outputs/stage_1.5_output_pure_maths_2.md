## Section 1 — The Theoretical Toolbox (a “Math Stack” that strictly generalizes \( \mathbb{F}_2 \) layouts)

### First: what the seed *really* commits to (so we know what to generalize)
**Linear Layouts** is (mathematically) a calculus of **linear maps on bit-vectors** over \( \mathbb{F}_2 \), i.e., the world of homomorphisms between **finite 2-groups** like \( (\mathbb{Z}/2\mathbb{Z})^k \). It explicitly leans on the fact that many GPU parameters and Triton layout subdivisions are **powers of two**, and it explicitly calls out that **power-of-two shapes** are a primary limitation (mitigated by padding+masking), while **flips/slices** require extending to **affine layouts**. ([arxiv.org](https://arxiv.org/html/2505.23819v3))

**LEGO** lives in a different layer: a **layout-expression frontend** that emphasizes **(mostly) bijective permutations** of index space (plus extensions for partial tiles/injective mappings), and it uses those to generate indexing expressions / instantiate templates (then the backend does the real HW lowering). ([arxiv.org](https://arxiv.org/html/2505.08091))

So the “theoretical arsenal” should not just be “bigger matrices” — it must enlarge *three* axes:

```
(1) Layout algebra:   F2-linear  ->  Z / Z_m linear  ->  affine  ->  piecewise/partial
(2) Schedule algebra: sync DAG   ->  async partial orders / resources / proxies
(3) Search mechanism: heuristics ->  constraint solving / rewriting / synthesis
```

---

### Bottleneck 1: **Non-power-of-two / prime dimensions + ragged extents** (padding+masking cliff)
Linear Layouts explicitly restricts to power-of-two shapes (mitigate via larger tensors + mask OOB). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
That’s poison for LLM decode + MoE where *the* shape is “whatever the batch/router produced”.

#### Theory A: **Finite Abelian Groups / \( \mathbb{Z} \)-Modules + Smith Normal Form (SNF)**
**Why it works**
- Replace “bit-vectors over \( \mathbb{F}_2 \)” with **mixed-radix index groups**: represent an index space as a product of cyclic groups  
  \[
  G \;\cong\; \mathbb{Z}_{n_0} \times \mathbb{Z}_{n_1} \times \cdots
  \]
  with \( n_i \) not necessarily powers of two.
- A layout becomes a **group homomorphism** (or module homomorphism) between such products.
- Classification + invertibility conditions become **gcd / divisibility** questions instead of “rank over \( \mathbb{F}_2 \)”.
- SNF gives a canonical decomposition of integer-linear maps (constructively!), which is the “right” analogue of Gaussian elimination over \( \mathbb{F}_2 \).

**Compiler payoff**
- You can *decide* contiguity/vectorization legality for **non-power-of-two strides** (e.g., stride 160) via modular constraints, instead of assuming bit-aligned structure.
- Layout conversion synthesis becomes “solve linear congruences” rather than “find right inverse in \( \mathbb{F}_2 \)”.

**Implementation sketch (Triton/MLIR-feasible)**
- Store layouts as integer matrices over \( \mathbb{Z} \) (or over \( \mathbb{Z}_m \) when reasoning about congruences like bank selection).
- Use SNF/HNF routines to:
  - check solvability of conversion,
  - produce a minimal-normal-form conversion plan,
  - derive contiguity/alignment constraints without padding-to-power-of-two.

#### Theory B: **Presburger Arithmetic / Polyhedral Model (parametric, piecewise affine)**
**Why it works**
- Polyhedral compilation represents iteration and access sets as **integer points in polyhedra**, and transformations as **affine** maps (with parameters). LEGO explicitly situates itself relative to polyhedral frameworks (Pluto/PPCG/etc.) as complementary. ([arxiv.org](https://arxiv.org/html/2505.08091))
- Ragged extents become *parameters*; partial tiles become **piecewise affine** regions (“interior tile” vs “edge tiles”).

**Compiler payoff**
- You stop treating raggedness as “masked waste”; you can generate:
  - one fast path for the **full interior** (no masks, vectorized),
  - one or more small slow paths for edges.
- You get a mathematically clean place to encode **“shape buckets”** for runtime specialization.

**Implementation sketch**
- In tt/ttg: attach parametric bounds; split kernel variants by symbolic constraints (e.g., \( L_{kv} \bmod 128 \)).
- Use JIT multi-versioning keyed on runtime shapes (fits Triton/PyTorch compilation caching).

---

### Bottleneck 2: **Flips, slices, negative strides, partial tiles** (layout is not a global bijection)
Linear Layouts says flips/slices aren’t expressible as linear layouts; affine layouts would capture them. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
LEGO’s base abstraction is “bijective mapping (permutation)”, with extensions for partial tiles/injective mappings. ([arxiv.org](https://arxiv.org/html/2505.08091))

#### Theory A: **Affine Groups \( \mathrm{AGL}(n,R) = R^n \rtimes \mathrm{GL}(n,R) \)** (semidirect products)
**Why it works**
- Extend \( x \mapsto Ax \) to \( x \mapsto Ax + b \).
- Offsets \(b\) are exactly “slice base pointers / sub-tiles”.
- Reflections (negative strides) correspond to multiplying by \( -1 \) (over \( \mathbb{Z} \)), which is impossible in pure \( \mathbb{F}_2 \) because \( -1 = 1 \).

**Compiler payoff**
- You can represent a *huge* class of practical layouts (slices, views, reinterprets) **without falling back** to “pad+mask and pray”.

**Implementation sketch**
- Extend layout IR nodes from “matrix” to “(matrix, translation)”.
- Keep conversions as group operations (compose/invert).
- Lowering uses adds + muls; for \(R=\mathbb{Z}\) you can still strength-reduce to shifts when factors are powers of two.

#### Theory B: **Groupoids / Inverse Semigroups (partial bijections)**
**Why it works**
- Masked tiles + ragged tensors are not total functions; they are **partial maps**.
- Inverse semigroups formalize “partial symmetries” (partial permutations), and groupoids formalize “invertible morphisms only where defined”.

**Compiler payoff**
- The compiler can reason about *where* an inverse exists (interior region) and avoid overpaying masks/shuffles globally.
- This gives a principled way to represent “validity domains” as first-class objects instead of ad-hoc predicates.

**Implementation sketch**
- Pair each layout with a domain predicate (often Presburger-definable).
- Propagate domains through ops; split kernels by domains (“interior”/“edge”).

---

### Bottleneck 3: **Hopper/Blackwell-era asynchrony (TMA + async proxy + barriers)** is not “just layout”
On Hopper-class GPUs, bulk tensor copies and some tensor-core ops are modeled via an **async proxy**, and ordering with normal loads/stores is *not* automatic; you need proxy fences. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai))  
PTX exposes `cp.async.bulk.tensor` with `.1d`–`.5d`, cluster destinations, and multicast. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
CUDA documents TMA pipelines using mbarriers and explicit fencing (`fence.proxy.async...`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/?utm_source=openai))

#### Theory A: **Petri Nets / Event Structures + Max-Plus (tropical) algebra**
**Why it works**
- TMA pipelines are naturally modeled as:
  - **events** (initiate copy, arrive mbarrier, wait, wgmma issue, fence),
  - **resources** (TMA engine, warp groups, shared buffers),
  - **tokens** (buffer slots / mbarrier transaction counts).
- Petri nets capture correctness + resource constraints. Max-plus algebra captures throughput / initiation interval (II) of pipelines.

**Compiler payoff**
- You can treat “transport” as a schedulable IR with legality constraints:
  - ensure mbarrier/wait correctness,
  - choose stage count (double-buffer/triple-buffer),
  - decide warp specialization boundaries.

**Implementation sketch (Triton backend)**
- Introduce a first-class “transport ops” dialect:
  - `tma.load(tile, tensormap, mbarrier)`
  - `mbarrier.wait`
  - `proxy.fence`
- Run a schedule-synthesis pass (constraint-based) that targets max overlap.

#### Theory B: **Separation Logic / Effect Systems for async-proxy correctness**
**Why it works**
- CUDA states that async-proxy ops and normal loads/stores are not automatically ordered; a **proxy fence is required** to synchronize across proxies. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai))
- This is exactly what effect systems were built for: track “this op reads shared via async proxy” vs “this op writes shared via generic proxy” and insert fences.

**Compiler payoff**
- Makes aggressive reordering/pipelining *safe* by construction.
- Prevents the “it works on A100, races on H100” class of bugs.

**Implementation sketch**
- Annotate IR ops with effects: `(proxy=generic|async, space=shared|global|cluster_shared, read|write)`.
- Add an effect-aware fence insertion + scheduling pass.

---

### Bottleneck 4: **Cluster-level distribution + multicast** (new locality tier beyond CTA)
PTX allows `cp.async.bulk.tensor` into `.shared::cluster` and has `.multicast::cluster` via a `ctaMask` (copy to multiple CTAs’ shared memory within a cluster). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
This breaks the seed’s implicit “CTA is the top locality” assumption.

#### Theory A: **Group actions + Wreath Products (hierarchical permutation structure)**
**Why it works**
- GPU hierarchy is inherently hierarchical: cluster → CTA → warp → thread → register.
- Hierarchical layouts are naturally described by **wreath products** (outer permutation of blocks + inner permutations within blocks).
- This provides canonical factoring: “what part of the mapping is cluster-level vs warp-level?”

**Compiler payoff**
- You can introduce a **cluster dimension** in the layout algebra without exploding ad-hoc cases.
- It enables systematic discovery of when multicast helps (common K/V tiles shared across CTAs).

**Implementation sketch**
- Extend distributed layout domain to include `cluster_cta_rank` as an axis.
- Add cluster-aware anchor ops and conversions.

#### Theory B: **Scoped/typed resource semantics (monoidal categories as the mathematical spine)**
**Why it works**
- The real problem is illegal composition: mixing CTA-scoped shared memory with cluster-scoped shared memory and synchronization.
- A compositional semantics (think “typed wiring diagram”) forces scope correctness.

**Compiler payoff**
- “Cluster DSMEM pointer” becomes a distinct type; only cluster-scoped ops can consume it.
- Prevents entire classes of lowering mistakes and clarifies IR invariants.

**Implementation sketch**
- In MLIR: use distinct address spaces / types for `shared::cta` vs `shared::cluster`.
- Verification pass enforces well-scoped use; codegen can assume legality.

---

### Bottleneck 5: **Gather/scatter + indirection (KV paging, MoE routing)** breaks “warp-contained gather”
Linear Layouts’ gather fast path requires that all axis elements of `src` and `index` reside **within the same warp**, otherwise it can’t do the shuffle-based optimization. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
That condition is brittle under routing/paging.

#### Theory A: **Permutation-group orbit / coset decomposition (computational group theory view)**
**Why it works**
- Treat a layout as an action of a subgroup of the symmetric group on index positions.
- “Warp-contained” becomes: the orbit of the axis under the subgroup “generated by warp-local moves” stays inside one warp.
- Cosets/orbits generalize the seed’s specific matrix-zero test into a hierarchy:
  - warp-contained,
  - CTA-contained,
  - cluster-contained.

**Compiler payoff**
- You can systematically pick between:
  - warp shuffles,
  - shared-memory transpose,
  - cluster DSMEM exchange,
  - or (on Hopper) transport-assisted patterns.

**Implementation sketch**
- Build a small “movement group” library over resource indices.
- Compute orbit partitions; choose the cheapest movement primitive per orbit size.

#### Theory B: **Synthesis: SMT/ILP + e-graph rewriting of movement plans**
**Why it works**
- Gather lowering is a search problem with hard constraints:
  - register budget (Register Pressure / occupancy cliffs),
  - latency overlap,
  - legal primitives (shuffle width, shared bank constraints, transport modes).
- PTX even exposes specialized tensor-copy load modes (e.g., variants include richer modes in newer docs), suggesting that “transport can sometimes express structured gathers.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/hopper-tuning-guide/parallel-thread-execution/index.html?utm_source=openai))

**Compiler payoff**
- Instead of hardcoding “if warp-contained then shuffle else shared”, you can *solve*:
  - “find a movement plan with cost < X and regs < R”.

**Implementation sketch**
- Encode candidate movement primitives as rewrite rules in an e-graph.
- Use an ILP/SMT cost model (instructions, registers, barrier waits) to select the optimal equivalence-class representative.

---

### Bottleneck 6: **Target-specific shared-memory/LDS bank-conflict rules (AMD wave64 phases)** invalidate a single “bank model”
AMD MI-series LDS access is wavefront-64 and the bank-conflict rule depends on the specific LDS instruction (e.g., `ds_write_b128` phases). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))  
A swizzle that is optimal under “NVIDIA-ish assumptions” can be wrong under these phase groupings.

#### Theory A: **Affine linear algebra over rings \( \mathbb{Z}/B\mathbb{Z} \)** (bank index = congruence class)
**Why it works**
- Bank index is fundamentally a congruence: \( \text{bank}(\text{addr}) = (\text{addr}/\text{word}) \bmod B \).
- Swizzles are affine transforms on coordinates; the natural home is the **affine group over a ring**, not \( \mathbb{F}_2 \).

**Compiler payoff**
- One parametric framework can emit:
  - XOR-like transforms for NVIDIA shared banks,
  - XOR-like transforms tuned to AMD’s lane-phase rules.

**Implementation sketch**
- Define a “bank objective” as modular distinctness constraints on address expressions.
- Solve in \( \mathbb{Z}/32\mathbb{Z} \) (or the right \(B\)) rather than bit-matrix-only.

#### Theory B: **Constraint solving with instruction-parametric lane partitions**
**Why it works**
- AMD docs explicitly specify phase groupings for `ds_write_b128`/`ds_read_b128` and show why naive layouts conflict and XOR fixes can help. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))
- This is naturally expressed as constraints:
  - For each phase group \(P\), require bank(addr(lane)) all distinct within \(P\).

**Compiler payoff**
- You can generate swizzles that are *provably* conflict-free for the exact instruction mix you emit.

**Implementation sketch**
- Emit a bank-model description per target ISA + instruction width.
- Feed it to a small SAT/SMT solver that searches over a restricted swizzle family (e.g., XOR of selected coord bits).

---

### Bottleneck 7 (cross-cutting): **Layout × schedule × conversion selection is a word problem / search problem**
Both seeds implicitly rely on heuristics to keep the search tractable. Linear Layouts uses right-inverse selection + heuristics; LEGO emphasizes compositional specs but still needs backend decisions. ([arxiv.org](https://arxiv.org/html/2505.23819v3))

#### Theory A: **Rewrite systems / Normal forms (computational group theory backbone)**
**Why it works**
- “Layout expressions” and “movement plans” form an algebra under composition.
- If you can define a terminating confluent rewrite system, you can canonicalize and compare plans efficiently (“normal form”).

**Why it’s not a random analogy**
- Recent group theory work shows practical power of rewrite systems: e.g., quadratic-time word problem solutions via length-preserving rewrite rules for classes of Artin groups. ([arxiv.org](https://arxiv.org/abs/2412.12195?utm_source=openai))

**Compiler payoff**
- Enables *deterministic* canonicalization of layout+conversion chains, shrinking autotuning space.

#### Theory B: **Equality saturation (e-graphs) + costed extraction**
**Why it works**
- E-graphs excel at exploring huge equivalence spaces under rewrite rules.
- You can encode:
  - algebraic identities (layout composition),
  - hardware identities (shuffle+permute equivalences),
  - schedule identities (commuting independent async ops, subject to proxy/fence rules).

**Compiler payoff**
- Lets you unify:
  - layout optimization,
  - conversion placement,
  - and transport scheduling
  into one global “best cost plan” extraction.

---

## Section 2 — Literature Scan (3–5 recent pure-math/group-theory breakthroughs that are unexpectedly relevant)

1) **McKay Conjecture proved (character degrees)**  
   Annals lists “The McKay Conjecture on character degrees” by Cabanes & Späth (accepted April 2025), proving the equality of certain character counts for any prime \( \ell \). ([annals.math.princeton.edu](https://annals.math.princeton.edu/articles/22056?utm_source=openai))  
   *Why it’s relevant:* it’s a modern template for “global invariants from local structure” — conceptually similar to deriving global layout/schedule guarantees from local (warp/CTA/cluster) constraints.

2) **Brauer’s Height Zero Conjecture completed (odd primes)**  
   Malle, Navarro, Schaeffer Fry, and Tiep complete the proof of Brauer’s Height Zero Conjecture (Annals 2024). ([annals.math.princeton.edu](https://annals.math.princeton.edu/2024/200-2/p04?utm_source=openai))  
   *Why it’s relevant:* heavy use of structural decomposition; inspires “reduce global optimization to a finite family of hard cases,” which is how you want to tame layout×schedule search (reduce to anchor patterns).

3) **Alperin–McKay + Brauer height zero for prime 2 completed**  
   Ruhstorfer completes the proof for the prime \(2\) case (Annals 2025). ([annals.math.princeton.edu](https://annals.math.princeton.edu/2025/201-2/p02?utm_source=openai))  
   *Why it’s relevant:* again, “hard corner case completion” mentality maps well to GPU backends where \(2\) is special (warp structure, bit-level mappings), but correctness/perf still hinges on handling the weird residual cases.

4) **Efficient rewrite systems for word problems in classes of Artin groups (quadratic time)**  
   A 2024 arXiv result proves word problem solvability using length-preserving rewrite rules (quadratic time) for certain Artin groups. ([arxiv.org](https://arxiv.org/abs/2412.12195?utm_source=openai))  
   *Why it’s relevant:* compilers need canonicalization/normal forms for composed transformations; this is the “pure math → algorithmic rewriting” bridge.

5) **PTX/CUDA formalization of async proxies and bulk tensor copies (math-adjacent but decisive for correctness)**  
   Not “pure math”, but it *forces* the right mathematical model of asynchrony: CUDA distinguishes generic vs async proxy and requires proxy fences for correct cross-proxy ordering. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai))  
   *Why it’s relevant:* it’s the concrete semantic constraint your “math of asynchrony” must encode; otherwise your schedule algebra is unsound.

---

If you want, I can turn this toolbox into a single coherent “successor abstraction” (a candidate ASPLOS’26/’27 pitch): **Affine/Modular Layouts + Transport Schedules**, where layouts live in \( \mathrm{AGL}(n,\mathbb{Z}) \) / \( \mathrm{AGL}(n,\mathbb{Z}_m) \) (with partial domains), and TMA/wgmma pipelines live in a Petri-net-like schedule IR with proxy-aware effect typing — all solved via equality saturation + constrained extraction, evaluated on H100/MI300 using TritonBench + Nsight Compute.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping](https://arxiv.org/html/2505.08091)
3. [3.2. Advanced Kernel Programming — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai)
4. [PTX ISA :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai)
5. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/?utm_source=openai)
6. [1. Introduction — PTX ISA 8.8 documentation](https://docs.nvidia.com/cuda/archive/13.0.1/hopper-tuning-guide/parallel-thread-execution/index.html?utm_source=openai)
7. [Understanding AMD GPU LDS and Bank Conflicts — Composable Kernel 1.1.0 Documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai)
8. [Rewriting in Artin groups without A\_3 or B\_3 subdiagrams](https://arxiv.org/abs/2412.12195?utm_source=openai)
9. [The McKay Conjecture on character degrees | Annals of Mathematics](https://annals.math.princeton.edu/articles/22056?utm_source=openai)
10. [Brauer’s Height Zero Conjecture | Annals of Mathematics](https://annals.math.princeton.edu/2024/200-2/p04?utm_source=openai)
11. [The Alperin–McKay and Brauer’s height zero conjecture for the prime 2 | Annals of Mathematics](https://annals.math.princeton.edu/2025/201-2/p02?utm_source=openai)