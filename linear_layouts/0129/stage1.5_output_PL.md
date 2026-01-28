## 0) Access verification (S3)

I verified I can fetch and parse all three seed artifacts you specified:

1. **Linear Layouts (arXiv HTML v3)** — accessible and parsable. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
2. **ISL / Integer Set Relations (arXiv HTML v1)** — accessible and parsable. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
3. **Categorical Foundations for CuTe Layouts (arXiv PDF v1)** — accessible and parsable. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

---

# 1) The Theoretical Toolbox (2 theories per Stage‑1 bottleneck)

I’m going to treat your Stage‑1 cliffs as *compiler obligations*:

- **Obligation A (Admissibility):** “Layout is valid” ⇒ “Layout is realizable by a hardware fast path (TMA/WGMMA/etc.)”.
- **Obligation B (Temporality):** “Mapping is correct” ⇒ “Asynchrony + barriers are correct *and* overlap‑optimal”.
- **Obligation C (Non‑\(2^n\)):** “Shape isn’t power‑of‑two / ragged” ⇒ “No padding‑induced tail effect cliffs”.
- **Obligation D (Bank semantics):** “Conflict‑free in the model” ⇒ “Conflict‑free under the **opcode‑specific** lane phasing.”
- **Obligation E (Compile-time):** “Formal reasoning exists” ⇒ “JIT/compile budgets aren’t destroyed.”

Below, each bottleneck gets **two distinct** theoretical frameworks that *generalize / subsume* the seed’s \(\mathbb{F}_2\) linear maps, and each framework is mapped immediately to a **concrete compiler optimization** (S2) and **silicon constraints** (S1).

---

## Bottleneck 1 — **TMA Descriptor Admissibility Gap** (Swizzle Atomicity + Descriptor Constraints)

### The cliff (hardware reality)
The seed’s \(\mathbb{F}_2\) layout language is “big”: arbitrary XOR‑mixing on address bits is allowed. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
But **TMA** is *descriptor-driven* and only accepts a constrained parameterization: rank is restricted, `globalAddress`/`globalStrides` have alignment/multiple constraints, and `swizzle` is a tiny enum family. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/hopper-tuning-guide/index.html))  
Once you cross that boundary, you fall off the **warp-specialized “single-thread issue”** path and land in SM-driven address-gen + loads/stores. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/hopper-tuning-guide/index.html))  

Also: PTX makes it explicit that `cp.async.bulk.tensor` is descriptor+coords, can be cluster-multicast, and is weakly ordered, completed via mbarriers. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

---

### **Theory A: Refinement Types / Proof-Carrying Layouts (Hardware-Refined Subtyping)**

**Core idea:** introduce a *semantic refinement*  
\[
\texttt{Layout} \; \supset \; \texttt{TmaLayout} \; \supset \; \texttt{TmaLayout\_Tiled}
\]
where membership in `TmaLayout` *carries proofs* of the CUDA Driver API constraints (rank, alignment, stride multiples, swizzle legality, inner-dimension bound, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

This turns “TMA cliff” into a **typed planning decision** instead of a late failure.

#### Why it generalizes \(\mathbb{F}_2\)
The \(\mathbb{F}_2\) map is still allowed as a *specification language*, but you introduce a *refinement judgment*:

\[
\Gamma \vdash L : \texttt{Layout}
\quad\text{and}\quad
\Gamma \vdash \textsf{admissible\_tma}(L) \Rightarrow
\Gamma \vdash L : \texttt{TmaLayout}
\]

That judgment is the missing axiom in Seed A (“valid layout ⇒ realizable”). Seed A explicitly embraces power-of-two + bit reasoning; the refinement is the bridge to descriptor legality. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

#### Concrete optimization (S2)
Once `TmaLayout` is a first-class type:
- **Layout search / propagation** can be constrained to stay in the `TmaLayout` subtype when the pipeline expects TMA.
- Equality of layouts is no longer purely algebraic; it becomes *algebra + admissibility*.

#### Implementation hook (C1)
- **MLIR dialect design:**  
  - `layout.layout` (general)  
  - `nvgpu.tma_desc` (opaque) with verifier enforcing driver constraints
  - `nvgpu.tma_load` returns a token (see Bottleneck 2)
- Lowering uses `cuTensorMapEncodeTiled` only when proofs exist; otherwise forces fallback (and surfaces cost).

#### Math-to-hardware diagram
```text
Seed A layout L (F2-linear, XOR-mixing)  ────────┐
                                                │  refinement / proof search
                                                v
                                      TmaLayout(L) ?  (alignment/stride/swizzle)
                                                │
                          yes ───────────────────┼─────────── no
                                                │
                                                v
     CUDA Driver API: cuTensorMapEncodeTiled  fallback: SM copy / cp.async / ld/st
     PTX: cp.async.bulk.tensor + mbarrier
```
TMA + warp specialization motivation is explicitly called out in the Hopper tuning guide. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/hopper-tuning-guide/index.html))  

---

### **Theory B: Constraint Programming / SMT Synthesis (Layout → Descriptor Solving)**

**Core idea:** treat descriptor parameters as *unknowns* and solve:
- variables: `globalStrides`, `boxDim`, `interleave`, `swizzle`, `elementStrides`, …
- constraints: alignment + multiples + bounds + inner-dimension vs swizzle size. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

This is exactly the right mathematical shape: mostly **linear arithmetic + congruences** (multiples of 16/32, bounded integers) with a small finite choice (swizzle enum).

A directly relevant 2026 datapoint: **Hexcute** formulates GPU layout synthesis as a constraint programming problem and solves it with a type-inference-based algorithm. ([2026.cgo.org](https://2026.cgo.org/details/cgo-2026-papers/12/Hexcute-A-Compiler-Framework-for-Automating-Layout-Synthesis-in-GPU-Programs))  

#### Why it supersedes \(\mathbb{F}_2\)
Instead of enumerating \(\mathbb{F}_2\) transformations and checking later, you solve *in the descriptor space*:
- the solver outputs either a **witness descriptor** or UNSAT.
- you can optimize a secondary objective: “minimize padding”, “maximize vectorization”, “minimize bank-conflict risk”.

#### Concrete optimization (S2)
- **TMA descriptor synthesis pass**:  
  input: `Layout` + desired tile shape + element type  
  output: either `tma_desc` + `cp.async.bulk.tensor` codegen, or fallback plan.

#### Hardware grounding
- Descriptor constraints and swizzle enum are explicit in the driver API. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- The actual issued instruction is `cp.async.bulk.tensor.*` which is descriptor+coords and can use cluster multicast. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

---

## Bottleneck 2 — **Spatial vs Temporal** (Async Copies, mbarriers, Warp Specialization)

### The cliff (hardware reality)
Hopper explicitly advertises TMA as an async engine enabling:
- avoiding registers for moves,
- avoiding SM instructions (single thread issues large moves),
- enabling warp specialization. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/hopper-tuning-guide/index.html))  

PTX makes the semantics unavoidable: `cp.async.bulk.tensor` is non-blocking, completes via mbarrier, and is treated as a **weak memory operation** with release semantics on completion at cluster scope. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

Seed A/B/C are fundamentally *spatial*; none of them is a **temporal type system** for “when data becomes visible.”

---

### **Theory A: Effect Systems + Linear/Session Types for Async Pipelines**

**Core idea:** represent async movement as an **effect** that produces a **capability/token** that must be consumed before use.

This is the “mathematical minimum” needed to make “layout selection is separable from schedule” false in the IR.

You already have an existence proof in infrastructure form: **MLIR’s `async` dialect** explicitly models async execution with `!async.token` and `async.await`. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/AsyncDialect/))  
And IREE’s **`stream` dialect** is explicitly about converting tensor programs into “explicitly scheduled asynchronous programs.” ([iree.dev](https://iree.dev/reference/mlir-dialects/Stream/))  

#### What you add (the theory-to-silicon bridge)
You *specialize* the token type to encode GPU protocol facts:

- `!nvgpu.mbarrier<scope=cluster>`  
- `!nvgpu.async_copy<bytes, scope, spaceSrc, spaceDst>`  

…and enforce typestate transitions:

- `arrive(mbarrier, bytes)` produces a “pending” state  
- `wait(mbarrier)` consumes “pending” and yields “available” for consumers

This is session-typing/typestate in spirit: the barrier is a communication channel with a protocol.

#### Concrete optimization (S2)
Once dependencies are explicit tokens, you can safely do:
- **software pipelining**: overlap `cp.async.bulk.tensor` with compute,
- **warp specialization scheduling**: segregate “copy warps” vs “compute warps” while preserving correctness,
- **wait minimization**: sink waits to last-use sites.

#### Why it’s hardware-correct
Because `cp.async.bulk.tensor` itself is weakly ordered and completion is defined via mbarrier semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

#### Math-to-hardware diagram
```text
Spatial mapping (layout L)  +  Temporal protocol (effects E)
                │                         │
                └──────────────┬──────────┘
                               v
                    Typed pipeline graph (DAG)
   node: t = tma_load(desc, coords) : token
   node: await(t) before smem use
   node: compute
```

---

### **Theory B: Axiomatic Memory Models + Proof/Translation Validation**

**Core idea:** use a formal memory model to *justify* reordering and fence/barrier insertion/elimination.

There is a strong starting point: NVIDIA Research published a **formal analysis of the PTX memory consistency model** (2019), including a mapping to scoped C++ primitives and mechanized checking. ([research.nvidia.com](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model))  
NVIDIA’s PTX interoperability guide explicitly references that mapping as proven correct. ([docs.nvidia.com](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/))  

#### Concrete optimization (S2)
- **Translation validation of async lowering**:  
  After lowering your typed async IR to PTX, validate that the required ordering edges are implemented via `mbarrier` operations and scopes correctly.
- **Fence minimization**:  
  If your effect system is conservative, the memory model can justify removing redundant ordering edges while preserving correctness.

This is how “math survives silicon”: you can be aggressive in scheduling without breaking weak ordering rules.

---

## Bottleneck 3 — **Power-of-2 Tyranny + Ragged / Prime Dimensions** (Tail Effect)

### The cliff (seed + workload collision)
Seed A explicitly notes the system’s *native comfort zone* is powers of two (warps, MMA tiles, Triton dims), and explicitly states a limitation: power-of-two shape restriction mitigated via padding + masking, with “affine layouts” as an extension target. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

But for ragged sequences (e.g., \([12, 1023, 7]\)), “lift to 1024 and mask” creates classic tail effect: wasted bandwidth + divergence + harder TMA tiling because descriptors want structured boxes. Descriptor constraints are explicit in the driver API. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

---

### **Theory A: Parametric Presburger / Polyhedral + Piecewise Affine Layouts (ISL-native)**

**Core idea:** represent the iteration domain and layout mapping as **(possibly piecewise) Presburger relations**, with symbolic parameters for dynamic sizes.

This is exactly what Seed B’s ISL relation perspective is setting you up for: it explicitly frames layouts as integer set relations and highlights quasi-affine extensions (needed for mod/XOR style constraints). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

#### Why it generalizes beyond \(\mathbb{F}_2\)
\(\mathbb{F}_2\) linear layouts are a *tiny decidable fragment* of Presburger arithmetic with congruences:
- Presburger handles arbitrary (non-\(2^n\)) sizes.
- Piecewise relations handle *raggedness* as unions of regions.

#### Concrete optimization (S2): **Core+Tail region compilation**
Instead of padding everything, you compute a **partition**:
- \(D_{\text{core}}\): the maximal rectangular region that satisfies TMA constraints (box sizes, alignment multiples).
- \(D_{\text{tail}}\): the remainder (small, predicated).

Then:
- compile \(D_{\text{core}}\) to **TMA** (`cp.async.bulk.tensor`) for structured tiles,
- compile \(D_{\text{tail}}\) to predicated scalar/vector loads.

This is a *semantic extension* of Seed A: from “single linear map” to “piecewise affine(+mod) map”.

#### Implementation hook (C1)
- MLIR: represent bounds in `affine`/`arith`/`scf` plus `shape` dialect for symbolic shapes. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/ShapeDialect/))  
- Use ISL only to *compute/verify the partition*, not to run an exponential search in the hot path.

---

### **Theory B: Abelian Group / Lattice Generalization (Mixed-Radix Layout Algebra)**

**Core idea:** Seed A uses the fact that coordinates are bit-vectors, i.e. the group \((\mathbb{Z}_2)^n\) (a vector space over \(\mathbb{F}_2\)). For general sizes, the “right” generalization is the finite abelian group
\[
G = \prod_{i=1}^d \mathbb{Z}_{N_i}
\]
and layouts as homomorphisms / affine maps on \(G\).

To reason about such maps canonically, integer matrix normal forms (e.g., **Smith normal form**) are the standard lattice/module tool. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Smith_normal_form))  

#### Why this matters (hardware-grounded)
- It lets you represent **prime or odd** sizes without forcing “next power-of-two” padding.
- It gives you a principled way to factor “bit-level swizzle on power-of-two factors” from “plain stride traversal on odd factors”.

#### Concrete optimization (S2): **Factor-aware lowering**
- If \(N = 2^k \cdot m\) with odd \(m\):  
  use \(\mathbb{F}_2\) swizzles only on the \(2^k\) substructure where it’s cheap (shifts/xors), and keep the odd component as affine stride traversal (no expensive modulo).

This is exactly the kind of “mixed-radix / mixed-algebra” lowering that avoids the seed’s padding cliff while staying hardware-friendly.

#### Where this lands in the IR
- CuTe/C++ templates: shape factorizations are natural.
- MLIR: encode factorization as attributes or as a `layout.factorized` type that lowers to arithmetic.

---

## Bottleneck 4 — **Bank Conflicts are Opcode-Indexed** (MI300 LDS lane phasing)

### The cliff (hardware reality)
AMD’s own materials emphasize:
- LDS is banked (32 banks, 4B),
- bank-conflict freedom depends on *which lanes are grouped* for a given instruction,
- `ds_write_b128` and `ds_read_b128` use different lane groupings,
- XOR-based transformations can eliminate conflicts. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  

This exactly matches your Stage‑1 statement: bank conflict is a property of \((\text{layout}, \text{instruction}, \text{lane-phase})\), not layout alone.

---

### **Theory A: Abstract Interpretation over Modular Domains (Instruction-Indexed Cost Semantics)**

**Core idea:** build an abstract domain that tracks addresses modulo the bank mapping function and checks conflicts per instruction-defined lane partitions.

At minimum:
- bank function: \(bank(a) = \left\lfloor a / 4 \right\rfloor \bmod 32\) (for 4B banks, 32 banks)
- instruction \(I\) defines an equivalence relation \(\sim_I\) that partitions lanes into groups
- conflict predicate:
\[
\exists \text{group } g, \exists \ell_1 \neq \ell_2 \in g : bank(addr(\ell_1)) = bank(addr(\ell_2))
\]

This is abstract interpretation with a **congruence domain** (values mod \(32\), mod \(128\), etc.), except the abstraction is parameterized by ISA semantics.

#### Concrete optimization (S2)
- Make “bank-conflict-free” a first-class cost/constraint during layout selection.
- For layouts expressed as \(\mathbb{F}_2\) maps, compute bank mapping by extracting low address bits and applying the linear map.

---

### **Theory B: Constraint-Based Swizzle Synthesis under ISA Lane Partition Constraints**

**Core idea:** directly synthesize an XOR (or small affine) swizzle that satisfies the opcode-specific lane grouping constraints.

AMD’s CK‑Tile materials give you the exact lane groups for `ds_read_b128` and `ds_write_b128` and describe XOR preshuffling as an alternative to padding. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  

So you can set up:
- unknown: small XOR transform \(T\) on indices (often fits Seed A’s “2 bits/column” swizzle envelope),
- constraints: conflict-free for each instruction’s lane groups,
- objective: minimize extra LDS footprint (avoid padding) and preserve vectorization.

#### Concrete optimization (S2)
- Replace “bank conflicts minimized” with “bank conflicts eliminated for the exact LDS opcodes emitted”.
- Emit different swizzles depending on whether the next stage uses MFMA reads or not (instruction-aware scheduling).

This is a clean compiler/synthesis bridge: *the ISA lane partition is part of the specification*.

---

## Bottleneck 5 — **Compile-Time Cliff** (ISL/rewrites/search blowups)

### The cliff
Seed B explicitly positions itself as foundational formalism for reasoning, not a performance-oriented optimizer. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
But in a JIT or aggressive autotuner, worst-case expensive relation operations can become a runtime cliff.

---

### **Theory A: Equality Saturation (E-graphs) + Costed Extraction**
**Core idea:** represent a space of equivalent layouts/transformations simultaneously and extract the best one under a cost model that includes “TMA-admissible” and “bank-conflict-free”.

This is exactly what `egg` enables: fast, extensible equality saturation with domain-specific analyses. ([mwillsey.com](https://www.mwillsey.com/papers/egg))  
And Tensat applies equality saturation to tensor graph superoptimization, showing practical improvements over sequential rewrite search. ([mwillsey.com](https://www.mwillsey.com/papers/tensat))  

#### Concrete optimization (S2)
- Use e-graphs for **layout normalization** and **legalization**:
  - rewrite \(\mathbb{F}_2\) layouts into a normal form that exposes affine strides + admissible swizzle,
  - run an e-class analysis that tags nodes with “TMA-admissible?” and “bank-conflict risk?”.

This is how you avoid exponential “try everything” while still exploring a rich equivalence space.

---

### **Theory B: Staged Compilation = Generate → Verify (Translation Validation as a Budgeting Tool)**
**Core idea:** use the heavy formalism (ISL/SMT) as a **checker** rather than a generator.

Workflow:
1. Generate candidate layouts by cheap heuristics (Seed A style).
2. Verify legality/equivalence via ISL relations (Seed B) or SMT in a restricted template.
3. Only if verification fails, expand search.

#### Concrete optimization (S2)
- Guarantees correctness while bounding compile time.
- Enables “fast path” compilation for common cases, with “slow path” only when needed.

This matches the practical stance implied by Seed B’s “foundation for future optimization strategies” framing. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

---

# 2) Literature Scan (2019–2026): 5 recent breakthroughs that plug directly into the arsenal

You asked for 3–5; here are **5** that are maximally load-bearing for *your exact bottlenecks*:

1. **Triton (MAPL/PLDI 2019): tile-centric GPU kernel language/compiler**  
   Why it matters: provides the operator-level substrate where layout IR and lowering decisions actually live. ([research.ibm.com](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations))  

2. **MLIR (2020): extensible multi-level IR infrastructure**  
   Why it matters: the “typed admissibility + effect system” plan is implementable specifically because MLIR is designed for staged lowering across dialects/abstractions. ([arxiv.org](https://arxiv.org/abs/2002.11054))  

3. **`egg` (POPL 2021): fast, extensible equality saturation**  
   Why it matters: gives you the machinery to normalize/repair layouts (e.g., rewrite toward TMA-admissible normal forms) with costed extraction. ([mwillsey.com](https://www.mwillsey.com/papers/egg))  

4. **Tensat (MLSys 2021): equality saturation for tensor graph superoptimization**  
   Why it matters: shows equality saturation can win in practice vs sequential rewrite search, which is directly analogous to “layout legalization/synthesis search.” ([mwillsey.com](https://www.mwillsey.com/papers/tensat))  

5. **Hexcute (CGO 2026): automated layout synthesis via constraint programming/type inference**  
   Why it matters: the closest match to your “instead of writing layouts, solve for them” vector—especially for descriptor admissibility and hardware intrinsic constraints. ([2026.cgo.org](https://2026.cgo.org/details/cgo-2026-papers/12/Hexcute-A-Compiler-Framework-for-Automating-Layout-Synthesis-in-GPU-Programs))  

---

## Optional “glue” readings (not part of the 3–5 list, but strategically important)

These are not “breakthrough list items” per your constraint, but they are the most directly actionable supporting foundations:

- **MLIR `async` dialect** for tokenized asynchrony modeling. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/AsyncDialect/))  
- **IREE `stream` dialect** for explicitly scheduled async programs (good mental model for host/device overlap and partitioning). ([iree.dev](https://iree.dev/reference/mlir-dialects/Stream/))  
- **Formal PTX memory model analysis (2019)** for making async lowering *provably* correct under weak ordering. ([research.nvidia.com](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model))  
- **MLIR `shape` dialect + shape inference docs** for symbolic shapes (ragged/dynamic). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/ShapeDialect/))  
- **Relax (TVM, 2023)** for cross-level dynamic shape abstractions (graph↔tensor program bridging). ([arxiv.org](https://arxiv.org/abs/2311.02103))  

---

## If you want the next step (Stage 2 framing)
I can now consolidate these into a single Stage‑2 “Theoretical Extension” plan:

- **Algebraic extension:** \(\mathbb{F}_2\) layouts → mixed-radix / piecewise Presburger layouts (core+tail).  
- **Temporal extension:** Layout IR → **effectful layout IR** with tokens + mbarrier protocol types.  
- **Hardware extension:** Layout IR → descriptor-refined subtypes + constraint-solver synthesis (TMA/WGMMA).  
- **Search extension:** e-graphs + solver + staged verification to bound compile time.

If you paste the exact Stage‑1 bottleneck list you want to prioritize (e.g., “TMA multicast” vs “WGMMA fragment constraints” vs “ragged KV-cache paging”), I’ll tailor the toolbox to those *and* propose the minimal MLIR dialect surface needed for an implementable prototype (C1), along with the evaluation metrics you required (C2).

---
Learn more:
1. [https://arxiv.org/html/2505.23819v3](https://arxiv.org/html/2505.23819v3)
2. [https://arxiv.org/html/2511.10374v1](https://arxiv.org/html/2511.10374v1)
3. [https://arxiv.org/pdf/2601.05972v1](https://arxiv.org/pdf/2601.05972v1)
4. [https://docs.nvidia.com/cuda/archive/12.6.0/hopper-tuning-guide/index.html](https://docs.nvidia.com/cuda/archive/12.6.0/hopper-tuning-guide/index.html)
5. [https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html)
6. [https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group\_\_CUDA\_\_TENSOR\_\_MEMORY.html](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
7. [https://2026.cgo.org/details/cgo-2026-papers/12/Hexcute-A-Compiler-Framework-for-Automating-Layout-Synthesis-in-GPU-Programs](https://2026.cgo.org/details/cgo-2026-papers/12/Hexcute-A-Compiler-Framework-for-Automating-Layout-Synthesis-in-GPU-Programs)
8. [https://mlir.llvm.org/docs/Dialects/AsyncDialect/](https://mlir.llvm.org/docs/Dialects/AsyncDialect/)
9. [https://iree.dev/reference/mlir-dialects/Stream/](https://iree.dev/reference/mlir-dialects/Stream/)
10. [https://research.nvidia.com/publication/2019-04\_formal-analysis-nvidia-ptx-memory-consistency-model](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model)
11. [https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/)
12. [https://mlir.llvm.org/docs/Dialects/ShapeDialect/](https://mlir.llvm.org/docs/Dialects/ShapeDialect/)
13. [https://en.wikipedia.org/wiki/Smith\_normal\_form](https://en.wikipedia.org/wiki/Smith_normal_form)
14. [https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
15. [https://www.mwillsey.com/papers/egg](https://www.mwillsey.com/papers/egg)
16. [https://www.mwillsey.com/papers/tensat](https://www.mwillsey.com/papers/tensat)
17. [https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations)
18. [https://arxiv.org/abs/2002.11054](https://arxiv.org/abs/2002.11054)
19. [https://arxiv.org/abs/2311.02103](https://arxiv.org/abs/2311.02103)
