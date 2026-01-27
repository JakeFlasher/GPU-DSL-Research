## Stage 1.5 — Theoretical Arsenal (Layouts × Asynchrony × Dynamism)

The two seed systems give you a **layout algebra**, but the modern “AI factory kernel” problem is:

\[
\textbf{Performance} \approx f(\text{Layout}) \times f(\text{Transport Schedule}) \times f(\text{Runtime Shape/Indirection})
\]

Linear Layouts explicitly admits its core expressiveness boundary: **power-of-two shapes** (mitigated via padding+masking), and that **flips/slices** need an **affine** extension. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
LEGO’s evaluation similarly selects **power-of-two square matrices** to avoid partial tiling/masking in Triton kernels. ([arxiv.org](https://arxiv.org/html/2505.08091))  
So your Stage-1 cliffs are *not* surprising—they’re the natural edges of “bit-linear layout algebra”.

Below is a *theoretical toolbox* that generalizes/supersedes the seed abstractions *without adding hardware*, and that is implementable in Triton/MLIR/PyTorch/CUTLASS.

---

### Software pipeline sketch (where the math should “land”)

```
PyTorch / TorchInductor / Triton Python
        |
        v
   TT / Linalg-on-Tensor IR
        |
        |   (A) Layout calculus  : affine + modular + piecewise
        |   (B) Transport calculus: async tasks + barriers + proxy fences
        |   (C) Runtime calculus  : inspector-executor + specialization cache
        v
 TTG / GPU IR  --->  PTX (cp.async.bulk.*, mbarrier, wgmma)  --->  SASS
```

The seed papers are strong in (A) *only*—and (A) is currently \(\mathbb{F}_2\)-centric. The cliffs happen when (B) and (C) dominate.

---

## 1) The Theoretical Toolbox

I’m going to align this directly with the Stage‑1 bottlenecks you listed (padding/masking explosion, TMA scheduling mismatch, gather/scatter indirection, cluster locality, target-specific bank behavior).

---

### Bottleneck: **Dynamic / ragged shapes + non‑power‑of‑two extents**  
*(“power-of-two restriction → padding/masking explosion”, decoding + MoE underfill)*

#### Theory A: **Parametric Polyhedral / Presburger + Piecewise‑Affine Layouts**
**Why it works**

- The right generalization of “bit-matrix layout” for arbitrary strides/extents is an **integer affine map** plus **guards**:
  - iteration domains as Presburger sets (linear inequalities with symbolic parameters),
  - address/layout maps as affine functions \(x \mapsto Ax + b\),
  - ragged edges as **piecewise** regions (different affine maps / different masking).
- This directly attacks the seed’s own stated limitation: flips/slices and non-power-of-two behavior naturally become affine/piecewise constructs rather than “pad to power-of-two and pray.” ([arxiv.org](https://arxiv.org/html/2505.23819v3))

**How it subsumes the seed**
- \(\mathbb{F}_2\) bit-linear layouts are the special case where:
  - all extents are powers of two,
  - arithmetic is effectively mod 2 on selected bits,
  - maps are linear and total.
- Polyhedral/PWB (piecewise-bounded) representations allow:
  - prime-sized dimensions,
  - ragged batches,
  - partial tiles without global padding.

**Implementation landing zone (real-metal feasible)**
- MLIR already has the conceptual plumbing for *affine maps*, and the Transform dialect has machinery for splitting/padding/tile packing workflows.
- Concrete precedent: SConvTransform explicitly handles “edge cases” by **splitting irregular regions and adjusting affine maps** in an MLIR Transform-based pipeline. ([arxiv.org](https://arxiv.org/abs/2511.18222))  
  (This is CPU-focused in that paper, but the *math+IR strategy* is the point.)

#### Theory B: **Staged Compilation (Partial Evaluation) + Shape‑Specialized Multi‑Versioning**
**Why it works**

- For LLM serving, “shape” is not a compile-time constant; it is a **runtime value** with a heavy-tailed distribution (batch, head_dim, page_size, tokens-per-expert).
- The clean theoretical framing is **partial evaluation / multi-stage programming**:
  - stage 1 (ahead-of-time): produce a kernel *generator* parameterized by shape symbols,
  - stage 2 (runtime): specialize for concrete shapes; cache code; pick tile sizes/layouts with minimal masking.

**Why this is not just “autotune”**
- Autotuning searches parameters; partial evaluation changes the *program structure* (e.g., removes masking, removes branches, constant-folds divisions/mods).
- It directly mitigates LEGO’s “we avoided masking to keep comparisons fair” evaluation gap: you don’t avoid masking by benchmark selection—you avoid it by specialization. ([arxiv.org](https://arxiv.org/html/2505.08091))

**Implementation landing zone**
- Triton already JITs; the missing piece is a principled **specialization key** (shape bucket + alignment + layout constraints) and an IR that can represent *guarded specializations*.

---

### Bottleneck: **Asynchronous transport mismatch (Hopper/Blackwell)**  
*(“layout engine reasons about conversions; Hopper wants TMA pipelines + async proxy ordering”)*
  
Hopper’s bulk async copies are not “just faster loads”—they have a distinct programming/semantic surface:
- `cp.async.bulk.*` has strict **16B alignment** and size constraints, explicit completion mechanisms, cluster multicast variants, etc. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.2/parallel-thread-execution/?utm_source=openai))  
- Ordering/visibility is **not free**; even `cp.async` groups require explicit waits for visibility guarantees. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.1.0/parallel-thread-execution/index.html?utm_source=openai))  
- Practitioners actively struggle with the async proxy visibility model (see the 2026 thread questioning missing explicit proxy fences). ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/why-arent-there-explicit-async-proxy-generic-proxy-fences-in-the-cuda-guide-tma-prefetching-example/357574?utm_source=openai))  

#### Theory A: **Task Graph Semantics / Event Structures (Dataflow for TMA + Tensor Cores)**
**Why it works**

- The math of async pipelines is a **partial order** of events, not a property of index maps.
- Model kernel as a **DAG of tasks** with explicit dependencies:
  - TMA copy tasks (issue → barrier completion),
  - compute tasks (wgmma/mma async),
  - epilogue tasks (stores, conversions),
  - synchronization edges (mbarriers, wait_group, proxy fences).
- Scheduling is then: topological order + resource constraints (warp specialization, barrier slots, register pressure).

**Recent “this is the right abstraction” signal**
- Cypress (PLDI 2025) is explicitly motivated by Hopper’s **asynchronous data movement unit (TMA)** + **asynchronous Tensor Core unit**, arguing you need warp-specialized producer/consumer pipelines and providing a task-based model with mapping specs. ([research.nvidia.com](https://research.nvidia.com/publication/2025-06_task-based-tensor-computations-modern-gpus?utm_source=openai))  

**Implementation landing zone**
- In MLIR/Triton terms: introduce a “transport dialect” (or a constrained subset) where:
  - `tma.copy(tile, tensormap, stage)` yields a **token**,
  - compute consumes tokens,
  - lowering emits `cp.async.bulk.tensor` + `mbarrier_expect_tx`/wait + correct group/wait semantics per CUDA/PTX. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

This is exactly the missing “(B) Transport calculus” in the seed pipeline.

#### Theory B: **Type-and-Effect / Session Types + Separation Logic for Pipeline Correctness**
**Why it works**

- Async pipelines have two failure classes:
  1) **semantic bugs** (use-before-ready, overwrite-before-consumed, missing fences),
  2) **performance bugs** (serialization, elected-thread mistakes, barrier misuse).
- A *type/effect* formulation can make these protocol constraints statically checkable:
  - effects encode “this stage buffer is produced/consumed,”
  - linear capabilities encode ownership of a shared-memory stage,
  - session types encode the protocol: `expect_tx → issue_copy → wait_ready → compute → release → reuse`.

**Why this matters on Hopper specifically**
- CUDA explicitly recommends using an *elected thread* (e.g., `elect_sync`) rather than a naive `if (threadIdx.x==0)` to avoid compiler-induced warp serialization in TMA initiation. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
  That’s exactly the sort of thing you want a compiler/type discipline to enforce rather than rely on folklore.

**Implementation landing zone**
- You don’t need a full proof assistant:
  - encode effects as SSA tokens + verifier passes (MLIR-style),
  - use lightweight separation-logic-inspired ownership checks for shared/cluster buffers,
  - then profile with Nsight Compute (real metal).

---

### Bottleneck: **Indirection-heavy gather/scatter (KV paging, MoE routing)**  
*(“layout algebra can’t linearize pointer chasing; shuffle rounds explode; coalescing collapses”)*

#### Theory A: **Inspector–Executor as a First-Class Semantics (Permutation/Partition Algebra)**
**Why it works**

- When addresses are defined by indirection tables, the core optimization is *not* a clever swizzle—it’s **reordering the work** to re-densify memory access.
- The theoretical framing:
  - represent routing/paging as a permutation \(p\) or a partition of tokens into bins (experts/pages),
  - apply \(p\) to pack sparse/irregular access into dense tiles,
  - run dense kernels with predictable layouts (where Linear Layouts-style machinery actually shines).

**Key point**
- This turns “gather/scatter” into:
  - a runtime **permutation synthesis** problem + a dense compute problem,
  - rather than a purely compile-time layout conversion problem.

**Implementation landing zone**
- PyTorch runtime (or a custom CUDA/Triton pre-pass) builds the permutation; Triton runs the dense expert/page kernels.
- Evaluatable on real GPUs via TritonBench MoE/attention kernels and end-to-end decode latency.

#### Theory B: **Equality Saturation over Data-Movement IR (Global Rewriting + Costed Extraction)**
**Why it works**

- Indirection pipelines have many semantics-preserving decompositions:
  - gather → shared staging → compute
  - sort+segment → block gather → compute
  - warp-contained shuffle gather vs shared vs global vector gather
- The right theory is not “pick one heuristic locally,” but **maintain equivalence classes** and choose globally profitable forms.

**Recent compiler momentum**
- `eqsat` (2025) proposes an MLIR-style IR/dialect for **equality saturation** integrated into the compilation flow, keeping e-graph state across transformations. ([ar5iv.org](https://ar5iv.org/html/2505.09363v2))  
- OOPSLA 2025 explicitly calls out the “abstraction gap” in bringing equality saturation to real-world ML compilers and SOTA hardware, i.e., exactly your setting. ([2025.splashcon.org](https://2025.splashcon.org/details/OOPSLA/42/Mind-the-Abstraction-Gap-Bringing-Equality-Saturation-to-Real-World-ML-Compilers?utm_source=openai))  

**Implementation landing zone**
- Represent layout conversions, gathers, and staging as rewriteable IR nodes.
- Use extraction cost terms that directly encode your Stage‑1 pain:
  - register pressure / live ranges,
  - number of shuffle rounds,
  - TMA eligibility (alignment, tensorMap constraints),
  - bank conflicts (target-specific).

---

### Bottleneck: **Cluster-level locality (Thread Block Clusters / DSMEM) and multi-level distribution**
*(“seed distributes within CTA/warp/thread; cluster adds a new tier”)*

#### Theory A: **Abstract Interpretation on a Sharding/Placement Lattice**
**Why it works**

- Cluster changes the question from “where is an element within a CTA” to “where is a tensor slice across CTAs in a cluster.”
- This is a classic dataflow problem:
  - define a lattice of placements \(\{\text{reg}, \text{shared::cta}, \text{shared::cluster}, \text{global}\}\),
  - propagate placement attributes through ops,
  - join/meet at merges, annotate conversions as needed.

**Recent signal (scale-out direction)**
- LLVM dev meeting 2024 highlights an MLIR “tensor propagation system” representing tensor shardings and propagation rules as MLIR attributes. ([llvm.org](https://www.llvm.org/devmtg/2024-10/?utm_source=openai))  
  That’s the same *lattice + propagation* idea, applied to sharding/placement.

**Implementation landing zone**
- Extend Triton’s layout engine with a second propagation: **placement/sharding propagation** over cluster-aware memory spaces, feeding into TMA multicast and DSMEM ops.

#### Theory B: **Wreath Products / Hierarchical Group Actions for Mapping**
**Why it works**

- LEGO is fundamentally hierarchical; GPUs are hierarchical; clusters add another rung.
- The clean group-theoretic model of “hierarchical permutation structure” is a **wreath product** (composition of symmetric actions across levels).
- Using wreath-product-like composition as the *semantic* model gives you:
  - a normal form for hierarchical mapping,
  - a way to represent “cluster × CTA × warp × thread × reg” mappings uniformly,
  - a path to solve for conversions as equations in a structured group.

**Recent pure-math tie-in (optional but relevant)**
- There is active work on solvability of equations in wreath products; e.g., Semigroup Forum (2025) surveys solvability of equations in wreath products and proves decidability results for certain diophantine problems in wreath products of abelian groups. ([link.springer.com](https://link.springer.com/article/10.1007/s00233-025-10511-8?utm_source=openai))  
  You don’t need these exact results, but it’s a strong hint that “solve conversion equations in a wreath-structured algebra” can be mathematically well-behaved.

---

### Bottleneck: **Target-specific bank conflicts (NVIDIA vs AMD wave64 phase rules)**
*(“swizzle optimality is bank-model dependent; wave64 introduces phased conflicts”)*
  
#### Theory A: **Modular Constraint Solving (SMT over Bitvectors + Integers)**
**Why it works**

- Bank mapping is a modular arithmetic function of addresses; many swizzles are XOR/affine transforms.
- For NVIDIA-style shared memory, many constraints are power-of-two friendly; for AMD wave64, conflict rules can depend on instruction width and lane grouping (phase rules).
- The correct abstraction is: solve for a transform \(T\) such that:
  - \(bank(addr(T(i)))\) spreads lanes uniformly under target’s bank function,
  - subject to vectorization/alignment constraints (e.g., TMA’s 16B alignment if you stage via TMA). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

**Implementation landing zone**
- Feed the solver with an ISA-parameterized bank model:
  - “bank function,”
  - “lane grouping,”
  - “preferred LDS/SMEM op widths.”
- Emit a swizzle as a compact affine/XOR transform (still fast in codegen).

#### Theory B: **Combinatorial Design View (Hash Families / Linear Codes for Lane→Bank Mapping)**
**Why it works**

- Swizzle design is essentially designing a low-cost hash from lane-id and coordinate bits to bank-id, minimizing collisions under the machine’s access pattern class.
- Linear XOR transforms are a family of universal hashes; “good swizzles” are those that behave like a good hash under the access distribution you care about.
- This gives you closed-form constructions (fast compile-time), and you can still validate/tune with real-metal microbenchmarks.

---

## 2) Literature Scan (2024–2026): recent work that maps onto these theories

Here are **5** recent items (mixing PL/compilers + one pure-math anchor) that are directly relevant to generalizing/superseding the seed papers.

1) **Cypress (PLDI 2025): Task-based tensor computations on modern GPUs**  
   - Motivated explicitly by Hopper having both **TMA (async data movement)** and **async Tensor Core** units; proposes a task-based model + mapping spec to generate warp-specialized pipelines. ([research.nvidia.com](https://research.nvidia.com/publication/2025-06_task-based-tensor-computations-modern-gpus?utm_source=openai))  
   - Relevance: strongest signal that the “right” abstraction is **task/event semantics**, not just layout algebra.

2) **MLIR Transform Dialect (2024–2025): controllable IR-based transformation system**  
   - The Transform dialect paper argues for making compiler transformations *programmable/composable* inside MLIR to regain control and integrate search methods. ([arxiv.org](https://arxiv.org/abs/2409.03864?utm_source=openai))  
   - Relevance: your toolbox (polyhedral splits, async scheduling, bank-model-specific rewrites) needs a **controllable transformation substrate**.

3) **`eqsat` (EGRAPHS 2025): Equality saturation as an MLIR-native dialect/IR**  
   - Integrates equality saturation concepts into an IR flow, aiming to keep e-graph state across compiler transforms. ([ar5iv.org](https://ar5iv.org/html/2505.09363v2))  
   - Relevance: replaces fragile local heuristics (layout conversion choices, gather lowering choices) with **global rewrite exploration + costed extraction**.

4) **HEC (USENIX ATC 2025): e-graph-based equivalence checking for compiler transformations**  
   - Uses e-graphs + MLIR frontend to verify transformations; reports finding real compiler bugs. ([arxiv.org](https://arxiv.org/abs/2506.02290))  
   - Relevance: if you start rewriting async pipelines + piecewise affine layouts aggressively, you’ll want a **scalable correctness backstop**.

5) **(Pure math anchor) Equations in wreath products (Semigroup Forum, 2025)**  
   - Surveys solvability of equations in wreath products and proves solvability for certain classes in wreath products of abelian groups. ([link.springer.com](https://link.springer.com/article/10.1007/s00233-025-10511-8?utm_source=openai))  
   - Relevance: supports using **wreath-product-style hierarchical mapping** (CTA/warp/thread/cluster) as a structured algebra where “solve for conversion” can be tractable, not an ad-hoc heuristic zoo.

---

### Minimal “required reading” specs (not breakthroughs, but they constrain everything)

If your proposal touches TMA, you cannot avoid these constraints:
- **CUDA Programming Guide (async copies / TMA)**: tensor maps, completion mechanisms, alignment rules, and the “single elected thread” initiation guidance. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
- **PTX ISA (`cp.async.bulk` + wait semantics)**: alignment/size constraints, multicast variants, and visibility rules. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.2/parallel-thread-execution/?utm_source=openai))  

These specs are the *ground truth* for any “math of asynchrony” you introduce.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping](https://arxiv.org/html/2505.08091)
3. [\[2511.18222\] Using MLIR Transform to Design Sliced Convolution Algorithm](https://arxiv.org/abs/2511.18222)
4. [PTX ISA 8.5](https://docs.nvidia.com/cuda/archive/12.6.2/parallel-thread-execution/?utm_source=openai)
5. [Parallel Thread Execution 8.1](https://docs.nvidia.com/cuda/archive/12.1.0/parallel-thread-execution/index.html?utm_source=openai)
6. [Why aren't there explicit async proxy<->generic proxy fences in the cuda guide TMA prefetching example? - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/why-arent-there-explicit-async-proxy-generic-proxy-fences-in-the-cuda-guide-tma-prefetching-example/357574?utm_source=openai)
7. [Task-Based Tensor Computations on Modern GPUs | Research](https://research.nvidia.com/publication/2025-06_task-based-tensor-computations-modern-gpus?utm_source=openai)
8. [4.11. Asynchronous Data Copies — CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai)
9. [\[2505.09363\] eqsat: An Equality Saturation Dialect for Non-destructive Rewriting](https://ar5iv.org/html/2505.09363v2)
10. [Mind the Abstraction Gap: Bringing Equality Saturation to Real-World ML Compilers (SPLASH 2025 - OOPSLA) - SPLASH 2025](https://2025.splashcon.org/details/OOPSLA/42/Mind-the-Abstraction-Gap-Bringing-Equality-Saturation-to-Real-World-ML-Compilers?utm_source=openai)
11. [The LLVM Compiler Infrastructure Project](https://www.llvm.org/devmtg/2024-10/?utm_source=openai)
12. [Equations in wreath products | Semigroup Forum | Springer Nature Link](https://link.springer.com/article/10.1007/s00233-025-10511-8?utm_source=openai)
13. [The MLIR Transform Dialect. Your compiler is more powerful than you think](https://arxiv.org/abs/2409.03864?utm_source=openai)
14. [\[2506.02290\] HEC: Equivalence Verification Checking for Code Transformation via Equality Saturation](https://arxiv.org/abs/2506.02290)