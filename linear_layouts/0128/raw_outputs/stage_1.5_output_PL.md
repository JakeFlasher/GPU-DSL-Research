## Stage 1.5 — Theoretical Arsenal to *supersede* \(\mathbb{F}_2\)-Layouts under (1) ragged shapes, (2) Hopper/Blackwell asynchrony, (3) irregular indirection

**One-line diagnosis:** Linear Layouts gives you a *spatial* algebra (bit-linear maps) that is extremely strong when (a) shapes are power-of-two and (b) “data movement” is mostly *layout conversion*. But Hopper-class performance is dominated by **spatio-temporal orchestration**: TMA pipelines, mbarriers, warp specialization, cluster DSMEM, and dynamic shape regimes. Linear Layouts itself calls out its core limitation: **power-of-two shapes**, with slicing/flipping not expressible without extending to affine layouts. ([arxiv.org](https://arxiv.org/html/2505.23819v3))

A useful mental model is:

```
Seed paper coverage (mostly):            Modern kernel reality (H100/B200/MI300):
  "where is element i?"                    "where is element i?"  +  "when/how does it move?"
      SPACE (layout)                           SPACE (layout)         TIME (async schedule)
  hw_idx  -> logical_idx                   hw_idx->logical_idx    tokens/barriers/pipeline stages
```

Below are **theoretical frameworks** you can actually compile into Triton/MLIR/PyTorch/CUTLASS without “new hardware” or simulators.

---

# 1) The Theoretical Toolbox

### Bottleneck: **Dynamic / ragged shapes → padding + masking explosion** (LLM decode, ragged attention, MoE underfilled tiles)

Linear Layouts is explicitly limited to power-of-two shapes, suggesting pad-to-power-of-two + mask; and it notes slicing/flipping needs affine layouts. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
LEGO, from the other direction, explicitly introduces mechanisms for **partial tiles** (ExpandBy) and acknowledges compile-time constraints in Triton (e.g., bounds must be known at compilation for certain constructs). ([arxiv.org](https://arxiv.org/html/2505.08091v2))

- **Theory A: Parametric Polyhedral Semantics (Presburger / Integer Sets) + Piecewise-Affine Layouts**
  - **Why it works:**  
    Raggedness is not “noise”; it’s *control-flow in the iteration space*. The polyhedral model naturally represents iteration domains as **parametric integer sets** with affine constraints (Presburger arithmetic). “Masking” becomes an explicit *guarded region* instead of an implicit afterthought.
  - **What you get vs. \(\mathbb{F}_2\):**
    - Layouts as \(x \mapsto A x + b\) over \(\mathbb{Z}\), plus **guards** (piecewise definitions).
    - Explicit modeling of **edge tiles** without padding to the next power of two.
    - A clean path to represent slicing/flipping directly (affine + piecewise), matching the extension Linear Layouts already points to. ([arxiv.org](https://arxiv.org/html/2505.23819v3))
  - **Compiler instantiation:**  
    In MLIR terms: “layout” becomes an `affine_map` + `IntegerSet` guards; lowering chooses between:
    - fast-path (full tile, no mask),
    - edge-path (masked) with explicit cost.
  - **Microarchitecture-aware payoff:**  
    You minimize wasted **HBM transactions** and reduce predicate-heavy instruction streams that trigger **SOL% cliffs** in small-batch decode.

- **Theory B: Staging / Partial Evaluation (Multi-Versioning JIT) as a *semantic* treatment of shape dynamism**
  - **Why it works:**  
    Many “dynamic” LLM-serving shapes are *not adversarially dynamic*—they cluster into buckets (batch sizes, head_dim, page_size, etc.). Treat shape parameters as **staging-time values**: generate a finite set of specialized kernels keyed on runtime shape buckets, with guards.
  - **Key detail:**  
    LEGO already does **range propagation** and uses solver-checked simplifications; it explicitly states Triton requires certain bounds known at compilation time, which is exactly what specialization satisfies. ([arxiv.org](https://arxiv.org/html/2505.08091v2))
  - **Compiler instantiation:**  
    - AOT generate a small portfolio of kernels; JIT fill gaps.
    - Use a **shape-cache** keyed by \((B, H, d, L_{kv}\ \text{bucket}, \text{page\_size})\).
  - **Microarchitecture-aware payoff:**  
    Fewer masked lanes → lower register pressure + better occupancy, which matters brutally in decode where the kernel grid is tiny.

---

### Bottleneck: **Non-power-of-two strides / prime dimensions → \(\mathbb{F}_2\) model mismatch**

\(\mathbb{F}_2\)-linear maps are naturally aligned with **bit structure** and power-of-two tiling; Linear Layouts explicitly relies on that alignment. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
LEGO, in contrast, leans into modulo/division algebra, and uses SMT (Z3) to prove side conditions for safe simplifications. ([arxiv.org](https://arxiv.org/html/2505.08091v2))

- **Theory A: Integer-Lattice Layout Algebra (Linear Maps over \(\mathbb{Z}\)) + Smith/Hermite Normal Forms**
  - **Why it works:**  
    General strides are integer-linear, not bit-linear. Model memory address as:
    \[
      \text{addr} = \text{base} + \langle s, i \rangle
    \]
    where \(s\) is a stride vector in bytes/elements, \(i\) is a multi-index. The right abstraction is a **\(\mathbb{Z}\)-module** / integer lattice, not a vector space over \(\mathbb{F}_2\).
  - **What you can do:**
    - Decide invertibility / existence of exact conversions using integer linear algebra (SNF/HNF).
    - Compute contiguity / vectorization legality via **gcd structure** of strides, not just “zero columns in a bit-matrix”.
  - **Compiler instantiation:**  
    - Add a “Z-affine layout” dialect next to Linear Layouts.
    - Use integer-normal-form transforms to synthesize conversion code or prove it must stage through shared memory.
  - **Microarchitecture-aware payoff:**  
    You stop forcing “pad to power-of-two” when the real issue is stride arithmetic; you also unlock vectorization opportunities when the last-dim stride supports it even for prime sizes.

- **Theory B: Mixed-Radix Index Algebra (Direct Products of Cyclic Groups) + Verified Strength Reduction**
  - **Why it works:**  
    Real shapes are often composite but not powers of two. Represent indices in a **mixed-radix** system (e.g., base \(b_0, b_1, \dots\)) rather than bits-only. Layout transforms become permutations/affine transforms over \(\mathbb{Z}_{b_k}\) factors *when carry is controlled*.
  - **The practical blocker:** div/mod are expensive.
  - **The PLT/solver answer:**  
    Use **range-aware rewrite systems** with solver-checked side conditions—exactly the pattern LEGO uses (custom simplifications + Z3 proofs from derived index ranges). ([arxiv.org](https://arxiv.org/html/2505.08091v2))
  - **Compiler instantiation:**  
    - Generate mixed-radix formulas, then aggressively rewrite to multiplies/shifts via proven bounds.
    - Emit “fast div” sequences (magic numbers) where legal.
  - **Microarchitecture-aware payoff:**  
    Keeps index math from becoming instruction-bound (a common decode cliff) while expanding the expressible layout space beyond \(\mathbb{F}_2\).

---

### Bottleneck: **Hopper/Blackwell asynchrony (TMA + mbarrier + warp specialization) is not first-class in a layout-only engine**

Linear Layouts’ layout engine is excellent at propagating layouts and inserting conversions, and it even optimizes `tl.gather` when the axis lives within one warp. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
But Hopper performance hinges on treating transport/pipelines as first-class. MLIR already has explicit async/token machinery and NVIDIA-specific ops (TMA, mbarrier, warpgroup mma) in `nvgpu`. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/AsyncDialect/?utm_source=openai))  
Also, **Tawa** explicitly targets this gap: it introduces “asynchronous references” (aref) and automatically partitions kernels into producer/consumer warp roles for warp specialization. ([arxiv.org](https://arxiv.org/abs/2510.14719))

- **Theory A: Tokenized Dataflow Semantics (Async/Futures) + Resource-Constrained Modulo Scheduling**
  - **Why it works:**  
    Asynchrony is naturally a **partial order**. Model each TMA copy, barrier arrive, and wgmma as an event producing/consuming tokens (futures). Then scheduling is a constrained optimization problem:
    - minimize stalls,
    - respect barrier protocols,
    - keep register pressure under occupancy thresholds.
  - **MLIR grounding (already exists):**
    - `!async.token` / async dialect defines token semantics. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/AsyncDialect/?utm_source=openai))
    - `gpu` dialect supports async dependencies/tokens. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/GPU/?utm_source=openai))
    - `nvgpu.tma.async.load/store` and `nvgpu.mbarrier.*` are explicit. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))
  - **Compiler instantiation:**  
    Build a pass that lowers from “layout + compute” to an explicit **pipeline IR**:
    ```
    (tile loop)
      issue TMA loads (producer warp)
      mbarrier expect_tx/arrive
      consumer warp: wgmma on stage k
      wait group / fence
    ```
  - **Microarchitecture-aware payoff:**  
    You stop paying register + instruction-issue bandwidth for copies, and you keep tensor pipes fed.

- **Theory B: Protocol Types / Typestate for Barriers (Session Types in SIMT clothing) + SMT-backed verification**
  - **Why it works:**  
    Most real Hopper bugs/perf cliffs are “protocol bugs”: wrong barrier phase, missing waits, wrong arrival counts, invalid overlap. Treat each barrier/TMA stage as a **typestate machine**:
    ```
    Barrier<State=Init> -> ExpectTx(nbytes) -> Arrive -> Wait -> (toggle parity) -> ...
    ```
    Session/typestate systems ensure the program follows a legal protocol.
  - **How it becomes implementable (2025 era):**  
    MLIR is increasingly moving toward explicit semantics + SMT tooling. “First-Class Verification Dialects for MLIR” makes semantics a first-class citizen to build dialect-agnostic SMT-backed tooling. ([pldi25.sigplan.org](https://pldi25.sigplan.org/details/pldi-2025-papers/60/First-Class-Verification-Dialects-for-MLIR?utm_source=openai))
  - **Compiler instantiation:**  
    - Encode TMA/mbarrier protocols in a verifier dialect or as attributes with a checker.
    - Prove “no deadlock / no use-before-ready / correct fence placement” at compile time (or at least catch violations).
  - **Microarchitecture-aware payoff:**  
    Enables aggressive reordering/pipelining because correctness is mechanically checked rather than folklore.

---

### Bottleneck: **Cluster-level locality (Thread Block Clusters / DSMEM) is outside CTA-only distributed layout models**

Linear Layouts’ distributed-layout family (in Triton’s model) is structurally constrained: columns are distinct powers of two or zero, i.e., a “permutation + optional zeros” style encoding. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
That’s perfect for warp/thread/register mapping, but it doesn’t naturally express **cluster rank** as a first-class locality tier.

- **Theory A: Region / Ownership Types for GPU Memory Spaces (Global / CTA shared / Cluster shared)**
  - **Why it works:**  
    Cluster DSMEM introduces new *lifetime and aliasing rules*. Region/ownership types (and separation-logic-style invariants) are the right math to express:
    - which warp/CTA owns which shared-memory region,
    - when it becomes visible to other CTAs,
    - when it is safe to reuse buffers (double/triple buffering across CTAs).
  - **Compiler instantiation:**  
    - Extend IR types: `memref<..., #gpu.address_space<workgroup|cluster|global>>`.
    - Add an ownership discipline so transformations (fusion, pipelining) don’t create DSMEM races.
  - **Microarchitecture-aware payoff:**  
    Correctly enables cluster-scope buffering / multicast patterns *without* over-synchronization.

- **Theory B: Group-Action / Hierarchical Sharding Algebra (Cluster \(\times\) CTA \(\times\) Warp \(\times\) Thread)**
  - **Why it works:**  
    Think of the execution hierarchy as a product of groups; a layout is a homomorphism from a “hardware index group” into an index space. \(\mathbb{F}_2\) is one instance (bit-vectors). Generalizing to include **cluster ID** as another factor gives you principled composition:
    \[
      G_{\text{cluster}} \times G_{\text{cta}} \times G_{\text{warp}} \times G_{\text{thread}} \to \text{TensorCoords}
    \]
  - **Compiler instantiation:**  
    - Add a “cluster dimension label” (exactly the kind of label Linear Layouts uses for bits, just extended to another locality tier).
    - Extend conversion synthesis to allow cross-CTA movement to be planned (DSM copies vs recompute vs refetch).
  - **Microarchitecture-aware payoff:**  
    You can *reason* about cluster-scope reuse and decide when it beats extra HBM traffic.

---

### Bottleneck: **Indirection-heavy gather/scatter (KV paging, MoE routing) breaks “regular tensor” layout reasoning**

Linear Layouts can optimize `tl.gather` with warp shuffles **only when the gather axis resides within one warp**, otherwise you fall off the fast path. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
This is exactly what KV paging and MoE routing violate: indices cross warps/CTAs and are pointer-chasing.

- **Theory A: Inspector–Executor as a First-Class Compilation Strategy (Sparse Polyhedral / Semilinear Sets + SMT)**
  - **Why it works:**  
    Indirection means the *iteration space depends on data*. The classic answer (sparse linear algebra, graph analytics) is inspector–executor:
    1. **Inspector:** analyze indices at runtime, pack/sort/group into contiguous blocks.
    2. **Executor:** run dense kernels on the packed representation (where Linear Layouts-style swizzles/TMA pipelines actually apply).
  - **Evidence that the theory scales:**  
    Sparse polyhedral frameworks already combine polyhedral scanning with SMT-based synthesis to match corresponding elements. ([arxiv.org](https://arxiv.org/abs/2208.11858))
  - **Compiler instantiation:**  
    - Generate a packing kernel in Triton (or CUDA) that transforms \((\text{indices}, \text{values})\) into “tiles per expert/page”.
    - Then dispatch a dense GEMM/FA kernel with aggressive layout+pipeline optimization.
  - **Microarchitecture-aware payoff:**  
    Converts “random accesses + long scoreboard stalls” into streaming TMA-friendly accesses, improving both DRAM efficiency and tensor-pipe utilization.

- **Theory B: Equality Saturation over *Data-Movement Plans* (E-graphs) + Cost Models that include register pressure**
  - **Why it works:**  
    For gather/scatter, there is no single best lowering: warp shuffles, shared staging, two-phase pack+compute, etc. Equality saturation lets you represent **all legal rewrites** and choose the cheapest under a realistic cost model.
  - **Recent MLIR enabler:**  
    DialEgg integrates Egglog-based equality saturation with MLIR in a dialect-agnostic way, making it plausible to apply e-graphs to layout/schedule rewriting inside MLIR pipelines. ([2025.cgo.org](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog?utm_source=openai))
  - **Compiler instantiation:**  
    - Encode rewrite rules: “shuffle-gather ⇄ shared-gather ⇄ pack+gather”.
    - Cost model uses: instruction count + predicted register pressure + expected memory coalescing.
  - **Microarchitecture-aware payoff:**  
    Avoids the “shuffle rounds explode → instruction-bound” cliff by selecting alternative plans when the warp-contained condition doesn’t hold.

---

### Bottleneck: **Register pressure / occupancy cliffs from layout conversions + wgmma fragments + epilogue fusion**

Linear Layouts already generates warp-shuffle conversions and sophisticated swizzles; that can trade shared-memory traffic for **more live registers**. Register pressure is the silent killer on H100/B200 when you combine wgmma fragments + pipelining.

- **Theory A: Resource-Aware Scheduling as Constrained Optimization (ILP / modulo scheduling with register-capacity constraints)**
  - **Why it works:**  
    Treat “register file usage per thread” as a hard capacity constraint and schedule conversions/prefetch/compute to minimize peak live ranges. This is classic compiler theory, but the novelty is making it **first-class for GPU kernel IR** with explicit async ops.
  - **Compiler instantiation:**  
    - Compute live-range peaks (or a proxy) at IR level.
    - Add constraints: “keep registers < R_threshold” to preserve occupancy.
    - Choose pipeline depth and conversion placement accordingly.

- **Theory B: Linear/Uniqueness Types for Temporaries (Liveness-by-construction)**
  - **Why it works:**  
    Linear types enforce single-use; uniqueness types enforce non-aliasing. Interpreted at the IR level, they let you:
    - prevent accidental duplication of large fragments,
    - enforce “consume then release” disciplines for pipeline stages,
    - structurally limit live ranges.
  - **Compiler instantiation:**  
    Use a type/effect annotation system on intermediate tensors (especially fragments/buffers) so transformations (fusion, reordering) can’t silently inflate liveness.

---

### Bottleneck: **Target-specific shared-memory banking (MI300 wave64 vs NVIDIA) requires a parametric bank model**

Linear Layouts gives a principled bank-conflict minimization algorithm expressed in linear algebra, explicitly modeling bank conflicts and aiming for maximal vectorization. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
But Stage 1 correctly flags the real issue: the bank model is **ISA-dependent** (lane grouping, instruction width, phased rules).

- **Theory A: Parametric Abstract Interpretation of Bank Conflicts**
  - **Why it works:**  
    Define an abstract domain parameterized by:
    - wave size,
    - bank mapping function,
    - transaction width (e.g., b128),
    - lane grouping phases.  
    Then “conflict freedom” is a property you can compute/approximate compositionally on layouts.
  - **Compiler instantiation:**  
    - Plug target parameters into the analyzer.
    - Feed back into swizzle synthesis and vectorization legality decisions.

- **Theory B: Constraint Solving / Combinatorial Design for Swizzles (Graph coloring view)**
  - **Why it works:**  
    Bank conflicts can be formulated as collisions in a bipartite/hypergraph: (lane, element) → bank. Swizzle is a permutation/XOR-like transform you choose to minimize collisions subject to constraints (vectorization, alignment).
  - **Compiler instantiation:**  
    - Encode as ILP/SMT (small sizes; tractable).
    - Or use search guided by measured feedback, but with solver guarantees for legality.

---

# 2) Literature Scan (2022–2026): 5 “breakthrough” directions that directly support the arsenal

Below are not just “papers to cite”—they map cleanly onto an implementable Triton/MLIR toolchain.

1. **Linear Layouts (ASPLOS ’26): \(\mathbb{F}_2\) matrix layouts + layout propagation + generic conversions**  
   Provides the seed abstraction, but also clearly identifies the key limitation (power-of-two; affine layouts needed for slicing/flipping). ([arxiv.org](https://arxiv.org/html/2505.23819v3))

2. **LEGO (CGO ’26 track / 2025 arXiv): layout expression language + partial tiles + SMT-checked simplification**  
   Valuable because it demonstrates: (a) *beyond bijections* via partial tiles/injective mappings, and (b) solver-checked rewrite rules using Z3 with propagated range constraints—exactly the “verified strength reduction” machinery you need for non-power-of-two indexing. ([arxiv.org](https://arxiv.org/html/2505.08091v2))

3. **Tawa (CGO ’26, arXiv 2025): automatic warp specialization via “asynchronous references (aref)”**  
   Directly addresses the “SIMT vs async hardware” mismatch by elevating warp-role communication/pipelining into an IR abstraction and compiling it automatically for H100-class hardware. ([arxiv.org](https://arxiv.org/abs/2510.14719))

4. **MLIR formalization wave: “First-Class Verification Dialects for MLIR” (PLDI 2025)**  
   Makes semantics and SMT-backed tooling first-class in MLIR—critical if we want to aggressively transform async pipelines (TMA/mbarrier/wgmma) without creating subtle ordering bugs. ([pldi25.sigplan.org](https://pldi25.sigplan.org/details/pldi-2025-papers/60/First-Class-Verification-Dialects-for-MLIR?utm_source=openai))

5. **MLIR optimization control & search: Transform Dialect (CGO 2025) + Equality Saturation integration (DialEgg, CGO 2025)**  
   - Transform dialect: makes compiler transformation *programmable* and composable, enabling systematic search over schedules/layouts rather than hard-coded passes. ([arxiv.org](https://arxiv.org/abs/2409.03864))  
   - DialEgg: brings equality saturation (Egglog) to MLIR in a dialect-agnostic way—key infrastructure for “solve, don’t write” layout/schedule planning. ([2025.cgo.org](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog?utm_source=openai))

---

## A “unified” pipeline sketch (how these theories compose into an implementable compiler plan)

```
          (Shape + Layout algebra)                  (Async schedule algebra)                 (Synthesis/verification)
PyTorch →  Triton/MLIR front-end  →  Layout IR  →    Pipeline IR (tokens)    →  lowering  →  PTX/SASS
            - piecewise affine         - async tokens (!async.token)           - nvgpu.tma + mbarrier ops
            - Z-affine lattices        - producer/consumer warp roles          - wgmma / warpgroup.mma
            - mixed-radix layouts      - modulo scheduling / buffering         - verified rewrites (SMT)
            - explicit masks/guards                                         - e-graph exploration (egglog)
```

The key “arsenal move” is to **stop treating layout as a static attribute** and instead treat **layout + transport schedule** as a jointly optimized *program*—rewritten, verified, and specialized under real hardware constraints.

If you want, in Stage 2 I can turn this into a concrete “replace the seed” plan: a **Spatio-Temporal Layout & Transport IR** inside Triton (tt/ttg) that (1) generalizes beyond \(\mathbb{F}_2\) to \(\mathbb{Z}\)/piecewise affine and (2) makes TMA/mbarrier pipelines first-class and synthesizable via e-graphs + SMT.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping](https://arxiv.org/html/2505.08091v2)
3. ['async' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/AsyncDialect/?utm_source=openai)
4. [\[2510.14719\] Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References](https://arxiv.org/abs/2510.14719)
5. ['gpu' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/GPU/?utm_source=openai)
6. ['nvgpu' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)
7. [First-Class Verification Dialects for MLIR (PLDI 2025 - PLDI Research Papers) - PLDI 2025](https://pldi25.sigplan.org/details/pldi-2025-papers/60/First-Class-Verification-Dialects-for-MLIR?utm_source=openai)
8. [\[2208.11858\] Polyhedral Specification and Code Generation of Sparse Tensor Contraction with Co-Iteration](https://arxiv.org/abs/2208.11858)
9. [DialEgg: Dialect-Agnostic MLIR Optimizer using Equality Saturation with Egglog (CGO 2025 - Main Conference) - CGO 2025](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog?utm_source=openai)
10. [\[2409.03864\] The MLIR Transform Dialect. Your compiler is more powerful than you think](https://arxiv.org/abs/2409.03864)