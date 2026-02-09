## Step 1 — Deep research & novelty check (what already exists, and what doesn’t)

### What exists (so we **don’t** accidentally claim novelty there)

- **Tokenized async-copy IR already exists** in mainstream toolchains:
  - MLIR’s **NVGPU dialect** has `nvgpu.device_async_copy`, `nvgpu.device_async_create_group`, and `nvgpu.device_async_wait` with an explicit **async token type**. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
  - Triton’s GPU dialect has `ttg.async_wait` with the familiar “wait until at most *num* groups outstanding” interface. ([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html?utm_source=openai))  

- **TMA descriptor creation is already represented**, but mostly by “call the driver”:
  - MLIR’s `nvgpu.tma.create.descriptor` explicitly **lowers to a runtime call** that eventually calls `cuTensorMapEncodeTiled`. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
  - The CUDA Driver API specifies the *encode-time* legality constraints (alignment, rank, dtype enums, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

- **Warp/warpgroup uniformity analysis exists** in general compiler infrastructure:
  - LLVM documents a **UniformityAnalysis** grounded in convergence and divergence analysis (abstract-interpretation style), including “temporal divergence.” ([llvm.org](https://llvm.org/docs/ConvergenceAndUniformity.html?utm_source=openai))  

- **The PTX memory model has a serious formalization pipeline already**:
  - Lustig et al. (ASPLOS 2019) provide a formal axiomatic model of PTX and a machine-checked Coq development (with Alloy testing). ([research.nvidia.com](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai))  

- **The key “correctness cliffs” are explicitly documented** (so we can build formal contracts against them):
  - `wgmma.*` requires warpgroup-uniform execution for `.aligned`, and misuse is UB; `wgmma.wait_group` is required before accessing certain registers. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - `cp.async.bulk.tensor` has per-arch/per-dtype byte-quanta restrictions and is weak-memory; completion has `.release` semantics at `.cluster` scope. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - `tensormap.replace` updates a 1024-bit tensor-map object and is a weak memory op on the whole object. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - `mbarrier`: using a non-barrier location is UB; and some waits require the `state` token to come from a prior arrive variant. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - TMEM (`tcgen05.*`) is dynamically allocated in power-of-two column blocks (unit 32 columns), with warp-uniform addressing constraints for `tcgen05.ld`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### What I did **not** find (i.e., room for true “math-first” novelty)

I found no evidence (in public docs / IR dialect docs / common compiler descriptions) of a GPU compiler whose **primary formulation** is any of the following:

1. **Tropical / max-plus algebra** as the *central scheduling calculus* for async copy + barrier + warpgroup MMA pipelines (with a correctness bridge to weak memory).  
2. **HyperLTL model checking** as a *proof backend* for warpgroup-uniformity obligations in GPU compilation (vs. dataflow-style uniformity). (AutoHyper exists as a tool, but not applied to this GPU-compiler problem.) ([arxiv.org](https://arxiv.org/abs/2301.11229?utm_source=openai))  
3. **Affine-lattice normalization (SNF/HNF) + proof-producing SMT** as an explicit *descriptor encodability synthesizer* for `CUtensorMap` / `cp.async.bulk.tensor` legality, producing reusable certificates/unsat cores rather than “call the driver and hope.” (Existing MLIR lowers to the driver call.) ([reviews.llvm.org](https://reviews.llvm.org/D155680))  

That gap is exactly where the following three directions live.

---

## Step 2 — Three math-first research directions

> **Constraint from you (P4):** each “Winning Factor” is *theoretical*, not “faster than Triton.”

---

# Direction 1 (Layout/Addressing Algebra): **Encodable Swizzle Groupoids + Dyadic-Capability Layouts**

### 1) Validated gap (from Stage‑1)
The seed layout math treats swizzles/layouts as composable mappings. Hardware requires **encodable** swizzles (finite vendor enums) and **resource protocols** for TMEM (dynamic allocation, power-of-two blocks, warp-uniform addressing). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

### 2) Math theory (primary novelty)
**Core idea:** turn “layout” into a *capability-carrying morphism* whose *algebra* bakes in encodability.

Define three layers:

1. **Encodable swizzle group** \(G_{\text{enc}}\).  
   Treat each hardware-exposed swizzle/interleave mode as an element of a **finite group/subgroup action** on address bits (or on 16B-chunk indices), rather than an arbitrary \( \mathbb{F}_2 \)-affine bijection. The point is:  
   - If \(g \in G_{\text{enc}}\), it is **encodable by construction** (no “swizzle is bijective but unencodable” failures).

2. **TMEM resource algebra** \(R_{\text{tmem}}\) as a **dyadic interval PCM**.  
   TMEM allocates columns in units of 32 columns, and the allocated column count must be a power of 2, within \([32,512]\). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
   Model allocatable blocks as dyadic intervals on a length‑512 line at base granularity 32. Disjoint union is partial → a **partial commutative monoid** (PCM).

3. **Semidirect product / action groupoid** \(R_{\text{tmem}} \rtimes G_{\text{enc}}\).  
   A “layout transform” becomes \((r, g)\):  
   - \(g\) rearranges address bits/lanes (swizzle/layout action),  
   - \(r\) transforms resource state (alloc/dealloc, capability splits/joins).

This gives a clean algebra for composition and equivalence:
- \((r_2, g_2)\circ(r_1, g_1) = (r_2 \circ g_2(r_1), g_2 g_1)\) (schematically),
- equivalence reduces to normal forms in a finite group + canonical dyadic decompositions.

### 3) Compiler artifact (“the solver/pass”)
**CapLayout Normalizer + Capability Verifier**

- **Normalizer:** computes canonical representatives of layout morphisms in \(R_{\text{tmem}} \rtimes G_{\text{enc}}\).  
- **Verifier:** checks that every TMEM access is guarded by a capability proving:
  - alloc size is legal (power-of-two * 32 columns, within bounds), ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - and `tcgen05.ld`’s “all threads in warp must use the same `taddr`” obligation holds for the region. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### 4) IR design (minimal but explicit)
Add a small dialect (or extension) with **first-class algebraic layout/capability values**:

- Types:
  - `!layout.enc<shape, space>` — an element of \(G_{\text{enc}}\) (finite).
  - `!tmem.cap<cols>` — a dyadic capability (cols ∈ {32,64,…,512}).
  - `!tmem.addr` — only constructible from `!tmem.cap`.

- Ops (illustrative):
  - `layout.enc.choose {mode = …}` — selects generator in \(G_{\text{enc}}\).
  - `layout.enc.compose(a,b) -> layout.enc`
  - `tmem.alloc(cols) -> !tmem.cap<cols>` and `tmem.dealloc(cap)`
  - `tmem.addr_of(cap, lane, col) -> !tmem.addr`
  - `tmem.ld(addr) -> …` requires `addr` built from a cap.
  - Optional: `tmem.addr_uniformize(addr) -> !tmem.addr_uniform` (a proof-carrying refinement).

### 5) Legality story (P1)
- **Encodability-by-construction:** restricting swizzles to \(G_{\text{enc}}\) eliminates the “math swizzle space too large” mismatch against finite hardware modes. (The finite enums are explicit in driver-level APIs / IR dialects.) ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- **TMEM legality:** the dyadic PCM enforces the PTX allocation rules (unit 32 columns, power-of-two, range bounds), and the typing rules prevent `tcgen05.ld` UB by construction (uniform `taddr` requirement becomes a type obligation). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### 6) Temporal story (P1)
Layouts become **effectful** in the presence of descriptor mutation and fences:

- PTX `tensormap.replace` is a weak memory op on a 1024-bit object. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- MLIR has `nvgpu.tma.fence.descriptor` precisely because *temporal ordering* matters after descriptor modification. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

So the math must include a *phase distinction*:
- `Desc[Dirty] --fence--> Desc[Ready]`  
and layout morphisms that depend on descriptors require `Ready`.

### 7) Lowering plan (P2)
- Lower `!layout.enc` choices to:
  - bit-manipulation codegen for swizzle/address transforms, or
  - fixed enum selections when targeting `nvgpu.tma.create.descriptor` / `cuTensorMapEncodeTiled` with supported swizzles. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
- Lower TMEM ops to `tcgen05.alloc/dealloc/ld/st` sequences (via NVVM intrinsics where available). TMEM rules are directly in PTX. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### 8) Evaluation plan (must include optimality proof / solver efficiency) (P2)
- **Optimality proof (within the encodable set):**  
  Because \(G_{\text{enc}}\) is finite, show: “the chosen swizzle minimizes a bank-conflict objective over all encodable swizzles,” by exhaustive enumeration (or group-theoretic pruning) with a formally defined cost.  
- **Solver efficiency metrics:**
  - canonicalization time per layout morphism,  
  - certificate size: (chosen group element ID + dyadic capability decomposition),  
  - % of TMEM regions verified without fallback.  
- **Correctness metric:** 0 occurrences of TMEM UB obligations violated (static proof), especially `tcgen05.ld` uniform `taddr`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### 9) Novelty check (P3)
- Existing IRs represent TMA/TMEM/wgmma ops, but they do **not** present a *unified algebraic object* that combines “encodable swizzle space” + “dyadic resource protocol” as the primary semantics. MLIR’s TMA descriptor creation lowers to the driver call, not to an algebraic synthesis/normalization. ([reviews.llvm.org](https://reviews.llvm.org/D155680))  

### Winning Factor (P4, theoretical)
**“Encodability-by-construction via a finite groupoid semantics”**: the key novelty is replacing “layouts as arbitrary bijections” with **layouts as morphisms in an encodable swizzle groupoid semidirect with a dyadic resource PCM**, yielding canonical forms and decidable equivalence that directly match hardware’s finite mode sets and allocation protocol.

---

# Direction 2 (Scheduling/Time Algebra): **Timed Pomsets + Max-Plus Throughput as a Certified Pipeline Scheduler**

### 1) Validated gap
Async tensor copies and warpgroup MMA are:
- **non-blocking** and **weak-memory**, with explicit completion semantics, ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- governed by **token protocols** (`mbarrier`, `commit_group`, `wait_group`), ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
which seed layout math does not model (it’s extensional, timeless).

### 2) Math theory (primary novelty)
A two-layer semantics that cleanly separates:
1. **Correctness (partial orders):** use **pomsets with preconditions / event-structure style semantics** to represent allowed reorderings under weak memory. ([2020.splashcon.org](https://2020.splashcon.org/details/splash-2020-oopsla/70/Pomsets-with-Preconditions-A-Simple-Model-of-Relaxed-Memory?utm_source=openai))  
2. **Performance (time):** use **max-plus (tropical) algebra** to compute optimal steady-state throughput and earliest-start schedules for the token dependency graph. ([link.springer.com](https://link.springer.com/article/10.1007/s10626-019-00294-w?utm_source=openai))  

The novelty is not “max-plus exists”—it’s the *bridge*:

> A compiler transformation is valid iff it is a **pomset refinement** (preserves allowed behaviors), and it is *optimal* (or bounded-optimal) with respect to a **max-plus model** of event timing.

### 3) Compiler artifact
**Pomset‑Checked Tropical Scheduler (PCTS)**

- **Input:** an IR region containing async copies, `mbarrier` ops, `wgmma` groups.  
- **Output:** a reordered / pipelined region + a proof artifact:
  - a pomset refinement certificate, and
  - a max-plus optimality certificate (e.g., critical cycle mean).

### 4) IR design
You can piggyback on existing token IRs, but add *timing semantics*:

- Represent operations as events with:
  - token deps (SSA),
  - memory-order annotations (weak/release),
  - and abstract latencies \(d(e)\).

Concretely, MLIR already has:
- async tokens and grouping ops in NVGPU, ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
- and `nvgpu.warpgroup.mma` that wraps the fence/commit/wait group structure. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

PCTS adds:
- `event.time<d>` attributes,
- and an explicit “reads become visible after wait” edge class.

### 5) Algorithm (P2)
**Phase A — build correctness pomset constraints**
- Extract events and relations:
  - program order (per thread),
  - token edges (SSA),
  - synchronizes-with edges from `mbarrier` completion (`.release`), ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - required waiting edges:
    - `wgmma.wait_group` must precede accumulator/input register reuse, else UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

Check: the transformed schedule is a **refinement** (no new behaviors).

**Phase B — compute max-plus schedule**
- Construct a precedence graph \(G=(V,E)\) where each edge has weight = latency.  
- Solve for:
  - earliest-start times (max-plus linear recurrences),
  - and steady-state initiation interval (II) via **maximum cycle mean** computations inside the max-plus model. ([link.springer.com](https://link.springer.com/article/10.1007/s10626-019-00294-w?utm_source=openai))  

### 6) Legality story (P1)
Legality is enforced as *temporal obligations*:

- `cp.async.bulk.tensor` is weak-memory and has explicit completion mechanism via `mbarrier::complete_tx::bytes` with release semantics at cluster scope. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- `mbarrier` tokens have provenance requirements; misuse is UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- `wgmma.*` requires warpgroup-uniform execution and correct wait ordering. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

So PCTS treats “legality” as a **set of mandatory edges** in the pomset. Any schedule that violates them is rejected.

### 7) Temporal story (P1)
This direction’s core contribution is explicitly temporal:

- The max-plus layer yields a **throughput-optimal** (or provably near-optimal) pipeline schedule under the extracted dependency constraints. ([link.springer.com](https://link.springer.com/article/10.1007/s10626-019-00294-w?utm_source=openai))  
- The pomset layer guarantees the schedule is consistent with weak memory and token semantics. ([2020.splashcon.org](https://2020.splashcon.org/details/splash-2020-oopsla/70/Pomsets-with-Preconditions-A-Simple-Model-of-Relaxed-Memory?utm_source=openai))  

### 8) Lowering plan (P2)
- Lower scheduled async ops to:
  - PTX `cp.async.bulk.*` / `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group`, ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - or MLIR NVGPU equivalents when upstream passes exist. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
- Lower warpgroup MMA to NVVM `wgmma.*` intrinsics (NVVM dialect has explicit fence ops). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVVMDialect/?utm_source=openai))  

### 9) Evaluation plan (must include optimality proof / solver efficiency) (P2)
- **Optimality proof:**  
  Show that PCTS achieves the **max-plus critical cycle mean bound** for the region’s dependency graph (or provide a certificate of the computed bound and the achieved II). ([link.springer.com](https://link.springer.com/article/10.1007/s10626-019-00294-w?utm_source=openai))  
- **Solver efficiency:**  
  - time to build pomset constraints,
  - time to compute max-plus schedule,
  - size of certificates.  
- **Correctness tests:** litmus-style tests keyed to PTX weak-memory + barrier semantics; note PTX memory model already has formal test infrastructure, which you can reuse as oracle inputs. ([github.com](https://github.com/NVlabs/ptxmemorymodel?utm_source=openai))  

### 10) Novelty check (P3)
- Existing compilers/IRs have **tokens and groups** (NVGPU/Triton). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
- Existing work has a **formal PTX memory model** (axiomatic/Coq). ([research.nvidia.com](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai))  
But I found no sign that GPU compilation currently uses:
- **pomset refinement as the correctness criterion** *and*
- **max-plus algebra as the scheduling objective/certificate**  
as the *primary formulation* for async pipelines.

### Winning Factor (P4, theoretical)
**“A certified scheduling calculus”**: the novelty is the **two-semantics composition**—pomset-based weak-memory correctness + max-plus optimality—yielding schedules that come with **proof objects** (refinement + throughput bound), not just heuristics.

---

# Direction 3 (Constraint/Legality Logic): **Affine-Lattice Encodability + Proof-Producing SMT for TMA / cp.async.bulk.tensor**

### 1) Validated gap
TMA and `cp.async.bulk.tensor` legality is not “just layout correctness”; it is:
- bounded inequalities (e.g., rank limits, box bounds),
- congruences (alignment/multiple-of),
- finite enums (swizzle/interleave/dtype),
- and arch gating,  
with hard UB cliffs. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

Existing IR support often delegates descriptor construction to runtime driver calls. ([reviews.llvm.org](https://reviews.llvm.org/D155680))  

### 2) Math theory (primary novelty)
Model descriptor encodability as:

\[
x \in P \cap (a + \Lambda) \cap \bigvee_{m \in \text{Modes}} M_m
\]

- \(P\): a rational polyhedron (bounds, rank constraints).  
- \(a+\Lambda\): an affine lattice capturing congruences (“multiple-of 16/32/128” etc).  
- \(M_m\): finite disjunction over enum choices (swizzle/interleave/dtype/arch mode).

Then solve with a **hybrid algebra+logic stack**:

1. **Algebraic normalization:** compute a basis / canonical form of \(\Lambda\) via SNF/HNF-style reasoning (so congruences become structured, not ad hoc).  
2. **Proof-producing SMT:** discharge the combined problem in QF_BV + QF_LIA + finite enums, extracting:
   - a satisfying assignment (descriptor parameters), or
   - an unsat core / proof object.

cvc5 explicitly supports **unsat cores and proofs** as retrievable artifacts, which is key for “compiler certificate” workflows. ([cvc5.github.io](https://cvc5.github.io/tutorials/beginners/outputs.html?utm_source=openai))  

### 3) Compiler artifact
**Descriptor Encodability Synthesizer (DES)**

Given a high-level tensor description (shape/strides/tile/dtype/mode), DES returns:

- **SAT:** a concrete `CUtensorMapEncode*` parameterization + a certificate (model + optional proof of constraint satisfaction).
- **UNSAT:** a small unsat core that maps back to “what to change” (tile shape, rank, dtype, swizzle).

### 4) IR design (P2)
Add explicit symbolic legality regions:

- `tma.encode_symbolic`:
  - inputs: symbolic dims/strides/box/enum vars,
  - outputs: `!tma.desc<b1024>` **plus a legality certificate handle**.

- `cpasync.tensor.typecheck`:
  - attaches the PTX legality constraints (including exact byte quanta) as a constraint object.

- `legal.prove`:
  - runs DES, producing either concrete attrs or an `illegal` diagnostic with unsat core.

This complements MLIR’s current “call the driver” approach. ([reviews.llvm.org](https://reviews.llvm.org/D155680))  

### 5) Algorithm (P2)
- Encode constraints from:
  - CUDA Driver API requirements for tensor maps, ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
  - PTX restrictions for `cp.async.bulk.tensor` (box sizes, alignments, swizzle sets), ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - plus memory-model constraints for descriptor mutation (`tensormap.replace` weak memory). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

Pipeline:
1. Partition constraints into (bounds, congruences, enums).
2. Normalize congruences (SNF/HNF layer) to reduce SMT burden.
3. Emit SMT-LIB and solve with proof/unsat-core enabled.
4. Reify model back into IR attributes + attach proof object hash.

### 6) Legality story (P1)
- Guarantees: if DES says “legal,” the descriptor respects the driver’s encode constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- Guarantees: if DES emits `cp.async.bulk.tensor`, it satisfies PTX’s per-dtype/per-arch constraints and weak-memory completion semantics assumptions used later by scheduling passes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### 7) Temporal story (P1)
DES is not purely spatial: it must account for temporal correctness around descriptor/state:

- `tensormap.replace` is weak memory; therefore any legalization that uses on-device descriptor mutation must require a fence/ordering protocol before first use. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- The legality checker can emit **typestate transitions**:
  - `desc: Dirty → Ready` only after `tma.fence.descriptor`. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

This makes descriptor encodability compatible with later temporal scheduling (Direction 2).

### 8) Lowering plan (P2)
- If all parameters are compile-time constants:
  - precompute descriptor bytes (or precompute encode arguments) and embed as constants / kernel params.
- Otherwise:
  - lower to runtime `cuTensorMapEncodeTiled` calls (as MLIR already does), but guarded by compile-time proved obligations when possible. ([reviews.llvm.org](https://reviews.llvm.org/D155680))  

### 9) Evaluation plan (must include optimality proof / solver efficiency) (P2)
- **Solver efficiency metrics:**
  - median/worst solver time per descriptor,
  - timeout rate under realistic autotuning search loads,
  - average unsat-core size and “actionability” (how often it maps to a single knob like tile or swizzle).
- **Soundness metric:**  
  - 0 mismatches: “DES says SAT” but driver rejects, or PTX constraints violated (validated by compilation+runtime microtests). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- **Certificate metric:**  
  - size/format of proof objects (cvc5 proof/unsat-core). ([cvc5.github.io](https://cvc5.github.io/tutorials/beginners/outputs.html?utm_source=openai))  

### 10) Novelty check (P3)
- MLIR already has a TMA descriptor op, but it **lowers to a runtime driver call**, not to a solver that can *prove encodability* and produce unsat cores/certificates. ([reviews.llvm.org](https://reviews.llvm.org/D155680))  
- PTX legality constraints (exact byte quanta, weak memory completion) are documented, but there’s no evidence they’re currently handled by a proof-producing encodability calculus in GPU compilation toolchains. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### Winning Factor (P4, theoretical)
**“Encodability as satisfiability with certificates”**: the novelty is making descriptor legality a first-class **decidable object**—an affine-lattice/polyhedron problem discharged by **proof-producing SMT**, yielding reusable certificates and minimal unsat cores for compiler search.

---

## Step 3 — Scoring decision matrix

| Direction | Math_Novelty (1–5) | Hardware_Relevance (1–5) | Impl_Risk (1–5) | Why_it_wins |
|---|---:|---:|---:|---|
| **D1: Encodable Swizzle Groupoids + Dyadic-Capability Layouts** | 4 | 4 | 3 | *Theoretical win:* **encodability-by-construction** via a finite groupoid/PCM semantics that unifies swizzles + TMEM allocation into one algebra with canonical forms and decidable equivalence. |
| **D2: Timed Pomsets + Max-Plus Certified Scheduling** | 5 | 5 | 4 | *Theoretical win:* a **two-layer certified calculus** (pomset refinement for weak memory + max-plus optimality for throughput), producing proof objects for both correctness and schedule optimality. |
| **D3: Affine-Lattice Encodability + Proof-Producing SMT** | 4 | 5 | 4 | *Theoretical win:* descriptor legality becomes a **proof-producing satisfiability problem** (lattice ∩ polyhedron ∩ finite enums) with unsat cores that drive compilation and autotuning deterministically. |

---

If you want the cleanest “proposal spine” for an ASPLOS/POPL submission: **Direction 2** is the most “mathematically singular” (one unified story for time + weak memory), while **Direction 3** is the most “compiler-contract immediate” (it turns today’s runtime legality cliffs into compile-time certificates).

---
Learn more:
1. ['nvgpu' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)
2. [TritonGPUOps — Triton documentation](https://triton-lang.org/main/dialects/TritonGPUOps.html?utm_source=openai)
3. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)
4. [Convergence And Uniformity — LLVM 23.0.0git documentation](https://llvm.org/docs/ConvergenceAndUniformity.html?utm_source=openai)
5. [A Formal Analysis of the NVIDIA PTX Memory Consistency Model. | Research](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai)
6. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
7. [AutoHyper: Explicit-State Model Checking for HyperLTL](https://arxiv.org/abs/2301.11229?utm_source=openai)
8. [⚙ D155680 \[mlir\]\[nvgpu\] Add \`tma.create.descriptor\` to create tensor map descriptor](https://reviews.llvm.org/D155680)
9. [Pomsets with Preconditions: A Simple Model olaxed Memory (SPLASH 2020 - OOPSLA) - SPLASH 2020](https://2020.splashcon.org/details/splash-2020-oopsla/70/Pomsets-with-Preconditions-A-Simple-Model-of-Relaxed-Memory?utm_source=openai)
10. [Analysis and control of max-plus linear discrete-event systems: An introduction | Discrete Event Dynamic Systems | Springer Nature Link](https://link.springer.com/article/10.1007/s10626-019-00294-w?utm_source=openai)
11. ['nvvm' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVVMDialect/?utm_source=openai)
12. [GitHub - NVlabs/ptxmemorymodel](https://github.com/NVlabs/ptxmemorymodel?utm_source=openai)
13. [SMT Solver Outputs — Satisfiability Modulo Theories: A Beginner's Tutorial documentation](https://cvc5.github.io/tutorials/beginners/outputs.html?utm_source=openai)
