## 1. Problem Validation: The Evidence

### 1.1 “Layout legality cliffs” are *real* (and user-visible), not a paper-invented strawman

**Developer pain is observable in public support channels:**

- **Swizzle legality is brittle enough that developers hit `CUDA_ERROR_INVALID_VALUE` with apparently “reasonable” settings.** In a Nov 10, 2025 NVIDIA Developer Forums thread, the author reports that enabling tensor-map swizzling returns `CUDA_ERROR_INVALID_VALUE` for some swizzle/tile combinations and explicitly says they *“can’t find any requirement for this in either the PTX ISA or the `cuTensorMapEncodeTiled` documentation.”* ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/cuda-error-invalid-value-when-creating-tensor-maps-with-swizzling/350966?utm_source=openai))  
  *Reviewer takeaway:* this is exactly the “legality cliff” story—small discrete choices (swizzle width vs inner dimension) abruptly turn an intended fast path into an API failure, and the debugging loop is documentation archaeology.

- **Even *linking* the tensor-map encode API becomes a practical integration bottleneck.** A CUTLASS feature request (May 31, 2024) describes PyTorch’s constraint of linking against the CUDA Driver API and calls out **a direct dependency on `cuTensorMapEncodeTiled` “causing issues,”** motivating adding it to a host adapter indirection layer. ([github.com](https://github.com/NVIDIA/cutlass/issues/1566?utm_source=openai))  
  A follow-up CUTLASS bug (Jul 10, 2024) reports **`undefined symbol: cuTensorMapEncodeTiled`** persisting even after the change, illustrating how “fast-path enablement” is coupled to fragile toolchain / ABI surfaces. ([github.com](https://github.com/NVIDIA/cutlass/issues/1624?utm_source=openai))

**The hardware/driver docs explicitly encode the cliff structure** (hard bounds + cross-field coupling):

- In CUDA Driver API **Tensor Map Object Management** (`cuTensorMapEncodeTiled`), the descriptor is gated by concrete requirements such as:
  - `tensorMap` address **64-byte aligned**,
  - `tensorRank` bounded (and/or constrained by interleave),
  - `boxDim[i] ≤ 256`,
  - `elementStrides[i] ≤ 8`,
  - and the famous cross-coupling **`interleave == CU_TENSOR_MAP_INTERLEAVE_32B ⇒ swizzle == CU_TENSOR_MAP_SWIZZLE_32B`**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  These aren’t “optimizations”; they are *admissibility constraints* that decide whether TMA is even expressible.

- The CUDA C++ Programming Guide adds **additional swizzle validity rules** beyond “rank/stride/box”: when applying a TMA swizzle, **global memory must be 128B aligned**, the **shared box’s inner dimension must satisfy the swizzle width table**, and *“If these requirements are not met, the instruction is considered invalid.”* (Section **10.29.3.2 “The Swizzle Modes”**). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?utm_source=openai))  
  *Reviewer takeaway:* legality is split across (at least) the Driver API and the Programming Guide; a compiler pass that only models one will still generate “invalid” code or hit runtime API failures.

### 1.2 “Async orchestration correctness” is also real (and the contract is nontrivial)

On Hopper+ fast paths, async copy and barrier semantics are not folklore—they are specified in PTX, and you can be *correct-by-layout* yet *wrong-by-protocol*.

- In PTX ISA **8.5**, `cp.async.bulk.tensor` is explicitly **non-blocking** and supports two completion mechanisms (section **9.7.10.24.9**):
  - `.mbarrier::complete_tx::bytes` (barrier-based completion),
  - `.bulk_group` (bulk async-group semantics). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html))  
- PTX also spells out **bulk-group sequencing**: `cp.async.bulk.commit_group` and `cp.async.bulk.wait_group` exist, and crucially there is **“no memory ordering guarantee” between operations within the same bulk async-group** (sections **9.7.10.24.12–13**). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html))  
  *Reviewer takeaway:* a compiler that “just emits async ops” without protocol tracking can silently introduce reordering bugs.

- For the barrier side: PTX ISA **8.5** defines the **tx-count tracking model** of `mbarrier`:
  - **expect-tx operation** increases tx-count (section **9.7.14.15.5.1**),
  - **complete-tx operation** decrements tx-count (section **9.7.14.15.5.2**),
  - plus the instructions `mbarrier.expect_tx` (**9.7.14.15.11**) and `mbarrier.complete_tx` (**9.7.14.15.12**). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html))  
  This makes “pipeline correctness” a resource-accounting invariant, not just a barrier placement heuristic.

**Evidence that tooling ecosystems treat this as a first-class complexity point:**
- NVIDIA CCCL tracked feature work explicitly around `mbarrier` completion semantics and `expect_tx` requirements (Sep 8, 2023 issue). ([github.com](https://github.com/NVIDIA/cccl/issues/419?utm_source=openai))  
- CCCL also publishes typed PTX wrappers for `cp.async.bulk.tensor` variants, which—by the sheer surface area—signals that correctness requires careful API modeling (and is not a one-off inline asm trick). ([nvidia.github.io](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html?utm_source=openai))

### 1.3 Blackwell TMEM adds *another* legality layer (collectives + allocation + lane partitions)

PTX ISA **9.1** introduces “Tensor Memory” + tcgen05, and the rules are harsh enough to be UB-triggering if mishandled:

- `tcgen05.alloc` has non-negotiable constraints (PTX ISA 9.1 **9.7.16.7.1**):
  - allocation unit is **32 columns** and **all lanes per column**,
  - `nCols` must be a **power of two** and in range **[32, 512]**,
  - allocation/deallocation is **warp-collective** (`.aligned` means all threads must execute uniformly; otherwise **behavior is undefined**). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/))
- **Lane partitioning:** in **9.7.16.8.1**, each warp in a warpgroup can access only a lane range (0–31, 32–63, 64–95, 96–127). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/))
- **Collective addressing UB:** `tcgen05.ld` requires **all threads in the warp to specify the same `taddr`** or behavior is undefined (PTX ISA 9.1 **9.7.16.8.3**). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/))

*Reviewer takeaway:* a “layout algebra” that ignores execution grain + collectivity constraints is incomplete on Blackwell; the metal introduces **grain-typed legality**, not just indexing math.

### 1.4 SOTA compiler stacks are *already* contorting around this (so the problem is current, not hypothetical)

A strong “problem is real” anchor is that mainstream kernel stacks call out TMA as both important and complex:

- PyTorch’s Hopper TMA deep dive (Hoque et al., PyTorch Blog, 2024) explicitly frames TMA as a new fully asynchronous copy engine and states: **CUTLASS has built-in TMA support**, while **Triton exposes TMA via an experimental API**. ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai))  
- PyTorch issues show **correctness bugs and illegal memory accesses** in the wild involving “on-device TMA” in Triton/Inductor flows (Jun 29, 2025). ([github.com](https://github.com/pytorch/pytorch/issues/157240?utm_source=openai))  
  Another issue reports wrong results depending on autotuner choices among TMA variants (Aug 6, 2025), underscoring that the space is both combinatorial and brittle. ([github.com](https://github.com/pytorch/pytorch/issues/159940?utm_source=openai))

**Net: the “Layout × Schedule × Legality × Async protocol” explosion is not contrived.** It is reflected in (a) specs with discrete constraints, (b) developer forum failures, and (c) production compiler + library integration bugs across 2024–2025.

---

## 2. Taxonomy of Approaches (The Landscape)

### Competitor Matrix

> Columns requested: Approach_Name | Method (Search/Heuristic/Templates) | Handles_TMA_Legality? | Handles_Async_Barriers? | Why_It_Falls_Short

| Approach_Name | Method (Search/Heuristic/Templates) | Handles_TMA_Legality? | Handles_Async_Barriers? | Why_It_Falls_Short |
|---|---|---:|---:|---|
| **Triton** — Tillet et al. (MAPL 2019) | Heuristic compiler + autotuning for tile params | **Partially / ad hoc** (via chosen tiles + runtime helper paths; not a legality *synthesizer*) | **Partially** (can generate async patterns; no compile-time proof of tx-count/phase correctness) | Triton is designed for productivity and performance via tiling; it is not (per paper) a solver for discrete descriptor-field constraints or a protocol verifier. ([pldi19.sigplan.org](https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations?utm_source=openai)) |
| **Linear Layouts** — Zhou et al. (arXiv 2025) | Algebraic layout conversion system (linear maps over \( \mathbb{F}_2 \)) | **No** (scope: layout representation + conversion; does not claim to emit `CUtensorMap` field witnesses) | **No** (does not claim to verify `mbarrier` tx-count / completion-mode contracts) | Solves “math of layouts” (and engineering bugs in layout conversion), not “metal legality + async protocol.” ([arxiv.org](https://arxiv.org/abs/2505.23819?utm_source=openai)) |
| **ISL layout relations** — Bhaskaracharya et al. (arXiv 2025) | Formal modeling using integer-set relations | **No** (does not address driver/PTX descriptor encoding constraints) | **No** | Explicitly pitched as unifying mathematical representations; doesn’t couple to concrete HW legality tables (swizzle enums, alignment, tx-count). ([arxiv.org](https://arxiv.org/abs/2511.10374?utm_source=openai)) |
| **Categorical CuTe layouts** — Carlisle et al. (arXiv 2026) | Categorical semantics + tractable-layout class | **No** | **No** | Valuable for algebra/canonicalization; does not encode TMA legality or tcgen05 collective rules. ([arxiv.org](https://arxiv.org/abs/2601.05972?utm_source=openai)) |
| **CUTLASS / CuTe ecosystem (NVIDIA)** (practical evidence via issues + PyTorch blog) | Templates + hand-engineered kernel families | **Yes for supported kernels** (TMA paths exist) | **Yes for supported kernels** (asynchronous pipeline paradigm in CUTLASS) | High performance but “by construction” for *specific kernel families*; not a general synthesis/proof framework. Integration friction (e.g., `cuTensorMapEncodeTiled` dependency) shows the boundary is brittle. ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai)) |
| **Ansor** — Zheng et al. (OSDI 2020) | Hierarchical search + evolutionary refinement + learned cost model | **No (not explicit)** | **No (not explicit)** | Designed for exploring scheduling/tiling spaces for tensor programs; does not model CUDA TMA descriptor admissibility constraints or mbarrier tx-count semantics as a correctness layer. ([usenix.org](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai)) |
| **TVM** — Chen et al. (OSDI 2018) | Graph + operator-level optimization; cost-model guided exploration | **No (not TMA-specific)** | **No (not `mbarrier`-protocol-specific)** | TVM targets broad portability; its paper focus is on end-to-end DL compilation and search. The CUDA TMA/tcgen05 legality layer postdates and is more discrete than the paper’s described model. ([usenix.org](https://www.usenix.org/conference/osdi18/presentation/chen?utm_source=openai)) |
| **Pluto** — Bondhugula et al. (PLDI 2008) | Polyhedral ILP scheduling for affine loop nests | **No** | **No** | Polyhedral excels at affine dependence + tiling; TMA legality includes non-affine, enum-coupled constraints (e.g., interleave/swizzle coupling) and async contracts absent from the polyhedral model. ([ece.lsu.edu](https://www.ece.lsu.edu/jxr/pluto/?utm_source=openai)) |
| **PPCG** — Verdoolaege et al. (ACM TACO 2013) | Polyhedral parallel code generation targeting CUDA | **No** (generates CUDA; not `CUtensorMap` descriptor synthesis) | **No** | PPCG produces CUDA from static control; it does not synthesize/verify modern Hopper/Blackwell fast-path descriptors or `mbarrier` tx-count protocols. ([llvm.googlesource.com](https://llvm.googlesource.com/llvm-project/%2B/refs/tags/llvmorg-11.1.0/polly/lib/External/ppcg/README?utm_source=openai)) |
| **egg** — Willsey et al. (POPL 2021) | Equality saturation library w/ e-class analyses | **Not domain-specific** | **Not domain-specific** | Provides the *machinery* (e-graphs, analyses), not the GPU legality theories or witness generation for CUDA descriptors/protocols. ([popl21.sigplan.org](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai)) |
| **Tensat** — Yang et al. (MLSys 2021) | EqSat for **tensor computation graph** rewriting | **No (graph-level)** | **No (graph-level)** | Optimizes computation graphs (phase ordering), not kernel-level layout/descriptor legality or PTX async protocols. ([mwillsey.com](https://www.mwillsey.com/papers/tensat?utm_source=openai)) |
| **EqSat + MCTS for tensor graphs** — Hartmann et al. (arXiv 2024; PACT 2024 claim) | EqSat + MCTS-guided e-graph construction | **No (graph-level)** | **No (graph-level)** | Still a graph-rewrite optimizer; nothing about emitting legal Hopper TMA descriptors or verifying `mbarrier`/bulk-group protocols. ([arxiv.org](https://arxiv.org/abs/2410.05534?utm_source=openai)) |
| **eqsat dialect** — Merckx et al. (arXiv 2025) | EqSat represented **inside MLIR** | **Not domain-specific** | **Not domain-specific** | Infrastructure for keeping e-graphs in IR; your novelty must be in *GPU legality analyses + solver-backed witness extraction*, not “EqSat in MLIR” per se. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai)) |
| **SEER** — Cheng et al. (arXiv 2023) | E-graph rewriting with MLIR for HLS superoptimization | **No (not CUDA TMA)** | **No (not PTX async)** | Strong prior art for “e-graphs in MLIR + performance search,” but targets HLS (hardware generation) not NVIDIA GPU legality + async copy/barrier protocols. ([arxiv.org](https://arxiv.org/abs/2308.07654?utm_source=openai)) |

---

## 3. Deep Dive: Equality Saturation in Compilers (2024–2026 reality check)

### 3.1 `egg` and what it *actually* gives you (and what it doesn’t)

- `egg` (Willsey et al., POPL 2021) contributes **rebuilding** and **e-class analyses**—the key practical hooks you need to carry analyses like “alignment lower bound” or “maybe-legal swizzle” through saturation. ([popl21.sigplan.org](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai))  
- However, `egg` is **intentionally domain-agnostic**; it doesn’t come with GPU legality semantics. So *any* claim that “LegalEGraph is novel” cannot be “we use egg,” it must be:  
  1) **the legality analysis lattice**, and  
  2) **a witness-producing extraction objective** that respects CUDA’s discrete constraints.

### 3.2 EqSat in MLIR (the “you will be scooped” line of attack)

- **eqsat dialect** (Merckx et al., arXiv 2025) is the cleanest “dangerous” foundation competitor: it argues for representing e-graphs *natively in MLIR IR* and maintaining them through compilation. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
  If you propose “EqSat as an MLIR pass,” reviewers will say: *Merckx already did that*. Your differentiator must therefore be **GPU-contract-carrying analyses and solver-backed witness generation**, not the e-graph container.

- **SEER** (Cheng et al., arXiv 2023) is another “dangerous” adjacent: it explicitly uses e-graphs + MLIR to explore equivalent implementations at scale for HLS. ([arxiv.org](https://arxiv.org/abs/2308.07654?utm_source=openai))  
  It sets a precedent that “MLIR + e-graphs = superoptimization platform,” so again: novelty must be your *domain-specific legality layer* and evaluation hooks.

### 3.3 EqSat for tensor graphs ≠ kernel legality (but reviewers will still cite it)

- **Tensat** (Yang et al., MLSys 2021) uses equality saturation to avoid phase-ordering problems in **deep learning computation graphs**. ([mwillsey.com](https://www.mwillsey.com/papers/tensat?utm_source=openai))  
- **Hartmann et al.** (arXiv 2024; “to be published in PACT ’24” per indexing) add MCTS guidance to build better e-graphs for tensor computation graphs. ([arxiv.org](https://arxiv.org/abs/2410.05534?utm_source=openai))  

These works matter because reviewers will ask: *“Why isn’t this just Tensat but lower-level?”* Your answer must be: because **kernel codegen fast paths are gated by hardware-admissibility constraints that don’t exist at graph level**—e.g., `CUtensorMap` field coupling and `mbarrier` tx-count protocols.

### 3.4 Has anyone integrated “legality predicates + SMT” into EqSat for GPUs already?

From the evidence surfaced here:

- The *EqSat-for-tensor-graphs* line (Yang MLSys’21; Hartmann PACT’24 claim) is explicitly about graph rewrite sequences and cost functions, not about emitting **concrete CUDA descriptor fields** under driver constraints. ([mwillsey.com](https://www.mwillsey.com/papers/tensat?utm_source=openai))  
- The *EqSat-in-MLIR* line (Merckx arXiv’25; Cheng arXiv’23) is infrastructure / HLS, not NVIDIA Hopper/Blackwell legality. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  

**But:** there is a near-miss: Merckx et al. (arXiv 2025, Julia IR) discuss **ILP-based extraction** under dominance constraints, demonstrating that “solver-in-the-loop extraction” is now mainstream EqSat technique. ([arxiv.org](https://arxiv.org/abs/2502.17075?utm_source=openai))  
So reviewers may argue: “ILP in extraction isn’t new.” That pushes you to emphasize that your solver is not just selecting among e-classes; it is **constructing a hardware witness** (descriptor fields + async-plan) that satisfies **discrete ISA/API constraints**.

---

## 4. Critical Novelty Defense

### 4.1 What *must* be your crisp value proposition (as a reviewer would demand)

Your proposal (“LegalEGraph/LegalSAT” + AsyncContract + TierGrain) survives **only if you are explicit that you are solving two orthogonal gaps** that current approaches treat as either heuristics or “handwritten library magic”:

1. **Hardware legality witness generation for TMA**  
   Example “hardware hook” constraint that must be first-class:  
   - `interleave == CU_TENSOR_MAP_INTERLEAVE_32B ⇒ swizzle == CU_TENSOR_MAP_SWIZZLE_32B` (Driver API). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
   A polyhedral compiler (Bondhugula PLDI’08; Verdoolaege TACO’13) doesn’t natively model enum-coupled legality; Ansor/TVM (Zheng OSDI’20; Chen OSDI’18) don’t describe such constraints in their optimization spaces. ([ece.lsu.edu](https://www.ece.lsu.edu/jxr/pluto/?utm_source=openai))  
   *Your differentiator:* legality is not “checked at the end”; it’s a **first-class analysis + solver witness** inside extraction.

2. **Async protocol correctness for generated schedules**  
   Example hard contract: `mbarrier` tx-count semantics (PTX ISA 8.5 §9.7.14.15.5.*; instructions §9.7.14.15.11–12). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html))  
   And the split completion mechanisms for `cp.async.bulk.tensor` (PTX ISA 8.5 §9.7.10.24.9) plus bulk-group ordering caveat (§9.7.10.24.12). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html))  
   *Your differentiator:* this is a **typestate/effect discipline** (AsyncContract) that prevents deadlocks/UB by construction.

### 4.2 The “Dangerous Three” (most similar) and how you must differentiate

#### (1) Merckx et al., **eqsat dialect** (arXiv 2025)
- **Why it’s dangerous:** it already sells “EqSat *inside MLIR*,” exactly the engineering story you might rely on. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
- **Your differentiation (must be stated like this):**  
  “Merckx et al. provide an IR substrate for e-graphs; we contribute a *domain semantics*—GPU legality analyses and witness extraction that synthesize `CUtensorMap` fields and async completion plans satisfying CUDA Driver + PTX contracts.”  
  Then immediately cite a concrete constraint you solve (e.g., interleave→swizzle coupling; 128B global alignment for swizzle; tx-count completeness), and show eqsat dialect doesn’t claim anything about those. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))

#### (2) Zhou et al., **Linear Layouts** (arXiv 2025)
- **Why it’s dangerous:** it is already “layout theory integrated into Triton,” and claims to avoid “quadratic explosion” in layout-to-layout conversions—reviewers may say your legality-aware rewrite system is just “Linear Layouts with a different engine.” ([arxiv.org](https://arxiv.org/abs/2505.23819?utm_source=openai))  
- **Your differentiation:**  
  “Linear Layouts formalize *spatial equivalence and conversion* of layouts; we target *metal admissibility*: synthesizing concrete driver/PTX descriptor fields under hard caps (rank/box/stride/swizzle enums) and coupling those choices to a correct async pipeline plan.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))

#### (3) Cheng et al., **SEER** (arXiv 2023)
- **Why it’s dangerous:** it is already “superoptimization with MLIR + e-graphs,” and it demonstrates end-to-end performance improvement—reviewers may say you are rebranding SEER for GPUs. ([arxiv.org](https://arxiv.org/abs/2308.07654?utm_source=openai))  
- **Your differentiation:**  
  “SEER targets HLS designs; our legality predicates are grounded in CUDA Driver/PTX constraints (e.g., TMA descriptor field constraints, `mbarrier` tx-count, tcgen05 collectives) and must produce *valid PTX-level artifacts* (tensor maps, barrier protocols, tcgen05 alloc/ld rules).” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))

### 4.3 The “Kill Logic” (how you survive review by these authors)

A reviewer who knows the above work will try to kill you with: “EqSat/MLIR/layouts are already done.”

You survive if you can truthfully claim (and then demonstrate with artifacts/metrics):

1. **You output a *witness* for hardware legality, not just an optimized expression.**  
   - e.g., produce `CUtensorMap` parameters satisfying hard constraints (`boxDim ≤ 256`, `elementStrides ≤ 8`, `tensorRank ≤ 5`, interleave↔swizzle coupling). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
   - and swizzle-mode constraints (128B global alignment + inner-dim table; otherwise “instruction invalid”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?utm_source=openai))  

2. **You produce a provably correct async plan under PTX semantics.**  
   - `cp.async.bulk.tensor` completion mechanism is chosen and scheduled coherently. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html))  
   - `mbarrier` tx-count is balanced via `expect_tx`/`complete_tx` discipline. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html))  

3. **On Blackwell, you enforce grain/collective legality that layout algebra misses.**  
   - `tcgen05.alloc` (nCols constraints + warp-collective `.aligned` UB). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/))  
   - `tcgen05.ld` uniform `taddr` rule and warpgroup lane partitions. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/))  

4. **You evaluate the right thing:** not just kernel speed, but “% kernels that become legal fast path” and “pipeline health” (stall reasons, eligible warps), because these are where legality/protocol systems should show wins.  
   (You already gesture at this; the proposal must commit to it with a crisp metric table and scripts.)

---

## 5. Recommended Citations (15–20 “must have”)

Below is a **BibTeX-ready** list (each item has enough fields to be translated into BibTeX; several sources even provide BibTeX directly).

### Core EqSat / e-graphs
1. **Willsey et al.** “egg: Fast and Extensible Equality Saturation.” *POPL 2021* (Proc. ACM PL). ([popl21.sigplan.org](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai))  
2. **Yang et al.** “Equality Saturation for Tensor Graph Superoptimization.” *MLSys 2021* (arXiv:2101.01332). ([mwillsey.com](https://www.mwillsey.com/papers/tensat?utm_source=openai))  
3. **Hartmann et al.** “Optimizing Tensor Computation Graphs with Equality Saturation and Monte Carlo Tree Search.” *arXiv 2024* (2410.05534), positioned for *PACT 2024*. ([arxiv.org](https://arxiv.org/abs/2410.05534?utm_source=openai))  
4. **Merckx et al.** “eqsat: An Equality Saturation Dialect for Non-destructive Rewriting.” *arXiv 2025* (2505.09363). ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
5. **Cheng et al.** “SEER: Super-Optimization Explorer for HLS using E-graph Rewriting with MLIR.” *arXiv 2023* (2308.07654). ([arxiv.org](https://arxiv.org/abs/2308.07654?utm_source=openai))  
6. **Merckx et al.** “Equality Saturation for Optimizing High-Level Julia IR.” *arXiv 2025* (2502.17075). ([arxiv.org](https://arxiv.org/abs/2502.17075?utm_source=openai))  

### Layout math (the seed + closest layout competitors)
7. **Zhou et al.** “Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using \( \mathbb{F}_2 \).” *arXiv 2025* (2505.23819). ([arxiv.org](https://arxiv.org/abs/2505.23819?utm_source=openai))  
8. **Bhaskaracharya et al.** “Modeling Layout Abstractions Using Integer Set Relations.” *arXiv 2025* (2511.10374). ([arxiv.org](https://arxiv.org/abs/2511.10374?utm_source=openai))  
9. **Carlisle et al.** “Categorical Foundations for CuTe Layouts.” *arXiv 2026* (2601.05972). ([arxiv.org](https://arxiv.org/abs/2601.05972?utm_source=openai))  

### Search / cost model baselines (what reviewers will expect)
10. **Chen et al.** “TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.” *OSDI 2018*. ([usenix.org](https://www.usenix.org/conference/osdi18/presentation/chen?utm_source=openai))  
11. **Zheng et al.** “Ansor: Generating High-Performance Tensor Programs for Deep Learning.” *OSDI 2020*. ([usenix.org](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai))  
12. **Tillet, Kung, Cox.** “Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.” *MAPL 2019 (PLDI workshop)*. ([pldi19.sigplan.org](https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations?utm_source=openai))  

### Polyhedral “why not” competitors
13. **Bondhugula et al.** “A Practical and Automatic Polyhedral Program Optimization System.” *PLDI 2008* (Pluto). ([ece.lsu.edu](https://www.ece.lsu.edu/jxr/pluto/?utm_source=openai))  
14. **Verdoolaege et al.** “Polyhedral parallel code generation for CUDA.” *ACM TACO 2013* (PPCG). ([llvm.googlesource.com](https://llvm.googlesource.com/llvm-project/%2B/refs/tags/llvmorg-11.1.0/polly/lib/External/ppcg/README?utm_source=openai))  

### Canonical LLM kernel family baseline
15. **Dao et al.** “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.” *NeurIPS 2022* (arXiv:2205.14135). ([papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html?utm_source=openai))  

### Hardware contracts (these must be cited with section numbers in your paper)
16. **NVIDIA.** *CUDA Driver API — Tensor Map Object Management* (Toolkit 13.1.1; `cuTensorMapEncodeTiled`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
17. **NVIDIA.** *CUDA C++ Programming Guide*, §10.29.3.2 “The Swizzle Modes” (global 128B alignment; inner-dimension validity → “instruction invalid”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programmex.html?utm_source=openai))  
18. **NVIDIA.** *PTX ISA 8.5*, §9.7.10.24.9 (`cp.async.bulk.tensor`) and §9.7.14.15.* (`mbarrier` tx-count model). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html))  
19. **NVIDIA.** *PTX ISA 9.1*, §9.7.16.7.1 (tcgen05 alloc/dealloc rules) and §9.7.16.8.* (TMEM access restrictions; `tcgen05.ld` uniform `taddr`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/))  
20. **Hoque et al.** “Deep Divee Hopper TMA Unit for FP8 GEMMs.” *PyTorch Blog* (2024). ([pytorch.org](https://pytorch.org/blog/hopper-tma-unit/?utm_source=openai))  

---

## Threats to Validity (what I’d ding you for if not addressed)

1. **“We’re first” risk:** EqSat-in-MLIR (Merckx’25) and e-graph superoptimization in MLIR (Cheng’23) mean you can’t claim novelty at the infrastructure level. Your claims must be *hardware-contract* novelty. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
2. **Versioning / moving-target risk:** the legality set is CUDA/PTX-version dependent; you must explicitly version the constraint model against (e.g.) CUDA Toolkit 13.1.1 driver API and PTX ISA 9.1. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
3. **Evaluation realism risk:** showing wins only on microbenchmarks is insufficient; you need evidence on real operator suites (and show **TMA-fastpath hit-rate** and **barrier stall reduction**, not only time). The PyTorch ecosystem already has TMA correctness issues in realistic compilation pipelines, which you can leverage as motivation and regression tests. ([github.com](https://github.com/pytorch/pytorch/issues/157240?utm_source=openai))  
4. **Scope creep risk for TMEM:** PTX ISA 9.1 tcgen05 rules are strict and UB-prone; a “TierGrain” prototype must show a concrete safety property (uniform `taddr`, lane partitioning respected) *and* demonstrate performance relevance. ([docs.nvidia.com](https://docs.nvidia.com/cudaallel-thread-execution/))  

---

### One last red-team note (document hygiene)
Your Stage-2 draft’s “Learn more” section includes malformed URLs (typos like `archiv13.1.0`, and broken `documentatis://`). Fix these before submission—reviewers treat broken citations as a proxy for shaky scholarship. (The correct canonical CUDA Driver API page is the one used above.) ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))

---
Learn more:
1. [CUDA\_ERROR\_INVLUE when creating tensor maps with swizzling - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/cuda-error-invalid-value-when-creating-tensor-maps-with-swizzling/350966?utm_source=openai)
2. [\[FEA\] Add cuTensorMapEncodeTiled to CudaHostAdapter · Issue #1566 · NVIDIA/cutlass · GitHub](https://github.com/NVIDIA/cutlass/issues/1566?utm_source=openai)
3. [\[BUG\] undefined symbol: cuTensorMapEncodeTiled on CUTLASS 3.5.1 · Issue #1624 · NVIDIA/cutlass · Gihttps://github.com/NVIDIA/cutlass/issues/1624?utm_source=openai)
4. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
5. [CUDA C++ Programming Guide (Legacy) — CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?utm_source=openai)
6. [PTX ISA 8.5](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html)
7. [\[FEA\]: Ampere mbarrier support for barriers with non-deflt completion function · Issue #419 · NVIDIA/cccl · GitHub](https://github.com/NVIDIA/cccl/issues/419?utm_source=openai)
8. [cp.async.bulk.tensor — CUDA Core Compute Libraries](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html?utm_source=openai)
9. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
10. [Deep Dive on the Hopper TMA Unit for FP8 GEMMs – PyTorch](https://pytorch.org/blog/hopper-tma-unit/?utm_source=o. [\[user triton\] on-device TMA + AOTI causes IMA with pytorch 2.8 branch · Issue #157240 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/157240?utm_source=openai)
12. [scaled\_mm Triton implementation causes wrong results on (at least) H100 · Issue #159940 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/159940?utm_source=openai)
13. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (MAPL 2019) - PLDI 2019](https://pldi19.n.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations?utm_source=openai)
14. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $\\mathbb{F}\_2$](https://arxiv.org/abs/2505.23819?utm_source=openai)
15. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/abs/2511.10374?utm_source=openai)
16. [Categorical Foundations for CuTe Layouts](https://arxiv.org/abs/2601.05972?utm_source=openai)
17. [Ansor: Generating High-Performance Tensor Programs for Deep Learning | USENIX](https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=openai)
18. [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning | USENIX](https://www.usenix.org/conference/osdi18/presentation/chen?utm_source=openai)
19. [PLUTO - An automatic parallelizer and locality optimizer for multicores](https://www.ece.lsu.edu/jxr/pluto/?utm_source=openai)
20. [polly/lib/External/ppcg/README - llvm-project - Git at Google](https://llvm.googlesource.com/llvm-project/%2B/refs/tags/llvmorg-11.1.0/polly/lib/External/ppcg/README?utm_source=openai)
21. [egg: Fast and Extensible Equality Saturation (POPL 2021 - POPL Research Papers) - POPL 2021](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation?utm_source=openai)
22. [mwillsey | tensat](https://www.mwillsey.com/papers/tensat?utm_source=openai)
23. [Optimizing Tensor Computation Graphs with Equality Saturation and Monte Carlo Tree Search](https://arxiv.org/abs/2410.05534?utm_source=openai)
24. [eqsat: An Equality Saturation Dialect for Non-destructive Rewriting](https://arxiv.org/abs/2505.09363?utm_source=openai)
25. [SEER: Super-Optimization Explorer for HLS using E-graph Rewriting with MLIR](https://arxiv.org/abs/2308.07654?utm_source=openai)
26. [Equality Saturation for Optimizing High-Level Julia IR](https://arxiv.org/abs/2502.17075?utm_source=openai)
27. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://papers.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html?utm_source=openai)
