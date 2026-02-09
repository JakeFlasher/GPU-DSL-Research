## WORK_PRODUCT

### 1) Table: Ground Truth Glossary

| Term | Definition (1–2 lines) | Where used | Source_ID (or UNVERIFIED) | Notes |
|---|---|---|---|---|
| CUDA Tile | Tile-based GPU programming model targeting portability for NVIDIA Tensor Cores; built on Tile IR spec/tools (incl. cuTile). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | SOTA baseline framing | N1 | NVIDIA blog states CUDA Tile was introduced in CUDA 13.1. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| Tile IR | Portable, low-level tile virtual machine + instruction set; models GPU as tile-based processor (vs PTX’s SIMT model). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html)) | Core semantics + legality | N3 | Spec includes memory model, stability guarantees, bytecode format. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| cuTile Python | Python-based DSL/programming model for writing tiled GPU kernels; aims to automatically leverage Tensor Cores and “tensor memory accelerators” while remaining portable across NVIDIA GPU architectures. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | Primary SOTA front-end | N2 | Docs release notes show 1.0.0 dated 2025-12-02. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/release_notes.html)) |
| Tile kernel (Tile IR) | Tile IR program function executed as \(N\) copies in parallel; entry point runs as parallel instances of “tile blocks.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) | Background / execution model | N3 | Unit differs from CUDA threads; mapping to hardware is compiler-handled. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) |
| Tile block (Tile IR) | Basic unit of execution: one logical tile thread operating over a multi-dimensional tile of data; hardware-thread mapping is abstracted. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) | Async/scheduling semantics | N3 | This abstraction matters for what “within a thread” means for races. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| Tensor (Tile IR) | \(n\)-D rectangular array with statically known rank/shape/element type; values are tensors or tensor views; global memory accessed via tensors. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) | Formal IR description | N3 | Pointer args are represented as rank-0 tensors (scalars) in examples. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) |
| Array (cuTile) | Mutable, global-memory tensor-like object with physical strided layout; kernel-side ops are mostly load/store between arrays and tiles. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | DSL data model | N2 | Host can pass PyTorch/CuPy objects as arrays (per docs). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) |
| Tile value (cuTile) | Immutable kernel-only value (no defined storage); dimensions must be compile-time constants and powers of two. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | DSL constraints + feasibility | N2 | Power-of-two restriction echoes layout-formalism constraints in seed papers. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |
| Scope (Tile IR memory model) | Memory operations can be scoped (e.g., `tile_block`, `device`, `sys`) or `weak`; non-weak scopes require an ordering. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Correctness / sync modeling | N3 | `weak` has strong “no concurrent comms” assumptions. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| Weak (Tile IR) | Operations without a scope are `weak`; weak ops cannot communicate between threads or unordered fragments; compiler may assume no concurrent access. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Legality constraints | N3 | Load-bearing for any “async overlap” story in Tile IR. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| Token (Tile IR) | Abstract value used to build dependencies between memory ops within the same tile-block thread; no runtime representation; cannot be stored/loaded/compared. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Async / dependency graph | N3 | Used by “token ordered operations” family. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| Token order (Tile IR) | Tile IR does **not** use program/data/control/address dependencies to order memory ops; tokens must be used, even if ordering seems redundant. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Central correctness invariant | N3 | Toolchain may optimize away apparent program deps; token deps are preserved. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| Memory ordering | Tile IR defines orderings incl. `weak`, `relaxed`, `release`, `acquire`, `acq_rel`; ordering + scope determine synchronization meaning. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Background + evaluation correctness | N3 | cuTile atomics expose a similar acquire/release vocabulary. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)) |
| cuTile `MemoryOrder` / `MemoryScope` | cuTile defines atomic memory order (relaxed/acquire/release/acq_rel) and scope (block/device/sys); without explicit sync, cross-thread ordering is not guaranteed. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)) | DSL-level sync semantics | N2 | cuTile docs point to Tile IR memory model for more detail. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)) |
| Triton-to-TileIR backend | Compiler bridge enabling Triton kernels to target CUDA Tile IR instead of PTX; preserves tile-level semantics by compiling “directly to CUDA Tile IR.” ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | SOTA compilation pipeline | N4 | Blog describes switching via env var + caching `.tileIR` artifacts. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| Tensor-of-pointer pattern (Triton) | Triton pattern that materializes tensors of pointers for access patterns; NVIDIA blog reports suboptimal performance on Tile IR backend (CUDA 13.1) and suggests descriptor-based alternatives. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Known limitation / workload selection | N4 | Proposed mitigation: use TMA load/store API w/ shape/strides/block_shape descriptors. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| Linear layouts (Triton) | Layout model using linear algebra over \( \mathbb{F}_2 \) (binary matrices) to represent/convert tensor layouts; integrated into Triton for codegen and robustness. ([arxiv.org](https://arxiv.org/abs/2505.23819)) | Seed baseline for layout reasoning | P1 | Paper notes power-of-two shape restriction as a primary limitation. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |
| CuTe layout | Layout abstraction mapping \(n\)-D logical coordinates to 1-D indices via shape/stride tuples (and supports swizzle-like bit manipulations). ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | Seed baseline for NVIDIA layout algebra | P2 | Used in NVIDIA ecosystem (e.g., CUTLASS) per seed papers. ([arxiv.org](https://arxiv.org/abs/2601.05972)) |
| ISL integer set relations | Unified mathematical representation using the Integer Set Library (ISL) to model and analyze both CuTe layouts and Triton linear layouts via integer set relations. ([arxiv.org](https://arxiv.org/abs/2511.10374)) | Seed baseline for verification/formalization | P2 | Pitch includes “formal analysis” and “correctness verification.” ([arxiv.org](https://arxiv.org/abs/2511.10374)) |

---

### 2) Table: SOTA Baseline Map (N1..N6, C1 optional)

| Source_ID | System | Abstraction Level | Key Semantics/Guarantees | Constraints/Assumptions | What It Solves | What It Does NOT Solve | Evidence (cite or UNVERIFIED) |
|---|---|---|---|---|---|---|---|
| N1 | CUDA Tile (concept page) | Programming model overview | Tile-based programming model targeting portability for NVIDIA Tensor Cores; positioned as simplifying optimized tiled kernels; based on Tile IR + cuTile. | Targeting NVIDIA platforms; details delegated to Tile IR spec + cuTile docs. | High-level framing + entry point to stack components (Tile IR, cuTile). | Does not define formal semantics/memory model; does not specify operation set or legality rules. | ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) |
| N2 | cuTile Python docs | User-facing Python DSL + execution/data/memory model | Python DSL for kernels; tiles vs arrays model; claims automatic leverage of Tensor Cores + tensor memory accelerators and portability across NVIDIA GPU architectures without code changes. | Tiles: immutable, kernel-only; tile dims compile-time constants and powers of two; memory model requires explicit synchronization for cross-thread ordering. | Practical authoring + explanation of DSL semantics + tuning/interop/debug hooks (docs). | Does not itself define Tile IR bytecode/VM semantics; defers detailed memory-model explanation to Tile IR spec. | ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) |
| N3 | Tile IR spec | Low-level virtual machine + instruction set + memory model + stability guarantees | Defines Tile IR as portable low-level tile VM/ISA; provides token-based ordering model; supplies stability/compatibility guarantees for bytecode and spec versions. | Token deps required for ordering (program deps don’t order); weak ops assume no concurrent comms; known issues in spec 13.1 docs; numerical stability limits across toolchains/targets. | Paper-grade formal anchor for legality (token/scopes/orderings), portability claims, and what “correct” means for transformations. | Known gaps: missing detailed memory-model examples, incomplete per-op bytecode encodings, limited atomics, missing cross-tile-block example. | ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html)) |
| N4 | NVIDIA blog: Triton-to-TileIR backend | Integration + roadmap + limitations (public narrative) | Triton-to-TileIR compiles Triton to CUDA Tile IR (instead of PTX), preserving tile-level semantics; presents roadmap/testing/benchmarking framing. | Currently: source-based compilation only; CUDA 13.1+; Blackwell GPUs; incomplete op support; tensor-of-pointer pattern can be slow; mitigations suggested (TMA / fallback). | Establishes a concrete SOTA baseline path (Triton → Tile IR) + explicit current limitations useful for workload selection and feasibility constraints. | Does not provide full op-support matrix in the blog text; early-stage backend implies moving target for evaluation. | ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| N5 | NVIDIA/cutile-python (repo) | Implementation + build/packaging | cuTile Python is open-source; build-from-source instructions and dependencies; includes C++ extension; CUDA Toolkit 13.1+ requirement; Apache 2.0 license noted. | Build requires modern toolchain (C++17, CMake, Python 3.10+); depends on CUDA Toolkit 13.1+. | Feasibility anchor: we can build/inspect/patch; enables measurement + prototyping. | Repo text does not replace spec for formal semantics; runtime/compiler internals may still be partly opaque. | ([github.com](https://github.com/NVIDIA/cutile-python)) |
| N6 | NVIDIA/cuda-tile (repo) | MLIR dialect + compiler infrastructure + bytecode tooling | States CUDA Tile IR is MLIR-based IR + compiler infra; aligned with CUDA Toolkit 13.1; provides dialect + Python bindings + bytecode serialization + conformance tests; describes producing/loading Tile IR bytecode + `tileiras` AOT option. | Requires MLIR/LLVM at a compatible commit; build complexity; toolchain coupling. | Feasibility + tooling: enables IR-level transformations, conformance testing, and compiler integration experiments. | Does not provide the full “user DSL” experience alone; does not guarantee backend availability for all GPUs/toolchains. | ([github.com](https://github.com/NVIDIA/cuda-tile)) |
| C1 (opt) | maknee TileIR internals blog | Secondary reverse-engineering narrative | Walkthrough of compilation pipeline (CuTile → MLIR/LLVM → SASS) and notes some details are undocumented and may change; based on CUDA 13.1. | Not authoritative; explicitly warns about undocumented/mutable details. | Useful orientation for “what to look at next” (passes/tools/env vars) when diving into the stack. | Cannot be treated as ground-truth semantics; may drift quickly as toolchain evolves. | ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)) |

---

### 3) Table: Claim Ledger v0 (10–25 claims; stable IDs)

**Claim_ID prefixes:** `C-P*` = seed-paper claims; `C-N*` = primary-source stack/platform claims; `C-I*` = **[INFERENCE]**.

| Claim_ID | Claim (1 sentence) | Status | Evidence pointers | Paper role | Risk if wrong |
|---|---|---|---|---|---|
| C-P1-01 | Linear Layouts models tensor layouts using linear algebra over \( \mathbb{F}_2 \). | VERIFIED | P1(arXiv abs): abstract/title. ([arxiv.org](https://arxiv.org/abs/2505.23819)) | Background + closest seed prior art for layout formalization | Novelty baseline becomes wrong / misframed |
| C-P1-02 | Linear Layouts represents layouts as binary matrices acting on bits of the hardware representation, enabling generic layout definitions and generic layout-to-layout conversions (avoiding “quadratic explosion”). | VERIFIED | P1(arXiv abs): abstract. ([arxiv.org](https://arxiv.org/abs/2505.23819)) | Motivation for “generic conversion” as valuable | Overclaiming benefit vs existing systems |
| C-P1-03 | Linear Layouts is integrated into Triton and includes a layout engine that can automatically choose and propagate layouts for Triton operations. | VERIFIED | P1(HTML): overview bullets on layout engine integration. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Baseline comparator: compiler-integrated layout reasoning | Misstating SOTA auto-layout capability |
| C-P1-04 | Linear Layouts reports reduced backend engineering effort and bug fixes in Triton’s legacy layout system (and evaluates vs legacy Triton). | VERIFIED | P1(arXiv abs): “reduce engineering effort…fixing…bugs”; P1(HTML): eval framing. ([arxiv.org](https://arxiv.org/abs/2505.23819)) | Problem reality + value proposition anchor | Weakens motivation if overstated |
| C-P1-05 | Linear Layouts claims 12% of bugs filed in Triton’s GitHub repository are layout-related. | VERIFIED | P1(HTML): intro statistic. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Motivation: layout is a real source of failures | If wrong, credibility hit (easy-to-check stat) |
| C-P1-06 | Linear Layouts identifies a primary limitation: restriction to power-of-two shapes (mitigated by masking out-of-boundary elements). | VERIFIED | P1(HTML): conclusions/limitations. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Scope constraints for any layout approach we propose | We design for non-existent constraint |
| C-P2-01 | The ISL paper proposes using integer set relations (ISL) to unify formal analysis for both CuTe layouts and Triton linear layouts, enabling correctness verification and cross-system reasoning. | VERIFIED | P2(arXiv abs): abstract. ([arxiv.org](https://arxiv.org/abs/2511.10374)) | Seed baseline for “verification via common formalism” | Misidentifying what ISL model can prove |
| C-P2-02 | The ISL paper models CuTe layouts as stride-based coordinate→index relations with swizzle bit manipulations and models Triton linear layouts as relations over \(F_2\)-style binary vector-space transforms. | VERIFIED | P2(arXiv abs): modeling details; P2(HTML): CuTe/linear layout definitions. ([arxiv.org](https://arxiv.org/abs/2511.10374)) | Technical grounding for “layout unification” | Wrong modeling details → wrong direction choice |
| C-P2-03 | The ISL paper implements layout manipulation algorithms (composition, inversion, complement) using ISL operations to preserve semantics. | VERIFIED | P2(arXiv abs): implementation claim. ([arxiv.org](https://arxiv.org/abs/2511.10374)) | Candidate baseline for “layout equivalence / transforms” | Overestimating tool availability/coverage |
| C-P3-01 | Categorical Foundations presents a categorical framework for a tractable class of CuTe layouts by defining categories Tuple and Nest whose morphisms give rise to layouts. | VERIFIED | P3(arXiv abs): abstract. ([arxiv.org](https://arxiv.org/abs/2601.05972)) | Seed baseline for “layout algebra foundations” | Misrepresenting paper’s scope (tractable class) |
| C-P3-02 | Categorical Foundations provides a Python implementation with tests that demonstrate alignment with CUTLASS behavior. | VERIFIED | P3(arXiv abs): implementation/tests claim. ([arxiv.org](https://arxiv.org/abs/2601.05972)) | Possible verification harness inspiration | If wrong, can’t reuse as validation scaffold |
| C-N1-01 | NVIDIA CUDA Tile is a tile-based GPU programming model targeting portability for NVIDIA Tensor Cores and is based on Tile IR and tools including cuTile Python. | VERIFIED | N1(dev page): CUDA Tile description + “based on Tile IR…including cuTile.” ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | SOTA baseline definition (what exists today) | Wrong baseline → novelty/positioning fails |
| C-N2-01 | cuTile is a parallel programming model and Python DSL that claims automatic use of advanced hardware (Tensor Cores + tensor memory accelerators) and portability across NVIDIA GPU architectures without code changes. | VERIFIED | N2(docs landing): cuTile description. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | SOTA front-end capability baseline | Overclaiming portability/feature coverage |
| C-N2-02 | In cuTile, arrays are mutable global-memory objects with strided layouts, while tiles are immutable kernel-only values whose dimensions are compile-time constants and powers of two. | VERIFIED | N2(docs landing): arrays vs tiles + power-of-two constraint. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | Hard feasibility constraints on any DSL/analysis | If wrong, we may overconstrain MVP |
| C-N2-03 | cuTile’s memory model permits compiler/hardware reordering and exposes atomic memory order and scope, with synchronization occurring per element. | VERIFIED | N2(memory model doc): reordering + per-element + order/scope. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)) | Background; aligns stack semantics with Tile IR | Wrong assumptions → incorrect legality tests |
| C-N3-01 | Tile IR is specified as a portable, low-level tile virtual machine and instruction set that models the GPU as a tile-based processor (unlike PTX’s SIMT model). | VERIFIED | N3(intro): definition + PTX contrast. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html)) | Core SOTA anchor for IR semantics | Misstating IR model breaks framing |
| C-N3-02 | Tile IR tile kernels execute as parallel instances of tile blocks, and the mapping of logical tile threads to hardware threads is abstracted and compiler-handled. | VERIFIED | N3(programming model): tile kernels + mapping abstraction. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) | Execution model assumptions for async scheduling | Wrong unit-of-execution assumptions |
| C-N3-03 | Tile IR’s memory model is derived from PTX and states weak operations cannot be used to communicate between threads (and are unsafe for unordered same-thread overlaps). | VERIFIED | N3(memory model): derived from PTX + weak communication restriction + hazards. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Legality constraints for transformations | Unsoundness if ignored |
| C-N3-04 | Tile IR requires explicit token dependencies for ordering: tokens have no runtime representation, program dependencies do not order memory ops, and token dependencies are preserved by the toolchain. | VERIFIED | N3(memory model): tokens + “program deps do not provide ordering” + toolchain may remove program deps. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Central “async correctness” primitive | Core idea collapses if false |
| C-N3-05 | Tile IR provides platform/compatibility guarantees including bytecode stability across conforming drivers and syntactic portability for programs conforming to spec vX.Y onto platforms supporting vX.Y+. | VERIFIED | N3(stability): bytecode stability + portability language. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html)) | Feasibility + evaluation portability story | Overpromising portability |
| C-N3-06 | Tile IR spec 13.1 release notes dated 2026-01-23 list known issues including missing cross-tile-block kernel examples, incomplete per-op bytecode encodings, limited atomics, and insufficient memory-model examples. | VERIFIED | N3(release notes): known issues + “Spec 13.1 (2026-01-23)”. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/release_notes.html)) | Risk register + “what’s not solved” baseline | If wrong, we may misjudge feasibility |
| C-N4-01 | Triton-to-TileIR backend compiles Triton kernels to CUDA Tile IR (instead of PTX) and currently requires source-based compilation with prerequisites including CUDA 13.1+ and Blackwell GPUs. | VERIFIED | N4(blog): backend description + prereqs + “source-based compilation only.” ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Feasibility + evaluation environment constraints | If wrong, hardware/tooling plan breaks |
| C-N4-02 | Triton-to-TileIR has known limitations (unsupported ops; tensor-of-pointer performance degradation) with suggested mitigations (fallback to SIMT; adopt TMA load/store API). | VERIFIED | N4(blog): limitations + mitigations + TMA descriptor sketch. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Workload selection + constraints for benchmarks | Evaluation invalid if ignored |
| C-N6-01 | The `cuda-tile` repo describes an open-source CUDA Tile IR ecosystem (MLIR dialect, Python bindings, bytecode serialization, conformance tests) aligned with CUDA Toolkit 13.1 and describes producing/loading Tile IR bytecode (incl. `tileiras`). | VERIFIED | N6(repo README): components + CTK 13.1 alignment + bytecode toolchain narrative. ([github.com](https://github.com/NVIDIA/cuda-tile)) | Feasibility anchor for building passes + tests | Wrong tooling assumptions derail MVP |
| C-I-01 | [INFERENCE] Any optimization that overlaps or reorders Tile IR memory operations must explicitly preserve/provide correct token dependencies (not rely on apparent program dependencies), or it can create data races. | UNVERIFIED | Inference from N3 token-order rules. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Core “legality checker / scheduler” design constraint | If false, proposed mechanism may be overcomplicated |
| C-I-02 | [INFERENCE] A promising research gap is to connect formal layout reasoning (P1/P2/P3) with Tile IR’s token/scope legality to build analyzable async overlap + diagnostic tooling for CUDA Tile stacks. | UNVERIFIED | Inference from seed papers + Tile IR token model. ([arxiv.org](https://arxiv.org/abs/2505.23819)) | Candidate thesis direction (Stage 2) | If already solved, novelty collapses |

---

### 4) Golden Snapshot (Carry-Forward)

#### P1 — Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using \( \mathbb{F}_2 \)
- **What it is:** arXiv paper (v3 revised 2025-10-22); also appears with an ASPLOS ’26 journal/conference reference in the HTML rendering. ([arxiv.org](https://arxiv.org/abs/2505.23819))  
- **What it guarantees/claims:** Models layouts via linear algebra over \( \mathbb{F}_2 \), uses binary matrices, enables generic conversions, integrates into Triton, improves robustness/engineering effort. ([arxiv.org](https://arxiv.org/abs/2505.23819))  
- **What it does NOT say:** It is not a CUDA Tile / Tile IR spec; it does not claim to define NVIDIA’s token/scope memory legality. ([arxiv.org](https://arxiv.org/abs/2505.23819))  
- **Why we care:** Strong seed baseline for “layout formalism + compiler integration,” plus explicit limitation (power-of-two shapes) that likely constrains any layout-centric proposal. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

#### P2 — Modeling Layout Abstractions Using Integer Set Relations
- **What it is:** arXiv preprint (submitted 2025-11-13) proposing ISL-based modeling for layout systems. ([arxiv.org](https://arxiv.org/abs/2511.10374))  
- **What it guarantees/claims:** Unified representation of CuTe + Triton linear layouts via integer set relations, enabling formal analysis + correctness verification; models swizzles + \(F_2\) linear-layout math; implements composition/inversion/complement. ([arxiv.org](https://arxiv.org/abs/2511.10374))  
- **What it does NOT say:** It does not claim integration with CUDA Tile / Tile IR toolchains; it’s a mathematical modeling/analysis layer rather than an NVIDIA stack spec. ([arxiv.org](https://arxiv.org/abs/2511.10374))  
- **Why we care:** Candidate foundation for a “layout legality/equivalence” engine, potentially reusable for proving transformations correct (or generating counterexamples). ([arxiv.org](https://arxiv.org/abs/2511.10374))  

#### P3 — Categorical Foundations for CuTe Layouts
- **What it is:** arXiv preprint (submitted 2026-01-09; 174 pages) focusing on categorical foundations for CuTe layout algebra. ([arxiv.org](https://arxiv.org/abs/2601.05972))  
- **What it guarantees/claims:** Defines categories Tuple and Nest; proves compatibility of categorical operations with layout operations; provides Python implementation + tests aligned with CUTLASS behavior. ([arxiv.org](https://arxiv.org/abs/2601.05972))  
- **What it does NOT say:** Does not claim to address Tile IR token ordering / memory scope semantics; not an async/scheduling paper. ([arxiv.org](https://arxiv.org/abs/2601.05972))  
- **Why we care:** Another (more abstract) layout-algebra baseline; potentially useful as an independently-developed “oracle” for layout operations in evaluation. ([arxiv.org](https://arxiv.org/abs/2601.05972))  

#### N1 — CUDA Tile (concept)
- **What it is:** NVIDIA Developer overview page positioning CUDA Tile as tile-based programming model for Tensor Core portability/performance. ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- **What it guarantees/claims:** CUDA Tile is based on Tile IR spec/tools; cuTile is user-facing language support (Python now; future C++ mentioned). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- **What it does NOT say:** Does not provide formal semantics/memory model; does not specify supported ops/targets in detail. ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- **Why we care:** Sets what “SOTA baseline” is for our proposal family and names the authoritative downstream specs/docs. ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  

#### N2 — cuTile Python docs
- **What it is:** Official NVIDIA documentation for cuTile Python DSL (execution model, data model, memory model, ops, debugging). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python))  
- **What it guarantees/claims:** Defines arrays vs tiles semantics; tiles are immutable + power-of-two compile-time dims; provides atomic memory order/scope; positions portability + leveraging hardware (Tensor Cores + tensor memory accelerators). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python))  
- **What it does NOT say:** Does not fully replace Tile IR spec for low-level ordering/token semantics; points to Tile IR for more detailed memory model discussion. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html))  
- **Why we care:** The user-facing programming contract + constraints we must respect; likely where evaluation kernels will be authored. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python))  

#### N3 — Tile IR spec
- **What it is:** NVIDIA specification for Tile IR (intro, type system, memory model, operations, stability, release notes). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html))  
- **What it guarantees/claims:** Defines token-based ordering (program deps don’t order); defines scopes/orderings; offers stability/compatibility guarantees for bytecode and spec versions. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_mohtml))  
- **What it does NOT say:** Release notes acknowledge gaps: missing cross-tile-block example section, incomplete per-op bytecode encodings, limited atomics, lack of detailed memory-model examples. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/release_notes.html))  
- **Why we care:** Primary ground truth for legality of async overlap/reordering and for what transformations are allowed. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  

#### N4 — NVIDIA blog: Triton-to-TileIR backend
- **What it is:** NVIDIA technical blog post (dated 2026-01-30) describing the Triton-to-TileIR backend and its roadmap/limitations. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- **What it guarantees/claims:** Triton can target CUDA Tile IR instead of PTX, preserving tile semantics; suggests early-stage roadmap and benchmarking vs PTX. ([developer.nvidia.com](tps://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- **What it does NOT say:** Blog doesn’t enumerate full support matrix in-text; states known limitations and that project is early-stage. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- **Why we care:** Concrete feasibility constraints for evaluation (CUDA 13.1+, Blackwell, source build) and known pitfalls  avoid (tensor-of-pointer). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

#### N5 — NVIDIA/cutile-python (repo)
- **What it is:** Open-source repository for cuTile Python; includes build instructions and licensing notes. ([github.com](https://github.com/NVIDIA/cutile-python))  
- **What it guarantees/claims:** Build requires CUDA Toolkit 13.1+ and a C++17/CMake/Python toolchain; licensed Apache 2.0. ([github.com](hps://github.com/NVIDIA/cutile-python))  
- **What it does NOT say:** Repo README does not by itself serve as the formal semantic spec for Tile IR or memory legality. ([github.com](https://github.com/NVIDIA/cutile-python))  
- **Why we care:** Ensures “MVP implementability” path exists (we can build/modify/measure). ([github.com](https://github.com/NVIDIA/cutile-python))  

#### N6 — NVIDIA/cuda-tile (repo)
- **What it is:** Open-source CUDA Tile IR dialect + tooling repo (MLIR-based), aligned with CUDkit 13.1. ([github.com](https://github.com/NVIDIA/cuda-tile))  
- **What it guarantees/claims:** Lists core components: dialect, Python bindings, bytecode serializer, conformance tests; documents build + integration and Tile IR bytecode tooling (`cuda-tile-translate`, `tileiras`). ([github.com](https://github.com/NVIDIA/cuda-tile))  
- **What it does NOT say:** Does not promise that all targets/hardware are supported equally; depends on compatible LLVM/MLIR/toolchain. ([github.com](https://github.com/NVIDIA/cuda-tile))  
- **Why we care:** Likely the most direct hook point for compiler passes and evaluation harnesses tied to conformance tests. ([github.com](https://github.com/NVIDIA/cuda-tile))  

#### C1 — maknee TileIR internals blog (secondary)
- **What it is:** Third-party blog post (“NVIDIA TileIR Internals…”) dated 2026-01-30, describing compilation pipeline and passes; explicitly notes undocumented details may change. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-InternaluTile-to-MLIR-LLVM-to-SASS/))  
- **What it guarantees/claims:** Orientation-level pipeline narrative (CuTile → various stages → SASS); warns about instability/undocumented behavior. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/))  
- **What it does NOT say:** Not a specification and not authoritative for semantics or guarantees. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SA)  
- **Why we care:** Useful debugging/introspection hints later, but must be treated as non-ground-truth. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/))  

---

### 5) Constraint Cliffs (max 7)

- If Tile IR truly requires **token dependencies** for memory ordering (and program deps don’t order), then our proposal **must** model/preserve token order; otherwise transformations can be unsound. ([docs.nvidia.com](https://docs.nvidia.com/cu/tile-ir/latest/sections/memory_model.html))  
- If `weak` memory ops cannot be used for inter-thread communication and assume no concurrent access, then any “async overlap” story must **either** avoid `weak` **or** correctly scope/order operations (not rely on weak comms). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
- If cuTile tile dimensions must be **compile-time constants and powers of two**, then any layout/autotiling approach must either conform oovide masking/packing strategies; otherwise it won’t compile/run. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python))  
- If Triton-to-TileIR currently requires **CUDA 13.1+ and Blackwell GPUs** and is source-build only, then evaluation must secure that environment (or choose baselines not requiring it), or the proposal can’t be empirically validated. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- If  IR spec 13.1 has **known documentation/spec gaps** (memory-model examples lacking; limited atomics; incomplete bytecode encodings), then MVP scope must avoid depending on underspecified features (or explicitly treat them as risks). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/release_notes.html))  
- If Tile IR does **not** guarantee bit-identical numerical results across toolchains/targets, then evaluation must include numeric-tolerance correctness checks (not strict equality across versions/targets). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html))  
- If tensor-of-pointer patterns are currently slow on Tile IR backend and TMA descriptors are a recommended mitigation, then workload selection and baselines must either avoid tensor-of-pointer or incorporate the descriptor rewrite; otherwise perf conclusions can be misleading. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

---

## CONTEXT_CAPSULE

```yaml
STATE_BEGIN
profile:
  project_name: "GPU Layout + Async Research Program"
  target_model: "gpt-5.2-pro (web UI)"
  operating_mode: "manual_state_v3 (web UI; milestone resets)"
  stage_plan: ["0", "1", "1.5", "2", "2.5", "3"]
  conference_targets: ["ISCA", "MICRO", "ASPLOS"]
  paper_genre: "research proposal"
  paper_acceptance_bar:
    - "Explicit novelty delta vs SOTA (named neighbors)"
    - "Feasibility scoped to an MVP w/ risks + mitigations"
    - "Evaluation plan has baselines + ablations + non-speed metrics"
    - "Claims mapped to evidence or explicit TODO verification steps"
  hard_constraints:
    - "No hallucinated citations"
    - "No invented technical specifics"
    - "Follow stage schemas exactly"
    - "Golden sources always present in STATE"
    - "Memory precedence: latest user > session > global; global is advisory"
  current_stage: 0
  last_updated: "2026-02-04"
  problem_statement_1liner: null
  current_best_thesis: null
  success_criteria:
    - "Reviewer can name closest prior art and see the delta in 30 seconds"
    - "Feasibility is believable without 'handwaving alias analysis / hardware assumptions'"
    - "Evaluation plan could be executed by a grad student in one semester"

GOLDEN_SOURCES:
  - id: "P1"
    kind: "seed_paper"
    title: "Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using \\mathbb{F}_2"
    url: "https://arxiv.org/html/2505.23819v3"
    last_verified: "2026-02-04"
  - id: "P2"
    kind: "seed_paper"
    title: "Modeling Layout Abstractions Using Integer Set Relations"
    url: "https://arxiv.org/html/2511.10374v1"
    last_verified: "2026-02-04"
  - id: "P3"
    kind: "seed_paper"
    title: "Categorical Foundations for CuTe Layouts"
    url: "https://arxiv.org/pdf/2601.05972v1"
    last_verified: "2026-02-04"

  - id: "N1"
    kind: "nvidia_primary"
    title: "CUDA Tile (concept)"
    url: "https://developer.nvidia.com/cuda/tile"
    last_verified: "2026-02-04"
  - id: "N2"
    kind: "nvidia_primary"
    title: "cuTile Python docs"
    url: "https://docs.nvidia.com/cuda/cutile-python"
    last_verified: "2026-02-04"
  - id: "N3"
    kind: "nvidia_primary"
    title: "Tile IR spec"
    url: "https://docs.nvidia.com/cuda/tile-ir/latest/"
    last_verified: "2026-02-04"
  - id: "N4"
    kind: "nvidia_primary"
    title: "Triton-to-TileIR backend (NVIDIA blog)"
    url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
    last_verified: "2026-02-04"
  - id: "N5"
    kind: "repo"
    title: "cuTile Python repo"
    url: "https://github.com/NVIDIA/cutile-python"
    last_verified: "2026-02-04"
  - id: "N6"
    kind: "repo"
    title: "cuda-tile repo"
    url: "https://github.com/NVIDIA/cuda-tile"
    last_verified: "2026-02-04"
  - id: "C1"
    kind: "community_secondary"
    title: "TileIR internals blog (secondary; not authoritative)"
    url: "https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/"
    last_verified: "2026-02-04"

GLOBAL_MEMORY:
  notes:
    - id: "GM-format"
      text: "Default deliverables: WORK_PRODUCT then CONTEXT_CAPSULE; Stage 3 overrides (LaTeX + capsule in comments)."
      last_update_date: "2026-02-04"
      keywords: ["format", "workflow"]
    - id: "GM-evidence"
      text: "Paper-grade claims require Claim_IDs + evidence pointers; unknowns must be UNVERIFIED + moved to OPEN_QUESTIONS."
      last_update_date: "2026-02-04"
      keywords: ["evidence", "claims"]

SESSION_MEMORY:
  notes:
    - id: "SM-0"
      text: "Stage 0 completed: glossary + baseline map + Claim Ledger v0 + golden snapshots + constraint cliffs created from P1..P3 and NVIDIA Tile stack sources."
      last_update_date: "2026-02-04"

VERDICT_LEDGER:
  items: []

CLAIM_LEDGER:
  items:
    - id: "C-P1-01"
      claim: "Linear Layouts models tensor layouts using linear algebra over F2."
      status: "VERIFIED"
      evidence: ["P1:arXiv_abs:abstract/title"]
      paper_section: "Background/Prior Art"
      delta_relevance: "seed baseline for layout formalization"
      last_update_date: "2026-02-04"
    - id: "C-P1-02"
      claim: "Linear Layouts represents layouts as binary matrices acting on bits, enabling generic layout definitions and conversions."
      status: "VERIFIED"
      evidence: ["P1:arXiv_abs:abstract"]
      paper_section: "Motivation"
      delta_relevance: "why generic conversion matters"
      last_update_date: "2026-02-04"
    - id: "C-P1-03"
      claim: "Linear Layouts is integrated into Triton and includes a layout engine that can automatically choose and propagate layouts."
      status: "VERIFIED"
      evidence: ["P1:html:overview bullets (layout engine integration)"]
      paper_section: "Related Work (Triton)"
      delta_relevance: "closest neighbor for compiler-integrated layout reasoning"
      last_update_date: "2026-02-04"
    - id: "C-P1-04"
      claim: "Linear Layouts reports reduced engineering effort and bug fixes in Triton's legacy layout system."
      status: "VERIFIED"
      evidence: ["P1:arXiv_abs:abstract"]
      paper_section: "Motivation"
      delta_relevance: "layout correctness/robustness is real pain point"
      last_update_date: "2026-02-04"
    - id: "C-P1-05"
      claim: "Linear Layouts claims 12% of Triton GitHub bugs are layout-related."
      status: "VERIFIED"
      evidence: ["P1:html:introduction (bug stat)"]
      paper_section: "Motivation"
      delta_relevance: "problem reality evidence"
      last_update_date: "2026-02-04"
    - id: "C-P1-06"
      claim: "Linear Layouts notes a primary limitation: restriction to power-of-two shapes (mitigable via masking)."
      status: "VERIFIED"
      evidence: ["P1:html:conclusion/limitations"]
      paper_section: "Limitations/Scope"
      delta_relevance: "constraints likely shared by tile systems"
      last_update_date: "2026-02-04"

    - id: "C-P2-01"
      claim: "ISL integer set relations are proposed as a unified representation for CuTe layouts and Triton linear layouts to enable formal analysis and correctness verification."
      status: "VERIFIED"
      evidence: ["P2:arXiv_abs:abstract"]
      paper_section: "Background/Prior Art"
      delta_relevance: "seed baseline for verification approach"
      last_update_date: "2026-02-04"
    - id: "C-P2-02"
      claim: "The ISL paper models CuTe stride+swizzle and Triton F2-style transforms via integer set relations."
      status: "VERIFIED"
      evidence: ["P2:arXiv_abs:abstract", "P2:html:intro definitions"]
      paper_section: "Background"
      delta_relevance: "ties layout systems under one math umbrella"
      last_update_date: "2026-02-04"
    - id: "C-P2-03"
      claim: "The ISL paper implements layout manipulation algorithms (composition/inversion/complement) using ISL operations to preserve semantics."
      status: "VERIFIED"
      evidence: ["P2:arXiv_abs:abstract"]
      paper_section: "Methods baseline"
      delta_relevance: "possible baseline for transform correctness"
      last_update_date: "2026-02-04"

    - id: "C-P3-01"
      claim: "Categorical Foundations defines categories Tuple and Nest to model a tractable class of CuTe layouts."
      status: "VERIFIED"
      evidence: ["P3:arXiv_abs:abstract"]
      paper_section: "Background/Prior Art"
      delta_relevance: "alternative formal foundation for CuTe algebra"
      last_update_date: "2026-02-04"
    - id: "C-P3-02"
      claim: "Categorical Foundations provides a Python implementation with tests aligned to CUTLASS behavior."
      status: "VERIFIED"
      evidence: ["P3:arXiv_abs:abstract"]
      paper_section: "Evaluation harness ideas"
      delta_relevance: "potential independent oracle for layout ops"
      last_update_date: "2026-02-04"

    - id: "C-N1-01"
      claim: "CUDA Tile is a tile-based GPU programming model targeting portability for NVIDIA Tensor Cores and is based on Tile IR and tools including cuTile Python."
      status: "VERIFIED"
      evidence: ["N1:CUDA Tile concept page"]
      paper_section: "SOTA Baseline"
      delta_relevance: "names what exists today in NVIDIA stack"
      last_update_date: "2026-02-04"
    - id: "C-N2-01"
      claim: "cuTile is a Python-based DSL/programming model that claims portability across NVIDIA GPU architectures and automatic leveraging of tensor cores and tensor memory accelerators."
      status: "VERIFIED"
      evidence: ["N2:cuTile docs landing"]
      paper_section: "SOTA Baseline"
      delta_relevance: "front-end baseline"
      last_update_date: "2026-02-04"
    - id: "C-N2-02"
      claim: "cuTile arrays are mutable global-memory objects with strided layouts; tiles are immutable kernel-only values with power-of-two compile-time dimensions."
      status: "VERIFIED"
      evidence: ["N2:cuTile docs (arrays vs tiles)"]
      paper_section: "Constraints/Model"
      delta_relevance: "hard constraints that affect MVP design"
      last_update_date: "2026-02-04"
    - id: "C-N2-03"
      claim: "cuTile memory model permits reordering and defines atomic memory order/scope with per-element synchronization."
      status: "VERIFIED"
      evidence: ["N2:memory_model"]
      paper_section: "Background (sync)"
      delta_relevance: "ties user-level atomic semantics to Tile IR model"
      last_update_date: "2026-02-04"

    - id: "C-N3-01"
      claim: "Tile IR is a portable, low-level tile virtual machine and instruction set that models GPUs as tile-based processors (unlike PTX SIMT)."
      status: "VERIFIED"
      evidence: ["N3:introduction"]
      paper_section: "Background"
      delta_relevance: "formal semantics anchor"
      last_update_date: "2026-02-04"
    - id: "C-N3-02"
      claim: "Tile IR tile kernels execute as parallel instances of tile blocks, and mapping to hardware threads is compiler-handled."
      status: "VERIFIED"
      evidence: ["N3:programming_model"]
      paper_section: "Background"
      delta_relevance: "execution-model assumptions"
      last_update_date: "2026-02-04"
    - id: "C-N3-03"
      claim: "Tile IR weak memory operations cannot be used to communicate between threads; compiler may assume weak tiles are not concurrently accessed."
      status: "VERIFIED"
      evidence: ["N3:memory_model:scopes"]
      paper_section: "Legality"
      delta_relevance: "critical legality constraint for async/reordering"
      last_update_date: "2026-02-04"
    - id: "C-N3-04"
      claim: "Tile IR uses tokens/token order for memory ordering; program dependencies do not order memory ops and may be optimized away, while token dependencies are preserved."
      status: "VERIFIED"
      evidence: ["N3:memory_model:tokens"]
      paper_section: "Legality/Core Idea"
      delta_relevance: "makes dependency modeling explicit"
      last_update_date: "2026-02-04"
    - id: "C-N3-05"
      claim: "Tile IR stability guarantees include bytecode stability and syntactic portability for programs conforming to spec vX.Y to platforms supporting vX.Y+."
      status: "VERIFIED"
      evidence: ["N3:stability"]
      paper_section: "Feasibility"
      delta_relevance: "portability story constraints"
      last_update_date: "2026-02-04"
    - id: "C-N3-06"
      claim: "Tile IR release notes for Spec 13.1 (2026-01-23) list known issues: missing cross-tile-block example, incomplete bytecode encodings, limited atomics, and missing detailed memory-model examples."
      status: "VERIFIED"
      evidence: ["N3:release_notes:spec13.1"]
      paper_section: "Risks/Threats"
      delta_relevance: "constraints + risk register"
      last_update_date: "2026-02-04"

    - id: "C-N4-01"
      claim: "Triton-to-TileIR compiles Triton kernels to CUDA Tile IR instead of PTX and currently requires CUDA 13.1+ and Blackwell GPUs with source-based compilation."
      status: "VERIFIED"
      evidence: ["N4:NVIDIA blog (2026-01-30)"]
      paper_section: "Baselines/Feasibility"
      delta_relevance: "defines evaluation environment constraints"
      last_update_date: "2026-02-04"
    - id: "C-N4-02"
      claim: "Triton-to-TileIR limitations include incomplete op support and tensor-of-pointer slowdowns; mitigations include SIMT fallback or adopting TMA load/store descriptors."
      status: "VERIFIED"
      evidence: ["N4:NVIDIA blog limitations section"]
      paper_section: "Baselines (limitations)"
      delta_relevance: "workload selection + threat to validity"
      last_update_date: "2026-02-04"

    - id: "C-N6-01"
      claim: "The cuda-tile repo provides an MLIR dialect/tooling ecosystem (Python bindings, bytecode serialization, conformance tests) aligned with CUDA Toolkit 13.1."
      status: "VERIFIED"
      evidence: ["N6:cuda-tile README"]
      paper_section: "Implementation Plan"
      delta_relevance: "MVP hook points for passes + testing"
      last_update_date: "2026-02-04"

    - id: "C-I-01"
      claim: "[INFERENCE] Any memory-overlap/reordering optimization in Tile IR must preserve or construct correct token dependencies (not rely on apparent program deps) to avoid races."
      status: "UNVERIFIED"
      evidence: ["Derived from N3:memory_model:tokens"]
      paper_section: "Core idea constraints"
      delta_relevance: "constrains design space for async scheduling"
      last_update_date: "2026-02-04"
    - id: "C-I-02"
      claim: "[INFERENCE] Research opportunity: combine formal layout reasoning (P1/P2/P3) with Tile IR token/scope legality to build analyzable async overlap + diagnostics for CUDA Tile stacks."
      status: "UNVERIFIED"
      evidence: ["Derived from P1,P2,P3 + N3 token model"]
      paper_section: "Thesis candidates (Stage 2)"
      delta_relevance: "candidate novelty direction"
      last_update_date: "2026-02-04"

EVAL_PLAN:
  status: "draft"
  metrics:
    - "end_to_end_speedup"
    - "compile_time_overhead"
    - "graph/token_complexity (nodes/edges/joins)"
    - "correctness/legality pass rate (and failure diagnostics quality)"
  baselines:
    - "SOTA baseline(s): CUDA Tile / cuTile / Tile IR toolchain"
    - "Naive serialization baseline"
    - "Conservative token-appending baseline (if applicable)"
  workloads: []
  ablations: []
  risks_to_validity: []

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: null
  stage1_5_toolbox: null
  stage2_directions: null
  stage2_5_novelty_audit: null
  stage3_assembly_pack: null
  stage3_paper: null

OPEN_QUESTIONS:
  - id: "OQ-01"
    question: "Which Tile IR operations are token-ordered and how do async/TMA-related ops produce/consume tokens (incl. wait semantics)?"
    why_load_bearing: "Core to 'async' legality + scheduler design."
    query_plan:
      - "Open N3 (Tile IR spec) -> '8. Operations' and search for: 'tko', 'token', 'tma', 'async', 'wait'."
      - "Extract the exact op signatures + token inputs/outputs for relevant load/store/async ops."
  - id: "OQ-02"
    question: "What is the authoritative current hardware/architecture support scope for CUDA Tile / Tile IR beyond what the Triton-to-TileIR blog states?"
    why_load_bearing: "Determines feasibility of evaluation hardware and baseline comparisons."
    query_plan:
      - "Re-check N3 '10. Stability' hardware support matrix and interpret what 'n/a' implies."
      - "Look for official CUDA Toolkit 13.1+ documentation/release notes explicitly listing CUDA Tile availability per architecture."
  - id: "OQ-03"
    question: "What is the exact unsupported/partially-supported operation list for Triton-to-TileIR (and how quickly is it changing)?"
    why_load_bearing: "Workload selection; prevents picking benchmarks that cannot compile."
    query_plan:
      - "From N4 blog, open the linked GitHub page for 'operations and features not yet supported'."
      - "Snapshot the list + date; map to planned workloads."
  - id: "OQ-04"
    question: "How exactly does cuTile Python lower/compile to Tile IR (pipeline boundaries: Python -> MLIR dialect -> bytecode -> driver JIT/AOT)?"
    why_load_bearing: "Determines where to implement passes and how to measure compile overhead."
    query_plan:
      - "Read N2 'Machine Representation' + 'Interoperability' sections."
      - "Cross-check with N6 tooling docs for bytecode production/loading."

NEXT_STAGE_HINT: "Start fresh chat; paste System Config + this capsule + Stage 1 prompt."
STATE_END

ARTIFACT_SUMMARY:
  - "Verified and updated GOLDEN_SOURCES titles for P1/P2/P3; set last_verified=2026-02-04 for all golden sources."
  - "Produced Ground Truth Glossary anchored to Tile IR spec + cuTile docs + seed papers."
  - "Mapped NVIDIA stack components (N1..N6) and a secondary internals blog (C1) into a baseline table with explicit 'does/does not solve'."
  - "Created Claim Ledger v0 with stable Claim_IDs (C-P*, C-N*, C-I*) and statuses."
  - "Identified constraint cliffs tied to token ordering, weak ops, power-of-two tile constraints, and Blackwell/CUDA 13.1 feasibility."
  - "Populated OPEN_QUESTIONS emphasizing async/TMA token semantics and Triton-to-TileIR op support constraints."
```

---
Learn more:
1. [https://developer.nvidia.com/cuda/tile](https://developer.nvidia.com/cuda/tile)
2. [https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
3. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html)
4. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory\_model.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)
5. [https://docs.nvidia.com/cuda/cutile-python](https://docs.nvidia.com/cuda/cutile-python)
6. [https://docs.nvidia.com/cuda/cutile-python/release\_notes.html](https://docs.nvidia.com/cuda/cutile-python/release_notes.html)
7. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog\_model.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)
8. [https://arxiv.org/html/2505.23819v3](https://arxiv.org/html/2505.23819v3)
9. [https://docs.nvidia.com/cuda/cutile-python/memory\_model.html](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)
10. [https://arxiv.org/abs/2505.23819](https://arxiv.org/abs/2505.23819)
11. [https://arxiv.org/html/2511.10374v1](https://arxiv.org/html/2511.10374v1)
12. [https://arxiv.org/abs/2601.05972](https://arxiv.org/abs/2601.05972)
13. [https://arxiv.org/abs/2511.10374](https://arxiv.org/abs/2511.10374)
14. [https://github.com/NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python)
15. [https://github.com/NVIDIA/cuda-tile](https://github.com/NVIDIA/cuda-tile)
16. [https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)
17. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html)
18. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/release\_notes.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/release_notes.html)
