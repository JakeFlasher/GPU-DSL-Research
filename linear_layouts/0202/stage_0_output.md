# WORK_PRODUCT

## 1) Table: Ground Truth Glossary

| Term | Seed-paper View | Hardware/Primary-Source View | SOTA Baseline View | Hard Constraints | Mismatch | Notes |
|---|---|---|---|---|---|---|
| **Tensor layout / layout abstraction** | **P1:** “tensor layouts” = mappings between logical tensors and hardware compute/memory resources; proposes a generic, composable layout representation (“Linear Layouts”) to avoid ad‑hoc, quadratic layout-conversion blowups. ([arxiv.org](https://arxiv.org/html/2505.23819v3))<br>**P2:** CuTe layouts + Triton linear layouts are widely used but isolated; proposes a unified representation via ISL integer set relations for formal reasoning/verification. ([arxiv.org](https://arxiv.org/html/2511.10374v1))<br>**P3:** layout = multi‑D logical coords → 1‑D memory coords; also used to describe SIMT thread partitioning and targeting specialized instructions (tensor cores). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | **Tile IR:** allocations must be contiguous (for `ptr<E>`); a **tile value** includes an *opaque* layout mapping element index space → linear index, but the **physical representation/layout is not specified or visible** to programs. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | **CUDA Tile / Tile IR:** positions tile programming as shifting from explicit SIMT thread mapping to compiler-managed mapping; global memory interaction is primarily via **views**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) | OOB reads/writes are UB in Tile IR; sub‑byte type layouts are restricted to contiguous. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | Seed papers make “layout” an explicit object for reasoning/conversion; Tile IR intentionally hides tile physical layout choices (compiler-owned). | **Inference:** bridging “layout algebra” to Tile IR likely means reasoning over **view shape/stride + compiler lowering choices**, not over a user-specified tile physical layout. |
| **Linear Layouts (Triton)** | **P1:** represents layouts as linear maps over $$\mathbb{F}_2$$ (binary matrices acting on bits); enables generic composition/inversion and generic conversions; integrated into Triton GPU backend with a layout engine that chooses/propagates layouts; reports correctness improvements + up to **1.40×** speedup (avg **1.07×** over **265** real-world cases). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Not a hardware spec, but claims generic lowering to hardware primitives (e.g., optimal swizzle discovery, warp-shuffle generation, generic lowering of intrinsics for the layout family). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **Referenced as baseline “industry standard” layout system** alongside CuTe in P2; distinct from CUDA Tile/Tile IR stack. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | N/A (paper-level formalism; no Tile-IR-style memory model described in the abstract/sections sampled). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Tile IR treats tile physical layout as opaque; Linear Layouts makes (a class of) layouts explicit and algebraic. | **Inference:** novelty risk for “layout representation” alone is high unless you exceed the expressivity/verification guarantees already in P1/P2/P3. |
| **CuTe layout (shape/stride; layout algebra)** | **P2:** CuTe layout maps N‑D coordinate space to 1‑D index via shape+stride tuples; supports hierarchical layouts + operations (composition/inverse/complement); CuTe swizzles do bit-level manipulations. ([arxiv.org](https://arxiv.org/html/2511.10374v1))<br>**P3:** formalizes a categorical framework for a tractable class; defines categories **Tuple** and **Nest** whose morphisms give layouts; proves compatibility with operations like composition/product/division; provides Python implementation aligned with CUTLASS behavior. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | Tile IR **tensor_view** explicitly carries shape/strides metadata (view-space), but tile’s internal physical layout remains opaque. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | CUDA Tile stack appears to favor **views + tile ops** rather than a user-exposed CuTe layout algebra. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | Creating a `tensor_view` is UB if shapes/strides exceed the view index bitwidth; OOB indexing is UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | CuTe exposes a rich, explicit layout algebra; Tile IR’s spec emphasis is on **views + token-ordered memory ops**, not on user-defining tile physical layout transformations. | **Inference:** “CuTe-like explicit layout algebra” may sit *above* Tile IR as a DSL/compiler pass, not inside Tile IR semantics. |
| **Swizzle** | **P2:** CuTe swizzle is a bijective mapping over an interval, defined via bit ops (XOR/AND/shifts/masks) to optimize access patterns. ([arxiv.org](https://arxiv.org/html/2511.10374v1))<br>**P1:** claims “automatic optimal swizzling discovery” with provable objectives (vectorization, bank conflicts). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Tile IR spec excerpts reviewed do not present “swizzle” as a first-class user-visible semantic primitive; optimization behavior is generally compiler/toolchain-controlled. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | CUDA Tile toolchains may include swizzle-related heuristics/controls, but **official semantics are not stated** on the CUDA Tile landing page. ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | N/A | Seed papers formalize swizzle as a mapping; Tile IR treats many layout-like decisions as compiler choices (and warns that unsupported features may be diagnosed or emulated). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html)) | **Secondary (C1):** community blog claims undocumented env vars like `TILEIR_ALWAYS_SWIZZLE`; treat as non-authoritative and unstable. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)) |
| **Tile (data fragment)** | **P3:** frames layouts as critical for mapping data + partitioning threads, especially for tensor-core targeting. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | **Tile IR:** a tile is an immutable N‑D array value with rank/shape/element type in its type; physical representation/layout is unspecified and chosen by compiler for efficiency. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | **cuTile Python:** tiles are immutable values with no defined storage; **tile dimensions must be compile-time constants and powers of two**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | cuTile tiles: compile-time constant, power-of-two dims. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | Seed papers often reason about mapping to warps/threads/registers; Tile IR/cUtile intentionally hide per-thread mapping and tile physical layout. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) | **Inference:** any “layout” research that requires controlling tile’s physical register layout may not be expressible at Tile IR language level (it’s compiler-owned). |
| **Tile-block / block-level execution (vs threads)** | P3 motivates SIMT/thread partitioning as a layout concern, but does not define Tile IR’s execution model. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | **Tile IR:** basic unit = **tile-block** (logical tile thread) executing a tile kernel instance; compiler abstracts mapping of grid + tile threads to hardware threads. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)) | **cuTile Python:** kernels run on a logical grid of blocks; scalar ops execute serially by a single thread of the block, array ops collectively in parallel; threads cannot be identified/manipulated; **explicit sync/communication within a block is not permitted**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) | cuTile forbids explicit intra-block sync; tile code is a restricted Python subset (no runtime). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) | Seed papers’ layout transforms may assume warp-/thread-level control; CUDA Tile stack pushes you to block/tile-level structure with compiler-managed mapping. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) | **Inference:** research that requires explicit warp-synchronous programming may need to be encoded via Tile IR ops/constraints rather than explicit thread IDs. |
| **Views (tensor_view / partition_view) & shape/stride descriptors** | **P2/P3:** shape+stride is central in CuTe; layout = coordinate→index mapping; operations defined over layouts/morphisms. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | **Tile IR:** views enrich pointers with metadata; `tensor_view` is logically (ptr, shape, strides, dimgroups); `partition_view` tiles a tensor_view into an index space; load/store via `load_view_tko` / `store_view_tko`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | **Triton-to-TileIR guidance:** suggests replacing “tensor-of-pointers” with descriptor-based loads/stores carrying (shape, strides, block_shape) to better fit Tile IR backend/TMA path. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | UB if shape/strides exceed view index bitwidth; OOB indexing UB (unless masked/padded view semantics apply). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | Seed papers formalize layout transformations as math objects; Tile IR makes shape/stride explicit at the **view** layer, while tile physical layout stays opaque. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | **Inference:** if your project’s “layout” includes *memory-layout descriptors* (shape/stride/block_shape), Tile IR provides a first-class hook (views). |
| **Token-ordered memory ops & tokens (TKO)** | Abstract-level snapshots of P1–P3 focus on layout modeling/verification/algebra, not an explicit token-based memory-ordering system. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **Tile IR memory model:** tokens build dependencies between memory ops within a tile-block thread; program dependencies do **not** order memory ops; tokens have no runtime representation; weak ops can’t communicate (except via token order within a tile-block thread). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | **Tile IR ops:** memory ops are token-ordered; ordering between any pair is undefined unless connected by tokens; `make_token` creates fresh token; `join_tokens` merges deps. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | Token-ordered ops are not constrained by program order; compiler may reorder unless constrained by tokens. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | Seed-paper layout rewrites that assume program-order safety can be illegal/incorrect under Tile IR unless token/ordering constraints are inserted correctly. | **Inference:** legality/temporal semantics is a “first-class” novelty axis precisely because Tile IR’s reordering model is explicit and non-trivial. |
| **Memory model: ordering + scope; data races** | P2 emphasizes correctness verification of layout transformations (via ISL), not GPU memory consistency rules. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | **Tile IR:** derived from PTX memory model; has scopes (`tile_block`, `device`, `sys`) and orderings (`weak`, `relaxed`, `release`, `acquire`, `acq_rel`); data races are UB; designed as a strict weakening of PTX and interoperable with PTX acquire/release patterns. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | **cuTile Python:** compiler/hardware may reorder; ordering across threads requires atomics with memory order (relaxed/acquire/release/acq_rel) and scope (block/device/sys); synchronization is per-element. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)) | “Weak” cannot synchronize across threads; scopes must cover communicating threads; programs with data races UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Seed-paper “layout correctness” ≠ memory consistency correctness; Tile IR requires explicit memory-order/scope discipline beyond layout semantics. | **Inference:** any “async” research must define token/memory-order interactions, not just data movement schedules. |
| **Asynchrony / pending memory accesses** | P1 targets efficient data movement/codegen, but Tile IR-style operational semantics for async progress is not its stated focus in the abstract excerpts. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **Tile IR operational semantics:** abstract machine state includes a set of **pending memory accesses** that make progress asynchronously to execution of Tile IR ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | cuTile: memory ops may be reordered for performance; no guaranteed cross-thread ordering without explicit synchronization. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)) | Token + memory-order/scope requirements; ordering of per-element memory-model operations from a tile op is left unspecified for implementation flexibility. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Seed-paper layout schedules don’t automatically provide legality under Tile IR’s async/reordering semantics. | **Inference:** “async research” can be grounded as: *new scheduling/overlap strategies + explicit legality constraints (tokens/scopes/orders)*. |
| **Tensor-of-pointers vs view/descriptor/TMA path** | P1 is inside Triton backend; P2 explicitly calls out Triton’s “linear layouts” vs CuTe; but “tensor-of-pointers” performance issues are CUDA Tile backend–specific. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Tile IR offers both pointer-tile gathers (`load_ptr_tko`) and view-based loads (`load_view_tko`); view loads/stores accept optimization hints including `allow_tma`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | **NVIDIA blog:** Triton “tensor-of-pointer” pattern is currently suboptimal on Tile IR backend (CUDA 13.1); suggests using descriptor/TMA API (shape/strides/block_shape) instead; incomplete op support. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Backend immaturity constraints: incomplete op support; requires CUDA 13.1+ and Blackwell GPUs (per blog). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Seed-paper layout IRs don’t address this backend-specific perf cliff; CUDA Tile stack pushes toward descriptor/view-based access patterns. | **Inference:** a research direction could be: “automatic rewrite from tensor-of-pointers to view/descriptor loads with legality + performance modeling,” but novelty must be argued vs existing compiler lowering work. |
| **Stability / portability / emulation** | **P1:** Triton backend targets GPUs “from various vendors” (portability context). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **Tile IR Stability:** defines stability/portability/compatibility; bytecode stability; programs conforming to vX.Y portable to platforms supporting vX.Y+; unsupported features are diagnosed or emulated to preserve semantics; determinism guarantee is scoped (fixed toolchain/config/target; within single tile-block thread). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html)) | **CUDA Tile / cuTile:** marketing-level portability across NVIDIA platforms/tensor cores; cuTile claims enabling latest hardware features “without requiring code changes.” ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | Portability ≠ feature availability; emulation/diagnostics permitted; different toolchain versions may change results. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html)) | Seed-paper portability is broader (multi-vendor toolchains); CUDA Tile portability is within NVIDIA platform, with explicit emulation/compatibility policy. | **Inference:** evaluation should separate “semantic portability” from “performance portability,” since the spec explicitly permits emulation. |

---

## 2) Table: SOTA Baseline Map

| Source_ID | System | Abstraction Level | Key Semantics | Constraints/Assumptions | What It Solves | What It Does NOT Solve | Evidence (cite or UNVERIFIED) |
|---|---|---|---|---|---|---|---|
| **N1** | CUDA Tile (landing/concept) | High-level programming model overview | Tile-based GPU programming model; targets portability for NVIDIA Tensor Cores; based on **Tile IR spec + tools**, including **cuTile** as user-facing language support (Python now; C++ “in the future”). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | Concept page; does not define formal semantics, memory model, or legality rules. ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | Names the stack + entry points; positions CUDA Tile as the portability/performance baseline for tile programming on NVIDIA. ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | No operational semantics; no memory model details; no guarantees about async legality, tokens, or view semantics. ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) |
| **N2** | cuTile Python (docs) | User-facing Python DSL + abstract machine | Execution: kernels over grid of logical blocks; scalar ops serial, array ops collective; threads not explicitly identified; no explicit intra-block sync. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html))<br>Data model: arrays are global-memory, strided, mutable; tiles are immutable values; **tile dims compile-time constants & powers of two**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python))<br>Memory: reordering allowed; synchronization via atomic memory order/scope; per-element granularity. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)) | Kernel caller must ensure arrays don’t alias + remain valid until kernel completion. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html))<br>Tile code is restricted Python subset (no runtime; objects immutable). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) | Practical DSL to write tile kernels; provides a disciplined model for tile/block parallelism and memory synchronization attributes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | Does not offer explicit thread-level programming model; does not permit explicit intra-block synchronization; does not itself fully specify Tile IR token semantics (points to Tile IR docs). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) | ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) |
| **N3** | Tile IR (spec) | Portable low-level tile VM / instruction set + formal memory model | Models GPU as tile-based processor; tile kernels run as grid of tile blocks; mapping to hardware threads abstracted. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html))<br>Defines operational semantics (incl. pending async memory accesses). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html))<br>Defines memory model (scopes/orderings, tokens, data-race UB, PTX interoperability). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))<br>Defines views as primary global-memory interface (`tensor_view`, `partition_view`, `load_view_tko/store_view_tko`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | Tile physical layout opaque; out-of-bounds UB; weak ops assume no concurrent accesses; ordering between memory ops undefined unless token-constrained. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | Paper-grade baseline for legality/temporal semantics: explicit tokens, scopes, orderings, UB conditions; stable/versioned target for DSLs/compilers. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/)) | Does not specify microarchitectural mapping; does not guarantee performance; hides tile physical layout decisions; feature availability may require emulation/diagnostics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)) | ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/)) |
| **N4** | Triton-to-TileIR backend (NVIDIA blog) | Compiler backend integration narrative + current limitations | Integrates CUDA Tile as Triton backend; can switch PTX↔Tile IR backend (env var mentioned) and select per-kernel backend. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))<br>Roadmap: dialect conversion, semantic validation, benchmarking. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))<br>Prereqs: CUDA **13.1+**, **Blackwell** GPUs; build from source; incomplete op support; tensor-of-pointer pattern currently suboptimal; suggests adopting TMA/descriptor API. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Early-stage backend; incomplete operation coverage; performance cliffs (tensor-of-pointer); hardware/toolchain prerequisites. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Concrete migration path for Triton → Tile IR; highlights real constraints for evaluation and for “layout + async” work (tensor-of-pointer vs descriptor). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Not a spec; does not guarantee semantics beyond linking to Tile IR; limitations may change quickly. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| **N5** | `NVIDIA/cutile-python` (repo) | Implementation repo (source) | README positions cuTile Python as a “programming language for NVIDIA GPUs” and points to official docs; example mentions CUDA toolkit **13.1+**. ([github.com](https://github.com/NVIDIA/cutile-python)) | Repo README is not the formal semantics; semantics live in docs/spec. ([github.com](https://github.com/NVIDIA/cutile-python)) | Reproducibility + implementation reference; testbed for experiments that must run “in the stack” rather than as a paper-only IR. ([github.com](https://github.com/NVIDIA/cutile-python)) | Does not itself specify the formal memory model/token semantics; must defer to Tile IR + docs. ([github.com](https://github.com/NVIDIA/cutile-python)) | ([github.com](https://github.com/NVIDIA/cutile-python)) |
| **N6** | `NVIDIA/cuda-tile` (repo) | Tile IR ecosystem components (MLIR-based) | README describes CUDA Tile IR as MLIR-based IR + compiler infrastructure for tiled computation patterns and tensor-core targeting. ([github.com](https://github.com/NVIDIA/cuda-tile))<br>Build instructions depend on LLVM/MLIR source/commit compatibility. ([github.com](https://github.com/NVIDIA/cuda-tile)) | Build/toolchain coupling; repo-level docs complement (but do not replace) Tile IR spec. ([github.com](https://github.com/NVIDIA/cuda-tile)) | Practical baseline for “what exists today” in open source for CUDA Tile IR tooling. ([github.com](https://github.com/NVIDIA/cuda-tile)) | Not a user-facing DSL; does not itself provide complete formal semantics (spec is in N3). ([github.com](https://github.com/NVIDIA/cuda-tile)) | ([github.com](https://github.com/NVIDIA/cuda-tile)) |
| **C1 (optional)** | Community “TileIR internals” blog | Secondary reverse-engineering / exploratory | Traces compilation pipeline cuTile → intermediate dialects → NVVM/LLVM → SASS; explicitly warns details are undocumented and may change; lists “undocumented environment variables” affecting compilation (e.g., preferring TMA, forcing swizzle). ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)) | Not authoritative; may be stale quickly; should not be treated as semantic truth. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)) | Useful for practical debugging hypotheses + internal naming, but only as *secondary* context. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)) | Does not provide spec guarantees; cannot be used to claim legality/semantics. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)) | ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)) |

---

## 3) Golden Snapshot (Carry-Forward)

### P1
- **What it is:** Research paper “Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $$\mathbb{F}_2$$” (ASPLOS ’26 context shown on the arXiv HTML). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **What it guarantees/claims:** Introduces “Linear Layouts” (binary-matrix/$$\mathbb{F}_2$$ model) for generic layout definition + layout-to-layout conversion; integrates into Triton GPU backend with a layout engine; reports correctness improvements and speedups (up to 1.40×, avg 1.07× over 265 real benchmarks) and new algorithms for swizzling/warp shuffle/intrinsics lowering. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **What it does NOT say:** Does not (in the abstract/contribution snippets reviewed) define a token-based GPU memory model or Tile IR–e legality constraints for async reordering. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **Why we care:** It’s a strong prior on “layout as a first-class mathematical object,” and a direct novelty competitor if your direction is “new layout representation/conversion.” ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

### P2
- **What it is:** NVIDIA-authored paper “Modeling Layout Abstractions Using Integer Set Relations.” ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **tees/claims:** Unifies CuTe layouts (shape/stride + swizzle) and Triton linear layouts (binary vector space transforms) via ISL integer set relations; aims at formal analysis/correctness verification and cross-system reasoning; implements ISL-based layout manipulation (composition/inversion/complement). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **What it does NOT say:** Explicitly states it is theoretical/foundational; goal is not performance optimizations per se. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **Why we care:** Provides a credible “verification substrate” baseline—important if your project claims legality/temporal semantics and correctness beyond speed. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

### P3
- **What it is:** Paper “Categorical Foundations for CuTe Layouts” (Colfax Research, Jan 2026). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **What it guarantees/claims:** Categorical framework (categories Tuple/Nest) for a tractable class of laines operations on morphisms and proves compatibility with corresponding layout operations; gives a complete characterization of layouts arising from the construction; provides Python implementation + tests aligned with CUTLASS behavior. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **What it does NOT say:** Not a CUDA Tile/Tile IR spec; does not define GPU memory ordering/async legality rules. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **Why we care:** Another strong “foundations + algra” baseline; novelty risk if you claim new formal layout algebra without exceeding this scope. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

### N1
- **What it is:** NVIDIA CUDA Tile landing page (top-level entry to the CUDA Tile stack). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- **What it guarantees/claims:** CUDA Tile is tile-based GPU programming model targeting portability for NVIDIA Tensor Cores; CUDA Tile is based on Tile IR + tools, including cuTile (Python now; fure C++). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- **What it does NOT say:** Does not provide formal semantics/memory model details (delegates to Tile IR spec/docs). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- **Why we care:** Defines the SOTA baseline target (what you must differentiate from). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  

### N2
- **What it is:** Official cuTile Python documentation (user-facing DSL spec). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python))  
- **What it guarantees/claims:** Defines execution model (block-level tile code; no thread IDs; no explicit intra-block sync), Python subset constraints, data model (arrays vs tiles), and memory model surface (atomic order/scope). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html))  
- **What it does NOT say:** Does not fully restate Tile IR’s token semantics; points to Tile IR docs for more detailed memory model explanatio ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html))  
- **Why we care:** Concrete “what users can write” baseline; any research direction must be expressible/compilable under these constraints if you claim integration. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html))  

### N3
- **What it is:** Tile IR specification (portable tile VM + instruction set + memory model). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/))  
- **Whatguarantees/claims:** Defines programming model, operational semantics, token-based memory ordering, memory scopes/orderings, UB conditions, and view-based global memory interface; includes explicit stability/portability policy. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html))  
- **What it does NOT say:** Does not expose tile physical layout decisions; does not guarantee performance; allows emulation/diagnostics for unsupported features. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html))  
- **Why we care:** This is the “legality/temporal semantics” baseline you must respect (and beat, if novelty claims hinge on legality). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  

### N4
- **What it is:** NVIDIA technical blog post on the Triton-to-TileIR backend (Jan 30, 2026). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-fpenai-triton/))  
- **What it guarantees/claims:** Backend enables Triton to target Tile IR; highlights roadmap (conversion/validation/benchmarking), prerequisites (CUDA 13.1+, Blackwell), and current limitations (unsupported ops; tensor-of-pointer perf issues; suggests descriptor/TMA approach). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- **What it does NOT say:** Not a formal spec; details may evolve rapidly with CUDA/toolchain releases. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- **Why we care:** Sets practical constraints for evaluation and for any “layout + async” direction that targets Triton as a front-end. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

### N5
- **What it is:** `NVIDIA/cutile-python` GitHub repository. ([gi.com](https://github.com/NVIDIA/cutile-python))  
- **What it guarantees/claims:** Points to official docs; gives installation/example hints (mentions CUDA toolkit 13.1+ in example context). ([github.com](https://github.com/NVIDIA/cutile-python))  
- **What it does NOT say:** Repo README is not the formal semantics; must defer to cuTile docs + Tile IR spec. ([github.com](https://github.com/NVIDIA/cutile-python))  
- **Why we care:** Practical codebase for running experiments and verifying what’s implementle. ([github.com](https://github.com/NVIDIA/cutile-python))  

### N6
- **What it is:** `NVIDIA/cuda-tile` GitHub repository (CUDA Tile IR ecosystem). ([github.com](https://github.com/NVIDIA/cuda-tile))  
- **What it guarantees/claims:** Describes an MLIR-based IR + compiler infrastructure for tile computation patterns/tensor-core optimizations; includes build instructions tied to LLVM/MLIR compatibility. ([github.com](https://github.com/NVIDIA/cuda-tile))  
- **What it does NOT say:** Does not substitute for the Tile IR spec; repo-level docs focus on tooling/build. ([github.com](https://github.com/NVIDIA/cuda-tile))  
- **Why we care:** Baseline for “what’s open + available” in the CUDA Tile IR toolchain. ([github.com](https://github.com/NVIDIA/cuda-tile))  

### C1
- **What it is:** Community blog post “NVIDIA TileIR Internals…” by Henry Zhu (Created Jan 30, 2026). ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/))  
- **What it guams:** Attempts to trace compilation pipeline cuTile → intermediate dialects → NVVM/LLVM → SASS; explicitly flags that some details are undocumented and may change; lists undocumented env vars (e.g., preferring TMA; forcing swizzle; delaying TMA store wait). ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/))  
- **What it does NOT say:** Not an official NVIDIA spec; cannot be used as authoritative semantics or stability guarantees. ([makneub.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/))  
- **Why we care:** Helpful for debugging hypotheses and understanding pipeline names, but must be treated as volatile/secondary. ([maknee.github.io](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/))  

---

## 4) Constraint Cliffs (max 5)

1. **Token-order is mandatory for memory legality in Tile IR:** program dependencies do *not* order memory ops; ordering between memory ops is undefined unless token-connected; compiler may reorder token-ordered ops unless constrained by tokens. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
2. **“Weak” is a strong assumption, not a default-safe mode:** weak ops can’t be used to communicate across threads; compiler may assume no concurrent accesses; data races are UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
3. **cuTile forxplicit intra-block synchronization and thread IDs:** any direction needing warp-/thread-level coordination must be expressed via the tile/block model and the IR/toolchain’s ordering primitives, not via explicit sync. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html))  
4. **Shape/layout safety is on you:** OOB access is UB; `tensor_view` creation is UB if shapes/strides exceed index bitwidth; arrays passed to kernels must not alias and must remain valid through kernel completi. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html))  
5. **Backend maturity + hardware gating can dominate evaluation:** Triton-to-TileIR requires CUDA 13.1+ and Blackwell GPUs (per blog), has incomplete op support, and currently suffers tensor-of-pointer performance issues (suggesting descriptor/TMA path). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

---

# CONTEXT_CAPSULE

```yaml
profile:
  project_name: "GPU Layout + Async Research Program"
  target_model: "gpt-5.2-pro (web UI)"
  operating_mode: "manual_state (no reliable API compaction)"
  stage_plan: ["0", "1", "1.5", "2", "2.5", "3"]

  hard_constraints:
    - "No hallucinated citations"
    - "No invented technical specifics"
    - "Exactly two deliverables: WORK_PRODUCT then CONTEXT_CAPSULE"
    - "Golden sources always present in STATE"

  current_stage: 0
  last_updated: "2026-02-03"

  problem_statement_1liner: null
  success_criteria:
    - "Novelty delta vs CUDA Tile / cuTile / Tile IR is explicit"
    - "Legality/temporal semantics are first-class (not handwaved)"
    - "Evaluation plan includes more than speedup"

GOLDEN_SOURCES:
  - id: "P1"
    kind: "seed_paper"
    title: "Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using 𝔽₂"
    url: "https://arxiv.org/html/2505.23819v3"
    last_verified: "2026-02-03"
  - id: "P2"
    kind: "seed_paper"
    title: "Modeling Layout Abstractions g Integer Set Relations"
    url: "https://arxiv.org/html/2511.10374v1"
    last_verified: "2026-02-03"
  - id: "P3"
    kind: "seed_paper"
    title: "Categorical Foundations for CuTe Layouts"
    url: "https://arxiv.org/pdf/2601.05972v1"
    last_verified: "2026-02-03"

  - id: "N1"
    kind: "nvidia_primary"
    title: "CUDA Tile (concept)"
    url: "https://developer.nvidia.com/cuda/tile"
    last_verified: "2026-02-03"
  - id: "N2"
    kind: "nvidia_primary"
    title: "cuTile Python docs"
    url: "https://docs.nvidia.com/cuda/cutile-python"
    last_verified: "2026-02-03"
  - id: "N3"
    kind: "nvidia_primary"
    title: "Tile IR spec"
    url: "https://docs.nvidia.com/cuda/tile-ir/latest/"
    last_verified: "2026-02-03"
  - id: "N4"
    kind: "nvidia_primary"
    title: "Triton-to-TileIR backend (NVIDIA blog)"
    url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
    last_verified: "2026-02-03"
  - id: "N5"
    kind: "repo"
    title: "cuTile Python repo"
    url: "https://github.com/NVIDIA/cutile-python"
    last_verified: "2026-02-03"
  - id: "N6"
    kind: "repo"
    title: "cuda-tile repo"
    url: "https://github.com/NVIDIA/cuda-tile"
    last_verified: "2026-02-03"

  - id: "C1"
    kind: "community_secondary"
    title: "TileIR internals blog (secondary)"
    url: "https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/"
    last_verified: "2026-02-03"

GLOBAL_MEMORY:
  notes:
    - id: "GM-format"
      text: "Always output exactly: WORK_PRODUCT then CONTEXT_CAPSULE."
      last_update_date: "2026-02-03"
      keywords: ["format", "workflow"]

    - id: "GM-precedence"
      text: "Memory precedence: latest user > session > global; global is advisory."
      last_update_date: "2026-02-03"
      keywords: ["memory", "precedence"]

    - id: "GM-golden"
      text: "Golden sources remain explicitly listed in STATE; don’t claim specifics unless provided/cited."
      last_update_date: "2026-02-0
      keywords: ["sources", "citations"]

    - id: "GM-tileir-token-order"
      text: "Tile IR memory ops are token-ordered; program order/dependencies do not order memory ops; ordering is undefined unless constrained by tokens."
      last_update_date: "2026-02-03"
      keywords: ["tileir", "tokens", "memory-model", "legality"]

    - id: "GM-tileir-weak-scope"
      text: "Tile IR: weak ops assume no concurrent access and cannot be used for cross-thread communication; scoped + ordered ops required for synchronization; data races are undefined behavior."
      last_update_date: "2026-02-03"
      keywords: ["tileir", "memory-model", "weak", "scope", "ordering"]

    - id: "GM-tileir-opaque-tile-layout"
      text: "Tile IR: tile physical layout/representation is not specified/visible; views (tensor_view/partition_view) expose shape+stride metadata for memory access."
      last_update_date: "2026-02-03"
      keywords: ["tileir", "tiles", "views", "layout"]

    - id: "GM-cutile-exec-constraints"
      text: "cuTile Python expresses block-level tile programming; threads are not user-addressable; explicit intra-block sync/communication is not permitted; tile dims are compile-time constants and powers of two."
      last_update_date: "2026-02-03"
      keywords: ["cutile", "execution-model", "constraints"]

SESSION_MEMORY:
  notes: []

VERDICT_LEDGER:
  items: []

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260203"
  stage1_gap_audit: null
  stage1_5_toolbox: null
  stage2_directions: null
  stage2_5_novelty_audit: null
  stage3_paper: null
  stage3_assembly_pack: null

OPEN_QUESTIONS:
  - id: "Q0-01"
    text: "Extract Tile IR’s full set of async/TMA-related ops (e.g., any explicit wait/commit primitives) and their token + memory-order interactions."
    priority: "high"
    suggested_queries:
      - "docs.nvidia.com tile-ir operations TMA wait token"
      - "Tile IR TMA store wait token order semantics"
    last_update_date: "2026-02-03"

  - id: "Q0-02"
    text: "Clarify how cuTile Python exposes (or hes) Tile IR tokens in practice; is token discipline inferred by compiler or user-controllable anywhere in cuTile?"
    priority: "high"
    suggested_queries:
      - "cuTile Python token ordered operations"
      - "cuda.tile make_token join_tokens"
    last_update_date: "2026-02-03"

  - id: "Q0-03"
    text: "Confirm the current hardware support constraints for Tile IR targets (Blackwell-only vs partial earlier arch support) and which features are emulated vs diagnosed."
    priority: "high"
    suggested_queries:
      - "Tile IR hardware support matrix Ampere Hopper Blackwell"
      - "CUDA Tile IR supported GPUs 13.1"
    last_update_date: "2026-02-03"

  - id: "Q0-04"
    text: "Map seed-paper layout representations (Linear Layouts / CuTe algebra / ISL relations) onto Tile IR 'view' semantics: what is the closest comparable object, and where is expressivity lost?"
    priority: "medium"
    suggested_queries:
      - "Tile IR tensor_view shape stride dimgroups semantics"
      - "Linear Layouts F2 matrix layout conversion vs tensor view"
    last_update_date: "2026-02-03"

  - id: "Q0-05"
    text: "Quantify baseline evaluation axes beyond speed that are feasible in this stack (e.g., correctness/verification coverage, determinism, memory-model litmus tests, robustness to toolchain versions)."
    priority: "medium"
    suggested_queries:
      - "Tile IR memory model litmus tests token order"
      - "Tile IR determinism guarantees stability section"
    last_update_date: "2026-02-03"

NEXT_STAGE_HINT:
  stage: "1"
  paste_next: |
    Paste the entire updated STATE from this CONTEXT_CAPSULE, plus your Stage 1 prompt.
    Suggested Stage 1 focus: Gap audit between (P1/P2/P3) layout formalisms and (N2/N3/N4) legality+async semantics, with an explicit “novelty delta vs Tile IR” matrix.
```

## ARTIFACT_SUMMARY
- Produced **WP0_20260203** Stage-0 fact sheet grounded in the golden sources (P1–P3, N1–N6, C1).
- Built a **Ground Truth Glossary** table aligning seed-paper layout formalisms with Tile IR / cmantics (tokens, views, memory model, stability).
- Built a **SOTA Baseline Map** table for CUDA Tile / cuTile / Tile IR / Triton-to-TileIR + repos (+ secondary internals blog).
- Created a carry-forward **Golden Snapshot** (3–6 bullets per golden source ID).
- Identified **5 Constraint Cliffs** (token legality, weak-scope pitfalls, cuTile execution constraints, UB conditions, backend maturity/hardware gating).
- Updated STATE: set `current_stage=0`, `last_updated=2026-02-03`, refreshed GOLDEN_SOURCES tits + `last_verified`, and wrote durable GLOBAL_MEMORY items.
- Added prioritized OPEN_QUESTIONS aimed at nailing down async/TMA semantics, token exposure, and hardware support boundaries.

## NEXT_STAGE_HINT
Stage **1**: run a **gap/novelty audit** that explicitly answers: “What can’t be expressed (or proven/legal) in CUDA Tile / Tile IR today, that our direction adds?” Use the updated STATE above as the only carried context.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensputation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
3. [Categorical Foundations for CuTe Layouts](https://arxiv.org/pdf/2601.05972v1)
4. [6. Semantics — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html)
5. [2. Programming Model — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)
6. [8. Operations — Tile IR](https://docs.nvidia.com/cuda/tile-irsections/operations.html)
7. [CUDA Tile | NVIDIA Developer](https://developer.nvidia.com/cuda/tile)
8. [10. Stability — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html)
9. [NVIDIA TileIR Internals: from CuTile to MLIR/LLVM to SASS | Henry Zhu](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)
10. [cuTile Python — cuTile Python](https://docs.nvidia.com/cuda/cutile-python)
11. [Execution Model — cuTile Python](https://docs.nvidia.ca/cutile-python/execution.html)
12. [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
13. [7. Memory Model — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)
14. [Memory Model — cuTile Python](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)
15. [Tile IR — Tile IR](https://docs.nvidia.com/cuda/tile-ist/)
16. [GitHub - NVIDIA/cutile-python: cuTile is a programming model for writing parallel kernels for NVIDIA GPUs](https://github.com/NVIDIA/cutile-python)
17. [GitHub - NVIDIA/cuda-tile: CUDA Tile IR is an MLIR-based intermediate representation and compiler infrastructure for CUDA kernel optimization, focusing on tile-based computation patterns and optimizations targeting NVIDIA tensor core units.](https://github.com/NVIDIA/cuda-tile)
