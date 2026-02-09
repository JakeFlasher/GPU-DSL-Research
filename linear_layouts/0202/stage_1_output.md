# WORK_PRODUCT

## 1) Table: **Axiom-vs-Hardware-vs-SOTA Matrix**

Ordered elephant-first by bottleneck class: **legality cliffs → temporal orchestration → memory wall → search/compile cost → expressivity (views/subviews) → availability**.

| Gap_ID | Seed_Axiom | Hardware_Feature | SOTA_System | What SOTA Solves | Why Still Fails | Required Extension | Evidence |
|---|---|---|---|---|---|---|---|
| **G2** (Legality cliff: token memory model) | **A:** Layouts as composable mappings + **generic conversions** (avoid case-by-case / quadratic conversion explosion). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **B:** GPU/compiler reordering is allowed; Tile IR memory model is derived from PTX-like behavior to permit optimizations. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | **C:** Tile IR: **program dependencies do *not* order memory ops**; **tokens must be used** to order token-ordered memory ops; weak ops assume no concurrent access; data races are UB. Also: “all memory ops are token-ordered; ordering undefined unless connected by tokens.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Makes the dependence structure explicit (tokens) and specifies order/scope for synchronization. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | **D (inference):** Seed layout optimizations (conversion, reordering, fusion) don’t *by themselves* carry a token+scope legality proof. Any pass that changes memory-op structure must reconstruct token constraints or risk **UB / “reads-from-future” style hazards**. | **Token-aware legality framework**: effect/type system for memory ops + token inference/repair + proof-carrying transforms (or equivalence checking) against Tile IR’s memory model. | Tile IR tokens + “program deps don’t order” + data-race UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| **G3** (Temporal orchestration: async/TMA pipelines) | **A:** Seed work optimizes layout conversions & data movement (incl. swizzle discovery / lowering to hardware primitives), but doesn’t provide a Tile-IR-token-shaped **temporal schedule object** (pipeline stages, prefetch distance) as a first-class artifact. **(inference)** ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **B:** Tile IR memory ops can be reordered unless token-constrained; spec explicitly notes token-ordered ops aren’t constrained by program order. Tile IR also has arch hints touching TMA usage (`allow_tma`, `latency`) for view loads/stores. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | **C:** Tile IR exposes token constructors (`make_token`, `join_tokens`) and load/store consume+produce tokens. cuTile forbids explicit intra-block sync/comm and thread IDs (limits user-authored schedule control). Triton-to-TileIR backend is early: incomplete op coverage; recommends shifting from tensor-of-pointers to descriptor/TMA-style APIs. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | Gives a *mechanism* (token partial order) to express dependencies and allow parallelism/reordering. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | **D (inference):** Mechanism ≠ **portable, composable orchestration**. There is no obvious, verifiable “async plan” layer that (1) composes with layout transforms, (2) generates correct token graphs, and (3) can be searched/tuned without correctness risk—especially if DSL level hides tokens. | Add a **temporal/orchestration IR** that compiles to tokens (and, where relevant, TMA choices) + verification hooks (litmus tests / static checks). Also clarify/standardize the async/TMA op set and token interactions. | Token ops + reordering note + TMA hints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) |
| **G1** (Memory wall + descriptor gap: swizzle/non-affine layouts) | **A:** Linear Layouts: layouts as **binary matrices acting on bits**; supports generic layout definitions + conversions; includes **swizzling** and codegen for data movement. ISL-relations paper explicitly models CuTe swizzles as bit-level manipulations (XOR/shift/AND). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **B:** TMA / “tensor memory accelerators” exist; TMA-style descriptor APIs can avoid materializing tensor-of-pointers; NVIDIA notes tensor-of-pointer pattern is suboptimal on Tile IR backend and suggests descriptor/TMA approach. Tile IR ops also mention `allow_tma` for view load/store hints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python)) | **C:** Tile IR `tensor_view` is **affine/stride-based** (baseptr + sum of $$i_m \cdot s_m$$) and is meant to expose strided structure; a tile-of-pointers explicitly has **no implied structure**; pointer-tile gather (`load_ptr_tko`) exists. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) | Provides (i) strided structured views and (ii) arbitrary pointer gather/scatter for irregular access. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) | **D (inference):** Many high-perf layouts use **non-affine / bit-level swizzles** (the seeds treat as first-class). Stride-only `tensor_view` can’t represent these. Falling back to pointer tiles loses structure and aligns with known “tensor-of-pointers” perf pathologies. | Introduce **first-class non-affine layout descriptors/views** (e.g., F₂-linear, ISL-relation-based, or categorical-normal-form-backed) + verified conversion between descriptor forms + lowering that preserves structure (enables TMA/coalescing) instead of collapsing into pointer tiles. | Stride-based view definition + pointer-tile non-structure + NVIDIA guidance re tensor-of-pointers + Tile IR TMA hints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) |
| **G5** (Search/compile cost: specialization explosion) | **A:** Seeds explicitly target systematic/generic layout handling (reduce engineering effort; avoid quadratic explosion; enable formal reasoning). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **B:** Performance features often require compile-time specialization; Tile IR tiles require static power-of-two extents; cuTile tiles likewise require compile-time constant, power-of-two dimensions; cuTile also supports constant embedding that forces per-value compilation. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) | **C:** cuTile: constant embedding ⇒ distinct kernel machine representation per constant value and compiled once per value; Tile IR: tile extents are power-of-two/static. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) | Enables aggressive optimization and stable performance by forcing static shapes/params. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) | **D (inference):** Auto-tuning/search across layouts & tile sizes becomes *compile-bound* (variant explosion). This is especially toxic if the research plan needs broad sweeps (layouts × schedules × shapes). | Treat compile/search as first-class: persistent caching, variant factoring, parametric layout descriptors, incremental compilation, and “verify once / reuse many” legality proofs. | cuTile constant embedding rules + tile constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) |
| **G4** (Expressivity: subview/layout algebra mismatch) | **A:** CuTe layout algebra includes operations like composition/logical product/division; ISL-relations approach supports composition/inversion/complement and models hierarchical layouts & swizzles; categorical paper formalizes these layout operations. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | **B:** Hardware benefits when the compiler knows structured access (strides/tiles). Tile IR introduces `tensor_view` + subview concept to expose structure. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) | **C:** Tile IR subviews: spec says it **currently provides a single subview** (`partition_view`) and is designed to support additional subview types in future. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) | Captures a very common pattern: partition a strided view into a grid of **non-overlapping** tiles (with optional masking). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) | **D (inference):** Seed algebra operations (e.g., overlapping/sliding/compose-able layouts) don’t map cleanly onto “only partition_view”. Falling back to pointer tiles again destroys structure and makes verification harder. | Extend subview taxonomy or add a **general index-relation subview** (ISL-style) + composition operators + explicit bounds/masking semantics that remain verifiable. | “Only partition_view for now” + seed ops catalog. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) |
| **G6** (Availability/portability gating: Blackwell-only in key backend) | **A:** Seeds aim for cross-system reasoning/unification (CuTe+Triton; Triton backend context). ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | **B:** Triton-to-TileIR backend prerequisites: **CUDA 13.1+ and NVIDIA Blackwell GPUs**; earlier architectures “enabled in upcoming CUDA releases.” ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | **C:** Backend is early: incomplete op support; known perf issues for tensor-of-pointer patterns; suggested mitigation via TMA APIs. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Provides a direct compilation path preserving tile semantics, positioned for next-gen features. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | **D (inference):** Research evaluation becomes hardware-gated and confounded by backend immaturity (hard to attribute wins/losses to the proposed layout+async method). | Multi-backend eval plan (PTX backend baselines, partial emulation), plus “capability tiers” so results are comparable even when Tile IR backend coverage is incomplete. | Backend prereqs + limitations called out by NVIDIA. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| **G7** (Control gap: thread/warp mapping & intra-block comm) | **A:** Linear Layouts includes generic lowering to hardware primitives (incl. **warp-shuffle generation**). Categorical CuTe foundations explicitly tie layouts to SIMT thread partitionings. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **B:** Hardware has intra-warp exchange primitives that seeds exploit (warp shuffles). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **C:** cuTile: tile programs describe **block-level** parallelism; threads can’t be identified/manipulated; **explicit intra-block sync/comm not permitted**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) | Raises abstraction level and lets compiler pick mapping/scheduling. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) | **D (inference):** If a new layout method requires explicit control of thread partitioning or communication patterns, cuTile-level expressivity may be insufficient (forcing compiler-internal changes or dropping to other layers). | Provide a verified “mapping layer” (annotations/constraints) or controlled low-level primitives that remain within Tile IR legality (tokens/scopes) while exposing needed control. | cuTile execution constraints + seed reliance on intra-warp exchange concepts. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) |

---

## 2) Stage-1 Verdict (≤10 bullets)

1) **Top gap = G2 (legality cliff):** Tile IR’s memory model makes **token order** the only reliable intra-thread ordering for memory ops; **program deps don’t order memory ops** and data races are UB, so layout/async transforms must be token-aware or they can silently become illegal. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
2) **#2 = G3 (temporal orchestration):** Tokens give a mechanism, but there’s no obvious *portable, composable* “async plan” object that can be searched/tuned and then lowered into a correct token graph—especially given cuTile’s restriction against explicit intra-block sync/comm and thread IDs. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html))  
3) **#3 = G1 (memory wall / descriptor gap):** Seed papers treat swizzles and generic conversions as first-class; Tile IR’s `tensor_view` is stride/affine, and pointer-tiles have no implied structure—pushing swizzles into pointer math tends to recreate the “tensor-of-pointers” performance cliff NVIDIA calls out (and motivates descriptor/TMA usage). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html))  
4) **Next bottleneck = G5 (search/compile cost):** cuTile constant embedding compiles per parameter value, and both cuTile and Tile IR require power-of-two, compile-time tile extents—so naive tuning/sweeps over layouts × schedules can be dominated by compilation, not runtime. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html))  
5) **G4 matters for “novelty delta vs Tile IR”:** Tile IR subviews are currently “partition_view-only,” while seed layout algebras support richer composition/product/division/hierarchy; mapping that expressivity without collapsing into pointer tiles is a central research gap. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html))  
6) **G6 is an evaluation reality:** Triton-to-TileIR is (per NVIDIA) CUDA 13.1+ and Blackwell-only today; older arch enablement is future-facing, and op coverage is incomplete—so evaluation must anticipate hardware gating and backend churn. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
7) **Implication for Stage 1.5/2:** The credible “novelty delta” is likely **not** “yet another layout algebra,” but a **layout+temporal+legality** bundle: a representation that can express swizzles/regularity *and* a verified lowering to Tile IR tokens/scopes and (where available) TMA paths. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html))  

---

## 3) Carry-Forward Gap List (compact)

| Gap_ID | 1-line statement | Required evidence to verify / tighten |
|---|---|---|
| G2 | Token memory model makes legality the dominant constraint on any layout/async transform. | Confirm full set of token-ordered memory ops & supported orderings/scopes for each (e.g., store variants, view ops); build token-order litmus suite aligned to Tile IR model. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| G3 | No first-class, verifiable async/TMA pipeline abstraction that composes with layouts and lowers to tokens. | Extract Tile IR’s async/TMA surface area (including any wait/commit primitives) + token interactions; determine whether/how cuTile exposes token control. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) |
| G1 | Stride-only `tensor_view` can’t represent bit-level swizzles; pointer fallback loses structure and can be slow. | Identify whether Tile IR has (or plans) non-affine/swizzle view descriptors; quantify pointer-tile vs view/TMA behavior on representative kernels; map seed swizzle formalisms to any Tile IR descriptor forms. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) |
| G5 | Compile/search cost is first-class due to power-of-two + constant-embedded specialization. | Measure compilation-cache behavior in cuTile; quantify variant explosion under tuning; identify which params can be “late-bound” safely without losing codegen quality. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) |
| G4 | Tile IR subview expressivity lags seed layout algebras (partition_view-only). | Confirm current + planned subview types and their semantics; prototype one missing subview (overlap/sliding) and show legality+lowering story. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)) |
| G6 | Evaluation is hardware-gated (Blackwell + CUDA 13.1 for Triton-to-TileIR backend). | Verify current (Feb 3, 2026) support matrix across cuTile/Tile IR tooling (beyond Triton-to-TileIR); define fallback baselines (PTX backend) to deconfound results. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| G7 | cuTile hides thread-level control needed by some seed-style mapping strategies. | Determine if Tile IR itself has primitives/annotations for thread partitioning or intra-block communication (within token legality); otherwise plan for a compiler-internal path. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/execution.html)) |

---

# CONTEXT_CAPSULE

## Updated STATE (paste-ready)

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

  current_stage: 1
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
    title: "Modeling Layout Abstractions Using Integer Set Relations"
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
      last_update_date: "2026-02-03"
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
      text: "Tile IR / cuTile: tiles are pure/immutable values without defined storage; structured memory access is expressed via views (shape+stride) or pointer tiles."
      last_update_date: "2026-02-03"
      keywords: ["tileir", "cutile", "tiles", "views", "layout"]

    # Stage-1 promotions (durable gap conclusions)
    - id: "GM-gap-top"
      text: "Stage-1 ranking: (1) token-order legality cliffs, (2) temporal orchestration for async/TMA via tokens, (3) structured layout descriptor gap (swizzle/non-affine vs stride-only views), then compile/search cost."
      last_update_date: "2026-02-03"
      keywords: ["gaps", "ranking", "legality", "async", "layouts"]

    - id: "GM-gap-G1-stride-only-view"
      text: "Tile IR tensor_view is affine stride-based; seed-style bit-level swizzles/non-affine layouts cannot be represented as tensor_view, pushing implementations toward pointer tiles and risking 'tensor-of-pointers' perf cliffs."
      last_update_date: "2026-02-03"
      keywords: ["gap", "views", "swizzle", "tensor_view", "tma"]

    - id: "GM-gap-G4-subview-limited"
      text: "Tile IR subviews: currently only partition_view is specified; seed layout algebras (CuTe ops, ISL relations) require richer subview/indexing forms (overlap/compose) to preserve structure."
      last_update_date: "2026-02-03"
      keywords: ["gap", "subview", "partition_view", "expressivity"]

    - id: "GM-gap-G5-compile-variants"
      text: "cuTile constant embedding compiles per distinct constant value; tile dims are compile-time power-of-two. Auto-tuning/search can become compile-cost dominated."
      last_update_date: "2026-02-03"
      keywords: ["gap", "compile-cost", "autotune", "specialization"]

    - id: "GM-gap-G6-blackwell-gating"
      text: "As of NVIDIA blog dated 2026-01-30: Triton-to-TileIR requires CUDA 13.1+ and Blackwell GPUs; older architectures expected in future CUDA releases."
      last_update_date: "2026-02-03"
      keywords: ["gap", "hardware-support", "blackwell", "cuda-13.1"]

SESSION_MEMORY:
  notes: []

VERDICT_LEDGER:
  items:
    - id: "V1-G2"
      decision: "Treat token-order legality as the primary bottleneck; any layout/async transform must be token+scope aware."
      rationale_pointers:
        - "Gap_ID: G2"
        - "Primary basis: Tile IR memory model + token-ordered memory ops"
      date: "2026-02-03"
      status: "active"

    - id: "V2-G3"
      decision: "Temporal orchestration needs a first-class, verifiable artifact (async plan) that lowers into Tile IR tokens and (where possible) TMA choices."
      rationale_pointers:
        - "Gap_ID: G3"
        - "Primary basis: token reordering rules + TMA-related hints + DSL constraints"
      date: "2026-02-03"
      status: "active"

    - id: "V3-G1"
      decision: "Structured layout descriptors beyond affine strides are required to carry seed swizzle/generic-conversion ideas into Tile IR without collapsing to pointer tiles."
      rationale_pointers:
        - "Gap_ID: G1"
        - "Primary basis: seed swizzle algebra vs tensor_view stride model + backend tensor-of-pointer limitations"
      date: "2026-02-03"
      status: "active"

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260203"
  stage1_gap_audit: "WP1_20260203"
  stage1_5_toolbox: null
  stage2_directions: null
  stage2_5_novelty_audit: null
  stage3_paper: null
  stage3_assembly_pack: null

OPEN_QUESTIONS:
  - id: "Q0-01"
    text: "Extract Tile IR’s full set of async/TMA-related ops (including any explicit wait/commit primitives) and their token + memory-order interactions."
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir load_view_tko store_view_tko"
      - "site:docs.nvidia.com tile-ir TMA wait token"
      - "Tile IR allow_tma latency OptimizationHints semantics"
    last_update_date: "2026-02-03"

  - id: "Q0-02"
    text: "Clarify how cuTile Python exposes (or hides) Tile IR tokens in practice; is token discipline inferred by compiler or user-controllable anywhere in cuTile?"
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com cutile token"
      - "cuda.tile token make_token join_tokens python"
    last_update_date: "2026-02-03"

  - id: "Q0-03"
    text: "Confirm current hardware support constraints for Tile IR targets (beyond Triton-to-TileIR): Blackwell-only vs partial earlier-arch support; which features are emulated vs diagnosed."
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir supported gpus"
      - "CUDA Tile IR supported architectures sm_90 sm_100"
    last_update_date: "2026-02-03"

  - id: "Q0-04"
    text: "Map seed-paper layout representations (Linear Layouts / CuTe algebra / ISL relations) onto Tile IR view/subview semantics: where expressivity is lost (especially swizzle/non-affine)."
    priority: "medium"
    suggested_queries:
      - "Tile IR tensor_view strides non-affine swizzle"
      - "partition_view dim_map semantics Tile IR"
    last_update_date: "2026-02-03"

  - id: "Q0-05"
    text: "Quantify feasible evaluation axes beyond speed in this stack: correctness/verification coverage, determinism, memory-model litmus tests, robustness to toolchain versions, compile-time cost."
    priority: "medium"
    suggested_queries:
      - "Tile IR memory model litmus tests token order"
      - "cuTile compilation cache constant embedding cost"
    last_update_date: "2026-02-03"

  - id: "Q1-01"
    text: "Locate the authoritative spec text for cuda_tile.load_view_tko / store_view_tko (signatures, token behavior, masking/padding, and how/when TMA is selected)."
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com \"load_view_tko\""
      - "site:docs.nvidia.com \"store_view_tko\""
    last_update_date: "2026-02-03"

  - id: "Q1-02"
    text: "Determine whether Tile IR has (or plans) a non-affine/swizzle-capable view/layout descriptor; if absent, design candidate extension that can lower to TMA or other structured paths."
    priority: "high"
    suggested_queries:
      - "Tile IR swizzle view type"
      - "cuda_tile tensor_view swizzle"
    last_update_date: "2026-02-03"

  - id: "Q1-03"
    text: "Measure/estimate compile/search bottleneck under cuTile constant embedding for realistic tuning loops (layout × schedule × tile sizes)."
    priority: "medium"
    suggested_queries:
      - "cuTile Constant embedding compiled once per value JIT caching"
    last_update_date: "2026-02-03"

NEXT_STAGE_HINT:
  stage: "1.5"
  paste_next: |
    Paste the entire updated STATE from this CONTEXT_CAPSULE.

    Stage 1.5 suggestion ("Toolbox"):
      - Build a legality + temporal-orchestratn toolbox:
        (1) Token-graph rules of thumb + anti-patterns (from Tile IR memory model hazards).
        (2) Minimal litmus-test suite plan (token ordering, weak vs scoped, acquire/release patterns).
        (3) A candidate intermediate representation for "async plans" that lowers to Tile IR tokens.
        (4) A mapping sketch from seed layout representations (F2 / ISL / CuTe categories) into Tile IR view/subview forms, including the swizzle gap.
```

## ARTIFACT_SUMMARY (this stage)

- Produced **Stage 1 Gap Audit** with stable gap IDs **G1–G7**.
- Built an **Axiom vs Hardware vs SOTA** matrix separating **Seed axioms (A)**, **primary/hardware claims (B)**, **SOTA coverage (C)**, and **deficits as labeled inference (D)**.
- Ranked bottlenecks elephant-first: **G2 legality** → **G3 temporal orchestration** → **G1 memory wall/descriptor** → **G5 compile/search** → **G4 subviews** → **G6 availability** → **G7 control gap**.
- Added **top-3** entries to **VERDICT_LEDGER** (V1–V3) tied to gaps G2/G3/G1.
- Promoted durable conclusions into **GLOBAL_MEMORY** (stride-only view gap; partition_view-only; compile-variant explosion; Blackwell gating).
- Updated **OPEN_QUESTIONS** with concrete verification items for async/TMA ops and `load_view_tko`/`store_view_tko` specs.
- Set **ARTIFACT_INDEX.stage1_gap_audit = WP1_20260203**.

## NEXT_STAGE_HINT (copy/paste)

Stage = **1.5**. Paste the **entire STATE** above, then request Stage 1.5 “Toolbox” focused on token-legality + temporal orchestrationapping seed layouts into Tile IR view/subview (and explicitly addressing the swizzle gap).

---
Learn more:
1. [https://arxiv.org/html/2505.23819v3](https://arxiv.org/html/2505.23819v3)
2. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory\_model.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)
3. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)
4. [https://docs.nvidia.com/cuda/cutile-python](https://docs.nvidia.com/cuda/cutile-python)
5. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html)
6. [https://docs.nvidia.com/cuda/cutile-python/execution.html](https://docs.nvidia.com/cuda/cutile-python/execution.html)
7. [https://arxiv.org/pdf/2601.05972v1](https://arxiv.org/pdf/2601.05972v1)
8. [https://arxiv.org/html/2511.10374v1](https://arxiv.org/html/2511.10374v1)
9. [https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
