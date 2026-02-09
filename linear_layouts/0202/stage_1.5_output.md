# WORK_PRODUCT

**Stage 1.5 re-grounding (goal + constraints)**  
- Target **top gaps** from STATE: **G2 (token-order legality), G3 (async/TMA orchestration), G1 (layout descriptor / swizzle gap)**.  
- For each Gap_ID: propose **Theory A**, **Theory B**, and a **Hybrid** (if composable), and map each to:  
  - **(a) IR extension** (new semantic object), **(b) algorithm/pass/solver**, **(c) prototype path**, **(d) metrics beyond speed**.  
- Non‑negotiable per gap: **SOTA hook** → can it compile to **Tile IR**? If yes, why not already done? If no, what **Tile IR limitation** blocks it?  
- Keep all non-obvious factual claims grounded in cited sources (no invented spec details).

## Table: Math/PL Toolbox

| Gap_ID | Theory_A | Theory_B | Hybrid | Mechanism | SOTA Hook | Metrics (beyond speedup) | Main Risk |
|---|---|---|---|---|---|---|---|
| **G2** (token-order legality cliffs) | **Token Effect Inference (TEI)** | **Token Regions / Structured Concurrency (TRSC)** | **TEI + TRSC** (infer by default, annotate when needed) | **A (TEI)**: IR = add explicit **AccessSet / Footprint** objects (e.g., region IDs + tile footprints) attached to memory ops; Algo = may-alias + footprint inference ⇒ synthesize minimal token edges (plus `join_tokens`) to order *conflicting* ops; Proto = post-pass on generated Tile IR: start from conservative single-chain tokenization, then **edge-elide** where non-overlap is proven (transitive reduction + hazard checks).  <br><br> **B (TRSC)**: IR = `tok.seq{}` / `tok.par{}` / `tok.fence(region)` blocks as *semantic* ordering constructs; Algo = region-to-SSA lowering: fork tokens, join via `join_tokens`, emit ordered/scoped ops where needed; Proto = implement as a front-end IR (MLIR dialect or python AST IR) that lowers into Tile IR tokens (`make_token`, `join_tokens`) + memory ordering/scope.  <br><br> **H**: IR = TRSC blocks + optional TEI “auto” mode; Algo = TEI fills in missing ordering, TRSC pins “must-order” edges; Proto = emit **explainable token graph** artifact alongside IR. | **Compiles to Tile IR today** by emitting/rewriting token-ordered ops with explicit tokens (`make_token`, `join_tokens`) and threading tokens through `*_tko` ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  <br><br> **Why not already done (robustly)?** Tile IR explicitly states **program dependencies do not order memory operations**; ordering must be expressed via tokens, so front-ends need nontrivial analysis to avoid either UB (under-order) or serialization (over-order). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | - **Litmus pass rate** for token-order hazards (e.g., “reads from the future” patterns) + scoped acquire/release patterns ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  <br>- **Data-race freedom coverage** (static proof obligations met; races are UB in Tile IR) ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  <br>- **Determinism stability** across toolchain versions/optimization levels (same outputs, same ordering constraints)  <br>- **Token graph complexity** (edges, join fan-in, longest dependency chain)  <br>- **Debuggability**: ability to print “why ordered” explanations  <br>- **Compile-time overhead** of TEI/TRSC passes | - TEI precision: must prove non-overlap; conservative fallback may erase parallelism  <br>- Under-order ⇒ UB/data races (high severity) ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  <br>- Over-order ⇒ kills overlap and makes G3 harder |
| **G3** (temporal orchestration for async/TMA via tokens) | **Async Plan IR (APIR)** | **Constraint-Solved Token Schedule (CSTS)** | **Heuristic APIR + solver/validator** | **A (APIR)**: IR = first-class **AsyncPlan** object: stages (load/compute/store), stage resources, stage-to-stage dependences, and per-access hints (latency class, “TMA allowed/forbidden”); Algo = modulo/pipeline scheduling + token-edge synthesis; emit per-arch optimization hints; Proto = a planner that lowers to Tile IR by: (i) inserting token structure, (ii) setting `optimization_hints` (e.g., `allow_tma`, `latency`, kernel `num_cta_in_cga`), and (iii) optionally producing a verifier report. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  <br><br> **B (CSTS)**: IR = declarative constraint model over token graph + resource/latency constraints; Algo = ILP/SMT: minimize critical path / spill / token live ranges subject to legality constraints (from G2 + dependences); Proto = generate constraints from TEI footprints + view index relations, solve, then emit concrete token threading + hint assignments.  <br><br> **H**: IR = APIR as the user-facing object, CSTS as back-end “hard cases” solver; Algo = heuristic schedule first, solver checks legality + suggests improvements. | **Compiles to Tile IR today**: Tile IR provides token-ordered memory ops and explicit token primitives; it also supports architecture-specific `optimization_hints` including `allow_tma` and `latency` (for `load_view_tko`/`store_view_tko`) plus `num_cta_in_cga` for kernels. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  <br><br> **Why not already done (as a verifiable artifact)?** In current stacks, scheduling + TMA selection is largely **implicit/heuristic** (and `allow_tma` is an optimization hint, not a semantic guarantee). cuTile exposes the knobs (`allow_tma`, `latency`) but not an explicit token/schedule object for external verification. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai)) | - **Overlap quality**: achieved load/compute overlap (stall-cycle proxy), pipeline steady-state %  <br>- **Plan validity**: no token-order hazards; meets acquire/release ordering requirements for scoped ops ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  <br>- **TMA utilization rate** (when allowed) vs fallback rate; sensitivity to `allow_tma`/`latency` hints ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  <br>- **Performance portability**: plan robustness across SM targets (plan deltas vs ByTarget-like specialization) ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai))  <br>- **Compile-time / solver time**; cacheability of plans  <br>- **Reproducibility**: same plan produced across runs (or explainable nondeterminism) | - Cost model wrong ⇒ “fast plan” slower or illegal  <br>- Solver blow-up (CSTS)  <br>- Hardware/backend may ignore hints; APIR may overfit to a backend behavior |
| **G1** (structured layout descriptors beyond affine strides; swizzle gap) | **F2 LinearLayout View (F2LV)** | **ISL Relation View (IRV)** | **Two-level “Affine + Swizzle Decorator” (ASD)** + categorical composition for reasoning | **A (F2LV)**: IR = new view/layout object whose semantics are a **bit-level linear map** (binary matrix acting on bits), inspired by Linear Layouts; includes composition/inversion at layout level; Algo = normalization + equivalence + conversion synthesis; Proto = implement in front-end as a layout engine that emits either: (i) a new Tile IR view-type implementer (future spec extension), or (ii) fallback codegen via pointer tiles for non-lowerable cases. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  <br><br> **B (IRV)**: IR = view described as an **integer set relation** (coordinate → index/offset), leveraging ISL operations for composition/inversion/complement; Algo = ISL-based canonicalization + legality checks + dependence queries; Proto = build a “layout-to-TileIR” translator that can (a) generate structured accesses when representable, else (b) generate gather/scatter pointer tiles. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  <br><br> **H (ASD)**: IR = keep Tile IR’s **tensor_view (shape+strides)** as base + attach a **swizzle decorator** (restricted, checkable class) to cover common non-affine patterns; use F2 for swizzle fragment + ISL for verification; use categorical “layout algebra” to make composition laws explicit and testable. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | **Can it compile to Tile IR?**  <br>- **Exact structured lowering:** **Not with current Tile IR view semantics**, because `tensor_view` is explicitly **strided** (affine: $$base + \sum i_m \cdot stride_m$$) and Tile IR currently only provides a single standardized subview pattern (partitioning into a grid of tiles). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/types.html?utm_source=openai))  <br>- **Fallback lowering:** Yes, by materializing a **tile of pointers** and using `load_ptr_tko`/`store_ptr_tko`, but that “tensor-of-pointers” style carries no structural layout info and is known to be performance-problematic in Tile IR backends; NVIDIA explicitly calls out tensor-of-pointer degradation and recommends descriptor/TMA-style structured loads/stores when possible. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/types.html?utm_source=openai))  <br><br> **Why not already done?** Tile IR’s standardized view path is designed around **shape+strides (+ limited partitioning/permutation)**; richer swizzle/non-affine descriptors would require either (i) a new view-type implementer in Tile IR (spec+compiler work), or (ii) accepting pointer-tile fallbacks. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) | - **Expressivity coverage**: % of target layouts representable (CuTe-style swizzles / Triton linear layouts) ([arxiv.org](https://arxiv.org/html/2511.10374v1))  <br>- **Structured lowering rate**: when can we keep `load_view_tko`/`store_view_tko` vs fall back to pointer tiles (proxy for TMA viability) ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  <br>- **Correctness of conversions**: proof/verification success rate for composition/inversion; aliasing/bijectivity checks ([arxiv.org](https://arxiv.org/html/2511.10374v1))  <br>- **Compile-time cost** (layout reasoning time; ISL/F2 solver time)  <br>- **Robustness**: fewer backend bugs / fewer “layout-related” miscompiles (mirrors motivation in linear layouts work) ([arxiv.org](https://arxiv.org/html/2505.23819v3))  <br>- **Determinism** of layout normalization/canonical forms | - Over-ambitious IRV (full ISL) may be too heavy for kernel-compile latency  <br>- Adding a new Tile IR view implementer is a high coordination cost (spec + compiler + tooling)  <br>- “Hybrid decorator” subset might miss important layouts, causing frequent fallbacks |

## Toolbox Verdicts (≤8)

- **G2 (legality): favor Hybrid (TEI + TRSC).** Tokens are the *only* reliable intra–tile-block-thread ordering mechanism for token-ordered memory ops (program dependencies don’t order), so we need an **inference default** plus an **annotation escape hatch** for hard cases and performance pinning. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
- **G2 implementation bias:** prototype as a **post-pass that relaxes a conservative token chain** (never under-order), because Tile IR already gives `make_token`/`join_tokens` and token threading; this minimizes UB risk while exploring parallelism. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
- **G3 (temporal orchestration): favor Theory A (Async Plan IR) with Hybrid validation.** Tile IR already exposes the knobs (`allow_tma`, `latency`, `num_cta_in_cga`) and token machinery; what’s missing is an explicit, inspectable **plan artifact** that can be verified and compared. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html?utm_source=openai))  
- **G3 “why hybrid”:** keep solver usage bounded—use CSTS only as a validator / local improver to avoid solver blow-ups and keep compile times predictable.  
- **G1 (layout): favor Hybrid ASD (Affine + Swizzle Decorator), backed by F2 + ISL.** Start with a **restricted swizzle decorator** that’s checkable and likely lowerable, but keep the *math backends* (F2/ISL) pluggable for future generalization and proof tooling. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/types.html?utm_source=openai))  
- **G1 SOTA alignment:** prioritize structured descriptors over pointer tiles because “tensor-of-pointer” materialization is an explicit known perf cliff on Tile IR backends; the toolbox should measure **structured-lowering rate** as a first-class metric. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

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

  current_stage: 1.5
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
      text: "Tile IR subviews: currently only partition_view is standardized; seed layout algebras (CuTe ops, ISL relations) require richer subview/indexing forms (overlap/compose) to preserve structure."
      last_update_date: "2026-02-03"
      keywords: ["gap", "subview", "partition_view", "expressivity"]

    - id: "GM-gap-G5-compile-variants"
      text: "HYPOTHESIS/UNVERIFIED: specialization over const tile shapes + hint params can cause auto-tuning/search loops to become compile-cost dominated; needs measurement and cache-aware design."
      last_update_date: "2026-02-03"
      keywords: ["gap", "compile-cost", "autotune", "specialization"]

    - id: "GM-gap-G6-blackwell-gating"
      text: "Verified (NVIDIA blog, 2026-01-30): Triton-to-TileIR backend requires CUDA 13.1+ and Blackwell GPUs; previous GPU architectures expected to be enabled in upcoming CUDA releases."
      last_update_date: "2026-02-03"
      keywords: ["gap", "hardware-support", "blackwell", "cuda-13.1"]

    # Stage-1.5 promotions (Toolbox choices)
    - id: "GM-toolbox-G2-TEI-TRSC"
      text: "Toolbox pick for G2: Hybrid legality approach = Token Effect Inference (auto tokenization / edge-elision from conservative chain) + Token Regions/Structured Concurrency (user/DSL annotations). Outputs an explainable token-graph artifact."
      last_update_date: "2026-02-03"
      keywords: ["toolbox", "G2", "tokens", "effect-system", "structured-concurrency"]

    - id: "GM-toolbox-G3-AsyncPlanIR"
      text: "Toolbox pick for G3: Async Plan IR as a first-class, verifiable schedule object that lowers to Tile IR tokens + optimization_hints (allow_tma/latency/num_cta_in_cga). Use solver/validator only for bounded hard cases."
      last_update_date: "2026-02-03"
      keywords: ["toolbox", "G3", "async", "tma", "tokens", "scheduling"]

    - id: "GM-toolbox-G1-AffinePlusSwizzleDecorator"
      text: "Toolbox pick for G1: Two-level layout descriptor = affine tensor_view base + restricted swizzle decorator. Back it with F2 linear maps and/or ISL relations for verification + canonicalization; lower to structured view ops when possible, else pointer-tile fallback (tracked as a metric)."
      last_update_date: "2026-02-03"
      keywords: ["toolbox", "G1", "layouts", "swizzle", "F2", "ISL", "views"]

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

    # Stage-1.5 new verdicts (toolbox selections)
    - id: "V4-G2"
      decision: "Select Hybrid TEI+TRSC as the core legality toolbox: conservative token chain + proven edge-elision, with structured concurrency annotations as an escape hatch."
      rationale_pointers:
        - "Gap_ID: G2"
        - "Toolbox: TEI (effect/footprint inference) + TRSC (token regions)"
        - "Reason: avoids UB while still enabling parallelism and explainability"
      date: "2026-02-03"
      status: "active"

    - id: "V5-G3"
      decision: "Select Async Plan IR as the core temporal orchestration artifact; lower plans to Tile IR tokens + optimization_hints; use solver/validator only in bounded cases."
      rationale_pointers:
        - "Gap_ID: G3"
        - "Toolbox: AsyncPlanIR + optional CSTS validation"
        - "Reason: makes overlap/scheduling decisions explicit and testable"
      date: "2026-02-03"
      status: "active"

    - id: "V6-G1"
      decision: "Select Two-level layout descriptor (Affine + Swizzle Decorator), backed by F2/ISL reasoning, as the layout expressivity bridge; track structured-lowering vs pointer-tile fallback."
      rationale_pointers:
        - "Gap_ID: G1"
        - "Toolbox: AffinePlusSwizzleDecorator + (F2, ISL) backends"
        - "Reason: minimal IR surface area with growth path toward richer view types"
      date: "2026-02-03"
      status: "active"

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260203"
  stage1_gap_audit: "WP1_20260203"
  stage1_5_toolbox: "WP1_5_20260203"
  stage2_directions: null
  stage2_5_novelty_audit: null
  stage3_paper: null
  stage3_assembly_pack: null

OPEN_QUESTIONS:
  - id: "Q0-01"
    text: "Clarify Tile IR’s async/TMA lowering semantics: allow_tma/latency exist as optimization_hints for load_view_tko/store_view_tko, but what exact structural/layout preconditions trigger TMA, and what (if any) implied waits/barriers are inserted by the backend?"
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir allow_tma latency semantics"
      - "site:docs.nvidia.com tile-ir load_view_tko allow_tma constraints"
      - "site:developer.nvidia.com Triton-to-TileIR TMA descriptor constraints"
    last_update_date: "2026-02-03"

  - id: "Q0-02"
    text: "cuTile Python vs Tile IR tokens: are token dependencies synthesized to preserve an implicit program-order semantics, or are there user-visible controls/diagnostics for token ordering?"
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com/cuda/cutile-python token"
      - "CUDA_TILE_LOGS CUTILEIR token"
      - "tileiras token join_tokens make_token"
    last_update_date: "2026-02-03"

  - id: "Q0-03"
    text: "Confirm current hardware support constraints for Tile IR targets (beyond Triton-to-TileIR): Blackwell-only vs partial earlier-arch support; which features are emulated vs diagnosed."
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir supported gpus"
      - "site:docs.nvidia.com cuda tile ir supported architectures sm_90 sm_100"
    last_update_date: "2026-02-03"

  - id: "Q0-04"
    text: "Map seed-paper layout representations (F2 linear layouts / ISL relations / CuTe categorical algebra) onto Tile IR view/subview semantics: identify exact expressivity loss and minimum viable extension surface."
    priority: "high"
    suggested_queries:
      - "Tile IR view_type implementers roadmap swizzle"
      - "site:docs.nvidia.com tile-ir partition_view dim_map masked semantics"
      - "Tile IR tensor_view non-affine swizzle"
    last_update_date: "2026-02-03"

  - id: "Q0-05"
    text: "Define evaluation axes beyond speed: correctness (litmus), determinism, robustness to toolchain versions, compile-time cost, explainability of token/layout artifacts."
    priority: "medium"
    suggested_queries:
      - "Tile IR memory model hazards litmus tests token order"
      - "cuTile TileCompilerTimeoutError compiler time metrics"
    last_update_date: "2026-02-03"

  - id: "Q1-5-01"
    text: "Tile IR view interface ambiguity: operations spec notes view_type is currently only implemented by partition_view, while semantics discuss tensor_view as a view. Confirm what load_view_tko/store_view_tko accept in practice and how tensor_view is meant to be accessed."
    priority: "medium"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir view_type partition_view only"
      - "site:docs.nvidia.com tile-ir load_view_tko tensor_view"
    last_update_date: "2026-02-03"

  - id: "Q1-5-02"
    text: "Quantify compile/search bottleneck in realistic tuning loops for this stack (layout × schedule × tile sizes × hint params), including cache behavior and timeouts."
    priority: "medium"
    suggested_queries:
      - "site:docs.nvidia.com/cuda/cutile-python TileCompilerTimeoutError"
      - "CUDA_TILE_COMPILER_TIMEOUT_SEC compile time"
      - "cuTile caching compiled kernels"
    last_update_date: "2026-02-03"

ARTIFACT_SUMMARY:
  - "Produced WP1_5_20260203: Math/PL toolbox table covering G2/G3/G1 with Theory A/B/Hybrid, mapped to IR extension + algorithm + prototype path + non-speed metrics."
  - "Selected favored approaches per gap (≤8 verdict bullets) and recorded them as durable GLOBAL_MEMORY notes."
  - "Added three new VERDICT_LEDGER items (V4-G2, V5-G3, V6-G1) capturing Stage-1.5 toolbox selections."
  - "Updated OPEN_QUESTIONS: removed the now-located 'load_view_tko/store_view_tko spec text' as a blocker; replaced with deeper semantic/lowering unknowns (TMA preconditions, cuTile token exposure, view interface ambiguity)."
  - "Kept compile-cost concern budowngraded to hypothesis and pushed measurement into OPEN_QUESTIONS."

NEXT_STAGE_HINT:
  stage: "2"
  paste_next: |
    Paste the entire updated STATE from this CONTEXT_CAPSULE.

    Stage 2 suggestion ("Directions synthesis"):
      Using the selected toolbox verdicts, propose 2–3 coherent research directions that:
        1) integrate G2 legality + G3 async planning + G1 layout descriptors into one end-to-end story,
        2) include a concrete lowering story to current Tile IR (and an explicit 'block-by-TileIR' delta when not possible),
        3) define an evaluation plan (correctness/determinism/compile-time/portability) with minimal benchmark kernels + litmus tests,
        4) state novelty deltas vs cuTile/Tile IR/Triton-to-TileIR + the seed papers (P1–P3).
```

---
Learn more:
1. [8. Operations — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)
2. [7. Memory Model — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)
3. [Performuning — cuTile Python](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai)
4. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
5. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
6. [https://arxiv.org/pdf/2601.05972v1](https://arxiv.org/pdf/2601.05972v1)
7. [5. Type System — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/types.html?utm_sourcai)
8. [8. Operations — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html?utm_source=openai)
9. [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
