# WORK_PRODUCT

## Stage goal + constraints (from STATE / Stage 2 contract)
- Produce **exactly 3** coherent research directions (**D1–D3**), each integrating **G2 legality + G3 async planning + G1 layout descriptors** into an end-to-end story.
- Each direction must include: **Gap_IDs**, **1-sentence thesis**, **core semantic novelty**, **artifact target + interface to Tile IR / cuTile**, **lowering plan**, **evaluation plan** (must include **speed, compile-time, legality pruning, stalls/BW**), and **explicit novelty delta vs N1..N6**.
- Avoid invented technical specifics; unknowns are called out as **blocked-by-TileIR** deltas and/or **Key unknowns**.

---

## 1) Direction Cards

### D1 — **Explainable Token Graph + AsyncPlanIR (“Legality-First Overlap”)**
- **Gap_IDs addressed:** **G2** (token-order legality cliffs), **G3** (temporal orchestration for async/TMA via tokens), **G1** (layout descriptor needed for effect reasoning + avoiding pointer-tile cliffs), plus partial **G5** (compile/search costs via pruning).
- **Thesis (1 sentence):** Build a **token-effect inference + structured concurrency** layer that emits an **explicit Async Plan** and a **minimal, explainable token graph**, then lowers into Tile IR tokens/hints without relying on undefined behavior.
- **Core semantic novelty (what new thing exists?):**
  - **Token Effect Inference (TEI):** computes per-op **read/write footprints + alias sets** (from views/layout descriptors) to derive a **conservative token chain** and then **prove safe edge-elision** (i.e., remove unnecessary order constraints) while staying within token-ordered legality.
  - **Token Regions / Structured Concurrency (TRSC):** user/DSL annotations for “pipeline”, “parallel region”, “join” that are **checked** (not trusted) against TEI, producing an **explainable token-graph artifact**.
  - **AsyncPlanIR (first-class schedule object):** an explicit plan containing **stages, issue points, and required waits/joins**; the compiler’s job is to **prove** the plan is token-legal and then **lower** it.
- **Artifact target (compiler/runtime) + interface with Tile IR / cuTile:**
  - **Artifact:** compiler analysis + transform pipeline that outputs:
    - (a) **Tile IR program** with required tokens and optimization_hints,
    - (b) a **sidecar “token graph + plan” artifact** for debugging/metrics (e.g., JSON/proto; format TBD).
  - **Interface:** sits **above Tile IR (N3)**; can be used by:
    - a standalone front-end that **emits Tile IR** for the **cuda-tile toolchain (N6)**, and/or
    - a wrapper/integration layer that uses **cuTile Python (N2/N5)** as the compilation entry point (exact hooks are a Stage-2.5 verification item).
- **Lowering plan (high-level):**
  1. **Input normalization:** represent all tensor accesses with **Affine+Swizzle layout descriptors** (G1 toolbox) even if the eventual lowering is stride-only or pointer-tile fallback.
  2. **Effect summarization:** compute footprint summaries: \((\text{reads}, \text{writes}, \text{address sets}, \text{scope})\) per memory op; detect may-alias conservatively.
  3. **Conservative tokenization:** emit a **total order** (token chain) over memory ops as a starting point (always legal).
  4. **Edge-elision:** use TEI disjointness/non-alias proofs to remove edges; enforce TRSC constraints (e.g., parallel regions require independence proofs).
  5. **AsyncPlanIR construction:** choose overlap schedule (prefetch distance, staging depth) *as a verifiable object*; insert explicit “join/wait” points in the plan.
  6. **Tile IR emission:** lower plan into Tile IR tokens + optimization_hints (e.g., allow_tma/latency when applicable), and ensure only token-legal reorderings are applied.
  - **Blocked-by-TileIR delta (explicit):**
    - **Token expressivity unknown:** if Tile IR tokens can only represent a **single chain** (vs DAG with join/split), plan overlap may be forced into overly-serial encodings.
    - **TMA semantics unknown:** attaching allow_tma/latency is possible, but the **exact preconditions and implied waits/barriers** are unknown (tracked already as Q0-01).
- **Evaluation plan (must include speed, compile-time, legality pruning, stalls/BW):**
  - **Kernels (minimal set):**
    - K1: **pipelined GEMM** (classic async load → shared → compute overlap),
    - K2: **attention block** (QKᵀ + softmax + PV) with pipelined loads,
    - K3: **transpose / swizzle stress** (layout-driven),
    - K4: **streaming memcpy + compute** (pure overlap microbench).
  - **Speed:** end-to-end throughput/latency vs baselines (cuTile/TileIR pipelines as available).
  - **Compile-time:** total compile wall-time; breakdown into **TEI**, **plan construction**, **IR emission**, **backend compile**; cold vs warm cache.
  - **Legality pruning:** measure:
    - (# candidate schedules/variants) → (# rejected by TEI/TRSC) → (# compiled) → (# valid+fast),
    - time saved by rejecting before backend compile.
  - **Stalls/BW:** profile and report:
    - achieved memory bandwidth vs theoretical,
    - dominant stall classes (compute vs memory vs sync/wait),
    - overlap quality (time waiting on async stages).
  - **Correctness/determinism extras (beyond required):**
    - token litmus tests where missing edges would be UB; system must refuse/repair such schedules;
    - multi-run determinism checks for the same seed/inputs.
- **Novelty delta vs SOTA (explicitly vs N1..N6):**
  - **vs N3 (Tile IR spec):** adds a **first-class effect/region/plan layer** *above* Tile IR that produces an **explainable token-graph artifact** and provably-safe edge-elision; Tile IR remains the lowering target.
  - **vs N2/N5 (cuTile Python + repo):** proposes **user-visible diagnostics + minimality proofs** for token constraints and explicit structured concurrency knobs; if cuTile already synthesizes tokens implicitly, the delta is **exposure + explainability + correctness certificates**.
  - **vs N1/N6 (CUDA Tile concept + cuda-tile repo):** introduces **AsyncPlanIR + TEI/TRSC** as explicit optimization objects, rather than relying on ad-hoc ordering decisions and opaque backend behavior.
  - **vs N4 (Triton→TileIR backend):** provides a semantics-first, inspectable pathway for overlap and legality decisions that can be compared across front-ends.

---

### D2 — **Affine+Swizzle Layout Descriptor → View Lowering (“Structure-Preserving Layouts for Async/TMA”)**
- **Gap_IDs addressed:** **G1** (stride-only view gap / swizzles), **G3** (async/TMA planning depends on layout), **G2** (token legality for pointer-tile fallback), plus **G4** (subview/indexing expressivity).
- **Thesis (1 sentence):** Make layouts a first-class, verifiable object (Affine + restricted Swizzle) so we can **systematically lower** to Tile IR views when possible, **predict when we’ll fall back** to pointer tiles, and feed that into AsyncPlanIR + TEI for both performance and legality.
- **Core semantic novelty (what new thing exists?):**
  - **Two-level layout descriptor (selected toolbox):**
    - Base: affine stride view (Tile IR-friendly),
    - Decorator: restricted **swizzle** (e.g., F₂-linear / small compositional class) with formal semantics.
  - **Canonicalization + equivalence checking:** use **F₂** linear maps and/or **ISL relations** to:
    - normalize layouts,
    - prove two layouts equivalent,
    - derive legal subviews/partitions.
  - **Lowering outcome as a metric:** every access/site gets a **structured-lowering verdict**:
    - “Lowered to view ops” vs “Pointer-tile fallback”, enabling optimization and evaluation.
- **Artifact target (compiler/runtime) + interface with Tile IR / cuTile:**
  - **Artifact:** a layout library + compiler pass that:
    - takes layout descriptors,
    - emits the “best” Tile IR view form available,
    - emits fallback pointer-based forms only when necessary (and records why).
  - **Interface:** supplies layout descriptors to:
    - **AsyncPlanIR** (for scheduling/prefetch decisions),
    - **TEI** (for footprint/alias proofs),
    - and finally emits Tile IR (N3) for compilation via N6 or via cuTile (N2/N5).
- **Lowering plan (high-level):**
  1. Encode swizzle decorator in a **restricted, checkable class** (start small; expand only with proofs).
  2. Canonicalize descriptor (F₂/ISL backend).
  3. Attempt **view lowering**:
     - if representable with available Tile IR view constructs → emit view ops,
     - else emit pointer-tile fallback + explicit address calc (tracked).
  4. Feed resulting access forms into **TEI** to ensure legality and into **AsyncPlanIR** to schedule overlap; attach allow_tma/latency hints only when preconditions are met (currently unknown).
  - **Blocked-by-TileIR delta (explicit):**
    - **View expressivity gap (G4/Q1-5-01):** if Tile IR only standardizes limited subview forms, many swizzle/subview compositions cannot be preserved.
    - **TMA preconditions (Q0-01):** without a precise contract, “TMA-friendly lowering” cannot be guaranteed—must be verified.
- **Evaluation plan (must include speed, compile-time, legality pruning, stalls/BW):**
  - **Kernels (minimal set):**
    - L1: GEMM with **swizzled shared layouts** (layout-sensitive),
    - L2: transpose + reshape chains,
    - L3: attention block where layout affects memory coalescing,
    - L4: gather/scatter microbench within restricted swizzle class.
  - **Speed:** throughput vs baseline layouts (stride-only and/or pointer-tile).
  - **Compile-time:** time spent in layout canonicalization + lowering; impact on variant count.
  - **Legality pruning:** how often TEI rejects pointer-tile fallback plans as unsafe under concurrency; how much search space is cut before compilation.
  - **Stalls/BW:** BW utilization and stall attribution across layouts; correlate with structured-lowering rate.
  - **Extra (high signal for G1):**
    - **Structured-lowering rate** (% lowered to view ops),
    - **Pointer-tile fallback rate** and its performance delta.
- **Novelty delta vs SOTA (explicitly vs N1..N6):**
  - **vs N3 (Tile IR):** introduces a **formal layout semantics layer** (Affine+Swizzle) that can *prove* when a transformation preserves meaning and when Tile IR view lowering is possible, rather than collapsing early to pointer forms.
  - **vs N2/N5 (cuTile Python):** adds explicit, verifiable layout canonicalization plus a tracked “fallback vs structured lowering” decision; even if cuTile supports some layouting, the delta is **formal equivalence + measurable lowering outcomes**.
  - **vs N1/N6 (CUDA Tile / cuda-tile):** provides a bridge from seed-paper layout algebras into Tile IR’s current view ecosystem with explicit expressivity-loss accounting.
  - **vs N4 (Triton→TileIR):** supplies a reusable layout descriptor that can be shared across multiple front-ends, reducing backend-specific layout heuristics.

---

### D3 — **Legality-Aware, Compile-Time-Bounded Autotuning (“Artifacts as Cache Keys”)**
- **Gap_IDs addressed:** **G5** (compile/search bottleneck hypothesis → make it first-class), plus integrates **G2** (legality pruning), **G3** (plan-space search), **G1** (layout-space search), and respects **G6** (hardware gating constraints).
- **Thesis (1 sentence):** Treat **(layout descriptor, async plan, token graph)** as canonical artifacts with stable hashes so we can **prune illegal variants early**, **cache aggressively**, and run autotuning under explicit compile-time budgets without losing correctness.
- **Core semantic novelty (what new thing exists?):**
  - **Artifact-canonicalization pipeline:** canonicalize layouts (F₂/ISL), normalize plans (AsyncPlanIR), and normalize token graphs (TEI-derived minimal constraints) to create **stable cache keys**.
  - **Multi-stage pruning with certificates:**
    - legality (TEI/TRSC),
    - feasibility (resource bounds inferred conservatively),
    - performance heuristics (plan/layout features),
    before invoking expensive backend compilation.
  - **Compile-time budget semantics:** tuning/search stops or adapts based on explicit budgets, not timeouts after the fact.
- **Artifact target (compiler/runtime) + interface with Tile IR / cuTile:**
  - **Artifact:** a tuning driver + cache layer that:
    - generates candidate layouts/plans,
    - runs TEI/TRSC legality checking,
    - emits Tile IR + hints,
    - compiles via N6 and/or via cuTile pipeline,
    - stores binaries + metadata keyed by canonical artifacts.
  - **Interface:** requires only the ability to (a) compile Tile IR, (b) measure compile time, (c) run/profiling harness.
- **Lowering plan (high-level):**
  1. Enumerate candidate **Affine+Swizzle** families (G1) + tile sizes.
  2. For each, enumerate **AsyncPlanIR** schedules (G3) and derive tokens via **TEI/TRSC** (G2).
  3. Prune illegal/pointless variants; cache survivors by canonical key.
  4. Emit Tile IR + optimization_hints; compile; run; profile; update ranking.
  - **Blocked-by-TileIR/toolchain delta (explicit):**
    - requires reliable hooks to record compile-time and reuse compiled artifacts across runs (exact mechanisms need verification in N2/N5/N6).
    - hardware gating (G6) may constrain profiling coverage; must design portable metrics or staged evaluation.
- **Evaluation plan (must include speed, compile-time, legality pruning, stalls/BW):**
  - **Speed:** best-of-budget performance vs baseline (no tuning / naive tuning) on a small kernel suite (reuse D1/D2 kernels).
  - **Compile-time:** total tuning wall-time under fixed budgets; compile-time per variant; cache hit rate across reruns.
  - **Legality pruning:** fraction of candidate variants rejected before compilation; reduction in backend compilation attempts.
  - **Stalls/BW:** verify chosen variants improve BW utilization and reduce dominant stalls; ensure tuning isn’t “optimizing” by accidentally serializing (token over-conservatism).
  - **Extra:** reproducibility of tuning outcome + sensitivity to toolchain version changes.
- **Novelty delta vs SOTA (explicitly vs N1..N6):**
  - **vs N2/N5 (cuTile Python):** elevates compile-time and legality into first-class tuning objectives with artifact hashing/caching; aims to prevent compile-dominated loops.
  - **vs N3 (Tile IR):** adds a tuning/search layer that reasons over tokens/hints rather than being limited to IR-level local transforms.
  - **vs N6 (cuda-tile):** provides systematic, explainable tuning logs (why pruned, what compiled, what stalled) and cross-run caching keyed by semantic artifacts.
  - **vs N4:** complements any front-end lowering by providing a uniform tuning/pruning/caching framework.

---

## 2) Decision matrix

Scales:
- **Novelty / Hardware Relevance:** 1 (low) → 5 (high)  
- **Risk:** 1 (low) → 5 (high)

| Direction | Novelty (1-5) | Hardware Relevance (1-5) | Risk (1-5) | Why it wins | Key unknowns |
|---|---:|---:|---:|---|---|
| **D1: Explainable Token Graph + AsyncPlanIR** | 4 | 5 | 3 | Best balance: attacks top-ranked bottleneck (**G2**) while enabling **G3**; can deliver value even if Tile IR view expressivity stays limited (falls back conservatively). | Token DAG/join expressivity in Tile IR; exact cuTile token exposure; exact TMA semantics for allow_tma/latency. |
| **D2: Affine+Swizzle → View Lowering** | 5 | 5 | 5 | Highest upside if structured lowering enables TMA-friendly accesses and avoids pointer-tile cliffs (**G1**), but depends heavily on current/near-term Tile IR view expressivity. | What view types/ops are actually accepted by load/store view ops; minimum IR extension surface; real TMA preconditions. |
| **D3: Compile-Time-Bounded Autotuning** | 3 | 4 | 4 | Strong practicality if compile/search really dominates (**G5**), but novelty depends on what existing tooling already provides; also requires robust toolchain integration. | Existence/quality of compilation caching hooks; ability to extract profiling counters reliably; hardware availability given gating (G6). |

---

## 3) Stage-2 Verdict (≤10 bullets)

- **Provisional winner: D1 (Explainable Token Graph + AsyncPlanIR).**
- It directly targets the **primary bottleneck (G2 legality/token-order cliffs)** and produces a **verifiable schedule artifact (G3)**, while using **G1 layouts** mainly as an input to effect inference (lower dependency on uncertain Tile IR view features).
- D1 also creates the *debuggable artifacts* (token graph + plan) that de-risk D2 and D3 later (shared instrumentation + legality certificates).
- D2 remains the **highest-novelty/highest-risk** bet; it should be pursued as a follow-on once view/TMA semantics are verified, or as a bounded “restricted swizzle subset” prototype.
- D3 is the “make it shippable” direction, contingent on confirming **compile/search dominance** in real loops (currently a hypothesis).
- **Stage 2.5 must kill/confirm (top items):**
  1. Whether Tile IR already provides (or implicitly implements) TEI/TRSC-like mechanisms and whether tokens can represent the needed **join/split/DAG** structure.
  2. The **exact** semantics and preconditions for **allow_tma/latency** and any implied waits/barriers (for both D1 and D2).
  3. What Tile IR view types are actually usable in practice (tensor_view vs partition_view ambiguity) and what minimal extension surface would be required for D2.
  4. Whether cuTile exposes enough hooks/diagnostics to make token/plan artifacts observable (or if we must build a parallel compilation pipeline).
  5. Quantify compile-time + caching behavior to justify D3 and to ensure D1’s analyses don’t become the new bottleneck.

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

  current_stage: 2
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
    # Workflow invariants
    - id: "GM-format"
      text: "Always output exactly: WORK_PRODUCT then CONTEXT_CAPSULE."
      last_update_date: "2026-02-03"
    - id: "GM-precedence"
      text: "Memory precedence: latest user > session > global; global is advisory."
      last_update_date: "2026-02-03"
    - id: "GM-golden"
      text: "Golden sources remain explicitly listed in STATE; don’t claim specifics unless provided/cited."
      last_update_date: "2026-02-03"

    # Tile IR semantics (as captured so far)
    - id: "GM-tileir-token-order"
      text: "Tile IR memory ops are token-ordered; program order/dependencies do not order memory ops; ordering is undefined unless constrained by tokens."
      last_update_date: "2026-02-03"
    - id: "GM-tileir-weak-scope"
      text: "Tile IR: weak ops assume no concurrent access and cannot be used for cross-thread communication; scoped + ordered ops required for synchronization; data races are undefined behavior."
      last_update_date: "2026-02-03"
    - id: "GM-tileir-opaque-tile-layout"
      text: "Tile IR / cuTile: tiles are pure/immutable values without defined storage; structured memory access is expressed via views (shape+stride) or pointer tiles."
      last_update_date: "2026-02-03"

    # Gap ranking + key gap statements
    - id: "GM-gap-top"
      text: "Gap ranking: (1) token-order legality cliffs (G2), (2) temporal orchestration for async/TMA via tokens (G3), (3) structured layout descriptor gap (G1), then compile/search cost (G5)."
      last_update_date: "2026-02-03"
    - id: "GM-gap-G1-stride-only-view"
      text: "Tile IR tensor_view is affine stride-based; seed-style bit-level swizzles/non-affine layouts can’t be represented directly → pointer-tile fallback and potential 'tensor-of-pointers' perf cliffs."
      last_update_date: "2026-02-03"
    - id: "GM-gap-G4-subview-limited"
      text: "Tile IR subviews: partition_view standardized; richer subview/indexing forms likely needed to preserve seed layout algebra structure."
      last_update_date: "2026-02-03"
    - id: "GM-gap-G5-compile-variants"
      text: "HYPOTHESIS/UNVERIFIED: specialization over const tile shapes + hint params may make tuning loops compile-cost dominated; needs measurement and cache-aware design."
      last_update_date: "2026-02-03"
    - id: "GM-gap-G6-blackwell-gating"
      text: "Verified (NVIDIA blog, 2026-01-30): Triton-to-TileIR backend requires CUDA 13.1+ and Blackwell GPUs; earlier architectures expected in upcoming CUDA releases."
      last_update_date: "2026-02-03"

    # Toolbox verdicts (Stage 1.5)
    - id: "GM-toolbox-G2-TEI-TRSC"
      text: "G2 toolbox: Hybrid TEI (auto tokenization + explainable edge-elision) + TRSC (token regions/structured concurrency annotations). Output: explainable token-graph artifact."
      last_update_date: "2026-02-03"
    - id: "GM-toolbox-G3-AsyncPlanIR"
      text: "G3 toolbox: Async Plan IR as a first-class, verifiable schedule object that lowers to Tile IR tokens + optimization_hints; bounded solver/validator only for hard cases."
      last_update_date: "2026-02-03"
    - id: "GM-toolbox-G1-AffinePlusSwizzleDecorator"
      text: "G1 toolbox: two-level layout descriptor = affine tensor_view base + restricted swizzle decorator; backed by F2/ISL reasoning; lower to view ops when possible else pointer-tile fallback (tracked as a metric)."
      last_update_date: "2026-02-03"

    # Stage 2: 3 directions (compact)
    - id: "GM-dir-D1"
      text: "D1: Explainable Token Graph + AsyncPlanIR (Legality-First Overlap): TEI/TRSC derives minimal token edges + verifiable async plan; lowers to Tile IR tokens/hints with sidecar artifacts."
      last_update_date: "2026-02-03"
    - id: "GM-dir-D2"
      text: "D2: Affine+Swizzle Layout Descriptor → View Lowering: formal layout canonicalization (F2/ISL) to preserve structure in Tile IR views; integrates TEI + AsyncPlan; track pointer-tile fallback explicitly."
      last_update_date: "2026-02-03"
    - id: "GM-dir-D3"
      text: "D3: Legality-Aware Compile-Time-Bounded Autotuning: canonicalize (layout, plan, token graph) as cache keys; prune illegal variants early; treat compile-time as an objective."
      last_update_date: "2026-02-03"

SESSION_MEMORY:
  notes: []

VERDICT_LEDGER:
  items:
    - id: "V1-G2"
      decision: "Treat token-order legality as the primary bottleneck; any layout/async transform must be token+scope aware."
      rationale_pointers: ["Gap_ID: G2"]
      date: "2026-02-03"
      status: "active"
    - id: "V2-G3"
      decision: "Temporal orchestration needs a first-class, verifiable artifact (async plan) that lowers into Tile IR tokens and (where possible) TMA choices."
      rationale_pointers: ["Gap_ID: G3"]
      date: "2026-02-03"
      status: "active"
    - id: "V3-G1"
      decision: "Structured layout descriptors beyond affine strides are required to carry seed swizzle/generic-conversion ideas into Tile IR without collapsing to pointer tiles."
      rationale_pointers: ["Gap_ID: G1"]
      date: "2026-02-03"
      status: "active"
    - id: "V4-G2"
      decision: "Select Hybrid TEI+TRSC as the core legality toolbox: conservative token chain + proven edge-elision, with structured concurrency annotations as an escape hatch."
      rationale_pointers: ["Gap_ID: G2", "Toolbox: TEI+TRSC"]
      date: "2026-02-03"
      status: "active"
    - id: "V5-G3"
      decision: "Select Async Plan IR as the core temporal orchestration artifact; lower plans to Tile IR tokens + optimization_hints; use solver/validator only in bounded cases."
      rationale_pointers: ["Gap_ID: G3", "Toolbox: AsyncPlanIR"]
      date: "2026-02-03"
      status: "active"
    - id: "V6-G1"
      decision: "Select Two-level layout descriptor (Affine + Swizzle Decorator), backed by F2/ISL reasoning, as the layout expressivity bridge; track structured-lowering vs pointer-tile fallback."
      rationale_pointers: ["Gap_ID: G1", "Toolbox: AffinePlusSwizzleDecorator"]
      date: "2026-02-03"
      status: "active"
    - id: "V7-Stage2"
      decision: "Stage-2 provisional winner: D1 (Explainable Token Graph + AsyncPlanIR)."
      rationale_pointers:
        - "Why: best balance of novelty/hardware relevance/risk; directly targets G2 and enables G3 with fewer dependencies on Tile IR view expressivity."
        - "D2 high upside but blocked by view/TMA semantics; D3 contingent on measured compile-time dominance."
      date: "2026-02-03"
      status: "active"

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260203"
  stage1_gap_audit: "WP1_20260203"
  stage1_5_toolbox: "WP1_5_20260203"
  stage2_directions: "WP2_20260203"
  stage2_5_novelty_audit: null
  stage3_paper: null
  stage3_assembly_pack: null

OPEN_QUESTIONS:
  - id: "Q0-01"
    text: "Clarify Tile IR’s async/TMA lowering semantics: what exact structural/layout preconditions trigger TMA when allow_tma/latency hints are present, and what (if any) implied waits/barriers are inserted."
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir allow_tma latency semantics"
      - "site:docs.nvidia.com tile-ir load_view_tko allow_tma constraints"
      - "site:developer.nvidia.com Triton-to-TileIR TMA descriptor constraints"
    last_update_date: "2026-02-03"
  - id: "Q0-02"
    text: "cuTile Python vs Tile IR tokens: are token dependencies synthesized implicitly, and are there user-visible controls/diagnostics for token ordering and scope/ordering strength?"
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com/cuda/cutile-python token"
      - "site:github.com/NVIDIA/cutile-python token"
    last_update_date: "2026-02-03"
  - id: "Q0-03"
    text: "Confirm current hardware support constraints for Tile IR targets beyond Triton-to-TileIR: which GPUs/SM versions are supported, and which features are gated/emulated/diagnosed."
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir supported gpus"
      - "site:docs.nvidia.com cuda tile ir supported architectures"
    last_update_date: "2026-02-03"
  - id: "Q1-5-01"
    text: "Tile IR view interface ambiguity: confirm what load_view_tko/store_view_tko accept in practice and how tensor_view vs partition_view are meant to be accessed/implemented."
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir load_view_tko tensor_view"
      - "site:docs.nvidia.com tile-ir view_type partition_view"
    last_update_date: "2026-02-03"
  - id: "Q2-01"
    text: "Token expressivity for D1: can Tile IR represent token DAGs (multiple tokens, join/split), or only linear token chains? What are the supported token-manipulation idioms (if any)?"
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir token join split"
      - "site:docs.nvidia.com tile-ir make_token join_tokens"
    last_update_date: "2026-02-03"
  - id: "Q0-04"
    text: "Map seed-paper layout representations (F2 linear layouts / ISL relations / CuTe algebra) onto Tile IR view/subview semantics; identify exact expressivity loss and minimum viable extension surface for D2."
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir partition_view dim_map masked semantics"
      - "Tile IR tensor_view non-affine swizzle"
    last_update_date: "2026-02-03"
  - id: "Q1-5-02"
    text: "Quantify compile/search bottleneck in realistic tuning loops (layout × schedule × tile sizes × hint params), including caching behavior and timeout modes, to justify/shape D3."
    priority: "medium"
    suggested_queries:
      - "site:docs.nvidia.com/cuda/cutile-python TileCompilerTimeoutError"
      - "sigithub.com/NVIDIA/cutile-python cache compile"
    last_update_date: "2026-02-03"

ARTIFACT_SUMMARY:
  - "Produced WP2_20260203: three Direction Cards (D1–D3) integrating G2 legality + G3 async planning + G1 layout descriptors with explicit lowering + evaluation plans."
  - "Added a decision matrix scoring novelty/hardware relevance/risk with key unknowns."
  - "Recorded Stage-2 provisional winner (D1) and added a new verdict ledger entry (V7-Stage2)."
  - "Updated GLOBAL_MEMORY with compact summaries of –D3."
  - "Refined OPEN_QUESTIONS with a new blocker on Tile IR token expressivity (DAG vs chain) needed for D1."

NEXT_STAGE_HINT:
  stage: "2.5"
  paste_next: |
    Paste the entire updated STATE from this CONTEXT_CAPSULE.

    Stage 2.5 suggestion ("Novelty + evidence audit"):
      Use web browsing on GOLDEN_SOURCES (N1..N6, P1..P3) to:
        1) Verify which pieces of D1/D2/D3 are already present (or explicitly planned) in CUDA Tile / cuTile / Tile IR / Triton-to-TileIR.
        2) Resolve the top ON_QUESTIONS (Q0-01, Q1-5-01, Q2-01, Q0-02) with direct quotes/snippets (<=25 words each).
        3) Produce an evidence-backed novelty table:
             Claim | Direction | Source supports? | Source contradicts? | Verdict (Novel / Not novel / Unclear) | Fix/reframe
        4) Update the provisional winner if any key claim is not novel or is blocked by immutable constraints.
```
