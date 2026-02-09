# WORK_PRODUCT

## 1) Table — Math/PL/Systems Toolbox (for top gaps: G1–G3)

| Gap_ID | Theory_A | Theory_B | Hybrid | New Artifact | SOTA Hook | MVP Build Path | Metrics (beyond speedup) | Main Risk | Mitigation |
|---|---|---|---|---|---|---|---|---|---|
| **G1** Token legality + token synthesis/minimization + diagnostics | **Token Effect System (TES)**: a type/effect layer over Tile IR that summarizes each op’s memory effects (R/W set, scope, ordering) + token I/O; checker enforces required token order; emits *actionable* diagnostics and suggested token wiring. | **Constraint/Solver Token Minimizer (CSTM)**: build “must-order” constraints (from may-alias + ordering rules) and solve (SMT/ILP) for minimal joins/edges or minimal token critical-path; use unsat cores / minimal witnesses for debugging. | **TES → constraints → greedy MVP; solver stretch**. (Pragmatic: make a correct conservative pass first; then optimize.) | **Token Effect Graph (TEG)**: explicit graph of token-ordered memory ops with “why” labels; plus **Token Normal Form (TNF)** (canonical SSA-like token plumbing). | **Tile IR semantics make this gap real:** program deps do **not** order memory ops; token deps must be used and are preserved. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) Ops section: *currently all memory ops are token-ordered* and ordering is undefined unless connected by tokens; `join_tokens` exists. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) **Triton-to-tile-IR explicitly says** the backend uses an unordered memory model; token semantics exist but “support for memory tokens will require extending the Triton APIs,” and suggests a conservative token-appending strategy. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) Tooling repo emphasizes dialect/bytecode/conformance rather than “token synthesis + minimality + diagnostics.” ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai)) | **MVP (2–4w):** MLIR pass over `cuda_tile.*_tko` ops to (1) build conservative alias classes (same view/pointer; optional user annotations), (2) enforce *single-thread* token legality rules, (3) insert TNF wiring (token chains + `join_tokens` at merges), (4) emit diagnostics with edge reasons.  **Stretch (8–12w):** CFG/loop-aware token SSA (phi-like joins), region-based alias analysis, and a solver-backed minimizer + counterexample generator. | - **token_graph_complexity:** #tokens, #edges, #`join_tokens`, token critical path  <br>- **legality pass rate** on litmus tests derived from Tile IR hazards (incl. weak vs scoped) ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) <br>- **diagnostic quality:** minimal explanation size / unsat-core size; “actionability” rubric <br>- compile-time overhead | Alias analysis imprecision ⇒ (a) false positives, or (b) over-serialization that kills overlap. | Ship in **tiers**: conservative-safe mode first; add annotations; measure precision/recall on **seeded-race** suites; provide “why” edges so users can refine. |
| **G2** Token-legal async/TMA-like pipelining + overlap optimizer | **Token-Aware Software Pipelining (TASP)**: schedule token-ordered memory ops earlier/later across loop iterations; place/rewire tokens to realize chosen overlap; exploit Tile IR view ops + optimization hints (`allow_tma`, `latency`) for TMA-eligible memory ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | **Temporal ILP Scheduler (TILP)**: encode time/stage for ops; constraints from TES (G1), resources, and latency; optimize overlap; output schedule + token wiring. | **Heuristic stage plan + local ILP**: do global list scheduling; run ILP on hot windows; validate with G1 legality checker. | **Temporal Tile Schedule (TTS)** + **Pipeline Certificate** (schedule + token wiring + assumptions). | Tile IR memory ops are reorderable unless constrained by tokens, so scheduling is meaningful. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) View load/store support **TMA-related hints** (`allow_tma`) + `latency`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) Triton-to-tile-IR notes known perf issues + introduces “occupancy” hint and TMA lowering support (but no generalized legality-aware pipeliner described). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) NVIDIA blog explicitly points to **adopting TMA load/store API** as a mitigation for tensor-of-pointer degradation and mentions “forthcoming optimization passes.” ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | **MVP (2–4w):** target 1–2 motifs (e.g., inner-K loop of block GEMM / attention tile) and implement: (1) rewrite tensor-of-pointers → view ops when possible, (2) schedule `load_view_tko` with `allow_tma`/`latency`, (3) compute overlaps with compute ops, (4) verify legality with G1 checker.  **Stretch (8–12w):** modulo scheduling for loops, cost model calibrated via microbenches + occupancy constraints, and auto-tuning of pipeline params. | - **overlap proxy metrics:** memory-stall cycles vs compute (hardware counters if available; else IR-level heuristic) <br>- **TMA-eligibility rate:** fraction of memory ops expressed as view ops with `allow_tma` ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) <br>- schedule robustness across shapes/dtypes <br>- compile-time + schedule-search time <br>- token graph blowup (regression guard) | Hints are “suggestions” and may not reliably map to real overlap; hardware access constraints (Blackwell, driver/toolkit requirements). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Treat as **schedule+legality framework** first (portable correctness); use microbench calibration; provide fallbacks; pin claims to CUDA 13.1-era toolchain and document assumptions. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |
| **G3** End-to-end layout → addressing/descriptor verification for Tile IR lowering | **ISL Layout Witness (ILW)**: represent layouts as integer set relations; compose/invert; derive expected index→address mapping; extract Tile IR view semantics and **prove equivalence** (or produce counterexample). ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | **Bit-level Refinement Checker (BRC)**: compile layout math into bit-vector address functions (works well for $$\mathbb{F}_2$$-style swizzles / bit tricks); compare against emitted Tile IR pointer arithmetic / view indexing via SMT; produce minimal counterexample indices. | **Hybrid certificate:** ISL for affine/stride regions; bit-level SMT for swizzles; unify into one “layout certificate” object. | **End-to-End Layout Certificate (E2E-LC)**: \{layout spec, domain constraints, derived address function / relation, descriptor params + proof/CE\}. | P2 unifies CuTe + Triton layouts via ISL relations and implements verified manipulation ops. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) P1 integrates linear layouts into Triton with a layout engine (selection/propagation), but not a Tile IR-lowering equivalence checker. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) P3 provides categorical foundations (Tuple/Nest) + Python impl aligned with CUTLASS behavior. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) Tile IR explicitly warns that invalid layouts / OOB (incl. “associating an invalid layout with a base pointer”) are UB, motivating verification at the emitted-access level. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) Tile IR views are the primary mechanism for global-memory interaction (`make_tensor_view` + `load_view_tko`/`store_view_tko`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | **MVP (2–4w):** implement ILW for strided layouts + simple swizzles; extract view/partition-view index mapping; check equivalence by (a) sampled testing + (b) ISL simplification where possible; emit counterexamples and “layout hazard” diagnostics (OOB/invalid-layout risk).  **Stretch (8–12w):** broaden to full CuTe+Triton families (ISL+$$\mathbb{F}_2$$ hybrids); synthesize TMA-friendly view shapes/strides when provably equivalent; integrate as validation pass in Triton-to-tile-IR conversion. | - **verification coverage:** % kernels/layouts certified (vs only tested) <br>- **CE quality:** smallest counterexample set; reproducible witness <br>- compile-time overhead <br>- bug-find rate on mutated lowerings <br>- UB hazard prevention rate (OOB/invalid-layout flagged) ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) | Semantic extraction from Tile IR view ops + solver scaling for swizzles. | Use staged checking: sample first, refine when suspicious; cache relations; hybrid fallback (ISL first, SMT only when needed). |

## 2) Toolbox Verdicts (≤10 bullets)

- **G1 (favored): TES→constraints hybrid**. TES gives a *language-level* notion of “what ordering is required,” aligned with the spec’s “program deps don’t order memory ops” rule. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) The constraint backend then supports *minimization* and *diagnostics* (unsat cores / minimal witnesses), turning token bugs from “silent miscompile risk” into actionable reports.
- **G1 novelty delta:** Triton-to-tile-IR explicitly calls out that token support needs API extensions and floats conservative token appending as a future solution. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) A principled legality + minimization + explanation framework is a clear “research-y” delta over “append tokens everywhere.”
- **G2 (favored): Token-Aware Software Pipelining (TASP) with a legality certificate**. Tile IR already exposes `allow_tma`/`latency` hooks on view ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) The defensible angle is: *schedule synthesis constrained by a formal token legality checker*, rather than ad-hoc reordering.
- **G2 novelty delta:** The ecosystem acknowledges missing/coming optimizations (blog’s “forthcoming optimization passes”; repo’s known performance issues). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) A pipeline pass that is (a) analyzable, (b) token-legal-by-construction, and (c) explainable, is a strong systems + PL delta.
- **G3 (favored): ISL witness + bit-level refinement hybrid**. P2 gives the unified relational baseline; P1/P3 give alternative math structures. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) The hybrid approach is the practical way to cover both affine-ish descriptors and bit-twiddly swizzles.
- **G3 novelty delta:** Tile IR explicitly makes “invalid layout / OOB” UB a user/toolchain hazard. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)) An *end-to-end* checker that validates the *emitted* Tile IR access patterns against the chosen layout math is a clear missing bridge between the formal layout papers and the Tile IR toolchain.
- **Cross-gap synergy:** G1’s token certificate becomes a *hard constraint* for G2 scheduling; G3’s layout certificate can drive safe rewrites from tensor-of-pointer to view/TMA-friendly accesses. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))
- **Most defensible proposal “shape”:** one unified “**Certificate Pack**” per kernel: \{Token Certificate (G1), Schedule Certificate (G2), Layout Certificate (G3)\}, each with counterexample-generating diagnostics.

## 3) Citation-needed / Query Plan (remaining killer citations / blockers)

Even with the above, the following are still load-bearing to verify in Stage 2 (direction selection + feasibility):

1. **Does CUDA Tile IR toolchain already include token inference/minimization passes?**  
   - Plan: inspect `N6` source (not just README) for passes mentioning `token`, `join_tokens`, `tko`, “ordering,” “scheduler,” “pipeline”; locate pass registration lists and docs. ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai))  
   - Output needed: “pass exists / does not exist,” limitations, and insertion points.

2. **Exact semantics/guarantees of `optimization_hints`: `allow_tma`, `latency`, and Triton-to-tile-IR `occupancy` hint**  
   - Plan: in Tile IR spec ops section, open the `OptimizationHints` definition and any normative text on whether hints are advisory vs required; find in Triton-to-tile-IR README/docs the precise meaning of `occupancy`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html))  
   - Output needed: what is a contract vs best-effort.

3. **Tile IR “views” semantics extraction for verification**  
   - Plan: in Tile IR ops section, deep-read `make_tensor_view`, `make_partition_view`, and the formal “Tensor View / Partition View” semantics pages referenced by the ops list; write a small semantic extractor. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html))  
   - Output needed: a precise mapping from view parameters + indices → addresses for use in G3 ILW.

4. **Workload selection constraints (Triton-to-tile-IR unsupported ops list, fast-moving)**  
   - Plan: from NVIDIA blog’s “unsupported operations” link + Triton-to-tile-IR docs, snapshot the current list + date, and map to candidate motifs (GEMM, attention, norms). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))

5. **Hardware/toolchain reproducibility scope (beyond Blackwell)**  
   - Plan: cross-check official cuTile prerequisites and repo statements about “13.1 Blackwell-only” and timelines for broader support; decide what the paper will claim vs what will be “future work.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cutile-python/quickstart.html?utm_source=openai))


# CONTEXT_CAPSULE

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
  current_stage: 1.5
  last_updated: "2026-02-04"
  problem_statement_1liner: null
  current_best_thesis: null
  success_criteria:
    - "Reviewer can name closest prior art and see the delta in 30 seconds"
    - "Feasibility is believable without 'handwaving alias analysis / hardware assumptions'"
    - "Evaluation plan could be executed by a grad student in one semester"

GOLDEN_SOURCES:
  - {id: "P1", kind: "seed_paper", title: "Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using \\mathbb{F}_2", url: "https://arxiv.org/html/2505.23819v3", last_verified: "2026-02-04"}
  - {id: "P2", kind: "seed_paper", title: "Modeling Layout Abstractions Using Integer Set Relations", url: "https://arxiv.org/html/2511.10374v1", last_verified: "2026-02-04"}
  - {id: "P3", kind: "seed_paper", title: "Categorical Foundations for CuTe Layouts", url: "https://arxiv.org/pdf/2601.05972v1", last_verified: "2026-02-04"}
  - {id: "N1", kind: "nvidia_primary", title: "CUDA Tile (concept)", url: "https://developer.nvidia.com/cuda/tile", last_verified: "2026-02-04"}
  - {id: "N2", kind: "nvidia_primary", title: "cuTile Python docs", url: "https://docs.nvidia.com/cuda/cutile-python", last_verified: "2026-02-04"}
  - {id: "N3", kind: "nvidia_primary", title: "Tile IR spec", url: "https://docs.nvidia.com/cuda/tile-ir/latest/", last_verified: "2026-02-04"}
  - {id: "N4", kind: "nvidia_primary", title: "Triton-to-TileIR backend (NVIDIA blog)", url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/", last_verified: "2026-02-04"}
  - {id: "N5", kind: "repo", title: "cuTile Python repo", url: "https://github.com/NVIDIA/cutile-python", last_verified: "2026-02-04"}
  - {id: "N6", kind: "repo", title: "cuda-tile repo", url: "https://github.com/NVIDIA/cuda-tile", last_verified: "2026-02-04"}
  - {id: "T1", kind: "repo", title: "Triton-to-TileIR backend repo (incubator)", url: "https://github.com/triton-lang/Triton-to-tile-IR", last_verified: "2026-02-04"}
  - {id: "C1", kind: "community_secondary", title: "TileIR internals blog (secondary; not authoritative)", url: "https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/", last_verified: "2026-02-04"}

GLOBAL_MEMORY:
  notes:
    - {id: "GM-format", text: "Default deliverables: WORK_PRODUCT then CONTEXT_CAPSULE; Stage 3 overrides (LaTeX + capsule in comments).", last_update_date: "2026-02-04"}
    - {id: "GM-evidence", text: "Paper-grade claims require Claim_IDs + evidence pointers; unknowns must be UNVERIFIED + moved to OPEN_QUESTIONS.", last_update_date: "2026-02-04"}

SESSION_MEMORY:
  notes:
    - {id: "SM-0", text: "Stage 0 completed: glossary + baseline map + Claim Ledger v0 + golden snapshots + constraint cliffs created from P1..P3 and NVIDIA Tile stack sources.", last_update_date: "2026-02-04"}
    - {id: "SM-1", text: "Stage 1 completed: Gap Audit produced G1..G8 ranked; top gaps focus on token legality (G1), async/TMA pipelining (G2), and end-to-end layout->addressing verification (G3).", last_update_date: "2026-02-04"}
    - {id: "SM-1.5", text: "Stage 1.5 completed: toolbox table produced for G1–G3 with TES/solver hybrid (G1), token-aware pipelining using allow_tma/latency hints (G2), and ISL+bit-level layout certificates (G3).", last_update_date: "2026-02-04"}

VERDICT_LEDGER:
  items:
    - id: "V-G1"
      verdict: "G1 is a top-3, acceptance-critical gap: need token legality checking + token synthesis/minimization + diagnostics for async/reordering in Tile IR."
      rationale: "Tile IR orders memory via tokens; incorrect/missing token constraints can cause silent races/miscompiles."
      last_update_date: "2026-02-04"
    - id: "V-G2"
      verdict: "G2 is a top-3 gap: need a principled temporal orchestration/pipelining optimizer (async/TMA-like overlap) constrained by legality."
      rationale: "Overlap/performance claims require an analyzable scheduler + legality constraints; TMA hints exist but need principled use."
      last_update_date: "2026-02-04"
    - id: "V-G3"
      verdict: "G3 is a top-3 novelty hook: connect formal layout reasoning (P1/P2/P3) to concrete Tile IR lowering (addressing/descriptor params) with end-to-end checking."
      rationale: "Delta is certifying emitted low-level accesses and enabling safe, performant layout choices."
      last_update_date: "2026-02-04"

    # Stage 1.5 theory selections
    - id: "V-1.5-G1"
      verdict: "Select Hybrid for G1: Token Effect System (legality + explanations) + constraint backend (minimization/unsat-core) with a greedy MVP."
      rationale: "Matches Tile IR spec: program deps don't order memory; tokens must be explicit. Hybrid yields both safety + optimization + debuggability."
      last_update_date: "2026-02-04"
    - id: "V-1.5-G2"
      verdict: "Select Theory A for G2 (Token-Aware Software Pipelining) with local ILP as stretch; always validated by G1 legality checker."
      rationale: "Tile IR exposes allow_tma/latency hints; scheduling must be token-legal and explainable; MVP can target 1–2 motifs."
      last_update_date: "2026-02-04"
    - id: "V-1.5-G3"
      verdict: "Select Hybrid for G3: ISL layout witness (affine/stride) + bit-level refinement (swizzles) to produce end-to-end Layout Certificates."
      rationale: "Covers broad layout families while avoiding solver blowups; directly targets Tile IR UB hazards (invalid layout/OOB)."
      last_update_date: "2026-02-04"

CLAIM_LEDGER:
  items:
    # Layout math baselines
    - {id: "C-P1-01", claim: "P1 presents linear layouts using linear algebra over \\mathbb{F}_2 and integrates them into Triton with a layout engine for layout choice/propagation.", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P2-01", claim: "P2 proposes ISL integer set relations as a unified representation for CuTe layouts and Triton linear layouts, enabling formal analysis/verification and implementing layout ops (composition/inversion/complement) via ISL.", status: "VERIFIED", evidence: ["P2"]}
    - {id: "C-P3-01", claim: "P3 defines categories Tuple and Nest whose morphisms give rise to a tractable class of CuTe layouts, and provides a Python implementation with tests aligned to CUTLASS behavior.", status: "VERIFIED", evidence: ["P3"]}

    # Tile IR / cuTile facts that became load-bearing for G1/G2
    - {id: "C-N3-07", claim: "Tile IR memory ops are token-ordered; ordering is undefined unless connected by tokens; program dependencies do not order memory ops.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-08", claim: "Tile IR provides join_tokens to produce a fresh token depending on multiple input tokens; memory ops consume/produce tokens (e.g., load_*_tko returns a result token).", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-09", claim: "Tile IR view loads/stores support optimization hints including allow_tma and latency.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N2-04", claim: "cuTile memory model permits reordering; without explicit synchronization there is no guaranteed ordering across threads; synchronization is per-element.", status: "VERIFIED", evidence: ["N2"]}

    # Toolchain / SOTA hooks
    - {id: "C-N6-01", claim: "cuda-tile repo provides CUDA Tile dialect, Python bindings, bytecode reader/writer, and conformance test suite aligned with CUDA Toolkit 13.1.", status: "VERIFIED", evidence: ["N6"]}
    - {id: "C-T1-01", claim: "Triton-to-TileIR backend describes an unordered memory model; memory token semantics exist but require Triton API extensions; notes incorrectness risks under aliasing/cross-tile-block reductions; suggests conservative token appending as a possible solution.", status: "VERIFIED", evidence: ["T1"]}
    - {id: "C-T1-02", claim: "Triton-to-TileIR backend includes performance tuning hints (occupancy) and supports lowering Triton host TMA APIs to CUDA Tile IR TMA APIs; documents known performance issues (e.g., small GEMM, tensor-of-pointer patterns).", status: "VERIFIED", evidence: ["T1"]}

    # Inferences (still useful, but keep explicitly UNVERIFIED)
    - {id: "C-I-01", claim: "[INFERENCE] A token-certificate (legality + minimality) can be treated as a compiler contract that enables aggressive reordering/scheduling while preserving correctness under the Tile IR memory model.", status: "UNVERIFIED", evidence: ["N3-derived"]}
    - {id: "C-I-02", claim: "[INFERENCE] A unified certificate pack (Token + Schedule + Layout) is a defensible novelty delta vs existing layout formalisms or existing Tile IR tooling in isolation.", status: "UNVERIFIED", evidence: ["P1/P2/P3/N3/N6/T1-derived"]}

EVAL_PLAN:
  status: "draft"
  metrics:
    - "end_to_end_speedup"
    - "compile_time_overhead"
    - "token_graph_complexity (#tokens, #join_tokens, token edges, token critical path)"   # non-negotiable (Stage 1.5)
    - "diagnostic_quality (minimal explanation size / counterexample quality)"            # non-negotiable (Stage 1.5)
    - "verification_coverage (fraction of kernels/layouts with certificates)"             # non-negotiable (Stage 1.5)
  baselines:
    - "SOTA baseline(s): CUDA Tile / cuTile / Tile IR toolchain"
    - "Conservative token-serialize baseline (single token chain over all memory ops)"    # non-negotiable (Stage 1.5)
    - "Naive program-order baseline (where applicable; expected illegal for Tile IR)"
  workloads: []
  ablations: []
  risks_to_validity:
    - "Hardware availability constraints (Blackwell/CC 10.x/12.x paths) can limit reproducibility."
    - "Spec/toolchain evolution during project can move targets; pin to a spec/toolkit version for claims."

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: "WP1_20260204"
  stage1_5_toolbox: "WP1_5_20260204"
  stage2_directions: null
  stage2_5_novelty_audit: null
  stage3_assembly_pack: null
  stage3_paper: null

OPEN_QUESTIONS:
  - id: "OQ-01"
    question: "What are the best-practice *examples* for token wiring across CFG joins/loops, and are there any official reference patterns beyond the spec text?"
    why_load_bearing: "Determines how to define a sound TNF for real kernels (not just straight-line code)."
    query_plan:
      - "In N3: search for token examples; check release notes for missing examples; scan conformance tests in N6 for token patterns."
  - id: "OQ-02"
    question: "What is the authoritative current support matrix (architectures + toolchain components like tileiras) for CUDA Tile / cuTile, and what parts remain Blackwell-only?"
    why_load_bearing: "Constrains evaluation feasibility and reproducibility claims."
    query_plan:
      - "Cross-check N2/N5 docs + N4 blog + T1 README; pin to CUDA 13.1-era claims."
  - id: "OQ-03"
    question: "What is the exact unsupported/partially-supported operation list for Triton-to-TileIR, and how quickly is it changing?"
    why_load_bearing: "Workload selection and avoiding dead-end benchmarks."
    query_plan:
      - "From N4 blog and T1 repo, locate and snapshot the canonical unsupported ops list; map to candidate motifs."
  - id: "OQ-04"
    question: "Where can compiler passes be inserted/measured in the cuTile Python → Tile IR pipeline (and in Triton-to-TileIR), and what are the stable hook points?"
    why_load_bearing: "Implementation feasibility + compile-time accounting."
    query_plan:
      - "Read N2 interoperability/machine representation; inspect N6 tooling entry points; inspect T1 conversion pass locations."
  - id: "OQ-05"
    question: "Does the current cuda-tile toolchain already implement token-dependency inference/minimization or an async pipeline scheduler? If yes, what are its limitations?"
    why_load_bearing: "Reframe novelty correctly ('new' vs 'limitation of existing')."
    query_plan:
      - "Inspect N6 source for token/schedule passes; search for pass registrations and docs; confirm with tests."
  - id: "OQ-06"
    question: "What debugging/diagnostic capabilities exist today for Tile IR token graphs and memory model errors (beyond conformance pass/fail)?"
    why_load_bearing: "Determines whether diagnostics are a new contribution or an integration/extension."
    query_plan:
      - "Scan N6 conformance output + debug flags; look for visualization/logging; identify gaps."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 2 prompt."
STATE_END

ARTIFACT_SUMMARY:
  - "Produced Stage 1.5 toolbox table for top gaps G1–G3 with Theory A/B/Hybrid, new artifacts, MVP+stretch paths, metrics, risks."
  - "Verified Tile IR token facts: program deps don't order memory ops; memory ops are token-ordered; join_tokens exists; view ops have allow_tma/latency hints."
  - "Pulled in Triton-to-tile-IR repo as SOTA hook: unordered memory model; token support needs API extensions; conservative token appending proposed; TMA lering and occupancy hints documented."
  - "Selected favored approaches: G1 TES+solver hybrid; G2 token-aware pipelining (with legality certificate); G3 ISL witness + bit-level refinement hybrid."
  - "Updated EVAL_PLAN with non-negotiable token-graph complexity + diagnostic quality + verification coverage; added conservative token-serialize baseline."
  - "Added golden source T1 (Triton-to-tile-IR repo) for future SOTA/limitations citations."
  - "Remaining blockers: confirm whether cuda-tile already has token/scheduling passes; extract view semantics precisely for G3; snapshot Triton-to-tile-IR unsupported ops list for workload selection."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 2 prompt."
```

---
Learn more:
1. [7. Memory Model — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)
2. [8. Operations — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html)
3. [GitHub - triton-lang/Triton-to-tile-IR: incubator repo for CUDA-TileIR backend]ps://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)
4. [GitHub - NVIDIA/cuda-tile: CUDA Tile IR is an MLIR-based intermediate representation and compiler infrastructure for CUDA kernel optimization, focusing on tile-based computation patterns and optimizations targeting NVIDIA tensor core units.](https://github.com/NVIDIA/cuda-tile?utm_source=openai)
5. [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
6. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
7. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
8. [Categorical Foundations for CuTe Layouts](https://arxiv.org/pdf/2601.05972v1)
9. [Quickstart — cuTile Python](https://docs.nvidia.com/cuda/archive/13.1.0/cutile-python/quickstart.html?utm_source=openai)
