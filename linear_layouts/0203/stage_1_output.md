# WORK_PRODUCT

## 1) Table: Axiom‑vs‑Hardware‑vs‑SOTA Matrix (ranked elephant‑first)

> **Reading note:** “SOTA_System(s)” is constrained to the provided stack IDs. Where a “missing” might already exist but is undocumented in the provided sources, I mark that as **gap‑candidate** and explicitly list the evidence needed to confirm.

| Gap_ID | Seed_Axiom | Hardware/Stack fact | SOTA_System(s) | What SOTA Solves | Why Still Fails | Required Extension | Evidence needed |
|---|---|---|---|---|---|---|---|
| **G1** | **Token dependencies, not program deps, enforce memory ordering;** token deps are preserved while program deps may be optimized away; **weak ops can’t be used for inter‑thread communication.** | Tile IR Spec **13.1 release notes** list known issues (incl. missing examples / incomplete encodings / limited atomics), implying correctness guidance is incomplete and implementers can diverge. | **N3, N2, N6** | N3 specifies token/scopes model; N2 exposes a memory model at cuTile level; N6 provides tooling + conformance tests. | **No end‑to‑end “legality checker + token synthesis/minimization” story is explicitly provided**: (a) aggressive async overlap requires *constructing* correct token edges; (b) missing examples increases miscompile risk; (c) conformance may not cover novel reorderings/schedules. | **(1)** Static legality checker for token/scopes/weak‑op constraints, **(2)** token‑graph synthesis + minimization (avoid “token spaghetti”), **(3)** diagnostic output that explains *why* illegal. | 1) Confirm whether N6 already includes any token‑inference/minimization passes (repo+docs audit). 2) Build a suite of “counterexample” kernels where missing token edges cause observable wrongness. 3) Show current toolchain is either (i) unsafely permissive or (ii) overly conservative. |
| **G2** | Tile kernels execute as parallel tile blocks; compiler maps to hardware threads ⇒ compiler *can* orchestrate time (pipelining) **if it can reason about ordering constraints.** | Triton→TileIR backend: **CUDA 13.1+ + Blackwell requirement**; limitations: incomplete op support; tensor‑of‑pointer slowdowns; suggested mitigations reference SIMT fallback / adopting TMA descriptors. | **N4, N1, N3, N2** | Working compilation path to Tile IR exists (N4); Tile IR gives a portable VM + ops (N3); cuTile claims auto use of tensor cores / memory accelerators (N2). | **Temporal orchestration gap:** no explicit, analyzable schedule search / pipelining optimizer is specified; current limitations can block overlap; legality depends on G1; performance portability unclear. | **Async/TMA pipelining optimizer**: schedule search constrained by legality (G1) + a cost model; robust fallback when ops unsupported; schedule/overlap introspection. | 1) Identify which async/TMA‑like ops exist and their token IO/wait semantics (OQ‑01). 2) Measure “baseline overlap” vs “achievable overlap” on representative motifs. 3) Confirm what scheduling decisions are currently fixed vs tunable in the toolchain. |
| **G3** | Layout formalisms (F2 / ISL / categorical) enable composition/inversion and are motivated by real bug pressure; layout correctness is a first‑order problem, not a polish task. | cuTile: arrays are strided global objects; tiles are immutable kernel‑only values with **power‑of‑two compile‑time** dimensions; Tile IR is a low‑level portable VM (not PTX SIMT). | **P1, P2, P3, N2, N3, N4** | P1: layout representation + Triton integration; P2: unifies CuTe+Triton layout abstractions via ISL relations; P3: formal foundation + test alignment to CUTLASS behavior; N2/N3/N4 provide the Tile IR stack. | **Cross‑layer “layout→addressing/TMA descriptor” correctness is not end‑to‑end guaranteed**: layout proofs/relations don’t automatically certify the concrete address calculations and memory‑accelerator descriptors emitted in Tile IR lowering. | **Proof‑carrying (or checker‑verified) lowering** tying layout algebra to emitted Tile IR address computations/descriptor parameters; equivalence checker between high‑level layout intent and low‑level accesses; optional auto‑layout selection informed by hardware ops. | 1) Pin down where layout metadata lives and where it’s lost in the lowering pipeline (OQ‑04). 2) Demonstrate at least one real/representative “layout mismatch” failure mode (bug, miscompile, or missed perf). 3) Show checker can validate known‑good layouts (avoid false positives). |
| **G4** | If layout bugs are common (P1) and ordering is token‑explicit (N3), then **debuggability** is a major adoption and research‑evaluation risk. | Tile IR Spec known issues include missing detailed memory‑model examples; conformance exists but may not produce human‑actionable root causes. | **N6, N3, P1, P3** | Conformance tests (N6) and improved layout system outcomes (P1) exist; P3 has tests aligned to CUTLASS behavior. | Failures can be **non‑local and non‑obvious** (token edges, scopes, layout conversions); current messages often become “compiler says no/segfault/wrong answer” without minimal counterexample or explanation. | **Diagnostics toolkit**: token‑graph visualization + “why illegal” witness, property‑based generator/shrinker for kernels, structured error taxonomy linking to spec clauses. | 1) Audit what diagnostics currently exist (N6 tooling). 2) Collect a corpus of failures + measure time‑to‑root‑cause without/with new tools. 3) Validate diagnostic precision (avoid noisy warnings). |
| **G5** | Automatic layout selection/propagation exists (P1), but combining **layout × schedule × legality checking** creates a combinatorial search and compile‑time risk. | Triton→TileIR is source‑based compilation; bytecode stability exists, but **compile latency** becomes user‑visible in JIT workflows. | **P1, N4, N6** | P1 gives a precedent for automated layout reasoning; N6 provides a place to implement passes; N4 implies a practical compile pipeline where overhead matters. | Without bounded search/caching, “better performance” can be negated by unacceptable compile time; overly conservative heuristics can kill performance. | **Bounded optimization framework**: cost model, caching/memoization, compile‑time instrumentation, early‑exit policies, and ablation‑friendly knobs. | 1) Baseline compile‑time breakdown today (wall time per pass). 2) Demonstrate scaling behavior as kernel param space grows. 3) Show predictable compile‑time envelope with acceptable perf loss. |
| **G6** | Weak ops can’t communicate across threads; compiler may assume weak tiles aren’t concurrently accessed ⇒ unsafe patterns must be flagged or rewritten. | cuTile permits reordering and defines atomic order/scope; Tile IR has scopes + tokens; limited atomics noted in spec release notes. | **N3, N2** | Semantics exist: tokens/scopes, atomic order/scope, weak vs strong notions. | **No explicit static “weak‑op misuse” checker** is described; absent tooling, users can write code that “seems to work” but is illegal and breaks under optimization. | **Race/legality analysis for weak ops** + auto‑upgrade (insert stronger ops/tokens/fences) or diagnostic warnings. | 1) Build litmus tests that separate legal vs illegal weak‑op communication. 2) Confirm current compiler transformations can trigger the bad behaviors. 3) Validate checker’s false‑positive rate. |
| **G7** | Both Linear Layouts and cuTile tile constraints point at **power‑of‑two** shapes as a first‑order restriction; masking is the stated mitigation. | cuTile: tiles have power‑of‑two compile‑time dimensions; Linear Layouts notes power‑of‑two limitation (mitigable by masking). | **N2, P1** | You can handle irregular shapes via masking/remainders; layouts can still be represented within the restricted domain. | Masking/remainder handling can impose performance cliffs and complicate reasoning; increases search space and makes “proof of equivalence” harder for mixed cases. | **Remainder‑tile strategy / mixed‑radix extension** integrated into layout algebra + scheduling; explicit treatment of tails in correctness + cost model. | 1) Empirical performance on irregular shapes vs padded/power‑of‑two. 2) Demonstrate correctness equivalence across tail handling. 3) Confirm what cuTile/Tile IR already supports for tails. |
| **G8** | Portability is a stated goal (VM + bytecode stability), but evaluation requires real, accessible execution targets. | Triton→TileIR currently requires **Blackwell + CUDA 13.1+**, potentially narrowing reproducibility/availability. | **N4, N3, N1** | Clear “where this is going” portability story; an actual implementation exists on a modern target. | Research evaluation risks becoming “works only on X”; hard to compare across architectures; baseline availability constraints can block experiments. | **Evaluation portability plan**: SIMT fallback baseline, or an emulation/simulation harness, or a sharply scoped claim set with reproducible environment packaging. | 1) Authoritative hardware support scope (OQ‑02). 2) Confirm minimal hardware needed for key motifs. 3) Establish comparable baselines (PTX/SIMT vs Tile IR). |

---

## 2) Table: Gap → Measurement Hook

| Gap_ID | What to measure | Minimal benchmark motif | Baseline(s) | Expected delta | Confounders | Threat-to-validity note |
|---|---|---|---|---|---|---|
| **G1** | (a) % illegal schedules detected, (b) false pos/neg vs oracle suite, (c) token‑graph size reduction vs naive, (d) compile‑time of checker | Microkernels with deliberate missing/incorrect token edges; include weak‑op misuse cases and cross‑scope cases | Current toolchain behavior (status quo N6 pipeline); “naive always‑serialize” token baseline; “naive append tokens everywhere” baseline | Fewer latent correctness bugs; enables safe reordering that was previously blocked or unsafe; smaller token graphs with same legality | Hard to build an oracle for concurrency; nondeterministic failures | If failures don’t manifest reliably on hardware, need differential testing / instrumented semantics; risk of overfitting to the test suite |
| **G2** | (a) End‑to‑end runtime, (b) memory stall cycles / achieved bandwidth, (c) overlap metric (copy/compute concurrency), (d) schedule search time | Pipelined tile GEMM‑like kernel with async global→shared stage and compute stage; optional epilogue | Serialized schedule; status‑quo cuTile/Triton→TileIR lowering; hand‑tuned schedule if available (as an upper bound) | Reduced memory stalls; improved utilization; maintain correctness under aggressive overlap | Power/clock variability; occupancy limits; differing register/shared usage due to schedule | Results may be hardware‑specific (Blackwell‑only initially); if async ops are limited/unstable, motif may change mid‑project |
| **G3** | (a) Layout equivalence pass rate, (b) mismatch detection (found bugs), (c) perf impact of auto‑layout vs default, (d) compile overhead of checking | Layout‑sensitive kernels: transpose+GEMM fusion, attention‑style block layouts, swizzle/stride conversions | Default layouts (cuTile strided arrays); Triton Linear Layouts engine alone (if applicable); “no checking” pipeline | Catch layout/addressing mismatches early; improve perf via better layout selection; preserve correctness | Algorithmic changes can masquerade as layout improvements; benchmark variance | If real “layout mismatch” bugs are rare, need seeded bug injection to validate detection without overstating real‑world prevalence |
| **G4** | (a) Time‑to‑root‑cause (median) on curated failures, (b) diagnostic precision/recall, (c) reduction in “unknown failure” category | “Failure zoo”: 20–50 small kernels failing due to token/scopes/layout; include minimized counterexamples | Current compiler diagnostics; conformance test output as‑is | Less developer time; higher fraction of failures explained with a specific actionable cause | User expertise variance; learning effects | Debuggability claims can be dismissed as anecdotal unless measured carefully (blinded tasks / fixed rubric) |
| **G5** | (a) compile time (p50/p95), (b) passes time breakdown, (c) search nodes explored, (d) perf vs compile‑time Pareto frontier | Parameter sweep: same kernel family across shapes/layout candidates/schedule knobs | Status‑quo pipeline; “bounded search off” vs “on”; caching off vs on | Predictable compile envelope with controllable perf tradeoff | Cache effects; machine load; driver JIT variance | If compile is partly in closed components, attribution may be hard; must isolate what you control in N6 pipeline |
| **G6** | (a) # of weak‑op misuse patterns detected, (b) runtime overhead after auto‑upgrade, (c) correctness under optimization | Litmus tests representing communication patterns (producer/consumer across scopes) | No checker; “always strong ops” baseline | Prevent silent races; minimal overhead when legal weak ops retained | Hard to map IR checks to real hardware outcomes; optimizer differences | If the spec evolves, “misuse” definitions can shift; keep checker aligned to a specific spec version (documented) |
| **G7** | (a) runtime on irregular shapes, (b) overhead vs padded, (c) correctness under tail handling, (d) complexity impact on layout+schedule search | GEMM/conv motifs with non‑power‑of‑two dimensions; tail tiles + masks | Masking/remainder handling as status quo | Better performance on tails or reduced overhead for irregular shapes; clearer correctness story | Padding can hide the problem; kernel selection impacts results | Reviewers may argue tails are “engineering”; must tie to correctness + portability + compile/search complexity to keep it research‑grade |
| **G8** | (a) reproducibility (can others run?), (b) cross‑arch comparability, (c) fallback overhead | Same kernel motifs run on TileIR path vs SIMT/PTX path (when available) | PTX/SIMT baseline; TileIR baseline on supported hardware | Broader evaluability; clearer baseline comparisons | Different backends change codegen beyond scheduling/layout | If fallback changes semantics, comparisons become apples‑to‑oranges; must define a strict equivalence contract |

---

## 3) Stage‑1 Verdict (≤12 bullets)

- **Top‑3 gaps (acceptance‑risk dominant):**
  - **G1 (Token legality + token synthesis/minimization + diagnostics)**: dominates because *correctness is non‑negotiable* and Tile IR explicitly decouples program deps from memory ordering; a single miscompile narrative can sink the paper.
  - **G2 (Async/TMA pipelining / temporal orchestration under legality)**: dominates because *performance claims require overlap*; without a principled scheduler/cost model, results risk being “hand‑tuned demo” or “no speedup.”
  - **G3 (End‑to‑end layout→addressing verification across the Tile IR lowering boundary)**: dominates novelty because it is the cleanest delta vs P1/P2/P3 alone: **connecting formal layout reasoning to the concrete Tile IR execution + memory model**.
- **Secondary but still load‑bearing:** **G5 compile/search cost** and **G4 debug/diagnostics**—reviewers increasingly demand *usability + predictable compilation*, not just peak speed.
- **Claims/unknowns that must be VERIFIED before Stage 2 directions (to avoid reinventing existing tooling or proposing impossible ops):**
  - **OQ‑01:** exact Tile IR op list for async/TMA‑like operations and their **token IO / wait semantics** (directly constrains G1+G2).
  - **OQ‑03:** authoritative unsupported/partial op list for Triton→TileIR (workload selection risk).
  - **OQ‑04:** the actual lowering pipeline boundaries (where to implement passes; how to measure compile overhead).
  - **NEW:** whether **N6 already contains** token‑dependency inference/minimization or scheduling passes (if it exists, the gap becomes “limitations of existing pass,” not “absence”).
  - **OQ‑02:** hardware support scope beyond Blackwell mention (evaluation feasibility and baseline plan).
- **Decision for Stage 2 direction search:** anchor directions around **(G1 + G2)** as the “safety + speed” core, and use **G3** as the “math‑backed novelty hook” tying P1/P2/P3 into the Tile IR world.

---

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
  current_stage: 1
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
  - {id: "C1", kind: "community_secondary", title: "TileIR internals blog (secondary; not authoritative)", url: "https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/", last_verified: "2026-02-04"}

GLOBAL_MEMORY:
  notes:
    - {id: "GM-format", text: "Default deliverables: WORK_PRODUCT then CONTEXT_CAPSULE; Stage 3 overrides (LaTeX + capsule in comments).", last_update_date: "2026-02-04"}
    - {id: "GM-evidence", text: "Paper-grade claims require Claim_IDs + evidence pointers; unknowns must be UNVERIFIED + moved to OPEN_QUESTIONS.", last_update_date: "2026-02-04"}

SESSION_MEMORY:
  notes:
    - {id: "SM-0", text: "Stage 0 completed: glossary + baseline map + Claim Ledger v0 + golden snapshots + constraint cliffs created from P1..P3 and NVIDIA Tile stack sources.", last_update_date: "2026-02-04"}
    - {id: "SM-1", text: "Stage 1 completed: Gap Audit produced G1..G8 ranked; top gaps focus on token legality (G1), async/TMA pipelining (G2), and end-to-end layout->addressing verification (G3).", last_update_date: "2026-02-04"}

VERDICT_LEDGER:
  items:
    - id: "V-G1"
      verdict: "G1 is a top-3, acceptance-critical gap: need token legality checking + token synthesis/minimization + diagnostics for async/reordering in Tile IR."
      rationale: "Tile IR orders memory via tokens; incorrect or missing tokens can cause silent races/miscompiles; spec issues increase risk."
      last_update_date: "2026-02-04"
    - id: "V-G2"
      verdict: "G2 is a top-3 gap: need a principled temporal orchestration/pipelining optimizer (async/TMA-like overlap) constrained by legality."
      rationale: "Without an analyzable scheduler + cost model, overlap/performance claims risk being ad-hoc or blocked by op limitations."
      last_update_date: "2026-02-04"
    - id: "V-G3"
      verdict: "G3 is a top-3 novelty hook: connect formal layout reasoning (P1/P2/P3) to concrete Tile IR lowering (addressing/descriptor params) with end-to-end checking."
      rationale: "Separately, layout math exists; the missing delta is certifying emitted low-level accesses and enabling safe, performant layout choices."
      last_update_date: "2026-02-04"

CLAIM_LEDGER:
  items:
    - {id: "C-P1-01", claim: "Linear Layouts models tensor layouts using linear algebra over F2.", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P1-02", claim: "Linear Layouts represents layouts as binary matrices acting on bits, enabling generic layout definitions and conversions.", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P1-03", claim: "Linear Layouts is integrated into Triton and includes a layout engine that can automatically choose and propagate layouts.", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P1-04", claim: "Linear Layouts reports reduced engineering effort and bug fixes in Triton's legacy layout system.", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P1-05", claim: "Linear Layouts claims 12% of Triton GitHub bugs are layout-related.", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P1-06", claim: "Linear Layouts notes a primary limitation: restriction to power-of-two shapes (mitigable via masking).", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P2-01", claim: "ISL integer set relations are proposed as a unified representation for CuTe layouts and Triton linear layouts to enable formal analysis and correctness verification.", status: "VERIFIED", evidence: ["P2"]}
    - {id: "C-P2-02", claim: "The ISL paper models CuTe stride+swizzle and Triton F2-style transforms via integer set relations.", status: "VERIFIED", evidence: ["P2"]}
    - {id: "C-P2-03", claim: "The ISL paper implements layout manipulation algorithms (composition/inversion/complement) using ISL operations to preserve semantics.", status: "VERIFIED", evidence: ["P2"]}
    - {id: "C-P3-01", claim: "Categorical Foundations defines categories Tuple and Nest to model a tractable class of CuTe layouts.", status: "VERIFIED", evidence: ["P3"]}
    - {id: "C-P3-02", claim: "Categorical Foundations provides a Python implementation with tests aligned to CUTLASS behavior.", status: "VERIFIED", evidence: ["P3"]}
    - {id: "C-N1-01", claim: "CUDA Tile is a tile-based GPU programming model targeting portability for NVIDIA Tensor Cores and is based on Tile IR and tools including cuTile Python.", status: "VERIFIED", evidence: ["N1"]}
    - {id: "C-N2-01", claim: "cuTile is a Python-based DSL/programming model claiming portability across NVIDIA GPU architectures and automatic leveraging of tensor cores and tensor memory accelerators.", status: "VERIFIED", evidence: ["N2"]}
    - {id: "C-N2-02", claim: "cuTile arrays are mutable global-memory objects with strided layouts; tiles are immutable kernel-only values with power-of-two compile-time dimensions.", status: "VERIFIED", evidence: ["N2"]}
    - {id: "C-N2-03", claim: "cuTile memory model permits reordering and defines atomic memory order/scope with per-element synchronization.", status: "VERIFIED", evidence: ["N2"]}
    - {id: "C-N3-01", claim: "Tile IR is a portable, low-level tile virtual machine and instruction set modeling GPUs as tile-based processors.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-02", claim: "Tile IR tile kernels execute as parallel instances of tile blocks; mapping to hardware threads is compiler-handled.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-03", claim: "Tile IR weak memory operations cannot be used to communicate between threads; compiler may assume weak tiles are not concurrently accessed.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-04", claim: "Tile IR uses tokens/token order for memory ordering; program dependencies do not order memory ops and may be optimized away, while token dependencies are preserved.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-05", claim: "Tile IR stability includes bytecode stability and syntactic portability for programs conforming to spec vX.Y to platforms supporting vX.Y+.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-06", claim: "Tile IR Spec 13.1 (2026-01-23) release notes list known issues (missing examples, incomplete encodings, limited atomics, etc.).", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N4-01", claim: "Triton-to-TileIR compiles Triton kernels to CUDA Tile IR instead of PTX; requires CUDA 13.1+ and Blackwell with source-based compilation.", status: "VERIFIED", evidence: ["N4"]}
    - {id: "C-N4-02", claim: "Triton-to-TileIR limitations include incomplete op support and tensor-of-pointer slowdowns; mitigations include SIMT fallback or adopting TMA load/store descriptors.", status: "VERIFIED", evidence: ["N4"]}
    - {id: "C-N6-01", claim: "The cuda-tile repo provides an MLIR dialect/tooling ecosystem (Python bindings, bytecode serialization, conformance tests) aligned with CUDA Toolkit 13.1.", status: "VERIFIED", evidence: ["N6"]}
    - {id: "C-I-01", claim: "[INFERENCE] Any memory-overlap/reordering optimization in Tile IR must preserve or construct correct token dependencies (not rely on apparent program deps) to avoid races.", status: "UNVERIFIED", evidence: ["N3-derived"]}
    - {id: "C-I-02", claim: "[INFERENCE] Research opportunity: combine formal layout reasoning (P1/P2/P3) with Tile IR token/scope legality to build analyzable async overlap + diagnostics for CUDA Tile stacks.", status: "UNVERIFIED", evidence: ["P1/P2/P3/N3-derived"]}

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
  risks_to_validity:
    - "Hardware availability constraints (e.g., Blackwell-only paths) can limit reproducibility."
    - "Spec/toolchain evolution during project can move targets; pin to a spec version for claims."

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: "WP1_20260204"
  stage1_5_toolbox: null
  stage2_directions: null
  stage2_5_novelty_audit: null
  stage3_assembly_pack: null
  stage3_paper: null

OPEN_QUESTIONS:
  - id: "OQ-01"
    question: "Which Tile IR operations are token-ordered and how do async/TMA-related ops produce/consume tokens (incl. wait semantics)?"
    why_load_bearing: "Constrains G1 legality and G2 scheduling; prevents proposing impossible optimizations."
    query_plan:
      - "Open N3 (Tile IR spec) -> search: 'token', 'tko', 'async', 'tma', 'wait'."
      - "Extract exact op signatures + token inputs/outputs; note scope/ordering rules."
  - id: "OQ-02"
    question: "What is the authoritative current hardware/architecture support scope for CUDA Tile / Tile IR (beyond the Triton-to-TileIR blog summary)?"
    why_load_bearing: "Determines evaluation feasibility and baseline plan (G8)."
    query_plan:
      - "Re-check N3 'stability' and any support matrices/notes."
      - "Look for official CUDA Toolkit 13.1+ docs listing CUDA Tile availability per architecture."
  - id: "OQ-03"
    question: "What is the exact unsupported/partially-supported operation list for Triton-to-TileIR (and how quickly is it changing)?"
    why_load_bearing: "Workload selection (G2/G8) and avoiding dead-end benchmarks."
    query_plan:
      - "From N4 blog, follow links to the canonical 'unsupported ops/features' list."
      - "Snapshot list + date; map to candidate motifs (GEMM/attention/etc.)."
  - id: "OQ-04"
    question: "How exactly does cuTile Python lower/compile to Tile IR (pipeline boundaries: Python -> MLIR dialect -> bytecode -> driver JIT/AOT), and where can passes be inserted/measured?"
    why_load_bearing: "Implementation feasibility and compile-time measurement (G5)."
    query_plan:
      - "Read N2 'Machine Representation' + 'Interoperability' sections."
      - "Cross-check with N6 docs/tooling for bytecode production/loading and pass hooks."
  - id: "OQ-05"
    question: "Does the current cuda-tile / Tile IR toolchain already implement token-dependency inference/minimization and/or an async pipeline scheduler? If yes, what are its limitations?"
    why_load_bearing: "If it exists, G1/G2 must be reframed as 'limitations of existing pass' not 'missing feature'."
    query_plan:
      - "Inspect N6 repo for passes mentioning tokens/scheduling; search for 'token', 'tko', 'schedule', 'pipeline', 'async'."
      - "Check docs/tests for mention of legality checking or dependency synthesis."
  - id: "OQ-06"
    question: "What debugging/diagnostic capabilities exist today for Tile IR token graphs and memory model errors (beyond conformance pass/fail)?"
    why_load_bearing: "Determines whether G4 is a new contribution or an extension/integration of existing tools."
    query_plan:
      - "Scan N6 conformance tests output formats and any debug flags."
      - "Look for visualization/logging tooling in N6; identify gaps in human-actionable explanations."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 1.5 prompt."
STATE_END
```

ARTIFACT_SUMMARY:
- "Stage 1 Gap Audit produced 8 ranked gaps (G1..G8) tied explicitly to Tile IR token semantics, async scheduling, and layout formalizations."
- "Top-3 gaps selected and recorded as verdicts: G1 (token legality + synthesis), G2 (async/TMA pipelining), G3 (layout→addressing end-to-end checking)."
- "Created Axiom-vs-Hardware-vs-SOTA matrix mapping each gap to: seed axiom, stack facts, what SOTA solves, why it fails, and required evidence."
- "Created Gap→Measurement Hook table specifying concrete benchmark motifs, baselines, confounders, and threats to validity."
- "OPEN_QUESTIONS upd with two new blockers (OQ-05, OQ-06) to verify whether key gaps are already partially solved in N6 tooling."
- "ARTIFACT_INDEX updated: stage1_gap_audit = WP1_20260204; current_stage set to 1."

NEXT_STAGE_HINT:
- "Fresh chat; paste: (1) Master System Configuration, (2) this CONTEXT_CAPSULE, (3) Stage 1.5 prompt (toolbox construction / feasibility MVP planning)."
