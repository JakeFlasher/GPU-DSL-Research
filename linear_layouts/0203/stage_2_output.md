# WORK_PRODUCT

## Direction Cards (D1..D3)

### D1 — Token Certificates for Tile IR (Token Effect System + Minimization)

Addresses (Gap_IDs)
- G1 (primary), G2 (enabler via legality checks), G3 (optional hook: layout ↔ alias/conflict constraints)

Thesis
- A compiler pass that **infers, checks, repairs, and minimizes Tile IR token wiring** can make unordered-memory Tile IR *safe-by-construction* while preserving performance by avoiding conservative serialization.

Novel semantic object / representation
- **Token Certificate**: a machine-checkable object consisting of:
  - **(a) Conflict relation** \(R\) over memory ops (may-alias/may-conflict, scoped per view/tile)
  - **(b) Token DAG** \(G_T\) (ops ↔ tokens) such that for every \((u,v) \in R\) requiring order, there is a token-path ordering \(u \prec_T v\)
  - **(c) Minimization objective** (e.g., minimize #tokens, #join_tokens, and token critical path)
  - **(d) Diagnostic payload** (unsat core / minimal unordered witness; suggested repair edits)
- Internal form: **Token Normal Form (TNF)** = SSA-like token phi/join at CFG merges + explicit loop-carried token recurrence.

Closest neighbors + delta
- CUDA Tile IR token semantics (N3) + toolchain/conformance (N6): **defines** tokens but does not (per current STATE; must confirm in 2.5) provide a **first-class, explainable token inference + minimization** pipeline.
- Triton-to-TileIR backend (T1): notes unordered memory + incorrectness risks; suggests **conservative token appending**.  
  **Delta:** (i) sound legality checking, (ii) repair, (iii) minimization, (iv) actionable diagnostics rather than “serialize everything”.
- Conceptual neighbors (non-TileIR-specific): effect systems / MemorySSA / dep-graph verifiers.  
  **Delta:** token-grounded semantics + optimization objective + direct integration with TileIR scheduling hints.

System / artifact
- **Artifact A1 (compiler pass):** `tileir-token-cert` (TileIR bytecode → TileIR bytecode + sidecar cert)
  - Input: Tile IR module (from cuTile Python, CUDA Tile compiler, or Triton-to-TileIR)
  - Output: (i) verified + optionally rewired Tile IR; (ii) Token Certificate (JSON); (iii) diagnostic report + token graph stats
- **Artifact A2 (dev tooling):** token-graph visualizer + “why ordered?” query (explain token path).
- Interface with stack:
  - cuTile Python: post-IR generation verifier/repair; usable in CI + conformance harness
  - Triton-to-TileIR: backend validation + optional optimization stage before final lowering

Guarantee + assumptions
- **Validated guarantee (soundness, scoped):**
  - If `tileir-token-cert` accepts a module, then **for every conservatively-identified conflicting pair of memory ops**, the module’s token wiring enforces an order consistent with that conflict relation.
  - If `repair` mode is enabled, produced rewiring is **token-legal** w.r.t. the same conflict relation.
- **Assumes:** the conflict relation is a sound over-approximation (alias/conflict analysis not unsound).
- **Out of scope (MVP):** cross-kernel ordering; atomics/fences beyond what Tile IR encodes; precise arbitrary pointer aliasing (handle conservatively first).

Implementation plan (MVP + stretch)
- **Weeks 1–4 (MVP):**
  - Parse Tile IR; extract CFG; build token SSA graph
  - Implement conservative conflict detection for view loads/stores (start coarse; refine if view region info available)
  - **Legality checker:** report unordered conflicting pairs (no token-path) with minimal witness
  - **Repair v0:** “safe serialization” (single token chain or per-basic-block chain + join_tokens at merges)
  - Emit token graph metrics (#tokens, #join_tokens, token critical path)
- **Weeks 5–12 (stretch):**
  - Add **minimization**: greedy edge pruning + (optional) ILP/SAT for small regions
  - Add **CFG joins/loops TNF** patterns (token-phis) + regression tests
  - Add stronger diagnostics (unsat-core-like slicing; repair suggestions)
  - Integrate into T1 backend path + N6 conformance suite workflow

Evaluation (metrics, baselines, ablations, workloads)
- **Metrics**
  - Speed: end-to-end runtime (kernel time; throughput)
  - Non-speed (≥3):
    - Compile-time overhead (pass time)
    - Token graph complexity (#tokens, #join_tokens, edges, critical path)
    - Diagnostic quality (explanation size; time-to-localize; witness minimality)
    - Verification coverage (% accepted; % repaired successfully)
    - Bug-finding effectiveness (# issues caught pre-runtime/conformance)
- **Baselines**
  - SOTA: CUDA Tile / cuTile / Tile IR pipeline as-is (N6/N2/N3 stack)
  - SOTA alt: Triton-to-TileIR backend default token strategy (T1)
  - Naive: conservative token-serialize (single token chain over all memory ops)
  - Naive sanity: no/incorrect tokens variants (expected illegal; demonstrates checker value)
- **Ablations (≥2)**
  - Repair-only vs repair+minimization
  - Coarse alias (everything conflicts) vs refined region-based conflict
  - Witness-only diagnostics vs witness+repair suggestions
- **Workloads / motifs (≥3)**
  - M1: double-buffered pipelined GEMM microkernel (async load → MMA loop)
  - M2: attention/softmax block (async loads + reductions)
  - M3: tensor-of-pointers gather/scatter (aliasing + ordering stress)
  - (Optional) M4: CFG join/if-else around memory ops (join_tokens stress)

Risks (top 2) + mitigations
- **R1: Conflict analysis too conservative ⇒ serialization ⇒ weak speed story.**
  - Mitigation: staged precision (region-based view analysis); optional annotations; report “why conservative”.
- **R2: CFG/loop token patterns subtle ⇒ unsound repair or excessive rejection.**
  - Mitigation: TNF token-phi design + unit tests from N6 conformance; structured-loops-first; safe fallback serialization + explicit warning.

---

### D2 — Token-Legal Async/TMA Pipeline Synthesis

Addresses (Gap_IDs)
- G2 (primary), G1 (dependency: legality checker), (optional) G3 (layout affects TMA feasibility)

Thesis
- A **token-aware software pipeliner** that synthesizes async/TMA overlap schedules for Tile IR—validated by token legality—can deliver predictable speedups without ad-hoc hand scheduling.

Novel semantic object / representation
- **Pipeline Schedule Certificate (PSC):**
  - Stage/time assignment for memory ops + compute ops (modulo schedule / steady-state)
  - Explicit **token constraints** derived from required ordering + pipeline stage edges
  - Parameters: prefetch distance, buffering depth, allowed reorders (uses allow_tma/latency hints when present)
  - Optional cost-model trace (why this schedule was chosen)

Closest neighbors + delta
- Tile IR hints (allow_tma, latency) (N3): knobs exist, but no **principled synthesizer with legality proof**.
- cuTile/CUTLASS hand pipelines: strong, manual. **Delta:** automation + explainability + legality validation.
- Triton/Triton-to-TileIR pipeline heuristics (T1): **Delta:** explicit token-legal schedule objective and PSC as reusable artifact.
- Classic modulo scheduling / polyhedral scheduling: **Delta:** concretely targets TileIR tokens + async/TMA ops.

System / artifact
- **Artifact B1:** `tileir-async-pipe` pass (TileIR → TileIR)
  - Reorders ops, inserts/adjusts async loads, rewires tokens, sets/uses allow_tma/latency hints
  - Always post-validated by D1 legality checker (or embedded checker)
- **Artifact B2:** schedule visualizer (pipeline stages, token critical path, overlap estimate)

Guarantee + assumptions
- **Validated guarantee (scoped):**
  - Output module is **token-legal** under the same conflict model used by the checker.
  - Schedule respects explicit token dependencies and preserves SSA/data dependencies.
- **Assumes:** latency hints are directionally correct; hardware/toolchain supports intended async/TMA ops.
- **Out of scope (MVP):** global optimal scheduling across arbitrary CFG; multi-kernel fusion; dynamic shape-dependent pipelines.

Implementation plan (MVP + stretch)
- **Weeks 1–4 (MVP):**
  - Pick 1 motif (structured loop) — start with pipelined GEMM
  - Implement heuristic pipelining: depth=2 buffering, fixed prefetch distance, join_tokens at loop boundaries
  - Integrate legality checker: reject/repair illegal schedules
  - Collect runtime + overlap proxy metrics (token critical path, issued async ops)
- **Weeks 5–12 (stretch):**
  - Add 2 more motifs (attention block; convolution/stencil)
  - Add local ILP/DP for stage assignment in small regions
  - Add occupancy/register/shared-memory constraints into cost model
  - Autotune discrete parameters (prefetch distance, buffering depth) with legality pruning

Evaluation (metrics, baselines, ablations, workloads)
- **Metrics**
  - Speed: throughput / kernel time
  - Non-speed (≥3):
    - Token critical path length (parallelism proxy)
    - Achieved overlap proxy (async issue rate; stall cycles if available)
    - Compile-time overhead
    - Register/shared-memory usage deltas
    - Verification coverage (% kernels successfully pipelined)
- **Baselines**
  - SOTA: current Tile IR lowering and tuning (N6 toolchain)
  - SOTA alt: hand-tuned pipelines where available (cuTile/CUTLASS-style reference)
  - Naive: no-overlap schedule (loads then compute, or compute then loads)
  - Naive: conservative token-serialize (destroys overlap opportunities)
- **Ablations (≥2)**
  - Disable legality-guided repair (expect failures/fallbacks)
  - Disable cost model (pure heuristic) vs cost-model-guided
  - Fix vs autotune prefetch distance/buffering depth
- **Workloads / motifs (≥3)**
  - N1: pipelined GEMM (TMA/async to shared → MMA loop)
  - N2: attention block (async loads for tiles + reduction)
  - N3: stencil/convolution tile (overlap halo loads + compute)

Risks (top 2) + mitigations
- **R1: Hardware-/toolkit-specific behavior makes results brittle.**
  - Mitigation: pin toolkit/cuda-tile versions; emphasize legality + PSC artifact, not universal speedups.
- **R2: Toolchain already pipelines similarly ⇒ novelty risk.**
  - Mitigation: Stage 2.5 audit; if overlap exists, reposition to “token-legal + explainable + certificate + diagnostics/minimization”.

---

### D3 — End-to-End Layout→View Descriptor Certificates (ISL + Bit-Refinement)

Addresses (Gap_IDs)
- G3 (primary), G1 (secondary: layout refines alias/conflict sets), (optional) G2 (layout enables/disables async/TMA)

Thesis
- A **Layout Certificate** that connects high-level layout algebra (ISL relations / \(\mathbb{F}_2\) linear layouts / categorical CuTe) to **concrete Tile IR view descriptors and address computations** can prevent silent OOB/descriptor bugs and unlock aggressive layout choices safely.

Novel semantic object / representation
- **Layout Certificate Pack (LCP):**
  - **(a) ISL witness**: logical indices → physical offsets relation
  - **(b) Bit-level refinement**: explicit mapping for swizzle/permutation bits (non-affine parts)
  - **(c) Proof obligations**: bounds (no OOB), alignment constraints, (optional) non-aliasing region facts
  - **(d) Linkage**: mapping from certificate terms to Tile IR view op params / descriptor fields

Closest neighbors + delta
- P1 (\(\mathbb{F}_2\) linear layouts) + Triton integration: formal layouts, but not necessarily tied to **Tile IR view descriptors**.
- P2 (ISL relations): unified representation + analysis. **Delta:** end-to-end checking against *emitted* Tile IR (descriptor/address-level) + bit-refinement.
- P3 (categorical CuTe): principled layout construction. **Delta:** certifying the *eventual* low-level view/address mapping and producing counterexamples.
- Existing cuTile/CUDA Tile view primitives (N2/N3): provide ops; **Delta:** formal witness + checker layer usable in CI and for safe transformations.

System / artifact
- **Artifact C1:** `tileir-layout-cert` analyzer/verifier
  - Consumes Tile IR (or cuTile view objects) + shape/tile constraints
  - Produces/validates LCP attached as metadata (or sidecar)
  - Emits counterexample index tuples on failure
- **Artifact C2:** reference address interpreter for a subset of view ops; validates by bounded enumeration + symbolic checks

Guarantee + assumptions
- **Validated guarantee (scoped):**
  - For supported subset, certificate validation implies **(i) in-bounds addressing** over tile domain and **(ii) equivalence** between high-level layout mapping and Tile IR’s effective address calculation.
- **Assumes:** shapes/strides are compile-time constants or within declared bounds; interpreter matches pinned toolchain semantics.
- **Out of scope (MVP):** fully general dynamic shapes; arbitrary pointer arithmetic outside view ops; global non-aliasing with unknown pointers.

Implementation plan (MVP + stretch)
- **Weeks 1–4 (MVP):**
  - Pick subset: strided/blocked affine + one simple swizzle family
  - Implement ISL witness construction for subset
  - Implement Tile IR view-address interpreter for subset; validate with bounded enumeration
  - Emit counterexamples + minimal “which dim caused OOB” reports
- **Weeks 5–12 (stretch):**
  - Add bit-refinement for more swizzles; integrate with P1/P3 constructors
  - Extend to TMA-related descriptor constraints where possible (conformance-driven)
  - Use certificates to enable transformations (layout selection/composition) with safety checks

Evaluation (metrics, baselines, ablations, workloads)
- **Metrics**
  - Speed: runtime where certificates enable aggressive layouts/TMA safely
  - Non-speed (≥3):
    - Verification coverage (% view ops/layouts certified)
    - Bug-finding rate (# OOB/invalid-descriptor cases detected)
    - Compile-time overhead (certificate gen + check)
    - Counterexample quality (minimality; debug time)
    - Certificate size/complexity (ISL relation size + bit-refinement size)
- **Baselines**
  - SOTA: current cuTile/CUDA Tile view lowering without certificates (N2/N3/N6)
  - Naive: bounded brute-force enumeration only (no ISL)
  - Naive: affine-only witness ignoring swizzle bits (expected to fail on swizzles)
- **Ablations (≥2)**
  - ISL-only (no bit-refinement) vs full LCP
  - Enumeration-only vs hybrid symbolic+bounded
  - Coarse vs fine tile-domain constraints
- **Workloads / motifs (≥3)**
  - L1: swizzled shared-memory GEMM
  - L2: attention tiling with transpose/layout transforms
  - L3: convolution/stencil tiles with padding/halo

Risks (top 2) + mitigations
- **R1: View/descriptor semantics complex/unstable ⇒ checker mismatch.**
  - Mitigation: pin toolchain; validate interpreter vs N6 conformance; keep subset small.
- **R2: Speed impact hard to show; looks like “just verification.”**
  - Mitigation: frame as enabling safe aggressive layouts/TMA; include case studies where certificate prevents conservative fallbacks.

---

## Decision Matrix

| Direction | Novelty(1-5) | Feasibility(1-5) | Eval Credibility(1-5) | Risk(1-5) | Why it wins | Key unknowns | Killer citation needed? |
|---|---:|---:|---:|---:|---|---|---|
| D1 Token Certificates | 4 | 4 | 5 | 2 | Foundational: makes unordered Tile IR safe + debuggable; can quantify token complexity + speed regained vs conservative serialization; enables D2. | Whether cuda-tile already has token inference/minimization; CFG/loop token patterns; practical alias/conflict model for view ops. | Yes: authoritative evidence on what current stack does/doesn’t provide + official token-wiring examples. |
| D2 Async Pipeline Synthesis | 3 | 3 | 3 | 4 | Clear speedup story if it works; ties to allow_tma/latency; strong systems angle. | True controllability via hints; availability of hand-tuned baselines; whether existing toolchain already pipelines similarly. | Yes: authoritative docs/spec on async/TMA legality + whether cuda-tile already schedules this way. |
| D3 Layout Certificates | 5 | 3 | 4 | 4 | Highest conceptual novelty: bridges P1/P2/P3 to concrete Tile IR descriptors; strong correctness/debug story; may unlock aggressive layouts. | Exact semantics/fields of view descriptors; useful-but-tractable subset; measurable speed impact. | Yes: authoritative view/address semantics sufficient to justify certificate obligations (else must scope hard). |

## Stage-2 Verdict (≤12 bullets)

- **Provisional winner: D1 (Token Certificates for Tile IR).**
- Why: targets acceptance-critical correctness (unordered memory unless tokens) and yields measurable artifacts (token complexity, diagnostics) with a credible evaluation (bugs caught + speed regained vs conservative serialization).
- D1 also provides a clean program arc: D1 (legality/minimization) → enables D2 (pipelining) and partially supports D3 (more precise conflict regions).
- Stage 2.5 must **confirm novelty**: does `cuda-tile` already implement token inference/minimization/diagnostics? If yes, what’s missing (objective, CFG/loop TNF, explainability)?
- Stage 2.5 must **extract authoritative token-wiring patterns** for CFG merges/loops from spec + conformance tests to ground TNF.
- Stage 2.5 must **pin evaluation versions** (toolkit + cuda-tile + cuTile) and confirm arch/support constraints for reproducibility.
- Stage 2.5 must **validate workload feasibility**: pick 3+ motifs supported in cuTile/Triton-to-TileIR today (avoid unsupported-op dead ends).
- Stage 2.5 must **collect killer citations** for: unordered memory model + token semantics (already in STATE), current lack (or limitations) of inference/minimization, and conservative token strategies in T1.

---

# CONTEXT_CAPSULE

STATE_BEGIN
profile:
  project_name: "GPU Layout + Async Research Program"
  target_model: "gpt-5.2-pro (web UI)"
  operating_mode: "manual_state_v3 (web UI; milestone resets)"
  stage_plan: ["0", "1", "1.5", "2", "2.5", "3"]
  conference_targets: ["ISCA", "MICRO", "ASPLOS"]
  paper_genre: "research proposal"
  hard_constraints:
    - "No hallucinated citations"
    - "No invented technical specifics"
    - "Follow stage schemas exactly"
    - "Golden sources always present in STATE"
  current_stage: 2
  last_updated: "2026-02-04"
  provisional_winner_direction: "D1"
  success_criteria:
    - "Closest prior art + explicit delta is obvious in <30 seconds"
    - "Feasibility believable without handwavy alias/hardware assumptions"
    - "Evaluation runnable by 1 grad student in ~1 semester"

GOLDEN_SOURCES:
  - {id: "P1", title: "Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2", url: "https://arxiv.org/html/2505.23819v3"}
  - {id: "P2", title: "Modeling Layout Abstractions Using Integer Set Relations", url: "https://arxiv.org/html/2511.10374v1"}
  - {id: "P3", title: "Categorical Foundations for CuTe Layouts", url: "https://arxiv.org/pdf/2601.05972v1"}
  - {id: "N1", title: "CUDA Tile (concept)", url: "https://developer.nvidia.com/cuda/tile"}
  - {id: "N2", title: "cuTile Python docs", url: "https://docs.nvidia.com/cuda/cutile-python"}
  - {id: "N3", title: "Tile IR spec", url: "https://docs.nvidia.com/cuda/tile-ir/latest/"}
  - {id: "N4", title: "Triton-to-TileIR backend (NVIDIA blog)", url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"}
  - {id: "N5", title: "cuTile Python repo", url: "https://github.com/NVIDIA/cutile-python"}
  - {id: "N6", title: "cuda-tile repo", url: "https://github.com/NVIDIA/cuda-tile"}
  - {id: "T1", title: "Triton-to-TileIR backend repo (incubator)", url: "https://github.com/triton-lang/Triton-to-tile-IR"}
  - {id: "C1", title: "TileIR internals blog (secondary; not authoritative)", url: "https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/"}

GLOBAL_MEMORY:
  notes:
    - {id: "GM-format", text: "Default deliverables: WORK_PRODUCT then CONTEXT_CAPSULE; Stage 3 overrides (LaTeX + capsule in comments).", last_update_date: "2026-02-04"}
    - {id: "GM-evidence", text: "Paper-grade claims require Claim_IDs + evidence pointers; unknowns must be UNVERIFIED + moved to OPEN_QUESTIONS.", last_update_date: "2026-02-04"}
    - {id: "GM-D1", text: "D1 (winner): Token Certificates for Tile IR (Token Effect System + Token Normal Form + minimization + diagnostics) as a TileIR pass; guarantees token-legal ordering for conservatively-identified conflicts; eval via speed regained vs conservative serialization + token-graph complexity + bug-finding.", last_update_date: "2026-02-04"}
    - {id: "GM-D2", text: "D2: Token-legal async/TMA pipeline synthesis producing a Pipeline Schedule Certificate; validated by D1 checker; eval via speed/overlap/resource pressure; higher risk due to hardware/toolchain variability.", last_update_date: "2026-02-04"}
    - {id: "GM-D3", text: "D3: Layout→View Descriptor Certificates (ISL witness + bit-level refinement) validated against TileIR view semantics; eval via verification coverage + bug-finding + (if possible) speed unlocked by safe aggressive layouts; novelty high, semantics risk.", last_update_date: "2026-02-04"}

SESSION_MEMORY:
  notes:
    - {id: "SM-0", text: "Stage 0: glossary + baseline map + Claim Ledger v0 + golden snapshots + constraint cliffs (P1..P3 + NVIDIA Tile stack).", last_update_date: "2026-02-04"}
    - {id: "SM-1", text: "Stage 1: Gap Audit; top gaps G1 token legality, G2 async/TMA pipelining, G3 end-to-end layout→addressing verification.", last_update_date: "2026-02-04"}
    - {id: "SM-1.5", text: "Stage 1.5: selected approaches: G1 TES+solver hybrid; G2 token-aware pipelining validated by G1; G3 ISL witness + bit-refinement hybrid.", last_update_date: "2026-02-04"}
    - {id: "SM-2", text: "Stage 2: produced D1..D3 direction cards; provisional winner D1.", last_update_date: "2026-02-04"}

VERDICT_LEDGER:
  items:
    - {id: "V-G1", verdict: "G1 top-3 gap: need token legality checking + synthesis/minimization + diagnostics.", last_update_date: "2026-02-04"}
    - {id: "V-G2", verdict: "G2 top-3 gap: principled async/TMA pipelining optimizer constrained by legality.", last_update_date: "2026-02-04"}
    - {id: "V-G3", verdict: "G3 top-3 novelty hook: end-to-end layout reasoning connected to Tile IR lowering/addressing.", last_update_date: "2026-02-04"}
    - {id: "V-2-WIN", verdict: "Provisional winner direction = D1 (Token Certificates for Tile IR).", rationale: "Most acceptance-critical + feasible + evaluable; foundational for D2 and supportive for D3.", last_update_date: "2026-02-04"}

CLAIM_LEDGER:
  items:
    - {id: "C-P1-01", claim: "P1: linear layouts over F2 integrated into Triton layout engine.", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P2-01", claim: "P2: ISL integer set relations unify CuTe + Triton linear layouts; enable analysis/verification + ops via ISL.", status: "VERIFIED", evidence: ["P2"]}
    - {id: "C-P3-01", claim: "P3: categorical foundations for CuTe layouts; Python impl + CUTLASS-aligned tests.", status: "VERIFIED", evidence: ["P3"]}
    - {id: "C-N3-07", claim: "Tile IR memory ops are token-ordered; ordering undefined unless connected by tokens; program deps do not order memory ops.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-08", claim: "Tile IR has join_tokens; memory ops consume/produce tokens.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-09", claim: "Tile IR view loads/stores support hints incl. allow_tma and latency.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-T1-01", claim: "T1: backend describes unordered memory model + token semantics; notes incorrectness risks and suggests conservative token appending.", status: "VERIFIED", evidence: ["T1"]}

EVAL_PLAN:
  status: "draft (to be specialized in 2.5)"
  metrics:
    - "end_to_end_speedup"
    - "compile_time_overhead"
    - "token_graph_complexity (#tokens, #join_tokens, token edges, token critical path)"
    - "diagnostic_quality (explanation size / counterexample quality)"
    - "verification_coverage (fraction of kernels/layouts with certificates)"
    - "bug_finding_effectiveness (#issues caught pre-runtime)"
  baselines:
    - "SOTA: CUDA Tile / cuTile / Tile IR toolchain default behavior"
    - "SOTA-alt: Triton-to-TileIR backend default token strategy"
    - "Naive: conservative token-serialize baseline (single token chain over all memory ops)"
    - "Naive: no/incorrect tokens (expected illegal; checker validation only)"
  workloads:
    - "pipelined GEMM microkernel (double buffering)"
    - "attention/softmax block (async loads + reduction)"
    - "tensor-of-pointers gather/scatter (alias stress)"
  ablations:
    - "minimization off vs on (D1)"
    - "coarse alias vs refined region-conflict model (D1)"
    - "diagnostics witness-only vs witness+repair suggestion (D1)"
  risks_to_validity:
    - "Toolchain/hardware availability constraints limit reproducibility; must pin versions and scope claims."

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: "WP1_20260204"
  stage1_5_toolbox: "WP1_5_20260204"
  stage2_directions: "WP2_20260204"
  stage2_5_novelty_audit: null
  stage3_assembly_pack: null
  stage3_paper: null

OPEN_QUESTIONS:
  - {id: "OQ-01", question: "Best-practice token wiring patterns for CFG joins/loops: are there official examples beyond spec prose?", why_load_bearing: "Defines Token Normal Form for real kernels.", next_action: "Scan N3 + N6 conformance/tests for token patterns; snapshot examples for paper."}
  - {id: "OQ-02", question: "Authoritative current support matrix (architectures + components) for CUDA Tile/cuTile; what is Blackwell-only?", why_load_bearing: "Evaluation feasibility + reproducibility claims.", next_action: "Cross-check N2/N4/N6/T1; pin exact toolkit version in proposal."}
  - {id: "OQ-03", question: "Canonical unsupported/partial support list for Triton-to-TileIR and how fast it changes.", why_load_bearing: "Workload selection and avoiding dead benchmarks.", next_action: "Snapshot T1 README/issues/docs + N4 blog; map motifs to supported subset."}
  - {id: "OQ-04", question: "Stable hook points to insert passes in cuTile Python → Tile IR and Triton-to-TileIR pipelines; how to measure compile cost.", why_load_bearing: "Implementation feasibility and honest compile-time accounting.", next_action: "Inspect N5/N6/T1 build + pass pipelines; identify minimal integration path."}
  - {id: "OQ-05", question: "Does cuda-tile already implement token dependency inference/minimization or async pipeline scheduling? If yes, what limitations remain?", why_load_bearing: "Determines true novelty of D1/D2.", next_action: "Search N6 source for token/schedule passes + docs; record evidence; adjust claims/deltas."}
  - {id: "OQ-06", question: "What debugging/diagnostic tooling exists today for token graphs and memory model errors (beyond pass/fail conformance)?", why_load_bearing: "Whether D1 diagnostics are new vs incremental.", next_action: "Scan N6 tooling flags/output; look for visualization/logging; document gaps."}
  - {id: "OQ-07", question: "What exact view/address semantics and descriptor invariants are specified for Tile IR view ops relevant to layout certificates?", why_load_bearing: "Feasibility boundary for D3 (and conflict analysis precision for D1).", next_action: "Extract precise semantics from N3/N2; if insufficient, scope to subset + conformance-driven semantics."}

NEXT_STAGE_HINT: "Fresh chat; paste this CONTEXT_CAPSULE + Stage 2.5 prompt (novelty audit / killer citations & unknowns resolution)."
STATE_END

ARTIFACT_SUMMARY:
  - "Created three concrete research directions: D1 Token Certificates (winner), D2 token-legal async/TMA pipelining, D3 layout→descriptor certificates."
  - "Selected provisional winner D1 based on acceptance-critical correctness + feasibility + evaluability."
  - "Added compact D1..D3 summaries to GLOBAL_MEMORY for milestone resets."
  - "Updated EVAL_PLAN with candidate motifs/workloads and D1-focused ablations."
  - "Updated VERDICT_LEDGER with V-2-WIN (provisional direction choice)."
  - "Extended OPEN_QUESTIONS with novelty-killer checks (existing passes?) and semantics-killer checks (view/descriptor invariants)."
  - "Set ARTIFACT_INDEX.stage2_directions = WP2_20260204."

NEXT_STAGE_HINT: "Fresh chat; paste this capsule + Stage 2.5 prompt."
