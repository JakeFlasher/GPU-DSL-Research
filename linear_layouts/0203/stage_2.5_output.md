## WORK_PRODUCT

### Stage goal + constraints (restate)
- **Goal (Stage 2.5):** rigorously audit D1/D2/D3 for **novelty vs closest neighbors**, feasibility, and evaluation credibility; then produce a **Stage‑3 Assembly Pack** that is paste‑ready and reviewer‑defensible.
- **Hard constraints:** no hallucinated citations; no invented toolchain specifics; if a key fact is uncertain, label **UNVERIFIED** + put a concrete verification plan into **OPEN_QUESTIONS**.
- **SOTA check is mandatory:** for each direction, explicitly answer: **“Already achievable/expressible in CUDA Tile stack (N1..N6)?”** If yes, novelty must be a different delta (certification/minimization/diagnostics/etc.).
- **Deliverables:** exactly the 5 WORK_PRODUCT items requested + updated CONTEXT_CAPSULE state.

---

### 1) Table — Core problem verification

| Core premise | Status (Verified/Unverified) | Evidence pointer | If false, impact | Fix |
|---|---|---|---|---|
| Tile IR memory operations are **token-ordered**; ordering between memory ops is **undefined unless connected by tokens** | **VERIFIED** | N3 ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai)) | D1 premise collapses (no “token legality” to check) | Pivot to different correctness target (e.g., view/address invariants) |
| Within a Tile Block Thread, **program dependencies do not provide ordering**; only token dependencies enforce ordering | **VERIFIED** | N3 ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)) | D1 “effect system” framing becomes less compelling | Reframe as *performance token minimization only* (still useful), or shift to D3 |
| Tokens are **abstract** (no runtime representation; cannot be compared or stored) | **VERIFIED** | N3 ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)) | Certificate format might need runtime instrumentation | Keep certificates compile-time; if runtime needed, scope to debug builds only |
| Tile IR has operations to create/merge tokens (e.g., `make_token`, `join_tokens`) and memory ops produce/consume tokens | **VERIFIED** | N3 ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai)) | D1 implementation path unclear | If absent in some subset, restrict to token-capable ops subset |
| Out-of-bounds behavior is **undefined** (so static reasoning must assume well-formed accesses or separately check bounds) | **VERIFIED** | N3 ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai)) | Alias/bounds proofs become meaningless if UB allowed | Add assumption to guarantee; optionally add a bounds-checking pre-pass (stretch) |
| CUDA Tile IR toolchain exists as an MLIR ecosystem with Python bindings + conformance tests, making “add a pass” plausible | **VERIFIED** | N6 ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai)) | Integration may require proprietary hooks | Scope MVP to standalone `cuda-tile-opt`/IR tooling; integrate later |
| Current public docs/repos clearly state a built-in **token dependency inference + minimization + diagnostics** pass already exists | **UNVERIFIED (likely false, but not proven)** | OQ-05 (needs code/doc evidence) | If it exists, D1 novelty risk increases sharply | Reposition delta to: *certificate object*, *minimality proof*, *explainable counterexamples*, *repair suggestions*, *measured compile/perf tradeoffs* |
| Practical performance evaluation is feasible on broadly available hardware | **UNVERIFIED / RISKY** | N4 + N5 suggest **Blackwell-only** in current state ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | Weakens “speedup” claims; reviewer skepticism | Make evaluation multi-tier: (A) compile-time+correctness on any machine, (B) runtime perf only if Blackwell available; be explicit |

---

### 2) Competitor baseline analysis (must include N1..N6 explicitly)

#### What the CUDA Tile stack already does (and what it doesn’t)
| Stack component | What they already do (ground truth) | What they can’t / don’t guarantee (for our thesis) | Where *our delta* lives |
|---|---|---|---|
| **N1 — CUDA Tile (concept page)** | Positions CUDA Tile as a tile-based model; CUDA Tile is based on Tile IR spec/tools; cuTile is user-facing language support (Python now; C++ later) ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile?utm_source=openai)) | No promise of *formal token legality checking*, token minimization, or actionable token diagnostics | Add a **correctness+quality layer**: check + certify + minimize token graphs |
| **N2 — cuTile Python docs** | Presents cuTile as Python DSL; claims automatic leverage of tensor cores + “tensor memory accelerators” and portability across NVIDIA architectures ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/?utm_source=openai)) | Public docs don’t specify token ordering strategy or offer user-visible token-debug tooling | D1 can surface **token semantics explicitly** as a cert/diagnostics tool even if cuTile “works” |
| **N3 — Tile IR spec** | Defines token-ordered memory model; tokens/join tokens; warns program deps don’t order memory ops; provides formal semantics for token order ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)) | Spec is not an optimizer: no algorithm for **inferring minimal necessary token edges**, no *certificate format*, no diagnostics UX | D1 builds a **compiler pass** that *operationalizes* N3 semantics into checkable artifacts |
| **N4 — Triton-to-TileIR backend (NVIDIA blog, Jan 30 2026)** | States Triton-to-TileIR is in active development; prerequisites include **CUDA ≥ 13.1** and **Blackwell GPUs**; notes limitations + perf gaps (e.g., tensor-of-pointer patterns) and suggests using **TMA APIs** ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | Not about token legality; not a proof/checker; also hints the ecosystem is young/unstable (evaluation risk) | D1 provides **semantic validation & debugging** that complements early backend development |
| **N5 — cuTile Python repo** | Documents requirements: driver **r580+**; notes `tileiras` “only supports Blackwell GPU with 13.1 release” (restriction to be removed later) ([github.com](https://github.com/NVIDIA/cutile-python?utm_source=openai)) | Does not advertise token legality diagnostics; hardware limits constrain evaluation portability | D1 can be evaluated **without runtime** for core results; perf eval is optional/limited |
| **N6 — cuda-tile repo** | Open-source CUDA Tile IR ecosystem: dialect, Python bindings, bytecode, conformance tests; aligned with CUDA Toolkit 13.1 ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai)) | Public README doesn’t establish any existing “token inference/minimization” pass; also repo says not accepting external contributions (integration risk) ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai)) | D1 can be built as **external pass/tool** against their IR + conformance tests; if upstreaming blocked, still publishable |

#### Direct “already achievable?” check (required)
- **D1 (Token Certificates):** *Token ordering itself* is already expressible (N3), and users/toolchains can wire tokens manually. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))  
  **Therefore novelty cannot be “tokens exist.”** Novelty must be **certifying + minimizing + diagnosing** token structure relative to a conflict model.
- **D2 (Async/TMA pipeline scheduling):** The stack acknowledges TMA usage and suggests rewriting to use TMA descriptors to address perf gaps (N4), and cuTile positions itself as leveraging tensor memory accelerators (N2). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
  **Therefore novelty must be “certified token-legal pipeline schedules,”** not “TMA exists.”
- **D3 (Layout→descriptor certificates):** Layout theory exists in P1–P3; Tile IR has view semantics and UB warnings (N3). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
  **Therefore novelty must be the bridge:** a *witness/certificate* that a high-level layout/view mapping refines to safe Tile IR addressing/descriptor invariants.

---

### 3) Per-direction audit table

Scales:
- **Novelty risk (1–10):** 10 = high risk “already done / not novel”
- **Feasibility risk (1–10):** 10 = high implementation risk
- **Eval credibility (1–10):** 10 = very credible for ISCA/MICRO/ASPLOS reviewers

| Direction | Closest neighbor(s) | Delta | Novelty risk | Feasibility risk | Eval credibility | Killer citation/evidence | Pivot if killed? |
|---|---|---:|---:|---:|---:|---|---|
| **D1: Token Certificates for Tile IR** | N3 token-ordered memory model; any existing “conservative token wiring” in toolchains (C‑T1‑01); CUDA Tile ecosystem (N6) | **New pass/tool** that (i) checks token legality vs conflict model, (ii) produces **certificate + counterexample**, (iii) **minimizes** token edges/joins under legality, (iv) emits **diagnostics/repair** | **4/10** (main risk: NVIDIA already has similar pass, UNVERIFIED) | **4/10** (checker + basic minimization feasible; deep alias precision harder) | **7/10** (strong static eval even without Blackwell; perf eval constrained) | Evidence of an existing public pass that already infers/minimizes token deps + good diagnostics (OQ‑05) | If “already exists,” pivot to **certificate format + minimality proof + UX** or to D2 “schedule certificates validated by D1” |
| **D2: Token-legal async/TMA pipeline synthesis** | N4 acknowledges TMA refactors and backend limitations; (implicit) existing pipelining in GPU kernels | Synthesize **pipeline schedule certificate** that is provably token-legal and resource-safe; validated by D1 checker | **6/10** (may be seen as “just an optimizer”) | **7/10** (hardware/resource modeling + backend instability) | **5/10** (hard to reproduce without Blackwell + exact toolchain) | Public docs showing cuTile/cuda-tile already performs similar schedule synthesis automatically | Pivot to: “use D1 to *validate* schedules produced by existing heuristics” (analysis-only) |
| **D3: Layout→descriptor/view certificates** | P1 linear layouts in Triton; P2 ISL relations; P3 categorical CuTe layouts; N3 view semantics + UB | Produce a witness that (layout/view) → (Tile IR addressing/descriptor invariants) is correct; catch OOB/stride/layout bugs before lowering | **7/10** (could be “too semantic” / under-specified) | **8/10** (needs precise semantics + integration point) | **4/10** (hard baselines; risk of semantics gaps) | If Tile IR spec lacks needed formal invariants / toolchain hides descriptor semantics (OQ‑07) | Pivot to narrower: **bounds/mask safety checker** for views (still useful, less semantic ambition) |

---

### 4) Strategic recommendation

**Choose final direction: D1 (Token Certificates for Tile IR).**

**Why reviewers should care (framing):**  
Tile IR’s memory model makes it *possible* to express safe parallelism, but also makes correctness *non-local* and brittle: within a tile block thread, the spec explicitly warns that program order/data deps do **not** enforce memory ordering—only tokens do. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)) As a result, compiler stacks and DSL backends face a harsh trade-off: either **over‑serialize** with conservative tokens (leaving performance on the table) or risk **silent mis-ordering** bugs that are hard to debug. D1 turns token ordering from an error-prone “implicit discipline” into a **checkable, minimizable, explainable certificate**, enabling (i) safe parallel reordering/overlap when proven non-conflicting and (ii) actionable counterexamples when unsafe—exactly the kind of acceptance‑critical correctness + performance story that fits ASPLOS/MICRO.

---

### 5) Stage‑3 Assembly Pack (paste‑ready)

#### Working Title
**Token Certificates for CUDA Tile IR: Minimal, Checkable Ordering for Token‑Ordered Memory**

#### Abstract (5–7 bullets)
- Tile IR uses a **token‑ordered memory model** where ordering between memory operations is undefined unless connected by tokens, and program dependencies do not impose ordering within a tile block thread. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))  
- This design enables aggressive parallel memory execution, but makes correctness depend on **explicit token wiring**, creating a new class of mis-ordering and “accidental serialization” bugs.
- We propose **Token Certificates**: a compiler pass that (i) derives a must‑order relation from a conservative conflict model, (ii) checks whether the program’s tokens enforce it, and (iii) emits a compact, machine‑checkable certificate.
- When the certificate fails, the pass produces an **explainable counterexample** (conflicting ops not ordered in token order) and a minimal repair suggestion.
- When the certificate succeeds, we optionally **minimize** token structure (tokens/joins/edges) while preserving legality, to recover overlap compared to conservative serialization.
- We implement the checker/minimizer as a pass in the CUDA Tile IR ecosystem (MLIR dialect + Python bindings + conformance tests). ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai))  
- We evaluate on representative kernels (GEMM double-buffering motif, attention/softmax blocks, gather/scatter alias stress) using correctness, token‑graph complexity, compile-time overhead, and (when available) end‑to‑end performance.

#### Contributions (3–5 bullets; each references Claim_IDs)
1. **Formalized “token legality” check** for Tile IR memory ops: a verifier that enforces the spec’s token‑ordered semantics and the “program deps don’t order memory ops” rule. (C‑N3‑07, C‑N3‑08)  
2. **Token Certificate format + checker**: emits/validates a compact witness of required token‑order constraints, enabling regression testing and toolchain debugging. (C‑N3‑07, C‑N3‑08)  
3. **Token Normal Form + minimization pass**: reduces unnecessary tokens/joins while preserving certified ordering under a conservative conflict model. (C‑N3‑07)  
4. **Actionable diagnostics**: counterexample generation + repair hints (where to add a token edge/join) to debug mis-ordering hazards. (C‑N3‑07; uses Tile IR debug-info plumbing where available) ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/debug_info.html?utm_source=openai))  
5. **Evaluation showing regained overlap** vs a naive conservative token chain baseline + compile-time overhead + bug-finding effectiveness. (EVAL_PLAN IDs in STATE)

#### System Overview (diagram description + components)
**Diagram (textual):** a left-to-right compiler pipeline.
1. **Input:** Tile IR module (from cuTile Python / Triton-to-TileIR / handwritten). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/?utm_source=openai))  
2. **(Optional) Conflict Model Builder:** conservative alias/region model over memory operations (views + pointer tiles) + user annotations.  
3. **Token Certificate Generator:** computes required must‑order constraints and proposes a token graph in **Token Normal Form** (with SSA-consistent tokens + `join_tokens` at CFG joins). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai))  
4. **Token Checker:** validates that tokens enforce required ordering (or outputs counterexample).  
5. **Token Minimizer:** reduces tokens/joins/edges while keeping checker true.  
6. **Outputs:** (a) rewritten Tile IR with minimized tokens, (b) certificate artifact (for CI/regression), (c) diagnostics report.

#### Key semantics/definitions that MUST appear
- **Token-ordered memory operation:** memory ops whose ordering is undefined unless related by token order. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai))  
- **Token order / waits-for order / happens-before (Tile IR terminology)** and the rule that **program dependencies don’t order memory ops**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))  
- **Conflict model** (conservative): when two ops “must be ordered” (e.g., may overlap + at least one write, within the modeled scope).  
- **Token graph:** nodes = memory ops; edges = token dependencies; `join_tokens` merges dependencies at control-flow joins. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai))  
- **Token Normal Form (TNF):** a restricted shape for token wiring (definition in paper) to make checking/minimization tractable.  
- **Certificate:** a machine-checkable representation of required order constraints (and/or the constructed token graph) that the checker validates.

#### Method sketch (algorithms/passes; invariants)
**Pass A — Extract memory events**
- Collect all token-ordered memory operations (loads/stores/atomics) and their (conservative) accessed regions.
- Record existing token SSA: which token each op consumes/produces; where `join_tokens` are placed. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai))  

**Pass B — Build conservative “must-order” constraints**
- For each pair of memory ops in a tile block thread, add constraint $$A \prec B$$ if:
  - modeled regions may overlap, and
  - at least one is a write / RMW, and
  - ordering is required under the chosen memory semantics/scope subset (explicitly scoped in paper).
- **MVP:** coarse region model (symbolic base + stride/shape where obvious; otherwise “may overlap”).  
- **Stretch:** refined model using Tile IR view metadata + layout certificates (future D3 hook).

**Pass C — Checker (Token Certificate Verifier)**
- Define: $$A \prec_{\text{token}} B$$ iff there is a token dependency path forcing $$A$$ before $$B$$ per Tile IR’s waits-for/token order definitions. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))  
- For each required $$A \prec B$$ constraint, verify $$A \prec_{\text{token}} B$$.  
- If any fail, output a **counterexample**: (A,B) pair + minimal explanation path (CFG location + missing edge).

**Pass D — Constructor + TNF**
- If tokens are missing, propose a token wiring in **TNF**:
  - introduce tokens with `make_token`, and enforce merges with `join_tokens` at CFG joins. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai))  
- Ensure SSA consistency: each token has a single defining op; all uses dominated by defs.

**Pass E — Minimizer**
- Objective: reduce **token graph complexity** (e.g., #tokens, #`join_tokens`, critical path length) while keeping checker true.
- **MVP:** greedy elimination of redundant edges/joins (re-check).  
- **Stretch:** structured optimization (e.g., formulate as constrained minimization over TNF).

**Invariants**
- **Soundness (w.r.t. conflict model):** all required must‑order constraints are enforced by token order.
- **No weakening across CFG merges:** token joins preserve dependencies needed for either predecessor path.

#### Scoped guarantee (“If accepted, we prove/validate X under assumptions Y”)
- **Guarantee:** If the Token Checker accepts a program + certificate, then **for every pair of token-ordered memory ops that the conflict model deems must‑ordered**, the program enforces that order in Tile IR **token order** (hence avoids the spec’s single-thread data-race hazard in that modeled scope). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))  
- **Assumptions (explicit):**
  - Conflict model is **conservative/sound** (may over-approximate overlap; must not miss true overlaps within scope).
  - Program does not rely on undefined behavior (e.g., out-of-bounds accesses) unless separately checked. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai))  
  - Scope restriction: initial version targets **within a tile block thread** token ordering; cross-block/global ordering is out of scope for the MVP.

#### Implementation & milestone plan (MVP vs stretch; dependencies)
**Dependencies**
- CUDA Tile IR ecosystem tooling + MLIR pass infrastructure (N6). ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai))  
- For runtime perf evaluation: CUDA 13.1+, driver r580+; current Blackwell constraints are likely. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  

**MVP (publishable core)**
1. Implement **Token Checker** (pass + standalone tool) with counterexample reporting.
2. Implement **TNF constructor** for straight-line + if/else CFGs using `join_tokens`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai))  
3. Implement **coarse conflict model** (safe over-approx; accept false positives).
4. Integrate with CUDA Tile IR test/conformance harness; add regression tests that intentionally break token order.

**Stretch**
- Minimizer beyond greedy; loop support; improved region model; integration into Triton-to-TileIR pipeline for automatic validation. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))

#### Evaluation plan (metrics + baselines + ablations + workloads + methodology)
**Metrics**
- End-to-end runtime speed (when hardware available)
- Compile-time overhead of certificate passes
- Token graph complexity: #tokens, #`join_tokens`, edges, critical path length
- Diagnostic quality: counterexample size + actionable location quality
- Bug-finding effectiveness: injected mis-orderings caught

**Baselines**
- “As-is” toolchain output (CUDA Tile / cuTile / Triton-to-TileIR, version pinned)
- Naive conservative serialization: single token chain over all memory ops
- No/incorrect tokens: expected checker failures (sanity)

**Workloads**
- Pipelined GEMM microkernel motif (double-buffered loads/stores)
- Attention/softmax block (loads + reductions)
- Tensor-of-pointers gather/scatter alias stress (matches known backend perf pain points) ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  

**Ablations**
- Minimization OFF vs ON
- Coarse conflict model vs refined (if implemented)
- Diagnostics: witness-only vs witness+repair suggestion

**Methodology**
- Publish scripts + IR dumps; use Tile IR debug info where possible. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/debug_info.html?utm_source=openai))  
- If Blackwell access limited: clearly separate results into (A) static/certification outcomes and (B) runtime perf.

#### Related work shortlist (with citations if browsing; else TODO)
- **Tile IR spec (token memory model, ops, debug info):** N3 ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))  
- **CUDA Tile ecosystem:** N1, N6 ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile?utm_source=openai))  
- **cuTile Python:** N2, N5 ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/?utm_source=openai))  
- **Triton-to-TileIR backend context + constraints:** N4 ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- **Layout foundations (background, future alias refinement hooks):** P1, P2, P3 ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **Triton (baseline DSL/compiler):** OpenAI Triton blog ([openai.com](https://openai.com/blog/triton/?utm_source=openai))  
- TODO (if space): effect systems / memory SSA / certified compilation precedents (needs citations)

#### Threats/limitations (1–5 bullets)
- Conflict/alias model may be too conservative → fewer optimizations; too weak → unsound certificates (must be explicitly scoped).
- Blackwell/toolchain availability may limit runtime evaluation generality. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- Token minimization could increase compile time; must report overhead honestly.
- If NVIDIA already provides similar internal tooling, novelty must shift to certificate UX + minimality guarantees (risk managed via OQ‑05).

#### Reviewer attack / response set (top 8 objections + crisp replies)
1. **“This is already handled by the CUDA Tile toolchain.”**  
   *Response:* We will explicitly audit and document existing passes; if present, we pivot to the **certificate artifact + minimality + diagnostics guarantees** not provided publicly (OQ‑05).
2. **“Tokens are just dependencies—this is trivial.”**  
   *Response:* Tile IR explicitly breaks the usual “single-thread program order” assumption; the checker operationalizes the spec and prevents a new class of silent mis-ordering hazards. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))
3. **“Your alias/conflict model is either too conservative or unsound.”**  
   *Response:* We scope the guarantee to a conservative model; we report false-positive rate and provide refinement hooks (views/layout metadata).
4. **“Where is the performance story?”**  
   *Response:* Compare against conservative token chain; show regained overlap + reduced token critical path; separate perf results from static benefits when hardware limited.
5. **“Why certificates—why not just run a pass?”**  
   *Response:* Certificates enable CI/regression and independent validation across toolchains; they make correctness portable across transformations.
6. **“Diagnostics are hand-wavy.”**  
   *Response:* We define the counterexample format (minimal unordered conflicting pair + missing token path) and measure diagnostic size/quality.
7. **“Evaluation depends on Blackwell; not reproducible.”**  
   *Response:* Core results are compile-time; we publish IR-level benchmarks + checker outcomes; runtime perf is an optional tier with pinned versions. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))
8. **“Scope is too narrow (only within tile block thread).”**  
   *Response:* That scope is exactly where the spec introduces the unusual hazard; extending scope is future work, but MVP is already high value. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))

---

## CONTEXT_CAPSULE

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
  current_stage: 2.5
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
    - {id: "GM-D1", text: "D1 (final): Token Certificates for Tile IR (Token Effect System + Token Normal Form + minimization + diagnostics) as a TileIR pass; guarantees token-legal ordering for conservatively-identified conflicts; eval via speed regained vs conservative serialization + token-graph complexity + diagnostics + bug-finding.", last_update_date: "2026-02-04"}
    - {id: "GM-D2", text: "D2: Token-legal async/TMA pipeline synthesis producing a Pipeline Schedule Certificate; validated by D1 checker; eval via speed/overlap/resource pressure; higher risk due to hardware/toolchain variability.", last_update_date: "2026-02-04"}
    - {id: "GM-D3", text: "D3: Layout→View Descriptor Certificates (ISL witness + bit-level refinement) validated against TileIR view semantics; eval via verification coverage + bug-finding + (if possible) speed unlocked by safe aggressive layouts; novelty high, semantics risk.", last_update_date: "2026-02-04"}
    - {id: "GM-2.5-focus", text: "Stage 2.5 novelty audit conclusion: D1 remains best for top-tier acceptance; main novey risk is unknown existence of built-in token inference/minimization/diagnostics in cuda-tile stack (must verify).", last_update_date: "2026-02-04"}

SESSION_MEMORY:
  notes:
    - {id: "SM-0", text: "Stage 0: glossary + baseline map + Claim Ledger v0 + golden snapshots + constraint cliffs (P1..P3 + NVIDIA Tile stack).", last_update_date: "2026-02-04"}
    - {id: "SM-1", text: "Stage 1: Gap Audit; top gaps G1 token legality, G2 async/TMA pipelining, G3 end-to-end layout→addressing verification.", last_upde_date: "2026-02-04"}
    - {id: "SM-1.5", text: "Stage 1.5: selected approaches: G1 TES+solver hybrid; G2 token-aware pipelining validated by G1; G3 ISL witness + bit-refinement hybrid.", last_update_date: "2026-02-04"}
    - {id: "SM-2", text: "Stage 2: produced D1..D3 direction cards; provisional winner D1.", last_update_date: "2026-02-04"}
    - {id: "SM-2.5", text: "Stage 2.5: novelty/feasibility/eval audit completed; Stage-3 Assembly Pack drafted for D1; clarified competitor baseline N1..N6; hardware eval risk noted (Blackwell/toolchain).", last_update_date: "2026-02-04"}

VERDICT_LEDGER:
  items:
    - {id: "V-G1", verdict: "G1 top-3 gap: need token legality checking + synthesis/minimization + diagnostics.", last_update_date: "2026-02-04"}
    - {id: "V-G2", verdict: "G2 top-3 gap: principled async/TMA pipelining optimizer constrained by legality.", last_update_date: "2026-02-04"}
    - {id: "V-G3", verdict: "G3 top-3 novelty hook: end-to-end layout reasoning connected to Tile IR lowering/addressing.", last_update_date: "2026-02-04"}
    - {id: "V-2-WIN", verdict: "Provisional winner direction = D1 (Token Certificates for Tile IR).", rationale: "Most acceptance-critical + feasible + evaluable; foundational for D2 and supportive for D3.", last_update_date: "2026-02-04"}
    - {id: "V-2.5-FINAL", verdict: "Final direction for Stage 3 = D1 (Token Certificates for Tile IR).", rationale: "Strongest correctness hook aligned with Tile IR spec; feasible as standalone checker+minimizer; evaluation credible even if runtime perf is hardware-limited.", last_update_date: "2026-02-04"}

CLAIM_LEDGER:
  items:
    - {id: "C-P1-01", claim: "P1: linear layouts over F2 integrated into Triton layout engine.", status: "VERIFIED", evidence: ["P1"]}
    - {id: "C-P2-01", claim: "P2: ISL integer set relations unify CuTe + Triton linear layouts; enable analysis/verification + ops via ISL.", status: "VERIFIED", evidence: ["P2"]}
    - {id: "C-P3-01", claim: "P3: categorical foundations for CuTe layouts; Python impl + CUTLASS-aligned tests.", status: "VERIFIED", evidence: ["P3"]}
    - {id: "C-N3-07", claim: "Tile IR memory ops are token-ordered; ordering undefined unless connected by tokens; program deps do not order memory ops.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-08", claim: "Tile IR has join_tokens; memory ops consume/produce tokens.", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-N3-09", claim: "Tile IR view loads/stores support optimization hints (incl. allow_tma/latency hints).", status: "VERIFIED", evidence: ["N3"]}
    - {id: "C-T1-01", claim: "T1: backend describes unordered memory model + token semantics; notes incorrectness risks and suggests conservative token appending.", status: "VERIFIED", evidence: ["T1"]}
    - {id: "C-N4-01", claim: "N4 (Jan 30, 2026): Triton-to-TileIR backend prerequisites include CUDA >= 13.1 and Blackwell GPUs; project in active development with unsupported ops/perf gaps.", status: "VERIFIED", evidence: ["N4"]}
    - {id: "C-N5-01", claim: "N5: cuTile Python requires NVIDIA Driver r580+; notes tileiras supports Blackwell-only in CUDA 13.1 release (restriction expected to lift later).", status: "VERIFIED", evidence: ["N5"]}
    - {id: "C-N6-01", claim: "N6: cuda-tile provides MLIR dialect + Python bindings + bytecode + conformance tests; aligned with CUDA Toolkit 13.1.", status: "VERIFIED", evidence: ["N6"]}
    - {id: "C-N2-01", claim: "N2: cuTile is a Python DSL claiming portability and automatic leverage of tensor cores and tensor memory accelerators.", status: "VERIFIED", evidence: ["N2"]}

EVAL_PLAN:
  status: "specialized to D1 (still draft)"
  metrics:
    - "end_to_end_speedup (when Blackwell available)"
    - "compile_time_overhead"
    - "token_graph_complexity (#tokens, #join_tokens, token edges, token critical path)"
    - "diagnostic_quality (witness size, localization quality, repair usefulness)"
    - "verification_coverage (fraction of kernels certified under conflict model)"
    - "bug_finding_effectiveness (#seeded token bugs caught pre-runtime)"
  baselines:
    - "CUDA Tile / cuTile / Tile IR toolchain default behavior (version-pinned)"
    - "Naive conservative token chain over all memory ops"
    - "No/incorrect tokens (checker should fail; diagnostic harness)"
  workloads:
    - "pipelined GEMM microkernel motif (double-buffered loads/stores)"
    - "attention/softmax block (async-ish loads + reduction ordering)"
    - "tensor-of-pointers gather/scatter alias stress"
  ablations:
    - "minimization off vs on"
    - "coarse conflict model vs refined region model"
    - "diagnostics witness-only vs witness+repair suggestion"
  risks_to_validity:
    - "Runtime perf evaluation may be constrained by CUDA 13.1 + Blackwell/tool availability; must separate static vs runtime results."

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: "WP1_20260204"
  stage1_5_toolbox: "WP1_5_20260204"
  stage2_directions: "WP2_20260204"
  stage2_5_novelty_audit: "WP2_5_20260204"
  stage3_assembly_pack: |
    id: "WP3_ASSEMBLY_20260204"
    working_title: "Token Certificates for CUDA Tile IR: Minimal, Checkable Ordering for Token‑Ordered Memory"
    abstract_bullets:
      - "Tile IR requires explicit tokens for ordering; program deps do not order memory ops."
      - "Token Certificates: checker + certificate + counterexample diagnostics."
      - "Optional minimization of tokens/joins/edges under legality."
      - "Evaluate: token-graph complexity, compile-time, bug finding, perf (if hardware available)."
    scoped_guarantee: "If checker accepts, all conflt-model must-order pairs are enforced in token order (within tile block thread scope), assuming no UB and sound conflict model."
    evaluation_core: "Compare vs conservative token chain; report compile overhead + diagnostic quality; perf tier if Blackwell available."
  stage3_paper: null

OPEN_QUESTIONS:
  - {id: "OQ-01", question: "Best-practice token wiring patterns for CFG joins/loops: are there official examples beyond spec prose?", why_load_bearing: "Defines Token Normal Form for real kernels.", next_action: "Scan Tile IR spec examples + cuda-tile tests (if accessible) for canonical join_tokens patterns; snapshot for paper."}
  - {id: "OQ-02", question: "Authoritative current support matrix (architectures + components) for CUDA Tile/cuTile; what is Blackwell-only today vs relaxed in newer releases?", why_load_bearing: "Evaluation feasibility + reproducibility claims.", next_action: "Pin CUDA toolkit + driver versions; cross-check N4 blog + N5 repo + NVIDIA docs; record exact constraints and dates."}
  - {id: "OQ-03", question: "Canonical unsupported/partial support list for Triton-to-TileIR and how fast it changes.", why_load_bearing: "Workload selection and avoiding dead benchmarks.", next_action: "Snapshot N4 blog + Triton-to-TileIR repo README/issues; map chosen motifs to supported subset."}
  - {id: "OQ-04", question: "Stable hook points to insert passes in cuTile Python → Tile IR and Triton-to-TileIR pipelines; how to measure compile cost.", why_load_bearing: "Implementation feasibility and honest mpile-time accounting.", next_action: "Inspect N6 tools + python bindings; identify minimal standalone checker path first, then integration."}
  - {id: "OQ-05", question: "Does cuda-tile/cuTile/tileiras already implement token dependency inference/minimization and/or token diagnostics? If yes, what limitations remain?", why_load_bearing: "Primary novelty-killer for D1.", next_action: "Search N6 repo (lib/tools/tests) + docs for token analysis passes; if exists, redefine delta to certificates/minimality/UX."}
  - {id: "OQ-06", question: "What debugging/diagnostic tooling exists today for token graphs and memory model errors (beyond pass/fail conformance)?", why_load_bearing: "Whether D1 diagnostics are new vs incremental.", next_action: "Scan N6 tooling flags/output + any docs; catalogue current UX."}
  - {id: "OQ-07", question: "What exact view/address semantics and descriptor invariants are specified for Tile IR view ops relevant to layout certificates?", why_load_bearing: "Feasibility boundary for D3 (and conflict analysis precision for D1).", next_action: "Extract precise semantics from N3; if insufficient, scope to subset + conformance-driven semantics."}

NEXT_STAGE_HINT: "Fresh chat; paste this CONTEXT_CAPSULE + Stage 3 prompt + Stage-3 Assembly Pack."
STATE_END

ARTIFACT_SUMMARY:
  - "Completed Stage 2.5 novelty/feasibility/eval audit for D1–D3 with explicit SOTA check against N1..N6."
  - "Chose final direction D1 (Token Certificates for Tile IR) and added V-2.5-FINAL to VERDICT_LEDGER."
  - "Identifi the main novelty-killer risk: possible existing token inference/minimization/diagnostics in cuda-tile stack (kept as OQ-05)."
  - "Pinned key feasibility constraints from public sources: CUDA >= 13.1 and likely Blackwell/runtime constraints; separated static vs runtime evaluation tiers."
  - "Produced Stage-3 Assembly Pack (WP3_ASSEMBLY_20260204) and stored a compact copy in ARTIFACT_INDEX."
  - "Extended CLAIM_LEDGER with N2/N4/N5/N6 factual claims needed for Stage 3."
  - "Updated EVAL_PLAN to a D1-specialized version emphasizing token-graph metrics + diagnostics + bug-finding, with perf as optional tier."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 3 prompt + Stage-3 Assembly Pack."

---
Learn more:
1. [8. Operations — Tile IR](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html?utm_source=openai)
2. [7. Memory Model — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)
3. [GitHub - NVIDIA/cuda-tile: CUDA Tile IR is an MLIR-based intermte representation and compiler infrastructure for CUDA kernel optimization, focusing on tile-based computation patterns and optimizations targeting NVIDIA tensor core units.](https://github.com/NVIDIA/cuda-tile?utm_source=openai)
4. [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)
5. [CUDA Tile | NVIDIA Developer](https://developer.nvidia.com/cuda/tile?utm_source=openai)
6. [cuTile Python — cuTile Python](https://docs.nvidia.com/cuda/cutile-python/?utm_source=openai)
7. [GitHub - NVIDIA/cutile-python: cuTile is a programming model for writing parallel kernels for NVIDIA GPUs](https://github.com/NVIDIA/cutile-python?utm_source=openai)
8. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
9. [9. Debug Info — Tile IR](https://docs.nvidia.com/cuda/tile-ir/sections/dnfo.html?utm_source=openai)
10. [Introducing Triton: Open-source GPU programming for neural networks | OpenAI](https://openai.com/blog/triton/?utm_source=openai)
