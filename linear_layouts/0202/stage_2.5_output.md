## WORK_PRODUCT

### Stage 2.5 goal + constraints (re-grounding)

- Goal: **rigorous novelty + SOTA feasibility audit** for D1/D2/D3 against **CUDA Tile / cuTile / Tile IR / Triton-to-TileIR**.
- Must: **no hallucinated citations**, and keep novelty claims as **deltas** when the baseline already supports something.
- Must: include **N1..N6 explicitly**, and resolve top OPEN_QUESTIONS with **≤25-word quotes/snippets**.
- Must: produce a **Stage‑3 Assembly Pack** (title, abstract bullets, contributions, semantics/defs, method sketch, eval plan+baselines, related work shortlist, threats).

---

### 1) Table — Core problem verification (evidence-backed)

| Core claim we rely on | Evidence (baseline) | Verified? | Implication for directions | “If already in Tile IR/cTile, novelty must be…” |
|---|---|---:|---|---|
| Memory ops are **token-ordered**; ordering otherwise undefined | Tile IR ops doc: “ordering … is undefined unless connected by tokens.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html?utm_source=openai)) | Yes | D1 must treat token graph legality as first-class | Not “tokens exist”; novelty is token-graph *synthesis/optimization + explanation* |
| **Program dependencies do not order** memory ops; tokens required even if “redundant” | Tile IR memory model: “Program dependencies … do not provide ordering … Tokens must be used…” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai)) | Yes | D1: legality tooling must reason on tokens, not control/data deps | Delta: “explainable legality proof” rather than “ordering model” |
| Tile IR tokens can express a **DAG (fork/join)**, not only chains | Tile IR ops: `make_token` creates fresh token; `join_tokens` depends on inputs. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) | Yes | D1 can target DAG schedules without extending Tile IR | Delta: how to *derive* good DAGs automatically + validate |
| Weak vs scoped operations + data-race UB is real | Tile IR memory model: weak ops can’t communicate; data races are UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai)) | Yes | D1’s validator needs to flag “looks sequential but isn’t” patterns | Delta: early illegality detection + counterexample traces |
| Views carry **shape/strides** (affine) but physical rep is hidden | Tile IR semantics: tensor_view tuple “(ptr, shape, strides, dimgroups)”; layout not visible. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/semantics.html?utm_source=openai)) | Yes | D2: non-affine / swizzle layouts are not obviously representable | Delta: a *restricted* extension or proof-guided lowering strategy |
| TMA use is a **hint**, not a semantic guarantee | cuTile perf: “may be lowered to use TMA… when … supports it.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai)) | Yes | D1/D2 must treat TMA as optional lowering; plan must tolerate fallback | Delta: plan validity independent of TMA selection |
| cuTile Python hides pointers; load/store expose **order/latency/allow_tma** (not tokens) | `ct.load`/`ct.store` signatures show `order`, `latency`, `allow_tma`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.load.html?utm_source=openai)) | Yes | D1: token graph is likely compiler-internal; explainability is missing at user level | Delta: surface token legality as an inspectable artifact |
| cuTile synchronization knobs are **for atomics** (order/scope), not general load/store | cuTile memory model: “two attributes for atomic operations: Memory Order… Memory Scope…” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)) | Yes | D1 must model cross-thread comm via scoped atomics or explicit protocol, not weak loads | Delta: tooling that prevents “accidentally weak” comm patterns |
| Hardware/toolchain gating is currently tight (Blackwell/CUDA 13.1 era) | cuTile Quickstart prerequisites list CC 10.x/12.x + CUDA 13.1+. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/quickstart.html?utm_source=openai)) and Triton-to-TileIR requires CUDA 13.1+ + Blackwell. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Yes | Evaluation must specify target + portability caveats | Delta: demonstrate correctness/benefit within the supported window |

---

### 2) Competitor baseline analysis (explicitly N1..N6)

#### Baseline table (what exists vs what is missing)

| ID | What it is | What it *already provides* (relevant facts) | Direct overlap risk vs D1/D2/D3 | What still looks unaddressed (room for novelty) |
|---|---|---|---|---|
| **N1** | CUDA Tile concept page | CUDA Tile is tile-based model; **based on Tile IR**; cuTile is user-facing (Python now, C++ future). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) | Low (it’s framing) | No “how to reason about token legality / schedules” story here |
| **N2** | cuTile Python docs | Tiles are **immutable values**; tile shapes compile-time constants; load/store move between arrays↔tiles. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python?utm_source=openai)) | Medium (D3: tuning hooks exist) | No explicit token graph control/inspection; scheduling/legality is implicit |
| **N3** | Tile IR spec | Formal memory model + tokens; memory ops token-ordered; token DAG ops (`make_token`, `join_tokens`); views are shape/strides-based; `allow_tma`/`latency` hints exist. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai)) | High overlap on *mechanisms* (tokens, hints) | Missing: **automatic minimal tokenization**, **proof/explanations**, **schedule artifacts** |
| **N4** | NVIDIA blog: Triton-to-TileIR backend | Active dev includes “semantic validation” & benchmarking; limitations: tensor-of-pointer degradation; suggests adopting **TMA load/store API** to avoid tensor-of-pointers. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Medium: they may build validators | Still no public “proof-carrying schedule / explainable token graph” artifact described |
| **N5** | cuTile Python repo | Confirms Tile IR basis + current Blackwell gating for `tileiras` in CUDA 13.1 era. ([github.com](https://github.com/NVIDIA/cutile-python)) | Low | Repo doesn’t advertise token-graph tooling at the user level (needs verification for passes) |
| **N6** | cuda-tile repo | Open-source MLIR dialect ecosystem + Python bindings + bytecode + conformance tests; aligned with CUDA 13.1. ([github.com](https://github.com/NVIDIA/cuda-tile)) | Medium: there may exist internal passes | Readme doesn’t claim explainable token-graph synthesis; novelty depends on what tooling exists in-tree |

#### “Is it already expressible / achievable?” (explicit check)

- **Token DAG schedules**: *Expressible* in Tile IR (make/fork/join tokens). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
  ⇒ If D1 claimed “we enable token DAGs”, that’s **not novel**.
- **TMA usage control**: cuTile & Tile IR both treat it as **hint/permission**, not a semantics guarantee. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai))  
  ⇒ If D2 claimed “we ensure TMA”, that’s **not a valid semantic promise**; must be “plan works with/without TMA”.
- **Layout expressivity**: Tile IR tensor_view is shape/strides (affine) + dimgroups; no public spec support for general non-affine/swizzle descriptors. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/semantics.html?utm_source=openai))  
  ⇒ D2 novelty could exist, but it risks being “yet another layout formalism” unless tightly scoped + shown to matter for Tile IR lowering.

---

### 3) Resolve top OPEN_QUESTIONS with ≤25‑word snippets

Below: each question → **evidence snippet(s)** (≤25 words) + what it implies.

#### Q0‑01 — Tile IR async/TMA lowering semantics & implied waits/barriers

- Snippet A (cuTile hint semantics): “may be lowered to use TMA … when the target architecture supports it.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai))  
- Snippet B (Tile IR hint semantics): “`allow_tma` – **suggest** whether to use TMA …” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  

**Resolution status:** *Partially resolved.* Public docs describe **permission/hints**, not the exact **structural preconditions** or any implied **wait/barrier insertion**. Those remain a Stage‑3 citation blocker.

#### Q1‑5‑01 — Tile IR view interface ambiguity (tensor_view vs partition_view; what load_view_tko accepts)

- Snippet A (tensor_view value model): “A tensor view value is logically a tuple of (ptr, shape, strides, dimgroups).” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/semantics.html?utm_source=openai))  
- Snippet B (partition_view model): “partition view … tuple of (tensor_view, tile_size).” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/semantics.html?utm_source=openai))  
- Snippet C (load_view_tko interface): “`load_view_tko` … loads a tile from a tile view.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  

**Resolution status:** *Mostly resolved at the semantic level.* Remaining ambiguity (for Stage‑3 precision): whether `load_view_tko` is specified/implemented uniformly for `tensor_view` (rank‑0 index) vs only “sub-views”.

#### Q2‑01 — Token expressivity (DAG vs linear chain; join/split idioms)

- Snippet A (join): “`join_tokens` … produces a fresh token which depends on all input tokens.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
- Snippet B (fresh token): “`make_token` … creates a fresh token with no prior dependencies.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  

**Resolution status:** *Resolved.* Tile IR supports **fork/join DAGs** using token reuse + `join_tokens`.

#### Q0‑02 — cuTile Python vs Tile IR tokens: implicit synthesis? user controls/diagnostics?

- Snippet A (cuTile ordering without sync): “without explicit synchronization, there is no guaranteed ordering of memory accesses across threads.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html))  
- Snippet B (cuTile sync knobs): “cuTile provides two attributes for atomic operations: Memory Order … Memory Scope …” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/memory_model.html))  

**Resolution status:** *Partially resolved.* We have evidence that **atomics** expose order/scope; we do **not** have a public “token graph debugging” interface. Inference (needs confirmation): since Tile IR requires tokens, and cuTile load/store don’t expose tokens, tokenization must be compiler-internal.

---

### 4) Per-direction novelty + feasibility audit (closest neighbors, delta, risk, killer evidence)

**Risk score meaning:** 1 = very safe novelty; 10 = high risk it’s “already done / obvious / hard to defend”.

#### Direction D1 — Explainable Token Graph + AsyncPlanIR (Legality‑First Overlap)

| Field | Audit |
|---|---|
| Closest neighbors | Tile IR’s **token ordering + memory model** itself (mechanism). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai)) Also Triton-to-TileIR’s roadmap mentions “semantic validation”. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| What is already achievable in stack? | You can write Tile IR with token DAGs (`make_token`, `join_tokens`) and order memory ops explicitly. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) |
| Our delta (must be the novelty) | **Automatic token-graph synthesis + minimization** (edge elision) with a **proof/validator** grounded in Tile IR memory model; plus **explainability artifact** (graph + justification + counterexample traces). Not “tokens”. |
| Feasibility | High: output remains valid Tile IR; does not require changing the spec—just a compiler/tool pass + sidecar. |
| Novelty risk | **5/10** (moderate). Dependence analysis exists broadly; but Tile IR’s “program deps don’t order” + token DAG legality makes a focused, explainable, proof-carrying tool defensible. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai)) |
| Killer evidence type (would collapse novelty) | A public CUDA Tile pass that (a) **auto-infers minimal token edges**, (b) emits **diagnostic token graphs**, (c) provides **formal validation outputs** comparable to our “proof-carrying” artifact. (Not found in docs yet.) |

#### Direction D2 — Affine+Swizzle Layout Descriptor → View Lowering

| Field | Audit |
|---|---|
| Closest neighbors | Linear Layouts (bit-level layout modeling over $$\mathbb{F}_2$$). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) ISL-based unified modeling of CuTe + linear layouts + swizzles. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) CuTe layout algebra foundations. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) Tile IR views are shape/strides (affine). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/semantics.html?utm_source=openai)) |
| Already achievable? | Affine layouts via tensor_view/partition_view are native. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/semantics.html?utm_source=openai)) TMA selection is hint-based. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai)) |
| Our delta | A **restricted swizzle decorator** + formal reasoning (ISL/$$\mathbb{F}_2$$) that *preserves* structure through lowering—plus a measurable “fallback-to-pointer-tile” cliff metric. |
| Feasibility | Medium/High risk: may require either (a) new view types, or (b) encoding swizzles as address transforms without breaking semantics. Public spec doesn’t spell out such a descriptor. |
| Novelty risk | **7/10** (high). Theoretical layout modeling is already an active area; novelty must be “what Tile IR needs + what it can’t express + minimal extension + evidence of impact.” |
| Killer evidence type | Tile IR spec (or cuda-tile repo) already includes a **non-affine/swizzle view** or a documented “linear layout” representation and lowering path to TMA/views. Not found in the docs we accessed. |

#### Direction D3 — Legality-aware compile-time bounded autotuning

| Field | Audit |
|---|---|
| Closest neighbors | cuTile provides ByTarget + hints (latency/allow_tma) and kernel config knobs. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai)) Triton-to-TileIR caches artifacts (.tileIR) per blog. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| Already achievable? | Architecture-conditional parameters + hints exist; compilers already cache compiled artifacts. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| Our delta | Make **compile time** a first-class objective; canonicalize & cache (layout+plan+token graph); prune illegal variants early using validator. |
| Feasibility | Medium: requires instrumentation + caching discipline + solver/validator budgets; also needs evidence compile-time dominates. |
| Novelty risk | **8/10** (very high) unless backed by measurements showing compile-time is the bottleneck and legality pruning materially helps. |
| Killer evidence type | Measurements showing compile-time is *not* limiting, or existing tooling already provides legality-aware cache keys + pruning. |

---

### 5) Strategic recommendation (final direction / pivot)

**Recommendation: finalize D1 (Explainable Token Graph + AsyncPlanIR), with one crucial reframing.**

1. **Reframe novelty correctly:** Tile IR already supports token DAGs via `make_token` and `join_tokens`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
   So the paper must not claim “we introduce token DAG scheduling”; it must claim:
   - **Token-graph synthesis** (derive a minimal, legal DAG from a higher-level program or conservative baseline),
   - **Proof/validator** tied explicitly to Tile IR memory model rules (tokens, happens-before, scopes, data-race UB). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai))
   - **Explainability** (why each edge exists; what breaks if removed; counterexample witness).

2. **Keep TMA as an optimization hint, not semantics:** both cuTile and Tile IR treat TMA as optional/hinted. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai))  
   AsyncPlanIR should remain valid under “TMA used” and “TMA not used”.

3. **Pull in layout research *as an analysis aid* (not as D2):** use ISL/$$\mathbb{F}_2$$ modeling (P2/P1) to prove **disjointness / non-alias** of tile footprints, enabling safe token edge elision. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
   This keeps the project anchored to the layout papers without betting the paper on extending Tile IR views.

4. **Defer D2/D3 to “future work” unless you find killer evidence:** D2 is tempting but high risk; D3 needs measurement.

---

## Stage‑3 Assembly Pack (paste‑ready)

### Working title
**Explainable Token-Graph Synthesis and Proof-Carrying Async Plans for CUDA Tile IR**

### Abstract (5–7 bullets)
- CUDA Tile IR orders memory operations using **explicit tokens**; program/data/control dependencies do not order memory operations, making legality non-obvious. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai))  
- This creates “legality cliffs”: small scheduling or lowering changes can introduce **undefined behavior** via data races or missing token order. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai))  
- We present **TEI/TRSC**, a token-edge inference framework that constructs a **minimal token dependency DAG** for Tile IR memory operations.
- We introduce **AsyncPlanIR**, a first-class schedule artifact that encodes overlap (prefetch/load/compute/store) subject to token legality and lowers to Tile IR tokens (`make_token`, `join_tokens`) plus performance hints (latency/allow_tma). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
- Our compiler emits an **explainable token graph** with per-edge justifications and counterexample traces for illegal edge deletions.
- Evaluation shows improved overlap opportunities with preserved correctness, reduced debugging time for ordering bugs, and competitive or improved runtime vs CUDA Tile baselines (cuTile / Triton-to-TileIR).

### Contributions (3–5 bullets)
1. **Explainable Token Graph Synthesis (TEI/TRSC):** derive token DAGs for Tile IR memory ops; prove legality under Tile IR memory model (token order, scopes, happens-before). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai))  
2. **AsyncPlanIR (proof-carrying schedule):** a verifiable plan that lowers to existing Tile IR token ops (`make_token`, `join_tokens`) and memory op hints (latency/allow_tma). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
3. **Validator + counterexamples:** bounded checker that rejects plans/token graphs that allow data races or violate required ordering; emits a witness trace.
4. **Usability artifact:** human-readable dependency graph + “why this edge exists” + “what breaks if removed”.

### System overview (artifact built)
- **Input:** Tile IR (or MLIR that lowers to Tile IR) produced by CUDA Tile stack (cuTile Python, Triton-to-TileIR, or other DSLs). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- **Output:**  
  - Tile IR program with **explicit token DAG** (using `make_token`, `join_tokens` and token-ordered memory ops), ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
  - Sidecar **ExplainableTokenGraph** (nodes=memory ops, edges=token deps, annotations=justification, footprint summaries, scope/order assumptions),  
  - Optional **AsyncPlanIR** schedule + validation certificate.

### Key semantics/definitions that MUST appear in the paper
- **Token-ordered operations:** memory ops’ relative order is undefined unless connected by tokens. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html?utm_source=openai))  
- **Token order vs program dependencies:** program deps don’t order memory ops; tokens must be used. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai))  
- **Scopes & memory ordering:** weak vs scoped operations; happens-before and data race UB framing. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai))  
- **Token DAG primitives:** `make_token`, `join_tokens`, and fork/join idioms. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
- **Hint semantics:** `allow_tma` / `latency` are performance hints; TMA is optional (“may be lowered”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai))  

### Method sketch (algorithm/passes)
1. **Parse & summarize memory operations:** identify token-ordered memory ops; record memory_ordering_semantics + scope, and view/pointer footprint descriptors. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html?utm_source=openai))  
2. **Footprint reasoning:** compute conservative overlap relations between ops (by view index space + shape/stride; optional ISL-based reasoning where applicable). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/semantics.html?utm_source=openai))  
3. **Conservative tokenization:** start from a safe token chain inside a region (sound-by-construction baseline).  
4. **Edge-elision (TEI):** remove token edges when proven unnecessary (disjoint footprints + memory model constraints).  
5. **Structured concurrency (TRSC):** introduce regions/annotations that group ops sharing a token discipline; materialize fork/join using token reuse + `join_tokens`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))  
6. **AsyncPlanIR synthesis:** choose a legal overlap schedule (load early, compute later, store after) subject to token constraints; attach `latency` / `allow_tma` hints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai))  
7. **Validator:** check for forbidden races / missing required ordering; emit witness on failure.

### Evaluation plan + baselines (must include CUDA Tile stack)
**Benchmarks**
- **Correctness-focused microbenchmarks:** Tile IR memory-model litmus patterns (“store buffering”, “load buffering”, intra-tile-block hazards) adapted to token ordering rules. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai))  
- **Kernel microbenchmarks:** load/compute/store pipelines, reductions, gather/scatter, transpose; include cases with and without view-based accesses.  
- **Real workloads:** Triton kernels compiled with Triton-to-TileIR; cuTile Python sample kernels; at least one attention-like pipeline where overlap matters. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

**Baselines (explicit)**
- **B0 (CUDA Tile stack “as-is”):** cuTile Python compiler output (no post-pass) + its recommended hints. ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- **B1 (Conservative tokens):** enforce a single token chain (maximal ordering) — correctness-first, minimal overlap.  
- **B2 (Triton-to-TileIR):** Tile IR backend output as currently available. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- **B3 (Optional):** Triton PTX backend as “legacy performance reference” (if comparable).

**Metrics (beyond speedup)**
- Runtime: throughput/latency, bandwidth utilization, achieved occupancy.  
- **Legality metrics:** number of token edges, graph width, validator failures caught, race counterexamples generated.  
- **Compile-time metrics:** pass runtime, number of variants pruned, cache hit rate if used.  
- **Debuggability:** time-to-root-cause for injected ordering bugs; size/clarity of explanation artifact.

**Experimental constraints**
- Explicitly state supported environments (CUDA 13.1 era; Blackwell gating for some toolchains). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

### Related work shortlist (cite if already in hand; otherwise TODO)
- Tile IR memory model + token ordering semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai))  
- CUDA Tile / cuTile Python overview (as the target ecosystem). ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile))  
- Layout reasoning foundations (optional analysis component): Linear Layouts (P1); ISL modeling (P2); CuTe layout algebra (P3). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- TODO (needs citations): Triton autotuning & compilation caching; polyhedral dependence analysis; software pipelining on GPUs.

### Threats / limitations (1–3 bullets)
- Public docs do not specify **exact TMA lowering preconditions** or implicit synchronization; AsyncPlanIR must be robust to backend variability. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai))  
- Token-graph minimization depends on conservative footprint/alias analysis; overly conservative analysis may limit benefits.  
- NVIDIA toolchain may add similar legality tooling; novelty must be defended as “explainable + proof-carrying + user-facing” rather than “we found tokens”.

---

## CONTEXT_CAPSULE

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

  current_stage: 2.5
  last_updated: "2026-02-03"

  problem_statement_1liner: "Make CUDA Tile IR token legality + async overlap explicit via explainable token graphs and proof-carrying async plans."
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

    # Verified Tile IR memory/token semantics (now evidence-backed)
    - id: "GM-tileir-token-order"
      text: "Tile IR: memory ops are token-ordered; ordering is undefined unless connected by tokens; program deps do not order memory ops."
      last_update_date: "2026-02-03"
    - id: "GM-tileir-token-dag"
      text: "Tile IR supports token DAGs via make_token + join_tokens (fork by reusing a token; join via join_tokens)."
      last_update_date: "2026-02-03"
    - id: "GM-tileir-weak-scope"
      text: "Tile IR: weak ops assume no concurrent access and cannot communicate; scoped + ordered ops required; data races are undefined behavior."
      last_update_date: "2026-02-03"

    # Verified hint semantics
    - id: "GM-allow-tma-hint"
      text: "cuTile/Tile IR: allow_tma + latency are performance hints; TMA use is optional ('may be lowered' / 'suggest'). Preconditions not specified in public docs."
      last_update_date: "2026-02-03"

    # Direction framing (updated)
    - id: "GM-dir-D1-final-framing"
      text: "D1 must not claim 'token DAG novelty' (already in Tile IR). Novelty is token-graph synthesis/minimization + validator + explainability + proof-carrying async plans."
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
    - id: "V6-G1"
      decision: "Layout expressivity beyond affine strides is useful, but Stage-2.5 priority is legality/temporal artifacts; layout reasoning can be used as analysis aid."
      rationale_pointers: ["Gap_ID: G1", "Stage2.5 audit"]
      date: "2026-02-03"
      status: "active"
    - id: "V8-Stage2.5"
      decision: "FINAL pick: D1 (Explainable Token Graph + AsyncPlanIR), reframed so novelty is synthesis/validation/explainability—not token DAG mechanics."
      ratnale_pointers:
        - "Tile IR already has make_token/join_tokens; D1 remains novel as a tooling layer (proof-carrying + explainable)."
        - "D2/D3 kept as future work unless new evidence shows a minimal extension is required."
      date: "2026-02-03"
      status: "active"

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260203"
  stage1_gap_audit: "WP1_20260203"
  stage1_5_toolbox: "WP1_5_20260203"
  stage2_directions: "WP2_20260203"
  stage2_5_novelty_audit: "WP2_5_20260203"
  stage3_paper: null
  stage3_assembly_pack: |
    Title: Explainable Token-Graph Synthesis and Proof-Carrying Async Plans for CUDA Tile IR
    Contributions: (1) token-graph synthesis+minimization w/ validator; (2) AsyncPlanIR lowering to Tile IR tokens+hints; (3) explainability + counterexamples.
    Eval: correctness litmus + kernel microbench + cuTile/Triton-to-TileIR baselines; metrics include runtime, compile-time, token-graph metrics, debug time.

OPEN_QUESTIONS:
  - id: "Q3-01"
    text: "TMA lowering details: what exact structural/layout constraints trigger TMA when allow_tma is enabled, and what implied waits/barriers exist (if any)?"
    priority: "high"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir allow_tma constraints TMA descriptor"
      - "site:docs.nvidia.com/cuda/cutile-python allow_tma lowering details"
      - "site:github.com/NVIDIA/cuda-tile TMA allow_tma"
    last_update_date: "2026-02-03"
  - id: "Q3-02"
    text: "Token tooling overlap risk: does cuda-tile or cuTile compiler already perform minimal token inference and/or emit token-graph diagnostics?"
    priority: "high"
    suggested_queries:
      - "site:github.com/NVIDIA/cuda-tile token graph"
      - "site:github.com/NVIDIA/cuda-tile join_tokens pass"
      - "site:github.com/NVIDIA/cutile-python token"
    last_update_date: "2026-02-03"
  - id: "Q3-03"
    text: "View interface precision: confirm whether load_view_tko/store_view_tko accept tensor_view directly (rank-0 index) vs only partition_view in current implementations."
    priority: "medium"
    suggested_queries:
      - "site:docs.nvidia.com tile-ir load_view_tko tensor_view"
      - "site:github.com/NVIDIA/cuda-tile load_view_tko tensor_view"
    last_update_date: "2026-02-03"
  - id: "Q3-04"
    text: "Evaluation harness: identify the most practical way to extract Tile IR from cuTile/Triton-to-TileIR and round-trip through tileiras/driver JIT for measurement."
    priority: "medium"
    suggested_queries:
      - "site:github.com/NVIDIA/cuda-tile cuda-tile-translate tileiras example"
      - "site:developer.nvidia.com Triton-to-TileIR cache .tileIR"
    last_update_date: "2026-02-03"

NEXT_STAGE_HINT:
  stage: "3"
  paste_next: |
    Paste the entire updated STATE from this CONTEXT_CAPSULE.

    Stage 3 ("Write the paper + full assembly pack refinement"):
      1) Expand the Stage-3 Assembly Pack into full sections (intro, model, algorithm, proofs/validator, eval).
      2) Close OPEN_QUESTIONS Q3-01..Q3-04 with citations (especially TMA + existing token passes).
      3) Add a precise threat model: what races are prevented, what remains UB, how validator treats weak ops/scopes.
```

### ARTIFACT_SUMMARY (this stage)
- Produced **WP2_5_20260203**: evidence-backed novelty + feasibility audit versus **CUDA Tile / cuTile / Tile IR / Triton-to-TileIR** (N1–N6).
- Verified Tile IR token DAG expressivity via `make_token` + `join_tokens`; updated novelty framing accordingly.
- Resolved/partially resolved top OPEN_QUESTIONS with **≤25-word** cited snippets (TMA hints; view models; tokenns; cuTile atomic memory model).
- Finalized strategic recommendation: **D1** as the Stage‑3 paper direction, explicitly as a tooling/verification/explainability layer.
- Produced a full **Stage‑3 Assembly Pack** in WORK_PRODUCT and stored a compact copy in STATE.

### NEXT_STAGE_HINT
See `NEXT_STAGE_HINT` in the YAML STATE above (Stage 3).

---
Learn more:
1. [8. Operations — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html?utm_source=openai)
2. [7. Memory Model — (https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html?utm_source=openai)
3. [8. Operations — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)
4. [6. Semantics — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/semantics.html?utm_source=openai)
5. [Performance Tuning — cuTile Python](https://docs.nvidia.com/cuda/cutile-python/performance.html?utm_source=openai)
6. [cuda.tile.load — cuTile Python](https://docs.nvidia.com/cuda/cutile-enerated/cuda.tile.load.html?utm_source=openai)
7. [Memory Model — cuTile Python](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)
8. [Quickstart — cuTile Python](https://docs.nvidia.com/cuda/cutile-python/quickstart.html?utm_source=openai)
9. [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
10. [CUDA Tile | NVIDIA Developer](https:veloper.nvidia.com/cuda/tile)
11. [cuTile Python — cuTile Python](https://docs.nvidia.com/cuda/cutile-python?utm_source=openai)
12. [GitHub - NVIDIA/cutile-python: cuTile is a programming model for writing parallel kernels for NVIDIA GPUs](https://github.com/NVIDIA/cutile-python)
13. [GitHub - NVIDIA/cuda-tile: CUDA Tile IR is an MLIR-based intermediate representation and compiler infrastructure for CUDA kernel optimization, focusing on tile-based computation patterns and optimizations targeting NVIDIA teor core units.](https://github.com/NVIDIA/cuda-tile)
14. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
15. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
16. [https://arxiv.org/pdf/2601.05972v1](https://arxiv.org/pdf/2601.05972v1)
