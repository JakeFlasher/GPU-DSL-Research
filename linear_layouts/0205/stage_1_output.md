## WORK_PRODUCT

### 1) Table: Axiom‑vs‑Tooling‑vs‑Gap Matrix

| Gap_ID | Premise/axiom | What tooling already measures | What is missing | Why reviewers care | Evidence needed | Measurement hook (metric/tool output) | Threat to validity |
|---|---|---|---|---|---|---|---|
| **G1 (Feasibility cliff)** | **A1:** Main claims require **spill/reg/occupancy observables on Tile‑executed kernels** (not just SIMT). | NCU measures spill/reg proxies for SIMT in principle: `launch__registers_per_thread` (C03), `derived__local_spilling_requests(_pct)` (C06), `sass__inst_executed_register_spilling*` (C05), `sass__inst_executed_local_loads/stores` (C05), achieved occupancy `sm__warps_active...` (C07), metric enumeration via `ncu --query-metrics` (C09). | **No current proof** that these metrics are **non‑`n/a` / stable** for **Tile workloads** on your Blackwell + NCU stack (OQ05). | If Tile metrics are `n/a`, paper collapses into runtime anecdotes (reviewers reject). | 1–2 Tile kernels where (i) Tile provenance is confirmed, (ii) ≥1 spill metric is numeric, (iii) repeated runs stable. | **Metric availability matrix** per kernel: attempt collection of the **non‑negotiable list**; record numeric vs `n/a` vs error; include provenance metric once verified (see G2). | False negatives from replay instability; “numeric” values that are meaningless for Tile; cherry‑picking kernels where metrics happen to work. |
| **G2 (Credibility cliff)** | **A2:** Backend comparisons must be **fail‑closed provenance‑clean**; TileIR→PTX fallback must not silently contaminate A/B. | Triton‑to‑tile‑IR documents `ENABLE_TILE=1` (C10) and fallback to PTX on Tile compilation bugs (C11). Blog/repo suggest cache artifacts differ (E06; but not yet a formal oracle in state). | Missing a **per‑kernel‑launch provenance oracle in your pipeline** + a **coverage/fallback rate ledger** (OQ02). Also missing verification that NCU exposes/exports a usable “Tile vs SIMT” label in *your* workflow (new OQ06). | Without defensible provenance, reviewers treat all deltas as potentially mislabeled execution. | Demonstrate: under `ENABLE_TILE=1`, classify launches into {Tile, SIMT/fallback, unknown}; publish “coverage rate” and exclude unknowns. | Primary hook (target): **NCU per‑launch execution‑model label** (metric name claimed in proposal review; must verify in your stack). Secondary hook: **artifact corroboration** (cache extension / dump outputs) + cache isolation. | Cache reuse causing cross‑condition artifact mixing; range aggregation hiding per‑launch variation; multi‑kernel TB ops mixing models. |
| **G3 (Feasibility for H2)** | **A3:** Occupancy hint must be **user‑controllable** and **applied** to test H2. | Repo states TileIR adds **occupancy hint 1..32, default 1** (C13). | Missing: **exact Triton user‑level API / knob** to set/sweep the hint, and proof it is **not ignored** (OQ03). | H2 is a major “knob‑mechanism” contribution; if untestable, novelty shrinks. | 2‑point proof: occupancy=1 vs occupancy=k changes at least one measurable quantity (occupancy/runtimes/spills) with other knobs fixed. | Sweep occupancy (once API found): collect `sm__warps_active...` (C07), `launch__registers_per_thread` (C03), spill metrics (C05/C06), runtime. | Confounded by kernel shape changes, num_ctas, or algorithmic differences; hint applies only to certain kernel classes. |
| **G4 (Mechanism validity)** | **A4:** “Spills” must be separated from generic local memory traffic; metrics must be validated **under Tile**. | NCU separates local LD/ST counts from spill‑caused instruction counts (C02, C05) and provides derived local spilling requests (C06). | Missing: empirical validation that (i) derived spill metrics correlate with SASS spill counters on Tile, and (ii) local arrays don’t masquerade as spills (MB2‑style calibration). | Reviewers will attack interpretation (“you measured local traffic, not spills”). | A calibration triad (local‑array only vs forced‑spill only vs both) with expected signature, on both backends where possible. | Collect: `derived__local_spilling_requests(_pct)` (C06), `sass__inst_executed_register_spilling*` (C05), `sass__inst_executed_local_loads/stores` (C05), `launch__stack_size` (C26). | Compiler DCE/optimization breaking “forced spill”; Tile lowering introduces different instruction mix; replay pass skew. |
| **G5 (Profiling perturbation)** | **A5:** NCU replay/multipass must not invalidate mechanism trends; overhead must be characterized. | NCU supports metric enumeration (C09) and standard metric collection; NCU warns about register metric caveats (C04) and exposes occupancy metrics (C07). | Missing: measured **NCU overhead and stability protocol** for Tile runs (minimal metric set vs full set, replay sensitivity). | MICRO/ISCA/ASPLOS expect explicit handling of measurement bias. | An ablation: runtime (no NCU) vs runtime (NCU minimal metrics) vs runtime (NCU expanded metrics) + variance over repeats. | Runtime via CUDA events (EVAL_PLAN); NCU metric collection runs with two metric sets; compare deltas/variance. | Clock/power drift; data‑dependent kernels changing across replay passes; TB ops with dynamic kernels. |
| **G6 (Register metric interpretation)** | **A6:** Must not over‑interpret `launch__registers_per_thread` as live regs; need explainable triangulation. | NCU warns reg count can exceed live due to allocation holes/ABI/instruction constraints (C04) and exposes `launch__stack_size` (C26); spill metrics exist (C05/C06). | Missing: paper‑grade interpretation rules + plots that triangulate regs vs stack size vs spill counters across backends. Also “Live Registers” metric name/usage not pinned (C25 lacks metric ID). | Avoids “reg count ⇒ liveness” criticism; strengthens mechanism attribution. | Show cases where regs/thread rises without spills, and cases where spills rise with stack size; document interpretation rubric. | Collect: `launch__registers_per_thread` (C03), `launch__stack_size` (C26), spill counters (C05/C06), plus identify “Live Registers” field via `ncu --query-metrics` search (needs verification). | Misleading correlations; backend changes instruction mix; different calling conventions between models. |
| **G7 (Toolchain pinning/compat)** | **A7:** TileIR profiling depends on correct GPU/driver/CUDA/NCU coupling; must be pinned and checked. | You can query versions via standard tooling (not yet recorded in state); repo claims CUDA 13.1 + Blackwell constraints (C12). | Missing: **automated environment manifest + preflight checks**, including GPU SM/CC and any min driver/NCU constraints (OQ01, new OQ11). | Without pinning, results are non‑reproducible and feasibility can silently fail. | A single manifest file per run containing GPU, driver, CUDA, NCU, Triton commit; plus preflight that fails fast. | Measurement hook: `nvidia-smi` dump + CUDA runtime/toolkit version + `ncu --version` + `ncu --query-metrics` snapshot + Triton/Triton-to-tile-IR commit hash. | Hidden upgrades mid‑project; mixed nodes; “it worked once” but not reproducible. |
| **G8 (Confound factorization)** | **A8:** Must factor **num_ctas** and math modes (approx/FTZ) as explicit experimental factors. | Repo exposes `TILEIR_ENABLE_APPROX`, `TILEIR_ENABLE_FTZ` (C14) and states `num_ctas=2` can enable 2CTA MMA (C16). | Missing: factorial design + reporting so backend deltas aren’t really “MMA mode” or “math mode.” | Reviewers will attribute speedups/slowdowns to hidden mode changes, not backend. | Ablations: backend × num_ctas × approx/FTZ for at least one compute‑heavy motif. | Hook: sweep env vars + `num_ctas`; collect runtime + reg/spill/occupancy metrics (EVAL_PLAN). | Numeric differences can break correctness checks; compute path changes instruction mix. |
| **G9 (Artifact trail reproducibility)** | **A9:** Need auditable artifact trail mapping “kernel config → compiled artifact(s)” for provenance and reproduction. | Upstream Triton documents dump/override env vars (not in claim ledger yet; mentioned in prior review doc). Tile repo/blog suggest distinct cache artifacts (E06). | Missing: verification that **dump/override** works in Triton‑to‑tile‑IR fork and captures Tile artifacts; plus a stable manifest format (OQ08). | Artifact trail upgrades methodology from “trust me” to reviewable. | Demonstrate artifact capture for both backends; show mapping from kernel hash to artifact filenames; show cache isolation prevents cross‑mix. | Hook: run with dump/override env vars (once verified) + file system manifest; corroborate with NCU provenance label when available. | Artifact formats may drift; cache path changes; mixed compilation stages. |
| **G10 (Ecological validity + mapping)** | **A10:** TB subset must be non‑cherry‑picked and per‑kernel mapping must exist (TB ops often multi‑kernel). | TritonBench has a standard op entrypoint (E04). | Missing: deterministic TB subset rule + per‑kernel mapping to provenance+metrics (multi‑kernel range export details) (new OQ07). | Reviewers distrust cherry‑picked ops and aggregate metrics over heterogeneous kernels. | A pinned TB commit + selection rule + per‑kernel table: kernel name/hash → execution model → key metrics. | Hook: NCU range profiling/export per kernel launch (once pipeline confirmed) + TB harness wrapper that logs kernels. | Range aggregation hides outliers; TB updates change kernels; non‑determinism in autotuning. |

---

### 2) Table: Gap → Microbench Motif

| Gap_ID | Minimal microbench motif | Control knobs (tile size, unroll, etc.) | Expected observable delta | Baseline(s) | Confounders |
|---|---|---|---|---|---|
| **G1** | **MB0 “HelloTile” + MB1 reg‑pressure ladder** (single‑kernel, long enough to profile) | backend (`ENABLE_TILE`), reg‑pressure knob (unroll / #accumulators), fixed problem size | For Tile‑executed runs: non‑negotiable metrics are numeric (not `n/a`); spill metrics become nonzero past a knee | Same kernel under `ENABLE_TILE=0` (SIMT) | Replay instability; kernel too short; dead‑code elimination |
| **G2** | **Provenance harness test**: run same kernel under `ENABLE_TILE=0` and `ENABLE_TILE=1`; log artifacts; (optionally) TB multi‑kernel op | cache isolation on/off; `TRITON_ALWAYS_COMPILE` (if verified); multi‑kernel vs single | Coverage ledger shows {Tile, SIMT} buckets, and unknowns go to exclusion; fallback bucket measurable if it occurs | SIMT run is known‑SIMT control; Tile run should be known‑Tile if stack works | Cache reuse; kernel name collisions; per‑range aggregation hiding per‑launch |
| **G3** | **MB3 occupancy‑hint sweep kernel** (compute‑heavy, stable runtime) | occupancy hint 1..32 (once API found), reg‑pressure constant, num_ctas fixed | Achieved occupancy and/or runtime changes monotonically in some region; reg/spill tradeoff visible | Default occupancy hint (1) + PTX backend | API missing/ignored; num_ctas confounds; thermal drift |
| **G4** | **MB2 spill vs local traffic triad**: (i) explicit local array, (ii) forced spill, (iii) both | array size/indexing; live‑range forcing; reg‑pressure knob | Derived spill pct high in forced‑spill; local LD/ST rises in local‑array even when spill pct stays low | Run triad on PTX and Tile | Compiler optimizing away spill pressure; Tile lowering changes memory ops |
| **G5** | **MB5 NCU overhead/replay ablation** (single kernel, long) | metric‑set size: minimal vs expanded; repeats; warmup | Runtime overhead quantified; metric variance increases with expanded set | No‑NCU runtime baseline | Clock variability; background load |
| **G6** | **MB1 reg ladder + stack correlation** | unroll / accumulators; keep memory traffic fixed | `launch__registers_per_thread` increases smoothly; `launch__stack_size` and spill counters jump at spill onset | Backend A/B | “holes” in reg alloc; different ABI between models |
| **G7** | **MB‑Preflight**: compile+run one Tile kernel + record manifest | none (just environment capture) | Either hard fail early (missing prerequisites) or produce manifest attached to every run | N/A | Multi‑node drift; container vs host mismatches |
| **G8** | **MB4 MMA/dot kernel** sensitive to `num_ctas` | `num_ctas ∈ {1,2}`, backend, (optional) approx/FTZ toggles | Separate “backend effect” from “2CTA MMA effect”; interactions visible | num_ctas=1 as control | Kernel algorithm changes; correctness diffs under approx |
| **G9** | **Artifact capture check**: run 1 kernel with dump/override env vars | dump dir; override dir; always‑compile flag (if verified) | Artifacts produced for both backends; stable naming; reproducible mapping | Upstream Triton as reference | Fork semantics drift; artifacts incomplete |
| **G10** | **TB‑Range motif**: 1 multi‑kernel TritonBench op under NCU range mode | op choice; deterministic shapes; autotune off/on | Per‑kernel table shows mixed kernels and their provenances; avoids aggregate ambiguity | A simple TB op (single kernel) | Autotuner variability; kernel fusion changes across commits |

---

### 3) Stage‑1 Verdict (≤12 bullets)

- **Top‑3 gaps dominating acceptance risk**
  - **G1 (Tile metric availability / `n/a` cliff):** If spill+reg metrics aren’t numeric for Tile, there is no mechanism attribution → paper likely rejected.
  - **G2 (Fail‑closed provenance + fallback accounting):** If you can’t prove “what ran,” A/B results are indefensible, especially with documented TileIR→PTX fallback.
  - **G3 (Occupancy hint controllability):** H2 is a high‑value knob story; if the hint can’t be set or is ignored, novelty shrinks and evaluation becomes thin.

- **Claim_IDs that must be VERIFIED before Stage 2 directions**
  - **C10–C11** are already VERIFIED (backend toggle + fallback hazard) but must be **operationalized** into a measured provenance ledger (Stage2 will depend on it).
  - **C06/C05/C07/C26** must be validated as **collectable on Tile** in your environment (new UNVERIFIED “environment‑instantiated” claims needed; see capsule additions C27+).
  - **C13 (occupancy hint existence)** is VERIFIED, but the **user‑accessible API + effect** must be VERIFIED (new claim C29; OQ03).
  - **C25 (“Live Registers exists”)** needs the **exact metric ID / section export path** verified before being used in plots.

---

## CONTEXT_CAPSULE

```text
STATE_BEGIN
profile:
  project_name: "TileSpill: TileIR vs PTX register pressure & spilling (Blackwell microbench)"
  target_model: "gpt-5.2-pro (web UI)"
  operating_mode: "manual_state_v4_2_tilespill (web UI; milestone resets; monotonic capsule)"
  stage_plan: ["0", "1", "1.5", "2", "2.5", "3"]

  conference_targets: ["ISCA", "MICRO", "ASPLOS"]
  paper_genre: "microarchitecture + toolchain characterization proposal"
  hard_constraints:
    - "No hallucinated citations"
    - "No invented metric names/tool behavior"
    - "Microarch scope lock: bare-metal microbench + NCU + PTX/TileIR A/B"
    - "Capsule monotonic: ledgers append-only; capsule must not shrink without explanation"

  current_stage: 1
  last_updated: "2026-02-04"

  scope_lock:
    research_question_1liner: "On Blackwell GPUs, quantify and attribute how Triton PTX vs Triton CUDA Tile IR backends change register allocation/spilling and how that impacts runtime/occupancy under controlled tuning knobs."
    hypotheses:
      - id: "H1"
        statement: "Backend switch (ENABLE_TILE=0/1) changes regs/thread + spill metrics + runtime."
        sweeps: ["backend: ENABLE_TILE ∈ {0,1}"]
        metrics: ["launch__registers_per_thread", "sass__inst_executed_register_spilling*", "derived__local_spilling_requests*", "runtime_us_cuda_events"]
      - id: "H2"
        statement: "TileIR occupancy hint sweep (1..32) trades regs/spills vs achieved occupancy and runtime."
        sweeps: ["tileir_hint: occupancy ∈ {1..32}"]
        metrics: ["launch__registers_per_thread", "derived__local_spilling_requests_pct", "sm__warps_active.avg.pct_of_peak_sustained_active", "runtime_us_cuda_events"]
      - id: "H3"
        statement: "Register-pressure parameter sweeps show spill-onset thresholds; thresholds differ across backends."
        sweeps: ["unroll factor", "tile size", "reduction dim"]
        metrics: ["sass__inst_executed_register_spilling*", "derived__local_spilling_requests*", "runtime_us_cuda_events"]
      - id: "H4"
        statement: "Tensor-of-pointer vs descriptor/TMA-oriented path changes reg pressure/spills and runtime; direction may be backend-dependent."
        sweeps: ["API style: tensor-of-pointer vs descriptor/TMA (when applicable)"]
        metrics: ["launch__registers_per_thread", "sass__inst_executed_register_spilling*", "derived__local_spilling_requests*", "runtime_us_cuda_events"]

    primary_knobs_to_sweep:
      - "backend: PTX vs TileIR (ENABLE_TILE=0/1)"
      - "TileIR occupancy hint: occupancy=1..32 (per-kernel hint; default 1)"
      - "Triton hint: num_ctas ∈ {1,2} (Blackwell dot/dense workloads control)"
      - "tensor-of-pointer vs descriptor/TMA rewrite (where applicable)"
      - "kernel parameters controlling register pressure (unroll, tile size, reduction dim)"

    excluded_topics:
      - "hardware redesign / RTL"
      - "full model training as primary evidence"
      - "unmeasurable compiler internals"
      - "Tile IR memory-model correctness (tokens/scopes) as primary contribution"

    success_criteria:
      - "Reproducible A/B switch with backend-used detection (avoid TileIR→PTX fallback contamination)"
      - "At least 1 actionable finding about spills/regs on Blackwell with clear metric signature + knob recommendation"
      - "Public-ish microbench slice + scripts that produce plots/tables"
      - "Evaluation includes controls + threats to validity (math modes, toolchain pinning, repeats)"

  environment_inventory:
    gpus_available:
      - name: "H100"
        notes: "PTX baseline + cross-arch sanity; TileIR backend likely unavailable"
        cc: null
      - name: "RTX 5090"
        notes: "Blackwell-class: run PTX vs TileIR A/B"
        cc: null
      - name: "B200"
        notes: "Datacenter Blackwell: run PTX vs TileIR A/B"
        cc: null
      - name: "GB10"
        notes: "Blackwell-family? confirm exact SKU/cc"
        cc: null

    toolchain_to_freeze:
      cuda_version: null
      driver_version: null
      triton_version: null
      python_version: null
      ncu_version: null
      tileir_stack_version: null
      env_vars:
        ENABLE_TILE: "0/1 (TileIR backend switch in Triton-to-tile-IR repo)"
        TILEIR_ENABLE_APPROX: "0/1 (TileIR approx mode; default disabled per repo)"
        TILEIR_ENABLE_FTZ: "0/1 (TileIR FTZ; default disabled per repo)"

GOLDEN_SOURCES:
  - id: "NV-NCU"
    kind: "nvidia_primary"
    title: "Nsight Compute Profiling Guide"
    url: "https://docs.nvidia.com/nsight-compute/ProfilingGuide/"
    last_verified: "2026-02-04"
  - id: "NV-PTX"
    kind: "nvidia_primary"
    title: "PTX ISA docs"
    url: "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"
    last_verified: "2026-02-04"
  - id: "NV-TILEIR-MM"
    kind: "nvidia_primary"
    title: "Tile IR memory model"
    url: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html"
    last_verified: "2026-02-04"
  - id: "NV-TILE-BLOG"
    kind: "nvidia_primary"
    title: "NVIDIA blog: Triton-to-TileIR backend"
    url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
    last_verified: "2026-02-04"
  - id: "TILE-REPO"
    kind: "repo"
    title: "triton-lang/Triton-to-tile-IR"
    url: "https://github.com/triton-lang/Triton-to-tile-IR"
    last_verified: "2026-02-04"
  - id: "CUDA-TILE-REPO"
    kind: "repo"
    title: "NVIDIA/cuda-tile"
    url: "https://github.com/NVIDIA/cuda-tile"
    last_verified: "2026-02-04"
  - id: "TB"
    kind: "benchmark_harness"
    title: "meta-pytorch/tritonbench"
    url: "https://github.com/meta-pytorch/tritonbench"
    last_verified: "2026-02-04"
  - id: "P1"
    kind: "seed_paper"
    title: "Linear Layouts (Triton)"
    url: "https://arxiv.org/html/2505.23819v3"
    last_verified: null
  - id: "P2"
    kind: "seed_paper"
    title: "ISL layout relations"
    url: "https://arxiv.org/html/2511.10374v1"
    last_verified: null
  - id: "P3"
    kind: "seed_paper"
    title: "Categorical CuTe layouts"
    url: "https://arxiv.org/pdf/2601.05972v1"
    last_verified: null

GLOBAL_MEMORY:
  notes:
    - id: "GM-format"
      text: "Deliverables: WORK_PRODUCT then CONTEXT_CAPSULE; Stage 3 outputs LaTeX only + capsule in comments."
      last_update_date: "2026-02-04"
    - id: "GM-capsule"
      text: "Capsule is monotonic: ledgers append-only; shrink forbidden unless justified with pointers."
      last_update_date: "2026-02-04"
    - id: "GM-scope"
      text: "Primary scope: register pressure/spills/local memory/occupancy; evidence via NCU + PTX/TileIR A/B on Blackwell."
      last_update_date: "2026-02-04"

SESSION_MEMORY:
  notes:
    - id: "SM-0-scope-freeze"
      text: "Stage 0 scope frozen: microarch-visible reg/spill/occupancy effects only; exclude Tile IR memory-model correctness."
      last_update_date: "2026-02-04"

GAP_LEDGER:
  items:
    - id: "G1"
      date: "2026-02-04"
      title: "Tile metric availability / n.a. feasibility cliff"
      rank_bucket: "Feasibility cliff (elephant)"
      status: "OPEN"
      linked_open_questions: ["OQ05", "OQ06", "OQ09"]
    - id: "G2"
      date: "2026-02-04"
      title: "Fail-closed per-launch provenance + fallback accounting"
      rank_bucket: "Evaluation credibility cliff (elephant)"
      status: "OPEN"
      linked_open_questions: ["OQ02", "OQ06", "OQ07", "OQ08"]
    - id: "G3"
      date: "2026-02-04"
      title: "TileIR occupancy hint controllability + non-ignored proof"
      rank_bucket: "Feasibility/novelty cliff (elephant)"
      status: "OPEN"
      linked_open_questions: ["OQ03"]
    - id: "G4"
      date: "2026-02-04"
      title: "Spill vs local traffic disambiguation (metric semantics under Tile)"
      rank_bucket: "Mechanism validity"
      status: "OPEN"
      linked_open_questions: ["OQ05", "OQ09"]
    - id: "G5"
      date: "2026-02-04"
      title: "Profiling perturbation/replay stability protocol (esp. Tile)"
      rank_bucket: "Evaluation credibility"
      status: "OPEN"
      linked_open_questions: []
    - id: "G6"
      date: "2026-02-04"
      title: "Register metric interpretation (allocated vs live; stack correlation)"
      rank_bucket: "Explainability"
      status: "OPEN"
      linked_open_questions: ["OQ06"]
    - id: "G7"
      date: "2026-02-04"
      title: "Toolchain pinning + compatibility preflight (GPU/driver/CUDA/NCU)"
      rank_bucket: "Feasibility hygiene (reviewer expectation)"
      status: "OPEN"
      linked_open_questions: ["OQ01", "OQ11"]
    - id: "G8"
      date: "2026-02-04"
      title: "Confound factorization: num_ctas=2 and approx/FTZ as explicit factors"
      rank_bucket: "Threat-to-validity control"
      status: "OPEN"
      linked_open_questions: []
    - id: "G9"
      date: "2026-02-04"
      title: "Artifact capture/override semantics in TileIR fork"
      rank_bucket: "Reproducibility/provenance support"
      status: "OPEN"
      linked_open_questions: ["OQ08"]
    - id: "G10"
      date: "2026-02-04"
      title: "TritonBench subset + per-kernel mapping (multi-kernel ops)"
      rank_bucket: "Ecological validity + mapping"
      status: "OPEN"
      linked_open_questions: ["OQ07"]

VERDICT_LEDGER:
  items:
    - id: "V01"
      date: "2026-02-04"
      verdict: "Scope frozen to measurable register pressure/spilling via Nsight Compute + runtime; correctness/memory-model work is explicitly excluded."
      confidence: "high"
    - id: "V02"
      date: "2026-02-04"
      verdict: "Primary experimental factor is backend selection using ENABLE_TILE (TileIR) vs default PTX, with explicit guard against TileIR→PTX fallback contamination."
      confidence: "medium"
    - id: "V03"
      date: "2026-02-04"
      verdict: "Feasibility elephant: must prove NCU returns numeric spill/reg/occupancy metrics for Tile-executed kernels (metric availability matrix gate)."
      confidence: "high"
    - id: "V04"
      date: "2026-02-04"
      verdict: "Credibility elephant: must implement fail-closed provenance per kernel launch and quantify fallback/coverage; otherwise PTX vs TileIR attribution is not defensible."
      confidence: "high"
    - id: "V05"
      date: "2026-02-04"
      verdict: "Novelty/feasibility elephant: occupancy hint must be user-controllable and demonstrably applied; otherwise H2 and a core knob-story weaken substantially."
      confidence: "medium"

CLAIM_LEDGER:
  items:
    - id: "C01"
      scope_tag: "ACTIVE"
      claim: "PTX .local is per-thread private memory; under ABI it is stack-allocated; accessed via ld.local/st.local."
      status: "VERIFIED"
      evidence: ["NV-PTX"]
    - id: "C02"
      scope_tag: "ACTIVE"
      claim: "NCU separates local load/store counts from spill-caused instruction counts; local traffic is not automatically spills."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C03"
      scope_tag: "ACTIVE"
      claim: "NCU provides launch__registers_per_thread and launch__registers_per_thread_allocated."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C04"
      scope_tag: "ACTIVE"
      claim: "NCU warns launch__registers_per_thread can exceed maximum live registers."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C05"
      scope_tag: "ACTIVE"
      claim: "NCU provides sass__inst_executed_register_spilling and mem_local/shared breakdowns."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C06"
      scope_tag: "ACTIVE"
      claim: "NCU provides derived__local_spilling_requests and derived__local_spilling_requests_pct."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C07"
      scope_tag: "ACTIVE"
      claim: "NCU achieved occupancy metric is sm__warps_active.avg.pct_of_peak_sustained_active (SM>=7.0)."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C08"
      scope_tag: "ACTIVE"
      claim: "NCU exposes theoretical occupancy metric sm__maximum_warps_per_active_cycle_pct."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C09"
      scope_tag: "ACTIVE"
      claim: "NCU metric availability can be enumerated via ncu --query-metrics."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C10"
      scope_tag: "ACTIVE"
      claim: "Triton-to-tile-IR enables CUDA Tile IR backend via ENABLE_TILE=1."
      status: "VERIFIED"
      evidence: ["TILE-REPO"]
    - id: "C11"
      scope_tag: "ACTIVE"
      claim: "Triton-to-tile-IR may fall back to PTX backend when a TileIR compilation bug occurs."
      status: "VERIFIED"
      evidence: ["TILE-REPO"]
    - id: "C12"
      scope_tag: "ACTIVE"
      claim: "Triton-to-tile-IR repo states CUDA 13.1-only features and Blackwell-only support in CUDA 13.1."
      status: "VERIFIED"
      evidence: ["TILE-REPO"]
    - id: "C13"
      scope_tag: "ACTIVE"
      claim: "TileIR backend adds occupancy hint (1..32, default 1)."
      status: "VERIFIED"
      evidence: ["TILE-REPO"]
    - id: "C14"
      scope_tag: "ACTIVE"
      claim: "TileIR backend disables approx and FTZ by default; TILEIR_ENABLE_APPROX and TILEIR_ENABLE_FTZ enable them."
      status: "VERIFIED"
      evidence: ["TILE-REPO"]
    - id: "C15"
      scope_tag: "ACTIVE"
      claim: "TileIR backend in CUDA 13.1 does not expose num_warps yet; some norm kernels may spill and slow down."
      status: "VERIFIED"
      evidence: ["TILE-REPO"]
    - id: "C16"
      scope_tag: "ACTIVE"
      claim: "Repo claims num_ctas=2 is critical for dense dot workloads enabling 2CTA MMA on Blackwell."
      status: "VERIFIED"
      evidence: ["TILE-REPO"]
    - id: "C17"
      scope_tag: "ACTIVE"
      claim: "cuda-tile repo documents cuda-tile-translate and tileiras workflow for bytecode->cubin, plus JIT option."
      status: "VERIFIED"
      evidence: ["CUDA-TILE-REPO"]
    - id: "C18"
      scope_tag: "ACTIVE"
      claim: "Define spills primarily by spill-specific NCU metrics, not local loads/stores alone."
      status: "INFERENCE"
      evidence: ["NV-NCU"]
    - id: "C19"
      scope_tag: "ACTIVE"
      claim: "Must detect backend-used to avoid TileIR→PTX fallback contaminating results."
      status: "INFERENCE"
      evidence: ["TILE-REPO"]
    - id: "C20"
      scope_tag: "ACTIVE"
      claim: "Must align/report math modes (appr/FTZ) across backends as confound control."
      status: "INFERENCE"
      evidence: ["TILE-REPO"]
    - id: "C21"
      scope_tag: "ACTIVE"
      claim: "Pressure sweeps should show spill onset where spill metrics become nonzero."
      status: "INFERENCE"
      evidence: []
    - id: "C22"
      scope_tag: "ACTIVE"
      claim: "Spill onset curves differ across PTX vs TileIR for at least one motif."
      status: "INFERENCE"
      evidence: []
    - id: "C23"
      scope_tag: "ACTIVE"
      claim: "TileIR occupancy hint sweep will move reg/spill metrics and correlate with achieved occupancy/runtime for compute-heavy kernels."
      status: "INFERENCE"
      evidence: ["TILE-REPO", "NV-NCU"]
    - id: "C24"
      scope_tag: "ACTIVE"
      claim: "Tensor-of-pointer vs descriptor/TMA paths can change reg pressure/spills."
      status: "INFERENCE"
      evidence: ["TILE-REPO"]
    - id: "C25"
      scope_tag: "ACTIVE"
      claim: "NCU exposes Live Registers distinct from launch__registers_per_thread to mitigate over-interpretation."
      status: "VERIFIED"
      evidence: ["NV-NCU"]
    - id: "C26"
      scope_tag: "ACTIVE"
      claim: "NCU provides launch__stack_size as a per-launch metric."
      status: "VERIFIED"
      evidence: ["NV-NCU"]

    - id: "C27"
      scope_tag: "ACTIVE"
      claim: "In our Blackwell + NCU stack, a per-launch execution-model provenance label (SIMT vs Tile) is available and exportable for each kernel launch."
      status: "UNVERIFIED"
      evidence: []
    - id: "C28"
      scope_tag: "ACTIVE"
      claim: "In our Blackwell + NCU stack, the non-negotiable spill/reg metrics (C03/C05/C06/C07/C26) return numeric values (not n/a) for Tile-executed kernels."
      status: "UNVERIFIED"
      evidence: []
    - id: "C29"
      scope_tag: "ACTIVE"
      claim: "Triton-to-tile-IR exposes a stable, user-accessible API to set the TileIR occupancy hint per kernel configuration, and changing it measurably affects execution."
      status: "UNVERIFIED"
      evidence: []
    - id: "C30"
      scope_tag: "ACTIVE"
      claim: "Upstream Triton dump/override env vars for compilation artifacts (e.g., kernel dump/override/always-compile) work equivalently in the Triton-to-tile-IR fork and capture TileIR artifacts."
      status: "UNVERIFIED"
      evidence: []
    - id: "C31"
      scope_tag: "ACTIVE"
      claim: "NCU provides shared-spill derived metrics analogous to derived__local_spilling_requests for completeness (exact metric IDs to confirm)."
      status: "UNVERIFIED"
      evidence: []

EVIDENCE_LEDGER:
  items:
    - id: "E01"
      date: "2026-02-04"
      source_id: "NV-NCU"
      url: "https://docs.nvidia.com/nsight-compute/ProfilingGuide/"
      notes: "Verified metric names: launch__registers_per_thread; sass__inst_executed_register_spilling; sass__inst_executed_local_loads/stores; launch__stack_size; derived__local_spilling_requests(_pct)."
    - id: "E02"
      date: "2026-02-04"
      source_id: "NV-PTX"
      url: "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"
      notes: "Verified PTX local state space description (private per-thread; stack under ABI; ld.local/st.local)."
    - id: "E03"
      date: "2026-02-04"
      source_id: "TILE-REPO"
      url: "https://github.com/triton-lang/Triton-to-tile-IR"
      notes: "Verified ENABLE_TILE=1 switch; fallback to PTX; occupancy hint 1..32; approx/FTZ env vars; Blackwell-only support in CUDA 13.1; num_warps not exposed note."
    - id: "E04"
      date: "2026-02-04"
      source_id: "TB"
      url: "https://github.com/meta-pytorch/tritonbench"
      notes: "Verified TritonBench purpose + basic usage."
    - id: "E05"
      date: "2026-02-04"
      source_id: "CUDA-TILE-REPO"
      url: "https://github.com/NVIDIA/cuda-tile"
      notes: "Verified cuda-tile-translate and tileiras workflow reference."
    - id: "E06"
      date: "2026-02-04"
      source_id: "NV-TILE-BLOG"
      url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
      notes: "Verified blog-level context: backend bridges Triton to Tile IR; benchmarking/validation workstreams."
    - id: "E07"
      date: "2026-02-04"
      source_id: "NV-TILEIR-MM"
      url: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html"
      notes: "Verified memory model exists; stored mainly as scope-exclusion reference."

EXPERIMENT_LEDGER:
  items: []

EVAL_PLAN:
  status: "draft (stage1 updated: added provenance + stack + overhead ablation as non-negotiables)"
  methodology:
    - "Bare-metal runs; warmup; repeat; report median + variability."
    - "Pin toolchain versions + env vars; treat backend toggles as experimental factors."
    - "Use ncu --query-metrics to confirm metric availability per version."
    - "Guardrail: detect TileIR->PTX fallback and treat as separate bucket."
    - "NEW: Require a metric availability matrix gate for Tile before large sweeps (G1)."
    - "NEW: Include NCU overhead/replay sensitivity ablation (minimal vs expanded metric set) (G5)."
  metrics:
    - "runtime_us_cuda_events"
    - "launch__registers_per_thread"
    - "launch__registers_per_thread_allocated"
    - "derived__local_spilling_requests"
    - "derived__local_spilling_requests_pct"
    - "sass__inst_executed_register_spilling"
    - "sass__inst_executed_register_spilling_mem_local"
    - "sass__inst_executed_register_spilling_mem_shared"
    - "sass__inst_executed_local_loads"
    - "sass__inst_executed_local_stores"
    - "sm__warps_active.avg.pct_of_peak_sustained_active"
    - "launch__stack_size"
    - "PROVENANCE (UNVERIFIED METRIC ID): per-launch execution-model label (SIMT vs Tile) (see C27 / OQ06)"
  baselines:
    - "PTX backend (ENABLE_TILE=0) vs TileIR backend (ENABLE_TILE=1) on same Blackwell GPU"
    - "TileIR occupancy hint sweep (occupancy=1..32) (blocked on OQ03)"
    - "Triton num_ctas sweep (1 vs 2) where relevant"
    - "NEW: NCU minimal-metrics vs expanded-metrics overhead/stability baseline (G5)"
  workloads:
    - "TritonBench subset (select ops with reduction/norm + GEMM motifs) (selection rule TBD; see G10)"
    - "Custom microbench kernels sweeping register pressure"
  ablations:
    - "TILEIR_ENABLE_APPROX=0/1 (controlled)"
    - "TILEIR_ENABLE_FTZ=0/1 (controlled)"
  risks_to_validity:
    - "backend feature drift across CUDA/Triton versions"
    - "register count metric overstates live regs; interpret carefully"
    - "silent TileIR->PTX fallback contaminates A/B comparisons"
    - "clock/frequency variability"
    - "NCU replay/overhead perturbs runtime and possibly spilling behavior"

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: "WP1_20260204"
  stage1_5_toolbox: null
  stage2_directions: null
  stage2_5_audit: null
  stage3_assembly_pack: null
  stage3_paper: null
  microbench_repo: null
  measurement_scripts: null
  ncu_reports_dir: null
  plots_dir: null

OPEN_QUESTIONS:
  active:
    - id: "OQ01"
      question: "What are the exact compute capabilities / sm_ targets for RTX 5090, B200, GB10, and which tileiras --gpu-name values should be used?"
      impacts: ["C12", "C17", "env_template", "G7"]
      query_plan_ref: "Q1"
    - id: "OQ02"
      question: "How do we robustly detect per-kernel whether TileIR backend was used vs fell back to PTX (no silent contamination)?"
      impacts: ["C11", "C19", "G2"]
      query_plan_ref: "Q2"
    - id: "OQ03"
      question: "What is the exact Triton API to set/sweep TileIR occupancy hint (occupancy=1..32) per kernel config?"
      impacts: ["C13", "C29", "H2", "G3"]
      query_plan_ref: "Q3"
    - id: "OQ04"
      question: "What are PTX backend defaults for approx/FTZ, and how do we align/report math modes across backends?"
      impacts: ["C20", "G8"]
      query_plan_ref: "Q4"
    - id: "OQ05"
      question: "On our Blackwell machines/NCU version, are all minimum spill/occupancy metrics available and stable via ncu --query-metrics?"
      impacts: ["C05", "C06", "C07", "C09", "C28", "G1", "G4"]
      query_plan_ref: "Q5"

    - id: "OQ06"
      question: "Does our Nsight Compute version expose and export a per-launch execution-model provenance label (SIMT vs Tile), and what is the exact metric name/field and export path?"
      impacts: ["C27", "G2", "G6", "EVAL_PLAN"]
      query_plan_ref: "Q6"
    - id: "OQ07"
      question: "How do we export per-kernel-launch (per-instance) metrics in NCU range mode for multi-kernel TritonBench ops (avoid aggregate-only ambiguity)?"
      impacts: ["G2", "G10", "C27"]
      query_plan_ref: "Q7"
    - id: "OQ08"
      question: "Do Triton compilation dump/override hooks (kernel dump/override/always-compile) work in the Triton-to-tile-IR fork, and what artifacts are produced for Tile kernels?"
      impacts: ["C30", "G2", "G9"]
      query_plan_ref: "Q8"
    - id: "OQ09"
      question: "Does NCU provide shared-spill derived metrics (and/or other spill-attribution metrics) applicable to Tile workloads, and what are the exact metric IDs?"
      impacts: ["C31", "G1", "G4"]
      query_plan_ref: "Q9"
    - id: "OQ11"
      question: "What are the minimum driver/CUDA/NCU constraints for CUDA Tile profiling on our systems, and do all Blackwell machines meet them (preflight)?"
      impacts: ["G7", "toolchain_to_freeze"]
      query_plan_ref: "Q11"
  closed: []

DELTA_LOG:
  - "Set profile.current_stage=0; last_updated=2026-02-04"
  - "Added hypotheses H1-H4"
  - "Seeded CLAIM_LEDGER with C01-C26"
  - "Verified and recorded evidence E01-E07; updated GOLDEN_SOURCES last_verified for NV-NCU/NV-PTX/NV-TILEIR-MM/NV-TILE-BLOG/TILE-REPO/CUDA-TILE-REPO/TB"
  - "Added OPEN_QUESTIONS OQ01-OQ05"
  - "Set ARTIFACT_INDEX.stage0_fact_sheet=WP0_20260204"
  - "Stage 1: added GAP_LEDGER entries G1-G10 (elephant-first ordering)"
  - "Stage 1: appended VERDICT_LEDGER V03-V05 (top-3 gap verdicts)"
  - "Stage 1: appended CLAIM_LEDGER C27-C31 as UNVERIFIED environment-instantiated prerequisites"
  - "Stage 1: added OPEN_QUESTIONS OQ06-OQ09, OQ11 as verification blockers"
  - "Stage 1: updated EVAL_PLAN to add provenance metric (UNVERIFIED ID), launch__stack_size, and NCU overhead ablation as non-negotiables"
  - "Stage 1: set ARTIFACT_INDEX.stage1_gap_audit=WP1_20260204"
  - "Stage 1: updated profile.current_stage=1"

CAPSULE_HEALTH:
  counts:
    verdicts: 5
    claims: 31
    evidence_items: 7
    experiments: 0
    gaps: 10
    open_questions_active: 10
  capsule_token_estimate: 4400
  notes:
    - "Stage 1 capsule updated; no shrink."
    - "Top elephants: G1 (Tile metric availability), G2 (fail-closed provenance), G3 (occupancy hint controllability)."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 1.5 prompt."
STATE_END
```
