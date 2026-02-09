# WORK_PRODUCT

## 1) Table: Microarch Toolbox (Implementable Artifacts)

> **Artifact ID convention:** A# = new implementable artifact; X# = planned MVP experiment; K# = kernel/motif.

| Gap_ID | Theory_A (measurement/attribution) | Theory_B (control/optimization) | Hybrid | New Artifact (a) | MVP Build Path (2–4 wks) (c) | Stretch Path (8–12 wks) (c) | Metrics / hooks (d) | Main Risk | Mitigation | Closest neighbor + delta type (e) |
|---|---|---|---|---|---|---|---|---|---|---|
| **G1** Tile metric availability / `n/a` cliff | **A1:** *Metric Availability Matrix Gate* — empirical “numeric vs `n/a` vs error” matrix for **Tile-executed** kernels (and SIMT), per GPU/toolchain. | **B1:** *Observable-substitution ladder* — if key spill metrics fail on Tile, fall back to **stack + SASS + runtime** (still mechanism-ish). | **H1:** Gate first; if missing, downgrade claims + pivot to what is measurable (explicitly). | **A1_metric_matrix/**: `metric_matrix.py`, `metrics_min.yaml`, `report.md` (table + raw exports). | **Week 1:** implement minimal kernel K0 + NCU runner; collect **minimal metric list** on RTX 5090 for `ENABLE_TILE∈{0,1}`. **Week 2:** add B200 if available; produce table(s). | Expand across NCU versions + driver/toolkit versions; publish matrix as dataset artifact (“early adopter map”). | Non‑negotiables: `launch__registers_per_thread`, `derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling*`, `sass__inst_executed_local_loads/stores`, `sm__warps_active...`, `launch__stack_size`, runtime. | NCU “supports Tile” but your specific metrics come back `n/a`/nonsense. | Use **minimal metric set**, long steady-state kernel (K5), log `n/a` explicitly, avoid overcollection/replay sensitivity (ties to G5). | Neighbor: NCU docs define metrics; **delta: first empirical “metric feasibility gate” + availability matrix** for Tile workloads (publishable characterization artifact). |
| **G2** Fail‑closed per‑launch provenance + fallback accounting | **A2:** *Fail-closed provenance ledger* — classify each kernel launch as expected backend vs not; report **coverage + fallback rate** as first-class results. | **B2:** *Provenance hardening controls* — per-run cache isolation, force recompilation, one-op-per-process runs, deterministic harness. | **H2:** A2 (label) + B2 (controls) + optional artifact corroboration if available (G9). | **A2_provenance_guard/**: `run_with_guard.py`, `manifest_schema.json`, `coverage_report.md`. | **Week 1:** implement manifest logging + cache isolation + per-run directory layout; add “unknown/ambiguous => exclude” logic. **Week 2–3:** run X02 on K0/K1 + 1 TB op; generate fallback/coverage tables. | Add “provenance certificate” per plotted point (hashes + env + kernel list). Optional: integrate lightweight compile-hook logging (if feasible) to reduce dependence on NCU for provenance outside profiled runs. | Provenance hook (UNVERIFIED): per-launch SIMT vs Tile label; plus kernel name/launch IDs; runtime; metric completeness. | Provenance label export might be missing / aggregated-only / hard to parse. | Make NCU provenance **required** for any point used in core claims until proven; otherwise treat as **exploratory**. Use strict cache/process isolation to reduce mixed-mode. | Neighbor: TILE-REPO documents fallback hazard; NCU (likely) exposes provenance metric; **delta: formal fail-closed protocol + measured fallback/coverage rates** (turn hazard into results). |
| **G3** TileIR occupancy hint controllability + “not ignored” proof | **A3:** *Occupancy-hint control-surface discovery + proof* — locate the knob, sweep 1..32, show measurable change (occupancy metric and/or runtime/spills). | **B3:** *Alternative occupancy controls* — if hint isn’t user-settable, pivot to knobs you can control: `num_ctas`, kernel shape, register-pressure knobs; treat “hint unexposed” as negative finding. | **H3:** A3 early go/no-go; if no control surface, demote H2 and strengthen H1/H3/H4 instead. | **A3_occ_hint_probe/**: repo search notes + minimal reproducer + “proof plot” script. | **Week 1 (must be early):** find the API/knob (or conclude “not user-exposed”); run 2-point test (occ=1 vs occ=32) on one compute-heavy kernel; check deltas. | If knob exists: incorporate into autotune sweep + interaction analysis (with reg pressure). If not: document limitation + propose workaround recommendations (`num_ctas`, etc.). | `sm__warps_active...`, runtime, `launch__registers_per_thread`, spill metrics; (optionally any occupancy-limit metric if verified later). | The hint may exist only internally or be silently ignored. | Treat as **MVP milestone X03**: pass/fail decision; publish negative result as “API gap” evidence if needed. | Neighbor: TILE-REPO README says hint exists; **delta: mapping from README claim → user-level control + measured effect proof** (or documented non-exposure). |
| **G4** Spill vs local-traffic disambiguation (metric semantics under Tile) | **A4:** *Metric validation microbench triad* — design cases where (i) spills dominate, (ii) local traffic not due to spilling (SIMT calibration), (iii) both; validate classifier rules. | **B4:** *Spill-isolation kernel design rules* — keep memory traffic constant while increasing live values; prevent DCE; include stack size. | **H4:** Adopt a **spill classifier** that requires agreement between derived spill % and SASS spill insts (plus stack). | **A4_spill_classifier/**: `classifier_rules.md`, calibration kernels (Triton + optional CUDA SIMT), analysis script. | **Week 2:** run on SIMT baseline to sanity-check NCU metrics; **Week 3–4:** attempt same logic on Tile-executed kernels; produce “agreement/discordance matrix.” | Extend to more motifs + TB kernels; publish “metric interpretation under Tile” guidance + exclusions. | `derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling*`, `sass__inst_executed_local_loads/stores`, `launch__stack_size`, runtime. | Hard to create “local-not-spill” cases purely in Triton; Tile semantics might differ. | Use CUDA SIMT kernel(s) strictly for calibration; for Tile, focus on spill/no-spill contrasts + discordance rules + explicit exclusions. | Neighbor: NCU defines metrics; **delta: empirical validation + discordance handling rules**, especially under new execution model. |
| **G5** Profiling perturbation / replay stability protocol | **A5:** *Overhead + stability ablation* — quantify NCU overhead and metric variance: runtime-only vs NCU-minimal vs NCU-expanded. | **B5:** *Two-phase measurement protocol* — separate “mechanism runs” (NCU) from “runtime runs” (no NCU); join via provenance + kernel hash. | **H5:** Always report overhead; collect mechanism metrics with minimal packs; runtime separately. | **A5_ncu_overhead/**: scripts + report template (tables/plots). | **Week 2:** implement long steady kernel K5; run X05 across 2–3 kernels for PTX vs TileIR; report slowdown + variance. | Develop recommended “metric packs” that minimize passes; add automated detection of unstable metrics (flag). | runtime_us_cuda_events; repeatability stats; key mechanism metrics above. | NCU changes schedule/spills; reviewers distrust any runtime under profiler. | Treat profiled runtime as **not primary**; use runtime-only for performance claims; use NCU for attribution with overhead disclosed. | Neighbor: NCU guide warns replay issues; **delta: concrete, Tile-aware overhead/stability protocol + disclosure**. |
| **G6** Register metric interpretation (allocated vs live; stack correlation) | **A6:** *Triangulation plots* — allocated regs vs (NCU “live regs” metric) vs stack size vs spill metrics; identify “holes/ABI” cases. | **B6:** *Design reg-ladder to increase true liveness* — strengthen inference by controlling actual liveness. | **H6:** Use “reg pressure ladder” + spill classifier to interpret. | **A6_reg_interp/**: analysis notebook/script + plot templates. | **Week 3:** implement K1 reg ladder; collect regs/thread + stack + spill; add live-reg metric once exact ID is verified (OQ12). | Add SASS/PTX inspection step (nvdisasm) to annotate causes of reg inflation. | `launch__registers_per_thread`, `launch__registers_per_thread_allocated`, `launch__stack_size`, spill metrics; **Live-reg metric ID TBD**. | Live-reg metric ID/export path unknown; interpretation can overreach. | Add OQ12; until resolved, stick to allocated regs + stack + spill insts as triangulation. | Neighbor: NCU warns allocated regs ≠ liveness; **delta: paper-grade interpretation guide + empirical plots on PTX vs TileIR**. |
| **G7** Toolchain pinning + compatibility preflight | **A7:** *Preflight + manifest* — record and gate on GPU/driver/CUDA/NCU/Triton versions; compute capability; env vars; commit hashes. | **B7:** *Repro harness* — conda env + pinned repos + run scripts; make re-run friction low. | **H7:** Preflight is mandatory; repro harness is stretch. | **A7_env_preflight/**: `preflight.py`, `manifest.json`, run directory convention. | **Week 1:** implement manifest logging for every run; include `nvidia-smi`, `nvcc --version`, `ncu --version`, python deps, GPU props. | Add container recipe + automated “known-good stack” checks once OQ11 resolved. | N/A (meta), but enables all metrics credibility. | Version drift silently invalidates comparisons. | Hard-fail gates: “unknown stack => exploratory only.” | Neighbor: generic reproducibility checklists; **delta: Tile-specific coupling + automated preflight artifacts**. |
| **G8** Confound factorization (`num_ctas`, approx/FTZ, etc.) | **A8:** *Factorial design + interaction reporting* — make confounds explicit factors, not hidden defaults. | **B8:** *Confound fixing policy* — main plots fix confounds; only ablation section varies them. | **H8:** Minimal factorial for key kernels; fix elsewhere. | **A8_factor_matrix/**: YAML run matrix + analysis script (effects). | **Week 2–3:** run 2×2 backend × `num_ctas` for 1 kernel; approx/FTZ fixed; report interaction if present. | Expand to approx/FTZ + occupancy hint (if G3 succeeds); DOEs to limit runs. | runtime + regs/spills/occupancy; plus coverage. | Combinatorial explosion; cherry-picking accusations. | Pre-register run matrix + deterministic selection; partial factorial. | Neighbor: benchmarking hygiene; **delta: explicit, published confound ledger + interaction quantification** for this toolchain transition. |
| **G9** Artifact capture/override semantics in TileIR fork | **A9:** *Artifact probe* — verify what dump/override hooks produce in the fork for PTX vs TileIR; index artifacts per kernel hash. | **B9:** *Fallback artifact path* — if hooks don’t work, rely on whatever does (cache dirs) + NCU provenance; patch fork only as stretch. | **H9:** “2-of-3 provenance” (NCU label + manifest + artifact) for high-value plots if feasible. | **A9_artifact_probe/**: env-var sweep script + file tree snapshot + hashes. | **Week 2:** run one kernel with dump/override env vars; document outputs; decide what’s reliable. | Build artifact indexer (kernel → PTX/SASS/tile bytecode/cubin) and ship with dataset. | N/A (meta), but supports provenance + reproducibility. | Fork may not honor upstream hooks; artifacts may be incomplete. | Treat as optional corroboration; do not base core claims solely on artifacts. | Neighbor: upstream Triton dump/override docs; **delta: validated behavior + artifact schema for TileIR fork**. |
| **G10** TritonBench subset + per-kernel mapping (multi-kernel ops) | **A10:** *TB wrapper + mapping* — deterministic op subset, isolate one op per process, profile all launches, map kernels → op, compute provenance coverage + mechanism summaries. | **B10:** *TB as external validity only* — keep mechanism claims on microbenches; TB used for “does it matter” case studies. | **H10:** Microbench for mechanisms + TB for case studies with strict provenance coverage reporting. | **A10_tb_wrapper/**: selection rule doc + wrapper + kernel-name map + aggregation logic. | **Week 3–4:** pick 3 ops; run with NCU minimal pack; output: per-op kernel list + % Tile/SIMT + key metrics per kernel. | Expand op set, automate kernel clustering, publish mapping + raw NCU exports. | Same mechanism metrics + provenance + coverage. | NCU per-instance export / kernel naming may be hard; multi-kernel noise. | One-op-per-process; deterministic selection; if per-instance export missing, treat TB results as exploratory. | Neighbor: TritonBench exists; **delta: provenance-aware profiling wrapper + per-kernel attribution + coverage accounting**. |

### P0) Standard data-collection + parsing pipeline (used by A1/A2/A4/A5/A10)

**P0 steps (algorithm/pipeline requirement b):**
1. **P0.1 Preflight** (A7): collect toolchain + GPU properties into `manifest.json`.
2. **P0.2 Build run directory**: per-run unique output dir; record env vars (`ENABLE_TILE`, approx/FTZ toggles, etc.).
3. **P0.3 Execute** kernel/op with warmup + repeats; measure **runtime_us_cuda_events**.
4. **P0.4 Profile (NCU)** with **minimal metric pack**; export machine-readable results (format TBD; verify via `ncu --help`).
5. **P0.5 Parse** exported results into a canonical row format:
   - keys: run_id, kernel_name, launch_idx, backend_expected, backend_observed (if available), metrics…
6. **P0.6 Fail-closed filtering** (A2): any row with ambiguous provenance or missing non-negotiable metric → excluded from main plots; counted in coverage stats.
7. **P0.7 Report**: tables + plots + “coverage/fallback” summaries + overhead ablation.

---

## 2) Toolbox Verdicts (≤10 bullets)

- **V-FAV(G1): A1 is mandatory** — implement **Metric Availability Matrix Gate** before investing in sweeps. This turns the biggest feasibility cliff into a week-1 measurable table (novelty = “Tile metric feasibility characterization artifact”).
- **V-FAV(G2): A2 + strict exclusion accounting is mandatory** — “fail-closed provenance” plus **fallback/coverage rate** should be a *headline result*, not hygiene. Without it, PTX vs TileIR attribution is not reviewable.
- **V-FAV(G3): A3 is a week-1 go/no-go** — if occupancy hint is not user-controllable or is ignored, **demote H2** early and re-center novelty on H1/H3/H4 plus provenance + metric validation.
- **V-FAV(G4): A4 spill-classifier calibration is required for mechanism claims** — reviewers will reject “local loads/stores == spills”; you need validated triangulation rules and discordance handling (especially for Tile).
- **V-FAV(G5): A5 overhead/stability ablation is required disclosure** — makes runtime claims defensible and prevents “profiler changed behavior” critiques from collapsing the paper.
- **V-FAV(G6): A6 triangulation improves explainability** — allocate vs stack vs spill insts yields a credible mechanism story even if “live regs” metric ID lags.
- **V-FAV(G7): A7 manifest logging becomes a hard gate** — prevents silent toolchain drift from invalidating comparisons; also enables artifact-level reproducibility.
- **V-FAV(G8): Use A8 minimal factorial (backend × `num_ctas`) early** — specifically for dot/GEMM-like motifs where README flags `num_ctas=2` as critical; report interactions explicitly.
- **V-FAV(G10): Use A10 as case-study harness, not as mechanism discovery** — do mechanisms on microbenches; TB establishes external validity with coverage accounting.
- **V-FAV(G9): Treat artifacts as corroboration only until verified** — integrate A9 once you confirm fork semantics; don’t let missing dumps block MVP.

---

## 3) Citation-needed / Query Plan (if needed)

> Goal: resolve UNVERIFIED items that block A2/A3/A6/A10 and tighten citations for camera-ready.

### QP-1 (maps to OQ06/OQ07): NCU provenance + per-launch export
- **What to verify:** exact field/metric name for “SIMT vs Tile” per launch; how to export **per-instance** values in range profiling; stable kernel identifiers in export.
- **Plan:**  
  1) Inspect local NCU docs installed with your version; search for “execution_model” and “per instance” / “range” export behavior.  
  2) Run `ncu --help` and capture the help text as an artifact (store under A7 manifest).  
  3) Empirical probe: profile a program that launches **two kernels** back-to-back; confirm export includes both launches separately (launch_idx increments).

### QP-2 (maps to OQ03): TileIR occupancy hint control surface
- **What to verify:** exact API/knob to set occupancy hint 1..32 in Triton-to-tile-IR from user code; whether it’s per-kernel config and not ignored.
- **Plan:**  
  1) Repo code search (`grep -R "occupancy"`, `"hint"`, `"tile"`).  
  2) Identify user-facing Python decorator/config entrypoint if any.  
  3) Empirical probe: set two extreme values and check for any systematic change in occupancy metric / runtime / spills under identical workload.

### QP-3 (maps to OQ09): shared spilling derived metrics IDs
- **What to verify:** exact metric IDs for shared spilling request attribution (if any) in your NCU version; applicability to Tile workloads.
- **Plan:**  
  1) `ncu --query-metrics | grep spilling` (artifact the full list).  
  2) Attempt collection in A1 matrix; record numeric vs `n/a`.

### QP-4 (maps to OQ11): minimum driver/CUDA/NCU constraints for CUDA Tile profiling
- **What to verify:** hard version constraints (driver/toolkit/NCU) and whether all your Blackwell machines meet them.
- **Plan:**  
  1) Add checks into A7 preflight: driver version, CUDA toolkit, NCU version.  
  2) Cross-check against NVIDIA release notes/docs (web or local) and then encode as **assertions** in preflight once confirmed.

### QP-5 (maps to OQ12): “Live Registers” exact metric ID + export
- **What to verify:** exact NCU metric ID for “Live Registers” and whether it’s available for Tile workloads.
- **Plan:**  
  1) Search NCU metric reference (local doc) for “Live Registers”.  
  2) `ncu --query-metrics | grep -i live` and attempt collection in A1.

---

# CONTEXT_CAPSULE

```yaml
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

  current_stage: 1.5
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
      linked_open_questions: ["OQ06", "OQ12"]
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
      verdict: "Credibility elephan must implement fail-closed provenance per kernel launch and quantify fallback/coverage; otherwise PTX vs TileIR attribution is not defensible."
      confidence: "high"
    - id: "V05"
      date: "2026-02-04"
      verdict: "Novelty/feasibility elephant: occupancy hint must be user-controllable and demonstrably applied; otherwise H2 and a core knob-story weaken substantially."
      confidence: "medium"
    - id: "V06"
      date: "2026-02-04"
      verdict: "Stage 1.5 choice: Adopt A1 (Metric Availability Matrix Gate) as the Week-1 MVP gate; no large sweeps until it passes for at least one Tile-executed kernel."
      confidence: "high"
    - id: "V07"
      date: "2026-02-04"
      verdict: "Stage 1.5 choice: Adopt A2 (Fail-closed provenance ledger + fallback/coverage reporting) as a publication constraint; ambiguous provenance points are excluded and counted."
      confidence: "high"
    - id: "V08"
      date: "2026-02-04"
      verdict: "Stage 1.5 choice: Treat occupancy-hint controllability (A3) as an early go/no-go; if not controllable or ignored, demote H2 and strengthen H1/H3/H4."
      confidence: "medium"
    - id: "V09"
      date: "2026-02-04"
      verdict: "Stage 1.5 choice: Adopt A4 (spill-vs-local disambiguation via classifier + discordance rules) as required for mechanism claims, especially under Tile."
      confidence: "medium-high"
    - id: "V10"
      date: "2026-02-04"
      verdict: "Stage 1.5 choice: Adopt A5 two-phase measurement + overhead ablation: runtime claims from runtime-only runs; mechanism claims from NCU minimal metric packs with overhead disclosed."
      confidence: "high"
    - id: "V11"
      date: "2026-02-04"
      verdict: "Stage 1.5 choice: Make A7 (toolchain preflight + manifest logging) mandatory for every run directory (pins, versions, env vars, GPU props)."
      confidence: "high"

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
      status: "INFERENCE"     evidence: ["TILE-REPO"]
    - id: "C20"
      scope_tag: "ACTIVE"
      claim: "Must align/report math modes (approx/FTZ) across backends as confound control."
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
  items:
    - id: "X01"
      date: "2026-02-04"
      status: "PLANNED (MVP gate)"
      title: "Metric Availability Matrix Gate (A1)"
      goal: "Show at least one Tile-executed kernel with numeric (non-n/a) values for core spill/reg/occupancy metrics."
      depends_on: ["OQ05", "OQ06", "OQ11"]
      outputs: ["table: metric -> numeric/n-a/error", "raw NCU export", "manifest.json"]
    - id: "X02"
      date: "2026-02-04"
      status: "PLANNED (MVP gate)"
      title: "Fail-Closed Provenance + Coverage/Fallback Ledger (A2)"
      goal: "Per-kernel-launch provenance classification; compute coverage + fallback rate under ENABLE_TILE=1."
      depends_on: ["OQ02", "OQ06", "OQ07", "OQ08"]
      outputs: ["coverage table", "excluded-point accounting", "per-run manifest"]
    - id: "X03"
      date: "2026-02-04"
      status: "PLANNED (go/no-go)"
      title: "Occupancy Hint Control-Surface Probe (A3)"
      goal: "Demonstrate occupancy hint is user-settable and not ignored (or record negative result)."
      depends_on: ["OQ03"]
      outputs: ["two-point test plot/table", "API snippet or 'not exposed' note"]
    - id: "X04"
      date: "2026-02-04"
      status: "PLANNED"
      title: "Spill-vs-Local Disambiguation Calibration (A4)"
      goal: "Validate spill classifier rules and handle discordance; attempt under Tile."
      depends_on: ["OQ05", "OQ09"]
      outputs: ["classifier_rules.md", "agreement/discordance matrix"]
    - id: "X05"
      date: "2026-02-04"
      status: "PLANNED"
      title: "NCU Overhead + Stability Ablation (A5)"
      goal: "Quantify profiler overhead and metric variance for PTX vs TileIR across minimal vs expanded metric packs."
      depends_on: []
      outputs: ["overhead table", "variance summary", "recommended metric pack"]
    - id: "X06"
      date: "2026-02-04"
      status: "PLANNED"
      title: "Register Pressure Ladder (K1) PTX vs TileIR (A6)"
      goal: "Produce spill-onset curves and reg/stack/spill triangulation under A/B."
      depends_on: ["OQ06", "OQ12"]
      outputs: ["spill onset plot(s)", "reg vs stack correlation plot(s)"]
    - id: "X07"
      date: "2026-02-04"
      status: "PLANNED"
      title: "TritonBench Wrapper Case Study (A10)"
      goal: "Per-op kernel mapping + provenance coverage + mechanism summary for a small deterministic TB subset."
      depends_on: ["OQ07"]
      outputs: ["per-op kernel list", "coverage table", "case-study plot(s)"]

EVAL_PLAN:
  status: "draft (stage1.5 updated: added gates + microbench motifs + fail-closed inclusion rules)"
  gates:
    - id: "GATE-1"
      name: "Metric Availability Matrix"
      pass_condition: "At least one Tile-executed kernel returns numeric values for the core spill/reg/occupancy metrics; document any n/a."
      artifact: "A1_metric_matrix"
    - id: "GATE-2"
      name: "Fail-Closed Provenance Coverage"
      pass_condition: "Any datapoint in main plots has unambiguous per-launch provenance (or is excluded and counted); report coverage + fallback."
      artifact: "A2_provenance_guard"
    - id: "GATE-3"
      name: "Occupancy Hint Control Surface"
      pass_condition: "Either (a) show knob is user-settable + not ignored, or (b) demote H2 with an explicit negative finding."
      artifact: "A3_occ_hint_probe"
  methodology:
    - "Bare-metal runs; warmup; repeat; report median + variability."
    - "Pin toolchain versions + env vars; treat backend toggles as experimental factors."
    - "Use ncu --query-metrics to confirm metric availability per version."
    - "Guardrail: detect TileIR->PTX fallback and treat as separate bucket."
    - "Require a metric availability matrix gate for Tile before large sweeps (G1)."
    - "Include NCU overhead/replay sensitivity ablation (minimal vs expanded metric set) (G5)."
    - "Two-phase measurement: runtime-only runs for performance claims; NCU runs for mechanism attribution; join via manifest + kernel identity."
    - "Correctness gate (control, not primary contribution): outputs must match between backends within tolerance or the kernel is excluded."
  microbench_motifs:
    - id: "K0"
      name: "SmokeTile"
      purpose: "Minimal kernel to establish Tile execution + NCU collection path."
    - id: "K1"
      name: "RegPressureLadder"
      purpose: "Monotonic register-pressure knob sweep; detect spill onset knee."
    - id: "K5"
      name: "SteadyStateLongKernel"
      purpose: "Long duration to reduce replay variability; used for overhead/stability ablations."
    - id: "K6"
      name: "ConfoundFactorKernel"
      purpose: "Kernel where num_ctas=2 matters (dense dot motif) for interaction tests."
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
    - "PROVENANCE (UNVERIFIED ID): per-launch execution-model label (SIMT vs Tile) (see C27 / OQ06)"
    - "OPTIONAL (UNVERIFIED IDs): shared-spill derived metrics (see C31 / OQ09)"
    - "OPTIONAL (UNVERIFIED ID): Live Registers metric ID (see C25 / OQ12)"
  baselines:
    - "PTX backend (ENABLE_TILE=0) vs TileIR backend (ENABLE_TILE=1) on same Blackwell GPU"
    - "TileIR occupancy hint sweep (occupancy=1..32) (blocked on OQ03)"
    - "Triton num_ctas sweep (1 vs 2) where relevant"
    - "NCU minimal-metrics vs expanded-metrics overhead/stability baseline (G5)"
  workloads:
    - "Custom microbench kernels sweeping register pressure (K0/K1/K5/K6)"
    - "TritonBench subset (selection rule TBD; see G10) — used for case studies, not primary mechanism discovery"
  ablations:
    - "TILEIR_ENABLE_APPROX=0/1 (controlled)"
    - "TILEIR_ENABLE_FTZ=0/1 (controlled)"
    - "backend x num_ctas interaction where applicable"
  inclusion_rules:
    - "Any datapoint in a main plot must hav (1) pass correctness gate; (2) unambiguous provenance (GATE-2); (3) non-n/a non-negotiable metrics (or is excluded and counted)."
  risks_to_validity:
    - "backend feature drift across CUDA/Triton versions"
    - "register count metric overstates live regs; interpret carefully"
    - "silent TileIR->PTX fallback contaminates A/B comparisons"
    - "clock/frequency variability"
    - "NCU replay/overhead perturbs runtime and possibly spilling behavior"

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: "WP1_20260204"
  stage1_5_toolbox: "WP1_5_20260204"
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
    - id: "OQ12"
      question: "What is the exact NCU metric ID for 'Live Registers' (C25), and is it available/exportable per-launch for Tile workloads?"
      impacts: ["G6", "EVAL_PLAN", "X06"]
      query_plan_ref: "QP-5"
  closed: []

DELTA_LOG:
  - "Stage 1.5: proposed Microarch Toolbox artifacts A1-A10 mapped to top gaps G1-G10 with MVP and stretch paths."
  - "Stage 1.5: added VERDICT_LEDGER V06-V11 selecting favored toolbox choices (A1/A2/A3/A4/A5/A7 as mandatory/early)."
  - "Stage 1.5: updated EVAL_PLAN with explicit gates (GATE-1..3), microbench motifs (K0/K1/K5/K6), inclusion rules (fail-closed), and two-phase measurement protocol."
  - "Stage 1.5: added EXPERIMENT_LEDGER planned MVP experiments X01-X07."
  - "Stage 1.5: set ARTIFACT_INDEX.stage1_5_toolbox=WP1_5_20260204."
  - "Stage 1.5: added OPEN_QUESTIONS OQ12 (Live Registers metric ID/export for Tile)."
  - "Stage 1.5: updated GAP_LEDGER G6 linked_open_questions to include OQ12."
  - "Stage 1.5: updated profile.current_stage=1.5."

CAPSULE_HEALTH:
  counts:
    verdicts: 11
    claims: 31
    evidence_items: 7
    experiments: 7
    gaps: 10
    open_questions_active: 11
  capsule_token_estimate: 5400
  notes:
    - "Stage 1.5 capsule updated; no shrink."
    - "MVP gates explicitly defined: metric availability (X01), provenance coverage (X02), occupancy hint go/no-go (X03)."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 2 prompt."
STATE_END
```
