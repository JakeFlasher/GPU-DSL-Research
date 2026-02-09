## WORK_PRODUCT

**Newly-verified leverage that directly collapses Gap G1 risk:**

- **NCU exposes `launch__execution_model`** as a launch metric, explicitly distinguishing **SIMT vs Tile** execution model per kernel launch. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai))  
- NVIDIA’s TileIR backend blog describes a **cache-level provenance signal**: when the Tile IR backend is active, Triton caches compiled kernels with **`.tileIR`** file extensions (vs `.cubin` for SIMT backend). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- NCU 2025.4 **added support for profiling CUDA tile workloads** and introduced a **Tile section**. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html?utm_source=openai))  

### 1) Table: Microarch Toolbox (Implementable Artifacts)

> Legend: “New Artifact” uses **K###** IDs (tooling kits). MVP = 2–4 weeks. Stretch = 8–12 weeks.

| Gap_ID | Theory_A (measurement / attribution) | Theory_B (control / optimization) | Hybrid | New Artifact | MVP Build Path | Stretch Path | Metrics | Main Risk | Mitigation | Closest neighbor + delta type |
|---|---|---|---|---|---|---|---|---|---|---|
| **G1** Backend provenance + TileIR→PTX fallback detection | **A1:** Treat provenance as **measured**, not assumed: collect NCU launch metrics incl. `launch__execution_model` (SIMT vs Tile) and store alongside every datapoint. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | **B1:** **Fail-closed harness**: if provenance missing/ambiguous, discard run; optionally enforce “always compile” and cache hygiene (toolchain-dependent). | **H1:** NCU execution-model label **+** cache artifact fingerprint (`.tileIR` vs `.cubin`) for quick pre-check. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | **K001 ProvenanceGuard** | (1) Run kernel twice with `ENABLE_TILE` toggled; (2) profile minimal metric set; (3) parse report → `backend_used={SIMT,Tile}`; (4) emit CSV row + JSON manifest. | Add binary hashing + per-kernel cache artifact capture; integrate into sweep runner so every row is provenance-verified. | `launch__execution_model` ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) + runtime + `launch__registers_per_thread` + spill metrics + SASS corroboration | NCU overhead / inability to profile in some replay modes | Use minimal sections; separate “runtime-only” runs from “NCU provenance runs”; require periodic provenance audits for runtime-only sweeps | **Neighbor:** manual env-var toggles + ad-hoc inspection. **Delta:** automated, per-row *measured* provenance + fail-closed policy (reviewer-proof). |
| **G2** NCU spill/reg metric visibility on CUDA tile workloads | **A2:** Build a **capability matrix**: for each backend/device, attempt to collect required metrics; record success/`n/a`/error. | **B2:** Define **tiered metric sets** (Tier0: launch-only; Tier1: spill+SASS; Tier2: full). Auto-downgrade if metrics unavailable. | **H2:** Capability matrix drives auto-selection + emits “feasibility badge” for each experiment. | **K002 NCUCapabilityProbe** | Script runs a tiny SIMT kernel and a tiny Tile kernel; collects the core metric list and logs whether each metric is returned vs `n/a`. | Expand to multiple NCU versions + replay modes; add regression tests for “metric drift” across upgrades. | Core: `launch__registers_per_thread`, `derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling`, local load/store SASS counts (plus `launch__execution_model`). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Confounding failures: permissions (NVGPUCTRPERM), replay incompatibility | Record error codes + replay mode + driver; provide “known-good” profiling recipe per platform | **Neighbor:** NCU docs list metrics, but don’t guarantee collectability for your workload/device. **Delta:** empirical availability map + automated guards. |
| **G3** Occupancy hint controllability + sweep logging | **A3:** Black-box validate hint: sweep N=1..32; observe monotone/non-monotone shifts in spills/regs/runtime; require provenance fields to prove hint was applied (compile metadata or observable effect). (Hint existence in README is known; *API remains OQ01*.) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | **B3:** Optimization: once controllable, fit simple response model (runtime vs spills vs hint) and pick hint under constraints. | **H3:** Use capability probe + provenance + sweep tool → “Occupancy Hint Profile Series” (your own, not NCU’s). | **K003 OccupancyHintSweeper** | As soon as API is found: run 1 kernel across hints 1..32 with fixed sizes; output plots (hint→spills, hint→runtime). | Generalize across kernels (microbench + TritonBench subset); produce heuristic: choose hint based on reg pressure regime. | Runtime + spill metrics + regs; optionally add occupancy/utilization metrics **after** verifying names via `--query-metrics`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | API not exposed or hint ignored → novelty collapses | Make MVP explicitly include “API proof”: minimal repro showing hint settable + changes `launch__execution_model` remains Tile and spills shift | **Neighbor:** Triton autotune over `num_warps/num_stages`. **Delta:** first systematic occupancy-hint tradeoff curves for TileIR with spill attribution. |
| **G4** Numeric-mode confounding (approx/FTZ) | **A4:** Treat numeric mode as a *measured factor*: attach env-var state + output error stats for every run; add micro-test for FTZ/subnormals. (TileIR approx/FTZ env vars are documented in README.) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | **B4:** Control: mainline experiments run in **one fixed numeric policy**; approx/FTZ only in a dedicated ablation block. | **H4:** Provenance row schema includes numeric-mode fields; correctness gate is required for any performance claim. | **K004 NumericModeGuard** | Add per-run correctness check (bitwise for integer microbenches; tolerance for FP kernels); log env vars + error norms. | Add targeted FTZ detection microbench (subnormal inputs) to show when FTZ materially changes outputs/perf. | Error metrics + runtime + provenance fields; keep spill metrics constant across numeric ablations. | False attributions (perf changes due to precision, not backend) | Separate factorial design: backend × numeric mode; do not mix modes inside same sweep | **Neighbor:** ad-hoc “it seems correct”. **Delta:** explicit numeric confound policy + quantified correctness gates. |
| **G5** Spill vs non-spill local memory attribution protocol | **A5:** Multi-signal classifier: treat “spills” as present only when derived spill metrics + spilling SASS instructions align; treat remaining local traffic as “local arrays/other”. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | **B5:** Control via calibration kernels: (i) forced spill ladder, (ii) local-array-without-spill ladder; build thresholds. | **H5:** Calibration → classifier → apply to TritonBench kernels; output “local traffic decomposition” table. | **K005 LocalAttributionSuite** | Implement 2–3 microbenches that sweep register pressure and local array size; collect core metrics; fit simple decision rules. | Add auto-report that tags each kernel-run with {spill-dominant, local-array-dominant, mixed} + confidence. | `derived__local_spilling_requests(_pct)` + `sass__inst_executed_register_spilling` + local load/store counts. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Local arrays can still induce spills → ambiguous | Use paired designs and hold everything but the intended factor constant; require confidence scoring, not binary labels | **Neighbor:** NCU shows local + spill metrics but interpretation is ad hoc. **Delta:** calibrated attribution protocol (reviewer doesn’t argue). |
| **G6** `launch__registers_per_thread` interpretation (holes/ABI constraints) | **A6:** Never interpret reg allocation alone: require corroboration by spills or occupancy/throughput shifts; report “reg pressure index” = f(regs, spills, runtime). | **B6:** Control: cross-check PTX path with `ptxas`-reported regs/spills (where available) and calibrate correlation to NCU metrics; for Tile path rely on SASS + derived metrics. | **H6:** Analyzer emits “reg claim confidence” + warns on holes/ABI risks. | **K006 RegPressureAnalyzer** | Microbench sweeps (unroll, tile size) → record regs + spills + runtime; show cases where regs change without spills → document interpretation rule. | Extend to per-instruction attribution (TTIR/TTGIR dumps) if available; publish a checklist for MICRO skepticism. | `launch__registers_per_thread` + spill metrics + runtime; avoid claims without corroboration. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Reviewer attacks “regs metric is misleading” | Bake the caveat into analysis + require corroboration; show counterexamples explicitly | **Neighbor:** single-metric reg reporting. **Delta:** conservative multi-signal methodology + confidence reporting. |
| **G7** Generalizable descriptor/TMA rewrite evaluation | **A7:** Paired-kernel study: pointer-tensor vs descriptor/TMA style; measure reg/spill deltas and runtime deltas under both backends. NVIDIA blog provides a concrete rewrite template. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | **B7:** Optimization: create a “rewrite cookbook” and encode as style guidelines or lints; identify when rewrite helps/hurts. | **H7:** Microbench + TritonBench case studies to show transferability. | **K007 DescriptorRewriteBench** | Implement 1 canonical kernel pair (GEMM-like) from blog pattern; run PTX vs TileIR; capture NCU spill/reg metrics. | Expand to 3–5 kernels and quantify “help regions” (stride patterns, tile sizes). | Runtime + reg/spill metrics + memory metrics; correctness checks. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Hard to ensure equivalent semantics/perf | Use identical math + verify outputs; isolate only address-generation style differences | **Neighbor:** single blog example. **Delta:** systematic quantification + generality claims with evidence. |
| **G8** Version pinning + artifact capture | **A8:** Measurement hygiene: per-run manifest captures toolchain + GPU + env vars + NCU version + backend provenance. | **B8:** Control: pinned environments (conda lock / container) + standardized directory layout for reports/plots. | **H8:** “Repro pack” generator that bundles manifest + scripts + reduced datasets. | **K008 ReproBundle** | Add `manifest.json` + directory conventions; ensure every plot/table has an artifact pointer. | Release “paper appendix pack”: minimal scripts to regenerate figures from raw NCU reports. | (Meta) not a perf metric; but store *all* perf + spill metrics and provenance with versions. | Miss a key version/knob → irreproducible | Manifest schema review + unit test: fail if required fields missing | **Neighbor:** README-level “versions used”. **Delta:** fail-closed manifest + artifact pointers for every figure. |
| **G9** Hardware clarity + Blackwell-specific confounds (e.g., `num_ctas=2` / 2CTA mode MMA) | **A9:** Hardware probe table: cc/SMs/regs/SMEM + NCU support evidence; treat device as factor in every plot. | **B9:** Control confounds: default `num_ctas=1` unless explicitly sweeping; isolate 2CTA effects in a dedicated ablation. (2CTA behavior mentioned in TileIR README; *meaning remains OQ05*.) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | **H9:** Dataset annotation includes `{device, num_ctas, execution_model}` so confounds are visible. | **K009 HWProbe+CTAAblation** | Produce hardware table for RTX5090/B200/H100 + run 1 dense-dot microbench sweeping `num_ctas`. | Add microarch interpretation work: does 2CTA change spills/regs or just compute throughput? | Runtime + reg/spill + execution_model + (candidate) tensor-core utilization metrics (verify names). | Misattribute “2CTA mode” effects to backend | Treat `num_ctas` as explicit factor; do not compare across different `num_ctas` settings | **Neighbor:** ad-hoc “this GPU is Blackwell”. **Delta:** explicit confound isolation + annotated dataset. |
| **G10** Microbench → TritonBench operator linkage | **A10:** Motif taxonomy: map each microbench knob to operator kernels (norm/reduction/GEMM) and show matching spill signatures. | **B10:** Control: select a small TritonBench subset known to stress regs/spills; build stable input sizes + seeds. | **H10:** Case study format: microbench predicts spill regime → TritonBench confirms → attribution supported by NCU. | **K010 MotifMap+TBSubsetRunner** | Pick 3–5 ops; run PTX vs TileIR; collect spill/reg metrics + runtime; show 1–2 aligned findings with microbench. | Expand to more ops; build regression model predicting spill risk from kernel features. | Runtime + core spill/reg metrics + provenance. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Ecological validity challenge | Make mapping explicit + show at least one “prediction succeeded” and one “failed” with explanation | **Neighbor:** TritonBench reports speed. **Delta:** mechanism-level (regs/spills/local) attribution + backend-controlled A/B. |

### 2) Toolbox Verdicts (≤10 bullets)

- **G1 (favored):** Build **K001 ProvenanceGuard** around **NCU `launch__execution_model`** as the authoritative “backend_used” label; add `.tileIR` cache fingerprint as a fast pre-check. This is the cleanest “reviewer-doesn’t-argue” provenance. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai))  
- **G2 (favored):** Do a **capability matrix first** (K002) so we never discover mid-paper that Tile workloads can’t collect spill metrics on your stack. Feasibility becomes a table, not a hope. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html?utm_source=openai))  
- **G3 (favored):** Treat occupancy hint as a **measurable knob** only after you have “API proof”; then K003 produces the central novelty plot: **hint → spills → runtime**. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **G4 (favored):** Enforce a **numeric-mode policy**: one fixed policy for mainline results; approx/FTZ only as explicit ablation with correctness gates (K004). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **G5 (favored):** Build calibration microbenches + a conservative spill classifier (K005) so “local memory traffic” never gets mis-sold as “spills” without corroboration. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai))  
- **G6 (favored):** Adopt a **multi-signal reg-pressure index** and forbid claims from `launch__registers_per_thread` alone; this inoculates against the known “holes/ABI” critique. (Keep as analysis rule baked into K006.)  
- **G7 (favored):** Make descriptor/TMA rewrite a **paired-kernel experiment** (K007) and report its impact on regs/spills, not just speed—this is the highest-transfer “how to write kernels differently” story. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- **G8 (favored):** Ship a **repro bundle** (K008) early; MICRO/ASPLOS reviewers penalize missing manifests more than missing minor optimizations.  
- **G9 (favored):** Treat `num_ctas` (and “2CTA mode MMA”) as a **confound to isolate**, not a tuning tweak to mix into backend comparisons. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **G10 (favored):** Do **one tight linkage case study** (microbench → TritonBench operator) with NCU spill/reg evidence; this is the minimum ecological-validity bar without ballooning scope. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai))  

### 3) Citation-needed / Query Plan (remaining killers)

(These are the **blocking unknowns** you still need to nail down with primary sources *and/or* a one-hour empirical probe on your own box.)

- **Q01 / OQ01 (Occupancy hint API):**  
  - Goal: exact user-facing way to set the TileIR **`occupancy` hint (1–32)** mentioned in README. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
  - Web targets: `triton-lang/Triton-to-tile-IR` **source** (search for “occupancy” in Python config objects / `Config` / `AttrsDescriptor`) + any official docs in repo.  
  - Empirical fallback: set candidate APIs, then verify effect using NCU metrics + `launch__execution_model` stays Tile.
- **Q02 / OQ02 (Metric collectability on real Tile kernels):**  
  - Goal: confirm that **the exact spill metrics** are non-`n/a` on **Tile execution model** for your Blackwell devices.  
  - Method: run K002 probe; record `n/a` vs values; keep raw NCU reports as artifacts.  
  - Web targets: Nsight Compute docs/release notes for any “tile workload limitations” section. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html?utm_source=openai))  
- **Q03 / OQ03 (Best-practice backend detection without NCU):**  
  - Goal: a lightweight provenance method when you don’t want to run NCU on every datapoint (runtime-only sweeps).  
  - Web targets: NVIDIA blog text about `.tileIR` cache artifacts + Triton cache format details. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
  - Empirical: verify `.tileIR` presence correlates with NCU `launch__execution_model=Tile`.
- **Q05 / OQ05 (“2CTA mode MMA” meaning + observability):**  
  - Goal: what exactly changes when `num_ctas=2`, and which NCU metrics best detect it.  
  - Web targets: NVIDIA architecture docs / CUDA Tile docs; otherwise scope it as “observed performance regime change” with explicit ablation only. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **Q10 / OQ06 (RTX 5090 / B200 / GB10 cc+SM counts):**  
  - Goal: authoritative device properties table + NCU support matrix; then freeze which devices are in-scope for claims.  
  - Method: local `deviceQuery`/`nvidia-smi -q` + NCU `launch__device_id` sanity. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai))  

---

## CONTEXT_CAPSULE

```yaml
STATE_VERSION: "manual_state_v4_2_tilespill@2026-02-04"
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
    - "Microarch scope lock: bare-metal microbench + NCU + PTX vs TileIR A/B"
    - "Capsule monotonic: ledgers append-only; shrink forbidden unless justified with pointers"
  current_stage: 1.5
  last_updated: "2026-02-04"
  scope_lock:
    research_question_1liner: "On Blackwell, quantify and attribute PTX-vs-TileIR backend differences in register allocation and spilling using NCU metrics, via controlled sweeps (backend toggle, TileIR occupancy hint, pointer-style rewrite, register-pressure knobs)."
    hypotheses:
      - id: "H1"
        statement: "Backend sweep (ENABLE_TILE=0/1) changes launch__registers_per_thread and spill metrics (derived__local_spilling_requests/_pct, sass__inst_executed_register_spilling) and runtime on same kernel+GPU."
        variables_to_sweep: ["backend: ENABLE_TILE=0 vs 1"]
        metrics_to_observe:
          ["runtime", "launch__execution_model", "launch__registers_per_thread", "derived__local_spilling_requests", "derived__local_spilling_requests_pct", "sass__inst_executed_register_spilling", "sass__inst_executed_local_loads", "sass__inst_executed_local_stores"]
      - id: "H2"
        statement: "TileIR occupancy hint (N=1..32) exposes a performance vs spill tradeoff measurable by runtime and spill metrics."
        variables_to_sweep: ["tileir_occupancy_hint: 1..32"]
        metrics_to_observe:
          ["runtime", "derived__local_spilling_requests(_pct)", "sass__inst_executed_register_spilling", "launch__registers_per_thread"]
        status: "BLOCKED_UNVERIFIED_API_FOR_SETTING_OCCUPANCY_HINT"
      - id: "H3"
        statement: "Descriptor/TMA rewrite reduces register pressure and spilling vs tensor-of-pointers under TileIR."
        variables_to_sweep: ["pointer_style: tensor-of-pointer vs descriptor/TMA"]
        metrics_to_observe:
          ["runtime", "launch__registers_per_thread", "derived__local_spilling_requests(_pct)", "sass__inst_executed_register_spilling"]
      - id: "H4"
        statement: "Reduction/norm kernels with large reduction dim spill more on TileIR due to missing num_warps exposure."
        variables_to_sweep: ["workload_class: gemm-like vs norm/reduction-largeR", "backend: ENABLE_TILE=0 vs 1"]
        metrics_to_observe:
          ["runtime", "derived__local_spilling_requests_pct", "sass__inst_executed_register_spilling"]
    primary_knobs_to_sweep:
      - "backend: PTX vs TileIR (ENABLE_TILE=0/1)"
      - "occupancy hint (TileIR backend) sweep"
      - "tensor-of-pointer vs descriptor/TMA rewrite (where applicable)"
      - "kernel parameters controlling register pressure (unroll, tile size, reduction dim)"
    excluded_topics:
      - "hardware redesign / RTL"
      - "full model training as primary evidence"
      - "unmeasurable compiler internals"
    success_criteria:
      - "At least 1 strong, reproducible finding about spills/regs that changes how kernels should be written/tuned"
      - "A public-ish microbench slice + scripts that produce plots/tables"
      - "Evaluation includes controls + threats to validity"
environment_inventory:
  gpus_available:
    - name: "H100"
      notes: "PTX baseline + cross-arch sanity; TileIR backend may be unavailable"
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
    cuda_version: null   # must be >= 13.1 for TileIR backend
    driver_version: null
    triton_version: null
    python_version: null
    ncu_version: null    # prefer >= 2025.4 for tile workload profiling
    tileir_stack_version: null
    env_vars:
      ENABLE_TILE: null
      TILEIR_ENABLE_APPROX: null
      TILEIR_ENABLE_FTZ: null

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
    notes: "Canonical path uses /latest/; older URL without /latest/ may fail."
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
    last_verified: "2026-02-04"
  - id: "P2"
    kind: "seed_paper"
    title: "ISL layout relations"
    url: "https://arxiv.org/html/2511.10374v1"
    last_verified: "2026-02-04"
  - id: "P3"
    kind: "seed_paper"
    title: "Categorical CuTe layouts"
    url: "https://arxiv.org/pdf/2601.05972v1"
    last_verified: "2026-02-04"

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

VERDICT_LEDGER:
  items:
    - id: "V001"
      date: "2026-02-04"
      gap_id: "G1"
      verdict: "EVAL_CREDIBILITY_CLIFF"
      rationale: "Without per-run backend provenance (TileIR vs PTX, plus fallback detection), A/B results are not reviewable."
      unblock_requires: ["C025", "OQ03", "OQ07"]
    - id: "V002"
      date: "2026-02-04"
      gap_id: "G2"
      verdict: "FEASIBILITY_CLIFF"
      rationale: "Core spill/reg metrics must be collectable for tile workloads under NCU; otherwise evaluation reduces to runtime-only anecdotes."
      unblock_requires: ["C023", "OQ02"]
    - id: "V003"
      date: "2026-02-04"
      gap_id: "G3"
      verdict: "NOVELTY_CRITICAL"
      rationale: "Occupancy hint is described as critical, but without a sweepable API and logging, key tradeoff story and ablation are impossible."
      unblock_requires: ["C024", "OQ01"]
    - id: "V004"
      date: "2026-02-04"
      gap_id: "G1"
      verdict: "FAVORED_APPROACH"
      rationale: "Use NCU launch__execution_model (SIMT vs Tile) as authoritative backend_used label; supplement with Triton cache extension (.tileIR vs .cubin) and fail-closed harness."
      unblock_requires: ["OQ03", "OQ07"]
    - id: "V005"
      date: "2026-02-04"
      gap_id: "G2"
      verdict: "FAVORED_APPROACH"
      rationale: "Build an empirical NCU metric-availability matrix on tile workloads (capability probe) before committing to large sweeps."
      unblock_requires: ["OQ02"]
    - id: "V006"
      date: "2026-02-04"
      gap_id: "G3"
      verdict: "FAVORED_APPROACH"
      rationale: "Treat occupancy hint as novelty centerpiece only after API proof; then sweep 1..32 and report spill/runtime tradeoff curves."
      unblock_requires: ["OQ01"]
    - id: "V007"
      date: "2026-02-04"
      gap_id: "G5"
      verdict: "FAVORED_APPROACH"
      rationale: "Calibrate and classify local memory traffic into spill-dominant vs local-array-dominant using derived spill metrics + SASS corroboration."
      unblock_requires: []
    - id: "V008"
      date: "2026-02-04"
      gap_id: "G7"
      verdict: "FAVORED_APPROACH"
      rationale: "Run paired pointer-vs-descriptor/TMA kernels and quantify reg/spill reductions and performance impacts across backends."
      unblock_requires: []

GAP_LEDGER:
  items:
    - id: "G1"
      rank: 1
      title: "Backend provenance + TileIR→PTX fallback detecti"
      category: "evaluation_credibility"
      status: "OPEN"
      blocks: ["H1", "All A/B claims"]
    - id: "G2"
      rank: 2
      title: "NCU spill/reg metric visibility on CUDA tile workloads"
      category: "feasibility"
      status: "OPEN"
      blocks: ["H1", "EVAL_PLAN core metrics"]
    - id: "G3"
      rank: 3
      title: "Occupancy hint controllability + sweep logging"
      category: "novelty_and_attribution"
      status: "OPEN"
      blocks: ["H2"]
    - id: "G4"
      rank: 4
      title: "Numeric-mode confounding (approx/FTZ) policy for fair A/B"
      category: "attribution"
      status: "OPEN"
      blocks: ["H1"]
    - id: "G5"
      rank: 5
      title: "Spill vs non-spill local memory attribution protocol"
      category: "explainability"
      status: "OPEN"
      blocks: ["Spill interpretation in all results"]
    - id: "G6"
      rank: 6
      title: "launch__registers_per_thread interpretation (holes/ABI constraints)"
      category: "explainability"
      status: "OPEN"
      blocks: ["Reg-pressure claims"]
    - id: "G7"
      rank: 7
      title: "Generalizable descriptor/TMA rewrite evaluation"
      category: "novelty_transfer"
      status: "OPEN"
      blocks: ["H3"]
    - id: "G8"
      rank: 8
      title: "Version pinning + artifact capture for reproducibility"
      category: "reproducibility"
      status: "OPEN"
      blocks: ["paper reproducibility expectations"]
    - id: "G9"
      rank: 9
      title: "Hardware clarity (cc/SKU) + Blackwell-specific mode confounds"
      category: "scope_control"
      status: "OPEN"
      blocks: ["cross-SKU claims", "OQ05/OQ06"]
    - id: "G10"
      rank: 10
      title: "Microbench → TritonBench operator linkage (ecological validity)"
      category: "evaluation_strength"
      status: "OPEN"
      blocks: ["success_criteria case study"]

CLAIM_LEDGER:
  items:
    - id: "C001"
      scope_tag: "ACTIVE"
      claim: "CUDA local memory is thread-private and used when automatic variables don’t fit in registers or when registpilling occurs."
      status: "VERIFIED"
      evidence: ["E001"]
      paper_role: "A"
      risk_if_wrong: "Misclassify spill vs non-spill local traffic."
    - id: "C002"
      scope_tag: "ACTIVE"
      claim: "Local memory resides in device memory and has similar latency to global memory."
      status: "VERIFIED"
      evidence: ["E001"]
      paper_role: "A"
      risk_if_wrong: "Wrong performance interpretation of local/spills."
    - id: "C003"
      scope_tag: "ACTIVE"
      claim: "NCU defines derived__local_spilling_requests and derived__local_spilling_requests_pct as spill-to-local metrics."
      status: "VERIFIED"
      evidence: ["E002"]
      paper_role: "A"
      risk_if_wrong: "Wrong metric selection/interpretation."
    - id: "C004"
      scope_tag: "ACTIVE"
      claim: "NCU provides sass__inst_executed_register_spilling and local load/store SASS instruction counts."
      status: "VERIFIED"
      evidence: ["E002"]
      paper_role: "A"
      risk_if_wrong: "Lose corroboration of derived spill metrics."
    - id: "C005"
      scope_tag: "ACTIVE"
      claim: "launch__registers_per_thread may exceed maximum live registers due to allocation holes and ABI/instruction constraints."
      status: "VERIFIED"
      evidence: ["E003"]
      paper_role: "A"
      risk_if_wrong: "Over-interpret reg allocation differences."
    - id: "C006"
      scope_tag: "ACTIVE"
      claim: "NCU 2025.4 added support for profiling CUDA tile workloads and introduced a Tile section."
      status: "VERIFIED"
      evidence: ["E007", "E012"]
      paper_role: "A"
      risk_if_wrong: "TileIR path not measurable with NCU."
    - id: "C009"
      scope_tag: "ACTIVE"
      claim: "TileIR backend can be enabled via environment variable ENABLE_TILE=1."
      status: "VERIFIED"
      evidence: ["E004", "E005"]
      paper_role: "B"
      risk_if_wrong: "A/B toggle incorrect."
    - id: "C010"
      scope_tag: "ACTIVE"
      claim: "TileIR backend requires CUDA 13.1+ and Blackwell GPUs (initially)."
      status: "VERIFIED"
      evidence: ["E004", "E005", "E008"]
      paper_role: "B"
      risk_if_wrong: "Wrong hardware/toolchain requirements."
    - id: "C012"
      scope_tag: "ACTIVE"
      claim: "TileIR backend may fall back to PTX when a compilation bug occurs; experiments must detect backend."
      status: "VERIFIED"
      evidence: ["E005"]
      paper_role: "B/C"
      risk_if_wrong: "Invalid comparisons due to silent fallback."
    - id: "C014"
      scope_tag: "ACTIVE"
      claim: "num_warps is not exposed yet in TileIR backend; some Norm kernels may spill for large reduction dims."
      status: "VERIFIED"
      evidence: ["E005"]
      paper_role: "B"
      risk_if_wrong: "Misattribute slowdown and spills."
    - id: "C015"
      scope_tag: "ACTIVE"
      claim: "TileIR backend provides an occupancy hint (1–32) described as critical."
      status: "VERIFIED"
      evidence: ["E005"]
      paper_role: "B/C"
      risk_if_wrong: "Miss key tuning knob."
    - id: "C020"
      scope_tag: "ACTIVE"
      claim: "TrinBench provides runnable operator benchmarks via python run.py --op <name>."
      status: "VERIFIED"
      evidence: ["E006"]
      paper_role: "A/C"
      risk_if_wrong: "No real-workload anchors."
    - id: "C021"
      scope_tag: "ACTIVE"
      claim: "TileIR backend disables approx and FTZ by default; can be enabled via TILEIR_ENABLE_APPROX=1 and TILEIR_ENABLE_FTZ=1."
      status: "VERIFIED"
      evidence: ["E005"]
      paper_role: "B"
      risk_if_wrong: "Confounded numeric/ISA differences."
    - id: "C022"
      scope_tag: "ACTIVE"
      claim: "A Blackwell-specific mode referred to as '2CTA mode MMA' may exist and may be triggered/related to num_ctas=2; its meaning and performance impact must be understood or explicitly scoped out."
      status: "UNVERIFIED"
      evidence: []
      paper_role: "B/C"
      risk_if_wrong: "Misattribute Blackwell execution-mode effects to compiler backend."
    - id: "C023"
      scope_tag: "ACTIVE"
      claim: "On Blackwell tile workloads generated via TileIR, Nsight Compute can collect the required spill/reg metrics (launch__registers_per_thread, derived__local_spilling_requests(_pct), sass__inst_executed_register_spilling, sass__inst_executed_local_loads/stores)."
      status: "UNVERIFIED"
      evidence: []
      paper_role: "A (evaluation feasibility)"
      risk_if_wrong: "Core evaluation collapses (no mechanism metrics)."
    - id: "C024"
      scope_tag: "ACTIVE"
      claim: "Triton user code/config exposes a stable way to set TileIR occupancy hint (1–3 and the hint is actually applied (not ignored), enabling reproducible sweeps."
      status: "UNVERIFIED"
      evidence: []
      paper_role: "B/C (novelty + attribution)"
      risk_if_wrong: "Cannot evaluate critical knob; weaker paper."
    - id: "C025"
      scope_tag: "ACTIVE"
      claim: "There is a robust, per-kernel, per-run backend provenance method to guarantee whether TileIR was used or a PTX fallback occurred, enabling fail-closed A/B experiments."
      status: "VERIFIED"
      evidence: ["E010", "E011"]
      paper_role: "A (evaluation credibility)"
      risk_if_wrong: "A/B comparisons can be invalid without detection."
    - id: "C026"
      scope_tag: "ACTIVE"
      claim: "NCU exposes launch__execution_model (SIMT vs Tile), enabling runtime-confirmed provenance of execution model per kernel launch."
      status: "VERIFIED"
      evidence: ["E010"]
      paper_role: "A (provenance)"
      risk_if_wrong: "Backend detection remains ambiguous."
    - id: "C027"
      scope_tag: "ACTIVE"
      claim: "When Tile IR backend is active, Triton caches compiled kernels with .tileIR file extensions (vs .cubin for SIMT backend)."
      status: "VERIFIED"
      evidence: ["E011"]
      paper_role: "A/B provenance aid"
      risk_if_wrong: "Cache-fingerprint provenance could mislead."

EVIDENCE_LEDGER:
  items:
    - id: "E001"
      source_id: "NV-NCU"
      kind: "doc"
      pointer: "NCU Profiling Guide: local memory definition + spilling causes"
      url: "https://docs.nvidia.com/nsight-compute/ProfilingGuide/"
      status: "VERIFIED"
    - id: "E002"
      source_id: "NV-NCU"
      kind: "doc"
      pointer: "NCU Profiling Guide metric definitions: derived__local_spilling_requests(_pct), sass__inst_executed_register_spilling, local loads/stores"
      url: "https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html"
      status: "VERIFIED"
    - id: "E003"
      source_id: "NV-NCU"
      kind: "doc"
      pointer: "Nsight Compute docs: launch__registers_per_thread caveat (holes/ABI/hardware constraints)"
      url: "https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html"
      status: "VERIFIED"
    - id: "E004"
      source_id: "NV-TILE-BLOG"
      kind: "blog"
      pointer: "NVIDIA blog (Jan 30, 2026): ENABLE_TILE=1, prereqs CUDA 13.1+ and Blackwell, limitations, TMA rewrite example"
      url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
      status: "VERIFIED"
    - id: "E005"
      source_id: "TILE-REPO"
      kind: "repo_readme"
      pointer: "Triton-to-tile-IR README: ENABLE_TILE=1, occupancy hint 1–32, missing num_warps, fallback to PTX, approx/FTZ env vars, num_ctas=2 note"
      url: "https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file"
      status: "VERIFIED"
    - id: "E006"
      source_id: "TB"
      kind: "repo_readme"
      pointer: "TritonBench README: python run.py --op gemm"
      url: "https://github.com/meta-pytorch/tritonbench"
      status: "VERIFIED"
    - id: "E0"
      source_id: "NV-NCU"
      kind: "release_note_forum"
      pointer: "Nsight Compute 2025.4 forum post: support for profiling CUDA tile workloads + Tile section"
      url: "https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614"
      status: "VERIFIED"
    - id: "E008"
      source_id: "CUDA-13.1-RN"
      kind: "release_notes"
      pointer: "CUDA 13.1 release notes: introduces CUDA Tile; initial release targets Blackwell GPUs"
      url: "https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html"
      status: "VERIFIED"
    - id: "E009"
      source_id: "NV-TILEIR-MM"
      kind: "spec"
      pointer: "Tile IR memory model: weak ops, scopes, ordering, tokens, PTX interoperability"
      url: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html"
      status: "VERIFIED"
    - id: "E010"
      source_id: "NV-NCU"
      kind: "doc"
      pointer: "NCU 2025.4 Profiling Guide Launch Metrics includes launch__execution_model (SIMT vs Tile) + metric query guidance"
      url: "https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html"
      status: "VERIFIED"
    - id: "E011"
      source_id: "NV-TILE-BLOG"
      kind: "blog"
      pointer: "NVIDIA blog: Verify Tile IR compilation section mentions Triton cache .tileIR extensions when Tile backend active"
      url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
      status: "VERIFIED"
    - id: "E012"
      source_id: "NV-NCU"
      kind: "release_notes_doc"
      pointer: "Nsight Compute Release Notes (docs): 2025.4 adds CUDA tile workload profiling and Tile section"
      url: "https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html"
      status: "VERIFIED"

EXPERIMENT_LEDGER:
  items: []

EVAL_PLAN:
  status: "draft"
  non_negotiables_added_stage1:
    - id: "NN1"
      requirement: "Per-run backend provenance: every measured datapoint must include a backend_used/provenance artifact; fail-closed if ambiguous (addresses G1)."
    - id: "NN2"
      requirement: "Metric feasibility check on tile workloads: before large sweeps, demonstrate required NCU spill/reg metrics are collectable with ENABLE_TILE=1 (addresses G2)."
    - id: "NN3"
      requirement: "All reports must include both derived spill metrics and SASS corroboration metrics (derived__local_spilling_requests(_pct) + sass__inst_executed_register_spilling + local loads/stores)."
    - id: "NN4"
      requirement: "Backend_used must be determined from NCU launch__execution_model (SIMT vs Tile) for all NCU-profiled runs; store it per kernel instance (strengthens G1)."
  methodology:
    - "Bare-metal runs; warmup; repeat; report median + variability."
    - "Pin toolchain versions + env vars; treat backend toggles as experimental factors."
    - "Use NCU metric query capability to confirm metric availability per version and per workload type (SIMT vs tile)."
    - "Detect TileIR vs PTX fallback explicitly; fail-closed if backend ambiguous."
    - "Record numeric-mode env vars (TILEIR_ENABLE_APPROX/FTZ) and keep them fixed unless performing a dedicated ablation."
  microbench_motifs_added_stage1_5:
    - id: "MB1"
      motif: "Reg-pressure ladder (sweep unroll / accumulators / tile sizes) to induce and then relieve spills"
      purpose: "Calibrate spill metrics and show backend differences"
    - id: "MB2"
      motif: "Local-array vs spill disambiguator (local traffic without spilling vs forced spilling)"
      purpose: "Support spill attribution protocol (G5)"
    - id: "MB3"
      motif: "Pointer-tensor vs descriptor/TMA rewrite pair"
      purpose: "Quantify reg/spill delta transfer (G7)"
    - id: "MB4"
      motif: "num_ctas ablation microbench (1 vs 2) on dot-like workloads"
      purpose: "Isolate 2CTA confound (G9)"
  metrics:
    - "runtime"
    - "launch__execution_model"
    - "launch__registers_per_thread"
    - "derived__local_spilling_requests (+ pct)"
    - "sass__inst_executed_register_spilling (+ local_loads/local_stores)"
  baselines:
    - "PTX backend (ENABLE_TILE=0) vs TileIR backend (ENABLE_TILE=1) on same Blackwell GPU"
    - "TileIR occupancy hint sweep (1..32) (BLOCKED until C024/OQ01 resolved)"
    - "tensor-of-pointer vs descriptor/TMA rewrite (when applicable)"
  workloads:
    - "TritonBench subset (select ops with reduction/norm + GEMM motifs)"
    - "Custom microbench kernels sweeping register pressure"
  ablations:
    - "Dedicated TileIR numeric-mode ablation: toggle TILEIR_ENABLE_APPROX / TILEIR_ENABLE_FTZ only as a controlled factor (not mixed into main A/B)."
    - "Dedicated num_ctas ablation to isolate 2CTA mode (if applicable)."
  risks_to_validity:
    - "backend feature drift across CUDA/Triton versions"
    - "register count metric overstates live regs; interpret carefully"
    - "clock/frequency variability"
    - "TileIR backend fallback to PTX can invalidate A/B unless detected"
    - "NCU profiling overhead/replay can perturb runtime and potentially code scheduling"

OPEN_QUESTIONS:
  active:
    - id: "OQ01"
      status: "OPEN"
      statement: "Exact API/syntax to set TileIR occupancy hint in Triton user code/config."
      blocks: ["H2", "C015", "C024", "G3"]
      plan: ["Q01"]
    - id: "OQ02"
      status: "OPEN"
      statement: "Which NCU spill/register metrics are collectable on CUDA tile workloads on Blackwell in NCU 2025.4+ (non-n/a)."
      blocks: ["C003", "C004", "C006", "C023", "G2"]
      plan: ["Q02", "Q08"]
    - id: "OQ03"
      status: "OPEN"
      statement: "Best-practice backend detection to guarantee no TileIR->PTX fallback in measured runs (esp. runtime-only sweeps without NCU). Candidate: correlate Triton cache .tileIR vs NCU launch__execution_model."
      blocks: ["C012", "G1"]
      plan: ["Q03", "Q04"]
    - id: "OQ04"
      status: "OPEN"
      statement: "Scope/identities of 'XXXNorm' kernels impacted by missing num_warps; workaround options."
      blocks: ["H4", "C014"]
      plan: ["Q06"]
    - id: "OQ05"
      status: "OPEN"
      statement: "Blackwell-specific meaning and measurement of '2CTA mode MMA' tied to num_ctas=2."
      blocks: ["C022", "G9"]
      plan: ["Q09"]
    - id: "OQ06"
      status: "OPEN"
      statement: "Confirm compute capability + SM counts for RTX 5090, B200, GB10; validate NCU support matrix."
      blocks: ["Env template completeness", "G9"]
      plan: ["Q10"]
    - id: "OQ07"
      status: "OPEN"
      statement: "How to extract/persist per-kernel compilation artifacts/provenance from Triton (incl. cache behavior) to support fail-closed backend_used labeling for runtime-only sweeps."
      blocks: ["G1"]
      plan: ["Q11"]
    - id: "OQ08"
      status: "OPEN"
      statement: "Define and validate a numeric-mode control policy for PTX vs TileIR A/B (approx/FTZ), including output-check strategy."
      blocks: ["G4"]
      plan: ["Q12"]
  closed: []

DELTA_LOG:
  - "Set profile.current_stage=1.5."
  - "ARTIFACT_INDEX.stage1_5_toolbox set to WP1_5_20260204."
  - "Added VERDICT_LEDGER entries V004..V008 selecting favored toolbox approaches for G1/G2/G3/G5/G7."
  - "Upgraded C025 to VERIFIED based on NCU launch__execution_model and TileIR cache fingerprint evidence."
  - "Added new VERIFIED claims C026 (NCU launch__execution_model provenance) and C027 (.tileIR cache extension when Tile backend active)."
  - "Added EVIDENCE entries E010 (NCU launch__execution_model) and E011 (blog .tileIR cache), and E012 (official NCU release notes)."
  - "Updated C006 evidence to include E012."
  - "EVAL_PLAN updated: added NN4 (execution-model provenance) + added Stage1.5 microbench motifs MB1..MB4."

CAPSULE_HEALTH:
  counts:
    verdict_ledger: 8
    claim_ledger: 19
    evidence_ledger: 12
    experiment_ledger: 0
    open_questions_active: 8
    open_questions_closed: 0
  capsule_token_estimate: 5400
  monotonicity: "OK (stage1 → stage1.5 growth)"

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 2 prompt."
```

---
Learn more:
1. [2. Profiling Guide — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html_source=openai)
2. [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
3. [1. Release Notes — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html?utm_source=openai)
4. [GitHub - triton-lang/Triton-to-tile-IR: incubator repo for CUDA-TileIR backend](https://github.com/triton-lang/Triton-to-tile-IR?tab=readmov-file)
