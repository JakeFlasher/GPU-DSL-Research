# WORK_PRODUCT

## 1) Direction Cards (D1..D3)

## D1 — Spill provenance + attribution (separate “true spills” from other local traffic)

### Thesis
A calibrated microbench ladder plus a spill-vs-local attribution protocol will reveal **systematic, reproducible differences** in register allocation and spilling between **PTX (ENABLE_TILE=0)** and **TileIR (ENABLE_TILE=1)** on Blackwell, as measured by NCU spill/register metrics.

### Research question
Under controlled register-pressure and kernel-structure sweeps, **when** does TileIR spill “earlier/more” than PTX (or vice versa), and can we **attribute** observed local traffic to *spills* vs *non-spill local* with a fail-closed methodology?

### Hypotheses (H#)
- **H1 (backend shifts reg allocation)**
  - **Sweep:** `backend ∈ {ENABLE_TILE=0, ENABLE_TILE=1}` on identical kernels + inputs
  - **NCU metrics:** `launch__execution_model`, `launch__registers_per_thread`
  - **Expected trend:** TileIR run shows `launch__execution_model=Tile` and a **different** (expected: **higher**) `launch__registers_per_thread` vs PTX/SIMT for the same tuning parameters.

- **H2 (spill “cliff” shifts with backend)**
  - **Sweep:** `reg_pressure_knob` (e.g., unroll / accumulator count / tile size) × `backend`
  - **NCU metrics:** `derived__local_spilling_requests`, `derived__local_spilling_requests_pct`, `sass__inst_executed_register_spilling`, `sass__inst_executed_local_loads`, `sass__inst_executed_local_stores`, plus `runtime`
  - **Expected trend:** As reg pressure increases, spill metrics stay low then **increase sharply**; the “knee” occurs at a **lower reg-pressure setting** under TileIR than PTX.

- **H3 (local traffic ≠ spills; protocol is discriminative)**
  - **Sweep:** `local_array_size` in a kernel designed to use local arrays (without forcing spills) vs a kernel designed to force spills (using MB2 controls)
  - **NCU metrics:** `derived__local_spilling_requests_pct` vs `sass__inst_executed_local_loads/stores`
  - **Expected trend:** Non-spill-local case shows **high local loads/stores** but **near-zero** derived spill %; spill-forced case shows **high derived spill %** and **high register-spilling SASS**.

- **H4 (norm/reduction sensitivity under TileIR)**
  - **Sweep:** `reduction_dim` (small→large) × `backend` on norm/reduction motifs (custom + TritonBench class)
  - **NCU metrics:** spill metrics above + `launch__registers_per_thread` + `runtime`
  - **Expected trend:** TileIR spill % grows **more steeply** with reduction size than PTX (directional expectation: TileIR worse at large-R), consistent with “missing tuning exposure” constraints.

### Closest neighbors + delta type
- **Closest neighbors**
  - NVIDIA Nsight Compute workflow: using `launch__registers_per_thread` + derived spill metrics as indicators (standard practice).
  - NVIDIA TileIR/Triton Tile backend public materials: enablement + caveats + fallback risk (baseline facts, not new).
  - Generic GPU microbench methodology: controlled sweeps, attribution, confound control (baseline technique).
- **Delta type:** **(a) new attribution method**  
  (backend-provenance + calibrated “spill vs non-spill local” discriminator + spill-cliff analysis across backends)

### Artifact target(s)
- **Mandatory**
  - Microbench suite slice: **MB1 + MB2 + MB4** (plus harness glue)
  - Measurement scripts: `ncu` capture + provenance logging + sweep automation
  - Analysis report: plots/tables for reg/spill curves + attribution outcomes
- **Optional**
  - Lightweight classifier script that labels each datapoint: `{spill-dominant, local-array-dominant, mixed}` based on NCU metrics

### Implementation plan (weeks 1–4; weeks 5–12)
- **Weeks 1–2 (first plots)**
  - Build **fail-closed provenance** into harness: require `launch__execution_model` for NCU runs; store cache fingerprints for runtime-only sweeps.
  - Run **NCU metric feasibility matrix** on Blackwell Tile workloads (resolve metric n/a risk early).
  - Implement MB1 ladder; produce first curves: `regs/thread` + `spill%` vs reg-pressure knob for PTX vs TileIR.

- **Weeks 3–4 (case studies)**
  - Implement MB2 disambiguator; validate protocol (H3).
  - Apply to **5–15 TritonBench ops (category-selected)**; pick at least:
    - one GEMM-like, one attention-like, one norm/reduction-largeR.
  - Produce 1–2 “deep dive” case studies with **SASS corroboration** metrics.

- **Weeks 5–12 (stretch)**
  - Cross-check on **2 Blackwell SKUs** (RTX 5090 vs B200) to rule out SKU quirks.
  - Add optional pointer-style ablation (MB3) as a second story.
  - If occupancy hint API becomes controllable, add sweep curves as an ablation rather than a dependency.

### Evaluation (metrics, baselines, workloads, methodology, GPU matrix)
- **Metrics (non-negotiable)**
  - `runtime`
  - `launch__execution_model`
  - `launch__registers_per_thread`
  - `derived__local_spilling_requests`, `derived__local_spilling_requests_pct`
  - `sass__inst_executed_register_spilling`
  - `sass__inst_executed_local_loads`, `sass__inst_executed_local_stores`

- **Baselines**
  - **PTX vs TileIR** on Blackwell: `ENABLE_TILE=0` vs `ENABLE_TILE=1`
  - **Occupancy hint sweep 1..32** (included as baseline; **blocked until API proof**)
  - **Descriptor rewrite** where applicable (secondary baseline; not required for MVP)

- **Workloads**
  - **TritonBench ops: 5–15 (exact op names TBD; use categories TB_OP## until verified)**
    - TB_OP01: GEMM / matmul class
    - TB_OP02: attention class
    - TB_OP03: softmax class
    - TB_OP04: layernorm class
    - TB_OP05: rmsnorm class
    - TB_OP06: reduction-sum class
    - TB_OP07: reduction-max class
    - TB_OP08: fused elementwise chain class
    - TB_OP09: gather/scatter (pointer-heavy) class
    - TB_OP10: normalization variant (alt) class
  - **Custom motifs (3)**
    - MB1 reg-pressure ladder
    - MB2 local-array vs spill disambiguator
    - MB4 `num_ctas` (1 vs 2) ablation to isolate Blackwell execution-mode confounds

- **Methodology**
  - Warmup + repeats; report median + variability.
  - Pin toolchain + env vars; keep numeric-mode fixed unless doing a dedicated ablation.
  - For every NCU datapoint: store `launch__execution_model`; **fail-closed** if ambiguous.
  - Treat NCU overhead separately: collect runtime both with and without profiling when needed.

- **GPU matrix**
  - **RTX 5090 (Blackwell-class):** primary iteration + full A/B (ENABLE_TILE=0/1) + initial metric feasibility
  - **B200 (Blackwell datacenter):** confirmatory runs for key findings + larger-scale sweeps
  - **H100:** PTX-only sanity checks (methodology + spill-vs-local protocol behavior), plus “is effect Blackwell-only?” context

### Risks + mitigations
- **Risk 1:** Required NCU spill/register metrics show as n/a for Tile workloads (feasibility cliff).
  - **Mitigation:** early **metric feasibility matrix** (weeks 1–2); if partial, adjust metric set but keep derived spill + SASS corroboration as core gate.
- **Risk 2:** Silent TileIR→PTX fallback contaminates A/B.
  - **Mitigation:** **fail-closed provenance**: require `launch__execution_model` in NCU runs; for runtime-only sweeps, persist compilation/cache fingerprints and reject ambiguous runs.

---

## D2 — Occupancy-hint tradeoff curves + empirically validated selection heuristic

### Thesis
If the TileIR occupancy hint is truly applied, then sweeping it (1..32) will produce **predictable reg/spill/throughput tradeoffs**, enabling a **simple, validated heuristic** that picks near-optimal hint values per kernel class.

### Research question
How does **TileIR occupancy hint** reshape register allocation and spilling, and can we select a good hint using only NCU-observable signals (regs/thread + spill%) without expensive autotuning?

### Hypotheses (H#)
- **H1 (hint controls a reg budget → spill tradeoff)**
  - **Sweep:** `tileir_occupancy_hint ∈ [1..32]` under `ENABLE_TILE=1`
  - **NCU metrics:** `launch__registers_per_thread`, spill metrics, `runtime`
  - **Expected trend:** Increasing hint (toward higher occupancy) drives **lower regs/thread** but **higher spill%**; runtime shows a **U-shape** (too-low occupancy vs too-many spills).

- **H2 (knee exists and is stable within a kernel class)**
  - **Sweep:** occupancy hint × representative inputs within a kernel class (e.g., fixed shape family)
  - **NCU metrics:** `derived__local_spilling_requests_pct` vs `runtime`
  - **Expected trend:** There is a consistent “knee” where spill% rises sharply; best runtime lies **just before** the knee.

- **H3 (class-dependent optima; reductions/norms are more spill-limited)**
  - **Sweep:** workload class (GEMM-like vs norm/reduction-largeR) × occupancy hint
  - **NCU metrics:** spill metrics + `runtime`
  - **Expected trend:** Norm/reduction kernels prefer **lower hint** (spill-avoidance) while GEMM-like kernels tolerate **higher hint** (latency hiding).

- **H4 (heuristic beats default or fixed hint)**
  - **Sweep:** selection policy `{default, fixed hint, heuristic}` across workloads
  - **NCU metrics:** `runtime` (+ check spill% as mechanism evidence)
  - **Expected trend:** Heuristic improves median runtime vs default with bounded spill% increase (or reduced spill%).

### Closest neighbors + delta type
- **Closest neighbors**
  - Vendor guidance that occupancy/registers/spills trade off (general).
  - TileIR backend public docs describing an occupancy hint (mentions exist; no published curves/heuristics).
  - Standard Triton tuning practices (num_warps/num_stages) for SIMT kernels (not the same knob).
- **Delta type:** **(d) a scoped mitigation heuristic/pass that is empirically validated**  
  (“choose occupancy hint via knee-finding on regs/spill% curves”)

### Artifact target(s)
- **Mandatory**
  - Microbench + TritonBench sweep harness for occupancy hint
  - Measurement scripts (NCU capture per hint)
  - Analysis report with tradeoff curves + heuristic evaluation
- **Optional**
  - A small “hint selector” tool (offline) that emits recommended hint per kernel signature

### Implementation plan (weeks 1–4; weeks 5–12)
- **Weeks 1–2 (first plots)**
  - **Blocker-first:** prove controllable occupancy hint API + prove it affects `launch__registers_per_thread` (otherwise direction collapses).
  - Generate first curves on 3 microbenches (MB1-style ladder + one reduction motif).

- **Weeks 3–4 (case studies)**
  - Sweep 5–15 TritonBench ops; cluster into classes; show per-class knees + best hints.
  - Compare heuristic vs default.

- **Weeks 5–12 (stretch)**
  - Cross-SKU stability: RTX 5090 vs B200.
  - Combine with descriptor rewrite (MB3) to see if hint optima shift when reg pressure drops.
  - Write a short “selection guide” (rules + supporting plots).

### Evaluation (metrics, baselines, workloads, methodology, GPU matrix)
- **Metrics:** same non-negotiables (runtime + regs/thread + derived spill + SASS spill/local loads/stores).
- **Baselines**
  - PTX vs TileIR (ENABLE_TILE=0/1)
  - **Occupancy hint sweep 1..32 (core factor)**
  - Descriptor rewrite where applicable (secondary)
- **Workloads**
  - TritonBench 5–15 ops (TB_OP01..TB_OP10 categories; exact names TBD)
  - Custom motifs (3): MB1, MB2, MB4 (plus an occupancy-hint sweep wrapper)
- **Methodology**
  - Warmup/repeats; fixed numeric-mode.
  - Fail-closed backend provenance for any datapoint used in curves.
- **GPU matrix**
  - Blackwell (RTX 5090 + B200): required for occupancy hint; full sweep
  - H100: PTX-only comparisons for “knob uniqueness” context

### Risks + mitigations
- **Risk 1:** Occupancy hint API is not accessible/stable or hint is ignored (C024 currently unverified).
  - **Mitigation:** make “API proof + effect on regs/thread” a week-1 gate; if it fails, pivot D2 into a *secondary ablation* rather than the main direction.
- **Risk 2:** Curves are highly shape-dependent (poor generalization).
  - **Mitigation:** constrain claims to well-defined shape families + publish the selection method and confidence bounds; report per-class rather than “universal” heuristic.

---

## D3 — Descriptor/TMA rewrite as a register-pressure + spill mitigation (TileIR vs PTX differential benefit)

### Thesis
Replacing tensor-of-pointers addressing with descriptor/TMA-style addressing yields a **larger reduction in reg pressure and spills under TileIR** than under PTX, producing measurable speedups on pointer-heavy kernels and a practical “write kernels this way” mitigation.

### Research question
For pointer-heavy access patterns, how much does a descriptor/TMA rewrite reduce `regs/thread` and spill metrics, and is the benefit **backend-dependent** (TileIR > PTX)?

### Hypotheses (H#)
- **H1 (rewrite reduces regs/spills under TileIR)**
  - **Sweep:** `pointer_style ∈ {tensor-of-pointers, descriptor/TMA}` under `ENABLE_TILE=1`
  - **NCU metrics:** `launch__registers_per_thread`, derived spill metrics, SASS spill/local loads/stores, `runtime`
  - **Expected trend:** Descriptor/TMA version shows **lower regs/thread**, **lower spill%**, **fewer spill SASS inst**, and improved runtime.

- **H2 (TileIR benefits more than PTX)**
  - **Sweep:** `backend ∈ {PTX, TileIR}` × `pointer_style`
  - **NCU metrics:** same as H1
  - **Expected trend:** The delta (tensor-of-pointers → descriptor) is **larger** under TileIR than under PTX.

- **H3 (benefit grows with pointer complexity)**
  - **Sweep:** `pointer_tensor_rank/size` (e.g., number of pointers carried per thread) × pointer_style
  - **NCU metrics:** regs/thread + spill metrics
  - **Expected trend:** Tensor-of-pointers shows rising regs/thread and spill% with pointer complexity; descriptor version grows more slowly.

- **H4 (runtime tracks spill SASS more than raw regs/thread)**
  - **Sweep:** across kernels/inputs; compare correlation of runtime with (a) regs/thread vs (b) spill/local SASS counts
  - **NCU metrics:** `runtime` vs `sass__inst_executed_register_spilling` (+ local loads/stores)
  - **Expected trend:** Runtime improvement aligns more strongly with reduced spill/local instruction counts than with reg count alone.

### Closest neighbors + delta type
- **Closest neighbors**
  - Public TileIR backend examples that demonstrate a descriptor/TMA rewrite (baseline exemplar, not generalized).
  - Standard “reduce register pressure by changing data representation” folklore in GPU programming.
- **Delta type:** **(d) a scoped mitigation heuristic/pass that is empirically validated**  
  (“use descriptor/TMA pattern when pointer tensors exceed threshold; quantify backend-dependent wins”)

### Artifact target(s)
- **Mandatory**
  - Microbench pair MB3 (tensor-of-pointers vs descriptor/TMA) + at least one real-op adaptation
  - Measurement scripts (A/B, NCU capture, provenance)
  - Analysis report with per-backend deltas + generalization evidence
- **Optional**
  - A small, scoped source-to-source helper (template/code pattern) to standardize the rewrite in Triton kernels

### Implementation plan (weeks 1–4; weeks 5–12)
- **Weeks 1–2 (first plots)**
  - Implement MB3 pair; validate outputs match (correctness gate).
  - Run PTX vs TileIR A/B on Blackwell; produce first delta table: regs/thread, spill%, runtime.

- **Weeks 3–4 (case studies)**
  - Identify 2–4 TritonBench ops in pointer-heavy / memory-indirection categories (exact names TBD) and apply rewrite where feasible.
  - Run occupancy hint sweep as secondary ablation if API is available.

- **Weeks 5–12 (stretch)**
  - Broaden to more ops; derive a “when it helps” rule using simple predictors (pointer tensor size, indexing structure).
  - Cross-SKU replication: RTX 5090 vs B200.
  - Integrate with D1 attribution protocol to avoid mislabeling local traffic.

### Evaluation (metrics, baselines, workloads, methodology, GPU matrix)
- **Metrics:** same non-negotiables (runtime + execution_model + regs/thread + derived spill + SASS spill/local loads/stores).
- **Baselines**
  - PTX vs TileIR (ENABLE_TILE=0/1)
  - Pointer style A/B (core factor)
  - Occupancy hint sweep 1..32 (secondary; blocked until API proof)
- **Workloads**
  - TritonBench 5–15 ops total, but emphasize pointer-heavy classes (TB_OP09-like)
  - Custom motifs (3): **MB3 + MB1 + MB2**
- **Methodology**
  - Correctness checks for rewrite variants (output compare).
  - Fail-closed backend provenance and fixed numeric-mode.
- **GPU matrix**
  - Blackwell (RTX 5090 + B200): required for TileIR + main claims
  - H100: PTX-only “rewrite helps even without TileIR?” context (optional)

### Risks + mitigations
- **Risk 1:** Descriptor/TMA rewrite is not broadly applicable across TritonBench ops (limited ecological validity).
  - **Mitigation:** (i) explicitly scope the claim to a defined kernel pattern class, (ii) demonstrate at least 2 real-op adaptations, (iii) provide a decision rule for applicability.
- **Risk 2:** Performance deltas confounded by backend numeric-mode defaults (approx/FTZ) or other toggles.
  - **Mitigation:** lock numeric-mode env vars for all A/B comparisons; run a dedicated numeric-mode ablation only as a separate experiment.

---

## 2) Decision Matrix

Scales: **Novelty/Feasibility/Eval Credibility:** 1(low)–5(high). **Risk:** 1(low risk)–5(high risk).

| Direction | Novelty(1-5) | Feasibility(1-5) | Eval Credibility(1-5) | Risk(1-5) | Why it wins | Key unknown evidence needed |
|---|---:|---:|---:|---:|---|---|
| **D1** | 4 | 4 | 5 | 2 | Directly addresses G1/G2/G5 with a publishable *method + findings* even if occupancy hint is blocked; produces defensible, attribution-grade results reviewers can audit. | **OQ02:** do required NCU spill/reg metrics work on Tile workloads? **OQ03/OQ07:** provenance robustness for runtime-only sweeps. **OQ10:** toolchain pin. |
| **D2** | 5 | 2 | 4 | 5 | Potentially strongest “knob story” + heuristic if controllable; could become standout novelty. | **OQ01/C024:** occupancy hint API + proof it changes regs/thread and affects performance. Without this, direction collapses. |
| **D3** | 4 | 3 | 4 | 3 | Concrete mitigation story; compelling if it generalizes beyond a toy kernel and shows TileIR-specific leverage. | Applicability breadth: can we adapt ≥2 real TritonBench ops? Also needs **OQ02** metric availability and numeric-mode controls (**OQ08**). |

**Provisional winner: D1.**

---

## 3) Stage-2 Verdict (≤12 bullets)

- **Winner:** **D1 (spill provenance + attribution)** as primary direction; D3 as best secondary story if MB3 generalizes.
- D1 is the only direction that remains publishable even if **occupancy hint control (OQ01)** is delayed.
- **Immediate gate (week 1):** resolve **OQ02** with an NCU metric feasibility matrix on Blackwell Tile workloads; fail early if core metrics are n/a.
- Keep **backend provenance fail-closed** as a hard rule for every plotted datapoint (NCU `launch__execution_model` where possible).
- Treat **TileIR→PTX fallback** as a first-class confound; reject ambiguous runs, don’t “average it out.”
- Use MB2 to prevent a common reviewer critique: “local loads/stores might be non-spill local arrays.”
- Run A/B primarily on **RTX 5090** for iteration speed; replicate key curves on **B200** to show stability.
- Use **H100** only for PTX-only sanity checks and to validate that the attribution protocol isn’t Blackwell-specific.
- Do not mix numeric-mode toggles into main A/B; isolate them as a dedicated ablation (OQ08).
- Stage-2 output locks: **metrics set, baselines, workload-category targets, and microbench motifs MB1/MB2/MB4**.

---

# CONTEXT_CAPSULE

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
  current_stage: 2
  last_updated: "2026-02-04"
  scope_lock:
    research_question_1liner: "On Blackwell, attribute PTX-vs-TileIR backend differences in register allocation and spilling using fail-closed provenance plus calibrated microbench ladders and Nsight Compute spill/register metrics, and produce reproducible guidance for when kernels spill (and how to avoid it)."
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
      - id: "H5"
        statement: "Under a calibrated reg-pressure ladder, the spill 'knee/cliff' (derived__local_spilling_requests_pct rise) occurs at different reg-pressure settings under TileIR vs PTX, enabling causal attribution of backend impact."
        variables_to_sweep: ["reg_pressure_knob: unroll/accumulators/tile sizes", "backend: ENABLE_TILE=0 vs 1"]
        metrics_to_observe:
          ["runtime", "launch__registers_per_thread", "derived__local_spilling_requests_pct", "sass__inst_executed_register_spilling", "sass__inst_executed_local_loads", "sass__inst_executed_local_stores"]
        status: "PLANNED_STAGE2_D1_PRIMARY"
      - id: "H6"
        statement: "A local-array vs spill disambiguator (MB2) can validate that derived spill metrics track true spill-to-local, while local load/store SASS counts alone can include non-spill local traffic."
        variables_to_sweep: ["local_array_size", "forced_spill_toggle", "backend: ENABLE_TILE=0 vs 1"]
        metrics_to_observe:
          ["derived__local_spilling_requests_pct", "sass__inst_executed_register_spilling", "sass__inst_executed_local_loads", "sass__inst_executed_local_stores"]
        status: "PLANNED_STAGE2_D1_PRIMARY"
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
  stage2_directions: "WP2_20260204"
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
    - id: "V009"
      date: "2026-02-04"
      gap_id: "S2"
      verdict: "PROVISIONAL_WINNER_D1"
      rationale: "Direction D1 (spill provenance + attribution) best addresses evaluation credibility and feasibility cliffs (G1/G2/G5) and yields publishable, auditable results even if occupancy hint control remains blocked."
      unblock_requires: ["OQ02", "OQ03", "OQ07", "OQ10"]

GAP_LEDGER:
  items:
    - id: "G1"
      rank: 1
      title: "Backend provenance + TileIR→PTX fallback detection"
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
      category: "novey_and_attribution"
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
      claim: "CUDA local mory is thread-private and used when automatic variables don’t fit in registers or when register spilling occurs."
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
      risk_if_wrong: "Wrong performance interetation of local/spills."
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
      rk_if_wrong: "Miss key tuning knob."
    - id: "C020"
      scope_tag: "ACTIVE"
      claim: "TritonBench provides runnable operator benchmarks via python run.py --op <name>."
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
      claim: "Triton user code/config exposes a stable way to set TileIR occupancy hint (1–32) and the hint is actually applied (not ignored), enabling reproducible sweeps."
      status: "UNVERIFIED"
      evidence: []
      paper_role: "B/C (novelty + attribution)"
      risk_if_wrong: "Cannot evaluate critical knob; weaker paper."
    - id: "C025"
      scope_tag: "ACTIVE"
      claim: "There is a robust, per-kernel, per-run backend provenance method to guarantee whether TileIR was used or a PTX faback occurred, enabling fail-closed A/B experiments."
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
      pointer: "TritonBench README: python run.py --op gem
      url: "https://github.com/meta-pytorch/tritonbench"
      status: "VERIFIED"
    - id: "E007"
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
    - id: "NN5"
      requirement: "Include MB2-style calibration to disambiguate spill-dominant vs non-spill local traffic; publish the attribution rule used in plots/tables (addresses G5 reviewer risk)."
  methodology:
    - "Bare-metal runs; warmup; repeat; report median + variability."
    - "Pin toolchain versions + env vars; treat backend toggles as experimental factors."
    - "Use NCU metric query capability to confirm metric availability per version and per workload type (SIMT vs tile)."
    - "Detect TileIR vs PTX fallback explicitly; fail-closed if backend ambiguous."
    - "Record numeric-mode env vars (TILEIR_ENABLE_APPROX/FTZ) and keep them fixed unless performing a dedicated ablation."
    - "For any runtime-only sweep used in conclusions: persist a provenance artifact (cache fingerprint + compile logs) or do not include in main figures."
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
    - "TritonBench subset: 5–15 ops spanning GEMM-like, attention-like, norm/reduction-largeR, and pointer-heavy classes (exact op names TBD; maintain a TB_OP## category map until verified)."
    - "Custom microbench kernels: MB1 + MB2 + MB4 required for MVP; MB3 optional/secondary."
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
    - id: "OQ09"
      status: "OPEN"
      statement: "Finalize the exact TritonBench op list (5–15) and create TB_OP## category map (GEMM/attention/norm/reduction/pointer-heavy) with a stable selection procedure."
      blocks: ["G10"]
      plan: ["Q13"]
    - id: "OQ10"
      status: "OPEN"
      statement: "Pin and record the actual toolchain versions (CUDA toolkit, driver, Triton, Python, Nsight Compute, TileIR stack) used for all reported results; define how scripts capture them."
      blocks: ["G8", "reproducibility expectations"]
      plan: ["Q14"]
  closed: []

DELTA_LOG:
  - "Updated profile.current_stage: 1.5 → 2."
  - "Updated scope_lock.research_question_1liner to emphasize spill provenance + attribution as primary story."
  - "Added scope_lock hypotheses H5 (spill cliff shift) and H6 (spill-vs-local attribution calibration) aligned with Stage-2 winner D1."
  - "ARTIFACT_INDEX.stage2_directions set to WP2_20260204."
  - "Added VERDICT_LEDGER V009: provisional winner D1."
  - "EVAL_PLAN updated: added NN5 (MB2 calibration requirement) and refined workload requirement to 5–15 TritonBench ops with TB_Ocategory map."
  - "OPEN_QUESTIONS added: OQ09 (TritonBench op list/category mapping) and OQ10 (toolchain pinning)."

CAPSULE_HEALTH:
  counts:
    verdict_ledger: 9
    claim_ledger: 19
    evidence_ledger: 12
    experiment_ledger: 0
    open_questions_active: 10
    open_questions_closed: 0
  capsule_token_estimate: 6200
  monotonicity: "OK (stage1.5 → stage2 growth)"

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 2.5 prompt."
