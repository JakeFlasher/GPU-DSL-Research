# WORK_PRODUCT

## (1) Core premise verification table (premises that must hold for a MICRO/ISCA/ASPLOS-grade story)

**Legend:**  
- Status: **VERIFIED / INFERENCE / UNVERIFIED** (matches current `CLAIM_LEDGER`)  
- Evidence pointers use existing `EVIDENCE_LEDGER` / `GOLDEN_SOURCES` IDs (no new citations introduced here).

| Premise ID | Premise (tight statement) | Claim_IDs | Status | Evidence pointers | Audit action (what to do next) | If it fails (paper impact) |
|---|---|---:|---|---|---|---|
| P01 | Triton-to-tile-IR backend can be enabled via `ENABLE_TILE=1` to run TileIR codegen vs default PTX backend A/B. | C10 | VERIFIED | E03 / TILE-REPO | None (already basis of A/B). Reconfirm in run manifest logging. | No A/B factor → project collapses. |
| P02 | TileIR compilation may **fall back** to PTX (so naive A/B comparisons can be silently contaminated). | C11 | VERIFIED | E03 / TILE-REPO | Implement fail-closed provenance (X02). | Results become non-attributable → reviewer rejection. |
| P03 | Nsight Compute exposes the core register/spill/occupancy/stack metrics we need (names pinned). | C03–C09, C26 | VERIFIED | E01 / NV-NCU | Lock a minimal metric pack; include `ncu --query-metrics` snapshot in artifacts. | Without these metrics, mechanism attribution weakens; pivot to runtime-only + SASS heuristics (higher risk). |
| P04 | Local memory traffic is **not equal** to spills; we must use spill-specific metrics to identify spilling. | C02, C05, C06, C18 | MIXED (C02/5/6 VERIFIED; C18 INFERENCE) | E01 / NV-NCU | Codify A4 classifier + discordance rules (X04). | Mechanism claims become hand-wavy (“local == spill”) → credibility hit. |
| P05 | PTX `.local` is per-thread private; under ABI it is stack-allocated; accessed via `ld.local`/`st.local` (grounding for spill ↔ stack ↔ local). | C01 | VERIFIED | E02 / NV-PTX | Use as definitional background; avoid over-claiming mapping from PTX to Tile. | Background becomes shaky; still survivable but weaker explanation. |
| P06 | For **Tile-executed kernels**, NCU returns numeric (non-`n/a`) values for non-negotiable metrics (regs/thread, spill metrics, occupancy, stack). | C28 | UNVERIFIED | — | Run X01 (A1 gate) on Blackwell + Tile kernel. | Feasibility cliff: if `n/a`, D1 still possible but becomes mostly runtime + indirect inference; paper bar much harder. |
| P07 | We can export a **per-launch execution-model provenance label** (SIMT vs Tile) or equivalent unambiguous indicator. | C27 | UNVERIFIED | — | Close OQ06 via tooling probe; if absent, implement secondary provenance (OQ02/OQ08). | Credibility cliff: without provenance, main plots must exclude ambiguous points → might shrink dataset drastically. |
| P08 | TileIR backend exposes an occupancy hint knob range (1..32, default 1). | C13 | VERIFIED | E03 / TILE-REPO | Keep as *potential* factor; do not depend on it for the core direction. | If absent in practice, D2 collapses; D1 unaffected. |
| P09 | The occupancy hint knob is actually user-settable via a stable Triton API and is not ignored (measurably affects scheduling/regs/occupancy). | C29 | UNVERIFIED | — | Close OQ03 with minimal two-point probe (X03). | D2 becomes negative result only; keep as limitation/roadmap. |
| P10 | Approx/FTZ modes are disabled by default under TileIR and can be toggled via env vars; math modes must be controlled as confounds. | C14, C20 | MIXED (C14 VERIFIED; C20 INFERENCE) | E03 / TILE-REPO | Add math-mode alignment to manifest + ablation (A7/A8). | Reviewers will attribute differences to math mode, not backend. |
| P11 | `num_ctas=2` is an important control factor for some dense dot workloads on Blackwell; include as an explicit interaction factor (not a hidden “magic speedup”). | C16 | VERIFIED | E03 / TILE-REPO | Treat as confound factor (G8); include as controlled sweep only where relevant. | Without controlling, you can misattribute runtime/spills to backend. |
| P12 | TritonBench can provide a realistic kernel subset (5–15 ops) for ecological validity beyond microbenches. | (NEW) C32 | VERIFIED (per existing evidence note) | E04 / TB | Close OQ13 to avoid invented op names; pin commit; record selection rules (X10). | Without TB subset, reviewers call it “toy microbench only.” |
| P13 | `cuda-tile-translate` / `tileiras` exist as lower-level tooling for Tile IR artifact inspection/harnessing (optional support). | C17 | VERIFIED | E05 / CUDA-TILE-REPO | Optional: only use if it helps provenance/artifact capture. | Not fatal; just reduces introspection options. |

**Core audit conclusion:**  
- **D1** stands or falls primarily on **P06/P07** (NCU metric availability for Tile + provenance signal), which are already your top elephants (G1/G2).  
- **D2/D3** are **second-order** and should be explicitly **conditional** (do not make them the spine).

---

## (2) Competitor / baseline analysis (what exists already, and what your delta must be)

### Baselines / “already doable?” checklist

| ID | Existing tool / approach | Already doable today? | What it gives | What it does *not* give (for TileSpill) | How TileSpill must differentiate |
|---|---|---:|---|---|---|
| B01 | Standard Nsight Compute profiling of CUDA kernels | Yes | Registers/thread, occupancy, local traffic, spill-specific counters (per NCU docs). | **Backend provenance** under ENABLE_TILE=1; Tile-vs-PTX attribution under fallback; curated microbench ladder + TB subset with controls. | D1: provenance-first protocol + coverage/fallback ledger + publishable dataset + scripts. |
| B02 | Compile-time reports (e.g., register count / spills) from toolchain | Partially | Fast static numbers per kernel. | May not apply to TileIR path; compile-time spill count may not match runtime; no occupancy/hardware counters. | Treat as *supporting* signal only; mechanism claims anchored in NCU + runtime. |
| B03 | Static SASS inspection (count local loads/stores) | Yes | A second view of local traffic patterns. | Local != spill; no direct separation of “local arrays” vs “spills”. | Use only as triangulation; primary spill definition via `sass__inst_executed_register_spilling*` + `derived__local_spilling_requests*` (C05/C06). |
| B04 | Triton autotuning / schedule knobs | Yes | Finds fast configs empirically. | Doesn’t explain *why* (regs/spills) or attribute backend differences; doesn’t track fallback contamination. | You provide the attribution layer + fail-closed logging + mechanism tables. |
| B05 | Microbench suites for GPUs | Yes | Isolated motifs can reveal mechanisms. | Existing suites won’t target **TileIR-vs-PTX backend switch** + occupancy hint + fallback coverage. | Your novelty: *TileIR-specific* A/B methodology + controlled knobs + Blackwell focus. |
| B06 | “Just run TritonBench” | Yes | Realistic kernels / operator families. | TB ops can be multi-kernel; profiling can aggregate; no inherent provenance guard. | You contribute deterministic TB subset enumeration + per-kernel mapping + range-mode export plan (OQ07). |

**Competitor audit verdict:** a reviewer can plausibly say “this is just profiling.”  
Your defense must be explicit: **the publishable unit is the methodology + attribution + curated benchmark slice + validated heuristic(s)** under a **new, failure-prone backend boundary** (TileIR + fallback + metric availability).

---

## (3) Per-direction audit table (neighbors, delta, risk scores, “already doable?”)

**Risk scale:** 1 = low risk / strong, 10 = high risk / weak.

| Direction | One-liner | Closest neighbors (tool/practice) | Explicit delta (what’s new) | Already doable with existing tools? | Novelty risk | Feasibility risk | Eval credibility risk | Primary blockers (OQs / Gates) | Fail-soft plan |
|---|---|---|---|---:|---:|---:|---:|---|---|
| **D1** | **Provenance-first** characterization of regs/spills/occupancy for **TileIR vs PTX** on Blackwell, with fail-closed coverage + spill classifier. | Standard NCU profiling + “A/B compile flag” benchmarking. | Turns a fragile A/B into a **publishable measurement protocol**: provenance ledger + fallback accounting + metric availability matrix + spill-vs-local classifier + curated motifs + TB subset. | **Yes (but incomplete)**: tools exist; novelty is in *how you make it defensible* under fallback + n/a metrics. | 6/10 | 4/10 (depends on P06/P07) | 4/10 (drops to 7/10 if provenance/metric gates fail) | OQ02/OQ06/OQ08; GATE-1/2; X01/X02 | If provenance label missing, pivot to “coverage-aware dataset” (exclude ambiguous) + report excluded mass. |
| **D2** | Response surface: sweep **TileIR occupancy hint (1..32)** × backend × `num_ctas` → derive a simple heuristic for spill/occupancy/runtime tradeoff. | Occupancy tuning best practices; autotuning. | TileIR-specific, Blackwell-specific mapping from **occupancy hint → reg/spill/occ** signatures; potentially a validated heuristic. | **Maybe**: only if knob is exposed + not ignored + metrics available for Tile. | 7/10 | 7/10 | 6/10 | OQ03 (knob API), OQ05/OQ06 (metrics + provenance), GATE-3, X03/X08 | If knob isn’t controllable: publish as negative result + remove from main contributions; keep as appendix. |
| **D3** | Addressing-style microbench: **descriptor/TMA-oriented path vs tensor-of-pointer** addressing → reg pressure/spills/runtime A/B (backend-dependent). | Known concept: addressing mode impacts regs; TMA/descriptor lowering discussions. | First systematic, controlled measurement of **addressing-style ↔ spills** under TileIR vs PTX on Blackwell; can explain “why Tile spills” (if seen). | **Maybe**: only if control surface exists and is isolatable. | 7/10 | 8/10 | 7/10 | OQ14 (force addressing style), OQ05/OQ06, X09 | Keep as stretch case study; do not gate acceptance on it. |

**Audit decision:** Keep **D1** as the spine; treat **D2/D3** as *conditional appendices/case studies*.

---

## (4) Strategic recommendation (final direction)

### Recommendation: **Lock D1 as Stage-3 spine** (with conditional D2/D3 add-ons)

**Why D1 wins under MICRO/ISCA/ASPLOS skepticism:**
- Directly addresses your elephants **G1/G2**: metric n/a feasibility + fallback contamination (V03/V04/V12).
- Produces a publishable artifact even if knobs are missing: **a defensible methodology + dataset + negative findings** (e.g., coverage rates, metric availability).
- Provides reviewer-grade credibility hooks: **GATE-1** (metric availability), **GATE-2** (fail-closed provenance), **A5** (NCU overhead ablation), **A7** (toolchain manifests).

**Scope discipline:**  
- Do **not** promise “we tune occupancy hint” unless OQ03 is closed.  
- Do **not** promise addressing-style isolation unless OQ14 is closed.

---

## (5) Stage-3 Assembly Pack (paste-ready; no new claims later)

### SP0 — Working title
**TileSpill: Provenance-First Measurement of Register Spilling in Triton CUDA Tile IR vs PTX on Blackwell GPUs**

---

### SP1 — Abstract (5–7 bullets; claim-safe / no numbers)
- **Motivation:** CUDA Tile IR is an emerging backend for Triton on Blackwell GPUs; changing the backend can change code generation choices and thus **register pressure and spilling**. (C10/C11, E03)
- **Problem:** Simple PTX-vs-TileIR A/B experiments are **not defensible** without (i) proving the backend actually executed (fallback contamination), and (ii) verifying that Nsight Compute returns non-`n/a` spill/occupancy metrics for Tile-executed kernels. (C11/C19/C28, G1/G2)
- **Approach:** We build a **provenance-first measurement pipeline** that is fail-closed per launch: we record backend-used or exclude the datapoint and report coverage/fallback rates. (V07, X02)
- **Mechanism attribution:** We use Nsight Compute’s **spill-specific metrics** and register/stack/occupancy counters, and we explicitly separate “local memory traffic” from “register spilling,” validated via calibration motifs. (C02–C06, C26, X04)
- **Benchmarks:** We evaluate on a curated set of **microbench motifs** (spill onset ladders, local-array calibration, long steady-state kernels) plus a deterministic **TritonBench subset** pinned to a commit. (EVAL_PLAN.motifs, X10)
- **Controls:** We pin toolchain + env vars, separate runtime-only vs NCU runs, quantify profiling overhead, and control confounds such as math modes and relevant Triton hints (`num_ctas`). (V10/V11, C14/C16, X05)
- **Outcome:** A reproducible dataset and methodology that attributes when and how TileIR changes regs/spills/occupancy/runtime versus PTX on Blackwell, including negative results if knobs/metrics are unavailable. (V12, GATE-1/2/3)

---

### SP2 — Contributions (3–5 bullets; must reference Claim_IDs)
- **Contrib-1 (Methodology / credibility):** A **fail-closed provenance ledger** for `ENABLE_TILE ∈ {0,1}` experiments that detects and reports TileIR→PTX fallback contamination; ambiguous datapoints are excluded and counted. (C10, C11, C19; X02; GATE-2)
- **Contrib-2 (Feasibility gate + reproducibility):** A **Metric Availability Matrix Gate** and minimal NCU metric pack for reg/spill/occupancy/stack attribution (including scripted `ncu --query-metrics` capture). (C03–C09, C26; X01; GATE-1)
- **Contrib-3 (Mechanism classifier):** A **spill-vs-local disambiguation classifier** grounded in spill-specific counters and stack metrics, calibrated on motifs designed to trigger local arrays without spilling (and vice versa where possible). (C02, C05, C06, C26, C01; X04)
- **Contrib-4 (Benchmark slice):** An open, pinned microbench suite (K0/K1/K2/K5/K6) plus a deterministic TritonBench subset wrapper with per-op kernel mapping notes and manifest logging. (V11; EVAL_PLAN.motifs; X10)

*(Conditional / only if OQ03/OQ14 close)*  
- **Contrib-5 (Optional heuristic/case study):** An occupancy-hint response surface or addressing-style case study that yields a validated knob heuristic for reducing spills without undue occupancy loss. (C13/C29 and/or C24/C14; X08/X09)

---

### SP3 — Background (what TileIR backend is + why spills matter) **with evidence pointers**
**TileIR backend in Triton (what it is):**
- Triton-to-tile-IR provides a CUDA Tile IR backend; experiments toggle backend via `ENABLE_TILE`. (C10, E03)
- The repo documents both enablement and the possibility of fallback to PTX under bugs, motivating provenance-first evaluation. (C11, E03)
- TileIR adds control surfaces such as an occupancy hint and math-mode env vars (approx/FTZ), which must be treated as confounds/knobs. (C13, C14, E03)

**Why register spills matter (what they do):**
- Spills and other per-thread local state live in PTX `.local` (stack-allocated under ABI) and manifest as local-memory instructions, which can inflate memory traffic and reduce performance. (C01, E02)
- Nsight Compute provides **spill-specific counters** (`sass__inst_executed_register_spilling*`) and derived spill request metrics; local loads/stores alone are insufficient evidence of spilling. (C02, C05, C06, E01)

---

### SP4 — Method (measurement pipeline + controls; fail-closed)
**M0. Toolchain preflight + manifest (mandatory per run directory)**  
- Record: GPU model + clocks if available, driver, CUDA, Triton-to-tile-IR commit, Python, Nsight Compute version, env vars (`ENABLE_TILE`, `TILEIR_ENABLE_APPROX`, `TILEIR_ENABLE_FTZ`, etc.). (V11, EVAL_PLAN.methodology)

**M1. Two-phase runs (separate performance vs mechanism)**
- **Phase A (runtime-only):** CUDA events runtime; warmup + repeats; report median + variability. (V10)
- **Phase B (NCU mechanism):** profile with minimal metric pack; quantify overhead vs Phase A; do not claim runtime from NCU unless overhead accounted. (X05)

**M2. Provenance-first backend-used detection (fail-closed)**
- Primary goal: per-launch backend-used classification; if ambiguous, exclude datapoint and record exclusion reason. (V07, X02)
- Implementation path is **open** until OQ06/OQ02/OQ08 close; do not promise a specific mechanism in the paper without verification. (C27 UNVERIFIED; OQ02/OQ06/OQ08)

**M3. Spill attribution rule-set (classifier)**
- Primary spill signals: `sass__inst_executed_register_spilling*` + `derived__local_spilling_requests*` (C05/C06).
- Triangulation signals: `launch__stack_size`, local loads/stores, regs/thread. (C03, C26, C02)
- Discordance handling: explicit rules (A4) + calibration motif (K2) to prevent “local==spill” errors. (X04)

**M4. Controls / confounds**
- Report and (when possible) align math modes across backends; treat `TILEIR_ENABLE_APPROX/FTZ` as explicit factors. (C14/C20, G8)
- Where relevant, include `num_ctas` interaction tests for dense dot motifs. (C16)

---

### SP5 — Evaluation plan (exact metrics, baselines, workloads, GPU matrix)

**Metrics (exact IDs as currently planned)**
- Runtime: `runtime_us_cuda_events`
- Registers: `launch__registers_per_thread`, `launch__registers_per_thread_allocated` (C03)
- Spills: `sass__inst_executed_register_spilling`, `_mem_local`, `_mem_shared` (C05)  
  + `derived__local_spilling_requests`, `_pct` (C06)
- Local traffic: `sass__inst_executed_local_loads`, `sass__inst_executed_local_stores` (C02)
- Occupancy: `sm__warps_active.avg.pct_of_peak_sustained_active`, `sm__maximum_warps_per_active_cycle_pct` (C07/C08)
- Stack: `launch__stack_size` (C26)
- Provenance label: **UNVERIFIED export path** (C27; OQ06)

**Baselines / ablations**
- Backend A/B: `ENABLE_TILE=0` vs `ENABLE_TILE=1` (C10)
- Provenance accounting: report coverage and fallback/exclusion rates for `ENABLE_TILE=1` (C11, X02)
- Math modes: `TILEIR_ENABLE_APPROX`, `TILEIR_ENABLE_FTZ` as controlled ablations (C14)
- `num_ctas`: 1 vs 2 where relevant (C16)
- Profiler overhead: minimal vs expanded metric packs (X05)
- Conditional:
  - Occupancy hint sweep (1..32) only if OQ03 closes (C29; X08)
  - Addressing-style A/B only if OQ14 closes (X09)

**Workloads**
- Microbench motifs (custom): K0/K1/K2/K5/K6 (EVAL_PLAN.microbench_motifs)
- TritonBench subset: **5–15 ops** selected after deterministic enumeration at pinned commit (OQ13; X10). Use TB-OP placeholders until closed.

**GPU matrix (current inventory; compute capability fields remain TBD until OQ01 closes)**
| GPU | Role | Backend A/B | Status |
|---|---|---:|---|
| RTX 5090 | Primary Blackwell desktop | Yes | Available (cc TBD; OQ01) |
| B200 | Primary datacenter Blackwell | Yes | Available (cc TBD; OQ01) |
| GB10 | Auxiliary Blackwell-family? | Yes (if supported) | Needs SKU/cc confirmation (OQ01) |
| H100 | Cross-arch sanity baseline | PTX baseline | Available; TileIR likely unavailable (as per current notes) |

---

### SP6 — Threats / limitations (explicit)
- **T1 (Provenance contamination):** TileIR→PTX fallback can silently contaminate A/B unless per-launch provenance is fail-closed. (G2, X02)
- **T2 (Metric n/a / partial visibility):** Key spill/occupancy metrics may be `n/a` for Tile-executed kernels depending on NCU/CUDA stack. (G1, X01)
- **T3 (Profiler perturbation):** NCU replay/overhead can change runtime and possibly spilling behavior; runtime claims must come from runtime-only runs. (G5, X05, V10)
- **T4 (Register metric interpretation):** `launch__registers_per_thread` may not equal live registers; interpret alongside stack/spill metrics; avoid over-claiming causality. (G6, C04/C25, OQ12)
- **T5 (Microbench representativeness):** Microbench motifs risk overfitting; mitigate with a TritonBench subset and explicit mapping notes. (G10, X10)
- **T6 (Version drift):** Toolchain/backend behavior can change rapidly; mitigate with pinned versions, manifest logging, and explicit re-run instructions. (G7, V11)

---

### SP7 — Reviewer attack / response set (10 items; paste-ready)
- **RA01 Attack:** “This is just Nsight Compute profiling; no novelty.”  
  **Response:** Novelty is **provenance-first methodology + coverage ledger + metric availability gate + spill classifier + curated benchmark slice** specific to TileIR-vs-PTX under fallback risk. (X01/X02/X04, GATE-1/2)
- **RA02 Attack:** “Your TileIR results might actually be PTX fallback.”  
  **Response:** Fail-closed provenance; ambiguous datapoints excluded and reported; coverage/fallback rates are first-class results. (V07, X02)
- **RA03 Attack:** “Local memory traffic ≠ spilling; your mechanism claims are invalid.”  
  **Response:** We define spills via spill-specific counters and derived spill requests; local loads/stores are only triangulation; classifier validated by K2 calibration motif. (C02/C05/C06, X04)
- **RA04 Attack:** “Profiler overhead invalidates runtime and maybe spilling behavior.”  
  **Response:** Two-phase protocol; runtime-only for performance claims; NCU-only for mechanisms; overhead quantified and disclosed. (V10, X05)
- **RA05 Attack:** “Results are Blackwell-specific and won’t generalize.”  
  **Response:** Scope is explicitly Blackwell microarch characterization; cross-arch sanity on H100 is included for contrast, but we do not claim universality. (scope_lock)
- **RA06 Attack:** “Without an occupancy-hint study, you lack knobs/insights.”  
  **Response:** The core contribution is attribution under backend switch; occupancy hint is explicitly conditional on OQ03 and reported as negative if unavailable. (GATE-3, X03)
- **RA07 Attack:** “Toolchain version drift makes results non-reproducible.”  
  **Response:** Every run directory includes manifest logging (toolchain versions, env vars, GPU props); scripts recreate tables/plots. (V11, A7)
- **RA08 Attack:** “Microbenches are toys.”  
  **Response:** Microbenches are mechanism probes; ecological validity comes from a pinned TritonBench subset with deterministic selection and per-op mapping notes. (X10, OQ13)
- **RA09 Attack:** “Register counts are misleading; you can’t attribute causality.”  
  **Response:** We avoid single-metric attribution; use joint signatures across regs/thread, spill-specific counters, derived spill requests, stack size, and occupancy. (C03–C07, C26)
- **RA10 Attack:** “Your dataset is biased because you exclude ambiguous datapoints.”  
  **Response:** Exclusions are counted and categorized; coverage itself is reported and discussed as a limitation and as a practical engineering reality of the backend. (X02, V07)

---

### SP8 — Source index (IDs only; URLs live in CASE FILE)
- NV-NCU, NV-PTX, TILE-REPO, NV-TILE-BLOG, CUDA-TILE-REPO, TB (see `GOLDEN_SOURCES` in CONTEXT_CAPSULE)

---

# CONTEXT_CAPSULE

```yaml
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

  current_stage: 2.5
  last_updated: "2026-02-04"

  scope_lock:
    research_question_1liner: "On Blackwell GPUs, quantify and attribute how Triton PTX vs Triton CUDA Tile IR backends change register allocation/spilling (and fallback/coverage) and how that impacts runtime/occupancy under controlled tuning knobs, using fail-closed per-launch provenance and validated NCU spill metrics."
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
      - id: "H5"
        statement: "Under ENABLE_TILE=1, a nonzero fraction of launches execute as SIMT due to fallback/mixed execution; fallback/coverage varies by kernel category and must be measured (fail-closed provenance)."
        sweeps: ["backend enablement: ENABLE_TILE=1 across kernel set; classify by per-launch execution model"]
        metrics: ["(UNVERIFIED export path) per-launch execution-model label", "coverage_rate_tile", "fallback_rate_simt"]

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
      - "Report provenance coverage + fallback rate for ENABLE_TILE=1 and count excluded/ambiguous points"

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
      linked_open_questions: ["OQ07", "OQ13"]

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
      verdict: "Feasility elephant: must prove NCU returns numeric spill/reg/occupancy metrics for Tile-executed kernels (metric availability matrix gate)."
      confidence: "high"
    - id: "V04"
      date: "2026-02-04"
      verdict: "Credibility elephant: must implement fail-closed provenance per kernel launch and quantify fallback/coverage; otherwise PTX vs TileIR attribution is not defensible."
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
    - id: "V12"
      date: "2026-02-04"
      verdict: "Stage 2 provisional winner: D1 (Provenance-first spill attribution across execution models) because it remains publishable under feasibility cliffs (metric n/a, occ_hint not exposed, descriptor path unavailable) and directly addresses G1/G2 credibility elephants."
      confidence: "high"
    - id: "V13"
      date: "2026-02-04"
      verdict: "Stage 2.5 audit: Lock D1 as Stage-3 spine (provenance-first A/B + metric availability gate + spill classifier + microbench+TB slice). Treat D2 (occ_hint response surface) and D3 (addressing-style case study) as conditional appendices gated on OQ03/OQ14."
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
      status: "INFERENCE"
      evidence: ["TILE-REPO"]
    - id: "C20"
      scope_tag: "ACTIVE"
      claim: "Must align/rept math modes (approx/FTZ) across backends as confound control."
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
    - id: "C32"
      scope_tag: "ACTIVE"
      claim: "TritonBench can serve as an ecological-validity workload source; a deterministic subset can be selected and pinned by commit."
      status: "VERIFIED"
      evidence: ["TB"]

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
      depends_on: ["OQ07", "OQ13"]
      outputs: ["per-op kernel list", "coverage table", "case-study plot(s)"]
    - id: "X08"
      date: "2026-02-04"
      status: "PLANNED (direction D2 stretch)"
      title: "Occupancy-hint response surface sweep (occ_hint × backend × num_ctas)"
      goal: "Map achieved occupancy vs regs/spills/runtime across occ_hint=1..32; derive/validate a simple heuristic if knob is controllable."
      depends_on: ["OQ03", "OQ05", "OQ06"]
      outputs: ["response surface plots", "heuristic candidate + validation table"]
    - id: "X09"
      date: "2026-02-04"
      status: "PLANNED (direction3 stretch)"
      title: "Addressing-style microbench (descriptor/block-pointer vs tensor-of-pointer) PTX vs TileIR"
      goal: "Isolate addressing-style effect on regs/spills/local traffic; test backend dependence."
      depends_on: ["OQ14", "OQ05", "OQ06"]
      outputs: ["K3 motif code", "A/B plots + mechanism interpretation"]
    - id: "X10"
      date: "2026-02-04"
      status: "PLANNED (hygiene)"
      title: "TritonBench op enumeration + deterministic subset selection"
      goal: "Avoid invented op names; select 5–15 ops by category using repo enumeration at pinned commit; record mapping rules."
      depends_on: ["OQ13"]
      outputs: ["tb_subset.json", "selection_rules.md", "pinned_commit.txt"]

EVAL_PLAN:
  status: "draft (stage2: directions D1/D2/D3; D1 locked at stage2.5)"
  gates:
    - id: "GATE-1"
      name: "Metric Availability Matrix"
      pass_condition: "At least one Tile-executed kernel returns numeric values for the core spill/reg/occupancy metrics; document any n/a."
      artift: "A1_metric_matrix"
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
    - "TritonBench subset must be deterministic + pinned; do not name ops beyond verified ones until enumerated (OQ13)."
  microbench_motifs:
    - id: "K0"
      name: "SmokeTile"
      purpose: "Minimal kernel to establish Tile execution + NCU collection path."
    - id: "K1"
      name: "RegPressureLadder"
      purpose: "Monotonic register-pressure knob sweep; detect spill onset knee."
    - id: "K2"
      name: "LocalArrayVsSpill"
      purpose: "Calibration motif to disambiguate spill-specific metrics vs generic local traffic (local arrays)."
    - id: "K3"
      name: "AddrStyleMotif"
      purpose: "Addressing-style A/B (tensor-of-pointer vs descriptor/block-pointer) to probe register pressure mechanisms when applicable."
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
    - "sm__maximum_warps_per_active_cycle_pct"
    - "PROVENANCE (UNVERIFIED export path): per-launch execution-model label (SIMT vs Tile) (see C27 / OQ06)"
    - "OPTIONAL (UNVERIFIED IDs): shared-spill derived metrics (see C31 / OQ09)"
    - "OPTIONAL (UNVERIFIED ID): Live Registers metric ID (see C25 / OQ12)"
  baselines:
    - "PTX backend (ENABLE_TILE=0) vs TileIR backend (ENABLE_TILE=1) on same Blackwell GPU"
    - "TileIR occupancy hint sweep (occupancy=1..32) (blocked on OQ03; if not possible, report negative result)"
    - "Triton num_ctas sweep (1 vs 2) where relevant"
    - "Descriptor/block-pointer vs tensor-of-pointer addressing style (when applicable; blocked on OQ14)"
    - "NCU minimal-metrics vs expanded-metrics overhead/stability baseline (G5)"
  workloads:
    - "Custom microbench kernels (K0/K1/K2/K3/K5/K6)"
    - "TritonBench subset: 5–15 ops selected deterministically after enumeration at pinned commit (OQ13). Start from verified `--op gemm`; represent remaining as TB-OP02..TB-OP15 placeholders until closed."
  ablations:
    - "TILEIR_ENABLE_APPROX=0/1 (controlled)"
    - "TILEIR_ENABLE_FTZ=0/1 (controlled)"
    - "backend x num_ctas interaction where applicable"
  inclusion_rules:
    "Any datapoint in a main plot must have: (1) pass correctness gate; (2) unambiguous provenance (GATE-2); (3) non-n/a non-negotiable metrics (or is excluded and counted)."
  risks_to_validity:
    - "backend feature drift across CUDA/Triton versions"
    - "register count metric overstates live regs; interpret carefully"
    - "silent TileIR->PTX fallback contaminates A/B comparisons"
    - "clock/frequency variability"
    - "NCU replay/overhead perturbs runtime and possibly spilling behavior"
    - "TritonBench multi-kernel ops complicate per-kernel attribution unless per-instance exports are used"

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: "WP1_20260204"
  stage1_5_toolbox: "WP1_5_20260204"
  stage2_directions: "WP2_20260204"
  stage2_5_audit: "WP2_5_20260204"
  stage3_assembly_pack: "WP2_5_20260204_ASSEMBLY"
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
    - id: "OQ13"
      question: "How do we deterministically enumerate TritonBench ops at a pinned commit and select a 5–15 op subset (without inventing op names), including category coverage and multi-kernel mapping notes?"
      impacts: ["G10", "X07", "X10", "EVAL_PLAN.workloads"]
      query_plan_ref: "Q13"
    - id: "OQ14"
      question: "What is the exact Triton control surface to force descriptor/block-pointer (TMA-oriented) vs tensor-of-pointer addressing style on Bckwell, and does the Triton-to-tile-IR fork preserve/enable it?"
      impacts: ["H4", "X09", "EVAL_PLAN.baselines"]
      query_plan_ref: "Q14"
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
  - "Stage 2: produced 3 research directions (D1/D2/D3) and selected provisional winner D1."
  - "Stage 2: updated scope_lock research question to explicitly include provenance coverage/fallback measurement as first-class output; added hypothesis H5."
  - "Stage 2: expanded EVAL_PLAN motifs with K2 (LocalArrayVsSpill) and K3 (AddrStyleMotif); expanded baselines/workloads language to avoid invented TB op names."
  - "Stage 2: appended experiments X08-X10 (occ_hint response surface; addressing-style motif; TB subset enumeration) and added open questions OQ13-OQ14."
  - "Stage 2: set ARTIFACT_INDEX.stage2_directions=WP2_20260204; updated profile.current_stage=2."
  - "Stage 2.5: performed MICRO/ISCA/ASPLOS-grade audit (neighbors/deltas + risk scoring) and locked D1 as Stage-3 spine; D2/D3 conditional."
  - "Stage 2.5: produced Stage-3 Assembly Pack (paste-ready) and set ARTIFACT_INDEX.stage3_assembly_pack=WP2_5_20260204_ASSEMBLY."
  - "Stage 2.5: added Claim C32 (TritonBench subset as workload source) based on existing evidence E04/TB."
  - "Stage 2.5: updated profile.current_stage=2.5; appended verdict V13."

CAPSULE_HEALTH:
  counts:
    verdicts: 13
    claims: 32
    evidence_items: 7
    experiments: 10
    gaps: 10
    open_questions_active: 13
  capsule_token_estimate: 7000
  notes:
    - "Stage 2.5 audit completed; capsule monotonic (no shrink)."
    - "Stage-3 spine locked: D1 provenance-first A/B + gates."
    - "Assembly Pack produced and pointer stored."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 3 prompt + Assembly Pack."
```
