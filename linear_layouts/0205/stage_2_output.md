## WORK_PRODUCT

### 1) Direction Cards

---

### D1) **Provenance-first spill attribution across execution models (TileIR vs PTX)**
#### Thesis
A fail-closed provenance + metric-validation protocol can make **TileIR vs PTX** spill/occupancy comparisons on Blackwell **reviewable**, and will reveal **nontrivial fallback/coverage** and **distinct spill-onset behavior** that is otherwise invisible.

#### Research question
On Blackwell, **when `ENABLE_TILE=1`**, what fraction of launches actually execute as **Tile vs SIMT (fallback)**, and—restricted to unambiguous provenance—how do **register allocation, stack, and spill metrics** differ between TileIR and PTX across controlled microbench motifs and a TritonBench subset?

#### Hypotheses (H#)
- **H1 (Coverage / fallback is nonzero and workload-dependent).**  
  - **Sweep:** backend toggle `ENABLE_TILE ∈ {0,1}` across kernels; within `ENABLE_TILE=1`, classify launches by execution model (Tile vs SIMT).  
  - **NCU metrics:** *provenance label* (UNVERIFIED export path; see OQ06), plus corroboration via spill/reg metrics availability (Gate-1).  
  - **Expected trend:** Under `ENABLE_TILE=1`, **some launches will be SIMT** (fallback bucket) and the **Tile coverage rate varies** by kernel category (pointer-heavy / reductions worse than GEMM-like).

- **H2 (Spill metrics can be validated and remain discriminative under Tile).**  
  - **Sweep:** microbench knob that induces (i) forced spills vs (ii) explicit local arrays (spill-vs-local disambiguation).  
  - **NCU metrics:** `derived__local_spilling_requests`, `derived__local_spilling_requests_pct`, `sass__inst_executed_register_spilling`, `sass__inst_executed_local_loads`, `sass__inst_executed_local_stores`, `launch__stack_size`.  
  - **Expected trend:** Forced-spill variant increases **spill-specific** metrics (derived + SASS spill) and often `launch__stack_size`; local-array variant increases **local LD/ST** without proportional increase in spill-specific metrics.

- **H3 (Backend changes spill onset and stack behavior in a reg-pressure ladder).**  
  - **Sweep:** reg-pressure ladder knob(s) (e.g., unroll factor / accumulator count / tile size) × backend `ENABLE_TILE ∈ {0,1}`.  
  - **NCU metrics:** `launch__registers_per_thread`, `launch__registers_per_thread_allocated`, `derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling(_mem_local/_mem_shared)`, `launch__stack_size`, `sm__warps_active.avg.pct_of_peak_sustained_active`, runtime (CUDA events).  
  - **Expected trend:** Both backends show a **knee** where spill metrics become nonzero; **knee location differs** (either earlier or later) and correlates with changes in `launch__stack_size` and achieved occupancy.

- **H4 (Mechanism attribution predicts runtime deltas only after provenance gating).**  
  - **Sweep:** backend `ENABLE_TILE ∈ {0,1}`, plus metric-pack size (minimal vs expanded) for overhead/stability.  
  - **NCU metrics:** same as above + `sm__maximum_warps_per_active_cycle_pct` (theoretical occupancy).  
  - **Expected trend:** Without gating, runtime deltas are noisy/misleading; **with fail-closed provenance + stable metric pack**, runtime correlates directionally with spill/stack growth and/or achieved occupancy changes.

#### Closest neighbors + delta type
- **Neighbors:** NCU Profiling Guide (spill/register/local semantics), Triton-to-tile-IR README (fallback hazard, knobs), “Producing wrong data…” (measurement bias), GPU microbenchmarking tradition (mechanism isolation).  
- **Delta type:** **(a) new attribution method** (fail-closed provenance + spill-metric validation protocol applied to an emerging execution model transition).

#### Artifact target(s)
- **Mandatory**
  - **Microbench suite slice**: `K0 SmokeTile`, `K1 RegPressureLadder`, `K2 LocalArrayVsSpill` (new), `K5 SteadyStateLongKernel`.
  - **Measurement scripts**: run harness producing (i) runtime-only, (ii) NCU minimal-pack, (iii) NCU expanded-pack, plus manifest logging.
  - **Analysis report**: “coverage/fallback ledger + metric availability matrix + spill classifier validation + spill-onset plots”.
- **Optional**
  - “Fail-closed provenance ledger library” (small Python module) that merges per-run manifest with NCU exports and excludes ambiguous points.

#### Implementation plan (weeks 1–4; weeks 5–12)
- **Weeks 1–2 (first plots)**
  - Implement **Gate-1 Metric Availability Matrix** (X01) on Blackwell (RTX 5090 and/or B200).
  - Implement **Gate-2 Fail-Closed Provenance/Coverage** (X02) skeleton: per-run manifest + excluded-point accounting.
  - Produce first plots: `K1` spill onset (PTX vs TileIR) **only if** provenance + metrics pass.
- **Weeks 3–4 (case studies)**
  - Implement `K2 LocalArrayVsSpill` and produce a **discordance matrix** (X04).
  - Wrap **5–15 TritonBench ops** with provenance gating (X07), starting with verified `--op gemm` and expanding after deterministic enumeration (OQ13).
- **Weeks 5–12 (stretch)**
  - Broaden kernel categories (reduction/norm/pointer-heavy), add more TB ops.
  - If occupancy hint control is confirmed (OQ03), add an occupancy-hint ablation to every main plot; otherwise record negative result and keep as limitation.

#### Evaluation (metrics, baselines, workloads, methodology, GPU matrix)
- **Metrics (non-negotiable)**
  - Runtime: `runtime_us_cuda_events`
  - Registers: `launch__registers_per_thread`, `launch__registers_per_thread_allocated`
  - Spills: `derived__local_spilling_requests`, `derived__local_spilling_requests_pct`, `sass__inst_executed_register_spilling`, plus `_mem_local`/`_mem_shared`
  - Local traffic: `sass__inst_executed_local_loads`, `sass__inst_executed_local_stores`
  - Occupancy + stack: `sm__warps_active.avg.pct_of_peak_sustained_active`, `sm__maximum_warps_per_active_cycle_pct`, `launch__stack_size`
  - Provenance: per-launch execution model (SIMT vs Tile) **(export path UNVERIFIED; Gate-2 depends on OQ06)**
- **Baselines**
  - **PTX vs TileIR** (`ENABLE_TILE=0/1`)
  - **Occupancy hint sweep** (1..32) *attempted*; if not controllable, negative result is reported (ties to GATE-3)
  - **Descriptor rewrite** when applicable (Direction D3 expands this; here it’s “if available”)
- **Workloads**
  - **TritonBench:** 5–15 ops selected deterministically after enumeration at pinned commit (start with `gemm`; others = TB-OP02..TB-OP15 placeholders until OQ13 closed)
  - **Custom motifs (≥3):** `K0`, `K1`, `K2`, `K5` (plus `K6` as needed for `num_ctas` interaction)
- **Methodology**
  - Warmup + repeats; report median + variability
  - Pin toolchain versions + env vars; isolate caches per run
  - Two-phase measurement: runtime-only vs NCU minimal metric pack; overhead reported (X05)
  - Fail-closed inclusion: correctness gate + provenance gate + non-`n/a` metric gate (excluded points are counted)
- **GPU matrix**
  - **Blackwell (RTX 5090, B200, GB10 if confirmed):** full A/B (`ENABLE_TILE=0/1`) + gates + TB subset.
  - **H100:** PTX-only sanity baselines (reg-pressure ladder shape, metric stability), used to debug harness and demonstrate “not Blackwell-specific measurement artifacts.”

#### Risks + mitigations
- **R1:** Tile-executed kernels return `n/a` for key spill/reg metrics (feasibility cliff).  
  **Mitigation:** Gate-1 early; if `n/a`, pivot to (i) SASS-level spill instruction metrics only, (ii) stack size + local LD/ST + occupancy, and explicitly scope claims to “available metrics”.
- **R2:** Provenance ambiguity (fallback, caching, multi-kernel ops) contaminates comparisons.  
  **Mitigation:** Gate-2 fail-closed policy; per-run cache isolation; per-kernel mapping; explicit fallback/coverage reporting.

---

### D2) **Occupancy-hint × register allocation: a TileIR-specific tuning space**
#### Thesis
TileIR introduces a **new tuning axis (occupancy hint)** that changes the **register/spill/occupancy** trade-off relative to PTX; mapping this axis yields a defensible, empirically validated heuristic for selecting occupancy hints (and `num_ctas=2` where relevant).

#### Research question
For TileIR-executed kernels on Blackwell, how does sweeping **occupancy hint (1..32)** alter **regs/thread, spill metrics, achieved occupancy, and runtime**, and how do these trade-offs differ from PTX (and interact with `num_ctas`)?

#### Hypotheses (H#)
- **H1 (Occupancy hint increases achieved occupancy but induces spill pressure).**  
  - **Sweep:** `occ_hint ∈ {1..32}` under `ENABLE_TILE=1`; hold kernel shape fixed.  
  - **NCU metrics:** `sm__warps_active.avg.pct_of_peak_sustained_active`, `launch__registers_per_thread(_allocated)`, `derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling`, `launch__stack_size`, runtime.  
  - **Expected trend:** As `occ_hint` increases, achieved occupancy increases; allocated regs/thread tends to decrease; spill/stack tends to **increase after a knee**.

- **H2 (Runtime is U-shaped vs occupancy hint; optimal differs by kernel class).**  
  - **Sweep:** `occ_hint ∈ {1..32}` × kernel class (compute-heavy vs memory-bound).  
  - **NCU metrics:** same as H1 + local LD/ST counts.  
  - **Expected trend:** Compute-heavy kernels improve until spill knee then degrade; memory-bound kernels are flatter and may not benefit.

- **H3 (`num_ctas=2` interaction is large for dense dot-like workloads, especially under TileIR).**  
  - **Sweep:** `num_ctas ∈ {1,2}` × `occ_hint` × backend `ENABLE_TILE ∈ {0,1}` on a dense-dot motif and GEMM-like TB ops.  
  - **NCU metrics:** runtime, regs/thread, spill metrics, achieved occupancy.  
  - **Expected trend:** Under TileIR, `num_ctas=2` shifts the optimal `occ_hint` and reduces runtime for dot/GEMM-like workloads; the effect is smaller or different under PTX.

- **H4 (A simple heuristic can be validated).**  
  - **Sweep:** compare heuristic-chosen `occ_hint` vs default `occ_hint=1` across TB subset.  
  - **NCU metrics:** runtime + spill/occupancy metrics.  
  - **Expected trend:** Heuristic improves median runtime without increasing spill metrics beyond a threshold; failures correspond to spill classifier flags.

#### Closest neighbors + delta type
- **Neighbors:** Triton-to-tile-IR README (occupancy hint existence + importance; `num_ctas=2` note), standard occupancy/spill literature, NCU metrics practices.  
- **Delta type:** **(d) a scoped mitigation heuristic/pass that is empirically validated** (occupancy-hint selection heuristic with measured mechanism signatures).

#### Artifact target(s)
- **Mandatory**
  - Microbench: `K1 RegPressureLadder`, plus `K6 ConfoundFactorKernel` (dense dot) for `num_ctas` interaction.
  - Measurement scripts: grid sweep runner (`occ_hint × num_ctas × backend`) + provenance gating.
  - Analysis report: “occupancy-hint response surfaces + heuristic validation on TB subset”.
- **Optional**
  - Tiny autotuner module: chooses `occ_hint` from a small candidate set based on early metric probes (only if overhead manageable).

#### Implementation plan (weeks 1–4; weeks 5–12)
- **Weeks 1–2 (first plots)**
  - Close/attempt **OQ03** (occupancy hint control surface). If not possible: record negative finding and downgrade D2.
  - Run 2-point probe: `occ_hint={1,32}` to check “not ignored” behavior (X03).
- **Weeks 3–4 (case studies)**
  - Full sweep `occ_hint=1..32` on 2–3 kernels (one microbench, one TB op).
  - Add `num_ctas` interaction sweep for dot/GEMM motif.
- **Weeks 5–12 (stretch)**
  - Build/validate heuristic; run across 10–15 TB ops; produce “wins/losses explained by spill/occupancy metrics.”

#### Evaluation (metrics, baselines, workloads, methodology, GPU matrix)
- **Metrics:** same non-negotiable set as D1.
- **Baselines:** PTX vs TileIR; occupancy hint sweep is the main axis; descriptor rewrite when applicable.
- **Workloads:** TB 5–15 ops (needs OQ13); custom motifs: `K0`, `K1`, `K6` (+ `K5` for stability).
- **Methodology:** same gating; plus response-surface plots (occ_hint vs spill/occupancy/runtime).
- **GPU matrix:** Blackwell required for Tile; H100 for PTX-only sanity + measurement overhead checks.

#### Risks + mitigations
- **R1:** Occupancy hint is not user-controllable or is ignored.  
  **Mitigation:** early go/no-go (X03); if negative, reposition as “negative result: knob not exposed” and pivot to D1/D3 contributions.
- **R2:** NCU replay/overhead distorts fine-grained response surfaces.  
  **Mitigation:** longer steady-state kernels (`K5`), minimal metric packs, repeatability checks, and runtime-only confirmation runs.

---

### D3) **Addressing style (descriptor/TMA vs tensor-of-pointer) as a register-pressure lever—backend-dependent**
#### Thesis
The **addressing style** used in Triton kernels (descriptor/block-pointer/TMA-oriented vs tensor-of-pointer) can be a first-order driver of **register allocation and spilling**, and the effect can **flip** between PTX and TileIR backends.

#### Research question
When we switch between (A) tensor-of-pointer style addressing and (B) descriptor/block-pointer style addressing (when available), how do **regs/thread, local traffic, spills, and runtime** change under PTX vs TileIR on Blackwell, and can we isolate the mechanism with microbench motifs and confirm it on TritonBench ops?

#### Hypotheses (H#)
- **H1 (Descriptor-style reduces address register pressure in regular access patterns; benefit is larger under TileIR).**  
  - **Sweep:** addressing style `addr_style ∈ {ptr_tensor, descriptor}` × backend `ENABLE_TILE ∈ {0,1}` on a regular blocked load/store motif.  
  - **NCU metrics:** regs/thread, spill metrics, local LD/ST, runtime, achieved occupancy.  
  - **Expected trend:** Descriptor-style lowers regs/thread and spill metrics; runtime improves; effect is more pronounced under TileIR if it better leverages descriptor lowering.

- **H2 (Irregular/gather patterns still cause reg pressure/spills; descriptor may not help).**  
  - **Sweep:** access pattern regularity (contiguous vs gather/scatter) × `addr_style` × backend.  
  - **NCU metrics:** same as H1 + `launch__stack_size`.  
  - **Expected trend:** Irregular patterns increase regs/thread and local/spill metrics in both backends; descriptor-style does not rescue (or may worsen due to setup overhead).

- **H3 (Interaction with occupancy hint / achieved occupancy).**  
  - **Sweep:** `occ_hint ∈ {1..32}` × `addr_style` × backend (if occ_hint controllable).  
  - **NCU metrics:** achieved occupancy, regs/thread, spills, runtime.  
  - **Expected trend:** When occupancy hint pushes for higher occupancy, pointer-heavy style hits spill knee earlier; descriptor-style shifts the knee.

#### Closest neighbors + delta type
- **Neighbors:** NVIDIA TileIR/Triton blog (descriptor/TMA motivation at a high level), existing Triton block-pointer practices (if applicable), spill/occupancy profiling practices.  
- **Delta type:** **(c) new insight about TileIR vs PTX codegen consequences** (addressing-style as a backend-dependent reg/spill lever).

#### Artifact target(s)
- **Mandatory**
  - Microbench motifs: `K3 AddrStyleMotif` (new: ptr vs descriptor variant), plus `K1 RegPressureLadder` as calibration.
  - Measurement scripts: compile/run both styles; cache isolation; provenance gating.
  - Analysis report: “addressing-style deltas (regs/spills/local traffic/runtime) with backend dependence.”
- **Optional**
  - Small refactoring guide (“if you see signature S, prefer descriptor-style”) grounded in NCU signatures.

#### Implementation plan (weeks 1–4; weeks 5–12)
- **Weeks 1–2 (first plots)**
  - Close **OQ14**: identify the exact Triton code pattern/API that reliably toggles descriptor/block-pointer path on Blackwell and in TileIR backend.
  - Implement `K3` minimal motif and run PTX vs TileIR for one access pattern.
- **Weeks 3–4 (case studies)**
  - Add irregular access pattern variant; run `addr_style × backend` matrix; validate with spill classifier (from D1 tooling).
  - Attempt mapping onto 2–3 TB ops that are pointer-heavy vs blocked (names TBD after OQ13).
- **Weeks 5–12 (stretch)**
  - Expand to more ops; incorporate occupancy hint interactions if OQ03 closes; produce decision rules.

#### Evaluation (metrics, baselines, workloads, methodology, GPU matrix)
- **Metrics:** same non-negotiable set as D1.
- **Baselines:** PTX vs TileIR; occupancy hint sweep attempted; descriptor rewrite is the main axis here.
- **Workloads:** TB 5–15 ops (OQ13); custom motifs: `K1`, `K3`, `K5` (+ `K2` to separate spills from local arrays).
- **Methodology:** same gating and two-phase runs; correctness gate enforced.
- **GPU matrix:** Blackwell for primary claims; H100 for “does addressing-style affect regs/spills in PTX-only too” sanity check.

#### Risks + mitigations
- **R1:** Descriptor/block-pointer control surface is not available or not stable in the TileIR fork.  
  **Mitigation:** treat as “when applicable”; if unavailable, report negative finding and narrow to ptr-style only; keep D1 as backbone.
- **R2:** Confounding changes in instruction mix (not just registers) dominate.  
  **Mitigation:** design motifs so memory traffic and FLOPs are matched; report achieved bytes/FLOPs proxy; use reg-pressure ladder as baseline.

---

### 2) Decision Matrix

| Direction | Novelty(1-5) | Feasibility(1-5) | Eval Credibility(1-5) | Risk(1-5) | Why it wins | Key unknown evidence needed |
|---|---:|---:|---:|---:|---|---|
| **D1** | 4 | **4** | **5** | 2 | Directly attacks the two “elephants”: **metric availability** and **fail-closed provenance**; produces publishable artifacts even if some knobs (occ_hint/descriptor) fail. | OQ05/OQ06: Tile metrics numeric + provenance export path; OQ02/OQ07: robust per-launch mapping; OQ08: artifact hooks in fork. |
| **D2** | 3 | 2 | 3 | **4** | High upside if occ_hint is real + controllable; yields a concrete tuning heuristic. But it can collapse if the knob isn’t exposed or metrics are unstable. | OQ03: occupancy hint API + “not ignored” evidence; OQ05/OQ06: Tile metrics numeric + provenance labeling; `num_ctas` controllability in TB subset. |
| **D3** | 4 | 2 | 4 | 4 | Potentially strong backend-dependent insight, but depends on an **unverified control surface** (descriptor path) and careful motif matching. | OQ14: how to force descriptor/block-pointer path; OQ05/OQ06: Tile metrics numeric + provenance; OQ13: TB ops that exercise the pattern. |

**Provisional winner: D1.**

---

### 3) Stage-2 Verdict (≤12 bullets)

- Pick **D1** as provisional winner: it is the only direction that **still yields a paper-grade result** if occupancy hint and descriptor rewrite both fail.  
- D1 turns “benchmarking hygiene” into a contribution by making **fallback/coverage and metric feasibility** first-class measured outputs.  
- D1’s deliverables (metric matrix + provenance ledger + spill classifier validation) are reusable beyond TileIR and match MICRO/ISCA/ASPLOS skepticism.  
- D2 is retained as a **stretch track** contingent on closing **OQ03** (control surface) quickly; otherwise it becomes a negative result.  
- D3 is retained as a **secondary insight track** contingent on **OQ14** (descriptor control surface); it can plug into D1’s infrastructure once available.  
- Week-1 priority remains unchanged: **Gate-1 (metric availability)** and **Gate-2 (fail-closed provenance)** on a single kernel, on Blackwell.  
- Evaluation plan will treat any ambiguous provenance / `n/a` spill metrics as **excluded and counted**, not hand-waved.  
- TritonBench usage will be **deterministic and pinned**; op names beyond `gemm` are “TBD until enumerated,” avoiding invented specifics.  
- H100 is used for **harness sanity + stability checks**, not for Tile claims.  
- The paper narrative becomes: “**what you can trust** when comparing a new execution model backend, and what the backend switch **actually changes** once you trust it.”

---

## CONTEXT_CAPSULE

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

  current_stage: 2
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
      scope_tag: "TIVE"
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
      depends_on: ["OQ07", "OQ13"]
      outputs: ["per-op kernel list", "coverage table", "case-study plot(s)"]
    - id: "X08"
      date: "2026-02-04"
      status: "PLANNED (direction D2 stretch)"
      title: "Occupancy-hint response surface sweep (occ_hint × backend × num_ctas)"
      goal: "Map achieved oupancy vs regs/spills/runtime across occ_hint=1..32; derive/validate a simple heuristic if knob is controllable."
      depends_on: ["OQ03", "OQ05", "OQ06"]
      outputs: ["response surface plots", "heuristic candidate + validation table"]
    - id: "X09"
      date: "2026-02-04"
      status: "PLANNED (direction D3 stretch)"
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
      outputs: ["tb_subset.json", "selection_rules.md", "pinned_mmit.txt"]

EVAL_PLAN:
  status: "draft (stage2: directions D1/D2/D3; D1 provisional winner)"
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
    - "TritonBench subset: 5–15 ops selected deterministically after emeration at pinned commit (OQ13). Start from verified `--op gemm`; represent remaining as TB-OP02..TB-OP15 placeholders until closed."
  ablations:
    - "TILEIR_ENABLE_APPROX=0/1 (controlled)"
    - "TILEIR_ENABLE_FTZ=0/1 (controlled)"
    - "backend x num_ctas interaction where applicable"
  inclusion_rules:
    - "Any datapoint in a main plot must have: (1) pass correctness gate; (2) unambiguous provenance (GATE-2); (3) non-n/a non-negotiable metrics (or is excluded and counted)."
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
    - id: "OQ13"
      question: "How do we deterministically enumerate TritonBench ops at a pinned commit and select a 5–15 op subset (without inventing op names), including category coverage andulti-kernel mapping notes?"
      impacts: ["G10", "X07", "X10", "EVAL_PLAN.workloads"]
      query_plan_ref: "Q13"
    - id: "OQ14"
      question: "What is the exact Triton control surface to force descriptor/block-pointer (TMA-oriented) vs tensor-of-pointer addressing style on Blackwell, and does the Triton-to-tile-IR fork preserve/enable it?"
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

CAPSULE_HEALTH:
  counts:
    verdicts: 12
    claims: 31
    evidence_items: 7
    experiments: 10
    gaps: 10
    open_questions_active: 13
  capsule_token_estimate: 6200
  notes:
    - "Stage 2 capsule updated; no shrink."
    - "Provisional winner D1: provenance-first spill attribution protocol."
    - "Added motifs K2/K3 and hygiene items to avoid invented TritonBench op names."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 2.5 prompt."
STATE_END
