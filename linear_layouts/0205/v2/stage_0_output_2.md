# WORK_PRODUCT

## 1) Table: Axiom-vs-Tooling-vs-Gap Matrix

> Legend: references like `C00x`/`E00x` point into the provided STATE’s ledgers.

| Gap_ID | Premise / axiom | What tooling already measures (explicit vs STATE/GOLDEN_SOURCES) | What is missing | Why reviewers care | Evidence needed (to prove gap is real) | Measurement hook (metric/tool output + how to collect) | Threat to validity |
|---|---|---|---|---|---|---|---|
| **G1** | **Novelty requires attribution, not “we profiled spills.”** The paper must show a **backend-caused** change in spill onset/pressure on Blackwell (TileIR vs PTX). | **NCU** can report:<br>• registers/thread: `launch__registers_per_thread` (C004/E004)<br>• occupancy limit by regs: `launch__occupancy_limit_registers` (C004/E004)<br>• spill instruction count: `sass__inst_executed_register_spilling_mem_local` (C005/E005)<br>• derived spill requests + pct: `derived__local_spilling_requests`, `derived__local_spilling_requests_pct` (C006/E006)<br>PTX side can emit spill/local-mem warnings via PTX compiler options (C021/E021) passed through Triton `PTXAS_OPTIONS` (C022/E022). | **A mechanistic, controllable experiment** that (a) isolates register-pressure as the independent variable, (b) provides **backend provenance**, and (c) demonstrates a **repeatable** spill-threshold delta attributable to backend differences (not kernel changes). Also missing: compile-time diagnostics parity for TileIR (tileiras) vs ptxas (UNVERIFIED as parity). | Without this, reviewers can dismiss as “NCU already tells you spills” → no novelty. Needs a crisp delta statement: *what* differs, *why*, *how measured*, *how controlled*. | • Show *same high-level motif* produces different $$R$$/thread and spill metrics under `ENABLE_TILE=0` vs `1` on a Blackwell GPU (links to G3).<br>• Demonstrate monotonic pressure ladder and identify **spill onset** (first non-zero spill metric) in each backend.<br>• Demonstrate effect survives repeats and isn’t an artifact of NCU replay (C007/E007). | **Primary**: Nsight Compute 2025.4 collection of (at least) `launch__registers_per_thread`, `launch__occupancy_limit_registers`, `sass__inst_executed_register_spilling_mem_local`, `derived__local_spilling_requests`, `derived__local_spilling_requests_pct` (C004–C006).<br>**Corroboration** (PTX backend only): capture ptxas warning output using options in C021 via `PTXAS_OPTIONS` (C022). | • Backend might also change memory path (e.g., TMA-related rewrites) confounding runtime (C015/E015).<br>• NCU replay/multi-pass can skew small kernels (C007/E007).<br>• Different occupancy/resource limits could drive reg allocation changes (ties to G5). |
| **G2** | **Feasibility cliff:** NCU spill metrics must be **available (non-`n/a`) and stable** on each target Blackwell GPU + driver/toolchain. | NCU 2025.4 defines the needed metrics (C003–C006/E003–E006). | **No proof** they are supported / permission-unblocked / stable on *your exact* Blackwell systems (C029 is UNVERIFIED). | If metrics are `n/a` or unstable, evaluation collapses; paper becomes speculative. | • For each target GPU: confirm metrics return numeric values for a known-spilling kernel and are consistent across reruns.<br>• Record whether any metrics require privileged profiling modes (if applicable). | **Metric feasibility probe:** run a minimal kernel that intentionally spills (e.g., high live-range ladder) and collect the spill metrics listed in G1; check for `n/a`, zeros, or inconsistent values across repeats (ties to C029/OQ002). | • Metrics may depend on driver version, permissions, or NCU replay configuration.<br>• “No spills” could be real or could indicate metric unsupported—must force spilling. |
| **G3** | **Evaluation credibility cliff:** A/B must be **fail-closed** on backend provenance. If you can’t prove which backend compiled the kernel, comparisons are not credible. | TileIR backend enablement via `export ENABLE_TILE=1` is documented (C014/E014, C016/E016). Blog mentions `.tileIR` cache artifacts (C014/E014). | **Per-kernel backend selection** mechanism is UNVERIFIED (C028/OQ001). Also missing: a **robust provenance record** per run (e.g., artifact hashes, cache traces) to prevent silent fallback or cache cross-contamination. | Reviewers will attack: “How do you know those results are TileIR vs PTX and not cache/fallback/toolchain drift?” | • Determine whether backend choice can be controlled per-kernel or only per-process (C028).<br>• Demonstrate that enabling TileIR produces expected artifacts (e.g., `.tileIR` cache) and disabling does not, for the *same* benchmark harness run.<br>• Establish a provenance log schema that is sufficient to reproduce. | **Provenance hook:** record `{ENABLE_TILE value, presence/paths of `.tileIR` cache artifacts, toolchain versions, GPU id}` per run (C014). Enforce “fail-closed”: if artifact/provenance mismatch, discard run. | • Cache artifacts could persist across runs and contaminate provenance unless isolated per-run.<br>• If selection is global-only, A/B within a single process may be impossible (requires process isolation). |
| **G4** | **Controllability gap:** Need a register-pressure control lever that is *symmetric enough* across backends to support causal claims about spill onset. | PTX/CUDA C++ supports `__maxnreg__` and `--maxrregcount` (C020/E020). Triton supports `PTXAS_OPTIONS` to pass options to ptxas (C022/E022). | **TileIR side equivalent** to “cap registers” is not in verified sources. If absent, must rely on code-level pressure only, which weakens causal interpretability. | Without controllability, reviewers can argue your ladder isn’t well-calibrated or comparable; results become anecdotal. | • Verify whether tileiras/TileIR pipeline exposes any register cap / spill-report option (currently unknown).<br>• If not available, show code-level pressure ladder yields predictable changes in `launch__registers_per_thread` and spill metrics anyway. | **PTX backend:** sweep `--maxrregcount` (C020) via `PTXAS_OPTIONS` (C022) and observe `launch__registers_per_thread` + spill metrics (C004–C006).<br>**TileIR backend:** sweep code-level pressure; optionally sweep TileIR occupancy hint (C017/E017) as an indirect lever (ties to G5). | • `--maxrregcount` changes scheduling/perf beyond spills; must measure occupancy/resource limits alongside regs/spills.<br>• Occupancy hint may confound with other compiler heuristics (unknown). |
| **G5** | **Interpretability gap:** TileIR-specific **occupancy hint** must be tied to observable reg allocation + spill behavior in a way reviewers accept. | TileIR backend exposes an occupancy hint knob in range 1..32 (C017/E017). NCU provides occupancy limit by regs (C004/E004). TileIR backend reportedly does not expose `num_warps` in CUDA 13.1, adding occupancy instead (C018/E018). | Missing: mapping from “hint value” → achieved occupancy/resource limits/reg pressure/spills; and a methodology to align occupancy across backends when knobs differ. | Otherwise, occupancy becomes an uncontrolled confound: reviewers can say deltas are occupancy artifacts, not backend spill behavior. | • Show occupancy hint sweep produces measurable, interpretable changes in `launch__occupancy_limit_registers`, `launch__registers_per_thread`, and spill metrics in at least one motif.<br>• Provide controls demonstrating occupancy isn’t limited by something else (e.g., shared memory), or at least report those limits (if available). | Sweep occupancy hint (C017) and collect `launch__registers_per_thread`, `launch__occupancy_limit_registers`, and spill metrics (C004–C006). | • Occupancy limit might be dominated by non-register resources; interpreting “occupancy vs regs” can be wrong if other limits bind.<br>• Backend differences in codegen could shift other resource usage. |
| **G6** | **Confound isolation gap:** Need to separate “spill/local memory effects” from other TileIR-vs-PTX performance deltas (e.g., address/descriptor patterns). | NVIDIA blog flags tensor-of-pointer pattern as suboptimal under TileIR and suggests descriptor/TMA rewrite (C015/E015). | Missing: isolation methodology so that the measured deltas can be attributed to spills rather than to memory access form differences. | If not isolated, reviewer can attribute your result to a known pattern mismatch rather than to spill behavior. | • Demonstrate at least one compute-heavy / memory-light motif where performance correlates with spill metrics (not global memory effects).<br>• Demonstrate at least one memory-touching motif where access pattern is held constant across backends. | Use a compute-heavy ladder motif (few global loads/stores, many live registers). Primary observables: runtime + spill metrics (C005–C006). | • Even “compute-heavy” kernels can change instruction mix and scheduling across backends; runtime differences may not be solely spills.<br>• Without extra memory-traffic metrics (not yet in STATE), attribution is weaker. |
| **G7** | **Measurement credibility gap:** NCU replay/multi-pass and small-kernel variability can invalidate microbench conclusions if not handled. | NCU warns about replay/multi-pass caveats and confounds (C007/E007). | Missing: a standardized *paper-grade* protocol for run duration, warmup/repeats, variance bounds, and reporting. | Reviewers in MICRO/ISCA/ASPLOS will reject “single-run” microbench plots; they expect variance + methodology. | • Show that measured metrics and runtime are stable (e.g., low variance) for representative motifs and that kernels are long enough to avoid extreme replay artifacts.<br>• Provide explicit controls (fixed problem size, repeat count, isolation). | For each configuration: perform repeated runs; report distribution/CI for runtime and key metrics (G1). (Exact NCU CLI flags must be verified locally; do not assume.) | • Thermal/clock variability; background processes; NCU overhead; replay variability. |
| **G8** | **Scope lock gap (framing):** TileIR/CUDA Tile claims must be **Blackwell-only**; cross-arch comparisons must be framed as non-tile baselines. | STATE already locks TileIR claims to Blackwell-only (C008–C010/E008–E010). H100 is listed as baseline-only (environment_inventory). | Missing: explicit guardrails in evaluation so no accidental “TileIR on non-Blackwell” inference creeps in. | Reviewers will punish scope creep or ambiguous architecture claims. | • Confirm target GPUs are actually Blackwell-class on the machines (C030 is UNVERIFIED / OQ003).<br>• Ensure any non-Blackwell runs are clearly labeled “PTX backend only; non-tile baseline.” | Record GPU identity + compute capability on each machine; gate TileIR runs to Blackwell devices only (ties to OQ003). | • Misidentifying SKU/cc causes invalid claims; mixed device runs can pollute plots. |

---

## 2) Table: Gap → Microbench Motif

| Gap_ID | Minimal microbench motif | Control knobs (examples; verify exact API surface in your stack) | Expected observable delta | Baseline(s) | Confounders |
|---|---|---|---|---|---|
| **G1** | **Register-pressure ladder**: compute-heavy kernel with tunable live-value count (e.g., N independent accumulators kept live across loop body). | • live accumulator count `N_live`<br>• unroll factor / loop trip count<br>• operand reuse pattern (to tune live ranges) | As `N_live` increases: `launch__registers_per_thread` increases, then spill metrics transition from 0 → >0; **spill onset threshold** differs between TileIR vs PTX (hypothesis C026). | • PTX backend (`ENABLE_TILE=0`)<br>• TileIR backend (`ENABLE_TILE=1`) | • Occupancy/resource constraints change with code shape (G5).<br>• Backend may change instruction scheduling; ensure motif stays structurally identical. |
| **G2** | **Metric feasibility probe**: a single known-spilling configuration from the ladder (choose a high `N_live`). | • none beyond choosing a spilling config<br>• problem size / duration | Metrics must be non-`n/a` and repeatable across reruns on each GPU. | Same kernel on each target GPU (RTX 5090 / B200 / GB10 once confirmed). | • Kernel too small → replay artifacts (G7).<br>• Not spilling enough → false “unsupported metric” suspicion. |
| **G3** | **Backend provenance harness**: compile+run the same motif twice under different backend settings in isolated runs; capture artifacts + hashes. | • `ENABLE_TILE` value (0/1)<br>• cache directory isolation per run (if configurable) | Presence/absence (and stability) of `.tileIR` artifacts for TileIR path (C014) plus consistent performance/metric signature per backend. | • Separate processes per backend (until C028 resolved). | • Cross-run cache contamination; silent fallback; accidental mixed toolchain versions. |
| **G4** | **Reg-cap sweep (PTX)** + **pressure sweep (TileIR)**: on PTX backend, force spills via explicit register cap; on TileIR, use code-level pressure to reach spill region. | • PTX: `--maxrregcount` via `PTXAS_OPTIONS` (C020,C022)<br>• TileIR: `N_live` ladder (and possibly occupancy hint as indirect lever, C017) | PTX: decreasing max regs should increase spill metrics even at lower `N_live` (sanity check). TileIR: compare whether similar spill region is reachable and how onset differs. | PTX backend without cap; PTX backend with cap. TileIR backend at matched `N_live`. | • Reg cap changes more than spills (occupancy, scheduling).<br>• TileIR may not have a direct reg cap (unknown). |
| **G5** | **Occupancy hint sweep** on a fixed kernel (choose one ladder point near spill onset). | • TileIR occupancy hint 1..32 (C017)<br>• keep code constant | Changes in `launch__occupancy_limit_registers` and possibly `launch__registers_per_thread` and spill metrics; may shift spill onset / magnitude (hypothesis C027). | TileIR backend at default hint vs swept hints; PTX backend as reference. | • Other resource limits may dominate occupancy.<br>• Hint may affect multiple compiler passes; interpret cautiously. |
| **G6** | **Compute-only vs memory-touching pair**: (A) compute-heavy ladder (minimal memory), (B) memory-touching but fixed-access-pattern kernel with tunable compute pressure. | • choose motif A vs B<br>• `N_live` ladder on both | If runtime deltas track spill metrics even in compute-only motif, supports “spills matter” attribution; memory-touching motif checks that conclusions aren’t just memory-path rewrite effects. | PTX vs TileIR for both motifs. | • Memory kernel may still trigger backend-specific transformations (C015). |
| **G7** | **Duration sweep**: add an internal repeat loop to extend kernel duration; reprofile to see if metrics stabilize. | • internal iteration count<br>• total work size | Variance in runtime/metrics should drop as kernel duration increases; exposes replay/multi-pass sensitivity. | Long-duration configuration as “methodology baseline.” | • Too-long kernels risk thermal drift; must watch clocks/temps (tooling outside current STATE). |
| **G8** | **Device identity + gating check**: tiny kernel that logs device attributes out-of-band (host-side) and gates TileIR runs. | • device selection | Prevents accidental TileIR claims on non-Blackwell; ensures plots are labeled by device+toolchain. | N/A (methodological). | • Misreporting GPU info if scripts run in container with restricted visibility; must verify on host. |

---

## 3) Stage-1 Verdict (≤12 bullets)

- **V1 (G1, novelty-critical):** Without a **controlled register-pressure ladder + backend A/B** that yields a crisp “spill onset threshold” comparison, the work degenerates into “NCU profiling” → low novelty risk is **high**.
- **V2 (G2, feasibility cliff):** The entire evaluation hinges on **NCU spill metrics being supported** on the actual Blackwell targets; if they return `n/a` or are unstable, Stage 2 directions must pivot.
- **V3 (G3, credibility cliff):** Backend provenance must be **fail-closed**. If you cannot prove “this run used TileIR” vs “this run used PTX,” reviewers will not trust deltas.
- **Top-3 gaps that dominate acceptance risk:** **G1 (novelty)**, **G2 (feasibility)**, **G3 (credibility)**.
- **Secondary but likely reviewer attack surfaces:** **G4 (control symmetry)** and **G5 (occupancy hint interpretability)**.
- **Must-VERIFY before Stage 2 directions (Claim_IDs):**
  - **C028** (per-kernel backend selection beyond global `ENABLE_TILE`) → determines A/B methodology viability.
  - **C029** (NCU spill metrics availability on each target Blackwell GPU) → feasibility gate.
  - **C030** (confirm target GPUs are Blackwell-class with recorded compute capability) → scope correctness gate.
- **Stage-2 readiness condition (practical):** complete the **metric feasibility probe** (G2) + a minimal **A/B provenance harness** (G3) before proposing 3 directions, otherwise Stage 2 becomes speculative.
- **Kill-switch suggestion:** if C029 fails on the primary Blackwell system, the project must either (a) find alternate supported metrics or (b) narrow claims to what can be measured (no spill attribution claims).

---

# CONTEXT_CAPSULE

```yaml
STATE_VERSION: "manual_state_v4_3_tilespill_blackwell_cuda13_1_lock@2026-02-04"

HARD_VERSION_LOCK:
  cuda_toolkit_min: "13.1"
  ptx_isa_min: "9.1"
  ncu_min: "2025.4"
  tile_arch_scope: "Blackwell-only"
  allowlist_domains:
    - "cookbook.openai.com"
    - "developers.openai.com"
    - "platform.openai.com"
    - "openai.github.io"
    - "docs.nvidia.com"
    - "developer.nvidia.com"
    - "forums.developer.nvidia.com"
    - "github.com/triton-lang"
    - "github.com/NVIDIA"
    - "github.com/meta-pytorch"
  denylist_patterns:
    - "docs.nvidia.com/cuda/archive/12."
    - "ptx-isa-version-8"
    - "medium.com"
  output_audit_banned_strings:
    - "CUDA 12"
    - "PTX 8."
    - "PTX ISA 8"

profile:
  project_name: "TileSpill: TileIR vs PTX register pressure & spilling (Blackwell microbench)"
  target_model: "gpt-5.2-pro (web UI)"
  operating_mode: "manual_state_v4_3_tilespill_blackwell_cuda13_1_lock"
  stage_plan: ["0", "1", "1.5", "2", "2.5", "3"]
  conference_targets: ["ISCA", "MICRO", "ASPLOS"]
  paper_genre: "microarchitecture + toolchain characterization proposal"
  hard_constraints:
    - "No hallucinated citations"
    - "No invented metric names/tool behavior"
    - "Microarch scope lock: bare-metal microbench + NCU + PTX vs TileIR A/B"
    - "Capsule monotonic: ledgers append-only; shrink forbidden unless justified with pointers"
    - "HARD_VERSION_LOCK: CUDA>=13.1, PTX>=9.1, NCU>=2025.4, Blackwell-only for tile claims"
  current_stage: 1
  last_updated: "2026-02-05"

environment_inventory:
  gpus_available:
    - name: "RTX 5090"
      notes: "Planned primary Blackwell-class for TileIR vs PTX A/B; MUST confirm cc/device ID on machine"
      cc: null
    - name: "B200"
      notes: "Datacenter Blackwell; MUST confirm cc/device ID on machine"
      cc: null
    - name: "GB10"
      notes: "SKU/cc unknown; treat as UNVERIFIED until checked"
      cc: null
    - name: "H100"
      notes: "Cross-arch baseline only; NOT for TileIR claims"
      cc: null

  toolchain_to_freeze:
    cuda_version: null
    driver_version: null
    ptx_isa_version: "9.1 (doc-verified; still record actual toolchain mapping)"
    ncu_version: null
    triton_version: null
    python_version: null
    tileir_stack_version: null
    env_vars:
      ENABLE_TILE: "0/1 (record per run)"
      TILEIR_ENABLE_APPROX: "0/1 (TileIR backend; record per run)"
      TILEIR_ENABLE_FTZ: "0/1 (TileIR backend; record per run)"

GOLDEN_SOURCES:
  - id: "OAI-MEM"
    kind: "openai_primary"
    title: "Context Personalization (state-based memory lifecycle + guardrails)"
    url: "https://cookbook.openai.com/examples/agents_sdk/context_personalization"
    last_verified: "2026-02-05"
    version_gate: "N/A (OpenAI doc)"

  - id: "OAI-SESS"
    kind: "openai_primary"
    title: "Session Memory (trimming vs summarization tradeoffs)"
    url: "https://developers.openai.com/cookbook/examples/agents_sdk/session_memory/"
    last_verified: "2026-02-05"
    version_gate: "N/A (OpenAI doc)"

  - id: "CUDA-13.1-RN"
    kind: "nvidia_primary"
    title: "CUDA Toolkit 13.1 Release Notes"
    url: "https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html"
    last_verified: "2026-02-05"
    version_gate: "CUDA==13.1.x required"

  - id: "NV-PTX-9.1"
    kind: "nvidia_primary"
    title: "PTX ISA 9.1 docs"
    url: "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"
    last_verified: "2026-02-05"
    version_gate: "PTX>=9.1 required"

  - id: "NV-NCU-2025.4"
    kind: "nvidia_primary"
    title: "Nsight Compute 2025.4 Profiling Guide"
    url: "https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html"
    last_verified: "2026-02-05"
    version_gate: "NCU>=2025.4 required"

  - id: "NV-TILEIR-MM"
    kind: "nvidia_primary"
    title: "Tile IR spec (memory model) — versioned 13.1"
    url: "https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html"
    last_verified: "2026-02-05"
    version_gate: "Tile IR Spec 13.1 (CUDA>=13.1 context)"

  - id: "NV-TILE-BLOG"
    kind: "nvidia_primary"
    title: "NVIDIA blog (Jan 30, 2026): Triton-to-TileIR backend"
    url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
    last_verified: "2026-02-05"
    version_gate: "CUDA>=13.1 + Blackwell"

  - id: "GPU-OPT"
    kind: "main_academic"
    title: "Optimization Techniques for GPU Programming (OUT_OF_ALLOWLIST; non-citable pointer)"
    url: "https://dl.acm.org/doi/pdf/10.1145/3570638"
    last_verified: null
    version_gate: "OUT_OF_ALLOWLIST (dl.acm.org)"

  - id: "TILE-REPO"
    kind: "repo"
    title: "triton-lang/Triton-to-tile-IR"
    url: "https://github.com/triton-lang/Triton-to-tile-IR"
    last_verified: "2026-02-05"

  - id: "CUDA-TILE-REPO"
    kind: "repo"
    title: "NVIDIA/cuda-tile"
    url: "https://github.com/NVIDIA/cuda-tile"
    last_verified: "2026-02-05"

  - id: "TRITON-REPO"
    kind: "repo"
    title: "triton-lang/triton"
    url: "https://github.com/triton-lang/triton"
    last_verified: "2026-02-05"

  - id: "TB"
    kind: "benchmark_harness"
    title: "meta-pytorch/tritonbench"
    url: "https://github.com/meta-pytorch/tritonbench"
    last_verified: "2026-02-05"

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260205"
  stage1_gap_audit: "WP1_20260205"
  stage1_5_toolbox: null
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
      date: "2026-02-05"
      gap_ids: ["G1"]
      verdict: "Novelty-critical: require controlled backend-attributed spill-onset characterization; otherwise this becomes 'just profiling'."
      rationale: "NCU already exposes registers/thread and spill metrics; contribution must be mechanism-attributed and controlled (not observational)."
      blocking_claim_ids: []
      unblock_actions:
        - "Build register-pressure ladder motif and measure spill-onset threshold under ENABLE_TILE=0 vs ENABLE_TILE=1 with fail-closed provenance."
        - "Report both runtime and spill metrics; show repeatability under NCU replay caveats."

    - id: "V002"
      date: "2026-02-05"
      gap_ids: ["G2"]
      verdict: "Feasibility cliff: verify NCU spill metrics availability/stability on each target Blackwell GPU before Stage 2."
      rationale: "If key spill metrics are n/a/unsupported/unstable, evaluation plan collapses or must pivot to weaker observables."
      blocking_claim_ids: ["C029", "C030"]
      unblock_actions:
        - "Run metric feasibility probe on each target GPU; confirm non-n/a values for spill metrics and repeatability."

    - id: "V003"
      date: "2026-02-05"
      gap_ids: ["G3"]
      verdict: "Credibility cliff: backend provenance must be fail-closed; resolve per-kernel vs per-process backend selection (C028) to design clean A/B."
      rationale: "Without definitive provenance and control, backend comparisons are not credible and are vulnerable to reviewer attacks (cache/fallback)."
      blocking_claim_ids: ["C028"]
      unblock_actions:
        - "Determine concrete mechanism to select backend (per-kernel or per-process only)."
        - "Implement provenance logging (ENABLE_TILE value + artifact presence/hashes) and discard ambiguous runs."

CLAIM_LEDGER:
  items:
    - id: "C001"
      claim: "PTX local state space (.local) is per-thread private memory accessed via ld.local and st.local."
      status: "VERIFIED"
      evidence: ["E001"]
      role: "A: glossary ground truth"
      risk_if_wrong: "low"
    - id: "C002"
      claim: "PTX memory hierarchy includes per-thread local memory, per-CTA shared memory, and global memory accessible to all threads."
      status: "VERIFIED"
      evidence: ["E002"]
      role: "A: glossary ground truth"
      risk_if_wrong: "low"
    - id: "C003"
      claim: "NCU states local memory is used when automatic variables don't fit registers and when register spilling occurs."
      status: "VERIFIED"
      evidence: ["E003"]
      role: "A: spill definition"
      risk_if_wrong: "low"
    - id: "C004"
      claim: "NCU defines launch__registers_per_thread and launch__occupancy_limit_registers as launch/occupancy metrics."
      status: "VERIFIED"
      evidence: ["E004"]
      role: "A: measurement method"
      risk_if_wrong: "low"
    - id: "C005"
      claim: "NCU defines spill instruction metrics including sass__inst_executed_register_spilling_mem_local and local load/store instruction counts."
      status: "VERIFIED"
      evidence: ["E005"]
      role: "A: measurement method"
      risk_if_wrong: "low"
    - id: "C006"
      claim: "NCU defines derived__local_spilling_requests and derived__local_spilling_requests_pct."
      status: "VERIFIED"
      evidence: ["E006"]
      role: "A: measurement cross-check"
      risk_if_wrong: "medium"
    - id: "C007"
      claim: "NCU warns about replay/multi-pass caveats and confounds for small/variable workloads and shared resources."
      status: "VERIFIED"
      evidence: ["E007"]
      role: "A: threats-to-validity seed"
      risk_if_wrong: "medium"
    - id: "C008"
      claim: "CUDA 13.1 introduces CUDA Tile (Tile IR + cuTile) and initial release targets Blackwell GPUs."
      status: "VERIFIED"
      evidence: ["E008"]
      role: "B: scope lock"
      risk_if_wrong: "high"
    - id: "C009"
      claim: "CUDA 13.1 introduces tileiras translating Tile IR bytecode into SASS."
      status: "VERIFIED"
      evidence: ["E009"]
      role: "B: backend pipeline fact"
      risk_if_wrong: "medium"
    - id: "C010"
      claim: "CUDA 13.1 release notes indicate Tile-IR AS compiler currently supports only Blackwell-class devices."
      status: "VERIFIED"
      evidence: ["E010"]
      role: "B: scope lock"
      risk_if_wrong: "high"
    - id: "C011"
      claim: "Tile IR spec has release notes indicating Spec 13.1 dated 2026-01-23."
      status: "VERIFIED"
      evidence: ["E011"]
      role: "B: version gate"
      risk_if_wrong: "medium"
    - id: "C012"
      claim: "Tile IR memory model PTX interoperability states it is intended as a strict weakening of the PTX memory model."
      status: "VERIFIED"
      evidence: ["E012"]
      role: "B: semantics/correctness"
      risk_if_wrong: "medium"
    - id: "C013"
      claim: "NVIDIA blog (2026-01-30) states Triton-to-TileIR requires CUDA>=13.1 and Blackwell GPUs."
      status: "VERIFIED"
      evidence: ["E013"]
      role: "B: scope lock"
      risk_if_wrong: "high"
    - id: "C014"
      claim: "NVIDIA blog shows enabling Tile IR backend with export ENABLE_TILE=1 and discusses .tileIR cache artifacts."
      status: "VERIFIED"
      evidence: ["E014"]
      role: "B: provenance heuristic"
      risk_if_wrong: "medium"
    - id: "C015"
      claim: "NVIDIA blog flags tensor-of-pointer pattern as suboptimal on TileIR backend with CUDA 13.1 and suggests TMA descriptor rewrite."
      status: "VERIFIED"
      evidence: ["E015"]
      role: "C: workload-pattern baseline"
      risk_if_wrong: "medium"
    - id: "C016"
      claim: "Triton-to-TileIR repo documents ENABLE_TILE=1 enabling TileIR backend and claims it uses only CUDA 13.1 features."
      status: "VERIFIED"
      evidence: ["E016"]
      role: "B: implementation fact"
      risk_if_wrong: "high"
    - id: "C017"
      claim: "Triton-to-TileIR repo documents TileIR-specific occupancy hint (1..32) and references num_ctas tuning."
      status: "VERIFIED"
      evidence: ["E017"]
      role: "C: knob set definition"
      risk_if_wrong: "medium"
    - id: "C018"
      claim: "Triton-to-TileIR repo states num_warps not exposed in CUDA 13.1 for TileIR backend and occupancy is added."
      status: "VERIFIED"
      evidence: ["E018"]
      role: "C: sweep constraint"
      risk_if_wrong: "medium"
    - id: "C019"
      claim: "Triton-to-TileIR repo documents TILEIR_ENABLE_APPROX and TILEIR_ENABLE_FTZ env vars (disabled by default)."
      status: "VERIFIED"
      evidence: ["E019"]
      role: "C: sweep constraint"
      risk_if_wrong: "medium"
    - id: "C020"
      claim: "CUDA C++ doc defines __maxnreg__ and --maxrregcount and states __launch_bounds__ and __maxnreg__ cannot both apply."
      status: "VERIFIED"
      evidence: ["E020"]
      role: "C: controlled pressure lever"
      risk_if_wrong: "medium"
    - id: "C021"
      claim: "PTX Compiler API doc lists --warn-on-spills, --warn-on-local-memory-usage, and --verbose options."
      status: "VERIFIED"
      evidence: ["E021"]
      role: "C: compile-time corroboration"
      risk_if_wrong: "medium"
    - id: "C022"
      claim: "Triton repo documents PTXAS_OPTIONS to pass extra options to ptxas."
      status: "VERIFIED"
      evidence: ["E022"]
      role: "C: tooling hook"
      risk_if_wrong: "medium"
    - id: "C023"
      claim: "TritonBench is a collection of PyTorch operators/custom ops and provides a harness to measure performance."
      status: "VERIFIED"
      evidence: ["E023"]
      role: "A/C: workload source"
      risk_if_wrong: "low"
    - id: "C024"
      claim: "PTX docs tie PTX ISA 9.1 to CUDA 13.1 (driver r590), satisfying PTX>=9.1 normative gate."
      status: "VERIFIED"
      evidence: ["E024"]
      role: "A: version gate"
      risk_if_wrong: "high"
    - id: "C025"
      claim: "Combining NCU spill instruction counts and derived spill request metrics yields a robust operational definition of spilling for evaluation."
      status: "INFERENCE"
      evidence: ["E004", "E005", "E006"]
      role: "D: method justification"
      risk_if_wrong: "medium"
    - id: "C026"
      claim: "TileIR vs PTX backends will differ in spill onset thresholds on at least one controlled register-pressure ladder motif on Blackwell."
      status: "INFERENCE"
      evidence: []
      role: "D: novelty hypothesis"
      risk_if_wrong: "high"
    - id: "C027"
      claim: "TileIR occupancy hint will measurably shift register allocation and spill metrics for some compute-intensive kernels."
      status: "INFERENCE"
      evidence: ["E017"]
      role: "D: mechanism hypothesis"
      risk_if_wrong: "high"
    - id: "C028"
      claim: "Per-kernel backend selection mechanism beyond global ENABLE_TILE is implied in the blog but not specified in verified sources."
      status: "UNVERIFIED"
      evidence: ["E013"]
      role: "D: measurement harness requirement"
      risk_if_wrong: "high"
    - id: "C029"
      claim: "NCU spill metrics will be available (non-n/a) on all target Blackwell GPUs under our driver/toolchain."
      status: "UNVERIFIED"
      evidence: ["E005"]
      role: "D: feasibility gate"
      risk_if_wrong: "high"
    - id: "C030"
      claim: "RTX 5090, B200, and GB10 are Blackwell-class and satisfy TileIR backend requirements; must be confirmed on the actual machines."
      status: "UNVERIFIED"
      evidence: []
      role: "D: feasibility gate"
      risk_if_wrong: "high"

EVIDENCE_LEDGER:
  items:
    - id: "E001"
      source_id: "NV-PTX-9.1"
      doc_version: "PTX ISA 9.1"
      accessed_date: "2026-02-05"
      arch_scope: "PTX normative (generic)"
      url: "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"
      notes: "Local state space (.local) definition + ld.local/st.local."
    - id: "E002"
      source_id: "NV-PTX-9.1"
      doc_version: "PTX ISA 9.1"
      accessed_date: "2026-02-05"
      arch_scope: "PTX normative (generic)"
      url: "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"
      notes: "Memory hierarchy description including per-thread local memory."
    - id: "E003"
      source_id: "NV-NCU-2025.4"
      doc_version: "Nsight Compute 2025.4"
      accessed_date: "2026-02-05"
      arch_scope: "Profiler (generic; verify per GPU)"
      url: "https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html"
      notes: "Local memory definition and register spilling mention."
    - id: "E004"
      source_id: "NV-NCU-2025.4"
      doc_version: "Nsight Compute 2025.4"
      accessed_date: "2026-02-05"
      arch_scope: "Profiler metrics (generic; verify per GPU)"
      url: "https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html"
      notes: "launch__registers_per_thread and launch__occupancy_limit_registers and related launch metrics."
    - id: "E005"
      source_id: "NV-NCU-2025.4"
      doc_version: "Nsight Compute 2025.4"
      accessed_date: "2026-02-05"
      arch_scope: "Profiler metrics (generic; verify per GPU)"
      url: "https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html"
      notes: "sass__inst_executed_* local loads/stores and register spilling metrics."
    - id: "E006"
      source_id: "NV-NCU-2025.4"
      doc_version: "Nsight Compute 2025.4"
      accessed_date: "2026-02-05"
      arch_scope: "Profiler metrics (generic; verify per GPU)"
      url: "https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html"
      notes: "derived__local_spilling_requests and derived__local_spilling_requests_pct."
    - id: "E007"
      source_id: "NV-NCU-2025.4"
      doc_version: "Nsight Compute 2025.4"
      accessed_date: "2026-02-05"
      arch_scope: "Profiler methodology"
      url: "https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html"
      notes: "Replay/multi-pass and small-kernel caveats; shared-resource confounds."
    - id: "E008"
      source_id: "CUDA-13.1-RN"
      doc_version: "CUDA Toolkit 13.1.0"
      accessed_date: "2026-02-05"
      arch_scope: "CUDA 13.1 feature scope"
      url: "https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html"
      notes: "CUDA Tile intro and initial Blackwell target."
    - id: "E009"
      source_id: "CUDA-13.1-RN"
      doc_version: "CUDA Toolkit 13.1.0"
      accessed_date: "2026-02-05"
      arch_scope: "CUDA 13.1 feature scope"
      url: "https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html"
      notes: "tileiras existence and role (Tile IR bytecode -> SASS)."
    - id: "E010"
      source_id: "CUDA-13.1-RN"
      doc_version: "CUDA Toolkit 13.1.0"
      accessed_date: "2026-02-05"
      arch_scope: "Blackwell-only for Tile-IR AS (per release notes known issues)"
      url: "https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html"
      notes: "Tile-IR AS compiler Blackwell-only limitation."
    - id: "E011"
      source_id: "NV-TILEIR-MM"
      doc_version: "Tile IR Spec 13.1"
      accessed_date: "2026-02-05"
      arch_scope: "Tile IR spec (versioned)"
      url: "https://docs.nvidia.com/cuda/tile-ir/13.1/sections/release_notes.html"
      notes: "Release notes show Spec 13.1 dated 2026-01-23."
    - id: "E012"
      source_id: "NV-TILEIR-MM"
      doc_version: "Tile IR Spec 13.1"
      accessed_date: "2026-02-05"
      arch_scope: "Tile IR spec (versioned)"
      url: "https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html"
      notes: "PTX interoperability and strict-weakening statement."
    - id: "E013"
      source_id: "NV-TILE-BLOG"
      doc_version: "Blog post dated 2026-01-30"
      accessed_date: "2026-02-05"
      arch_scope: "Blackwell + CUDA>=13.1"
      url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
      notes: "Prereqs: CUDA 13.1+ and Blackwell; backend integration overview."
    - id: "E014"
      source_id: "NV-TILE-BLOG"
      doc_version: "Blog post dated 2026-01-30"
      accessed_date: "2026-02-05"
      arch_scope: "Blackwell + CUDA>=13.1"
      url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
      notes: "ENABLE_TILE=1 enable snippet; .tileIR cache artifacts."
    - id: "E015"
      source_id: "NV-TILE-BLOG"
      doc_version: "Blog post dated 2026-01-30"
      accessed_date: "2026-02-05"
      arch_scope: "Blackwell + CUDA>=13.1"
      url: "https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/"
      notes: "Tensor-of-pointer perf issue and TMA descriptor rewrite direction."
    - id: "E016"
      source_id: "TILE-REPO"
      doc_version: "Repo main (pin by commit in artifacts)"
      accessed_date: "2026-02-05"
      arch_scope: "TileIR backend impl for CUDA 13.1"
      url: "https://github.com/triton-lang/Triton-to-tile-IR"
      notes: "ENABLE_TILE toggle; CUDA 13.1 feature-only claim; Blackwell-only support note."
    - id: "E017"
      source_id: "TILE-REPO"
      doc_version: "Repo main (pin by commit in artifacts)"
      accessed_date: "2026-02-05"
      arch_scope: "TileIR backend impl"
      url: "https://github.com/triton-lang/Triton-to-tile-IR"
      notes: "occupancy hint definition and range; num_ctas note."
    - id: "E018"
      source_id: "TILE-REPO"
      doc_version: "Repo main (pin by commit in artifacts)"
      accessed_date: "2026-02-05"
      arch_scope: "TileIR backend impl"
      url: "https://github.com/triton-lang/Triton-to-tile-IR"
      notes: "num_warps not exposed; occupancy used instead."
    - id: "E019"
      source_id: "TILE-REPO"
      doc_version: "Repo main (pin by commit in artifacts)"
      accessed_date: "2026-02-05"
      arch_scope: "TileIR backend impl"
      url: "https://github.com/triton-lang/Triton-to-tile-IR"
      notes: "TILEIR_ENABLE_APPROX and TILEIR_ENABLE_FTZ env vars."
    - id: "E020"
      source_id: "CUDA-13.1-RN"
      doc_version: "CUDA Toolkit 13.1.0"
      accessed_date: "2026-02-05"
      arch_scope: "CUDA C++ language extension behavior"
      url: "https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/cpp-language-extensions.html"
      notes: "__maxnreg__ and --maxrregcount and incompatibility with __launch_bounds__."
    - id: "E021"
      source_id: "CUDA-13.1-RN"
      doc_version: "CUDA Toolkit 13.1.0"
      accessed_date: "2026-02-05"
      arch_scope: "Compiler option names (CUDA 13.1 archive)"
      url: "https://docs.nvidia.com/cuda/archive/13.1.0/hopper-tuning-guide/ptx-compiler-api/index.html"
      notes: "Option names: --warn-on-spills, --warn-on-local-memory-usage, --verbose."
    - id: "E022"
      source_id: "TRITON-REPO"
      doc_version: "Repo main (pin by commit in artifacts)"
      accessed_date: "2026-02-05"
      arch_scope: "Triton compiler env var surface"
      url: "https://github.com/triton-lang/triton"
      notes: "PTXAS_OPTIONS env var exists for NVIDIA backend."
    - id: "E023"
      source_id: "TB"
      doc_version: "Repo main (pin by commit in artifacts)"
      accessed_date: "2026-02-05"
      arch_scope: "Benchmark harness"
      url: "https://github.com/meta-pytorch/tritonbench"
      notes: "TritonBench description and CLI usage."
    - id: "E024"
      source_id: "NV-PTX-9.1"
      doc_version: "PTX ISA 9.1"
      accessed_date: "2026-02-05"
      arch_scope: "PTX release history"
      url: "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"
      notes: "PTX ISA 9.1 tied to CUDA 13.1 (driver r590) in the PTX doc."

EXPERIMENT_LEDGER:
  items: []

EVAL_PLAN:
  status: "draft"
  non_negotiables:
    - "Per-run backend provenance (fail-closed if ambiguous)"
    - "Metric feasibility probe on tile workloads before large sweeps"
    - "Version pinning: CUDA>=13.1, PTX>=9.1, NCU>=2025.4; record versions in artifacts"
    - "No CUDA<13.1 or PTX<9.1 sources for Tile IR/CUDA Tile claims"
    - "Spill evaluation must include both a direct spill instruction metric (sass__inst_executed_register_spilling_mem_local) and a derived spill request metric (derived__local_spilling_requests / pct) when available."
    - "Every microbench plot comparing backends must include the A/B baselines: ENABLE_TILE=0 vs ENABLE_TILE=1, with provenance artifacts recorded."
    - "Report NCU replay/multi-pass caveats and include repeatability statistics for at least the key ladder motif."

OPEN_QUESTIONS:
  active:
    - id: "OQ001"
      claim_ids: ["C028"]
      question: "What is the concrete per-kernel mechanism/API to select PTX vs TileIR backend within the same process, beyond global ENABLE_TILE?"
      query_plan_refs: ["Q001"]
      status: "active"
    - id: "OQ002"
      claim_ids: ["C029"]
      question: "Are NCU spill metrics (e.g., sass__inst_executed_register_spilling_mem_local, derived__local_spilling_requests) available and stable on each target Blackwell GPU with our driver/toolchain, or do they return n/a / require special permissions?"
      query_plan_refs: ["Q002", "Q006"]
      status: "active"
    - id: "OQ003"
      claim_ids: ["C030"]
      question: "Confirm each target GPU SKU (RTX 5090, B200, GB10) is Blackwell-class and record compute capability + device attributes; GB10 is ambiguous."
      query_plan_refs: ["Q003"]
      status: "active"
    - id: "OQ004"
      claim_ids: []
      question: "Does tileiras / the TileIR compilation pipeline expose compile-time diagnostics analogous to ptxas spill/local-memory warnings (for corroboration), and can Triton surface/capture them?"
      query_plan_refs: ["Q004"]
      status: "active"
  closed: []

DELTA_LOG:
  - id: "DL001"
    date: "2026-02-05"
    changes:
      - "Set profile.current_stage=0; last_updated=2026-02-05; ARTIFACT_INDEX.stage0_fact_sheet=WP0_20260205."
      - "Added Claim Ledger v0: C001–C030."
      - "Added Evidence Ledger entries: E001–E024 (all version-gated where applicable)."
      - "Updated NV-TILEI URL to versioned Tile IR spec 13.1 path for determinism."
      - "Fixed invalid YAML in GOLDEN_SOURCES by normalizing GPU Optimization entry to id=GPU-OPT and labeling OUT_OF_ALLOWLIST."
      - "Seeded OPEN_QUESTIONS: OQ001–OQ003 and mapped QUERY_PLAN Q001–Q008."

  - id: "DL002"
    date: "2026-02-05"
    changes:
      - "Advanced to Stage 1: set profile.current_stage=1; ARTIFACT_INDEX.stage1_gap_audit=WP1_20260205."
      - "Added Stage-1 gap-audit verdicts V001–V003 focusing on: novelty attribuG1), metric feasibility (G2), and backend provenance/control (G3)."
      - "Appended OPEN_QUESTIONS OQ004 (tileiras/TileIR compile-time diagnostics parity) as an evaluation corroboration blocker."
      - "Updated EVAL_PLAN non_negotiables to explicitly require dual spill metrics (direct + derived) when available, A/B baselines, and repeatability reporting."

CAPSULE_HEALTH:
  ledger_counts:
    claims: 30
    evidence: 24
    open_questions_active: 4
    experiments: 0
  approx_tokens: 3500
  notes: "Stage 1 capsule update appended verdicts + one new open question; no shrink."

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 1.5 prompt."
```
