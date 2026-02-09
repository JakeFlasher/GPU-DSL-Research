## WORK_PRODUCT

### 1) Table: Axiom-vs-Tooling-vs-Gap Matrix

| Gap_ID | Premise/axiom | What tooling already measures | What is missing | Why reviewers care | Evidence needed | Measurement hook (metric/tool output) | Threat to validity |
|---|---|---|---|---|---|---|---|
| **G1** (elephant #1) | **A/B comparisons are invalid unless you can prove which backend actually produced the measured code** (TileIR vs PTX) *per kernel/run*. | TileIR backend toggle exists (**ENABLE_TILE=1**) (C009/E004,E005). TileIR→PTX fallback can happen on compilation bugs (C012/E005). NCU 2025.4 adds Tile workload profiling / Tile section (C006/E007). | **Deterministic provenance**: per-kernel “backend_used” ground truth, plus a **fail-closed** run policy when fallback/ambiguity occurs. Also: cache interactions (compile artifacts reused across ENABLE_TILE changes) are not yet controlled. | Reviewers will reject any spill/reg delta claim if there’s a plausible “you accidentally measured PTX both times” explanation. This is an **evaluation credibility cliff**. | (1) Demonstrate at least one ambiguous/fallback scenario is possible in practice (even if rare). (2) Show your provenance method detects it. (3) Show runs marked “TileIR” are consistently TileIR across repeats. | **Two required artifacts per run**: (a) compilation provenance artifact (log/artifact hash; exact mechanism TBD), and (b) NCU report metadata sanity signal (e.g., presence/absence of Tile section; *sanity only*, not sole proof) (C006/E007). | Triton cache/JIT reuse can defeat naive A/B toggles. Multi-process compilation races. NCU itself can perturb runtime; provenance must be independent of profiler. |
| **G2** (elephant #2) | **The paper’s core claims require reg+spill metrics on Tile workloads**, not just on “classic” CUDA kernels. | NCU defines spill metrics and SASS spill/local instruction counters (C003–C004/E002) and registers-per-thread metric with interpretation caveats (C005/E003). NCU supports profiling CUDA tile workloads starting 2025.4 (C006/E007). | **Unproven visibility:** whether the specific “must-have” metrics (**launch__registers_per_thread**, **derived__local_spilling_requests(_pct)**, **sass__inst_executed_register_spilling**, **sass__inst_executed_local_loads/stores**) are actually collectable (non-N/A) for TileIR-generated tile workloads on Blackwell in your environment. | If these metrics can’t be collected, the proposal collapses into “runtime only” anecdotes → **feasibility cliff** and **reviewer skepticism** (“no mechanism, only symptoms”). | NCU report (or exported CSV) for **ENABLE_TILE=1** runs showing the exact metric names above present with values, across ≥2 kernels (one spilling, one non-spilling). | NCU metric query output confirming availability + NCU report/CSV containing: `launch__registers_per_thread`, `derived__local_spilling_requests`, `derived__local_spilling_requests_pct`, `sass__inst_executed_register_spilling`, `sass__inst_executed_local_loads`, `sass__inst_executed_local_stores` (C003–C005/E002,E003). | Metrics may be architecture-/mode-restricted; tool may substitute derived estimates; counter collection may require replay and distort runtime comparisons. |
| **G3** (elephant #3) | **If a knob is “critical” (occupancy hint), the paper must be able to sweep it reproducibly** to show tradeoffs and to attribute performance/spills. | TileIR backend describes an **occupancy hint (1–32)** and calls it critical (C015/E005). | **Controllability gap:** exact user-facing API/syntax to set occupancy hint in Triton code/config is currently unknown (OQ01). Also missing: evidence the hint is applied (not ignored) and a method to log the applied value per run. | Without this, you lose the cleanest mechanism-based story (“hint trades occupancy vs spilling”) and a key ablation → **novelty + evaluation strength hit**. | (1) Minimal code snippet that sets occupancy hint. (2) Sweep shows monotonic/structured changes in runtime and spill/reg metrics across hint values. (3) Evidence that the hint value is actually consumed by the backend (not just accepted). | Microbench sweep over hint ∈ [1..32] while collecting: runtime + `launch__registers_per_thread` + spill metrics (`derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling`) (C003–C005/E002,E003). | Hint may be ignored for some kernels, overridden internally, or interact with tile size/unroll such that effects are non-monotonic; caching can mask compile-time changes. |
| **G4** | **Backend A/B must control numerical-mode confounders** if they change codegen (and therefore registers/spills). | TileIR disables approx + FTZ by default; can be enabled via `TILEIR_ENABLE_APPROX=1` and `TILEIR_ENABLE_FTZ=1` (C021/E005). | **Fairness policy missing:** how to ensure PTX vs TileIR comparisons are not dominated by approx/FTZ differences; also missing: whether these toggles materially change register/spill metrics for your kernels. | Reviewers can argue you measured “math mode differences,” not register allocation/spill behavior → attribution undermined. | Show at least one kernel where toggling approx/FTZ changes instruction mix and/or reg/spill metrics; then specify and follow a fixed policy across all experiments. | Controlled ablation on TileIR backend: toggle approx/FTZ and collect runtime + spill/reg metrics (same set as G2). Track env vars in metadata (C021/E005). | PTX backend may not have a matching toggle, so “perfect alignment” may be impossible; numeric drift may invalidate “same semantics” assumption if outputs aren’t checked. |
| **G5** | **Local-memory traffic must be correctly attributed to spilling**, not mistaken for legitimate local arrays/stack use. | NCU explains local memory and that it’s used when variables don’t fit in registers / spilling occurs (C001/E001). NCU provides spill metrics and local load/store instruction counts (C003–C004/E002). | **Explainability gap:** a validated interpretation protocol linking (a) derived spill requests, (b) SASS spill/local instruction counters, and (c) performance—plus microbench design rules that avoid accidental explicit local arrays. | If you can’t convincingly say “this is spilling,” reviewers will treat spill claims as speculative. | Microbench where you intentionally increase live value count (register pressure) while keeping memory access pattern fixed, and show correlated increases in: derived spill requests + local load/store counts + runtime. | Sweep a **register-pressure knob** and record: `launch__registers_per_thread`, `derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling`, `sass__inst_executed_local_loads/stores`, runtime (C003–C005/E002,E003). | Compiler may introduce local stack objects or change scheduling; derived metric may not perfectly match actual traffic; local loads/stores can also come from non-spill local usage. |
| **G6** | **Registers-per-thread is not “live registers.”** Interpretation must reflect holes/ABI constraints. | NCU warns `launch__registers_per_thread` can exceed maximum live regs due to holes/ABI/hardware constraints (C005/E003). | **Attribution gap:** method to separate “real reg pressure” from allocation artifacts when comparing PTX vs TileIR; avoid overclaiming “TileIR uses more regs” when it’s holes. | ISCA/MICRO reviewers will attack simplistic interpretations; you need a defensible measurement story. | Identify cases where reg/thread changes without spill changes, and present a conservative interpretation framework (e.g., rely more on spill counters for pressure-related conclusions). | Joint analysis: treat `launch__registers_per_thread` as context; anchor “pressure” conclusions on spill metrics (`derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling`) + runtime. (C003–C005/E002,E003). | Misclassification risk: you may attribute a runtime delta to reg count while it’s due to memory/coalescing/TMA differences or different instruction selection. |
| **G7** | **If you claim “descriptor/TMA rewrite reduces spilling,” it must be systematic and not a one-off.** | NVIDIA blog shows a TMA/descriptor rewrite example and positions it as relevant to TileIR performance (E004). TileIR backend exists and is Blackwell+CUDA 13.1+ (C009–C010/E004,E005,E008). | **Generalization + controllability missing:** a repeatable recipe to produce “same semantics” kernels in two pointer styles (tensor-of-pointers vs descriptor/TMA), across ≥2 kernels / ≥1 TritonBench op, with version-locked code. | Reviewers want an actionable takeaway; otherwise it’s just “we saw a neat demo once.” | Two implementations per kernel (pointer-style A/B) with output checks + consistent reductions in spill metrics and runtime under TileIR; show whether PTX backend sees the same effect. | For each kernel variant collect: runtime + `launch__registers_per_thread` + spill metrics (G2 set). Run under both ENABLE_TILE=0 and 1. (C003–C005/E002,E003; C009/E004,E005). | Rewrite may change the memory system behavior (not just registers), so performance deltas may not be attributable purely to spilling; output equivalence tests are mandatory. |
| **G8** | **Toolchain drift kills characterization papers** unless you pin versions and store artifacts. | GOLDEN_SOURCES exist (NCU docs, TileIR docs/blog/repos, TritonBench) (E001–E009). | **Reproducibility gap:** pinned versions (CUDA/driver/Triton/NCU), scripted runs, stored raw NCU reports, and a metadata schema (env vars, GPU SKU, backend provenance). | Reproducibility is a primary reviewer axis; “works on my machine” is a reject. | Demonstrate re-running the same suite (same commit, same versions) yields stable deltas (within variance bounds). Provide artifact bundle pointers. | Measurement harness emits: tool versions + env vars + GPU identity + NCU `*.ncu-rep` (or exported CSV) + run logs. (No new metric names required.) | GPU clocks/thermals, driver updates, and compiler nondeterminism can cause drift; NCU overhead can change scheduling and thus spills. |
| **G9** | **Blackwell-only scope requires explicit boundaries and cross-SKU checks**; unknown Blackwell-specific modes (e.g., “2CTA mode MMA”) can confound. | TileIR initial targeting is Blackwell (C010/E004,E005,E008). Missing num_warps noted (C014/E005). GPU inventory lists RTX 5090 / B200 / GB10 but cc unknown (env inventory + OQ06). | **Hardware clarity gap:** confirm compute capability + device properties; decide whether results generalize across RTX 5090 vs B200; and resolve meaning/measurement of “2CTA mode MMA” (OQ05). | Reviewers will ask: “Is this a compiler effect or a SKU/mode effect?” and “Does it hold beyond one card?” | Confirm device properties (cc, SM count, clocks) and replicate key deltas on ≥2 Blackwell GPUs if available. Resolve what “2CTA mode MMA” is and whether it changes reg/spill behavior. | Run the same A/B microbench and collect the same metric set on RTX 5090 and B200; store device metadata alongside NCU reports. (Metric set unchanged.) | Different clocks/memory configs can dominate runtime; architectural modes may change occupancy/resource allocation and indirectly affect spilling. |
| **G10** | **Microbench findings must map to real operators** (TritonBench subset) with a clear motif linkage. | TritonBench provides runnable operators via `python run.py --op <name>` (C020/E006). | **Ecological validity gap:** operator selection that covers motifs (GEMM-like, norm/reduction-heavy, pointer-heavy), and a demonstrated “microbench → operator fix” transfer (e.g., occupancy hint or pointer rewrite reduces spills). | Otherwise reviewers label the work “toy kernels + profiler screenshots.” | At least 1 strong case study: a TritonBench op exhibits the same reg/spill signature as a microbench motif; applying the identified knob reduces spills and improves runtime. | Run selected TritonBench ops under ENABLE_TILE=0/1; profile with NCU collecting the same spill/reg metrics and runtime. (C020/E006 + G2 metric set.) | TritonBench harness overhead; op may not be TileIR-compatible; dynamic shapes/heuristics can change kernel selection across runs. |

---

### 2) Table: Gap → Microbench Motif

| Gap_ID | Minimal microbench motif | Control knobs (tile size, unroll, etc.) | Expected observable delta | Baseline(s) | Confounders |
|---|---|---|---|---|---|
| **G1** | **Backend-provenance probe**: 1–2 tiny kernels compiled/run with ENABLE_TILE=0 vs 1, plus a “stress” kernel likely to hit edge cases (e.g., large reduction dim). | `ENABLE_TILE ∈ {0,1}`; (optionally) reduction dim, unroll, tile sizes to stress compiler. | Provenance artifacts differ between backends; detection method flags any fallback/ambiguity; measured metrics only trusted when provenance is unambiguous. | PTX backend (ENABLE_TILE=0) as known baseline path; TileIR (ENABLE_TILE=1). | Compilation cache reuse; inconsistent JIT; stress kernel might not compile at all (then it’s not a fallback test). |
| **G2** | **Metric-availability probe**: a single kernel compiled with TileIR and profiled. Include one spilling and one non-spilling configuration. | Register-pressure knob (e.g., number of live accumulators / unroll) to induce spilling vs no spilling. | NCU produces non-N/A values for required metrics; spill metrics increase only in spilling config. | Same kernel with low pressure (non-spilling) vs high pressure (spilling). | Some metrics may be unsupported for tile workloads; profiling replay can distort runtime. |
| **G3** | **Occupancy-hint sweep kernel**: GEMM-like or reduction kernel with moderate pressure where tuning is meaningful. | Occupancy hint 1..32 (API TBD); plus tile sizes / stages / unroll; keep problem size fixed. | Tradeoff curve: runtime vs spill metrics; potentially changes in `launch__registers_per_thread` and spill onset. | Default hint (whatever backend chooses) or a mid value (e.g., 16) as reference. | Hint may be ignored; compile caching; interplay with tile sizes can invert trends. |
| **G4** | **Numeric-mode sensitivity kernel**: includes operations likely to switch instruction selection under approx/FTZ (exact op TBD). | `TILEIR_ENABLE_APPROX ∈ {0,1}`, `TILEIR_ENABLE_FTZ ∈ {0,1}` under TileIR; keep inputs fixed and check outputs. | If modes affect codegen, you’ll see runtime and potentially reg/spill deltas; use this to justify a fixed policy. | TileIR defaults (approx off, FTZ off) vs enabled variants. | Numeric differences can break “same semantics” comparisons; PTX backend may not match modes. |
| **G5** | **Live-variable sweep**: arithmetic kernel that keeps many temporaries live to force spills, avoiding explicit local arrays. | #accumulators (K), unroll factor, tile size; keep memory ops constant. | Clear spill onset: `derived__local_spilling_requests_pct` and `sass__inst_executed_register_spilling` rise with K; local load/store counts rise; runtime degrades. | Low-K (no spill) configuration. | Compiler may optimize away temps; register allocation strategy differs by backend and by minor code changes. |
| **G6** | **Holes-vs-pressure sanity motif**: create two configs with similar spill behavior but different `launch__registers_per_thread` (or vice versa) to motivate conservative interpretation. | Small syntactic variants (control-flow, inlining, minor math changes) while holding live-variable intent similar. | Demonstrate non-1:1 mapping between reg/thread and spill; motivates relying on spill counters as primary “pressure” evidence. | Compare variants under same backend; then compare across backends. | Hard to guarantee only “holes” changed; variants may change scheduling/memory. Treat as interpretability aid, not core result. |
| **G7** | **Pointer-style A/B**: same functional kernel written as tensor-of-pointers vs descriptor/TMA (when feasible). | pointer_style ∈ {tensor-of-pointer, descriptor/TMA}; tile sizes; problem size. | Under TileIR: lower `launch__registers_per_thread` and spill metrics for descriptor/TMA, plus runtime improvement (if true). | Tensor-of-pointers as baseline; run also under PTX to test whether effect is TileIR-specific. | Memory system changes confound; ensure output equivalence and comparable alignment/strides. |
| **G8** | **Repeatability harness**: run the same kernel many times, with and without NCU profiling. | warmup count; repeats; (optional) fixed clocks if available; profiling on/off. | Quantified variance; establish “minimum detectable effect” and confidence intervals for deltas. | Unprofiled timing baseline; profiled runs separately reported. | Thermal throttling; background OS noise; NCU overhead changing behavior. |
| **G9** | **Cross-SKU replication**: run the same “best” microbench on RTX 5090 and B200 (if both available). | none besides backend toggle and same knobs; keep inputs fixed. | Same-direction deltas in spill/reg metrics between PTX vs TileIR across SKUs (or explicit statement of SKU-specific behavior). | Within each GPU: ENABLE_TILE=0 vs 1 A/B. | Different SM counts/clocks/memory; must normalize reporting and treat runtime separately from mechanism metrics. |
| **G10** | **TritonBench motif mapping**: pick 2–3 ops representing motifs (GEMM-like, norm/reduction, pointer-heavy) and show microbench-predicted knob helps at least one op. | backend toggle; occupancy hint (if available); pointer rewrite (if applicable); problem sizes. | At least one op shows reduced spill metrics and improved runtime when applying identified knob under TileIR. | PTX backend and/or default TileIR settings as baseline. | Operator may dispatch different kernels across runs; overhead; some ops may not be TileIR-compatible. |

---

### 3) Stage-1 Verdict (≤12 bullets)

- **Top-3 gaps (acceptance-risk dominant):**
  - **G1 (backend provenance / fallback detection)** is the #1 **evaluation credibility cliff**: without per-run backend proof, PTX-vs-TileIR spill claims are not reviewable.
  - **G2 (NCU metric visibility on tile workloads)** is the #1 **feasibility cliff**: if required spill/reg metrics aren’t collectable on TileIR-generated workloads, the proposal loses mechanism-level evaluation.
  - **G3 (occupancy hint controllability)** is the #1 **novelty + strength lever**: the README calls the hint “critical,” but without a sweepable API you can’t deliver the clean tradeoff story reviewers will expect.
- **Secondary but still high-impact gaps:** G4 (numeric-mode confounding) and G5/G6 (spill + reg metric interpretation) are the main ways reviewers can argue your attribution is confounded or overstated.
- **Must-verify Claim_IDs before Stage 2 directions (currently UNVERIFIED):**
  - **C023** (required spill/reg metrics are collectable on Tile workloads under NCU for Blackwell).
  - **C024** (exact Triton user-facing API exists to set TileIR occupancy hint; sweep works).
  - **C025** (robust per-kernel backend provenance method; fail-closed against fallback/caching).
  - **C022** (what “2CTA mode MMA / num_ctas=2” means and whether it affects evaluation; or explicitly scope it out).

---

## CONTEXT_CAPSULE

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
  current_stage: 1
  last_updated: "2026-02-04"
  scope_lock:
    research_question_1liner: "On Blackwell, quantify and attribute PTX-vs-TileIR backend differences in register allocation and spilling using NCU metrics, via controlled sweeps (backend toggle, TileIR occupancy hint, pointer-style rewrite, register-pressure knobs)."
    hypotheses:
      - id: "H1"
        statement: "Backend sweep (ENABLE_TILE=0/1) changes launch__registers_per_thread and spill metrics (derived__local_spilling_requests/_pct, sass__inst_executed_register_spilling) and runtime on same kernel+GPU."
        variables_to_sweep: ["backend: ENABLE_TILE=0 vs 1"]
        metrics_to_observe:
          ["runtime", "launch__registers_per_thread", "derived__local_spilling_requests", "derived__local_spilling_requests_pct", "sass__inst_executed_register_spilling", "sass__inst_executed_local_loads", "sass__inst_executed_local_stores"]
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
      claim: "CUDA local memory is thread-private and used when automatic variables don’t fit in registers or when register spilling occurs."
      status: "VERIFIED"
      evidence: ["E001"]
      paper_role: "A"
      risk_if_wrong: "Misclay spill vs non-spill local traffic."
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
      evidence: ["E007"]
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
      claim: "There is a robust, per-kernel, per-run backend provenance method to guarantee whether TileIR was used or a PTX fallback occurred (including cache interactions), enabling fail-closed A/B experiments."
      status: "UNVERIFIED"
      evidence: []
      paper_role: "A (evaluation credibility)"
      risk_if_wrong: "A/B comparisons can be invalid without detection."

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
      pointer: "Triton-to-tile-IR README: ENABLE_TILE=1, occupancy hint 1–32, missing num_warps, fallback to PTX, approx/FTZ env vars, known issues"
      url: "https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file"
    status: "VERIFIED"
    - id: "E006"
      source_id: "TB"
      kind: "repo_readme"
      pointer: "TritonBench README: python run.py --op gemm"
      url: "https://github.com/meta-pytorch/tritonbench"
      status: "VERIFIED"
    - id: "E007"
      source_id: "NV-NCU"
      kind: "release_note_forum"
      pointer: "Nsight Compute 2025.4 release note: support for profiling CUDA tile workloads + Tile section"
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
  methodology:
    - "Bare-metal runs; warmup; repeat; report median + variability."
    - "Pin toolchain versions + env vars; treat backend toggles as experimental factors."
    - "Use NCU metric query capability to confirm metric availability per version and per workload type (SIMT vs tile)."
    - "Detect TileIR vs PTX fallback explicitly (cache artifacts + logs + report metadata); fail-closed if backend ambiguous."
    - "Record numeric-mode env vars (TILEIR_ENABLE_APPROX/FTZ) and keep them fixed unless performing a dedicated ablation."
  metrics:
    - "runtime"
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
      statement: "Which NCU spill/register metrics are collectable on CUDA tile workloads on Blackwell in NCU 2025.4+."
      blocks: ["C003", "C004", "C006", "C023", "G2"]
      plan: ["Q02", "Q08"]
    - id: "OQ03"
      status: "OPEN"
      statement: "Best-practice backend detection to guarantee no TileIR->PTX fallback in measured runs."
      blocks: ["C012", "C025", "G1"]
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
      statement: "How to extract/persist per-kernel compilation artifacts/provenance from Triton (incl. cache behavior) to support fail-closed backend_used labeling."
      blocks: ["C025", "G1"]
      plan: ["Q11"]
    - id: "OQ08"
      status: "OPEN"
      statement: "Define and validate a numeric-mode control policy for PTX vs TileIR A/B (approx/FTZ), including output-check strategy."
      blocks: ["G4"]
      plan: ["Q12"]
  closed: []

DELTA_LOG:
  - "Set profile.current_stage=1."
  - "ARTIFACT_INDEX.stage1_gap_audit set to WP1_20260204."
  - "Added GAP_LEDGER entries G1..G10 (ranked elephant-first)."
  - "Added VERDICT_LEDGER entries V001..V003 for top gaps (G1..G3)."
  - "Added UNVERIFIED Claim_IDs: C022 (2CTA mode MMA), C023 (NCU metrics on tile workloads), C024 (occupancy hint API), C025 (backend provenance method)."
  - "Updated OPEN_QUESTIONS blocks to reference new gaps/claims; added OQ07 (provenance artifacts) and OQ08 (numeric-mode policy)."
  - "Updated EVAL_PLAN with Stage-1 non-negotiables (backend provenance + tile-metric feasibility check + dual spill corroboration)."

CAPSULE_HEALTH:
  counts:
    verdict_ledger: 3
    claim_ledger: 17
    evidence_ledger: 9
    experiment_ledger: 0
    open_questions_active: 8
    open_questions_closed: 0
  capsule_token_estimate: 4300
  monotonicity: "OK (stage0 → stage1 growth)"

NEXT_STAGE_NT: "Fresh chat; paste capsule + Stage 1.5 prompt."
