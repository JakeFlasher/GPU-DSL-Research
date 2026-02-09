## 1) Executive Summary (1 paragraph)

The proposal’s core hazard is real and well-documented: Triton-to-TileIR is enabled via `ENABLE_TILE=1`, can *fall back to the NVIDIA PTX backend on compilation bugs*, and Nsight Compute (NCU) 2025.4+ can label each kernel launch’s **execution model** as **SIMT vs Tile** via `launch__execution_model`, making “fail-closed provenance” both necessary and practically achievable (**Problem reality: Green**). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) The novelty is credible but at risk of being perceived as “just good methodology” unless you explicitly elevate (i) *provenance coverage/fallback-rate measurement* and (ii) the *metric-feasibility gate* into first-class contributions (**Novelty: Yellow**). Feasibility is the main knife-edge: NCU supports profiling CUDA tile workloads (driver ≥ 590), but the docs do **not** guarantee that all spill/register mechanism metrics are non‑`n/a` for tile workloads, so your “metric availability matrix” gate is essential (**Feasibility: Yellow (can become Red if metrics are missing)**). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)) Evaluation strength is directionally solid (mechanism metrics + calibrated microbenches + explicit confounds), but it needs stronger hardening around (a) cache isolation, (b) correctness/output checks given TileIR’s unordered-by-default memory semantics, and (c) profiling perturbation/replay artifacts to meet MICRO/ISCA/ASPLOS skepticism (**Evaluation strength: Yellow**). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))

---

## 2) Claim-by-Claim Verification Table

> **Legend**:  
> **Verified** = explicitly supported by primary sources.  
> **Partially Verified** = some elements supported; the “guarantee” portion is not.  
> **Unverified** = not confirmed from available sources; needs an empirical check or deeper code inspection.  
> **Incorrect** = contradicted by sources.

| Claim (verbatim or tightly paraphrased) | Where in proposal | Status | Evidence (citation + short excerpt/interpretation, with version/date context) | Notes (why it matters; caveats; impact) |
|---|---|---|---|---|
| CUDA “local memory” is thread-private and used when automatic variables don’t fit in registers or when register spilling occurs. | Motivation; Background §Spill/reg observables | **Verified** | NCU Profiling Guide v2025.4.1 defines **local memory** as *thread-private* and lists cases including “any variable if the kernel uses more registers than available (register spilling).” ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) | Foundational for your spill-vs-local-traffic framing; reviewers will demand this be grounded. |
| Local memory resides in device memory and has similar performance characteristics/latency to global memory. | Motivation | **Verified** | NCU Profiling Guide v2025.4.1: “local memory space resides in device memory… similar performance characteristics as accesses to global memory.” ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) | Supports why spill→local hurts; but remember L1/L2 caching can modulate “global-like.” |
| NCU defines `derived__local_spilling_requests` and `_pct` as spill-to-local observables. | Abstract; Method; Metrics | **Verified** | NCU Profiling Guide v2025.4.1: `derived__local_spilling_requests` = requests to L1 for register spilling to local; `_pct` = percent of local requests due to spilling. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) | Good choice; ties “spills” to something closer to root cause than generic local LD/ST counts. |
| NCU provides corroborating SASS counters: `sass__inst_executed_register_spilling` and local LD/ST instruction counts. | Abstract; Hypotheses; Evaluation outputs | **Verified** | NCU Profiling Guide v2025.4.1 lists `sass__inst_executed_local_loads/stores` and `sass__inst_executed_register_spilling` (+ mem_local/mem_shared variants). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) | Strong triangulation angle: derived requests vs executed spill instructions. Also consider collecting *shared* spilling metrics too. |
| `launch__registers_per_thread` can exceed maximum live registers due to allocation holes and ABI/instruction constraints; treat as necessary-not-sufficient. | Background §Spill/reg caveats; Controls | **Verified** | Nsight Compute docs (2022.4) explicitly warn that `launch__registers_per_thread` can be “significantly higher” than max live due to “holes… ABI restrictions… particular hardware instructions.” ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html)) | Critical to prevent over-claiming “liveness” from allocation. You should bake this caveat into the interpretation guide and plots. |
| NCU 2025.4 added support for profiling CUDA tile workloads and introduced a Tile section. | Background §NCU support | **Verified** | Nsight Compute Release Notes / Updates in 2025.4: “Added support for profiling CUDA tile workloads” and “Introduced a new Tile section…” ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html)) | Establishes feasibility of using NCU for Tile kernels at all. Still doesn’t guarantee every metric works. |
| CUDA Tile profiling is supported with driver versions **590+**. | Background §NCU support | **Verified** | Nsight Compute Release Notes include: “CUDA Tile profiling is supported with driver versions 590 and above.” ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)) | This can become a silent blocker; must be in your “toolchain pinning” checklist. |
| NCU exposes `launch__execution_model` that reports SIMT vs Tile; and `launch__*` metrics are per kernel launch / instanced per launch for ranges. | Provenance protocol; Non-negotiables | **Verified** | NCU Profiling Guide v2025.4.1: `launch__execution_model` = “SIMT or Tile” and `launch__*` metrics are collected “per kernel launch”; range results are instanced per launch. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) | This is your strongest “authoritative provenance” primitive. Important nuance: for range profiling, ensure you export the **per-instance** values, not only aggregates. |
| CUDA 13.1 introduces CUDA Tile (Tile IR + `cuTile`) and the initial release targets Blackwell GPUs. | Background §TileIR basics; toolchain prereqs | **Verified** | CUDA Toolkit 13.1 Release Notes (New Features) explicitly: “CUDA 13.1 introduces CUDA Tile…” and “initial release targets Blackwell GPUs.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) | Validates the “Blackwell + CUDA 13.1+” constraint as a platform requirement. |
| Triton-to-TileIR requires CUDA 13.1+ and Blackwell GPUs (initially). | Background; Scope | **Verified** | NVIDIA blog (Jan 30, 2026) lists prerequisites: “CUDA 13.1 or higher” and “NVIDIA Blackwell GPUs.” ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Matches CUDA 13.1 notes; good to cite both in camera-ready. |
| TileIR backend enablement uses `ENABLE_TILE=1`. | Background §TileIR toggles; Method factors | **Verified** | Triton-to-tile-IR repo README: “enable… by setting `ENABLE_TILE=1`.” NVIDIA blog uses the same toggle. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Verified. Make sure your scripts log it and isolate caches across changes. |
| When Tile IR backend is active, Triton cache artifacts use `.tileIR` extensions vs `.cubin` for SIMT. | Provenance protocol | **Verified** | NVIDIA blog: “When the Tile IR backend is active, Triton caches compiled kernels with `.tileIR` file extensions instead of… `.cubin`… SIMT backend.” ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Useful as *corroboration*, not a sole oracle: cache reuse and mixed-mode processes can still bite. |
| TileIR can fall back to PTX backend on compilation bugs; experiments must detect fallback. | Motivation; Threats to validity | **Verified** | Triton-to-tile-IR README: “When a compilation bug occurs… it falls back to the NVIDIA PTX backend.” ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | This is the central provenance hazard; absolutely worth foregrounding. |
| TileIR disables approx and FTZ by default; can be enabled with `TILEIR_ENABLE_APPROX=1` / `TILEIR_ENABLE_FTZ=1`. | Background §TileIR toggles; Controls | **Verified** | Triton-to-tile-IR README explicitly states approx/FTZ disabled by default and gives env vars. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Strong confound-control handle. Still need **output checking** to ensure toggles don’t change correctness. |
| TileIR introduces an **occupancy hint** taking N=1..32 (default 1) and is “critical.” | Hypotheses (H2); Background | **Verified (existence)** | Repo README: occupancy hint 1–32, default 1, “critical,” expects N active thread blocks per SM. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | The hint exists, but how users set/sweep it from Triton Python is not shown in README → see next row. |
| There is a stable, user-accessible API to set/sweep TileIR occupancy hint from Triton code, and it is applied (not ignored). | Hypotheses H2; Feasibility | **Unverified** | README asserts the hint exists and is tunable, but does not document the *user-level control surface*. We did not find an explicit API description in accessible sources. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | This is correctly flagged “UNVERIFIED” in your state. De-risk by locating the control in code/docs and proving it affects `launch__` occupancy limits or runtime. |
| `num_ctas=2` enables “2CTA mode MMA” on Blackwell for dense dot-like workloads; confounds backend comparisons. | Method factors; MB4 | **Verified** | Repo README: “Setting `num_ctas=2` is critical… enables 2CTA mode MMA on Blackwell architecture.” ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Strongly supports MB4 as an attribution control. You should treat it as a separate experimental factor in plots. |
| Triton has dump/override hooks (`TRITON_KERNEL_DUMP`, `TRITON_DUMP_DIR`, `TRITON_KERNEL_OVERRIDE`, `TRITON_OVERRIDE_DIR`, `TRITON_ALWAYS_COMPILE`) usable for artifact capture/provenance. | Provenance protocol; Artifacts | **Verified (in upstream Triton)** | Triton repo README lists these env vars and “Kernel Override Steps,” including `TRITON_ALWAYS_COMPILE=1`. ([github.com](https://github.com/triton-lang/triton)) | Verified upstream. **Open question**: ensure the Triton-to-tile-IR fork preserves/extends these semantics for Tile artifacts (possible drift). |
| TritonBench supports `python run.py --op <name>` entrypoint. | TritonBench anchors | **Verified** | TritonBench README “Basic Usage”: “python run.py --op gemm.” ([github.com](https://github.com/meta-pytorch/tritonbench)) | Verified; good anchor for ecological validity. Add provenance capture around it (since TB itself doesn’t guarantee backend provenance). |
| NCU spill/register metrics listed as “non-negotiable outputs” are collectable (non‑`n/a`) for Tile workloads on Blackwell under NCU 2025.4+. | Feasibility gate; Non-negotiables | **Unverified** | Release notes guarantee *tile workload profiling support* but do not guarantee that specific spill metrics are supported/meaningful for tile workloads. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)) | This is your biggest feasibility cliff. The proposed “metric availability matrix” is the right early gate. |
| “Fail-closed provenance” using `launch__execution_model` + artifact corroboration is sufficient to make PTX-vs-TileIR comparisons reviewable. | Abstract; Provenance protocol | **Partially Verified** | `launch__execution_model` is well-defined (SIMT vs Tile). `.tileIR` cache artifacts exist. But “sufficient” depends on handling failure modes (cache reuse, mixed kernels, range aggregation). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) | I agree with the direction; harden with explicit per-run cache isolation + per-launch exports + exclusion accounting (“coverage rate”). |

---

## 3) State-of-the-Art Analysis (Competitors / Prior Art)

### 3.1 Official docs/tools (NCU, CUDA Toolkit release notes, CUDA Tile / Tile IR docs): normative guarantees vs tool behavior

**Nsight Compute (NCU) — what it normatively guarantees**
- **Execution-model provenance exists as a first-class metric.** NCU v2025.4.1 defines `launch__execution_model` as “SIMT or Tile” and states that `launch__*` metrics are **per kernel launch** and instanced per launch in range mode. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  *Implication:* your “authoritative per-launch label” is not folklore—it is explicitly described by the tool.
- **Local memory semantics and spill causality are documented.** The Profiling Guide explicitly ties local memory traffic to (a) compiler placement of some automatic variables (arrays/large structs, non-constant indexing) and (b) “register spilling” when register demand exceeds availability, and it explicitly states local memory lives in device memory with global-like characteristics. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  *Implication:* you can cite NCU rather than re-explain local memory from scratch.
- **Spill metrics are explicitly defined.** NCU provides both derived spill request metrics (`derived__local_spilling_requests`, `_pct`) and SASS-executed spill instruction counts (`sass__inst_executed_register_spilling` and its variants), plus generic local LD/ST instruction counts. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  *Implication:* your triangulation plan is aligned with tool-provided intent.
- **Replay and multi-pass collection can produce misleading values if kernels aren’t stable across passes.** NCU warns that out-of-range metrics can happen when replay passes see different work distribution; it suggests increasing workload duration and/or collecting fewer metrics together. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  *Implication:* you should treat NCU metric collection itself as a confound and adopt a minimal-metrics protocol for the feasibility gate.

**Nsight Compute 2025.4 Tile profiling support — what it guarantees (and what it doesn’t)**
- Release notes state NCU 2025.4 added **support for profiling CUDA tile workloads** and introduced a **Tile section**; they also specify a minimum driver (590+) for CUDA Tile profiling. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html))  
- However, the release notes do **not** promise that *every* metric you care about is supported for tile workloads—only that tile workloads can be profiled and that a Tile summary section exists. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html))  
  *Implication:* your “metric availability matrix” gate is not overkill; it’s the correct interpretation of what’s actually promised.

**CUDA Toolkit 13.1 release notes — what it normatively guarantees**
- CUDA 13.1 introduces **CUDA Tile** (Tile IR + `cuTile`), notes that it includes pipeline/fatbin/driver updates, and explicitly states the **initial release targets Blackwell GPUs**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html))  
  *Implication:* the “Blackwell + CUDA 13.1+” prerequisite is not a blog claim; it’s in release notes.

**CUDA Tile IR docs — what it normatively says about ordering**
- Tile IR documentation describes memory operations as **token-ordered** and says ordering between memory ops is undefined unless tokens establish order; this matches the “unordered by default” theme in Triton-to-tile-IR’s README. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html))  
  *Implication:* correctness checking is not optional. If your kernels rely on ordering/alias assumptions, TileIR can change semantics unless tokens are inserted. This is a *major hidden confound* missing from the current proposal text (see recommendations).

### 3.2 NVIDIA/Triton ecosystem (Triton-to-TileIR, TritonBench, cuda-tile): what is explicit today vs implicit; documented hazards/limitations

**Triton-to-tile-IR repo (industrial “ground truth” for hazards)**
- **Enablement & fallback:** `ENABLE_TILE=1` enables TileIR; on compilation bug it **falls back to NVIDIA PTX backend**. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- **Known performance limitations & missing knobs:** README states `num_warps` is not exposed yet and suggests some “XXXNorm” kernels may degrade due to register spilling; it introduces an `occupancy` hint (1–32, default 1) and emphasizes `num_ctas=2` enabling “2CTA mode MMA” on Blackwell. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- **Numeric-mode knobs:** approx and FTZ are disabled by default and controllable by env vars. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- **Memory model hazard:** it states CUDA Tile IR “supports only an unordered memory model” and calls out cases that may produce incorrect results (aliasing, cross-block reductions) unless scripts are updated. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
  *Takeaway:* You already have a strong “hazard dossier” from the repo. Your proposal currently emphasizes fallback and numeric mode, but it under-weights the **unordered memory model** correctness risk, which can ruin measurement attribution if not controlled.

**NVIDIA blog (Jan 30, 2026) — explicit provenance hints**
- The blog explains a practical **artifact fingerprint**: when TileIR is active, Triton caches compiled kernels with `.tileIR` extensions instead of `.cubin`. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
  *Takeaway:* This is excellent corroboration for runtime-only sweeps, but you still need cache isolation and per-kernel mapping to prevent false provenance.

**cuda-tile repo — competitor baseline / ecosystem context**
- NVIDIA’s `cuda-tile` repo positions CUDA Tile IR as an MLIR-based IR + compiler ecosystem aligned with CUDA 13.1 and describes the toolchain around bytecode and `tileiras`. ([github.com](https://github.com/NVIDIA/cuda-tile))  
  *Takeaway:* This provides a plausible long-term path for *artifact-level provenance* (tile bytecode → tileiras → cubin) analogous to PTX→ptxas, and it gives context for why execution model changes can affect register allocation and spilling.

**Triton (upstream) — artifact capture hooks**
- Upstream Triton documents environment variables for dumping and overriding compilation stages (`TRITON_KERNEL_DUMP`, `TRITON_DUMP_DIR`, `TRITON_KERNEL_OVERRIDE`, `TRITON_OVERRIDE_DIR`, `TRITON_ALWAYS_COMPILE`). ([github.com](https://github.com/triton-lang/triton))  
  *Takeaway:* These can underwrite an auditable compilation-artifact trail—but you must verify they still work (or are adapted) in the Triton-to-tile-IR fork and that they capture TileIR artifacts (not just PTX/cubin).

**TritonBench — workload anchor**
- TritonBench provides a standardized operator benchmark harness invoked via `python run.py --op <name>`. ([github.com](https://github.com/meta-pytorch/tritonbench))  
  *Takeaway:* Great ecological validity anchor, but by itself it does **not** solve provenance (multi-kernel ops, mixed backends); you must wrap TB with your provenance/metrics harness.

### 3.3 Academic prior art: closest conceptual matches

Here’s the “closest match” landscape relative to your *methods* (not necessarily your exact TileIR-vs-PTX question):

**(A) Register pressure / spilling tradeoffs and alternatives**
- **RegDem (2019)**: shows that **register pressure limits occupancy**, and that spilling policy/location matters (they spill to shared memory via SASS translation). ([arxiv.org](https://arxiv.org/abs/1907.02894))  
  *Relevance:* Helps you justify why “spills” are a mechanism-level performance driver and why comparing backends’ register allocation is meaningf*(B) Microbenchmarking for mechanism discovery**
- **Wong et al., ISPASS 2010 (“Demystifying GPU microarchitecture through microbenchmarking”)** is a classic template for designing microbenchmarks to isolate hidden architectural behaviors, and it ships benchmark code. ([stuffedcow.net](https://www.stuffedcow.net/research/cudabmk))  
  *Relevance:* Supports your microbenchmark-first stance, and provides precedent that “carefully designed microbenches + counters” is publishable work.

**(C) Reproducibility/provenance and measurement bias**
- **Mytkowicz et al., ASPLOS 2009 (“Producing wrong data…”)**: demonstrates that seemingly irrelevant experimental setup changes can introduce large measurement bias; proposes detection/avoidance strategies. ([research.ibm.com](https://research.ibm.com/publications/producing-wrong-data-without-doing-anything-obviously-wrong))  
  *Relevance:* This is strong rhetorical ammunition for “fail-closed provenance” and for logging+randomization, especially when toolchain fallback and caching can silently change the executed code.
- **van der Kouwe et al., 2018 (“Benchmarking Crimes…”)**: provides a taxonomy of benchmarking mistakes and emphasizes reproducibility/comparability failures. ([arxiv.org](https://arxiv.org/abs/1801.02381))  
  *Relevance:* Helps you justify a confound register and explicit exclusion criteria.

**(D) Register file focus (less direct, but supports importance)**
- **GREENER (2017)** addresses compile-time analysis of register usage for energy, showing how hard register behavior is to infer without tools and analysis. ([arxiv.org](https://arxiv.org/abs/1709.04697))  
  *Relevance:* Not about spilling attribution directly, but supports the broader thesis that register behavior is a first-order concern requiring careful methodology.

---

### 3.4 Metric interpretation guide (what each observable means, how to triangulate, and what disagreements imply)

This is the “paper-ready” interpretation guide you should include (or closely mirror) in your methodology section.

#### Provenance metric
- **`launch__execution_model`**: NCU-defined execution model for the kernel launch: **SIMT or Tile**. Treat as the *primary oracle* for “what ran” in NCU-profiled runs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  **Failure mode:** range profiling aggregates can hide per-launch variation unless you export **per-instance** values.

#### Register allocation proxy
- **`launch__registers_per_thread`**: registers allocated per thread. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  **Caveat:** may exceed maximum live registers due to allocation holes / ABI / instruction constraints—so it’s a proxy for allocation pressure, not exact liveness. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html))  
  **Interpretation:** use as a *monotonic indicator* in a reg-pressure ladder (MB1), but don’t claim “live regs” without additional evidence.

#### Spill-specific derived metrics
- **`derived__local_spilling_requests`** and **`derived__local_spilling_requests_pct`**: derived attribution of local-memory requests to spilling-to-local. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  **Interpretation:** Use `_pct` for *classification* (“spill-dominant” vs “local-traffic-dominant”), and absolute requests for “magnitude.”

- **Also collect shared spilling**: NCU defines `derived__shared_spilling_requests` and `_pct`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  **Why:** If a backend (or future toolchain) spills to shared memory, local-only metrics miss it.

#### SASS corroboration
- **`sass__inst_executed_register_spilling`**: SASS instructions executed “as a result of register spilling,” with breakdowns for spilling to local vs shared. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  **Interpretn:** If derived spill metrics rise but SASS spill instructions stay flat, suspect (i) metric unsupported/`n/a` behavior, (ii) replay pass inconsistency, or (iii) classification error.

#### Generic “local traffic” context (not spill-specific)
- **`sass__inst_executed_local_loads` / `sass__inst_executed_local_stores`**: counts local LD/ST instructions executed. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  **Interpretation:** necessary context, but not sufficient to claiilling because local arrays and other compiler decisions can generate local traffic without “spills.”

#### Replay/overhead caveat
- NCU warns that replay passes can yield out-of-range or surprising metrics if the kernel’s work distribution varies across passes; mitigate by longer steady-state kernels and fewer simultaneous metrics. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
  **Interpretation:** your feasibility gate should run a single-kernel microbench long enough stable and should collect only the target metrics first.

---

## 4) Novelty Assessment

### 4.1 Core novelty claim (my interpretation)

The proposal’s core novelty is **a generalizable, fail-closed provenance + mechanism-attribution methodology for comparing compiler backends that can silently change execution models**, operationalized specifically for **Triton PTX/SIMT vs Triton-to-TileIR/Tile** on Blackwell, using (i) an **authoritative per-launch execution-model label** (`launch__execution_model`), () **artifact corroboration** (`.tileIR` vs `.cubin`, dump/override hooks), (iii) a **metric-feasibility gate** for newly supported tile profiling, and (iv) **calibrated microbenchmarks** that expose **spill knees** and disambiguate spills from generic local traffic.

### 4.2 Prior-art comparison matrix

| Proposal feature | Closest prior art | What’s missing there | Delta here |
|---|---|---|---|
| Per-launch execution-model provenance using `launch__execution_model` (SIMT vs Tile) | NCU defines the metriand its meaning. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) | Prior work/tools often *report* metrics but don’t enforce provenance as a fail-closed publication rule. | You elevate provenance to a **publication constraint** (exclude ambiguous points), which is especially relevant with documented fallback. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| Corroborate runtime provenance with build/cache artifacts (`.tileIR` vs `.cubin`) | NVIDIA blog sugges using cache extensions to verify TileIR compilation. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Blog gives a tip, not a systematic audit trail format; doesn’t address mixed kernels/caching pitfalls. | You formalize artifact corroboration and propose fail-closed exclusion rules + (optionally) dump/override hooks. ([github.com](https://github.com/triton-lang/triton)) |
| Metric-feasibility gate (“metric availabi matrix”) for tile workloads | NCU release notes announce tile profiling support. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)) | No guarantee that *specific* spill metrics are supported/valid for tile workloads. | You explicitly treat this as a *feasibility cliff* and propose an empirical matrix before committing to sweeps. |
| MB2 disambiguator: local array vs forced spill | General GPU microbenchmarking methodology exists; e.g., ISPASS microbenchmarking tradition. stuffedcow.net](https://www.stuffedcow.net/research/cudabmk)) | Prior microbench suites aren’t tailored to validate **NCU’s spill attribution metrics** under a new execution model. | You use MB2 specifically to validate that derived spill metrics correlate with true spilling vs generic local traffic—important for reviewer trust. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) |
| Explicit handling of TileIR→PTX fallback as a central threat | Triton-to-tile-IR README documlback. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Most compiler comparison papers assume a single fixed backend/ISA path. | You make fallback detection and provenance auditability the “first claim,” not an appendix footnote. |

### 4.3 Novelty risk (what reviewers may call incremental) + how to sharpen

**Likely reviewer critique:** “This is solid benchmarking hygiene, not a research contribution.”  
**How to sharpen (concretely):**
1. **Make “fallback rate” and “provenrate” first-class results.**  
   Report: *What fraction of attempted TileIR compilations actually execute as Tile vs silently fall back to SIMT?* This directly converts a methodology hazard into an empirical characterization result. Grounding: fallback is documented. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))
2. **Promote the metric-feasibility matrix into a reusable artifact.**  
   NCU added tile profiling support; early adopters will face “is this metric real or `n/a`?” issueublished matrix across NCU versions + driver versions is a concrete, reusable output. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html))
3. **Tie MB2 to a publishable “metric validation” claim.**  
   E.g., “On Blackwell Tile workloads, `derived__local_spilling_requests_pct` tracks SASS spill instruction deltas under forced-spill controls, and diverges under explicit local arrays.” That’s a tool+method characterization contribution. ([docs.nvidia.com](https://docs./nsight-compute/ProfilingGuide/))
4. **Explicitly position as a template for emerging execution models beyond TileIR.**  
   CUDA Tile is new in CUDA 13.1 and targets Blackwell first. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) Your method is plausibly reusable for future “mixed execution model” transitions.

---

## 5) Feasibility & Risk Register

### 5.1 Top 5 technical risks + mitigations

1) **Risk: Core spill/reg metrics are `n/a` or unsuppd for Tile workloads under NCU (feasibility cliff).**  
   - *Why it’s real:* NCU 2025.4 guarantees tile workload profiling support, but does not guarantee metric parity with SIMT profiling. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html))  
   - *Mitigation (early gate):* implement the “metric availability matrix” on the smallest possible Tile kernel, collecting only the non-negotiable metrics first (no giant section sets). Use `--query-metrics` to confirm the metrist and attempt collection, recording `n/a` outcomes. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
   - *De-risk milestone:* within week 1, show one Tile kernel where `launch__execution_model=Tile` and at least one spill metric returns a numeric value.

2) **Risk: Provenance ambiguity in runtime-only sweeps due to caching/mixed kernels/fallback.**  
   - *Why it’s real:* fallback is documented, and Triton caches compiled artifacts (with different extensions). ([github.com](tps://github.com/triton-lang/Triton-to-tile-IR))  
   - *Mitigation:*  
     - Isolate cache per run (separate `TRITON_HOME` or wipe cache), and optionally force recompilation (`TRITON_ALWAYS_COMPILE=1`). ([github.com](https://github.com/triton-lang/triton))  
     - Treat `.tileIR` vs `.cubin` as corroboration, not the oracle; keep a policy that main claims require NCU provenance. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
     - Log a per-kernel manifest: kernel hash/name → expected backend → observed `launch__execution_model` (for profiled samples) → artifact filenames.

3) **Risk: Occupancy hint sweep (H2) is not controllable from user code or is silently ignored.**  
   - *Why it’s real:* README asserts the hint exists and is critical, but does not document the user-facing API in the accessible text. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
   - *Mitigation:* Treat as optional uven. Add a minimal “occupancy-hint control proof” experiment: set the hint in two ways (if found), and show a change in `launch__occupancy_limit_*` or runtime/spill curves while holding other knobs fixed. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
   - *De-risk milestone:* close OQ01 with a code snippet + evidence that the hint changes something measurable.

4) **Risk: Correctness confounds from TileIR’s unordered-by-default memory ordering (not just numeric mode).**- *Why it’s real:* Triton-to-tile-IR README explicitly warns about unordered memory model and incorrect results under aliasing or cross-block reductions; Tile IR docs emphasize token-ordered memory ops unless tokens impose order. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
   - *Mitigation:*  
     - Add an always-on output check for microbenches and TB subset (within tolerance), and exclude any kernel where TileIR changes semantics (or add token-ordering fixes if available).  
     Prefer initial microbenches that are single-kernel, no aliasing, no cross-CTA reductions.

5) **Risk: Profiling perturbation / replay artifacts change schedule, spilling behavior, or metric validity.**  
   - *Why it’s real:* NCU explicitly discusses replay, multi-pass collection, and metric anomalies when behavior differs across passes. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
   - *Mitigation:*  
     - Separate “mechanism metrics runs” (NCU) from “runtime swe” (no NCU), but ensure provenance for any datapoint that enters main claims.  
     - Keep NCU metric sets minimal for the feasibility gate; increase only after confirming stability.

### 5.2 What would make the project fail (and how to de-risk early)

**Hard fail conditions (for the intended paper):**
- You cannot obtain **any** reliable spill-related mechanism metric for Tile-executed kernels (even if runtime differences exist). Then the work collapses into anecdotal runtime comparisons. ([docs.nvidia.c](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html))  
- You cannot robustly detect fallback/mixed execution, so PTX vs TileIR attribution is not defensible. (This is less likely given `launch__execution_model`, but you must implement per-launch extraction correctly.) ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))

**First prototype milestone (strong de-risk plan):**  
A single-page report containing:
1. One kernel with `ENABLE_TILE=1` where NCU reports `launch__execution_model=Tile`. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
2. A table showing which of your target metrics are numeric vs `n/a`.  
3. A paired `ENABLE_TILE=0` run for the same kernel and the same metric set.

### 5.3 Metric availability matrix template + minimal experiment plan

**Template (per kernel, per toolchain):**

| Field | Example values |
|---|---|
| GPU + driver | RTX 5090 + driver 590.xx (must be ≥ 590 for tile profiling) ([docs.nvidia.com](https://docs.nvidia.com/nght-compute/ReleaseNotes/index.html)) |
| CUDA toolkit | 13.1.x (Tile initial release targets Blackwell) ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) |
| NCU version | 2025.4.x (tile workload profiling supported) ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)) |
| Backend toggle | `ENABLE_TILE=0/1` ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| Provenance | `launch__execution_model` = SIMT/Tile ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)) |
| Metrics attempted | list of spill/reg metrics |
| Metric status | numeric / `n/a` / error |
| Sanity checks | e.g., derived spill vs SASS spill consistency |
| Notes | replay passes, kernel duration, etc. |

**Minimal plan to populate it:**
- Start with one tiny kernel (e.g., vector add) and one “reg-pressure ladder” kernel (MB1) to ensure you have both low- and high-pressure regimes.
- Run NCU with a metric list containinlaunch__execution_model`, `launch__registers_per_thread`, `derived__local_spilling_requests(_pct)`, `sass__inst_executed_register_spilling`, `sass__inst_executed_local_loads/stores`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
- Export raw results per kernel launch; confirm per-launch provenance in range mode if used. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))

---

## 6) Recommendations to Strengthen the Proposal

1) **Add a correctness/output-validation policy as a first-class control (not just numeric mode).**  
   TileIR’s unordered memory model is explicitly called out as a correctness risk; without output checks you risk comparing “fast wrong” vs “slow right.” ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
   *Actionable edit:* add a “Correctness Gate” subsection: every kernel must pass validation under both backends before inclusion.

2) **Expand provenance from a “label” into a measurable result: rort fallback rate and provenance coverage.**  
   Since fallback is documented, quantify it: *how often does `ENABLE_TILE=1` actually execute as Tile?* ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
   *Actionable edit:* add a planned plot/table: “% launches with execution_model=Tile vs SIMT under ENABLE_TILE=1, by kernel category.”

3) **Harden runtime-only provenance: isolate caches and force recompile when needed.**  
   Triton caching + `.tileIR`/`.cubin` artifacts are useful but mislead if reused across runs. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
   *Actionable edit:* require per-run cache directory isolation (or cleanup) and log `TRITON_ALWAYS_COMPILE`/cache path.

4) **Extend the metric set to include shared spilling, not only local spilling.**  
   NCU provides `derived__shared_spilling_requests(_pct)` and SASS `*_mem_shared` breakdowns. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
   *Actionable edit:* include shared spilling in “non-negotiable outputs” to avoid blind spots.

5) **Make “metric discordance rules” explicit (what you do when metrics disagree).**  
   Example rule: if `derived__local_spilling_requests_pct` increases but `sass__inst_executed_register_spilling` does not, classify the datapoint as “suspect” and exclude from main claims or investigate with a reduced metric set (replay sensitivity). ([docs.nvidia.com](https://doc/nsight-compute/ProfilingGuide/))  

6) **Strengthen MB1 to control for algorithmic confounds introduced by the reg-pressure knob.**  
   Unrolling/extra accumulators can change instruction count and memory behavior.  
   *Actionable edit:* add a “reg-pressure knob design principles” paragraph: keep memory traffic constant, prevent dead-code elimination, and report achieved FLOPs/bytes so reviewers can see you didn’t change the workload class.

7) **Strengthen MB2 with a 3-way calibration (not just 2-*  
   Use:  
   - (i) explicit local array (non-spill local traffic),  
   - (ii) forced spill (spill traffic),  
   - (iii) both combined,  
   and show your classifier still separates them. Ground the “local arrays can cause local memory” premise in NCU’s local-memory description. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  

8) **Add a small “tool overhead / replay sensitivity” ablation early.**  
   NCU replay can perturb results; the Profiling Guide explains ues due to replay variability. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  
   *Actionable edit:* include a plot: runtime (no NCU) vs runtime (NCU minimal metric set) for 2–3 kernels.

9) **Tighten TritonBench subset selection into a reproducible rule.**  
   TritonBench is easy to invoke, but operator choice can be accused of cherry-picking. ([github.com](https://github.com/meta-pytorch/tritonbench))  
   *Actionable edit:* define deterministic selection criteria (e.g., o op per category: GEMM-like, attention-like, norm/reduction, pointer-heavy) and pin TB commit + submodules.

10) **Explicitly log the driver/toolkit/NCU coupling constraints in the experiment manifest.**  
   Tile profiling needs driver ≥ 590; CUDA Tile starts in 13.1 and targets Blackwell initially; these constraints must be pinned. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html))  

---

## 7) Annotated Bibliography (12–18 items)

> Access date for all items below: *6-02-04** (unless otherwise indicated in the source itself).

1) **NVIDIA Nsight Compute Profiling Guide (v2025.4.1)** — authoritative definitions for `launch__execution_model`, spill metrics, local memory semantics, and SASS spill counters; core citation base for your metric interpretation and provenance claims. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/))  

2) **NVIDIA Nsight Compute Release Notes / “Updates in 2025.4”** — documents that NCU 2025.4 added CUDA tile wprofiling and introduced the Tile section; establishes minimum viability for profiling Tile kernels. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html))  

3) **NVIDIA Developer Forums: “Nsight Compute 2025.4 is now available” (Dec 5, 2025 post)** — secondary-but-official announcement reinforcing tile workload support and summarizing key features; useful for date-stamping tool capability in the paper narrative. ([forums.developer.nvidia.com](httpsums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614))  

4) **CUDA Toolkit 13.1 Release Notes (v13.1.0)** — normative statement that CUDA 13.1 introduces CUDA Tile (Tile IR + cuTile) and that the initial release targets Blackwell GPUs; anchors hardware/toolchain prerequisites. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html))  

5) **NVIDIA Technical Blog (Jan 30, 2026): “Advancing GPU Programming with the CUDA Tile IR Backfor OpenAI Triton”** — explains how to enable TileIR in Triton-to-TileIR, and provides the `.tileIR` vs `.cubin` cache fingerprint guidance plus TMA/descriptor rewrite motivation. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

6) **triton-lang/Triton-to-tile-IR (README)** — primary industrial source for the documented hazards: `ENABLE_TILE=1`, fallback-to-PTX on compilation bugs, missing `num_warps`, occupancy h32, `num_ctas=2` enabling 2CTA mode MMA, approx/FTZ toggles, and unordered memory model warnings. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  

7) **NVIDIA/cuda-tile (repo README)** — official open-source CUDA Tile IR ecosystem aligned with CUDA 13.1; useful context for artifact-level provenance and the tile toolchain (`tileiras`, bytecode). ([github.com](https://github.com/NVIDIA/cuda-tile))  

8) **CUDA Tile IR documentation (Semantics / Operations)** — describes token-ordered me operations and undefined ordering without tokens; supports your correctness/confound discussion for Tile execution. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/local/sections/semantics.html))  

9) **triton-lang/triton (README: dump/override env vars)** — documents `TRITON_KERNEL_DUMP`, `TRITON_DUMP_DIR`, `TRITON_KERNEL_OVERRIDE`, `TRITON_OVERRIDE_DIR`, `TRITON_ALWAYS_COMPILE`; supports your artifact capture and provenance hardening story (verify applicability in the TileIR fork). ([github.co(https://github.com/triton-lang/triton))  

10) **meta-pytorch/tritonbench (README)** — documents `python run.py --op gemm` usage and library integration; supports the “real operator anchors” part of the evaluation plan. ([github.com](https://github.com/meta-pytorch/tritonbench))  

11) **Sakdhnagool et al., “RegDem: Increasing GPU Performance via Shared Memory Register Spilling” (arXiv:1907.02894, 2019)** — directly relevant register-pressure/occupancy/spilling prior art; supports why spilling  policy matter and provides conceptual framing for spill tradeoffs. ([arxiv.org](https://arxiv.org/abs/1907.02894))  

12) **Wong et al., “Demystifying GPU Microarchitecture through Microbenchmarking” (ISPASS 2010; paper+code release)** — establishes precedent that carefully designed GPU microbenchmarks can reveal mechanisms and are publishable; good template for MB1/MB2 design philosophy. ([stuffedcow.net](https://www.stuffedcow.net/research/cudabmk))  

13) **Mytkowicz et al., “Producing wrong datt doing anything obviously wrong!” (ASPLOS 2009)** — classic measurement-bias paper; strengthens your argument that silent confounds (like fallback/caching/tool replay) can invalidate conclusions unless controlled and logged. ([research.ibm.com](https://research.ibm.com/publications/producing-wrong-data-without-doing-anything-obviously-wrong))  

14) **van der Kouwe et al., “Benchmarking Crimes: An Emerging Threat in Systems Security” (arXiv:1801.02381, 2018)** — provides a structured taxonomy of ng failures; supports a confound register and explicit exclusion rules (your fail-closed posture). ([arxiv.org](https://arxiv.org/abs/1801.02381))  

15) **Jatala et al., “GREENER: A Tool for Improving Energy Efficiency of Register Files” (arXiv:1709.04697, 2017)** — register-file-focused analysis tool paper; useful background emphasizing the importance and difficulty of reasoning about registers without careful instrumentation. ([arxiv.org](https://arxiv.org/abs/1709.04697))  

16) **Intel oneAPI GPUization Guide: “Optimizing Register Spills” (doc, 2024–2025 versions listed)** — industrial guidance on practical knobs (unrolling, live ranges, large arrays) that directly informs MB1/MB2 knob selection and confound avoidance, even if on a different GPU ecosystem. ([intel.com](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/optimizing-register-spills.html))

---
Learn more:
1. [https://github.com/triton-lang/Triton-to-tile-IR](https://github.com/triton-lang/Triton-IR)
2. [https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)
3. [https://docs.nvidia.com/nsight-compute/ProfilingGuide/](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
4. [https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html)
5. [https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html)
6. [https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)
7. [https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
8. [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)
9. [https://github.com/meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench)
10. [https://docs.nvidia.com/cuda/tile-ir/sections/operations.html](https://docs.nvidia.com/cuda/tile-ir/sections/operations.html)
11. [https://github.com/NVIDIA/cuda-tile](https://github.com/NVIDIA/cuda-tile)
12. [https://arxiv.org/abs/1907.02894](https://arxiv.org/abs/1907.02894)
13. [https://www.stuffedcow.net/research/cudabmk](https://www.stuffedcow.net/research/cudabmk)
14. [https://research.ibm.com/publications/producing-wrong-data-without-doing-anything-obviously-wrong](https://research.ibm.com/publications/producing-wrong-data-without-doing-anything-obviously-wrong)
15. [https://arxiv.org/abs/1801.02381](https://arxiv.org/abs/1801.02381)
16. [https://arxiv.org/abs/1709.04697](https://arxiv.org/abs/1709.04697)
17. [https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614)
18. [https://docs.nvidia.com/cuda/tile-ir/local/sections/semantics.html](https://docs.nvidia.com/cuda/tile-ir/local/sections/semantics.html)
19. [https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/optimizing-register-spills.html](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/optimizing-register-spills.html)
