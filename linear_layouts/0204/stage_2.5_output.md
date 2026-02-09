# WORK_PRODUCT

## 1) Core premise verification table (fail-closed)

| Premise ID | Core premise (ties to Claim_IDs) | Status | Evidence | What breaks if false / next test |
|---:|---|---|---|---|
| P01 | **TileIR backend exists for Triton and is toggleable** via `ENABLE_TILE=1` (vs PTX path). (C009) | VERIFIED | Triton-to-tile-IR README states `ENABLE_TILE=1`. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | If false → no A/B. **Next:** trivial smoke kernel compile + run with/without env var; log backend provenance. |
| P02 | **Prereqs:** TileIR backend requires **CUDA 13.1+** and **Blackwell GPUs** (initially). (C010) | VERIFIED | CUDA 13.1 release notes describe CUDA Tile and initial Blackwell target; NVIDIA blog lists CUDA 13.1+ and Blackwell prereqs for Triton-to-TileIR. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html?utm_source=openai)) | If false → toolchain scope wrong. **Next:** record exact `nvcc --version`, driver, GPU cc/SKU; enforce in harness. |
| P03 | **Silent fallback risk:** TileIR compilation can **fall back to PTX** on compilation bug; experiments must detect backend-used. (C012) | VERIFIED | README explicitly states fallback behavior in “ChangeList.” ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | If false → provenance simpler; if true (it is) → **must fail-closed**. **Next:** implement provenance gate per-run. |
| P04 | **Provenance signal from profiler:** Nsight Compute has `launch__execution_model` = **SIMT or Tile** per kernel launch. (C026) | VERIFIED | `launch__execution_model` definition in NCU metrics reference. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | If false → provenance cliff (G1). **Next:** confirm it’s populated (not N/A) on your TileIR kernels. |
| P05 | **Cache fingerprint:** When Tile backend active, Triton cache uses `.tileIR` vs `.cubin`. (C027) | VERIFIED | NVIDIA blog “Verify Tile IR compilation” section. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | If false → runtime-only sweeps need other provenance. **Next:** cache-dir scrape + hash per kernel build. |
| P06 | **Profiler supports tile workloads:** NCU 2025.4 adds “profiling CUDA tile workloads” + a “Tile” section. (C006) | VERIFIED | NCU 2025.4 updates in release notes. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html?utm_source=openai)) | If false → feasibility cliff (G2). **Next:** run NCU 2025.4+ on TileIR kernel and confirm report includes Tile section. |
| P07 | **Spill meaning:** Local memory is thread-private; spills and oversized locals cause local traffic; local resides in device memory and behaves like global latency. (C001,C002) | VERIFIED | NCU profiling guide “Memory” section on local memory + spilling + locality. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | If false → spill attribution invalid. **Next:** MB2 disambiguator must empirically validate classification anyway. |
| P08 | **Spill metrics exist:** `derived__local_spilling_requests(_pct)` and SASS spill/local-load/store counters exist & define spill-to-local vs local traffic. (C003,C004) | VERIFIED | Metric definitions for derived spill + SASS executed spill/local-load/store. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | If false → evaluation collapses (G2/G5). **Next:** metric query + collect on PTX backend first, then Tile backend. |
| P09 | **Numeric-mode confound knobs exist:** TileIR disables approx + FTZ by default; can enable via env vars. (C021) | VERIFIED | README lists `TILEIR_ENABLE_APPROX` and `TILEIR_ENABLE_FTZ`. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | If uncontrolled → “backend effect” may be numeric/ISA effect. **Next:** lock policy + output-check harness. |
| P10 | **Occupancy hint exists (1–32)** and is described as critical for TileIR backend. (C015) | VERIFIED (existence only) | README describes `occupancy` hint range and motivation. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | If cannot be *set* → H2 blocked. **Next:** resolve OQ01 by inspecting Triton API surface / Config fields locally. |
| P11 | **Occupancy hint is sweepable from user code and actually applied.** (C024) | UNVERIFIED | (No doc we can cite that shows exact user-level API + enforcement.) | If false → can’t make occupancy-hint a centerpiece. **Next:** minimal code snippet that (a) sets hint, (b) logs it, (c) shows systematic metric shift. |
| P12 | **2CTA mode confound exists:** `num_ctas=2` “enables 2CTA mode MMA on Blackwell” (dot-like workloads). (C022) | VERIFIED (existence claim) | README mentions `num_ctas=2` and 2CTA mode MMA. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | If misunderstood → could confound PTX-vs-TileIR claims. **Next:** MB4 ablation (num_ctas=1 vs 2) with backend-held-fixed. |
| P13 | **“Must-have” feasibility:** the required spill/reg metrics are **collectable (non-n/a) on Tile workloads** on your Blackwell GPUs. (C023) | UNVERIFIED | (This is inherently empirical; NCU docs don’t guarantee per-metric availability per execution model.) | If false → main contribution must pivot to runtime-only + indirect signals (weak). **Next:** build a metric-availability matrix (OQ02). |

**Audit summary:** the **measurement hooks exist** (NCU + TileIR backend + provenance signal), but the **two remaining paper-killers** are still empirical: **(i)** metric availability on Tile workloads (C023/OQ02) and **(ii)** occupancy hint controllability (C024/OQ01).

---

## 2) Competitor / baseline analysis (MICRO/ISCA/ASPLOS skepticism)

### 2.1 “Already doable with existing tools?” check (required)

| Check | Answer | So where is novelty allowed to live? |
|---|---|---|
| Can someone already flip PTX vs TileIR? | Yes (`ENABLE_TILE=1`). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | Not novel: “TileIR exists.” Novelty must be **methodology + attribution + artifacts**. |
| Can someone already measure registers/spills? | Yes: NCU defines local memory + spill metrics + SASS spill/local-load/store counters. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | Not novel: “NCU has metrics.” Novelty must be **(a) calibrated ladders**, **(b) fail-closed provenance**, **(c) validated heuristic** linking metrics → actionable guidance. |
| Can someone already profile tile workloads? | Yes in NCU 2025.4+. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html?utm_source=openai)) | Novelty must be in **how** to make claims reviewable (controls + provenance + confound isolation). |

### 2.2 Closest neighbors (named) + explicit delta type

| Neighbor ID | Closest neighbor | What they cover | Delta type for TileSpill |
|---:|---|---|---|
| N01 | **Nsight Compute docs + metrics** | Defines spill/local-memory concepts + exposes metrics; now supports tile workloads. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | **Methodology delta:** turn tool capability into **auditable A/B evidence** (provenance + calibration + attribution). |
| N02 | **NVIDIA TileIR/Triton blog announcement** | Explains TileIR backend, prereqs, cache fingerprint, and hints at performance issues (tensor-of-pointer) + TMA rewrite. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | **Measurement delta:** convert guidance into **quantified reg/spill deltas** + reproducible scripts + cross-operator mapping. |
| N03 | **Triton-to-tile-IR repo README** | States new hints (`occupancy`), missing `num_warps`, fallback to PTX, numeric knobs, `num_ctas=2` note. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | **Attribution delta:** isolate which observed effects are (a) backend, (b) hint choice, (c) confounds (2CTA, numeric). |
| N04 | **Triton compiler ecosystem (main repo)** | Already provides compiler inspection hooks (e.g., dump/override knobs) enabling artifact capture. ([github.com](https://github.com/triton-lang/triton?utm_source=openai)) | **Reproducibility delta:** define **fail-closed provenance artifacts** for runtime-only sweeps, not just “dump for debugging.” |
| N05 | **Multi-level / tile-based DSL research (e.g., ML-Triton, Hexcute)** | Focus on language/compiler design & performance; not specifically TileIR-vs-PTX spill attribution on Blackwell. ([arxiv.org](https://arxiv.org/abs/2503.14985?utm_source=openai)) | **Scope delta:** we are not proposing new compiler; we provide **hardware-facing characterization** + tuning heuristics grounded in NCU counters. |

**Key novelty positioning (reviewer-proof):**
- “We used existing tools” is *true*; novelty is **a fail-closed experimental protocol** that survives skepticism (G1/G2/G5) and yields **actionable spill-avoidance guidance**.

---

## 3) Per-direction audit table (with “already doable?” + risk scores)

**Scoring convention:** risk is **1 (low)** → **10 (high)**.

| Dir | Direction (one-liner) | Closest neighbors | Explicit delta | Already doable w/ existing tools? | Novelty risk | Feasibility risk | Eval credibility risk |
|---:|---|---|---|---:|---:|---:|---:|
| D1 | **Spill provenance + attribution:** PTX vs TileIR backend changes reg allocation/spills; prove via calibrated microbenches + NCU + fail-closed provenance. | N01–N04 | **Methodology delta:** (i) backend-used provenance, (ii) reg-pressure ladder (spill cliff), (iii) spill-vs-local disambiguation protocol, (iv) TB slice linkage | Yes | 5 | 4 | 3 *(if provenance+metrics collected)* |
| D2 | **Occupancy-hint tradeoff curves:** sweep occupancy hint 1–32 to map runtime vs spills. | N03 | **Attribution delta:** new knob unique to TileIR backend; show systematic knee/pareto frontier | Partially (knob exists, API not proven) | 6 | 8 *(blocked by OQ01/C024)* | 5 |
| D3 | **Descriptor/TMA rewrite case study:** tensor-of-pointer vs descriptor/TMA reduces reg pressure & spills under TileIR. | N02 | **Measurement delta:** quantify reg/spill impact across kernels, not just show code rewrite | Yes | 7 *(blog already suggests it)* | 5 | 4 |
| D4 | **2CTA-mode confound isolation:** map `num_ctas` (1 vs 2) impact to avoid misattributing backend effects. | N03 | **Credibility delta:** preempt “you measured a Blackwell mode toggle, not backend effect” | Yes | 6 | 6 | 5 |
| D5 | **Operator survey (TritonBench slice):** categorize which real ops spill under TileIR vs PTX and why, using the attribution protocol. | N01 + benchmark suites | **Ecological-validity delta:** connect microbench insights to real operators w/ the same counters | Yes | 6 | 5 | 4 |

**Audit call:** D1 is the only direction that (a) directly resolves G1/G2/G5 and (b) stays publishable even if D2 fails (occupancy hint API remains blocked).

---

## 4) Strategic recommendation (final direction)

### Final direction choice: **D1 (Spill provenance + attribution)**

**Why D1 wins (MICRO/ISCA/ASPLOS bar):**
- **Credibility gate:** D1 *requires* and operationalizes fail-closed provenance (G1) using `launch__execution_model` + cache fingerprint; without this, the work is unreviewable. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **Feasibility gate:** even if occupancy hint remains blocked, D1 still yields a strong paper via MB1/MB2 + NCU counters (regs + spills + local traffic). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- **Novelty framing that survives “you just ran Nsight Compute”:** D1 contributes a *validated* attribution workflow (spill vs local-array) and a calibrated “spill cliff” methodology, not tool usage.

**How other directions fit without derailing:**
- D4 becomes a **threat-to-validity control** (mandatory small ablation).
- D3 becomes a **secondary case study** (nice-to-have novelty transfer).
- D2 stays a **stretch goal**; include only if OQ01/C024 resolved with a demonstrably applied hint.
- D5 becomes the **ecological validity** section of D1.

---

## 5) Stage-3 Assembly Pack (paste-ready; do not add new conceptual claims later)

### 5.1 Working title
**TileSpill: Fail-Closed Spill Attribution for Triton’s CUDA TileIR Backend vs PTX on Blackwell**

### 5.2 Abstract (5–7 bullets)
- CUDA 13.1 introduces CUDA Tile / Tile IR, and Triton now has an experimental TileIR backend alongside the traditional PTX path. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html?utm_source=openai))  
- Changing compilation backend can change **register allocation decisions** and **spilling**, but TileIR can also fall back to PTX, making naïve A/B benchmarking unreliable. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- We present a **fail-closed provenance pipeline** that labels each measured kernel launch with an execution model (SIMT vs Tile) and rejects ambiguous runs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- We build **calibrated microbench ladders** that sweep register pressure to expose spill “knees/cliffs,” and a companion microbench that separates spill-induced local traffic from intentional local-memory use. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- Using Nsight Compute’s spill/local counters, we quantify backend-dependent changes in `launch__registers_per_thread`, spill-to-local request rates, and spill-related SASS instruction counts. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- We apply the same attribution protocol to a curated subset of real Triton operators (TritonBench slice) to test ecological validity and to derive actionable “when will this spill?” guidance.  
- We release scripts, metric sets, and provenance artifacts to reproduce the figures under pinned toolchain versions, acknowledging rapid version drift in the TileIR stack. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  

### 5.3 Contributions (3–5 bullets, referencing Claim_IDs)
- **K1 (Fail-closed backend provenance):** A per-run, per-kernel provenance label derived from Nsight Compute `launch__execution_model` (C026) plus Triton cache fingerprinting `.tileIR` vs `.cubin` (C027), designed to detect TileIR→PTX fallback (C012) under `ENABLE_TILE` toggling (C009).  
- **K2 (Spill attribution protocol):** A metric-driven method that treats local-memory traffic as ambiguous by default (C001/C002) and attributes “true spilling” using `derived__local_spilling_requests(_pct)` (C003) corroborated by SASS spill counters and local load/store counts (C004).  
- **K3 (Calibrated microbench ladders):** MB1/MB2 kernels that sweep register pressure and local-memory usage to empirically validate that chosen NCU spill metrics track spill events (supports G5; ties to C003/C004).  
- **K4 (Backend A/B characterization):** A reproducible evaluation plan comparing PTX backend vs TileIR backend on Blackwell GPUs (C010) with controls for numeric-mode confounds (`TILEIR_ENABLE_APPROX/FTZ`) (C021).  
- **K5 (Transfer to real workloads):** A curated TritonBench operator slice (C020) plus a TB_OP taxonomy showing how spill signals correlate with performance regressions across backends.

### 5.4 Background (TileIR backend + why spills matter) — **cited**
**What TileIR backend is (scope-relevant):**
- CUDA 13.1 introduces **CUDA Tile** and **CUDA Tile IR** as a tile-based programming model/IR, with the initial release targeting Blackwell GPUs. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html?utm_source=openai))  
- NVIDIA provides an incubating Triton backend that targets TileIR instead of PTX, enabled via environment variable selection and verified via `.tileIR` cache artifacts. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

**Why register pressure and spills are central:**
- Nsight Compute documents that CUDA **local memory** is thread-private and is used when automatic variables don’t fit in registers or when register spilling occurs; local memory resides in device memory and behaves like global-latency access. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- Nsight Compute provides both **derived spill-to-local metrics** and **SASS instruction counters** for spill/local operations, enabling spill attribution beyond “local loads/stores happened.” ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  

### 5.5 Method (measurement pipeline + controls)

**M1. Provenance-first harness (fail-closed):**
- Factor A: backend toggle via `ENABLE_TILE ∈ {0,1}` (C009). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- For every NCU-profiled datapoint, record `launch__execution_model` and assert expected value (SIMT for PTX backend, Tile for TileIR backend). If mismatch or N/A → discard run (NN1/NN4). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- For runtime-only sweeps, record compile/cache artifacts (e.g., `.tileIR` vs `.cubin`) as secondary provenance; treat disagreement with NCU provenance as a hard error. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- Record whether compilation fallback occurred (TileIR→PTX) per README-stated behavior; runs with fallback are excluded from PTX-vs-TileIR comparisons. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  

**M2. Controls (confound management):**
- **Numeric mode:** keep `TILEIR_ENABLE_APPROX` and `TILEIR_ENABLE_FTZ` fixed unless performing a dedicated ablation (NN policy), since TileIR disables them by default. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **2CTA mode confound:** include a `num_ctas` ablation (1 vs 2) for dot-like workloads because documentation states `num_ctas=2` enables “2CTA mode MMA” on Blackwell. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **Profiler perturbation:** separate “runtime measurements” (outside NCU) from “mechanism measurements” (inside NCU), and report both (threat mitigation).  
- **Reproducibility knobs:** pin toolchain versions; optionally use NCU clock-control features introduced in 2025.4 to reduce frequency noise (where supported). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html?utm_source=openai))  

**M3. Microbench design (auditable ladders):**
- MB1: reg-pressure ladder (unroll/accumulator/tile-size knobs) → induce spill cliff.  
- MB2: local-array vs spill disambiguator → validate that derived spill metrics change when spills change, even if local loads/stores exist for other reasons. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- MB4: `num_ctas` ablation to isolate 2CTA mode confound. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  

### 5.6 Evaluation (exact metrics + baselines + workloads + GPU matrix)

**E1. Exact Nsight Compute metrics (non-negotiable set):**
- `launch__execution_model` (provenance). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- `launch__registers_per_thread` (register allocation proxy). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- `derived__local_spilling_requests` and `derived__local_spilling_requests_pct`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- `sass__inst_executed_register_spilling`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- `sass__inst_executed_local_loads`, `sass__inst_executed_local_stores`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- **Optional if available:** `sass__inst_executed_register_spilling_mem_local` (more specific corroboration). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  

**E2. Baselines / ablations (must report):**
- **Primary baseline:** PTX backend (`ENABLE_TILE=0`) vs TileIR backend (`ENABLE_TILE=1`) on the *same* Blackwell GPU, holding inputs and numeric mode constant. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **Confound ablation:** `num_ctas ∈ {1,2}` for dot-like workloads (2CTA mode). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **Optional (only if OQ01 resolved):** occupancy hint sweep `occupancy ∈ [1..32]` (TileIR backend). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **Optional (case study):** tensor-of-pointer vs descriptor/TMA rewrite for impacted workloads (transfer story). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  

**E3. Workloads:**
- **Microbenches (required for MVP):** MB1, MB2, MB4.  
- **Realistic kernels:** TritonBench subset (5–15 ops), selected by a documented procedure (TB_OP## taxonomy). (C020)  
  - Selection procedure (locked now): choose ops spanning GEMM-like, reduction/norm-like, and pointer-heavy memory patterns; record exact op names + commit hash before final results.

**E4. GPU matrix (minimum):**
- **Primary (Blackwell):** RTX 5090 (consumer) + B200 (datacenter). *(cc/SKU details captured as artifacts; OQ06).*  
- **Secondary sanity (SIMT only acceptable):** H100 for PTX-backend control runs (no TileIR requirement implied).  
- **Optional:** GB10 once SKU/cc confirmed (OQ06).

**E5. Deliverables / artifacts (what gets published):**
- NCU reports directory (raw + merged), metric CSV extracts, plots, provenance logs, and toolchain “version manifest” per run. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html?utm_source=openai))  

### 5.7 Threats / limitations (explicit)
- **T1 Microbench validity:** microbenches may not reflect operator mixes; mitigated by TritonBench slice (D5 embedded).  
- **T2 Version drift:** TileIR backend is incubating; behavior may change across CUDA/Triton commits; mitigated by strict version pinning and reporting. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **T3 Profiler perturbation:** NCU replay/patching can change timing; mitigated by separating timing runs from mechanism runs and reporting overhead. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- **T4 Register metric interpretation:** register count is a proxy; mitigated by reporting both register proxy + spill-to-local derived metrics + SASS corroboration. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- **T5 Numeric-mode confounds:** approx/FTZ settings differ by default; mitigated by fixed env policy + output checks. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **T6 Hardware-mode confounds:** `num_ctas=2` can enable Blackwell-specific behavior; mitigated by MB4 ablation and by excluding confounded cases from backend-attribution claims. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  

### 5.8 Reviewer attack / response set (10 items)

| Attack ID | Likely reviewer attack | Response (what we will show) |
|---:|---|---|
| R01 | “This is just Nsight Compute screenshots.” | We contribute a **fail-closed experimental protocol** + calibrated microbench suite + validated spill attribution heuristic; tools are baseline. |
| R02 | “How do you know TileIR was actually used (no fallback)?” | Every plotted datapoint includes `launch__execution_model` provenance; ambiguous runs are rejected. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| R03 | “Runtime differences could be numeric-mode differences.” | Hold `TILEIR_ENABLE_APPROX/FTZ` fixed and perform output checks; numeric toggles only in dedicated ablation. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |
| R04 | “Local loads/stores aren’t necessarily spills.” | We use derived spill metrics + SASS spill counters + MB2 validation to disambiguate spill vs local-array traffic. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| R05 | “NCU overhead invalidates your perf conclusions.” | Separate unprofiled runtime runs from profiled mechanism runs; quantify overhead; use NCU only for attribution. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| R06 | “Your ‘registers per thread’ is not a reliable live-range measure.” | We treat it as a proxy and anchor conclusions on spill metrics; reg count only supports trends, not absolute live regs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| R07 | “Your results are Blackwell-SKU specific.” | Include at least one consumer + one datacenter Blackwell GPU; report full env manifests; scope claims accordingly. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html?utm_source=openai)) |
| R08 | “This is a compiler paper without compiler insight.” | We explicitly frame as **hardware-facing characterization**: what backend choice does to resource usage, with auditable evidence and guidance—not new compilation algorithms. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| R09 | “Microbench results don’t generalize to real ops.” | Demonstrate transfer via TritonBench slice; publish per-op spill signatures and correlate with slowdowns. |
| R10 | “You might be measuring 2CTA mode MMA / num_ctas effects.” | We include `num_ctas` ablation and exclude confounded settings from backend-attribution claims. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |

---

# CONTEXT_CAPSULE

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
  current_stage: 2.5
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
  - id: "TRITON-REPO"
    kind: "repo"
    title: "triton-lang/triton"
    url: "https://github.com/triton-lang/triton"
    last_verified: "2026-02-04"
    notes: "Used to document dump/override knobs and compilation artifact capture."

ARTIFACT_INDEX:
  stage0_fact_sheet: "WP0_20260204"
  stage1_gap_audit: "WP1_20260204"
  stage1_5_toolbox: "WP1_5_20260204"
  stage2_directions: "WP2_20260204"
  stage2_5_audit: "WP2_5_20260204"
  stage3_assembly_pack: "WP3_AP_20260204"
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
    - id: "V010"
      date: "2026-02-04"
      gap_id: "S2"
      verdict: "FINAL_DIRECTION_D1"
      rationale: "Stage-2.5 audit: D1 remains the only direction that simultaneously (i) is reviewable via fail-closed provenance, (ii) remains publishable if occupancy hint API is blocked, and (iii) directly measures mechanisms (regs/spills) via NCU rather than runtime anecdotes."
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
      blocks: ["success_creria case study"]

CLAIM_LEDGER:
  items:
    - id: "C001"
      scope_tag: "ACTIVE"
      claim: "CUDA local memory is thread-private and used when automatic variables don’t fit in registers or when register spilling occurs."
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
      claim: "TritonBench provides runnable operator benchmarks via python run.py --op <name>."
      status: "VERIFIED"
      evidence: ["E006"]
      paper_role: "A/C"
      risk_if_wrong: "No real-workload anchors."
    - id: "C021"
      scope_tag: "ACTIVE"
      claim: "TileIR backend disables approx and FTZ by dault; can be enabled via TILEIR_ENABLE_APPROX=1 and TILEIR_ENABLE_FTZ=1."
      status: "VERIFIED"
      evidence: ["E005"]
      paper_role: "B"
      risk_if_wrong: "Confounded numeric/ISA differences."
    - id: "C022"
      scope_tag: "ACTIVE"
      claim: "A Blackwell-specific mode referred to as '2CTA mode MMA' is documented in the TileIR backend repo as enabled by setting num_ctas=2 for dense dot-related workloads; its meaning and performance impact must be understood or explicitly scoped out."
      status: "VERIFIED"
      evidence: ["E005"]
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
    - id: "C028"
      scope_tag: "ACTIVE"
      claim: "Triton provides environment-variable knobs to dump/override compilation artifacts (e.g., TRITON_KERNEL_DUMP, TRITON_DUMP_DIR, TRITON_KERNEL_OVERRIDE, TRITON_OVERRIDE_DIR, TRITON_ALWAYS_COMPILE), which can support per-kernel artifact capture for provenance and debugging."
      status: "VERIFIED"
      evidence: ["E013"]
      paper_role: "A (artifact capture path for runtime-only sweeps)"
      risk_if_wrong: "Harder to build provenance artifacts outside NCU."

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
      url: "https://github.com/triton-lang/Triton--tile-IR?tab=readme-ov-file"
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
    - id: "E013"
      source_id: "TRITON-REPO"
      kind: "repo_readme"
      pointer: "Triton repo: env vars for dumping/overriding compiled artifacts (TRITON_KERNEL_DUMP, TRITON_DUMP_DIR, etc.)"
      url: "https://github.com/triton-lang/triton"
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
    - "Dedicated TileIR numeric-mode ablation: toggle TILEIENABLE_APPROX / TILEIR_ENABLE_FTZ only as a controlled factor (not mixed into main A/B)."
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
      blocks: ["G9", "confound isolation for backend attribution"]
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
      notes: "Partially mitigated by Triton dump/override env vars (C028), but applicability to TileIR fork must be validated."
    - id: "OQ08"
      status: "OPEN"
      statement: "Define and validate a numeric-mode control policy for PTX vs TileIR A/B (approx/FTZ), including output-check strategy."
      blocks: ["G4"]
      plan: ["Q12"]
    - id: "OQ09"
      status: "OPEN"
      statement: "Finalize the exact TritonBench op list (5–15) and create a TB_OP## category map (GEMM/attention/norm/reduction/pointer-heavy) with a stable selection procedure."
      blocks: ["G10"]
      plan: ["Q13"]
    - id: "OQ10"
      status: "OPEN"
      statement: "Pin and record the actual toolchain versions (CUDA toolkit, driver, Triton, Python, Nsight Compute, TileIR stack) used for all reported results; define how scripts capture them."
      blocks: ["G8", eproducibility expectations"]
      plan: ["Q14"]
  closed: []

DELTA_LOG:
  - "Updated profile.current_stage: 2 → 2.5."
  - "Added ARTIFACT_INDEX.stage2_5_audit = WP2_5_20260204."
  - "Added ARTIFACT_INDEX.stage3_assembly_pack = WP3_AP_20260204."
  - "Added VERDICT_LEDGER V010: final direction choice D1 after Stage-2.5 audit."
  - "Updated CLAIM_LEDGER: C022 status UNVERIFIED → VERIFIED (doc states num_ctas=2 enables 2CTA mode MMA)."
  - "Added GOLDEN_SOURCES.TRITON-REPO and EVIDENCE_LEDGER E013; addedIM_LEDGER C028 on Triton artifact dump knobs."
  - "OPEN_QUESTIONS unchanged in count; OQ07 annotated with partial mitigation via C028 (still requires validation on TileIR fork)."

CAPSULE_HEALTH:
  counts:
    verdict_ledger: 10
    claim_ledger: 20
    evidence_ledger: 13
    experiment_ledger: 0
    open_questions_active: 10
    open_questions_closed: 0
  capsule_token_estimate: 7600
  monotonicity: "OK (stage2 → stage2.5 growth)"

NEXT_STAGE_HINT: "Fresh chat; paste capsule + Stage 3 prompt + Assemblyack."
```

---
Learn more:
1. [GitHub - triton-lang/Triton-to-tile-IR: incubator repo for CUDA-TileIR backend](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)
2. [CUDA Toolkit 13.1 - Release Notes — Release Notes 13.1 documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html?utm_source=openai)
3. [2. Profiling Guide — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)
4. [Advancing GPU Prmming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
5. [Updates in 2025.4 — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/topics/updates-2025-4.html?utm_source=openai)
6. [2. Profiling Guide — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)
7. [GitHub - triton-langton: Development repository for the Triton language and compiler](https://github.com/triton-lang/triton?utm_source=openai)
8. [ML-Triton, A Multi-Level Compilation and Language Extension to Triton GPU Programming](https://arxiv.org/abs/2503.14985?utm_source=openai)
