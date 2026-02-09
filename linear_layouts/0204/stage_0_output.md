# WORK_PRODUCT

## 0) Scope Freeze Card (must be first)

### Research question (1 sentence)
**RQ0:** On NVIDIA Blackwell GPUs, **how does Triton’s CUDA Tile IR backend (enabled via `ENABLE_TILE=1`) change register allocation and register spilling/local-memory traffic vs the PTX backend**, and can these deltas be **attributed to explicit, user-sweepable knobs** (TileIR `occupancy` hint, pointer-style rewrite to descriptors/TMA, and register-pressure sweeps) measured with Nsight Compute? ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))

### Hypotheses H1..H4 (each 1 sentence; must name sweep vars + metrics)

- **H1 (Backend delta exists):** Sweeping **backend** ∈ {PTX (`ENABLE_TILE=0`), TileIR (`ENABLE_TILE=1`)} on the **same Blackwell GPU + same kernel** will produce a measurable change in **register allocation / spill signals**, observable via `launch__registers_per_thread`, `derived__local_spilling_requests(_pct)`, and `sass__inst_executed_register_spilling`, with corresponding runtime shifts. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai))  
- **H2 (TileIR occupancy knob is performance-critical):** Sweeping TileIR **`occupancy` hint** \(N ∈ [1,32]\) will expose a **non-monotonic** tradeoff between runtime and spill pressure, observable via runtime + `derived__local_spilling_requests(_pct)` + `sass__inst_executed_register_spilling`, because the hint is explicitly described as “critical” and intended to control active CTAs per SM. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **H3 (Pointer-style rewrite reduces reg pressure/spills on TileIR):** Sweeping **memory access style** ∈ {tensor-of-pointers, descriptor/TMA rewrite} (within the same kernel algorithm) will **reduce registers/spills** under TileIR, observable via lower `launch__registers_per_thread` and lower spill metrics, because tensor-of-pointer patterns are explicitly called out as a known performance problem and TMA-style descriptor rewrites are the suggested mitigation. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- **H4 (Certain reductions/norms spill due to missing `num_warps` exposure):** Sweeping **workload class** ∈ {GEMM-like, reduction/norm with large reduction dim} will show **disproportionate spilling** on TileIR for the reduction/norm class (vs PTX), observable via increased `derived__local_spilling_requests_pct` and `sass__inst_executed_register_spilling`, consistent with the backend limitation that `num_warps` is “not exposed yet” and can cause spilling for some Norm kernels. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  

### Excluded topics (3–7 bullets)
- Hardware redesign / RTL / new SM microarchitecture proposals (disallowed scope).  
- Full model training or end-to-end ML training runs as primary evidence (disallowed scope).  
- Simulator-first evaluation (allowed only as secondary sanity; not primary).  
- Compiler-internal speculation without measurable hooks in NCU + runtime.  
- Claims about exact Blackwell register-file sizes/partitioning unless explicitly sourced (avoid guessing).  

### Success criteria (3–6 bullets)
- **SC1 (Measurable delta):** At least one **reproducible** PTX-vs-TileIR finding where spill/regs metrics move materially and correlate with runtime changes on a Blackwell GPU. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai))  
- **SC2 (Attribution):** Demonstrate attribution to at least one explicit knob: TileIR `occupancy` sweep and/or descriptor/TMA rewrite and/or register-pressure sweep (with controlled A/B). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **SC3 (Paper-grade method):** A minimal microbench slice + scripts that emit tables/plots with runtime + `launch__registers_per_thread` + spill metrics + local memory instruction counts. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai))  
- **SC4 (Tooling feasibility gate):** Nsight Compute version is confirmed to support profiling CUDA tile workloads (TileIR path) on the target Blackwell system(s). ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai))  

### Evaluation environment template (fields may be unknown, but must exist)

| Field group | Field | Value (TBD allowed) |
|---|---|---|
| Hardware | GPU model | TBD (RTX 5090 / B200 / etc) |
| Hardware | GPU arch / compute capability | TBD (must be Blackwell for TileIR path per prereqs) ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| Hardware | SM count / clocks / power limit | TBD |
| Hardware | Host CPU + RAM | TBD |
| Software | OS + kernel | TBD |
| Software | NVIDIA driver version | TBD |
| Software | CUDA toolkit version | **≥ 13.1 required for TileIR backend** ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| Software | Triton version | TBD (pin exact commit/tag) |
| Software | Triton-to-TileIR stack version | TBD (pin commit of `triton-lang/Triton-to-tile-IR`) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |
| Tooling | Nsight Compute version | **≥ 2025.4 recommended for “CUDA tile workloads” profiling support** ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai)) |
| Tooling | NCU CLI invocation | TBD (store exact command-lines + section/metrics list) |
| Experimental factors | Backend | `ENABLE_TILE` ∈ {0,1} ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| Experimental factors | TileIR occupancy hint | \(N ∈ [1,32]\) (how set = **UNVERIFIED; must resolve**) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |
| Experimental factors | Pointer style | tensor-of-pointer vs descriptor/TMA rewrite ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| Experimental factors | Register pressure knobs | unroll / tile sizes / reduction dims (kernel-specific) |
| Controls | Warmup + repeats | TBD (e.g., warmup 10, repeats 50; report median/IQR) |
| Controls | Clock control | TBD (document any boost/lock settings; if used) ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai)) |
| Artifacts | NCU reports directory | TBD |
| Artifacts | Kernel sources + configs | TBD |

---

## 1) Table: Ground Truth Glossary

| Term | Definition (1–2 lines) | Where used | Source_ID (or UNVERIFIED) | Notes |
|---|---|---|---|---|
| Local memory | Thread-private memory space used for automatic vars that don’t fit in registers or when spilling occurs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Claims C001–C004; interpreting spill traffic | NV-NCU | Local resides in device memory; access resembles global latency. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| Register spilling | Compiler-generated loads/stores to preserve values when register pressure exceeds available registers. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Metrics interpretation; microbench design | NV-NCU | NCU has explicit spill metrics (derived + SASS). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) |
| Register pressure | Demand for registers (live values) vs hardware/ABI constraints; high pressure increases spill risk. | Hypotheses H1–H4 | NV-NCU | “Most important resource under compiler control is number of registers used.” ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) |
| `launch__registers_per_thread` | NCU “launch” metric reporting allocated registers per thread; can be larger than max live regs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai)) | Primary reg-allocation signal | NV-NCU | Must not over-interpret; corroborate with other evidence. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai)) |
| Live registers | Source-level metric concept: number of registers that must be kept valid at a code location. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2023.2/NsightCompute/index.html?utm_source=openai)) | Corroboration for C005/C007 | NV-NCU | Helps interpret “holes” caveat. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2023.2/NsightCompute/index.html?utm_source=openai)) |
| Allocation “holes” | Register allocation fragmentation due to ABI / instruction constraints, inflating allocated count vs live regs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai)) | Threat to validity | NV-NCU | Explicitly called out in Nsight Compute docs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai)) |
| `derived__local_spilling_requests` | Count of executed instructions + L1 requests made due to register spilling to local memory. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Primary spill-volume metric | NV-NCU | Not “all local memory,” specifically spill-related. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) |
| `derived__local_spilling_requests_pct` | Percent of total local-memory requests to L1 due to register spilling. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Normalized spill signal | NV-NCU | Useful across kernels with differing local traffic baselines. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) |
| `sass__inst_executed_register_spilling` | SASS instruction count for loads/stores executed as a result of register spilling. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Instruction-level corroboration | NV-NCU | Prefer with local load/store counts for triangulation. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) |
| `sass__inst_executed_local_loads/stores` | Counts of executed local memory load/store instructions. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Separating “local” vs “spill” | NV-NCU | Local accesses can exist without being spills (arrays, etc.). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| CTA occupancy | CTAs per SM limited by threads, registers, shared memory, barriers, etc. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Motivates occupancy hint sweep | NV-NCU | Occupancy is a resource limiter; interacts with registers. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) |
| TileIR backend | Triton backend targeting CUDA Tile IR instead of PTX (Triton-to-TileIR). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | Project core A/B factor | NV-TILE-BLOG / TILE-REPO | Incubator, active development, limitations. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| `ENABLE_TILE` | Env var switch to enable CUDA Tile IR backend in Triton-to-tile-IR repo. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | Backend A/B toggle | TILE-REPO | Must record in experiment metadata. |
| TileIR `occupancy` hint | Backend-specific hint \(N=1..32\) expecting N active CTAs per SM; described as “critical.” ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | Hypothesis H2 | TILE-REPO | **How to set in code = UNVERIFIED** (needs doc/examples). |
| Tensor-of-pointer | Triton pattern building a tensor of element pointers inside kernel; called out as suboptimal on TileIR backend. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | Hypothesis H3 | NV-TILE-BLOG / TILE-REPO | Key A/B rewrite axis. |
| TMA / TMA API | Suggested mitigation: avoid tensor-of-pointers by using descriptor/TMA load/store style. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | Hypothesis H3 | NV-TILE-BLOG / TILE-REPO | Specific API surface must be pinned per Triton version. |
| Tensor descriptor | Structured metadata passed for TMA-backed operations (e.g., descriptor-based loads/stores). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | H3 rewrite path | NV-TILE-BLOG | Exact Triton API name varies; do not assume beyond cited examples. |
| Unordered memory model | TileIR backend functional note: supports unordered memory model; ordering not implicit. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | Workload selection constraints | TILE-REPO / NV-TILEIR-MM | Impacts correctness for aliasing / cross-block reductions. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |
| Tokens / token order | Tile IR uses tokens to establish ordering; program dependencies don’t order memory ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)) | Correctness constraints | NV-TILEIR-MM | Avoid racy microbenches or explicitly order if needed. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)) |
| Backend fallback | TileIR repo notes: on compilation bug, backend can fall back to PTX. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | A/B validity | TILE-REPO | Must detect to avoid false “TileIR” results. |
| TritonBench | Suite of PyTorch operators + harness; runnable via `python run.py --op ...`. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai)) | Real-workload anchors | TB | Use subset with norms/reductions + GEMM motifs. |

---

## 2) Table: Tooling & Evaluation Baseline Map

| Source_ID | What it is | What it measures/guarantees | Key limitations | Evidence (cite or UNVERIFIED) |
|---|---|---|---|---|
| NV-PTX | NVIDIA PTX ISA specification (programming model, state spaces, etc.). | Defines PTX-level model and semantics used by compilers and tooling. | PTX != final SASS; hardware behavior can differ; not a spill metric source by itself. | ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) |
| NV-NCU | Nsight Compute Profiling Guide + docs. | Defines local memory, spill meaning, and provides metric definitions for spill-related counters and SASS instruction counts. | `launch__registers_per_thread` can overstate live regs; metric availability varies by GPU/tool version. | Local memory definition: ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)); spill metrics: ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)); register caveat: ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai)) |
| NV-NCU-2025.4 | Nsight Compute 2025.4 release note (forum). | States added support for profiling CUDA tile workloads + new Tile section + Tile↔source correlation (limited). | Does not guarantee all metrics supported for tile workloads; need empirical `--query-metrics`. | ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai)) |
| TB-1 | meta-pytorch/tritonbench benchmark suite. | Provides runnable operator benchmarks (`python run.py --op ...`) to anchor evaluation beyond synthetic microbenches. | Not a microarch tool; operator coverage/configs may change; must pin commit. | ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai)) |
| NV-TILE-BLOG | NVIDIA blog introducing Triton-to-TileIR backend (Jan 30, 2026). | Documents `ENABLE_TILE=1`, prereqs (CUDA 13.1+, Blackwell), limitations, tensor-of-pointer perf issue + TMA mitigation example. | Blog-level; not a formal spec; details can drift vs repo. | ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| NV-TILEIR-MM | CUDA Tile IR “Memory Model” spec (latest). | Defines tile IR memory ordering, scopes, and tokens; describes weak operations and token ordering requirements. | Not a performance guide; does not directly define register/spill behavior. | ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)) |
| TILE-REPO | triton-lang/Triton-to-tile-IR repo (README). | Specifies env vars, known issues, tuning tips (`occupancy` 1–32), missing `num_warps`, fallback-to-PTX behavior, approx/FTZ toggles. | Incubator; behavior may change; README may be ahead/behind implementation. | ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |
| CUDA-TILE-REPO | NVIDIA/cuda-tile repo. | Example of producing Tile IR bytecode + compiling via `tileiras`; states CUDA 13.1+ + compatible driver required. | Not Triton-specific; focuses on Tile IR tooling/examples. | ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai)) |
| CUDA-13.1-RN | CUDA Toolkit 13.1 release notes. | Introduces CUDA Tile (Tile IR + cuTile) and notes initial release targets Blackwell GPUs. | High-level; not Triton-specific; no spill metric discussion. | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html?utm_source=openai)) |

---

## 3) Table: SOTA Baseline Map (microarch-specific)

| Baseline_ID | Baseline description | Why it is a fair comparator | What it cannot explain | Evidence |
|---|---|---|---|---|
| B0 | **PTX backend** run: same Triton kernel, `ENABLE_TILE=0`. | Same high-level kernel; isolates backend/codegen differences. | Does not isolate individual compiler passes; only end-to-end delta. | Backend switch exists via env var. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| B1 | **TileIR backend** run: same Triton kernel, `ENABLE_TILE=1`. | Direct A/B vs B0 on same GPU + same kernel source. | TileIR backend may have unsupported ops / perf issues; may fall back to PTX. | ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| B2 | **TileIR `occupancy` sweep**: N=1..32 vs default N=1. | Single-knob sweep explicitly called “critical”; stays within TileIR backend. | Needs exact mechanism to set hint; effect may be workload-specific. | ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |
| B3 | **Pointer-style A/B**: tensor-of-pointers vs descriptor/TMA rewrite (same math). | Targets a known TileIR performance cliff and a documented mitigation; isolates a concrete code pattern. | Rewrite may change instruction mix beyond registers/spills (e.g., memory pipes). | Tensor-of-pointer issue + TMA rewrite described. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| B4 | **TileIR + `num_ctas` ablation** (e.g., 1 vs 2) on dot/GEMM-like workloads. | `num_ctas` called “critical” for dense dot on Blackwell; plausible interaction with regs/occupancy. | Not universal; may not apply to reductions/norms. | ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) |
| B5 | **Cross-arch sanity**: run PTX backend microbench on H100 to validate harness (not for TileIR claims). | Ensures measurement pipeline works; helps debug nondeterminism. | Cannot support TileIR-vs-PTX claims on Blackwell. | (Methodological control; no web claim) |
| B6 | **Metric triangulation**: use both derived spill metrics + SASS spill instruction counters. | Reduces risk of misinterpreting local memory or reg allocation counts. | Still correlational; does not prove causality without controlled sweeps. | Metric definitions exist. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) |

---

## 4) Table: Claim Ledger v0 (15–30 claims)

**Status legend:** VERIFIED = directly supported by cited primary sources; INFERENCE = logically required methodology choice; UNVERIFIED = needs a concrete query/experiment.

| Claim_ID | Claim (1 sentence) | Status | Evidence pointers | Paper role (A/B/C/D) | Risk if wrong |
|---|---|---|---|---|---|
| C001 | CUDA local memory is thread-private and is used when automatic variables don’t fit in registers or when register spilling occurs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | VERIFIED | NV-NCU | A (tooling ground truth) | Misclassify spill vs non-spill local traffic. |
| C002 | Local memory resides in device memory and has similar latency to global memory. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | VERIFIED | NV-NCU | A | Over/underestimate performance impact of spills. |
| C003 | Nsight Compute defines `derived__local_spilling_requests` and `derived__local_spilling_requests_pct` as spill-to-local-memory request metrics. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | VERIFIED | NV-NCU | A | Wrong metric selection → wrong conclusions. |
| C004 | Nsight Compute provides SASS instruction counts for spilling, including `sass__inst_executed_register_spilling` and local load/store counts. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | VERIFIED | NV-NCU | A | Lose ability to corroborate derived metrics. |
| C005 | `launch__registers_per_thread` can be significantly higher than maximum live registers due to allocation holes and ABI/instruction constraints, so it must be corroborated. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai)) | VERIFIED | NV-NCU | A | False narrative (“TileIR uses more regs”) from artifact. |
| C006 | Nsight Compute 2025.4 added support for profiling CUDA tile workloads and introduced a Tile summary section. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai)) | VERIFIED | NV-NCU-2025.4 | A (feasibility gate) | Project infeasible if Tile workloads can’t be profiled. |
| C007 | `derived__local_spilling_requests` is specifically about register spilling to local memory, not all local memory accesses. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | VERIFIED | NV-NCU | A | Conflate local arrays with spills. |
| C008 | In CUDA, CTA occupancy is limited by physical resources including registers, threads, shared memory, and barriers, so register count can constrain occupancy. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | VERIFIED | NV-NCU | A | Misinterpret occupancy hint effects. |
| C009 | Triton-to-TileIR backend can be enabled with environment variable `ENABLE_TILE=1`. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | VERIFIED | NV-TILE-BLOG / TILE-REPO | B (stack fact) | Can’t run A/B if enable path wrong. |
| C010 | Triton-to-TileIR requires CUDA 13.1+ and Blackwell GPUs (initially), and is in active development with limitations. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | VERIFIED | NV-TILE-BLOG / TILE-REPO / CUDA-13.1-RN | B | Wrong platform assumptions; wasted setup time. |
| C011 | The TileIR backend README states it “only uses features available in CUDA 13.1” and only supports Blackwell GPU in CUDA 13.1. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | VERIFIED | TILE-REPO | B | Confuse unsupported platforms with “bugs.” |
| C012 | The TileIR backend may fall back to PTX on compilation bugs, so experiments must detect the active backend. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | VERIFIED | TILE-REPO | B/C | False A/B comparisons if fallback occurs. |
| C013 | Tensor-of-pointer load/store patterns are a known TileIR performance issue; rewriting to TMA/descriptor style is a recommended mitigation. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-progrming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | VERIFIED | NV-TILE-BLOG / TILE-REPO | B/C | Miss the key performance lever in evaluation. |
| C014 | `num_warps` is not exposed yet in TileIR backend; for some Norm kernels with large reduction dims, performance may degrade due to register spilling. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | VERIFIED | TILE-REPO | B | Misattribute norm slowdowns to “TileIR is bad” generically. |
| C015 leIR backend introduces an `occupancy` hint that accepts integer N=1..32 and is described as “critical.” ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | VERIFIED | TILE-REPO | B/C | Without this knob, tuning space is too small. |
| C016 | Tile IR memory operations may be reordered; ordering is undefined unless established by tokens; weak operations cannot be used for inter-thread communication. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/ry_model.html?utm_source=openai)) | VERIFIED | NV-TILEIR-MM | B (correctness constraints) | Incorrect kernels invalidate perf data. |
| C017 | A valid PTX vs TileIR comparison must keep kernel algorithm + problem sizes fixed and change only backend/tuning knobs. | INFERENCE | Method constraint | D (inference) | Confounded comparisons. |
| C018 | `launch__registers_per_thread` should be interpreted alongside spill metrics and (when possible) live-register source metrics to avoid “holes” artifacts. ([docsdia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai)) | INFERENCE | NV-NCU | D | Wrong causal attribution. |
| C019 | NCU metric availability and meaning can vary by GPU architecture/tool version, so the metric set must be confirmed on the target systems (e.g., with `--query-metrics`). ([developer.nvidia.com](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/?utm_source=openai)) | INFERENCE | NV blog (NCU usage) | D | Missing metrics → broken evaluation plan. |
| C020 | TritonBench is a real suite of PyTorch operators with runnable commands like `python run.py --op gemm`. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai)) | VERIFIED | TB-1 | A/C | No real-workload anchor. |
| C021 | The TileIR backend disables approx and FTZ by default and provides env vars to enable them (`TILEIR_ENABLE_APPROX=1`, `TILEIR_ENABLE_FTZ=1`). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | RIFIED | TILE-REPO | B (controls) | Uncontrolled numeric/ISA differences confound A/B. |
| C022 | TileIR backend supports `num_ctas` and calls it “critical” for dense dot-related workloads (2CTA mode MMA on Blackwell). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)) | VERIFIED | TILE-REPO | B/C | Miss important Blackwell-specific tuning. |
| C023 | TileIR compiled kernels may be cached with `.tileIR` extensions rather than `.cubin`, providing a potential backend-detec signal. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) | VERIFIED | NV-TILE-BLOG | B/C | Can’t reliably detect fallback/backend. |
| C024 | CUDA Tile IR memory model is derived from PTX memory model and is intended as a strict weakening to allow PTX interoperability. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)) | VERIFIED | NV-TILEIR-MM B | Incorrect assumptions about ordering semantics. |

---

## 5) Golden Snapshot (Carry-Forward)

### NV-NCU (Nsight Compute Profiling Guide + docs)
- **What it is:** NVIDIA’s primary documentation for Nsight Compute metrics and profiling interpretation. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **What it claims/guarantees:** Defines local memory and spilling causes; defines spill-related derived metrics and SASS spill counters; warns about `launchregisters_per_thread` interpretation. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **What it does NOT say:** Does not guarantee that every metric is available on every GPU/tool/version or on tile workloads specifically (must confirm on target). ([developer.nvidia.com](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/?utm_source=openai))  
- **Why we care:** Supplies “reviewer-doesn’t-argue” definitions + metric hooks foster/spill/local-memory claims.

### NV-PTX (PTX ISA docs)
- **What it is:** NVIDIA PTX ISA specification (SIMT programming model, state spaces, memory model, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **What it claims/guarantees:** Formalizes PTX-level semantics, including state spaces and memory consistency model at the PTX level. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **What it does NOT say:** Does not directly specify final SASS scheduling or actual reg allocation decisions; does not directly provide spill counters.  
- **Why we care:** Baseline “PTX backend” semantics anchor; helps constrain what PTX-vs-TileIR comparisons mean.

### NV-TILEIR-MM (Tile IR memory model spec)
- **What it is:** Formal memory model for CUDA Tile IR (ordering, scopes, tokens). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))  
- **What it claims/guarantees:*ken ordering is required for ordering; weak ops cannot be used for communication; model is derived from PTX and intended as a strict weakening. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai))  
- **What it does NOT say:** Not a performance tuning guide; does not explain register allocation/spills.  
- **Why we care:** Workload selection and correctness constraints (avoid races/aliasing pitfalls) for valid perf measurements.

### NV-TILE-BLOG (NVIDIA blog: Triton-to-TileIR backend, Jan 30, 2026)
- **What it is:** NVIDIA technical blog describing integration of CUDA Tile IR backend into Triton (incubator repo). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- **What it claims/guarantees:** Prereqs (CUDA 13.1+, Blackwell); enable via `ENABLE_TILE=1`; limitations; tensor-of-pointer perf issue; suggests TMA-based rewrite. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- **What it does NOT say:** Does not provide a complete, stable API spec for all tuning knobs; details may drift.  
- **Why we care:** Authoritative “as-of-date” description + concrete mitigation direction for H3.

### TILE-REPO (triton-lang/Triton-to-tile-IR)
- **What it is:** Incubator GitHub repo adding CUDA Tile IR backend to Triton; README includes tuning and limitationsgithub.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **What it claims/guarantees:** `ENABLE_TILE=1`; CUDA 13.1-only features; Blackwell-only in 13.1; known perf/functional issues; `occupancy` hint (1–32); missing `num_warps`; fallback to PTX; approx/FTZ env vars. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **What it does NOT say:** README alone doesn’t fully specify how to set each hint in user code (needs examples/docs). ([gitcom](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **Why we care:** Primary knob list and “limitations we must design around.”

### CUDA-TILE-REPO (NVIDIA/cuda-tile)
- **What it is:** NVIDIA repo with Tile IR tooling/examples (bytecode, translation, `tileiras`, driver loading). ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai))  
- **What it claims/guarantees:** Tile IR bytecode can be produced/compiled; prerequisites include supported CUDA device + CUDA 13+ compatible driver. ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai))  
- **What it does NOT say:** Not about Triton register spills; not a benchmarking harness.  
- **Why we care:** Toolchain grounding and fallback path for minimal TileIR “hello world” validation.

### TB (TritonBench)
- **What it is:** Collection of PyTorch operators and benchmark harness for Triton performance evaluation. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **What it ms/guarantees:** Provides install steps and a simple CLI to benchmark ops (`python run.py --op gemm`). ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **What it does NOT say:** Not focused on microarchitectural attribution; must add NCU capture + controls.  
- **Why we care:** Realistic operator anchors beyond synthetic microbenches.

### P1 (Linear Layouts — Triton)
- **What it is:** Paper on “Linear Layouts” integrated with Triton; includes evaluation with micro-bens and real benchmarks (ASPLOS’26 per header). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **What it claims/guarantees:** Proposes a layout representation and integration in Triton; demonstrates performance/correctness improvements in that domain. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **What it does NOT say:** Not primarily about TileIR vs PTX spill behavior on Blackwell; not a spill-metric methodology paper.  
- **Why we care:** Neighbor work for “Triton backend/codegen charaization”; helps position novelty as spill/regs attribution vs layout modeling.

### P2 (ISL layout relations)
- **What it is:** Paper modeling CuTe layouts and Triton linear layouts using ISL integer set relations (abstract + methods). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **What it claims/guarantees:** Unifies layout abstractions mathematically for formal analysis and verification. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **What it does NOT say:** Not about register spillinmetrics or TileIR backend behavior.  
- **Why we care:** Related-work context if reviewers ask “isn’t this already solved?” → answer: that’s layouts, not spill attribution across backends.

### P3 (Categorical CuTe layouts)
- **What it is:** Paper on categorical framework for CuTe layout algebra; includes a Python implementation aligned with CUTLASS behavior (per abstract snippet). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **What it claims/guarantees:** Provides theoretical framework aerization of tractable layouts; reports implementation + tests alignment. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **What it does NOT say:** Not about TileIR backend or NCU-based spill measurement methodology.  
- **Why we care:** Another “layout math” neighbor to explicitly bracket as **not** our novelty.

---

## 6) Constraint Cliffs (max 10; microarch-focused)

- **K01:** If Nsight Compute cannot profile Tile workloads on the target system, we cannot collect spill/reg metrics for TileIR proposal fails**; require NCU with tile workload support (≥ 2025.4). ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai))  
- **K02:** If TileIR backend silently falls back to PTX, PTX-vs-TileIR comparisons become invalid → we must implement backend-detection checks (cache artifacts + logs + NCU metadata). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **K03:** If the TileIR upancy` hint cannot be set programmatically in our Triton version, H2 cannot be tested → must find the exact API or redesign around other knobs. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **K04:** If `launch__registers_per_thread` is treated as “live regs,” conclusions may be wrong due to allocation holes → always triangulate with spill counters + live-register source metrics when possible. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/mpute/index.html?utm_source=openai))  
- **K05:** If derived spill metrics are misread as “all local memory,” we will misattribute local arrays/structs as spills → always include SASS spill counters and local load/store metrics. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **K06:** If unordered memory model limitations affect correctness for a chosen workload, any performance data is untrustworthy → start with race-free microbenches and carefully  TritonBench ops. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
- **K07:** If descriptor/TMA rewrite changes algorithmic work or memory traffic patterns beyond the intended axis, attribution to “register spilling reduction” is weak → control math + inputs + outputs and report instruction/memory deltas. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- **K08 we compare across different GPUs (RTX vs B200) without holding clocks/limits constant, deltas can be dominated by SKU differences → primary A/B must be on **the same GPU**.  
- **K09:** If CUDA/Triton/TileIR repo versions drift, results aren’t reproducible → pin versions/commits and store full environment manifests. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- **K10:** If key NCU metrics arailable on the target GPU/tool build (metrics vary), evaluation plan must adapt (alternate metrics/sections) → must preflight via `--query-metrics`. ([developer.nvidia.com](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/?utm_source=openai))  

---

## 7) QUERY_PLAN (targeted; mapped to Claim_IDs)

**Goal:** eliminate remaining UNVERIFIED load-bearing gaps (mostly “how to set knobs / confirm metric availability for tile workloads”).

1. **Q01 (API for occupancy hint):** o set TileIR `occupancy` hint in Triton-to-tile-IR (code example / config field name / launch arg)?” → maps to **C015, H2**. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
2. **Q02 (NCU metric availability for tile workloads):** “Do `derived__local_spilling_requests` and `sass__inst_executed_register_spilling` collect for CUDA tile workloads in NCU 2025.4+? Any limitations?” → **C003, C004, C006**. ([forums.developer.nvidia.com](https://forums.developer.nvidiidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai))  
3. **Q03 (Backend detection):** “What is the most reliable way to detect TileIR vs PTX backend at runtime (report metadata / cache artifact / compilation log)?” → **C012, C023**. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
4. **Q04 (Fallback frequency/conditions):** “Under what conditions does TileIR backend fall back to PTX; is there a flag to disable fallback (fail-fast)?” → **C01b.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
5. **Q05 (TileIR functional correctness hazards):** “Concrete examples of aliasing/cross-block reduction scripts needing updates under unordered memory model; recommended safe subsets.” → **C016, K06**. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
6. **Q06 (num_warps limitation scope):** “Which ‘XXXNorm’ kernels (layernorm/rmsnorm/groupnorm?) are impacted; is there a workaround H4**. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
7. **Q07 (Metric triangulation best practice):** “NCU guidance on combining `launch__registers_per_thread` with Live Registers and spill counters; pitfalls.” → **C005, C018**. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai))  
8. **Q08 (Tile workload profiling workflow):** “NCU 2025.4 tile workload profiling: required flags/sections; any differenceT kernels.” → **C006**. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai))  
9. **Q09 (Blackwell-specific `num_ctas=2` guidance):** “What exactly is ‘2CTA mode MMA on Blackwell’; how to ensure we are in that mode; does it change register pressure?” → **C022**. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
10. **Q10 (Compute capability inventory):** “Confirm compute capability and SM counts for RTX 5090 / B200 / GB10; verify NCU support matrix.” → environment template completeness; supports **SC4/K01**. ([developer.nvidia.com](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/?utm_source=openai))  

---

# CONTEXT_CAPSULE

```yaml
STATE_VERSION: "manual_state_v4_2_tilespill@2026-02-04"
profile:
  project_name: "TileSpill: TileIR vs PTX register pressure & spilling (Blackwell microbench)"
  target_model: "gpt-5.2-pro (web UI)"
  opera_mode: "manual_state_v4_2_tilespill (web UI; milestone resets; monotonic capsule)"
  stage_plan: ["0", "1", "1.5", "2", "2.5", "3"]
  conference_targets: ["ISCA", "MICRO", "ASPLOS"]
  paper_genre: "microarchitecture + toolchain characterization proposal"
  hard_constraints:
    - "No hallucinated citations"
    - "No invented metric names/tool behavior"
    - "Microarch scope lock: bare-metal microbench + NCU + PTX vs TileIR A/B"
    - "Capsule monotonic: ledgers append-only; shrink forbidden unless justified with pointers"
  current_stage: 0
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
  stage1_gap_audit: null
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
  items: []

CLAIM_LEDGER:
  items:
    - id: "C001"
      scope_tag: "ACTIVE"
      claim: "CUDA local memory is thread-private and used when automatic variables don’t fit in registers or when register spilling occurs."
      status: "VERIFIED"
      evidence: ["E001"]
      paper_role: "A"
      risk_if_wrong: isclassify spill vs non-spill local traffic."
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
      evidence: E006"]
      paper_role: "A/C"
      risk_if_wrong: "No real-workload anchors."
    - id: "C021"
      scope_tag: "ACTIVE"
      claim: "TileIR backend disables approx and FTZ by default; can be enabled via TILEIR_ENABLE_APPROX=1 and TILEIR_ENABLE_FTZ=1."
      status: "VERIFIED"
      evidence: ["E005"]
      paper_role: "B"
      risk_if_wrong: "Confounded numeric/ISA differences."

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
  methodology:
    - "Bare-metal runs; warmup; repeat; report median + variability."
    - "Pin toolchain versions + env vars; treat backend toggles as experimental factors."
    - "Use NCU --query-metrics to confirm metric availability per version and per workload type (SIMT vs tile)."
    - "Detect TileIR vs PTX fallback explicitly (cache artifacts + logs + report metadata)."
  metrics:
    - "runtime"
    - "launch__registers_per_thread"
    - "derived__local_spilling_requests (+ pct)"
    - "sass__inst_executed_register_spilling (+ local_loads/local_stores)"
  baselines:
    - "PTX backend (ENABLE_TILE=0) vs TileIR backend (ENABLE_TILE=1) on same Blackwell GPU"
    - "TileIR occupancy hint sweep (1..32)"
    - "tensor-of-pointer vs descriptor/TMA rewrite (when applicable)"
  workloads:
    - "TritonBench subset (select ops with reduction/norm + GEMM motifs)"
    - "Custom microbench kernels sweeping register pressure"
  ablations:
    - "disable/enable backend-specific env vars (FTZ/approx) only as a controlled ablation"
  risks_to_validity:
    - "backend feature drift across CUDA/Triton versions"
    - "register count metric overstates live regs; interpret carefully"
    - "clock/frequency variability"
    - "TileIR backend fallback to PTX can invalidate A/B"

OPEN_QUESTIONS:
  active:
    - id: "OQ01"
      status: "OPEN"
      statement: "Exact API/syntax to set TileIR occupancy hint in Triton user code/config."
      blocks: ["H2", "C015"]
      plan: ["Q01"]
    - id: "OQ02"
      status: "OPEN"
      statement: "Which NCU spill/register metrics are collectable on CUDA tile workloads on Blackwell in NCU 2025.4+."
      blocks: ["C003", "C004", "C006", "SC4"]
      plan: ["Q02", "Q08"]
    - id: "OQ03"
      status: "OPEN"
      statement: "Best-practice backend detection to guarantee no TileIR->PTX fallback in measured runs."
      blocks: ["C012", "K02"]
      plan: ["Q03", "Q04"]
    - id: "OQ04"
      status: "OPEN"
      statement: "Scope/identities of 'XXXNorm' kernels impacted by missing num_warps; workaround options."
      blocks: ["H4", "C014"]
      plan: ["Q06"]
    - id: "OQ05"
      status: "OPEN"
      statement: "Blackwell-specific meaning and measurement of '2CTA mode MMA' tied to num_ctas=2."
      blocks: ["C022"]
      plan: ["Q09"]
    - id: "OQ06"
      status: "OPEN"
      statement: "Confirm compute capability + SM counts for RTX 5090, B200, GB10; validate NCU support matrix."
      blocks: ["Env template completeness", "K01"]
      plan: ["Q10"]
  closed: []

DELTA_LOG:
  - "Set profile.current_stage=0 and last_updated=2026-02-04."
  - "ARTIFACT_INDEX.stage0_fact_sheet set to WP0_20260204."
  - "Added CLAIM_LEDGER entries C001..C006, C009..C012, C014..C015, C020..C022."
  - "Added EVIDENCE_LEDGER entries E001..E009."
  - "Seeded OPEN_QUESTIONS OQ01..OQ06 (occupancy hint API, tile-metric availability, fallback detection, norm spill scope, 2CTA mode, GPU inventory)."

CAPSULE_HEALTH:
  counts:
    verdict_ledger: 0
    claim_ledger: 13
    evidence_ledger: 9
    experiment_ledger: 0
    open_questions_active: 6
    open_questions_closed: 0
  capsule_token_estimate: 3200
  monotonicity: "OK (stage0 baseline)"

NEXT_STAGE_HINT: "Start fresh chat; paste System Config + this capsule + Stage 1 prompt."
```

---
Learn more:
1. [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)
2. [Nsight Compute :: Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai)
3. [GitHub - triton-lang/Triton-to-tile-IR: incubator repo for CUDA-TileIR backend](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file)
4. [2. Profiling Guide — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)
5. [NVIDIA® Nsight™ Compute 2025.4 is now available - Nsight Compute - NVIDIA Developer Forums](httforums.developer.nvidia.com/t/nvidia-nsight-compute-2025-4-is-now-available/353614?utm_source=openai)
6. [2. Profiling Guide — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)
7. [Nsight Compute :: Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/2023.2/NsightCompute/index.html?utm_source=openai)
8. [7. Memory Model — Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html?utm_source=openai)
9. [GitHubeta-pytorch/tritonbench: Tritonbench is a collection of PyTorch custom operators with example inputs to measure their performance.](https://github.com/meta-pytorch/tritonbench?utm_source=openai)
10. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
11. [GitHub - NVIDIA/cuda-tile: CUDA Tile IR is an MLIR-based intermediate representation and compiler infrastructure for CUDA kernel optimization, focusing on tile-based computation patterns and opmizations targeting NVIDIA tensor core units.](https://github.com/NVIDIA/cuda-tile?utm_source=openai)
12. [CUDA Toolkit 13.1 - Release Notes — Release Notes 13.1 documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html?utm_source=openai)
13. [Using Nsight Compute to Inspect your Kernels | NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/?utm_source=openai)
14. [Linear Layouts: Robust Code Generation of Efficient Tsor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
15. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
16. [Categorical Foundations for CuTe Layouts](https://arxiv.org/pdf/2601.05972v1)
