# WORK_PRODUCT

## 0) Scope Freeze Card (must be first)

### Research question (frozen; 1 sentence)
**RQ0:** On **NVIDIA Blackwell** GPUs, how does compiling the *same* Triton kernels via **PTX backend vs CUDA Tile IR backend** change **register allocation and spilling**, and how do those changes translate into measurable **runtime + occupancy** trade-offs under controlled tuning knobs?

### Hypotheses (frozen; 2–4; each names sweeps + metrics)
- **H1 (Backend effect on spills):** Sweeping **backend ∈ {PTX, TileIR}** (via `ENABLE_TILE=0/1`) will cause statistically significant shifts in **`launch__registers_per_thread`**, **`sass__inst_executed_register_spilling{,_mem_local,_mem_shared}`**, **`derived__local_spilling_requests{,_pct}`**, and **kernel runtime (CUDA events)** on at least one Blackwell GPU. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- **H2 (TileIR occupancy-hint ↔ reg/spill trade-off):** Sweeping TileIR **`occupancy` hint N ∈ {1..32}** will measurably move **`launch__registers_per_thread`** and spill metrics (**`sass__inst_executed_register_spilling*`**, **`derived__local_spilling_requests*`**) and correlate with **achieved occupancy `sm__warps_active.avg.pct_of_peak_sustained_active`** and runtime. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- **H3 (Kernel-level pressure sweep reveals spill onset):** Sweeping **register-pressure knobs** (e.g., unroll factor / tile size / reduction dimension) will produce a **spill-onset threshold** (spill metrics transition from ~0 to >0) that is **backend-dependent**, observable via **`sass__inst_executed_register_spilling*`** and **`derived__local_spilling_requests*`**, with a corresponding runtime inflection. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **H4 (API style changes reg pressure):** For kernels where both are possible, switching **legacy tensor-of-pointer load/store** vs **descriptor/TMA-oriented path** will change **register allocation/spill metrics** and runtime; the direction may differ by backend. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  

### Excluded topics (frozen; 3–7 bullets)
- Functional correctness / memory-model legality (Tile IR tokens, ordering, data races) as a primary contribution. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
- New hardware proposals (no RTL / no “we need a new register file”).  
- Full-model training as evidence (only microbenches + operator benchmarks).  
- Claims that require undocumented compiler internals (we only use *observable artifacts*: NCU + emitted code + runtime).  
- Cross-vendor performance portability (AMD/Intel) as a main axis.  
- “PTX ISA semantics” beyond what is needed to interpret **local memory / stack / spills**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### Success criteria (frozen; 3–6 bullets)
- **SC1:** A reproducible A/B harness that can switch **PTX vs TileIR** and confirm which backend actually ran (no silent-mislabeled results). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- **SC2:** At least **1 strong finding** on Blackwell that is actionable for kernel writers/tuners, stated as: *“If you see X (metrics), do Y (knob/code change)”* and supported by NCU spill metrics + runtime. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **SC3:** A public-ish artifact slice (scripts + microbench subset + plot/table generator) producing at minimum: **regs/thread**, **spill inst**, **spill requests**, **achieved occupancy**, **runtime**. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **SC4:** Controls documented: toolchain versions, env vars, warmup/repeats, and math-mode alignment (approx/FTZ). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  

### Evaluation environment template (fields frozen; values may be unknown/TBD)
| Field | Value (TBD allowed) | Notes |
|---|---|---|
| Date captured | YYYY-MM-DD | Must be logged per run |
| Machine | hostname + CPU + RAM |  |
| OS / kernel |  |  |
| GPU | SKU(s): RTX 5090 / B200 / etc | record PCI IDs if possible |
| GPU compute capability / `sm_` | TBD | needed for `tileiras --gpu-name` style targets ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai)) |
| Driver version | TBD | TileIR stacks may require newer drivers (log exact) |
| CUDA toolkit | TBD (expect 13.1+ for TileIR backend in this repo) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | record `nvcc --version` & paths |
| Nsight Compute (NCU) | version + CLI path | metrics vary by version; must log |
| Python | version |  |
| PyTorch | version | TritonBench defaults to nightly by default install path ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai)) |
| Triton baseline | version/commit | for PTX backend baseline |
| Triton-to-tile-IR | commit hash | must be pinned |
| Backend toggle | `ENABLE_TILE` ∈ {0,1} ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | log effective value per run |
| TileIR math-mode toggles | `TILEIR_ENABLE_APPROX`, `TILEIR_ENABLE_FTZ` ∈ {0,1} ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | must be recorded (confound control) |
| TileIR tuning hint | `occupancy` ∈ {1..32} ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | must be recorded per kernel config |
| Triton tuning | `num_ctas` (incl. 1,2) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | important for Blackwell dot workloads |
| Runtime timing | CUDA events (median of N runs) | N >= 30 recommended for low-variance kernels |
| NCU metrics (minimum set) | see below | verify via `ncu --query-metrics` ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |

**Minimum metric set (frozen names):**  
`launch__registers_per_thread`, `launch__registers_per_thread_allocated`, `sass__inst_executed_register_spilling`, `sass__inst_executed_register_spilling_mem_local`, `sass__inst_executed_register_spilling_mem_shared`, `derived__local_spilling_requests`, `derived__local_spilling_requests_pct`, `sm__warps_active.avg.pct_of_peak_sustained_active`, plus `sass__inst_executed_local_loads` / `sass__inst_executed_local_stores` for context. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  

---

## 1) Table: Ground Truth Glossary  
*(A) Primary tooling facts + (B) TileIR stack facts; anything speculative is marked UNVERIFIED in Source_ID)*

| Term | Definition (1–2 lines) | Where used | Source_ID (or UNVERIFIED) | Notes |
|---|---|---|---|---|
| Register pressure | Demand for registers; high pressure can force the compiler to spill values to memory. | RQ0, H1–H4 | NV-NCU (partial) | Operationalized via spill metrics + regs/thread. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| `launch__registers_per_thread` | NCU per-launch metric: number of registers allocated per thread. | H1, H2, reporting | NV-NCU | Treat as **allocation**, not exact liveness. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| Live registers | NCU “Live Registers”: number of registers that must be kept valid at a code location; can differ from total allocated regs. | Threats/interpretation | NV-NCU | Used to avoid over-interpreting regs/thread. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2024.1/NsightCompute/index.html?utm_source=openai)) |
| Register spilling | Compiler-generated loads/stores due to insufficient registers; NCU exposes explicit spill instruction metrics. | Definition of “spill” | NV-NCU | Use spill-specific metrics, not just local ops. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| `sass__inst_executed_register_spilling` | NCU SASS metric: number of store+load instructions executed due to register spilling (with mem-local/shared breakdowns). | H1–H3, plots | NV-NCU | Primary spill signal. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| `derived__local_spilling_requests` | NCU derived source metric: executed instructions + L1 requests made for register spilling to local memory (+ `%` variant). | H1–H3, plots | NV-NCU | “Requests” lens complements instruction counts. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) |
| Local memory / PTX `.local` | PTX `.local` is per-thread private memory; under ABI it is stack-allocated; accessed with `ld.local`/`st.local`. | Interpretation of “local” | NV-PTX | Local memory != always spill (could be explicit locals). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) |
| `sass__inst_executed_local_loads/stores` | NCU SASS metrics: number of local memory load/store instructions executed. | Context metrics | NV-NCU | Useful to separate “local traffic” vs “spill-specific traffic”. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| Achieved occupancy | NCU metric `sm__warps_active.avg.pct_of_peak_sustained_active` (maps from nvprof “achieved_occupancy” for SM≥7.0). | H2 reporting | NV-NCU | Runtime occupancy signal; not always perf predictor. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightComputeCli/index.html?utm_source=openai)) |
| TileIR backend (Triton-to-tile-IR) | Triton backend that targets CUDA Tile IR (instead of PTX), enabled in this repo via `ENABLE_TILE=1`. | Setup + baselines | TILE-REPO | Must detect fallback-to-PTX. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| `ENABLE_TILE` | Environment variable used by Triton-to-tile-IR repo to enable the CUDA Tile IR backend. | Baseline switch | TILE-REPO | A/B factor for H1. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| TileIR `occupancy` hint | TileIR backend hint: integer **1..32**, default 1, expressing expected active CTAs per SM. | H2 | TILE-REPO | A/B/C knob for microbench tuning. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| TileIR approx/FTZ toggles | Repo states TileIR disables approx and FTZ by default; env vars enable them. | Controls | TILE-REPO | Must avoid confounding vs PTX baseline. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |

---

## 2) Table: Tooling & Evaluation Baseline Map  
*(A) PTX/NCU/TritonBench + (B) TileIR stack facts)*

| Source_ID | What it is | What it measures/guarantees | Key limitations | Evidence (cite or UNVERIFIED) |
|---|---|---|---|---|
| **NV-PTX** | NVIDIA PTX ISA documentation | Defines PTX state spaces like `.local` and stack behavior under ABI | PTX docs do **not** tell you final SASS register allocation/spills; only semantics | ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) |
| **NV-NCU** | Nsight Compute Profiling Guide / metric reference | Defines metric names like `launch__registers_per_thread`, `sass__inst_executed_register_spilling`, `derived__local_spilling_requests*` | Some metrics are derived; SASS patching / replay can perturb; `launch__registers_per_thread` can overstate live regs | ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| **NV-NCU-CLI** *(not in GOLDEN_SOURCES list but used as evidence)* | Nsight Compute CLI docs (metric mapping) | Confirms achieved occupancy mapping: `sm__warps_active.avg.pct_of_peak_sustained_active` | Metric availability depends on chip + NCU version; must query on target machine | ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightComputeCli/index.html?utm_source=openai)) |
| **TB-1** | TritonBench repo (PyTorch operator benchmarks) | Provides a suite + harness to benchmark operators (e.g., `run.py --op gemm`) | Not a microbench; complex kernels confound attribution; needs subset selection | ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai)) |
| **TILE-REPO** | Triton-to-tile-IR incubator repo | Documents backend switch `ENABLE_TILE=1`, fallback behavior, new hint `occupancy` (1..32), and known spill-related issue note | Early-stage; functional/perf issues; only Blackwell supported in CUDA 13.1 per README | ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| **NV-TILE-BLOG** | NVIDIA technical blog on Triton-to-TileIR | High-level description + roadmap; confirms env-var based backend switch concept | Blog-level; not a spec; may omit exact knobs/edge cases | ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)) |
| **CUDA-TILE-REPO** | NVIDIA/cuda-tile repo | Documents producing Tile IR bytecode and compiling with `tileiras` or JIT | Not directly about register spilling; mainly toolchain plumbing | ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai)) |
| **NV-TILEIR-MM** | Tile IR memory model spec | Normative semantics; notes relationship to PTX memory model | Not directly about spill/regs; relevant only for scope exclusion | ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |

---

## 3) Table: SOTA Baseline Map (microarch-specific)  
*(C) Baselines; focus is what we compare against, not “prior papers”)*

| Baseline_ID | Baseline description | Why it is a fair comparator | What it cannot explain | Evidence |
|---|---|---|---|---|
| **B1** | **Same kernel**, same inputs, same GPU, switch backend via `ENABLE_TILE=0/1` | Isolates backend/codegen path as primary factor | Silent fallback can contaminate “TileIR” bucket; must detect | TileIR enable + fallback described in repo README ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| **B2** | TileIR backend, sweep **`occupancy` hint = 1..32** | Directly targets a TileIR-specific tuning knob; tests reg/spill ↔ occupancy tradeoffs | Does not guarantee “best” schedule; might trade other bottlenecks | Occupancy hint range+default documented ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| **B3** | Sweep register-pressure knobs (unroll/tile/reduction dim) within each backend | Lets us map spill onset curves and attribute backend differences | Kernel becomes different algorithmically if sweep changes memory traffic too much | (Method) must be designed in microbench suite; no external evidence needed |
| **B4** | Use Triton hint **`num_ctas` ∈ {1,2}** where applicable | README claims `num_ctas=2` is critical for some Blackwell dot workloads; fair to include as tuned baseline | May change kernel structure/latency hiding beyond spills | Repo claim about `num_ctas=2` & 2CTA MMA ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| **B5** | “Local memory traffic” metrics vs “spill-specific” metrics | Prevents false inference that all local ops are spills | Still indirect; doesn’t tell *which* values spilled | NCU exposes both local load/store counts and spill-specific counts ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) |
| **B6** | TritonBench subset vs custom microbenches | Balances realism (operator kernels) with attribution (microbenches) | TritonBench kernels may be too complex to isolate mechanisms | TritonBench scope/usage from README ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai)) |

---

## 4) Table: Claim Ledger v0 (15–30 claims)  
*(A) Primary tooling facts; (B) TileIR stack facts; (D) INFERENCE explicitly labeled in Status)*

| Claim_ID | Claim (1 sentence) | Status | Evidence pointers | Paper role | Risk if wrong |
|---|---|---|---|---|---|
| **C01** | PTX `.local` is per-thread private memory; under ABI it is stack-allocated and accessed via `ld.local`/`st.local`. | **VERIFIED** | NV-PTX ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | Definition (local/stack) | Misinterpret “local” metrics/spills |
| **C02** | NCU distinguishes **local memory load/store instruction counts** from **spill-caused instruction counts**, so “local traffic” is not automatically “spilling.” | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Measurement interpretation | Over/under-count spills |
| **C03** | NCU provides `launch__registers_per_thread` (and `_allocated`) as per-kernel launch register allocation metrics. | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Core metric hook | No consistent reg metric |
| **C04** | NCU warns `launch__registers_per_thread` can be higher than maximum live registers due to holes/ABI/hardware constraints. | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.1/NsightCompute/index.html?utm_source=openai)) | Threats-to-validity guardrail | Wrong interpretation of “regs” |
| **C05** | NCU provides `sass__inst_executed_register_spilling` plus mem-local/shared breakdown spill metrics. | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Primary spill signal | No reliable spill indicator |
| **C06** | NCU provides `derived__local_spilling_requests` and `_pct` as spill-to-local request metrics. | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)) | Secondary spill signal | Miss spills that manifest as requests |
| **C07** | Achieved occupancy metric name in NCU is `sm__warps_active.avg.pct_of_peak_sustained_active` (SM≥7.0 mapping). | **VERIFIED** | NV-NCU-CLI ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightComputeCli/index.html?utm_source=openai)) | Occupancy control/response | Occupancy axis mis-measured |
| **C08** | NCU exposes theoretical occupancy via metrics including `sm__maximum_warps_per_active_cycle_pct`. | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Context metric | Wrong “theoretical” baseline |
| **C09** | NCU supports enumerating metric availability via `ncu --query-metrics`. | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Feasibility gate | Scripts break across versions |
| **C10** | In Triton-to-tile-IR repo, users enable the CUDA Tile IR backend by setting `ENABLE_TILE=1`. | **VERIFIED** | TILE-REPO ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Experimental factor definition | No real A/B switch |
| **C11** | Repo states: when a compilation bug occurs with TileIR backend, it falls back to the NVIDIA PTX backend. | **VERIFIED** | TILE-REPO ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Validity threat | Mislabel TileIR runs |
| **C12** | Repo states TileIR backend uses CUDA 13.1 features only and supports only Blackwell GPUs in CUDA 13.1. | **VERIFIED** | TILE-REPO ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Feasibility constraint | Can’t run A/B on target |
| **C13** | Repo introduces TileIR backend hint `occupancy` accepting integers 1..32 (default 1). | **VERIFIED** | TILE-REPO ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Key tuning knob | No tunable knob → weak paper |
| **C14** | Repo states TileIR backend disables approx + FTZ by default; env vars `TILEIR_ENABLE_APPROX` / `TILEIR_ENABLE_FTZ` enable them. | **VERIFIED** | TILE-REPO ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Control variable definition | Confounded perf comparisons |
| **C15** | Repo states `num_warps` is not exposed yet (CUDA 13.1) and some large-reduction norm kernels may spill registers and lose performance. | **VERIFIED** | TILE-REPO ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Motivation for spills focus | Wrong target symptom |
| **C16** | Repo claims `num_ctas=2` is critical for dense dot workloads because it enables 2CTA mode MMA on Blackwell. | **VERIFIED (as “repo claims”)** | TILE-REPO ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Baseline knob | Miss key baseline; unfair eval |
| **C17** | NVIDIA/cuda-tile describes producing Tile IR bytecode with `cuda-tile-translate` and compiling AoT with `tileiras` (or JIT via driver API). | **VERIFIED** | CUDA-TILE-REPO ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai)) | Artifact plumbing | Can’t inspect/attribute codegen |
| **C18** | Our paper will define “spill” primarily via **spill-specific NCU metrics** (`sass__inst_executed_register_spilling*`, `derived__local_spilling_requests*`), not by local loads/stores alone. | **INFERENCE (design choice)** | C05–C06 motivate feasibility ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Definition of outcome variables | Ambiguous results / reviewer pushback |
| **C19** | Any A/B pipeline must log a **backend-used** indicator to detect TileIR→PTX fallback and exclude/mask those runs. | **INFERENCE** | C11 motivates need ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Validity guardrail | False conclusions from mixed buckets |
| **C20** | Because TileIR defaults disable approx/FTZ, we must align math-mode settings across backends or report them as an explicit ablation. | **INFERENCE** | C14 ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Confound control | “Perf difference is precision” |
| **C21** | Pressure sweeps (unroll/tile/reduction) will show a measurable spill onset when spill metrics move from ~0 to >0. | **INFERENCE** | Uses NCU spill metrics (C05–C06) ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Microbench design goal | No clean curves → weak claims |
| **C22** | Spill onset threshold and slope will differ between PTX and TileIR backends for at least one motif. | **INFERENCE** | — | Novelty target | Null result risk |
| **C23** | Sweeping TileIR `occupancy` hint will change reg allocation/spilling and correlate with achieved occupancy and runtime for compute-heavy kernels. | **INFERENCE** | C13 + occupancy metric name C07 ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Key tuning finding | “Occupancy hint irrelevant” |
| **C24** | Switching tensor-of-pointer vs descriptor/TMA paths can change register pressure/spills due to address materialization differences. | **INFERENCE** | Repo notes tensor-of-pointer APIs are a known perf issue ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Mechanism hypothesis | Wrong attribution |
| **C25** | NCU exposes a “Live Registers” view distinct from `launch__registers_per_thread`, so we can avoid equating allocated regs with peak live regs. | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2024.1/NsightCompute/index.html?utm_source=openai)) | Threats-to-validity mitigation | Reg story collapses |
| **C26** | NCU includes `launch__stack_size`, which can help track stack usage (often related to `.local` allocations). | **VERIFIED** | NV-NCU ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)) | Auxiliary metric | Miss stack/local confound |

---

## 5) Golden Snapshot (Carry-Forward)  
*(3–6 bullets each; keep as stable anchors)*

### NV-NCU — Nsight Compute Profiling Guide
- **What it is:** Official NCU docs + metric reference. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **Claims/guarantees:** Defines metric names for regs/thread and spill-related metrics (spill instruction counts and spill request metrics). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
- **Does NOT say:** It does not prove causality (“why did the compiler spill?”), only measurement surfaces.  
- **Why we care:** It is the **measurement contract** for our microarch paper (metric names must match). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  

### NV-PTX — PTX ISA docs
- **What it is:** Normative PTX ISA reference. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Claims/guarantees:** Defines `.local` as per-thread private memory and stack allocation under ABI. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- **Does NOT say:** How PTX maps to SASS or actual register allocation decisions.  
- **Why we care:** Ground truth for interpreting “local memory” and stack-related effects. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

### NV-TILEIR-MM — Tile IR memory model
- **What it is:** Normative Tile IR memory model spec (semantics). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
- **Claims/guarantees:** Defines scope/order concepts and relation to PTX memory model. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
- **Does NOT say:** Anything about register allocation/spilling.  
- **Why we care:** **Scope boundary**: we explicitly exclude correctness/ordering from primary contribution. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  

### NV-TILE-BLOG — NVIDIA blog on Triton-to-TileIR
- **What it is:** High-level overview and roadmap of Triton-to-TileIR backend. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- **Claims/guarantees:** Positions backend as a switchable alternative to PTX and discusses benchmarking/validation as workstreams. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
- **Does NOT say:** Exact env var names or full tuning API surface.  
- **Why we care:** External motivation + legitimacy for PTX vs TileIR A/B benchmarking. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  

### TILE-REPO — triton-lang/Triton-to-tile-IR
- **What it is:** Incubator repo adding CUDA Tile IR backend to Triton. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- **Claims/guarantees:** `ENABLE_TILE=1` enables TileIR backend; fallback to PTX on compile bug; new `occupancy` hint 1..32; notes spill risk in some kernels; documents approx/FTZ toggles. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- **Does NOT say:** That its defaults are “fair” for perf comparisons (we must control).  
- **Why we care:** This is our **A/B mechanism** and also directly flags spilling as an issue worth measuring. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  

### CUDA-TILE-REPO — NVIDIA/cuda-tile
- **What it is:** Open-source CUDA Tile IR dialect/tools repo. ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai))  
- **Claims/guarantees:** Documents `cuda-tile-translate` → bytecode and `tileiras` → cubin (AoT), plus JIT loading. ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai))  
- **Does NOT say:** How generated code affects register pressure.  
- **Why we care:** Artifact extraction/inspection pathway if we need to debug codegen deltas. ([github.com](https://github.com/NVIDIA/cuda-tile?utm_source=openai))  

### TB — meta-pytorch/tritonbench
- **What it is:** Benchmark suite of PyTorch operators with example inputs to evaluate Triton performance. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **Claims/guarantees:** Provides install + operator-level benchmark runner; includes multiple operator sources/submodules. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- **Does NOT say:** Anything about register spilling directly; we must instrument.  
- **Why we care:** Realistic workload slice to complement microbenches. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

### P1 — Linear Layouts (Triton) (seed paper)
- **What it is:** Seed paper listed in project state.  
- **Claims/guarantees:** **UNVERIFIED in this stage** (not used for microarch spill scope yet).  
- **Does NOT say (expected):** Likely not about TileIR vs PTX spill characterization (to be checked later).  
- **Why we care:** Possible background if layout choices impact register pressure (future, optional).

### P2 — ISL layout relations (seed paper)
- Same status as P1 (kept, but not scope-critical in Stage 0).

### P3 — Categorical CuTe layouts (seed paper)
- Same status as P1 (kept, but not scope-critical in Stage 0).

---

## 6) Constraint Cliffs (max 10; microarch-focused)
1) **If** TileIR runs can silently fall back to PTX, **then** we must implement a backend-used detector (or invalidate those runs), or the whole A/B study collapses. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
2) **If** TileIR defaults differ in approx/FTZ vs PTX backend defaults, **then** we must align math modes or treat them as ablations to avoid confounded “performance wins.” ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
3) **If** spill-specific metrics (`sass__inst_executed_register_spilling*`, `derived__local_spilling_requests*`) are unavailable on Blackwell/our NCU version, **then** we must revise the metric set (or the spill focus fails). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
4) **If** `launch__registers_per_thread` is used as the only “reg pressure” metric, **then** reviewer skepticism is justified (NCU warns it can overstate live regs); we must report spill metrics and/or live registers. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.1/NsightCompute/index.html?utm_source=openai))  
5) **If** our kernels are memory-bound, **then** spill effects may be masked; we must include compute-heavy motifs where spills clearly matter.  
6) **If** TileIR `occupancy` hint does not materially affect codegen, **then** H2 fails; we must find alternate controllable knobs or re-scope contributions. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
7) **If** `num_warps` remains unavailable in the TileIR backend we evaluate, **then** we must not claim parity with PTX tuning knobs; instead, we measure consequences and recommend alternatives. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
8) **If** TritonBench kernels are too complex to attribute spill mechanisms, **then** we must rely on microbench motifs and use TritonBench only as a “realism check.” ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
9) **If** we cannot pin toolchain versions (CUDA/driver/Triton/NCU), **then** results won’t be reviewable/reproducible (MICRO/ISCA bar).  
10) **e can’t produce plots/tables within ~2–4 weeks (MVP), **then** the project is not feasible on schedule (requires scope tightening).

---

## 7) QUERY_PLAN (targeted; map queries → Claim_IDs / hypotheses)

1) **Q1:** Confirm compute capability / `sm_` targets for **RTX 5090, B200, GB10** and what `tileiras --gpu-name sm_XXX` should be. → (env template, C12, C17)  
2) **Q2:** Determine how to **programmatically detect** “TileIR backend used vs PTX fallback” (logs, artifact type, compiler flag, emignature). → (C11, C19, SC1) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
3) **Q3:** Locate the exact **Triton API surface** for setting TileIR **`occupancy` hint** (decorator arg? meta? attribute?) and how to sweep it per kernel. → (C13, H2) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
4) **Q4:** Verify PTX backend defaults for **approx/FTZ**, and define a **comparison protocol** that either aligns or reports mode deltas. → (C14, C20) ([github.com](https://gcom/triton-lang/Triton-to-tile-IR))  
5) **Q5:** On the target NCU version, run `ncu --query-metrics` on Blackwell to confirm availability of the **minimum metric set** (spill metrics + achieved occupancy). → (C05–C09; feasibility gate) ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
6) **Q6:** Validate that `sass__inst_executed_register_spilling_mem_local` and `_mem_shared` behave as expected on Blackwell (nonzero only when spill happens). → (C05, C18) .nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
7) **Q7:** Identify a minimal microbench kernel family that can **monotonically sweep register pressure** without changing memory traffic too much (e.g., controlled unroll + register arrays). → (H3, C21)  
8) **Q8:** Determine whether TileIR backend’s “tensor-of-pointer APIs are slow” correlates with reg pressure/spills vs other factors (e.g., memory coalescing), to avoid wrong attribution. → (C24, H4) ([githtps://github.com/triton-lang/Triton-to-tile-IR))  
9) **Q9:** Confirm how `num_ctas=2` is set in Triton for relevant ops and how to record it in artifacts; determine if it changes spill metrics in practice. → (C16, B4, H* controls) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
10) **Q10:** Check whether newer CUDA releases beyond 13.1 expose `num_warps` for TileIR backend; decide whether scope stays pinned to CUDA 13.1 for determinism. → (C15, feasibility)

---

# CONTEXT_CAPSULE

`ml
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

  current_stage: 0
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
        metrics: ["launch__reers_per_thread", "derived__local_spilling_requests_pct", "sm__warps_active.avg.pct_of_peak_sustained_active", "runtime_us_cuda_events"]
      - id: "H3"
        statement: "Register-pressure parameter sweeps show spill-onset thresholds; thresholds differ across backends."
        sweeps: ["unroll factor", "tile size", "reduction dim"]
        metrics: ["sass__inst_executed_register_spilling*", "derived__local_spilling_requests*", "runtime_us_cuda_events"]
      - id: "H4"
        statement: "Tensor-of-pointer vs descriptor/TMA-oriented path changes reg pressure/spills and runtime; direction may be backend-dependent."
        sweeps: ["API style: tensor-of-pointer vs descriptor/TMA (when applicable)"]
        metrics: ["launch__registers_per_thread", "sass__inst_executed_register_spilling*", "runtime_us_cuda_events"]

    primary_knobs_to_sweep:
      - "backend: PTX vs TileIR (ENABLE_TILE=0/1)"
      - "TileIR occupancy hint: occupancy=1..32 (per-kernel hint; default 1)"
      - "Triton hint: num_ctas ∈ {1, (Blackwell dot/dense workloads control)"
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
       c: null
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

CLAIM_LEDGER:
  items:
    - id: "C01"
      scope_tag: "ACTIVE"
      claim: "PTX .local is per-thread private memory; under ABI it is stack-allocated; accessed via ld.local/st.local."
      status: "VERIFIED"
      evidence: ["NV-PTX"]
    - id: "C02"
      scope_tag: "ACTIVE"
      claim: "NCU separates local load/store counts from spill-caed instruction counts; local traffic is not automatically spills."
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
      claim: "Must align/report math modes (approx/FTZ) across backends as confound control.      status: "INFERENCE"
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
  items: []

EVAL_PLAN:
  status: "draft"
  methodology:
    - "Bare-metal runs; warmup; repeat; report median + variability."
    - "Pin toolchain versions + env vars; treat backend toggles as experimental factors."
    - "Use ncu --query-metrics to confirm metric availability per version."
    - "Guardrail: detect TileIR->PTX fallback and treat as separate bucket."
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
  baselines:
    - "PTX backend (ENABLE_TILE=0) vs TileIR backend (ENABLE_TILE=1) on same Blackwell GPU"
    - "TileIR occupancy hint sweep (occupancy=1..32)"
    - "Triton num_ctas sweep (1 vs 2) where relevant"
  workloads:
    - "TritonBench subset (select ops with reduction/norm + GEMM motifs)"
    - "Custom microbench kernels sweeping register pressure"
  ablations:
    - "TILEIR_ENABLE_APPROX=0/1 (controlled)"
    - "TILEIR_ENABLE_FTZ=0/1 (controlled)"
  risks_to_validity:
    - "backend feature drift across CUDA/Triton versions"
    - "register count metric overstates live regs; interpret carefully"
    - "silent TileIR->PTX fallback contaminates A/B comparisons"
    - "clock/frequency variability"

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

OPEN_QUESTIONS:
  active:
    - id: "OQ01"
      question: "What are the exact compute capabilities / sm_ targets for RTX 5090, B200, GB10, and which tileiras --gpu-name values should be used?"
      impacts: ["C12", "C17", "env_template"]
      query_plan_ref: "Q1"
    - id: "OQ02"
      question: "How do we robustly detect per-kernel whether TileIR backend was used vs fell back to PTX (no silent contamination)?"
      impacts: ["C11", "C19", "SC1"]
      query_plan_ref: "Q2"
    - id: "OQ03"
      question: "What is the exact Triton API to set/sweep TileIR occupancy hint (occupancy=1..32) per kernel config?"
      impacts: ["C13", "H2"]
      query_plan_ref: "Q3"
    - id: "OQ04"
      question: "What are PTX backend defaults for approx/FTZ, and how do we align/report math modes across backends?"
      impacts: ["C20"]
      query_plan_ref: "Q4"
    - id: "OQ05"
      question: "On our Blackwell machines/NCU version, are all minimum spill/occupancy metrics available and stable via ncu --query-metrics?"
      impacts: ["C05", "C06", "C07", "C09"]
      query_plan_ref: "Q5"
  closed: []

DELTA_LOG:
  - "Set profile.current_stage=0; last_updated=2026-02-04"
  - "Added hypotheses H1-H4"
  - "Seeded CLAIM_LEDGER with C01-C26"
  - "Verified and recorded evidence E01-E07; updated GOLDEN_SOURCES last_verified for NV-NCU/NV-PTX/NV-TILEIR-MM/NV-TILE-BLOG/TILE-REPO/CUDA-TILE-REPO/TB"
  - "Added OPEN_QUESTIONS OQ01-OQ05"
  - "Set ARTIFACT_INDEX.stage0_fact_sheet=WP0_20260204"

CAPSULE_HEALTH:
  counts:
    verdicts: 2
    claims: 26
    evidence_items: 7
    experiments: 0
    open_questions_active: 5
  capsule_token_estimate: 3100
  notes:
    - "Stage 0 capsule created (no shrink constraints apply yet)."

NEXT_STAGE_HINT: "Start fresh chat; paste System Config + this capsule + Stage 1 prompt."
STATE_END
```

---
Learn more:
1. [GitHub - triton-lang/Triton-to-tile-IR: incubator repo for CUDA-TileIR backend](https://github.com/triton-lang/Triton-to-tile-IR)
2. [2. Profiling Guide — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai)
3. [7. Memory Model — Tile IR](https://docsdia.com/cuda/tile-ir/latest/sections/memory_model.html)
4. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
5. [GitHub - NVIDIA/cuda-tile: CUDA Tile IR is an MLIR-based intermediate representation and compiler infrastructure for CUDA kernel optimization, focusing on tile-based computation patterns and optimizations targeting NVIDIA tensor core units.](https://github.com/NVIDIA/cuda-tile?utm_source=openai)
6. [GitHub - meta-pytorch/tritonbenchTritonbench is a collection of PyTorch custom operators with example inputs to measure their performance.](https://github.com/meta-pytorch/tritonbench?utm_source=openai)
7. [3. Nsight Compute — NsightCompute 12.4 documentation](https://docs.nvidia.com/nsight-compute/2024.1/NsightCompute/index.html?utm_source=openai)
8. [2. Profiling Guide — NsightCompute 13.1 documentation](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html?utm_source=openai)
9. [Nsight Compute CLI :: Nsight Computeumentation](https://docs.nvidia.com/nsight-compute/2022.4/NsightComputeCli/index.html?utm_source=openai)
10. [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton | NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai)
11. [Nsight Compute :: Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/2022.1/NsightCompute/index.html?utm_source=openai)
