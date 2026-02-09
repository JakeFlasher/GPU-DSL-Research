## WORK_PRODUCT

### 0) Scope Freeze Card (must be first)

**RQ (frozen, 1 sentence).**  
On NVIDIA Blackwell GPUs using CUDA ≥ 13.1, how does Triton’s CUDA Tile IR backend change **register allocation + register spilling (local memory traffic)** vs Triton’s PTX backend, and which backend-specific knobs (e.g., `occupancy`, `num_ctas`, approx/FTZ toggles) causally govern that shift as observed in Nsight Compute spilling metrics? ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html))

**H1 (backend A/B spill onset).**  
Sweep **(V)** `ENABLE_TILE ∈ {0,1}` × **(V)** register-pressure ladder parameter `K` (e.g., live-value count / unroll factor) and observe **(M)** `launch__registers_per_thread`, `launch__occupancy_limit_registers`, `sass__inst_executed_register_spilling_mem_local`, `derived__local_spilling_requests_pct`; **prediction:** spill metrics rise at different `K` thresholds for TileIR vs PTX. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))

**H2 (TileIR occupancy hint governs spills).**  
With TileIR enabled, sweep **(V)** TileIR `occupancy ∈ {1..32}` × **(V)** `num_ctas ∈ {1,2}` and observe **(M)** the same spill + register metrics; **prediction:** higher requested concurrent residency (occupancy hint) shifts register allocation/spills (trade-off vs parallelism), visible in `launch__registers_per_thread` and spill instruction counts. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))

**H3 (tensor-of-pointer vs TMA-descriptor changes reg pressure).**  
On TileIR backend, sweep **(V)** code pattern `{tensor-of-pointer, TMA descriptor rewrite}` × **(V)** tile sizes (e.g., `BLOCK_M/N/K`) and observe **(M)** `launch__registers_per_thread` + spill metrics; **prediction:** descriptor rewrite reduces pointer-materialization pressure, reducing spills and local-memory instruction counts. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))

**H4 (TileIR approx/FTZ toggles change codegen → spills).**  
On TileIR backend, sweep **(V)** `TILEIR_ENABLE_APPROX ∈ {0,1}` × **(V)** `TILEIR_ENABLE_FTZ ∈ {0,1}` and observe **(M)** registers/spills + runtime; **prediction:** math-mode toggles change instruction selection and can measurably change register count and spill volume in pressure-sensitive kernels. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))

**Excluded topics (frozen).**
- Any tile-backend claims on non‑Blackwell GPUs (non-goal; can be control baseline only). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html))  
- New compiler/IR designs without measurement hooks (we only characterize + attribute).  
- Full model training as evidence (too confounded).  
- Simulation-only evidence.  
- Non-bare-metal shared-counter environments as primary evidence (MIG/vGPU/MPS only as explicitly labeled threats-to-validity if unavoidable). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))

**Success criteria (frozen).**
- SC1: Produce at least one **register-pressure ladder plot** showing backend-dependent spill onset (TileIR vs PTX) using NCU spill metrics + register counts. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- SC2: Demonstrate a **causal knob** (TileIR `occupancy`, `num_ctas`, approx/FTZ) that shifts spill metrics in the expected direction with repeatable magnitude. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- SC3: Provide an evaluation template that pins toolchain + metrics and fails closed on backend provenance ambiguity (TileIR vs PTX). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- SC4: For at least one TritonBench operator, attribute a performance delta to a measurable change in register/spill behavior (not just “it’s faster”). ([github.com](https://github.com/meta-pytorch/tritonbench))

**Evaluation environment freeze template (fields + commands to record; fields may be unknown now).**

**(F) Hardware/driver**
- GPU(s): `nvidia-smi -L` (record PCI IDs), `nvidia-smi -q` (record driver, clocks, power limits).  
- Compute capability / SM version: record from a CUDA sample like `deviceQuery` (store full output). *(UNVERIFIED here; must be captured on the machine — see OQ/C022).*

**(F) Toolchain versions (must satisfy min constraints)**
- CUDA Toolkit version: `nvcc --version` (must be ≥ 13.1). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html))  
- Nsight Compute: `ncu --version` (must be ≥ 2025.4).  
- PTX ISA version (normative refs): record from PTX docs as 9.1 and keep URL pinned. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- Tile toolchain presence: record absolute paths + hashes for `tileiras`, `ptxas`, `libnvvm.so` (TileIR backend depends on these from CUDA 13.1). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html))  

**(F) Repo + Python stack**
- `python --version`, `pip freeze`, `pip show triton` (or PyTorch-bundled Triton).  
- Pin repos by commit: `git rev-parse HEAD` for Triton-to-TileIR + TritonBench + Triton fork. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  

**(F) Backend/tuning knobs recorded per run**
- `ENABLE_TILE` (requested backend). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
- TileIR knobs (if used): `TILEIR_ENABLE_APPROX`, `TILEIR_ENABLE_FTZ`, TileIR `occupancy`, Triton `num_ctas`. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- NCU collection recipe: `--set` / `--section` / `--metrics` list, plus `--query-metrics` snapshot to avoid typos/renames. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  

---

### 1) Table: Ground Truth Glossary

| Term | Definition (1–2 lines) | Where used | Source_ID (or UNVERIFIED) | Notes |
|---|---|---|---|---|
| `.local` state space | PTX local state space is private per-thread memory; accessed via `ld.local`/`st.local`. | Defining “local memory” and spill target | NV-PTX-9.1 | PTX local definition + access ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) |
| Local memory (CUDA) | Thread-private memory in device memory; used when automatic variables don’t fit registers or for other compiler reasons (incl. register spilling). | “Spill” meaning + runtime symptom | NV-NCU-2025.4 | Also notes local has global-like latency; arranged for coalescing. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| Register spilling | Condition where registers exceed availability and values are spilled (e.g., to local memory). | Core phenomenon measured | NV-NCU-2025.4 | Explicitly called out as “register spilling.” ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| Spill instructions (NCU) | NCU exposes counts of executed instructions due to spilling (e.g., to local memory). | Primary dependent variable | NV-NCU-2025.4 | Use `sass__inst_executed_register_spilling_mem_local`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| Spill request metrics (derived) | NCU exposes derived metrics for spill-related requests to L1 (e.g., `derived__local_spilling_requests` and pct). | Cross-check spill signal vs instruction counts | NV-NCU-2025.4 | Use both absolute + pct where possible. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| Registers per thread | Static launch metric: number of registers allocated per thread (`launch__registers_per_thread`). | Register pressure axis | NV-NCU-2025.4 | Also have occupancy-limit metrics due to registers. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| Occupancy limit due to registers | NCU metric: `launch__occupancy_limit_registers`. | Mechanism link reg→occupancy | NV-NCU-2025.4 | Helps attribute perf deltas to reg pressure. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| Stack size | NCU metric `launch__stack_size` (stack during launch). | Secondary symptom of local/stack use | NV-NCU-2025.4 | Not spill-specific (arrays/ABI can also use stack). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| CUDA Tile | CUDA 13.1 introduces CUDA Tile (CUDA Tile IR + cuTile); initial release targets Blackwell. | Toolchain scope lock | CUDA-13.1-RN | Key “Blackwell-only” guardrail. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) |
| Tile IR | Tile IR is a portable tile VM/instruction set; spec is versioned (Spec 13.1). | Alternative backend semantics | NV-TILEIR-MM | Use versioned docs (13.1). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/index.html)) |
| `tileiras` | CUDA 13.1 introduces `tileiras` translating Tile IR bytecode → SASS. | Backend provenance + compilation pipeline | CUDA-13.1-RN | Tool exists; still need to record installed binary hash. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) |
| Tile IR memory model ↔ PTX | Tile IR memory model is derived from PTX; intended as a strict weakening; defines PTX interoperability rules. | Correctness constraints; avoid aliasing hazards | NV-TILEIR-MM | Potential confound in real kernels; microbenches should avoid aliasing. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html)) |
| Unordered/weak defaults (TileIR backend repo) | Repo notes unordered memory model by default; tokens exist for ordering control; aliasing/cross-tile-block reductions can be incorrect without care. | Correctness guardrails for benchmarks | TILE-REPO | Treat as “implementation note,” not full spec. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| `ENABLE_TILE` | Env var used to enable Tile IR backend (switch compilation pipeline); blog uses `export ENABLE_TILE=1`. | A/B toggle (PTX vs TileIR) | NV-TILE-BLOG / TILE-REPO | Also used to detect cache artifacts. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| `.tileIR` cache artifacts | When Tile IR backend active, Triton caches kernels with `.tileIR` extensions (blog claim). | Backend provenance heuristic | NV-TILE-BLOG | Still need to validate on our install path (empirical). ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| TileIR `occupancy` hint | Tile backend adds `occupancy` hint: integer N from 1 to 32 active thread blocks per SM. | Independent variable in H2 | TILE-REPO | `num_warps` not exposed yet (changes sweep design). ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| `num_ctas` hint | Existing Triton hint; repo claims `num_ctas=2` critical for dense dot workloads on Blackwell. | Control knob / confound in comparisons | TILE-REPO | Must log it per run. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| `TILEIR_ENABLE_APPROX` | TileIR disables approx by default; can enable via env var. | H4 sweep | TILE-REPO | Verify interaction with register pressure empirically. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| `TILEIR_ENABLE_FTZ` | TileIR disables FTZ by default; can enable via env var. | H4 sweep | TILE-REPO | Same: treat as codegen knob; measure. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| Tensor-of-pointer pattern | Pattern where tensors are composed of pointers for memory access; blog reports suboptimal performance on TileIR backend with CUDA 13.1. | H3 code-pattern axis | NV-TILE-BLOG | Blog suggests rewriting toward TMA descriptor API. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| `__maxnreg__` / `--maxrregcount` | Register usage controls: `__maxnreg__` sets per-kernel max regs; `--maxrregcount` caps regs file-wide. | Controlled spill induction | CUDA-PG-13.1 | Useful for synthetic ladders / forcing spill. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/cpp-language-extensions.html)) |
| `--warn-on-spills` / `--warn-lmem-usage` | Compiler warnings for spills/local-memory use; `--verbose` prints codegen stats. | Compile-time spill detection (supporting evidence) | NV-PTXCOMP-API (CUDA 13.1) | Use via `PTXAS_OPTIONS` plumbing when available. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/hopper-tuning-guide/ptx-compiler-api/index.html)) |

---

### 2) Table: Tooling & Evaluation Baseline Map

| Source_ID | What it is | What it measures/guarantees | Key limitations | Evidence (cite or UNVERIFIED) |
|---|---|---|---|---|
| NV-PTX-9.1 | PTX ISA specification (version 9.1) | Normative semantics for state spaces like `.local` and memory hierarchy; ties PTX ISA 9.1 to CUDA 13.1 (driver branch r590). | Does **not** specify how Triton/ptxas will allocate registers or spill for a given kernel; ISA ≠ compiler behavior. | ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) |
| NV-NCU-2025.4 | Nsight Compute Profiling Guide (2025.4) | Defines metric names for registers, occupancy limits, local/spill instruction counts, and derived spill request metrics; documents replay/multi-pass caveats. | Some metrics require replay; small/variable kernels can yield misleading values; other GPU engines can confound chip-global resources. | ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| TB-1 (TB) | TritonBench benchmark suite | Provides a collection of PyTorch operators/custom ops + harness to measure performance; includes example invocation (`python run.py --op ...`). | It’s an application-level harness (PyTorch integration, mixed kernels); not a controlled microbench by default. | ([github.com](https://github.com/meta-pytorch/tritonbench)) |
| CUDA-13.1-RN | CUDA Toolkit 13.1 release notes | Establishes existence + scope: CUDA Tile (Tile IR + cuTile), initial Blackwell target; introduces `tileiras` (Tile IR bytecode→SASS); notes Tile-IR AS supports only Blackwell-class devices (current). | Release notes are high-level; not a full spec of codegen/regalloc behavior. | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) |
| NV-TILEIR-13.1 (spec) | Tile IR specification (versioned) | Provides normative Tile IR VM model and memory model; states PTX interoperability / “strict weakening” intent. | Does not directly define register allocation strategy or spilling heuristics. | ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/index.html)) |
| NV-TILE-BLOG | NVIDIA technical blog (Jan 30, 2026) | States prerequisites (CUDA 13.1+, Blackwell), shows enabling (`ENABLE_TILE=1`), describes `.tileIR` cache artifacts, notes tensor-of-pointer perf issue + TMA rewrite direction. | Blog-level guidance; “per-kernel backend selection” is mentioned but mechanism details are not specified (UNVERIFIED details). | ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| TILE-REPO | triton-lang/Triton-to-tile-IR repo | Implementation: env vars (`ENABLE_TILE`, `TILEIR_ENABLE_APPROX`, `TILEIR_ENABLE_FTZ`), fallback to PTX on compilation bugs, new tuning knob `occupancy`, limitation: `num_warps` not exposed in CUDA 13.1, Blackwell-only support claim. | Incubator repo; behavior can change by commit; must pin commit hash in artifacts. | ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| CUDA-TILE-REPO | NVIDIA/cuda-tile repo | CUDA Tile IR MLIR-based IR + tooling; points to versioned Tile IR docs (13.1). | Not Triton-specific; won’t tell you Triton backend toggle mechanics. | ([github.com](https://github.com/NVIDIA/cuda-tile)) |
| TRITON-REPO | triton-lang/triton repo | Provides env var `PTXAS_OPTIONS` to pass options to `ptxas` (NVIDIA). | Not TileIR-specific; doesn’t guarantee TileIR backend integration (that’s separate). | ([github.com](https://github.com/triton-lang/triton)) |
| cuTile Python | docs for cuTile Python DSL | Defines cuTile as a Python DSL/model; has release notes section with version 1.0.0 (2025-12-02). | Not directly used for Triton backend spill study (mostly terminology/adjacent). | ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/)) |
| CUDA Tile webpage | High-level CUDA Tile landing page | Defines CUDA Tile as tile-based programming model; links to Tile IR + cuTile docs. | Marketing/overview; not normative for metrics. | ([developer.nvidia.com](https://developer.nvidia.com/cuda/tile)) |

---

### 3) Table: SOTA Baseline Map (microarch-specific)

| Baseline_ID | Baseline description | Why it is a fair comparator | What it cannot explain | Evidence |
|---|---|---|---|---|
| BL-P0 | **PTX backend**: run the same Triton kernel with Tile backend disabled (`ENABLE_TILE=0`). | Same source-level kernel; isolates backend choice. | If TileIR backend silently falls back or if compile options differ, it’s not a clean A/B unless provenance is logged. | `ENABLE_TILE` toggle exists; backend is switchable. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| BL-T0 | **TileIR backend**: run the same kernel with `ENABLE_TILE=1`. | Direct A/B vs BL-P0; same GPU + same toolchain; backend swap only. | TileIR backend feature gaps / unsupported ops may force code changes; then “same kernel” assumption breaks. | Prereqs + enable path described. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| BL-T1 | TileIR **tuning sweep**: `occupancy ∈ {1..32}` (and `num_ctas` if applicable). | Knobs are backend-native; tests whether performance delta is knob-driven vs “backend magic.” | Doesn’t isolate *why* occupancy changes regalloc; needs metrics (`launch__registers_per_thread`, spilling metrics). | Knob definition + range stated. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) |
| BL-RCAP | **Register cap control**: use `__maxnreg__` or `--maxrregcount` to intentionally push kernels into/out of spilling regimes. | Provides a controlled way to traverse reg-pressure regimes without changing algorithm. | Caps can change scheduling/occupancy; not a pure “spill only” knob. | CUDA doc defines these controls and constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/cpp-language-extensions.html)) |
| BL-SPILL-MET | **Spill metric triage**: record `launch__registers_per_thread`, `launch__occupancy_limit_registers`, spill instruction metrics, and derived spill request pct. | Establishes measurable “spill” definition independent of runtime noise. | Metrics may be `n/a` or unstable for tiny kernels; needs feasibility probe. | NCU metrics + names are specified. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) |
| BL-PAT | **Code-pattern control**: tensor-of-pointer vs TMA descriptor rewrite (TileIR backend). | Matches blog-flagged perf pitfall; gives a grounded workload transform knob. | Blog notes performance, not explicitly register pressure; must measure to attribute. | Pattern + rewrite shown in blog. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) |
| BL-APP | **TritonBench operator case study**: pick 1–2 ops where spills are present and compare backends/knobs. | Connects microbench mechanisms to real-ish operators. | TritonBench has system-level confounds (PyTorch overhead, mixed kernels); requires careful isolation. | TritonBench purpose + usage documented. ([github.com](https://github.com/meta-pytorch/tritonbench)) |

---

### 4) Table: Claim Ledger v0 (15–30 claims)

**Legend:** Status ∈ {VERIFIED, UNVERIFIED, INFERENCE}.  
Buckets are embedded in **Paper role** as `A/B/C/D` per stage requirement:
- **A** = Primary tooling facts (PTX, NCU, TritonBench)
- **B** = Tile IR stack facts (CUDA Tile, Tile IR, Triton-to-TileIR)
- **C** = SOTA baseline map / known levers
- **D** = INFERENCE (explicit)

| Claim_ID | Claim (1 sentence) | Status | Evidence pointers | Paper role | Risk if wrong |
|---|---|---:|---|---|---|
| C001 | PTX local state space (`.local`) is per-thread private memory and is accessed via `ld.local` and `st.local`. | VERIFIED | E001 ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | A: Glossary ground truth | Low |
| C002 | PTX memory hierarchy specifies each thread has private local memory, CTA has shared memory, and all threads can access global memory. | VERIFIED | E002 ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | A: Glossary ground truth | Low |
| C003 | Nsight Compute states local memory accesses occur for automatic variables that don’t fit into registers or when the kernel uses more registers than available (register spilling). | VERIFIED | E003 ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | A: Spill definition | Low |
| C004 | Nsight Compute exposes `launch__registers_per_thread` and `launch__occupancy_limit_registers` as launch/occupancy metrics. | VERIFIED | E004 ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | A: Measurement method | Low |
| C005 | Nsight Compute exposes spill instruction metrics including `sass__inst_executed_register_spilling_mem_local` and local memory instruction counts (`sass__inst_executed_local_loads/stores`). | VERIFIED | E005 ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | A: Measurement method | Low |
| C006 | Nsight Compute exposes derived spill request metrics such as `derived__local_spilling_requests` and `derived__local_spilling_requests_pct`. | VERIFIED | E006 ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | A: Cross-check metric | Medium |
| C007 | Nsight Compute warns that replayed / multi-pass collection and small/variable kernels can produce confusing or misaligned metric values and chip-global confounds. | VERIFIED | E007 ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | A: Threats-to-validity seed | Medium |
| C008 | CUDA Toolkit 13.1 introduces CUDA Tile (CUDA Tile IR and cuTile) and states the initial release targets Blackwell GPUs. | VERIFIED | E008 ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) | B: Scope lock evidence | High |
| C009 | CUDA 13.1 introduces `tileiras`, a compiler translating Tile IR bytecode into GPU machine instructions (SASS). | VERIFIED | E009 ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) | B: Backend pipeline fact | Medium |
| C010 | CUDA 13.1 release notes state the Tile-IR AS compiler currently supports only Blackwell-class devices (and has limitations). | VERIFIED | E010 ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) | B: Scope lock evidence | High |
| C011 | Tile IR spec is versioned and has release notes indicating “Spec 13.1 (2026-01-23)”. | VERIFIED | E011 ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/release_notes.html)) | B: Version gate | Medium |
| C012 | Tile IR memory model PTX interoperability section states it is intended to be a strict weakening of the PTX memory model. | VERIFIED | E012 ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html)) | B: Correctness + semantics | Medium |
| C013 | NVIDIA blog (Jan 30, 2026) states Triton-to-TileIR adoption requires CUDA 13.1+ and Blackwell GPUs. | VERIFIED | E013 ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | B: Scope lock evidence | High |
| C014 | NVIDIA blog shows enabling Tile IR backend via `export ENABLE_TILE=1` and suggests `.tileIR` cache artifacts when active. | VERIFIED | E014 ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | B: Backend provenance | Medium |
| C015 | NVIDIA blog reports tensor-of-pointer pattern has suboptimal performance on Tile IR backend with CUDA 13.1 and proposes rewriting using a TMA descriptor API. | VERIFIED | E015 ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | B/C: Workload selection rationale | Medium |
| C016 | Triton-to-TileIR repo claims Tile IR backend in this repo uses only features available in CUDA 13.1 and is enabled via `ENABLE_TILE=1`. | VERIFIED | E016 ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | B: Implementation fact | High |
| C017 | Triton-to-TileIR repo documents a TileIR-specific tuning hint `occupancy` (N=1..32) and references `num_ctas` as a critical hint in some workloads. | VERIFIED | E017 ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | C: Knob set definition | Medium |
| C018 | Triton-to-TileIR repo states TileIR backend does not support `num_warps` in CUDA 13.1 but adds `occupancy`. | VERIFIED | E018 ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | C: Sweep design constraint | Medium |
| C019 | Triton-to-TileIR repo documents `TILEIR_ENABLE_APPROX` and `TILEIR_ENABLE_FTZ` env vars (disabled by default) to enable approx/FTZ. | VERIFIED | E019 ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | C: Sweep design constraint | Medium |
| C020 | CUDA C++ doc defines `__maxnreg__` and states `--maxrregcount <N>` can control register usage; `__launch_bounds__` and `__maxnreg__` cannot both apply to the same kernel. | VERIFIED | E020 ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/cpp-language-extensions.html)) | C: Controlled pressure lever | Medium |
| C021 | PTX Compiler API doc lists `--warn-on-spills`, `--warn-on-local-memory-usage`, and `--verbose` for codegen statistics/warnings. | VERIFIED | E021 ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/hopper-tuning-guide/ptx-compiler-api/index.html)) | C: Compile-time corroboration | Medium |
| C022 | Triton repo documents `PTXAS_OPTIONS` to pass extra options to `ptxas` (NVIDIA backend). | VERIFIED | E022 ([github.com](https://github.com/triton-lang/triton)) | C: Implementation hook | Medium |
| C023 | TritonBench is a collection of PyTorch custom operators / operators used to evaluate Triton performance and provides a runnable benchmarking harness. | VERIFIED | E023 ([github.com](https://github.com/meta-pytorch/tritonbench)) | A/C: Workload source | Low |
| C024 | PTX documentation release history ties PTX ISA 9.1 to CUDA 13.1 (driver r590), satisfying PTX ≥ 9.1 version gate for normative PTX references. | VERIFIED | E024 ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | A: Version gate | High |
| C025 | Using NCU’s spill instruction counts + derived spill request metrics together provides a robust operational definition of “spilling” for this proposal’s evaluation. | INFERENCE | (Uses E004–E006) ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | D: Method justification | Medium |
| C026 | TileIR vs PTX backends will differ in spill onset thresholds on at least one controlled register-pressure ladder motif on Blackwell. | INFERENCE | — | D: Core novelty hypothesis | High |
| C027 | TileIR `occupancy` hint will measurably shift register allocation and spill metrics for some compute-intensive kernels. | INFERENCE | (Knob exists: E017) ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | D: Mechanism hypothesis | High |
| C028 | The blog’s statement that users can select backend per-kernel implies an API/mechanism beyond global `ENABLE_TILE`, but the exact mechanism is not specified in the cited material. | UNVERIFIED | E013 (statement w/o mechanism) ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | D: Measurement harness requirement | High |
| C029 | NCU spill metrics (e.g., `sass__inst_executed_register_spilling_mem_local`) will be available (non-`n/a`) on all target Blackwell GPUs under our exact driver/toolchain. | UNVERIFIED | Metrics exist (E005) but availability is empirical ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)) | D: Feasibility risk | High |
| C030 | The listed lab GPUs (RTX 5090, B200, GB10) are Blackwell-class and meet TileIR backend requirements; this must be confirmed by device attributes on the actual machines. | UNVERIFIED | — | D: Feasibility gate | High |

---

### 5) Golden Snapshot (Carry-Forward)

> Format per ID: **What it is** | **Claims/guarantees** | **Does NOT say** | **Why we care**

- **OAI-MEM** ([cookbook.openai.com](https://cookbook.openai.com/examples/agents_sdk/context_personalization))  
  - What it is: OpenAI cookbook pattern for state-based memory + guardrails.  
  - Claims/guarantees: State-based memory supports structured authoritative fields + precedence rules and recommends guardrails at distillation/consolidation/injection.  
  - Does NOT say: Nothing about CUDA/Triton; not a profiling method.  
  - Why we care: Our web-UI “manual state” process needs deterministic ledgers and strict “no invention” lifecycle.

- **OAI-SESS** ([developers.openai.com](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory/))  
  - What it is: OpenAI cookbook example for session trimming (keep last N user turns).  
  - Claims/guarantees: Defines a “turn” and shows automatic trimming in a custom session object.  
  - Does NOT say: No claims about long-term persistence correctness for research artifacts.  
  - Why we care: Our chat will be long; we need a safe manual compaction habit.

- **CUDA-13.1-RN** ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html))  
  - What it is: CUDA Toolkit 13.1 release notes.  
  - Claims/guarantees: Introduces CUDA Tile (Tile IR + cuTile), targets Blackwell initially, introduces `tileiras` bytecode→SASS; notes Blackwell-only support in current Tile-IR AS.  
  - Does NOT say: Doesn’t quantify spill behavior or performance; not a compiler spec.  
  - Why we care: Hard version + arch gate; establishes what exists in CUDA 13.1.

- **NV-PTX-9.1** ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
  - What it is: PTX ISA documentation (includes PTX ISA 9.1 release history).  
  - Claims/guarantees: Normative definitions for `.local` state space and memory hierarchy; ties PTX ISA 9.1 to CUDA 13.1.  
  - Does NOT say: No direct statements about ptxas register allocator heuristics.  
  - Why we care: Ground truth for what “local” means; version-gated normative PTX references.

- **NV-NCU-24** ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
  - What it is: Nsight Compute profiling guide (2025.4).  
  - Claims/guarantees: Provides metric names for registers/occupancy/spilling/local memory; documents CLI organization and replay caveats; shows `--query-metrics` and `--set full` behavior.  
  - Does NOT say: Doesn’t guarantee any metric is supported on any given GPU/driver combo (must test).  
  - Why we care: Evaluation feasibility hinges on spilletrics existing and being collectable.

- **NV-TILEIR-MM** (Tile IR spec 13.1 memory model) ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html))  
  - What it is: Versioned Tile IR specification sections (memory model + release notes).  
  - Claims/guarantees: Tile IR memory model derived from PTX; includes PTX interoperability and tokens; spec versioning exists (Spec 13.1 dated).  
  - Does NOT say: No guarantees about how TileIR codegen affects register allocation/spills.  
  - Why we care: Correctness caveats + semantic differences can confound “same kernel” comparisons.

- **NV-TILE-BLOG** ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))  
  - What it is: NVIDIA technical blog describing Triton-to-TileIR backend.  
  - Claims/guarantees: Requires CUDA 13.1+ + Blackwell; shows `ENABLE_TILE=1`; mentions `.tileIR` cache artifacts; flags tensor-of-pointer performance issue and TMA rew direction.  
  - Does NOT say: Doesn’t fully specify the per-kernel backend selection mechanism (we must verify in code/docs).  
  - Why we care: Provides enabling path + known pitfalls to build controlled case studies.

- **GPU-OPT** *(OUT_OF_ALLOWLIST; non-citable in this project’s governance)*  
  - What it is: ACM paper pointer in the template state.  
  - Claims/guarantees: UNVERIFIED here (domain not in allowlist).  
  - Does NOT say: N/A.  
  - Why we care: Keep as “external reading” only unernance changes.

- **TILE-REPO** ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
  - What it is: triton-lang/Triton-to-tile-IR incubator repo (implementation).  
  - Claims/guarantees: Backend toggle (`ENABLE_TILE`), CUDA 13.1 feature-only claim, Blackwell-only in CUDA 13.1, fallback to PTX on bugs, new knobs (`occupancy`), env vars for approx/FTZ.  
  - Does NOT say: Doesn’t guarantee stability across commits; must pin hash.  
  - Why we care: Defines actual knobs and failure modes weust log/guard.

- **CUDA-TILE-REPO** ([github.com](https://github.com/NVIDIA/cuda-tile))  
  - What it is: NVIDIA/cuda-tile open repo for CUDA Tile IR tooling (MLIR dialect etc).  
  - Claims/guarantees: Points to versioned Tile IR spec; describes IR/dialect ecosystem.  
  - Does NOT say: Not a Triton backend guide; won’t define `ENABLE_TILE`.  
  - Why we care: Cross-check naming/structure and use as an in-governance pointer to spec docs.

- **TRITON-REPO** ([github.com](https://github.com/triton-lang/tron))  
  - What it is: Triton language/compiler development repo.  
  - Claims/guarantees: Documents `PTXAS_OPTIONS` environment variable for NVIDIA backend.  
  - Does NOT say: Doesn’t mention TileIR backend specifics (separate project).  
  - Why we care: Enables compile-time corroboration (verbose/warn-spills) through ptxas.

- **TB** ([github.com](https://github.com/meta-pytorch/tritonbench))  
  - What it is: meta-pytorch/tritonbench benchmark suite.  
  - Claims/guarantees: Collection of operators/ctom ops with example inputs; runnable harness (`python run.py --op ...`).  
  - Does NOT say: Not a microarchitectural isolation suite; needs careful isolation.  
  - Why we care: Provides real-ish workloads for a final case study.

---

### 6) Constraint Cliffs (≤10; microarch-focused)

- CC1: If **CUDA toolkit < 13.1**, TileIR backend/toolchain (`tileiras`, CUDA Tile) is out of scope → proposal must fail fast or re-scope away from TileIR. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/-toolkit-release-notes/index.html))  
- CC2: If the target GPU is not **Blackwell-class**, current TileIR toolchain support may not hold → TileIR claims must be dropped or labeled as future work. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html))  
- CC3: If NCU cannot collect **spill metrics** (e.g., `sass__inst_executed_register_spilling_mem_local`) on our setup, the core evaluation collapses → must pivot to alternative instrumentation and downgrade ms. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- CC4: If kernels are too short/variable, NCU replay/multi-pass effects can distort metrics → microbench must batch/scale to stable durations. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html))  
- CC5: If TileIR backend silently **falls back** to PTX, comparisons are invalid unless provenance is fail-closed. ([github.com](https://github.com/triton-lang/Triton-to-tilIR))  
- CC6: If `num_warps` is not exposed in TileIR (as stated), any evaluation that assumes `num_warps` sweep must be rewritten to use `occupancy`/`num_ctas` knobs. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  
- CC7: If tensor-of-pointer pattern dominates performance on TileIR (known issue), then “TileIR vs PTX” deltas may just be that pattern → must include descriptor rewrite control. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-thetile-ir-backend-for-openai-triton/))  
- CC8: If memory aliasing/cross-tile-block reductions cause correctness hazards under TileIR unordered defaults, affected workloads must be excluded or rewritten to avoid undefined/incorrect behavior. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))  

---

### 7) QUERY_PLAN (only for UNVERIFIED load-bearing claims)

| Query_ID | Targeted query (include version/arch) | Maps to Claim_IDs | Intended source domains |
|---|---|---|---|
| Q001 | “Triton-toileIR per-kernel backend selection API how to choose backend per kernel” | C028 | github.com/triton-lang, developer.nvidia.com, docs.nvidia.com |
| Q002 | “Nsight Compute 2025.4 Blackwell support sass__inst_executed_register_spilling_mem_local n/a” | C029 | docs.nvidia.com, forums.developer.nvidia.com |
| Q003 | “CUDA compute capability RTX 5090” and “CUDA compute capability B200” and “CUDA compute capability GB10” | C030 | developer.nvidia.com, docs.nvidia.com |
| Q004 | “tileiras versilag how to print version tileiras CUDA 13.1” | (env freeze feasibility; supports C008–C010 provenance) | docs.nvidia.com, forums.developer.nvidia.com |
| Q005 | “Triton-to-TileIR .tileIR cache extension location TRITON cache directory” | C014 (provenance robustness) | developer.nvidia.com, github.com/triton-lang |
| Q006 | “NCU derived__local_spilling_requests interpretation register spilling to local memory” | C025 | docs.nvidia.com |
| Q007 | “Tile IR backend occupancy hint semantics and intregister allocation” | C027 | github.com/triton-lang, docs.nvidia.com |
| Q008 | “Tile IR backend approx FTZ semantics TILEIR_ENABLE_APPROX TILEIR_ENABLE_FTZ impact” | C019/H4 | github.com/triton-lang, docs.nvidia.com |

---

## CONTEXT_CAPSULE

```yaml
STATE_VERSION: "manual_state_v4_3_tilespill_blackwell_cuda13_1_lock@2026-02-04"

HARD_VERSION_LOCK:
  cuda_toolkit_min: "13.1"
  ptx_isa_min: "9.1"
  ncu_min: "2025.4"
  tile_arch_scope: "Blackwell-only"
  allowlist_domains:
    - "cookbook.openai.com" "developers.openai.com"
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
  current_stage: 0
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
    title: "Tile IR spec (memory model) âersioned 13.1"
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
  closed: []

DELTA_LOG:
  - id: "DL001"
    date: "2026-02-05"
    changes:
      - "Set profile.current_stage=0; last_updated=2026-02-05; ARTIFACT_INDEX.stage0_fact_sheet=WP0_20260205."
      - "Added Claim Ledger v0: C001–C030."
      - "Added Evidence Ledger entries: E001–E024 (all version-gated where applicable)."
      - "Updated NV-TILEIR-MM URL to versioned Tile IR spec 13.1 path for determinism."
      - "Fixed invalid YAML in GOLDEN_SOURCES by normalizing GPU Optimization entry to id=GPU-OPT and labeling OUT_OF_ALLOWLIST."
      - "Seeded OPEN_QUESTIONS: OQ001–OQ003 and mapped QUERY_PLAN Q001–Q008."

CAPSULE_  ledger_counts:
    claims: 30
    evidence: 24
    open_questions_active: 3
    experiments: 0
  approx_tokens: 2900
  notes: "Initial Stage-0 capsule; no shrink constraints applicable yet."

NEXT_STAGE_HINT: "Start fresh chat; paste System Config + this capsule + Stage 1 prompt."
```

---
Learn more:
1. [https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)
2. [https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
3. [https://github.com/triton-lang/Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR)
4. [https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html)
5. [https://github.com/meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench)
6. [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
7. [https://docs.nvidia.com/cuda/tile-ir/13.1/index.html](https://docs.nvidia.com/cuda/tile-ir/13.1/index.html)
8. [https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory\_model.html](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/memory_model.html)
9. [https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/cpp-language-extensions.html](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
10. [https://docs.nvidia.com/cuda/archive/13.1.0/hopper-tuning-guide/ptx-compiler-api/index.html](https://docs.nvidia.com/cuda/archive/13.1.0/hopper-tuning-guide/ptx-compiler-api/index.html)
11. [https://github.com/NVIDIA/cuda-tile](https://github.com/NVIDIA/cuda-tile)
12. [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)
13. [https://docs.nvidia.com/cuda/cutile-python/](https://docs.nvidia.com/cuda/cutile-python/)
14. [https://developer.nvidia.com/cuda/tile](https://developer.nvidia.com/cuda/tile)
15. [https://docs.nvidia.com/cuda/tile-ir/13.1/sections/release\_notes.html](https://docs.nvidia.com/cuda/tile-ir/13.1/sections/release_notes.html)
16. [https://cookbook.openai.com/examples/agents\_sdk/context\_personalization](https://cookbook.openai.com/examples/agents_sdk/context_personalization)
17. [https://developers.openai.com/cookbook/examples/agents\_sdk/session\_memory/](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory/)
