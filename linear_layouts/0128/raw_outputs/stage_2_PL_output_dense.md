## 0) Internal outline (constraints + what we’re allowed to change)

### Hard constraints (from your config)
- **Real-metal only**: evaluation must run on real GPUs (H100/A100/B200/MI300) using **Triton kernels / TritonBench / Nsight Compute / end-to-end latency** (no gem5/GPGPU-Sim/FPGA).  
- **No new hardware**: we must exploit existing features (TMA, mbarrier, warp specialization, clusters/DSMEM, Tensor Cores, etc.).  
- **Seed paper full text must be from arXiv HTML** (we have it). ([arxiv.org](https://arxiv.org/html/2505.23819v3))

### Stage 1 “Hardware Reality” bottlenecks (from your Stage 1 memo)
1. **Asynchrony gap**: layout engines optimize “where”, but Hopper/Blackwell require optimizing **“when/how”** (TMA + barriers + warp specialization).  
2. **Dynamic/ragged shapes**: power-of-two assumptions → padding/masking overhead in real inference.  
3. **Indirection**: paging/MoE routing breaks contiguity → vectorization collapses; shuffles can become instruction-bound.

### Stage 1.5 “Theoretical Arsenal” we can actually operationalize
- **Piecewise-affine / Presburger sets** (ragged edges as guarded iteration domains).  
- **\(\mathbb{Z}\)-module / integer-lattice algebra** (replace \(\mathbb{F}_2\) bit-matrices when shapes/strides aren’t powers of two; use SNF/GCD structure for vectorization legality).  
- **Tokenized dataflow IR** (async events/tokens for copy→barrier→compute; schedule with resource constraints).  
- **Typestate/session types** (static correctness for mbarrier/fence protocols).  
- **Equality saturation / e-graphs** (global search over data-movement lowerings).  
- **SMT-based swizzle/bank-conflict solving** with target-parametric bank models.

---

## 1) Deep research validation (what’s true on H100/Blackwell/MI300, and what already exists)

### 1.1 What the seed paper actually claims / omits
**Linear Layouts (seed)**:
- Models layouts as linear maps over \(\mathbb{F}_2\) and integrates into Triton backend; provides conversion, swizzle, gather optimizations, etc. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **Explicitly states the primary limitation is restriction to power-of-two shapes**, mitigated by masking; flipping/slicing require affine extensions. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Gather fast-path uses warp shuffles **only if the gathered axis stays within a warp** (otherwise the optimization doesn’t apply). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Notably: the seed showing **layout codegen** does *not* model TMA / mbarrier / async-proxy fencing as first-class objects (no “async token” semantics).

So Stage 1’s “layout != schedule” critique is consistent with the seed’s scope.

### 1.2 Hopper/Blackwell reality: TMA + clusters + fences are real and matter
- Hopper introduces **TMA** as a more sophisticated async copy engine than Ampere async copies: transfers **1D–5D tensors** between global↔shared and also between shared regions of SMs in a cluster; avoids register moves; a single thread can issue large transfers while the block continues work. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  
- PTX exposes this via **`cp.async.bulk.tensor.*`**, taking a **tensor map (descriptor)** + coordinates + **mbarrier** completion mechanism. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
- Correctness/perf requires **proxy-ordering fences**: generic-proxy writes aren’t automatically visible to async proxy without an explicit fence; commonly expressed as `fence.proxy.async.shared::cta` in practice. ([accelerated-computing.academy](https://accelerated-computing.academy/fall25/resources/tma-interface/?utm_source=openai))  
- Thread Block **Clusters + Distributed Shared Memory (DSMEM)** are in CUDA’s model (Hopper cc≥9.0) and remain in Blackwell; blocks in a cluster are co-scheduled and can access each other’s shared memory. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/01-introduction/programming-model.html?utm_source=openai))  
- Blackwell tuning guidance: clusters are supported; **B200 allows non-portable cluster size 16** (opt-in), and occupancy guidance explicitly calls out `cudaOccupancyMaxActiveClusters`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/blackwell-tuning-guide/))  

### 1.3 What’s already “solved” in the ecosystem (second-order leads)
This is the novelty filter: if we propose “add TMA”, we’re late.

- **Triton already has a TMA surface**:
  - `tl.make_tensor_descriptor` creates a descriptor; on NVIDIA with TMA support it’s backed by TMA hardware; there are alignment/stride/contiguity constraints (e.g., base 16B aligned, last dim contiguous). ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
  - Triton has a **TMA-based persistent matmul tutorial**. ([triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html?utm_source=openai))  
  - There’s active discussion/instability around the API (“experimental” descriptor interfaces). ([github.com](https://github.com/triton-lang/triton/issues/6177?utm_source=openai))  
- **Warp specialization support has landed/been productized** for Triton users via PyTorch/Triton releases (Hopper-focused). ([pytorch.org](https://pytorch.org/blog/warp-specialization/?utm_source=openai))  
- Research has already moved to async-first models:
  - **Cypress (PLDI’25)**: task-based sequential semantics lowered to warp-specialized pipelines using TMA/Tensor Cores. ([research.nvidia.com](https://research.nvidia.com/publication/2025-06_task-based-tensor-computations-modern-gpus?utm_source=openai))  
  - **Tawa (arXiv’25)**: compiler that generates warp-specialized code with “asynchronous references (aref)”. ([arxiv.org](https://arxiv.org/abs/2510.14719?utm_source=openai))  
- Dynamic shape/padding pain is being addressed directly:
  - **TMA-Adaptive FP8 Grouped GEMM (arXiv’25)**: explicitly targets says “padding each group to a fixed alignment” overhead; uses a descriptor pool + runtime selection to cover residual cases without padding. ([arxiv.org](https://arxiv.org/abs/2508.16584?utm_source=openai))  
- For indirection / paged KV:
  - **PagedAttention (vLLM)** makes KV cache allocation non-contiguous and dynamic (OS paging analogy), improving throughput but changing access regularity. ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  
  - **vAttention (ASPLOS’25)** keeps KV cache contiguous in *virtual* memory using CUDA virtual memory APIs (avoids rewriting kernels for paging), and reports notable throughput gains. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/?utm_source=openai))  
- Equality saturation in “real compilers” is now a live topic:
  - **DialEgg (CGO’25)** integrates MLIR with Egglog eq-sat. ([2025.cgo.org](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog?utm_source=openai))  
  - **eqsat dialect (arXiv’25 / EGRAPHS’25)** embeds e-graphs in MLIR. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
  - “Mind the Abstraction Gap…” applies eq-sat to XLA graphs and discusses cost modeling/scalability. ([experts.illinois.edu](https://experts.illinois.edu/en/publications/mind-the-abstraction-gap-bringing-equality-saturation-to-real-wor/?utm_source=openai))  
- AMD bank-conflict reality is target-specific and phase-based:
  - MI-series wavefront is **64 lanes**, LDS has **32 banks**, and “bank-conflict free” depends on instruction width + lane grouping phases (`ds_read_b128`, `ds_write_b128`). ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html?utm_source=openai))  

**Implication:** novelty is not “TMA exists” or “warp specialization exists”, but **how to co-optimize layout + async transport + dynamism under real compiler constraints**, and how to do it in Triton/MLIR with robust correctness.

---

# 2) Three research directions (Stage 1 gap + Stage 1.5 theory → new artifact)

I’ll follow your formula explicitly:

> **Proposal = (Validated HW gap) + (Stage 1.5 math/PL theory) → (New compiler/runtime artifact)**

---

## Direction 1 — **Z-Layouts: Showing that \(\mathbb{F}_2\) is the wrong base ring for inference**
### “Guarded \(\mathbb{Z}\)-Module Layouts + Mixed-Radix Vectorization for Ragged Serving”

#### 2.1 Define the gap (seed insufficiency)
- Seed Linear Layouts is powerful *when* layout math is naturally bit-linear; but it **restricts to power-of-two shapes** and punts raggedness to “define larger tensors + mask”. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Modern serving workloads are **structurally ragged** (dynamic KV cache, variable sequence length), and state-of-the-art systems explicitly treat KV cache as dynamic/paged (PagedAttention). ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  
- Worse, many performance-relevant constraints are *not bit-linear*:
  - TMA descriptors impose alignment/contiguity constraints (e.g., last dim contiguous). ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
  - Grouped GEMM padding/alignment overhead is real enough that recent work targets **eliminating padding requirements** via descriptor pools + runtime selection. ([arxiv.org](https://arxiv.org/abs/2508.16584?utm_source=openai))  

So “mask it” is not just a perf smell; it actively blocks systematic vectorization and TMA-friendly transfers for ragged boundaries.

#### 2.2 Apply the Stage 1.5 theory (explicit math)
- **Replace \(\mathbb{F}_2\) matrices with \(\mathbb{Z}\)-modules**:
  - Layout becomes an integer linear map \(x \mapsto Ax + b\) over \(\mathbb{Z}\), not XOR on bits.
  - Use **Smith Normal Form (SNF)** / GCD structure of \(A\) to derive:
    - maximal contiguous subspaces (vectorization width),
    - legality of reshapes/tiles under non-power-of-two dims,
    - when an inverse exists (or a right-inverse for surjections).
- **Piecewise-affine + guards (Presburger sets)** for ragged edges:
  - \(f(x) = A_i x + b_i\) when guard \(g_i(x)\) holds (e.g., tail tiles, page boundaries).
- **Mixed-radix indexing**:
  - Instead of “bit slicing”, represent indices in bases \((b_0,b_1,\dots)\) matching runtime shapes, enabling strength reduction and avoiding expensive div/mod where possible.

This is *exactly* the “Superseding power-of-two” plank from your Stage 1.5 §1, but grounded in compiler implementability.

#### 2.3 Define the mechanism (compiler/runtime artifact)
**Artifact:** a Triton backend pass + runtime specialization layer:

1. **`ttg.zlayout` IR**: represent layouts as guarded integer affine maps (plus a “layout domain” as a Presburger set / bounds).
2. **Vectorization solver**:
   - compile-time compute SNF/GCD facts on **symbolic strides** where possible,
   - otherwise emit a small set of versions keyed on alignment/shape buckets (partial evaluation).
3. **Ragged lowering**:
   - generate **tail-specialized kernels** (no masks) for common residual sizes,
   - otherwise generate guarded vectorized paths with minimal predicate overhead.

##### Pipeline diagram (software-visible)
```ascii
 PyTorch / Inductor Shapes (dynamic)
           |
           v
   [ Shape Bucketer ]
   key = (M%128, N%64, alignment, page_size, ...)
           |
   +-------+--------------------+
   |                            |
   v                            v
[Version A: no tails]     [Version B: guarded tails]
(no masks, max vec)       (piecewise affine guards)
   |                            |
   +------------+---------------+
                v
      Triton TTGIR (with zlayout)
                |
      [Z-Module Vectorization Pass]
      (SNF/GCD → vec width + legal tiles)
                |
      LLVM/PTX emission
      (vector ld/st, predication only on fringes)
```

#### 2.4 Implementation sketch (what to build in 3–4 months)
- **Where in Triton:** implement as an MLIR pass at/near the existing layout engine boundary (where Linear Layouts currently run), but gated behind a feature flag (so we can compare). Seed paper’s integration point is in the backend pipeline; we reuse that integration style. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **Core data structures**
  - `GuardedAffineLayout`:
    - list of pieces \((g_i, A_i, b_i)\),
    - domain bounds \((0 \le x_j < S_j)\) as parameters.
- **Core algorithms**
  - *Vector width derivation*:
    - for each memory access map, compute stride lattice; derive largest \(w\) such that addresses are contiguous and aligned to \(w \cdot \text{elem\_bytes}\).
  - *Tail specialization*:
    - learn buckets from profiling (or just compile for common residuals: 0, 16, 32, 48 for FP16 tiles, etc).
- **Hardware hook-ups**
  - If using TMA, ensure descriptor constraints are satisfied; otherwise lower to conventional vector loads.
  - Integrate with descriptor-pool idea (tell, don’t guess): recent work shows “descriptor pool + runtime selection” is viable for residual handling. ([arxiv.org](https://arxiv.org/abs/2508.16584?utm_source=openai))  

#### 2.5 Evaluation plan (real metal)
- **Microbenchmarks (TritonBench-like)**
  - ragged transpose / copy with varying tail sizes (simulate decoding tails),
  - grouped GEMM with random group sizes (MoE-ish),
  - KV-cache page gather microbench (paged block table).
- **End-to-end**
  - LLM inference decode throughput under continuous batching with variable seq lengths (vLLM-style), and compare:
    - baseline Triton kernels vs zlayout versions.
- **Nsight Compute counters**
  - achieved memory throughput,
  - vector load efficiency,
  - predicate overhead,
  - occupancy / reg usage.

#### 2.6 Novelty check
- This is **not** “just do masks”: the seed already says masks exist. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- This is **not** “descriptor pools exist”: that exists for grouped GEMM, but we generalize to a compiler-level representation and unify with layout/vectorization reasoning. ([arxiv.org](https://arxiv.org/abs/2508.16584?utm_source=openai))  
- The novelty is the **ring change** (\(\mathbb{F}_2 \to \mathbb{Z}\)) and making raggedness first-class (guards + mixed radix), so the compiler can reason about vectorization/TMA legality systematically.

---

## Direction 2 — **Make “transport schedule” a first-class IR, not a hand-coded kernel trick**
### “Tokenized TMA Pipelines for Triton with Typestate-Verified Barriers”

#### 2.1 Define the gap (seed insufficiency)
- Hopper performance depends on **TMA + async overlap**: TMA transfers 1D–5D tensors, avoids register moves, and supports cluster transfers. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  
- Correctness and performance require **fence/proxy ordering** between generic and async proxies; this is subtle enough to warrant explicit guidance and is commonly expressed as `fence.proxy.async.shared::cta`. ([accelerated-computing.academy](https://accelerated-computing.academy/fall25/resources/tma-interface/?utm_source=openai))  
- The seed Linear Layouts pass is a spatial algebra + conversion generator; it does showing warp-shuffle conversions and swizzling, but it does not elevate **async pipeline structure** into something the compiler can optimize globally. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

Second-order lead / contradiction resolution:
- Triton *already* exposes TMA descriptors and a TMA-based matmul tutorial. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
- Triton/PyTorch have shipped **automated warp specialization**. ([pytorch.org](https://pytorch.org/blog/warp-specialization/?utm_source=openai))  
So the gap is: **TMA exists, but schedule synthesis + correctness-by-construction + cross-op generalization is still missing** (and the API is still evolving/experimental). ([github.com](https://github.com/triton-lang/triton/issues/6177?utm_source=openai))  

#### 2.2 Apply the Stage 1.5 theory (explicit PL mechanisms)
- **Task-graph semantics with tokens**:
  - model `tma.copy` as producing a token \(t\),
  - compute consumes \(t\),
  - barriers/fences are explicit edges in the dependency DAG.
- **Typestate / session types** for pipeline protocol:
  - shared-memory stage is a linear resource with states:
    \[
    Empty \to InFlight \to Full \to Drained \to Empty
    \]
  - compiler statically verifies no use-before-ready and no overwrite-before-consume.
- **Resource-aware scheduling**:
  - choose pipeline depth and warp roles subject to **register pressure** and shared memory limits.

This is “Cypress/Tawa-like semantics”, but implemented as an **optimization layer inside Triton**, not a new language.

#### 2.3 Define the mechanism (compiler artifact)
**Artifact:** an MLIR dialect extension + Triton backend pass:
- `ttg.async_token` (or extend `triton-nvidia-gpu`) with ops like:
  - `async.tma_load(desc, coords, smem_slot) -> !async.token`
  - `async.mbarrier_arrive(token, bar)`
  - `async.mbarrier_wait(bar)`
  - `async.proxy_fence()` (inserted when required)
- A **scheduler** that:
  1. partitions warps into producer/consumer roles (warp specialization),
  2. allocates a circular buffer of \(D\) stages in SMEM/DSMEM,
  3. emits a modulo schedule that overlaps:
     - `cp.async.bulk.tensor` (TMA) ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
     - compute (`wgmma`/Blackwell MMA variants) as available.

##### Dataflow diagram (tokenized pipeline)
```ascii
      [desc + coords]                          [tensorcore fragment]
             |                                         |
             v                                         v
   async.tma_load(...) -> token t0         compute(dot) waits on t0
             |                                         |
             v                                         v
   async.mbarrier_arrive(t0)                 epilogue / store
             |
             v
     (typestate transitions ensure:
      no overwrite of stage k until consumer releases it)
```

#### 2.4 Implementation sketch (concrete hooks)
- **Leverage existing PTX interface reality**
  - Lower async ops to PTX `cp.async.bulk.tensor.*` + mbarrier completion mechanism. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
  - Enforce `async_proxy_fence` when generic-proxy initializes barriers or SMEM data before TMA observes it. ([accelerated-computing.academy](https://accelerated-computing.academy/fall25/resources/tma-interface/?utm_source=openai))  
- **Clusters/DSMEM**
  - Add placement types: `Shared::CTA` vs `Shared::Cluster` so the pass can target `.shared::cluster` forms of `cp.async.bulk.tensor` where profitable. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/01-introduction/programming-model.html?utm_source=openai))  
  - Blackwell-specific: optionally explore cluster size 16 on B200 via opt-in and see when it’s worth it. ([docs.nvidia.com](https://docs.nvidia.com/cuda/blackwell-tuning-guide/))  
- **Interoperation with existing Triton features**
  - The pass should consume high-level tiling chosen by existing MMA pipeline machinery (NVIDIA blog notes Triton already extends MMA pipelining for Blackwell). ([developer.nvidia.com](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/?utm_source=openai))  
  - It should also be compatible with Triton’s descriptor API constraints. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  

#### 2.5 Evaluation plan (real metal)
- **Microbenchmarks**
  - TMA load/store bandwidth kernels (vary tensor rank 2–5),
  - pipelined matmul/attention tiles comparing:
    - baseline Triton TMA examples vs tokenized scheduler output.
- **TritonBench**
  - persistent matmul variants (existing) as a baseline harness. ([triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html?utm_source=openai))  
- **Metrics**
  - reg pressure / occupancy,
  - mbarrier wait stalls,
  - achieved overlap (copy engine busy while tensor core busy),
  - instruction count vs throughput.

#### 2.6 Novelty check
- Cypress and Tawa validate task-based / aref abstractions for Hopper. ([research.nvidia.com](https://research.nvidia.com/publication/2025-06_task-based-tensor-computations-modern-gpus?utm_source=openai))  
- Our novelty is **embedding this into Triton’s compiler backend** as a *general scheduling pass* that composes with layout propagation (Linear Layouts) and can target clusters/DSMEM—turning “expert kernel structure” into an optimizable artifact.  
- Also: correctness is enforced via typestate-like verification rather than “hope your fences are right.”

---

## Direction 3 — **Stop treating gather/scatter as “just another layout conversion”**
### “Inspector–Executor + E-Graph Search for Indirection-Aware Data Movement”

#### 2.1 Define the gap (seed insufficiency)
- Linear Layouts’ gather optimization is explicitly conditional: it uses warp shuffles only if the gathered axis stays within one warp. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Real inference stacks increasingly introduce **indirection**:
  - PagedAttention uses paged/non-contiguous KV blocks (logical→physical mapping), which changes access regularity. ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  
  - vAttention shows that even *systems papers* are now forced to reason about memory layout/contiguity at the CUDA virtual-memory layer to avoid kernel rewrites. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/?utm_source=openai))  
- Once indices cross warp boundaries, shuffle-based conversion becomes either impossible or too instruction-heavy.

So a purely static “layout inversion finds contiguity → vectorize” model breaks down when the access function depends on runtime values (indices).

#### 2.2 Apply the Stage 1.5 theory
- **Inspector–Executor**:
  - inspector computes a permutation (and block descriptors) that packs sparse/indirect accesses into dense tiles.
- **Equality saturation over a data-movement IR**:
  - maintain an e-graph of equivalent lowering strategies:
    - direct gather,
    - warp-shuffle gather,
    - shared-memory staging + swizzle,
    - inspector-executor pack/unpack,
    - cluster/DSMEM staging (if available).
  - use cost model driven by:
    - shuffle rounds (instruction count),
    - register pressure,
    - bank conflicts (target-specific),
    - TMA descriptor overhead.
- **Target-specific bank models**:
  - AMD wave64 + phase-based LDS rules. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html?utm_source=openai))  
  - NVIDIA shared memory and TMA/cluster behavior as per tuning guides. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  

#### 2.3 Define the mechanism (compiler+runtime artifact)
**Artifact:** “Data Movement Planner” spanning runtime + compiler:

1. **Runtime inspector** (lightweight GPU kernel):
   - input: index tensor / block table / expert assignment,
   - output: permutation says “these tokens belong to tile \(i\)”, plus per-tile metadata.
2. **Executor kernel** (Triton):
   - runs on dense tiles,
   - uses vectorized loads or TMA where legal,
   - writes results in permuted order, then scatters back.

3. **Compiler search** (MLIR):
   - represent each candidate lowering as a rewrite in an e-graph engine (DialEgg/eqsat style). ([2025.cgo.org](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog?utm_source=openai))  
   - use MLIR Transform dialect to orchestrate where to apply which strategy (so we can keep it programmable/debuggable). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/Transform/?utm_source=openai))  

##### Pipeline diagram (planner + execution)
```ascii
   (indices / routing)
          |
          v
   [Inspector Kernel]
   - histogram/bucket
   - prefix-sum
   - permutation P
          |
          v
   [Executor Kernel (dense)]
   - vector/TMA loads
   - compute
   - write dense out
          |
          v
   [Scatter / Unpermute]
```

#### 2.4 Implementation sketch
- **IR choice**
  - Either:
    - add a small MLIR “data-movement” dialect (gather/scatter/permute primitives), or
    - treat Triton IR patterns as the rewrite domain and translate to MLIR eqsat.  
- **Equality saturation infrastructure**
  - DialEgg shows how to connect MLIR constructs to Egglog. ([2025.cgo.org](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog?utm_source=openai))  
  - eqsat dialect shows native e-graph-in-MLIR approach. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  
  - “Mind the Abstraction Gap” demonstrates cost-model-driven eqsat in a production ML compiler setting (XLA), which is exactly the playbook we need. ([experts.illinois.edu](https://experts.illinois.edu/en/publications/mind-the-abstraction-gap-bringing-equality-saturation-to-real-wor/?utm_source=openai))  
- **Bank-conflict models**
  - Encode bank mapping constraints per target:
    - NVIDIA: reuse existing shared mem analysis + swizzle ideas from Linear Layouts (spatial part). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
    - AMD: incorporate documented wave64 phase groups. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html?utm_source=openai))  

#### 2.5 Evaluation plan
- **Microbenchmarks**
  - synthetic gather where indices:
    - stay within warp (seed fast-path should win),
    - cross warps (our planner should win),
    - random vs clustered distributions (stress inspector overhead).
- **System benchmarks**
  - paged KV attention kernels (PagedAttention-style). ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  
  - compare also to vAttention-style approach where possible (contiguous virtual memory baseline). ([microsoft.com](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/?utm_source=openai))  
- **Metrics**
  - instruction count / IPC (to see instruction-bound shuffle explosions),
  - achieved memory bandwidth,
  - overhead of inspector stage vs savings.

#### 2.6 Novelty check
- Yes, people already permute tokens for MoE in practice; the novelty is:
  1. **making it compiler-visible and searchable**, not handwritten,
  2. selecting among multiple *equivalent* lowering strategies via eq-sat,
  3. integrating TMA/clusters and bank-conflict models across NVIDIA+AMD.

---

# 3) Decision matrix (3 directions scored)

Scales:
- **Novelty**: 1 (incremental) → 5 (new systems+math application)
- **HW relevance**: 1 (generic) → 5 (directly unlocks H100/B200/MI300 features)
- **Implementation risk**: 1 (low) → 5 (high) for a 3–4 month prototype

| Direction | Theoretical Novelty | Hardware Relevance | Implementation Risk | Why |
|---|---:|---:|---:|---|
| **D1: Z-Layouts (guarded \(\mathbb{Z}\)-module)** | 5 | 4 | 4 | Big abstraction jump beyond \(\mathbb{F}_2\); directly targets raggedness + vectorization legality; but SNF/guards + Triton integration is nontrivial. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |
| **D2: Tokenized TMA Transport IR + typestate** | 4 | 5 | 3 | Extremely aligned with Hopper/Blackwell reality (TMA/mbarrier/clusters/fences); closest path to ASPLOS-style HW/SW interface paper; but must be careful not to duplicate existing Triton WS/TMA heuristics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai)) |
| **D3: Inspector–Executor + E-graph movement planner** | 5 | 4 | 5 | Strong PL/systems combo (eq-sat + runtime permutations) and directly addresses indirection; but end-to-end complexity (runtime+compiler+cost model) is high for 3–4 months. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |

---

## If you want the “best ASPLOS fitâsentence
**Direction 2** is the cleanest ASPLOS story: **TMA + clusters are real hardware mechanisms; we build a compiler IR/scheduler that makes them systematically usable and correct**, while composing with spatial layout optimizations (Linear Layouts) rather than replacing them. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  

If you want, I can turn **one** of these into a full ASPLOS-style paper skeleton next (problem → key idea → design → impation → evaluation → related work), including a concrete list of Triton/MLIR files/passes to modify and 6–8 microbench kernels to write for TritonBench.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [1. NVIDIA Hopper Tuning Guide — Hopper Tuning Guide 13.1 documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai)
3. [PTX ISA :: CUDA Toolkit Documentation](https://docs./cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai)
4. [6.S894](https://accelerated-computing.academy/fall25/resources/tma-interface/?utm_source=openai)
5. [1.2. Programming Model — CUDA Programming Guide](https://docs.nvidia.cn/cuda/cuda-programming-guide/01-introduction/programming-model.html?utm_source=openai)
6. [1. NVIDIA Blackwell Tuning Guide — Blackwell Tuning Guide 13.1 documentation](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
7. [triton.language.make\_tensorscriptor — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai)
8. [Persistent Matmul — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html?utm_source=openai)
9. [Functional interface for TMA descriptors · Issue #6177 · triton-lang/triton · GitHub](https://github.com/triton-lang/triton/issues/6177?utm_source=openai)
10. [Enabling advanced GPU features in PyTorch – Warization – PyTorch](https://pytorch.org/blog/warp-specialization/?utm_source=openai)
11. [Task-Based Tensor Computations on Modern GPUs | Research](https://research.nvidia.com/publication/2025-06_task-based-tensor-computations-modern-gpus?utm_source=openai)
12. [Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References](https://arxiv.org/abs/2510.14719?utm_source=openai)
13. [TMA-Adaptive FP8 Grouped GEMM: Eliminating Padding Requirements in Low-Precision Training and Inference on Hoer](https://arxiv.org/abs/2508.16584?utm_source=openai)
14. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180?utm_source=openai)
15. [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention - Microsoft Research](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/?utm_source=openai)
16. [DialEgg: Dialect-Agnostic MLIR Optimizer using Equality Saturation with Egglog (CGO 2025 - Main Conference) - CGO 2025](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog?utm_source=openai)
17. [eqsat: An Equality Saturation Dialect for Non-destructive Rewriting](https://arxiv.org/abs/2505.09363?utm_source=openai)
18. [Mind the Abstraction Gap: Bringing Equality Saturation to Real-World ML Compilers - Illinois Experts](https://experts.illinois.edu/en/publications/mind-the-abstraction-gap-bringing-equality-saturation-to-real-wor/?utm_source=openai)
19. [Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile Framework — ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html?utm_source=openai)
20. [OpenAI Triton on NVIDIA Blackwell Boosts AI Performance and Programmability | NVIDIA Technical Blog](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/?utm_source=openai)
21. [Transform Dialect - ML](https://mlir.llvm.org/docs/Dialects/Transform/?utm_source=openai)
