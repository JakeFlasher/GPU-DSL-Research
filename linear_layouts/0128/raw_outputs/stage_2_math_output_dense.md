## 0) Internal outline (constraints + theory hooks)

### Non‑negotiable constraints (from your system config)
- **Real-metal only**: evaluate on actual GPUs (H100/A100/MI300/etc.) with **TritonBench, Nsight Compute / rocprof-compute, and end-to-end latency**. No simulators / FPGA.  
- **No new hardware**: must exploit **existing** features (TMA, mbarrier, warp specialization, cluster/DSMEM, cache controls), or mitigate existing limitations.  
- **Implementation feasibility**: must be buildable in **Triton / PyTorch / CUTLASS** within ~3–4 months for an ASPLOS-style artifact.

### Stage 1 bottlenecks (what we must “make obsolete”)
From **Stage 1**:  
1) **Padding/raggedness cliff** (LLM decode, KV cache ragged extents)  
2) **Indirection cliff** (PagedAttention paging, MoE routing => pointer chasing / gather/scatter)  
3) **Instruction cliff** (shuffle-round explosion; conversions become instruction-bound)  
4) **Async transport cliff** (Hopper/Blackwell performance depends on **TMA + barriers + proxy ordering**)  
5) **Hierarchy cliff** (Clusters + DSMEM introduce new locality tiers; compiler needs to reason about them)

### Stage 1.5 “math stack” hooks (what we can realistically operationalize)
From **Stage 1.5** (picking the most implementable “math → compiler” upgrades):
- **Ragged shapes**: move from $$\mathbb{F}_2$$ to **$$\mathbb{Z}$$-module / mixed-radix** reasoning via **Smith Normal Form (SNF)/HNF**, plus **Presburger / polyhedral domains**.
- **Asynchrony**: model as **events + partial orders** (Trace theory) + correctness (Petri nets) + throughput (Max-Plus), plus a **PL effect system** for proxy fences.
- **Indirection**: model gathers as **relations** (Boolean semiring), then **factorize** into permutations + block-diagonal structure; use **Cayley-graph search** for minimal shuffle sequences when the plan is “warp exchange”.

---

## 1) Deep research phase: validate Stage 1 gaps against manuals + recent literature

### 1.1 Seed A (Linear Layouts) has an explicit power-of-two limitation + only intra-warp gather fast path
- Linear Layouts explicitly states its **primary limitation is restriction to power-of-two shapes**, mitigated by “define larger tensors + masking,” and that **flips/slices aren’t expressible** without affine extension. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Its optimized `tl.gather` lowering is explicitly **only when the gathered axis resides within a warp**; otherwise the warp-shuffle optimization doesn’t apply. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

This directly matches Stage 1’s “ragged shapes” + “gather only intra-warp” cliffs.

### 1.2 Seed B (LEGO) sidesteps partial tiles/masking in evaluation and is frontend-only
- LEGO’s Triton evaluation **uses power-of-two square matrices** and selects configurations that **avoid partial tiling and masking** “for a fair comparison.” ([arxiv.org](https://arxiv.org/html/2505.08091))  
- LEGO frames itself as a **frontend tool** that generates indexing expressions and **relies on backend compiler frameworks** for loop/space reordering. ([arxiv.org](https://arxiv.org/html/2505.08091))  
- LEGO also notes linear layouts are internal to Triton and **don’t support user-defined nonlinear bijections**, reinforcing the frontend/backend split. ([arxiv.org](https://arxiv.org/html/2505.08091))  

This matches Stage 1’s “LEGO blind spot on dynamic/masked shapes” and “cannot force HW scheduling.”

### 1.3 Hopper/Blackwell reality: peak kernels require TMA descriptors + async barriers + proxy fences (layout ≠ schedule)
**TMA is not just a layout choice; it is a transport engine with its own correctness rules.**
- NVIDIA Hopper introduces **TMA** for async multidimensional tensor copies using a **copy descriptor**, explicitly to reduce address-gen overhead and avoid register/SM instruction usage. ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/?utm_source=openai))  
- Triton exposes this via **`tl.make_tensor_descriptor`**, whose doc spells out key constraints: base must be **16B aligned**, leading strides have **16B-multiple constraints**, last dim contiguous, and only **2–5D** supported; loads/stores are “backed by TMA hardware” on NVIDIA GPUs with TMA. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
- The PTX ISA defines `cp.async.bulk.tensor` as a **non-blocking** copy with **mbarrier completion** variants, introduced in PTX 8.0 and requiring **sm_90+**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html?utm_source=openai))  
- CUDA’s advanced programming guide introduces **async proxy semantics**: bulk-asynchronous copies with TMA (and some tensor core ops) operate in the **async proxy**, and **proxy fences are required** to order them vs normal loads/stores; CUDA also shows `fence.proxy.async.shared::cta` used to make barrier init visible to subsequent bulk copies. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai))  

So: even if Linear Layouts perfectly chooses swizzles and warp shuffles, it still doesn’t model the **transport pipeline** (descriptor legality, barrier accounting, proxy fences, stage depth). That’s exactly Stage 1’s “TMA conflict”.

### 1.4 Clusters + DSMEM are first-class on Hopper (compiler must reason about a new locality tier)
- CUDA defines **Thread Block Clusters (cc 9.0)** and **Distributed Shared Memory (DSMEM)**: blocks in a cluster can read/write/atomically access each other’s shared memory, and you must `cluster.sync()` to guarantee concurrency and safety. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.3.1/cuda-c-programming-guide/index.html?utm_source=openai))  
- Hopper tuning guide states TMA can transfer tensors **between shared memory regions of different SMs in the same cluster** (i.e., cluster-aware transport exists in hardware). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html?utm_source=openai))  

This validates Stage 1’s “clusters/DSMEM gap”: the HW gives new movement options; neither seed treats them as first-class schedule + layout decisions.

### 1.5 AMD MI-series: LDS bank conflicts depend on instruction width + phased lane groups (not NVIDIA’s model)
- AMD ROCm docs explain LDS is banked and conflicts depend on instruction width; e.g., `ds_write_b128` is conflict-free iff there are no conflicts **within each 8-lane phase group** for a 64-lane wavefront; XOR-based preshuffling is a key technique. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))  

So any “unified swizzle solver” needs a **parametric bank model**, not hard-coded NVIDIA assumptions—matching Stage 1.

### 1.6 LLM serving confirms raggedness + indirection are not edge cases
- vLLM / PagedAttention: KV cache memory per request is huge and **grows/shrinks dynamically**, and PagedAttention explicitly uses **paged/non-contiguous** allocation to reduce waste. ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  
- Pensieve generalizes paged attention to support attention with a GPU cache spread over **non-contiguous memory**. ([arxiv.org](https://arxiv.org/abs/2312.05516?utm_source=openai))  

This is the “indirection breaks contiguity assumptions” in Stage 1, backed by systems literature.

### 1.7 MoE serving confirms “permutation + dispatch + combine” is the core cost center
- DeepSpeed’s MoE writeup explicitly calls out that MoE requires **sorting tokens by expert (sparse einsum / permutation)** and re-sorting back, plus communication; it frames these as data-layout transformations and kernel design problems. ([microsoft.com](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/?utm_source=openai))  
- NVIDIA Megatron-Core’s MoE token dispatcher docs describe **routing_map preprocessing, token permutation, all-to-all dispatch, postprocess sorting by expert, and unpermutation**, with fused kernels in some backends. ([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html?utm_source=openai))  
- DeepEP is a concrete expert-parallel comm library that exposes dispatch layout calculation and async dependency management (events), showing the ecosystem already treats dispatch as a specialized runtime/kernel problem. ([github.com](https://github.com/deepseek-ai/DeepEP?utm_source=openai))  

So: an ASPLOS-grade opportunity is to turn these **ad-hoc fused dispatch kernels** into a **compiler-recognized relational primitive** with layout/schedule synthesis.

---

## 2) Scaffold & plan: map bottlenecks → Stage 1.5 theory → compiler artifact

```
(Stage 1 gap)                 (Stage 1.5 theory)                               (new artifact)
--------------------------------------------------------------------------------------------------------------
Ragged extents / padding  -->  Z-modules + SNF/HNF + Presburger domains   -->  Piecewise Modular/Affine Layout IR
Async transport (TMA)      -->  Petri nets + Trace monoids + Effect sys   -->  Transport-Schedule IR + Fence/Barrier synthesis
Indirection (MoE/Paging)   -->  Relations (Boolean semiring) + factorize  -->  Relation-aware lowering: permute↔compute↔unpermute
Shuffle-round explosion    -->  Cayley graph shortest paths               -->  Optimal warp-exchange synthesizer (codegen pass)
Vendor bank quirks (AMD)   -->  Modular affine constraints over Z/BZ      -->  Parametric bank-conflict solver backend
```

Below are three **distinct** research directions built from this scaffold.

---

## 3) Three high-value research directions (each = Gap + Theory → Artifact)

---

# Direction 1 — **MALP**: Mixed‑Radix Affine Layouts with Piecewise Domains (SNF + Presburger)  
**Target:** kill the **padding cliff** for ragged inference shapes without abandoning Linear Layouts’ composability.

### A) Define the gap (why seed is insufficient)
- Linear Layouts admits the **power-of-two restriction** and relies on **masking** for nonconforming shapes; flips/slices require affine extensions. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- LEGO’s evaluation largely avoids the hard case (partial tiles / masking) by choosing **power-of-two** matrices. ([arxiv.org](https://arxiv.org/html/2505.08091))  
- Real inference workloads (PagedAttention-style) are explicitly **dynamic + non-contiguous** in KV management, so “pad to the next 2^k tile” is not a rounding error—it’s a throughput limiter. ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  

### B) Apply the theory (explicit Stage 1.5 math move)
**Replace $$\mathrm{GL}(n,2)$$ layouts with**:
1) **Mixed-radix index modules**: $$\mathbb{Z}_{n_0} \times \cdots \times \mathbb{Z}_{n_k}$$  
2) **Integer-linear maps + congruences**, solved/canonicalized via **Smith Normal Form (SNF)** / HNF  
3) **Presburger domains** to represent “valid interior vs ragged edge” regions as first-class constraints

This is Stage 1.5’s ladder steps **(3) mixed-radix** + **(4) affine/polyhedral** made concrete.

### C) Define the mechanism (software/compiler artifact)
**Artifact:** a Triton backend pass + IR extension:  
> **MALP IR** = `(layout_relation, domain_constraints, cost_model)`  
where `layout_relation` can be:
- a bijection (permutation-like)
- an affine map (slices, negative strides)
- a piecewise affine map (interior vectorized; boundary masked)

**How it integrates with Linear Layouts instead of replacing it**
- Keep existing $$\mathbb{F}_2$$ Linear Layouts as the **fast path** when extents are powers of two.
- When extents are not powers of two, lift the analysis to **integer relations** and compute:
  - maximal contiguous vectors under modular constraints
  - legal reshapes/permutes under domain constraints
  - conversion plans using SNF “multipliers” (explicit codegen)

**Implementation trick (practical, 3–4 month viable)**
- Use **ISL-style integer set relations** internally for composition/inversion/complement of layout relations, because (a) it already fits Presburger reasoning and (b) it’s a known compiler substrate.  
  This is aligned with recent work modeling Triton linear layouts and CuTe layouts using integer set relations for formal manipulation. ([arxiv.org](https://arxiv.org/abs/2511.10374?utm_source=openai))  

### D) “What does codegen look like?”
You emit **two paths**:
- **Interior kernel**: vectorized loads/stores (no masks), possibly eligible for TMA descriptor usage if constraints are met  
- **Boundary kernel**: minimal masking / scalar cleanup

This matches GPU reality: avoid paying boundary costs on every iteration.

### E) Prototype plan (Triton-focused)
**MVP in 3–4 months**
1) Add a layout analysis module that can represent:
   - (a) bit-linear layouts (existing)
   - (b) affine integer relations + domains (new)
2) Implement:
   - SNF-based “contiguity width” computation for common cases (2D/3D)
   - piecewise split: interior tile set vs boundary tiles
3) Hook into `convert_layout` lowering to choose:
   - warp shuffle (if warp-contained)
   - shared-memory transpose (if not)
   - (later) TMA-backed movement if descriptor constraints are satisfiable

**Stretch goals**
- Extend to negative strides / views (affine offsets), aligning with Linear Layouts’ stated affine extension direction. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

### F) Evaluation plan (real metal, TritonBench-first)
- **Microbench:** ragged transpose / masked loads for non-power-of-two shapes (token counts, head dims).  
- **TritonBench:** prioritize kernels sensitive to masking/padding (attention variants, embedding/gather-like). Linear Layouts already reports TritonBench usage; use the same harness for credibility. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **Metrics:** achieved global load/store width, L2/HBM bandwidth, instruction count deltas.

### G) ASCII pipeline diagram
```text
[Triton TTIR/TTG]
      |
      v
[Layout Propagation]
  |   \
  |    \--(if power-of-2)--> [Existing F2 Linear Layouts] --> codegen
  |
  +--(else)--> [MALP: Integer Relation Builder]
                  |
                  v
        [SNF/HNF + Presburger Domain Split]
           |                      |
           v                      v
   [Interior Vector Path]   [Boundary Mask Path]
           \                      /
            \                    /
             v                  v
            [Unified Lowering + Autotune (optional)]
                         |
                         v
                  [LLVM/PTX Backend]
```

### H) Novelty check (second-order leads)
- Existing seeds either (a) stay in $$\mathbb{F}_2$$ and accept masking/padding ([arxiv.org](https://arxiv.org/html/2505.23819v3)), or (b) generate indexing expressions without backend-enforced piecewise lowering ([arxiv.org](https://arxiv.org/html/2505.08091)).  
- Recent ISL-based modeling work focuses on **formal representation/manipulation**, not “ragged-first codegen + costed splitting.” ([arxiv.org](https://arxiv.org/abs/2511.10374?utm_source=openai))  

---

# Direction 2 — **TAS**: Transport‑Aware Scheduling IR for TMA (Petri nets + Trace + Effects)  
**Target:** make “layout ≠ schedule” obsolete by giving Triton a first-class **async transport** layer.

### A) Define the gap
- Hopper/Blackwell performance hinges on **TMA**: multidimensional async copies driven by descriptors; it reduces register/SM instruction overhead. ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/?utm_source=openai))  
- Correctness is not “just insert a barrier”:  
  - `cp.async.bulk.tensor` is non-blocking and completes via **mbarrier mechanisms**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html?utm_source=openai))  
  - CUDA introduces **async proxy** semantics; TMA lives in the **async proxy**, and you need **proxy fences** to order async operations vs normal loads/stores. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai))  
- Seeds do not treat:
  - descriptor legality constraints (alignment/stride restrictions) ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
  - barrier transaction accounting (`mbarrier.expect_tx`, wait) ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html?utm_source=openai))  
  - proxy fences / ordering domains ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai))  
  as first-class scheduling decisions.

### B) Apply the theory
From Stage 1.5 bottleneck #3:
- **Petri nets**: model producer/consumer buffers, barrier tokens, and resource capacities (shared memory stages).  
- **Trace monoids**: model legal reorderings (commuting independent ops) to push copies earlier without violating dependencies.  
- **Effect system**: type/effect track `(proxy=async|generic, space=shared|global, rw)` so the compiler inserts the *minimum necessary* `fence.proxy.*` to remain correct.

### C) Define the mechanism (compiler artifact)
**Artifact:** a new internal IR in TritonGPU: **TransportScheduleIR**.

It explicitly represents:
- `TMA_ISSUE(desc, coords, smem_dst, mbarrier)`
- `MBARRIER_INIT / EXPECT_TX / TRY_WAIT`
- `PROXY_FENCE(async↔generic)` edges
- stage buffers: `smem[stage_id]` (double/triple buffering)

It then runs a **schedule synthesis pass**:
1) Build dependency graph (compute uses tile → must wait on that stage’s barrier flip)  
2) Solve for stage depth and placement (max-plus / critical path)  
3) Emit PTX-level constructs via existing lowering paths (Triton already has descriptor machinery and descriptor ops). ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  

### D) Prototype plan (grounded in current Triton surface area)
You can build this on top of what Triton already exposes:
- `tl.make_tensor_descriptor` creates TMA descriptors and uses TMA-backed loads/stores on supporting NVIDIA GPUs. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
- Triton dialect already includes descriptor scatter ops lowered to NVIDIA TMA scatter on supported targets, suggesting descriptor-aware lowering is already an accepted pattern. ([triton-lang.org](https://triton-lang.org/main/dialects/TritonOps.html?utm_source=openai))  

**MVP (3–4 months)**
- Restrict to **2D** and **3D** tiles; restrict to CTA-shared (no cluster DSMEM yet).
- Implement:
  - legality checker for descriptor constraints (16B alignment; stride constraints) ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
  - barrier + proxy fence insertion skeleton following CUDA guidance ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai))  
  - stage depth heuristic guided by max-plus estimates (or a small DP)  
- Validate with microbench + Nsight Compute.

**Stretch (ASPLOS “volume 2” story)**
- Add **cluster-aware transport**:
  - TMA can copy into shared memory of multiple SMs in a cluster; CUDA clusters + DSMEM provide the synchronization and address space. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html?utm_source=openai))  

### E) ASCII dataflow diagram (what the compiler “sees”)
```text
             (per K-iteration)
   +----------------------------------+
   |  TMA issue tile(k+S) into smem[S]|
   |     cp.async.bulk.tensor ...     |
   |     mbarrier.expect_tx           |
   +-----------------+----------------+
                     |
                     v
           +-------------------+
           | mbarrier.try_wait |
           +---------+---------+
                     |
                     v
        +---------------------------+
        |  WGMMA / MMA on smem[S]   |
        |  (async proxy too!)       |
        +---------------------------+
                     |
                     v
        +---------------------------+
        | Epilogue / stores (generic|
        | proxy) + fence/proxy sync |
        +---------------------------+
```

### F) Evaluation plan
- Compare against:
  1) baseline Triton kernels using TMA descriptors but manual staging choices  
  2) existing warp-specialized kernels without synthesized schedule  
- Use:
  - **Nsight Compute**: overlap indicators, barrier stalls, issue utilization
  - end-to-end: attention/matmul kernels from Triton tutorials (persistent matmul, grouped gemm) that already use descriptors. ([triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html?utm_source=openai))  

### G) Novelty check
- The manuals define the primitives (TMA, mbarrier, proxy fences) but do not provide a compiler algorithm for **automatic schedule synthesis**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai))  
- Existing seeds optimize layout conversions and indexing, but **do not elevate transport schedule to an IR with correctness typing**.

---

# Direction 3 — **RELAX**: Relation‑Aware Lowering for Indirection (MoE + Paging) with Optimal Exchange Synthesis  
**Target:** make “gather breaks vectorization” obsolete by compiling indirection into **structured movement plans**.

### A) Define the gap
- Linear Layouts’ gather optimization is explicitly conditional: only if the gathered axis **resides within a warp**, it uses warp shuffles. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- LLM serving reality: KV cache is often **paged/non-contiguous** (PagedAttention), so indirection is inherent. ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  
- MoE serving reality: dispatch is fundamentally **token permutation → expert compute → unpermutation**, often fused with comm; DeepSpeed and Megatron explicitly describe these data-layout transforms and dedicated kernels. ([microsoft.com](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/?utm_source=openai))  

So we need a compiler artifact that recognizes “indirection patterns” as first-class, instead of treating them as opaque gathers.

### B) Apply the theory
From Stage 1.5 bottleneck #5 and #6:
1) Model indirection as a **relation** $$R \subseteq X \times Y$$ (sparse 0–1 matrix).  
2) Try to **factorize** it into:
   $$R \approx \Pi_2 \circ D \circ \Pi_1$$  
   where $$\Pi$$ are permutations and $$D$$ is block-diagonal / block-sparse (dense inner kernels become possible).  
3) When the movement is warp-contained, synthesize the exchange using a **Cayley-graph shortest path** over the available shuffle generators (minimize shuffle rounds → avoid instruction cliffs).

### C) Define the mechanism (compiler/runtime artifact)
**Artifact:** a two-level system:
1) **Compiler pass (“Relation Extractor”)** detects patterns:
   - MoE token routing maps
   - paged KV index arrays
   - gather/scatter pairs around dense compute
2) **Planner (“Relation Factorizer”)** chooses a strategy:
   - **Plan A (permute-execute-unpermute)**: generate a permutation that groups by page/expert, then run dense kernels on blocks
   - **Plan B (warp exchange)**: if group fits within a warp, generate minimal shuffle sequence
   - **Plan C (shared memory transpose / staged)**: if cross-warp, stage through shared with bank-aware swizzle (parametric for NVIDIA vs AMD)

This directly attacks:
- Stage 1 Cliff 2 (indirection)  
- Stage 1 Cliff 3 (instruction-round explosion)

### D) Concrete integration points (what you can build now)
- **For MoE**: reuse ecosystem reality instead of fighting it:
  - DeepSpeed/Megatron-style dispatch already defines the semantic phases; the compiler can fuse/recognize and specialize the local permutation kernels. ([microsoft.com](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/?utm_source=openai))  
  - DeepEP already exposes dispatch layout computation + async dependency hooks, which we can treat as a runtime backend. ([github.com](https://github.com/deepseek-ai/DeepEP?utm_source=openai))  

- **For paging**: recognize “gather from pages” and choose:
  - reorder by page-id when possible (inspector-executor)
  - otherwise, keep gather but use best movement primitive

- **For Blackwell-forward path**: PTX-level support exists for TMA tensor operations including tile gather/scatter variants in newer SM targets (e.g., SM_100 listings). ([nvidia.github.io](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html?utm_source=openai))  
  (Even if your primary evaluation is H100, this makes the work “Blackwell-ready” without new hardware assumptions.)

### E) Prototype plan (3–4 months)
**MVP**
1) Implement a **single-GPU relation factorizer** for MoE:
   - Input: `(token_indices -> expert_id)` map
   - Output: permutation indices + segment offsets  
2) Generate:
   - a Triton kernel that performs **stable group-by expert** (fast radix/binning)
   - expert MLP kernel on grouped tokens
   - unpermute kernel  
3) Add **warp-contained detection**:
   - if segment size <= warp tile, use shuffle-based exchange (reuse/extend Linear Layouts gather logic + optimal synthesis). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**Stretch**
- Multi-GPU: integrate with Megatron/DeepEP dispatch APIs for the all-to-all stage; compiler specializes the local permute/unpermute and selects overlap points. ([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html?utm_source=openai))  

### F) Evaluation plan
- **Microbench**:
  - synthetic gathers where index distribution crosses warp boundaries (to trigger failure of intra-warp-only paths) ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **MoE layer benchmarks**:
  - end-to-end latency for dispatch + expert MLP + combine  
- **Tooling**:
  - Nsight Compute / rocprof-compute instruction mix + memory throughput
  - Compare to baseline permute kernels in framework (Megatron/DeepSpeed-style)

### G) ASCII pipeline diagram
```text
[PyTorch/Inductor Graph]
      |
      v
[Detect relation: gather/scatter around dense op]
      |
      v
[Factorize R ≈ Π2 D Π1]
   |          |         |
   |          |         +--> if warp-contained: [Cayley shortest-path shuffle]
   |          |
   |          +--> if block-contained: [Shared-memory staged + swizzle]
   |
   +--> else: [Permutation kernels + dense block compute]
      |
      v
[Triton codegen + autotune knobs]
```

### H) Novelty check
- Systems already use specialized MoE dispatch kernels, but they are not derived from a general **relation algebra + compiler plan extractor**; they’re hand-designed per stack. ([microsoft.com](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/?utm_source=openai))  
- Linear Layouts optimizes gather only in a specific intra-warp case; RELAX generalizes to cross-warp and “paged/non-contiguous” regimes by changing the abstraction (relation instead of function). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

---

## 4) Decision matrix (scores + rationale)

Scoring: **5 = best** for Novelty/Relevance, **5 = highest risk** for Implementation Risk.

| Direction | Theoretical Novelty | Hardware Relevance (H100/Blackwell) | Implementation Risk (3–4 mo) | Notes |
|---|---:|---:|---:|---|
| **(1) MALP (SNF + Presburger)** | 4/5 | 4/5 | 3/5 | Directly attacks Linear Layouts’ stated power-of-two limitation ([arxiv.org](https://arxiv.org/html/2505.23819v3)) and real ragged/paged serving ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai)). Risk is moderate because scope can be limited to 2D/3D + interior/boundary splitting. |
| **(2) TAS (Transport Schedule IR for TMA)** | 4/5 | 5/5 | 4/5 | Most aligned with Hopper/Blackwell reality: descriptors + mbarrier + proxy fences are mandatory semantics ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai)). Risk is correctness complexity, but MVP can target CTA-only and reuse Triton descriptor APIs ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai)). |
| **(3) RELAX (Relations + factorization + optimal exchange)** | 5/5 | 4/5 | 4/5 | Strong ASPLOS angle: unify MoE/paging indirection into a compiler-recognized relational primitive; grounded in existing MoE pipelines (permute/all-to-all/unpermute) ([microsoft.com](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/?utm_source=openai)). Risk: building a robust inspector-executor and cost model quickly. |

---

## If you want the “most ASPLOS-proof” direction to lead with
If you can only push one into a paper-quality artifact by March 2026 timelines, I’d lead with:

1) **TAS (Direction 2)** as the flagship “HW/SW interface” contribution (TMA + proxy fences + schedule synthesis), because it’s the clearest *Hopper/Blackwell-specific* compiler gap backed by official semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/advaprogramming.html?utm_source=openai))  
2) **MALP (Direction 1)** as the “LLM serving realism” extension (ragged/piecewise) that makes Linear Layouts less brittle. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
3) **RELAX (Direction 3)** as either a second paper or a strong “systems workload” section if you can stabilize the runtime inspector story in time. ([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html?utmopenai))  

If you want, I can turn one chosen direction into an **ASPLOS-ready proposal skeleton** (problem statement → key insight → design → implementation plan → eval plan → risks) and explicitly map it onto Triton passes (`tt`/`ttg`/LLVM/PTX) and profiling metrics (Nsight sections to screenshot).

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [LEGO: A Layout Expression Language for Code Generation cal Mapping](https://arxiv.org/html/2505.08091)
3. [NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/?utm_source=openai)
4. [triton.language.make\_tensor\_descriptor — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai)
5. [PTX ISA 8.4](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html?utm_source=openai)
6. [3.2.dvanced Kernel Programming — CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/advanced-kernel-programming.html?utm_source=openai)
7. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.3.1/cuda-c-programming-guide/index.html?utm_source=openai)
8. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html?utm_source=openai)
9. [Understanding AMD GPU LDS and Bank Conflicts — Composable Kerne1.0 Documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai)
10. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180?utm_source=openai)
11. [Stateful Large Language Model Serving with Pensieve](https://arxiv.org/abs/2312.05516?utm_source=openai)
12. [DeepSpeed: Advancing MoE inference and training to power next-generation AI scale - Microsoft Research](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/?utm_source=openai)
13. [core.transformer.moe.token\_dispatcher — Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html?utm_source=openai)
14. [GitHub - deepseek-ai/DeepEP: DeepEP: an efficient expert-parallel communication library](https://github.com/deepseek-ai/DeepEP?utm_source=openai)
15. [Modeling Laut Abstractions Using Integer Set Relations](https://arxiv.org/abs/2511.10374?utm_source=openai)
16. [CUDA C++ Programming Guide — CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html?utm_source=openai)
17. [TritonOps — Triton documentation](https://triton-lang.org/main/dialects/TritonOps.html?utm_source=openai)
18. [Group GEMM — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html?utm_source=openai)
19async.bulk.tensor — CUDA Core Compute Libraries](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html?utm_source=openai)
