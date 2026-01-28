- **(C) Your Stage‑0 “constraint cliffs” framing is directionally correct:** the *dominant* gaps are not “better math for layouts”, but **legality + orchestration + bandwidth/latency roofs** that the seed formalisms mostly treat as out‑of‑scope. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **(A) Linear Layouts is the only seed paper that explicitly targets backend codegen** (Triton integration, completeness under shape ops); the other two are explicitly *foundational / modeling* rather than performance. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **(B) Hopper/H100 reality:** layout usefulness is gated by (i) **TMA tensor-map admissibility** and (ii) **async orchestration** (`cp.async.bulk.tensor` + `mbarrier` lifecycle/tx-count), not by expressiveness of a mathematical layout algebra. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **(B) Blackwell reality:** there is a new on‑chip tier (**Tensor Memory / TMEM**) with **dynamic allocation + warpgroup lane partitioning**, which turns “layout” into a *resource management + addressing* problem. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- **(C) Compute‑vs‑bandwidth asymmetry is extreme on both Hopper and Blackwell** (and even more on MI300X), so “layout” wins only when it materially increases reuse, vectorization, and overlap (pipeline depth), not when it merely makes mappings more elegant. ([nvidia.com](https://www.nvidia.com/en-eu/data-center/h100/))  
- **(B) NVIDIA swizzle is not “any permutation”:** TMA swizzle has **finite modes** and **alignment/inner-dimension validity requirements**; if the layout formalism can’t prove those, it can’t promise you reach the fast path. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html))  
- **(B) AMD MI300 reality:** LDS bank conflicts + MFMA‑driven access patterns are first‑order constraints, with profiling tool support for bank-conflict metrics—again, *hardware cost model*, not just mapping expressiveness. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  
- **(C) Practical bottleneck ranking (preview):** **(B) asynchrony/orchestration** and **(A) memory wall** dominate; **(C) descriptor legality** is a cliff; **(D) compile/search** is second-order unless you brute-force. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  

---

## Step 0 — Access Check (Seed Papers)

| Seed paper | Access | Readability notes |
|---|---:|---|
| (A) *Linear Layouts: Robust Code Generation…* | ✅ | HTML loads with full TOC + sections 1–9. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |
| (A) *Modeling Layout Abstractions Using Integer Set Relations* | ✅ | HTML loads (TOC + full text). ([arxiv.org](https://arxiv.org/html/2511.10374v1)) |
| (A) *Categorical Foundations for CuTe Layouts* | ✅ | PDF loads (174 pages). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) |

---

## Step 1 — Seed Truth (per paper)

### 1) Linear Layouts (ASPLOS’26) — what it *actually* asserts

**(A) Axioms / primitives (layout meaning):**  
- A layout is modeled as a **linear function over \(\mathbb{F}_2\)** acting on **bit‑vectors** (explicitly framed as binary‑matrix transformations on the hardware representation). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**(A) Closure / completeness guarantees (what the math buys):**  
- Claims **completeness under Triton’s shape operators** and integration into Triton’s backend layout engine (i.e., this is not purely paper math; it’s a compiler claim). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**(A) Explicit limitations (quoted/located):**  
- “Primary limitation” is **restriction to power‑of‑two shapes**; mitigation via masking.  
- **Flipping and slicing are not expressible** as linear layouts; suggested extension: **affine layouts**. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**(C) Concrete compiler artifact mapping + metric:**  
- Artifact: **Triton GPU backend layout propagation + layout conversion lowering** (the paper positions linear layouts as a backend representation). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Metric you can measure:  
  - vectorization width chosen (e.g., `.v4`/`.v8` patterns),  
  - shared-memory bank conflict rate (Nsight metric family `l1tex__data_bank_conflicts*`),  
  - instruction count for layout conversion (shuffle/permute). *(These are (C) because the paper doesn’t name Nsight counters, but it’s the direct mapping.)*  

---

### 2) ISL Relations (NVIDIA) — unification, not performance

**(A) Axioms / primitives:**  
- Models **CuTe layouts** and **Triton linear layouts** as **integer set relations (ISL)** to enable unified reasoning across systems. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- For the linear‑layout side, it explicitly relies on **binary vector space** structure and notes a key precondition: **dimension sizes must be powers of two**. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**(A) Closure properties (operations supported):**  
- Implements/claims a “complete suite” of layout manipulation algorithms using ISL operations: **composition, inversion, complement** (as a modeling + correctness story). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**(A) Explicit limitations (quoted/located):**  
- It explicitly says the work is “fundamentally theoretical and foundational” and that its **goal is not performance optimizations**. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- It also states a representability caveat: **if an index mapping is not strictly affine, “no straightforward layout representation exists for the given shape”** (though another shape of same size might work). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Complexity note: ISL ops have **worst‑case exponential complexity**, but they argue practical dimension counts are small in DL workloads. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**(C) Concrete compiler artifact mapping + metric:**  
- Artifact: an **ISL-based “isl-layout” translation / analysis toolchain**; usable as a compiler analysis/verification pass, not a kernel fast‑path. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Metric: compile‑time cost vs tensor rank / tiling depth (they point to “at most ~24 dims” under typical rank + tiling). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

---

### 3) Categorical Foundations for CuTe Layouts — “tractable” only

**(A) Axioms / primitives:**  
- Introduces categories (e.g., **Tuple** and **Nest**) with **functors** (inclusion + flattening) and claims an **adjoint equivalence** between these categorical views. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- Defines how a **nested tuple morphism encodes a layout** via a construction, and how to recover a “standard representation.” ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

**(A) Closure properties (what it closes under):**  
- At minimum, categorical composition gives you **compositionality of encoded layouts** (i.e., morphism composition corresponds to composing layout structure), and flattening is compatible with flattening layouts. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

**(A) Explicit limitations (quoted/located):**  
- They explicitly contemplate a “more tractable” category **not equivalent to Tuple** and **leave it to future work**. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- Key boundary: **a layout is encodable by their morphism construction iff it is “tractable”** (Proposition 3.2.2.9). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

**(C) Concrete compiler artifact mapping + metric:**  
- Artifact: **CuTe layout algebra** as used in CUTLASS-style codegen; categorical normal forms can reduce “case explosions” in compiler rewrite rules. *(Inference: paper is foundational; CUTLASS/CuTe usage is the practical target.)* ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Metric: number of distinct layout-conversion code paths (rewrite count) and compile-time normalization cost.

---

## Step 2 — Hardware Truth (H100 + Blackwell + MI300): top 5 dominating constraints

### (B) Constraint #1 — **TMA / `CUtensorMap` descriptor legality** (H100 / Hopper+)

**(B) Manual claim (what hardware requires):**  
A `CUtensorMap` (TMA tensor map) is **opaque** and must be created through CUDA APIs/PTX; using TMA requires satisfying **hard admissibility constraints**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

#### (B) Required `CUtensorMap` legality constraints (non‑exhaustive but must‑check)  ✅ *S2 satisfied here*

From `cuTensorMapEncodeTiled` (tiled TMA descriptor):  
- **Descriptor alignment:** `tensorMap` address **64B aligned**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Rank bounds:** `tensorRank` non‑zero and **\(\le 5\)**; if `interleave != NONE` then `tensorRank \(\ge 3\)`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Global base alignment:** `globalAddress` **16B aligned**; additional **32B alignment** when `interleave==32B` and for certain packed data types. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Dims:** `globalDim[i]` **non‑zero** and **\(\le 2^{32}\)**; packed types impose divisibility on `globalDim[0]`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Strides (bytes):** `globalStrides[i]` **multiple of 16** and **\(<2^{40}\)**; additional **multiple of 32** for `interleave==32B` and some packed types. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Box bounds:** `boxDim[i]` **non‑zero** and **\(\le 256\)**; if `interleave==NONE`, then `boxDim[0] * elementSize` must be **multiple of 16 bytes**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Element strides:** each `elementStrides[i]` **non‑zero** and **\(\le 8\)**; when `interleave==NONE`, **dimension‑0 stride is ignored** (“TMA doesn’t support the stride for dimension zero”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Swizzle legality coupling:**  
  - Swizzle modes are enumerated, and swizzle+interleave have restrictions (e.g., `interleave==32B` forces `swizzle==32B`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - When `interleave==NONE` and swizzle is used, **inner dimension must be \(\le\) swizzle size**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

From the CUDA Programming Guide’s TMA swizzle section (additional constraints often missed in “layout math”):  
- **Global memory alignment for swizzle:** “Global memory must be aligned to **128 bytes**.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html))  
- **Shared memory alignment for TMA swizzle:** shared memory is **required to be 128B aligned** when using TMA; the swizzle mapping has **fixed 16B granularity**; and violating inner-dimension requirements makes the instruction **invalid**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html))  

**(C) Compiler artifact mapping:**  
- Artifact: CUDA driver call `cuTensorMapEncodeTiled(...)` + PTX `cp.async.bulk.tensor` consuming `[tensorMap, tensorCoords]`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- Metric: static legality predicate (boolean) + dynamic “effective sectors/transactions” if you fall back to normal loads.

---

### (B) Constraint #2 — **Async tensor-copy contract (`cp.async.bulk.tensor`) + completion semantics**

**(B) Manual claim:**  
- `cp.async.bulk.tensor` initiates **asynchronous tensor copies** with explicit syntax over **1d–5d** and explicit completion mechanism options. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- It is **Hopper+**: “Requires `sm_90` or higher.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- For some ISA/type variants there are **hard box/stride/alignment/coordinate multiple** constraints (example restrictions shown for `.b6p2x16` on specific targets). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- Bulk async-group completion is explicit (`commit_group` / `wait_group`) and **does not guarantee memory ordering between copies within a group**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  

**(C) Compiler artifact mapping:**  
- Artifact: Triton/CUDA codegen must decide between  
  - **TMA path** (`cp.async.bulk.tensor … .mbarrier::complete_tx::bytes`) and  
  - **fallback** (`ld.global` → `st.shared` or `cp.async` classic). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- Metric (Nsight Compute): stalls from async underfill (`smsp__stall_*` families), and reduced tensor pipe utilization when WGMMA starves. *(Metric naming is Nsight‑tooling specific; concept is invariant.)*

---

### (B) Constraint #3 — **`mbarrier` lifecycle + tx-count phase completion (temporal orchestration is enforced)**

**(B) Manual claim:**  
- `mbarrier` is an **opaque object in memory** with a strict lifecycle: using operations other than `init` on an uninitialized object is **undefined behavior**; using non‑mbarrier ops on an initialized object is also **UB**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- Valid init `count` range is **\[1, \(2^{20}-1\)\]**, and **phase completion requires** both “pending arrivals == 0” **and** “tx-count == 0.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- Hopper introduces **tx-count tracking** to coordinate asynchronous transaction completion. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  

**(C) Compiler artifact mapping:**  
- Artifact: pipelined kernels encode a **software pipeline** with explicit `mbarrier.expect_tx` and the auto `complete_tx` side effect of `cp.async.bulk.tensor`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- Metric: barrier‑related stall fraction; incorrect counts lead to deadlock or bubbles (functional + performance cliff).

---

### (B) Constraint #4 — **Tensorcore operand descriptor encoding limits (WGMMA/MMA)**

**(B) Manual claim:**  
- Shared-memory multiplicands use a **64-bit matrix descriptor**, and **descriptor contents must be the same across all warps in the warpgroup**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- The descriptor encodes offsets with `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4`, implying **16-byte granularity** (quantization of representable byte offsets). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- Swizzle mode is encoded in a small field with **enumerated valid values**; some bit patterns are explicitly invalid. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  

**(C) Compiler artifact mapping:**  
- Artifact: codegen must pick tile shapes/strides and shared-memory base addresses such that the descriptor is encodable (16B granularity) and consistent across the warpgroup. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- Metric: if you miss encodability, you don’t just get slower—you may be forced off the WGMMA path entirely (tensor pipe utilization collapse).

---

### (B) Constraint #5 — **On-chip tiering & resource management (Blackwell TMEM vs SMEM; MI300 LDS + AGPR pressure)**

**(B) Blackwell TMEM (new tier):**  
- Tensor Memory is **dynamically allocated**; allocation unit is **32 columns**, allocated column count must be **power of two**, and allocation is by a **single warp** in a CTA. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- Warpgroup restriction: each warp can access only a **lane subset** (0–31, 32–63, 64–95, 96–127). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- `tcgen05.ld` requires **all threads in the warp specify the same `taddr`** (otherwise UB). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  

**(B) NVIDIA shared memory (baseline tier):**  
- Shared memory has **32 banks**; bank conflicts serialize accesses (with broadcast exceptions). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html))  

**(B) AMD MI300: LDS + MFMA (and how it constrains layout):**  
- LDS bank conflict behavior is explicitly documented (MI-series: **32 banks**, 4‑byte width; conflicts serialize). ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  
- MFMA is the throughput engine; MFMA fragments live in registers and may use **AGPRs**, changing register pressure/occupancy constraints. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html))  
- MLIR exposes AMD sparse MFMA as a first-class op with **specific supported MNK shapes** on gfx942 (MI300-class), encoding operand packing constraints at the IR level. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/AMDGPU/))  

**(C) Compiler artifact mapping:**  
- NVIDIA: choose between SMEM vs TMEM residency and allocate/manage TMEM (tcgen05 alloc/dealloc). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- AMD: choose LDS layouts that are bank-conflict-safe *under the MFMA read patterns* (often XOR-swizzle/padding). ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  
- Metric: achieved occupancy (register-limited), plus bank-conflict cycles (Nsight: shared bank conflicts; ROCm: LDS bank conflict metrics). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/tutorial/profiling-by-example.html))  

---

## Step 3 — Elephant-in-the-room diagnosis (ranked)

### (C) First, quantify the roofline pressure (compute vs bandwidth)

Using published peak specs (not normalized for dense vs sparse unless stated):

| Device | Peak tensor compute | HBM bandwidth | (C) Implied FP8 “compute/bw” threshold |
|---|---:|---:|---:|
| (B) NVIDIA H100 (SXM) | 3,958 TFLOPS FP8 (with sparsity) | 3.35 TB/s | \(\approx 1182\) FLOPs/byte |
| (B) NVIDIA HGX/DGX B200 (8 GPUs) | 72 PFLOPS FP8/FP6 (sparse spec) | 64 TB/s | \(\approx 1125\) FLOPs/byte |
| (B) AMD MI300X (OAM) | 20.9 PFLOPs FP8 | 5.3 TB/s | \(\approx 3943\) FLOPs/byte |

Sources for the raw numbers. ([nvidia.com](https://www.nvidia.com/en-eu/data-center/h100/))  

Interpretation. **Most real kernels are below \(\sim 10^3\) FLOPs/byte at FP8 unless they reuse data heavily**, which means “layout” only matters insofar as it enables reuse and overlap (tiling, residency, pipelining). *(This paragraph is (C) inference from the table.)*

### (C) Ranked bottlenecks (with falsification-first rationale)

1. **(B) Latency hiding / asynchrony orchestration (B)**  
   If you don’t pipeline TMA/async copies + barriers correctly, tensor cores starve; the ISA explicitly forces you to manage completion (`commit_group`/`wait_group`, `mbarrier` tx-count). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  

2. **(B) Memory bandwidth wall (A)**  
   Peak compute is orders of magnitude larger than HBM bandwidth; layouts that don’t increase locality/reuse can’t win. ([nvidia.com](https://www.nvidia.com/en-eu/data-center/h100/))  

3. **(B) Descriptor legality cliffs (C)**  
   TMA and WGMMA have hard encodability constraints (rank/stride/box/alignment; descriptor 16B quantization; finite swizzle legality). This is a *binary cliff* not a smooth cost model. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

4. **(B) Bank conflicts (SMEM/LDS) (and finite swizzle modes)**  
   Even when bandwidth is available, bank conflicts and invalid swizzle choices serialize traffic. NVIDIA and AMD both document bank structures and conflict behavior, and AMD exposes conflict metrics in tooling. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html))  

5. **(B) Register pressure / fragment packing → occupancy collapse**  
   WGMMA matrix descriptors and MFMA/AGPR usage make register footprint a hard constraint; layouts that look “nice” but inflate live ranges reduce occupancy and make (1) worse. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  

6. **(A) Compilation/search cost (D)**  
   Seed formalisms can reduce conversion explosion, but if legality predicates (TMA/mbarrier/descriptor encodings) aren’t integrated, the search space still explodes in practice. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

7. **(C) Dynamic memory management (E)** (KV cache, fragmentation, paging)  
   Important for end-to-end LLM serving, but it’s a different layer than these layout papers. *(No strong citation in the provided sources; keep as inference.)*

---

## Seed‑vs‑Hardware Matrix (requested)

> Column discipline: **Seed columns are (A)**, **hardware columns are (B)**, **diagnosis/extension columns are (C)**.

| Seed_Math_Concept | Explicit_Seed_Limit | Hardware_Feature | Hardware_Legality_Constraint | Performance_Cliff_Mode | Minimal_Relaxation/Extension |
|---|---|---|---|---|---|
| (A) “Layout as linear map over \(\mathbb{F}_2\)” (Triton linear layouts) ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | (A) Power‑of‑two shape restriction; flip/slice not expressible; suggests affine extension ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | (B) TMA tensor maps + `cp.async.bulk.tensor` fastpath ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | (B) Rank \(\le 5\), stride multiple‑of‑16, boxDim \(\le 256\), alignment rules; swizzle validity/alignment rules ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | (C) Layout is “expressible” but **not TMA‑admissible** ⇒ forced fallback path (more instructions, lower BW, less overlap) | (C) Add a **constraint solver layer**: layout → (tensorMap params) + proof of admissibility; treat as first‑class in IR (e.g., MLIR attrs) |
| (A) Generic swizzle constructions for bank conflicts (seed treats swizzle as mapping) ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | (A) Not stated as “finite-mode + alignment-gated” (seed focus is expressiveness/optimality) ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | (B) TMA swizzle modes (finite) + validity/alignment + 16B granularity ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)) | (B) “Instruction invalid” if inner-dimension requirements fail; global memory aligned 128B; SMEM aligned 128B ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)) | (C) “Optimal” swizzle in algebra may be **unrealizable** in hardware mode set; or realizable only with padding/alignment overhead | (C) Restrict swizzle search space to hardware modes; expose padding/alignment as explicit cost terms |
| (A) ISL integer set relations unify CuTe + linear layouts ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | (A) Goal explicitly **not performance optimization** ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | (B) Bulk async groups + no ordering guarantee ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)) | (B) No memory ordering guarantee among tensor copies in same bulk group ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)) | (C) Correctness of mapping ≠ correctness/performance of **temporal schedule**; pipeline bugs or bubbles persist | (C) Extend relation model with a **schedule dimension** (pipeline stages, barrier phases) and resource constraints |
| (A) CuTe layout inversion/complement/composition as formal ops ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | (A) Non‑strictly affine mapping ⇒ “no straightforward layout representation” for a given shape ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | (B) WGMMA shared-memory descriptor encoding ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)) | (B) Descriptor offsets quantized: encode uses `>> 4` (16B); swizzle field enumerated with invalid bit patterns ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)) | (C) Inverse/composition may yield offsets/strides that are **not encodable** ⇒ can’t use WGMMA operand descriptors | (C) Add **quantization-aware algebra**: legality as “representability in descriptor bitfields” |
| (A) Categorical encoding of “tractable layouts” via morphisms ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | (A) Only tractable layouts encodable; broader “tractable category” deferred ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | (B) TMA descriptor admissibility (structural constraints) ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | (B) Hard bounds on rank/box/stride/alignment for tensor maps ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | (C) Even “tractable” algebraically may violate TMA admissibility; conversely TMA admissible is a strict subset of “nice” layouts | (C) Define a new “hardware-tractable” subcategory: objects are layouts + witnesses for admissible param tuples |
| (A) “Completeness under Triton shape ops” ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | (A) Requires power-of-two; affine extension needed for slice/flip ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | (B) `mbarrier` tx-count phase completion rules ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)) | (B) Phase completes only when pending arrivals == 0 and tx-count == 0; strict lifecycle rules (UB otherwise) ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)) | (C) Layout completeness doesn’t prevent **pipeline deadlocks** or stalls; temporalorrectness is separate | (C) Add a “temporal type system”: pipeline stages with verified tx accounting |
| (A) Bank conflict optimization as goal (seed) ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | (A) Seed doesn’t commit to AMD LDS phase rules (different from NVIDIA SMEM model) ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | (B) AMD LDS bank mapping + phase-based conflict rules (ds_read/write_b128) ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-confEADME.html)) | (B) Conflicts depend on wave lane grouping + instruction width; MI-series LDS has 32 banks ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)) | (C) A swizzle “optimal” under NVIDIA’s 32-bank model can still be bad on AMD due to different phase groupings | (C) Parameterize bank-cost model by ISA + instruction width; integrate rocprof bank-conflict metrics into autotuning |
| (A) Layout reasoning is system-independent (mapping abstn) ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | (A) (ISL paper) explicitly not about performance; (categorical) focuses on tractability ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | (B) Blackwell TMEM is dynamically allocated & warpgroup-partitioned ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)) | (B) Allocation unit 32 columns, power-of-two; single-warp alloc; lane partitions; collective access rules ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)) | (C) Layout isn’t just mapping; it’s **resource allocation + addressability + deallocation** | (C) Extend IR with explicit TMEM allocation objects + lifetime analysis (like SMEM but dynamic + warpgroup-scoped) |

---

## 3 Performance Cliffs (SASS-level intuition) — requested

### Cliff 1 — **Async pipeline underfilled / barrier misuse ⇒ tensor cores starve**
- **Symptom (profiler terms):**  
  - Low tensor-pipe utilization MA/MMA issue rate low),  
  - High “waiting”/barrier stall fraction,  
  - Poor overlap of memory and compute. *(Tool-specific names vary; the phenomenon is deterministic.)*  
- **(B) Hardware root cause:** bulk async groups require explicit commit/wait and provide no ordering guarantee; mbarrier phase completion is tx-count sensitive. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- **(C) Formalism blind spot:** seed layout algebras prove map equivalence but don’t model **tx-count accounting** or **phase completion**.  
- **(C) What hand-tuned kernels do:** multi-stage SMEM pipelines: `cp.async.bulk.tensor … .mbarrier::complete_tx::bytes`, `mbarrier.expect_tx`, carefully chosen wait distances, and WGMMA scheduled to consume tiles as soon as ready. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  

### Cliff 2 — **Shared/LDS bank conflict storm ⇒ effective bandwidth collapses**
- m:**  
  - NVIDIA: elevated shared bank conflict counters (bank-conflict serialization). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html))  
  - AMD: rocprof-compute bank-conflict cycles rise; bank conflict rate increases with active lanes. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/tutorial/profiling-by-example.html))  
- **(B) Hardware root cause:**  
  - NVIDIA SMEM: 32 banks, 32-bit word mapping, conflicts serialize. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html))  
  - AMD LDS: 32 banks (MI-series), instruction-width-dependent “phase” rules; XOR-based swizzle often used to eliminate conflicts. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  
- **(C) Formalism blind spot:** “bank conflicts” are not a single uniform cost function across vendors; the seed abstractions don’he ISA-dependent phase grouping rules.  
- **(C) What hand-tuned kernels do:** NVIDIA uses TMA swizzle modes (finite) when legal; AMD uses XOR preshuffle / padding tuned to MFMA read patterns. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html))  

### Cliff 3 — **Descriptor encodability/legality cliff ⇒ lose the fast path entirely**
- **Symptom:**  
  - Sudden step-function drop: kernel falls back to scalar/vector LD/ST path, more instructions, leverlap.  
  - Sometimes: compile-time rejection or runtime “invalid instruction” if constraints violated. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html))  
- **(B) Hardware root cause:**  
  - TMA tensor map has strict rank/stride/box/alignment constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - WGMMA matrix descriptor encodes offsets in 16B chunks and restricts sle bit patterns. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- **(C) Formalism blind spot:** seed models treat layout as mapping; they don’t make “descriptor representability” a first-class type constraint.  
- **(C) What hand-tuned kernels do:** pick tile sizes and alignments that satisfy (i) tensorMap admissibility and (ii) descriptor encodability; sometimes introduce padding solely to cross legality thresholds. ([docs.nvidia.com](httpss.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

---

## Stage‑1 Verdict (10 bullets: correct vs incomplete vs wrong)

1. **(C) Correct:** You identified “arbitrary layout/stride” vs **TMA admissibility** as a real cliff; CUDA explicitly imposes rank/stride/box/alignment rules. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
2. **(C) Correct:** You flagged swizzle as **finite and validity/alignmend**; CUDA PG makes invalidity/alignment explicit. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html))  
3. **(C) Correct:** You flagged async copy as a *typed/boxed/aligned contract*; PTX shows variant-specific hard constraints and sm-target gating. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
4. **(C) Correct:** You called out “mma/wgmma needs special layout” but the deeper issue is *criptor bitfield limits** (16B quantization + enumerated swizzles). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
5. **(C) Correct:** You called out Blackwell TMEM as more than “special memory”; PTX defines explicit allocation + lane partition + collective access rules. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
6. **(C) Incomplete:** Your Stage‑0 swizzle set “none/32/64/128 that the Driver API enumerates additional 128B atomicity variants (e.g., `_ATOM_*`) with datatype-specific support. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
7. **(C) Incomplete:** The real orchestration bottleneck is not just “async copy exists,” but **mbarrier tx-count phase completion + UB lifecycle rules**, which dominate correctness and performance. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/inhtml?highlight=st+async))  
8. **(C) Incomplete:** For Blackwell, the *architectural* bottleneck is now also **tiering decisions (SMEM vs TMEM) and TMEM allocation lifetime**, not just layout mapping. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
9. **(C) Incomplete:** AMD MI300 needs explicit treatment: LDS bank conflicts have ISA-dependent phase rules and profiling support; MFMA/AGPR pressure makes “layout conversion” a register-allocation lem too. ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html))  
10. **(C) Wrong only in scope expectation:** expecting seed papers to encode *all* legality + scheduling is mismatched to their stated goals—two of them explicitly frame themselves as foundational rather than performance systems papers. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

---

### What must be completed in the next stage
- (C) Formalize a **“hardware-admissible laype”**: a layout term plus a proof object that it can be lowered to (i) `CUtensorMap` params and (ii) WGMMA descriptor fields (quantization + swizzle legality).  
- (C) Add a **temporal schedule IR**: pipeline stages annotated with tx-count accounting + wait distances, and verify against PTX mbarrier semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async))  
- (C) Use profiling-driven cost models: Nsight shared bank conflicts on NVIDIA; rocprof L bank conflict metrics on AMD; incorporate into autotuning rather than pure algebraic optimality. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/tutorial/profiling-by-example.html))

---
Learn more:
1. [https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group\_\_CUDA\_\_TENSOR\_\_MEMORY.html](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
2. [https://arxiv.org/html/2505.23819v3](https://arxiv.org/html/2505.23819v3)
3. [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=st+async)
4. [https://www.nvidia.com/en-eu/data-center/h100/](https://www.nvidia.com/en-eu/data-center/h100/)
5. [https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
6. [https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
7. [https://arxiv.org/html/2511.10374v1](https://arxiv.org/html/2511.10374v1)
8. [https://arxiv.org/pdf/2601.05972v1](https://arxiv.org/pdf/2601.05972v1)
9. [https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
10. [https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware\_implementation.html](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html)
11. [https://mlir.llvm.org/docs/Dialects/AMDGPU/](https://mlir.llvm.org/docs/Dialects/AMDGPU/)
12. [https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/tutorial/profiling-by-example.html](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/tutorial/profiling-by-example.html)
