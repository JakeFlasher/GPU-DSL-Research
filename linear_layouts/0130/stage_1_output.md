- **C (Context Verdict):** The three seed formalisms are *spatially* strong (they formalize coordinate→index maps and/or algebra over them) but remain *hardware-legality weak*: they do not, by themselves, guarantee you can even *encode* the layout into H100/Blackwell descriptors (TMA, WGMMA/TCGen05, TMEM). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **C (Context Verdict):** The most consequential missing dimension is **temporal orchestration** (async copy + barrier phase/tx-count contracts). Layout equivalence is irrelevant if your pipeline deadlocks, stalls, or is forced into a slow path. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html))  
- **C (Context Verdict):** On **H100**, the “layout usefulness frontier” is dominated by **TMA eligibility** (tensor-map legality + swizzle inner-dimension + alignment) and by **mbarrier completion correctness**—not by abstract closure under layout operations. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **C (Context Verdict):** On **Blackwell**, the on-chip tiering story changes: **TMEM** (tensor memory) plus `tcgen05.*` shifts the balance away from register-only accumulation, but introduces new legality/collective constraints (warpgroup lane partitions, per-warp collective addressing, alloc/dealloc rules). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  
- **C (Context Verdict):** The compute-vs-bandwidth gap keeps widening: even with higher HBM bandwidth, peak tensor-core rates grow faster, so layout work is mostly about **increasing arithmetic intensity** (reuse) and **feeding compute** via async pipelines. ([nvidia.com](https://www.nvidia.com/en-gb/data-center/h100/))  
- **C (Context Verdict):** For **MI300 (CDNA 3)** the correctness/perf “grain” is **wave64**; any layout formalism intended to be portable must explicitly reason about warp32 vs wave64 distribution, and that is not present in the seed math as stated. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.0/conceptual/pipeline-metrics.html))  
- **C (Context Verdict):** The “bottleneck” is usually not a single item: for kernel generators, it is a **ranked stack**: (1) orchestration correctness, (2) descriptor legality cliffs, then (3) bandwidth/occupancy ceilings. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html))  
- **C (Context Verdict):** The right research target is a **typed constraint system** that sits *between* seed layout algebra and codegen, emitting *both* legality proofs (descriptor encodings) and performance predictions (Nsight/rocprof metrics). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVVMDialect/))  

---

## Step 0 — Access Check (seed papers are readable)

| Paper | Access | Evidence |
|---|---|---|
| (1) *Linear Layouts: Robust Code Generation… using* \( \mathbb{F}_2 \) | ✅ Readable (arXiv HTML) | Page opens with full TOC + content. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |
| (2) *Modeling Layout Abstractions Using Integer Set Relations* | ✅ Readable (arXiv HTML) | Page opens with TOC + sections 1–8. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) |
| (3) *Categorical Foundations for CuTe Layouts* | ✅ Readable (arXiv PDF) | PDF opens; 174 pages. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) |

---

## Step 1 — Seed Truth (per paper)

### Paper 1: Linear Layouts (\( \mathbb{F}_2 \)) — *Seed Truth*

**A (Seed Paper Claim): Axioms / core objects**
- Layouts are modeled as **linear maps over** \( \mathbb{F}_2 \): “binary matrices acting on the bits” of hardware representations (their unifying abstraction for swizzles/permutations/broadcast-like structure). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- The compiler target is explicit: the formalism is integrated into **Triton’s GPU backend (MLIR-based)** to drive layout propagation + lowering. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**A (Seed Paper Claim): Closure properties**
- They claim (and prove) a **closed family of distributed layouts** under Triton shape ops (forward/backward closure under reshape/transposes/split/join/broadcast-like ops). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**A (Seed Paper Claim): Explicit limitations (exact section + quote)**
- In **§8 Conclusions**, they state: “**primary limitation … restriction to power-of-two shapes**” and that “**flipping and slicing are not expressible**” (proposed remedy: affine extension). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**C (Our Inference): What this guarantees / what it does *not* (compiler artifact + metric)**
- Guarantees are **about spatial reasoning** (layout equivalence, construction, closure under shape ops).  
- It does *not* guarantee **descriptor encodability** into TMA/TMEM/WGMMA forms; you still need a legality layer that can be checked/constructed at codegen time (e.g., emit `cp.async.bulk.tensor` only when TMA constraints pass). The measurable fallout is: kernel shifts from “async-copy fed” to “LD/ST fed,” observable as reduced memory/SM throughput. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

---

### Paper 2: ISL Integer-Set Relations — *Seed Truth*

**A (Seed Paper Claim): Axioms / core objects**
- A CuTe layout is treated as an \(n\)-D coordinate space → 1-D index mapping parameterized by **shape/stride**, modeled as **integer set relations** in ISL. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- ISL relations support **composition/inverse/domain/range** as first-class operations (their algebraic “closure engine”). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**A (Seed Paper Claim): Closure properties**
- Closure is inherited from ISL: layout operations are implemented as ISL relation operations (compose, inverse, etc.). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**A (Seed Paper Claim): Explicit limitations / assumptions (exact section + quote)**
- In the CuTe composition discussion, they explicitly assume away CuTe’s “implicit layout promotion” corner: “**we assume … no holes … no implicit layout promotion is necessary**.” ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- In **§8 Conclusions**, they state the scope: “**focuses on establishing mathematical foundations rather than runtime performance optimization**.” ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- In **§6 Complexity**, they acknowledge worst-case cost: ISL operations “**exhibit worst-case exponential complexity**” (and argue typical ranks are small). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**C (Our Inference): What this guarantees / what it does *not* (compiler artifact + metric)**
- This is strongest at **correctness predicates** (domain/range bounds synthesis to avoid OOB under composition holes).  
- It still does not model **hardware-legal descriptor fields** (alignment enums, swizzle legality tables, barrier semantics). Without that, the measurable gap appears as “provably correct mapping” but “cannot emit the fast instruction,” i.e., misses throughput. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

---

### Paper 3: Categorical Foundations (CuTe) — *Seed Truth*

**A (Seed Paper Claim): Axioms / core objects**
- They build categories **Tuple** and **Nest**, whose morphisms “give rise to layouts,” explicitly **focusing on a class of tractable layouts**. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- “Tractable” is defined by an explicit arithmetic condition (divisibility structure on strides after sorting). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

**A (Seed Paper Claim): Closure properties**
- They define operations on morphisms and prove compatibility with layout operations (composition, logical product/division) *within the tractable class*; they also give algorithms for composing tractable layouts via refinements. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

**A (Seed Paper Claim): Explicit limitations (exact section + quote)**
- The limitation is structural: they explicitly restrict to tractable layouts; moreover, they give a concrete counterexample of a flat layout that is “**not tractable**.” ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

**C (Our Inference): What this guarantees / what it does *not* (compiler artifact + metric)**
- This gives **compile-time algebraic structure** (good for canonicalization, equivalence, predictable composition).  
- It does not address the “GPU reality layer”: descriptor encodings, async barriers, and memory-tier constraints. Thus, it cannot predict (or ensure) improvements in Nsight tables like SM throughput, memory pipes, bank conflicts, or occupancy. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html))  

---

## Step 2 — Hardware Truth (H100 + Blackwell + MI300): Top 5 hardware-enforced constraints

> Format per constraint: **A (Seed)** / **B (Manual/Official)** / **C (Inference with compiler artifact + metric)**.

### 1) TMA / `CUtensorMap` legality (descriptor cliffs dominate eligibility)

**A (Seed Paper Claim):**
- Paper 1’s layout algebra is linear-over-\(\mathbb{F}_2\) and only later mentions TMA as a hardware boundary in evaluation; it does not claim to enforce `CUtensorMap` field legality. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Paper 2 models layouts/swizzles as integer relations; it does not claim to model CUDA Driver tensor-map encoding constraints. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Paper 3 focuses on CuTe layout algebra (tractable layouts), not on driver/PTX descriptor packing rules. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

**B (Hardware Manual Claim): `CUtensorMap` / TMA legality constraints you must satisfy**  
(Enumerated per rule S2)

- **Rank bounds**: `tensorRank` must be \(\le 5\); if `interleave != NONE`, then `tensorRank >= 3` (and the im2col API has its own stricter rank requirements). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Global address alignment**: `globalAddress` must be **16B-aligned**, and **32B-aligned** when `interleave == 32B` (and additional packed-type cases require stricter alignment). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Tensor sizes**: `globalDim[i]` must be non-zero and \(\le 2^{32}\); packed sub-byte formats impose extra multiples (e.g., `globalDim[0]` multiple of 128 or 2 depending on type). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Strides**: `globalStrides[i]` (bytes) must be a **multiple of 16** and \(< 2^{40}\); becomes **multiple of 32** under `interleave==32B` and for certain packed types. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Box bounds**: `boxDim[i]` must be non-zero and \(\le 256\); if `interleave==NONE`, `boxDim[0]*elementSize` must be a multiple of **16B**; some packed types force `boxDim[0]==128`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Element strides**: `elementStrides[i]` must be non-zero and \(\le 8\); and **when `interleave==NONE`, `elementStrides[0]` is ignored** (“TMA doesn’t support stride for dimension 0”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Swizzle is enumerated + cross-constrained**: driver exposes swizzle enums (NONE/32B/64B/128B + atomic variants) and imposes constraints like: `interleave==32B` ⇒ `swizzle==32B`; plus packed-type restrictions on allowed swizzles; plus “inner dimension” bounds (e.g., 64B swizzle ⇒ inner dim ≤ 64). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Programming-guide swizzle validity**: applying a TMA swizzle requires **global memory 128B alignment** and inner-dimension requirements; otherwise the instruction is “invalid.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html))  

**C (Our Inference): compiler artifact + measurable metric**
- **Compiler artifact mapping:**  
  - Triton/MLIR/CUTLASS must *type-check* layouts into a “TMA-encodable layout type” before emitting `cp.async.bulk.tensor` + encoded tensor-map.  
- **Metric symptom:** if you fail legality, kernels typically lose TMA and fall back to conventional global→shared movement, visible as increased DRAM traffic (`dram__bytes_read.sum.per_second`) and/or reduced SM throughput (`sm__throughput.pct_of_peak_sustained_active`). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2019.5.1/NsightComputeCli/index.html))  

---

### 2) `cp.async.bulk.tensor` + completion mechanisms (temporal legality, not just spatial layout)

**A (Seed Paper Claim):**
- Seed formalisms treat “copy/reorder” largely as compositional mappings; they do not encode the **completion mechanism contract** (mbarrier vs bulk-group semantics). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**B (Hardware Manual Claim):**
- PTX defines `cp.async.bulk.tensor` with **two distinct completion mechanisms**: `.mbarrier::complete_tx::bytes` or `.bulk_group`, and these have different sequencing/visibility implications. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  
- PTX also defines extensive **type-/arch-specific restrictions** for tensor copy instructions (notably for sub-byte packed types and swizzle-mode support, coordinate constraints, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  

**B (Hardware Manual Claim): `mbarrier` correctness is a contract**
- `mbarrier` is an opaque `.b64` object, **8-byte aligned**, in shared memory. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html))  
- Phase completion requires **both** pending-arrival count **and** tx-count to reach zero; `expect-tx` increments tx-count and `complete-tx` decrements it. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html))  

**C (Our Inference): compiler artifact + measurable metric**
- **Compiler artifact mapping:** NVVM/MLIR has explicit ops for mbarrier tx-count tracking (e.g., `expect-tx`), which must be scheduled consistently with async copies. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVVMDialect/))  
- **Metric symptom:** broken/under-filled async pipelines show up as latency-bound behavior: “Achieved Occupancy” may look fine (`sm__warps_active.avg.pct_of_peak_sustained_active`) yet “No Eligible”/stalling dominates and memory pipes are underutilized. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750))  

---

### 3) Shared-memory bank conflicts + swizzle validity (hardware bank model, not “any permutation”)

**A (Seed Paper Claim):**
- Paper 1 explicitly targets bank conflicts and even gives an “optimal swizzling” algorithm in its model; however, the seed math does not itself enforce **enumerated swizzle legality** of TMA modes. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Paper 2 models swizzles via bit-level relations; it does not assert alignment/inner-dimension legality as required by CUDA TMA swizzle modes. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**B (Hardware Manual Claim):**
- Shared memory has **32 banks**, successive 32-bit words map to successive banks; bank conflicts serialize requests. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html))  
- Nsight Compute explicitly treats “Bank Conflicts” as a measurable serialization factor (requests split into multiple conflict-free requests). ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2021.3/ProfilingGuide/index.html))  
- TMA swizzle patterns are **valid only under specific alignment + inner-dimension requirements**; otherwise instruction is invalid. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html))  

**C (Our Inference): compiler artifact + measurable metric**
- **Compiler artifact mapping:** in Triton this manifests as layout-selection + emission of swizzled shared-memory layouts (or padding), and in CuTe/CUTLASS as choosing compatible swizzle/interleave modes for TMA copy objects.  
- **Metric symptom:** Nsight Compute “Shared Memory” tables show elevated “Bank Conflicts” / “Wavefronts,” even if DRAM bandwidth is fine. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2021.3/ProfilingGuide/index.html))  

---

### 4) Warp/warpgroup/wavefront collectives constrain layout distribution

**A (Seed Paper Claim):**
- Seed papers reason about distributing indices over “threads/warps” abstractly, but do not encode **collective execution constraints** of certain instructions (all threads must participate with consistent operands). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

**B (Hardware Manual Claim): Blackwell TMEM is collective + lane-partitioned**
- PTX: Tensor Memory is dynamically allocated; allocation is in **columns (min 32, power-of-two)**, and a column allocation allocates **all 128 lanes**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  
- PTX: TMEM is partitioned so each warp in a warpgroup can access only a **lane range** (warp0: 0–31, … warp3: 96–127). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  
- PTX: `tcgen05.ld` is warp-collective: **all threads in the warp must specify the same `taddr`**, else behavior is undefined. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  

**B (Hardware Manual Claim): MI300/CDNA has a different grain**
- ROCm docs: on CDNA accelerators the **wavefront size is 64** work-items (wave64). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.0/conceptual/pipeline-metrics.html))  

**C (Our Inference): compiler artifact + measurable metric**
- **Compiler artifact mapping:** portability requires an explicit “execution-grain type” (warp32 vs wave64 vs warpgroup128) in the IR, because it changes fragment mapping, legality of collectives, and bank behavior.  
- **Metric symptom:** mismatched grain shows up as divergence/underutilization and “Achieved Occupancy” not translating into throughput. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750))  

---

### 5) Compute-vs-bandwidth asymmetry + on-chip tiering (SMEM vs TMEM vs registers)

**A (Seed Paper Claim):**
- Paper 2 explicitly says it is not optimizing runtime performance; Paper 3 is math/structure-first; Paper 1 proposes adding “hardware measurements” in future work. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

**B (Hardware Manual/Official Claim): the asymmetry is real (spec numbers)**
- **H100**: HBM bandwidth is on the order of **3.35 TB/s** (SXM) with very high tensor-core peak rates. ([nvidia.com](https://www.nvidia.com/en-gb/data-center/h100/))  
- **Blackwell**: HBM bandwidth per GPU is cited as **8 TB/s**, and NVIDIA explicitly positions memory scaling as critical for large-model inference. ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  
- **MI300X**: HBM3 bandwidth per OAM is cited as **5.3 TB/s** with very high FP16/BF16 peak. ([amd.com](https://www.amd.com/en/products/accelerators/instinct/mi300/platform.html))  
- **Blackwell on-chip tiering**: PTX defines a distinct **Tensor Memory** with separate access + allocation semantics (not “just shared memory”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  

**C (Our Inference): compiler artifact + measurable metric**
- **Compiler artifact mapping:** the layout system must co-design (tile sizes, staging depth, and memory tier) to raise arithmetic intensity; otherwise you remain bandwidth-bound regardless of spatial elegance.  
- **Metric symptom:** Nsight Compute classifies kernels via throughput/occupancy indicators; roofline-like reasoning uses DRAM metrics (`dram__bytes_read.sum.per_second`) vs SM throughput (`sm__throughput…`). ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750))  

---

## Table — Seed-vs-Hardware Matrix (required)

> Convention inside cells: **A:** Seed Paper Claim, **B:** Hardware Manual/Official Claim, **C:** Our Inference.

| Seed_Math_Concept | Explicit_Seed_Limit | Hardware_Feature | Hardware_Legality_Constraint | Performance_Cliff_Mode | Minimal_Relaxation/Extension |
|---|---|---|---|---|---|
| A: Layouts as linear maps over \( \mathbb{F}_2 \) (Paper 1) ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | A: Power-of-two shapes; flip/slice not expressible without affine extension (§8). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | B: TMA tensor maps + `cp.async.bulk.tensor` | B: `globalAddress` 16B/32B aligned; `globalStrides` multiple of 16/32; `boxDim[i]≤256`; swizzle/interleave coupling + inner-dimension caps. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | C: “TMA-or-bust” cliff: if illegal, you fall back to higher instruction count + lower overlap, losing BW/SM throughput. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2019.5.1/NsightComputeCli/index.html)) | C: Add a **legality-typed layout IR** that can synthesize `CUtensorMap` fields + reject/repair (pad, change box, change swizzle). |
| A: Layouts as ISL relations (Paper 2) ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | A: Assumes away “holes” / implicit promotion in composition (“we assume… no holes”). ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | B: `mbarrier` tx-count completion gating async ops | B: `.b64`, 8B aligned; phase completes only when pending arrivals == 0 **and** tx-count == 0. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html)) | C: “Looks correct but stalls” cliff: spatial mapping correct, but async pipeline never reaches steady-state. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750)) | C: Extend ISL model with **temporal events** (issue/commit/wait + tx-count accounting) tied to emitted NVVM/PTX ops. |
| A: Categorical tractable layouts (Paper 3) ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | A: Restricted to “tractable” class; non-tractable example exists. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | B: Blackwell TMEM (`tcgen05.*`) | B: TMEM is dynamically allocated in columns; lane access partitioned by warp; `tcgen05.ld` requires same `taddr` across warp. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html)) | C: “Tier-mismatch” cliff: ignoring TMEM yields register pressure/occupancy collapse; using TMEM without constraints yields UB/slowdowns. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2019.5.1/NsightComputeCli/index.html)) | C: Add a **memory-tier type system** (Reg/SMEM/TMEM) with legality + cost model; map categorical ops to CuTe/CUTLASS primitives + Nsight metrics. |
| A: Swizzle as algebraic transform (all seeds, implicitly) ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | A: Paper 2 focuses on foundations, not perf; Paper 1’s swizzle is algorithmic, not ISA-enum constrained. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | B: TMA swizzle patterns are enumerated and validity-checked | B: Global memory 128B aligned; inner dimension must meet table requirements; else instruction invalid. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)) | C: Bank-conflict cliff: perf collapses via split transactions; Nsight “Bank Conflicts/Wavefronts” rise. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2021.3/ProfilingGuide/index.html)) | C: Add **enumerated swizzle typing** + constraint solving (alignment, inner-dim, interleave coupling). |
| A: Portability via abstract “threads over data” (implicit) | A: No explicit encoding of warp32 vs wave64 distribution | B: MI300/CDNA uses wave64 | B: Wavefront size is always 64 on CDNA accelerators. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.0/conceptual/pipeline-metrics.html)) | C: “Portability cliff”: a layout good on warp32 can be structurally wrong/slow on wave64 due to different collaboration granularity. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.0/conceptual/pipeline-metrics.html)) | C: Make execution grain explicit in IR (warp/wave/warpgroup) + expose as tuning dimension with profiler feedback. |
| A: “Math-first” modeling (Paper 2 & 3) ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | A: “Foundations rather than runtime performance optimization.” ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | B: Hardware is throughput-classified in profilers | B: Nsight compute profiles occupancy/throughput and classifies bottlenecks; achieved occupancy metric exists (`sm__warps_active.avg.pct_of_peak_sustained_active`). ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750)) | C: “Cost-model gap”: without a metric-linked model, the math cannot rank layouts/schedules in the space that matters. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750)) | C: Integrate a **measurement-backed cost model** (Nsight/rocprof counters) as a first-class semantics layer. |

---

## 3 Performance Cliffs (SASS-level intuition) — required

### Cliff 1 — **TMA Eligibility / TensorMap Legality Cliff** (H100/Hopper-class)

- **Symptom (Nsight Compute metrics / tables):**  
  - DRAM throughput rises but SM throughput doesn’t: `dram__bytes_read.sum.per_second` high while `sm__throughput.pct_of_peak_sustained_active` remains modest. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2019.5.1/NsightComputeCli/index.html))  
  - You don’t see the expected async-copy feeding behavior; kernels look “memory-throughput bound” rather than compute-fed. ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/metric-references-and-description/111750))  
- **A (Formalism blind spot):**  
  - Seed layout equivalence does not imply “TMA-encodable”: rank/stride/box/swizzle legality is outside the algebra unless explicitly modeled. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **C (What hand-tuned kernels do instead):**  
  - They design tiles so `CUtensorMap` constraints hold (stride multiples, `boxDim` caps, alignment) and pick a legal swizzle/interleave combo so `cp.async.bulk.tensor` is admile. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

---

### Cliff 2 — **Async Orchestration / `mbarrier` tx-count Cliff** (H100 + beyond)

- **Symptom (Nsight Compute metrics / reasoning):**  
  - “Achieved Occupancy” can be non-trivial (`sm__warps_active.avg.pct_of_peak_sustained_active`), yet kernel is latency-bound because warps are frequently not eligible (pipeline bubbles). ([forums.developer.nvidia.com](https://forums.developeia.com/t/metric-references-and-description/111750))  
- **A (Formalism blind spot):**  
  - Seed models treat copies as mappings; they do not model that `mbarrier` phases complete only when **pending arrivals AND tx-count are zero**. ([llvm.org](https://llvm.org/docs/NVPTXUsage.html))  
- **C (What hand-tuned kernels do instead):**  
  - Use `cp.async.bulk.tensor … mbarrier::complete_tx::bytes` and manage `expect-tx/complete-tx` consistently so producer/consumer phases advance (classic multi-stage pipeling). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  

---

### Cliff 3 — **On-chip Tiering Cliff (Registers vs SMEM vs TMEM)** (Blackwell emphasis)

- **Symptom (Nsight Compute metrics / tables):**  
  - Register-limited occupancy: “Achieved Occupancy” (`sm__warps_active.avg.pct_of_peak_sustained_active`) is low and “Block Limit registers” dominates (Nsight UI terminology). ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/mrences-and-description/111750))  
- **A (Formalism blind spot):**  
  - None of the seed papers’ core formalisms treat **memory tier choice** as part of the semantics (where do fragments/accumulators live?), yet this dictates register pressure and instruction selection. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **B (What hardware requires):**  
  - Blackwell TMEM has strict collective constraints (warpgroup lane partition; warp-collective `taddr` for `tcgen05.ld`) and explicit allocation semaics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  
- **C (What hand-tuned kernels do instead):**  
  - They exploit TMEM + `tcgen05.*` to shift live state off registers (when legal) and structure the kernel around warpgroup collectives. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  

---

## Step 3 — Elephant-in-the-room diagnosis (ranked)

> Ranking is **C (Our Inference)**, justified by **B evidence**.

1. **(B) Latency hiding / asynchrony orchestration**  
   - Because modern fast paths *are defined by async pipelines* (`cp.async.bulk.tensor` + `mbarrier` completion); correctness + throughput hinge on phase/tx-count contracts. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html))  
2. **(C) Descriptor legality cliffs**  
   - Because TMA usage is gated by strict tensor-map legality (alignment/stride/box/swizzle rules). If you miss, the compiler cannot legally emit the instruction, forcing a slow path. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
3. **(A) Memory bandwidth wall (HBM)**  
   - Because peak tensor-core compute has grown dramatically relative to HBM bandwidth (H100 vs Blackwell vs MI300X), so many kernels are bandwidth/IO dominated unless layout+tiling raises reuse. ([nvidia.com](https://www.nvidia.com/en-gb/data-center/h100/))  
4. **(D) Compilation/search cost**  
   - Because the legal+fast subset is narrow and the search space (tile × swizzle × stage count × grain) is combinatorial; Paper 2 even notes worst-case exponential costs for relation ops, and Paper 1 positions future integration of “hardware measurements” (i.e., tuning). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
5. **(E) Dynamic memory management (KV cache / fragmentation) — secondary for *layout* papers**  
   - Critical at system level, but orthogonal to the seed formalisms as presee of the three provides semantics for allocator/fragmentation-aware layout choices. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

---

## Stage-1 Verdict (10 bullets: correct / incomplete / wrong)

1. **Correct (C):** Calling out **descriptor legality cliffs** as real is directionally right; the driver API has hard caps (e.g., `boxDim≤256`, stride multiples). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
2. **Correct (C):** Treang **swizzle as enumerated/validity-checked** (not arbitrary permutation) is correct for TMA: inner-dimension + alignment requirements make invalid encodings possible. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html))  
3. **Correct (C):** The “barrier is more than synchronization” point is correct: **tx-count must reach zero** for phase completion in `mbarrier`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html))  
4.orrect (C):** Highlighting Blackwell **TMEM as a special tier** is correct; PTX specifies a distinct memory region with allocation + lane restrictions. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html))  
5. **Incomplete (C):** Any Hopper/Blackwell audit must explicitly include **completion-mechanism choice** for `cp.async.bulk.tensor` (mbarrier vs bulk_group), not just “alignment.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallelead-execution/index.html))  
6. **Incomplete (C):** Seed Paper 2 contains a *seed-level* semantic cliff: **CuTe implicit layout promotion / holes** breaks naïve relational composition—this matters for memory safety and should be first-class in the gap audit. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
7. **Incomplete (C):** The Blackwell story must explicitly address **compute-vs-bandwidth asymmetry** (HBM bandwidth scaling relative to compute) as the dominant global constraint. ([developer.nvidcom](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  
8. **Incomplete (C):** MI300 portability requires explicit **wave64** reasoning; treating “warp” as universal is a portability bug for layout distribution. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.0/conceptual/pipeline-metrics.html))  
9. **Potentially wrong unless scoped (C):** Any statement like “tensorRank is always in \([1,5]\)” is **API-de* (tiled vs im2col variants differ); rank constraints must be stated per encoding API. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
10. **Wrong as a performance narrative (C):** Framing layouts as the *only* bottleneck is wrong; in practice, **orchestration correctness** and **tiering/occupancy** dominate many failures even when a layout map is “correct.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-ution/index.html))  

--- 

If you want Stage 2, I can turn this into an explicit “typed IR” sketch: **(Layout × Grain × Tier × AsyncContract)** with (i) a decidable legality checker that emits `CUtensorMap`/descriptor fields, and (ii) a minimal counter set to close the loop with Nsight/rocprof.

---
Learn more:
1. [https://arxiv.org/html/2505.23819v3](https://arxiv.org/html/2505.23819v3)
2. [https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html](https://docs.nvidia.com/cudve/12.1.1/parallel-thread-execution/index.html)
3. [https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group\_\_CUDA\_\_TENSOR\_\_MEMORY.html](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
4. [https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html)
5. [https://www.nvidia.com/en-gb/data-center/h100/](https://www.nvidia.com/en-gb/data-center/h100/)
6. [https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.0/conceptual/pipeline-metrics.html](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.0/conceptual/pipeline-metrics.html)
7. [https://mlir.llvm.org/docs/Dialects/NVVMDialect/](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)
8. [https://arxiv.org/html/2511.10374v1](https://arxiv.org/html/2511.10374v1)
9. [https://arxiv.org/pdf/2601.05972v1](https://arxiv.org/pdf/2601.05972v1)
10. [https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)
11. [https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group\_\_CUDA\_\_TENSOR\_\_MEMORY.html](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
12. [https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group\_\_CUDA\_\_TENSOR\_\_MEMORY.html](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
13. [https://docs.nvidia.com/cuda/cuda-driver-api/group\_\_CUDA\_\_TENSOR\_\_MEMORY.html](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
14. [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
15. [https://docs.nvidia.com/nsight-compute/2019.5.1/NsightComputeCli/index.html](https://docs.nvidia.com/nsight-compute/2019.5.1/NsightComputeCli/index.html)
16. [https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html)
17. [https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html)
18. [https://forums.developer.nvidia.com/t/metric-references-and-description/111750](https://forums.developer.nvidia.com/t/metric-references-and-description/111750)
19. [https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
20. [https://docs.nvidia.com/nsight-compute/2021.3/ProfilingGuide/index.html](https://docs.nvidia.com/nsight-compute/2021.3/ProfilingGuide/index.html)
21. [https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
22. [https://www.amd.com/en/products/accelerators/instinct/mi300/platform.html](https://www.amd.com/en/products/accelerators/instinct/mi300/platform.html)
23. [https://llvm.org/docs/NVPTXUsage.html](https://llvm.org/docs/NVPTXUsage.html)
