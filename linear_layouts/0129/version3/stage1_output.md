## Context Verdict (6–10 bullets, falsification-first)

- **1) “Layout algebra ≠ hardware legality.”**
  - **[A] Seed**: Layouts are modeled as total (or at least well-defined) mappings in a math domain (linear maps over $$\mathbb{F}_2$$, integer relations, categorical morphisms). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
  - **[B] Hardware**: Hopper/Blackwell *refuse* certain layouts at the descriptor level (e.g., CuTensorMap/TMA alignment/stride/boxDim constraints). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - **[C] Inference**: Any “layout abstraction” used for codegen must carry a *feasibility predicate* (descriptor legality + instruction operand legality), not just an index map.

- **2) The biggest modern gap is temporal orchestration, not index math.**
  - **[A] Seed**: Linear Layouts proves closure for *shape ops* (reshape/transpose/join/split/broadcast) but that’s “spatial” closure. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
  - **[B] Hardware**: TMA copies are *asynchronous* and completion is wired to `mbarrier::complete_tx::bytes`; the code must pipeline these. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  
  - **[C] Inference**: Layout formalisms that don’t model “time” will systematically miss the main performance cliff on H100/B200-class kernels.

- **3) Descriptor legality cliffs are real and brutal on sm90+.**
  - **[A] Seed**: The seeds treat “layout conversion” as a semantics-preserving transformation problem. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
  - **[B] Hardware**: CuTensorMap demands strict alignment (e.g., 16B/32B), stride multiples (16/32), boxDim bounds, swizzle constraints, etc. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - **[C] Inference**: A compiler can have a “best” layout mathematically that is *unrepresentable* as a legal descriptor—so the optimization must be legality-aware.

- **4) Blackwell shifts the bottleneck toward memory efficiency + on-chip tiering.**
  - **[B] Hardware**: Blackwell Ultra advertises **15 PFLOPS** NVFP4 dense compute but **8 TB/s** HBM bandwidth (2.4× H100), and adds **256 KB TMEM per SM**. ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  
  - **[B] Hardware**: TMEM has warpgroup-lane access restrictions and special `tcgen05.*` load/store shapes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html))  
  - **[C] Inference**: “Layouts” now include *where intermediates live* (SMEM vs TMEM), not just a mapping into registers/shared.

- **5) AMD MI300 has different first-order constraints (LDS + MFMA distribution).**
  - **[B] Hardware**: LDS bank mapping + phase rules differ by instruction granularity (`ds_read_b128` / `ds_write_b128`). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))  
  - **[B] Hardware**: MFMA is wavefront-level; operands are distributed across lanes; AGPR/VGPR storage is finite (up to 512 KiB combined per CU on supported arch). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html?utm_source=openai))  
  - **[C] Inference**: A “portable” layout abstraction must parameterize bank rules + operand distribution, not assume NVIDIA-like behaviors.

- **6) The seeds’ explicit limitations already admit the core mismatch.**
  - **[A] Seed**: Linear Layouts: “primary limitation… power-of-two shapes”; flipping/slicing not expressible without affine extension. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
  - **[A] Seed**: ISL-relations: linear-layout binary spaces require power-of-two dims; ISL ops can be worst-case exponential. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
  - **[A] Seed**: Categorical CuTe: focuses on “tractable layouts”; only tractable layouts arise from their category construction. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
  - **[C] Inference**: None of the seeds claim to solve descriptor legality or temporal scheduling; treating them as “enough for peak kernels” is a category error.

---

## Step 0 — Access Check (all 3 seed URLs opened & readable)

| Seed | URL | Format | Access result |
|---|---|---:|---|
| 1 | `https://arxiv.org/html/2505.23819v3` | HTML | Readable (full TOC + body). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |
| 2 | `https://arxiv.org/html/2511.10374v1` | HTML | Readable (full TOC + body). ([arxiv.org](https://arxiv.org/html/2511.10374v1)) |
| 3 | `https://arxiv.org/pdf/2601.05972v1` | PDF (174 pp) | Readable (text extracted + searchable). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) |

> **[C] Note on “read them in entirety”**: I programmatically traversed each document end-to-end (TOC + full text extraction) and then deep-read the sections that define axioms/operations/limits and the “corner-case” parts (e.g., closure proofs, tractable/non-degenerate conditions, complexity notes). Where claims depend on exact wording, I cite the exact section lines.

---

## Step 1 — Seed Truth (per paper)

### Seed 1: *Linear Layouts* (Zhou et al., 2026)

#### Axioms / Core formalism
- **[A] Axiom A1 (representation)**: A *Linear Layout* is a **linear map between labeled vector spaces over $$\mathbb{F}_2$$**. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **[A] Axiom A2 (algebra)**: Core operators include **composition** and **product** (block-diagonal “direct sum”), plus **left division** and **right inverse** (for surjective layouts). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **[C] Compiler artifact mapping**: This is meant to be the *layout IR* inside Triton’s GPU backend (tt/ttg lowering); i.e., replace ad-hoc per-layout methods with matrix reasoning.  
  - **Hardware metric to track**: emitted vector width (e.g., 128-bit vs 32-bit loads/stores), number of `convert_layout` ops, and shared-memory instruction count (Nsight Compute: shared transactions + bank-conflict counters).

#### Closure properties / completeness
- **[A] Completeness C1 (distributed layouts)**: “Every distributed layout is a linear layout”; distributed layouts are characterized as surjective maps whose matrices are permutation-like with optional zero columns. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **[A] Completeness C2 (memory layouts)**: “Every memory layout is a linear layout”; memory layouts are invertible linear layouts with constraints on column bit-popcount (0 or 1 bits). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **[A] Closure C3 (shape ops)**: The family of distributed layouts is **forward/backward closed** under Triton shape ops (transpose/reshape/join/split/expand_dims/broadcast), enabling “no-op” layout propagation through those ops. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

#### Explicit limitations (must point to exact section)
- **[A] Limitation L1 (power-of-two)**: Quote (≤25 words): “*The primary limitation of linear layouts is the restriction to power-of-two shapes…*” (Section 8, Conclusions). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **[A] Limitation L2 (non-linearity: flip/slice)**: Quote (≤25 words): “*Operations such as flipping and slicing are not expressible as linear layouts…*” and they propose an “affine layouts” extension. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **[A] Limitation L3 (swizzle algorithm scope)**: Appendix notes the bank-conflict/swizzle algorithm focuses on vectorization “for simplicity” (explicitly excluding broader intrinsic targeting in that section). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

---

### Seed 2: *Modeling Layout Abstractions Using Integer Set Relations* (Bhaskaracharya et al., 2025)

#### Axioms / Core formalism
- **[A] Axiom A1 (unifying representation)**: Represent CuTe layouts, CuTe swizzles, and Triton linear layouts as **integer set relations** in ISL (polyhedral-style relations between coordinate/index spaces). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **[A] Axiom A2 (operation set)**: ISL supports operations on relations (composition, inverse, domain/range) and on sets (union/intersection/difference). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **[A] Axiom A3 (quasi-affine support)**: ISL’s quasi-affine extension supports floor/ceil divisions, enabling division-heavy mappings. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

#### Closure properties
- **[A] Closure C1 (layout ops)**: CuTe’s key ops—composition/inverse/complement—are first-class in their model, and complement is described as “filling gaps” in a target index interval. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **[C] Compiler artifact mapping**: This naturally maps to an MLIR/polyhedral analysis pass that emits/rewrites `memref`/`vector` layout maps, or to CuTe layout reasoning in CUTLASS generators.  
  - **Hardware metric to track**: transaction count (global coalescing), bank-conflict wavefronts, or “descriptor feasibility” as constraints within the relation domain.

#### Explicit limitations (exact sections)
- **[A] Limitation L1 (power-of-two for linear-layout binary spaces)**: Quote (≤25 words): “*Note that the dimension sizes in the co-ordinate and index spaces must be powers of two.*” (Section 2.3.1). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **[A] Limitation L2 (compile-time worst-case)**: Quote (≤25 words): “*relation composition and lexicographic minimum exhibit worst-case exponential complexity*” (Section 6, Complexity). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **[A] Limitation L3 (not a runtime performance paper)**: Quote (≤25 words): “*focuses on establishing mathematical foundations rather than runtime performance optimization*” (Conclusions). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

---

### Seed 3: *Categorical Foundations for CuTe Layouts* (Carlisle et al., 2026)

#### Axioms / Core formalism
- **[A] Axiom A1 (categorical encoding)**: Define categories **Tuple** and **Nest** whose morphisms encode (flat/nested) layouts; define operations on morphisms and prove compatibility with layout operations. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **[A] Axiom A2 (tractability as a semantic restriction)**: They focus on a “naturally occurring class of tractable layouts.” ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

#### Closure properties / algebraic guarantees
- **[A] Closure C1 (layout algebra mirrored)**: The framework aims to mirror CuTe’s algebra (composition, logical product, logical division) via morphism operations, with proofs of compatibility. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **[C] Compiler artifact mapping**: This maps most directly to CuTe/CUTLASS metaprogramming (layout composition/product/division) and to compile-time verification/normalization passes.  
  - **Hardware metric to track**: ability to produce legal operand layouts for tensor-core instructions and minimize shared/LDS bank conflicts in staging.

#### Explicit limitations (exact sections / statements)
- **[A] Limitation L1 (scope: tractable-only)**: Quote (≤25 words): the paper presents its framework “*focusing on… tractable layouts*.” ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **[A] Limitation L2 (exact characterization)**: Quote (≤25 words): “*there exists a nested tuple morphism f encoding L iff L is tractable*.” ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
- **[A] Limitation L3 (pathologies excluded)**: They explicitly exclude degeneracies (shape entry $$s_i=1$$ with nonzero stride) to avoid non-unique representations. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

---

## Seed-vs-Hardware Matrix (required)

> **Claim-type key** (enforces S1):
> - **Seed_Math_Concept / Explicit_Seed_Limit = [A]**
> - **Hardware_Feature / Hardware_Legality_Constraint = [B]**
> - **Performance_Cliff_Mode / Minimal_Relaxation/Extension = [C]**

| Seed_Math_Concept | Explicit_Seed_Limit | Hardware_Feature | Hardware_Legality_Constraint | Performance_Cliff_Mode | Minimal_Relaxation/Extension |
|---|---|---|---|---|---|
| Linear layouts as $$\mathbb{F}_2$$-linear maps on **bit indices** ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | “Primary limitation… **power-of-two shapes**” ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Hopper TMA / CuTensorMap descriptors | `globalAddress` **16B aligned** (32B if interleave=32B or packed types); `globalStrides` multiples of **16/32**; rank/boxDim/stride bounds ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | Illegal descriptor ⇒ no `cp.async.bulk.tensor` path; fallback to scalar/vector `ld.global` + more staging | Add **descriptor-feasibility constraints** to the layout domain; allow masked “super-layout” as seed suggests ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |
| Distributed layout ≈ permutation matrix (+ zero columns) ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Flips/slices not expressible as linear layouts ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Blackwell TMEM (Tensor Memory) | TMEM access is partitioned: each warp in a warpgroup accesses only specific lanes; `tcgen05.st` requires **all threads have same `taddr`** else undefined ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html)) | Operand staging into TMEM becomes serialization / undefined behavior if layout doesn’t respect warpgroup lane ownership | Extend layout type system with **warpgroup-lane ownership** + collective-address invariants |
| Memory layouts as invertible linear layouts; swizzling is linear ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | Appendix swizzle algo focuses on vectorization “for simplicity” ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | TMA swizzle modes | Swizzle is one of `CUtensorMapSwizzle`; shared-memory box inner dimension must be **≤ swizzle span**, swizzle maps 16B chunks to bank subgroups ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai)) | Bank-conflict cliff: wrong swizzle/padding ⇒ extra “wavefronts” in SMEM/LDS ops | Make swizzle a **first-class parameter** with legality + cost model (bank wavefront count) |
| ISL integer set relations unify CuTe + Triton layouts ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | ISL ops can be worst-case exponential; focus is not runtime tuning ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | CuTensorMap legality is a polyhedral constraint set | `boxDim[i]≤256`; `boxDim[0]*elemSize` multiple of 16 (interleave none); `elementStrides[i]∈[1,8]`; inner-dim ≤ swizzle size ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | Search explores “great” layouts that are descriptor-illegal; time wasted + fallback kernels | Add **legality constraints into the relation**; prune using ISL before codegen |
| ISL quasi-affine (floor/ceil) extensions ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | Linear-layout binary spaces still require pow2 dims ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | AMD LDS bank mapping depends on instruction width | LDS has 32/64 banks of 4B; bank conflicts evaluated in **phases** for `ds_*_b128` ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai)) | “Looks affine” but bank conflicts explode under instruction-specific phase rules | Extend cost model with **instruction-granularity bank mapping** (per op width) |
| Categorical encoding via Tuple/Nest morphisms ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | Only **tractable** layouts arise (iff tractable) ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | AMD MFMA operand distribution + register tiers | MFMA is wavefront-level; operands distributed across lanes; MFMA uses VGPR/AGPR (finite), up to 512 KiB combined per CU on supported arch ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html?utm_source=openai)) | Layout conversion requires extra VGPR/LDS moves; MFMA pipeline underfed | Add an “MFMA-feasible” subcategory that encodes **lane-distribution constraints** explicitly |
| Non-degeneracy constraints (exclude $$s_i=1$$ with stride≠0) ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | Pathologies must be excluded to keep 1–1 morphism ↔ layout mapping ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | TMA elementStrides/boxDim semantics | TMA ignores stride for dim0 when `interleave=NONE`; when `elementStrides[i]≠1`, TMA loads `ceil(boxDim[i]/elementStrides[i])` elements; to load N, set `boxDim[i]=N*elementStrides[i]` ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | “Silent overfetch” cliff: descriptor legal but moves more data than intended ⇒ bandwidth waste | Seed-level “layout” must include **iterator semantics** (boxDim × elementStride) and cost it |

---

## Step 2 — Hardware Truth (H100 + Blackwell + MI300): Top 5 hardware-enforced constraints

> Each item is **[B] Hardware Manual/Blog claim** (S1).

### 1) TMA / CuTensorMap descriptor legality (H100/Hopper+)
- **[B] What it is**: Hardware-validated tensor map descriptors used by `cp.async.bulk.tensor*` (TMA engine).
- **[B] Why it dominates layouts**: If the layout you want can’t be encoded legally, the compiler can’t emit the TMA path—period. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))

**Legality constraints (S2-required enumeration, CUDA docs):**
- **Alignment**
  - `globalAddress` must be **16-byte aligned**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - If `interleave = CU_TENSOR_MAP_INTERLEAVE_32B`, then `globalAddress` must be **32-byte aligned**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - Certain packed types (`*_ALIGN16B`) also require **32-byte alignment**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Strides**
  - `globalStrides[i]` (bytes) must be a multiple of **16** (and < $$2^{40}$$); with `interleave_32B` (or certain packed types) strides must be multiple of **32**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Tensor rank / dims**
  - `tensorRank` is bounded (≤5) and interleave may require rank ≥3. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - `globalDim[i]` must be nonzero and ≤ $$2^{32}$$; packed types impose multiples on `globalDim[0]` (e.g., multiple of 128 for some). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **`boxDim` rules**
  - `boxDim[i]` must be nonzero and ≤ **256**; plus alignment constraints like `{boxDim[0] * elementSize}` multiple of **16B** when `interleave=NONE`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - Packed types can require `boxDim[0]=128`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **`elementStrides` rules**
  - Each `elementStrides[i]` must be nonzero and ≤ **8**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - When `interleave=NONE`, TMA ignores `elementStrides[0]` (“TMA doesn’t support the stride for dimension zero”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - If some `elementStrides[i]≠1`, TMA loads `ceil(boxDim[i]/elementStrides[i])` elements; to load N, set `boxDim[i]=N*elementStrides[i]`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **Swizzle / interleave / type interactions**
  - Swizzle patterns (type `CUtensorMapSwizzle`) define how 16B chunks map to shared-memory bank groups, and the shared-memory box inner dimension must be **≤ swizzle span**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

---

### 2) Async pipeline semantics: `cp.async.bulk.tensor` + `mbarrier::complete_tx::bytes` (H100/Hopper+)
- **[B] What it is**: PTX-level async tensor copy instruction family where completion is tracked via an `mbarrier` object. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  
- **[B] Hard constraint**: Your layout pipeline must supply:
  - a legal tensor map (`CUtensorMap`),  
  - correct tensor coordinates,  
  - a valid barrier object,  
  - correct staging order (producer/consumer), otherwise you serialize or deadlock at the barrier boundary (semantically). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  

---

### 3) Blackwell on-chip tiering: SMEM vs TMEM + TMEM access restrictions (Blackwell)
- **[B] What it is**: Blackwell introduces **Tensor Memory (TMEM)** as a warp-synchronous on-chip storage (256 KB per SM in Blackwell Ultra). ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  
- **[B] Hard constraint**: TMEM is partitioned by warp within a warpgroup; not all warps can access all lanes (e.g., warp 0 lanes 0–31, warp 1 lanes 32–63, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html))  
- **[B] Hard constraint**: TMEM load/store has fixed shapes (`tcgen05.ld/st` shapes like `.16x64b`, `.16x128b`, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html))  
- **[B] Correctness constraint**: For `tcgen05.st`, “all the threads in the warp must specify the same value of `taddr` … otherwise behavior is undefined.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html))  

---

### 4) Blackwell compute-vs-bandwidth asymmetry (Blackwell)
- **[B] Claim**: Blackwell Ultra advertises large compute gains (dense NVFP4 PFLOPS scale) while HBM bandwidth is **8 TB/s** per GPU (2.4× H100 per the blog’s comparison). ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  
- **[B] Consequence**: More kernels become **bandwidth / data-movement bound** unless they exploit locality (TMEM/SMEM reuse) and minimize conversion traffic. ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  

---

### 5) MI300: LDS bank-conflict rules + MFMA operand distribution / register tiers
- **[B] LDS bank constraints**: LDS is organized into **32 or 64 banks** (4B width), with bank conflicts determined by address mapping; conflict check depends on the instruction width and is evaluated in phases for `ds_*_b128`. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))  
- **[B] MFMA constraints**: MFMA instructions are **wavefront-level**; operands are distributed across threads and stored in VGPRs (and on some arch, AGPRs). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html?utm_source=openai))  
- **[B] Register-tier constraint**: Supported architectures provide up to **512 KiB combined register storage per CU** (VGPR + AGPR) and MFMA throughput depends on feeding these units. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html?utm_source=openai))  

---

## Step 3 — Elephant-in-the-room diagnosis (ranked; with citations)

> **All items are [C] Inference**, justified by **[A] seed limits** + **[B] hardware constraints**.

1) **(B) Latency hiding / asynchrony orchestration**
   - **Why #1**: Hopper/Blackwell’s fastest data movement and tensorcore paths are explicitly asynchronous (`cp.async.bulk.tensor…mbarrier::complete_tx`). If you don’t pipeline and stage correctly, you cannot approach peak, regardless of how “good” the static layout map is. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  

2) **(C) Descriptor legality cliffs**
   - **Why #2**: CuTensorMap legality is a *binary gate*: illegal ⇒ path unavailable. The constraints are numerous (alignment, stride multiples, boxDim, swizzle interaction), so “layout search” frequently hits infeasible regions unless constrained. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

3) **(A) Memory bandwidth wall (increasingly so on Blackwell)**
   - **Why #3**: Blackwell scales compute heavily and adds TMEM specifically to improve reuse; the blog’s numbers show bandwidth doesn’t scale at the same rate as compute, so kernels that spill or over-move data (layout conversions, transposes, attention-like ops) hit bandwidth sooner. ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  

4) **(D) Compilation/search cost**
   - **Why #4**: The ISL paper explicitly notes worst-case exponential costs for certain relation ops, and the framework is foundational rather than runtime-optimized; naive integration into autotuners without pruning can become a bottleneck. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

5) **(E) Dynamic memory management (KV cache / fragmentation)**
   - **Why #5**: The Blackwell blog explicitly frames memory capacity as critical for avoiding KV-cache offloading and enabling long context / high-concurrency inference, indicating system-level memory pressure can dominate end-to-end even if kernels are optimal. ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  

6) **(Other) Cross-vendor semantic mismatch (NVIDIA TMA/TMEM vs AMD LDS/MFMA)**
   - **Why**: Different bank rules + operand distributions mean “one layout algebra” is insufficient unless it parameterizes hardware cost/legality; AMD’s LDS+MFMA constraints are materially different from NVIDIA’s TMA+TMEM constraints. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))  

---

## 3 Performance Cliffs (SASS-level intuition) (required)

### Cliff 1 — **TMA descriptor illegality → slow-path global→shared staging**
- **Symptom (Nsight Compute style)** *(C inference)*:
  - No `cp.async.bulk.tensor*` in SASS; instead many `LDG`/`ST` sequences + more instructions.
  - High `smsp__warp_issue_stalled_long_scoreboard` (waiting on memory) and low tensor/pipe utilization.
- **Formalism blind spot** *(C)*:
  - Seeds treat a layout as an index mapping; legality constraints (alignment/boxDim/swizzle/type coupling) are not part of the mapping. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **What hand-tuned kernels do** *(B facts + C inference)*:
  - Reshape tiles to satisfy CuTensorMap: enforce 16B/32B base alignment and 16/32 stride multiples; pick boxDims ≤256 and obey `boxDim[i]=N*elementStrides[i]` when strided; choose a legal swizzle and keep inner dimension ≤ swizzle span. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
  - Then pipeline `cp.async.bulk.tensor…mbarrier::complete_tx::bytes` stages. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  

---

### Cliff 2 — **Baict explosion in SMEM/LDS → “wavefronts” and throughput collapse**
- **Symptom (profiler terms)** *(C)*:
  - NVIDIA: elevated shared-memory bank conflict counters (e.g., `l1tex__data_bank_conflicts*`) and reduced shared throughput.
  - AMD: LDS bank conflicts per phase; `ds_read_b128` shows conflict unless access follows the documented phase groups.
- **Formalism blind spot** *(C)*:
  - “Swizzle” is modeled abstractly, but real bank behavior depends on (i) instruction width and (ii) bank grouping  explicitly documents phase groupings). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))  
- **What hand-tuned kernels do** *(B facts + C inference)*:
  - AMD: choose layouts that are conflict-free under `ds_{read,write}_b128` phase rules, or apply XOR preshuffle instead of padding when possible. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))  
  - NVIDIA TMA: pick swizzle modes that map 16B chunks to bank subgroups and respect the “inner dimension ≤ swizzle span” rule. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  

---

### Cliff 3 — **Blackwell TMEM misuse / non-collective addressing → undefined behavior or forced staging**
- **Symptom (profiler terms)** *(C)*:
  - Tensor-core pipe underutilized, unexpec around data movement; extra SMEM traffic where TMEM was expected.
- **Formalism blind spot** *(C)*:
  - Seeds don’t encode *warpgroup-lane ownership* or the “collective address” requirement for TMEM store.
- **What hand-tuned kernels do instead** *(B facts + C inference)*:
  - Respect TMEM’s warpgroup partition: each warp accesses only its lanes; use the supported `tcgen05.ld/st` shapes; ensure warp-wide uniform `taddr` for `tcgen05.st`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.el-thread-execution/index.html))  
  - Use TMEM to keep intermediate results close (256 KB per SM in Blackwell Ultra) and reduce off-chip traffic. ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  

---

## Stage-1 Verdict (10 bullets: correct vs incomplete vs wrong)

1) **Correct**: Layout abstractions *must* model bit-level permutations/swizzles to be useful on modern tensor-core kernels. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
2) **Correct**: “Descriptor legality cliffs” are a first-order optimization constraint on Hopper/Blackwell (not just a perf detail). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
3) **Correct**: Temporal orchestration (async copies + barriers) dominates whether a good layout becomes a good kernel. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  
4) complete**: Linear-layout closure under shape ops is valuable, but it does not address *instruction feasibility* (TMA/TMEM/MFMA operand rules). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
5) **Incomplete**: ISL relations unify reasoning across CuTe/Triton, but the seed explicitly does not claim runtime optimization; you still need a hardware cost/legality model. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
6) **Incomplete**: Categorical CuTe foundations are powerful for normalization/verification, but the scope is tractable layouts; many “real” hardware operand layouts may sit outside without extensions. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
7) **Correct**: Bank conflicts remain a dominant layout-driven cliff on both NVIDIA (swizzle patterns) and AMD (LDS phase rules). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai))  
8) **Correct**: Blackwell’s on-chip tiering (TMEM) changes the definition of â: it’s now also *placement* and *collective access discipline*. ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  
9) **Wrong / overconfident**: Treating “layout math” as sufficient to reach peak without modeling barriers, staging depth, and instruction granularity is not supported by the hardware docs. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))*Correct but needs scoping**: “Memory wall” is real, but on Blackwell the more precise story is “compute rises faster than bandwidth → data movement and reuse become the limiter,” especially for layout conversions/attention-like ops. ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
ver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
3. [1. Introduction — PTX ISA 8.7 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai)
4. [Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era | NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
5. [1. Introduction — PTX 8.7 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html)
6. [Understanding AMD GPU LDS and Bank Conflicts — Composable Kernel 1.2.0 Documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai)
7. [Hardware implementation — HIP 7.2.0 Documentation](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html?utm_source=openai)
8. [Modeling Layouttractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
9. [Categorical Foundations for CuTe Layouts](https://arxiv.org/pdf/2601.05972v1)
10. [4.11. Asynchronous Data Copies — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html?utm_source=openai)
