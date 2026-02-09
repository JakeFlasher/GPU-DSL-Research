## 1. Core Problem Verification

The “validated gaps” in your Stage‑2 directions map cleanly onto *real* NVIDIA correctness cliffs. What’s *not* real is any implication that NVIDIA/CUDA “doesn’t handle it at all”—the vendor handles many pieces via **runtime encoding**, **typed wrappers**, and **ISA-defined protocols**. The open gap is: **static, proof-producing compilation contracts** (certificates / cores), and **compositional semantics that justify reordering**.

### Table — Reality check against NVIDIA docs + current toolchains

| Problem Claim | Hardware Reality (What docs say) | Existing Tooling (What Triton/CUDA/MLIR do) | Verdict |
|---|---|---|---|
| **TMA / `CUtensorMap` legality is “mixed constraints” and causes compilation cliffs** | `cuTensorMapEncode*` imposes hard encode-time constraints (alignment, bounds, finite enums). It **returns error codes** on invalid values; tensor maps are opaque objects. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)) | **MLIR NVGPU**: `nvgpu.tma.create.descriptor` *calls* `cuTensorMapEncodeTiled` (runtime), and has `nvgpu.tma.fence.descriptor` for safe use after host-side modification. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)) **CUTLASS**: relies on `cuTensorMapEncodeTiled` in host adapter (engineering pain: symbol/indirection issues in PyTorch integration). ([github.com](https://github.com/NVIDIA/cutlass/issues/1566?utm_source=openai)) **Triton**: exposes `tl.make_tensor_descriptor` with constraints like base 16B alignment + stride restrictions; backed by TMA on supported GPUs. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai)) | **Partial Gap.** *Vendor solves “is this encodable?” at runtime* (return `CUDA_ERROR_INVALID_VALUE`). What’s **not solved** is compile-time *proof/certificate* and actionable *unsat core / diagnosis* for autotuners. |
| **`cp.async.bulk.tensor` legality is brittle (exact byte quanta, arch gating, UB cliffs)** | PTX ISA specifies **exact restrictions** (e.g., Box-Size[0] must be exactly 64B/96B for some types; coordinate multiples; alignment; supported swizzles; arch-specific restrictions like `sm_120a` changes). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source=openai)) | NVIDIA provides **CCCL/libcudacxx PTX wrappers** documenting per-instruction ISA/SM version gating and variants; still doesn’t enforce all shape/stride legality beyond what the programmer/compiler supplies. ([nvidia.github.io](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html?utm_source=openai)) MLIR/NVVM can represent these ops and barriers (engineering). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)) | **Valid Gap.** The op exists and is documented, but the “math-first legality synthesizer + certificate” isn’t a standard compiler component today. |
| **Warpgroup uniformity + fence protocol for `wgmma.*` is a UB cliff** | PTX explicitly says `.aligned` requires **warpgroup-uniform execution**, otherwise **undefined behavior**; `wgmma.fence` must appear at specific points (and is `.sync.aligned` itself). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | **MLIR NVGPU** adds `nvgpu.wargroup.mma` to wrap `wgmma.fence/commit/wait` sequences (engineering abstraction), but this does **not** by itself prove uniformity of the controlling predicates. ([reviews.llvm.org](https://reviews.llvm.org/D158434?utm_source=openai)) **LLVM** has a general **UniformityAnalysis** line (not PTX-warpgroup-specific proof obligations, but relevant). ([llvm.org](https://llvm.org/docs/ConvergenceAndUniformity.html?utm_source=openai)) | **Valid Gap.** Fence insertion is *engineering-solvable*, but **uniformity obligations** are semantic / hyperproperty‑ish and not “solved by vendor.” |
| **`mbarrier` object validity + state-token provenance is UB** | PTX: using any `mbarrier` op except `init` on an uninitialized object is UB; and certain waits require the `state` operand to originate from specific arrive variants (“otherwise behavior is undefined”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | **MLIR NVGPU** has first-class `!nvgpu.mbarrier.*` types + token types, and ops like `mbarrier.create`, `mbarrier.arrive.*`, `mbarrier.test.wait`. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)) **LLVM NVPTX** intrinsics document provenance constraints (state must come from `arrive` on same barrier object) and UB conditions if address spaces are wrong. ([llvm.org](https://llvm.org/docs/NVPTXUsage.html?utm_source=openai)) | **Partial Gap.** IR can encode tokens (good engineering), but compilers generally do **not** produce *proof artifacts* that the token protocol is correct across transformations. |
| **TMEM (`tcgen05.*`) is dynamic / finite / collective; uniform `taddr` constraints matter** | PTX: TMEM allocation is dynamic; allocation unit is 32 columns; allocated columns must be power-of-two; `tcgen05.ld` requires **warp‑uniform `taddr`** else UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | **LLVM NVPTX** has `llvm.nvvm.tcgen05.alloc.*` intrinsics and documents `ncols` must be power-of-two; represents TMEM as a special address space. ([llvm.org](https://llvm.org/docs/NVPTXUsage.html?utm_source=openai)) | **Open & Hard.** Vendor provides instructions; LLVM provides intrinsics. Static **capability/lifetime** + **warp-uniform operand proof** is not standard tooling. |

**Bottom line:** None of the five cliffs are imaginary. CUDA/PTX define them crisply, but today’s mainstream toolchains largely treat them as “well-typed if you used the right wrapper / passed correct parameters,” not “proved correct with a reusable certificate.”

---

## 2. Competitor Analysis (The “Heuristic” Baseline)

Below I’m explicitly separating **Engineering Solutions** (IR support, wrappers, runtime calls) from **Formal Solutions** (decidable logics + proof artifacts).

### Side-by-side: Triton vs MLIR‑NVGPU/NVVM vs CUTLASS vs TVM

| System | What it supports (relevant to your gaps) | How it enforces correctness | What’s missing vs your “math-first” directions |
|---|---|---|---|
| **Triton** | *TMA descriptors*: `tl.make_tensor_descriptor` with documented alignment/stride constraints; produces descriptor-backed loads/stores on NVIDIA GPUs with TMA support. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  *NVIDIA-specific IR*: Triton NVIDIA GPU dialect includes async TMA copy ops and mbarrier arrive. ([triton-lang.org](https://triton-lang.org/main/dialects/TritonNvidiaGPUOps.html?utm_source=openai)) | **Engineering**: front-end constraints + backend lowering. Typically relies on runtime + codegen invariants rather than emitting *proof artifacts*. | No *proof-producing legality solver* for full `CUtensorMap` / `cp.async.bulk.tensor` legality; no unsat-core explanations; no pomset‑level semantic justification for reordering/scheduling. |
| **MLIR (NVGPU + NVVM)** | *Tokens + async IR*: has device async copy tokens, mbarrier ops/tokens, TMA create descriptor, descriptor fence. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  *Warpgroup MMA wrapper*: `nvgpu.wargroup.mma` abstracts `wgmma.fence/commit/wait`. ([reviews.llvm.org](https://reviews.llvm.org/D158434?utm_source=openai)) | **Engineering**: (1) explicit ops/types, (2) dialect verifiers, (3) lowers TMA descriptor creation to **runtime driver call** `cuTensorMapEncodeTiled`. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)) | No algebraic encodability calculus; no proof-producing legality synthesis; no formal weak-memory/pomset correctness criterion for reordering token pipelines; uniformity obligations aren’t discharged by a hyperproperty backend. |
| **CUTLASS** | Strong engineered kernels for Hopper/Blackwell features; uses driver API tensor-map encoding. Evidence: feature request/issues around integrating `cuTensorMapEncodeTiled` in host adapter. ([github.com](https://github.com/NVIDIA/cutlass/issues/1566?utm_source=openai)) | **Engineering**: C++ templates + handwritten invariants; runtime encoding through driver API. | No formal correctness certificates; “legality” is mostly “we wrote it carefully + it works + tests.” Not a solver. |
| **TVM** | Has explicit TIR intrinsics for `cp.async`, `commit_group`, `wait_group`, and barrier init; also `ptx_cp_async_bulk`. ([tvm.apache.org](https://tvm.apache.org/docs/reference/api/doxygen/namespacetvm_1_1tir_1_1builtin.html?utm_source=openai)) | **Engineering**: intrinsic-level lowering; schedule primitives; mostly relies on user schedule + passes. | No clear first-class support (in core docs) for **TMA tensor-map descriptors / `cp.async.bulk.tensor`** legality; no proof objects or unsat cores; no hyperproperty uniformity checking. (At least not shown in current docs.) ([tvm.apache.org](https://tvm.apache.org/docs/reference/api/doxygen/namespacetvm_1_1tir_1_1builtin.html?utm_source=openai)) |

### Important “vendor solved it?” nuance (CUDA 12.x/13.x reality)

NVIDIA is actively moving these features into **typed wrappers** (CCCL/libcudacxx) and keeping arch/ISA gating up to date (e.g., the documented changelog notes changes in which SM variants enable multicast wrappers). ([nvidia.github.io](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html?utm_source=openai))

That **does not equal** a *formal* solution:
- wrappers don’t emit reusable certificates,
- they don’t produce unsat cores,
- they don’t provide a proof notion for schedule transformations.

---

## 3. Direction Audits

I’m going to try to “kill” each direction by finding prior art that already does the core idea. Where I can’t, I’ll give the closest neighbors (SoTA anchors) and what you must cite.

### Direction 1 — *Encodable Swizzle Groupoids + Dyadic-Capability Layouts*

#### Closest existing paper (SoTA neighbor)
**Axe: A Simple Unified Layout Abstraction for Machine Learning Compilers**  
- Venue: **arXiv**  
- Year: **2026**  
- URL: `https://arxiv.org/abs/2601.19092` ([arxiv.org](https://arxiv.org/abs/2601.19092?utm_source=openai))  
**Why it’s close:** It is explicitly about *unifying* layout abstractions across axes/hierarchies in ML compilers.

#### The Delta (They did X, you do Y)
- **Axe**: a *unified layout abstraction* (expressive coordination of placement/layout).  
- **Your Direction 1**: a *finite encodability-by-construction semantics*:
  - swizzles restricted to *hardware encodable* finite modes (matches driver enums),  
  - TMEM modeled as a **dyadic resource algebra / capability protocol**, and  
  - composition/equivalence via groupoid + PCM normalization.

This “capability + finite encodability” combination is **not** something I see in Axe’s abstract framing (Axe is about unification and compilation structure, not hardware legality certificates). ([arxiv.org](https://arxiv.org/abs/2601.19092?utm_source=openai))

#### Novelty killers / threats (what could invalidate novelty)
1. **NVIDIA already defines swizzle modes as finite enums**, with constraints tying swizzle/interleave to dtype and inner dimension bounds. That undercuts any claim that “encodable swizzles are novel”—only the *algebraic formulation* might be novel. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
2. **Engineering ecosystems are already building “swizzle constructors” and TMEM wrappers** (e.g., libraries expose “make_swizzle(mode=32B/64B/128B)”). This attacks novelty if you overclaim “no one models swizzles as objects.” (But: that’s engineering, not formal semantics.) ([docs.modular.com](https://docs.modular.com/mojo/kernels/layout/swizzle/make_swizzle/?utm_source=openai))  
3. **PTX already specifies TMEM as dynamic with strict rules**; merely restating those rules is not a publishable contribution. Your contribution has to be the *algebra + decidable equivalence + capability typing* that makes illegal states unrepresentable. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  

#### Novelty Risk Score (1–10)
**7/10** (fairly novel framing, but easy to be dismissed as “types + enums + buddy allocator,” unless you deliver a crisp equivalence theory + certificates).

#### Killer citation (must cite)
**Axe: A Simple Unified Layout Abstraction for Machine Learning Compilers** (arXiv 2026).  
URL: `https://arxiv.org/abs/2601.19092` ([arxiv.org](https://arxiv.org/abs/2601.19092?utm_source=openai))  

*(Also, in the actual paper you’ll inevitably cite NVIDIA’s tensor map + swizzle enum spec, but those are docs, not “killer papers.”)* ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))

---

### Direction 2 — *Timed Pomsets + Max-Plus Throughput as a Certified Pipeline Scheduler*

This direction is the most likely to get “novelty‑sniped,” because **software pipelining / modulo scheduling** already has a long history of computing throughput bounds and initiation intervals (II)—often using **cycle constraints** that are isomorphic to max‑plus / maximum cycle mean reasoning.

#### Closest existing papers (you need two anchors)
1) **Iterative modulo scheduling: an algorithm for software pipelining loops**  
- Venue: **MICRO 1994** (27th Annual International Symposium on Microarchitecture)  
- Year: **1994**  
- DOI: `10.1109/MICRO.1994.717412`  
- URL: `https://dblp.org/rec/conf/micro/Rau94` ([dblp.org](https://dblp.org/rec/conf/micro/Rau94?utm_source=openai))  
**Why it’s close:** It is classic throughput/II-focused scheduling with correctness constraints (dependences/resources). This will be the reviewer’s first instinct: “this is just modulo scheduling.”

2) **A Formal Analysis of the NVIDIA PTX Memory Consistency Model**  
- Venue: **ASPLOS 2019**  
- Year: **2019**  
- DOI: `10.1145/3297858.3304043`  
- URL: `https://doi.org/10.1145/3297858.3304043` ([scinapse.io](https://www.scinapse.io/papers/2935389012?utm_source=openai))  
**Why it’s close:** It’s the canonical formal semantics anchor for PTX weak memory.

You *also* need a pomset/event-structure anchor if you build correctness on pomset refinement:

3) **Pomsets with Preconditions: A Simple Model of Relaxed Memory**  
- Venue: **OOPSLA 2020**  
- Year: **2020**  
- DOI: `10.1145/3428262`  
- URL: `https://doi.org/10.1145/3428262` ([2020.splashcon.org](https://2020.splashcon.org/details/splash-2020-oopsla/70/Pomsets-with-Preconditions-A-Simple-Model-of-Relaxed-Memory?utm_source=openai))  

#### The Delta (They did X, you do Y)
- **Rau/MICRO’94**: scheduling theory for pipelining loops (throughput), but not GPU weak-memory token protocols or formal memory-model refinement. ([dblp.org](https://dblp.org/rec/conf/micro/Rau94?utm_source=openai))  
- **Lustig/ASPLOS’19**: PTX memory model formalization + compilation mapping proofs, but not a compiler pass that uses this semantics as a *schedule legality checker* for async tensor pipelines. ([research.nvidia.com](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai))  
- **Jagadeesan/Jeffrey/Riely OOPSLA’20**: pomset model of relaxed memory + compiler optimizations, but not GPU-specific token protocols and throughput-optimal scheduling. ([2020.splashcon.org](https://2020.splashcon.org/details/splash-2020-oopsla/70/Pomsets-with-Preconditions-A-Simple-Model-of-Relaxed-Memory?utm_source=openai))  

Your **actual novel claim** must be:

> A GPU pipeline scheduler that produces (a) a **semantic refinement certificate** under a relaxed-memory model appropriate for PTX async operations, and (b) a **throughput optimality certificate** (e.g., achieving a max-plus bound) *for the same transformed region*.

To keep it honest: PTX already spells out that these ops are weak/ordered in specific ways and have UB constraints (e.g., wgmma uniformity/fence). Your correctness layer must model those explicitly, not “assume SC.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

#### Novelty killers / threats
1. **“Max-plus scheduling” ≈ “classic initiation interval bounds.”** If your novelty is just “we used maximum cycle mean,” reviewers will cite Rau (and many others) and reject. ([dblp.org](https://dblp.org/rec/conf/micro/Rau94?utm_source=openai))  
2. **Existing IRs already use tokens** (`async_copy`, groups, waits). If you merely add a scheduler pass without formal semantics + proof objects, it’s incremental. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
3. **PTX semantics are already formalized** (ASPLOS’19). If you don’t reuse/extend that or clearly justify your model choice, a reviewer will call your semantics “hand-wavy.” ([research.nvidia.com](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai))  

#### Novelty Risk Score (1–10)
**5/10** (high upside, but a big target: parts of it are “known,” and you must make the *composition + certificate* unambiguously new).

#### Killer citation (must cite)
**A Formal Analysis of the NVIDIA PTX Memory Consistency Model** (ASPLOS 2019).  
DOI `10.1145/3297858.3304043`. ([scinapse.io](https://www.scinapse.io/papers/2935389012?utm_source=openai))

---

### Direction 3 — *Affine-Lattice Encodability + Proof-Producing SMT for TMA / `cp.async.bulk.tensor`*

This is the most “PLDI-friendly” direction if you can convincingly show:
- soundness vs NVIDIA’s encode API + PTX legality,
- solver performance that supports autotuning loops,
- and *actionable* unsat cores that map to knobs (tile, swizzle, dtype, ranks).

#### Closest existing papers (SoTA neighbors)
1) **cvc5: A Versatile and Industrial-Strength SMT Solver**  
- Venue: **TACAS 2022**  
- Year: **2022**  
- DOI: `10.1007/978-3-030-99524-9_24`  
- URL: `https://doi.org/10.1007/978-3-030-99524-9_24` ([link.springer.com](https://link.springer.com/chapter/10.1007/978-3-030-99524-9_24?utm_source=openai))  
**Why it’s close:** it is your plausible proof/unsat-core backend.

2) **CoNST: Code Generator for Sparse Tensor Networks**  
- Venue: **arXiv**  
- Year: **2024**  
- URL: `https://arxiv.org/abs/2401.04836` ([arxiv.org](https://arxiv.org/abs/2401.04836?utm_source=openai))  
**Why it’s close:** demonstrates “tensor codegen decisions as SMT constraints” (even if not about NVIDIA TMA legality).

#### “Core Problem” reality check (is it already solved by vendor?)
Vendor side: **Yes, in the narrow sense** that the driver provides `cuTensorMapEncodeTiled`/`EncodeIm2col` and will reject invalid descriptors at runtime. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
But:
- it does **not** give you a proof object,
- it does **not** give you an unsat core,
- it does **not** help a compiler/autotuner systematically explore the parameter space (beyond trial-and-error calls).

Also, PTX adds additional weak-memory + legality requirements for tensor-map mutation (`tensormap.replace` is weak memory on the 1024-bit object). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))

#### The Delta (They did X, you do Y)
- **Vendor APIs**: *decide legality* by returning `CUDA_ERROR_INVALID_VALUE`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- **MLIR NVGPU**: encodes descriptor creation as a runtime call (`nvgpu.tma.create.descriptor`). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
- **Your proposal**: a solver pass that treats encodability as  
  $$P \cap (a+\Lambda) \cap \bigvee_m M_m$$  
  and outputs **(a)** a satisfying assignment + certificate, or **(b)** an unsat core mapped to layout knobs.

Crucially, cvc5 can provide unsat cores / proofs in principle (and documents `get-unsat-core` / `get-proof`). ([cvc5.github.io](https://cvc5.github.io/tutorials/beginners/outputs.html?utm_source=openai))

#### Novelty killers / threats
1. If your solver just re-encodes NVIDIA’s constraints and returns SAT/UNSAT, reviewers may say: “this is an expensive reimplementation of `cuTensorMapEncodeTiled`.” Your defense must be: **certificates + unsat cores + compiler integration** (esp. for autotuning). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
2. If you don’t show that the constraint system is stable across CUDA updates (new dtypes, swizzle modes, arch gating), reviewers will worry about bitrot. The CUDA Driver API evolves (e.g., newer toolkits add packed datatypes and swizzle restrictions). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
3. If you can’t demonstrate robust performance under autotuning workloads, it’ll be dismissed as “SMT toy.” (CoNST is a warning: SMT in tensor compilation exists; you must show *why your domain is different*.) ([arxiv.org](https://arxiv.org/abs/2401.04836?utm_source=openai))  

#### Novelty Risk Score (1–10)
**6/10** (SMT-in-compilers exists; *proof-producing* + *GPU legality* + *unsat-core-guided search* is the plausible novelty).

#### Killer citation (must cite)
**cvc5: A Versatile and Industrial-Strength SMT Solver** (TACAS 2022).  
DOI `10.1007/978-3-030-99524-9_24`. ([link.springer.com](https://link.springer.com/chapter/10.1007/978-3-030-99524-9_24?utm_source=openai))  

---

## 4. Strategic Recommendation

### Best “Academic Market Fit” (ASPLOS/POPL/PLDI)

I’d rank them for *top-tier acceptance probability* as:

1) **Direction 3 (Legality/Logic)** — **Best PLDI/POPL fit**
- Why: crisp contribution, measurable, and addresses a real pain: *autotuning + legality cliffs* for TMA / tensor async copies. The delta vs vendor is defensible: **certificates + unsat cores + deterministic legality** instead of runtime trial and error. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- Risk mitigation: scope carefully (start with `cuTensorMapEncodeTiled` + a subset of `cp.async.bulk.tensor` modes/types); show 0 mismatch vs driver + PTX legality tests. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

2) **Direction 2 (Pomset + Max-Plus)** — **High upside ASPLOS/POPL, but hardest to execute**
- Why: if you actually deliver *semantic refinement + throughput certificates* for tokenized GPU pipelines, it’s a “mathematically singular” story. ([research.nvidia.com](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai))  
- The adversarial reviewer attack will be: “max-plus scheduling is old (Rau), PTX memory model exists (Lustig), so what’s new?” Your answer must be the *composition* and *proof objects* for modern PTX async/TMA/wgmma pipelines. ([dblp.org](https://dblp.org/rec/conf/micro/Rau94?utm_source=openai))  

3) **Direction 1 (Encodable groupoids + TMEM capabilities)** — **Most conceptually novel, but highest “POPL skepticism” risk**
- Why: It can look like “nice math, unclear necessity.” You must show concrete compiler leverage: canonicalization, decidable equivalence, and *UB elimination by construction* for TMEM/wgmma operand constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- If you can produce a tight formal statement like “all well-typed programs cannot trigger TMEM UB (`tcgen05.ld` uniform `taddr`),” that becomes POPL‑worthy.

### Practical recommendation: a hybrid spine that de-risks publication

If you want the **most publishable** arc with minimum risk:
- **Lead with Direction 3** (legalize + certify descriptors and tensor-copy admissibility).  
- Add a smaller, defensible piece of Direction 2: *token‑graph scheduling with a mechanically checkable certificate of respecting required edges* (don’t overclaim full pomset semantics on day 1).  
- Use Direction 1 as a *typed IR contract layer* for TMEM (capabilities) once you have traction.

That composition matches the reality that **today’s compilers already have the ops**, but not the *proof-carrying legality layer* nor the *certificate story*. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))

---
Learn more:
1. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)
2. ['nvgpu' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)
3. [\[FEA\] Add cuTensorMapEncodeTiled to CudaHostAdapter · Issue #1566 · NVIDIA/cutlass · GitHub](https://github.com/NVIDIA/cutlass/issues/1566?utm_source=openai)
4. [triton.language.make\_tensor\_descriptor — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai)
5. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source=openai)
6. [cp.async.bulk.tensor — CUDA Core Compute Libraries](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html?utm_source=openai)
7. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
8. [⚙ D158434 \[MLIR\]\[NVGPU\] Adding \`nvgpu.wargroup.mma\` Op for Hopper GPUs](https://reviews.llvm.org/D158434?utm_source=openai)
9. [Convergence And Uniformity — LLVM 23.0.0git documentation](https://llvm.org/docs/ConvergenceAndUniformity.html?utm_source=openai)
10. [User Guide for NVPTX Back-end — LLVM 22.0.0git documentation](https://llvm.org/docs/NVPTXUsage.html?utm_source=openai)
11. [TritonNvidiaGPUOps — Triton documentation](https://triton-lang.org/main/dialects/TritonNvidiaGPUOps.html?utm_source=openai)
12. [tvm: tvm::tir::builtin Namespace Reference](https://tvm.apache.org/docs/reference/api/doxygen/namespacetvm_1_1tir_1_1builtin.html?utm_source=openai)
13. [Axe: A Simple Unified Layout Abstraction for Machine Learning Compilers](https://arxiv.org/abs/2601.19092?utm_source=openai)
14. [make\_swizzle | Modular](https://docs.modular.com/mojo/kernels/layout/swizzle/make_swizzle/?utm_source=openai)
15. [dblp: Iterative modulo scheduling: an algorithm for software pipelining loops.](https://dblp.org/rec/conf/micro/Rau94?utm_source=openai)
16. [A Formal Analysis of the NVIDIA PTX Memory Consistency Model | Performance Analytics](https://www.scinapse.io/papers/2935389012?utm_source=openai)
17. [Pomsets with Preconditions: A Simple Model of Relaxed Memory (SPLASH 2020 - OOPSLA) - SPLASH 2020](https://2020.splashcon.org/details/splash-2020-oopsla/70/Pomsets-with-Preconditions-A-Simple-Model-of-Relaxed-Memory?utm_source=openai)
18. [A Formal Analysis of the NVIDIA PTX Memory Consistency Model. | Research](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model?utm_source=openai)
19. [cvc5: A Versatile and Industrial-Strength SMT Solver | Springer Nature Link](https://link.springer.com/chapter/10.1007/978-3-030-99524-9_24?utm_source=openai)
20. [CoNST: Code Generator for Sparse Tensor Networks](https://arxiv.org/abs/2401.04836?utm_source=openai)
21. [1. Introduction — PTX ISA 8.7 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=opena
22. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)
23. [SMT Solver Outputs — Satisfiability Modulo Theories: A Beginner's Tutorial documentation](https://cvc5.github.io/tutorials/beginners/outputs.html?utm_source=openai)
