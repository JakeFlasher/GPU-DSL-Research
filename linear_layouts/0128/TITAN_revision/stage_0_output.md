## Forensic hardware fact‑check of **TITAN** (PL_proposal.tex)

This review treats as **hardware‑mapping claims** anything that asserts (explicitly or implicitly) that a TITAN IR construct (token, typestate, fence, barrier) can be **lowered** to **specific GPU primitives** on NVIDIA H100 (SM90) and AMD MI300 (CDNA3) without semantic gaps.

Primary grounding sources used:
- **NVIDIA PTX ISA** (CUDA 12.9.1 / PTX ISA 8.8) for `cp.async.bulk{.tensor}`, `mbarrier`, and `fence.proxy.*`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
- **CUDA C++ Programming Guide** (12.6.0) for *TMA usage patterns*, alignment tables, and proxy‑fence guidance. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html))  
- **CUDA Driver API** for tensor‑map (TMA descriptor) encoding requirements. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **ROCm docs + LLVM AMDGPU docs** for MI300 wavefront/LDS and `s_waitcnt` / `s_barrier` semantics (since AMD CDNA3 ISA PDFs are not fully public in one canonical place). ([rocm.docs.amd.com](https://rocm.docs.amd.com/en/docs-6.1.2/reference/gpu-arch-specs.html))  

---

## Targeted checks (your “Specific Verification Targets”)

### 1) **TMA alignment / swizzle requirements**
**Finding:** The proposal *does not* account for the hard alignment rules that `cp.async.bulk.tensor` (TMA) inherits from the tensor map + CC 9.0 requirements.

Concrete requirements (CC 9.0, multi‑dim bulk tensor copy):
- **Shared memory destination address must be 128‑byte aligned.** ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html))  
- **Global memory address must be 16‑byte aligned** (baseline), with **global strides multiples of 16 bytes**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html))  
- Swizzle modes impose additional constraints; CUDA’s async-copies guidance summarizes that for CC9 swizzle patterns, **global alignment becomes 128 bytes** for swizzled modes and the pattern repeats at 256/512/1024 bytes depending on mode. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html))  

**Verdict:** **False by omission** (proposal doesn’t mention alignment/padding and ring‑buffer stage alignment constraints).

---

### 2) **Barrier granularity vs. per‑token storage**
**Finding:** `mbarrier` can support **per‑stage** synchronization cheaply, but **not the exact API shape shown in the proposal**.

Hardware facts:
- An `mbarrier` object is `.b64`, **8‑byte aligned**, lives in `.shared`, and is **user‑defined** (limited by shared memory, not a fixed small barrier index set). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
- `mbarrier` is **phase‑based** and can track async transactions via **tx‑count**, so you can reuse one barrier over many iterations (i.e., per pipeline stage, across phases). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
- **Critical mismatch:** waiting is not just `wait(bar)`; PTX `mbarrier.test_wait` / `mbarrier.try_wait` checks completion of a phase identified by a **state token returned by `mbarrier.arrive`**, or by phase parity. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

**Verdict:** **Valid in principle** (per-stage barriers are feasible and cheap), but **the proposal’s token<->barrier API is oversimplified**.

---

### 3) **AMD portability: `s_waitcnt` vs token/capability**
**Finding:** There is an **impedance mismatch**. `s_waitcnt` is fundamentally **counter-based**, not a “wait on a specific transfer handle”.

- `s_waitcnt` waits on **counts of outstanding operations** (`vmcnt`, `lgkmcnt`, etc.). It does not identify an individual memory operation or DMA transfer. ([llvm.org](https://llvm.org/docs/AMDGPU/gfx90a_waitcnt.html))  

So, a TITAN `!token` that semantically names *“this specific copy”* cannot be mapped 1:1 unless you (a) force each tokenized “copy” to be the only in‑flight op in its counter class (which **serializes**), or (b) weaken the abstraction to stage‑level ordering (token ≈ “all ops before this point”).  

**Verdict:** **False** if the proposal intends *true per‑copy capability tokens* on AMD.

---

### 4) **Proxy fences: when is `fence.proxy.async` needed?**
**Finding:** The proposal identifies the need for proxy fences, but its rule is incomplete: it must model **(at least) three relevant proxy domains on Hopper** and the **directionality + implicit fences**.

Hard facts:
- `cp{.reduce}.async.bulk` operations run in the **async proxy**; cross‑proxy use of a memory location requires **`fence.proxy.async`** between generic and async proxies. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
- **Completion** of a `cp{.reduce}.async.bulk` op is followed by an **implicit generic↔async proxy fence**, making results visible to the generic proxy once completion is observed. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
- `cp.async.bulk.tensor` reads the tensor map via the **tensormap proxy** (separate from generic and async). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
- Ordering between generic writes to tensor maps and tensormap‑proxy reads uses **`fence.proxy.tensormap::generic`** (or the fused `tensormap.cp_fenceproxy`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
- CUDA’s guide explicitly calls out `fence.proxy.async.shared::cta` after barrier init and before bulk async copies, and also before async reads of shared memory after generic writes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html))  

**Verdict:** **Oversimplified**: the proposal needs explicit handling of **tensormap vs async vs generic** proxies and must exploit the **implicit fence on completion** to avoid over‑fencing.

---

## Claim Verification Matrix

> Columns: Proposal_Claim \| Hardware_Reality (PTX/CDNA3) \| Verdict \| Citation

| Proposal_Claim | Hardware_Reality (PTX/CDNA3) | Verdict | Citation |
|---|---|---|---|
| **“TMA bulk tensor transfers (global↔shared) …”** | `cp.async.bulk.tensor` exists for global→shared (`.shared::cta` or `.shared::cluster`). Shared→global exists but uses **bulk async‑group** completion (not mbarrier). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Oversimplified | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **“… and cluster shared↔shared via DSMEM”** | CUDA defines **Distributed Shared Memory** for thread block clusters (a.k.a. `shared::cluster`). PTX has bulk copies involving `.shared::cluster` (e.g., `.shared::cta → .shared::cluster`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)) | Valid | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)) |
| **“Use tensor-map descriptor for 2D–5D tiling.”** | `cp.async.bulk.tensor` takes `[tensorMap, tensorCoords]` and supports `.1d`…`.5d`. Tensor map is a **128‑byte** opaque object. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Valid | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **“Tensor-map descriptor metadata writes may require proxy fences.”** | Tensor map is accessed in **tensormap proxy**, and PTX exposes `fence.proxy.tensormap::generic` and `tensormap.cp_fenceproxy` for establishing ordering. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Valid (but under‑specified) | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **“Completion is synchronized via mbarrier arrive/wait.”** | mbarrier‑based completion exists, but PTX “wait” is `mbarrier.test_wait` / `mbarrier.try_wait`, and it checks completion of a phase identified by a **state token returned by `mbarrier.arrive`** (or parity). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Oversimplified | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **IR sketch: `copy_tensor -> token; mbarrier_arrive(bar, token); mbarrier_wait(bar)`** | On NVIDIA, the natural “token” for waiting is the **mbarrier phase state** from `mbarrier.arrive{.expect_tx}`; the copy signals completion via `complete_tx` to the barrier. Waiting requires the state/parity. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Oversimplified (API mismatch) | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **“Per-stage barriers (ring buffer stage i has barrier i)”** | `mbarrier` objects are **user‑defined** and only limited by shared memory; size is `.b64` (8 bytes aligned). Phases + tx-count make “per stage, many iterations” feasible. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Valid | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **“TMA descriptors (tensor maps) are first-class; created via CUDA APIs.”** | Driver API `cuTensorMapEncodeTiled` creates tensor maps; requirement: tensorMap address aligned **64 bytes** (descriptor storage rule). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | Valid | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) |
| **TMA alignment target: “128‑byte alignment/swizzle requirements of `cp.async.bulk.tensor`.”** | For CC 9.0 multi‑dim bulk tensor async copy: **shared address 128B aligned**, global address 16B aligned, strides multiple 16B. Swizzle patterns impose stronger global alignment (128B) and repeat boundaries. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)) | Proposal misses this ⇒ False-by-omission | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)) |
| **“Proxy fences are required; TITAN inserts minimal fences.”** | PTX explicitly: bulk async copies run in **async proxy**; accessing same location across proxies needs **`fence.proxy.async`**; completion implies implicit generic↔async fence. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Valid | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **Proxy-fence example logic: “GenericWrites ⇒ AsyncReads”** | CUDA guide gives concrete cases: after initializing a shared barrier, use `fence.proxy.async.shared::cta` so bulk async ops operate on initialized barrier; also after generic writes to shared memory that will be read by bulk async ops, fence is required. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)) | Oversimplified (must include data-buffer writes + scope/state-space) | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)) |
| **“Typestate accounts for 3 proxies (Generic, Async, Tensor).”** | Hopper has **(at least)** generic proxy, async proxy, and tensormap proxy exposed via PTX (`fence.proxy.async…`, `fence.proxy.tensormap::generic…`). Proposal text mentions generic/async but not tensormap explicitly. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Oversimplified / incomplete | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **“TITAN IR needs only barrier ops for completion.”** | PTX supports **two completion mechanisms**: async-group and mbarrier-based; additionally, `cp.async.bulk.tensor` shared→global uses `.bulk_group` completion. IR needs group commit/wait ops in addition to mbarrier ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | False (missing mechanism) | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **“Correctness-by-construction prevents use-before-ready / overwrite-before-consume.”** | PTX explicitly states: after starting async copy, modifying source/tensor descriptor or reading destination before completion is **undefined behavior**; a typestate/token discipline can enforce this if it also covers descriptor lifetimes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) | Valid (if extended to tensorMap) | ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)) |
| **“AMD MI300 uses wave64 and LDS.”** | MI300X/MI300A: wavefront size **64**, LDS **64 KiB**. ([rocm.docs.amd.com](https://rocm.docs.amd.com/en/docs-6.1.2/reference/gpu-arch-specs.html)) | Valid | ([rocm.docs.amd.com](https://rocm.docs.amd.com/en/docs-6.1.2/reference/gpu-arch-specs.html)) |
| **“AMD path: tokens correspond to outstanding memory operations; consume via `s_waitcnt`.”** | `s_waitcnt` waits until **counts** of outstanding operations match thresholds (vmcnt/lgkmcnt/expcnt). This is not a named-handle mechanism. ([llvm.org](https://llvm.org/docs/AMDGPU/gfx90a_waitcnt.html)) | Oversimplified | ([llvm.org](https://llvm.org/docs/AMDGPU/gfx90a_waitcnt.html)) |
| **AMD portability target: “Does `s_waitcnt` wait a specific transfer/token?”** | No: it’s counter‑based (wait for outstanding operations count). You can’t directly “wait(t)” for one transfer unless you constrain scheduling so that token uniquely maps to a counter threshold (often serializing). ([llvm.org](https://llvm.org/docs/AMDGPU/gfx90a_waitcnt.html)) | False (for per‑transfer tokens) | ([llvm.org](https://llvm.org/docs/AMDGPU/gfx90a_waitcnt.html)) |
| **“AMD wave64-compatible barriers” analogous to `mbarrier`** | AMD has workgroup execution barriers (`s_barrier` and related intrinsics), but LLVM documents that barriers synchronize execution and may require `s_waitcnt …(0)` before `s_barrier` on some targets; barriers are not transaction-counted completion objects like NVIDIA `mbarrier`. ([llvm.org](https://llvm.org/docs/AMDGPUUsage.html)) | Oversimplified | ([llvm.org](https://llvm.org/docs/AMDGPUUsage.html)) |

---

## Hardware Mismatch List (what the proposal gets wrong or dangerously underspecifies)

1) **TMA legality constraints are missing (alignment/stride/swizzle).**  
   - Multi‑dim TMA requires **128B shared alignment**; ring‑buffer stage bases must preserve it, which implies either (a) stage size multiple-of-128, or (b) per‑stage padding. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html))  
   - Swizzle modes can imply **repeat‑boundary effects (256/512/1024B)** and stronger global alignment (128B), impacting both allocation and consumer indexing. ([docs.nvidia.cn](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html))  

2) **Barrier API shape mismatch (token is required for wait).**  
   The proposal’s `mbarrier_wait(bar)` omits the **phase token/state** that PTX requires (`mbarrier.test_wait/try_wait` checks a phase given by state or parity). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

3) **Missing the second completion mechanism (bulk async-groups) — especially for stores.**  
   `cp.async.bulk.tensor` shared→global uses `.bulk_group` completion; CUDA’s guide explicitly says shared→global completion isn’t tracked by shared barrier, but by bulk async-group commit/wait. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

4) **Proxy model incomplete (“3 proxies” is real; proposal mostly models 2).**  
   You need at minimum:
   - generic ↔ async: `fence.proxy.async.*` ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
   - generic ↔ tensormap: `fence.proxy.tensormap::generic.*` or `tensormap.cp_fenceproxy` ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  
   The proposal’s text talks about “metadata written by generic then consumed by async proxy” but does not explicitly include **tensormap proxy** semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

5) **Incorrect fence placement opportunities if implicit fences aren’t modeled.**  
   PTX: completion of bulk async copies is followed by an **implicit generic↔async proxy fence**. If TITAN doesn’t model this, it may insert redundant fences that cost overlap. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

6) **Cluster/barrier corner-case:** `mbarrier` ops are limited for `shared::cluster`.  
   PTX summarizes that “other” mbarrier ops aren’t supported on `shared::cluster` objects; this matters if TITAN intends per-stage barriers in cluster memory with remote waits. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

7) **AMD “token” semantics do not map cleanly to `s_waitcnt`.**  
   `s_waitcnt` cannot select a specific transfer; it gates on outstanding op counts, which pushes TITAN either toward serialization or toward weakening token semantics to coarse stage-level constraints. ([llvm.org](https://llvm.org/docs/AMDGPU/gfx90a_waitcnt.html))  

---

## Fatal Flaws (abstractions the hardware cannot support efficiently)

1) **AMD: per-transfer `!token` (capability-style) is not representable by `s_waitcnt` without serializing.**  
   Because `s_waitcnt` is counter-based, not handle-based, the compiler cannot “wait(token_i)” precisely while allowing other outstanding ops to overlap. Any faithful lowering either waits for *too much* (stalling overlap) or forces schedule constraints that defeat pipelining. ([llvm.org](https://llvm.org/docs/AMDGPU/gfx90a_waitcnt.html))  

2) **NVIDIA: claiming a single mbarrier-based token protocol covers *both* directions of TMA (global→shared and shared→global) is wrong.**  
   Shared→global `cp.async.bulk.tensor` uses `.bulk_group` completion, not mbarrier; without explicit IR support for bulk async-groups, TITAN can’t correctly model completion/wait for stores. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

If TITAN restricts itself to **global→shared** only (load side), flaw (2) becomes a scope limitation rather than fatal; but the proposal currently states “global ↔ shared”.

---

## What would make the proposal pass a microarchitecture sanity check (minimal fixes)

If you want TITAN to be hardware-faithful **and** plausibly performance-neutral on NVIDIA, the proposal should explicitly add:

1) **A legality/constraint module for TMA**  
   - Stage base alignment (128B), stage stride constraints, padding rules, and swizzle-mode constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html))  

2) **A correct barrier token model**  
   - Represent the **mbarrier phase token/state** explicitly in IR (what CUDA C++ calls `barrier::arrival_token` / what PTX returns from `mbarrier.arrive`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

3) **First-class async-group completion ops**  
   - For `.bulk_group`: `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group{_read}` analogs, since this is the completion model for at least shared→global tensor copies. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

4) **Proxy fences modeled as a 3‑proxy effect system**  
   - generic↔async and generic↔tensormap, including exploiting the *implicit fence on completion* to avoid over-fencing. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html))  

On AMD, the proposal must **weaken** the token semantics (token ≈ stage-level ordering point, not specific transfer) or introduce a different backend contract than “capability token”.

---

If you want, I can produce a *revised IR op set* that is actually isomorphic to the PTX mechanisms (mbarrier phase token + bulk_group wait token + explicit proxy fence kinds) and mark exactly which ops are NVIDIA-only vs AMD-lowerable.

---
Learn more:
1. [https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)
2. [https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)
3. [https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group\_\_CUDA\_\_TENSOR\_\_MEMORY.html](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
4. [https://rocm.docs.amd.com/en/docs-6.1.2/reference/gpu-arch-specs.html](https://rocm.docs.amd.com/en/docs-6.1.2/reference/gpu-arch-specs.html)
5. [https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html](https://docs.nvidia.cn/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
6. [https://llvm.org/docs/AMDGPU/gfx90a\_waitcnt.html](https://llvm.org/docs/AMDGPU/gfx90a_waitcnt.html)
7. [https://llvm.org/docs/AMDGPUUsage.html](https://llvm.org/docs/AMDGPUUsage.html)
