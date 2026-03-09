Below is the complete revised 8-section proposal.

---

## 1. Revised title and one-sentence thesis

**ATLAS-GEMM: Proof-Carrying Retargeting and Impossibility Certificates for Fused `sm_90a` WGMMA → `gfx942` MFMA Microkernels**

**Thesis.** For one narrow kernel family—single-workgroup FP16×FP16→FP32 GEMM with fused row-bias add—ATLAS will either produce a **small, independently checkable translation certificate** from NVIDIA `sm_90a` PTX WGMMA code to AMD `gfx942`/CDNA3 MFMA code, or a **small, independently checkable impossibility certificate** proving that no such exact retargeting exists under explicit resource/synchronization budgets. The source side is asynchronous, warpgroup-level `wgmma`; the target side is wavefront-level MFMA over 64-lane wavefronts, with FP16→FP32 shapes such as `32x32x8` and `16x16x16`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html))

**Why this revision is stronger.** I am no longer claiming a general cross-GPU retargeter. The paper is now about one exact boundary where source and target really differ: `wgmma` is collective over a warpgroup of four warps and uses `wgmma.fence`, `fence.proxy.async`, `commit_group`, and `wait_group`, while CDNA3 MFMA is collective over 64-lane wavefronts. That mismatch is the paper. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html))

---

## 2. Narrowed problem statement and running example

### Kernel family
I narrow to exactly this family:

- dense GEMM microkernels
- FP16 inputs, FP32 accumulation
- **fused row-bias add** as the only fusion in scope
- one workgroup only
- no atomics, no split-K, no inter-workgroup communication
- no approximate FP equivalence
- no TMA / `cp.async` / attention / softmax / FP8 in the main paper

### Source/target pair
- **Source:** NVIDIA `sm_90a` PTX kernels using `wgmma.mma_async`
- **Target:** AMD `gfx942`/CDNA3 kernels using MFMA intrinsics/instructions

### Running example
The running example throughout the paper is:

\[
D_{64\times 64} = A_{64\times K} \times B_{K\times 64} + \text{bias}_{64}
\]

with `K` a multiple of 16.

One source inner step is a `wgmma.mma_async ... m64n64k16.f32.f16.f16`; one target realization decomposes that logical step into an ordered set of CDNA3 `32x32x8` MFMA steps across two 64-lane wavefronts in one 128-thread workgroup. The source uses one warpgroup (4 contiguous warps = 128 threads); the target uses two wavefronts (2×64 = 128 threads). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html))

### What is out of scope
This version explicitly drops:

- universal retargeting across many GPU pairs
- arbitrary layouts
- arbitrary fusion patterns
- approximate numeric equivalence
- end-to-end LLM kernels

That scope cut is deliberate: the paper becomes a PLDI paper about **exact retargetability and exact no-go results** for one family.

---

## 3. Precise semantic contract: `Obs_exact`

This is the main fix.

For this paper, `Obs` is **not** value-only.  
For any well-formed input state `σ`, define:

\[
Obs_{exact}(K,\sigma) = (Out, GlobalEff, FPTrace, SyncReq)
\]

### 3.1 `Out` — value semantics
`Out` is the final FP32 value of every element of the `64×64` output tile in global memory.

**Preserved exactly:** bitwise equality of final FP32 outputs.

---

### 3.2 `GlobalEff` — memory effects
`GlobalEff` records:

- the exact global-memory read footprint: the source tile of `A`, source tile of `B`, and bias slice
- the exact global-memory write footprint: the output tile `D`
- the fact that there are **no other global writes**

**Preserved exactly:** same global read/write regions, same output region, no extra writes.

**Not preserved exactly:** internal shared-memory/LDS addresses. Those are treated as implementation-private scratch space.

---

### 3.3 `FPTrace` — floating-point order
For each output element `(i,j)`, `FPTrace[i,j]` is the exact ordered symbolic FP computation:

\[
[fma(A[i,0],B[0,j]), \ldots, fma(A[i,K-1],B[K-1,j]), add(bias[i])]
\]

**Preserved exactly:**

- same sequence of multiplicative contributions
- same order of accumulation
- same placement of the fused bias add

So the paper preserves **exact FP order**, not just “same real-number result up to tolerance.”

**Consequence:** any reassociation, reduction-tree change, or numerically “close enough” rewrite is **out of scope** and will either fail certification or require an impossibility certificate under the exact contract.

---

### 3.4 `SyncReq` — synchronization obligations
`SyncReq` is the minimal workgroup-local happens-before relation required for legality:

- shared/LDS producer before matrix-op consumer
- accumulator producer before bias/store consumer
- any wait required before reusing accumulator fragments
- collective participation constraints for the matrix op phases

This is where source-side `wgmma.fence`, `fence.proxy.async`, `commit_group`, and `wait_group` matter: PTX explicitly treats these as ordering requirements for legal use of shared-memory inputs and accumulator registers, and accessing accumulators before the relevant `wait_group` is undefined. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html))

**Preserved exactly:** all source legality-relevant ordering obligations must hold in the target.

**Allowed:** the target may add **stronger** ordering, but any extra ordering is recorded as explicit cost (e.g., “+1 barrier”).

---

### 3.5 What `Obs_exact` does **not** preserve
It does **not** preserve:

- cycle timing
- exact issue order of independent operations
- exact register numbers
- exact shared-memory/LDS addresses
- approximate FP equivalence

So `Obs_exact` is stronger than value equivalence, but weaker than a full microarchitectural trace. That is the intended tractability point.

---

## 4. Simplified IR: a tiny relational-linear tile IR

I simplify the old “universal relational-linear IR” into a **7-op tile IR**.

### Core idea
The IR has only two novel ingredients:

1. **Relational layout annotations**  
   Each tile/fragment carries a relation `φ` from logical matrix coordinates to physical lane/register ownership.

2. **Linear ownership tokens**  
   Accumulators and scratch tiles are linear: they cannot be silently duplicated or dropped.

### IR ops
- `load_tile`
- `barrier`
- `view(φ)`  
- `mma(shape)`
- `wait`
- `bias_row_add`
- `store_tile`

That is enough for this paper.

### Running example in the IR

**Source IR**
```text
a  = load_tile A[m:m+64, k:k+16]
b  = load_tile B[k:k+16, n:n+64]
barrier p0
a' = view(a, φ_src_A)
b' = view(b, φ_src_B)
acc1 = mma<m64,n64,k16>(a', b', acc0, order=[k..k+15])
wait g0
acc2 = bias_row_add(acc1, bias[m:m+64])
store_tile D[m:m+64, n:n+64] <- acc2
```

**Target IR**
```text
a  = load_tile A[m:m+64, k:k+16]
b  = load_tile B[k:k+16, n:n+64]
barrier p0
b' = view(b, φ_tgt_B)          # or explicit relayout if needed
acc00 = mma<m32,n32,k8>(...)
acc01 = mma<m32,n32,k8>(...)
...
acc07 = mma<m32,n32,k8>(...)
wait p1
acc2 = bias_row_add(join(acc00..acc07), bias[m:m+64])
store_tile D[m:m+64, n:n+64] <- acc2
```

### Why this is clearer
The IR is no longer “universal” in the grand sense. It is universal only across the two ISAs in this paper’s kernel family. That is a feature, not a bug.

---

## 5. Translation certificates: small and independently checkable

A translation certificate `Π` has six fields:

1. **Metadata**  
   shape, types, workgroup shape, budgets

2. **Decomposition witness**  
   how one source logical `mma` step is covered by target MFMA steps

3. **Layout witness**  
   piecewise-affine maps `φA, φB, φAcc, φOut`

4. **FP-order witness**  
   ordered `k`-slice list for each target subtile

5. **Sync witness**  
   phase graph showing target ordering satisfies `SyncReq`

6. **Cost summary**  
   extra wavefronts, extra relayouts, extra barriers, spills

### Example certificate skeleton
```text
shape = 64x64x16
src_mma = wgmma.m64n64k16
tgt_basis = 8 x mfma.m32n32k8

decomp =
  (wave0, tile[0:32,0:32], k[0:8])
  (wave0, tile[0:32,0:32], k[8:16])
  ...
  (wave1, tile[32:64,32:64], k[8:16])

fp_order = [0:8, 8:16]

sync =
  load -> barrier0 -> mfma* -> wait0 -> bias -> store

cost =
  extra_barriers = 0
  extra_relayouts = 1
  spills = 0
```

### Checker
The checker is standalone and does **no search**. It only verifies:

- exact tile coverage
- no duplicate/drop of linear fragments
- layout bijection
- exact `FPTrace`
- same `GlobalEff`
- target ordering satisfies `SyncReq`
- budget/cost accounting

So the certificate is:

- **small**: schema-sized, not execution-trace-sized
- **independent**: checkable without rerunning synthesis
- **first-class artifact**: source + target + certificate is enough

---

## 6. Impossibility certificates as a primary contribution

This is now co-equal with translation.

### Query form
For source kernel `S` and budget `β`, ask:

\[
\exists T \in Lang(\beta).\; Obs_{exact}(T)=Obs_{exact}(S)
\]

If yes: return translation certificate `Π`.

If no: return impossibility certificate `Κ`.

### Impossibility certificate format
`Κ = (Core, Derivation, RepairLabel)`

- **Core**: a minimal unsat core of layout/coverage/order/sync/resource constraints
- **Derivation**: replayable linear-arithmetic / graph contradiction proof
- **RepairLabel**: the smallest missing capability, e.g.
  - `+1 wavefront`
  - `+1 relayout`
  - `+1 barrier`
  - `relax FP order`

### Canonical negative results the paper will seek
1. **No 1-wavefront exact translation**  
   sanity-check impossibility: a single 64-lane wavefront cannot realize the running example under the chosen MFMA basis.

2. **No zero-relayout exact translation**  
   more interesting: the source logical fragment ownership and the target MFMA ownership relation may be non-isomorphic unless an explicit relayout is inserted.

3. **No zero-extra-sync exact translation**  
   if the target must strengthen ordering to satisfy source legality obligations, ATLAS proves that “+1 barrier/wait” is necessary.

### Why this matters
The paper’s claim is no longer “we translate many kernels.”  
It is:

> For every kernel in scope, ATLAS returns either a proof of exact retargetability or a proof of exact non-retargetability under explicit budgets.

That is much stronger scientifically than “the search timed out.”

---

## 7. Evaluation plan

### Benchmarks
A realistic corpus is **10–12 kernels**, not 100+:

- 4 hand-written source microkernels for the running family
- 4 layout/sync variants designed to stress negative cases
- 2–4 compiler-emitted kernels, only if they fit the exact subset

### Questions
1. **Correctness:**  
   Does the checker accept only valid certificates?

2. **Compactness:**  
   How large are `Π` and `Κ`?

3. **Coverage:**  
   For the corpus, how often do we get translation vs impossibility?

4. **Diagnostic value:**  
   Do impossibility certificates identify the right missing capability (`+relayout`, `+barrier`, etc.)?

5. **Performance:**  
   On positive cases, how close is generated MFMA code to a hand-tuned AMD baseline?

### Success criteria
A successful paper does **not** need 100% translation rate.  
It needs:

- every kernel gets **either** `Π` **or** `Κ`
- certs are small and fast to check
- impossibility results are nontrivial and actionable
- positive cases are not grossly slower than expert MFMA code

---

## 8. Feasible 15-month plan, risks, and final contributions

### Months 1–3
- freeze subset and running example
- build source and target lifters into the tiny tile IR
- formalize `Obs_exact`

**Deliverable:** semantics note + 2 lifted kernels

### Months 4–6
- implement standalone checker
- define concrete certificate schema
- prove checker sound at the tile-IR level

**Deliverable:** first end-to-end checked certificate for the running example

### Months 7–9
- implement synthesis/search for positive certificates
- support target decomposition into `32x32x8` MFMA basis
- support fused row-bias

**Deliverable:** 4–6 positive translations

### Months 10–12
- encode impossibility queries
- generate unsat-core-based impossibility certificates
- add repair labels

**Deliverable:** 3–4 strong negative results

### Months 13–15
- full evaluation
- paper writing
- artifact packaging

### Months 16–18 (buffer / optional extension)
- add one extra epilogue (e.g., residual add or ReLU)
- add one extra source shape
- polish proof/mechanization

---

### Main risks and mitigations

- **Risk:** full PTX/AMDGPU semantics are too large  
  **Mitigation:** soundness is relative to validated lifting into the tiny tile IR.

- **Risk:** search explodes  
  **Mitigation:** fixed kernel family, fixed source/target pair, fixed target basis.

- **Risk:** exact FP order is too strong  
  **Mitigation:** that is intentional; negative cases become publishable impossibility results.

- **Risk:** performance is weaker than expert code  
  **Mitigation:** the paper’s primary contribution is proof-carrying translation / no-go certificates, not absolute autotuning wins.

---

## Final claimed contributions

1. A **precise** semantic contract that preserves:
   - value semantics
   - global memory effects
   - exact FP order
   - synchronization obligations

2. A **tiny relational-linear tile IR** tailored to one real retargeting boundary.

3. **Small, independent translation certificates**.

4. **Small, independent impossibility certificates** with explicit repair labels.

5. A realistic, artifact-friendly PLDI paper on one narrowed application: fused GEMM+bias retargeting from `sm_90a` WGMMA to `gfx942` MFMA.

If you want, I can next turn this into a **PLDI-style 1-page abstract + introduction**, or into a **full 6–8 page project proposal draft**.
