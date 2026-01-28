## Stage 2 — Theoretical Extension & Implementation Strategy  
*(Principal Compiler Theorist & Systems Architect; ASPLOS/PLDI/ISCA focus — “Mathematical elegance must survive hardware reality.”)*

### 0) Access verification (Rule S3)

I verified that all three seed artifacts are reachable and parseable:

1. **Linear Layouts (\(\mathbb{F}_2\))** — arXiv HTML v3 loads (full section structure visible). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
2. **ISL / Integer Set Relations** — arXiv HTML v1 loads (sections on CuTe swizzle, linear layouts, ISL ops). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
3. **Categorical Foundations (CuTe)** — arXiv PDF v1 loads (Jan 2026; definition of tractable layouts is extractable). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

I also verified the key hardware manuals / interfaces used to validate Stage‑1 bottlenecks:

- **CUDA Driver API: `cuTensorMapEncodeTiled` / TensorMap constraints** ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- **PTX ISA: `cp.async.bulk{,.tensor}` + memory ordering + completion mechanisms** ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
- **Hopper tuning guide: TMA + warp specialization rationale** ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  
- **AMD LDS bank conflict lane‑phasing rules (MI‑class)** ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html?utm_source=openai))  

---

## 1) Deep research phase — validating Stage‑1 “bottlenecks” against current docs + literature

Stage 1 claimed three “performance cliffs.” Below is a **hardware-grounded verification** (not just restating the math).

### 1.1 Cliff A (validated): **Descriptor admissibility gap is real and hard-coded in the API**

**Stage‑1 claim:** “A bijective / ‘good’ layout in the \(\mathbb{F}_2\) algebra may be un-encodable as a TMA TensorMap descriptor → you fall off the TMA fast-path.”

**Validation:** `cuTensorMapEncodeTiled` imposes *structural constraints that are not ‘layout algebra’ constraints*:

- TensorMap is only supported on sufficiently new GPUs (cc ≥ 9.0 is called out), and rank is capped (≤ 5). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- Strides must satisfy **coarse byte-multiple constraints** (multiples of 16B, sometimes 32B depending on interleave / packed types). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- `boxDim` is bounded (≤ 256), and when `interleave == NONE`, the inner box byte-size must be a multiple of 16B. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- **Swizzle is a finite enumeration**, and newer docs explicitly expose *swizzle atomicity variants* (e.g., ATOM modes). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- There’s also a concrete “quirk” that is extremely compiler-relevant: when `interleave == NONE`, **TMA ignores / doesn’t support stride for dimension 0** (so the descriptor language is not “general affine in all dims”). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

So the Stage‑1 framing is correct: **“layout validity” ≠ “descriptor realizability.”** TMA is a *typed interface*, not a generic address function.

Now connect to seed math: Linear Layouts deliberately works in a broad space of invertible linear maps (and even restricts to “1–2 bits/column” for “memory layouts”). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
That restriction still permits XOR mixing patterns that have no representation as `(globalStrides, boxDim, swizzleEnum, alignment)`. Your Stage‑1 “admissibility subtype” is not optional; it is demanded by the API surface. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

### 1.2 Cliff B (validated): **Spatial-only layout reasoning is insufficient because PTX async ops are weakly ordered + barrier-typed**

**Stage‑1 claim:** seeds model *where*, not *when*; but Hopper pipelines require explicit barrier choreography.

**Validation:** PTX is explicit:

- `cp.async.bulk` is non-blocking and uses **mbarrier completion** in some forms; it is also treated as a **weak memory operation**, and the completion has defined **release semantics** at a specified scope. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
- `cp.async.bulk.tensor` similarly has explicit completion mechanisms; the doc distinguishes `.mbarrier::complete_tx::bytes` vs `.bulk_group`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  

And the Hopper tuning guide makes the performance intent explicit: **TMA exists to enable warp specialization** where a single thread issues bulk movement and others compute; but this only works if the pipeline and synchronization are correct. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  

So: even if you had a perfect “bank-conflict-free” spatial swizzle, **the wrong placement of waits / arrives collapses overlap** into dead time.

### 1.3 Cliff C (validated, and the landscape moved): **Power-of-two + raggedness is a real workload collision, and the community is actively patching it**

Seed A explicitly states the **power-of-two restriction** and the “lift + mask” mitigation. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
Seed B also states the binary vector space requirement: dimension sizes must be powers of two to fit the representation. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

On the “hardware reality” side: you cannot simply “mask away” in the descriptor world without consequences, because TMA box/stride alignment rules constrain what “masked” even means at the transfer granularity. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

**Novelty check:** there is already visible momentum on “ragged TMA”:

- Triton added “write-only ragged TMAs” and notes testing on **H100 and GB200** (i.e., Hopper and Blackwell-class). ([github.com](https://github.com/triton-lang/triton/pull/7792))  
- Triton release notes (3.5.0) also mention “general ragged TMAs” and “automatic bounds checking using higher-dimensional TMA descriptors.” ([newreleases.io](https://newreleases.io/project/github/triton-lang/triton/release/v3.5.0))  
- Academic work exists on removing padding requirements using a **descriptor pool + runtime selection** (“TMA-Adaptive FP8 Grouped GEMM”). ([arxiv.org](https://arxiv.org/abs/2508.16584?utm_source=openai))  

Interpretation: the cliff is real, and the ecosystem is converging on **descriptor families + runtime dispatch** as the practical bridge.

### 1.4 (Cross-vendor validation): **Bank conflict freedom is instruction-indexed on AMD (lane phasing), so a single layout-only predicate is wrong**

Stage‑1’s MI300 point is well supported: AMD LDS conflicts depend on lane grouping *per instruction*.

- ROCm blog explicitly lists **different lane groups for `ds_write_b128` vs `ds_read_b128`** when defining “conflict-free.” ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html?utm_source=openai))  
- CK docs reinforce the same lane-phase grouping story and explain access is divided into phases. ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai))  

So any “bank conflict minimization” objective must be parameterized by `(opcode, width, lane-phase)`, not just the address mapping.

---

## 2) Novelty landscape — what already exists (so we don’t reinvent it)

To stay PLDI/ASPLOS-novel in 2026, we must acknowledge existing pieces:

1. **MLIR already has a TMA / WGMMA surface** (`nvgpu.tma.create.descriptor`, `nvgpu.tma.async.load`, warpgroup descriptor ops). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
   *Meaning:* adding “a TMA op” is not novel; the missing part is **synthesis + correctness + integration with formal layout search**.

2. **Warp specialization automation is already a live research line.**  
   - *Tawa* introduces “asynchronous references (aref)” to automate warp specialization on H100. ([arxiv.org](https://arxiv.org/abs/2510.14719?utm_source=openai))  
   - There’s also a performance model for warp specialization kernels. ([arxiv.org](https://arxiv.org/abs/2506.11209?utm_source=openai))  

   *Meaning:* proposing “a pipeline abstraction” must differentiate: e.g., **integrate layout algebra + descriptor admissibility + effect typing**.

3. **Ragged TMA support is landing in production toolchains** (Triton), and padding elimination via descriptor pooling exists as a paper. ([github.com](https://github.com/triton-lang/triton/pull/7792))  

   *Meaning:* our “power-of-two tyranny” direction must generalize from ad-hoc raggedness to a **formal mixed-radix/piecewise layout semantics** that compilers can reason about and verify.

---

## 3) Scaffold & plan — mapping validated gaps → “Stage 1.5” theories → artifacts

You referenced a “Stage 1.5 theoretical arsenal,” but I don’t have that attachment in this chat. I’ll therefore use (a) the theories implied by Stage‑1 and (b) what the current literature suggests. If you paste Stage‑1.5, I’ll remap precisely.

### Table — Gap → Theory → Compiler artifact (the synthesis formula)

| Validated Hardware Gap (Stage 1) | Math / PL Theory Lever (Stage 1.5 candidates) | New Compiler/Runtime Artifact |
|---|---|---|
| **Descriptor admissibility gap** (TMA/WGMMA require strict descriptor constraints; general \(\mathbb{F}_2\) layouts can be unencodable) ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)) | **Refinement types** (“layout is valid” vs “layout is TMA-admissible”), plus **template-based synthesis** (SMT / Presburger + finite enum search), plus **normal-form/factorization** (rewrite \(\mathbb{F}_2\) layouts into allowed swizzle atoms) ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | **Descriptor-Refined Layout IR + Synthesis Pass**: `Layout ⟶ (TensorMapDescriptor | fail)` and `Layout ⟶ (WGMMA_Descriptor | convert_layout)` integrated into Triton/MLIR (NVGPU backend) ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)) |
| **Spatial vs temporal** (PTX async is weakly ordered; correctness/perf require tokens + barrier choreography) ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai)) | **Effect systems / linear types** for async tokens, plus **dataflow IR** (e.g., aref-style) that supports automatic warp specialization and pipeline scheduling ([arxiv.org](https://arxiv.org/abs/2510.14719?utm_source=openai)) | **Effectful Async Layout Dialect**: SSA tokens for TMA/cp.async, verified ordering + auto-pipelining + warp partitioning |
| **Power-of-two tyranny & raggedness** (padding+mask is semantically OK but performance-toxic; descriptor constraints further restrict) ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | **Mixed-radix / integer lattice** extensions (e.g., Smith Normal Form as canonicalization), plus **piecewise affine relations** (ISL), plus **runtime descriptor pooling** ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | **Piecewise Layouts + Descriptor Pooling**: compile “core + tails” regions into a small finite set of descriptors + runtime select; verify equivalence with ISL |

---

## 4) Three research directions (distinct, high-value)

Each direction follows:  
**Proposal = (Hardware gap) + (Math/PL theory) → (New compiler/runtime artifact)**

---

# Direction 1 — **Descriptor-Refined Layout Subtyping + Synthesis (TMA/WGMMA contracts)**  
*(ASPLOS/PLDI core)*

### 4.1 Gap (why the seeds are insufficient)

**Seed A (Linear Layouts)** gives a huge layout language: “memory layouts” are **invertible linear layouts** with columns having 1 or 2 nonzero bits. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
That space includes XOR-mixing patterns that are perfectly valid bijections but have **no representation** as a TMA TensorMap descriptor.

TMA descriptor legality is not “nice-to-have”; it is hard-coded:

- limited rank and tile sizes, strict byte-alignment/stride multiple constraints ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
- finite swizzle enums and (newer PTX/CUDA) swizzle-atomicity submodes ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

Similarly, WGMMA uses **descriptors with swizzle modes and stride fields**, and the acceptable layouts are sharply restricted (not “all \(\mathbb{F}_2\) layouts”). MLIR even has explicit warpgroup descriptor ops as a reflection of this reality. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

**Bottom line:** the seed formalisms lack a *hardware-refined subtype system*:
\[
\texttt{Layout} \;\supset\; \texttt{TMA\_Layout} \quad\text{and}\quad \texttt{Layout} \;\supset\; \texttt{WGMMA\_OperandLayout}
\]
and without this, “layout search” can select a layout that triggers a **fast-path cliff**.

### 4.2 Apply the theory (Stage 1.5 lever)

**(A) Refinement types for layouts (hardware contracts).**  
Model descriptor legality as a refinement predicate:
\[
\texttt{TMA\_Admissible}(L, \text{dtype}, \text{tile}, \text{align}) 
\]
and similarly for WGMMA operand descriptors:
\[
\texttt{WGMMA\_AdmissibleA}(L),\ \texttt{WGMMA\_AdmissibleB}(L)
\]

**(B) Template-based synthesis + finite-enum search.**  
Express TMA as a template family:
- affine traversal via `globalStrides`
- plus `swizzle ∈ Enum`
- plus alignment/box constraints

Then solve:
\[
\exists \texttt{params}. \;\; L \equiv \texttt{TensorMap}(\texttt{params})
\]

**(C) Normal forms / factorization (“Swizzle Atomicity Normal Form”).**  
Treat TMA/WGMMA swizzles as *a small set of generators* (e.g., 32B/64B/128B + atomicity variants), and attempt to rewrite a general \(\mathbb{F}_2\) layout into:
\[
L \;=\; (\underbrace{L_{\text{desc}}}_{\text{TMA/WGMMA encodable}}) \circ (\underbrace{L_{\text{resid}}}_{\text{pure shuffle/permutation}})
\]
where \(L_{\text{resid}}\) is implemented by shuffles/permutes.

This directly targets the Stage‑1 “swizzle atomicity mismatch” problem, and is grounded in the PTX definition of atomicity submodes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html?utm_source=openai))  

### 4.3 Define the mechanism (software artifact)

**Artifact:** *Descriptor-Refined Layout IR (DRLIR)* + *Synthesis pass*

#### (1) IR design (MLIR/Triton feasible)
- Add a type/attribute layer:
  - `#layout.linear_f2<Matrix>` (Seed A)
  - `#layout.isl_rel<Relation>` (Seed B)
  - `#layout.cute_morphism<...>` (Seed C)
  - **Refinements:** `#layout.tma<swizzle, interleave, l2promo, oob>` and `#layout.wgmma_operand<role=A|B, swizzleMode, leadingDim, stride>`
- Lowering emits `nvgpu.tma.create.descriptor` / `nvgpu.tma.async.load` only if refinement proof exists. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

#### (2) Synthesis pipeline
1. **Lift layout** into a common representation (ISL relation) — this is exactly Seed B’s unification story. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
2. **Try to match/synthesize** a TensorMap descriptor:
   - enumerate swizzle/interleave options
   - solve for strides and boxDims
   - validate alignment constraints (from Driver API)
3. If success: emit NVGPU TMA ops; else:
   - compute factorization \(L = L_{\text{desc}} \circ L_{\text{resid}}\)
   - emit TMA for \(L_{\text{desc}}\) and shuffle for \(L_{\text{resid}}\)

#### (3) Where the seed math plugs into codegen
- Seed A gives algebraic composition/inversion; we use it to compute candidate \(L_{\text{desc}}\) quickly.
- Seed B (ISL) is the equivalence checker: confirm synthesized descriptor mapping equals target relation. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Seed C’s “tractable” subset becomes the *fast-path detection*: tractable/affine-ish layouts are likely encodable; non-tractable are “needs factorization or fallback.” ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

### 4.4 Math-to-hardware mapping diagram (subtyping + synthesis)

```text
Seed layout (huge):
  L : coords -> offsets
  (F2-linear, ISL relation, or CuTe morphism)

          typecheck/synthesize
                 |
                 v
  ┌───────────────────────────────────────────┐
  │  TMA_Layout<L>  (refined subtype)          │
  │   - rank <= 5, boxDim <= 256               │  (Driver API)
  │   - globalStrides multiple-of-16B(/32B)    │
  │   - swizzle ∈ finite enum (+ atomicity)    │  (PTX/CUDA)
  └───────────────────────────────────────────┘
                 |
                 v
  nvgpu.tma.create.descriptor / tma.async.load  (NVGPU dialect)
```

*(The refinement boundary is a first-class compiler decision, not a late “lowering failure.”)* ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  

### 4.5 Feasibility (C1) and evaluation plan (C2)

**Implementation plan (3–4 months prototype):**
- Build pass in Triton backend or MLIR-based pipeline that:
  - converts layout to ISL relation (reuse Seed B approach)
  - searches descriptor template space
  - emits NVGPU TMA ops when possible

**Evaluation (TritonBench + microbench):**
- Benchmarks: GEMM staging, transpose/convert_layout, FlashAttention-like tiles; plus ragged cases for “fail → factorize.”  
- Metrics:
  - **Speed** (end-to-end + kernel time)
  - **Compilation time** (synthesis + ISL checks)
  - **Code size** (PTX/SASS instruction count, registers)
  - **TMA utilization** (Nsight: TMA-related throughput counters)
- Use TritonBench operator suite as harness. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

**Expected paper contribution:** a *typed, proof-refined lowering* from a large formal layout language to a small hardware language.

---

# Direction 2 — **Effectful Async Layout IR (tokens, warp specialization, and schedule synthesis)**  
*(ASPLOS/PLDI, with an eye toward MICRO if you emphasize pipeline semantics)*

### 5.1 Gap (why the seeds are insufficient)

Seeds A/B/C are “spatial”: they reason about mappings. None give a first-class way to typecheck *asynchronous visibility*.

But PTX explicitly defines async copies as weakly ordered, completed via mbarrier/bulk-group. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
And Hopper’s tuning guidance frames TMA as enabling warp-specialized producer/consumer kernels (some warps issue transfers, others compute). ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  

So the missing link is not “more layout expressiveness,” but **a temporal/effect layer that composes with layout algebra**.

### 5.2 Apply the theory (Stage 1.5 lever)

**(A) Linear/effect types for async tokens.**  
Treat async operations as producing linear resources:

- `tma.load : (...) -> token<TMA, bytes, scope>`
- `mbarrier.wait : token -> ()`

**Invariant:** tokens must be consumed before dependent use; the IR enforces it.

**(B) Dataflow representation of warp specialization (“aref-style”).**  
Tawa introduces “asynchronous references” to express warp-level communication and automate warp specialization. ([arxiv.org](https://arxiv.org/abs/2510.14719?utm_source=openai))  
We can treat this as *exactly the missing temporal counterpart* to Seed layouts.

### 5.3 Define the mechanism (software artifact)

**Artifact:** *AsyncLayout IR* = (Layout algebra) ⊗ (Async effect system)

Concrete MLIR plan:
- Extend Triton GPU IR or add a dialect that:
  1. Represents layout transforms as pure ops (`layout.apply`, `layout.compose`).
  2. Represents movement as effectful ops:
     - `async.copy.tma(layout, coords) -> token`
     - `async.copy.cp_async(layout, ...) -> token`
     - `async.await(token)`
  3. Encodes warp roles explicitly:
     - `warp.partition {role = producer|consumer}`

Lowering:
- Emit PTX `cp.async.bulk{,.tensor}` and mbarrier sequences from effectful ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  
- Use NVGPU’s existing token types as the carrier (don’t reinvent token typing). ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  

Key novelty vs “just NVGPU tokens exist”:  
**the schedule is derived jointly with layout choice**, not after the fact. Layout decisions change:
- how many bytes per stage,
- whether you can use TMA at all,
- whether your shared-memory layout supports WGMMA descriptors,
- which barriers are needed.

### 5.4 Math-to-hardware mapping diagram (effects + layout)

```text
(Layout L)       (Async primitive P)         (Barrier protocol B)
    |                    |                          |
    └───────(compose as a single IR morphism)───────┘
                         |
                         v
            Effect-typed op:  Morphism = (L, P, B)
                         |
                         v
          PTX: cp.async.bulk(.tensor) + mbarrier ops
```

This is the formal way to say: **“layout selection is not separable from schedule.”** ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai))  

### 5.5 Feasibility + evaluation

**Feasibility:** moderate risk but bounded if you reuse:
- NVGPU dialect ops for TMA + barrier types ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai))  
- existing Triton warp specialization infrastructure / aref work (already active in toolchains) ([newreleases.io](https://newreleases.io/project/github/triton-lang/triton/release/v3.5.0))  

**Evaluation plan:**
- Compare:
  1. baseline Triton layouts + hand-managed staging
  2. auto-warp-specialization without layout-aware scheduling
  3. AsyncLayout IR (joint optimization)
- Metrics:
  - Speedup
  - Stall cycles on barriers / memory pipeline
  - Instruction count + registers
  - Compilation time impact

**Publishable claim:** an effect system that is *not merely for correctness* but also exposes a search space for performance (pipeline depth, specialization partition).

---

# Direction 3 — **Mixed-Radix / Piecewise Layout Semantics + Descriptor Pooling for Raggedness**  
*(ASPLOS systems angle; can also be pitched as “compiler/runtime co-design”)*

### 6.1 Gap (why the seeds are insufficient)

Seed A explicitly admits:
- **restriction to power-of-two shapes** and proposes “larger tensor + mask.” ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

Seed B can model richer relations, but it doesn’t provide a hardware-shaped lowering strategy. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

Seed C’s tractable layouts are stride/divisibility structured, which aligns with affine addressing, but not with dynamic ragged batches unless you add a dependent/range layer. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

Meanwhile, the ecosystem indicates “ragged + TMA” is important and nontrivial:
- Triton PR explicitly targets ragged TMAs and tests on H100/GB200. ([github.com](https://github.com/triton-lang/triton/pull/7792))  
- A paper proposes descriptor pools + runtime selection to eliminate padding in grouped GEMM. ([arxiv.org](https://arxiv.org/abs/2508.16584?utm_source=openai))  

So the gap is: **a formal semantics for piecewise (core+tail) layouts that the compiler can verify and lower to a finite set of descriptors.**

### 6.2 Apply the theory (Stage 1.5 lever)

**(A) Replace pure \(\mathbb{F}_2\) with mixed-radix integer lattices.**  
You want a representation that naturally supports non-\(2^n\) extents:
- represent shapes in mixed radix, not just bit-vectors
- represent layout transforms as integer matrices + modular constraints

**Canonicalization tool:** **Smith Normal Form (SNF)** (or related lattice normal forms) to:
- detect when a mapping is equivalent to “nice” strided form + bounded permutation
- compute a *structured decomposition* that can be lowered to descriptors + residual shuffle

**(B) Piecewise affine relations (ISL) as the semantic glue.**  
Instead of “lift and mask,” model:
\[
\text{Domain} = \text{CoreRect} \;\cup\; \text{TailSlices}
\]
and carry this as a first-class object in the IR, so codegen can produce:
- one TMA fast-path kernel for the core
- one/two tail kernels (or predicated epilogue)
- OR a descriptor pool + runtime selection

### 6.3 Define the mechanism (software artifact)

**Artifact:** *Piecewise Layout IR + Descriptor Pool Runtime*

#### Compiler side
1. Analyze dynamic shapes (seq lengths / group sizes) and choose a partition:
   - **core** region satisfies TMA constraints (inner dimension multiple-of-16B etc.) ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai))  
   - **tail** regions cover the ragged remainder
2. Precompute a **small descriptor family**:
   - either via “higher-dimensional descriptor tricks” (as Triton ragged TMA does) ([newreleases.io](https://newreleases.io/project/github/triton-lang/triton/release/v3.5.0))  
   - or via descriptor pools (as in TMA-Adaptive grouped GEMM) ([arxiv.org](https://arxiv.org/abs/2508.16584?utm_source=openai))  
3. Use ISL equivalence checks to verify that the union of piecewise regions equals the intended mapping (Seed B’s core strength). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  

#### Runtime side
- A lightweight selector chooses descriptor \(D_k\) based on runtime tail size (e.g., `seq_len mod tile`).

### 6.4 Math-to-hardware mapping diagram (core+tail + descriptor family)

```text
Ragged domain (dynamic):
  D = CoreRect  ∪  Tail_1 ∪ Tail_2 ∪ ...

Compile-time:
  synthesize TensorMapDescriptor(CoreRect)
  synthesize {TensorMapDescriptor(Tail_i)}_i  (small finite family)

Runtime:
  pick descriptor based on length bucket
  issue cp.async.bulk.tensor + mbarrier
```

This is the “semantic upgrade” over “pad to 1024 + mask”: you **bound the tail cost**, rather than paying it everywhere. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

### 6.5 Feasibility + evaluation

**Feasibility:** medium. Why?
- The ecosystem already shows ragged TMA mechanisms exist (so you’re not inventing PTX sequences). ([github.com](https://github.com/triton-lang/triton/pull/7792))  
- The novelty is the *formal piecewise semantics + synthesis strategy* rather than ad-hoc ragged checks.

**Evaluation:**
- Use TritonBench + ragged attention / grouped GEMM workloads. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  
- Metrics:
  - Speedup
  - **wasted bytes moved** (estimate from executed tiles vs useful)
  - tail overhead ratio
  - compilation time (descriptor family synthesis)
  - code size (multi-kernel / multi-descriptor)

---

## 7) Decision matrix (scoring the three directions)

Scale: **1 (low) … 5 (high)**.  
Implementation risk: **5 = highest risk**, **1 = lowest risk**.

| Direction | Theoretical Novelty | Hardware Relevance | Implementation Risk | Rationale |
|---|---:|---:|---:|---|
| **D1: Descriptor-Refined Layout Subtyping + Synthesis** | 4 | 5 | 3 | Strong PLDI story: refinement types + synthesis for a *real hardware type system* (TensorMap/WGMMA). Highly relevt to TMA constraints. NVGPU already provides ops, reducing risk, but synthesis/equivalence is nontrivial. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)) |
| **D2: Effectful Async Layout IR (tokens + schedule synthesis)** | 3 | 5 | 4 | Hardware relevance is maximal (TMA/cp.async + warp specialization). But novelty must be carefully differentiated from Tawa/aref; risk is higher due to scheduling + correctness + integration complexity. ([arxiv.org](https://arxiv.org/abs/2510.14719?utm_source=openai)) |
| **D3: Mixed-Radix/Piecewise Layouts + Descriptor Pooling** | 4 | 4 | 3 | Clear answer to “power-of-two tyranny.” Some prior art exists (descriptor pools, ragged TMA support), but a *formal piecewise semantics + compiler synthesis* is publishable. Risk moderate (runtime+compiler). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |

---

## 8) Recommended synthesis path (how I’d pitch for ASPLOS/PLDI)

If the goal is a sinherent thesis that reviewers see as inevitable:

> **Unify layout algebra with hardware-typed descriptors and effect-typed asynchrony.**  
> Layouts are not merely functions \(L\); they are **proof-refined, effectful morphisms** that either inhabit the TMA/WGMMA admissible sublanguage or carry an explicit, costed fallback.

A clean 3-paper arc (matching your deliverables) could be:
1. **PLDI:** D1 (refinement + synthesis)  
2. **ASPLOS:** D2 (effectful async scheduling integrated with layout types)  
3. **MICRO/ISCA adjunct:** D3 or the AMD instruction-indexed bank model as a cross-vendor validation story ([rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html?utm_source=openai))  

---

## One clarifying request (so I can align perfectly with your pipeline)

I did not receive the **Stage‑1.5 output attachment** in this chat. If you paste it (even just the headings / key frameworks), I will:
- replace my “Stage‑1.5 candidates” with your exact arsenal,en novelty claims against that set,
- and produce a sharper “Gap ↔ Theory ↔ Artifact” mapping with fewer degrees of freedom.

If you want, I can also produce a **prototype plan** (exact MLIR passes, where to hook in Triton’s pipeline, and what the minimal working subset is) for whichever direction you choose.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [Modeling Layout Abstractions Using Integer Set ttps://arxiv.org/html/2511.10374v1)
3. [Categorical Foundations for CuTe Layouts](https://arxiv.org/pdf/2601.05972v1)
4. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)
5. [PTX ISA :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html?utm_source=openai)
6. [1. NVIDIA Hopper Tuning Guide — Hopper Tuning Guide 13.1 documentation](https://do.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai)
7. [Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile Framework — ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html?utm_source=openai)
8. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)
9. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/12.9.0uda-driver-api/group__CUDA__TENSOR__MEMORY.html?utm_source=openai)
10. [\[FRONTEND\] Support for write-only ragged TMAs by apgoucher · Pull Request #7792 · triton-lang/triton · GitHub](https://github.com/triton-lang/triton/pull/7792)
11. [triton-lang/triton v3.5.0 on GitHub](https://newreleases.io/project/github/triton-lang/triton/release/v3.5.0)
12. [TMA-Adaptive FP8 Grouped GEMM: Eliminating Padding Requirements in Low-Precision Training and Inference on Hopper](https://arxiv.org/abs/2508.16584?utm_sou=openai)
13. [Understanding AMD GPU LDS and Bank Conflicts — Composable Kernel 1.1.0 Documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html?utm_source=openai)
14. ['nvgpu' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=openai)
15. [Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References](https://arxiv.org/abs/2510.14719?utm_source=openai)
16. [A Performance Model for Warp Specializati Kernels](https://arxiv.org/abs/2506.11209?utm_source=openai)
17. [1. Introduction — PTX ISA 8.8 documentation](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html?utm_source=openai)
18. [GitHub - meta-pytorch/tritonbench: Tritonbench is a collection of PyTorch custom operators with example inputs to measure their performance.](https://github.com/meta-pytorch/tritonbench?utm_source=openai)
