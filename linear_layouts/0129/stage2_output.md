## Stage 2 — Hardware‑Validated Scaffold → 3 Research Directions (ASPLOS/PLDI/ISCA/MICRO‑ready)

### 0) Access verification (S3) + what I actually retrieved

I verified fetch + parse of all three seed artifacts (and pulled the specific sections needed for Stage‑2 synthesis):

- **Seed A (Linear Layouts, arXiv HTML v3)**: accessible; extracted the formal definitions of *Distributed Layouts* and *Memory Layouts*, and the explicit statement of the *power‑of‑two restriction + “lift + mask” mitigation*. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **Seed B (ISL relations, arXiv HTML v1)**: accessible; extracted the claim that both CuTe layouts and Triton linear layouts can be translated to ISL integer relations; and that bit‑level swizzles/XOR are modeled via quasi‑affine/modular constraints. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- **Seed C (Categorical CuTe foundations, arXiv PDF v1, 174 pages)**: accessible; extracted the paper’s high‑level contract: a categorical framework over categories **Tuple** and **Nest**, focusing on a class of **tractable layouts**, with a reference implementation aligned with CUTLASS behavior (Jan 2026). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

For “hardware reality” validation, I also retrieved current vendor manuals/docs and *used them as ground truth* for the cliffs:

- **CUDA Driver API 12.9**: exact **CuTensorMap (TMA) descriptor constraints** (`cuTensorMapEncodeTiled`): rank bounds, alignment/stride congruences, `boxDim` bounds, and the finite swizzle enum family incl. ATOM modes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- **PTX ISA docs**: async copy grouping, the “no ordering within an async‑group” rule, and mbarrier completion semantics for async/bulk ops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  
- **Hopper tuning guide**: TMA’s architectural promise (single‑thread issue, no SM instruction pressure, warp specialization). ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  
- **Triton docs (tensor descriptors)**: `tl.make_tensor_descriptor` constraints and the fact that loads/stores *are backed by TMA hardware* when supported. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
- **AMD ROCm LDS docs/blog**: bank mapping (32 banks × 4B) + opcode‑indexed lane grouping for `ds_read_b128` vs `ds_write_b128` (the “instruction × lane‑phase” cliff). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html?utm_source=openai))  
- **Blackwell context**: CUDA/Blackwell family‑specific feature model (architecture/family targeting issues you must respect to unlock the newest paths). ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/?utm_source=openai))  
- **Evaluation harness**: Meta’s **TritonBench** as a practical perf suite for Triton operators (installation + running ops). ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

---

## 1) Stage‑1 bottlenecks — re‑validated against manuals (Deep Research Phase deliverable)

### 1.1 Hardware‑grounded “axioms that break”

| Stage‑1 Cliff / Broken Axiom | What the manuals actually guarantee | Why it’s a **cliff**, not a slope |
|---|---|---|
| **(A) “Any valid layout is realizable by TMA”** | `cuTensorMapEncodeTiled` imposes: `tensorRank ≤ 5` (and additional rank constraints when interleave ≠ NONE), **16B/32B alignment**, `globalStrides` multiples of 16/32, `boxDim[i] ≤ 256`, and **finite swizzle enums** (32B/64B/128B + ATOM variants). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | Crossing the boundary loses the *TMA engine*: Hopper explicitly sells that a **single thread issues large moves** and the SM can keep computing while the copy is in flight. Lose admissibility → fall back to SM‑issued loops. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai)) |
| **(B) “Layout choice separable from schedule”** | PTX: async ops have **no ordering within a committed group**, and visibility is tied to `wait_group`/`wait_all` or **mbarrier completion**. Misplacing waits produces stalls or hazards; “program order” is not the model. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai)) | This is correctness *and* performance: late wait → use‑before‑ready bug; early wait → pipeline collapse. |
| **(C) “Power‑of‑2 restriction + masking is fine”** | Seed A itself states: *primary limitation is restriction to power‑of‑two shapes*, mitigated by “define larger tensors and mask out‑of‑boundary elements.” ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | For ragged / non‑2^n, masking induces **tail effect** and damages regularity required by TMA boxes and alignment/`boxDim` constraints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) |
| **(D) “Bank conflict freedom is a property of layout alone”** | AMD documents show: LDS is banked; and conflict rules depend on opcode lane grouping (`ds_write_b128` vs `ds_read_b128` use different lane partitions). ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html?utm_source=openai)) | A layout can be “proved conflict‑free” under the wrong predicate yet still conflict in silicon. |

---

## 2) Scaffold & Plan — Stage‑1.5 theories mapped to *validated* bottlenecks

### 2.1 “Obligation → Theory → Compiler artifact” map

| Compiler Obligation (from Stage‑1 cliffs) | Stage‑1.5 Theory Lever | Concrete artifact (C1‑implementable) |
|---|---|---|
| **O1: Admissibility** — `Layout` ⇒ `Tma/Wgmma fast path` | **Refinement types / proof‑carrying layouts** + **SMT/CP synthesis** | MLIR/Triton: a `!layout.tma_desc` refined type + `layout→descriptor` synthesis pass that either produces a witness descriptor or fails early with a costed fallback plan |
| **O2: Temporality** — mapping correctness ⇒ async protocol correctness + overlap optimal | **Effect systems / linear tokens** + **memory‑model aware validation** | MLIR dialect: tokenized `tma.async_load → token`; `await(token)`; verifier ensures dominance and correct mbarrier usage; scheduler sinks waits |
| **O3: Non‑2^n / ragged** — avoid padding tail effect | **Piecewise Presburger domains** + **mixed‑radix modules** | IR extension: `layout.piecewise{core,tail}` + runtime bucketing; compile core with TMA (boxes), tail with predicated/vector path |
| **O4: Bank semantics** — conflict‑free under opcode lane‑phasing | **Abstract interpretation over modular domains** + constraint‑based XOR swizzle synthesis | Analysis pass `bank_conflict(layout, opcode, vecwidth)`; integrate into layout search / extraction cost |
| **O5: Compile‑time budgets** | **Equality saturation (eqsat)** + staged “generate→verify” | Use eqsat to normalize/rewrite layouts into descriptor‑friendly normal forms; invoke solver only on a small candidate set |

Key novelty pivot: **make the fast path a type**, not a late lowering accident.

---

# 3) Three Research Directions (the “Proposal = Gap + Theory → Artifact” synthesis)

Below are three *distinct* directions. Each is designed to be prototype‑able in ~3–4 months inside Triton/MLIR, and evaluable on TritonBench. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

---

## Direction 1 — **Descriptor‑Certified Layouts**: Proof‑Carrying TMA/WGMMA Lowering from \(\mathbb{F}_2\) Layout Specs

### (1) Define the Gap (validated)

Seed A’s layout language is intentionally broad:

- **Distributed layout**: surjective linear map with “at most one 1 per column” and no repeated non‑zero columns (permutation matrix with zero columns). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- **Memory layout**: invertible linear map with 1–2 non‑zero bits per column (swizzle envelope). ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

But **TMA** is a *descriptor‑driven* engine with a hard admissibility fragment:

- `tensorRank ≤ 5`, alignment and stride congruences, `boxDim ≤ 256`, and **finite enumerated swizzles** (including ATOM variants). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- Hopper’s performance claim depends on staying in this fragment: **single thread issues large data movement to TMA**, enabling warp specialization and avoiding SM instruction pressure. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  
- Triton itself exposes this as `tl.make_tensor_descriptor`, which requires 16B alignment + stride properties and is TMA‑backed when supported. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  

**Therefore:** Seed A can generate layouts that are *mathematically bijective* but **not encodable as a tensor descriptor**, forcing a fallback path and losing the intended “warp specialization + offloaded copy” advantage. This is exactly Stage‑1’s *descriptor admissibility cliff*, now grounded in driver constraints.

### (2) Apply the Theory (Stage‑1.5 lever)

**Refinement types / proof‑carrying layouts** + **constraint‑based synthesis** + **equality saturation as a normalizer**.

Concretely:

- Treat Seed‑A layouts as a *specification language*: \(L : (\mathbb{Z}_2)^n \to (\mathbb{Z}_2)^m\).  
- Introduce a refinement judgment:

\[
\Gamma \vdash L : \texttt{Layout}
\quad\wedge\quad
\Gamma \vdash \exists D.\; \textsf{EncodesTMA}(D, L)
\Rightarrow
\Gamma \vdash (L,D) : \texttt{TmaLayout}
\]

where `EncodesTMA` is **exactly** the CUDA Driver API constraints + a semantic equivalence check (or restricted equivalence template). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  

We use **eqsat** (MLIR equality saturation dialect) as the practical engine to rewrite layouts into “descriptor‑friendly” normal forms before attempting synthesis. ([arxiv.org](https://arxiv.org/abs/2505.09363?utm_source=openai))  

### (3) Define the Mechanism (software artifact)

**Artifact:** a Triton/MLIR pass pipeline that turns general linear layouts into **witnessed TMA descriptors** (or fails early with a costed plan).

#### Mechanism overview
1. **Normalize** Seed‑A layout expressions with eqsat:
   - push XOR mixing into low‑order bits when possible
   - factor permutations from swizzles
   - canonicalize reshape/split/join compositions  
   (Goal: reduce the space before solver, and expose affine‑ish stride structure.)

2. **Descriptor synthesis (witness generation)**:
   - Search a restricted template family matching `cuTensorMapEncodeTiled`:
     - choose rank mapping, block shapes, `boxDim`, `elementStrides`, `interleave`, `swizzle`
     - enforce alignment / stride congruences and `boxDim` bounds  
     ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
   - Output either:
     - **SAT witness**: a `tma_desc` object (or a `tl.make_tensor_descriptor` form) ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
     - **UNSAT**: mark the layout as “TMA‑inadmissible” and force alternative pipelines

3. **Proof‑carrying IR**:
   - Attach the witness as an IR attribute:
     - `tma.desc{rank, strides, boxDim, swizzle, interleave, oobFill}`
   - Later passes are *checkers*, not searchers (“generate→verify” staging).

#### MLIR/Triton integration sketch (C1)
- Extend TritonGPU lowering with:
  - `triton_gpu.layout_spec` (general)
  - `triton_gpu.tma_desc` (refined)
  - `triton_gpu.tma_load/store` (only legal if desc present)
- Lower `tma_load/store` to the descriptor‑backed path (`cp.async.bulk.tensor...` in PTX terms) when possible, otherwise to cp.async or ld/st fallback. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.1.0/parallel-thread-execution/index.html?utm_source=openai))  

#### Math‑to‑Hardware diagram
```text
Seed-A Layout L  (F2-linear, XOR mixing)
        |
        |  eqsat normalization  (rewrite toward descriptor-friendly form)
        v
   Normal-form L'
        |
        |  synthesize witness D s.t. EncodesTMA(D, L')
        |     (alignment/stride/boxDim/swizzle enum constraints)
        v
   (L', D) : TmaLayout     OR     UNSAT -> fallback plan
        |
        v
 PTX/Backend:
   - TMA path: descriptor-backed bulk tensor ops + barriers
   - else: SM-issued cp.async / ld/st loops
```

### Evaluation plan (C2, on TritonBench)
Use **meta‑pytorch/tritonbench** as the primary harness (it already wraps PyTorch + Triton and runs per‑op benchmarks). ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

**Metrics:**
- **Speedup** vs baseline Triton (and vs “linear layouts without descriptor refinement”).
- **Fast‑path hit rate**: % of candidate layouts that synthesize a valid TMA descriptor.
- **Compilation time**: total compile time + time spent in eqsat + time spent in solver.
- **Code size**: PTX size / SASS instruction count proxy (or `nvdisasm` count).
- **Bank conflict rate**: Nsight Compute shared memory bank conflict metrics on kernels that stage through SMEM.
- **TMA utilization proxy**: count of descriptor‑backed loads/stores (or emitted descriptor ops).

### Novelty vs closest known work
- **Hexcute (CGO’26)** already frames layout synthesis as constraint programming with type‑inference flavor and targets dataflow/pipelining; so “solver for layouts” is no longer novel by itself. ([2026.cgo.org](https://2026.cgo.org/details/cgo-2026-papers/12/Hexcute-A-Compiler-Framework-for-Automating-Layout-Synthesis-in-GPU-Programs?utm_source=openai))  
- Our novelty is **proof‑carrying *hardware‑descriptor* subtyping** *inside* a Seed‑A‑style \(\mathbb{F}_2\) layout algebra + eqsat‑based normalization to make synthesis tractable—explicitly designed to eliminate *hardware path cliffs* (TMA/WGMMA admissibility as a type). This is a *PLDI/ASPLOS* pitch.

---

## Direction 2 — **Effect‑Typed Async Layout IR**: Making `cp.async`/TMA/mbarrier Semantics First‑Class (Spatial → Temporal)

### (1) Define the Gap (validated)

Stage‑1’s friction “spatial vs temporal” is directly confirmed by PTX:

- async groups complete in commit order, but **no ordering between operations within an async‑group**. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  
- bulk async ops require explicit completion tracking (async‑group or **mbarrier**), and “visibility” is tied to `wait_group`/`wait_all` or mbarrier completion. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  

And Hopper’s advertised performance model relies on **overlap + warp specialization**: one warp issues TMA while others compute. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  

Seed A/B/C give you the “where” mapping; they do not give you a typed story for the “when” protocol. That’s why “layout selection separable from schedule” fails on silicon.

### (2) Apply the Theory

**Effect systems / linear capabilities (token discipline)** + **translation validation against the PTX memory model**.

Key idea: an async copy is not just an address mapping; it is a *protocol* producing a consumable capability:

\[
\texttt{tma\_load} : (\texttt{Desc}, \texttt{Coords}) \to \texttt{Token}
\quad;\quad
\texttt{await} : \texttt{Token} \to ()
\]

and *smem consumers* are illegal unless dominated by `await`.

### (3) Define the Mechanism

**Artifact:** an MLIR dialect + scheduler that turns “implicit staging” into a typed async DAG.

#### IR surface (C1‑feasible)
- `nvgpu.tma.async_load %desc, %coords -> !nvgpu.token<region, scope>`
- `nvgpu.token.await %t`
- `nvgpu.mbarrier.init/arrive/test_wait` (typed phases)

This is explicitly designed to match PTX’s requirement that completion must be observed via waits/mbarriers, and that group operations are weakly ordered. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai))  

#### Compiler passes
1. **Async extraction**: detect load→compute pipelines (e.g., shared‑memory staged matmul) and lift into tokenized ops.
2. **Dependency legalization**: insert/verify `await` so every consumer is ordered.
3. **Overlap scheduler**:
   - sink `await` to last‑use
   - hoist `async_load` as early as dependencies permit
   - optionally partition warps into “copy warps” and “compute warps,” consistent with Hopper’s intended usage. ([docs.nvidia.com](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai))  
4. **Translation validation**:
   - after lowering to PTX, validate that each token corresponds to an actual wait edge (`wait_group` or `mbarrier.test_wait`), and no SMEM access occurs before completion is observed.

#### Math‑to‑Hardware diagram
```text
Layout L (spatial)                 PTX reality (temporal)
-----------------                  ---------------------
addr = L(i)             +          async ops are weakly ordered
                                    completion via wait/mbarrier

Effect-typed IR:
  t = tma.async_load(desc, coords) : token
  ... compute independent work ...
  await(t)  // must dominate smem read
  use(smem)
```

### Evaluation plan (C2)
Run on TritonBench ops that already stress pipelining (matmul/attention‑like kernels) and compare:

- **Speedup / throughput** (baseline vs tokenized scheduling).
- **Overlap metrics** (stall reasons around waits).
- **Compilation time** (extra passes).
- **Code size** (tokenization can reduce redundant barriers or increase metadata; measure both).
- **Bank conflict rate** (ensure scheduling doesn’t force worse SMEM access patterns).

TritonBench provides a standard harness for these operator benchmarks. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

### Novelty / venue fit
- **ASPLOS/PLDI**: This is a type/effect system that makes weakly‑ordered async semantics *explicitly checkable* and *optimizable*, aligned with PTX’s documented semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mbar&utm_source=openai))  
- Distinct from Direction 1: Direction 1 is “can we *encode* a layout as a descriptor?”; Direction 2 is “can we *schedule* the descriptor path correctly and profitably?”

---

## Direction 3 — **Piecewise Mixed‑Radix Layouts for Ragged / Non‑\(2^n\) Shapes**: Core+Tail Compilation that Preserves TMA Regularity

### (1) Define the Gap (validated)

Seed A explicitly states the limitation:

- “primary limitation … restriction to power‑of‑two shapes,” mitigated by “define larger tensors and mask out‑of‑boundary elements.” ([arxiv.org](https://arxiv.org/html/2505.23819v3))  

That’s a semantic patch, but it’s performance‑toxic for ragged/non‑\(2^n\) workloads because:

- TMA descriptors require **box‑shaped traversals with strict bounds and congruences** (`boxDim ≤ 256`, `boxDim[0]*elemSize` multiple of 16 when interleave NONE, strides multiple of 16/32, etc.). ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- Triton’s own descriptor API bakes in the same reality: base must be 16B aligned, leading strides multiples of 16B, last dim contiguous. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  

So “lift to next power of two + mask” tends to (i) waste work (tail effect) and (ii) either break descriptor preconditions or force you into non‑TMA code paths for tails.

### (2) Apply the Theory

**Mixed‑radix modules / finite abelian groups** + **piecewise Presburger domains**.

Operationally: replace a single global \(\mathbb{F}_2\) isomorphism with a **piecewise layout**:

\[
\texttt{Layout} \equiv (\texttt{Layout}_{core} \;\uplus\; \texttt{Layout}_{tail})
\]

where `core` is chosen to be **TMA‑admissible**, and `tail` is handled by a separate (predicated) lowering.

### (3) Define the Mechanism

**Artifact:** a “core+tail” layout compiler pass that *constructs a TMA‑friendly core region* and isolates tails so they don’t poison the whole kernel.

#### Mechanism steps
1. **Domain decomposition**
   - Given runtime sizes (or symbolic bounds), compute:
     - `D_core`: maximal rectangular region coverable by TMA tiles (respecting `boxDim`/alignment/stride constraints)
     - `D_tail`: remainder (small)
2. **Core path**
   - Emit `tl.make_tensor_descriptor` for `D_core` and run TMA loads/stores for that region. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai))  
3. **Tail path**
   - Emit a predicated epilogue (or a second kernel variant) for `D_tail` with vector/scalar loads.
4. **Optional mixed‑radix factorization**
   - For dimension \(N\) not power‑of‑two:
     - factor \(N = 2^k \cdot m\)
     - apply Seed‑A \(\mathbb{F}_2\) swizzles only on the \(2^k\) factor (cheap XOR)
     - keep the odd factor affine/stride‑based (avoid costly general modulo)  
   This is the “math survives silicon” bridge: preserve low‑bit swizzle benefits without forcing global padding.

#### Math‑to‑Hardware diagram
```text
Ragged / non-2^n domain D
        |
        |  piecewise decomposition (Presburger / heuristic tiling)
        v
D = D_core ⊎ D_tail
   |          |
   |          +--> predicated ld/st (small, localized divergence)
   v
TMA-admissible core tiles
   |
   +--> encode descriptor (alignment/stride/boxDim/swizzle)
   |
   v
cp.async.bulk.tensor / descriptor-backed loads
```

### Evaluation plan (C2)
Use TritonBench operators with size variability + add ragged‑shape variants:

- **Speedup** vs “pad+mask everywhere” baseline.
- **Tail efficiency**: ratio of useful elements vs loaded elements (can be derived from bytes moved vs ideal).
- **TMA coverage**: fraction of work in `D_core` executed via descriptor path.
- **Compilation time**: overhead of decomposition + extra kernels.
- **Code size**: single kernel with epilogue vs two kernels (trade‑off).
- **Bank conflict rate**: Nsight Compute metrics for SMEM accesses in the core path.

TritonBench provides the benchmarking harness to run these systematically. ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

### Novelty / relation to seeds
- Seed A acknowledges masking as mitigation but does not provide a **semantic core/tail split** that preserves fast‑path regularity. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Seed B (ISL relations) makes piecewise domains representable, but it is positioned as a correctness/unification formalism, not a hardware‑typed lowering strategy. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Seed C’s “tractable layouts” categorical story provides a clean algebra for composability and a tested implementation aligned with CUTLASS behavior, but it does not encode TMA descriptor constraints; core/tail introduces the missing hardware refinement layer. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  

---

# 4) Decision Matrix (3 directions scored)

Scoring scale (explicit):
- **Theoretical Novelty**: 1 = incremental, 5 = genuinely new math→systems application
- **Hardware Relevance**: 1 = marginal, 5 = directly unlocks H100/Blackwell fast paths (TMA/warp specialization)
- **Implementation Feasibility (inverse risk)**: 1 = very risky >4 months, 5 = feasible in 3–4 months

| Direction | Theoretical Novelty | Hardware Relevance | Feasibility (3–4 months) | Why |
|---|---:|---:|---:|---|
| **D1: Descriptor‑Certified Layouts** | **4** | **5** | **3** | Strong novelty in *proof‑carrying descriptor subtyping* + eqsat normalization; directly targets TMA constraints (driver‑API‑grounded). Moderate risk: solver+IR plumbing. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) |
| **D2: Effect‑Typed Async Layout IR** | **5** | **5** | **2** | Most “PLDI‑ish” contribution: token/effect system matching PTX weak ordering + mbarrier semantics; but correctness and lowering are subtle; higher prototype risk. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mbar&utm_source=openai)) |
| **D3: Piecewise Mixed‑Radix Core+Tail** | **4** | **4** | **4** | Clear attack on Seed‑A’s explicit power‑of‑two limitation; practical and implementable via epilogue splitting and descriptor‑friendly core tiles. Needs careful heuristics but less “new IR machinery” than D2. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |

---

## If you want a single “top‑tier” unified proposal
A very strong ASPLOS/PLDI submission is **D1 + D2** as the core thesis (“fast path is a type; time is a token”), with **D3** as the dynamic‑shape case study that demonstrates the system matters for real ragged inference.

If you tell me which operator family you want as the headline evaluation (e.g., FlashAttention‑style attention, persistent matmul, MoE routing), I’ll tighten the 3 directions into a single narrative with:
- an MLIR dialect surface (minimal ops/attrs),
- a staged lowering pipeline,
- and a concrete TritonBench experiment matrix (exact ops, sizes, Nsight metrics, compile‑time budgets).

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [Modeling Layout Abstractions Using Integer Set Relations](https://arxiv.org/html/2511.10374v1)
3. [Categorical Foundations for CuTe Layouts](https://arxiv.org/pdf/2601.05972v1)
4. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
5. [1. Introduction — PTX ISA 8.7 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=openai)
6. [1. NVIDIA Hopper Tuning Guide — Hopper Tuning Guide 13.1 documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html?utm_source=openai)
7. [triton.language.make\_tensor\_descriptor — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html?utm_source=openai)
8. [Hardware implementation — HIP 7.2.0 Documentation](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html?utm_source=openai)
IDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features | NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/?utm_source=openai)
10. [GitHub - meta-pytorch/tritonbench: Tritonbench is a collection of PyTorch custom operators with example inputs to measure their performance.](https://github.com/meta-pytorch/tritonbench?utm_source=openai)
11. [eqsat: An Equality Saturation Dialect for Non-destructive Rewriting](https://arxiv.org/abs/2505.09363?utm_source=openai)
12. [Parallel Thread Execution 8.1](https://docs.nvidia.com/cuda/archive/12.1.0/parallel-thread-execution/index.html?utm_source=openai)
13. [Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU Programs (CGO 2026 - Main Conference) - CGO 2026](https://2026.cgo.org/details/cgo-2026-papers/12/Hexcute-A-Compiler-Framework-for-Automating-Layout-Synthesis-in-GPU-Programs?utm_source=openai)
14. [1. Introduction — PTX ISA 9.documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mbar&utm_source=openai)
