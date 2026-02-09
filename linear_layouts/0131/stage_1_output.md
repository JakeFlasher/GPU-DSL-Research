## Theoretical Verdict (summary)

**(A) Seed Paper Claim.**  
Across the three seed papers, “layout” is treated as a *static mathematical object* (a map / relation / morphism) enabling composition, inversion, and correctness reasoning for *coordinate ↔ index* transformations—either as a linear map over $$\mathbb{F}_2$$ (Paper 1), an ISL integer-set relation (Paper 2), or categorical morphisms producing CuTe layouts (Paper 3). ([arxiv.org](https://arxiv.org/html/2505.23819v3))

**(B) Hardware Manual Claim.**  
Modern NVIDIA GPU correctness requires *more than layout*: (i) **encode-time legality** for tensor-map/TMA descriptors (alignment, rank limits, stride and box constraints, finite swizzle/interleave modes), (ii) **instruction-form legality** for `cp.async.bulk.tensor` including per-type exact byte quanta and architecture gating, (iii) **warpgroup-wide uniformity + mandatory ordering** for `wgmma.*` (else UB), (iv) **synchronization object invariants + token provenance** for `mbarrier.*` (else UB), and (v) **dynamic resource allocation** for Blackwell Tensor Memory (TMEM) including power-of-two allocation rules and warp/warpgroup access restrictions. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))

**(C) Theoretical Deficit (Expressivity Gap, compiler-facing).**  
The seed math cannot *represent* (let alone guarantee) key hardware states: **(1)** descriptor legality as a mixed system of bounded inequalities + congruences + finite enums, **(2)** async completion + weak-memory effects, **(3)** warpgroup-uniform control-flow predicates, and **(4)** dynamic allocation/lifetime of on-chip resources. Concretely: you need a compiler artifact stack that includes a *legality solver pass* (bitvector + Presburger/congruence), an *effectful IR* for async ops/barriers (tokens as SSA values), a *SIMT-uniformity analysis* for warpgroup ops, and a *resource/liveness system* for TMEM alloc/dealloc.

---

## Step 0 — Access Check (Open & Cite)

**(A) Seed Paper Claim.**  
The audit requires reading the exact arXiv versions listed in the prompt.

**(B) Hardware Manual Claim.**  
N/A.

**(C) Theoretical Deficit.**  
N/A.

### Verified access (opened successfully)

| Artifact | Access verified? | Evidence |
|---|---:|---|
| Paper 1: arXiv HTML `2505.23819v3` | Yes | Opened and parsed. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) |
| Paper 2: arXiv HTML `2511.10374v1` | Yes | Opened and parsed. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) |
| Paper 3: arXiv PDF `2601.05972v1` | Yes | Opened and parsed (174 pages). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) |

---

## Step 1 — Seed Truth (Axioms)

I extract “axioms” as the *primitive mathematical commitments* the papers build correctness on, and I annotate the compiler artifact each axiom corresponds to.

### Paper 1 (Linear Layouts; $$\mathbb{F}_2$$ matrices)

**(A) Seed Paper Claim.**  
1) **Layout = linear map over $$\mathbb{F}_2$$.**  
A “Linear Layout” is defined as a linear map between labeled vector spaces over $$\mathbb{F}_2$$. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
**Compiler artifact mapping:** a layout is representable as a *binary matrix* object in the compiler IR (or a derived basis-vector representation), enabling algebraic composition.

2) **Composition / product / inverses are matrix constructions.**  
Composition is matrix multiplication over $$\mathbb{F}_2$$; products are block-diagonal combinations; inverses/right-inverses exist under algebraic conditions. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
**Compiler artifact mapping:** layout conversion becomes “compute matrix (or basis) + generate code” rather than case-by-case lowering.

3) **Arithmetic model is bitwise: XOR is addition, AND is multiplication.**  
The paper explicitly grounds $$\mathbb{F}_2$$ arithmetic in bitwise XOR/AND. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
**Compiler artifact mapping:** codegen can target `xor`, `and`, shifts, permutes, shuffles.

4) **Claimed coverage: “distributed layouts” and “memory layouts” are linear layouts.**  
They assert completeness-style statements such as “Blocked layouts are linear layouts,” “MMA layouts are linear layouts,” and “Every memory layout is a linear layout.” ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
**Compiler artifact mapping:** layout engine can treat many layout families uniformly (compose/invert).

### Paper 2 (ISL integer set relations unifying CuTe + Linear Layouts)

**(A) Seed Paper Claim.**  
1) **CuTe layouts = shape/stride mapping (coordinate → 1D index).**  
CuTe layouts map $$d$$-D coordinates to a 1-D index via shape/stride tuples (stride-based calculation). ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
**Compiler artifact mapping:** a CuTe `Layout<Shape,Stride>` lowers to an affine-ish address/index computation graph.

2) **CuTe swizzle = explicit bit-manipulation bijection.**  
They define a swizzle as a bijection on a bounded interval using XOR/AND/shift. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
**Compiler artifact mapping:** swizzle can lower to bitwise IR ops; correctness can be checked as bijectivity/invertibility on the interval.

3) **Triton linear layouts = binary vector space maps; dimension sizes must be powers of two.**  
They explicitly state the binary-vector-space view and note that dimension sizes in coordinate/index spaces must be powers of two. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
**Compiler artifact mapping:** layout legality includes “power-of-two sizing” constraints before you can even interpret the map as $$\mathbb{F}_2$$-linear.

4) **ISL modeling axiom: layouts/swizzles become integer set relations (conjunctions of affine constraints).**  
ISL relations are defined via conjunctions of affine constraints; ISL provides composition/inverse/etc. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
**Compiler artifact mapping:** “layout reasoning” becomes polyhedral-style reasoning (domain/range, composition, inversion) at compile time.

### Paper 3 (Categorical CuTe foundations: categories Tuple & Nest)

**(A) Seed Paper Claim.**  
1) **Layouts are functions induced by shape/stride tuples (flat layouts).**  
They define a flat layout function (in the text) of the form  
$$\Phi_L(x)=x_1 d_1+\cdots + x_m d_m$$  
with $$x_j$$ computed via mixed-radix decomposition (floor/div/mod by earlier shape components). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
**Compiler artifact mapping:** a flat layout is a specific integer arithmetic program (div/mod + linear combination).

2) **Two categories (Tuple, Nest); morphisms give rise to layouts; operations correspond to layout algebra.**  
Abstract states they define categories Tuple and Nest whose morphisms give rise to layouts and prove compatibility with layout operations. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))  
**Compiler artifact mapping:** layout algebra operations (compose/product/division) correspond to categorical constructions; could justify canonicalization/normalization passes.

---

## Step 2 — Hardware Truth (Top 5 Constraints)

Below are the “top 5” constraints because they create *hard correctness cliffs* (UB / invalid encoding / impossible scheduling) and are repeatedly hit by compiler lowering for modern kernels.

### 1) TMA / `CUtensorMap` descriptor legality (encode-time)

**(B) Hardware Manual Claim.**  
Tensor map objects are opaque and must be created/modified via CUDA APIs; encode-time requirements include (non-exhaustive): `tensorMap` 64B alignment; rank constraints (tiled rank ≤ 5; im2col rank 3–5); `globalAddress` 16B aligned (32B in specific modes/types); `globalDim[i]` non-zero and ≤ $$2^{32}$$ (plus packed-type congruences); `globalStrides[i]` multiple of 16 and < $$2^{40}$$ (tightened to 32 in some cases); `boxDim[i]` non-zero and ≤ 256 (plus 16B multiple constraints on inner dimension in some modes); `elementStrides[i]` 1..8; finite swizzle/interleave modes with additional restrictions (e.g., inner-dimension ≤ swizzle size; datatype-specific allowed swizzles). ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
PTX additionally specifies tensor-map objects are 1024 bits and `tensormap.replace` is a weak memory op over the whole object. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

### 2) `cp.async.bulk.tensor` instruction legality (PTX-level)

**(B) Hardware Manual Claim.**  
`cp.async.bulk.tensor` is a non-blocking async tensor copy that takes a tensor map plus coordinates; it has `.dim ∈ {1d..5d}`, supports different dst/src state spaces, uses an `mbarrier` completion mechanism for global→shared, has target ISA gating (requires `sm_90+`, with additional qualifiers gated by later targets), and has UB if (for example) `.shared::cta` destination address is not in executing CTA shared memory. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
For sub-byte / packed types, PTX imposes exact byte-quantized rules like Box-Size[0] must be exactly 64B/96B (type-dependent), Tensor-Size[0] multiple-of quanta, per-dimension stride alignment, coordinate congruences, and finite supported swizzle sets (again arch/type dependent). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
PTX also states the copy is treated as a weak memory operation and the complete-tx on the mbarrier has `.release` semantics at `.cluster` scope. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

### 3) `wgmma.*` legality: warpgroup-uniform control + mandatory ordering fences

**(B) Hardware Manual Claim.**  
The mandatory `.aligned` qualifier means *all threads in the warpgroup must execute the same `wgmma.mma_async`*; if used under a condition that is not warpgroup-uniform, behavior is undefined. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
`wgmma.fence` must be issued by all warps of the warpgroup at specific points (e.g., before first `wgmma.mma_async`), otherwise behavior is undefined. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

### 4) `mbarrier.*` object validity + token provenance

**(B) Hardware Manual Claim.**  
Performing any `mbarrier` op (except `mbarrier.init`) on a location that does not contain a valid mbarrier object is undefined behaviour; plus state-space/address-window constraints apply. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
Some wait forms require the `state` operand be the result of a prior arrive variant; otherwise behavior is undefined. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

### 5) Blackwell Tensor Memory (TMEM) is dynamic, finite, and has access restrictions

**(B) Hardware Manual Claim.**  
On `sm_100a`/`sm_100f`, Tensor Memory is a 2D matrix of **512 columns × 128 rows per CTA**, each cell 32 bits. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
Tensor Memory is **dynamically allocated**: allocated by a single warp; allocation unit is 32 columns; number of columns allocated must be a power of two; access restrictions partition lanes by warp within warpgroup; `tcgen05.ld` requires all threads in the warp use the same `taddr` else UB; `tcgen05.shift` requires lane alignment to 32. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

---

## Step 3 — Expressivity Gap (Constraint-by-Constraint)

I give for each constraint: (A) what the seed axioms *say*, (B) what hardware *requires*, and (C) the explicit “Expressivity Gap” + a compiler artifact mapping (S4).

---

### Constraint 1: TMA / `CUtensorMap` descriptor legality (encode-time)

**(A) Seed Paper Claim.**  
- Paper 1: a layout is a $$\mathbb{F}_2$$-linear map (binary matrix) enabling composition/inversion; it treats “memory layouts” as invertible linear layouts. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Paper 2: CuTe layouts/swizzles and Triton linear layouts can be represented as ISL integer set relations; swizzle is a bijection definable via XOR/AND/shift. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Paper 3: layouts arise from algebraic/categorical constructions; the core object is a layout function induced by shape/stride structure, and operations like composition/product/division are studied abstractly. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))

**(B) Hardware Manual Claim.**  
A legal TMA tensor map must satisfy a *large conjunction* of encode-time requirements: 64B alignment for the descriptor object; rank bounds (≤5; im2col ≥3); global address alignment (16B/32B depending on interleave/type); bounds on `globalDim` (≤ $$2^{32}$$), bounds and congruences for `globalStrides` (multiple-of 16/32, < $$2^{40}$$), bounds on `boxDim` (≤256) plus inner-dimension multiple-of constraints; `elementStrides` constraints; plus **finite** allowed swizzle/interleave modes with datatype-specific restrictions and inner-dimension ≤ swizzle-size rules. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
The tensor-map object is 1024 bits in PTX, and mutation uses `tensormap.replace` which is a weak memory operation over the whole 1024-bit object. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**(C) Theoretical Deficit.**  
**Expressivity Gap (explicit):** the seed formalisms model a **layout mapping** $$f : \text{Coord} \to \text{Index}$$ (or its inverse), but **TMA legality is a predicate over a *descriptor state***  
$$\text{LegalTMA}(\texttt{tensorRank}, \texttt{globalDim}, \texttt{globalStrides}, \texttt{boxDim}, \texttt{elementStrides}, \texttt{interleave}, \texttt{swizzle}, \texttt{dtype}, \texttt{alignments}, \ldots).$$  
This predicate mixes: bounded inequalities (e.g., ≤ $$2^{32}$$, ≤256), congruences (multiple-of 16/32/128), and *finite enumerations* (swizzle/interleave sets). None of the seed axioms make this legality predicate first-class: they can describe **how** indices map, but not **whether a tensor-map encoding exists** for that mapping and datatype/mode.

**Proof sketch of insufficiency:**  
- In Paper 1’s axiom set, “layout = linear map over $$\mathbb{F}_2$$” is a statement about algebraic structure of $$f$$, not about satisfiability of encode-time constraints on a *descriptor object* or its fields. A binary matrix does not carry (nor constrain) `tensorRank≤5`, descriptor alignment, or stride byte-multiplicity requirements. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- In Paper 2’s axiom set, ISL relations can represent affine constraints, but the paper’s *semantic target* is “correctness of layout transformations,” not “existence of a legal `CUtensorMap` encoding with vendor enums + datatype restrictions.” The hardware requires a domain-specific finite-mode legality filter and bit-congruence checks keyed by `CUtensorMapDataType`. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- In Paper 3’s categorical framework, the objects/morphisms capture layout algebra; the encode-time constraints (alignment, rank caps, swizzle-mode enums) are *external* to the categorical structure. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))

**Concrete compiler artifact mapping (what you actually need):**  
- A **TMA legality checker + synthesizer pass** (e.g., `TmaDescriptorLegalize`) that takes `(tensor shape/strides, tile shape, dtype, chosen interleave/swizzle)` and runs a satisfiability check in a combined theory:
  - bitvector constraints for alignments / low-bit zeros,  
  - Presburger inequalities for bounds,  
  - finite-domain constraints for enums,  
  - plus special-case rules for packed types. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
- The output of this pass must be either:
  1) a proven-legal `CUtensorMapEncode*` parameterization, or  
  2) a compile-time failure requiring alternative tiling/layout selection.

---

### Constraint 2: `cp.async.bulk.tensor` legality (instruction admissibility + exact byte quanta)

**(A) Seed Paper Claim.**  
- Paper 1: swizzles and memory layouts can be expressed as $$\mathbb{F}_2$$-linear maps, and the compiler can search for “optimal swizzling” and generate code for layout conversion. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Paper 2: swizzles are bijections definable via XOR/AND/shift; layouts are modeled as relations suitable for composition and inversion. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Paper 3: layout operations are algebraic; composition/division/product correctness is described abstractly. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))

**(B) Hardware Manual Claim.**  
`cp.async.bulk.tensor` is a *non-blocking async tensor copy* whose form and legality depend on `.dim`, `.dst/.src` spaces, `.load_mode`, completion mechanism (mbarrier), and target architecture; it has UB for invalid destination address-space cases. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
For sub-byte/packed types, PTX imposes hard “byte-quantized” equalities and congruences (e.g., Box-Size[0] must be exactly 64B or 96B, Tensor-Size[0] multiples, global address 16B aligned, tensor stride 16B aligned, first coordinate multiple-of 64, supported swizzle set finite and arch-dependent). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
Additionally, PTX specifies weak-memory semantics + release semantics for the completion operation. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**(C) Theoretical Deficit.**  
**Expressivity Gap (explicit):** seed layout math treats “tensor copy” as (at most) a mapping between address/offset spaces, but `cp.async.bulk.tensor` legality depends on **(i)** *instruction-form typing* (target ISA gating + qualifiers), **(ii)** *exact byte-quanta constraints* keyed to datatype and architecture, and **(iii)** *async completion effects* (barrier completion with release semantics). A static relation $$f$$ cannot encode “Box-Size[0] must be exactly 96B for type X” because that constraint is not a property of the mapping—it is a property of the *chosen instruction encoding and dtype mode*.

**Proof sketch of insufficiency:**  
- The seed axioms do not include a notion of *instruction admissibility judgment*  
  $$\Gamma \vdash \texttt{cp.async.bulk.tensor}(\ldots) : \text{ok}$$  
  parameterized by (arch, dtype, mode, swizzle, sizes, alignments). They reason about mapping correctness assuming the operation exists. PTX makes “operation exists” conditional, with UB otherwise. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- The weak-memory + barrier-release semantics are **temporal/effectful**, not functional. ISL relations / $$\mathbb{F}_2$$ matrices do not model happens-before, completion, or release/acquire patterns. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**Concrete compiler artifact mapping:**  
- An IR operation `async_tensor_copy` must carry *typed attributes* `{dim, dtype, mode, swizzle, arch_min}` and produce a **completion token** (SSA value) representing the mbarrier phase/bytes completion.  
- A legality pass must check the PTX constraints (Box-Size exactness, stride alignment, coordinate congruences) before lowering. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- A scheduler / memory-model pass must enforce correct use of the completion token (e.g., wait before consuming data), reflecting the weak-memory + release semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

---

### Constraint 3: `wgmma` warpgroup-uniform control + mandatory fences

**(A) Seed Paper Claim.**  
- Paper 1 claims MMA/WGMMA layouts are representable as linear layouts; layout conversions can be generated generically. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Paper 2 positions layouts as unified relations; it can model swizzles and linear layouts; it focuses on mapping correctness. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Paper 3 studies layout algebra compatibility with CuTe operations and “alignment with CUTLASS behavior” (semantic agreement). ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))

**(B) Hardware Manual Claim.**  
PTX states: the mandatory `.aligned` qualifier indicates all threads in the warpgroup must execute the same `wgmma.mma_async`; conditional use must be warpgroup-uniform else undefined behavior. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
PTX also requires `wgmma.fence` at specific points; missing it yields undefined behavior. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**(C) Theoretical Deficit.**  
**Expressivity Gap (explicit):** a layout formalism can represent **where fragments live** (thread/register mapping), but it cannot represent the warpgroup-wide predicate  
$$\text{Uniform}_{WG}(p)$$  
required to justify executing `wgmma.*` under predicate $$p$$, nor can it represent the *required ordering constraints* (“a fence must appear before first wgmma, and between certain register accesses”). These are properties of **control flow + program order**, not of a coordinate→index mapping.

**Proof sketch of insufficiency:**  
- A function/relation $$f$$ does not encode whether a program’s control predicate is warpgroup-uniform; two programs can implement the same layout mapping but differ in divergence. PTX makes divergence a correctness cliff (UB). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- Likewise, a layout algebra does not encode required fence placement; two programs with identical layout mappings but different instruction orderings differ in legality. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**Concrete compiler artifact mapping:**  
- A **warpgroup-uniformity analysis** pass that proves conditions feeding `wgmma.*` are uniform across the 128-thread warpgroup (or rewrites code to enforce uniformity).  
- A **fence inference/verification pass** that inserts/validates `wgmma.fence` according to the PTX rules, using def-use of accumulator/input registers and program order. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

---

### Constraint 4: `mbarrier` object validity + state-token provenance

**(A) Seed Paper Claim.**  
- Paper 2: ISL relations represent mappings; operations include composition/inverse; correctness is about mapping preservation. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
- Paper 1: layout conversions and async-related codegen are discussed at a high level, but the core semantics are still mapping-based. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Paper 3: categorical operations reason about layout algebra, not synchronization objects. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))

**(B) Hardware Manual Claim.**  
PTX: using an uninitialized/non-mbarrier location with `mbarrier` ops (except init) is undefined behaviour; and some wait forms require `state` to be produced by a prior arrive variant—otherwise UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**(C) Theoretical Deficit.**  
**Expressivity Gap (explicit):** seed layout formalisms cannot represent the runtime *barrier object state machine* nor the *provenance constraint*  
$$\texttt{state} \in \text{Image}(\texttt{mbarrier.arrive*})$$  
required by PTX. A layout relation $$f$$ has no notion of “this SSA value must originate from that instruction” and no notion of “this shared-memory location contains a valid barrier object at this point in time.”

**Proof sketch of insufficiency:**  
- ISL relations and $$\mathbb{F}_2$$ matrices are extensional models of mappings. `mbarrier` legality is *intensional*: it depends on program history (was init executed? did arrive execute to produce this token?) and on memory object validity. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**Concrete compiler artifact mapping:**  
- Model `mbarrier` in IR with **typed tokens**: `arrive` returns a token value of type `mbarrier_state<scope>`; `test_wait` consumes that token.  
- Add a **barrier object lifetime analysis** that ensures the underlying shared-memory slot is initialized before use and invalidated/reused safely. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

---

### Constraint 5: Blackwell TMEM dynamic allocation + warp/warpgroup access restrictions

**(A) Seed Paper Claim.**  
- Paper 1: discusses “tensor memory” as a memory layout target and claims memory layouts are linear layouts; also ties specialized units (like Tensor Memory on Blackwell) to the need for layout conversion. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
- Paper 3: focuses on layout algebra and tractable layouts derived from categorical morphisms. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1))

**(B) Hardware Manual Claim.**  
TMEM is a dedicated on-chip memory: 512 columns × 128 rows per CTA on `sm_100a/sm_100f`; it is dynamically allocated by a single warp; allocation unit 32 columns; allocated column count must be power of two; access restrictions partition lanes by warp; `tcgen05.ld` requires warp-uniform `taddr` else UB; additional alignment constraints exist (`tcgen05.shift` lane aligned to 32). ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

**(C) Theoretical Deficit.**  
**Expressivity Gap (explicit):** a layout mapping can describe an addressing function into TMEM (e.g., lane/column indexing), but it cannot represent **allocation state**  
$$\text{AllocState} : \{\text{columns}\} \to \{\text{free},\text{allocated}\}$$  
nor the constraint that alloc/dealloc are *dynamic*, potentially blocking, and must be paired before kernel exit. Nor can it represent the *warp-uniform operand constraint* (“all threads in warp must use same `taddr`”) as part of a pure coordinate→index map.

**Proof sketch of insufficiency:**  
- Layout algebra assumes memory is an always-available set of addresses. TMEM is a **resource protocol**: you must allocate a power-of-two block (in 32-column units) before the address is even meaningful, and you must follow warp/warpgroup collective rules for access. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
- Therefore, “correctness of the layout function” is insufficient to imply “legal TMEM program.”

**Concrete compiler artifact mapping:**  
- Introduce explicit IR ops `tmem.alloc` / `tmem.dealloc` producing a *capability* value used to form valid `taddr`s.  
- Use a **linear/affine type discipline** (or region-based liveness) so every allocation is deallocated on all paths.  
- Use a **warp-uniformity check** for operands like `taddr` (similar to `wgmma` uniformity) to guarantee collective access requirements. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

---

## Axiom-vs-Hardware Matrix (required)

| Seed_Axiom | Hardware_Feature | Why_Axiom_Fails | Required_Math_Extension |
|---|---|---|---|
| Layout is a $$\mathbb{F}_2$$-linear map (binary matrix). ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | `CUtensorMap` encode-time legality: rank≤5, 64B descriptor alignment, stride/box bounds, finite swizzle/interleave modes. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)) | Linear-map semantics does not express **descriptor-field predicates** (bounds + congruences + finite enums). A legal mapping may have **no legal descriptor**. | Add a **legality constraint layer**: mixed bounded Presburger + congruences (bitvectors) + finite enums; implement as SMT/ISL+bitvector checker in a compiler pass. |
| Swizzle modeled as XOR/AND/shift bijection on an interval. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | TMA swizzle is an *enumerated* finite set with datatype restrictions + inner-dimension ≤ swizzle size. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html | Seed swizzle space is **too large** and does not encode “only these modes exist.” You can prove bijection yet still be unencodable. | Add **finite-mode selection** + constraints tying mode↔dtype↔dimensions; treat swizzle choice as a constrained optimization problem. |
| ISL relations are conjunctions of affine constraints; provide composition/inverse. ([arxiv.org](https://arxiv.org/html/2511.10374v1)) | `wgmma.mma_async` requires warpgroup-uniform execution (`.aligned`) else UB. ([docs.nvidia.com//docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | ISL relation describes data mapping, not **SIMT control-flow uniformity**. Two programs with same relation can differ in divergence, changing legality. | Add **SIMT uniformity analysis** (predicate abstraction) + IR regions marked `warpgroup_uniform`; verify or rewrite. |
| Layout conversion correctness via algebraic composition. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | `wgmma.fence` required at specific program points; missing it is UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | Composition laws do not encode **program order** constraints between instructions and registers. | Add **effect/ordering semantics** (instruction-level happens-before) + a fence inference/verification pass. |
| Layout is a static mapping; memory is a set of addresses. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | TMEM is dynamically allocated (power-of-two blocks, 32-column units), blocking; access restrictions; warp-uniform `taddr` required else UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | Static mapping cannot represent **allocation state/lifetime** or collective operand constraints. | Add **resource-aware IR** (alloc/dealloc + capabilities) + linear types / liveness + warp-uniform operand checking. |
| “Tensor copy” treated as a conceptual movement operation. ([arxiv.org](https://arxiv.org/html/2505.23819v3)) | `cp.async.bulk.tensor` has per-type **exact bequalities** (Box-Size[0]=64B/96B), alignment/congruence rules, arch gating, and weak-memory completion semantics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | Mapping semantics omits **instruction admissibility** and **async completion effects**; cannot express “this op is UB unless constraints hold.” | Add an **instruction typing judgment** + legality checker; model async completion as SSA tokens with memory-order effects. |
| Layout algebra captures correct of index computation. ([arxiv.org](https://arxiv.org/pdf/2601.05972v1)) | `mbarrier` requires object validity and token provenance; otherwise UB. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | Layout algebra cannot express “this shared address currently holds a valid barrier object” nor “this token must originate from arrive.” | Add **operational semantics** / effect system; represent barrier state tokens in IR; verify initialization and token flow. |

---
 4 — Elephant-in-the-room diagnosis (rank bottlenecks)

### (A) Seed Paper Claim.  
The seed papers focus on representing layouts and proving correctness of transformations (composition/inversion/compatibility), enabling code generation and cross-system reasoning. ([arxiv.org](https://arxiv.org/html/2505.23819v3))

### (B) Hardware Manual Claim.  
Blackwell-era systems increase on-chip/shared/L2 resources and support HBM3/HBM3e; NVIDIA guidance emphasizes coalescing and minimizing redundant global memory cess. ([docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html))  
DGX B200 system-level specs illustrate extreme compute vs memory bandwidth: FP8 Tensor Core performance and HBM3e bandwidth are both large, but compute scales to tens of PFLOPS while memory bandwidth is tens of TB/s. ([nvidia.com](https://www.nvidia.com/en-eu/data-center/dgx-b200/?utm_source=openai))

### (C) Theoretical Deficit (ranked).  
I rank bottlenecks by “what breaks first” when you try to r a modern kernel end-to-end:

1) **(C1) Legality cliffs (highest correctness risk).**  
   - UB conditions (`wgmma` uniformity/fence, `mbarrier` validity, TMEM rules, `cp.async.bulk.tensor` exact byte constraints) can silently miscompile.  
   - Seed formalisms don’t define an instruction admissibility judgment or a resource/lifetime discipline, so they cannot *prove absence of UB*.  
   **Compiler artifact:** legality/type-checking + effect/resource system must run *before* any optimization using the laut algebra.

2) **(C2) Latency/async semantics (highest scheduling risk).**  
   - Async copies and barrier completions are weak-memory/effectful; performance requires pipelining and correct wait placement. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
   - Seed mapping semantics is “timeless,” so it can’t express pipeline hazards.  
   **Compiler artifact:** token-based async IR + scheduler that reasons about dependency distance and ordering.

3) **(C3) Memol (highest throughput limiter; Blackwell compute-vs-bandwidth asymmetry).**  
   - DGX B200 specs (FP8 PFLOPS vs HBM TB/s) imply you need very high arithmetic intensity to saturate compute; otherwise you’re bandwidth-bound. ([nvidia.com](https://www.nvidia.com/en-eu/data-center/dgx-b200/?utm_source=openai))  
   - Even with improved cache/shared configurations, global memory is still a hard limiter; hence the emphasis on coalescing and reducing redundant memory traffic. ([docs.nvidia.com](https://docs.nvia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html))  
   **Compiler artifact:** layout math *helps* by enabling better swizzles/reuse, but only once legality + async scheduling are handled.

---

## Stage-1 Verdict (10 bullets)

1) **(A)** The seed papers make layout *composable* as math objects (matrices, relations, morphisms), enabling generic conversion reasoning. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
2) **(B)** Hardware correctness is governed by *admissibility constraints* (alignment, bounds, enums, arch gating) and *UB-triggering uniformity/order rules*, not just mapping correctness. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
3) **(C)** The central gap is that the seed math is **extensional** (functions/relations), while GPU legality is **intensional/effectful** (program order, tokens, allocation state).

4) **(A)** Paper 1’s $$\mathbb{F}_2$$ linearization is powerful for bit-level swizzles and layout conversion algorits. ([arxiv.org](https://arxiv.org/html/2505.23819v3))  
5) **(B)** TMA/`CUtensorMap` and `cp.async.bulk.tensor` legality depends on bounded integer fields and exact byte quanta tied to dtype/arch. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html))  
6) **(C)** Therefore, “linear map correctness” does not imply “descriptor encodability” or “instruction legality”; a solver-backed legality pass is required.

7) **(A)** Paper 2’s ISL modeling gives a to unify CuTe and Triton layout reasoning in a single relational framework. ([arxiv.org](https://arxiv.org/html/2511.10374v1))  
8) **(B)** But PTX requires warpgroup-uniform execution and mandatory fences for `wgmma`, which are control-flow and ordering properties. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
9) **(C)** ISL relations alone cannot express “uniform across warpgroup” or “fence must appear between these accesses”; you need SIMT-uniformity + rder analyses.

10) **(B)** Blackwell introduces TMEM as a dynamically allocated, finite on-chip resource with strict allocation and access rules, while system specs underscore compute vs bandwidth asymmetry. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))  
    **(C)** Any “layout-only” theory misses that end-to-end correctness/performance requires *resource protocols* (alloc/dealloc) and *async pipelines*, not just index algebra.

--- 

If you want Stage 2, I cann the above into a concrete “compiler contract” checklist: a minimal set of IR types/ops + analyses that, together, close each expressivity gap while preserving the seed papers’ algebraic benefits.

---
Learn more:
1. [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using ₂](https://arxiv.org/html/2505.23819v3)
2. [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
3. [Modeling Layout Abstractions Using Set Relations](https://arxiv.org/html/2511.10374v1)
4. [Categorical Foundations for CuTe Layouts](https://arxiv.org/pdf/2601.05972v1)
5. [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
6. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html)
7. [DGX B200: The Foundation for Your AI Factory | NVIDIA](https://www.nvidia.com/en-eu/data-center/dgx-b200/?utm_source=openai)
