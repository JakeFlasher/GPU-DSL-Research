## Three research directions (Stage‑2 synthesis)

Below are **3 distinct, high-value directions** that follow your synthesis formula:

\[
\textbf{Proposal} = (\text{Stage‑1 Hardware Gap}) + (\text{Stage‑1.5 Theory}) \rightarrow (\textbf{New Compiler/Runtime Artifact})
\]

Each is **real‑metal evaluable** (H100/B200/MI300-class GPUs), **no new hardware**, and implementable in **Triton / PyTorch / CUTLASS** within an ASPLOS/ISCA/MICRO prototype scope.

---

# Direction 1 — **Transport‑First Codegen**: TMA/Cluster/Async‑Proxy become first‑class IR objects

### 1) Define the Gap (why Linear Layouts / LEGO are insufficient)

**Stage‑1 gap:** Linear Layouts is *layout‑conversion‑centric*. It is strong at mapping tensor coordinates to threads/warps/shared with \( \mathbb{F}_2 \) bit‑linear maps, but on Hopper/Blackwell the limiting factor is often **transport orchestration**:

- Keeping **wgmma/mma pipes fed** depends on **TMA bulk movement**, **mbarriers**, and **warp specialization**.
- Correctness and performance depend on **async proxy ordering** (bulk TMA ops + some tensor ops live in an async proxy and require proxy fences).
- Cluster/DSMEM introduces a new locality tier; layouts alone don’t decide *who* moves data, *when*, and *with what ordering constraints*.

**Failure mode:** You can have a “perfect” swizzle/convert plan yet still hit **low SOL Tensor%** because the kernel is **schedule‑bound** (barrier stalls, scoreboard stalls, insufficient overlap), not “indexing‑bound.”

---

### 2) Apply the Theory (Stage‑1.5 math concept)

Use **asynchrony / concurrency math** as the missing abstraction:

- **Trace theory / Mazurkiewicz traces (partial commutation monoids):** model legal reorderings of events when independent.
- **Petri nets + max‑plus (tropical) algebra:** model pipeline throughput and resource constraints (TMA engines, shared buffers, warp roles).
- **Effect systems / separation‑logic‑style reasoning:** make async‑proxy correctness *compositional* (insert proxy fences by construction rather than by brittle patterns).

This is the key “seed‑rendering” move: **layouts stop being the only first‑class thing**; *transport events and their legality model* become first-class too.

---

### 3) Define the Mechanism (new compiler/runtime artifact)

#### Artifact: **Triton “Transport IR” + Proxy‑Aware Scheduler**
Add a transport sub‑dialect in TritonGPU (ttg) (or adjacent dialect) whose ops explicitly represent Hopper‑era movement:

- `ttg.tma.load(tile, tensormap, mbarrier, dst=shared::cta|shared::cluster)`
- `ttg.mbarrier.init/arrive/wait`
- `ttg.proxy.fence(async<->generic)`
- `ttg.warp_specialize(role=loader|compute)`
- (optional) `ttg.tma.multicast(ctaMask)`

Then run a **schedule synthesis pass** that chooses:

- **stage count** (double/triple buffering),
- **warp specialization split** (copy warps vs compute warps),
- **ordering** (where proxy fences go),
- **conversion placement** (do you convert in regs, via shared, or via transport staging),
- and (on cluster kernels) **cluster‑shared staging / multicast** decisions.

#### What “math” looks like in the implementation (not just words)
- Build an **event graph**:
  - nodes = transport/compute events,
  - edges = must‑happen‑before from effects + barriers + data deps.
- Define an **independence relation** \(I\) for trace rewriting:
  - allow commute when they touch disjoint resources / proxies.
- Use a Petri‑net‑like resource model:
  - tokens = free buffer stages,
  - places = shared buffers + mbarriers,
  - transitions = issue TMA, wgmma consumes stage, etc.
- Use a **max‑plus throughput estimator** to pick stage count and overlap that maximizes steady‑state initiation interval under register/shared constraints.

#### ASCII pipeline sketch (software stack)
```
Triton Python
   |
tt IR  (math/layout intent)
   |
   |  [Layout Engine: still useful]
   |   - choose anchor layouts
   |   - conversions + swizzles
   v
ttg IR (TritonGPU)
   |
   |  NEW: Transport IR construction
   |   - identify eligible global<->shared tiles
   |   - emit tensormap constraints + tma ops + barriers
   |
   |  NEW: Proxy-aware schedule synthesis
   |   - trace rewriting / fence insertion
   |   - stage count + warp specialization
   v
LLVM/NVVM -> PTX -> SASS (cp.async.bulk.tensor + mbarrier + wgmma)
```

---

### Feasibility filter (3–4 month prototype scope)

**Scope-limited MVP that still publishes:**
- Only target **H100** first.
- Only handle **global→shared TMA** for common 2D tiles used in GEMM/FA.
- Only implement **double-buffering** + a small set of canonical schedules.
- Insert proxy fences using a conservative effect system (sound > complete).

**Implementation plan (concrete):**
1. Add TTG ops and verification for well‑scoped proxy usage.
2. Lower TTG transport ops to inline PTX (or LLVM intrinsics) for `cp.async.bulk.tensor` + `mbarrier`.
3. Add scheduler pass that chooses between:
   - baseline ld/st staging,
   - cp.async (Ampere style),
   - TMA bulk (Hopper).
4. Hook into `tt.dot` lowering points (mma/wgmma anchors).

---

### Real‑metal evaluation plan (TritonBench + Nsight Compute)
- **Microbenchmarks:** pipelined tiled GEMM, shared‑staged attention blocks, wgmma inner loops with RHS update patterns.
- **TritonBench:** FlashAttention forward/backward; quantized GEMM variants if available.
- **Nsight Compute metrics:**
  - SOL Tensor / Tensor pipe active,
  - barrier stalls, scoreboard stalls,
  - eligible warps per cycle,
  - register count / occupancy cliffs,
  - achieved memory throughput vs roofline.

**Primary claim:** “Layout‑optimal kernels become transport‑optimal; tensor pipes stay fed on Hopper without hand‑written CUDA schedules.”

---

---

# Direction 2 — **Modular + Affine Layouts**: generalize \( \mathbb{F}_2 \) to mixed‑radix + piecewise domains (ragged decode/MoE)

### 1) Define the Gap

**Stage‑1 gap:** Linear Layouts is fundamentally tuned for **power‑of‑two shapes** and bit‑sliced reasoning:

- Ragged decode, MoE routing, and serving shapes are not powers of two.
- Padding+masking isn’t a “small overhead” in decode; it’s often *the* cost (wasted loads, predication, low utilization).
- Flips/slices/negative strides and “partial tiles” are not naturally represented by pure \( \mathbb{F}_2 \)-linear maps.

LEGO can express more general mappings but (a) lives mostly in the frontend and (b) does not solve backend legality/cost for vectorization, conversions, and hardware-specific staging.

---

### 2) Apply the Theory (Stage‑1.5 concepts)

Replace “bit‑linear algebra” with **integer/modular + affine** tools that strictly generalize it:

- **Integer lattices / finitely generated abelian groups + Smith Normal Form (SNF) / Hermite Normal Form (HNF):**
  - solve conversion existence,
  - synthesize right inverses in non‑power‑of‑two regimes,
  - decide contiguity/alignment as modular constraints, not bit rank.
- **Affine groups \( \mathrm{AGL}(n,\mathbb{Z}) \) / \( \mathrm{AGL}(n,\mathbb{Z}_m) \):**
  - represent slices (translation \(b\)), flips (multiplication by \(-1\)), general strides.
- **Presburger arithmetic / piecewise affine domains:**
  - make “interior tile” vs “edge tile” a first‑class split (multi-versioning),
  - treat masks as domain constraints rather than pervasive predication.

---

### 3) Define the Mechanism (new compiler/runtime artifact)

#### Artifact: **Triton “ModAffine Layout Engine” + Domain‑Split Kernel Multi‑Versioning**

Introduce a new layout object:

\[
x \mapsto A x + b \quad (\text{over } \mathbb{Z} \text{ or } \mathbb{Z}_m), \quad \text{with domain } D \subseteq \mathbb{Z}^n
\]

Where:
- \(A\) is a small integer matrix (dimensions are tiny in practice: 2–6 axes),
- \(b\) is an integer translation (slice offset),
- \(m\) captures extents / bank moduli / alignment constraints,
- \(D\) is a Presburger-definable validity domain (for partial tiles).

#### What the pass actually does
1. **Domain extraction pass (tt level):**
   - derive constraints like \(0 \le i < L_{kv}\), \(L_{kv}\) ragged per batch element, etc.
2. **Layout solving pass (tt→ttg):**
   - use HNF/SNF-style elimination to:
     - compute vectorizable width given alignment constraints,
     - synthesize conversion plans without forcing power-of-two padding.
3. **Kernel multi-versioning:**
   - generate:
     - **fast interior kernel**: no masks, maximal vectorization,
     - **edge kernel(s)**: minimal masking, only where domain requires.
4. **Runtime dispatch (PyTorch/Triton caching):**
   - choose variant based on shape bucket predicates:
     - e.g., \(L_{kv} \bmod 128 = 0\), \( \text{head\_dim} \in \{64,128\} \), etc.

#### ASCII view (piecewise specialization)
```
Runtime shapes (B, Lq, Lkv[i], head_dim, ...)
   |
   |  [Presburger predicate evaluation]
   |    - interior?   (full tiles)
   |    - edge?       (tail tiles)
   v
 +-------------------+     +-------------------+
 |  Kernel Variant A |     |  Kernel Variant B |
 |  (interior)       |     |  (edge)           |
 |  - no masks       |     |  - minimal masks  |
 |  - max vector ld  |     |  - limited vec ld |
 +-------------------+     +-------------------+
            \                 /
             \               /
              v             v
               GPU execution (higher SOL%, fewer cliffs)
```

---

### Feasibility filter (3–4 month prototype)

**MVP strategy:** keep SNF/HNF machinery minimal and practical.
- Matrices are tiny; implement a robust HNF (or “good enough” SNF subset) with gcd elimination in C++.
- Restrict initial support to:
  - 2D/3D layouts (common in attention/GEMM tiles),
  - modular reasoning for alignment and strides,
  - affine translations for slices.

**Why this can beat “full polyhedral compilation” in time:** we’re not doing general loop transforms; we’re doing **layout conversion + vectorization legality** with a domain split.

---

### Real‑metal evaluation plan
- **TritonBench workloads:** ragged decode attention, MoE pre/post kernels, embedding lookups where partial tiles dominate.
- **Metrics:**
  - reduction in masked instruction fraction,
  - vector width achieved on global loads/stores,
  - register pressure change (occupancy cliffs),
  - end-to-end latency per token (decode).

**Primary claim:** “Stop paying the padding tax; specialize kernels to ragged serving shapes without losing backend layout intelligence.”

---

---

# Direction 3 — **Indirection‑Aware Compilation**: treat KV paging / MoE routing as *relations*, not layouts

### 1) Define the Gap

**Stage‑1 gap:** Both seeds are strongest when “logical indices are regular.” LLM serving is often the opposite:

- KV-cache paging: **pointer chasing** / block tables break contiguity.
- MoE routing: scatter/gather dominates; tokens-per-expert is highly nonuniform.
- Linear Layouts’ gather fast path is brittle (warp-contained axis) and shuffle rounds can become instruction-bound.
- Even a perfect layout cannot fix **physical non-contiguity**—you need **data reordering** or **hierarchical movement plans**.

---

### 2) Apply the Theory (Stage‑1.5 concepts)

Use “indirection math” rather than “layout math”:

- **Relations as algebra (Boolean semiring / sparse 0–1 matrices):**
  - gather/scatter are relations \(R \subseteq X \times Y\), not bijections.
- **Inspector–executor = relation factorization:**
  - approximate \(P \approx \Pi_2^\top \, \mathrm{BlkDiag}(\text{dense}) \, \Pi_1\) where \(\Pi\) are permutations.
- (Optional extension for movement inside GPU)
  - **Orbit/coset decomposition** for deciding warp/CTA/cluster-contained movement,
  - **e-graph + ILP/SMT extraction** for choosing between shuffle/shared/cluster/TMA staging plans under register constraints.

---

### 3) Define the Mechanism (new compiler/runtime artifact)

#### Artifact: **Pack‑Execute Runtime + “Indirect Axis” Compiler Contract**

Add a compiler-visible annotation: **`indirect_axis`**.

When the compiler sees:
- an index array used for gather/scatter,
- or a KV page table access pattern,

it can choose between two lowering modes:

**Mode A — Direct gather (status quo):**
- acceptable only for small/warp-contained patterns.

**Mode B — Pack‑Execute (new):**
1. **Inspector step (runtime, cheap):**
   - compute a permutation that groups tokens by:
     - expert id (MoE),
     - KV page id / block id (paging),
     - or both.
2. **Pack kernel:**
   - gather into a packed contiguous buffer (now regular).
3. **Dense compute kernel:**
   - apply Linear/ModAffine layouts and (if Direction 1 exists) TMA pipelines.
4. **Unpack/scatter kernel:**
   - scatter outputs back to original order.

#### ASCII dataflow
```
indices (expert_id / page_id) + activations
          |
          |  [Inspector: build permutation + segment offsets]
          v
  +-------------------+
  | Pack (gather)     |  -> contiguous tiles
  +-------------------+
          |
          v
  +-------------------+    (now layout-friendly)
  | Dense compute     |    - wgmma/mma
  | + layout engine   |    - TMA pipeline possible
  +-------------------+
          |
          v
  +-------------------+
  | Unpack (scatter)  |
  +-------------------+
          |
          v
   original ordering restored
```

#### Key compiler/runtime contracts (to keep it implementable)
- Provide a small runtime API:
  - `plan = build_pack_plan(indices, bucket_policy)`
  - `pack(plan, x)`, `unpack(plan, y)`
- The compiler emits both direct and pack-execute variants and uses a **cost model** to dispatch:
  - if tokens-per-expert < threshold → direct,
  - else pack-execute.
- For H100: pack kernel can use **bulk movement** (where applicable) or at least improved coalescing.
- For MI300: pack staging uses LDS with **target-parametric XOR swizzles** (you can reuse the bank-model framework from Stage‑1.5 even if not the primary contribution here).

---

### Feasibility filter (3–4 months)

**MVP: MoE only or KV paging only (pick one).**
- MoE is usually easier because “expert id” is a clean key for sorting/segmentation.
- Implement inspector with:
  - histogram + prefix sum + scatter (GPU-side) or CPU-side for small batch.
- Integrate into PyTorch runtime as an opt-in path for Triton kernels.

---

### Real‑metal evaluation plan
- **TritonBench MoE:** measure end-to-end layer latency under realistic token-per-expert skew.
- **KV paging microbench:** vary page size, locality, batch size; measure decode step time.
- **Profiling signatures:**
  - fewer long-scoreboard stalls (better coalescing),
  - higher L2 utilization,
  - lower shuffle instruction fraction (less in-warp routing),
  - improved SOL DRAM% *and* SOL SM% (less divergence).

**Primary claim:** “Make indirection regular enough that layout + transport optimizations finally apply.”

---

---

# Decision matrix (score + rationale)

Scoring: **1 (low) → 5 (high)**.  
Implementation risk: **5 = lowest risk / easiest**, **1 = highest risk**.

| Direction | Theoretical Novelty | Hardware Relevance (H100/B200 features) | Implementation Risk (3–4 months) | Why |
|---|---:|---:|---:|---|
| **1) Transport‑First Codegen (TMA/Proxy/Cluster IR)** | 4 | 5 | 2 | Novel because it elevates *scheduling semantics* (traces/effects/Petri nets) into the compiler stack; maximal relevance because it directly targets TMA + async proxy + clusters; risk is high due to correctness/fence subtleties and TMA descriptor constraints. |
| **2) ModAffine Layouts (SNF/HNF + Presburger domains)** | 5 | 4 | 3 | Strong “seed-obsoleting” math move: \( \mathbb{F}_2 \rightarrow \mathbb{Z}/\mathbb{Z}_m \) + affine + piecewise domains; very relevant to ragged serving and non-power-of-two shapes; medium risk because integer normal forms must be implemented carefully but matrices are small and domain splitting is practical. |
| **3) Indirection‑Aware Pack‑Execute (relations + inspector–executor)** | 4 | 4 | 4 | Novel application of relation algebra/factorization framing; very relevant to MoE/KV paging (major real bottlenecks); comparatively lower risk because an MVP can be built as a runtime transformation with clear wins on skewed distributions, though overhead tuning is nontrivial. |

---

## How these three fit together (optional “portfolio” view)

They are intentionally orthogonal, but compose into an “AI Factory Triton” stack:

- **Dir‑3** makes irregular workloads **regular enough** (packing) to expose dense structure.
- **Dir‑2** makes dense structure **representable without power-of-two hacks** (mod/affine + piecewise).
- **Dir‑1** makes dense, representable tiles **actually reach peak** (TMA + proxy‑correct scheduling).

If you want, I can turn any one direction into an ASPLOS-grade “paper skeleton” (IR design, algorithm, prototype plan, evaluation figures, Nsight section, and related work positioning vs Triton-Linear + LEGO).
