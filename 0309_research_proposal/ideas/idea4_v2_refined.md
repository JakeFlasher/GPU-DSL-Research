Below is the drastically narrowed **one-paper** version.

## 1. Title

**ATLAS-RC: Replica–Caching Duality for Immutable Tensor Tiles**

## 2. Abstract

This paper no longer claims a universal cross-scale layout algebra. It isolates one precise claim: **for immutable/read-mostly tensor tiles, inter-GPU replication and on-chip caching are the same copy operator instantiated at different hierarchy cuts**. We formalize this claim in the smallest fragment where it is both provably true and compiler-useful: a two-level hierarchy `GPU -> TMEM`, read-only tensors over a fixed execution region, and a fixed post-routing consumer set. The main theorem proves that replica placement and cache placement are observation-equivalent up to scope, and differ only in which budget they consume (HBM/network versus TMEM capacity). We then show one concrete optimization around a single running example: **hot-expert placement in MoE inference**. The planner can legally trade an extra GPU replica of a hot expert weight tensor for a larger TMEM-resident tile in the local expert kernel, jointly reducing remote token traffic and local weight streaming. The artifact is intentionally small: a proof-backed planner over a few existing kernel variants plus one multi-GPU MoE prototype. This paper does **not** include a full compiler, E-graph rewriting, general code generation, or NCCL lowering.

## 3. Key Insight / Thesis Statement

**In the immutable fragment, a copy is just a copy.**  
The only difference between a GPU replica and a TMEM cache entry is **where in the hardware hierarchy the copy is materialized** and **which resource budget it consumes**.

That statement becomes formally true in the smallest useful setting:

- immutable or read-mostly tensor tiles,
- one planning window with no writes,
- two hierarchy cuts: `GPU` and `TMEM`,
- one fixed running example: **MoE expert weights after routing is known**.

This is enough to yield one new optimization that current split systems do not express in one search space:  
**replicate a hot expert across more GPUs if that makes a larger TMEM tile legal for the local expert GEMM.**

## 4. Technical Approach

### 4.1 Running example: one hot MoE expert

Consider MoE inference on 4–8 GPUs. After top-k routing, expert weights are fixed and read-only for the whole expert-execution region. Suppose expert `e*` is hot.

Two legal plans exist:

- **Plan A:** one owner GPU for `e*`; tokens are shuffled there; the owner runs a grouped/smaller-tile expert kernel.
- **Plan B:** `e*` is replicated on 2 GPUs; more tokens are served locally; each replica can run a dedicated larger-TMEM-tile kernel.

The paper’s question is narrow: **can those two plans be described as the same copy operation at different cuts, and can that theorem drive a useful optimizer?**

### 4.2 Smallest formal fragment: Read-Mostly Copy Calculus

We define a minimal calculus over a tree-shaped hierarchy:

`world -> GPU -> TMEM`

- Let `τ` range over tiles of an immutable tensor `W`.
- Let `C ⊆ Tile × GPU × Phase` be the fixed consumer set for one execution region.
- A placement is:

`L = (home, R_gpu, R_tmem)`

where:

- `home(τ)` is the canonical owning GPU,
- `R_gpu(τ)` is the set of extra GPU-resident copies of `τ`,
- `R_tmem(τ, g)` is the set of TMEM-resident copies of `τ` on GPU `g`.

Define one operator:

`copy_at(cut, τ, S)`

which duplicates tile `τ` into descendants `S` at hierarchy cut `cut`.

Then:

- **GPU replication** = `copy_at(GPU, ...)`
- **TMEM caching** = `copy_at(TMEM, ...)`

A consumer observes correct data if an identical copy of `τ` is reachable before its phase executes. Because the region is immutable, there is no coherence or writeback state.

### 4.3 Main theorem: Replica–Caching Duality

**Replica–Caching Duality Theorem.**  
In the Read-Mostly Copy Calculus, for any immutable placement `L` and cuts `c1, c2 ∈ {GPU, TMEM}`,

`Obs(copy_at(c1, copy_at(c2, L))) = Obs(copy_at(c2, copy_at(c1, L)))`

So replication and caching are not two different semantic operators. They are one operator, indexed by scope.

**Why the theorem is true**
- tiles are immutable in the region,
- reads care only about reachability of identical values,
- duplication order does not change observable reads.

**Why the theorem is useful**
Equivalent plans differ only in **budget accounting**, not correctness:

- GPU copies charge HBM/network budget,
- TMEM copies charge on-chip capacity budget.

That gives a compiler-safe exchange rule: move copies upward or downward across cuts, preserve semantics, then optimize cost.

### 4.4 Compiler-useful corollary: MoE replication ↔ TMEM tile size

For a hot expert `e`, the planner chooses:

- replication factor `r_e ∈ {1, 2, 4}`,
- kernel variant `k ∈ {small, medium, large}` with TMEM tile size `t(k)`.

Feasibility constraints are simple:

- `r_e * |W_e| <= extra_HBM_budget`
- `tokens_e / r_e >= batch_min(k)`
- `tmem_footprint(k) <= TMEM_budget_per_replica`

Objective:

- reduce remote token traffic,
- reduce local weight streaming,
- improve end-to-end tokens/s.

This yields a small **exchange curve** for each hot expert: as `r_e` increases, a larger TMEM tile may become legal. The planner asks:

**“Is one more GPU copy of `W_e` worth the larger TMEM tile it unlocks?”**

That is the single concrete optimization of the paper.

Current systems can usually do only one side at a time:

- distributed planners choose expert replication,
- kernel autotuners choose local tile size.

They do **not** expose this as one legality-preserving transformation over the same immutable tensor copies.

### 4.5 Prototype and explicit non-goals

The prototype is intentionally limited.

**We will build**
- a small planner implementing the theorem and exchange rule,
- a MoE runtime hook that can replicate selected hot experts,
- 2–3 existing expert-kernel variants with different TMEM tile sizes.

**We will not build**
- a general layout IR,
- E-graph rewriting,
- automatic kernel synthesis,
- general multi-GPU communication lowering,
- a mutable-tensor theory,
- a universal ATLAS optimizer.

Communication primitives and kernel codegen are treated as fixed infrastructure, not contributions.

## 5. Expected Contributions

- **A minimal formal core** for immutable tensor placement over `GPU -> TMEM`.
- **A precise Replica–Caching Duality theorem** showing that caching and replication are the same scope-indexed copy operator in this fragment.
- **A compiler-useful exchange rule** that collapses a cross-level search into a small discrete optimization.
- **One concrete optimization**: trading GPU replication of hot MoE experts for larger TMEM tiles in local expert kernels.
- **One focused prototype result** showing when this joint optimization beats split placement-only or kernel-only baselines.

## 6. Evaluation Plan

### Scope and timeline: 12–18 months, realistically 15 months

**Months 1–4**
- formalize the Read-Mostly Copy Calculus,
- prove the duality theorem and exchange lemma,
- build a small oracle enumerator for tiny cases.

**Months 5–7**
- derive the MoE exchange-curve optimizer,
- calibrate a simple cost model using routing skew, HBM budget, and TMEM capacity.

**Months 8–11**
- integrate with an MoE inference runtime,
- reuse 2–3 existing expert-kernel variants,
- implement selective hot-expert replication.

**Months 12–15**
- run experiments,
- compare against split baselines,
- write the paper.

**Stretch months 16–18**
- extra model, more routing traces, stronger ablations.

### Prototype

- Single-node, 4–8 GPU MoE inference prototype.
- Primary target: Blackwell-class TMEM if available.
- Fallback: the same theorem with SMEM as the lower cut.

### Benchmarks

- Synthetic routing distributions from uniform to Zipf-skewed.
- One real MoE workload with hot-expert behavior.
- Microbenchmarks for isolated expert execution plus end-to-end MoE block latency.

### Baselines

- **Placement-only:** expert replication heuristic, fixed kernel.
- **Kernel-only:** autotuned tile size, fixed expert placement.
- **Oracle:** brute-force joint grid search on small instances.

### Metrics

- tokens/s and latency,
- remote token bytes,
- local HBM bytes,
- TMEM occupancy,
- planner overhead,
- gap to oracle on small cases.

### Success criteria

- planner matches or nearly matches oracle on small instances,
- beats both split baselines under skewed routing,
- correctly predicts when replication should **not** be used.

## 7. Target Venue and Why

**PLDI**.

This is now primarily a **formalism + compiler-usefulness** paper, not a broad systems umbrella. The core claim is a small theorem about immutable tensor copies; the MoE prototype exists to prove that the theorem enables a new optimization, not to present a full production compiler. That scope fits PLDI much better than the original ATLAS-wide framing.

## 8. Potential Weaknesses and Mitigations

- **Weakness: the formal core may look too small.**  
  **Mitigation:** make that a strength. The paper explicitly asks for the *smallest* fragment where the duality is both true and useful.

- **Weakness: restricted to immutable/read-mostly tensors.**  
  **Mitigation:** target MoE inference, where the assumption is exact; discuss training only as future work via immutable windows between updates.

- **Weakness: benefit may require skewed routing and hot experts.**  
  **Mitigation:** characterize the threshold regime and include negative results on near-uniform routing.

- **Weakness: TMEM-specific demonstration may seem hardware-narrow.**  
  **Mitigation:** the theorem is scope-indexed; TMEM is just the strongest demo. A SMEM fallback keeps the paper valid.

- **Weakness: reviewers may still confuse this with the original ATLAS umbrella or Idea 2.**  
  **Mitigation:** rename it **ATLAS-RC**, state explicit non-goals up front, and remove all claims about universal layout algebra, cross-arch lowering, and full-system unification.

- **Weakness: “no existing system can express this” could sound overstated.**  
  **Mitigation:** phrase the claim precisely: existing systems can hand-compose the two decisions, but they do not expose them as one formal copy transformation with one legality rule and one search space.

If you want, I can next turn this into a **1-page submission pitch**, a **PLDI-style extended abstract**, or a **figure-by-figure paper outline**.
