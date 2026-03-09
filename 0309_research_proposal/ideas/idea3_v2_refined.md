Below is a fully revised version with all 8 sections updated.

---

## 1. Title

**Cache-as-Layout: Exact On-Chip Reuse and Probabilistic L2 Bounds for Synthesizing Hierarchy-Aware GPU Kernels**

## 2. Abstract

Modern GPU kernels are limited as much by traversal order as by arithmetic tiling, but existing compiler treatments blur together two very different hierarchy levels: **software-managed storage** (registers, shared memory, TMEM) and **hardware-managed caches** (especially L2). This proposal refocuses Cache-as-Layout around that split. We treat each hierarchy level as a named layout, but claim **exactness only where the compiler controls residency**. For shared memory and TMEM, quotient/complement factorization yields exact formulas for live occupancy, bank/slot conflicts, and spill-free schedules. For L2, we state the Complementary Reuse Theorem only under an explicit idealized model: fixed line size, known set mapping, finite associativity, true LRU replacement, isolated access stream, and no foreign CTA traffic. Under that model, same-set reuse distance is determined by schedule order over the complement axes; in 2D separable cases, serpentine/Sawtooth emerges as an exact corollary. Real hardware is then treated as a deviation from this ideal via calibrated replacement slack and CTA-interference distributions, yielding **probabilistic miss bounds and robust schedule ranking**, not universal optimality claims. For 3D+ complements, Morton/Hilbert-style traversals are presented only as heuristics. Finally, we integrate cache-aware traversal with **CuTeFuse-style fusion**, so fused pipelines jointly choose on-chip carrier layouts and outer traversal orders. The resulting claim is narrower but stronger: **exact theorems for software-managed levels, explicit-model theorems for ideal L2, quantified deviation on real GPUs, and a practical path to compiler-generated cache-aware fused kernels**.

## 3. Key Insight / Thesis Statement

**Thesis:** quotient/complement is the right algebraic interface between layout and reuse, but its meaning depends on the hierarchy level.

- For **software-managed levels** (SMEM/TMEM), quotient/complement gives an **exact liveness and conflict calculus**.
- For **hardware-managed L2**, quotient/complement gives an **ideal reuse calculus** under a stated cache model, plus a **calibrated probabilistic bound** on real hardware.

So the contribution is not “all cache behavior is just layout.”  
It is:

> **Layout determines the access trace; hardware policy and interference determine how much of that trace becomes reuse.**  
> The compiler can prove the first part exactly where it controls state, and bound the second part where hardware controls state.

**Scope of the formal core:** read-mostly dense or block-sparse tensor kernels with static tilings and tractable CuTe/Axe layouts. We prove exact results for on-chip levels, give explicit-model results for L2, and treat higher-dimensional traversal synthesis as heuristic unless formally derived.

**Non-incremental leap:** the compiler does not just tile for tensor cores and then autotune loop order. It derives a small, algebraically structured schedule family from resident-object quotients and complement axes, then composes that with fusion decisions.

## 4. Technical Approach

### A. Represent each hierarchy level as a named layout, but split deterministic from probabilistic levels

Let \(I\) be the dynamic iteration space and define the usual composed access map

\[
A_h = H_h \circ L_{\text{data}} \circ L_{\text{alg}} : I \rightarrow \mathcal{R}_h \times \mathcal{O}_h
\]

for hierarchy level \(h\), where:

- \(\mathcal{R}_h\): resident object identity at level \(h\),
- \(\mathcal{O}_h\): intra-object offset.

From this we derive:

- **resident quotient** \(Q_h\): which object is being touched,
- **offset map** \(O_h\): where inside that object,
- **complement coordinates** \(K_h\): iteration directions not absorbed by \(Q_h\).

Using CuTe/Axe named axes, these maps stay symbolic across CTA, warpgroup, warp, lane, stage, line, bank, slot, and cluster axes.

We then split the hierarchy:

1. **Deterministic levels:** registers, shared memory, TMEM  
   Compiler controls allocation, release, and layout.
2. **Probabilistic levels:** L2  
   Hardware controls replacement and sees interference from other CTAs/SMs.
3. **Out of formal core:** undocumented hardware-managed L1 behavior  
   We treat this as part of measured deviation, not theorem scope.

This separation is the main revision relative to the original proposal.

---

### B. Exact theorems for software-managed levels (SMEM/TMEM)

For \(h \in \{\text{SMEM}, \text{TMEM}\}\), assume model \(\mathcal{M}_{\text{det}}(h)\):

- capacity \(C_h\) is known,
- bank/slot mapping is known,
- objects are explicitly promoted and explicitly released,
- there is **no hidden replacement**,
- we target the common tractable class where each on-chip object has a contiguous live interval under a legal tiled schedule.

#### Theorem 1 (Deterministic Complement Theorem)
For any legal schedule \(\sigma\) in this class, the exact peak live occupancy is

\[
\Lambda_h(\sigma)=\max_t \left|\{Q_h(i)\;|\; i \text{ has been loaded by time } t \text{ and not yet last-used}\}\right|.
\]

A spill-free implementation exists **iff** \(\Lambda_h(\sigma)\le C_h\).

Moreover, within a schedule class that preserves intra-object order and reorders only complement coordinates \(K_h\), minimizing complement-frontier growth minimizes \(\Lambda_h\). Bank/slot conflict degree is an exact function of \(O_h\), so swizzles can be synthesized as exact low-bit constraints rather than heuristics.

**Interpretation.**
- For SMEM/TMEM, there is no “cache miss theorem.”  
  There is an **exact promotion/liveness theorem**.
- If the live frontier fits, each object is loaded once.
- If it does not fit, the compiler knows exactly where spills/reloads arise.

This is especially attractive for TMEM and fused pipelines, where the compiler also controls stage structure.

---

### C. State the Complementary Reuse Theorem only under an explicit idealized L2 model

We now state the central theorem under a precise model, not for “real L2 in general.”

#### Idealized L2 model \(\mathcal{M}_{L2}^{\mathrm{iso}}\)

- line size \(B\),
- \(S\) sets,
- associativity \(W\),
- capacity \(C = BSW\),
- deterministic set map \(\psi\) from line ID to set (or slice,set),
- **true LRU** replacement within each set,
- isolated access stream from one logical kernel stream or persistent CTA stream,
- no foreign insertions from other CTAs/SMs,
- no prefetch/coherence surprises,
- read-mostly accesses,
- schedule class \(\Sigma\) preserves dependences and fixed intra-line order.

#### Theorem 2 (Complementary Reuse Theorem under \(\mathcal{M}_{L2}^{\mathrm{iso}}\))
For any reuse event \(j\) of line \(\ell_j\), let \(d_j^\sigma\) be the number of **distinct same-set lines** accessed between the two uses of \(\ell_j\) under schedule \(\sigma\). Then:

- reuse \(j\) is a hit **iff** \(d_j^\sigma < W\),
- total L2 misses equal
\[
M_{L2}(\sigma)=M_{\text{compulsory}}+\sum_j \mathbf{1}[d_j^\sigma \ge W].
\]

Because the schedules in \(\Sigma\) differ only in how they traverse complement coordinates \(K_{L2}\), L2 miss behavior within \(\Sigma\) is fully determined by complement-induced same-set reuse distance. Therefore, any schedule minimizing those distances is miss-optimal **within \(\Sigma\)** under \(\mathcal{M}_{L2}^{\mathrm{iso}}\).

This directly answers the reviewer’s critical question:  
**the theorem is claimed only for isolated, set-associative, true-LRU L2 with known mapping and fixed schedule class.**

#### Sawtooth as an explicit corollary
In the 2D attention-style case where:

- \(Q_{L2}\) identifies resident KV lines/tiles,
- \(K_{L2}\) is the Q-block traversal grid,
- set mapping is separable enough for wavefront parity to control same-set reuse distance,
- legal schedules are wavefront-respecting,

the frontier-minimizing schedule is parity-reversed serpentine.  
That is precisely **Sawtooth**.

So Sawtooth is still a theorem-backed result, but now only under a stated model and schedule class.

---

### D. Real hardware: calibrated probabilistic bounds for L2, not exact claims

Real GPUs deviate from \(\mathcal{M}_{L2}^{\mathrm{iso}}\) because of:

- pseudo-LRU or undocumented victim choices,
- slice hashing quirks,
- prefetch or hidden policy effects,
- other CTAs/SMs inserting into the same sets.

We model this with two deviation terms for each reuse event \(j\):

- \(X_j\): foreign same-set insertions during the reuse window,
- \(R_j\): effective replacement slack relative to ideal LRU.

Then the real miss event is approximated by

\[
\Pr[\text{miss}_j] = \Pr[d_j^\sigma + X_j + R_j \ge W].
\]

This yields two outputs:

1. **ideal prediction** from \(d_j^\sigma\),
2. **real-hardware bound** from empirical or learned distributions over \(X_j\) and \(R_j\).

So for L2, the compiler will optimize either:

- expected misses,
- an upper confidence bound on misses,
- or schedule ranking robust to deviation.

This is a weaker but much more defensible claim than “layout exactly predicts cache behavior.”

---

### E. Quantifying hardware deviation

A core part of the paper is to measure the gap between the ideal theorem and real GPUs.

We will infer a hardware descriptor in stages:

1. **Single-stream microbenchmarks**  
   Recover effective line size, set count, associativity, and approximate set/slice mapping.
2. **Replacement probes**  
   Estimate how far real replacement deviates from LRU; derive an effective slack model \(R_j\).
3. **Controlled interference probes**  
   Run calibrated adversary CTAs to estimate \(X_j\) as a function of resident CTA count, kernel mix, and reuse window length.

We will report three deviation metrics:

- **count error:** predicted vs measured misses/bytes,
- **ranking fidelity:** correlation between predicted and measured schedule ordering,
- **optimization regret:** slowdown of the schedule chosen by the model relative to the best measured schedule.

The goal is not perfect counter prediction.  
The goal is to show that the theorem gives the right **schedule structure**, and that a small calibrated residual closes the remaining gap.

---

### F. Traversal, tiling, swizzling, and CTA interference

#### Tiling
Choose \(T_{L2}, T_{SMEM}, T_{TMEM}, T_{REG}\) subject to:

- exact occupancy constraints for SMEM/TMEM,
- bank/slot conflict constraints,
- L2 associativity-margin constraints under the ideal model,
- occupancy/register constraints.

#### Traversal
The compiler will search only a structured family induced by \(K_h\):

- monotone scans,
- parity-reversed serpentine,
- skewed wavefronts,
- CTA coloring / phase staggering,
- cluster-local traversal orders.

For **1D and 2D complements**, we will make theorem-backed claims where derivable.  
For **3D+ complements**, Morton/Hilbert/Gray-like traversals are **heuristic candidates only**. They will be scored empirically or by the calibrated model, not advertised as optimal.

#### Swizzling
Low-bit address behavior is handled with linear swizzles over bank/set bits.  
For SMEM/TMEM this is exact conflict control; for L2 it is a model-guided heuristic to reduce set hot spots.

#### CTA interference
CTA interference matters only at probabilistic levels.

- For **SMEM/TMEM**, non-cooperating CTAs are irrelevant because storage is partitioned; cooperative 2-CTA kernels are modeled jointly and exactly.
- For **L2**, interference enters through \(X_j\). We will reduce it using:
  - persistent-kernel modes for validation,
  - CTA coloring,
  - phase-staggered work queues,
  - cluster-local scheduling when available.

So multi-SM contention is not ignored; it is pushed into an explicit interference term and, where possible, controlled by scheduling.

---

### G. Integration with CuTeFuse-style kernel fusion

The reviewer’s “Ideas 1+3” suggestion is exactly right. The near-term version of this project should integrate **fusion** and **cache-aware traversal**.

The combined workflow is:

1. **CuTeFuse-style fusion** derives a common on-chip carrier layout \(B\) for intermediate tensors.
2. The carrier \(B\) becomes a **deterministic hierarchy level** governed by Theorem 1.
3. Cache-as-Layout then chooses the **outer traversal** over external tensors to maximize L2 reuse while respecting the fused pipeline’s on-chip live-state constraints.

#### Example: fused attention
For \(QK^T \rightarrow\) online softmax \(\rightarrow PV\):

- fusion removes score-matrix materialization,
- registers/TMEM/SMEM hold score/probability fragments and reduction state,
- the outer schedule over K/V pages or tiles is then chosen by the L2 model,
- Sawtooth-like parity reversal becomes one candidate when the fused layout exposes the same 2D complement structure.

#### Example: MoE pipeline
For route/pack \(\rightarrow\) grouped GEMM \(\rightarrow\) combine:

- fusion changes which expert tiles remain on-chip,
- traversal over expert/page order can be made L2-aware,
- the deterministic theorem prevents over-fusing into an occupancy disaster.

This fusion integration is also the feasibility story: the first implementation can target a few high-value fused pipelines rather than “all GPU kernels.”

## 5. Expected Contributions

- **A corrected formal split of the hierarchy**: exact compiler-controlled on-chip levels vs probabilistic hardware-managed L2.
- **Theorem 1**: an exact quotient/complement liveness/conflict theorem for shared memory and TMEM.
- **Theorem 2**: the Complementary Reuse Theorem under an explicit isolated set-associative L2 model, with Sawtooth as a 2D corollary.
- **A deviation-calibrated extension** that quantifies how real GPUs differ from the ideal theorem via replacement slack and CTA interference.
- **An honest synthesis story**: 1D/2D theorem-backed traversal, 3D+ Morton/Hilbert-style traversals as heuristics.
- **A fusion-aware compiler path** that combines CuTeFuse-style carrier synthesis with cache-aware outer traversal.
- **A practical reduction of search space** from unconstrained autotuning to a small algebraically generated schedule family plus residual calibration.

## 6. Evaluation Plan

### Platforms
- **Minimum viable core:** H100/H200-class GPUs for L2 + SMEM results.
- **Stretch goal:** one Blackwell-class platform for TMEM extension.
- If Blackwell access slips, the paper remains complete with H100/H200 L2+SMEM and treats TMEM as an extension.

### Workloads
- FlashAttention / paged attention / decode-prefill kernels
- Grouped GEMM / split-K GEMM
- MoE routing + packing + grouped GEMM
- Block-sparse attention / sparse tiled kernels
- Synthetic kernels designed to isolate specific complement structures

### Baselines
- Hand-written CUTLASS/CuTe kernels
- Triton autotuned kernels
- Manual Sawtooth-style schedules
- Unfused kernels and CuTeFuse-only fused kernels
- Template baselines with bank swizzles but no complement reasoning

### Metrics
- Throughput / latency
- DRAM bytes and L2 hit rate
- Shared-memory bank conflicts
- TMEM slot occupancy / spills / bubbles
- Prediction error for live occupancy and misses
- Schedule-ranking correlation
- Optimization regret
- Compile time and search-space reduction

### Critical experiments
1. **Exactness on SMEM/TMEM**  
   Validate exact occupancy/bank/slot predictions and spill-free feasibility.
2. **Ideal-theorem validation for L2**  
   Use isolated or persistent-kernel settings to test whether same-set reuse distance predicts misses.
3. **Hardware deviation study**  
   Quantify the gap from ideal LRU/set model to real hardware on each platform.
4. **CTA interference study**  
   Vary resident CTA count and synthetic co-runners; validate probabilistic miss bounds.
5. **Honest higher-D study**  
   Compare Morton/Hilbert/Gray/skewed-lexicographic candidates as heuristics, with no optimality claim.
6. **Fusion integration**  
   Show fused attention and one MoE pipeline where on-chip exactness + L2 traversal jointly matter.
7. **Ablation**  
   Remove complement reasoning, interference calibration, fusion integration, or swizzle synthesis and quantify loss.

### Concrete 12–18 month milestones

**Months 0–3**
- Build microbenchmark suite for line/set/way/bank inference
- Implement symbolic \(Q_h/O_h/K_h\) extraction for a CuTe-based kernel subset
- Deliverable: hardware descriptor + synthetic validation kernels

**Months 4–6**
- Implement exact SMEM analyzer and swizzle solver
- Add TMEM abstraction if hardware is available
- Deliverable: exact occupancy/bank-conflict prediction on real kernels

**Months 7–10**
- Implement idealized L2 theorem and 1D/2D traversal synthesis
- Automatically recover Sawtooth on an attention kernel
- Deliverable: theorem-backed L2 scheduler in isolated mode

**Months 11–14**
- Add replacement-slack and CTA-interference calibration
- Evaluate ranking fidelity and regret on real workloads
- Deliverable: robust L2 scheduling under real hardware deviation

**Months 15–18**
- Integrate with CuTeFuse-style fusion for attention and one MoE pipeline
- End-to-end evaluation, ablations, and artifact cleanup
- Deliverable: submission-ready system and paper

This milestone plan is intentionally staged so the paper is still publishable if TMEM or full fusion integration arrives late.

## 7. Target Venue and Why

**Primary target: ASPLOS.**

This revised version is no longer just a formal compiler idea. It is:

- a precise theorem under an explicit cache model,
- an exact on-chip compiler analysis,
- a calibrated study of where real GPUs deviate from that model,
- and a system that synthesizes schedules for important kernels and fused pipelines.

That compiler + architecture + measurement mix fits ASPLOS best.

**Fallback if scope narrows:** CGO  
If the final artifact lands as “exact SMEM + ideal/calibrated L2 + fused attention case study,” CGO would also be a strong fit.

## 8. Potential Weaknesses and Mitigations

- **Weakness: the main theorem is not true for arbitrary real GPU caches.**  
  **Mitigation:** say this explicitly. The theorem is only claimed for \(\mathcal{M}_{L2}^{\mathrm{iso}}\); real hardware is handled by measured deviation bounds.

- **Weakness: undocumented L2 hashing/replacement may make count prediction noisy.**  
  **Mitigation:** optimize for schedule ranking and regret, not only raw miss-count accuracy; infer an effective descriptor from microbenchmarks.

- **Weakness: CTA interference may dominate and swamp schedule effects.**  
  **Mitigation:** model interference explicitly, validate in persistent/isolated modes first, then add robust scheduling and CTA coloring.

- **Weakness: higher-dimensional traversals are not formally solved.**  
  **Mitigation:** restrict formal claims to 1D/2D; present Morton/Hilbert/Gray only as heuristic schedule families.

- **Weakness: fusion may improve reuse but hurt occupancy.**  
  **Mitigation:** fuse only when deterministic live-state analysis says the carrier fits; allow partial fusion/materialization.

- **Weakness: Blackwell/TMEM access risk could delay the full story.**  
  **Mitigation:** make H100/H200 L2+SMEM the publishable core and treat TMEM as a stretch extension.

- **Weakness: scope could still expand too far.**  
  **Mitigation:** keep the first paper focused on read-mostly tensor kernels, 1D/2D complement structures, and two fused pipeline case studies.

---

If you want, I can next turn this into a **1-page extended abstract**, a **paper outline with theorem statements/proof sketches**, or a **rebuttal-style response to the reviewer**.
