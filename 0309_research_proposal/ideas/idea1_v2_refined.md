Below is the **complete revised 8-section proposal**.

---

## 1. Title

**CuTeFuse: Sound Legality Certificates and Cache-Aware Traversal for Pure Tiled Producer-Consumer GPU Pipelines**

---

## 2. Abstract

Current GPU fusion systems largely decide fusibility from operator templates, and they often conflate **correctness** with **performance**. This proposal narrows scope to a kernel class where fusion can be reasoned about rigorously: **pure tiled producer-consumer (PTPC) kernels with explicit synchronization semantics**. In CuTeFuse, each kernel boundary is represented as a layout transducer from logical tile coordinates to cooperative-group, on-chip storage, and stage coordinates. For an intermediate tensor \(X\), CuTeFuse uses CuTe division and complement to derive a **legality certificate** proving that producer and consumer factor through a common on-chip carrier and that the residual transfer is realizable using a target-specific local transform basis (register permutation, warp shuffle, shared-memory transpose/swizzle, TMEM relay, barriers, and optionally fixed-size cluster exchange). The certificate also includes dependence, synchronization, reduction-contract, and static resource checks.

The central claim is now deliberately weaker and sharper: **within the PTPC-\((\mathcal L,\mathcal R)\) fragment, the legality certificate is sound and complete for boundary elimination without materialization**; outside that fragment it is conservative, not complete. Profitability is handled by a separate optimization layer over only those regions already proven legal. To address reuse, CuTeFuse also incorporates a restricted **cache-aware traversal** module: quotient/complement reasoning over outer schedule axes produces reuse signatures and a small family of legal wavefront/serpentine traversals, used only for cost modeling and schedule selection, not for the soundness theorem. The result is a feasible compiler research agenda that can both discover profitable fusions and explain precisely why apparently layout-compatible kernels still should not be fused.

---

## 3. Key Insight / Thesis Statement

**Revised thesis:** For kernels in the **PTPC-\((\mathcal L,\mathcal R)\)** fragment, a boundary is legally eliminable **iff** there exists a **legality certificate** proving:

1. a common on-chip carrier layout exists,
2. producer-to-consumer residual motion is realizable by local transforms in \(\langle \mathcal L\rangle\),
3. synchronization and dependence order are preserved,
4. reduction semantics satisfy a declared certified contract in \(\mathcal R\), and
5. static register/shared-memory/TMEM bounds are respected.

**Whether that legal fusion should be chosen is a separate profitability problem**, guided by occupancy, synchronization overhead, and cache-aware traversal/reuse estimates.

This replaces the old overly broad “layout-compatible iff fusible” claim with a **restricted but defensible relative iff theorem**.

---

## 4. Technical Approach (detailed)

### 4.1 Restricted kernel class: the PTPC fragment

CuTeFuse will target a formally defined fragment:

### **PTPC-\((\mathcal L,\mathcal R)\)**
A kernel family is in this fragment if:

1. **Pure tiled producer-consumer structure**  
   Intermediates are SSA-like values: no aliasing, no atomics, no side effects except declared outputs.

2. **Static tile domains and layouts**  
   Tile domains are finite Presburger sets, and boundary layouts are affine or piecewise-affine CuTe/Axe-style maps.

3. **Explicit synchronization semantics**  
   The IR contains barrier/fence/stage events explicitly. Communication occurs only within a fixed cooperative group: a CTA, warpgroup, or a fixed-size cluster with hardware-supported synchronization.

4. **Local communication basis**  
   Residual transfer must be expressible using a target-specific local transform library \(\mathcal L\), e.g.  
   \[
   \mathcal L = \{\text{identity, reg permute, warp shuffle, smem transpose/swizzle, TMEM relay, barrier, optional cluster exchange}\}.
   \]

5. **Certified reduction contracts**  
   Reductions are allowed only if they preserve either:
   - the same reduction tree as the unfused form, or
   - a certified summary-state algebra from \(\mathcal R\) (e.g. online softmax state, rowwise sum/max, Welford).

This scope cut directly answers the reviewer’s core question: **soundness and completeness are claimed only for this restricted fragment.**

---

### 4.2 Legality certificates, not profitability certificates

For an intermediate tensor \(X\) between producer \(K_p\) and consumer \(K_c\), define layout transducers

\[
P_X, C_X : D_X \rightarrow G \times M \times T
\]

where \(D_X\) is the logical tile domain, \(G\) is cooperative-group identity, \(M\) is on-chip storage coordinate, and \(T\) is stage/time.

A **legality certificate** is

\[
\mathcal C_{\text{legal}}(X) = (B_X, U_p, U_c, \delta, \sigma, \rho).
\]

It proves:

1. **Common carrier**  
   There exists an on-chip carrier layout \(B_X\) such that CuTe division/complement yields residual maps \(U_p,U_c\) with
   \[
   P_X = U_p \circ B_X,\qquad C_X = U_c \circ B_X.
   \]

2. **Local realizability**  
   The residual transfer is implementable locally:
   \[
   U_c \circ U_p^{-1} \in \langle \mathcal L \rangle.
   \]

3. **Dependence and sync preservation**  
   \(\delta\) embeds producer release events to consumer acquire events while preserving the original partial order.

4. **Stage schedule**  
   \(\sigma\) assigns tiles to pipeline stages with explicit barriers/fences.

5. **Static resource feasibility**  
   \(\rho\) proves that all live ranges fit architectural bounds:
   registers, shared memory, TMEM, and required residency.

This certificate is **only about correctness/legal realizability**. It says nothing about whether fusion is faster.

---

### 4.3 Soundness and relative completeness

The proposal’s central theorem becomes:

> **Relative iff theorem.**  
> For PTPC-\((\mathcal L,\mathcal R)\) kernels, a boundary \(X\) is eliminable without materialization **iff** a legality certificate \(\mathcal C_{\text{legal}}(X)\) exists.

More precisely:

- **Soundness:** if a certificate exists, CuTeFuse can synthesize a fused kernel whose visible outputs and memory effects refine the unfused semantics.
- **Relative completeness:** if a PTPC fused implementation exists whose communication uses only transforms in \(\langle \mathcal L\rangle\) and reductions in \(\mathcal R\), then a certificate can be extracted.

Outside PTPC-\((\mathcal L,\mathcal R)\)—for dynamic permutations, general cross-CTA irregularity, unsupported reductions, or side-effectful kernels—the system remains **sound but conservative**.

---

### 4.4 Failure cases: where layout compatibility is necessary but not sufficient

A major revision is that CuTeFuse will explicitly model and report **failure witnesses**. Examples:

1. **Resource-feasible layout, but illegal fusion**  
   A producer and consumer may share a carrier layout, yet keeping producer accumulators, intermediate state, and consumer fragments live simultaneously may exceed register or TMEM bounds.  
   **Witness:** `resource_overflow(reg/smem/tmem)`.

2. **Layout-compatible, but synchronization-inexpressible**  
   A consumer may need values produced by another CTA or split-K partition without an allowed cooperative barrier/exchange primitive.  
   **Witness:** `sync_gap(cross_group)`.

3. **Layout-compatible, but reduction semantics mismatch**  
   Naive tilewise softmax is not equivalent to globally correct softmax unless an online summary-state contract is used.  
   **Witness:** `reduction_contract_missing`.

4. **Layout-compatible, but incompatible fan-out/lifetime**  
   One intermediate may feed two consumers with incompatible stage orders or lifetimes.  
   **Witness:** `lifetime_conflict(fanout)`.

5. **Legal, but should still not be fused**  
   Fusion may be correct yet reduce occupancy or destroy outer-cache reuse.  
   **Witness:** not a legality failure, but a profitability rejection: `negative_profit`.

This directly answers the reviewer’s request to show that **layout compatibility is necessary, but not sufficient**.

---

### 4.5 Profitability is a separate layer

Once legality certificates are computed, CuTeFuse builds a **legal fusion hypergraph**:

- nodes: operator instances with candidate lowerings,
- legal hyperedges: only regions whose internal boundaries all have legality certificates.

A separate cost model then selects among legal regions using features such as:

- HBM bytes eliminated,
- local transform cost,
- stage/barrier count,
- occupancy penalty,
- tensor-core issue efficiency,
- live-range pressure margin,
- cache-aware traversal score.

So the compiler obeys a strict rule:

- **No certificate ⇒ fusion is illegal.**
- **Certificate exists ⇒ fusion is legal, but may still be rejected as unprofitable.**

For linear chains, region selection can be solved exactly with DP. For general DAGs, CuTeFuse will use bounded-search hypergraph optimization.

---

### 4.6 Incorporating cache-aware traversal without overclaiming

The near-term combination with the strongest payoff is exactly what the reviewer suggested: **fusion + cache-aware traversal**.

For a legal fused region and hierarchy level \(h\), define

\[
A_h = H_h \circ L_{\text{data}} \circ L_{\text{sched}},\quad
Q_h = A_h \div T_h,\quad
K_h = \operatorname{comp}(Q_h).
\]

Interpretation:

- \(Q_h\): what remains resident at level \(h\),
- \(K_h\): which traversal directions cause eviction/revisit.

CuTeFuse will use this in a deliberately limited way:

- For **registers/shared memory/TMEM**, quotient/complement yields hard lifetime and residency constraints.
- For **L2**, it yields a **reuse signature** used only in the profitability model and schedule search.

To keep the work feasible and technically sound, traversal synthesis will be restricted to a small family:

- monotone,
- serpentine/parity-reversed,
- wavefront/skewed,
- a few low-bit swizzles.

This is enough to capture the key near-term cases—especially attention-style KV reuse—without overclaiming a full cache-optimality theorem.

---

### 4.7 Compiler realization

Implementation will reuse existing ecosystem pieces rather than inventing a full stack from scratch:

- **Front-end IR:** a PTPC subset extracted from CuTe/CUTLASS-style kernels and a restricted Triton lowering.
- **Certificate engine:** canonicalize layouts, compute common carriers via CuTe division/complement, and emit independently checkable legality certificates or failure witnesses.
- **Planner:** choose profitable legal regions and traversal orders.
- **Backend:** synthesize fused kernels via CUTLASS/CuTe templates first; Triton backend second.

The initial scope will focus on:

- GEMM → pointwise → GEMM,
- GEMM → rowwise reduction/norm,
- \(QK^T \rightarrow\) online softmax \(\rightarrow PV\),
- selected bucketized/static-routing MoE pipelines.

---

## 5. Expected Contributions

1. **A formal restricted kernel class** for GPU fusion: PTPC-\((\mathcal L,\mathcal R)\).
2. **A legality-certificate abstraction** for fusion, with a precise soundness/completeness claim inside that class.
3. **A clean separation of legality and profitability**, avoiding the prior overclaim.
4. **Failure witnesses** explaining why layout-compatible kernels may still be non-fusible.
5. **A cache-aware traversal layer** that combines fusion with reuse-sensitive schedule selection without making it part of the soundness theorem.
6. **A practical compiler prototype** capable of rediscovering a meaningful subset of hand-optimized fused pipelines.

---

## 6. Evaluation Plan and 12–18 Month Implementation Plan

### Evaluation

**Workloads**
- GEMM → bias/GELU → GEMM
- GEMM → RMSNorm/LayerNorm → residual
- \(QK^T \rightarrow\) online softmax \(\rightarrow PV\)
- selected MoE routing/packing/grouped-GEMM pipelines with restricted routing assumptions

**Hardware**
- Primary: H100
- Extension: Blackwell/B200 if available

**Baselines**
- TorchInductor/Triton fusion
- CUTLASS/cuBLASLt epilogue fusion
- TensorRT/XLA-style graph fusion
- Hand-tuned kernels where available (e.g. FlashAttention family)

**Metrics**
- throughput / latency
- HBM bytes
- kernel count / launch overhead
- occupancy, registers, smem, TMEM
- L2 hit rate / reuse counters
- compile time
- certificate success/failure breakdown

**Ablations**
- no legality certificates
- legality without failure witnesses
- legality + cost, but no cache-aware traversal
- no reduction contracts
- no cluster/TMEM extension
- no global region planner

### 12–18 month plan

**Months 1–3:**  
Define PTPC IR, explicit stage/event semantics, and canonical layout normalization.  
**Milestone M1:** legality checker for pairwise non-reduction boundaries.

**Months 4–6:**  
Implement legality certificates and failure witnesses for pointwise and simple rowwise reductions.  
**Milestone M2:** correct accept/reject behavior on GEMM → pointwise and GEMM → norm microbenchmarks.

**Months 7–9:**  
Generate fused single-CTA kernels using registers/shared memory; add certified reduction contracts (rowwise max/sum, online softmax, Welford).  
**Milestone M3:** end-to-end legal synthesis for attention microkernels and GEMM → norm chains.

**Months 10–12:**  
Add profitability layer and global planning for chains; integrate cache-aware traversal scores and restricted wavefront/serpentine schedule synthesis.  
**Milestone M4:** measurable wins on H100 over template fusion baselines.

**Months 13–15:**  
Optional extension to fixed-size cluster / TMEM / Blackwell-specific primitives if hardware is available.  
**Milestone M5:** partial rediscovery of FA-style multi-stage pipelines.

**Months 16–18:**  
Full evaluation, artifact cleanup, theorem writeup, and paper submission.  
**Fallback if Blackwell is unavailable:** submit a strong H100 paper with the cluster/TMEM path framed as future extension.

This plan is intentionally realistic: the core paper does **not** depend on full Blackwell coverage.

---

## 7. Target Venue and Why

**Primary target: PLDI.**

Why this is now stronger for PLDI:

- the formal core is sharper,
- the theorem is properly scoped,
- legality is separated from profitability,
- failure explanations are first-class,
- the implementation is substantial but still believable.

**Backup target:** ASPLOS, if the final paper becomes more systems/performance-heavy than proof-heavy.

---

## 8. Potential Weaknesses and Mitigations

- **Weakness: scope is narrower than the original vision.**  
  **Mitigation:** make that a virtue; PTPC still covers important kernels in attention, MLP, norms, and some MoE paths.

- **Weakness: completeness is only relative to \((\mathcal L,\mathcal R)\).**  
  **Mitigation:** state this explicitly and make transform/reduction libraries extensible.

- **Weakness: L2/cache reasoning is approximate.**  
  **Mitigation:** keep cache-aware traversal out of the legality theorem; use it only for schedule ranking.

- **Weakness: some dynamic MoE/top-k patterns remain outside the fragment.**  
  **Mitigation:** start with bucketized/static variants and emit conservative failure witnesses otherwise.

- **Weakness: global region selection on DAGs is still hard.**  
  **Mitigation:** exact optimization on chains, bounded heuristic search on DAGs, with legality certificates drastically pruning the space.

- **Weakness: hardware-specific primitives may limit portability.**  
  **Mitigation:** parameterize the framework by \(\mathcal L\); core results use reg/shuffle/smem on H100, while TMEM/cluster exchange are optional extensions.

---

If you want, I can next turn this into either a **PLDI-style extended abstract**, a **2-page project pitch**, or a **paper outline with theorem statements and proof sketches**.
