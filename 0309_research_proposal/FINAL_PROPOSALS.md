# Final Research Proposals: ML Compilers, CuTe/CUTLASS Layouts, and GPU Architecture
## Generated via iterative GPT-5.4:xhigh Codex ideation + review loop

---

## Summary & Ranking

| Rank | Title | Venue | Reviewer Verdict | Status |
|------|-------|-------|-----------------|--------|
| 1 | **ATLAS-GEMM**: Proof-Carrying Retargeting & Impossibility Certificates | PLDI | **Strong Accept** | Ready for paper draft |
| 2 | **CuTeFuse**: Sound Legality Certificates for Automatic Fused GPU Pipelines | PLDI | **Weak Accept** | Ready for paper draft |
| 3 | **Cache-as-Layout**: Exact On-Chip Reuse & Probabilistic L2 Bounds | ASPLOS | **Weak Accept** | Ready for paper draft |
| 4 | **ATLAS-RC**: Replica-Caching Duality for Immutable Tensor Tiles | PLDI | Borderline | Needs stronger MoE eval |
| 5 | **Closing the Attention Complement**: Predicting TMEM & Row-Reduction Pipelines | MICRO | Borderline | Needs sensitivity analysis |

---

## Idea 1 (Rank #2): CuTeFuse

**Full title**: CuTeFuse: Sound Legality Certificates and Cache-Aware Traversal for Pure Tiled Producer-Consumer GPU Pipelines

**Core thesis**: For kernels in the PTPC-(L,R) fragment, a boundary is legally eliminable **iff** a legality certificate exists proving: common on-chip carrier, local realizability of residual transfer, synchronization/dependence preservation, certified reduction contracts, and static resource feasibility. Profitability is a separate optimization layer.

**Key contributions**:
- Formal PTPC kernel fragment with sound/complete fusibility certificates
- Clean separation of legality vs profitability
- Failure witnesses explaining why layout-compatible kernels may not be fusible
- Cache-aware traversal integrated via quotient/complement reuse signatures
- Practical compiler targeting attention, MLP, norm, and MoE pipelines

**Target**: PLDI | **Timeline**: 18 months

**Reviewer final note**: "Quantify how much real workload coverage the PTPC fragment buys you."

---

## Idea 2 (Rank #1): ATLAS-GEMM

**Full title**: ATLAS-GEMM: Proof-Carrying Retargeting and Impossibility Certificates for Fused sm_90a WGMMA → gfx942 MFMA Microkernels

**Core thesis**: For FP16 GEMM+bias microkernels, ATLAS either produces a small, independently checkable translation certificate from NVIDIA WGMMA to AMD MFMA, or a small impossibility certificate proving no exact retargeting exists under explicit budgets.

**Key contributions**:
- Precise Obs_exact semantic contract (value + memory effects + exact FP order + sync)
- Tiny 7-op relational-linear tile IR for one real retargeting boundary
- Small, independently checkable translation certificates
- Impossibility certificates with repair labels (+relayout, +barrier, etc.)
- First proof-carrying cross-vendor tensor-core kernel retargeting

**Target**: PLDI | **Timeline**: 15 months

**Reviewer final note**: "Do not broaden the scope again; the narrowness is now a strength."

---

## Idea 3 (Rank #3): Cache-as-Layout

**Full title**: Cache-as-Layout: Exact On-Chip Reuse and Probabilistic L2 Bounds for Synthesizing Hierarchy-Aware GPU Kernels

**Core thesis**: Quotient/complement is the right algebraic interface between layout and reuse, but its meaning depends on hierarchy level. Exact theorems for software-managed SMEM/TMEM; explicit-model theorems for ideal L2 with calibrated probabilistic bounds on real hardware.

**Key contributions**:
- Theorem 1: Exact quotient/complement liveness/conflict for SMEM/TMEM
- Theorem 2: Complementary Reuse Theorem under explicit isolated L2 model (Sawtooth as 2D corollary)
- Calibrated probabilistic deviation bounds via replacement slack + CTA interference
- Joint synthesis of tiling, traversal, and swizzling across hierarchy levels
- Integration with CuTeFuse-style kernel fusion

**Target**: ASPLOS | **Timeline**: 18 months

**Reviewer final note**: "Make the deterministic/probabilistic split the centerpiece of the paper."

---

## Idea 4 (Rank #4): ATLAS-RC

**Full title**: ATLAS-RC: Replica-Caching Duality for Immutable Tensor Tiles

**Core thesis**: For immutable/read-mostly tensor tiles, inter-GPU replication and on-chip caching are the same copy operator instantiated at different hierarchy cuts. This enables a joint optimization: trade GPU replication of hot MoE experts for larger TMEM tiles in local expert kernels.

**Key contributions**:
- Minimal Read-Mostly Copy Calculus over GPU→TMEM hierarchy
- Replica-Caching Duality theorem: copy_at(c1, copy_at(c2, L)) = copy_at(c2, copy_at(c1, L))
- Compiler-useful exchange rule collapsing cross-level search
- One concrete MoE optimization: replication ↔ TMEM tile size tradeoff

**Target**: PLDI | **Timeline**: 15 months

**Reviewer final note**: "Make the MoE optimization decisive, not illustrative."

---

## Idea 5 (Rank #5): Closing the Attention Complement

**Full title**: Closing the Attention Complement: Predicting Next-Generation TMEM and Row-Reduction Pipelines from Layout Analysis

**Core thesis**: CuTe complement/division applied to FlashAttention identifies the non-MMA residual (rowmax, rowsum, exp, rescale, persistent state) as the closure gap. Retrospective analysis shows this would have predicted TMEM from Hopper/FA-3 data. Forward prediction: 384KB TMEM/SM + 1 row-streaming complement port.

**Key contributions**:
- Layout-closure criterion narrowed to attention complement hardware
- Retrospective validation: Hopper/FA-3 bottlenecks predict TMEM-like storage
- Explicit costed forward prediction grounded in FA-4 roofline numbers
- One RTL TMEM/complement block artifact

**Target**: MICRO | **Timeline**: 18 months

**Reviewer final note**: "Add strong sensitivity analysis; if prediction is unstable, narrow the claim further."

---

## Recommended Combinations

The reviewer suggested two strong paper combinations:

1. **Best near-term combo: CuTeFuse + Cache-as-Layout** (Ideas 1+3)
   - Algebraic fusion legality + cache-aware traversal synthesis in one compiler
   - Already partially integrated in the revised proposals

2. **Best PLDI combo: CuTeFuse + ATLAS-GEMM** (Ideas 1+2)
   - Use ATLAS-GEMM's relational-linear certificate machinery as formal core
   - CuTeFuse as the main application demonstrating the formalism

---

## Source Papers

1. CuTe Layout Representation and Algebra (NVIDIA, Cecka, 2603.02298)
2. Categorical Foundations for CuTe Layouts (Colfax, 2601.05972)
3. Linear Layouts over F2 (OpenAI, 2505.23819)
4. Modeling Layout Abstractions Using ISL (NVIDIA, 2511.10374)
5. Axe Layout (CMU/NVIDIA/Princeton, 2601.19092)
6. Hexcute (U Toronto/NVIDIA, 2504.16214)
7. FlashAttention-4 (Princeton/Meta/Colfax/NVIDIA, 2603.05451)
8. Sawtooth Wavefront Reordering (U Rochester, 2601.16032)

---

## Files

All detailed proposals and reviews are in `/ideas/`:
- `idea[1-5]_*.md` — Original proposals
- `idea[1-5]_v2_refined.md` — Revised proposals after reviewer feedback
- `review_round1.md` — First review
- `review_round2.md` — Second review (final)
