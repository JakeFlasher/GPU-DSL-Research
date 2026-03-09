# Research Idea Generation Workflow Plan
## ML Compilers / NVIDIA CuTe Layout / GPU Mathematical Foundations

### Overview
Iterative multi-agent workflow using GPT-5.4:xhigh (via /ask-codex) to generate, review, and refine 5 novel research ideas suitable for top-tier venues (MLSys, ISCA, ASPLOS, MICRO, PLDI).

---

## Phase 1: Idea Generation (5 Parallel Codex Agents)

Each agent receives the FULL context of all 8 papers and is tasked with generating ONE novel research idea from a specific angle:

### Agent 1 — "Layout-Aware Automatic Kernel Fusion"
**Angle**: Combine CuTe's layout algebra (composition, complement, division) with Hexcute's constraint-based synthesis and FlashAttention-4's pipeline co-design insights to propose automatic discovery of fusible kernel sequences through layout compatibility analysis.
**Key papers**: CuTe Algebra, Hexcute, FlashAttention-4

### Agent 2 — "Cross-Architecture Provably-Correct Layout Translation"
**Angle**: Leverage the categorical framework (Tuple/Nest), ISL unification, and F2 linear layouts to propose provably correct, automatic layout translation between different GPU vendors/architectures (NVIDIA, AMD, Intel) and across generations.
**Key papers**: Categorical Foundations, ISL Modeling, Linear Layouts

### Agent 3 — "Formal Cache Hierarchy Optimization via Layout Algebra"
**Angle**: Use Sawtooth's empirical cache analysis + CuTe's formal algebra + Axe's multi-scale abstraction to propose a framework that automatically derives cache-optimal tiling and access patterns directly from layout specifications.
**Key papers**: Sawtooth Wavefront, CuTe Algebra, Axe

### Agent 4 — "Unified Algebraic Framework from Threads to Clusters"
**Angle**: Synthesize Axe's named-axis multi-scale abstraction, CuTe's hierarchical layout algebra, and the categorical/ISL formalisms into a single algebraic framework that spans thread→warp→CTA→cluster→multi-GPU, enabling end-to-end layout optimization.
**Key papers**: Axe, CuTe Algebra, Categorical Foundations, ISL Modeling

### Agent 5 — "Layout Algebra-Driven Hardware-Software Co-Design"
**Angle**: Use FlashAttention-4's Blackwell bottleneck analysis + CuTe algebra + Hexcute synthesis to propose using layout algebra as a formal language for hardware-software co-design — deriving optimal tensor core ISA extensions, memory hierarchy parameters, and interconnect topologies.
**Key papers**: FlashAttention-4, CuTe Algebra, Hexcute

---

## Phase 2: Review (1 Reviewer Codex Agent)

A single reviewer agent receives ALL 5 ideas and evaluates each against:
1. **Novelty**: Is this genuinely new, not incremental?
2. **Feasibility**: Can this be implemented and evaluated?
3. **Significance**: Would top venues (MLSys/ISCA/ASPLOS) accept this?
4. **Technical Soundness**: Are the mathematical foundations solid?
5. **Clarity**: Is the idea well-articulated?

The reviewer provides structured feedback with scores (1-10) and specific improvement suggestions.

---

## Phase 3: Refinement (Iterative Loop)

For each idea that needs improvement:
1. Send reviewer feedback back to the ORIGINAL idea agent (via /ask-codex)
2. Agent refines the idea based on feedback
3. Send refined idea back to reviewer
4. Repeat until reviewer scores ≥ 8/10 on all criteria

### Convergence Criteria
- All 5 ideas score ≥ 8/10 on Novelty, Feasibility, Significance
- Reviewer explicitly states "satisfied" for each idea
- Maximum 3 refinement rounds per idea

---

## Phase 4: Final Output

Consolidated document with 5 polished research proposals, each containing:
- Title
- Abstract (200 words)
- Key Insight / Thesis
- Proposed Methodology
- Expected Contributions
- Target Venue
- Evaluation Plan

---

## Execution Timeline
- Round 0: Launch 5 idea-generation agents in parallel
- Round 1: Launch reviewer agent on all 5 ideas
- Round 2+: Iterative refinement until convergence
