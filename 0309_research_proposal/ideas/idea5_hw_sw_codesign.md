## 1. Title

**Architecting for Layout Closure: Inverting CuTe Algebra to Synthesize Tensor-Core ISAs, TMEM Hierarchies, and GPU Networks**

## 2. Abstract

FlashAttention-4 on Blackwell exposes a structural flaw in current accelerator design: tensor-core throughput scaled aggressively, but the complementary resources required by attention—row-wise normalization, staging, persistent tile state, and data movement—did not. The resulting bottlenecks are not accidental; they reflect a deeper mismatch between the workload’s layout algebra and the hardware’s supported resource algebra. I propose **Layout Closure Co-Design**, a framework that treats CuTe/linear/Axe layouts as a formal specification language for both software demand and hardware supply. The key step is to represent an architecture itself as a hierarchical layout over compute units, memories, interconnect, and time, then **invert Hexcute-style synthesis**: given a workload family, solve for the minimal hardware basis whose supported layout operations close the workload under composition, division, complement, and inversion. The solver emits concrete architectural artifacts: tensor-core complement instructions for reductions and normalization, TMEM-like intermediate memories when quotient layouts reveal persistent tile state, bank-swizzle and shuffle networks from bit-level layout gaps, and interconnect topologies from shard/replica axes. This turns layout algebra from a kernel-tuning abstraction into a hardware-specification language, enabling future GPUs to be designed from workload layout requirements rather than post hoc bottleneck debugging.

## 3. Key Insight / Thesis Statement

**A tensor architecture should not be sized around peak MMA throughput; it should be sized so that its hardware layout algebra is closed over the critical morphisms of its target workload family.**  
In this view, CuTe’s **complement/division** operations directly identify the *missing hardware basis*—the exact compute, memory, and communication resources that must exist so the end-to-end kernel can execute without expensive materialization through SMEM, SFUs, L2, or inter-GPU collectives.

## 4. Technical Approach

### A. Represent workload demand as a unified layout algebra

The first step is to lift “layout” from a memory-addressing abstraction to a full **program-demand IR**. I would build a unified representation that combines:

- **CuTe hierarchical layouts** for thread-value, tile, and memory mappings,
- **linear layouts** for bit-level swizzles, XOR bank mappings, and warp-shuffle structure,
- **ISL relations** as the common executable substrate for composition/inversion/complement,
- **Axe named axes** for lane/warp/register/memory/GPU/distribution structure,
- and one extra ingredient: a **temporal axis** representing pipeline stage, wavefront order, and reuse distance.

This gives a layout relation of the form:

\[
L_s : (\text{semantic tensor axes}) \rightarrow (\text{resource axes}, \text{time axes})
\]

for every stage \(s\) in a kernel.

For FlashAttention, the semantic axes are not just \((m,n,k)\), but also row/column tiles, head, sequence block, pipeline stage, and reduction state. That matters because the non-MMA part of attention is not “miscellaneous scalar work”; it is a very specific **quotient layout**: row-wise max/sum/exp/rescale over the axes left over after factoring out the MMA morphisms. Blackwell’s imbalance can therefore be stated algebraically: hardware scaled the bilinear contraction morphism, but not its complement.

### B. Represent hardware itself as a layout family

The central non-incremental move is to treat hardware as a **first-class layout object**. A GPU configuration becomes a parameterized family of layouts:

\[
H(\theta)=\{L_{TC}, L_{REG}, L_{SMEM}, L_{TMEM}, L_{SFU}, L_{NoC}, L_{L2}\}
\]

where \(\theta\) contains microarchitectural parameters such as tensor-core tile shapes, number of reduction/exponential lanes, TMEM bank count/depth, SMEM ports, shuffle-network richness, and interconnect topology.

Examples:

- \(L_{TC}\): maps tensor fragments and epochs to tensor-core pipes/cycles.
- \(L_{TMEM}\): maps persistent accumulator tiles to banks/rows/ports/time.
- \(L_{SFU}\): maps unary/reduction complements to normalization lanes.
- \(L_{NoC}\): maps shard/replica communication to links, VCs, and schedule slots.
- \(L_{L2}\): maps CTA wavefronts to slice reuse structure.

Now software layouts and hardware layouts live in the **same algebra**. This is the key bridge: architecture search becomes a factorization problem in the same language compilers already use for kernel layout synthesis.

### C. Define “layout closure” as the design objective

Let \(\mathcal{W}\) be the closure of the workload’s critical layout morphisms under composition, product, inversion, division, and complement. Let \(\mathcal{B}_\theta\) be the primitive morphisms supported by hardware \(H(\theta)\). A hardware design is **layout-closed** for a workload family if every critical morphism in \(\mathcal{W}\) can be implemented by a short, throughput-balanced factorization through \(\mathcal{B}_\theta\).

The important quantity is the **closure gap**: the minimal complement that must be materialized because current hardware lacks the right primitive. That gap is not just a yes/no property; its *shape* determines hardware parameters:

- the quotient’s reduction extent gives required reduction-tree width,
- the quotient’s pointwise volume gives required exp/unary throughput,
- the quotient’s live state gives required TMEM capacity,
- the producer/consumer division gives SMEM/TMEM bandwidth and bank geometry,
- the shard/replica complement gives required network bisection and multicast support.

For FlashAttention-4, this predicts exactly why “2x more MMA” is insufficient. The attention pipeline’s initiation interval shifts to the **complement basis**: row normalization, state persistence, and tile movement. In other words, the architecture overscaled one generator of the workload algebra while leaving the other generators unchanged.

### D. Invert Hexcute: synthesize hardware from software constraints

Hexcute synthesizes thread/value layouts under fixed hardware constraints. I would invert this. The software layouts are fixed; the hardware parameters become symbolic unknowns.

The solver takes as input:

1. a workload family (e.g., long-context attention, MoE, grouped GEMM, sequence-parallel training),
2. extracted layout relations from CuTe-DSL/Triton/MLIR,
3. area/power/timing constraints,
4. candidate primitive classes for compute/memory/network.

It then solves for \(\theta\) such that the workload algebra closes with minimum cost. Concretely, it chooses:

- number and width of complement lanes per tensor-core cluster,
- whether a TMEM-like level must exist, and its bank/depth/port structure,
- SMEM bank count and swizzle family,
- lane-permute / warp-shuffle network richness,
- collective primitives and GPU topology induced by Axe D/R/O axes.

This is where the categorical and tractability results matter: they give canonical normal forms and keep the search space from exploding. ISL provides exact executability for both stride-based and bit-level transforms. Linear-layout analysis covers bank swizzles and shuffles. Axe extends the same machinery across devices.

### E. The derived ISA: a layout-basis ISA, not a fixed MMA ISA

The next-gen ISA should not expose only one hardwired primitive (`mma`). It should expose a **small layout basis table** plus 3 primitive families:

1. **map**: bilinear/contraction-style tensor operations over a layout ID,
2. **reduce-complement**: reductions/unary-normalization over quotient axes,
3. **reindex/collective**: composition/inversion/permutation/multicast/scatter over layout IDs.

For attention, the synthesized basis would likely include:
- QK contraction,
- rowwise max/sum/exp-rescale recurrence,
- PV contraction,
- accumulator rescale/update,
- persistent-tile moves between TC and TMEM.

This is still hardware-friendly because the basis is **compiled ahead of time** for a workload family; it is not an unrestricted dynamic descriptor machine.

### F. How TMEM fits naturally

In this framework, **TMEM is not an ad hoc Blackwell feature**. It is the inserted object that appears when the quotient between producer and consumer layouts has:
- high temporal reuse,
- tensor-fragment structure aligned with TC tiles,
- too much live state for registers,
- and too much layout mismatch to efficiently materialize through SMEM.

So TMEM is algebraically derivable: it is the cheapest storage object that restores closure between compute and complement stages.

### G. Beyond attention: interconnect and cache topology

Using Axe’s shard/replica/offset decomposition, the same solver can derive communication hardware. For MoE, if the quotient over the GPU axis is an all-to-all transpose, the solver may prefer a topology and collective engine optimized for expert routing rather than generic NCCL-style collectives. For cache behavior, Paper 8’s sawtooth schedule can be modeled as a **temporal layout transform**; if the time-axis complement dominates, the framework can synthesize reorder queues or L2 slice-aware prefetch support.

## 5. Expected Contributions

- **A new formalism:** hardware architectures represented as hierarchical layouts in the same algebra as CuTe/Triton programs.
- **Layout closure theory for accelerators:** a definition of closure gap that maps directly to missing compute, memory, and interconnect resources.
- **Inverse Hexcute synthesis:** a solver that infers hardware constraints and architectural knobs from workload layout requirements.
- **A new ISA/microarchitecture template:** layout-basis tensor cores with complement pipelines, derived TMEM, and layout-aware collectives.
- **A compelling case study:** showing that FlashAttention-4’s Blackwell bottlenecks and TMEM-like remedies emerge automatically from algebraic analysis.

## 6. Evaluation Plan

**Prototype stack**
- Extend CuTe/ISL-based layout extraction from CuTe-DSL and Triton kernels.
- Build a closure-gap analyzer and inverse synthesis solver.
- Feed outputs into a cycle model plus area/power estimates.
- Implement at least one RTL prototype: a tensor-core complement pipe + TMEM controller.

**Workloads**
- FlashAttention-2/4 variants, long-context attention, GQA/MQA.
- MoE inference/training kernels.
- Grouped GEMM and fused epilogues.
- Sequence/expert/data-parallel multi-GPU kernels.

**Baselines**
- Hopper-like and Blackwell-like configurations.
- Manually balanced “more SFU / more SMEM” variants.
- Conventional DSE driven by op counts rather than layout closure.

**Questions**
1. Does closure gap predict real bottlenecks on existing kernels?
2. Can the framework rediscover human-designed features like TMEM or 2-CTA style persistence?
3. Under fixed area/power, do synthesized architectures outperform heuristic scaling?
4. Do synthesized features generalize across workload families, or overfit?

**Metrics**
- Throughput, energy, area efficiency, utilization balance, L2 miss rate, compiler mapping quality, and solver time.

## 7. Target Venue and Why

**ASPLOS**

This idea is strongest as an ASPLOS paper because it is inherently **cross-layer**: it introduces a formal language/IR, a compiler-style synthesis method, and a concrete hardware co-design story. ISCA/MICRO would like the architecture, PLDI would like the algebra, and MLSys would like the workload impact—but ASPLOS is where the full “layout algebra as a hardware specification language” thesis fits most naturally.

## 8. Potential Weaknesses and Mitigations

- **Weakness: overfitting to attention-heavy workloads.**  
  **Mitigation:** optimize over a workload family set with held-out kernels; report Pareto fronts, not one-point designs.

- **Weakness: layout algebra may miss dynamic control flow or sparsity effects.**  
  **Mitigation:** augment layouts with profiled guards/probabilistic weights and validate against dynamic traces.

- **Weakness: area/timing predictions could be too approximate.**  
  **Mitigation:** synthesize at least the complement pipe/TMEM path to RTL and calibrate the analytical model.

- **Weakness: a layout-basis ISA may seem too radical for adoption.**  
  **Mitigation:** compile layout descriptors to a small static basis table and show backward-compatible lowering to existing tensor-core pipelines.

- **Weakness: search space may explode.**  
  **Mitigation:** use categorical normal forms, tractable-layout restrictions, and hierarchical decomposition by compute/memory/network axes.

If you want, I can also turn this into a **1-page extended abstract** or a **full 3–4 page workshop-style proposal**.
