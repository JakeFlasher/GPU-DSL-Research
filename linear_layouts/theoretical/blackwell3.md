# Blackwell “Tensor Hopper”-style operator analysis: a reproducible workflow + initial results for B100 / B200 / “GH10”

## Quick clarification: “GH10” vs “GB10”
I can’t find an NVIDIA **“GH10”** Blackwell GPU designation in publicly accessible materials. What NVIDIA *does* ship and document is **GB10 Grace Blackwell Superchip**, used in **NVIDIA DGX Spark** (formerly “Project DIGITS”). ([nvidia.com](https://www.nvidia.com/en-us/products/workstations/dgx-spark/?utm_source=openai))  

Below, I’ll treat your “GH10” as **GB10** (DGX Spark). If you meant something else (e.g., GB200, GH200, or a China export SKU), tell me and I’ll adapt the workflow.

---

# 1) Analysis workflow (rigorous, “don’t miss anything”)

This section is the “lab manual”: it’s meant to be followed step-by-step and to produce a professional, reproducible report.

## 1.1 Define *exactly* what you will report (scope + metrics)
**Deliverable:** a one-page “scope & definitions” section, so later numbers don’t get mixed.

1) **Targets / SKUs**
   - Datacenter: **B100**, **B200**
   - Desktop superchip: **GB10 (DGX Spark)**
   - (Optional) also track: HGX baseboards (8× GPUs), DGX systems, GB200 superchips.

2) **Ops you will analyze (minimum set)**
   - **Dense GEMM** (training/inference core)
   - **Attention block** (FlashAttention-style)
   - **Bandwidth-bound kernels** (STREAM-like)
   - **New Blackwell-specific engines**: TMEM + tcgen05 pipeline; Decompression Engine (DE)

3) **Primary metrics (must be consistent everywhere)**
   - Peak compute: FP16/BF16, FP8, FP4 (dense and/or sparsity where relevant)
   - Peak memory bandwidth: HBM (or LPDDR for GB10), plus *sustained* bandwidth
   - **PAI (Peak Arithmetic Intensity)** in FLOPs/Byte
   - Instruction micro-metrics:
     - **single-instruction latency** (dependency chain)
     - **reciprocal throughput** (steady-state issue rate)
   - Kernel-level metrics:
     - achieved TFLOPS / %peak
     - achieved BW / %peak
     - occupancy / warp residency
     - pipeline stall breakdown (Nsight Compute)

---

## 1.2 Build a “Golden Facts Table” with provenance
**Deliverable:** `facts.csv` (or markdown table) where every number has:
- SKU name
- value
- unit
- “dense vs sparsity”
- data type
- source link + date
- notes (e.g., “DGX system reports 180GB/GPU; other sources say 192GB”)

**Rule:** if two reputable sources disagree, keep **both**, annotate why it might differ (rounding, usable vs physical memory, early vs shipping specs, etc.).

Example of a real discrepancy you *must* record:
- Some sources describe B200 with **192GB HBM3e**, while NVIDIA DGX B200 system specs total **1,440GB**, implying **180GB/GPU**. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

---

## 1.3 Roofline-first pass (hardware-level sanity check)
**Deliverable:** a roofline section + a small script/notebook that computes PAI for each SKU.

1) Choose “compute peak” consistently (e.g., FP16 dense Tensor Core).
2) Choose memory peak consistently (HBM peak bandwidth, not NVLink).
3) Compute:
\[
\text{PAI}=\frac{\text{Peak FLOP/s}}{\text{Peak Byte/s}}\;\;[\text{FLOPs/Byte}]
\]
4) Interpret:
- PAI is a *hardware* tipping point: kernels with arithmetic intensity below it are *more likely* bandwidth-bound (at that memory level).

---

## 1.4 Decompose peak compute into microarchitecture knobs (what you did for Hopper)
**Deliverable:** a section that explains “where peak TFLOPS comes from”, and a plan to fill missing parameters.

Use:
\[
\text{Peak TFLOPS} = N_\text{(Tensor engines)} \times \frac{\text{FLOPs}}{\text{engine}\cdot\text{cycle}} \times f_\text{clock}
\]

For Blackwell, you usually **won’t** have all pieces from marketing specs (especially clocks, per-SM organization, etc.). So your workflow must include **two ways** to fill gaps:

### Path A — documentation extraction
- NVIDIA architecture briefs / whitepapers / product briefs (datacenter + RTX)
- CUDA/PTX ISA docs for Blackwell
- vendor system guides (DGX, HGX)
- reputable press with tables sourced from NVIDIA slides (AnandTech-level)

### Path B — inference from measurement
- Measure sustained TC throughput on a single SM and full GPU
- Measure clocks during kernel execution (lock clocks if possible)
- Infer effective “FLOPs/cycle” per SM from:
\[
\text{FLOPs/cycle} \approx \frac{\text{Measured FLOPs/s}}{f_\text{SM}\cdot N_\text{SM}}
\]
(where \(f_\text{SM}\) is the real execution clock during the benchmark, not the boost headline)

---

## 1.5 Instruction-level microbenchmarking (Blackwell-specific)
**Deliverable:** `tcgen05_latency.csv`, `tcgen05_throughput.csv`, plus methodology text.

Blackwell changes the “core object” of analysis:
- Hopper: **wgmma** (warp-group, 128 threads)
- Blackwell: **tcgen05.mma** (warp-level scope; issued as **single-thread instructions** enabling per-thread scheduling) ([arxiv.org](https://arxiv.org/html/2512.02189v1))

### What to measure (minimum)
For each **tile shape** you care about and each **precision mode**:
1) **Single-instruction latency (SI-LAT)**
   - Create a true dependency chain so the compiler can’t overlap.
2) **Reciprocal throughput**
   - Many independent instructions to fill the pipeline.
3) **Accumulator mode effect**
   - FP16 accumulate vs FP32 accumulate (often halves throughput on Blackwell, per published microbench data). ([arxiv.org](https://arxiv.org/html/2512.02189v1))
4) **Operand placement**
   - A/B from SMEM vs TMEM
   - Accumulators in TMEM (typical on Blackwell)

### Required controls (don’t skip)
- Fix problem size to avoid launch overhead dominating
- Warm-up runs
- Lock clocks if possible
- Use `asm volatile` / SASS validation to ensure the expected instruction is emitted
- Record driver + CUDA version

---

## 1.6 Memory hierarchy microbenchmarking (must include TMEM)
**Deliverable:** `mem_latency.csv`, `mem_bw.csv`, `tmem_bw.csv`, plus a brief model.

You need at least:
- HBM bandwidth (sustained STREAM-like)
- L2 size/behavior (miss latency vs hit)
- SMEM bandwidth/latency
- **TMEM characteristics** (new on Blackwell)

Published microbenchmark work reports:
- **TMEM is 256KB per SM**
- and provides very high on-chip bandwidth (**16 TB/s read**, **8 TB/s write per SM**) and is described as additive with L1/SMEM rather than competing. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

Whether the “per SM” interpretation matches your own measurements is exactly why this stage exists.

---

## 1.7 Build the “tipping point” models (operator-level, like your Hopper wgmma analysis)
**Deliverable:** For each core op (GEMM, attention subkernel, etc.), a small algebraic model that compares:
- compute time vs
- data-movement time (at the *relevant* memory tier)

For Hopper you compared wgmma compute cycles vs SMEM load cycles.  
For Blackwell you will often compare:
- **tcgen05.mma compute** vs
- **tcgen05.cp / tcgen05.ld/st data movement** into TMEM/SMEM vs
- HBM fetch (when operands don’t fit on-chip)

---

## 1.8 Extend beyond GEMM (required by your prompt)
**Deliverable:** at least two more “template-applied” analyses.

Recommended Blackwell-specific extensions:

1) **FlashAttention / attention block**
   - Use software pipelining + warp specialization methodology.
   - There is recent work that explicitly targets Hopper and Blackwell, showing the importance of jointly optimizing SWP and WS schedules for FlashAttention-like kernels. ([arxiv.org](https://arxiv.org/abs/2512.18134?utm_source=openai))

2) **Decompression Engine (DE)**
   - Blackwell adds a dedicated DE; published microbench work reports throughput/latency behaviors and suggests it shifts optimal design for data analytics pipelines. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

3) (Optional) **Sparse SpMV with hardware decompression** as a case study.

---

## 1.9 Validation loop (to keep it academic)
**Deliverable:** “prediction vs measurement” plots and error discussion.

For each model:
- predict whether compute- vs memory-bound
- predict the scaling trend vs tile size / batch size
- measure actual kernel
- explain mismatches (e.g., scheduling, contention, register pressure, bank conflicts, etc.)

---

# 2) Theoretical analysis (Blackwell B100 / B200 / GB10)

## 2.1 Hardware snapshot (peaks + bandwidths)

Below are *publicly reported* peak figures sufficient to do the same “roofline-style” first pass as your Hopper markdown.

### Datacenter Blackwell (B100 / B200)
A widely circulated spec table (from reputable press coverage of NVIDIA’s Blackwell announcement) lists for **both B100 and B200**:
- **HBM3e bandwidth:** 8 TB/s  
- **Memory:** 192GB  
- **FP16 dense tensor:** B100 1.8 PFLOPS, B200 2.2 PFLOPS (often rounded; some vendor guides show 2.25 PFLOPS for B200) ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data?utm_source=openai))

For B200 specifically, an HGX B200 product guide reports:
- **FP16 Tensor Core:** 2.25 / 4.5 PFLOPS (dense / sparsity)
- **HBM3e bandwidth:** 7.7 TB/s (this is slightly below “8 TB/s” headline figures) ([lenovopress.lenovo.com](https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu?utm_source=openai))

### GB10 (DGX Spark)
NVIDIA DGX Spark specs report:
- **FP4 tensor performance:** up to 1 PFLOP (noted as theoretical with sparsity)
- **Unified system memory bandwidth:** 273 GB/s
- **System memory:** 128GB LPDDR5x ([nvidia.com](https://www.nvidia.com/en-us/products/workstations/dgx-spark/?utm_source=openai))

---

## 2.2 Peak Arithmetic Intensity (PAI) calculations (same method as Hopper)

### Definition
\[
\text{PAI}=\frac{\text{Peak FLOP/s}}{\text{Peak Byte/s}}
\]

### B100 (FP16 dense Tensor Core)
Using **1.8 PFLOPS** FP16 dense and **8 TB/s** HBM bandwidth: ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data?utm_source=openai))
\[
\text{PAI}_{\text{B100, FP16}}=\frac{1.8}{8}=0.225\;\text{PFLOP/TB} = 225\;\text{FLOPs/Byte}
\]

### B200 (FP16 dense Tensor Core)
Using **2.25 PFLOPS** FP16 dense and **8 TB/s** headline bandwidth (or 7.7 TB/s effective): ([lenovopress.lenovo.com](https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu?utm_source=openai))
- With 8 TB/s:
\[
\text{PAI}_{\text{B200, FP16}}=\frac{2.25}{8}=281.25\;\text{FLOPs/Byte}
\]
- With 7.7 TB/s:
\[
\text{PAI}_{\text{B200, FP16}}=\frac{2.25}{7.7}\approx 292.21\;\text{FLOPs/Byte}
\]

### GB10 (DGX Spark) (FP4 headline)
Using **1 PFLOP FP4** and **273 GB/s** bandwidth: ([nvidia.com](https://www.nvidia.com/en-us/products/workstations/dgx-spark/?utm_source=openai))
\[
\text{PAI}_{\text{GB10, FP4}}=\frac{1}{0.273}\approx 3.663\;\text{PFLOP/TB}=3663\;\text{FLOPs/Byte}
\]

**Interpretation caution:** this GB10 figure is based on a *sparsity-enabled FP4 peak*, so treat it as an upper bound for a “headline roofline,” not a guaranteed sustained point. ([nvidia.com](https://www.nvidia.com/en-us/products/workstations/dgx-spark/?utm_source=openai))

---

## 2.3 “Peak TFLOPS is made of what?” (how to do the Hopper-style decomposition on Blackwell)

### The formula (same as Hopper)
\[
\text{Peak TFLOPS} = N_\text{tensor} \times \frac{\text{FLOPs}}{\text{tensor}\cdot\text{cycle}} \times f_\text{clock}
\]

### What’s different on Blackwell: the instruction model
Blackwell changes the Tensor Core programming model and pipeline:
- It introduces **tcgen05.mma** PTX instructions (warp-level scope), described as breaking from Hopper’s **wgmma** warp-group approach; and it introduces **TMEM** as a dedicated tensor memory tier. ([arxiv.org](https://arxiv.org/html/2512.02189v1))
- The published microbenchmark analysis explicitly contrasts Hopper’s warp-group wgmma with Blackwell’s warp-level tcgen05. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

### Practical workflow to fill unknown microparameters (B100/B200)
Because public sources often omit “tensor cores per SM” and real clocks:
1) Use known **SM count** when available (B200: reported as **148 SMs** in published analysis). ([arxiv.org](https://arxiv.org/html/2512.02189v1))
2) Measure real clocks under load (lock if possible).
3) Measure sustained tcgen05 throughput → infer effective FLOPs/cycle per SM.
4) Cross-check against vendor peak PFLOPS.

---

## 2.4 Instruction-level analysis (the “wgmma latency table” equivalent for Blackwell)

### 2.4.1 SASS mapping: tcgen05 is precision-specific
Published instruction mapping for Blackwell tcgen05.mma indicates precision-dependent SASS families, including **OMMA** for FP4, which is described as new to Blackwell (wgmma is Hopper-only). ([arxiv.org](https://arxiv.org/html/2512.02189v1))

### 2.4.2 Latency: Hopper wgmma scales with tile width; Blackwell tcgen05 is ~constant
A key microarchitectural difference that directly affects “operator intuition”:

- **Hopper wgmma** (warp-group) single-instruction latency examples:  
  m64n64k16 → 32 cycles, m64n128k16 → 64, m64n256k16 → 128  
- **Blackwell tcgen05.mma** (warp) single-instruction latency:  
  m64n64k16 → ~11 cycles, m128n128k16 → ~11.3, m256n256k16 → ~11.4 ([arxiv.org](https://arxiv.org/html/2512.02189v1))

**Why this matters for “Hopper-style threshold reasoning”:**
- On Hopper, increasing \(N\) could visibly increase the instruction’s latency.
- On Blackwell, tile size is reported to affect throughput more than latency, which suggests the old “latency doubles at \(N\ge \dots\)” mental model may no longer apply cleanly. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

### 2.4.3 Throughput & accumulator choice: FP32 accumulate can cost ~2×
A published characterization table shows that switching to FP32 accumulation can roughly halve throughput (example shown for FP16 input). ([arxiv.org](https://arxiv.org/html/2512.02189v1))

**Operator-level takeaway:** on Blackwell, “accumulator precision” is a first-class performance knob, not a minor detail.

---

## 2.5 Memory-system change you must incorporate: TMEM
Your Hopper markdown centered on SMEM bandwidth and wgmma’s loading behavior.  
For Blackwell, **TMEM** changes that entire picture:

Published microbenchmark analysis reports:
- TMEM is a **dedicated 256KB on-chip memory per SM** for tensor operations.
- It reports very high **TMEM bandwidth** (16 TB/s read, 8 TB/s write per SM) and states it operates additively with L1/SMEM bandwidth rather than competing. ([arxiv.org](https://arxiv.org/html/2512.02189v1))
- It also reports improved end-to-end cache-miss access latency figures (TMEM-access path discussed as ~420 cycles vs ~1000-cycle Hopper global latency in that analysis). ([arxiv.org](https://arxiv.org/html/2512.02189v1))

**Practical implication:** for many “tensor-heavy” kernels, the dominant optimization problem shifts from “how to feed wgmma from SMEM efficiently” to “how to stage/retain intermediate tensors in TMEM and overlap tcgen05.cp data movement with tcgen05.mma compute.”

---

# 3) Practice guidance (Blackwell-appropriate analogs to the Hopper SwapAB story)

Your Hopper markdown’s key “action item” was:  
> use latency/throughput analysis to predict whether a layout change (SwapAB) will help on a specific SKU.

For Blackwell, the *spirit* stays the same, but the *mechanism* changes.

## 3.1 Reframing SwapAB for Blackwell
### What SwapAB was exploiting on Hopper
On Hopper, wgmma latency/throughput behavior could change sharply with \(N\), so making the “effective \(N\)” larger (via transposition / SwapAB) could push you into a better regime.

### Why Blackwell is different
Published results indicate Blackwell’s tcgen05.mma single-instruction latency is near-constant across much larger tile sizes than Hopper wgmma. ([arxiv.org](https://arxiv.org/html/2512.02189v1))  
So if your “SwapAB thesis” was primarily about *reducing the number of small-\(N\) instructions whose latency balloons*, that lever may be weaker on Blackwell.

**However, SwapAB can still matter** for reasons that remain valid:
- improving memory coalescing / access patterns
- matching *preferred tile shapes* (especially 64×64 guidance around TMEM efficiency)
- improving producer/consumer locality when you keep intermediates in TMEM

## 3.2 The Blackwell-native optimization you should add: “TMEM-resident pipelines”
A published analysis claims TMEM is intended to reduce reliance on shared/register paths, and recommends decomposing GEMMs into **64×64 tiles** to maximize TMEM utilization, warning that tiles that are too small or too large can underutilize the interface or trigger multi-phase transfers. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

So the “Blackwell SwapAB” decision workflow should be:

1) **Can I re-tile the kernel to a 64×64-ish tile without SwapAB?**  
2) **Can I keep intermediates in TMEM across fused ops?**  
3) If not, **does swapping operands make the *better-tilable* dimension become the “contiguous / reused” one**?

This becomes especially important in fused transformer blocks where you’d like to avoid round-tripping intermediate matrices to HBM.

---

# 4) Extending the template to other operations (and applying it)

You asked to extend beyond GEMM; here are two Blackwell-relevant extensions that match the same “theory → microbench → model → guidance” pattern.

## 4.1 Attention / FlashAttention-style kernels (pipeline scheduling lens)
### Why it belongs in this report
- Attention is a canonical “mixed reuse” workload: some stages are compute-dense, others are bandwidth- or latency-sensitive. ([arxiv.org](https://arxiv.org/html/2502.13113))
- Recent work emphasizes that **software pipelining (SWP)** and **warp specialization (WS)** must be jointly optimized, and reports results across Hopper and Blackwell for FlashAttention-like schedules. ([arxiv.org](https://arxiv.org/abs/2512.18134?utm_source=openai))

### Blackwell attention workflow (concrete steps)
1) **Decompose attention into stages** (e.g., QKᵀ tile, softmax/update, PV tile).
2) For each stage, compute:
   - FLOPs
   - bytes moved at each tier (HBM↔L2↔SMEM/TMEM)
3) Decide the pipeline strategy:
   - which warps are “load/transform” specialists (tcgen05.cp / global loads / layout transforms)
   - which warps are “compute” specialists (tcgen05.mma)
4) Validate with Nsight:
   - Tensor utilization
   - memory pipe utilization
   - barrier/synchronization stalls
5) Iterate tile sizes toward TMEM-friendly tiles (64×64 guidance) where possible. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

**Report output:** show how the predicted bottleneck shifts as you:
- change precision (FP8 vs FP4)
- change accumulation (FP16 vs FP32)
- change staging strategy (keep K/V in TMEM vs reload)

## 4.2 Data analytics / sparse workflows: Decompression Engine (DE)
### Why it belongs
NVIDIA’s Blackwell architecture materials highlight a **Decompression Engine**, and published microbenchmark analysis reports detailed DE throughput/latency behavior and batching effects. ([nvidia.com](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/))

### “Same template” applied
1) **Hardware peak**: treat DE as a new fixed-function throughput roof for “compressed bytes/s”.
2) **Microbench**:
   - vary chunk sizes and concurrency
   - measure output throughput vs latency
3) **Model**:
   - identify input-bandwidth vs output-throughput ceilings
4) **Guidance**:
   - pick chunk size + concurrency near the empirical “pipeline depth”
   - fuse decompression with downstream ops when feasible

Published results report strong dependence on chunk size and concurrency and imply that DE performance is shaped by bandwidth constraints and batching behavior. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

---

# 5) What you can already write as “results” today (B200-focused) vs what must be measured (B100/GB10)

## 5.1 What is reasonably well supported for B200 (from public microbenchmark analysis)
You can already include (with citations):
- B200 is described as dual-die Blackwell with **148 SMs** and HBM3e stacks. ([arxiv.org](https://arxiv.org/html/2512.02189v1))
- tcgen05.mma latency is reported around **11 cycles** and near-constant across large tiles, versus Hopper wgmma scaling with tile width. ([arxiv.org](https://arxiv.org/html/2512.02189v1))
- TMEM is reported as **256KB/SM** and extremely high bandwidth, shifting the “feed the tensor cores” problem. ([arxiv.org](https://arxiv.org/html/2512.02189v1))
- Sustained bandwidth numbers (e.g., STREAM triad) and the rough match to peak-bandwidth ratios can be cited for calibration. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

## 5.2 What you should treat as “to be measured / verified” for B100 and GB10
Even if peaks are published, these are commonly missing or SKU-dependent:

### B100
- SM count / clocks / enabled units (often not in public tables)
- Whether tcgen05 latencies match B200 exactly or shift with clocks/partitioning
- Sustained bandwidth (does it hit the same ~7.4–7.5 TB/s region?)

### GB10 (DGX Spark)
- Tensor instruction mix and achievable sustained performance under the 240W platform constraints
- Real attainable memory BW under load (LPDDR5x)
- Whether tcgen05/TMEM behaves identically to datacenter Blackwell (it may not)

---

# 6) Suggested report outline (drop-in template)
If you want a “mimic the Hopper markdown but academic” structure, use:

1. **Introduction**
   - Why Blackwell changes operator optimization assumptions (tcgen05 + TMEM + DE).
2. **Hardware overview**
   - Table: peaks, bandwidths, PAI for B100/B200/GB10.
3. **Theory: roofline + peak decomposition**
   - PAI calculations
   - “peak TFLOPS is made of …” workflow (even if some parameters are TBD)
4. **Instruction-level microarchitecture**
   - tcgen05 mapping + latency/throughput
   - accumulator precision impact
5. **Memory hierarchy characterization**
   - HBM sustained BW
   - TMEM size/bandwidth + how it changes the model
6. **Operator case studies**
   - GEMM (with/without SwapAB)
   - Attention block (SWP/WS scheduling)
   - DE-enabled data pipeline (optional)
7. **Actionable guidelines**
8. **Limitations + missing parameters + how to measure**
9. **Appendix**
   - experimental methodology (clock locking, measurement harness, statistical practice)
   - provenance table (facts + sources)

---

## If you answer 3 questions, I can tailor the “performed analysis” much more tightly
1) Are you targeting **training**, **inference**, or both? (accumulator choice + precision matters a lot on Blackwell)
2) For GEMM, what are the dominant shapes? (typical \(m,n,k\), and whether you compute \(AB^T\) frequently)
3) Confirm: “GH10” = **GB10 (DGX Spark)**?

If you confirm these, I can:
- choose representative tile shapes,
- write the SwapAB decision logic in the same style as your Hopper example (with concrete \(m,n,k\) scenarios),
- and expand the attention/decompression sections into full “worked examples” with explicit compute/byte accounting.

---
Learn more:
1. [A Grace Blackwell AI supercomputer on your desk | NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/?utm_source=openai)
2. [Microbenchmarking NVIDIA’s Blackwell Architecture: An in-depth Architectural Analysis](https://arxiv.org/html/2512.02189v1)
3. [Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs](https://arxiv.org/abs/2512.18134?utm_source=openai)
4. [NVIDIA Blackwell Architecture and B200/B100 Accelerators Announced: Going Bigger With Smaller Data](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data?utm_source=openai)
5. [ThinkSystem NVIDIA HGX B200 180GB 1000W GPU Product Guide > Lenovo Press](https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu?utm_source=openai)
6. [Harp: A Taxonomy for Heterogeneous and Hierarchical Processors for Mixed-reuse Workloads](https://arxiv.org/html/2502.13113)
7. [The Engine Behind AI Factories | NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)