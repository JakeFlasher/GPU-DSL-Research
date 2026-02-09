# Tensor Core Instruction Latency and Optimization Analysis on NVIDIA Blackwell GPUs

## Abstract (≤ 200 words)

This report replicates—and extends—the Hopper-era Tensor Core instruction‑latency analysis workflow to NVIDIA Blackwell, focusing on B100, B200, and the GB10 Grace‑Blackwell Superchip. We first verify SKU naming against current NVIDIA press materials and explicitly flag missing public datasheets and password‑protected NVIDIA PDFs. We then collect per‑SKU peak tensor throughput and memory bandwidth, derive peak arithmetic intensity (PAI), and decompose peak throughput into a Tensor Core unit (TCU) per‑cycle model with confidence tags (VERIFIED | ESTIMATED | UNVERIFIED). The analysis pivots from Hopper’s warp‑group `wgmma` to Blackwell’s warp‑level `tcgen05.mma` instruction family and incorporates Blackwell‑specific Tensor Memory (TMEM), whose bandwidth and synchronization costs alter the memory‑vs‑compute boundary and the profitability of Hopper‑style SwapAB strategies. Finally, we extend the methodology to FlashAttention‑style kernels and reduced‑precision (FP8/FP4) tensor operations, producing cross‑architecture comparison tables and practical kernel‑development guidance while enumerating all uncertainties and open measurement needs.

---

## 0. Verification Gate (SKU names & die designations)

### 0.1 Product names (verification against NVIDIA materials)

**B200** is explicitly referenced by NVIDIA as the **“NVIDIA B200 Tensor Core GPU”** in the March 18, 2024 NVIDIA press release discussing the GB200 Grace Blackwell Superchip. ([investor.nvidia.com](https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Blackwell-Platform-Arrives-to-Power-a-New-Era-of-Computing/default.aspx))

**GB10** is a real NVIDIA product: NVIDIA’s newsroom describes **“GB10 Superchip”** as a Grace‑Blackwell system‑on‑chip (SoC) with a Blackwell GPU and Grace CPU, used for developer‑desktop systems (e.g., Project DIGITS / DGX Spark). ([nvidianews.nvidia.com](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips))  
**Important clarification:** GB10 is not a discrete data‑center GPU SKU in the same sense as B100/B200; it is a **CPU+GPU superchip** aimed at desktop/developer systems. We therefore treat GB10 as a Blackwell‑architecture target whose per‑SM microarchitectural details are largely **not publicly enumerated**.

**B100**: within the set of NVIDIA sources we could access (including the investor press release above), **B100 was not explicitly named**. The name “B100” is widely used in industry reporting, but the absence of an accessible NVIDIA public datasheet/whitepaper page in our source set means the **exact public‑facing product naming and many low‑level parameters remain UNVERIFIED** in this report.

### 0.2 Chip die designations (“GB100”, “GB200”, “GB10”)

NVIDIA’s accessible press materials in our source set **do not publicly state** the internal GPU die codename for the B200 GPU (community sources often use “GB100”, but that is not confirmed here). Consequently:

- **B100 die designation:** **UNVERIFIED** (no accessible NVIDIA document in this run explicitly maps B100 → a particular die codename).
- **B200 die designation:** **UNVERIFIED** (same).
- **GB10** is itself the packaging/product designation “GB10 Superchip” and is therefore **VERIFIED as a real chip product name**, but its internal GPU die codename is **UNVERIFIED** from accessible NVIDIA materials. ([nvidianews.nvidia.com](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips))

### 0.3 Accessibility constraint (affects confidence tags downstream)

Several NVIDIA Blackwell PDFs reachable via NVIDIA shortlinks / `nvdam.widen.net` in our run were **password‑protected**, preventing direct extraction of official tables (e.g., a “Blackwell datasheet” and “Blackwell architecture technical brief” endpoints). This forces reliance on (i) accessible NVIDIA webpages and blogs, and (ii) Tier‑1 research microbenchmarks for low‑level parameters. ([nvdam.widen.net](https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703))

**Assumption policy:** whenever an official NVIDIA table is inaccessible or absent, we (a) mark the parameter **ESTIMATED** or **UNVERIFIED**, and (b) propagate that tag into all derived quantities.

---

## 1. Introduction & Motivation

Operator performance engineering on NVIDIA GPUs increasingly hinges on instruction‑granular understanding of Tensor Core execution. The Hopper‑era “operator‑developer workflow” commonly begins with coarse roofline reasoning (peak tensor TFLOPS vs HBM bandwidth), then sharpens into microarchitectural decomposition (Tensor Core count × per‑cycle work × clock), and finally converges on instruction‑level latency/throughput studies that reveal *where* the hardware saturates and *why* an optimization such as SwapAB works on one SKU yet fails on another.

Blackwell changes the ground rules. According to Tier‑1 Blackwell microbenchmarking work, Blackwell introduces (i) a new PTX Tensor Core instruction family (`tcgen05.*`) and (ii) a new on‑chip storage tier, **Tensor Memory (TMEM)**, which can source Tensor Core operands and hold accumulators, with per‑SM read bandwidth reported at **16 TB/s** and write bandwidth **8 TB/s**. ([arxiv.org](https://arxiv.org/html/2512.02189v1)) In addition, the same Tier‑1 work reports a shift in Tensor Core programming granularity: from Hopper’s **warp‑group** MMA (`wgmma`, 128 threads) to Blackwell’s **warp‑level** MMA (`tcgen05.mma`, 32 threads), with dramatically reduced *single‑instruction latency* and near‑constant latency across tile sizes. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

These changes strongly suggest that Hopper‑tuned heuristics—especially those built around shared‑memory bandwidth plateaus and N‑dimension‑dependent `wgmma` latency scaling—must be re‑validated for Blackwell. The goal of this report is therefore to provide a *single, self‑contained methodology* that kernel developers can apply when transitioning from Hopper to Blackwell, and to explicitly document what can be concluded from public sources versus what remains unknown.

---

## 2. Background: The Hopper Baseline Analysis (brief recap)

### 2.1 The Hopper workflow being replicated

The Hopper analysis workflow (as in the provided Hopper write‑up) can be summarized as seven stages:

1. **Stage A (spec collection):** gather per‑SKU peak tensor throughput (FP16 Tensor TFLOPS) and memory bandwidth.  
2. **Stage B (PAI):** compute peak arithmetic intensity (PAI) as  
   \[
   \text{PAI} = \frac{\text{Peak Tensor TFLOPS}}{\text{HBM Bandwidth (TB/s)}} \quad [\text{FLOPs/Byte}].
   \]
3. **Stage C (TFLOPS decomposition):** write peak throughput as  
   \[
   \text{TFLOPS} = N_{\text{TCU}} \cdot F_{\text{TCU}} \cdot f,
   \]
   where \(N_{\text{TCU}}\) is the number of Tensor Core units, \(F_{\text{TCU}}\) is FLOPs per TCU per cycle, and \(f\) is the boost clock in cycles/s.
4. **Stage D (instruction latency profiling):** microbenchmark a Tensor Core instruction family (Hopper: `wgmma.mma_async.<shape>.f32.f16.f16`) across tile shapes, especially varying \(N\).  
5. **Stage E (threshold derivation):** build a two‑component model:
   \[
   \text{Latency}(\text{shape}) \approx \max(\text{Memory\_cycles}, \text{Compute\_cycles}),
   \]
   and solve for the \(N\) where compute dominates memory.
6. **Stage F (SwapAB):** explain why swapping \(A\) and \(B\) in small‑\(M\) GEMMs changes which instruction shapes are used and can reduce effective latency on some SKUs.
7. **Stage G:** practical guidance.

### 2.2 What changes on Blackwell?

Blackwell retains the broad roofline logic (Stages A–C) but substantially modifies the instruction and memory layers (Stages D–F). In particular:

- Hopper’s core instruction of interest, `wgmma`, is explicitly described as Hopper‑only in Tier‑1 Blackwell microbenchmarking results; Blackwell’s central PTX instruction is `tcgen05.mma`. ([arxiv.org](https://arxiv.org/html/2512.02189v1))
- Blackwell introduces TMEM and a new synchronization/data‑movement regime that affects instruction scheduling and pipeline design. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

Consequently, while we replicate the Hopper workflow stages, we also **extend** them by (i) making the memory term architecture‑aware (SMEM vs TMEM), and (ii) treating “latency” carefully, distinguishing *single‑instruction dependency‑chain latency* from *steady‑state pipe‑filled latency*.

---

## 3. Blackwell Hardware Specifications (Phase 1 / Stage A)

This section constructs the Phase‑1 table required by the prompt, with confidence tags attached to each cell. Where NVIDIA public documents were unavailable (or password‑protected), we either (a) infer from Tier‑1 measurement papers, or (b) mark the value unknown.

### 3.1 Phase‑1 table: Blackwell targets (B100, B200, GB10)

**Table 3.1 — Blackwell Phase‑1 parameter table (with confidence tags)**

| Parameter | B100 | B200 | GB10 (Grace‑Blackwell Superchip) |
|---|---:|---:|---:|
| SMs | **148 (ESTIMATED)** (assumed same SM count as B200; no accessible NVIDIA table) | **148 (UNVERIFIED)** (Tier‑1 microbenchmark paper) ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | **UNKNOWN (UNVERIFIED)** (NVIDIA does not publish SM count on DGX Spark page) ([nvidia.com](https://www.nvidia.com/en-us/project-digits/)) |
| Tensor Cores / SM | **4 (ESTIMATED)** (assumed continuity with Blackwell Ultra SM description) ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)) | **4 (ESTIMATED)** (same assumption) ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)) | **UNKNOWN (UNVERIFIED)** (only “5th Generation Tensor Cores” stated) ([nvidia.com](https://www.nvidia.com/en-us/project-digits/)) |
| Total Tensor Cores / GPU | **592 (ESTIMATED)** (=148×4) | **592 (ESTIMATED)** (=148×4) | **UNKNOWN (UNVERIFIED)** |
| Boost Clock (MHz) for FP16/BF16 | **UNKNOWN (UNVERIFIED)** (no accessible NVIDIA spec; see §4 for scenario‑based derivations) | **UNKNOWN (UNVERIFIED)** | **UNKNOWN (UNVERIFIED)** |
| Peak FP16 Tensor TFLOPS | **1800 (UNVERIFIED)** (industry reporting; not in accessible NVIDIA table) ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data)) | **2250 dense / 4500 sparse (VERIFIED at baseboard level)**: HGX B200 reports FP16/BF16 Tensor Core in “sparse” terms with dense = ½. Per‑GPU dense = 2.25 PFLOPS. ([nvidia.com](https://www.nvidia.com/en-us/data-center/hgx)) | **UNKNOWN for FP16 (UNVERIFIED)**; NVIDIA publishes “up to 1 PFLOP FP4” for DGX Spark/GB10. ([nvidia.com](https://www.nvidia.com/en-us/project-digits/)) |
| HBM generation & capacity | **HBM3e 192 GB (UNVERIFIED)** ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data)) | **HBM3e 192 GB (UNVERIFIED)** (Tier‑1 paper) ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | **LPDDR5x 128 GB unified (VERIFIED)** ([nvidia.com](https://www.nvidia.com/en-us/project-digits/)) |
| Memory Bandwidth (TB/s) | **8.0 (UNVERIFIED)** ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data)) | **8.0 (ESTIMATED/UNVERIFIED)** (NVIDIA blog gives 8 TB/s as per‑GPU bandwidth for Blackwell Ultra memory subsystem; Tier‑1 paper reports sustained 7.48 TB/s in STREAM) ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)) | **0.273 (VERIFIED)** (=273 GB/s) ([nvidia.com](https://www.nvidia.com/en-us/project-digits/)) |
| SMEM size per SM (KB) | **256 (ESTIMATED)** (assumed matches B200’s reported SMEM capacity) ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | **256 (UNVERIFIED)** (Tier‑1 paper references 256 KB SMEM capacity per SM) ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | **UNKNOWN (UNVERIFIED)** |
| SMEM bandwidth (Bytes/cycle/SM) | **UNKNOWN (UNVERIFIED)** (no public value located) | **UNKNOWN (UNVERIFIED)** | **UNKNOWN (UNVERIFIED)** |

**Interpretation notes.**  
1) The B200 SM count (148) is supported by Tier‑1 Blackwell microbenchmark analysis describing a dual‑die B200 with 148 SMs. ([arxiv.org](https://arxiv.org/html/2512.02189v1))  
2) The Phase‑1 table includes SMEM bandwidth because the Hopper methodology uses it for the memory‑vs‑compute threshold. Unfortunately, we did not locate an authoritative Blackwell SMEM bytes/cycle/SM figure in accessible NVIDIA documentation; downstream threshold results that depend on SMEM bytes/cycle are therefore **scenario‑based** and tagged.  
3) GB10 is treated as “GB10 Superchip” (not corrected to “GB100/GB200”) because GB10 is confirmed as a real NVIDIA Grace‑Blackwell product; however, its GPU‑internal details are not publicly enumerated at the same level as data‑center accelerators. ([nvidianews.nvidia.com](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips))

### 3.2 Hopper equivalents (for direct comparison)

**Table 3.2 — Hopper comparison table (H100 PCIe, H100 SXM5, HGX H20)**

| Parameter | H100 PCIe | H100 SXM5 | HGX H20 |
|---|---:|---:|---:|
| SMs | **114 (VERIFIED)** ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | **132 (VERIFIED)** ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | **UNKNOWN (UNVERIFIED)** (not in accessible NVIDIA table in this run) |
| Tensor Cores / SM | **4 (VERIFIED)** ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | **4 (VERIFIED)** ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | **UNKNOWN (UNVERIFIED)** |
| Total Tensor Cores / GPU | **456 (VERIFIED)** ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | **528 (VERIFIED)** ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | **UNKNOWN (UNVERIFIED)** |
| Boost Clock (MHz) for FP16/BF16 | **1620 (UNVERIFIED)** (from Hopper baseline write‑up) | **≈1830 (ESTIMATED)** (back‑solved assuming \(F_{\text{TCU}}=1024\); see §4) | **UNKNOWN (UNVERIFIED)** |
| Peak FP16 Tensor TFLOPS (dense) | **756 (UNVERIFIED)** (from Hopper baseline write‑up) | **989.4 (CONSISTENT with ½ of 1,979 “with sparsity”)** ([nvidia.com](https://www.nvidia.com/en-eu/data-center/h100/)) | **148 (UNVERIFIED)** (from Hopper baseline write‑up) |
| Memory bandwidth (TB/s) | **2.039 (UNVERIFIED)** (Hopper baseline write‑up) | **3.35 (VERIFIED)** ([nvidia.com](https://www.nvidia.com/en-eu/data-center/h100/)) | **4.0 (UNVERIFIED)** (Hopper baseline write‑up) |
| SMEM size per SM (KB) | **256 combined L1/SMEM (VERIFIED for Hopper SM)** ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | **256 combined L1/SMEM (VERIFIED)** ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | **UNKNOWN (UNVERIFIED)** |
| SMEM bandwidth (Bytes/cycle/SM) | **128 (UNVERIFIED)** (Hopper baseline write‑up) | **128 (UNVERIFIED)** | **128 (UNVERIFIED)** |

**Caveat:** the Hopper baseline values for H20 and some clock values are carried from the provided Hopper document because we did not locate accessible official tables for H20 in this run. Those values remain **UNVERIFIED**.

---

**CHECKPOINT — Phase 1**  
• Key values: B200 has 148 SMs and 192 GB HBM3e; TMEM is 256 KB/SM with 16 TB/s read, 8 TB/s write (Tier‑1). ([arxiv.org](https://arxiv.org/html/2512.02189v1))  
• Open questions: official B100/B200 boost clocks; Blackwell SMEM bytes/cycle; GB10 SM/TC counts.  
• Carry-forward: treat Blackwell instruction as `tcgen05.mma`; treat TMEM as first‑class in the memory term. ([arxiv.org](https://arxiv.org/html/2512.02189v1))  

---

## 4. Derived Metrics: PAI and FLOPs/TCU/cycle (Phase 2 / Stages B & C)

### 4.1 Peak Arithmetic Intensity (PAI)

Using the prompt’s definition (tensor throughput divided by off‑chip bandwidth):

\[
\text{PAI} \;=\; \frac{\text{Peak Tensor TFLOPS}}{\text{HBM Bandwidth (TB/s)}} \quad [\text{FLOPs/Byte}].
\]

**Blackwell (FP16, dense):**
- **B100:**  
  \[
  \text{PAI}_{\text{B100}} \approx \frac{1800}{8.0} = 225 \;\text{FLOPs/Byte}
  \]
  (inputs UNVERIFIED). ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data))
- **B200:**  
  \[
  \text{PAI}_{\text{B200}} \approx \frac{2250}{8.0} = 281.25 \;\text{FLOPs/Byte}
  \]
  (TFLOPS derived from HGX B200 sparse spec with dense = ½; bandwidth partially UNVERIFIED). ([nvidia.com](https://www.nvidia.com/en-us/data-center/hgx))
- **GB10:** FP16 peak is **not publicly specified** on the DGX Spark page; therefore **PAI(FP16) is UNVERIFIED/UNKNOWN**. ([nvidia.com](https://www.nvidia.com/en-us/project-digits/))  
  (Supplementary: FP4 “up to 1 PFLOP” and 273 GB/s imply FP4‑PAI ≈ 3663 FLOPs/Byte, but that is *not* the requested FP16 PAI.)

**Hopper baselines (from the provided Hopper write‑up):**
- **H20:** \(148/4 = 37\) FLOPs/Byte (given baseline).
- **H100 PCIe:**  
  \[
  \text{PAI}_{\text{H100 PCIe}} \approx \frac{756}{2.039} \approx 370.77 \;\text{FLOPs/Byte}.
  \]
- **H100 SXM5:**  
  \[
  \text{PAI}_{\text{H100 SXM}} \approx \frac{989.4}{3.352} \approx 295.17 \;\text{FLOPs/Byte}.
  \]

**Implication.** Blackwell’s FP16 dense PAI (225–281 FLOPs/Byte in the UNVERIFIED/ESTIMATED B100/B200 spec set) sits **below** Hopper H100 PCIe’s ~371 and near Hopper H100 SXM’s ~295, indicating that “HBM‑vs‑tensor” balance has not shifted uniformly; the B200’s enormous FP4/FP8 throughput gains do not automatically translate into higher FP16 PAI, and many FP16 kernels may still be bandwidth‑sensitive.

### 4.2 FLOPs/TCU/cycle decomposition (Stage C)

We use:

\[
F_{\text{TCU}} \;=\; \frac{\text{Peak TFLOPS}}{N_{\text{TCU}}\cdot f_{\text{boost}}}
\]
where \(f_{\text{boost}}\) is in THz.

**Hopper reference (sanity check).** Using the Hopper baseline workflow for H100 PCIe (values consistent with the provided Hopper write‑up):

- \(N_{\text{TCU}} = 114 \times 4 = 456\). ([developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/))  
- \(\text{Peak FP16 TFLOPS} = 756\) (baseline).  
- \(f_{\text{boost}} = 1.620\,\text{GHz} = 0.001620\,\text{THz}\) (baseline).  
Then:
\[
F_{\text{TCU}} \approx \frac{756}{456\times 0.001620}
= \frac{756}{0.73872}
\approx 1023.39 \approx 1024 \;\text{FLOPs/TCU/cycle}.
\]
This reproduces the Hopper baseline conclusion.

**Blackwell challenge: missing public boost clocks.** Because accessible NVIDIA documentation does not publish per‑SKU tensor boost clocks for B100/B200/GB10 in our source set, we present a **scenario‑based** derivation.

Let \(N_{\text{TCU}} \approx 148 \times 4 = 592\) (ESTIMATED) and let peak FP16 (dense) be 1800 TFLOPS (B100) or 2250 TFLOPS (B200). Then:

- **B200 (symbolic):**
  \[
  F_{\text{TCU,B200}} \approx \frac{2250}{592 \cdot f_{\text{boost}}(\text{THz})}.
  \]
  If \(f_{\text{boost}} = 1.85\,\text{GHz}\), then \(F_{\text{TCU}}\approx 2054\).  
  If \(f_{\text{boost}} = 1.98\,\text{GHz}\), then \(F_{\text{TCU}}\approx 1919\).  
  These values suggest **≈2× Hopper** per‑cycle capability, but the exact integer is **ESTIMATED**.

- **B100 (symbolic):**
  \[
  F_{\text{TCU,B100}} \approx \frac{1800}{592 \cdot f_{\text{boost}}(\text{THz})}.
  \]
  At \(f_{\text{boost}} = 1.60\,\text{GHz}\), \(F_{\text{TCU}}\approx 1900\).

**Operational takeaway.** Even with clock uncertainty, Blackwell’s *per‑TCU per‑cycle* work for FP16 is plausibly in the **~1900–2100 FLOPs/cycle** band under the common assumption of 592 TCUs. That strongly hints at a **widened Tensor Core datapath** relative to Hopper’s ~1024 FLOPs/cycle/TCU.

### 4.3 What does this suggest about the Tensor Core array (Stage 4.3)?

Tier‑1 Blackwell microbenchmarking provides a stronger microarchitectural clue than the TFLOPS decomposition alone: the reported **single‑instruction latency** for Blackwell `tcgen05.mma` is approximately **11 cycles** and remains nearly constant across tile sizes (e.g., from \(m64n64k16\) to \(m256n256k16\)), while Hopper `wgmma` single‑instruction latency grows linearly with tile width (e.g., 32 → 64 → 128 cycles for increasing \(N\)). ([arxiv.org](https://arxiv.org/html/2512.02189v1))

A natural interpretation is that Blackwell’s Tensor Core pipeline behaves more like a **spatially scaled array** (more parallel work per cycle at similar depth) rather than Hopper’s more **temporal** behavior where larger tiles effectively consume more cycles. This aligns with the paper’s conclusion that tile size affects **throughput** more than **latency** on Blackwell, which is consistent with widening and/or increased concurrent utilization rather than deeper pipelining. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

---

**CHECKPOINT — Phase 2**  
• Key values: Blackwell FP16 dense PAI ≈ 225 (B100) and 281 (B200) FLOPs/Byte (UNVERIFIED inputs); Hopper PAI: H20=37, H100 PCIe≈371, H100 SXM≈295. ([nvidia.com](https://www.nvidia.com/en-us/data-center/hgx))  
• Open questions: B100/B200 tensor boost clocks and exact TCU count/SM for data‑center Blackwell.  
• Carry-forward: latency evidence suggests Blackwell TC is spatially widened; plan models that incorporate TMEM and warp‑level `tcgen05`. ([arxiv.org](https://arxiv.org/html/2512.02189v1))  

---

## 5. Instruction Latency Model (Phase 3 / Stage D)

### 5.1 Blackwell Tensor Core instruction identification (Step 5.1)

**Hopper baseline instruction (reference):** Hopper uses warp‑group MMA instructions (`wgmma.*`) that are issued by a **warp group** (128 threads, commonly 4 warps). This is the instruction family used in the original Hopper analysis.

**Blackwell instruction family:** Tier‑1 Blackwell microbenchmarking identifies a new PTX instruction family, **`tcgen05`**, with the MMA instruction **`tcgen05.mma`**. It further states a change in granularity: Hopper’s `wgmma` is warp‑group‑level, while Blackwell’s `tcgen05.mma` is **warp‑level**. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

**Operand placement model:** Unlike Hopper’s requirement that operands traverse shared memory before Tensor Core consumption, Blackwell’s `tcgen05` path allows MMA instructions to source operands from **SMEM or TMEM**, and it writes accumulators into **TMEM** as part of its asymmetric dataflow. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

**Supported shapes (publicly evidenced in Tier‑1 measurement):** the Tier‑1 paper reports measured latencies for shapes including:
- `tcgen05.mma m64n64k16` (warp scope),
- `tcgen05.mma m128n128k16`,
- `tcgen05.mma m256n256k16`,  
and also reports a precision‑sweep table for `m64n8k16`. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

**Methodological extension vs Hopper:** Because Blackwell introduces explicit TMEM loads/stores and synchronization, the relevant “instruction latency” for kernel developers can be:
1) **Single‑instruction dependency‑chain latency** (what Tier‑1 reports), and/or  
2) **Steady‑state pipe‑filled issue latency** (what the Hopper write‑up measured for `wgmma`).  

In the absence of a published Blackwell “pipe‑filled” `tcgen05` latency table in accessible sources, we treat Tier‑1 SI‑LAT as the primary measured data and explicitly flag comparability limits.

### 5.2 Latency table construction (Step 5.2)

#### 5.2.1 Measured Blackwell SI‑LAT (Tier‑1)

Tier‑1 reports SI‑LAT (cycles) for Blackwell `tcgen05.mma` that is roughly constant (~11–11.4 cycles) across large tile sizes, and it reports FP16 SI‑LAT ≈ 11.2 cycles for `m64n8k16`. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

#### 5.2.2 Required format (Hopper‑style N sweep)

The prompt requests the Hopper‑style shape sweep:
`m64n8k16`, `m64n16k16`, `m64n32k16`, `m64n64k16`, `m64n128k16`, `m64n256k16`.

Only a subset is explicitly measured for Blackwell in accessible Tier‑1 sources; therefore, we fill the missing shapes using the *measured invariance trend* and mark them **ESTIMATED**.

**Table 5.1 — Instruction latency (cycles). Blackwell entries are SI‑LAT; Hopper entries are pipe‑filled baseline values.**

| Instruction Shape | B100 | B200 | GB10 | H100 SXM (baseline) | H20 (baseline) |
|---|---:|---:|---:|---:|---:|
| m64n8k16 | 11.2 (ESTIMATED*) | **11.2 (UNVERIFIED, measured SI‑LAT)** ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | UNKNOWN | 18 | 18 |
| m64n16k16 | 11.2 (ESTIMATED*) | 11.2 (ESTIMATED) | UNKNOWN | 20 | 32 |
| m64n32k16 | 11.2 (ESTIMATED*) | 11.2 (ESTIMATED) | UNKNOWN | 24 | 64 |
| m64n64k16 | 11.0 (ESTIMATED*) | **11.0 (UNVERIFIED, measured SI‑LAT)** ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | UNKNOWN | 32 | 128 |
| m64n128k16 | 11.1 (ESTIMATED*) | 11.1 (ESTIMATED) | UNKNOWN | 64 | 256 |
| m64n256k16 | 11.2 (ESTIMATED*) | 11.2 (ESTIMATED) | UNKNOWN | 128 | 512 |
| **Blackwell‑only:** m128n128k16 | 11.3 (ESTIMATED*) | **11.3 (UNVERIFIED, measured SI‑LAT)** ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | UNKNOWN | — | — |
| **Blackwell‑only:** m256n256k16 | 11.4 (ESTIMATED*) | **11.4 (UNVERIFIED, measured SI‑LAT)** ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | UNKNOWN | — | — |

\*B100 entries are “architecture‑inherited SI‑LAT” estimates: the Tier‑1 latency data is for B200; we assume instruction pipeline depth in cycles is shared across B100/B200 unless proven otherwise.

**Interpretation.** The striking contrast is that Hopper’s baseline `wgmma` latencies scale strongly with \(N\), while Blackwell’s `tcgen05` SI‑LAT is reported to be almost constant across tile sizes. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

### 5.3 Memory‑bound vs compute‑bound threshold (Step 5.3 / Stage E)

The Hopper write‑up’s threshold logic relied on modeling `wgmma` as reading \(A\) and \(B\) tiles from shared memory each instruction, then comparing that SMEM read time to compute time. On Blackwell, that model must be generalized, because the MMA can source operands from **TMEM** and the overall pipeline can overlap data movement.

We therefore provide **two threshold derivations**:

#### 5.3.1 Threshold under a Hopper‑style “SMEM‑sourced MMA” assumption (UNVERIFIED)

If Blackwell `tcgen05.mma` were to read both \(A\) and \(B\) directly from SMEM per instruction, then:

- FP16 element size: 2 bytes.
- \(A\) bytes for \(m64k16\):  
  \[
  A_{\text{bytes}} = 64 \times 16 \times 2 = 2048.
  \]
- \(B\) bytes for \(n\times k = N\times16\):  
  \[
  B_{\text{bytes}} = N \times 16 \times 2 = 32N.
  \]
- Memory cycles:
  \[
  T_{\text{mem}}(N) = \left\lceil \frac{2048}{B_{\text{SMEM}}} \right\rceil + 
                       \left\lceil \frac{32N}{B_{\text{SMEM}}} \right\rceil.
  \]
But **\(B_{\text{SMEM}}\)** (bytes/cycle/SM) is not publicly available in our source set; thus no numeric threshold can be VERIFIED.

Notably, the reported SI‑LAT ≈ 11 cycles for Blackwell is **lower** than the Hopper‑style SMEM read time one would get by assuming Hopper’s 128 B/cycle (which would be 18 cycles at \(N=8\)). This mismatch is evidence that the critical path is *not* “raw SMEM read of A+B per instruction” in the same way as Hopper.

#### 5.3.2 Threshold under a Blackwell‑appropriate “TMEM‑staged operands” model (supported by Tier‑1)

Tier‑1 Blackwell microbenchmarking reports that TMEM provides **16 TB/s read bandwidth per SM** and **8 TB/s write bandwidth per SM**, additive with L1/SMEM bandwidth. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

Let \(B_{\text{TMEM,read}} = 16\text{ TB/s}\). Converting to bytes/cycle requires a clock; because the boost clock is not publicly specified here, we leave it symbolic, \(f\) cycles/s:

\[
B_{\text{TMEM,read}}^{(\text{bytes/cycle})} = \frac{16\times 10^{12}}{f}.
\]

Then the “operand supply” time for \(A\) and \(B\) when already staged in TMEM is:

\[
T_{\text{mem,TMEM}}(N) \approx \frac{2048 + 32N}{B_{\text{TMEM,read}}^{(\text{bytes/cycle})}}
= \frac{(2048 + 32N)\, f}{16\times10^{12}}.
\]

For any plausible GHz‑class \(f\), this term is far below 1–2 cycles for \(N \le 256\), i.e., negligible compared to the measured ~11‑cycle SI‑LAT of `tcgen05.mma`. Thus, **the instruction is compute‑dominated for all \(N\) in the Hopper sweep** when operands are TMEM‑resident.

**Conclusion (per SKU).**
- **B200:** under the TMEM‑staged model, the “memory→compute transition \(N\)” is effectively **below the smallest relevant \(N\)**; i.e., there is **no transition within \(N \in [8,256]\)** and the operation is compute‑dominated. This is consistent with the reported constant SI‑LAT across tile sizes. ([arxiv.org](https://arxiv.org/html/2512.02189v1))  
- **B100:** same conclusion is **ESTIMATED** (architecture‑inherited instruction pipeline).  
- **GB10:** **UNVERIFIED/UNKNOWN** because no TMEM bandwidth/SMEM bandwidth/latency measurements are published for GB10 in accessible sources; only its system memory bandwidth is published. ([nvidia.com](https://www.nvidia.com/en-us/project-digits/))

---

**CHECKPOINT — Phase 3**  
• Key values: Blackwell uses warp‑level `tcgen05.mma`; measured SI‑LAT ≈ 11 cycles and is nearly tile‑size invariant; TMEM is 256 KB/SM with 16 TB/s read and 8 TB/s write. ([arxiv.org](https://arxiv.org/html/2512.02189v1))  
• Open questions: Blackwell SMEM bytes/cycle; pipe‑filled `tcgen05` latency sweep comparable to Hopper; GB10 microbench data.  
• Carry-forward: treat “threshold \(N\)” as largely eliminated (compute‑dominated) when using TMEM staging, shifting optimization focus from N‑selection to scheduling/synchronization. ([arxiv.org](https://arxiv.org/html/2512.18134v1))  

---

## 6. Compute vs Memory Threshold Analysis (Phase 3 continuation / Stage E)

### 6.1 Why Hopper’s “N‑threshold” exists

On Hopper, `wgmma` latency in the baseline analysis increases with \(N\) once the instruction ceases to be limited by shared‑memory operand fetch and becomes limited by Tensor Core arithmetic throughput. The crucial point is that Hopper’s instruction execution is tightly coupled to SMEM operand delivery in the measured steady‑state pipeline.

### 6.2 Why Blackwell weakens or removes that threshold (TMEM + warp‑level MMA)

Blackwell’s introduction of TMEM and warp‑level `tcgen05.mma` modifies both terms in the classic two‑component model:

1. **Memory term changes in kind.** Operand traffic can be staged into TMEM via `tcgen05.cp`, and accumulators are stored in TMEM. Tier‑1 work explicitly states that operands may be sourced from SMEM or TMEM and that accumulators are written to TMEM, making “SMEM read bandwidth” an incomplete descriptor of operand supply. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

2. **Compute term changes in scaling.** The reported SI‑LAT for `tcgen05.mma` is approximately constant across tile sizes, suggesting tile size affects *throughput* (work done per instruction) rather than *latency* (cycles until dependency resolves). ([arxiv.org](https://arxiv.org/html/2512.02189v1)) This is qualitatively different from Hopper, where larger `wgmma` tiles have larger SI‑LAT.

The combined effect is that the “transition at \(N=\text{some threshold}\)” that the Hopper analysis used to explain SwapAB behavior may not exist (or may occur at \(N\) values outside the conventional tile range) on Blackwell when kernels are structured to keep intermediate results and/or operands in TMEM.

### 6.3 Practical reinterpretation of the threshold concept for Blackwell

For Blackwell, a more actionable “threshold” for kernel developers is often **not** “SMEM‑bound vs Tensor‑bound within a single MMA instruction,” but rather:

- **TMEM synchronization threshold:** when the cost of TMEM loads/stores and their blocking synchronization becomes visible and forces warp specialization. OPT_PIPE explicitly notes that Blackwell requires different software pipelining and warp specialization strategies due to a faster Tensor Core and the synchronization required by Tensor Memory loads/stores. ([arxiv.org](https://arxiv.org/html/2512.18134v1))

- **Global‑to‑on‑chip staging threshold:** when the tile size and reuse are large enough that staging into TMEM amortizes the `tcgen05.cp` and synchronization overheads; Tier‑1 microbenchmarking suggests a preferred tile size regime (e.g., 64×64 elements) for TMEM efficiency. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

Thus, Blackwell shifts the “threshold logic” upward in the stack: the key transitions are now between *dataflow organizations* (SMEM‑only vs TMEM‑centric pipelines) and *scheduling strategies* (single‑warp‑role vs multi‑warp specialization), rather than between small‑N and large‑N `wgmma` shapes.

---

## 7. SwapAB Optimization on Blackwell (Phase 4 / Stage F)

We now replicate the Hopper SwapAB analysis for the concrete example \(m=16, n=64, k=16\) in a single‑SM scope.

### 7.1 Problem setup (Step 6.1)

We consider a GEMM of the form \(C = A B^T\) where \(A\in\mathbb{R}^{m\times k}\), \(B\in\mathbb{R}^{n\times k}\). The Hopper write‑up notes that if an instruction’s \(M\) dimension is fixed (e.g., \(M=64\)), then when \(m<64\) some compute is “wasted” unless the problem is transformed.

SwapAB rewrites:
\[
C = A B^T \quad \Rightarrow \quad C = (B A^T)^T,
\]
which effectively swaps the roles of \(m\) and \(n\) from the instruction‑selection perspective.

### 7.2 Strategy enumeration per prompt (Step 6.2)

We define \(T\) as the baseline instruction‑issue latency “on the plateau” (the minimal latency observed when the instruction is not further scaled by the \(N\) dimension). On Hopper H20, the baseline analysis treated `m64n16k16` as \(T\) and `m64n64k16` as \(4T\). On H100, both were treated as \(T\).

For Blackwell, Tier‑1 measurement indicates `tcgen05.mma` SI‑LAT is nearly tile‑size invariant, so we adopt the model:
\[
\text{Latency}(m64n16k16) \approx \text{Latency}(m64n64k16) \approx T.
\]
This is **ESTIMATED** for the unmeasured `n16` case but consistent with the reported invariance trend. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

For \(m=16,n=64,k=16\):

- **Strategy 1:** one large‑\(N\) instruction (e.g., `m64n64k16`), compute \(C=A B^T\): latency \(\approx T\).
- **Strategy 2:** tile in \(N\) using four `m64n16k16` instructions, compute \(C=A B^T\): latency \(\approx 4T\).
- **Strategy 3:** SwapAB and use one `m64n16k16` on the swapped problem, compute \(C=(B A^T)^T\): latency \(\approx T\).

### 7.3 Cross‑SKU summary table (Step 6.3)

**Table 7.1 — SwapAB strategy comparison**

| SKU | Strategy 1 | Strategy 2 | Strategy 3 | SwapAB Speedup |
|---|---:|---:|---:|---:|
| H100 SXM5 (baseline) | \(T\) | \(4T\) | \(T\) | \(1\times\) |
| H20 (baseline) | \(4T\) | \(4T\) | \(T\) | \(4\times\) |
| B100 | \(T\) (ESTIMATED) | \(4T\) (ESTIMATED) | \(T\) (ESTIMATED) | \(\approx 1\times\) |
| B200 | \(T\) (ESTIMATED; SI‑LAT invariance) | \(4T\) (ESTIMATED) | \(T\) (ESTIMATED) | \(\approx 1\times\) |
| GB10 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

### 7.4 Per‑SKU explanation (Step 6.4, ≤ 1 paragraph each)

**B200.** SwapAB is not expected to deliver Hopper‑H20‑style speedups because the underlying Blackwell MMA latency does not grow strongly with \(N\) in the available SI‑LAT data; thus choosing a “large‑\(N\)” instruction is not intrinsically penalized by increased instruction latency, eliminating the mechanism that made H20’s direct approach effectively \(4T\). Furthermore, TMEM‑centric dataflows move the dominant costs toward synchronization and staging rather than toward SMEM‑bandwidth plateaus. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

**B100.** Under the assumption that B100 shares the same `tcgen05` pipeline depth (in cycles) as B200, SwapAB should likewise offer little benefit. Any remaining performance differences are more likely to come from (i) different clocks/power limits or (ii) data‑movement overheads (TMEM allocation/copy) rather than from an \(N\)-dependent MMA instruction latency curve.

**GB10.** No instruction‑latency or per‑SM microarchitecture measurements were found in accessible sources. Because GB10 is a superchip with unified LPDDR5x memory bandwidth (273 GB/s) and unknown on‑chip details, SwapAB profitability cannot be predicted reliably from public information. ([nvidia.com](https://www.nvidia.com/en-us/project-digits/))

### 7.5 Relation to Hopper PRs and expected Blackwell impact (Step 6.5)

The Hopper write‑up cites real‑world kernel PRs where SwapAB improved performance on H20 but not on H100. That pattern is precisely what the Hopper threshold model explains: on H20, choosing a large \(N\) can push the instruction into a compute‑dominated regime where latency scales up with \(N\), making SwapAB profitable; on H100, large‑\(N\) already yields the low‑latency regime, reducing benefit.

On Blackwell, the available evidence indicates that the key latency‑scaling mechanism is no longer present at the instruction level (or is suppressed by TMEM staging and warp‑level MMA), so Hopper‑style SwapAB PRs should be treated as **architecture‑contingent** and re‑benchmarked. The optimization may still matter if it changes **dataflow** (e.g., which operand is reused and thus staged into TMEM), but not because of a Hopper‑like \(N\)-dependent MMA latency curve.

---

**CHECKPOINT — Phase 4**  
• Key values: Hopper SwapAB gives ~4× on H20 but ~1× on H100; on Blackwell (B100/B200) SwapAB is expected ≈1× because `tcgen05.mma` SI‑LAT is near tile‑size invariant and TMEM shifts bottlenecks to synchronization/staging. ([arxiv.org](https://arxiv.org/html/2512.02189v1))  
• Open questions: measured pipe‑filled `tcgen05` latency for `m64n16k16` vs `m64n64k16`; GB10 instruction behavior.  
• Carry-forward: Blackwell tuning focus: TMEM residency + software pipelining/warp specialization, not SwapAB for small‑M alone. ([arxiv.org](https://arxiv.org/html/2512.18134v1))  

---

## 8. Extended Operations (Phase 5)

This section repeats the threshold‑derivation and optimization‑impact analysis for **two additional operations** beyond the single GEMM case: (a) FlashAttention forward and (d) reduced‑precision Tensor Core operations (FP8/FP4). We additionally comment on MoE/grouped GEMM as a third case study.

### 8.1 Operation 1: Flash Attention forward pass (choice (a))

#### 8.1.1 Mapping to Tensor Core shapes (requirement 1)

A single‑SM FlashAttention‑style forward pass can be abstracted as a loop over key/value blocks:
\[
S_i = Q K_i^T,\quad P_i = \exp(S_i)\;\text{(plus normalization)},\quad O \mathrel{+}= P_i V_i.
\]
OPT_PIPE presents this structure explicitly in its scheduling discussion and evaluates Blackwell forward attention at FP16 with head dimension 128. ([arxiv.org](https://arxiv.org/html/2512.18134v1))

Let \(Q\) have shape \((M, K)\) and \(K_i\) have shape \((N, K)\) where:
- \(M = \text{block\_q}\),
- \(N = \text{block\_kv}\),
- \(K = \text{head\_dim}\) (often 128 in the cited experiments). ([arxiv.org](https://arxiv.org/html/2512.18134v1))

Each GEMM has FLOP count:
\[
\text{FLOPs}(QK^T) = 2MNK,\quad \text{FLOPs}(PV) = 2MNK.
\]
Thus attention’s Tensor Core work per block pair is approximately \(4MNK\) FLOPs, ignoring softmax overhead.

#### 8.1.2 Threshold derivation (requirement 2)

A Hopper‑style threshold compares compute time to the time required to stage operands into the on‑chip source used by Tensor Core instructions. On Blackwell, the crucial distinction is whether intermediate results and/or operands reside in **TMEM**.

Tier‑1 Blackwell microbenchmarking argues that intermediate results for chained matmuls (like attention’s \(QK^T\) followed by \(PV\)) should be kept in TMEM to exploit the extremely high TMEM read bandwidth. ([arxiv.org](https://arxiv.org/html/2512.02189v1)) We therefore define two regimes:

- **Regime A (SMEM‑centric):** operands staged into SMEM; accumulators spilled to SMEM/registers.
- **Regime B (TMEM‑centric):** operands and accumulators staged in TMEM; explicit TMEM loads/stores and synchronization occur.

Let the per‑block compute time be:
\[
T_{\text{comp}} \approx \frac{4MNK}{\text{TC\_FLOPs/cycle per SM}}\;\;[\text{cycles}],
\]
and let the staging time be:
\[
T_{\text{stage}} \approx \frac{\text{bytes staged}}{\text{effective on-chip bandwidth}}\;\;[\text{cycles}].
\]

Tier‑1 provides a concrete on‑chip bandwidth for TMEM: 16 TB/s read per SM. ([arxiv.org](https://arxiv.org/html/2512.02189v1)) This implies that when softmax and rescaling can be structured to reuse TMEM‑resident accumulators, the effective staging bandwidth is so high that \(T_{\text{stage}}\) is generally smaller than synchronization and compute costs.

Hence, the “threshold” for Blackwell attention is less about \(N\) (tile width) and more about **whether the software pipeline can hide TMEM load/store synchronization**. OPT_PIPE emphasizes that on Blackwell, “Tensor Memory loads and stores” introduce synchronization operations that fundamentally alter optimal software pipelining and warp specialization compared to Hopper. ([arxiv.org](https://arxiv.org/html/2512.18134v1))

#### 8.1.3 Optimization implications (requirement 3)

The key Blackwell implications for FlashAttention‑like kernels are:

1. **Warp specialization becomes mandatory rather than optional.** OPT_PIPE reports that Blackwell’s faster Tensor Core and the synchronization associated with Tensor Memory loads/stores require different scheduling strategies, with dedicated warp groups handling variable‑latency operations and TMEM‑related synchronization. ([arxiv.org](https://arxiv.org/html/2512.18134v1))

2. **TMEM residency changes the cost model of softmax/rescaling.** OPT_PIPE notes that reading accumulators from Tensor Memory for general computations requires explicit data movement into the register file, and the scheduling must account for blocking synchronization. ([arxiv.org](https://arxiv.org/html/2512.18134v1)) This adds a new “latency term” absent in Hopper’s SMEM‑only analysis.

3. **Tile size selection is constrained by TMEM efficiency rather than SMEM bytes/cycle.** Tier‑1 suggests an efficiency sweet spot around 64×64 element tiles for TMEM utilization (and warns of underutilization for smaller tiles and multi‑phase transfers for very large tiles). ([arxiv.org](https://arxiv.org/html/2512.02189v1))

#### 8.1.4 Blackwell vs Hopper comparison (requirement 4)

On Hopper, FlashAttention kernels commonly treat `wgmma` + SMEM bandwidth as the primary microarchitectural bottleneck and rely on dynamic warp interleaving to extract ILP. OPT_PIPE explicitly states that Hopper’s dynamic interleaving of warps can naturally provide instruction‑level parallelism, whereas on Blackwell the burden shifts toward static scheduling and explicit warp role assignment. ([arxiv.org](https://arxiv.org/html/2512.18134v1))

Thus, for attention kernels, Blackwell’s “threshold” is best understood as a **pipeline‑synchronization threshold** (TMEM load/store + rescale/exp overlap), not as a simple \(N\) boundary in MMA latency.

---

### 8.2 Operation 2: Reduced precision Tensor Core operations (choice (d): FP8 / FP4)

#### 8.2.1 Mapping (requirement 1)

We reuse the same MMA tile interpretation but change the operand size:
- FP16: 2 bytes/element,
- FP8: 1 byte/element,
- FP4: 0.5 bytes/element (packed).

#### 8.2.2 Threshold derivation (requirement 2)

A Hopper‑style SMEM bandwidth model yields memory time proportional to bytes moved. For a fixed \((M,K)=(64,16)\) and varying \(N\):

- FP16 bytes: \(A=2048\), \(B=32N\).
- FP8 bytes: \(A=1024\), \(B=16N\).
- FP4 bytes: \(A=512\), \(B=8N\).

If the on‑chip operand source is TMEM, Tier‑1 indicates the memory term becomes extremely small relative to compute latency and synchronization. ([arxiv.org](https://arxiv.org/html/2512.02189v1)) The key precision‑dependent effect then becomes: **throughput increases without comparable increases in latency**.

Tier‑1 Blackwell microbenchmarking reports that across precisions, latency varies only modestly (roughly 11.2–14.2 cycles), while throughput scales dramatically (e.g., FP16 ≈ 1929 TFLOPS, FP8 ≈ 3851 TFLOPS, FP4 ≈ 7702 TFLOPS in their reported measurements). ([arxiv.org](https://arxiv.org/html/2512.02189v1)) This suggests that reduced precision increases effective work per cycle primarily via widened parallelism rather than deeper pipelines—again pushing any memory‑vs‑compute threshold further toward the memory side (or eliminating it if TMEM staging dominates).

#### 8.2.3 Practical implications (requirement 3)

1. **Latency is not the tuning lever across precisions.** Because latency changes little, precision selection is chiefly about throughput and bandwidth pressure, not instruction‑latency hazards. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

2. **Accumulator precision can halve throughput.** The Tier‑1 paper states that moving from FP16 accumulation to FP32 accumulation halves throughput (FP16→FP32 accumulate) in their measurement set, implying an accumulator datapath bottleneck. ([arxiv.org](https://arxiv.org/html/2512.02189v1)) In practice, kernels requiring FP32 accumulation should expect reduced peak utilization and may shift bottlenecks back toward memory/synchronization.

3. **TMEM reuse is especially valuable at FP4/FP8.** Lower‑precision increases arithmetic throughput faster than HBM bandwidth increases; therefore, keeping intermediate results in TMEM and minimizing global traffic becomes essential to avoid a “roofline collapse” at low precision.

#### 8.2.4 Blackwell vs Hopper (requirement 4)

Hopper’s FP8 is delivered via the Transformer Engine and `wgmma`, but its instruction latency still scales with tile width and warp‑group synchronization. Blackwell’s warp‑level `tcgen05.mma` plus TMEM changes both synchronization and dataflow, and measured instruction latency appears largely decoupled from tile size. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

---

### 8.3 Operation 3 (additional): Grouped GEMM / MoE dispatch (choice (b))

MoE inference/training often involves many small GEMMs with small \(M\) (tokens per expert) and moderate \(N,K\). Hopper SwapAB can help when small \(M\) forces underutilization of fixed‑\(M\) MMA tiles. On Blackwell, the main constraint becomes **whether these tiny GEMMs can amortize TMEM allocation and synchronization** and whether warp specialization can hide variable‑latency operations (e.g., staging expert weights) without wasting Tensor Core cycles. OPT_PIPE’s discussion of Blackwell’s increased synchronization requirements supports the view that MoE kernels may need more explicit scheduling than on Hopper. ([arxiv.org](https://arxiv.org/html/2512.18134v1))

---

**CHECKPOINT — Phase 5**  
• Key values: For Blackwell attention, the dominant “threshold” is synchronization/dataflow (TMEM loads/stores) rather than an \(N\)-dependent MMA latency boundary; FP8/FP4 latency varies little while throughput scales strongly. ([arxiv.org](https://arxiv.org/html/2512.02189v1))  
• Open questions: public PTX ISA documentation enumerating all `tcgen05.mma` shapes; GB10 TMEM/latency behavior.  
• Carry-forward: kernel tuning should prioritize TMEM‑centric pipelines, warp specialization, and minimizing TMEM↔RF traffic for softmax/rescaling. ([arxiv.org](https://arxiv.org/html/2512.18134v1))  

---

## 9. Cross-Architecture Comparison (Phase 6)

This section produces the master comparison table requested. All Blackwell entries that depend on unknown boost clocks or SMEM bytes/cycle are tagged.

**Table 9.1 — Master comparison table (requested format)**

| Dimension | H100 SXM | H20 | B100 | B200 | GB10 |
|---|---:|---:|---:|---:|---:|
| Peak FP16 Tensor TFLOPS | 989.4 (baseline; dense) | 148 (baseline) | 1800 (UNVERIFIED) ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data)) | 2250 dense (HGX‑derived) ([nvidia.com](https://www.nvidia.com/en-us/data-center/hgx)) | UNKNOWN (FP16) ([nvidia.com](https://www.nvidia.com/en-us/project-digits/)) |
| Memory BW (TB/s) | 3.35 ([nvidia.com](https://www.nvidia.com/en-eu/data-center/h100/)) | 4.0 (baseline) | 8.0 (UNVERIFIED) ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data)) | 8.0 (UNVERIFIED/ESTIMATED) ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)) | 0.273 ([nvidia.com](https://www.nvidia.com/en-us/project-digits/)) |
| PAI (FLOPs/Byte) | ≈295.17 | 37 | 225 | 281.25 | UNKNOWN (FP16) |
| FLOPs/TCU/cycle | ≈1024 (baseline) | ≈256 (baseline) | UNKNOWN / scenario‑based | UNKNOWN / scenario‑based | UNKNOWN |
| SMEM BW (Bytes/cycle) | 128 (baseline assumption) | 128 (baseline) | UNKNOWN | UNKNOWN | UNKNOWN |
| Mem→Compute threshold \(N\) (FP16) | 64 (baseline) | 16 (baseline) | “None within 8–256” if TMEM‑staged (ESTIMATED) | “None within 8–256” if TMEM‑staged (UNVERIFIED) ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | UNKNOWN |
| SwapAB benefit (m16n64k16) | 1× | 4× | ≈1× (ESTIMATED) | ≈1× (ESTIMATED) | UNKNOWN |
| FP8 FLOPs/TCU/cycle (if avail) | ≈2× FP16 on Hopper (baseline) | ≈2× FP16 (baseline) | UNVERIFIED | UNVERIFIED | UNKNOWN |
| Mem→Compute threshold \(N\) (FP8) | similar‑order to FP16 under SMEM model | lower than FP16 on weak TCUs | “None within 8–256” if TMEM‑staged (ESTIMATED) | “None within 8–256” if TMEM‑staged (UNVERIFIED) ([arxiv.org](https://arxiv.org/html/2512.02189v1)) | UNKNOWN |

**Interpretation.** Blackwell’s most salient architectural change for this methodology is that the classic SMEM‑bandwidth‑limited plateau analysis becomes insufficient: TMEM and warp‑level MMA cause instruction latency to appear nearly tile‑size invariant in available SI‑LAT data, shifting the performance‑engineering center of gravity to scheduling and synchronization. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

---

**CHECKPOINT — Phase 6**  
• Key values: B200: 148 SMs, TMEM 256 KB/SM (16 TB/s read); `tcgen05.mma` SI‑LAT ≈ 11 cycles and tile‑size invariant; FP16 dense PAI ≈ 281 FLOPs/Byte (UNVERIFIED bandwidth). ([arxiv.org](https://arxiv.org/html/2512.02189v1))  
• Open questions: official B100/B200 clocks and SMEM bytes/cycle; B100 SM count; GB10 microarchitecture.  
• Carry-forward: cross‑arch guidance—Hopper: choose \(N\) to cross threshold; Blackwell: choose dataflow/synchronization strategy (TMEM) and warp specialization. ([arxiv.org](https://arxiv.org/html/2512.18134v1))  

---

## 10. Practical Guidance for Kernel Developers

The following guidance is the “actionable distillation” of the replicated Hopper workflow under Blackwell’s changes:

1. **Do not transplant Hopper’s \(N\)-threshold heuristics blindly.** On Hopper, choosing \(N\ge 64\) (H100) or applying SwapAB (H20) can move `wgmma` from SMEM‑dominated to compute‑dominated latency. On Blackwell, the available evidence indicates `tcgen05.mma` SI‑LAT is largely invariant with tile size, so the same lever may not exist. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

2. **Treat TMEM as the primary performance object, not an implementation detail.** TMEM has enormous per‑SM bandwidth and is additive with L1/SMEM in Tier‑1 measurements. Design kernels so that (i) accumulators remain in TMEM across chained operations and (ii) operands with reuse are staged into TMEM to reduce SMEM pressure and global traffic. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

3. **Expect synchronization and data movement to dominate for complex pipelines.** OPT_PIPE shows that Blackwell introduces synchronization operations around Tensor Memory loads/stores that force fundamentally different software pipelining and warp specialization strategies in attention kernels. This suggests that on Blackwell, “optimal schedule” is often about **hiding synchronization**, not about maximizing MMA tile \(N\). ([arxiv.org](https://arxiv.org/html/2512.18134v1))

4. **Precision scaling primarily changes throughput, not latency.** Tier‑1 measurements show latency changes modestly across precisions while throughput changes dramatically; consequently, low‑precision kernels can become bandwidth‑bound unless TMEM reuse and staging are strong. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

5. **Benchmark the *right* latency metric.** Hopper’s original analysis used steady‑state pipe‑filled latency. For Blackwell, the accessible data is SI‑LAT. When building your own Blackwell microbenchmarks, measure both:
   - dependency‑chain SI‑LAT for scheduling constraints, and  
   - steady‑state issue/throughput under realistic operand staging (SMEM vs TMEM) for kernel performance.

---

## 11. Limitations and Open Questions (with explicit list of ESTIMATED/UNVERIFIED items)

### 11.1 Inaccessible or missing official documents

- The NVIDIA “Blackwell datasheet” and “Blackwell architecture technical brief” endpoints encountered in this run were password‑protected, preventing extraction of official per‑SKU tables. ([nvdam.widen.net](https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703))  
  **Needed to confirm:** B100/B200 boost clocks, exact Tensor Core count/SM for data‑center Blackwell, SMEM bytes/cycle/SM, TMEM‑ISA shape set.

### 11.2 ESTIMATED/UNVERIFIED parameter list (non‑exhaustive but explicit)

- **B100 SM count:** ESTIMATED (assumed = 148).  
- **B100 tensor cores/SM and total tensor cores:** ESTIMATED.  
- **B100 peak FP16 TFLOPS and memory BW:** UNVERIFIED. ([anandtech.com](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data))  
- **B200 tensor cores/SM and total tensor cores:** ESTIMATED (4/SM).  
- **B200 per‑GPU HBM bandwidth 8 TB/s:** UNVERIFIED/ESTIMATED (supported by NVIDIA blog context; not a direct B200 datasheet here). ([developer.nvidia.com](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))  
- **Blackwell SMEM bandwidth (bytes/cycle/SM):** UNKNOWN (UNVERIFIED).  
- **GB10 SM count, tensor cores/SM, boost clocks, FP16 tensor TFLOPS:** UNKNOWN (UNVERIFIED). ([nvidia.com](https://www.nvidia.com/en-us/project-digits/))  
- **Blackwell `m64n16k16` / `m64n32k16` SI‑LAT values:** ESTIMATED based on invariance. ([arxiv.org](https://arxiv.org/html/2512.02189v1))

### 11.3 Methodological comparability limits

- Hopper baseline latencies in the provided write‑up are “pipe‑filled” (steady‑state) measurements, while Blackwell Tier‑1 latencies are “single‑instruction latency” (dependency‑chain SI‑LAT). These are *not* identical metrics, and a rigorous comparison requires Blackwell pipe‑filled `tcgen05.mma` sweeps.

### 11.4 High‑value experiments to resolve uncertainties

1. Publish/measure **Blackwell SMEM bytes/cycle/SM** and how `tcgen05.mma` consumes SMEM vs TMEM in steady state.  
2. Produce a Blackwell analogue of the Hopper `wgmma` latency sweep under continuous issue, controlling operand residency (SMEM‑only vs TMEM‑staged).  
3. Gather the same microbenchmarks for **GB10** to determine whether its Blackwell GPU slice matches B‑series behavior.

---

## 12. Conclusion

By replicating the Hopper instruction‑latency methodology and integrating Tier‑1 Blackwell microbenchmark evidence, we find that the central architectural shift from Hopper to Blackwell is not merely “more TFLOPS,” but a change in *how* Tensor Core work is issued and fed: Blackwell introduces warp‑level MMA via `tcgen05.mma` and a dedicated Tensor Memory (TMEM) tier with extremely high per‑SM bandwidth. ([arxiv.org](https://arxiv.org/html/2512.02189v1)) The most consequential consequence for optimization strategy is that Hopper’s N‑dimension‑dependent MMA latency behavior—which enabled large SwapAB wins on some Hopper SKUs—appears substantially weakened or eliminated in the available Blackwell SI‑LAT data, shifting profitability toward TMEM‑centric dataflow, software pipelining, and warp specialization to manage synchronization. ([arxiv.org](https://arxiv.org/html/2512.18134v1))

For B200 specifically, Tier‑1 measurements report nearly constant ~11‑cycle single‑instruction latency across tile sizes and precisions, while throughput scales strongly with reduced precision, reinforcing a spatial‑array interpretation and emphasizing memory/dataflow as the dominant constraint at low precision. ([arxiv.org](https://arxiv.org/html/2512.02189v1)) For GB10, while the product is real and well‑specified at the system level (memory bandwidth, unified memory, FP4 peak), GPU‑internal parameters are not publicly enumerated; thus all GB10‑internal conclusions remain UNVERIFIED. ([nvidia.com](https://www.nvidia.com/en-us/project-digits/))

The actionable recommendation is clear: on Blackwell, prioritize TMEM residency and schedule design first; treat Hopper‑era SwapAB heuristics as hypotheses to re‑benchmark, not as defaults.

---

## References (numbered, with URLs)

1. ARCH_BW — *Microbenchmarking NVIDIA’s Blackwell Architecture: An in-depth Architectural Analysis* (arXiv HTML).  
   `https://arxiv.org/html/2512.02189v1`

2. OPT_PIPE — *Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs* (arXiv HTML).  
   `https://arxiv.org/html/2512.18134v1`

3. NVIDIA HGX Platform page (HGX B200 baseboard specs incl. sparse vs dense footnote).  
   `https://www.nvidia.com/en-us/data-center/hgx/`

4. NVIDIA H100 product specifications (memory bandwidth, “with sparsity” tensor TFLOPS).  
   `https://www.nvidia.com/en-eu/data-center/h100/`

5. NVIDIA Hopper Architecture In-Depth (SMs and Tensor Cores per SM for H100 variants).  
   `https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/`

6. NVIDIA Technical Blog — *Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era* (TMEM per SM, memory bandwidth context).  
   `https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/`

7. NVIDIA Investor Press Release (Mar 18, 2024) — *NVIDIA Blackwell Platform Arrives to Power a New Era of Computing* (mentions “B200 Tensor Core GPUs” and “GB200 Grace Blackwell Superchip”).  
   `https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Blackwell-Platform-Arrives-to-Power-a-New-Era-of-Computing/default.aspx`

8. NVIDIA Newsroom — *NVIDIA Puts Grace Blackwell on Every Desk…* (GB10 Superchip description).  
   `https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips`

9. NVIDIA DGX Spark / Project DIGITS page (GB10 system specs: 128 GB unified memory, 273 GB/s, FP4 perf).  
   `https://www.nvidia.com/en-us/project-digits/`

10. NVIDIA RTX Blackwell GPU Architecture (PDF; Tier‑1 registry source, consumer/workstation oriented).  
   `https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf`

---
Learn more:
1. [https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Blackwell-Platform-Arrives-to-Power-a-New-Era-of-Computing/default.aspx](https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Blackwell-Platform-Arrives-to-Power-a-New-Era-of-Computing/default.aspx)
2. [https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips)
3. [https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703](https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703)
4. [https://arxiv.org/html/2512.02189v1](https://arxiv.org/html/2512.02189v1)
5. [https://www.nvidia.com/en-us/project-digits/](https://www.nvidia.com/en-us/project-digits/)
6. [https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
7. [https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data)
8. [https://www.nvidia.com/en-us/data-center/hgx](https://www.nvidia.com/en-us/data-center/hgx)
9. [https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
10. [https://www.nvidia.com/en-eu/data-center/h100/](https://www.nvidia.com/en-eu/data-center/h100/)
11. [https://arxiv.org/html/2512.18134v1](https://arxiv.org/html/2512.18134v1)