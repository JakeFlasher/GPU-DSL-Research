### Context recap (what Stage 1 implies for Stage 2)

The seed (*Linear Layouts*, ASPLOS’26 / arXiv:2505.23819v3) is fundamentally a **compiler-side algebraization** of tensor layouts: represent layouts as linear maps over \(\mathbb{F}_2\), then systematically derive conversions and pick data-movement primitives (warp shuffles, `ldmatrix/stmatrix`, vectorized ld/st, shared-memory staging, etc.).

Stage 1 also surfaced the key meta-fact: **no verified “descendants” yet** (paper is effectively brand-new as of Jan 26, 2026), so “gaps” here are not “X cited it and failed,” but “what breaks when you push this method into 2026 hardware/workloads where the bottleneck migrates.”

Below are **three ISCA/MICRO-grade gaps** that emerge when you treat Linear Layouts as “solved at the SM/warp level” but still missing **microarchitectural closure** across (a) chiplet NoCs + sliced L2s, (b) register/occupancy economics, (c) scratchpad/bank-mapping realism and evolution.

---

## 0) At-a-glance: the 3 research gaps

| Gap ID | What “fails” if you naïvely apply Linear Layouts | Bottleneck that replaces “layout conversion cost” | Mechanism direction (hypothesis) | Typical venue gravity |
|---|---|---|---|---|
| **G1** | Layouts optimize *within* an SM, but **create L2-slice/NoC hotspots** on chiplet/distributed-cache GPUs | **NoC + L2 slice bandwidth / queuing**, not warp-local data movement | **Programmable / compiler-steered cache-slice hashing** co-designed with layout algebra | ISCA/MICRO |
| **G2** | Replacing shared-mem conversions with shuffles increases **register pressure & live ranges**, causing **occupancy cliffs** | **Regfile capacity + warp scheduling**, not shared-mem bank conflicts | **Late-shuffle + “layout-as-renaming”** (avoid materializing conversions) | MICRO (pipeline detail), ISCA (mechanism) |
| **G3** | “Optimal swizzling” assumes stable bank mapping / transaction rules; new scratchpads (tensor memory, changed banks) break static optimality | **Scratchpad throughput variability + replays**, not conversion correctness | **Hardware-configurable \(\mathbb{F}_2\) address remap “views”** for shared memory + lightweight online selection | ISCA/MICRO |

I’ll now detail each gap with the required structure.

---

# G1 — Chiplet GPUs: Layout conversions “solve SM,” but blow up NoC/L2 slice behavior

## 1) Strawman failure
**Strawman:** “Just use the seed’s linear-layout conversion + optimal swizzle, and we’re done.”

**Failure mode on modern chiplet / distributed L2 GPUs:**
- The seed optimizes **warp/CTA-local movement** (shuffles vs shared memory) and **shared-memory bank conflicts**, but it does **not** model:
  - **Which L2 slice becomes the home** for a given global address,
  - **NoC path length / contention** to that slice,
  - slice-level **MSHR pressure** and queuing delays.
- A layout that improves coalescing or reduces shared-mem conflicts can *still* induce **address-bit regularity** that collapses many warps onto a subset of slices (classic hashing/pathology problem).
- On multi-die designs, that becomes **remote-slice traffic amplification**:
  - higher NoC utilization,
  - higher average memory access latency (queueing dominates),
  - sometimes lower effective HBM BW due to backpressure.

In short: the seed’s “layout win” is local; the **system-level home-placement** can become the new limiter.

## 2) The insight (non-obvious architectural observation)
Cache-slice selection in many designs is effectively a **(mostly) XOR/hash of address bits**. That is, it often behaves like a **linear (or affine) map over \(\mathbb{F}_2\)** on a subset of address bits.

So we have two composable maps:

```
(logical coords) --layout--> (address bits A) --slice-hash--> slice_id
                L (F2)                    H (often F2-ish)
```

If \(L\) makes certain low/high address bits *low-entropy* across a warp-group (common when you “regularize” access for vectorization), and \(H\) depends heavily on those bits, you get **slice collapse**.

**Key observation:** This is *the same algebraic regime as the seed paper*, but the seed stops at the SM boundary. The missing piece is **including “home mapping” into the layout objective**.

## 3) The hypothesis (concrete mechanism)
### Hypothesis: *Compiler-steered programmable slice hashing*, co-designed with linear layouts, reduces NoC/L2 hotspots without changing program semantics.

#### Mechanism sketch
Add a small number \(K\) of **selectable slice-hash modes** per kernel launch:

- Slice selection uses a configurable XOR network:
  - \(slice\_id = H_k(A)\), where \(k \in \{0,\dots,K-1\}\)
  - each \(H_k\) is a linear (or affine) map over selected address bits.

- **Critical correctness constraint:** you cannot arbitrarily change line “home” while lines are resident.
  - Practical approach: restrict reconfiguration to **kernel boundaries**, with a **flush/invalidate** of the affected cache(s) at launch (or require empty L2 via epoching).
  - This is realistic: GPU runtimes already have kernel boundaries and can afford some launch-time work for long kernels.

#### Why this is plausible hardware
- XOR networks on ~10–20 bits are tiny (area/power).
- Control state is a few dozen bytes per GPU (or per slice).
- Timing impact: a small XOR tree in the address routing stage (usually not on the critical path vs SRAM access).

#### How it leverages the seed
The compiler already has:
- the per-thread/per-warp address functions implied by layout algebra,
- the ability to compute which address bits vary across lanes/tiles,
- a natural objective: maximize entropy of slice_id across concurrent warps **subject to** coalescing constraints.

So the compiler can pick \(k\) (hash mode) that best “decorrelates” the access set.

### ASCII block diagram
```
              +-------------------+
coords(i,j) ->| Linear Layout L   |-> addr bits A[...]
              +-------------------+
                         |
                         v
              +-------------------+
              | Slice Hash H_k    |  (select k per kernel)
              +-------------------+
                         |
                         v
                    NoC routing -> L2 slice -> HBM
```

## Evaluation feasibility check
**Simulation infrastructure required:**
- Needs **cycle-accurate NoC + multi-slice L2** modeling.
- Feasible with:
  - **GPGPU-Sim** (core + memory) extended with:
    - multiple L2 slices,
    - a simple mesh/crossbar NoC (can integrate something BookSim-like), and
    - configurable slice-hash modes.
  - Alternatively (if you already have it): **gem5 Garnet** for NoC + a GPU front-end is harder; GPGPU-Sim is the pragmatic route.

**What to measure (must-haves for ISCA/MICRO credibility):**
- Per-slice: occupancy, MSHR utilization, queueing delay, hit rate
- NoC: injection rate, average latency, saturation point
- End-to-end: kernel time, achieved HBM BW, IPC
- Energy proxy: GPUWattch-style estimates for NoC + L2 + HBM (or at minimum: dynamic flit counts + SRAM access counts)

---

# G2 — Occupancy cliffs: “better conversions” can be slower due to regfile economics

## 1) Strawman failure
**Strawman:** Replace shared-memory conversions with warp shuffles/permutations whenever algebra says it’s legal.

**Failure mode in real fused kernels / LLM inference operators:**
- Shuffle-based conversions often:
  - increase the number of live temporaries,
  - extend live ranges (producer → conversion → consumer),
  - inflate **registers per thread**.
- GPUs are extremely sensitive to **register-limited occupancy**:
  - occupancy drops → less latency hiding,
  - memory stalls dominate,
  - you get a performance cliff even if instruction count drops.

So the seed’s local “instruction efficiency” objective can directly conflict with **warp residency**.

## 2) The insight
Many layout conversions are:
- **pure permutations** of registers (intra-thread), or
- **pure lane permutations** (warp shuffle patterns),
- with no arithmetic.

The *data* doesn’t change, just **where you read it from**.

Therefore, a large class of conversions can be reframed as:
- **operand selection / renaming** at the point of *use*,
- rather than **materializing** a converted tensor value that must stay live.

This is a classic microarchitectural principle: *make the consumer flexible instead of generating temporaries*.

## 3) The hypothesis (concrete mechanism)
### Hypothesis: *Late-shuffle (consumer-side lane remap) + register-view indirection* reduces register pressure and preserves occupancy while retaining the seed’s conversion optimality.

#### Mechanism A: Late-shuffle operand fetch
Instead of:
```
tmp = shfl(src, lane_map)
y   = f(tmp)
```

Support:
```
y = f( shfl(src, lane_map) )   // fused: consumer reads remote lane directly
```

Microarchitecturally:
- Add support for an instruction operand to specify **(source_lane, source_reg)**.
- The operand collector / register read stage fetches from the indicated lane via the existing shuffle cross-lane network.
- The shuffle result is **not written back** to a new register unless explicitly requested.

This reduces:
- instruction count (fewer explicit shuffles),
- register writes,
- most importantly: **temporary registers** that inflate reg usage.

#### Mechanism B: Register permutation as “view” (intra-thread)
For intra-thread register permutations (common in layout conversions):
- Support a small **register-group view descriptor**:
  - base physical registers + permutation mask
- Reads are remapped at decode/operand-collection time.
- Writes either go through the inverse mapping or force “materialization.”

This is more invasive, but you can scope it:
- only for compiler-declared “tensor tiles” (register groups),
- only for permutations (not arbitrary indexing).

### ASCII diagram (late-shuffle)
Baseline:
```
Lane RF ----shuffle net----> tmp RF ----> ALU/FMA ----> dst RF
         (write tmp)
```

Late-shuffle:
```
Lane RF ----shuffle net----> Operand Collector ----> ALU/FMA ----> dst RF
         (no tmp register file write)
```

#### Hardware overhead realism
- Shuffle networks already exist; the delta is:
  - extra mux/control to allow non-shuffle instructions to source an operand via shuffle path,
  - scoreboard/hazard tracking for remote-lane source,
  - possible pressure on operand collector bandwidth.
- Expectation: limit to **1 late-shuffle operand per instruction** (or per cycle) to contain complexity.
- Power: fewer RF writes can *save* energy; shuffle-net toggling cost must be modeled.

## Evaluation feasibility check
**Simulation infrastructure required:**
- Needs a GPU core pipeline model with:
  - register file port constraints,
  - shuffle unit throughput/latency,
  - occupancy model.
- **GPGPU-Sim** is sufficient if you extend:
  - instruction set / scheduling for late-shuffle,
  - RF write suppression,
  - warp stall breakdown.

**What to report (to avoid “compiler trick” criticism):**
- reg/thread distribution, achieved occupancy
- stall reason breakdown (scoreboard, memory, execution dep)
- shuffle unit utilization vs baseline
- performance across kernels that are:
  - shuffle-heavy conversions,
  - shared-mem heavy conversions,
  - mixed compute/memory (LLM attention blocks are a good stressor)

---

# G3 — Swizzle optimality is brittle: scratchpad banking and memory primitives evolve

## 1) Strawman failure
**Strawman:** The compiler computes an “optimal” swizzle (bank-conflict-free, max vectorization) assuming:
- known bank indexing,
- known transaction granularity (e.g., 128B segments),
- stable behavior across GPU generations.

**Failure mode:**
- Bank mapping and scratchpad microarchitecture are not guaranteed stable:
  - banks may be hashed,
  - special memories (tensor memory / new scratchpads) may have different conflict rules,
  - vector transaction decomposition rules can change,
  - replay mechanisms differ.
- Static swizzles can become “provably optimal for the wrong model,” leading to:
  - unexpected bank conflicts/replays,
  - regression vs simpler layouts,
  - portability pain (exactly what the seed tries to reduce).

## 2) The insight
The seed’s key mathematical object—**invertible linear maps over \(\mathbb{F}_2\)**—is *extremely cheap in hardware* if implemented as XOR networks on address bits.

Instead of forcing the compiler to “guess” the bank mapping model perfectly, push part of the mapping into the hardware as a **configurable address remap layer**. Then:
- you get portability across banking changes,
- you can adapt at runtime if needed,
- and you can keep the programming model aligned with the seed’s algebra.

## 3) The hypothesis (concrete mechanism)
### Hypothesis: Shared memory should support multiple **configurable \(\mathbb{F}_2\) address views** (“CLAR”: Configurable Linear/Affine Remapper).

#### Mechanism: CLAR views for shared memory
Add a per-CTA (or per-SM) remapper:

- For each view \(v\), define:
  - \(A' = M_v A \oplus c_v\) over \(\mathbb{F}_2\) for selected low-order address bits
  - \(M_v\) constrained to be invertible over the remapped bit subset (to avoid aliasing)
- Bank selection and row decode use \(A'\), not \(A\).

Expose to ISA:
- shared memory ld/st take an extra small **view-id** operand (2–4 bits).
- compiler chooses views for phases (e.g., store under one view, load under same view, or switch view for different access phases).

**Why this helps:**
- You can pick a view that minimizes conflicts for a particular access pattern **without rewriting the kernel**.
- If scratchpad banking changes across gens, the vendor can ship a mapping of “known good” \(M_v\) presets; compiler just selects a view ID.
- For kernels with multiple phases (common in attention), you can switch view IDs per phase with negligible overhead.

#### Hardware overhead realism
- Storage:
  - Suppose you remap 12–16 low bits (enough to affect bank + within-tile addressing).
  - A full 16×16 matrix is 256 bits; with 4 views that’s 128B per CTA worst-case (but you can compress by limiting structure, e.g., upper-triangular + identity on some bits).
- Logic:
  - XOR network per remapped bit: ~16*(fan-in) XOR gates.
- Timing:
  - Address generation stage adds XOR depth; can pipeline if needed.
- Power:
  - XOR is cheap, but toggles every shared-mem access; must be included in energy model.

### ASCII diagram
```
shared addr A ---> [ CLAR: A' = M_v*A XOR c_v ] ---> bank select / row decode ---> SRAM banks
                     ^ view-id (v)
```

## Evaluation feasibility check
**Simulation infrastructure required:**
- Needs detailed shared memory bank/replay modeling.
- **GPGPU-Sim** is again sufficient:
  - implement CLAR remap before bank selection,
  - measure bank conflicts, replays, and shared-mem throughput.
- Optional but strong: add a simple runtime policy that selects among a small set of views (static per kernel, or 2-phase).

**What to report:**
- bank conflict rate / replay count (per instruction, per warp)
- achieved shared-mem BW (bytes/cycle)
- kernel speedup breakdown: fewer replays vs better vectorization vs fewer instructions
- sensitivity across “architectures” by varying:
  - number of banks,
  - bank hashing policy,
  - transaction segmentation rules

---

# Decision matrix (scoring the 3 gaps)

| Gap | Novelty | Implementation Complexity | Venue Fit (ISCA vs ASPLOS) | Rationale (1–2 lines) |
|---|---|---|---|---|
| **G1: Slice-hash × layout co-design** | **High** | **Med** | **ISCA/MICRO** | Cross-layer but clearly microarchitectural: NoC/slice bottlenecks are first-order on chiplets; mechanism is small (hash modes) but needs strong NoC eval. |
| **G2: Late-shuffle / layout-as-renaming** | **Med–High** | **High** | **MICRO (strong), ISCA (ok)** | Deep pipeline interaction + RF/collector design; high payoff via occupancy preservation, but more invasive to GPU core model and ISA semantics. |
| **G3: Shared-mem CLAR views** | **High** | **Med** | **ISCA/MICRO (also ASPLOS-friendly)** | Clean mechanism with tractable cost model; tightly aligned with seed’s \(\mathbb{F}_2\) framing; eval is mostly within shared-mem model (easier than NoC). |

---

## If forced to pick “best first bet” for an ISCA/MICRO submission
- **Most balanced (novel + feasible):** **G3**  
  (tight mechanism, measurable wins, realistic overhead, simulator-friendly)
- **Highest upside (but eval-heavy):** **G1**  
  (if you can convincingly model NoC + L2 slices, this hits a 2026 pain point)
- **Riskiest but potentially most publishable in MICRO:** **G2**  
  (needs careful pipeline/ISA design and strong evidence it avoids occupancy cliffs without killing frequency)

If you want, in Stage 2.5 I can turn these into a **one-page “mechanism spec + baseline + ablation plan”** for the top choice (including the exact counters/plots reviewers will demand: stall breakdowns, replay histograms, slice utilization heatmaps, and sensitivity to bank/hash parameters).