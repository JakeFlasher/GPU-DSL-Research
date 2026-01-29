# Revised Problem Formulation (Hardware–Math Gap, made precise)

Modern GPU kernels increasingly rely on **software-managed pipelining**: overlapping global→shared transfers with tensor-core compute, often with **warp specialization** (dedicated “producer” warps issuing transfers, “consumer” warps computing). The core compiler problem is no longer just finding a good initiation interval—it is enforcing **hardware-faithful correctness** while preserving overlap.

We frame TITAN as addressing a specific **Hardware–Math Gap**: the semantic gap between (i) the *mathematical* pipeline schedule a compiler wants to express and verify, and (ii) the *actual ISA contracts* that make that schedule correct on real GPUs.

## The Hardware–Math Gap has two dominant failure modes

### (G1) Proxy Consistency Gap (memory visibility across proxy domains)
On NVIDIA Hopper-class GPUs, key asynchronous mechanisms execute in **distinct proxy domains** (at minimum: **generic**, **async**, and **tensormap**). Correctness requires not only ordering operations, but also **establishing cross-proxy visibility** with the right **proxy fences**, and exploiting **implicit fences on completion** to avoid over-fencing.

This is not a “performance detail”: missing a required proxy fence can produce **undefined behavior**, while inserting fences conservatively can destroy the very overlap the pipeline was meant to achieve.

> **TITAN’s stance:** Unlike prior pipeline IRs that treat “fence” as a monolithic primitive, TITAN models **proxy consistency explicitly**: proxy-domain effects are part of the IR contract, and the compiler proves that every cross-proxy read has a justified ordering path.

### (G2) ISA Mismatch Gap (handle/phase tokens vs counter completion)
Pipeline IRs often assume per-transfer **capability tokens**: “wait on exactly this copy.” That assumption matches some hardware mechanisms (e.g., phase/state-based completion), but not others.

- On NVIDIA, completion is naturally expressed using **phase/state** associated with barrier arrival (state tokens / parity discipline).
- On AMD, completion is fundamentally **counter-based** (`s_waitcnt`), not handle-based. A naïve “wait(token)” abstraction is not faithfully representable without either weakening semantics or serializing the schedule.

> **TITAN’s stance:** Portability requires acknowledging the mismatch. TITAN provides a **portability layer** that makes token semantics explicit and parameterized: exact (phase/state) tokens where hardware supports them, and **Virtual Token Mapping** (prefix/stream ordinals) where hardware provides only counters.

## What must be proven (and what earlier drafts implicitly assumed)

A correct pipeline system must rule out the reviewer-grade failures that occur in real kernels:

1. **Ring-buffer overwrite or deadlock under imbalance** (consumer slower than producer).
2. **WAR/UAF hazards** from decoupled tensor-core consumption (issue ≠ done reading).
3. **Multi-warp readers** (warp specialization) where “done reading” is collective, not scalar.
4. **Tail / predication hangs** (“wait on a token that never arrives”) from short loops and partial tiles.

Accordingly, TITAN’s problem is:

> Given a loop with per-iteration tiles and a target depth \(D\), generate a pipelined schedule and IR such that (i) safety holds for all executions (including imbalance and tails), (ii) the IR is **lowerable to the target ISA without semantic gaps**, and (iii) performance does not regress due to over-fencing, over-synchronization, or register-pressure blowups.

---

# Methodology (explicit contract + backend-parameterized tokens)

TITAN is built around a **single principle**: *correctness must be established at the IR level using a protocol that is (a) runtime-realizable and (b) hardware-faithful at lowering time.*

This is implemented via:

1. A **protocol-shaped IR** (stages, epochs, acquire/produce/wait/release).
2. An **augmented typestate** that verifies the protocol (including wrap-around, reader multiplicity, and compute completion).
3. A **portability layer** that defines what a “token” means per backend, including the required **AMD Virtual Token Mapping** to `s_waitcnt`.

The rest of the system (scheduler search, warp partitioning, unrolling) is constrained by these contracts rather than hand-waved around them.

---

# Revised System Architecture (The Fix)

## Architecture overview

TITAN compiles pipelined GPU kernels through four enforceable stages:

1. **Pipeline Construction (IR shaping):** rewrite candidate regions into a staged ring-buffer form with explicit acquire/produce/wait/release structure and guarded prologue/epilogue.
2. **WR–SMS Scheduling:** schedule producer/consumer roles with resource constraints and a register budget objective; reject or degrade configurations that violate budgets.
3. **Verification:** run (i) **augmented typestate** (protocol correctness) and (ii) **proxy/effects verification** (visibility correctness), plus hardware legality checks (e.g., TMA constraints).
4. **Lowering (backend-specific):** map the verified IR to NVIDIA or AMD primitives, using backend-parameterized token semantics.

Crucially: if any step fails, TITAN falls back to a deterministic baseline (e.g., \(D=2\), no warp specialization), rather than emitting “probably-correct” code.

---

## 1) Augmented Typestate (Formal Contract) — wrap-around, multi-reader, async compute

Earlier pipeline systems often modeled a stage as a scalar “Full/Empty” bit and assumed steady-state balance. TITAN instead verifies a **time-indexed protocol** with explicit backpressure and wrap-around semantics.

### 1.1 Time-indexed resources: stage \(s\) and epoch \(e\)

Let \(D\) be ring depth. Each logical iteration \(t\) maps to:

- stage index: \(s(t) = t \bmod D\)
- epoch: \(e(t) = \lfloor t / D \rfloor\)

TITAN’s IR and typestate never speak about “stage \(s\)” without also tracking **epoch \(e\)**. This is the critical move that eliminates ABA/parity ambiguity by construction.

We define the following affine/linear resources:

- \(\textsf{Free}(s,e)\): permission to overwrite stage \(s\) for epoch \(e\).
- \(\textsf{Slot}(s,e)\): *exclusive write capability* to the stage buffer for \((s,e)\). (Affine; cannot be duplicated.)
- \(\textsf{ReadyTok}(s,e)\): evidence that produced data for \((s,e)\) is complete and visible to consumers.
- \(\textsf{ReadPerm}(s,e)\): read permission for consumers; splittable across consumer warps.
- \(\textsf{ComputeTok}(s,e)\): evidence that compute consuming \((s,e)\) may still be reading stage memory asynchronously.
- \(\textsf{ConsumedTok}(s,e)\): evidence that all consumers are done reading stage memory for \((s,e)\).

### 1.2 Two-semaphore handshake (explicit backpressure)

For each \((s,e)\), correctness is governed by a **two-semaphore handshake**:

- **Acquire** blocks on \(\textsf{Free}(s,e)\) to obtain \(\textsf{Slot}(s,e)\).
- **Produce** (async copy) transitions the slot into a ready state and yields \(\textsf{ReadyTok}(s,e)\).
- **Consume** waits on \(\textsf{ReadyTok}(s,e)\), issues compute, then proves completion before releasing.
- **Release** produces \(\textsf{Free}(s,e+1)\) (i.e., frees the slot for the next epoch).

This replaces the “static partial order” assumption with a **runtime-realizable backpressure mechanism** that remains correct under imbalance (consumer slower than producer).

### 1.3 State machine (per \((s,e)\))

For each stage \(s\) and epoch \(e\):

\[
\textsf{Free}(s,e)
\rightarrow \textsf{InFlight}(s,e)
\rightarrow \textsf{Ready}(s,e)
\rightarrow \textsf{Reading}(s,e)
\rightarrow \textsf{Consumed}(s,e)
\rightarrow \textsf{Free}(s,e{+}1)
\]

Where:

- \(\textsf{InFlight}(s,e)\) corresponds to: have \(\textsf{Slot}(s,e)\) and outstanding async produce.
- \(\textsf{Ready}(s,e)\) corresponds to: have \(\textsf{ReadyTok}(s,e)\).
- \(\textsf{Reading}(s,e)\) corresponds to: consumers hold \(\textsf{ReadPerm}(s,e)\) and possibly \(\textsf{ComputeTok}(s,e)\).

### 1.4 Reader multiplicity (multi-warp correctness)

Warp specialization makes “the consumer” a **set of warps**, not a scalar thread. TITAN makes this explicit with either:

- **Fractional permissions (paper-clean):** \(\textsf{ReadPerm}(s,e)\) is splittable into \(k\) pieces, one per consumer warp; \(\textsf{ConsumedTok}(s,e)\) is derivable only after rejoining all pieces.
- **Collective release (implementation-friendly):** `stage.release` is specified as a **uniform collective** across consumer warps (or across a consumer-only barrier), and the verifier enforces non-divergence in the release path.

Either way, TITAN proves:

> A stage cannot become \(\textsf{Free}(s,e+1)\) until **all** consumer warps have completed their reads for \((s,e)\).

### 1.5 Async compute consumption (WAR/UAF closure)

On modern tensor-core pipelines, “issue compute” does not imply “done reading shared.” TITAN introduces a **compute-consumption token**:

- `compute.issue(...) -> ComputeTok(s,e)`
- `compute.wait(ComputeTok(s,e))` is required before `stage.release`.

This closes the classic WAR/UAF hole:

> Overwriting the stage buffer is illegal until both (i) all consumer warps have logically finished and (ii) the hardware compute pipeline has completed any deferred reads.

### 1.6 Fundamental safety invariant (no lapping)

The typestate plus runtime protocol enforces the ring-buffer boundedness invariant:

\[
\forall t:\quad \textsf{produced}(t) - \textsf{consumed}(t) \le D
\]

This is the precise condition that prevents wrap-around aliasing (ABA), and it is enforced operationally because `stage.acquire(s,e)` blocks until the corresponding `Free(s,e)` is available.

---

## 2) Prologue/Epilogue and Predication (no “wait that never arrives”)

Modulo-scheduled pipelines fail in practice on tails unless prologue/epilogue are explicit. TITAN therefore specifies a **single unified time loop** that makes production and consumption guards structurally consistent.

Let \(N\) be the number of logical tiles and \(D\) be depth. Define \(P = D-1\). TITAN generates:

- time loop \(t = 0 \ldots N+P-1\)
- produce tile index \(p = t\)
- consume tile index \(c = t-P\)

Producer executes only if \(p < N\). Consumer executes only if \(0 \le c < N\). Since \(p = c+P\), every consumer wait is paired with a dominating producer produce, by construction.

### Predication / partial tiles
TITAN requires one explicit semantics choice; both are verifiable:

1. **0-byte arrive semantics (default):** the producer always “arrives” and produces a `ReadyTok(s,e)` even if `bytes=0`. The consumer never hangs; correctness is maintained via masked compute or defined zero-fill behavior.
2. **Poison semantics:** producer can yield `ReadyPoison(s,e)`; consumers must branch to a safe path. The verifier enforces poison handling.

If neither is feasible (e.g., complex ragged epilogues), TITAN uses a **non-pipelined tail fallback** rather than risking hangs.

### Warp specialization: identical loop bounds
Producer and consumer roles must share the same outer loop bounds and must reach `pipeline.finalize`. TITAN explicitly forbids “producer exits early” patterns that strand consumers on synchronization objects.

---

## 3) Hardware Constraints (TMA legality, alignment, swizzling) **(required)**

TITAN treats hardware legality as a first-class verification step, not an appendix.

### 3.1 TMA (bulk tensor copy) constraints affect ring-buffer layout
When lowering staged copies to NVIDIA TMA-like bulk tensor operations, the ring buffer must satisfy:

- **Shared-memory destination alignment** constraints (e.g., 128-byte alignment requirements for multidimensional bulk tensor copies).
- **Global address and stride** constraints (e.g., baseline alignment and stride multiples), strengthened under **swizzle modes**.
- **Swizzle repeat boundaries** (e.g., repeating at 256/512/1024-byte granularities depending on mode), which constrain per-stage padding and indexing.

TITAN therefore includes a **TMA legality verifier** that checks stage base alignment and stage stride/padding rules. If illegal, TITAN either (i) pads the stage stride to restore legality, or (ii) falls back to a non-TMA path.

### 3.2 Barrier object constraints
Barrier objects and their placement (CTA shared vs cluster shared) impose additional constraints on where per-stage synchronization state can live and which operations are legal. TITAN’s lowering is constrained by these rules; legality is checked before scheduling commits to a configuration.

---

## 4) Hardware-faithful synchronization & proxy model (strength, not a liability)

TITAN’s IR is intentionally **isomorphic to real primitives**, avoiding “nice-looking” token APIs that cannot be lowered.

### 4.1 NVIDIA: phase/state-based barrier completion
On NVIDIA, waiting is not “wait(barrier)”; it is waiting on completion of a particular **phase/state**. TITAN models this explicitly by representing an arrival state token returned by the arrive operation, and requiring it for waits.

### 4.2 Two completion mechanisms (loads vs stores)
TITAN distinguishes:

- barrier-tracked completion (common for global→shared movement), and
- async-group completion (required for certain bulk operations, notably store-direction mechanisms).

This prevents the “one barrier protocol fits all directions” error.

### 4.3 Proxy domains as effects (generic / async / tensormap)
Every TITAN memory operation is annotated with a proxy domain, and TITAN inserts only the fences justified by the IR’s effect graph.

This is deliberately defensive:

- **Correctness:** we never rely on undocumented cross-proxy visibility.
- **Performance:** we avoid redundant fences by modeling **implicit proxy fences on completion** where the hardware provides them.

---

## 5) Portability Layer: NVIDIA tokens vs AMD Virtual Token Mapping

### 5.1 Contract: token semantics are backend-parameterized
TITAN does not claim that “a token means the same thing everywhere.” Instead, TITAN defines a common *verification interface* (“must not read before ready; must not overwrite before consumed”) and lets backends implement the token primitives with the strongest faithful semantics they can support.

### 5.2 AMD Virtual Token Mapping (counter-based completion) **(explicit)**
On AMD, `s_waitcnt` is counter-based, so per-transfer capability tokens are not faithfully representable in general. TITAN therefore uses **Virtual Token Mapping**:

- A token is a **stream ordinal** in a per-class issue stream: \(\textsf{VTok}(\textsf{class}, n)\).
- The verifier enforces **monotone waits** (no out-of-order waiting) within a class.
- Transfers belonging to a single stage are issued as a **contiguous group**, and the consumer waits the group token (stage-level granularity).

Operationally:

- Issuing an async op increments the class counter and yields \(\textsf{VTok}(\textsf{class}, n)\).
- Waiting on \(\textsf{VTok}(\textsf{class}, n)\) lowers to the appropriate `s_waitcnt` threshold for that class, advancing a single monotone “retirement frontier.”

If a candidate schedule would require selective waiting (e.g., waiting on “the middle” op while leaving older ones outstanding), it is rejected or rewritten (e.g., grouping, smaller \(D\), or disabling specialization). This makes AMD portability explicit, verifiable, and non-illusory.

---

# New Evaluation Plan (designed to defend against the known rejection risks)

TITAN’s evaluation must prove three things simultaneously:

1. **Correctness under adversarial kernel behaviors** (imbalance, predication, multi-warp consumption).
2. **Performance stability** (no cliffs from warp specialization; no over-fencing; controlled register pressure).
3. **Portability without semantic cheating** (AMD counter model acknowledged; performance impact quantified).

We structure the evaluation accordingly.

---

## A) Correctness validation (stress tests + negative tests)

### A1. Stress test: ring buffer saturation (imbalance)
**Goal:** Demonstrate that explicit backpressure prevents overwrite/ABA and avoids deadlock when consumers lag.

- Construct kernels where consumer throughput is artificially reduced:
  - heavy epilogue,
  - forced shared-memory bank conflicts,
  - controlled instruction-level stalls,
  - reduced occupancy via register inflation.
- Sweep \(D \in \{2,3,4,5\}\) and warp partitions.
- Metrics:
  - no hangs,
  - correctness of outputs (bitwise or tolerance as appropriate),
  - measured producer stall time at `stage.acquire` (should increase smoothly under imbalance, not corrupt).

**Ablation:** remove blocking acquire (or replace with non-blocking) to demonstrate the exact failure mode TITAN prevents.

### A2. ABA / wrap-around adversary
**Goal:** Demonstrate epoch-indexed safety.

- Use very small \(D\) (e.g., 2) and many iterations with randomized per-iteration stalls.
- Validate that consumers never observe a “reused parity” as a false-ready signal.

### A3. Multi-warp reader correctness
**Goal:** Demonstrate that a stage is not released until all consumer warps are done.

- Construct consumer regions where warps have unequal work (e.g., predicated compute, divergent epilogue).
- Validate that:
  - releases occur only after all warps complete,
  - verifier rejects intentionally broken code that attempts early release.

### A4. Async compute hazard test (WAR/UAF closure)
**Goal:** Show that compute-consumption tokens are necessary and effective.

- Implement a kernel variant where compute reads from shared in a pipelined manner (representative of tensor-core pipelines).
- Compare:
  1. with `compute.wait` enforced before `stage.release`,
  2. without it (unsafe variant).
- Validate that (2) can fail under stress, while (1) does not.

### A5. Prologue/epilogue and tail correctness
**Goal:** No “wait never arrives” hangs.

- Sweep \(N\) relative to \(D\): \(N < D\), \(N = D\), \(N \gg D\).
- Test ragged tiles and predication-heavy shapes.
- Compare the unified-loop pipelined path vs the fallback tail path; ensure both are correct and no-hang.

### A6. Proxy fence litmus suite (NVIDIA)
**Goal:** Make proxy consistency a measurable strength.

- Litmus patterns:
  - generic writes → async reads (shared buffer and barrier init),
  - generic writes → tensormap proxy reads (descriptor updates),
  - completion → generic consumption (validate reliance on implicit fence).
- Compare:
  - TITAN minimal fences (effect-driven),
  - conservative always-fence,
  - intentionally under-fenced (expected to be incorrect).
- Metrics:
  - correctness,
  - fence count,
  - overlap/performance impact.

### A7. TMA legality tests (alignment/swizzle)
**Goal:** Demonstrate the legality verifier and padding behavior.

- Generate stage layouts spanning alignments/strides/swizzle modes.
- Validate:
  - illegal cases are rejected or padded,
  - padded cases remain correct,
  - measure any padding overhead.

---

## B) Performance evaluation (end-to-end + ablations targeted to risks)

### B1. End-to-end benchmarks
Evaluate representative workloads that exercise staged async transfers and tensor-core compute:

- GEMM families across shapes (including small tiles and low-\(K\)),
- attention-style blocks (short-\(K\), heavy epilogue),
- fused epilogues (activation, reduction, stores),
- memory-bound kernels to test “don’t over-specialize” behavior.

Report:
- throughput (e.g., effective TFLOP/s or GB/s),
- achieved occupancy,
- register count (post-regalloc),
- shared memory footprint,
- fence and barrier instruction counts,
- tail overhead (when fallback triggers).

### B2. Ablation: warp specialization heuristic (avoid cliffs)
**Goal:** Prove TITAN does not regress on small tiles/low compute intensity.

- Compare:
  - always-on specialization,
  - TITAN heuristic (conditional),
  - always-off specialization.
- Focus on small tiles, short \(K\), and reg-tight configurations.

### B3. Ablation: register-aware scheduling
**Goal:** Show that reg pressure is controlled by design, not by hope.

- Compare:
  - naive modulo scheduling (no reg budget),
  - TITAN WR–SMS with reg budget + mitigation,
  - optional regalloc feedback variant (if implemented).
- Report occupancy stability and spill counts.

### B4. Ablation: proxy modeling (performance benefit of being correct)
**Goal:** Show TITAN’s explicit proxy modeling is not just “safer” but also avoids over-fencing.

- Minimal effect-driven fences vs conservative fences.
- Report overlap efficiency and throughput differences.

---

## C) Portability evaluation (AMD reality acknowledged, overhead quantified)

### C1. Ablation: AMD Virtual Token Mapping overhead
**Goal:** Quantify the cost of counter-based completion semantics.

- Compare on AMD:
  - TITAN Virtual Token Mapping (monotone waits, grouped per stage),
  - conservative `s_waitcnt(0)`-style full waits,
  - (where possible) a serialized per-transfer emulation baseline (expected slow).
- Measure:
  - overlap (in-flight operations),
  - stall cycles at waits,
  - end-to-end performance.

### C2. Semantic coverage + fallback rate
**Goal:** Show when TITAN can preserve fine-grained overlap and when it must degrade.

- Report the fraction of configurations rejected due to non-representable token patterns.
- Report which fallbacks trigger (smaller \(D\), grouped transfers, no warp specialization).

---

## D) Compile-time budget (to preempt “this is autotuning” critiques)

TITAN’s search is bounded and cached. We report:

- compile time distribution (median / p95) across benchmark suite,
- number of candidate schedules explored per kernel,
- cache hit rates by (tile shape, dtype, arch, fusion pattern),
- fallback frequency.

This makes the “small discrete search” claim concrete and defensible.

---

If you paste the actual **PL_proposal.tex** sections (even roughly), I can do a second pass that (1) preserves your original notation and narrative flow, (2) drops these rewritten blocks into the exact section structure you already have, and (3) aligns terminology so the IR op names, proofs, and evaluation checklist remain consistent across the full paper.
