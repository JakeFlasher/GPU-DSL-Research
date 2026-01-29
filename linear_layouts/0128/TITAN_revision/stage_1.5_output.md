## Proposal Patch Kit

| Vulnerability | Proposed_Fix (Theory/Algo) | Implementation_Cost | Citation_Support (Similar approaches) |
|---|---|---:|---|
| **1) S1: Ring-buffer backpressure underspecified ⇒ overwrite/deadlock/ABA** | **Make backpressure explicit**: a **two-semaphore (Ready/Free) handshake per stage** + **time-indexed (epoch) protocol** so producer cannot lap consumer. Define `stage_acquire` as *blocking* on `Free(s,epoch)`; define `Ready(s,epoch)` as copy-completion; prove invariant `produced - consumed ≤ D`. | Medium | Credit-based flow control; bounded ring-buffer protocols with sequence/epoch; GPU pipelining patterns (producer/consumer “pipeline” objects). |
| **2) S2: Typestate too weak for multi-warp readers, async tensor-core consumption, predication** | **Strengthen typestate with affine/time-indexed logic**: stage permissions become `Stage(s,epoch)`; add **reader multiplicity** (fractional/affine split) and a **compute-consumption token** that must be waited before release. Add predication-safe “arrive with 0 bytes / poison” semantics. | High | Affine/linear types (Rust-like); fractional permissions (separation logic); typestate; time/epoch-indexed protocols. |
| **3) Prologue/epilogue/partial-tile behavior not modeled ⇒ “wait never arrives” hangs** | **Define a 3-phase transformation** (prologue fill / steady / epilogue drain) using a single unified loop with guarded issue/wait. Ensure: for every consumer `wait_ready`, there exists a dominating producer “arrive/expect” even if `bytes=0`. Or **explicitly split tail tiles into a non-pipelined fallback** (safe baseline). | Medium | Classic software pipelining prologue/epilogue; modulo scheduling; CUTLASS-style pipeline fill/drain. |
| **4) Reg pressure hand-wavy ⇒ occupancy collapse/spills** | **Reg-aware software pipelining (non-negotiable F3)**: schedule with a reg-budget objective, use **iterative deepening on D/unroll**, add **rematerialization hints** + **live-range splitting** rules, and optionally **regalloc feedback** loop (cheap “retry with smaller D”). | High | Register-pressure-aware scheduling; integrated scheduling+RA literature; modulo scheduling with live-set cost. |
| **5) Warp specialization cliffs; no disable heuristic (S3)** | Add a **disable/degenerate heuristic** + cost model: only consider warp specialization if predicted overlap benefit exceeds lost compute throughput + sync overhead + occupancy drop. Enforce minimum `K_tiles`, minimum arithmetic intensity, and occupancy floor. | Medium | Roofline/occupancy models; GPU “split warp roles” heuristics (producer warp count small); CUTLASS multi-stage GEMM heuristics. |
| **6) IR token/barrier shape not isomorphic to NVIDIA (phase tokens, proxy domains, async-group completion)** | **Revise IR contract to be hardware-faithful**: represent **mbarrier arrival state token**, model **3 proxy domains** (generic/async/tensormap) in effects, and add **bulk async-group commit/wait** ops for store direction. Also add **TMA legality verifier** (alignment/stride/swizzle). | High | MLIR/LLVM target intrinsics; effect systems; explicit barrier-phase tokens; “commit/wait group” abstractions. |
| **7) AMD: `s_waitcnt` counter-based ⇒ per-transfer tokens not preservable** | **Virtual Token shim (non-negotiable F1)**: tokens become **prefix/stream ordinals** with **monotone waits** + **restricted scheduling window**. Lower to `s_waitcnt` thresholds per counter-class; forbid out-of-order token waits and “selective wait for op i” patterns. Group transfers per stage. | Medium | Counter-based completion (waitcnt) insertion strategies; “in-order retirement queue” / stream tokens; restricted-window async models. |
| **8) Compile-time: “small search” hides hard scheduling problem** | **Warp-Role Swing Modulo Scheduling (WR‑SMS)** + pruning: schedule steady-state once per (D, partition) with bounded II; early reject on legality/reg budget; cache schedules keyed by tile shape/arch. Always provide a baseline fallback (D=2, no specialization). | Medium | Swing Modulo Scheduling; iterative compilation/auto-tuning with caching; template-guided scheduling. |

---

# Patch 1 — Make ring-buffer backpressure and wrap-around *rigorous* (S1)

### What changes (semantic contract)
We replace the “compile-time partial order” story with an explicit **runtime protocol** that is *provably safe under imbalance* (consumer slower than producer) and that eliminates the parity/ABA ambiguity by construction.

### Core idea: **two-semaphore handshake per stage**
For each ring slot (stage) `s ∈ [0..D-1]`, maintain two logical semaphores:

- `Free(s, e)` — “slot `s` is free to write for epoch `e`”
- `Ready(s, e)` — “slot `s` contains valid produced data for epoch `e`”

Where **epoch** `e = ⌊t / D⌋` distinguishes wrap-around uses.

**Required invariant (the missing piece in Stage 1):**
\[
\forall t:\;\; \textsf{produced}(t) - \textsf{consumed}(t) \le D
\]
Equivalently, producer can never be more than `D` stages ahead. This is *exactly* the condition that prevents phase aliasing / ABA when using parity-based mechanisms.

### Concrete blocking rule (backpressure)
Define `stage_acquire(s,e)` as:

- **Blocking until** `Free(s,e)` is observed (not speculative),
- Then it grants exclusive write capability to stage buffer `Buf[s]` for epoch `e`.

When consumer is slower, producer simply blocks on `Free`, which is *the* explicit runtime backpressure mechanism the paper lacked.

### IR patch (minimal but explicit)
Replace the current “linear ownership fiction” (`ttg.async.stage_acquire/release` as compile-time-only) with time-indexed operations:

```mlir
// Acquire exclusive write capability to slot s for epoch e.
%slot = ttg.stage.acquire %pipe, %s, %e
  : (!ttg.pipe, index, index) -> !ttg.slot<s, e>

// Launch an async produce into the slot and obtain a ready token tied to (s,e).
%rdy = ttg.stage.async_produce %slot, %src_desc, %bytes, %proxy_info
  : (!ttg.slot<s,e>, !ttg.src, i32, !ttg.proxy) -> !ttg.ready<s,e>

// Consumer waits for ready(s,e), then computes.
ttg.stage.wait_ready %rdy : !ttg.ready<s,e> -> ()

// Release: signal Free(s, e+1) (or equivalently Free(s,e) for next reuse).
ttg.stage.release %slot, %consumed
  : (!ttg.slot<s,e>, !ttg.consumed<s,e>) -> ()
```

Key: `!ttg.slot<s,e>` is an **affine capability**: cannot be duplicated, must be released exactly once, and carries epoch.

### Runtime implementation options
- **NVIDIA (SM90/H100)**:  
  - `Ready(s,e)` is naturally implemented by **mbarrier tx-count completion** for that stage/epoch.
  - `Free(s,e)` can be implemented by either:
    1) a second `mbarrier` used as a phase semaphore (consumer arrives when done), or  
    2) a shared-memory `seq[s]` epoch counter (classic bounded ring) plus a *subset barrier* for consumer warps.
- **AMD (MI300/CDNA3)**:  
  - `Ready(s,e)` cannot be handle-based; it becomes **group/prefix completion** (see Patch 7).
  - `Free(s,e)` is a shared `seq[s]` + workgroup barrier.

### Why this eliminates ABA
ABA happens only if the producer can advance a stage’s “ready/free phase” by ≥2 without the consumer observing the intermediate phase. Under the invariant `produced-consumed ≤ D`, each slot’s epoch progresses by at most 1 between producer acquisitions. Therefore parity-based waiting is safe, or sequence-number waiting is exact.

---

# Patch 2 — Strengthen typestate with **Affine + Time-indexed** logic (S2) (non-negotiable F2)

Stage 1 is right: the current typestate assumes “one consumer, one read, synchronous consumption.” That’s not true for warp groups and pipelined tensor-core consumption.

We fix this by making the typestate **(a) time-indexed** and **(b) multiplicity-aware**, and by introducing a **compute-consumption token**.

---

## 2.1 Time-indexing: make wrap-around explicit
Define the stage resource type as:

- `StageBuf(s, e)` — “buffer slot `s` at epoch `e`”
- `ReadyTok(s, e)` — readiness for that exact epoch
- `ConsumedTok(s, e)` — proof the buffer is no longer read by compute

Now the state machine becomes (for each `(s,e)`):

\[
\textsf{Free}(s,e) 
\rightarrow \textsf{InFlight}(s,e)
\rightarrow \textsf{Ready}(s,e)
\rightarrow \textsf{Reading}(s,e)
\rightarrow \textsf{Consumed}(s,e)
\rightarrow \textsf{Free}(s,e+1)
\]

The verifier rejects any attempt to “use Ready(s,e)” with “StageBuf(s,e+1)” or similar wrap confusion.

---

## 2.2 Multi-warp readers: affine split/join
We need to express: “slot is free only after **all** consumer warps are done reading.”

### Option A (cleanest for a paper): *fractional permissions*
- After `wait_ready`, the compiler creates a read-permission that can be split into `k = |W_C|` pieces:
  - `R(StageBuf(s,e)) → R1 ⊗ R2 ⊗ ... ⊗ Rk`
- Each consumer warp consumes exactly one piece.
- Only after re-joining all pieces can we create `ConsumedTok(s,e)` and release the slot.

### IR sketch
```mlir
%rperm = ttg.stage.read_perm %slot : !ttg.slot<s,e> -> !ttg.rperm<s,e>

// Split into per-warp affine fragments (conceptually; can lower to structured control)
%p0, %p1, ... = ttg.rperm.split %rperm {parts = W_C} : ...

// Each consumer warp must return its piece:
%rperm2 = ttg.rperm.join %p0, %p1, ... : -> !ttg.rperm<s,e>

// Only after join + compute-wait:
%consumed = ttg.stage.consumed %rperm2, %ctok : -> !ttg.consumed<s,e>
```

### Option B (more implementable): **collective release**
Require `stage_release` to be executed by all consumer warps in lockstep (or after a consumer-only barrier). The verifier enforces “no divergence” in the specialized region (uniform control flow).

This is essentially the operationalization of fractional permissions.

---

## 2.3 Async compute consumption token
We introduce a **compute token** that models the fact that tensor-core pipelines can keep reading shared after issue.

### New IR concept
Any “async-ish” compute op that reads stage memory returns a token:

```mlir
%ctok = ttg.compute.issue %inputs_from_stage : -> !ttg.compute_token<s,e>
ttg.compute.wait %ctok : !ttg.compute_token<s,e> -> ()
```

The rule becomes:

> `stage_release(s,e)` requires `compute.wait(ctok)` (or proof compute is synchronous).

This plugs the exact WAR/UAF gap called out in Stage 1.

---

## 2.4 Predication / skipped copies: poison or 0-byte arrive
You need semantics that guarantee:

- consumers never wait forever, and
- consumers never treat “arrived but no data” as valid unless specified.

We add one of these **explicit** semantics (pick one and state it in the paper):

### Choice 1: **0-byte arrive is valid and produces “zero-filled” stage**
- Producer computes `bytes = pred ? full_bytes : 0`.
- `async_produce(slot, bytes)` always returns a `ReadyTok(s,e)`.
- If `bytes=0`, backend must ensure the stage memory is logically zero (either by prior initialization, or by a separate fast memset, or by predicated compute treating missing bytes as zero).

### Choice 2: **Poison state**
- If copy skipped, producer still signals readiness but sets state `ReadyPoison(s,e)`.
- Consumer must branch: either skip compute for that stage or use safe fallback.
- Verifier ensures poison is handled.

For matmul tails, **the simplest robust approach** is often: *split tail tiles into a fallback path and do not pipeline them*. But if TITAN wants to pipeline tails, it must pick Choice 1 or 2 explicitly.

---

# Patch 3 — Prologue/Epilogue/Short-loop correctness (S3-ish hang)

We add an explicit transformation that is correct for:

- `K_tiles < D`,
- predicated last tiles,
- producer/consumer warp specialization with different roles.

## 3.1 Single unified loop schema (recommended)
Let `N = num_k_tiles`.

Let `prefetch = D-1`.

We generate a single “time” loop `t = 0 .. N + prefetch - 1`:

- **Producer issues** for logical tile `p = t` (or `p = t` with offset depending on formulation) guarded by `p < N`.
- **Consumer consumes** logical tile `c = t - prefetch` guarded by `0 ≤ c < N`.

Concretely:

```c
for (t = 0; t < N + (D-1); ++t) {
  // Producer: produce tile p=t if in range
  if (t < N) {
    s = t % D; e = t / D;
    acquire_free(s,e);
    rdy[s,e] = async_produce(s,e, bytes(t));  // bytes can be 0 for predicated
  }

  // Consumer: consume tile c=t-(D-1) if in range
  c = t - (D-1);
  if (0 <= c < N) {
    s = c % D; e = c / D;
    wait_ready(rdy[s,e]);
    ctok = compute_issue_using_stage(s,e);
    compute_wait(ctok);
    release_free(s,e);
  }
}
finalize_barrier();
```

### Why this fixes the “wait never comes”
Because the consumer-side wait for `(s,e)` is guarded by `c < N`, and the producer-side `async_produce` for the same `(s,e)` is guarded by `t < N` where `t = c + (D-1)`. Those guards match by construction.

## 3.2 Warp specialization: keep loop bounds identical
If producer and consumer are in different warps, you still generate **the same `t` loop bounds** for both roles, but each role executes only its guarded side.

Critical rule to state and enforce:

> Producer warps must not “exit early” while consumer warps still need them for synchronization objects (e.g., shared barrier init/finalization).

Mechanism:
- Either place producer/consumer in the same `scf.for` with role predicates,
- Or emit an explicit `ttg.pipeline.finalize` that both roles must reach.

---

# Patch 4 — Register-aware software pipelining (S4) (non-negotiable F3)

The paper currently says “estimate and reject.” That won’t convince anyone who has watched a single extra live range halve occupancy.

We replace that with a concrete, implementable heuristic that interacts with the scheduler.

## 4.1 Add a first-class **RegBudget** and a live-set cost
### New analysis products
- `RegBase`: baseline registers for the kernel without pipelining (measured or estimated)
- `RegOpWeight(v)`: per-SSA value weight (scalars ~1, vector fragments >1, MMA fragments large)
- `MaxLiveWeight(τ)`: maximum weighted live set under schedule `τ`

### New acceptance criterion
A candidate schedule is valid only if:

1. `MaxLiveWeight(τ) ≤ RegBudget`, and
2. predicted occupancy (from regs + shared) stays above an explicit floor (e.g., ≥1 CTA/SM, or ≥2 depending on target).

## 4.2 Reg-aware WR‑SMS (ties into Patch 8)
During scheduling, use `MaxLiveWeight` as:
- a tie-breaker (prefer earlier kill / later def patterns),
- a pruning rule (if partial schedule already exceeds budget, abandon).

## 4.3 Concrete mitigation transformations (not just “reject”)
When reg pressure is too high, apply these *before* giving up:

1. **Reduce unroll first**, then reduce `D`.  
   Unroll tends to replicate address/predicate temporaries; it’s often the real culprit.
2. **Rematerialize pointers/indices**:  
   Add an IR hint on cheap affine expressions so RA can recompute rather than keep live.
   - `ttg.hint.rematerialize %idx_expr`
3. **Split live ranges around waits**:  
   If a value is only needed after `wait_ready`, ensure it is recomputed after the wait or stored to shared.
4. **Stage-localize temporaries**:  
   For producer, keep only stage pointer for “current s” live; compute `base + s*stride` on demand.

## 4.4 Optional but compelling: regalloc feedback loop
If you can afford it, do:
- compile candidate (fast path),
- read back actual `regs/thread`,
- if above threshold, re-run with smaller `D` or lower unroll.

This turns “hand-wavy estimate” into an engineering reality check.

---

# Patch 5 — Warp specialization: add a disable heuristic + cost model (S5 / S3)

We add a simple model that makes specialization conditional, plus a safe fallback.

## 5.1 Decision rule (practical, reviewer-friendly)
Enable warp specialization only if all hold:

1. `N = K_tiles ≥ N_min` (e.g., ≥ 4 or ≥ 2D)  
   → enough iterations to amortize role split and sync.
2. `ComputeCycles_per_tile` is not tiny (tile not too small).  
3. Predicted overlap gain exceeds predicted throughput loss:
   \[
   \Delta T \approx \max(0, L_\text{copy} - L_\text{compute}) 
   \;\;>\;\;
   \lambda \cdot \textsf{LostCompute}(W_P)
   \]
4. Occupancy after (regs, shared, barriers) ≥ floor.

## 5.2 Search space restriction
If the heuristic says “maybe,” search only a tiny neighborhood:
- `W_P ∈ {1,2}` (keep producer warps minimal)
- `D ∈ {2,3,4}` (bounded)
- unroll factors small

Otherwise:
- **disable specialization** and use “all-warps do both” pipeline or even a non-pipelined kernel.

This directly addresses the “sharp cliffs” concern: we explicitly define when we turn it off.

---

# Patch 6 — Make IR isomorphic to NVIDIA primitives + proxy domains + async-groups (S6)

Stage 0 already shows the mismatch: `mbarrier_wait(bar)` without phase state is not the real thing, proxy domains are incomplete, and stores need async-groups.

We patch the IR op set so the lowering is not a handwave.

## 6.1 NVIDIA: represent **arrival state token**
Add an explicit type `!nv.mbarrier.state` (or generic `!ttg.barrier_state`) returned by arrive:

```mlir
%st = ttg.nv.mbarrier.arrive_expect_tx %bar, %tx
  : (!ttg.nv.mbarrier, i32) -> !ttg.nv.mbarrier.state

// cp.async.bulk.tensor ... associated with %bar
ttg.nv.cp_async_bulk_tensor %tmap, %coords, %dst, %bar : ...

ttg.nv.mbarrier.wait %bar, %st : -> ()
```

This makes `ReadyTok` concrete on NVIDIA.

## 6.2 Add **bulk async-group** completion ops (for stores)
Introduce:

```mlir
ttg.nv.bulk_group.commit : -> ()
ttg.nv.bulk_group.wait %n : (i32) -> ()
```

And require that any shared→global bulk tensor copy uses this mechanism, not mbarrier.

## 6.3 Proxy domains as effects (generic/async/tensormap)
Make proxy visibility part of the IR semantics:

- Each memory operation is annotated with its proxy domain.
- Fence ops are typed:

```mlir
ttg.fence.proxy {from = "generic", to = "async", scope="cta", space="shared"} : -> ()
ttg.fence.proxy {from = "generic", to = "tensormap", scope="cta"} : -> ()
```

And importantly: model the **implicit proxy fence on completion** so we don’t over-fence.

## 6.4 Add a TMA legality verifier (alignment/stride/swizzle)
At pipeline construction time, assert:

- stage base alignment,
- stage stride multiples,
- swizzle-mode constraints,
- padding per stage to preserve alignment across ring slots.

In MLIR terms: attach layout attributes to the ring buffer allocation and check them in a verifier pass. If illegal, either:
- pad the stage stride, or
- fall back to non-TMA path.

This converts “false-by-omission” into an explicit legality module.

---

# Patch 7 — AMD mapping: **Virtual Token shim + restricted window** (S7) (non-negotiable F1)

We stop pretending AMD can do per-transfer handle waits. Instead, we formalize a compatible token model.

## 7.1 Redefine AMD token semantics: **prefix tokens**
On AMD, a token is not “this particular copy.” It is:

> **a position in a per-counter-class issue stream** (an ordinal), with **monotone consumption**.

Call it `!ttg.vtok<class, n>`.

### Rules the verifier enforces (this is the “restricted scheduling window”)
1. Tokens of the same class must be **waited in non-decreasing order**.
2. You may not “wait(token_i)” while allowing older tokens to remain un-waited (no out-of-order).
3. All async transfers for one pipeline stage must be issued as a **contiguous group**, and the consumer waits the group token.

This makes lowering to `s_waitcnt` well-defined.

## 7.2 Lowering strategy (compiler-side “retirement queue”)
The AMD backend maintains, per waitcnt-class `C`:

- `issued_C`: how many ops of class C have been issued since last full wait
- `frontier_C`: the last waited ordinal

When emitting a `vtok`:
- assign `n = issued_C` (or global ordinal),
- increment `issued_C`.

When waiting `vtok(n)`:
- emit `s_waitcnt C(issued_C - (n+1))` **or** conservatively `s_waitcnt C(0)` depending on instruction mix,
- update `frontier_C = n`.

(Exact immediate math is backend-dependent; the key is: the backend has a single monotone frontier and never tries to wait a “middle” op.)

## 7.3 What we say in the paper (crucial)
We explicitly state:

- **NVIDIA** backend supports *handle/phase-precise* readiness.
- **AMD** backend supports only *prefix/group* readiness; TITAN’s token semantics are parameterized by backend capability.

Then we adjust the claimed portability guarantee accordingly:
- correctness is preserved, but granularity differs.

This preempts the reviewer’s “portability overstated” critique.

---

# Patch 8 — Scheduler: Warp-Role Swing Modulo Scheduling (WR‑SMS) + pruning (S8)

We need to replace “small discrete search” with a concrete algorithm and a compile-time budget story.

## 8.1 Model: multi-role, resource-constrained modulo scheduling
Annotate each op with:
- `role ∈ {P (producer), C (consumer), Both}`
- resource usage (copy pipe, tensor pipe, barrier unit, etc.)
- latency

Create a dependence graph that includes:
- data deps,
- `Ready/Free` protocol deps,
- loop-carried deps with distance `D`.

## 8.2 Algorithm (WR‑SMS)
1. **Choose initiation interval (II)**:
   \[
   II = \max(II_\text{resource}, II_\text{recurrence})
   \]
2. Run a Swing Modulo Scheduling variant that:
   - schedules ops onto per-role timelines modulo `II`,
   - respects resource capacities per role,
   - uses priority = (criticality, low mobility, reg-pressure impact).
3. If fail:
   - increase `II` or reduce `D`, retry.
4. Validate:
   - typestate/effects verification,
   - reg budget (Patch 4),
   - hardware legality constraints (Patch 6).

## 8.3 Pruning + caching (compile-time story)
- Early prune any `(D, W_P, unroll)` that fails:
  - alignment legality,
  - shared memory budget,
  - reg budget estimate.
- Cache best-known configs keyed by:
  - arch, tile shape, dtype, epilogue fusion pattern.

Always keep a deterministic fallback:
- `D=2`, no warp specialization, conservative fences.

This addresses “either autotuning or handwave” by making it *bounded, deterministic, and cached*.

---

# How these patches jointly close the Stage‑1 “reject” set

- **S1 deadlock/overwrite/ABA** is closed by: explicit blocking `acquire` + invariant `≤D` + epoch typing.
- **S2 WAR/UAF + multi-warp readers** is closed by: compute tokens + multiplicity-aware release + uniform-control verification.
- **Tail hangs** are closed by: unified loop guards + “0-byte arrive or poison” semantics + finalize barrier.
- **Reg pressure** is handled by: reg-budgeted scheduling + mitigation + optional regalloc feedback.
- **Warp specialization cliffs** are handled by: enable/disable heuristic + constrained search.
- **NVIDIA mismatch** is closed by: mbarrier arrival token + proxy domain effects + bulk-group ops + TMA legality verifier.
- **AMD mismatch** is addressed by: Virtual Tokens + restricted window + explicitly weakened granularity claim.
- **Compile-time** is grounded by: WR‑SMS + pruning + caching + fallback.

If you want, I can also draft (1) a paper-ready “Formal Protocol” section (definitions + invariants + safety theorem statements) and (2) a minimal revised TITAN IR op table (generic + NV + AMD traits) so the text reads as *“this is the contract; this is the lowering; this is what is proven.”*
