1) **Executive Summary (1 paragraph)**  
**Verdict:** The *correctness problem is real and spec-backed* (**Problem reality: Green**)—Tile IR explicitly makes memory effects of token-ordered ops unordered unless connected by tokens, forbids relying on program dependencies for ordering, restricts `weak` for concurrency/communication, and declares data races UB (including hazards *within a single tile-block thread*). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) **Novelty:** The proposed artifacts are *plausibly publishable but at risk of being perceived as “engineering recomposition” of known ideas* (**Novelty: Yellow**): constraint→DAG synthesis/minimization echoes classic partial-order reduction/transitive reduction and SSA-style memory-dependence representations; however, packaging this specifically for Tile IR’s “program deps don’t order memory” cliff, with *explainable counterexamples* and a *checkable overlap plan* could be a meaningful contribution if the certificate/minimality goals are made crisp. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) **Feasibility:** Technically feasible in MLIR (**Feasibility: Yellow**) for reachability/token-graph construction, but *sound constraint generation* depends on alias/overlap reasoning for tiles/views and on carefully scoped claims about cross-thread communication and `weak`/scoped orderings; without those guardrails, you risk either unsoundness or “always serialize” conservatism. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) **Evaluation strength:** Directionally strong (**Evaluation: Yellow**) because you propose metrics beyond speed (graph complexity, compile overhead, time-to-debug), but you need more concrete motifs, baselines, and an experimentally credible debugging study design to convince skeptical PL/CGO reviewers.

---

2) **Claim-by-Claim Verification Table**

| Claim (verbatim or tightly paraphrased) | Where in the proposal (section) | Status | Evidence (citation + short excerpt/interpretation, with version/date context) | Notes |
|---|---|---|---|---|
| Tile IR memory operations are token-ordered; ordering is undefined unless established by token dependencies. | Abstract; Introduction; Background §“Token-ordered memory operations” | **Verified** | Tile IR Ops (v13.1 docs, accessed 2026-02-03): “Currently all memory operations are token-ordered; … undefined unless connected by tokens.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) | This is the core premise: legality is graph-based, not syntax-based. |
| Program dependencies (control/data/address) do **not** provide ordering for token-ordered memory operations; tokens must be used even if redundant. | Abstract; Introduction; Background §“Program dependencies do not order memory”; Validator section | **Verified** | Tile IR Memory Model (latest docs, accessed 2026-02-03): “Program dependencies … do not provide ordering … Tokens must be used … Program dependencies may be optimized away.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Strongly supports “non-local legality” and “rare nondeterminism” risk if edges are missing. |
| Tokens are abstract compile-time values: no runtime representation; cannot be compared, computed upon, or stored/loaded. | Background §“Token-ordered memory operations” | **Verified** | Tile IR Memory Model: tokens “have no concrete representation at runtime, cannot be compared … or stored/loaded.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Enables proof-carrying *static* certificates; also means debugging requires tool support (no runtime inspection). |
| Tokens build dependencies **within the same tile-block thread** (unit of token order). | Background; TokenGraph synthesis inputs | **Verified** | Tile IR Memory Model: tokens are for dependencies “within the same tile block thread.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Critical scoping point: TokenGraph is per tile-block thread; cross-thread correctness needs scopes/orderings. |
| Weak operations cannot be used for inter-thread communication; compiler may assume weakly accessed tiles are not concurrently accessed. | Background §“Scopes, ordering, UB”; Threat Model | **Verified** | Tile IR Memory Model (Scopes): “Weak operations cannot be used to communicate … The compiler may assume … `weak` are not concurrently accessed.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | This is a correctness cliff: using `weak` for sharing is UB/invalid assumptions. |
| Data races are defined and programs with data races have undefined behavior. | Background; Threat Model | **Verified** | Tile IR Memory Model (Data Races): conflicting accesses not HB/morally strong ⇒ “Programs with data races have undefined behaviour.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Validates your “missing edge → UB” framing. |
| Tile IR permits race hazards **within a single tile-block thread** (unlike typical “single-thread HB always holds”). | Introduction; Validator/counterexamples motivation | **Verified** | Tile IR Memory Model (Hazards 7.12.2): “not the case in Tile IR… data race … within a tile block thread … not ordered by token order” (incl. internal overlap). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | This is a powerful, reviewer-compelling motivation for explicit token-graph tooling. |
| `cuda_tile.join_tokens` joins tokens; consuming ops are ordered w.r.t. all joined tokens. | TokenGraph synthesis sketch; Background | **Verified** | Tile IR Ops (v13.1): join_tokens “depends on all input tokens… consume new token… ordered with respect to all joined tokens.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) | Confirms your fork/join DAG encoding model. |
| `cuda_tile.make_token` creates a fresh token with no prior deps (pure). | TokenGraph synthesis sketch | **Verified** | Tile IR Ops (v13.1): make_token “fresh token with no prior dependencies.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) | Supports “t0 = make_token” in pseudocode. |
| Token-ordered ops are not constrained by program order; compiler may reorder unless constrained by tokens. | Background; Validator explanation | **Verified** | Tile IR Ops (v13.1): “Token-ordered operations are not constrained by program order… compiler may reorder… unless… constrained by tokens.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) | Reinforces why “it looks ordered in SSA” is not a correctness argument for memory effects. |
| cuTile `allow_tma` and `latency` are optional performance hints; `allow_tma=True` means may lower to TMA; kernels run without specifying hints. | Background §“Performance hints”; AsyncPlanIR lowering | **Verified** | cuTile Performance Tuning (accessed 2026-02-03): `latency`/`allow_tma` described as “hints… optional… compile and run without specifying… may be lowered to use TMA.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html)) | Your “hints not semantics” framing is consistent with docs. |
| cuda-tile toolchain can translate MLIR→Tile IR bytecode (`cuda-tile-translate`) and compile AoT with `tileiras` (or JIT load). | Implementation plan; harness | **Verified** | NVIDIA/cuda-tile README (aligned w/ CUDA Toolkit 13.1): describes producing bytecode with `cuda-tile-translate` and compiling with `tileiras` / JIT via driver API. ([github.com](https://github.com/NVIDIA/cuda-tile)) | Supports feasibility of a benchmarking/inspection harness. |
| Triton-to-TileIR reality check: Tile IR backend currently has unordered global memory by default; missing token APIs can cause incorrect results under aliasing or cross-tile-block transactions; “conservative token appending” is a contemplated mitigation. | Integration points; evaluation baselines | **Verified** | triton-lang/Triton-to-tile-IR README (accessed 2026-02-03): “unordered memory model… not ordered by default… memory token semantics… require extending Triton APIs… may produce incorrect results… aliasing… across tile blocks… conservative rules to append memory tokens… performance loss.” ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) | Strong industrial evidence that your proposed artifacts address a live pain point. |
| Triton-to-TileIR is early-stage; requires CUDA 13.1+, Blackwell GPUs; roadmap includes semantic validation & benchmarking. | Introduction; Implementation plan | **Verified** | NVIDIA Technical Blog (Jan 30, 2026): describes prerequisites, early limitations, and road map items incl. “testing and validation” and “performance benchmarking.” ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) | Helps justify baselines and evaluation scope grounded in current ecosystem constraints. |

---

3) **State-of-the-Art Analysis (Competitors / Prior Art)**

### Official specs/docs (Tile IR memory model, ops, cuTile): normative vs implementation-defined
**Normative semantics (strongly supportive of your premise).**  
The Tile IR spec is unusually explicit that *memory effect order* for token-ordered operations is not inherited from program order or data/control/address dependencies; ordering is only defined by token dependencies (“waits-for” via tokens), and the toolchain may optimize away apparent program dependencies. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) This is precisely the kind of semantics where “looks right in SSA” is not a correctness argument—your proposal’s framing is not a rhetorical flourish; it is directly documented.

**Scoped vs weak operations and UB cliffs are first-class in the spec.**  
The memory model’s treatment of `weak` is stronger than “relaxed”: it explicitly prohibits using weak operations for communication between threads (and even between *fragments of the same tile block* not ordered by token order), and grants the compiler a non-concurrency assumption for weakly accessed tiles. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) Data races are defined and declared UB, which makes missing token edges a *hard correctness* issue (not “just” heisenbugs). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) The spec even flags a rare but important hazard: races can arise *within a single tile-block thread* due to internal overlap or missing token ordering—this is a uniquely sharp motivation for graph-based tooling and good diagnostics. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))

**Operations are designed to encode fork/join dependencies.**  
`make_token` and `join_tokens` give you exactly the SSA-level building blocks to encode partial orders (fork by reusing a token; join by `join_tokens`). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) This supports the feasibility of your pass, but also weakens novelty claims if you position “token DAG construction” as a contribution (your proposal correctly avoids that).

**Hints are documented as hints (not semantics).**  
cuTile’s `latency` and `allow_tma` are described as optional inputs that influence scheduling/lowering; `allow_tma=True` permits but does not require TMA usage, and kernels run without specifying hints. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html)) This supports your choice to treat them as performance metadata in AsyncPlanIR rather than proof obligations.

**Implementation-defined surface area remains.**  
The spec says “currently all memory operations are token-ordered.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)) That “currently” matters: if future versions introduce program-ordered families or different ordering modes, your TokenGraph/validator design should be robust to versioned semantics (e.g., feature-gate checks by Tile IR bytecode version).

### Compiler infra (MLIR/LLVM ecosystem): effect systems, async/transform dialect relevance, existing verifier patterns
**LLVM MemorySSA is the closest “shape” prior art for TokenGraph construction.**  
MemorySSA explicitly builds a pruned SSA form over memory, placing `MemoryPhi`s only where needed, and it tracks/updates that structure as IR is transformed. ([llvm.org](https://www.llvm.org/docs/MemorySSA.html)) Conceptually, a Tile IR token is “a memory state name,” and `join_tokens` behaves like a φ/join for concurrent predecessors. The key *delta* for your work is: Tile IR makes the “memory state name” *semantically mandatory* for ordering token-ordered operations (program order is irrelevant), and it couples this with scopes/ordering constraints and UB data-race rules. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) That coupling is not what MemorySSA was designed to validate.

Implication for reviewers: if you describe TokenGraph synthesis as “inventing memory SSA,” you will get dinged. If you describe it as “a MemorySSA-like construction specialized to a token-ordered GPU memory model, producing minimal, explainable token dependencies plus counterexamples,” you have a clearer research target.

**MLIR Async dialect provides a strong analogy for “plan tokens + completion + checkable dependencies.”**  
The Async dialect’s `!async.token` represents completion of an async operation, and `async.await` orders subsequent computation on that completion; the docs explicitly note tokens can express dependencies on side effects. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/AsyncDialect/)) This is highly relevant to AsyncPlanIR: reviewers already accept “token-like values as dependency carriers” in MLIR; the novelty must be in *what* you certify (Tile IR memory legality + overlap intent) and *how* you explain failures.

**Translation validation and counterexample culture exists in compiler tooling.**  
Pnueli et al.’s “translation validation” frames checking each compilation instance rather than proving the compiler correct once-and-for-all. ([cs.nyu.edu](https://cs.nyu.edu/home/people/in_memoriam/pnueli/transval-icalp98.html)) Alive2 operationalizes this idea at LLVM IR scale, providing translation validation wrappers and explicitly supporting counterexamples. ([github.com](https://github.com/AliveToolkit/alive2)) This is a strong precedent for your validator+counterexample workflow, albeit on a different semantic axis (refinement/equivalence vs token-order legality).

### CUDA Tile ecosystem and Triton-to-TileIR: what is explicit today vs implicit; documented hazards/limitations
**Toolchain/harness feasibility is good.**  
The open-source `cuda-tile` repo describes producing Tile IR bytecode via `cuda-tile-translate`, then compiling with `tileiras` or JIT-loading via CUDA driver APIs. ([github.com](https://github.com/NVIDIA/cuda-tile)) That is enough infrastructure to build: (a) a token-graph extractor/visualizer, (b) a validator pass runner, (c) a benchmarking harness with reproducibility hooks.

**Industrial reality check: token/scoping issues are *already* causing functional hazards upstream.**  
The Triton-to-TileIR incubator README states bluntly that global memory accesses are not ordered by default (unordered memory model) and that token semantics exist but need API extensions; it warns about incorrect results with aliasing and cross-tile-block transactions, and it lists “conservative rules to append memory tokens” as a potential mitigation—with an explicit performance downside. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) This is extremely valuable to your proposal: it shows (1) correctness gaps are not hypothetical, (2) naive “just add tokens” fixes exist but can be costly, and (3) there is demand for *principled, minimal* token insertion plus diagnostics.

**NVIDIA’s public Triton-to-TileIR messaging emphasizes validation/benchmarking as core workstreams.**  
The NVIDIA blog describes semantic validation and benchmarking as part of the development roadmap and positions the backend as early-stage with limitations. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)) That supports your evaluation plan’s “more than speedup” stance and suggests your artifacts could slot into a real pipeline.

### Academic prior art: closest conceptual matches (ordering constraint synthesis, validators, proof-carrying scheduling/ordering, explainable diagnostics)
**Ordering-constraint minimization has classic graph-theoretic roots.**  
Your “minimize token edges subject to constraints” maps cleanly onto transitive reduction / redundancy elimination in partial orders: transitive reduction seeks a smallest-edge graph preserving reachability. ([xlinux.nist.gov](https://xlinux.nist.gov/dads/HTML/transitiveReduction.html)) What’s missing in classic formulations is the *compilation artifact* mapping: turning a reduced constraint relation into SSA token threading with good join placement and stable diagnostics under IR rewrites.

**Schedule/plan representations are well-established (Halide is the canonical example).**  
Halide explicitly separates algorithm from schedule, using a schedule representation to choose points in the parallelism/locality trade-off space. ([halide-lang.org](https://halide-lang.org/)) AsyncPlanIR has a similar “plan as first-class artifact” flavor, but your plan targets *token legality + overlap* under a weak ordering model, not just locality/parallelism.

**Triton’s research lineage already frames tile-level IR + tile-level optimizations.**  
The MAPL/PLDI 2019 Triton paper describes a tile-centric language/IR and tile-level optimization passes for GPU code generation. ([research.ibm.com](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations)) Triton’s success means reviewers will ask: “Why not encode your overlap intent in the existing scheduler / pipeline constructs upstream?” Your answer should be: Tile IR’s token/scoping legality is a distinct semantic obligation that must survive lowering and subsequent transforms; it needs a checkable witness and targeted diagnostics at the Tile IR boundary.

**Proof-carrying / certifying compilation is foundational prior art.**  
Necula’s Proof-Carrying Code work is the obvious conceptual anchor: ship code with a machine-checkable proof of safety properties. ([people.eecs.berkeley.edu](https://people.eecs.berkeley.edu/~necula/papers.html)) Your “certificate for ordering/scheduling” can be positioned as a *domain-specific PCC*: not proving full functional correctness, but proving the presence of required token/scoping order constraints (and flagging UB hazards).

**Verified vs validated scheduling is active research.**  
Recent work on “Fully Verified Instruction Scheduling” emphasizes the gap between translation validation and fully mechanized scheduling proofs, explicitly noting the overhead/assurance trade-off. ([2024.splashcon.org](https://2024.splashcon.org/details/splash-2024-oopsla/82/Fully-Verified-Instruction-Scheduling)) This is directly relevant to your feasibility story: you can justify a “validator + certificate” approach as a pragmatic middle ground (stronger than ad hoc, cheaper than full mechanization), while acknowledging that “fully verified scheduling” exists as an upper bound comparison.

---

4) **Novelty Assessment**

**Core novelty claim (interpreted):**  
*Introduce a first-class, explainable correctness+performance artifact layer for Tile IR’s token-ordered memory model—(i) synthesize and minimize token dependency DAGs from explicit ordering intent; (ii) validate legality under Tile IR scope/ordering rules with actionable counterexamples; (iii) represent async overlap as a lowering-friendly plan that carries a checkable witness mapping plan constraints to token enforcement.*

### Prior-art comparison matrix

| Proposal feature | Closest prior art (evidence-backed) | What’s missing there (for your problem) | Delta here (what you must make crisp) |
|---|---|---|---|
| Constraint-driven token dependency construction over memory effects | LLVM **MemorySSA** (pruned SSA, φ placement, updates under IR motion) ([llvm.org](https://www.llvm.org/docs/MemorySSA.html)) | Not designed as a *semantic obligation* where program order is irrelevant for memory; not specialized to Tile IR’s scoped/weak rules and UB race definition. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Show how TokenGraph synthesis uses Tile IR’s normative “program deps don’t order” + per-tile overlap hazards to generate constraints and encode them via `make_token`/`join_tokens`. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| “Minimization” of ordering edges | Transitive reduction / minimal reachability-preserving edge sets ([xlinux.nist.gov](https://xlinux.nist.gov/dads/HTML/transitiveReduction.html)) | Classic transitive reduction ignores join placement cost models, critical path, SSA token plumbing, and debuggability constraints. | Define what “minimal” means (edge-minimal vs join-minimal vs critical-path-minimal) and justify a computable objective + approximation. |
| Validator with counterexamples for legality failures | **Translation validation** philosophy ([cs.nyu.edu](https://cs.nyu.edu/home/people/in_memoriam/pnueli/transval-icalp98.html)) + **Alive2** counterexample-driven IR validation ([github.com](https://github.com/AliveToolkit/alive2)) | Alive2 validates refinement/equivalence of transformations; it’s not a token/scoping legality checker. | Specialize counterexamples to Tile IR: show unordered conflicting events under Tile IR memory model, cite hazard patterns, and propose stable repairs in token terms. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) |
| Plan representation for overlap/pipelining | **Halide schedule** representation (algorithm vs schedule) ([gpbib.cs.ucl.ac.uk](https://gpbib.cs.ucl.ac.uk/gp-html/Ragan-Kelley_2013_PLDI.html)); tile-level compilation pipelines in Triton ([research.ibm.com](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations)); MLIR **async.token/await** idioms ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/AsyncDialect/)) | These do not natively target Tile IR token legality constraints where ordering is purely token-defined and `weak` has strong non-concurrency assumptions. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) | Make AsyncPlanIR precise about resources, stage edges, and what is being certified (token reachability + scope/order invariants), not just “naming edges.” |
| Proof-carrying certificate/witness | **Proof-Carrying Code** ([people.eecs.berkeley.edu](https://people.eecs.berkeley.edu/~necula/papers.html)); verified scheduling work emphasizes proof vs validation tradeoffs ([2024.splashcon.org](https://2024.splashcon.org/details/splash-2024-oopsla/82/Fully-Verified-Instruction-Scheduling)) | PCC typically targets safety/typing invariants and uses heavy proof infrastructure; verified scheduling is expensive. | Provide a *lightweight, checkable* certificate: enough to avoid re-running expensive analyses, while still meaningful and robust under transformations. |

### Novelty risk (what reviewers could say) and how to sharpen
1) **“This is just SSA threading / MemorySSA with different names.”**  
*Risk:* TokenGraph synthesis could look like reinventing pruned SSA for memory. ([llvm.org](https://www.llvm.org/docs/MemorySSA.html))  
*Sharpening:* Emphasize Tile IR’s semantic cliffs: program dependencies are explicitly non-ordering, `weak` forbids communication, and races can arise within one tile-block thread. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) Those are *not* standard assumptions in general-purpose SSA memory forms. Your validator/counterexample UX and scope-aware legality checks are the differentiator.

2) **“Transitive reduction is textbook; minimization is not research.”**  
*Risk:* If “minimization” is only transitive reduction, it’s a known algorithmic step. ([xlinux.nist.gov](https://xlinux.nist.gov/dads/HTML/transitiveReduction.html))  
*Sharpening:* Define a multi-objective notion: edge minimality + join_tokens count + critical path + stability under transforms + diagnostic quality. Show that the interesting part is *token-SSA construction under Tile IR semantics*, not the graph primitive.

3) **“AsyncPlanIR is a schedule annotation; proof-carrying is marketing.”**  
*Risk:* Without a concrete certificate, “proof-carrying” can be dismissed.  
*Sharpening:* Provide a concrete, small certificate format and checking algorithm (see below), and state precisely what it proves (reachability/scopes) and what it does not (alias truth, OOB). Tie it to established PCC framing. ([people.eecs.berkeley.edu](https://people.eecs.berkeley.edu/~necula/papers.html))

---

5) **Feasibility & Risk Register**

### Top 5 technical risks + mitigations

1) **Risk: Sound constraint generation depends on tile/view aliasing and *internal overlap* reasoning.**  
*Why it matters:* Tile IR explicitly warns about races within a tile-block thread due to internal overlap and missing token order. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) If `E_req` misses an overlap-based constraint, your synthesis can “prove” an unsafe program.  
*Mitigations:*  
- Start with **pointer-based ops only** and a conservative may-alias model; gradually add view-based ops with explicit overlap analysis.  
- Add an **“assumption surface”**: if aliasing/overlap is unknown, emit constraints that serialize (safe fallback) and report “precision loss” metrics.  
- Require (or infer) **non-alias annotations** for high-performance kernels; validate those assumptions separately.

2) **Risk: Over-claiming cross-thread correctness when tokens are per tile-block thread.**  
*Why it matters:* Tokens build dependencies “within the same tile block thread.” ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)) Inter-thread communication requires scoped/ordered operations; `weak` explicitly cannot communicate. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
*Mitigations:*  
- Make the validator’s guarantee explicit: “If accepted, all required intra-tile-block-thread order edges are token-enforced; scoped synchronization is used where specified; no `weak` is used for intended communication.”  
- Treat cross-tile-block properties as **out-of-scope unless** you also model release/acquire and scope relationships (and even then, be conservative).

3) **Risk: Control-flow joins and token SSA plumbing can balloon (join placement, path sensitivity).**  
*Why it matters:* Minimal constraint edges do not directly imply minimal `join_tokens` placements; pruned SSA construction is subtle. ([llvm.org](https://www.llvm.org/docs/MemorySSA.html))  
*Mitigations:*  
- Use a MemorySSA-inspired approach: introduce join points only where required by constraints (pruned form). ([llvm.org](https://www.llvm.org/docs/MemorySSA.html))  
- Define a clear invariant: “Every token-ordered op consumes exactly one token; every token has a single defining op (`make_token`, `join_tokens`, or a memory op result).”

4) **Risk: Counterexamples that are technically correct but not actionable or stable under transforms.**  
*Why it matters:* The value proposition hinges on explainability; unstable or noisy reports will be ignored.  
*Mitigations:*  
- Report counterexamples in the language of the spec: unordered conflicting events, lack of token reachability, and forbidden weak/scoped usage. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))  
- Use stable IDs/locations (bytecode debug info if available) and prefer **small cores**: minimal conflicting pair + minimal missing-edge set.

5) **Risk: Certificate preservation across passes (proof-carrying story breaks under lowering/optimization).**  
*Why it matters:* The Triton-to-TileIR README highlights that conservative token insertion may be needed during conversion and can affect performance. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) If later passes rewrite token structure, your certificate must be invalidated or updated.  
*Mitigations:*  
- Adopt translation-validation style contracts: either (a) re-validate after each token-affecting pass, or (b) require passes to update the witness. ([cs.nyu.edu](https://cs.nyu.edu/home/people/in_memoriam/pnueli/transval-icalp98.html))  
- Keep the certificate check **cheap** (reachability + attribute checks) so re-validation is practical.

### What would make the project fail, and how to de-risk early (first prototype milestone)
**Failure modes:** (i) alias/overlap constraints are too conservative → no performance wins; (ii) constraints are unsound → validator misses UB; (iii) counterexamples unusable; (iv) AsyncPlanIR is underspecified and cannot be checked; (v) compile-time overhead is unacceptable.

**De-risk milestone (recommended):**  
Build **Validator v0** for a restricted subset: pointer-based `load_ptr_tko`/`store_ptr_tko`, `make_token`, `join_tokens`, and memory attributes (`weak` vs scoped). Check:  
- token reachability for a user-provided `E_req`,  
- forbidden uses of `weak` for “communicating” edges (as defined by your plan),  
- generate a minimal unordered conflicting pair rt.  
Use `cuda-tile-translate` + `tileiras` harness from `cuda-tile` to run litmus kernels and confirm nondeterministic failures become deterministic diagnostics. ([github.com](https://github.com/NVIDIA/cuda-tile))

---

6) **Recommendations to Strengthen the Proposal**

1) **Define “minimal” precisely and pick one primary objective.**  
Right now “minimal token dependency DAG” could mean: transitive reduction, minimal `join_tokens`, minimal critical-path length, or minimal serialization cost. Citeive reduction as a baseline, then justify your chosen objective and approximation. ([xlinux.nist.gov](https://xlinux.nist.gov/dads/HTML/transitiveReduction.html))

2) **Make the constraint language explicit (what is an ordering spec?).**  
Add a formal-ish schema for `E_req`: e.g., edges labeled with reason `{overlap, communication, plan-stage, conservative-unknown-alias}` and required scope/ordering pairs. Tie these reasons directly to the Tile IR memory model statements (weak restriction, HB, UB races). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))

3) **Specify the validator’s acceptance guarantee as a theorem-like statement.**  
Example: “If validator accepts, then for every required edge `(A,B)` there exists a token ‘waits-for’ path from A to B, and no `weak` op participates in a required communication relation.” Keep it aligned with spec language (“waits-for”, “tile block thread”, HB/data race). ([docs.nvidia.com](https://docs.nvidia.com/cu/tile-ir/latest/sections/memory_model.html))

4) **Concrete AsyncPlanIR certificate format (design suggestion).**  
Add a small, checkable witness structure, e.g.:  
- `ops`: stable IDs for token-ordered ops (use MLIR loc or explicit symbol attrs).  
- `requires`: list of `(src_op_id, dst_op_id, kind)` edges (kind = plan edge / anti-alias / release-before / acquire-after).  
- `witness`: for each required edge, a *token path witness* as a sequence of token values/defining ops (or simply “reachable” plushed topological interval labels).  
**Checking algorithm:** build token-dependency graph from SSA, then for each required edge, verify reachability; verify scope/order attributes for edges marked “communication”; reject with smallest failing edge and show repair (“add join feeding dst token”). This is a lightweight PCC/translation-validation style artifact. ([people.eecs.berkeley.edu](https://people.eecs.berkeley.edu/~necula/papers.html))

5) **Clarify issue vs completion hazards for overlap.**  
TilanIR semantics to what Tile IR tokens actually order (memory model “waits-for” relation) and what they do not (e.g., pure compute). Ground it in the spec’s hazard examples (“reading from the future”, races within a thread). ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html))

6) **Add a “pass contract” table for compositional compilation.**  
For each transformation class (CSE, LICM-like motion, view lowering, async pipelining), state whether it may: redered ops, rewrite tokens, change scopes/orderings; and whether it must re-run validation or update certificates. Use MemorySSA’s “must update when IR changes” as an analogy. ([llvm.org](https://www.llvm.org/docs/MemorySSA.html))

7) **Strengthen baselines: include “conservative token appending” and “always-thread program order.”**  
The Triton-to-TileIR README explicitly contemplates conservative token appending as a mitigation with perf loss—this is a natural baseline and competitor. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR)) Also include a naive baseline: thread one token through all memory ops (full serialization), and measure critical path/parallelism metrics against your minimized TokenGraph.

8) **Make the debugging-time evaluation credible.**  
Specify: participant profile (compiler engineers vs kernel devs), tasks (fix missing edge vs wrong scope vs weak misuse), measured outcomes (time-to-localize, time-to-fix, number of iterations), and controls (with/without counterexample reports). Also record false positives/negatives.

9) **Add concrete kernel motifs & bug classes (revised plan sketch).**  
Pick 3–5 motifs and map them to specific Tile IR hazards:  
- **Double-buffered pipeline** (prefetch/compute/store) → missing join when reusing shared tile buffer;  
- **Reduction + epilogue** → release/acquire ordering + token edges around release;  
- **Softmax-like staged loads** → alias/overlap in views;  
- **Split-K accumulation** → cross-tile-block transacwhy `weak` is invalid for determinism. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))

---

7) **Annotated Bibliography (12–18 items)**  
*(All accessed 2026-02-03 unless otherwise noted.)*

1) **NVIDIA Tile IR — Memory Model (latest docs).** Defines token order, `weak` restrictions, data races as UB, and hazards including races within a single tile-block thread. This is your primary semantic foundation. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_modtml))

2) **NVIDIA Tile IR — Operations (v13.1 docs).** Normatively defines `make_token`, `join_tokens`, and states memory ops are token-ordered and unordered unless connected by tokens; also notes reordering vs program order. Use for precise op-level claims. ([docs.nvidia.com](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html))

3) **cuTile Python — Performance Tuning docs.** Documents `latency` and `allow_tma` as optional hints; `allow_tma=True` “may be lowered to use TMA.”s treating them as non-semantic metadata in AsyncPlanIR. ([docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python/performance.html))

4) **NVIDIA/cuda-tile (open-source repo README).** Practical harness: `cuda-tile-translate` → Tile IR bytecode; `tileiras` AoT or driver JIT. Also notes alignment with CUDA Toolkit 13.1. ([github.com](https://github.com/NVIDIA/cuda-tile))

5) **NVIDIA Technical Blog (Jan 30, 2026): “CUDA Tile IR backend for OpenAI Triton.”** Industrial context: backend integration,quisites, limitations, and emphasis on validation + benchmarking workstreams. Use for “ecosystem readiness” discussion. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/))

6) **triton-lang/Triton-to-tile-IR (incubator repo README).** Direct evidence of unordered memory hazards in practice: warns about incorrect results with aliasing/cross-tile-block transactions; notes missing token API and mentions “conservative tokending” as mitigation. Excellent baseline/competitor source. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR))

7) **MLIR Async Dialect docs.** `!async.token` + `async.await` are a close analogy for dependency tokens and for representing completion/side-effect dependencies in IR—useful when positioning AsyncPlanIR as an MLIR-native concept. ([mlir.llvm.org](https://mlir.llvm.org/docs/Dialects/AsyncDialect/))

8) **LLVM MemorySSA documentation (LLVM 22.0.0git docs).** Shows pruned SSA over my with `MemoryPhi` placement and update/invalidation considerations—closest compiler-infra analog to TokenGraph SSA token threading and join placement. ([llvm.org](https://www.llvm.org/docs/MemorySSA.html))

9) **NIST DADS: “transitive reduction” (+ reference to Aho–Garey–Ullman 1972).** Provides the canonical definition of transitive reduction as edge-minimal reachability preservation; use as baseline for “minimization” discussions. ([xlinux.nist.gov](https://xlinux.nist.gov/dads/HTML/transittml))

10) **Aho, Garey, Ullman — “The Transitive Reduction of a Directed Graph” (SIAM J. Computing, 1972) bibliographic record.** Anchors the classical result behind transitive reduction; cite when formalizing your minimization step. ([bibsonomy.org](https://www.bibsonomy.org/bibtex/2418379203ca4deaa011c53e653fd6195/georg.oettl))

11) **Pnueli, Siegel, Shtrichman — “Translation Validation …” (ICALP 1998 page).** Frames the validator-driven workflow: validate each compilation run post hoc withelevant precedent for your “certificate + validator” story. ([cs.nyu.edu](https://cs.nyu.edu/home/people/in_memoriam/pnueli/transval-icalp98.html))

12) **Alive2 (GitHub repo).** Concrete modern example of translation validation with counterexamples and UB-precise interpretation; useful precedent for “actionable counterexamples” and for discussing false positives/unsupported cases. ([github.com](https://github.com/AliveToolkit/alive2))

13) **Halide project site (PLDI 2013 paper listing).** Canonicale representation decoupled from algorithm; provides a strong conceptual comparator for AsyncPlanIR as a first-class plan artifact. ([halide-lang.org](https://halide-lang.org/))

14) **Ragan-Kelley et al. — Halide PLDI 2013 bibliographic entry (includes PDF link & DOI in record).** Use when you want a stable citation for “schedule representation describes points in a trade-off space.” ([gpbib.cs.ucl.ac.uk](https://gpbib.cs.ucl.ac.uk/gp-html/Ragan-Kelley_2013_PLDI.html))

15) **Tillet, Kung, Cox — TrPL/PLDI 2019) venue page / abstract.** Establishes Triton’s tile-level IR and optimization framing; important when contrasting “upstream scheduling intent” vs “Tile IR token legality obligations.” ([pldi19.sigplan.org](https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations))

16) **Necula — personal publications page (PCC pointers).** High-quality primary pointer to “Proof-Carrying Code (POPL’97)” and the broer certifying compilation line; cite to ground “proof-carrying” terminology. ([people.eecs.berkeley.edu](https://people.eecs.berkeley.edu/~necula/papers.html))

17) **Alastair Reid’s “Related Work” entry for Necula PCC (POPL’97 metadata).** Secondary but convenient structured citation for PCC venue/pages; use if you need quick bibliographic context. ([alastairreid.github.io](https://alastairreid.github.io/RelatedWork/papers/necula%3Apopl%3A1997/))

18) **Yang, Shirako, Sarkar — “Fully Verified Instruction Scheduling” (OOPSLA/SPLASH 2024 page).** Directly relevant comparator for scheduling/ordering proof vs validation; helps position your certificate approach on the assurance/effort spectrum. ([2024.splashcon.org](https://2024.splashcon.org/details/splash-2024-oopsla/82/Fully-Verified-Instruction-Scheduling))

---
Learn more:
1. [https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory\_model.html](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)
2. [https://docs.nvia.com/cuda/tile-ir/latest/13.1/sections/operations.html](https://docs.nvidia.com/cuda/tile-ir/latest/13.1/sections/operations.html)
3. [https://docs.nvidia.com/cuda/cutile-python/performance.html](https://docs.nvidia.com/cuda/cutile-python/performance.html)
4. [https://github.com/NVIDIA/cuda-tile](https://github.com/NVIDIA/cuda-tile)
5. [https://github.com/triton-lang/Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR)
6. [https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
7. [https://www.llvm.org/docs/MemorySSA.html](https://www.llvm.org/docs/MemorySSA.html)
8. [https://mlir.llvm.org/docs/Dialects/AsyncDialect/](https://mlir.llvm.org/docs/Dialects/AsyncDialect/)
9. [https://cs.nyu.edu/home/people/in\_memoriam/pnueli/transval-icalp98.html](https://cs.nyu.edu/home/people/in_memoriam/pnueli/transval-icalp98.html)
10. [https://github.com/AliveToolkit/alive2](https://github.com/AliveToolkit/alive2)
11. [https://xlinux.nist.gov/dads/HTML/transitiveReduction.html](https://xlinux.nist.gov/dads/HTML/transitiveReduction.html)
12. [https://halide-lang.org/](https://halide-lang.org/)
13. [https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations)
14. [https://people.eecs.berkeley.edu/~necula/papers.html](https://people.eecs.berkeley.edu/~necula/papers.html)
15. [https://2024.splashcon.org/details/splash-2024-oopsla/82/Fully-Verified-Instruction-Scheduling](https://2024.splashcon.org/details/splash-2024-oopsla/82/Fully-Verified-Instruction-Scheduling)
16. [https://gpbib.cs.ucl.ac.uk/gp-html/Ragan-Kelley\_2013\_PLDI.html](https://gpbib.cs.ucl.ac.uk/gp-html/Ragan-Kelley_2013_PLDI.html)
17. [https://www.bibsonomy.org/bibtex/2418379203ca4deaa011c53e653fd6195/georg.oettl](https://www.bibsonomy.org/bibtex/2418379203ca4deaa011c53e653fd6195/georg.oettl)
18. [https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations](https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations)
19. [https://alastairreid.github.io/RelatedWork/papers/necula%3Apopl%3A1997/](https://alastairreid.github.io/RelatedWork/papers/necula%3Apopl%3A1997/)
