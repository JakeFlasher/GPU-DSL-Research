Below is a full rewrite that addresses the reviewer’s six requested changes.

## 1. Title

**Closing the Attention Complement: Predicting Next-Generation TMEM and Row-Reduction Pipelines from Layout Analysis**

## 2. Abstract

I narrow the proposal to one subsystem only: the **TMEM-like intermediate memory plus the row-wise complement pipeline** used by FlashAttention-style attention at `M=N=D=128`. CuTe’s `logical_divide` and `complement` already let me factor an attention kernel into an MMA-carried tile and a leftover complement consisting of rowmax, rowsum, exp, rescale, and persistent intermediate state. citeturn21view0turn22view0 Hopper/FlashAttention-3 already exposed the missing hardware basis: H100 offered about **989 TFLOPS** FP16 matmul throughput versus about **3.9 TFLOPS** special-function throughput, and FA-3 improved overlap partly by spending more registers. citeturn5view2turn24view0 Blackwell then added **256 KB of Tensor Memory per SM**, and FlashAttention-4 explicitly uses TMEM to keep `S=QK^T` and `P` tiles live; its published B200 roofline for `M=N=D=128` is **1024 / 1024 / 768 cycles** for forward and **2560 / 1024 / 3328 cycles** for backward, for tensor-core / exp / shared-memory limits respectively. citeturn18view0turn26view0turn25view0turn13view2

The revised question is therefore concrete: **can layout-closure analysis recover TMEM as the right Hopper→Blackwell design move, and can it predict the smallest next-step TMEM/complement configuration that survives an explicit cost model?** I argue yes. The main predicted next feature is **384 KB TMEM per SM plus one row-streaming complement port**. In my first-pass normalized model, that design costs about **+4.8 area units** and **+2.2 power units** per SM, but reduces backward tile time from **3328** to about **2560 cycles**; for a `1 forward : 2 backward` training mix, that is about **20% fewer cycles** overall. If a future GPU also increases tensor-core throughput again, complement throughput should rise from today’s **16 effective exp ops/cycle** to **24–32** as a second, conditional knob. citeturn4view0turn13view2

## 3. Key Insight / Thesis Statement

For this paper, **layout closure means only one thing**: after factoring attention into MMA and non-MMA pieces, the non-MMA complement must execute **without spilling persistent tile state through SMEM** and **without becoming slower than the MMA roof**. CuTe already supplies the right algebra: `logical_divide` isolates the MMA-carried tile, while `complement` exposes the leftover row-wise axes. citeturn21view0turn22view0 I therefore reduce the hardware search to a tiny parameter vector,

\[
\theta = \{C_{\mathrm{TMEM}}, P_{\mathrm{row}}, W_{\mathrm{red}}, W_{\mathrm{exp}}\},
\]

where `C_TMEM` is TMEM capacity, `P_row` is a row-streaming TMEM read path into the complement pipeline, `W_red` is row-reduction width, and `W_exp` is effective exp throughput.

The closure test is:

\[
T_{\mathrm{comp}}(\theta) \le T_{\mathrm{MMA}}, \qquad
T_{\mathrm{move}}(\theta) \le T_{\mathrm{MMA}}, \qquad
\text{LiveState} \le C_{\mathrm{TMEM}}.
\]

If these hold, the subsystem is balanced. If not, the gap says exactly whether the next transistor budget should buy **more TMEM capacity**, **more TMEM-to-complement bandwidth**, or **more complement throughput**.

## 4. Technical Approach

### A. Narrow the model to one workload and one residual

I focus only on **BF16/FP16 training attention with `M=N=D=128`**, because FA-4 publishes detailed cycle roofs there. The software object is not the whole kernel; it is the **complement residual** between the `QK^T` / `PV` MMAs and the row-wise online-softmax recurrence. From CuTe layouts I extract two quantities: the persistent state that must survive between MMAs, and the row-major access order needed by max/sum/exp/rescale. CUTLASS 3.8 also makes this feasible in practice by exposing `tmem` as a first-class locale plus `smem->tmem` and `tmem->rmem` copy atoms. citeturn13view2turn21view0turn22view0turn19view0

### B. Retrospective validation: would this have predicted TMEM on Blackwell?

Using only Hopper-era evidence, the answer should already have been “the missing feature is persistent intermediate storage, not more MMA.” FA-3 reports that H100 has about **989 TFLOPS** FP16 matmul throughput but only about **3.9 TFLOPS** special-function throughput, and then shows progressively tighter GEMM/softmax overlap schemes whose benefit comes with rising register pressure. citeturn5view2turn24view0 In the revised framework, that pattern is exactly a closure-gap signature: the complement is no longer a tiny epilogue; it is a throughput-critical, stateful residual whose live tile does not fit comfortably in registers.

Blackwell later introduced precisely the kind of object this analysis would request: **256 KB TMEM per SM** and **UMMA/tcgen05 accumulation into TMEM**. FlashAttention-4 then uses TMEM to keep `S` and later `P` resident, explicitly notes that staged writes to TMEM relieve register pressure, and says that keeping accumulators in TMEM makes multiple MMAs in flight practical in backward. citeturn18view0turn26view0turn25view0 That gives the paper a clean retrospective claim: **from Hopper bottleneck analysis alone, the framework should have predicted a TMEM-like intermediate store of roughly hundreds of KB per SM with row-wise complement access semantics.** Blackwell’s actual TMEM is therefore not just post hoc inspiration; it is the historical validation target.

### C. Explicit performance and hardware cost model

I anchor the model in FA-4’s published roofline numbers for `M=N=D=128` on B200:

- Forward: `T_MMA = 1024`, `T_exp = 1024`, `T_SMEM = 768`
- Backward: `T_MMA = 2560`, `T_exp = 1024`, `T_SMEM = 3328` citeturn13view2

So the current bottleneck equations are:

\[
T_{\mathrm{fwd}}(\theta)=\max(T_{\mathrm{MMA}}, T_{\mathrm{comp}}, T_{\mathrm{move}})
\]
\[
T_{\mathrm{bwd}}(\theta)=\max(T_{\mathrm{MMA}}, T_{\mathrm{comp}}, T_{\mathrm{move}})
\]

with the published Blackwell baseline implying:

\[
T_{\mathrm{fwd}} = \max(1024,1024,768)=1024,\qquad
T_{\mathrm{bwd}} = \max(2560,1024,3328)=3328.
\]

The key inference is backward. The gap between backward shared-memory time and backward MMA time is

\[
3328 - 2560 = 768 \text{ cycles}.
\]

FA-4 uses **128 B/cycle** as the shared-memory bandwidth roof, so closing that gap requires eliminating about

\[
768 \times 128 = 98{,}304 \text{ B} \approx 96 \text{ KB}
\]

of per-tile shared-memory traffic. FA-4 also explicitly states that the current **256 KB TMEM cannot hold five full accumulators/intermediates at once**, so the kernel must reuse TMEM columns. citeturn13view2turn13view4 My inference is therefore that the smallest sensible next point is **384 KB TMEM/SM**, not 512 KB: `256 KB + 128 KB` clears the ≈96 KB gap and leaves modest bank/alignment headroom.

I use a simple normalized per-SM cost model. One **area unit** is the cost of one **32 KB TMEM bank**; one **power unit** is the dynamic cost of one active 32 KB bank at tile rate. Then

\[
A(\theta)= \frac{\Delta C_{\mathrm{TMEM}}}{32\text{ KB}} + 0.5\,P_{\mathrm{row}} + 0.3\,W_{\mathrm{red128}} + 0.8\,W_{\mathrm{exp16}}
\]

\[
P(\theta)= 0.35\frac{\Delta C_{\mathrm{TMEM}}}{32\text{ KB}} + 0.4\,P_{\mathrm{row}} + 0.4\,W_{\mathrm{red128}} + 0.8\,W_{\mathrm{exp16}}
\]

where `W_red128` counts one extra 128-row reduction engine and `W_exp16` counts one extra 16-op/cycle exp-assist group.

This yields two concrete design points:

1. **Validated next-step design (my main recommendation)**  
   - `C_TMEM = 384 KB/SM`  
   - `P_row = 1` extra row-streaming complement port  
   - `W_red128 = 1` extra max/sum row-reduction engine  
   - `W_exp16 = 0`  
   - **Cost:** `A = 4.8`, `P = 2.2`  
   - **Performance estimate:** `T_bwd ≈ 2560`, `T_fwd = 1024`, so a `1FWD:2BWD` mix goes from `7680` cycles to `6144` cycles, i.e. about **20% fewer cycles**.

2. **Aggressive-TC roadmap design (conditional)**  
   - Same as above, plus `W_exp16 = 1` so effective exp throughput rises from `16` to `32` ops/cycle  
   - **Cost:** `A = 5.6`, `P = 3.0`  
   - **Why:** forward is already exactly balanced at `1024` MMA vs `1024` exp cycles, so any future tensor-core increase without complement scaling will immediately make the kernel exp-bound. A `1.5×` TC roadmap needs about `24` effective exp ops/cycle; a `2×` roadmap needs about `32`. This is an inference from the published FA-4 roofline, not a direct paper claim. citeturn13view2

The non-obvious feature here is not “more SFUs.” It is **a larger TMEM with one row-streaming complement port**, because the published bottleneck is a combined **state-capacity + row-wise reduction** problem, not a generic scalar-throughput problem.

### D. One realistic hardware artifact

To keep feasibility realistic, I propose only **one RTL artifact**: a banked `384 KB` TMEM slice with one row-streaming port feeding a `rowmax/rowsum/exp/rescale` pipeline. The RTL will replay FA-4-like traces and validate the analytic model’s predicted `T_move` and `T_comp`. This is practical because Blackwell TMEM is already surfaced in CUTLASS tooling, and Nsight Compute now exposes Blackwell TMEM traffic in its memory chart for calibration. citeturn19view0turn23view0

## 5. Expected Contributions

- **A sharply narrowed closure criterion** for one real subsystem: attention complement hardware, not whole-GPU ISA/network co-design.
- **A retrospective result**: Hopper/FA-3 bottlenecks would have predicted a TMEM-like intermediate memory before Blackwell existed.
- **An explicit costed prediction** for the next GPU generation: `384 KB TMEM/SM + 1 row-streaming complement port` as the main recommendation, with complement exp width scaled only if tensor-core throughput rises again.
- **A concrete artifact**: one RTL TMEM/complement block, not a speculative full-chip redesign.

This contribution set is grounded because the historical inputs and validation targets are already public: Hopper/FA-3 exposes the bottleneck signature, while Blackwell/FA-4 exposes both TMEM and the per-tile roofline that the revised model uses. citeturn5view2turn24view0turn18view0turn25view0turn13view2

## 6. Evaluation Plan

**Months 1–4: extraction and calibration.** Build a CuTe-based complement extractor that outputs `LiveState`, row-access order, and per-stage bytes moved. Reuse CUTLASS/CuTe `tmem` support instead of inventing a new IR. Calibrate TMEM traffic on Blackwell with Nsight Compute’s TMEM memory chart. citeturn21view0turn22view0turn19view0turn23view0

**Months 5–8: retrospective validation.** Freeze the Blackwell side. Feed only Hopper/FA-3 data into the solver and ask whether it predicts a TMEM-like object, what rough size class it wants, and whether it asks for more MMA, more SMEM, or persistent intermediate storage. Then compare that blind prediction to Blackwell’s documented **256 KB TMEM/SM** and FA-4’s actual TMEM-heavy pipeline. citeturn5view2turn24view0turn18view0turn25view0

**Months 9–12: forward prediction.** Sweep `C_TMEM ∈ {256, 320, 384, 512} KB`, `P_row ∈ {0,1}`, and `W_exp ∈ {16,24,32}` under the normalized area/power model. Report the smallest design that satisfies the closure test, plus ablations against three alternatives: exp-only scaling, SMEM-only scaling, and TMEM-only scaling.

**Months 13–18: one RTL simulation.** Implement the `384 KB` TMEM slice plus row-streaming complement pipe, synthesize/simulate it, and compare simulated cycles against the analytic model. The paper’s success criterion is modest: one subsystem, one realistic trace-driven RTL validation, and one forward recommendation.

**Primary metrics.**  
1. Retrospective correctness: did the solver predict TMEM-like storage from Hopper evidence?  
2. Forward accuracy: does `384 KB` emerge as the smallest point that closes backward on FA-4’s roofline?  
3. Cost/performance: cycles per tile, normalized area, normalized power, and training-mix speedup.

## 7. Target Venue and Why

**MICRO**

The revised paper is now much more architecture-focused than the original version: one subsystem, one historical validation, one explicit cost model, and one RTL block. That is a better fit for MICRO than for a broader cross-layer venue.

## 8. Potential Weaknesses and Mitigations

- **Weakness: still attention-specific.**  
  **Mitigation:** I now claim only `d=128` FlashAttention-style training kernels as the main result. `d=64/256` become sensitivity studies, not headline claims.

- **Weakness: the area/power model is normalized, not final silicon mm²/pJ.**  
  **Mitigation:** that is intentional at proposal stage; the 13–18 month RTL phase calibrates the coefficients.

- **Weakness: hindsight bias in the retrospective claim.**  
  **Mitigation:** run the retrospective as a blind protocol: Hopper/FA-3 inputs first, Blackwell/FA-4 documents only afterward.

- **Weakness: TMEM is vendor-specific.**  
  **Mitigation:** the paper relies only on documented, already exposed interfaces—CUTLASS `tmem` support and profiled TMEM traffic in Nsight—not on undocumented hardware behavior. citeturn19view0turn23view0

- **Weakness: complement-pipe prediction depends on future tensor-core scaling.**  
  **Mitigation:** I make the paper’s main claim the `384 KB + row port` result, and treat `24–32` effective exp ops/cycle as a conditional second-stage recommendation tied to the vendor’s TC roadmap.

If you want, I can also turn this into a **1-page abstract**, a **3–4 page workshop proposal**, or a **patch to `ideas/idea5_hw_sw_codesign.md`**.
