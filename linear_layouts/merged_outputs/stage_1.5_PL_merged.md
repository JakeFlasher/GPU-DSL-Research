## **The "Stage 1.5" Theoretical Arsenal: Layouts × Asynchrony × Dynamism**

The seed systems (Linear Layouts and LEGO) provide a robust **spatial algebra** (bit-linear maps and layout propagation), but they hit performance cliffs when facing the modern "AI factory kernel" reality: dynamic shapes, asynchronous transport (TMA/Hopper), and complex indirection [1, 2].

The proposed solution shifts the performance model from pure layout algebra to a joint optimization of space and time:
\[
\textbf{Performance} \approx f(\text{Layout}) \times f(\text{Transport Schedule}) \times f(\text{Runtime Shape})
\]

---

### **1. Solving Dynamic & Ragged Shapes (Superseding Power-of-Two Restrictions)**
*Bottleneck: Padding explosion, decoding inefficiencies, and MoE underfill caused by the $\mathbb{F}_2$-centric restriction to power-of-two matrices [1, 2].*

#### **Theory A: Parametric Polyhedral & Piecewise-Affine Layouts**
To handle ragged edges and primes without global padding, the model generalizes from bit-matrix algebra to **integer affine maps with guards**:
*   **Polyhedral Semantics:** Iteration domains are modeled as **Presburger sets** (parametric integer sets with linear inequalities). This treats raggedness as control flow in the iteration space rather than noise [1, 2].
*   **Piecewise Definitions:** Layouts are defined as piecewise-affine functions ($x \mapsto Ax + b$) with explicit **guards** (masking regions). This allows slicing and flipping to be represented natively [1].
*   **Integer-Lattice Algebra:** For non-power-of-two strides (e.g., prime dimensions), the abstraction moves from a vector space over $\mathbb{F}_2$ to a **$\mathbb{Z}$-module (integer lattice)**. This allows determining vectorization legality via GCD structure rather than bit-matrix zero columns [2].
*   **Mixed-Radix Indexing:** Indices are represented in a mixed-radix system (bases $b_0, b_1...$) rather than purely binary, utilizing verified strength reduction (via SMT solvers) to handle division/modulo operations efficiently [2].

#### **Theory B: Staged Compilation & Partial Evaluation**
Since LLM shapes (batch, tokens) are runtime values but cluster into predictable buckets:
*   **Partial Evaluation:** Treat shape parameters as values for a kernel *generator*. The compiler produces a minimal portfolio of specialized kernels (e.g., removing masks and constant-folding divisions) [1].
*   **Shape-Specialized Multi-Versioning:** A JIT runtime selects kernels based on a specialization key ($B, H, d, \text{alignment}$) [2]. This mitigates the overhead of dynamic masking by changing the program structure itself rather than just autotuning parameters [1].

---

### **2. Solving Asynchronous Transport (Hopper/Blackwell Pipelines)**
*Bottleneck: Layout engines optimize conversion in space, but hardware like Hopper demands temporal pipelines (TMA, mbarrier, warp specialization) [1, 2].*

#### **Theory A: Task Graph Semantics & Tokenized Dataflow**
Asynchrony is modeled as a **partial order** of events rather than index maps:
*   **Event Structures:** Kernels are represented as DAGs of tasks (Copy $\to$ Barrier $\to$ Compute $\to$ Epilogue) connected by explicit tokens [1].
*   **Token Semantics:** Operations like `tma.copy` yield a **token** (future). Compute units consume these tokens, enabling the compiler to schedule using topological sorts and resource constraints (register pressure) [2].
*   **Warp Specialization:** The IR explicitly partitions the kernel into producer/consumer roles, modeled as communicating processes via "asynchronous references" (aref) [2].

#### **Theory B: Protocol Types & Separation Logic**
To prevent race conditions (use-before-ready) and performance bugs (serialization):
*   **Session Types / Typestate:** Pipeline stages are treated as typestate machines (e.g., `Empty \to ExpectTx \to Arrive \to Wait \to Full`). The compiler statically verifies that the protocol is followed [1, 2].
*   **Separation Logic:** Linear capabilities encode ownership of shared memory buffers (e.g., circular buffers in clusters), ensuring that a buffer is not overwritten before it is consumed [1, 2].

---

### **3. Solving Indirection (Gather/Scatter & MoE Routing)**
*Bottleneck: Pointer chasing breaks layout linearity; coalescing collapses and shuffle rounds explode [1, 2].*

#### **Theory A: Inspector–Executor & Permutation Synthesis**
Instead of treating gathers as static layout conversions, they are treated as **runtime permutation problems**:
*   **Inspector:** A runtime pre-pass analyzes indices to compute a permutation $p$ that packs sparse tokens into dense blocks [1, 2].
*   **Executor:** The kernel runs on dense tiles using standard affine layouts. This converts "random access" problems into streaming, TMA-friendly access problems [2].

#### **Theory B: Equality Saturation over Data-Movement IR**
Because there are many semantically equivalent ways to lower a gather (warp shuffle vs. shared mem vs. global gather):
*   **E-Graphs:** The compiler maintains equivalence classes of data movement plans.
*   **Cost Modeling:** A global search selects the lowering strategy based on register pressure, shuffle round count, and instruction usage, avoiding local heuristic traps [1, 2].

---

### **4. Solving Hierarchical Locality (Clusters & DSMEM)**
*Bottleneck: Seed systems distribute within Thread Blocks (CTAs), but Clusters introduce a new hierarchy tier with Distributed Shared Memory (DSMEM) [1, 2].*

#### **Theory A: Abstract Interpretation on Sharding Lattices**
*   **Placement Lattice:** Define a lattice of memory placements $\{\text{Reg}, \text{Shared::CTA}, \text{Shared::Cluster}, \text{Global}\}$.
*   **Propagation:** Compiler passes propagate these placement attributes to manage ownership and data movement across cluster boundaries (multicast/DSMEM) [1].

#### **Theory B: Wreath Products & Group Actions**
*   **Hierarchical Algebra:** The machine hierarchy is modeled as a **wreath product** of symmetric groups ($G_{\text{cluster}} \wr G_{\text{cta}} \wr G_{\text{warp}}$). This provides a structured algebraic way to solve for layout conversions across the entire hierarchy, rather than using ad-hoc heuristics for each level [1, 2].

---

### **5. Solving Target-Specific Bank Conflicts**
*Bottleneck: Optimal swizzling depends on ISA-specific rules (e.g., NVIDIA vs. AMD Wave64 phases) which vary by generation [1, 2].*

#### **Theory A: Modular Constraint Solving (SMT)**
*   **Parametric Models:** The bank mapping function is treated as an ISA-dependent parameter (e.g., XOR patterns, lane groupings).
*   **SMT Solvers:** The compiler uses SMT (Satisfiability Modulo Theories) to solve for swizzles $T$ such that $bank(addr(T(i)))$ spreads accesses uniformly across banks [1, 2].

#### **Theory B: Combinatorial Design**
*   **Hash Families:** Swizzles are viewed as low-cost hash functions. The compiler selects linear codes or XOR families that minimize collisions for the specific access patterns and bank architecture of the target hardware [1, 2].

---

### **6. Solving Register Pressure (Liveness)**
*Bottleneck: Aggressive buffering and layout conversions can explode register usage, killing occupancy (unique emphasis in [2]).*

*   **Resource-Aware Scheduling:** Modulo scheduling that treats register file capacity as a hard constraint [2].
*   **Linear/Uniqueness Types:** Enforcing "single-use" semantics on tensor fragments at the IR level to structurally limit live ranges and prevent accidental duplication of large fragments [2].

---

### **7. Implementation & Literature Roadmap**

**The Unified Software Pipeline:**
The implementation strategy lands this math into a concrete compiler stack (PyTorch $\to$ Triton/MLIR $\to$ PTX):

1.  **Layout Calculus (Space):** Piecewise-affine maps, $\mathbb{Z}$-lattices, mixed-radix algebras [1, 2].
2.  **Transport Calculus (Time):** Async tokens (`!async.token`), barrier protocols, and pipeline task graphs [1, 2].
3.  **Synthesis Engine:** Equality saturation (E-graphs) to explore lowering costs and SMT solvers to verify banking/tiling correctness [1, 2].

**Key Supporting Literature (2024–2026):**
*   **Cypress (PLDI 2025):** Validates the shift to Task-Based/Event semantics for Hopper TMA [1].
*   **Tawa (2025):** Demonstrates automatic warp specialization via "asynchronous references" [2].
*   **MLIR Transform & DialEgg (2025):** Provides the infrastructure for programmable search strategies and equality saturation within the compiler [1, 2].
*   **Equations in Wreath Products (2025):** Provides the pure-math foundation for solving hierarchical mapping equations [1].
  
### Appendix A: Theoretical Visualization of Stage 1.5 Abstractions

#### Figure 1: The Unified Performance Model (Space × Time × Shape)
*A visual representation of how Stage 1.5 extends the seed Layout Algebra (Space) with Transport (Time) and Dynamic Indirection to define modern kernel performance.*

```text
      [ STATIC COMPILE TIME ]                   [ DYNAMIC RUNTIME ]
      
     (1) SPATIAL LAYOUT        (2) TEMPORAL SCHEDULE      (3) RUNTIME SHAPE
     +-------------------+     +---------------------+    +------------------+
     | Linear Layouts &  |     |   Async Pipeline    |    |  Inspector /     |
     |   LEGO Algebra    |     |    (Token Flow)     |    |   Executor       |
     +-------------------+     +---------------------+    +------------------+
              |                           |                        |
              v                           v                        v
     +-------------------+     +---------------------+    +------------------+
     | Map: x -> Ax + b  |  x  |  DAG: {Copy, Sync,  |  x |  Val: {Batch,    |
     | (Piecewise/Affine)|     |        Compute}     |    |        Experts}  |
     +-------------------+     +---------------------+    +------------------+
              |                           |                        |
              +-------------+-------------+                        |
                            |                                      |
                            v                                      |
              +-----------------------------+                      |
              |   Kernel Synthesis Engine   | <--------------------+
              | (Equality Saturation / SMT) |
              +-----------------------------+
                            |
                            v
              +-----------------------------+
              |  Optimized PTX / Assembly   |
              | (TMA + WGMMA + MBarrier)    |
              +-----------------------------+
```
**Source:** Derived from the synthesis of Linear Layouts [1] and LEGO [2].

---

#### Figure 2: Parametric Polyhedral Layouts vs. Padding
*Contrasting the seed "Power-of-Two" limitation with the Stage 1.5 "Piecewise Affine" solution for ragged tensors (e.g., variable length sequences).*

```text
    (A) Seed Strategy: Global Padding       (B) Stage 1.5: Piecewise/Guarded
        (Linear Layouts / LEGO)               (Presburger Sets / Affine)

    Physical Memory (2^N aligned)           Physical Memory (Packed/Dense)
    +-------+-------+-------+----           +-------+-------+-------+
    | D0    | D1    | PAD   | ...           | D0    | D1    | D2    |
    | (10)  | (10)  | (XX)  |               | (10)  | (10)  | (10)  |
    +-------+-------+-------+----           +-------+-------+-------+
    | D2    | PAD   | PAD   | ...           | Guard:| x < B |       |
    | (10)  | (XX)  | (XX)  |               | Map:  | idx = |       |
    +-------+-------+-------+----           |       | B*h+w |       |
                                            +-------+-------+-------+
    Constraint: Dimensions must be          Constraint: Branching/Masking
    Power-of-Two (2^k). Wasted BW.          handled via Predicates/Guards.
                                            
    Algebra: GF(2) / Bit-Matrix             Algebra: Z-Module / Integer Lattice
    Addr = M * bits(idx)                    Addr = (Base + Stride * i) | Guard
```
**Source:** addressing limitations described in [1] and [2].

---

#### Figure 3: Async Transport & Tokenized Dataflow (Hopper Arch)
*Visualizing "Theory A: Task Graph Semantics." Operations are nodes in a DAG connected by asynchronous tokens (`!async.token`) representing data availability.*

```text
      PRODUCER (TMA Unit)                    CONSUMER (Warp Group)
      -------------------                    ---------------------
               |                                       |
    [ Op: tma.copy.bulk ]                              |
               |                                       |
               +---(produces)---> [ TOKEN ] --+        |
                                      |       |        |
                                  [ MBARRIER ]<--(arrives)
                                  (ExpectTx)  |        |
                                      |       |        |
                                      +-------+----(waits)
                                                       |
                                               [ Op: wgmma.mma_async ]
                                                       |
                                             (Consumes Tensor Core Pipe)
                                                       |
      [ Typestate Check ]                    [ Resource Check ]
      State: Empty -> Full                   Reg Pressure < Limit
```
**Source:** "Task Graph Semantics / Event Structures" [1] and "Resource-Constrained Modulo Scheduling" [2].

---

#### Figure 4: Inspector-Executor Model for Indirection
*Visualizing "Theory A: Indirection." Converting random access (Scatter/Gather) into dense compute via runtime permutation synthesis.*

```text
      LOGICAL VIEW (Sparse/Ragged)       PHYSICAL VIEW (Dense/Streaming)
      
      [ Token A ] (Expert 1)            +---------------------------+
           |                            | INSPECTOR (Runtime Pre-Pass)|
      [ Token B ] (Expert 2) ---------> | 1. Scan Indices             |
           |                            | 2. Compute Permutation (P)  |
      [ Token C ] (Expert 1)            +-------------+-------------+
                                                      |
                                                      v
      +---------------------+           +---------------------------+
      |      GATHER         |           | EXECUTOR (Dense Kernel)   |
      | (Random Access)     | <-------> | 1. Apply P -> Continuous  |
      |   High Latency      |           | 2. Run GEMM (TMA Friendly)|
      +---------------------+           | 3. Apply P' -> Scatter    |
                                        +---------------------------+
                                                      |
          "Implicit Layout"                   "Explicit Permutation"
       (Dependency on Values)                 (Algebraic Object)
```
**Source:** "Inspector–Executor as a First-Class Semantics" [1] and [2].

---

#### Figure 5: Hierarchical Locality (Wreath Products)
*Visualizing the layout distribution hierarchy. Elements are mapped through a composition of groups (Cluster $\wr$ CTA $\wr$ Warp).*

```text
    HIERARCHY LEVEL        MEMORY SPACE          ALGEBRAIC GROUP ACTION
    ===================================================================

    [ GRID / CLUSTER ]     DSMEM (Distributed)   G_cluster (Permutation)
            |              (Multicast/Remote)            |
            v                                            v
    [     CTA        ]     SMEM (Shared)         G_cta (Tile Mapping)
            |              (Banked/Swizzled)             |
            v                                            v
    [     WARP       ]     Registers             G_warp (Shuffle)
            |              (Warp Uniform)                |
            v                                            v
    [    THREAD      ]     Registers             G_thread (Lane ID)
                           (Private)

    -------------------------------------------------------------------
    Composite Layout L = l_cluster * l_cta * l_warp * l_thread
```
**Source:** "Wreath Products / Hierarchical Group Actions" [1].

---

#### Figure 6: Swizzling as Combinatorial Design (SMT)
*Visualizing "Theory B: Combinatorial Design." Using XOR-based swizzling to avoid bank conflicts by rotating access patterns.*

```text
    (A) LINEAR / STRIDED ACCESS          (B) XOR-SWIZZLED ACCESS (Hash)
        (High Conflict Probability)          (Conflict Minimization)

    Bank 0   Bank 1   Bank 2             Bank 0   Bank 1   Bank 2
    +----+   +----+   +----+             +----+   +----+   +----+
    | T0 |   | T1 |   | T2 |  <-- Row 0  | T0 |   | T1 |   | T2 |
    +----+   +----+   +----+             +----+   +----+   +----+
    | T0 |   | T1 |   | T2 |  <-- Row 1  | T2 |   | T0 |   | T1 | (Shifted)
    +----+   +----+   +----+             +----+   +----+   +----+
    | T0 |   | T1 |   | T2 |  <-- Row 2  | T1 |   | T2 |   | T0 | (Shifted)
    +----+   +----+   +----+             +----+   +----+   +----+
      ^
      |                                    ^
    CONFLICT!                            NO CONFLICT
    (Serialization)                      (Parallel Access)

    Constraint: bank(addr) = (addr >> 2) ^ (addr & mask)
    Solved via: SMT Solver / Integer Linear Lattice
```
**Source:** "Modular Constraint Solving" and "Combinatorial Design" [1] [2].
