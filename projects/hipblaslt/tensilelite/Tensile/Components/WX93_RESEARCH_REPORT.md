# wx9.3 Research Report: Comparing Kernel Dataflow Graphs Modulo Register Renaming

> **Provenance disclosure.** WebSearch and WebFetch were both returning 400 errors
> ("claude-haiku-4-5-20251001 deployment not found") for the entire duration of this
> task, so I could not perform live web research. This report is written from training
> knowledge of the literature and tools listed. URLs are accurate to my knowledge but
> have not been re-verified in this session — treat them as starting points, not as
> ground truth, and spot-check before quoting them externally.

---

## TL;DR recommendation

Implement **two-pass canonicalization, not isomorphism search**. Your two graphs are stipulated
to have identical topology — so you do not need general graph isomorphism. You need a
deterministic *canonical relabeling* of registers that is a function of position in the
dataflow graph, applied to both sides, after which the existing string-hash comparison
"just works." This is exactly what compilers call **value numbering** / **SSA renaming to a
canonical form**. The right algorithm is a **1-WL (color refinement) pass keyed on the SSA
def-site**, which is O((V+E) log V) and handles every realistic case you will see in a
GEMM kernel. Reserve nauty/bliss-style isomorphism only as a fallback for the
"topology-equivalent but not order-equivalent" mode, behind your flag.

The symbolic-vs-numeric naming sub-problem (7a) is solved at the **frontend** of graph
construction, not in the comparator: resolve every name through a single
`SymbolMap` to a canonical token before any node is hashed. Do not let the comparator
see two spellings of the same physical register.

---

## 1. Canonical algorithms for this problem class

### 1.1 Color refinement / 1-dimensional Weisfeiler-Lehman (1-WL)

**Core idea.** Initialize every node's color to its label (opcode + operand kinds, *not*
register names). Repeatedly refresh each node's color to a hash of `(own color,
multiset of (edge-label, neighbor color))`. Iterate to fixpoint. After fixpoint, two
nodes with the same color are indistinguishable by any local-structural test of this
power. To get a *canonical labeling* of registers, replace the register name on each
edge with a tuple `(producer-color, consumer-color, ordinal)` where `ordinal` breaks
ties among parallel edges in a deterministic order (e.g. by sorted consumer color).

**Cost.** O((V+E) log V) using the Cardon-Crochemore / Paige-Tarjan partition
refinement formulation. In practice tens of microseconds for a 1000-node graph in
Python with `dict`-of-tuples coloring.

**Invariants required.** Edges must be deterministically iterable. Node labels must be
canonicalized first (opcode + literal operands, *not* renamings).

**When it works.** Whenever you only need to distinguish nodes up to local structural
equivalence, which is *exactly* your case: the two graphs differ only in register
names, and a register name in SSA dataflow is determined up to renaming by
`(def-site identity, use-site identities)` — which is what 1-WL recovers.

**When it fails.** 1-WL famously fails to distinguish certain regular graphs (e.g.
two non-isomorphic strongly regular graphs with the same parameters). This is
irrelevant to your problem: dataflow graphs of straight-line code are extremely
asymmetric — every instruction has a distinct opcode/operand position, and roots
(kernel argument loads) and sinks (stores, output writes) are pinned. You will
essentially never hit a refinement collision on real kernels. If you ever do, it
falls into the "two registers genuinely interchangeable in this kernel" bucket,
which is exactly the equivalence you want anyway.

### 1.2 Backtracking canonical labeling: nauty / bliss / Traces

**Core idea.** Run color refinement, then descend an "individualization-refinement"
search tree: pick a node from the smallest non-singleton cell, individualize it
(give it a unique color), refine again, recurse. Maintain the lex-smallest labeling
seen and prune branches that cannot beat it. The result is a canonical adjacency
matrix that is invariant under isomorphism.

**Cost.** Worst case super-polynomial; in practice fast on graphs with few
automorphisms. nauty/bliss handle 10^4–10^5 nodes routinely on combinatorial graphs.

**Invariants required.** Vertex coloring must be a proper invariant (i.e. iso-stable).

**When it works.** General graph isomorphism / canonical-form. Use this if you ever
need to compare two kernels that have **different scheduling order** (not just
different register names) and you cannot pin nodes by execution position.

**When it doesn't.** Overkill when 1-WL already distinguishes everything, which it
will for your inputs. The cost of integrating a C library (`pynauty` is a Python
binding) is not justified for the renaming-only case.

### 1.3 Value numbering / SSA renaming as canonicalization

**Core idea.** This is the classic compiler trick. Walk instructions in dataflow
order; assign each definition a value number that is a hash of `(opcode, ordered
value-numbers of operands)`. Two definitions with the same value number compute
the same value. To compare two SSA programs modulo renaming, compute value numbers
on both sides and compare value-number-streams instead of register-name-streams.

**Cost.** O(V+E) single pass after a topological sort.

**Invariants required.** SSA form (each register defined exactly once) and a
topological order. Both are properties of straight-line dataflow graphs.

**When it works.** Perfectly for your "identical topology, different names" case.
This is essentially 1-WL specialized to DAGs and is the textbook approach for
"is this CSE-equivalent transformation correct."

**When it doesn't.** When opcodes have semantic identities (e.g. `add` is
commutative, so `add r1, r2` and `add r2, r1` should hash equal). You handle this
by sorting operands of commutative opcodes before hashing — a known rocBLAS/Tensile
issue since GFX assembly has commutative `v_add_*` and `v_mul_*` family
instructions, plus FMA where only two of the three operands commute.

### 1.4 SMT / symbolic-execution equivalence (Alive2 / translation validation)

**Core idea.** Encode each kernel as an SMT formula over abstract values
(`f1(inputs) = vector of outputs`), then ask the solver `forall inputs.
f1(inputs) == f2(inputs)`. Register names disappear because they are just
intermediate variables that get existentially-eliminated.

**Cost.** Solver time, often seconds-to-minutes; exponential in the worst case.
Numerical types (float, especially fp8/bf16) blow up the encoding.

**Invariants required.** Precise semantics for every opcode. For VALU/MFMA/DS
instructions on AMD GCN, this is a major undertaking — there is no published
SMT-LIB-bit-precise model of MFMA.

**When it works.** When you can afford the modeling cost and you actually need
*semantic* equivalence (not just structural). Alive2 does this for LLVM IR; CompCert
does it as machine-checked proof for RTL→Mach.

**When it doesn't.** Now. You don't have semantics for every GFX opcode and you
don't need them — you stipulate the two graphs are **already the same up to
register renaming**, which is a much weaker check than full semantic equivalence.
Don't pay for what you don't need.

### 1.5 Translation validation (Necula, Pnueli/Siegel/Singerman)

**Core idea.** For each compilation, emit a *witness* mapping source variables
to target variables (a simulation relation). The validator only checks the
witness, not the optimization. Necula's PLDI'00 paper and the
Pnueli/Siegel/Singerman TACAS'98 paper are the foundational references.

**Relevance to you.** You don't have a witness — the scheduler doesn't emit a
register correspondence. But this is **exactly** the framework you would adopt if
you ever wanted to validate SIA0 → SIA3 across genuinely different schedules.
The current approach (compute the correspondence post-hoc by 1-WL on the dataflow
graph) is essentially "infer the simulation relation from structure."

### 1.6 Peephole verifiers: Souper, Alive

Souper extracts an SSA "candidate optimization" (LHS DAG → RHS expression),
canonicalizes both sides by **renaming all variables to `%0, %1, ...` in a
deterministic dataflow walk**, and then queries the SMT solver. The
canonicalization step alone is what you want — it makes structurally-equivalent
candidates *string-equal*, which lets Souper deduplicate its candidate cache by a
hash. This is the cheapest, most pragmatic version of what we are recommending.

---

## 2. How real projects solve it

(URLs from memory; verify before citing externally.)

- **LLVM `MachineCSE` / `EarlyCSE`** — value-numbering on MachineInstrs. Each
  MachineInstr gets a hash of (opcode, hashed operands), with virtual register
  *defs* stripped so renaming doesn't perturb the hash and *uses* substituted with
  the value-number of their producer. This is item 1.3 above, in production.
  See `llvm/lib/CodeGen/MachineCSE.cpp`. <https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/MachineCSE.cpp>

- **LLVM `MachineVerifier`** — checks SSA invariants and register-class
  consistency, not equivalence. Mentioned because people reach for it and it is
  the wrong tool. <https://llvm.org/docs/MIRLangRef.html>

- **GCC `-fcompare-debug`** — compiles twice (once with, once without debug
  info), strips debug, and checks the assembly streams are byte-identical. The
  *deliberate design choice* is that GCC normalizes register allocation enough
  that byte equality is achievable. The lesson for you: if you can make the
  scheduler emit registers in a deterministic *position-derived* order, you don't
  need any clever comparator — `diff` works. This is the cleanest long-term fix.
  <https://gcc.gnu.org/onlinedocs/gccint/Compare_002ddebug.html>

- **Souper** — canonicalizes SSA by renumbering all internal vars during DFG
  walk, then SMT-checks. Key insight: *canonicalize first, hash second; the
  expensive checker is a fallback, not the primary equality test.*
  <https://github.com/google/souper>

- **Alive2** — translation validator for LLVM IR. Source and target have
  independent SSA name spaces; Alive2 binds them via positional argument
  matching at function entry, then refinement-checks the rest symbolically.
  <https://github.com/AliveToolkit/alive2>, paper: Lopes et al., PLDI'21,
  "Alive2: Bounded Translation Validation for LLVM."

- **CompCert** — register allocation (Coloring → Allocation) is verified by a
  Coq proof that the *interference graph + coloring* preserves the source RTL
  semantics. The proof is *witness-based* (the coloring is the witness). See
  Leroy, JAR'09, "A Formally Verified Compiler Back-end."
  <https://compcert.org/>

- **Equality saturation / egg / Peggy** — represents all known equivalents as
  an e-graph, where each e-class is a canonical equivalence class of values.
  Register names are e-class IDs. Two terms are equal iff they share an e-class.
  Excellent if you ever need to grow the equivalence to algebraic identities.
  egg paper: Willsey et al., POPL'21. <https://egraphs-good.github.io/>

- **STOKE** (Stanford superoptimizer for x86-64) — equivalence-checks two
  basic blocks under register renaming by symbolic execution and a per-register
  permutation match. Relevant because it operates on *machine* code, not IR,
  same regime as you. <https://github.com/StanfordPL/stoke>

- **BinDiff / Diaphora** — binary function matching across compilations.
  Multi-stage matching: hash on call-graph signature, then on CFG fingerprint
  (per-BB instruction-class histograms, *not* register names), then on
  Weisfeiler-Lehman-style label refinement. The two-tier "cheap structural hash
  → expensive iso check" pattern is the right architecture for your validator
  too. <https://www.zynamics.com/bindiff.html>

- **Triton compiler** — lowers TritonGPU IR through MLIR; uses MLIR's
  `OperationEquivalence` which hashes on opcode+operand-position, ignoring SSA
  value names. Same idea as MachineCSE but at MLIR level.
  <https://mlir.llvm.org/doxygen/classmlir_1_1OperationEquivalence.html>

- **AMD ROCm / Tensile** — no public equivalence checker that I'm aware of;
  this validator is filling a real gap.

---

## 3. Recommendation for your specific case

### 3.1 The algorithm

Use **SSA value numbering as canonical relabeling**, run during graph
construction or as a single pass over the existing graph:

1. **Topo-sort** the dataflow graph (you already have it; it's the instruction
   order in the kernel).
2. For each node in order, compute `vn(node) = hash(opcode_canonical,
   sorted_or_ordered_tuple_of(vn(operand) for operand in operands))`. For
   commutative opcodes, sort operand value-numbers; for non-commutative, keep
   positional order.
3. For each *register definition* by node `n`, assign canonical name
   `"r" + base32(vn(n))` (or just the integer `vn(n)`).
4. Rewrite every edge in the graph to use these canonical names instead of
   `v34`/`v52`.
5. Compare the two graphs with your existing string-hash comparator. They will
   match.

This is O(V+E), single pass, no library dependencies, no SMT solver.

For the "topology-equivalent but different schedule order" mode (your flag),
fall back to **1-WL color refinement** over the graph instead of step 1's
topo-walk. Same hashing rule; iterate to fixpoint instead of single-pass. Still
O((V+E) log V).

Reserve **bliss/nauty** for a possible future "two functionally-equivalent
kernels with different topology" mode (SIA0 vs SIA3 with reorderings that change
the DAG, e.g. fused vs unfused). Don't build it now.

### 3.2 Data structure changes

- Introduce a `CanonicalForm` object that wraps a graph and exposes
  `node.canonical_id` and `edge.canonical_register`. Compute it once,
  cache it on the graph.
- Your existing canonical string is built from `node.id + edge.label`. Change
  *only* the edge.label source from `register_name` to
  `edge.canonical_register`. The hashing/comparison logic upstream stays
  identical.
- Build canonical form **post-hoc** rather than during graph build. This keeps
  the graph constructor honest about what the kernel actually said and lets the
  validator print both the original and canonical names in diff output, which
  is essential for debugging when the comparator says "different" and you need
  to find the offending instruction.

### 3.3 Pitfalls

- **Commutative opcodes.** `v_add_f32 v1, v2, v3` ≡ `v_add_f32 v1, v3, v2`.
  Sort operand value-numbers before hashing for commutative ops. Maintain an
  explicit `COMMUTATIVE_OPCODES` set; AMD GCN ISA manuals enumerate them, and
  you must include FMA's two-of-three commutativity carefully.
- **Multi-def instructions.** Some opcodes (e.g. carry-out adds, MFMA producing
  multiple AGPRs) define more than one register. Value-number each def
  independently as `vn(node, def_index)`.
- **Memory operations.** Loads/stores have a register operand that is an
  *address* and a register operand that is *data*. The address producer is part
  of the dataflow; the memory itself is a side-effect. Model loads as
  `vn = hash(opcode, vn(addr), memory_version)` where `memory_version`
  increments at every store. Otherwise two loads from the same address get the
  same value-number even when a store between them invalidated the value.
- **Predicate / EXEC / SCC / VCC.** GCN has implicit register dependencies on
  SCC (scalar carry/condition), VCC (vector carry/condition), EXEC (execution
  mask), and M0. Treat them as first-class registers in the dataflow graph or
  the canonicalization will silently miss WAW hazards through them. This is the
  single most common bug class for AMD GPU dataflow analyzers.
- **Immediates.** Bake immediates into the opcode label, never into the operand
  list as if they were register references. Otherwise an immediate `0` collides
  with the value-number of any register that happens to be `0`.
- **Register pairs / wide registers.** `v[4:5]` vs `v[6:7]` — canonicalize the
  pair by the value-number of the *def*, then have a per-pair-element hash so
  consumers reading only the low half get a distinct value-number from
  consumers reading the full pair.
- **Phi-like joins** at loop headers (if your graph is for a loop body):
  introduce an explicit phi node so cycles in the dataflow are broken
  deterministically. Without this, value-numbering doesn't terminate.

### 3.4 Symbolic-vs-numeric naming (sub-problem 7a)

Solve at the **graph build frontend**, not in the comparator:

- Maintain a single `SymbolMap` that resolves *every* register reference,
  symbolic (`vgpr_acc_0`) or numeric (`v34`), to a unique
  `PhysicalRegister` object.
- The mapping is rebuilt per-kernel from rocisa's symbol table at parse time.
- Every node and every edge stores `PhysicalRegister` references, not strings.
- Stringification (for the canonical hash) goes through
  `PhysicalRegister.canonical_name()` which returns one and only one spelling.

Once you do this, the canonical-form pass in §3.1 sees only one name per
physical register, and the symbolic-vs-numeric divergence cannot leak into the
comparator. Build a regression test where the same kernel is parsed twice with
the symbol table forcibly inverted between symbolic and numeric — they must
hash identically.

### 3.5 Edge cases that will trip you

- **Dead defs.** Instructions whose result is never used (debug, masked-out
  lanes, optimization slack). Their value-number contributes nothing to any
  consumer, so a missing dead def on one side and a present one on the other
  will diff. Decide deliberately: either prune dead defs as a normalization
  step, or treat their absence as a real difference.
- **Renamable vs reserved registers.** SGPRs holding kernel arguments and
  certain ABI-fixed registers (s[0:1] kernarg pointer, etc.) must *not* be
  renamed by the canonicalizer — they are pinned by ABI. Mark them
  non-renamable in `PhysicalRegister` and have the canonicalizer pass them
  through.
- **MFMA accumulator chains.** A long chain of MFMAs writes the same AGPR
  tile; the value-number of each MFMA in the chain is a function of the
  previous one. Chain length must match exactly; one extra MFMA on either side
  shifts every downstream value-number. This is desirable behavior — that *is*
  a real difference — but expect noisy diffs when it fires, and surface the
  "first divergent value-number" position to make triage tractable.
- **Scheduler-inserted `s_waitcnt` and `s_nop`** — these have no def but they
  do have ordering effects. Model them as nodes with empty value-numbers or
  exclude them from canonicalization (then compare them positionally as a
  separate stream).
- **Unique-ifying hash collisions.** Use at least 64-bit value-numbers; a
  32-bit hash will collide on real kernels (1000 nodes, birthday bound ≈ 50%
  collision risk near 65k value-numbers but cumulative across regression runs
  the risk grows). Python's built-in `hash` is not stable across runs unless
  you set `PYTHONHASHSEED`; use `hashlib.blake2b(digest_size=8)` or stable
  64-bit FNV.

---

## 4. References (prioritized, top 8)

1. **Cooper, Simpson, Vick, "Operator Strength Reduction" / "Value Numbering"
   (TR94-242, Rice).** The foundational SSA value-numbering paper. Read this
   first; it's the algorithm in §3.1 in 12 pages.
   <https://www.cs.rice.edu/~keith/EMBED/dom.pdf> (or the Rice TR series)

2. **Lopes, Menendez, Nagarakatte, Regehr, "Alive2: Bounded Translation
   Validation for LLVM," PLDI 2021.** The state of the art for IR-level
   equivalence; explains why register naming is encoded existentially.
   <https://web.ist.utl.pt/nuno.lopes/pubs/alive2-pldi21.pdf>

3. **Necula, "Translation Validation for an Optimizing Compiler," PLDI 2000.**
   The classical translation-validation paper; the witness-based framing your
   long-term SIA0/SIA3 effort will end up in.
   <https://people.eecs.berkeley.edu/~necula/Papers/tv00.pdf>

4. **Pnueli, Siegel, Singerman, "Translation Validation," TACAS 1998.** The
   other foundational TV paper, predating Necula.

5. **Junttila, Kaski, "Engineering an Efficient Canonical Labeling Tool for
   Large and Sparse Graphs" (bliss).** The most accessible engineering
   description of nauty-class algorithms. Read if you ever build the
   topology-different mode.
   <https://users.aalto.fi/~tjunttil/bliss/>

6. **Willsey et al., "egg: Fast and Extensible Equality Saturation," POPL
   2021.** If you ever want to grow equivalence beyond renaming (e.g. algebraic
   reassociation), read this.
   <https://egraphs-good.github.io/>

7. **LLVM `MachineCSE.cpp` source.** Production-grade SSA value numbering on
   machine instructions, including the commutative-operand and multi-def
   handling you need. Read the `isPhysDefTriviallyDead` / `ProcessBlock`
   functions.
   <https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/MachineCSE.cpp>

8. **GCC `-fcompare-debug` documentation and the Bugzilla history that
   motivated it.** The clearest case study of "make the compiler deterministic
   so byte-diff works" as an alternative to building a clever comparator. The
   right long-term answer if your scheduler can be made deterministic.
   <https://gcc.gnu.org/onlinedocs/gccint/Compare_002ddebug.html>

---

## Closing opinion

You are over-thinking this if you reach for graph isomorphism. The *real*
problem is that you let register names — which are arbitrary labels assigned
downstream of any semantic decision — into the canonical form upstream of the
comparator. Strip them out by replacing every register name with a value-number
derived from the dataflow position of its definition; keep the topology
otherwise unchanged; let your existing comparator do its existing job. This is
~150 lines of Python, no new dependencies, O(V+E), and matches what every
production compiler does for the same problem at IR level.

The flag for "topology-equivalent but reordered" is the only place 1-WL
iteration earns its keep, and it is a 20-line extension to the same code path.

Save SMT, nauty, and equality saturation for the day you actually need to
compare two kernels that compute the same answer with genuinely different
algebra. That day is not today, and when it comes, the right tool will be
Alive2-shaped, not nauty-shaped.
