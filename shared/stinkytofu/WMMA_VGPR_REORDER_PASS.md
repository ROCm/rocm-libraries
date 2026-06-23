# Plan: WMMA Reorder Analysis Pass for VGPR Reduction

## Context

StinkyTofu operates on real physical register indices (non-SSA, assembly-level IR).
There is no register allocator — VGPR indices in the input are already fixed.

The GEMM loop body is 2× unrolled with double buffering:
- **First half**: wmma using `A_X0[0:63]`, `B_X0[0:63]`; ds_loads prefetch `A_X1[0:63]`, `B_X1[0:15]`
- **Second half**: wmma using `A_X1[0:63]`, `B_X1[0:63]`

**Problem**: Current first half uses B-major outer ordering → A_X0[i] lives until the very last B
group. A_X1 must sit in separate physical registers → 64 extra VGPRs.

**Opportunity**: Switching to A-major outer for the first half makes A_X0[i] dead after its last B
group iteration. A_X1[i] can then be aliased into A_X0[i]'s physical registers → 64 VGPRs saved.

**Scope of this pass**: Analysis only. The pass will:
1. Analyze the wmma sequence and determine the optimal reordering
2. Report which register groups can be aliased and how many VGPRs would be saved
3. Output the desired wmma ordering for a downstream pass/person to act on

The actual wmma reordering and register operand rewriting are out of scope for now.

---

## ABI Design: Three Independent Layers

There are three separate ABIs. Each can be swapped independently.

---

### Layer 1 — Liveness ABI (`IRegLivenessAnalysis`)

**Responsibility**: Given a basic block, compute live intervals for register groups.
Does not know anything about wmma ordering or aliasing decisions.

```cpp
// A register group identified by its base VGPR index and contiguous size.
struct RegGroup {
    unsigned base;
    unsigned size;
    bool operator==(const RegGroup&) const = default;
};

// Live interval in terms of instruction positions within the BB.
// "position" is an opaque index — consumers only compare for overlap.
struct RegInterval {
    unsigned first;  // first instruction position where group is live (def or first use)
    unsigned last;   // last instruction position where group is live (last use)
};

// ABI 1: liveness backend — swappable independently of the reorder algorithm.
class IRegLivenessAnalysis {
public:
    virtual ~IRegLivenessAnalysis() = default;

    // Compute live intervals for all register groups that appear as A/B sources in wmma.
    // Returns a map from group → interval.
    // Implementations may use wmma-only positions or full instruction positions —
    // callers treat the values as opaque ordinals and only compare them.
    virtual std::map<RegGroup, RegInterval> computeLiveness(
        const BasicBlock& bb,
        const std::vector<WmmaNode>& wmmaSeq) const = 0;
};

// ── Current: wmma-only ──────────────────────────────────────────────────────
// interval.first = index of first wmma that reads the group
// interval.last  = index of last  wmma that reads the group
// Fast; sufficient for the double-buffered GEMM pattern.
class WmmaIntervalLiveness : public IRegLivenessAnalysis { ... };

// ── Future: full backward dataflow ─────────────────────────────────────────
// interval.first = instruction index of the ds_load that defines the group
// interval.last  = instruction index of the last instruction (any type) that reads the group
// More precise; handles scalar/ds_write instructions that also touch A/B groups.
// class FullBackwardDataflowLiveness : public IRegLivenessAnalysis { ... };
```

---

### Layer 2 — Reorder Algorithm ABI (`IWmmaReorderAlgorithm`)

**Responsibility**: Given the wmma sequence and precomputed liveness intervals, decide the
optimal wmma ordering and which register groups can be aliased.
Does not know how liveness was computed.

```cpp
// ABI 2: reorder + alias algorithm — swappable independently of the liveness backend.
class IWmmaReorderAlgorithm {
public:
    virtual ~IWmmaReorderAlgorithm() = default;

    // Given the wmma sequence and their precomputed live intervals,
    // return the desired ordering and the aliasable register group pairs.
    struct AlgorithmResult {
        std::vector<WmmaNode> desiredOrder;      // permutation of wmmaSeq
        std::vector<AliasCandidate> aliases;      // which groups can share physical regs
    };

    virtual AlgorithmResult solve(
        const std::vector<WmmaNode>& wmmaSeq,
        const std::map<RegGroup, RegInterval>& liveness) const = 0;
};

// ── Current: A-major outer heuristic ───────────────────────────────────────
// Detects B-major outer pattern; produces A-major outer permutation.
// Finds (A_X0[i], A_X1[i]) alias pairs by checking non-overlapping intervals
// after the simulated reordering. O(n_wmma).
class AMajorOuterAlgorithm : public IWmmaReorderAlgorithm { ... };

// ── Future: interval graph coloring ────────────────────────────────────────
// Builds an interval conflict graph over all A/B groups; finds maximum independent
// sets to identify the most alias pairs. Works for arbitrary wmma patterns beyond 2× unroll.
// class IntervalGraphAlgorithm : public IWmmaReorderAlgorithm { ... };
```

---

### Layer 3 — Pass Output ABI (`WmmaReorderAnalysisResult`)

**Responsibility**: Stable data contract between this pass and downstream passes.
Never changes regardless of which liveness backend or algorithm is in use.

```cpp
struct RegReplacement {
    StinkyInstruction* inst;
    unsigned operandIdx;
    bool isSrc;
    StinkyRegister oldReg;
    StinkyRegister newReg;
};

struct WmmaReorderAnalysisResult {
    bool applicable;
    std::vector<StinkyInstruction*> desiredWmmaOrder;  // instruction pointers in new order
    std::vector<RegReplacement>     replacements;       // operand-level rewrite map
    unsigned                        totalVgprSaved;
};
```

---

### Pass wiring

```cpp
class StinkyWmmaReorderPass : public StinkyInstPass {
public:
    StinkyWmmaReorderPass(
        std::unique_ptr<IRegLivenessAnalysis>  liveness  = std::make_unique<WmmaIntervalLiveness>(),
        std::unique_ptr<IWmmaReorderAlgorithm> algorithm = std::make_unique<AMajorOuterAlgorithm>());

    // run() calls:
    //   1. liveness_->computeLiveness(bb, wmmaSeq)      → intervals
    //   2. algorithm_->solve(wmmaSeq, intervals)        → order + aliases
    //   3. buildReplacements(aliases, bb)               → RegReplacement list
    //   4. return WmmaReorderAnalysisResult{...}
};
```

Swapping either axis independently:
```cpp
// Swap liveness only (keep A-major heuristic):
auto pass = std::make_unique<StinkyWmmaReorderPass>(
    std::make_unique<FullBackwardDataflowLiveness>());

// Swap algorithm only (keep wmma-only liveness):
auto pass = std::make_unique<StinkyWmmaReorderPass>(
    std::make_unique<WmmaIntervalLiveness>(),
    std::make_unique<IntervalGraphAlgorithm>());
```

---

## Steps

### Step 1 — Collect and classify wmma instructions

Walk the basic block. Collect all `isXDLWMMA(inst)` instructions in program order.

```cpp
struct WmmaNode {
    StinkyInstruction* inst;
    unsigned aGroupBase;  // src0.idx (first VGPR index of A tile)
    unsigned bGroupBase;  // src1.idx
    unsigned cGroupBase;  // dest.idx
    unsigned groupSize;   // src0.num (e.g. 8 for v_wmma_f32_16x16x32_bf16)
};
std::vector<WmmaNode> wmmaSeq;  // all wmma in program order
```

Detect the first-half / second-half split: the wmma sequence naturally splits into two equal
halves where the first half uses one set of A/B register ranges and the second half uses another.
Identify the split point by checking when the A-group register base changes to a disjoint range.

### Step 2 — Build live intervals per A/B group (current ordering)

For each unique `aGroupBase` and `bGroupBase` in the first half, compute:
```
first_wmma[group] = earliest index in firstHalf where this group appears as src
last_wmma[group]  = latest  index in firstHalf where this group appears as src
interval[group]   = [first_wmma, last_wmma]
```

If any A group has `last_wmma - first_wmma == nB - 1` (spans all B iterations), this confirms
B-major outer ordering is in effect.

### Step 3 — Compute intervals under A-major outer reordering

Simulate the A-major outer ordering without mutating the IR:
```
for i in sorted(A_groups):
    for j in sorted(B_groups):
        virtual_index++
        record virtual_index as wmma position for (A[i], B[j])
```

Compute `simulated_interval[aGroup]` = [first virtual index, last virtual index].
In A-major outer, each A group's interval width = nB (spans exactly nB consecutive wmma positions),
and intervals for different A groups are non-overlapping.

### Step 4 — Find aliasable pairs

Two groups R (first half) and R' (second half) are aliasable if, under the simulated reordering,
`simulated_last_wmma[R] < first_wmma[R']` (R's interval ends before R' is first used).

Because the second half starts at index N/2 (after all first-half wmma), and under A-major outer
each A_X0[i] interval ends before the second half begins, **all** (A_X0[i], A_X1[i]) pairs are
aliasable.

Build the alias result:
```cpp
struct AliasCandidate {
    StinkyRegister canonical;   // keep these physical regs (A_X0[i])
    StinkyRegister aliasable;   // can be remapped to canonical (A_X1[i])
    unsigned vgprSaved;         // canonical.num
};
std::vector<AliasCandidate> aliasCandidates;
unsigned totalVgprSaved = sum of aliasable.num for each candidate;
```

### Step 5 — Output the analysis result

Produce a `WmmaReorderAnalysisResult` struct returned from the pass:

```cpp
// One entry per instruction operand that needs to change.
// The downstream rewriter applies this map mechanically — no re-analysis needed.
struct RegReplacement {
    StinkyInstruction* inst;   // which instruction to patch
    unsigned operandIdx;        // index into srcRegs or destRegs
    bool isSrc;                 // true = srcRegs, false = destRegs
    StinkyRegister oldReg;      // current register (e.g. A_X1[i])
    StinkyRegister newReg;      // replacement register (e.g. A_X0[i])
};

struct WmmaReorderAnalysisResult {
    // Whether the optimization applies to this basic block
    bool applicable;

    // Desired wmma ordering: a permutation of the original wmma instruction pointers.
    // Downstream reorder pass applies this order directly to the BB.
    std::vector<StinkyInstruction*> desiredWmmaOrder;

    // Flat per-operand replacement map.
    // Downstream register renaming pass applies this list directly — no further analysis needed.
    std::vector<RegReplacement> replacements;

    // Summary: how many VGPRs are freed if replacements are applied.
    unsigned totalVgprSaved;
};
```

The `replacements` vector covers:
- **ds_load destinations**: instructions currently writing into A_X1[i] → rewrite dest to A_X0[i]
- **second-half wmma sources**: instructions reading A_X1[i] → rewrite src to A_X0[i]

Log a summary (debug/info level):
```
[WmmaReorderPass] Applicable: true
[WmmaReorderPass] 8 aliasable A-group pairs → 64 VGPRs saved
[WmmaReorderPass] 72 register operand replacements emitted
```

---

## Pass Structure

New files:
- `shared/stinkytofu/include/stinkytofu/transforms/asm/StinkyWmmaReorderPass.hpp`
- `shared/stinkytofu/src/transforms/asm/StinkyWmmaReorderPass.cpp`
- **Class**: `StinkyWmmaReorderPass : public StinkyInstPass`
- **Returns**: `WmmaReorderAnalysisResult` via `PassContext` or `AnalysisManager`
- **Gate**: New `PassFeatureConfig` flag `enableWmmaVgprReorderAnalysis` (off by default)

Runs **before** `StinkyDAGSchedulerPass`. Does not mutate any instructions or register operands.

---

## Key APIs to reuse

| Purpose | Location |
|---|---|
| Identify wmma | `isXDLWMMA()` — `StinkyAsmIR.hpp` |
| Read src/dest regs | `inst->getSrcRegs()`, `inst->getDestRegs()` |
| Iterate per-DWORD | `forEachRegUnit()` — `RegisterKey.hpp` |
| BB instruction list (read-only) | `bb->getInstructions()` |
| Pass base class | `StinkyInstPass` — `PassManager.hpp` |
| Pass feature config | `PassFeatureConfig` — `PassContext.hpp` |

---

## Verification

1. Unit test: feed a synthetic wmma sequence (B-major outer, 8×8) and verify the pass reports:
   - `applicable = true`
   - `totalVgprSaved = 64`
   - `aliasCandidates` has 8 entries, each with the correct (A_X0[i], A_X1[i]) pairing
   - `desiredWmmaOrder` matches the expected A-major outer permutation
2. Unit test: feed an already-A-major-outer sequence → `applicable = false` (or 0 VGPRs saved).
3. Run on `raw_asm_roundtrip.s` loop body and check the logged output matches expectations.
