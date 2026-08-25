# WMMA VGPR Reorder Pass

## Overview

`StinkyWmmaVgprReorderPass` is a read-only analysis pass that detects VGPR savings
in N-buffered GEMM kernels by identifying when wmma instructions can be reordered
to allow cross-pool register aliasing.

The pass never mutates any instruction or register operand. All output is in
`WmmaReorderAnalysisResult` for downstream passes to act on.

---

## Problem

In an N-buffered GEMM loop body the wmma instructions are split into N pools, each
using its own set of registers for the pool-varying operand (either A or B depending
on the kernel). When that operand is the *outer* loop dimension its registers stay
live across all iterations of the inner dimension, preventing cross-pool aliasing and
wasting VGPRs.

Making the pool-varying operand the *inner* dimension (grouping its instructions
contiguously within each pool) tightens its liveness interval so that pool N's
registers die before pool 0's next use. The two pools can then be aliased to the
same physical registers.

---

## Three-Layer ABI

The pass is structured as three independent layers. Each can be swapped without
touching the others.

### Layer 1 — `IRegLivenessAnalysis`

Computes live intervals for the A and B register groups referenced by the wmma
sequence.

```cpp
class IRegLivenessAnalysis {
public:
    virtual std::map<RegGroup, RegInterval> computeLiveness(
        const BasicBlock&            bb,
        const std::vector<WmmaNode>& wmmaSeq) const = 0;
};
```

**Built-in implementation**: `WmmaIntervalLiveness`

Fast wmma-only backend. The interval for a register group is
`[first wmma index that reads it, last wmma index that reads it]`.
Sufficient for standard N-buffered GEMM patterns.

A full-instruction liveness backend can be substituted here without changing
anything else.

---

### Layer 2 — `IWmmaReorderAlgorithm`

Given the pre-grouped pools and precomputed live intervals, decides the optimal
reordering and which register groups can be aliased.

```cpp
class IWmmaReorderAlgorithm {
public:
    struct Result {
        std::vector<WmmaNode>       desiredOrder;
        std::vector<AliasCandidate> aliases;
    };

    virtual Result solve(
        const std::vector<std::vector<WmmaNode>>& pools,
        const std::map<RegGroup, RegInterval>&    liveness) const = 0;
};
```

`pools` arrives already grouped and relabeled by `detectABIndices` so that
`WmmaNode::aGroup` is always the pool-varying operand and `WmmaNode::bGroup`
is always pool-shared.

**Built-in implementation**: `PoolVaryingReorderAlgorithm`

Checks whether the pool-varying operand (A) has inflated liveness
(`interval width > number of distinct B groups`). If so, reorders each pool
via `reorderContiguously` to make A groups contiguous, then pairs up
corresponding A groups across pools as alias candidates.

---

### Layer 3 — `WmmaReorderAnalysisResult`

The stable output contract consumed by all downstream passes.

```cpp
struct WmmaReorderAnalysisResult {
    bool                          applicable;
    std::vector<StinkyInstruction*> desiredWmmaOrder;  // permutation of wmma pointers
    std::vector<RegReplacement>   replacements;         // per-operand rewrite map
    unsigned                      totalVgprSaved;
};
```

Retrieve after the pass runs:

```cpp
const WmmaReorderAnalysisResult* result = getWmmaReorderResult(bb);
```

---

## Pool Tagging

TensileLite stamps each wmma instruction with a `WmmaPoolData` modifier at
generation time so the pass can group instructions by pool without heuristics.

```cpp
struct WmmaPoolData : public TypedModifier<WmmaPoolData> {
    uint32_t poolIndex = 0;
};
```

If any wmma instruction is missing this modifier the pass bails out for the
entire basic block. Partial tagging indicates a misconfigured pipeline and the
pass must not proceed on incomplete information.

---

## A/B Detection

`detectABIndices` determines which src operand index is pool-varying (A) vs
pool-shared (B) by checking register group intersection across pools. If src0's
groups appear in multiple pools it is pool-shared and src1 is A; otherwise src0
is A.

This means the pass handles both the standard case (hardware A is pool-varying)
and the symmetric case (hardware B is pool-varying) without any special-casing.

---

## Injecting Custom Implementations

Both the liveness backend and the algorithm are injected at construction:

```cpp
auto pass = createStinkyWmmaVgprReorderPass(
    std::make_unique<MyLivenessBackend>(),
    std::make_unique<MyReorderAlgorithm>());
```

Passing `nullptr` for either uses the built-in default.
