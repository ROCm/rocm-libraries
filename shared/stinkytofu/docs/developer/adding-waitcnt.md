# Adding New Instructions to WaitCntPass

Quick guide for extending WaitCntPass to support new memory instructions.

## Key Difference

- **Loads**: Precise register-level tracking (optimal performance)
- **Stores**: Conservative scalar counting only (no address analysis available)

---

## Adding Load Instructions

**File:** 'StinkyConfigurableWaitCntPass.cpp'

### 1. Add Classification Helper

'''cpp
inline bool isScratchLoad(const StinkyInstruction& inst) {
    return inst.getHwInstDesc()->has(InstFlag::SCRATCH_LOAD);
}
'''

### 2. Extend MemoryOperationState

'''cpp
struct MemoryOperationState {
    int scratchLoadCount = 0;
    std::vector<StinkyRegister> outstandingScratchLoads;
    // ... existing fields
};
'''

### 3. Update incrementForInst()

'''cpp
else if(isScratchLoad(inst)) {
    for(const auto& destReg : inst.destRegs) {
        auto oldSize = outstandingScratchLoads.size();
        outstandingScratchLoads.erase(
            std::remove_if(outstandingScratchLoads.begin(),
                          outstandingScratchLoads.end(),
                          [&destReg](const StinkyRegister& r) {
                              return r.isOverlap(destReg);
                          }),
            outstandingScratchLoads.end()
        );
        if(outstandingScratchLoads.size() == oldSize)
            scratchLoadCount++;
        outstandingScratchLoads.push_back(destReg);
    }
}
'''

### 4. Update Other Methods

'''cpp
// applyWaitCnt()
applyCount(wait.slcnt, scratchLoadCount, outstandingScratchLoads);

// mergeFrom() - Smart merge with canonical ordering
scratchLoadCount = std::max(scratchLoadCount, other.scratchLoadCount);
if(outstandingScratchLoads != other.outstandingScratchLoads) {
    if(oldCount == other.scratchLoadCount &&
       outstandingScratchLoads.size() == other.outstandingScratchLoads.size()) {
        // Same count, different order -> create canonical ordering (sort by register index)
        std::vector<StinkyRegister> merged = other.outstandingScratchLoads;
        std::sort(merged.begin(), merged.end());
        outstandingScratchLoads = merged;
    } else {
        // Different counts -> conservative merge, clear lists
        outstandingScratchLoads.clear();
    }
}

// isEmpty()
return ... && scratchLoadCount == 0 && ...;

// statesEqual() - Must compare register list contents (not just sizes)
if(a.scratchLoadCount != b.scratchLoadCount) return false;
if(a.outstandingScratchLoads != b.outstandingScratchLoads) return false;  // Full equality check

// isMemoryLoad()
return ... || isScratchLoad(inst);
'''

---

## Adding Store Instructions

**[!] IMPORTANT:** Stores use a simpler, conservative approach (no register tracking).

### Why Conservative?

Without address analysis, we can't determine if stores conflict:
'''assembly
ds_write [v0], v2  ; Write to address in v0
ds_write [v1], v3  ; Does [v0] == [v1]? Unknown!
'''

### Implementation

'''cpp
// 1. Add classification
inline bool isScratchStore(const StinkyInstruction& inst) {
    return inst.getHwInstDesc()->has(InstFlag::SCRATCH_STORE);
}

// 2. Add ONLY scalar counter (NO register vector!)
struct MemoryOperationState {
    int scratchStoreCount = 0;  // [x] Yes
    // std::vector<StinkyRegister> outstandingScratchStores;  // [ ] NO!
};

// 3. Simple increment
else if(isScratchStore(inst)) {
    scratchStoreCount++;  // That's it!
}

// 4. Update other methods
// applyWaitCnt()
if(wait.sscnt.has_value() && *wait.sscnt < scratchStoreCount)
    scratchStoreCount = *wait.sscnt;

// mergeFrom()
scratchStoreCount = std::max(scratchStoreCount, other.scratchStoreCount);

// isEmpty()
return ... && scratchStoreCount == 0 && ...;

// At barriers: wait for ALL
waitReq.sscnt = WAIT_COMPLETE;  // Always wait all
'''

---

## Control Flow Merge Behavior

**Critical:** The pass uses **multi-path analysis** to achieve optimal wait counts even with multiple predecessors.

### Core Strategy: Multi-Path Analysis

Instead of merging predecessor states early, the pass:
1. **Keeps all predecessor states separate**
2. **For each instruction**, computes wait requirements from EACH path
3. **Takes the minimum** (most restrictive) across all paths
4. **Then merges** states conservatively for internal block processing

This ensures **maximum precision** for GPU performance.

### Single Predecessor -> **Precise**

When a block has only **one predecessor**, state is directly used:

'''
block1:
  ds_read v0, v1, v2, v3
  v
block2:
  v_fmac v4, v0, v2  <- dlcnt=1 (precise!)
'''

**Result:** Full register tracking preserved -> Optimal 'dlcnt' values [x]

### Multiple Predecessors -> **Multi-Path Analysis (Precise!)**

When a block has **2+ predecessors**, multi-path analysis computes optimal waits:

'''
      entry
       /  \
   block1     block2
   (v0,v1)    (v2,v3,v4)
       \     /
       block3:
         v_fmac v5, v0, v1  <- Analyzes BOTH paths!
'''

**How it works:**
'''cpp
void insertCrossBlockDependencyWaitCounts() {
    // Keep predecessor states SEPARATE (no early merge)
    std::vector<MemoryOperationState> predecessorStates;

    for(auto it = insts_.begin(); ...; ++it) {
        // For EACH predecessor, compute what wait is needed
        WaitCntRequirement finalReq;
        for(const auto& predState : predecessorStates) {
            WaitCntRequirement pathReq = computeWaitRequirementForPath(predState, inst);
            finalReq.merge(pathReq);  // Takes minimum (most restrictive)
        }

        if(finalReq.isValid()) {
            insertWaitCnt(it, finalReq);
            // NOW merge for internal processing
            mergePredecessorStates();
            return;
        }
    }
}
'''

**Result:** Optimal 'dlcnt' for ALL cases (no conservative fallback) [x]

### Preloop + Loop -> **Multi-Path Analysis (Precise!)**

Loop entry paths get optimal handling through multi-path analysis:

'''
block1 (preloop):
  ds_read v0, v1, v2, v3
  v
block2 (loop):
  v_fmac v4, v0, v2  <- dlcnt=1 (optimal!)
  ds_read v0, v2, v1, v3
  v (back-edge to itself)
'''

**How it works:**

'''
Path 1 (preloop):  [v0, v1, v2, v3] -> Using v0, v2 -> needs dlcnt=1
Path 2 (loop):     [v0, v2, v1, v3] -> Using v0, v2 -> needs dlcnt=2

Final: min(1, 2) = 1  <- Takes most restrictive (lowest)
'''

This ensures **both** paths are satisfied with minimal waiting!

### Why Multi-Path Analysis?

**GPU Performance is Critical:** Every cycle of waiting impacts performance.

**Old approach (conservative merge):**
'''
Path A: [v0, v1]       (2 outstanding)
Path B: [v2, v3, v4]   (3 outstanding)
Merge: count=3, list=[] (cleared)
Result: dlcnt=0 (wait for ALL) [ ]
'''

**New approach (multi-path analysis):**
'''
Path A: [v0, v1]       -> v_fmac uses v0 -> needs dlcnt=1
Path B: [v2, v3, v4]   -> v_fmac uses v0 -> not present, no wait
Final: min(1, IGNORE) = 1  <- Optimal! [x]
'''

### Summary Table

| Scenario | Predecessors | Wait Strategy | Example 'dlcnt' |
|----------|--------------|---------------|-----------------|
| **Linear chain** | 1 (non-loop) | Direct use | 'dlcnt=1' |
| **Self-loop** | 1 (itself) | Direct use | 'dlcnt=2' |
| **Preloop + loop** | 2 (1 entry + 1 back-edge) | Multi-path analysis | 'dlcnt=1' [x] |
| **Diamond merge** | 2+ different non-loop | Multi-path analysis | 'dlcnt=1' [x] |

**Key Insights:**
- **Multi-path analysis** = Always precise, never conservative
- **GPU performance** = Minimal waiting = Maximum throughput
- **Handles all CFG patterns** without falling back to 'dlcnt=0'
- **After first wait**: States are merged conservatively for internal block processing (acceptable since the critical cross-block dependency is already handled optimally)

---

## Testing

**File:** 'ConfigurableWaitCntPassTest.cpp'

'''cpp
TEST_F(ConfigurableWaitCntPassTest, ScratchLoadBasic) {
    BasicBlock* bb = func->createBasicBlock("test");

    createScratchLoad(bb, arch, 0);
    createVAddF32(bb, arch, 1, 0, 2);  // Uses v0

    runWaitCntPass();

    // Verify s_waitcnt was inserted before v_add
    auto* waitInst = findWaitCntBefore(vaddInst);
    ASSERT_NE(waitInst, nullptr);
    EXPECT_EQ(waitInst->getWaitCntData().slcnt, 0);
}
'''

**Test scenarios:**
- Basic load-use dependency
- Multiple outstanding loads
- Register reissue
- Cross-block dependency (single predecessor)
- Multi-predecessor merge (diamond CFG)
- Loop convergence (preloop + loop)

---

## Checklist

### Loads
- [ ] Add 'isScratchLoad()' helper
- [ ] Add 'scratchLoadCount' + 'outstandingScratchLoads'
- [ ] Update 'incrementForInst()' with reissue handling
- [ ] Update 'applyWaitCnt()', 'mergeFrom()', 'statesEqual()', 'isEmpty()'
- [ ] Update 'isMemoryLoad()' classification
- [ ] Add tests

### Stores
- [ ] Add 'isScratchStore()' helper
- [ ] Add 'scratchStoreCount' only (NO vector!)
- [ ] Update 'incrementForInst()' with simple increment
- [ ] Update 'applyWaitCnt()', 'mergeFrom()', 'isEmpty()'
- [ ] Barrier handling: 'waitReq.sscnt = WAIT_COMPLETE'
- [ ] Add tests

---

## Common Mistakes

'''cpp
// [ ] WRONG: Register tracking for stores
std::vector<StinkyRegister> outstandingScratchStores;

// [x] CORRECT: Only scalar counter
int scratchStoreCount = 0;

// [ ] WRONG: Precise store wait
waitReq.sscnt = 2;

// [x] CORRECT: Wait all at barriers
waitReq.sscnt = WAIT_COMPLETE;
'''

---

## Summary

| | Loads | Stores |
|---|---|---|
| **Counter** | [x] Yes | [x] Yes |
| **Register Vector** | [x] Yes | [ ] No |
| **Precise Waits** | [x] 'dlcnt=2' | [ ] 'dscnt=0' only |
| **Complexity** | High | Low |

**Loads:** Precise tracking (dest register known)
**Stores:** Conservative (memory address unknown)

---

## See Also

- [Architecture Overview](architecture.md) - System architecture and pass pipeline
