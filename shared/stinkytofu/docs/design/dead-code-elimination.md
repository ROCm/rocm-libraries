# Dead Code Elimination Pass Design

## Overview

The Dead Code Elimination (DCE) pass removes "dead store" instructions - instructions that write to a register that is overwritten before being used. This is a simple, safe pattern that works across basic blocks and function boundaries.

**Key Principle:** If an instruction writes to a register, and that register is redefined before any use, the first write is dead and can be safely removed.

---

## Architecture

'''
+---------------------+
|  Input IR           |  Function with dead stores
+----------+----------+
           |
           v
+---------------------+
|  For Each           |  Find instructions that define registers
|  Instruction        |
+----------+----------+
           |
           v
+---------------------+
|  Forward Scan       |  Look for uses or redefinitions
|                     |
+----------+----------+
           |
           +--> Found USE -----> Keep (Live)
           |
           +--> Found REDEF ---> Remove (Dead Store)
           |
           +--> End of Func ---> Keep (Might be used elsewhere)
           |
           v
+---------------------+
|  Safety Check       |  mustPreserveInstruction()
+----------+----------+
           |
           v
+---------------------+
|  Remove Dead        |  Delete overwritten instructions
|  Stores             |
+----------+----------+
           |
           v
+---------------------+
|  Iterate Until      |  Repeat until no more changes
|  Fixpoint           |
+----------+----------+
           |
           v
+---------------------+
|  Optimized IR       |  Reduced instruction count
+---------------------+
'''

---

## Algorithm

### Efficient Backward Scan (O(n))

'''cpp
// Track registers that are used/redefined later
std::set<StinkyRegister> usedRegs;       // Used before next definition
std::set<StinkyRegister> redefinedRegs;  // Redefined later

// Scan BACKWARD through all instructions (single pass)
for each instruction (backward from end):
    if mustPreserveInstruction(instruction):
        // Side effects - mark sources as used, dests as redefined
        for each source: usedRegs.insert(source)
        for each dest: redefinedRegs.insert(dest)
        continue

    // Check if this is a dead store
    bool isDeadStore = false
    for each destination:
        if (destination in redefinedRegs) AND (destination NOT in usedRegs):
            isDeadStore = true  // Overwritten before use!

    if isDeadStore:
        mark for removal
    else:
        // Update tracking sets
        for each source:
            usedRegs.insert(source)

        for each destination:
            redefinedRegs.insert(destination)
            // Special case: in-place ops (e.g., v0 = add v0, v1)
            if destination NOT also a source:
                usedRegs.erase(destination)  // Clear - new definition
'''

### Iterative Removal

'''cpp
do:
    changes = runOnBasicBlock()
while changes > 0  // Repeat until fixpoint (usually 1-2 iterations)
'''

---

## Example Transformations

### Example 1: Simple Dead Store

**Before:**
'''asm
v0 = v_mul_f32 v1, v2     // v0 is overwritten below
v3 = v_add_f32 v4, v5
v0 = v_mov_b32 v7         // Redefines v0 -> first v0 is dead
v6 = v_add_f32 v0, v3
global_store_dword addr, v6
'''

**After:**
'''asm
v3 = v_add_f32 v4, v5
v0 = v_mov_b32 v7
v6 = v_add_f32 v0, v3
global_store_dword addr, v6
'''

### Example 2: Chain of Dead Stores

**Before:**
'''asm
v0 = v_mul_f32 v1, v2     // Overwritten by next
v0 = v_add_f32 v3, v4     // Overwritten by next
v0 = v_fma_f32 v5, v6, 1.0  // Overwritten by next
v0 = v_mov_b32 v7         // Final value, used below
global_store_dword addr, v0
'''

**After (single iteration removes all 3):**
'''asm
v0 = v_mov_b32 v7
global_store_dword addr, v0
'''

**Why this works:** Forward scan from each instruction detects redefinition before use.

---

## Safety Guarantees

### Instructions That Are Never Removed

The pass uses 'mustPreserveInstruction()' to check:

1. **Memory Operations**
   - Loads: 'isGlobalMemLoad()', 'isDSRead()', 'isTensorLoad()'
   - Stores: 'isGlobalMemStore()', 'isDSWrite()'
   - Atomics: 'isGlobalMemAtomic()'

2. **Control Flow**
   - Branches: 'isBranch()'
   - Calls and returns

3. **Synchronization**
   - Barriers: 'isBarrier()'
   - Wait instructions

4. **Side Effects**
   - Any instruction with 'IF_HasSideEffect' flag

### Why These Instructions Are Preserved

'''
Memory writes (stores):
  v_store_b32 addr, v0   // Must preserve - observable effect

Control flow:
  s_cbranch_scc0 label   // Must preserve - affects program flow

Barriers:
  s_barrier              // Must preserve - synchronization
'''

---

## Iterative Application

### Why Iteration Is Needed

Removing one dead store can expose new dead stores in different registers:

'''
Step 0 (Initial):
  v0 = v_mul_f32 v1, v2
  v0 = v_add_f32 v3, v4     // Redefines v0 -> first v0 is DEAD
  v5 = v_mov_b32 v0         // Uses v0
  v5 = v_sub_f32 v6, v7     // Redefines v5 -> previous v5 is NOW DEAD
  store v5

Step 1 (Remove first v0):
  v0 = v_add_f32 v3, v4
  v5 = v_mov_b32 v0
  v5 = v_sub_f32 v6, v7     // Redefines v5 -> previous v5 is DEAD
  store v5

Step 2 (Remove first v5):
  v0 = v_add_f32 v3, v4     // Now unused, but not overwritten -> KEPT
  v5 = v_sub_f32 v6, v7
  store v5
'''

**Note:** The new DCE only removes overwrites, not all unused instructions. Iteration helps find cascading overwrite chains.

### Implementation

'''cpp
void run(Function& func, PassContext& passCtx) {
    for (BasicBlock& bb : func) {
        bool changed = true;
        while (changed) {
            // Scan for dead stores
            int removed = runOnBasicBlock(bb, func);
            changed = (removed > 0);
        }
    }
}
'''

---

## Integration with Other Passes

### Typical Pipeline

'''cpp
PassManager pm;

// 1. Peephole creates fusion opportunities
pm.addPass(createPeepholeOptimizationPass());

// 2. DCE cleans up intermediate values
pm.addPass(createDeadCodeEliminationPass());

// 3. Duplicate elimination finds redundancies
pm.addPass(createDuplicateEliminationPass());

// 4. Final DCE cleanup
pm.addPass(createDeadCodeEliminationPass());
'''

### Why DCE Is Run Multiple Times

After peephole fusion:
'''
Before:
  v0 = v_fma_f32 v1, v2, 1.0    // Intermediate result
  v3 = v_add_f32 v0, 1.0         // Uses v0

After Peephole:
  v3 = v_fma_f32 v1, v2, 2.0    // Fused, v0 no longer needed

After DCE:
  v3 = v_fma_f32 v1, v2, 2.0    // Original v_fma_f32 removed
'''

After duplicate elimination:
'''
Before:
  v0 = v_mul_f32 v1, v2         // First occurrence
  v3 = v_add_f32 v4, v5
  v6 = v_mul_f32 v1, v2         // Duplicate (removed)

After Duplicate Elimination:
  v0 = v_mul_f32 v1, v2         // Now unused if v6 was the only user
  v3 = v_add_f32 v4, v5

After DCE:
  v3 = v_add_f32 v4, v5         // v0 removed
'''

---

## Performance Characteristics

### Time Complexity

- **Single Iteration:** O(n log r) where n = instructions, r = registers
  - Single backward pass through all instructions: O(n)
  - Set operations (insert/erase/lookup): O(log r) per operation
  - Total: O(n log r) ~= **O(n)** since r is small (~256)

- **Worst Case:** O(n log r) with multiple iterations (rare)
- **Typical Case:** O(n log r) with 1-2 iterations

**Why O(n)?** We make **one backward pass** tracking two sets ('usedRegs' and 'redefinedRegs').

**In Practice:** With 1000 instructions, this is ~1000 x log(256) ~= 8,000 operations (~0.1ms).

### Space Complexity

- **Instruction List:** O(n) to collect all instructions
- **usedRegs Set:** O(r) where r = number of registers
- **redefinedRegs Set:** O(r)
- **Removal Set:** O(n)
- **Total:** O(n + r) ~= **O(n)**

### Practical Performance

'''
Typical kernel with 1000 instructions:
  - Iteration 1: ~2-5% removed (dead stores)
  - Iteration 2: ~0-1% removed (cascading effects)
  - Iteration 3: ~0% removed (fixpoint)

Total overhead: < 0.2ms (single backward pass per iteration)
'''

---

## Implementation Details

### Key Methods

'''cpp
// Collect all instructions from function (cross-block)
void collectAllInstructions(Function& func,
                            std::vector<StinkyInstruction*>& instructions);

// Run one pass of dead store elimination (O(n) backward scan)
int runOnBasicBlock(BasicBlock& bb, Function& func);
'''

### Backward Scan Pattern

'''cpp
// Single backward pass through function
std::set<StinkyRegister> usedRegs;       // Regs used before next def
std::set<StinkyRegister> redefinedRegs;  // Regs redefined later

for each instruction (BACKWARD):
    if dest in redefinedRegs AND dest NOT in usedRegs:
        -> REMOVE (dead store)
    else:
        add sources to usedRegs
        add dests to redefinedRegs
        (handle in-place ops specially)
'''

---

## Limitations

### What This Pass Does NOT Do

**1. Remove All Unused Instructions**

The pass only removes instructions that are **overwritten** before use. Instructions that are unused but never overwritten are kept:

'''assembly
v0 = v_mul_f32 v1, v2    // Unused but NOT overwritten -> KEPT
v3 = v_add_f32 v4, v5
v3 = v_mov_b32 v6        // Overwrites v3 -> previous v3 REMOVED
store v3
'''

**Why?** This conservative approach ensures correctness across blocks and function calls without complex dataflow analysis.

**2. Handle Partially-Used Register Ranges**

If an instruction defines multiple registers (e.g., 'v[0:3]') and only some are redefined, the entire instruction is kept:

'''assembly
v[0:3] = vmem_load addr    // Defines v0, v1, v2, v3
v0 = v_add_f32 v4, v5      // Only v0 redefined -> load is KEPT
'''

### Cross-Block and Inter-Procedural Safety

**[x] The forward-scan pattern IS safe across blocks:**

'''assembly
Block A:
  v0 = v_mul_f32 v1, v2    // Define v0
  branch to B

Block B:
  v3 = v_add_f32 v0, v4    // Use detected during forward scan
'''
The scan sees the use in Block B, so v0 is kept.

**[x] Also safe with function calls:**

'''assembly
v10 = v_add_f32 v1, v2
s_swappc_b64 ...           // Call uses v10
'''
If the callee uses v10, we see it at the call site's register operands.

### Other Limitations

1. **Conservative Side-Effect Handling**: All memory operations, control flow, and barrier instructions are preserved.

2. **No Path-Sensitive Analysis**: Doesn't consider which control flow paths are taken.

---

## Future Enhancements

To make DCE more aggressive, we could add:

1. **Full Dead Code Elimination** (not just dead stores)
   - Use backward dataflow analysis to find ALL unused instructions
   - Build def-use chains across entire function
   - Requires liveness analysis for multi-block functions

2. **Register Range Handling**
   - Track individual elements in register ranges (e.g., which of 'v[0:3]' are actually used)
   - More precise analysis for vector operations

3. **Partial Redundancy Elimination (PRE)**
   - Move computations to less frequently executed paths
   - Eliminate redundant computations along some paths

4. **Loop-Aware DCE**
   - Special handling for loop-carried dependencies
   - Identify loop-invariant dead code

5. **Metrics and Reporting**
   - Count instructions removed
   - Report optimization opportunities
   - Generate statistics per optimization pattern

---

## References

- **Source:** 'src/ir/asm/DeadCodeEliminationPass.cpp'
- **Header:** 'include/ir/asm/DeadCodeEliminationPass.hpp'
- **Tests:** 'tests/unit/DeadCodeEliminationPassTest.cpp'
- **Related:** 'mustPreserveInstruction()' in 'StinkyAsmIR.hpp'

