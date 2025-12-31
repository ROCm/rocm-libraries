# Duplicate Elimination Pass Design

## Overview

The Duplicate Elimination pass identifies and eliminates redundant computations (Common Subexpression Elimination). This optimization recognizes when the same computation is performed multiple times and reuses the first result.

**Key Principle:** If two instructions compute the same value using the same opcode and operands, eliminate the duplicate and reuse the original result.

---

## Architecture

'''
+---------------------+
|  Input IR           |  Function with duplicate computations
+----------+----------+
           |
           v
+---------------------+
|  Instruction        |  Build signatures (opcode + operands)
|  Signature          |
+----------+----------+
           |
           v
+---------------------+
|  Duplicate          |  Match signatures
|  Detection          |
+----------+----------+
           |
           v
+---------------------+
|  Operand            |  Verify operands not modified
|  Verification       |
+----------+----------+
           |
           v
+---------------------+
|  Replace Uses       |  Redirect to original result
+----------+----------+
           |
           v
+---------------------+
|  Remove             |  Delete duplicate instructions
|  Duplicates         |
+----------+----------+
           |
           v
+---------------------+
|  Optimized IR       |  Reduced redundancy
+---------------------+
'''

---

## Algorithm

### Phase 1: Build Instruction Signatures

'''cpp
struct InstructionSignature {
    uint32_t opcode;                    // Unified opcode
    std::vector<StinkyRegister> srcRegs; // Source operands

    bool operator==(const InstructionSignature& other) const {
        return opcode == other.opcode && srcRegs == other.srcRegs;
    }
};
'''

### Phase 2: Find Duplicates

'''cpp
signatureMap: Map<Signature, StinkyInstruction*>
duplicates: Map<StinkyInstruction*, StinkyInstruction*>

for each instruction:
    if mustPreserveInstruction(instruction):
        continue  // Skip side-effecting instructions

    signature = buildSignature(instruction)

    if signature in signatureMap:
        original = signatureMap[signature]
        if operandsUnmodified(original, instruction):
            duplicates[instruction] = original
    else:
        signatureMap[signature] = instruction
'''

### Phase 3: Operand Verification

'''cpp
bool areOperandsUnmodified(original, duplicate):
    origPos = getPosition(original)
    dupPos = getPosition(duplicate)

    for each sourceReg in duplicate.srcRegs:
        if sourceReg was redefined between origPos and dupPos:
            return false

    return true
'''

### Phase 4: Replace and Remove

'''cpp
for each (duplicate, original) in duplicates:
    dupDest = duplicate.destRegs[0]
    origDest = original.destRegs[0]

    // Replace all uses of dupDest with origDest
    replaceRegisterUses(dupDest, origDest)

    // Remove duplicate instruction
    remove(duplicate)
'''

---

## Example Transformations

### Example 1: Simple Duplicate

**Before:**
'''asm
v0 = v_mul_f32 v1, v2     // First occurrence
v3 = v_add_f32 v4, v5
v6 = v_mul_f32 v1, v2     // Duplicate: same opcode + operands
v7 = v_sub_f32 v6, v0
'''

**After:**
'''asm
v0 = v_mul_f32 v1, v2     // Original kept
v3 = v_add_f32 v4, v5
// v6 removed, uses replaced with v0
v7 = v_sub_f32 v0, v0
'''

### Example 2: Multiple Duplicates

**Before:**
'''asm
v0 = v_mul_f32 v1, v2     // First occurrence
v3 = v_mul_f32 v1, v2     // Duplicate 1
v5 = v_mul_f32 v1, v2     // Duplicate 2
v7 = v_add_f32 v0, v3
v9 = v_sub_f32 v5, v7
'''

**After:**
'''asm
v0 = v_mul_f32 v1, v2     // Original kept
// All duplicates removed, uses replaced with v0
v7 = v_add_f32 v0, v0
v9 = v_sub_f32 v0, v7
'''

### Example 3: Operand Modified (Not Duplicate)

**Before:**
'''asm
v0 = v_mul_f32 v1, v2     // Uses v1
v3 = v_add_f32 v4, v5
v1 = v_sub_f32 v6, v7     // v1 MODIFIED
v8 = v_mul_f32 v1, v2     // Different v1! Not a duplicate
'''

**After (No Change):**
'''asm
v0 = v_mul_f32 v1, v2
v3 = v_add_f32 v4, v5
v1 = v_sub_f32 v6, v7
v8 = v_mul_f32 v1, v2     // Kept - operand was modified
'''

---

## Safety Guarantees

### Instructions That Are Never Duplicated

The pass uses 'mustPreserveInstruction()' to skip:

1. **Memory Operations**
   - Loads may have different addresses
   - Stores have observable side effects
   - Atomics cannot be duplicated

2. **Control Flow**
   - Branches affect program flow
   - Cannot be considered redundant

3. **Side Effects**
   - Instructions with 'IF_HasSideEffect'

### Operand Modification Tracking

'''
Position Tracking:
  0: v_fma_f32 v0, v1, v2, 1.0   // First occurrence
  1: v_add_f32 v3, v4, v5
  2: v1 = v_sub_f32 v6, v7       // v1 MODIFIED at position 2
  3: v_fma_f32 v8, v1, v2, 1.0   // Not duplicate (v1 changed)

Check: Are operands from position 0 still valid at position 3?
  - v1 was redefined at position 2 (between 0 and 3)
  - Result: NOT a duplicate
'''

---

## Instruction Signature Matching

### Hash-Based Lookup

'''cpp
struct InstructionSignatureHash {
    size_t operator()(const InstructionSignature& sig) const {
        size_t hash = std::hash<uint32_t>{}(sig.opcode);
        for (const auto& reg : sig.srcRegs) {
            hash ^= reg.hash() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

std::unordered_map<InstructionSignature,
                   StinkyInstruction*,
                   InstructionSignatureHash> signatureMap;
'''

### What Makes Instructions Identical?

**Same:**
- Same unified opcode
- Same source registers (exact match)
- Same immediate constants

**Different:**
- Different opcode
- Different source registers
- Different register order (e.g., commutative operations treated separately)

'''
Identical:
  v0 = v_add_f32 v1, 2.0
  v3 = v_add_f32 v1, 2.0

Different (opcode):
  v0 = v_add_f32 v1, v2
  v3 = v_mul_f32 v1, v2

Different (operands):
  v0 = v_add_f32 v1, v2
  v3 = v_add_f32 v1, v4

Note: Commutative operations not currently optimized
  v0 = v_add_f32 v1, v2
  v3 = v_add_f32 v2, v1    // Could be same, but not detected
'''

---

## Integration with Dead Code Elimination

### Combined Effect

'''
Initial:
  v0 = v_mul_f32 v1, v2     // Will become unused
  v3 = v_add_f32 v4, v5
  v6 = v_mul_f32 v1, v2     // Duplicate
  v8 = v_fma_f32 v6, v7, 1.0

After Duplicate Elimination:
  v0 = v_mul_f32 v1, v2     // Now unused (was used by v6)
  v3 = v_add_f32 v4, v5
  // v6 removed, replaced with v0
  v8 = v_fma_f32 v0, v7, 1.0

After DCE:
  v3 = v_add_f32 v4, v5
  v8 = v_fma_f32 v0, v7, 1.0
'''

### Optimal Pass Order

'''cpp
PassManager pm;

// 1. Peephole fusion (may create duplicates)
pm.addPass(createPeepholeOptimizationPass());

// 2. Remove dead code from fusion
pm.addPass(createDeadCodeEliminationPass());

// 3. Eliminate duplicates
pm.addPass(createDuplicateEliminationPass());

// 4. Remove newly dead originals
pm.addPass(createDeadCodeEliminationPass());
'''

---

## Performance Characteristics

### Time Complexity

- **Signature Build:** O(n) where n = number of instructions
- **Hash Lookup:** O(1) average per instruction
- **Total:** O(n) average case

### Space Complexity

- **Signature Map:** O(n) worst case (all unique)
- **Duplicate Map:** O(n) worst case (all duplicates)
- **Total:** O(n)

### Practical Performance

'''
Typical kernel with 1000 instructions:
  - Duplicates found: ~5-15%
  - Hash collisions: < 1%
  - Processing time: < 1ms
'''

---

## Implementation Details

### Data Structures

'''cpp
class DuplicateAnalysis {
    // Track all instructions with positions
    std::vector<StinkyInstruction*> instructions;
    std::unordered_map<StinkyInstruction*, int> instPosition;

    // Track register redefinitions
    std::unordered_map<StinkyRegister, StinkyInstruction*> killedRegs;
};
'''

### Key Methods

'''cpp
// Analyze BasicBlock
void analyze(BasicBlock& bb);

// Find all duplicates
std::unordered_map<StinkyInstruction*, StinkyInstruction*>
findDuplicates();

// Check if operands unchanged
bool areOperandsUnmodified(StinkyInstruction* orig,
                          StinkyInstruction* dup) const;
'''

---

## Limitations

### 1. Local CSE Only

Operates within BasicBlocks. Does not track values across BasicBlock boundaries.

'''
BasicBlock A:
  v0 = v_mul_f32 v1, v2    // First occurrence
  branch to B

BasicBlock B:
  v3 = v_mul_f32 v1, v2    // Could reuse v0, but not detected
'''

### 2. Order-Sensitive

Does not recognize commutative operations:

'''
Not Detected as Duplicate:
  v0 = v_add_f32 v1, v2
  v3 = v_add_f32 v2, v1    // Same result, different order
'''

### 3. No Value Numbering

Does not recognize equivalent computations through different paths:

'''
Not Detected:
  v0 = v_add_f32 v1, 1.0
  v2 = v_add_f32 1.0, v1   // Logically same, syntactically different
'''

### 4. Single BasicBlock Scope

Cannot eliminate duplicates across function calls or complex control flow.

---

## Future Enhancements

1. **Global CSE**
   - Track values across BasicBlock boundaries
   - Use dominator tree for safe elimination
   - Handle loop-invariant computations

2. **Commutative Awareness**
   - Normalize commutative operations
   - Recognize 'v_add(a,b) == v_add(b,a)'

3. **Value Numbering**
   - Assign value numbers to expressions
   - Detect semantic equivalence
   - Handle algebraic identities

4. **Partial Redundancy Elimination**
   - Insert computations to eliminate partial redundancy
   - More aggressive optimization

5. **Profile-Guided CSE**
   - Use profiling data to guide elimination
   - Optimize for hot paths

---

## References

- **Source:** 'src/ir/asm/DuplicateEliminationPass.cpp'
- **Header:** 'include/ir/asm/DuplicateEliminationPass.hpp'
- **Tests:** 'tests/unit/DuplicateEliminationPassTest.cpp'
- **Related:** 'mustPreserveInstruction()' in 'StinkyAsmIR.hpp'
- **Related:** Dead Code Elimination pass

