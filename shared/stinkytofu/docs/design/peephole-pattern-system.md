# Peephole Pattern System Design

## Overview

The peephole optimization system transforms declarative patterns into efficient C++ matcher code at build time.

**Key Principle:** Pattern -> Generated Code -> Optimized IR

---

## Architecture

'''
+---------------------+
|  .pattern File      |  User writes declarative patterns
|  (Declarative)      |
+----------+----------+
           |
           | Build Time
           v
+---------------------+
|  TableGen           |  Parses and generates code
|  (Code Generator)   |
+----------+----------+
           |
           v
+---------------------+
|  .inc File          |  Generated C++ matcher code
|  (C++ Code)         |
+----------+----------+
           |
           | Compile Time
           v
+---------------------+
|  Peephole Pass      |  Generic pass applies patterns
|  (Never Changes)    |
+----------+----------+
           |
           | Runtime
           v
+---------------------+
|  Optimized IR       |  Transformed instructions
+---------------------+
'''

---

## Pattern -> Generated Code Transformation

### Example Pattern

'''
pattern AddFMAFusion_F32 {
  match {
    $fma = v_fma_f32 $result, $a, $b, $fma_const
    $add = v_add_f32 $dst, $add_const, $result
  }
  constraints {
    HasOneUse($result)
    IsConstant($fma_const)
    IsConstant($add_const)
  }
  rewrite {
    $new = AddConstants($fma_const, $add_const)
  }
}
'''

### Generated C++ Code

'''cpp
class AddFMAFusion_F32 : public PatternMatcher {
public:
    std::optional<RewriteResult> tryMatchAndRewrite(
        StinkyInstruction* inst,
        const MatchContext& context) override
    {
        // 1. Match root instruction (last in pattern)
        if (inst->getUnifiedOpcode() != GFX::v_add_f32)
            return std::nullopt;

        // 2. Extract operands
        StinkyRegister dst = inst->destRegs[0];
        StinkyRegister add_const = inst->srcRegs[0];
        StinkyRegister result = inst->srcRegs[1];

        // 3. Find defining instruction for $result
        auto defIt = context.defMap.find(result);
        if (defIt == context.defMap.end())
            return std::nullopt;
        StinkyInstruction* fma = defIt->second;

        // 4. Check FMA opcode
        if (fma->getUnifiedOpcode() != GFX::v_fma_f32)
            return std::nullopt;

        // 5. Extract FMA operands
        StinkyRegister a = fma->srcRegs[0];
        StinkyRegister b = fma->srcRegs[1];
        StinkyRegister fma_const = fma->srcRegs[2];

        // 6. Check constraints
        if (context.useCount.find(result)->second != 1)
            return std::nullopt;  // HasOneUse
        if (!isConstant(fma_const))
            return std::nullopt;  // IsConstant
        if (!isConstant(add_const))
            return std::nullopt;  // IsConstant

        // 7. Apply rewrite (fold constants)
        double new_const = AddConstants(
            getConstantValue(fma_const),
            getConstantValue(add_const)
        );

        // 8. Modify instructions in-place
        fma->destRegs[0] = dst;
        fma->srcRegs[2] = StinkyRegister(new_const);

        // 9. Return instructions to remove
        RewriteResult result;
        result.instructionsToRemove = {inst};
        result.applied = true;
        return result;
    }

    const char* getName() const override {
        return "AddFMAFusion_F32";
    }
};
'''

---

## Code Generation Breakdown

### 1. Pattern Structure -> Class Structure

'''
Pattern:                          Generated:
+--------------+                  +--------------------------+
| pattern Name |    =========>    | class Name : public      |
| {            |                  |       PatternMatcher {   |
|   match      |                  |   tryMatchAndRewrite()   |
|   constraints|                  |   getName()              |
|   rewrite    |                  | };                       |
| }            |                  +--------------------------+
+--------------+
'''

### 2. Match Section -> Matching Code

**Pattern:**
'''
match {
  $fma = v_fma_f32 $result, $a, $b, $c
  $add = v_add_f32 $dst, $const, $result
}
'''

**Generated Code:**
'''cpp
// Match root (last instruction)
if (inst->getUnifiedOpcode() != GFX::v_add_f32)
    return std::nullopt;

// Extract root operands
StinkyRegister dst = inst->destRegs[0];
StinkyRegister const = inst->srcRegs[0];
StinkyRegister result = inst->srcRegs[1];

// Find defining instruction
auto defIt = context.defMap.find(result);
if (defIt == context.defMap.end())
    return std::nullopt;
StinkyInstruction* fma = defIt->second;

// Check opcode
if (fma->getUnifiedOpcode() != GFX::v_fma_f32)
    return std::nullopt;

// Extract operands
StinkyRegister a = fma->srcRegs[0];
StinkyRegister b = fma->srcRegs[1];
StinkyRegister c = fma->srcRegs[2];
'''

### 3. Constraints -> Guard Code

| Pattern Constraint | Generated Code |
|-------------------|----------------|
| 'HasOneUse($reg)' | 'if (useCount.find($reg)->second != 1) return nullopt;' |
| 'IsConstant($reg)' | 'if (!isConstant($reg)) return nullopt;' |
| 'SameValue($r1, $r2)' | 'if (!($r1 == $r2)) return nullopt;' |
| 'DifferentValue($r1, $r2)' | 'if ($r1 == $r2) return nullopt;' |

**Example:**
'''
Pattern:                          Generated:
+------------------+              +--------------------------------+
| constraints {    |              | // Check constraints           |
|   HasOneUse($x)  |  =========>  | auto it = useCount.find($x);   |
|   IsConstant($c) |              | if (it->second != 1) return;   |
| }                |              | if (!isConstant($c)) return;   |
+------------------+              +--------------------------------+
'''

### 4. Rewrite -> Transformation Code

The codegen analyzes the rewrite block to generate instruction transformation code. The process is fully pattern-driven--no hardcoded pattern-specific logic exists in the generator.

#### Code Generation Process

**Input:** Rewrite block from pattern
'''
rewrite {
  $new_const = MulConstants($mul_const, $add_const)
  $new_fma = v_fma_f32 $mul_result, $mul_const, $a, $new_const
  replace $add with $new_fma
  remove $mul
}
'''

**Steps:**

1. **Analyze `replace` and `remove` statements** to determine:
   - Which instruction to modify in-place (`$add`)
   - Which instruction to remove (`$mul`)

2. **Parse CreateInst statement** (`$new_fma = v_fma_f32 ...`) to extract:
   - New instruction opcode (`v_fma_f32`)
   - Destination operand (`$mul_result`)
   - Source operands (`$mul_const`, `$a`, `$new_const`)

3. **Generate transformation code** that:
   - Changes the instruction opcode
   - Updates destination register (if present)
   - Resizes and updates source registers
   - Handles constant folding

#### Generated Transformation Code

**For the above pattern, generates:**
'''cpp
// 1. Fold constants
double new_const = MulConstants(
    getConstantValue(mul_const),
    getConstantValue(add_const)
);

// 2. Change opcode (ADD -> FMA)
const HwInstDesc* newDesc = getMCIDByUOp(GFX::v_fma_f32, context.arch);
add_inst->updateHwInstDesc(newDesc);

// 3. Update destination (if instruction has one)
if (!add_inst->destRegs.empty()) {
    add_inst->destRegs[0] = mul_result;
}

// 4. Resize srcRegs to accommodate new instruction's operands
size_t firstSrcIdx = add_inst->destRegs.empty() ? 0 : 1;
size_t numSrcRegs = 4 - firstSrcIdx;  // v_fma_f32 has 3 sources
if (add_inst->srcRegs.size() < numSrcRegs) {
    add_inst->srcRegs.resize(numSrcRegs);
}

// 5. Update source registers sequentially
size_t srcIdx = 0;
add_inst->srcRegs[srcIdx++] = mul_const;
add_inst->srcRegs[srcIdx++] = a;
add_inst->srcRegs[srcIdx++] = StinkyRegister(static_cast<double>(new_const));

// 6. Return instruction for removal
result.instructionsToRemove = {mul_inst};
'''

#### Key Design Features

**Opcode Transformation:**
- Patterns can transform instruction types (e.g., `v_add_f32` -> `v_fma_f32`)
- Uses `context.arch` to lookup new instruction descriptor
- Updates instruction metadata (issue/latency cycles)

**Dynamic Destination Handling:**
- Runtime check: `destRegs.empty()` determines if instruction writes to a register
- Supports both register-writing instructions (ALU) and non-writing instructions (stores, branches)

**Robust Source Register Updates:**
- `srcIdx` counter tracks which source slot to fill, independent of operand array structure
- Automatically resizes `srcRegs` if new instruction needs more operands
- Works for instructions with or without destinations

**Constant Folding:**
- Detects operands from builtin calls (`MulConstants`, `AddConstants`)
- Wraps folded constants in `StinkyRegister(static_cast<double>(...))` constructor

#### Complete Example: ADD+MUL Fusion

**Pattern:**
'''
peephole pattern ADDMULFusion_F32 {
  match {
    $add = v_add_f32 $add_result, $add_const, $a
    $mul = v_mul_f32 $mul_result, $mul_const, $add_result
  }
  constraints {
    HasOneUse($add_result)
    IsConstant($add_const)
    IsConstant($mul_const)
  }
  rewrite {
    $new_const = MulConstants($mul_const, $add_const)
    $new_fma = v_fma_f32 $mul_result, $mul_const, $a, $new_const
    replace $add with $new_fma
    remove $mul
  }
}
'''

**Generated Matcher:**
'''cpp
class ADDMULFusion_F32 : public PatternMatcher {
public:
    std::optional<RewriteResult> tryMatchAndRewrite(
        StinkyInstruction* inst,
        const MatchContext& context) override
    {
        // ... match and constraint checks ...

        // Fold constants
        double new_const = MulConstants(
            getConstantValue(mul_const),
            getConstantValue(add_const)
        );

        // Transform ADD into FMA
        const HwInstDesc* newDesc = getMCIDByUOp(GFX::v_fma_f32, context.arch);
        add_inst->updateHwInstDesc(newDesc);

        if (!add_inst->destRegs.empty()) {
            add_inst->destRegs[0] = mul_result;
        }

        size_t firstSrcIdx = add_inst->destRegs.empty() ? 0 : 1;
        size_t numSrcRegs = 4 - firstSrcIdx;
        if (add_inst->srcRegs.size() < numSrcRegs) {
            add_inst->srcRegs.resize(numSrcRegs);
        }

        size_t srcIdx = 0;
        add_inst->srcRegs[srcIdx++] = mul_const;
        add_inst->srcRegs[srcIdx++] = a;
        add_inst->srcRegs[srcIdx++] = StinkyRegister(static_cast<double>(new_const));

        result.instructionsToRemove = {mul_inst};
        return result;
    }
};
'''

**Result:**
- Input: `v_add_f32 v0, 2.0, v1` + `v_mul_f32 v2, 3.0, v0`
- Output: `v_fma_f32 v2, 3.0, v1, 6.0` (2 instructions -> 1, constant folded)

#### Benefits

- **Zero Hardcoding**: No pattern name checks or special cases in GenPatterns.cpp
- **Extensible**: Supports any instruction transformation declaratively
- **Safe**: Runtime checks prevent out-of-bounds access
- **Correct by Construction**: Operand mapping derived from pattern syntax
- **Architecture-Aware**: Properly handles per-architecture instruction descriptors

---

## Def-Use Analysis (Runtime Infrastructure)

### Position-Aware Definition Tracking

The peephole pass uses **position-aware def-use analysis** to handle complex scenarios like in-place operations and instruction ordering.

#### Problem: In-Place Operations

When a register is both source and destination, naive analysis fails:

'''
v_fma_f32 v0, v1, v2, 1.0   // defMap[v0] = fma_inst
v_add_f32 v0, 1.0, v0       // defMap[v0] = add_inst (OVERWRITES!)
'''

When matching the ADD, we need to find the FMA that defined v0, but 'defMap[v0]' returns the ADD itself!

#### Solution: Context-Aware Definition Lookup

The 'DefUseAnalysis' class tracks:
1. **All definitions** of each register (not just the last one)
2. **Instruction positions** for ordering
3. **Context-aware lookup** that only returns definitions BEFORE the current instruction

'''cpp
class DefUseAnalysis {
    // Track ALL definitions with positions
    std::map<StinkyRegister, std::vector<StinkyInstruction*>> defMap;
    std::unordered_map<StinkyInstruction*, int> instPosition;

    // Find definition that comes BEFORE currentInst
    StinkyInstruction* getDefiningInstBefore(
        const StinkyRegister& reg,
        StinkyInstruction* beforeInst) const;

    // Build context-aware defMap for specific instruction
    std::unordered_map<StinkyRegister, StinkyInstruction*>
    getDefMapBefore(StinkyInstruction* beforeInst) const;
};
'''

#### Example: In-Place Fusion

'''
Position 0: v_fma_f32 v0, v1, v2, 1.0   // Defines v0
Position 1: v_add_f32 v0, 1.0, v0       // Uses v0 (position 0), defines v0 (position 1)

When matching ADD at position 1:
  - getDefMapBefore(add_inst) returns defMap with v0 -> fma_inst
  - Pattern matches successfully!
  - Result: v_fma_f32 v0, v1, v2, 2.0
'''

#### Instruction Order Verification

Only definitions that come **before** the current instruction are visible:

'''
WRONG ORDER (correctly rejected):
Position 0: v_add_f32 v0, 1.0, v1       // Uses v1 (not defined yet!)
Position 1: v_fma_f32 v1, v2, v3, 1.0   // Defines v1

When matching ADD at position 0:
  - getDefMapBefore(add_inst) -> v1 not in defMap
  - Pattern fails to match (correct behavior)
'''

'''
CORRECT ORDER:
Position 0: v_fma_f32 v1, v2, v3, 1.0   // Defines v1
Position 1: v_add_f32 v0, 1.0, v1       // Uses v1 from position 0

When matching ADD at position 1:
  - getDefMapBefore(add_inst) -> v1 -> fma_inst
  - Pattern matches successfully!
'''

#### Iterative Pattern Application

Patterns are applied iteratively until no more matches are found:

'''cpp
void run(Function& func, PassContext& passCtx) {
    for (BasicBlock& bb : func) {
        bool changed = true;
        while (changed) {
            // Re-analyze after each fusion
            DefUseAnalysis defUse;
            defUse.analyze(bb);

            // Apply patterns
            int fusionsInBB = runOnBasicBlock(bb, defUse);
            changed = (fusionsInBB > 0);
        }
    }
}
'''

This handles cascading optimizations where one fusion creates new opportunities.

---

## Matching Strategy

### Bottom-Up Matching

Patterns match from **last instruction first** (bottom-up):

'''
Pattern Order:                    Matching Order:
+-----------------+              +-----------------+
| $fma = v_fma    |  +-----+     | $add = v_add    | <- Match first
| $add = v_add    |  |  2  |     | $fma = v_fma    | <- Match second
+-----------------+  |  1  |     +-----------------+
                     +-----+
'''

**Why?** The pass iterates instructions forward, so we match the last instruction in the pattern first, then walk backward through def-use chains.

### Def-Use Chain Walking

'''
IR:
+-----------------------------+
| v_fma_f32 v0, a, b, 1.0     | <- Defines v0
| ...                         |
| v_add_f32 v2, 1.0, v0       | <- Uses v0 (matches here)
+-----------------------------+
                |
                | defMap.find(v0)
                v
+-----------------------------+
| v_fma_f32 v0, a, b, 1.0     | <- Found!
+-----------------------------+
'''

---

## Constraint Implementation

### HasOneUse

**Purpose:** Ensure intermediate value can be safely eliminated.

**Pattern:**
'''
constraints {
  HasOneUse($tmp)
}
'''

**Generated:**
'''cpp
auto it = useCount.find($tmp);
if (it == useCount.end() || it->second != 1)
    return std::nullopt;
'''

**Data Flow:**
'''
DefUseAnalysis:
+--------------------------------+
| v_fma_f32 v0, ...              |
| v_add_f32 v1, ..., v0 <- use 1  |
| v_mul_f32 v2, ..., v0 <- use 2  |
+--------------------------------+
        |
        v
useCount[v0] = 2  <- HasOneUse fails!

+--------------------------------+
| v_fma_f32 v0, ...              |
| v_add_f32 v1, ..., v0 <- use 1  |
+--------------------------------+
        |
        v
useCount[v0] = 1  <- HasOneUse passes!
'''

### IsConstant

**Purpose:** Check if register holds a literal value.

**Pattern:**
'''
constraints {
  IsConstant($c)
}
'''

**Generated:**
'''cpp
if (!isConstant($c))
    return std::nullopt;

// Where isConstant is:
inline bool isConstant(const StinkyRegister& reg) {
    return reg.dataType == StinkyRegister::Type::LiteralDouble ||
           reg.dataType == StinkyRegister::Type::LiteralInt;
}
'''

---

## Optimization Pipeline

### Single Pattern Execution

'''
+------------------+
| Input IR         |
| v_fma v0, a,b,1  |
| v_add v2, 1, v0  |
+--------+---------+
         |
         v
+------------------+
| Pattern Match    |  tryMatchAndRewrite(v_add, ...)
| - v_add found    |
| - v_fma found    |
| - Constraints OK |
+--------+---------+
         |
         v
+------------------+
| Apply Rewrite    |  Fold: 1.0 + 1.0 = 2.0
| Modify v_fma     |  fma->destRegs[0] = v2
| Mark v_add dead  |  fma->srcRegs[2] = 2.0
+--------+---------+
         |
         v
+------------------+
| Output IR        |
| v_fma v2, a,b,2  |
+------------------+
'''

### Multi-Pattern Execution

'''
For each instruction:
    For each pattern (in order):
        if pattern.match(inst):
            pattern.rewrite(inst)
            break  <- First match wins
'''

---

## Pattern Registry

All patterns are automatically registered:

**Generated:**
'''cpp
inline std::vector<std::unique_ptr<PatternMatcher>>
createAllPatterns() {
    std::vector<std::unique_ptr<PatternMatcher>> patterns;
    patterns.push_back(std::make_unique<AddFMAFusion_F32>());
    patterns.push_back(std::make_unique<AddFMAFusion_F16>());
    // ... more patterns ...
    return patterns;
}
'''

**Used by pass:**
'''cpp
class PeepholeOptimizationPass {
    std::vector<std::unique_ptr<PatternMatcher>> patterns;

    void init() {
        patterns = patterns::createAllPatterns();
    }

    void run() {
        for (auto* inst : instructions) {
            for (auto& pattern : patterns) {
                if (auto result = pattern->tryMatchAndRewrite(inst, ctx)) {
                    removeDeadInstructions(result->instructionsToRemove);
                    break;
                }
            }
        }
    }
};
'''

---

## Complete Example: Add+FMA Fusion

### Input

'''
Pattern File:
+-----------------------------------------+
| pattern AddFMAFusion_F32 {              |
|   match {                               |
|     $fma = v_fma_f32 $r, $a, $b, $c1    |
|     $add = v_add_f32 $d, $c2, $r        |
|   }                                     |
|   constraints {                         |
|     HasOneUse($r)                       |
|     IsConstant($c1)                     |
|     IsConstant($c2)                     |
|   }                                     |
|   rewrite {                             |
|     $new = AddConstants($c1, $c2)       |
|   }                                     |
| }                                       |
+-----------------------------------------+
'''

### Generated Code (Simplified)

'''
Generated C++:
+--------------------------------------------------+
| class AddFMAFusion_F32 : public PatternMatcher { |
|   std::optional<RewriteResult>                   |
|   tryMatchAndRewrite(inst, ctx) {                |
|                                                  |
|     // 1. Match v_add_f32 (root)                |
|     if (opcode != v_add_f32) return nullopt;    |
|                                                  |
|     // 2. Extract operands                      |
|     dst = inst->destRegs[0];                    |
|     c2 = inst->srcRegs[0];                      |
|     r = inst->srcRegs[1];                       |
|                                                  |
|     // 3. Find FMA defining r                   |
|     fma = ctx.defMap[r];                        |
|     if (fma->opcode != v_fma_f32) return null;  |
|                                                  |
|     // 4. Extract FMA operands                  |
|     a = fma->srcRegs[0];                        |
|     b = fma->srcRegs[1];                        |
|     c1 = fma->srcRegs[2];                       |
|                                                  |
|     // 5. Check constraints                     |
|     if (useCount[r] != 1) return null;          |
|     if (!isConstant(c1)) return null;           |
|     if (!isConstant(c2)) return null;           |
|                                                  |
|     // 6. Fold constants                        |
|     new_const = c1 + c2;                        |
|                                                  |
|     // 7. Rewrite in-place                      |
|     fma->destRegs[0] = dst;                     |
|     fma->srcRegs[2] = new_const;                |
|                                                  |
|     // 8. Mark ADD for removal                  |
|     return {.instructionsToRemove = {inst}};    |
|   }                                              |
| };                                               |
+--------------------------------------------------+
'''

### IR Transformation

'''
Before:                           After:
+----------------------+          +----------------------+
| v_fma_f32 v0, a,b,1  |          | v_fma_f32 v2, a,b,2  |
| v_add_f32 v2, 1, v0  |          +----------------------+
+----------------------+                    |
         |                                  |
         |                                  v
         v                          +--------------+
+----------------+                  | 2 -> 1        |
| Pattern Match  |                  | instructions |
| - Constraints  |                  | v0 freed     |
| - Fold 1+1=2   |                  +--------------+
+----------------+
'''

---

## Uniform Pattern Interface

All generated patterns share the same interface:

'''
+-------------------------------------+
|         PatternMatcher              | <- Base class
|  (uniform interface for all)        |
+-------------------------------------+
| + tryMatchAndRewrite(inst, ctx)     |
| + getName() -> string                |
+------------+------------------------+
             |
             | inherits
             |
    +--------+--------+------------+
    |                 |            |
+---v-----------+ +--v--------+ +-v----------+
| AddFMAFusion  | | MulFusion | | YourPattern|
|     _F32      | |   _F32    | |            |
+---------------+ +-----------+ +------------+
'''

**Benefits:**
- Pass treats all patterns uniformly
- Adding patterns requires zero code changes to pass
- Patterns can be reordered, enabled/disabled easily

---

## Performance Characteristics

### Time Complexity

- **Pattern Matching:** O(k) per instruction, where k = pattern size (typically 2-3)
- **Def-Use Lookup:** O(1) with hash maps
- **Overall:** O(n x p x k) where:
  - n = number of instructions
  - p = number of patterns (typically < 10)
  - k = pattern size (typically 2-3)

### Space Complexity

- **Def-Use Maps:** O(n) space
- **Pattern Matchers:** O(p) space (pattern count)
- **Overhead:** Minimal (shared pointers)

### Optimization

The generated code is highly optimized:
- Early exit on opcode mismatch
- Guard clauses avoid deep computation
- In-place modification (no allocation)
- Constant folding at compile time when possible

---

## Limitations

See [Known Issues](../known-issues.md) for cross-block use-def and other limitations.

### Current Limitations

1. **Pattern Scope:** Limited to 2-3 instructions
   - Good for: Local fusions, constant folding
   - Not for: Global optimizations spanning many instructions

2. **Rewrite Flexibility:** Currently automatic for fusion patterns
   - Good for: Add+FMA, Mul+Add style fusions
   - Extensible: Can add more rewrite logic to generator

3. **Constraint Types:** Fixed set of constraints
   - Available: HasOneUse, IsConstant, SameValue, DifferentValue
   - Extensible: Can add more constraint types

### Future Extensions

**Can be added:**
- Distance constraints (e.g., within N instructions)
- Register class constraints
- Architecture-specific patterns
- Cost model integration

---

## Design Principles

### 1. Declarative Over Imperative

**User writes:** What to match and transform
**System generates:** How to match and transform efficiently

### 2. Separation of Concerns

**Pattern File:** Optimization knowledge
**Generator:** Code generation logic
**Pass:** Generic execution framework

### 3. Zero Manual Registration

Patterns are automatically:
- Parsed at build time
- Generated into C++
- Registered in pattern registry
- Applied by pass

### 4. Type Safety

Generated C++ code is:
- Strongly typed
- Compiler-checked
- Optimized by C++ compiler

---

## Summary

### What You Write

'''
pattern Name {
  match { ... }
  constraints { ... }
  rewrite { ... }
}
'''

### What You Get

'''cpp
class Name : public PatternMatcher {
    std::optional<RewriteResult> tryMatchAndRewrite(...) {
        // Efficient matching code
        // Constraint checking
        // In-place rewriting
        // Returns dead instructions
    }
};
'''

### How It Works

'''
.pattern file -> TableGen -> .inc file -> Compiled -> Pass applies patterns
'''

### Key Benefits

- **Declarative:** Write patterns, not C++ code
- **Fast:** Generated code is optimized
- **Safe:** Type-checked, compiler-verified
- **Automatic:** Zero manual registration
- **Extensible:** Add patterns without touching pass

The system transforms high-level pattern descriptions into efficient, type-safe C++ code automatically!

---

## Def-Use Analysis for Register Reuse

### Background

The peephole optimization pass uses def-use analysis to determine if it's safe to fuse instructions. A key constraint is 'HasOneUse(register)', which checks if a register value has exactly one use before fusion.

### Problem: Register Reuse

In non-SSA IR with register reuse, the same physical register can be redefined multiple times:

'''asm
v10 = v_fma_f32 v1, v2, 1.0      # def #1 at position 9
v10 = v_add_f32 1.0, v10         # def #2 at position 10, uses def #1
v10 = v_mul_f32 v0, v10          # def #3 at position 11, uses def #2
v1  = v_mul_f32 0.5, v10         # position 12, uses def #3
'''

**Naive approach:** Count all uses of register 'v10' -> finds 3 uses (positions 10, 11, 12)
**Problem:** This counts uses of ALL definitions, not just def #1!

**Correct approach:** Count only uses of def #1 -> finds 1 use (position 10)

### Solution: Per-Definition Use Counting

The 'getUseCountForDef()' method implements **live range-aware use counting**:

'''cpp
int getUseCountForDef(StinkyInstruction* defInst, const StinkyRegister& reg) const {
    // 1. Find position of this definition
    int defPos = instPosition[defInst];

    // 2. Find next redefinition of same register
    int nextDefPos = INT_MAX;
    for (auto* laterDefInst : defMap[reg]) {
        int laterPos = instPosition[laterDefInst];
        if (laterPos > defPos && laterPos < nextDefPos) {
            nextDefPos = laterPos;
        }
    }

    // 3. Count uses in live range [defPos, nextDefPos]
    int count = 0;
    for (auto* useInst : useMap[reg]) {
        int usePos = instPosition[useInst];
        // Include redefining instruction (in-place operations)
        if (usePos > defPos && usePos <= nextDefPos) {
            count++;
        }
    }
    return count;
}
'''

### Key Design Decisions

#### 1. Live Range Termination

A register's live range ends at the **next redefinition**:
- Position 9: 'v10' def #1 starts
- Position 10: 'v10' def #2 starts -> def #1's live range ends
- Uses at positions 11+ don't count for def #1

#### 2. In-Place Operations

The condition 'usePos <= nextDefPos' (not '<') handles in-place operations:

'''asm
v10 = v_fma_f32 ...     # def #1 at pos 9
v10 = v_add_f32 1.0, v10  # pos 10: uses def #1, then redefines
'''

Position 10 is BOTH:
- A use of def #1 (reads 'v10')
- The next redefinition (writes 'v10')

The '<=' ensures this use is counted.

### Impact

**Before Fix:**
- GELU optimization: 13 -> 13 instructions (no change)
- Pattern rejected due to 'HasOneUse' failing (counted 3 uses instead of 1)

**After Fix:**
- GELU optimization: 13 -> 12 instructions (1 instruction eliminated)
- FMA+ADD fusion works correctly with register reuse
- Pattern correctly identifies single use

### Example: GELU Optimization

'''asm
# Before optimization
9:  v10 = v_fma_f32 -2.0, v10, 1.0    # def #1
10: v10 = v_add_f32 1.0, v10          # def #2, uses def #1
11: v10 = v_mul_f32 v0, v10           # def #3, uses def #2
12: v1  = v_mul_f32 0.5, v10          # uses def #3

# After optimization (FMA+ADD fused)
9:  v10 = v_fma_f32 -2.0, v10, 2.0    # Constants folded: 1.0 + 1.0 = 2.0
11: v10 = v_mul_f32 v0, v10           # Now uses fused result
12: v1  = v_mul_f32 0.5, v10
'''

### Testing

See 'tests/unit/PeepholeOptimizationPassTest.cpp':
- 'Fusion_WithRegisterReuse': Verifies fusion works with register reuse
- 'NoFusion_RegisterReuseWithMultipleUsesBeforeRedef': Verifies fusion correctly prevented when multiple uses exist

### Related Commits

- **Fix:** commit '38b661d4fb' - Implements 'getUseCountForDef()'
- **Tests:** commit '087930d10a' - Adds comprehensive test coverage

