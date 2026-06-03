# How to Add Peephole Optimization Patterns

This guide shows you how to add new optimization patterns to the StinkyTofu compiler.

## Quick Start

**Three steps:**
1. Write pattern in '.pattern' file
2. Run 'make'
3. Done!

## Example: Adding a Multiply-Add Fusion

### Step 1: Write the Pattern

Edit 'src/transforms/asm/PeepholePatterns.pattern':

'''
//===----------------------------------------------------------------------===//
// Pattern: Multiply-Add Fusion (F32)
//
// Transforms:
//   v_mul_f32 v0, a, b
//   v_add_f32 v2, v0, c
// Into:
//   v_fma_f32 v2, a, b, c
//===----------------------------------------------------------------------===//

pattern MulAddFusion_F32 {
  match {
    $mul = v_mul_f32 $tmp, $a, $b
    $add = v_add_f32 $dst, $tmp, $c
  }

  constraints {
    HasOneUse($tmp)
  }

  rewrite {
    $new_const = AddConstants($fma_const, $add_const)
    // Pattern system automatically handles the transformation
  }
}
'''

### Step 2: Build

'''bash
cd build
make -j$(nproc)
'''

**What happens:**
- [x] Pattern is parsed
- [x] C++ matcher code is generated
- [x] Pattern is automatically registered
- [x] Pass will try your pattern on every instruction

### Step 3: Test

Your pattern is now active! It will match and transform automatically.

'''
[PeepholePass] Loaded 3 pattern(s)
  [Peephole] Applied MulAddFusion_F32
'''

---

## Pattern Syntax

### Basic Structure

'''
pattern PatternName {
  match {
    // Instructions to match
  }

  constraints {
    // Conditions that must be true
  }

  rewrite {
    // How to transform (currently automatic)
  }
}
'''

### Match Section

**Syntax:**
'''
$instruction_var = opcode $dest, $src1, $src2, ...
'''

**Rules:**
- First operand is **always** the destination register
- Pattern variables start with '$'
- Instructions are matched in temporal order (first executed first)

**Example:**
'''
match {
  // First: FMA instruction
  $fma = v_fma_f32 $result, $a, $b, $const1

  // Second: ADD instruction using FMA result
  $add = v_add_f32 $output, $const2, $result
}
'''

### Constraints Section

Available constraints:

| Constraint | Meaning | Example |
|------------|---------|---------|
| 'HasOneUse($reg)' | Register used exactly once | 'HasOneUse($tmp)' |
| 'IsConstant($reg)' | Register is a literal | 'IsConstant($const)' |
| 'SameValue($r1, $r2)' | Registers have same value | 'SameValue($a, $b)' |
| 'DifferentValue($r1, $r2)' | Registers differ | 'DifferentValue($x, $y)' |

**Example:**
'''
constraints {
  HasOneUse($tmp)      // $tmp only used by next instruction
  IsConstant($c1)      // $c1 must be a literal value
  IsConstant($c2)      // $c2 must be a literal value
}
'''

### Rewrite Section

Currently, rewrites are **automatic** for fusion patterns:
- Constants are folded
- Instructions are modified in-place
- Dead instructions are removed

**Built-in constant folding:**
'''
rewrite {
  $new = AddConstants($c1, $c2)    // c1 + c2
  $new = SubConstants($c1, $c2)    // c1 - c2
  $new = MulConstants($c1, $c2)    // c1 * c2
  $new = NegateConstant($c)        // -c
}
'''

---

## Complete Examples

### Example 1: Add+FMA Fusion (F32)

**Before:**
'''asm
v_fma_f32 v0, a, b, 1.0
v_add_f32 v2, 1.0, v0
'''

**After:**
'''asm
v_fma_f32 v2, a, b, 2.0    # v0 eliminated!
'''

**Pattern:**
'''
pattern AddFMAFusion_F32 {
  match {
    $fma = v_fma_f32 $fma_result, $a, $b, $fma_const
    $add = v_add_f32 $add_dst, $add_const, $fma_result
  }

  constraints {
    HasOneUse($fma_result)
    IsConstant($fma_const)
    IsConstant($add_const)
  }

  rewrite {
    $new_const = AddConstants($fma_const, $add_const)
  }
}
'''

### Example 2: Add+FMA Fusion (F16)

**Before:**
'''asm
v_fma_f16 v0, a, b, 0.5
v_add_f16 v1, 0.5, v0
'''

**After:**
'''asm
v_fma_f16 v1, a, b, 1.0
'''

**Pattern:**
'''
pattern AddFMAFusion_F16 {
  match {
    $fma = v_fma_f16 $fma_result, $a, $b, $fma_const
    $add = v_add_f16 $add_dst, $add_const, $fma_result
  }

  constraints {
    HasOneUse($fma_result)
    IsConstant($fma_const)
    IsConstant($add_const)
  }

  rewrite {
    $new_const = AddConstants($fma_const, $add_const)
  }
}
'''

---

## Tips & Best Practices

### 1. Always Use 'HasOneUse' for Intermediate Values

'''
[x] Good:
constraints {
  HasOneUse($tmp)    // Safe to eliminate $tmp
}

[ ] Bad:
// Missing HasOneUse - might eliminate value that's used elsewhere!
'''

### 2. Check for Constants When Folding

'''
[x] Good:
constraints {
  IsConstant($c1)
  IsConstant($c2)
}
rewrite {
  $result = AddConstants($c1, $c2)
}

[ ] Bad:
// Missing IsConstant - can't fold non-constant values!
'''

### 3. Name Variables Descriptively

'''
[x] Good:
$fma_inst = v_fma_f32 $fma_result, $a, $b, $const

[ ] Bad:
$i1 = v_fma_f32 $r1, $r2, $r3, $r4
'''

### 4. Add Comments Explaining the Optimization

'''
//===----------------------------------------------------------------------===//
// Pattern: NegateNegate
//
// Eliminates double negation: -(-x) => x
// Benefit: Eliminates 2 instructions, frees registers
//===----------------------------------------------------------------------===//
'''

### 5. Order Patterns by Specificity

More specific patterns should come before general ones:
'''
// Specific: mul by constant 1.0
pattern MulByOne { ... }

// General: any multiplication
pattern MulFusion { ... }
'''

### 6. Patterns Work Across Non-Consecutive Instructions

The pattern system uses **def-use analysis**, so patterns don't require strict consecutiveness:

'''
[x] This works fine:
  v_fma_f32 v0, v1, v2, 1.0
  v_mul_f32 v5, v6, v7         // Intervening instruction
  v_add_f32 v0, 1.0, v0        // Still matches and fuses!
'''

The pattern matcher automatically verifies that:
- Dependencies are correct (FMA comes before ADD)
- Intermediate registers haven't been overwritten

'''
[ ] This correctly doesn't match:
  v_fma_f32 v0, v1, v2, 1.0
  v_mul_f32 v0, v6, v7         // Overwrites v0!
  v_add_f32 v3, 1.0, v0        // Uses v0 from v_mul, not v_fma
'''

### 7. In-Place Operations Are Handled Correctly

Patterns work even when destination == source:

'''
[x] In-place fusion works:
  v_fma_f32 v0, v1, v2, 1.0    // Defines v0
  v_add_f32 v0, 1.0, v0        // Uses v0, writes v0
  => v_fma_f32 v0, v1, v2, 2.0 // Correctly fused!
'''

The pattern system uses **position-aware def-use analysis** to find the correct previous definition.

---

## Testing Your Pattern

### 1. Write Test IR

Create test input:
'''asm
v_fma_f32 v0, a, b, 1.0
v_add_f32 v2, 1.0, v0
'''

### 2. Run the Pass

'''cpp
auto pass = createPeepholeOptimizationPass();
pass->run(module);
'''

### 3. Check Output

Expected output:
'''asm
v_fma_f32 v2, a, b, 2.0
'''

Console should show:
'''
[PeepholePass] Loaded N pattern(s)
  [Peephole] Applied YourPatternName
'''

---

## Troubleshooting

### Pattern Not Matching?

**Check:**
1. [x] Instruction opcodes are correct (e.g., 'v_fma_f32', not 'v_fma')
2. [x] Operand order is correct (destination first)
3. [x] Constraints are satisfied
4. [x] Instructions appear in the right order

**Debug:**
'''bash
# Check generated code
cat build/PeepholePatterns.inc | grep "YourPatternName"

# Verbose build
make VERBOSE=1 | grep pattern
'''

### Build Errors?

**Common issues:**
- Syntax error in '.pattern' file -> Check pattern grammar
- Undefined opcode -> Verify opcode exists in GFX ISA
- Missing constraint -> Add required constraints

**Fix:**
'''bash
# Clean rebuild
cd build
rm PeepholePatterns.inc
make tablegen
make
'''

---

## What's Automatic

You **don't** need to:
- [ ] Write C++ matcher code
- [ ] Write C++ rewrite code
- [ ] Register patterns manually
- [ ] Update the pass
- [ ] Handle def-use analysis
- [ ] Remove dead instructions

The system handles all of this automatically! [x]

---

## What You Control

You **do** control:
- [x] Which instructions to match
- [x] What constraints to check
- [x] How constants are folded
- [x] Pattern priority (order in file)

---

## Advanced: Pattern Variants

### Same Pattern for Multiple Types

Create separate patterns for each type:
'''
pattern AddFMAFusion_F32 { ... v_fma_f32 ... }
pattern AddFMAFusion_F16 { ... v_fma_f16 ... }
pattern AddFMAFusion_F64 { ... v_fma_f64 ... }
'''

### Pattern Families

Group related patterns:
'''
//===----------------------------------------------------------------------===//
// FMA Fusion Family
//===----------------------------------------------------------------------===//

pattern AddFMAFusion_F32 { ... }
pattern MulAddFusion_F32 { ... }
pattern SubFMAFusion_F32 { ... }
'''

---

## File Locations

| File | Purpose |
|------|---------|
| 'src/transforms/asm/PeepholePatterns.pattern' | **Your patterns go here** |
| 'build/PeepholePatterns.inc' | Generated C++ code (auto) |
| 'src/transforms/asm/PeepholeOptimizationPass.cpp' | Generic pass (never edit) |
| 'tools/tablegen/GenPatterns.cpp' | Code generator (rarely edit) |

---

## Summary

**To add a pattern:**
1. Edit 'PeepholePatterns.pattern'
2. Run 'make'
3. Done!

**Pattern structure:**
'''
pattern Name {
  match { ... }       // Instructions to find
  constraints { ... } // Conditions to check
  rewrite { ... }     // How to transform
}
'''

**The system automatically:**
- Parses your pattern
- Generates efficient C++ code
- Registers your pattern
- Applies it during compilation

**You just write the pattern!** !

---

## Next Steps

- See [Architecture Overview](architecture.md) for system architecture
- See [Pattern Grammar Reference](pattern-grammar.md) for complete syntax reference
- See 'src/transforms/asm/PeepholePatterns.pattern' for real examples

---

## Questions?

Common patterns are already implemented. Just copy and modify them for your needs!
