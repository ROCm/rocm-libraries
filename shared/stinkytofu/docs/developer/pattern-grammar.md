# Pattern Grammar Reference

Complete syntax reference for writing peephole optimization patterns.

## Quick Reference

### Pattern Structure

'''
pattern PatternName {
  match {
    // Instructions to match
  }
  constraints {
    // Conditions to check
  }
  rewrite {
    // How to transform
  }
}
'''

### Variable Syntax

- '$variable' - Pattern variable (starts with '$')
- '$inst' - Instruction variable (represents the instruction itself)
- '$reg' - Register/operand variable

---

## Match Block

### Syntax

'''
match {
  $inst_var = opcode $dest, $src1, $src2, ...
}
'''

### Rules

1. **First operand is destination** (always)
2. **Variables start with '$'**
3. **Instructions listed in temporal order** (first executed -> last executed)

### Examples

#### Single Instruction Match

'''
match {
  $mov = v_mov_b32 $dst, $src
}
'''

#### Two Instruction Match (Data Flow)

'''
match {
  $mul = v_mul_f32 $tmp, $a, $b        // First: multiply
  $add = v_add_f32 $dst, $tmp, $c      // Second: add (uses $tmp)
}
'''

#### With Constants

'''
match {
  $fma = v_fma_f32 $result, $a, $b, $const   // $const can be literal
  $add = v_add_f32 $dst, 1.0, $result        // Or hardcoded value
}
'''

### Variable Types

| Variable | Represents | Example |
|----------|------------|---------|
| '$inst' | The instruction itself | '$fma', '$add_inst' |
| '$reg' | Register or operand | '$tmp', '$result', '$a' |
| '$const' | Constant operand | '$c1', '$fma_const' |

**Note:** All variables are just names - the system figures out what they represent from context.

---

## Constraints Block

### Syntax

'''
constraints {
  ConstraintName($arg1)
  ConstraintName($arg1, $arg2)
}
'''

### Available Constraints

#### 'HasOneUse($register)'

Check if register has exactly one use.

**Use case:** Safe to eliminate intermediate result.

'''
constraints {
  HasOneUse($tmp)    // $tmp only used once
}
'''

**Example:**
'''
match {
  $mul = v_mul_f32 $tmp, $a, $b
  $add = v_add_f32 $dst, $tmp, $c
}
constraints {
  HasOneUse($tmp)    // $tmp only used by $add
}
'''

#### 'IsConstant($operand)'

Check if operand is a constant (not a register).

**Use case:** Required for constant folding.

'''
constraints {
  IsConstant($c1)    // Must be literal value
}
'''

**Example:**
'''
match {
  $fma = v_fma_f32 $result, $a, $b, $c
}
constraints {
  IsConstant($c)     // Ensure $c is a constant
}
'''

#### 'SameValue($reg1, $reg2)'

Check if two operands are the same register.

**Use case:** Detect in-place operations.

'''
constraints {
  SameValue($dst, $src)    // In-place: dst = dst + x
}
'''

#### 'DifferentValue($reg1, $reg2)'

Check if two operands are different registers.

**Use case:** Avoid incorrect transformations.

'''
constraints {
  DifferentValue($a, $b)   // Ensure different regs
}
'''

### Multiple Constraints

'''
constraints {
  HasOneUse($tmp)
  IsConstant($c1)
  IsConstant($c2)
  DifferentValue($a, $b)
}
'''

---

## Rewrite Block

### Constant Folding

#### 'AddConstants($c1, $c2)'

Add two constants.

'''
rewrite {
  $sum = AddConstants($c1, $c2)
}
'''

**Example:**
'''
$c1 = 1.0, $c2 = 1.0  ->  $sum = 2.0
'''

#### 'SubConstants($c1, $c2)'

Subtract constants: 'c1 - c2'

'''
rewrite {
  $diff = SubConstants($c1, $c2)
}
'''

#### 'MulConstants($c1, $c2)'

Multiply constants.

'''
rewrite {
  $product = MulConstants($c1, $c2)
}
'''

#### 'NegateConstant($c)'

Negate constant: '-c'

'''
rewrite {
  $neg = NegateConstant($c)
}
'''

### Current Behavior

**Currently, rewrites are automatic** for fusion patterns:
- Constants are folded
- Instructions modified in-place
- Dead instructions removed

The rewrite block mainly specifies **which constants to fold**.

---

## Complete Examples

### Example 1: Add+FMA Fusion

**Transform:**
'''asm
v_fma_f32 v0, a, b, 1.0
v_add_f32 v2, 1.0, v0
=>
v_fma_f32 v2, a, b, 2.0
'''

**Pattern:**
'''
pattern AddFMAFusion_F32 {
  match {
    $fma = v_fma_f32 $fma_result, $a, $b, $fma_const
    $add = v_add_f32 $add_dst, $add_const, $fma_result
  }

  constraints {
    HasOneUse($fma_result)    // Safe to eliminate
    IsConstant($fma_const)    // Can fold
    IsConstant($add_const)    // Can fold
  }

  rewrite {
    $new_const = AddConstants($fma_const, $add_const)
  }
}
'''

### Example 2: Multiply-Add Fusion

**Transform:**
'''asm
v_mul_f32 v0, a, b
v_add_f32 v2, v0, c
=>
v_fma_f32 v2, a, b, c
'''

**Pattern:**
'''
pattern MulAddFusion_F32 {
  match {
    $mul = v_mul_f32 $tmp, $a, $b
    $add = v_add_f32 $dst, $tmp, $c
  }

  constraints {
    HasOneUse($tmp)
  }

  rewrite {
    // Automatically transforms to FMA
  }
}
'''

### Example 3: Negate-Negate Elimination

**Transform:**
'''asm
v_mul_f32 v0, a, -1.0      // Negate
v_mul_f32 v1, v0, -1.0     // Negate again
=>
v_mov_b32 v1, a            // Just copy
'''

**Pattern:**
'''
pattern NegateNegate_F32 {
  match {
    $neg1 = v_mul_f32 $tmp, $a, -1.0
    $neg2 = v_mul_f32 $dst, $tmp, -1.0
  }

  constraints {
    HasOneUse($tmp)
  }

  rewrite {
    // Replaces with move
  }
}
'''

---

## Common Patterns

### Pattern: In-Place Operation

'''
match {
  $op = v_add_f32 $dst, $dst, $val   // dst = dst + val
}
constraints {
  SameValue($dst, $dst)  // Verify in-place
}
'''

### Pattern: Constant Propagation

'''
match {
  $mul = v_mul_f32 $dst, $src, 1.0   // Multiply by 1
}
constraints {
  IsConstant(1.0)
}
rewrite {
  // Replaces with move: dst = src
}
'''

### Pattern: Zero Elimination

'''
match {
  $mul = v_mul_f32 $dst, $src, 0.0   // Multiply by 0
}
constraints {
  IsConstant(0.0)
}
rewrite {
  // Replaces with: dst = 0
}
'''

---

## Operand Order

### Destination First, Always

'''
[x] Correct:
$inst = v_add_f32 $dst, $src1, $src2
                  ^^^^^
                  destination first

[ ] Wrong:
$inst = v_add_f32 $src1, $src2, $dst
'''

### Match Instruction Signature

For 'v_fma_f32 dest, src0, src1, src2':

'''
[x] Correct:
$fma = v_fma_f32 $dst, $a, $b, $c
                 ^^^^  ^^  ^^  ^^
                 dest  src0 src1 src2

[ ] Wrong:
$fma = v_fma_f32 $a, $b, $c, $dst
'''

---

## Temporal Order

Instructions must be listed in **execution order**:

'''
[x] Correct (top-to-bottom = first-to-last):
match {
  $mul = v_mul_f32 $tmp, $a, $b      // Executes first
  $add = v_add_f32 $dst, $tmp, $c    // Executes second
}

[ ] Wrong (reversed order):
match {
  $add = v_add_f32 $dst, $tmp, $c    // Can't use $tmp before it's defined!
  $mul = v_mul_f32 $tmp, $a, $b
}
'''

---

## Naming Conventions

### Instruction Variables

Use descriptive names:

'''
[x] Good:
$fma_inst, $add_inst, $mul, $store

[ ] Bad:
$i1, $i2, $x, $temp
'''

### Operand Variables

Name by role or purpose:

'''
[x] Good:
$result, $tmp, $intermediate, $fma_const, $add_const

[ ] Bad:
$r1, $r2, $x, $y
'''

### Example with Good Names

'''
pattern AddFMAFusion_F32 {
  match {
    $fma_inst = v_fma_f32 $fma_result, $a, $b, $fma_const
    $add_inst = v_add_f32 $add_dst, $add_const, $fma_result
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

## Comments

### Use Comments Liberally

'''
pattern Example {
  match {
    // First: FMA computes a*b + c
    $fma = v_fma_f32 $result, $a, $b, $c

    // Second: ADD adds constant to FMA result
    $add = v_add_f32 $dst, $const, $result
  }

  constraints {
    HasOneUse($result)    // Safe to eliminate $result
    IsConstant($c)        // Required for folding
    IsConstant($const)    // Required for folding
  }

  rewrite {
    // Fold constants: c + const
    $new_const = AddConstants($c, $const)
  }
}
'''

### Pattern Header Comments

'''
//===----------------------------------------------------------------------===//
// Pattern: AddFMAFusion_F32
//
// Fuses v_add_f32 into v_fma_f32 by folding constants.
//
// Before: v_fma_f32 v0, a, b, 1.0
//         v_add_f32 v2, 1.0, v0
//
// After:  v_fma_f32 v2, a, b, 2.0
//
// Benefit: 2 instructions -> 1, frees v0
//===----------------------------------------------------------------------===//
pattern AddFMAFusion_F32 {
  ...
}
'''

---

## Grammar Summary (EBNF)

For reference, here's the formal grammar:

'''ebnf
pattern        := 'pattern' IDENT '{' match_block constraints_block? rewrite_block '}'

match_block    := 'match' '{' match_stmt+ '}'
match_stmt     := VAR '=' IDENT VAR (',' VAR)*

constraints_block := 'constraints' '{' constraint+ '}'
constraint        := IDENT '(' VAR (',' VAR)* ')'

rewrite_block  := 'rewrite' '{' rewrite_stmt* '}'
rewrite_stmt   := VAR '=' IDENT '(' VAR (',' VAR)* ')'

VAR    := '$' IDENT
IDENT  := [a-zA-Z_][a-zA-Z0-9_]*
'''

**But you don't need to memorize this!** Just follow the examples.

---

## Validation

The parser checks:

- [x] All variables defined before use
- [x] Opcodes exist ('v_fma_f32', 'v_add_f32', etc.)
- [x] Constraint functions are valid
- [x] Rewrite operations reference matched variables
- [x] Syntax is correct

**Errors are caught at build time**, not runtime!

---

## Tips

### Start with Existing Patterns

Copy and modify existing patterns from 'PeepholePatterns.pattern'.

### Test Incrementally

1. Add pattern
2. Build
3. Check generated code: 'cat build/PeepholePatterns.inc'
4. Test with IR

### Use Simple Names

'''
[x] Good: $result, $tmp, $dst
[ ] Bad: $intermediate_computation_result_temporary_value
'''

### One Constraint Per Line

'''
[x] Good:
constraints {
  HasOneUse($tmp)
  IsConstant($c1)
  IsConstant($c2)
}

[ ] Bad:
constraints { HasOneUse($tmp) IsConstant($c1) IsConstant($c2) }
'''

---

## Common Mistakes

### Mistake 1: Wrong Operand Order

'''
[ ] Wrong:
$add = v_add_f32 $src1, $src2, $dst

[x] Correct:
$add = v_add_f32 $dst, $src1, $src2
'''

### Mistake 2: Missing Constraints

'''
[ ] Wrong (might crash if $tmp used multiple times):
match {
  $mul = v_mul_f32 $tmp, $a, $b
  $add = v_add_f32 $dst, $tmp, $c
}

[x] Correct:
match {
  $mul = v_mul_f32 $tmp, $a, $b
  $add = v_add_f32 $dst, $tmp, $c
}
constraints {
  HasOneUse($tmp)    // Required!
}
'''

### Mistake 3: Wrong Temporal Order

'''
[ ] Wrong (uses $tmp before it's defined):
match {
  $add = v_add_f32 $dst, $tmp, $c
  $mul = v_mul_f32 $tmp, $a, $b
}

[x] Correct:
match {
  $mul = v_mul_f32 $tmp, $a, $b
  $add = v_add_f32 $dst, $tmp, $c
}
'''

---

## File Location

**Where to write patterns:**
'''
src/transforms/asm/PeepholePatterns.pattern
'''

**Generated code (auto):**
'''
build/PeepholePatterns.inc
'''

---

## See Also

- **[Adding Peephole Patterns](adding-peephole-patterns.md)** - How-to guide with examples
- **[Architecture Overview](architecture.md)** - System architecture and pass pipeline
- **'src/transforms/asm/PeepholePatterns.pattern'** - Real pattern examples

---

## Quick Reference Card

'''
Pattern Structure:
  pattern Name { match {} constraints {} rewrite {} }

Match:
  $inst = opcode $dst, $src1, $src2, ...

Constraints:
  HasOneUse($reg)              // Exactly one use
  IsConstant($operand)         // Literal value
  SameValue($r1, $r2)          // Same register
  DifferentValue($r1, $r2)     // Different registers

Rewrite:
  $new = AddConstants($c1, $c2)    // c1 + c2
  $new = SubConstants($c1, $c2)    // c1 - c2
  $new = MulConstants($c1, $c2)    // c1 * c2
  $new = NegateConstant($c)        // -c

Variables:
  $inst       // Instruction variable
  $reg        // Register/operand variable
  $const      // Constant variable (use IsConstant)

Rules:
  1. Destination operand comes first
  2. Instructions in temporal order
  3. Variables start with $
  4. Add HasOneUse for intermediate values
  5. Add IsConstant for constant folding
'''

---

## Questions?

Just look at existing patterns in 'PeepholePatterns.pattern' and copy the structure!
