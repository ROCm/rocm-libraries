# Adding Intrinsics

This guide shows you how to add new intrinsics to StinkyTofu.

## What is an Intrinsic?

An intrinsic is a high-level operation that expands into multiple instructions. Examples:
- `ReluF32`: Implements ReLU activation (max(x, 0))
- `ClampF32`: Clamps values to a range [min, max]
- `SigmoidF32`: Applies sigmoid activation function

Intrinsics are automatically available throughout StinkyTofu via the `IntrinsicRegistry`.

## Quick Start

### Step 1: Define Your Intrinsic

Edit `src/ir/logical/Intrinsics.intrinsic`:

```
intrinsic ReluF32 {
    arguments {
        dest: vgpr_f32
        src: vgpr_f32
        temp: vgpr_f32
    }

    body {
        v[temp] = v_cmp_gt_f32(v[src], 0.0)
        v[dest] = v_cndmask_b32(0.0, v[src], v[temp])
    }

    comment "ReLU F32 activation"
    python_binding true
}
```

### Step 2: Rebuild

```bash
cd build
ninja
```

The build system automatically:
1. Parses `Intrinsics.intrinsic`
2. Compiles it to `intrinsics.st.bc` (binary format)
3. Makes it available to the runtime

### Step 3: Use Your Intrinsic

In TensileLite or any code generator:

```cpp
#include "stinkytofu/ir/logical/IntrinsicRegistry.hpp"

auto& intrinsics = IntrinsicRegistry::instance();

if (intrinsics.hasIntrinsic("ReluF32")) {
    auto pattern = intrinsics.lookup("ReluF32");
    // Use pattern to generate code
}
```

## Intrinsic Definition Syntax

### Basic Structure

```
intrinsic <Name> {
    arguments { ... }
    body { ... }
    comment "..."
    python_binding true/false
}
```

### Arguments Block

List all registers needed, including temporaries:

```
arguments {
    dest: vgpr_f32      // Output register
    src: vgpr_f32       // Input register
    temp: vgpr_f32      // Temporary register
    min_val: vgpr_f32   // Another input
}
```

**Important:** You must explicitly list temporary registers. StinkyTofu does not use SSA form, so register allocation happens before intrinsic expansion.

### Body Block

Define the instruction sequence using high-level IR operations:

```
body {
    v[temp] = v_max_f32(v[src], v[min_val])
    v[dest] = v_min_f32(v[temp], v[max_val])
}
```

**Syntax:**
- Register reference: `v[name]` for VGPR, `s[name]` for SGPR
- Operation: `operation(operands...)`
- Assignment: `dest = operation(...)`

### Metadata

```
comment "Brief description of what this intrinsic does"
python_binding true    // Generate Python API (future feature)
```

## Examples

### Simple Intrinsic (ReLU)

```
intrinsic ReluF32 {
    arguments {
        dest: vgpr_f32
        src: vgpr_f32
        temp: vgpr_f32
    }
    body {
        v[temp] = v_cmp_gt_f32(v[src], 0.0)
        v[dest] = v_cndmask_b32(0.0, v[src], v[temp])
    }
    comment "ReLU F32 activation"
    python_binding true
}
```

### Complex Intrinsic (Clamp)

```
intrinsic ClampF32 {
    arguments {
        dest: vgpr_f32
        src: vgpr_f32
        min_val: vgpr_f32
        max_val: vgpr_f32
        temp: vgpr_f32
    }
    body {
        v[temp] = v_max_f32(v[src], v[min_val])
        v[dest] = v_min_f32(v[temp], v[max_val])
    }
    comment "Clamp F32 to range [min, max]"
    python_binding true
}
```

## Testing Your Intrinsic

### Verify Compilation

```bash
cd build
./tools/intrinsic-compiler/intrinsic-compiler
# Should output: Compiled 5 intrinsics to intrinsics.st.bc (1117 bytes)
```

### Test Loading

```bash
./tools/intrinsic-compiler/demo_auto_load
# Should show your intrinsic in the list
```

### Add Unit Test

Edit `tests/unit/IntrinsicFlowTest.cpp`:

```cpp
TEST_F(IntrinsicFlowTest, MyNewIntrinsic) {
    auto& registry = IntrinsicRegistry::instance();
    ASSERT_TRUE(registry.hasIntrinsic("MyNewIntrinsic"));

    auto pattern = registry.lookup("MyNewIntrinsic");
    ASSERT_NE(pattern, nullptr);
    EXPECT_EQ(pattern->arguments.size(), 3);
    EXPECT_EQ(pattern->body.size(), 2);
}
```

## Tips

? **DO:**
- Use clear, descriptive names (`ReluF32`, not `r32`)
- List all temporaries in `arguments`
- Keep intrinsics focused (single operation)
- Add comments explaining what it does

? **DON'T:**
- Use SSA form (StinkyTofu isn't SSA-based)
- Assume register allocation happens during expansion
- Create very large intrinsics (consider breaking them up)
- Use assembly syntax (use high-level IR operations)

## Troubleshooting

**Intrinsic not found at runtime:**
- Rebuild: `ninja` in the build directory
- Check `intrinsics.st.bc` was generated
- Verify syntax in `Intrinsics.intrinsic`

**Parse errors:**
- Check matching braces `{ }`
- Ensure `arguments` and `body` are present
- Verify register syntax: `v[name]` or `s[name]`

**Build errors:**
- Run `ninja clean && ninja` to rebuild from scratch
- Check the TableGen output for syntax errors

## See Also

- [Architecture Overview](architecture.md) - System architecture and intrinsic system overview
- [IntrinsicRegistry API](../../include/stinkytofu/ir/logical/IntrinsicRegistry.hpp) - Runtime API
- Example intrinsics: `src/ir/logical/Intrinsics.intrinsic`
