# Virtual Registers User Guide

## Overview

Virtual registers enable template-based code generation in StinkyTofu. They allow you to create reusable instruction sequences with placeholder registers that can be mapped to actual physical registers during instantiation. This is particularly useful for generating activation functions, common code patterns, and library functions that need to work with different register allocations.

### Encoding

Virtual registers are distinguished from physical registers by the MSB (bit 31) of the register index ('reg.idx'). When the MSB is set, the register is virtual; the lower 31 bits hold the virtual index. This means that if a virtual register accidentally reaches the emitter or scheduler without being resolved, the extremely large index value (>= 2^31) will cause obvious failures. Use 'resolveVirtualToPhysical()' to convert virtual registers to physical before emission.

## Basic Usage

### Creating Virtual Registers

'''cpp
#include "stinkytofu/ir/asm/StinkyRegister.hpp"

using namespace stinkytofu;

// Create virtual VGPRs
auto vTemp0 = StinkyRegister::Virtual(0);      // Virtual v0
auto vTemp1 = StinkyRegister::Virtual(1);      // Virtual v1
auto vTemp2 = StinkyRegister::Virtual(2, 4);   // Virtual v[2:5] (4 consecutive)

// Create virtual SGPRs
auto sTemp0 = StinkyRegister::VirtualSGPR(0);  // Virtual s0
auto sTemp1 = StinkyRegister::VirtualSGPR(1);  // Virtual s1
'''

### Checking if Register is Virtual

'''cpp
StinkyRegister reg = StinkyRegister::Virtual(0);

if (reg.isVirtualister()) {
    // This is a virtual register needing remapping
}

// Physical registers return false
StinkyRegister physical(RegType::V, 10, 1);
assert(!physical.isVirtualister());

// Literals also return false
StinkyRegister literal(42);
assert(!literal.isVirtualister());
'''

### Resolving to Physical Registers

'''cpp
// Create a virtual register
auto vTemp = StinkyRegister::Virtual(0);

// Resolve to physical register v10
auto v10 = vTemp.resolveVirtualToPhysical(10);

// Resolve to physical register v20 (same virtual, different mapping)
auto v20 = vTemp.resolveVirtualToPhysical(20);

// After resolving, registers are physical
assert(!v10.isVirtualister());
assert(v10.reg.idx == 10);
'''

### Remapping Instructions

'''cpp
StinkyTofu st({12, 5, 0});  // gfx1250

// Create instruction with virtual registers
auto vDst  = StinkyRegister::Virtual(0);
auto vSrc0 = StinkyRegister::Virtual(1);
auto vSrc1 = StinkyRegister::Virtual(2);

auto insts = st.VAddU32(vDst, vSrc0, vSrc1, "add");
auto* inst = insts[0];

// Remap all virtual registers in the instruction
inst->remapRegisters(10, 0);  // VGPR offset: 10, SGPR offset: 0

// Now instruction uses v10, v11, v12
'''

## Module-Level Remapping

### In-Place Remapping

Use 'remapVirtualisters()' when you want to modify the original module:

'''cpp
StinkyTofu st({9, 4, 2});
auto module = st.createIRList("my_kernel");

// Add instructions with virtual registers
auto v0 = StinkyRegister::Virtual(0);
auto v1 = StinkyRegister::Virtual(1);
module->add(st.VMovB32(v0, v1, "move"));

// Remap in-place: v0->v10, v1->v11
module->remapVirtualisters(10, 0);

std::cout << module->emitAssembly() << std::endl;
// Output: v_mov_b32 v10, v11
'''

**When to use:**
- Single instantiation
- Don't need to preserve the original
- Modifying the module directly is acceptable

### Clone and Remap

Use 'cloneAndRemap()' when you need to preserve the original for reuse:

'''cpp
StinkyTofu st({9, 4, 2});

// Create template with virtual registers
auto templateModule = st.createIRList("template");
auto v0 = StinkyRegister::Virtual(0);
auto v1 = StinkyRegister::Virtual(1);
templateModule->add(st.VMovB32(v0, v1, "move"));

// Clone and remap to different locations
auto inst1 = templateModule->cloneAndRemap(10, 0);  // v10, v11
auto inst2 = templateModule->cloneAndRemap(20, 0);  // v20, v21
auto inst3 = templateModule->cloneAndRemap(30, 0);  // v30, v31

// Template remains unchanged (v0, v1)
auto inst4 = templateModule->cloneAndRemap(40, 0);  // Can reuse!
'''

**When to use:**
- Multiple instantiations from the same template
- Need to preserve the original template
- Working with shared templates (especially in Python)

## Working with VGPR and SGPR Offsets

Both 'remapVirtualisters()' and 'cloneAndRemap()' accept separate offsets for VGPRs and SGPRs:

'''cpp
StinkyTofu st({9, 4, 2});
auto module = st.createIRList("test");

// Mix of virtual VGPRs and SGPRs
auto vDst   = StinkyRegister::Virtual(0);   // Virtual VGPR
auto vSrc   = StinkyRegister::Virtual(1);   // Virtual VGPR
auto sConst = StinkyRegister::VirtualSGPR(0);   // Virtual SGPR

module->add(st.VAddU32(vDst, vSrc, sConst, "add"));

// Remap: VGPRs with offset 10, SGPRs with offset 5
module->remapVirtualisters(10, 5);

// Result: v_add_u32 v10, v11, s5
'''

## Mixed Virtual and Physical Registers

Virtual and physical registers can coexist in the same instruction:

'''cpp
StinkyTofu st({9, 4, 2});
auto module = st.createIRList("test");

auto vVirtual  = StinkyRegister::Virtual(0);  // Virtual
auto vPhysical = StinkyRegister(RegType::V, 100, 1); // Physical
auto literal   = StinkyRegister(42);                 // Literal

module->add(st.VAddU32(vVirtual, vPhysical, literal, "add"));

// Remap only affects virtual registers
module->remapVirtualisters(10, 0);

// Result: v_add_u32 v10, v100, 42
//         ^ remapped  ^ unchanged  ^ unchanged
'''

## Complete Example: Activation Function Template

Here's a complete example showing how to create and use a reusable activation function template:

'''cpp
#include "StinkyTofu.hpp"
#include <iostream>

using namespace stinkytofu;

// Create abs(x) template: abs(x) = max(x, -x)
std::shared_ptr<IRListModule> createAbsTemplate(StinkyTofu& st) {
    auto module = st.createIRList("abs_template");

    // Use virtual registers v0, v1, v2
    auto vInput  = StinkyRegister::Virtual(0);
    auto vOutput = StinkyRegister::Virtual(1);
    auto vTemp   = StinkyRegister::Virtual(2);

    // Generate: v2 = -v0, v1 = max(v0, v2)
    module->add(st.VSubI32(vTemp, StinkyRegister(0), vInput, "negate"));
    module->add(st.VMaxI32(vOutput, vInput, vTemp, "abs"));

    return module;
}

int main() {
    StinkyTofu st({12, 5, 0});  // gfx1250
    auto kernel = st.createIRList("my_kernel");

    // Create template once
    auto absTemplate = createAbsTemplate(st);

    // Instantiate at different register locations
    auto abs1 = absTemplate->cloneAndRemap(10, 0);  // Uses v[10:12]
    auto abs2 = absTemplate->cloneAndRemap(20, 0);  // Uses v[20:22]
    auto abs3 = absTemplate->cloneAndRemap(30, 0);  // Uses v[30:32]

    // Add to kernel
    kernel->addModule(*abs1);
    kernel->addModule(*abs2);
    kernel->addModule(*abs3);

    // Emit assembly
    std::cout << kernel->emitAssembly() << std::endl;
    /*
    Output:
        v_sub_i32 v12, 0, v10      // abs1: negate
        v_max_i32 v11, v10, v12    // abs1: abs
        v_sub_i32 v22, 0, v20      // abs2: negate
        v_max_i32 v21, v20, v22    // abs2: abs
        v_sub_i32 v32, 0, v30      // abs3: negate
        v_max_i32 v31, v30, v32    // abs3: abs
    */

    return 0;
}
'''

## Design Rationale

### Why Virtual Registers?

**Problem**: You want to generate reusable code patterns (like activation functions) without hardcoding specific register allocations.

**Solution**: Use virtual registers as placeholders during template generation, then map them to physical registers during instantiation.

**Benefits**:
- Generate once, instantiate many times
- No hardcoded register indices
- Templates are independent of surrounding code
- Clean separation of template definition and usage

### Comparison with TensileLite's Approach

| Aspect | TensileLite (Holder) | StinkyTofu (Virtual) |
|--------|----------------------|----------------------|
| Template creation | 'Holder(idx)' | 'StinkyRegister::Virtual(idx)' / 'VirtualReg(type, idx)' |
| Single instantiation | 'HolderToGpr(module, base, 'v')' | 'module->remapVirtualisters(base, 0)' |
| Multiple instantiation | Manual clone + 'HolderToGpr' | 'module->cloneAndRemap(base, 0)' |
| Phase | Two-phase (clone -> transform) | Single-phase (clone+remap) |
| Type safety | Runtime check | Compile-time + runtime |

## Limitations and Notes

### Modifier Cloning

Currently, modifiers (SDWA, DPP, VOP3P, etc.) are **not cloned** by 'cloneAndRemap()'. This is acceptable for simple activation functions but would need to be addressed for instructions with complex modifiers.

**Workaround**: If you need modifier cloning, manually clone and apply modifiers after remapping.

### Dependency Tracking

The 'users' and 'operandDefs' fields (dependency tracking) are **not copied** during cloning. This is intentional as dependencies should be rebuilt in the context where the cloned instructions are used.

### Register Overflow

Virtual register remapping does not check for register overflow. Ensure your offsets don't exceed available physical registers:

'''cpp
// BAD: May overflow if architecture only has 256 VGPRs
module->remapVirtualisters(250, 0);

// GOOD: Check available registers first
int availableVGPRs = getArchMaxVGPRs(archID);
int requiredVGPRs = getTemplateVGPRCount(template);
if (baseOffset + requiredVGPRs <= availableVGPRs) {
    module->remapVirtualisters(baseOffset, 0);
}
'''

## Best Practices

1. **Use descriptive names for templates**: 'abs_template', 'relu_template', etc.
2. **Document virtual register usage**: Comment which virtual registers are inputs/outputs
3. **Prefer 'cloneAndRemap()'**: Unless you're certain it's single-use
4. **Keep templates simple**: Avoid complex control flow or modifiers
5. **Test with multiple instantiations**: Ensure templates work correctly when reused

## See Also

- [StinkyAsmEmitter User Guide](asm-emitter.md) - Converting IR to assembly
- API Reference: 'include/stinkytofu/ir/asm/StinkyRegister.hpp'
- Unit Tests: 'tests/unit/VirtualisterTest.cpp', 'tests/unit/VirtualisterRemappingTest.cpp'
