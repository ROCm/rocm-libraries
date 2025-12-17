#!/usr/bin/env python3
"""
Example: Using StinkyIR (High-Level IR) vs StinkyAsmIR (Low-Level Assembly IR)

This demonstrates the two-tier architecture:
- StinkyAsmIR: Low-level assembly instructions
- StinkyIR: High-level functions that generate instruction sequences
"""

import sys
sys.path.insert(0, '../../build/python_module')
import stinkytofu
from stinkytofu import StinkyAsmIR, StinkyIR, vgpr, sgpr

def example_low_level_asm():
    """Example: Low-level assembly IR (StinkyAsmIR)"""
    print("=" * 70)
    print("Example 1: Low-Level Assembly IR (StinkyAsmIR)")
    print("=" * 70)
    
    st = StinkyAsmIR([9, 4, 2])  # Low-level assembly builder
    module = st.createIRList("low_level_operations")
    
    # Low-level: Manual instruction sequencing
    # Simple vector arithmetic
    inst = st.VAddU32(vgpr(0), vgpr(1), vgpr(2), "add")
    module.add(inst)
    
    inst = st.VMulF32(vgpr(3), vgpr(4), vgpr(5), "multiply")
    module.add(inst)
    
    inst = st.VSubU32(vgpr(6), vgpr(7), vgpr(8), "subtract")
    module.add(inst)
    
    asm = module.emitAssembly()
    print(asm)
    print("Note: Low-level gives you full control over every instruction")
    print()


def example_high_level_division():
    """Example: High-level IR (StinkyIR) - Division"""
    print("=" * 70)
    print("Example 2: High-Level IR (StinkyIR) - Vector Division")
    print("=" * 70)
    
    st = StinkyAsmIR([9, 4, 2])  # Low-level builder that creates and owns instructions
    ir = StinkyIR([9, 4, 2])     # High-level function generator
    
    # Same division, but using high-level function
    # Much simpler and automatically handles magic number calculation
    insts = ir.vectorStaticDivide(
        st,            # pass the builder
        qReg=0,        # quotient -> v0
        dReg=1,        # dividend -> v1
        divisor=7,     # compile-time constant
        tmpVgpr=[2, 3],  # temporary VGPRs
        comment="v0 = v1 / 7"
    )
    
    # High-level functions return multiple instructions
    # Use a low-level IRListModule to collect them
    module = st.createIRList("high_level_division")
    module.add(insts)  # Add all instructions at once
    
    asm = module.emitAssembly()
    print(asm)
    print()


def example_bpe_multiplication():
    """Example: Bytes-per-element multiplication"""
    print("=" * 70)
    print("Example 3: High-Level IR - BPE Multiplication")
    print("=" * 70)
    
    st = StinkyAsmIR([9, 4, 2])  # Low-level builder
    ir = StinkyIR([9, 4, 2])     # High-level generator
    
    # Multiply by bytes-per-element (common in address calculation)
    # Automatically optimizes for power-of-2 and special cases
    insts = ir.vectorMultiplyBpe(
        st,         # pass builder
        dstReg=10,
        srcReg=5,
        bpe=4.0,  # 4 bytes per element (FP32)
        comment="address calculation"
    )
    
    module = st.createIRList("bpe_multiply")
    module.add(insts)
    
    asm = module.emitAssembly()
    print(asm)
    print("Special case (bpe=0.75):")
    
    # Try a more complex BPE value
    insts = ir.vectorMultiplyBpe(st, dstReg=10, srcReg=5, bpe=0.75)
    module2 = st.createIRList("bpe_075")
    module2.add(insts)
    print(module2.emitAssembly())
    print()


def example_conditional_branch():
    """Example: Conditional branching"""
    print("=" * 70)
    print("Example 4: High-Level IR - Conditional Branching")
    print("=" * 70)
    
    st = StinkyAsmIR([9, 4, 2])  # Low-level builder
    ir = StinkyIR([9, 4, 2])     # High-level generator
    module = st.createIRList("conditional")
    
    # High-level: Branch if register is zero
    insts = ir.BranchIfZero(
        st,          # pass builder
        sgprName=10,
        tmpSgpr=20,
        label="zero_case",
        comment="branch on zero"
    )
    module.add(insts)
    
    # Some work for non-zero case
    inst = st.VAddU32(vgpr(0), vgpr(1), vgpr(2), "non-zero path")
    module.add(inst)
    
    # Create the label
    label = st.createLabel("zero_case")
    module.add(label)
    
    # Work for zero case
    inst = st.VMulF32(vgpr(0), vgpr(3), vgpr(4), "zero path")
    module.add(inst)
    
    asm = module.emitAssembly()
    print(asm)
    print()


def example_mixed_usage():
    """Example: Mixing low-level and high-level IR"""
    print("=" * 70)
    print("Example 5: Mixed Low-Level and High-Level IR")
    print("=" * 70)
    
    st = StinkyAsmIR([9, 4, 2])  # Low-level builder
    ir = StinkyIR([9, 4, 2])     # High-level generator
    module = st.createIRList("mixed")
    
    # Low-level: Load data
    inst = st.BufferLoadDword(vgpr(0), vgpr(10), "load data")
    module.add(inst)
    
    inst = st.SWaitCnt(vlcnt=0, comment="wait for load")
    module.add(inst)
    
    # High-level: Divide by stride
    insts = ir.vectorStaticDivide(
        st,          # pass builder
        qReg=1,
        dReg=0,
        divisor=16,
        tmpVgpr=[2, 3],
        comment="calculate index"
    )
    module.add(insts)
    
    # Low-level: Use result
    inst = st.VAddU32(vgpr(5), vgpr(1), vgpr(6), "adjust index")
    module.add(inst)
    
    asm = module.emitAssembly()
    print(asm)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("StinkyTofu Two-Tier Architecture Examples")
    print("  - StinkyAsmIR: Low-level assembly instructions")
    print("  - StinkyIR: High-level functions")
    print("=" * 70 + "\n")
    
    example_low_level_asm()
    example_high_level_division()
    example_bpe_multiplication()
    example_conditional_branch()
    example_mixed_usage()
    
    print("=" * 70)
    print("Benefits of Two-Tier Architecture:")
    print("  ✓ Low-level: Full control, hardware-specific optimizations")
    print("  ✓ High-level: Productivity, algorithm reuse, fewer bugs")
    print("  ✓ Mix both: Use high-level for complex operations, low-level for fine-tuning")
    print("=" * 70)

