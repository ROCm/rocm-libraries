#!/usr/bin/env python3
"""
Basic usage example for the StinkyTofu Python module.

This example demonstrates how to:
1. Create a StinkyTofu builder for a specific architecture
2. Add instructions using register operands
3. Emit assembly code with optional comments
"""

import sys
import os

# Add the build directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python_module'))

from stinkytofu import StinkyAsmIR, vgpr, sgpr, acc


def main():
    print("=" * 70)
    print("StinkyTofu Python Module - Basic Usage Example")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Example 1: Simple Vector ALU Operations
    # ========================================================================
    
    print("Example 1: Simple Vector ALU Operations")
    print("-" * 70)
    
    # Create a StinkyAsmIR builder for MI300A/MI300X (gfx942)
    st = StinkyAsmIR([9, 4, 2])
    module = st.createIRList("simple_valu")
    
    # Create and add vector ALU instructions
    module.add(st.VAddU32(vgpr(0), vgpr(1), vgpr(2), "c = a + b"))
    module.add(st.VMulF32(vgpr(3), vgpr(0), vgpr(4), "d = c * scale"))
    module.add(st.VAddU32(vgpr(5), vgpr(3), vgpr(6), "result = d + offset"))
    
    # Emit assembly with comments
    asm = module.emitAssembly(emit_comments=True)
    print(asm)
    print()
    
    # ========================================================================
    # Example 2: Scalar Instructions and Synchronization
    # ========================================================================
    
    print("Example 2: Scalar Instructions and Synchronization")
    print("-" * 70)
    
    st2 = StinkyAsmIR([9, 4, 2])
    module2 = st2.createIRList("scalar_sync")
    
    # Scalar operations
    module2.add(st2.SAbsI32(sgpr(0), sgpr(1), "compute absolute value"))
    module2.add(st2.SBarrier("wait for all threads"))
    module2.add(st2.VAddU32(vgpr(0), vgpr(1), vgpr(2), "continue after barrier"))
    
    asm2 = module2.emitAssembly(emit_comments=True)
    print(asm2)
    print()
    
    # ========================================================================
    # Example 3: Matrix Multiplication (MFMA)
    # ========================================================================
    
    print("Example 3: Matrix Multiplication (MFMA)")
    print("-" * 70)
    
    st3 = StinkyAsmIR([9, 4, 2])
    module3 = st3.createIRList("mfma_test")
    
    # Matrix multiply-accumulate using createMFMA
    # MFMA typically uses accumulator registers (acc)
    module3.add(st3.createMFMA(
        instType="f32",      # input type
        accType="f32",       # accumulator type
        m=16, n=16, k=4,     # M, N, K dimensions
        blocks=1,            # number of blocks
        mfma1k=False,        # not 1k variant
        acc=acc(0, 4),       # destination accumulator (4 registers)
        a=vgpr(0, 4),        # matrix A: v[0:3] (4 registers)
        b=vgpr(4, 4),        # matrix B: v[4:7] (4 registers)
        acc2=acc(0, 4),      # input accumulator (can be same as dest)
        comment="C = A * B + C"
    ))
    
    asm3 = module3.emitAssembly(emit_comments=True)
    print(asm3)
    print()
    
    # ========================================================================
    # Example 4: Register Ranges
    # ========================================================================
    
    print("Example 4: Register Ranges")
    print("-" * 70)
    
    st4 = StinkyAsmIR([9, 4, 2])
    module4 = st4.createIRList("register_ranges")
    
    # Use register ranges with count parameter
    module4.add(st4.VAddU32(
        vgpr(0, 4),   # v[0:3] - 4 consecutive registers
        vgpr(4, 4),   # v[4:7] - 4 consecutive registers
        vgpr(8, 4),   # v[8:11] - 4 consecutive registers
        "vectorized add"
    ))
    
    asm4 = module4.emitAssembly(emit_comments=True)
    print(asm4)
    print()
    
    # ========================================================================
    # Example 5: Complete Kernel Fragment
    # ========================================================================
    
    print("Example 5: Complete Kernel Fragment")
    print("-" * 70)
    
    st5 = StinkyAsmIR([9, 4, 2])
    module5 = st5.createIRList("complete_kernel")
    
    # Load data
    module5.add(st5.VAddU32(vgpr(0), vgpr(10), vgpr(11), "compute address"))
    
    # Synchronize
    module5.add(st5.SBarrier("sync before computation"))
    
    # Compute
    module5.add(st5.VMulF32(vgpr(1), vgpr(2), vgpr(3), "scale input"))
    module5.add(st5.VAddU32(vgpr(4), vgpr(1), vgpr(5), "add bias"))
    
    # Matrix multiply using createMFMA
    module5.add(st5.createMFMA(
        "f32", "f32", 16, 16, 4, 1, False,
        acc(0, 4), vgpr(6), vgpr(7), acc(0, 4),
        comment="accumulate results"
    ))
    
    # Synchronize
    module5.add(st5.SBarrier("sync before store"))
    
    # Store result
    module5.add(st5.VAddU32(vgpr(8), vgpr(12), vgpr(13), "compute store address"))
    
    asm5 = module5.emitAssembly(emit_comments=True)
    print(asm5)
    print()
    
    # ========================================================================
    # Example 6: Composite Instructions (Architecture-Aware Lowering)
    # ========================================================================
    
    print("Example 6: Composite Instructions (Architecture-Aware Lowering)")
    print("-" * 70)
    
    st6 = StinkyAsmIR([9, 4, 2])
    module6 = st6.createIRList("composite_example")
    
    # Some instructions may not be supported on all architectures
    # StinkyTofu automatically lowers them to equivalent sequences
    
    # VAddPKF32: packed add for F32 (adds two pairs of floats in parallel)
    # - On gfx942+ with v_pk_add_f32: generates 1 packed instruction
    # - On older architectures without support: generates 2 v_add_f32 instructions
    insts = st6.VAddPKF32(vgpr(0, 2), vgpr(2, 2), vgpr(4, 2), "packed add")
    print(f"  VAddPKF32 generated {len(insts)} instruction(s) for gfx942")
    print(f"  (gfx942 supports v_pk_add_f32, so it uses a single packed instruction)")
    
    # module.add() accepts lists of instructions
    module6.add(insts)
    
    # Add more instructions
    module6.add(st6.VAddU32(vgpr(6), vgpr(0), vgpr(1)))
    
    asm6 = module6.emitAssembly(emit_comments=True)
    print(asm6)
    print()
    
    # ========================================================================
    # Example 7: Assembly Without Comments
    # ========================================================================
    
    print("Example 7: Assembly Without Comments")
    print("-" * 70)
    
    st7 = StinkyAsmIR([9, 4, 2])
    module7 = st7.createIRList("no_comments")
    
    module7.add(st7.VAddU32(vgpr(0), vgpr(1), vgpr(2), "this comment will not appear"))
    module7.add(st7.VMulF32(vgpr(3), vgpr(0), vgpr(4), "neither will this"))
    
    # Emit without comments
    asm7 = module7.emitAssembly(emit_comments=False)
    print(asm7)
    print()
    
    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
