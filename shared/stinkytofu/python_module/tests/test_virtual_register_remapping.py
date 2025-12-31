#!/usr/bin/env python3
"""
Python tests for virtual register remapping at module level.
Tests IRListModule::remapVirtualRegisters and IRListModule::cloneAndRemap.
"""

import sys
import os

# Add stinkytofu module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python_module'))

import stinkytofu as st


def test_remap_virtual_registers_in_place():
    """Test in-place remapping of virtual registers in a module."""
    print("\n=== Test: remapVirtualRegisters (in-place) ===")
    
    # Create module with virtual registers
    arch = [9, 4, 2]  # GFX942
    builder = st.StinkyAsmIR(arch)
    module = builder.createIRList("test_template")
    
    # Create instruction with virtual registers
    v_temp0 = st.StinkyRegister.Virtual(0)
    v_temp1 = st.StinkyRegister.Virtual(1)
    s_temp0 = st.StinkyRegister.VirtualSGPR(0)
    
    # Add a V_ADD_U32 instruction: v[temp0] = v[temp1] + s[temp0]
    insts = builder.VAddU32(v_temp0, v_temp1, s_temp0, "test instruction")
    module.add(insts)
    
    print(f"Before remap: {str(module)}")
    
    # Remap in-place: vgprOffset=10, sgprOffset=5
    module.remapVirtualRegisters(10, 5)
    
    asm_after = str(module)
    print(f"After remap: {asm_after}")
    
    # Verify the registers were remapped
    assert "v10" in asm_after, "Expected v10 in output"
    assert "v11" in asm_after, "Expected v11 in output"
    assert "s5" in asm_after, "Expected s5 in output"
    
    print("✓ In-place remapping successful\n")


def test_clone_and_remap():
    """Test cloning a module and remapping virtual registers (template reuse)."""
    print("\n=== Test: cloneAndRemap (template reuse) ===")
    
    # Create template module
    arch = [9, 4, 2]  # GFX942
    builder = st.StinkyAsmIR(arch)
    template_module = builder.createIRList("activation_template")
    
    # Create instruction with virtual registers
    v_temp0 = st.StinkyRegister.Virtual(0)
    v_temp1 = st.StinkyRegister.Virtual(1)
    v_temp2 = st.StinkyRegister.Virtual(2)
    
    # Add multiple instructions: v[temp0] = v[temp1] + v[temp2]
    insts1 = builder.VAddF32(v_temp0, v_temp1, v_temp2, "template add 1")
    template_module.add(insts1)
    
    insts2 = builder.VMulF32(v_temp0, v_temp0, v_temp1, "template mul")
    template_module.add(insts2)
    
    template_asm = str(template_module)
    print(f"Template: {template_asm}")
    
    # Clone and remap for first instance: vgprOffset=10, sgprOffset=0
    instance1 = template_module.cloneAndRemap(10, 0)
    instance1_asm = str(instance1)
    print(f"Instance 1 (offset=10): {instance1_asm}")
    
    # Verify instance 1
    assert "v10" in instance1_asm, "Expected v10 in instance1"
    assert "v11" in instance1_asm, "Expected v11 in instance1"
    assert "v12" in instance1_asm, "Expected v12 in instance1"
    
    # Clone and remap for second instance: vgprOffset=20, sgprOffset=0
    instance2 = template_module.cloneAndRemap(20, 0)
    instance2_asm = str(instance2)
    print(f"Instance 2 (offset=20): {instance2_asm}")
    
    # Verify instance 2
    assert "v20" in instance2_asm, "Expected v20 in instance2"
    assert "v21" in instance2_asm, "Expected v21 in instance2"
    assert "v22" in instance2_asm, "Expected v22 in instance2"
    
    # Verify template is unchanged
    template_asm_after = str(template_module)
    print(f"Template (should be unchanged): {template_asm_after}")
    assert template_asm == template_asm_after, "Template should remain unchanged after cloning"
    
    # Verify template still has virtual registers (v[0:2])
    # Note: Virtual registers might be printed as v0, v1, v2, but they should be the *same*
    # as before cloning (this is a bit tricky to verify from ASM string alone)
    
    print("✓ Clone and remap successful (template reuse working)\n")


def test_mixed_physical_and_virtual():
    """Test remapping with a mix of physical and virtual registers."""
    print("\n=== Test: Mixed physical and virtual registers ===")
    
    arch = [9, 4, 2]  # GFX942
    builder = st.StinkyAsmIR(arch)
    module = builder.createIRList("mixed_template")
    
    # Mix of virtual and physical registers
    v_virtual = st.StinkyRegister.Virtual(0)
    v_physical = st.vgpr(100)  # Physical v100
    
    # Instruction: v[virtual] = v[physical] + v[virtual]
    insts = builder.VAddF32(v_virtual, v_physical, v_virtual, "mixed registers")
    module.add(insts)
    
    print(f"Before remap: {str(module)}")
    
    # Remap: only virtual registers should change
    module.remapVirtualRegisters(50, 0)
    
    asm_after = str(module)
    print(f"After remap: {asm_after}")
    
    # v[virtual=0] -> v50, v[physical=100] -> v100 (unchanged)
    assert "v50" in asm_after, "Expected v50 (remapped virtual)"
    assert "v100" in asm_after, "Expected v100 (unchanged physical)"
    
    print("✓ Mixed register remapping successful\n")


def test_sgpr_remapping():
    """Test SGPR virtual register remapping."""
    print("\n=== Test: SGPR virtual register remapping ===")
    
    arch = [9, 4, 2]  # GFX942
    builder = st.StinkyAsmIR(arch)
    module = builder.createIRList("sgpr_template")
    
    # Virtual SGPRs
    s_temp0 = st.StinkyRegister.VirtualSGPR(0, 2)  # s[0:1]
    s_temp2 = st.StinkyRegister.VirtualSGPR(2, 2)  # s[2:3]
    
    # S_ADD_U32 instruction
    insts = builder.SAddU32(s_temp0, s_temp0, s_temp2, "sgpr add")
    module.add(insts)
    
    print(f"Before remap: {str(module)}")
    
    # Remap: sgprOffset=10
    module.remapVirtualRegisters(0, 10)
    
    asm_after = str(module)
    print(f"After remap: {asm_after}")
    
    # s[0:1] -> s[10:11], s[2:3] -> s[12:13]
    assert "s[10:11]" in asm_after or "s10" in asm_after, "Expected s10 or s[10:11]"
    assert "s[12:13]" in asm_after or "s12" in asm_after, "Expected s12 or s[12:13]"
    
    print("✓ SGPR remapping successful\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Virtual Register Remapping (Python)")
    print("=" * 70)
    
    try:
        test_remap_virtual_registers_in_place()
        test_clone_and_remap()
        test_mixed_physical_and_virtual()
        test_sgpr_remapping()
        
        print("=" * 70)
        print("✅ All Python tests PASSED!")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print("=" * 70)
        print(f"❌ Test FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)

