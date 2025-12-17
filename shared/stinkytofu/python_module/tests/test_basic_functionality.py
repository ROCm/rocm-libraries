"""
Basic functionality tests for StinkyTofu Python bindings

Tests the core Python module imports and basic IR building.
"""

import pytest


def test_import_stinkytofu():
    """Test that we can import the stinkytofu module"""
    import stinkytofu
    assert stinkytofu is not None


def test_import_stinkyasm_ir():
    """Test that we can import StinkyAsmIR"""
    from stinkytofu import StinkyAsmIR
    assert StinkyAsmIR is not None


def test_create_stinkyasm_ir():
    """Test creating StinkyAsmIR instance"""
    from stinkytofu import StinkyAsmIR
    
    builder = StinkyAsmIR([9, 4, 2])
    assert builder is not None


def test_vgpr_sgpr_helpers():
    """Test vgpr() and sgpr() helper functions"""
    from stinkytofu import vgpr, sgpr
    
    # Test vgpr
    v0 = vgpr(0)
    assert v0 is not None
    
    v10 = vgpr(10)
    assert v10 is not None
    
    # Test sgpr
    s0 = sgpr(0)
    assert s0 is not None
    
    s20 = sgpr(20)
    assert s20 is not None


def test_create_ir_list():
    """Test creating IRListModule"""
    from stinkytofu import StinkyAsmIR
    
    builder = StinkyAsmIR([9, 4, 2])
    module = builder.createIRList("test_kernel")
    assert module is not None
    assert module.getName() == "test_kernel"


def test_add_instruction():
    """Test adding instructions to IRListModule"""
    from stinkytofu import StinkyAsmIR, vgpr
    
    builder = StinkyAsmIR([9, 4, 2])
    module = builder.createIRList("test_kernel")
    
    # Create a simple instruction (returns a list)
    inst = builder.VAddU32(vgpr(0), vgpr(1), vgpr(2), "add instruction")
    
    # Add it to the module
    module.add(inst)
    
    # Generate assembly
    asm = module.emitAssembly()
    assert "v_add_u32" in asm or "v_add_nc_u32" in asm


def test_emit_assembly():
    """Test emitting assembly from a module"""
    from stinkytofu import StinkyAsmIR, vgpr, sgpr
    
    builder = StinkyAsmIR([9, 4, 2])
    module = builder.createIRList("test_kernel")
    
    # Add a few instructions (each method returns a list)
    module.add(builder.VAddF32(vgpr(0), vgpr(1), vgpr(2), "add floats"))
    module.add(builder.SMulI32(sgpr(0), sgpr(1), sgpr(2), "mul scalars"))
    module.add(builder.SEndpgm("end"))
    
    asm = module.emitAssembly()
    
    # Verify instructions are present
    assert "v_add_f32" in asm
    assert "s_mul_i32" in asm
    assert "s_endpgm" in asm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

