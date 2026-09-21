"""
Basic tests for stinkytofu high-level IR Python bindings with shared_ptr
"""

import pytest
import sys
import os

# Add the build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/lib"))

import stinkytofu


class TestRegisterCreation:
    """Test register creation helper functions"""

    def test_vgpr(self):
        v0 = stinkytofu.vgpr(0)
        assert v0.reg_type == stinkytofu.RegType.V
        assert v0.index == 0
        assert v0.count == 1
        assert not v0.is_literal

    def test_sgpr(self):
        s0 = stinkytofu.sgpr(10, 2)
        assert s0.reg_type == stinkytofu.RegType.S
        assert s0.index == 10
        assert s0.count == 2

    def test_accvgpr(self):
        acc = stinkytofu.accvgpr(0, 4)
        assert acc.reg_type == stinkytofu.RegType.A  # AccVGPR uses RegType.A
        assert acc.index == 0
        assert acc.count == 4


class TestIRModule:
    """Test LogicalModule creation and management"""

    def test_create_module(self):
        module = stinkytofu.LogicalModule("test_kernel")
        assert module.getName() == "test_kernel"

    def test_add_instructions(self):
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)
        v2 = stinkytofu.vgpr(2)

        module = stinkytofu.LogicalModule("test_kernel")

        # Add instructions
        module.add(stinkytofu.VAddF32(v0, v1, v2))
        module.add(stinkytofu.VMulF32(v0, v1, v2))
        module.add(stinkytofu.VFmaF32(v0, v1, v2, v0))

        # Check module dump
        dump = module.dump()
        assert "test_kernel" in dump
        assert "Instructions: 3" in dump
        assert "VAddF32" in dump
        assert "VMulF32" in dump
        assert "VFmaF32" in dump

    def test_instruction_with_comment(self):
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)
        v2 = stinkytofu.vgpr(2)

        module = stinkytofu.LogicalModule("test")
        module.add(stinkytofu.VAddU32(v0, v1, v2, comment="This is a test comment"))

        dump = module.dump()
        assert "This is a test comment" in dump


class TestInstructionTypes:
    """Test different instruction types"""

    def test_vector_alu_instructions(self):
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)
        v2 = stinkytofu.vgpr(2)

        module = stinkytofu.LogicalModule("alu_test")

        # Integer operations
        module.add(stinkytofu.VAddU32(v0, v1, v2))
        module.add(stinkytofu.VSubU32(v0, v1, v2))
        module.add(stinkytofu.VMulLOU32(v0, v1, v2))

        # Float operations
        module.add(stinkytofu.VAddF32(v0, v1, v2))
        module.add(stinkytofu.VMulF32(v0, v1, v2))
        module.add(stinkytofu.VFmaF32(v0, v1, v2, v0))

        # FP16 operations
        module.add(stinkytofu.VAddF16(v0, v1, v2))
        module.add(stinkytofu.VMulF16(v0, v1, v2))

        dump = module.dump()
        assert "Instructions: 8" in dump

    def test_ds_instructions(self):
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)
        v2 = stinkytofu.vgpr(2)

        module = stinkytofu.LogicalModule("ds_test")

        # DS load/store instructions
        module.add(stinkytofu.DSLoadB32(v0, v1, comment="Load 32-bit"))
        module.add(stinkytofu.DSLoadB64(v0, v1, comment="Load 64-bit"))
        module.add(stinkytofu.DSStoreB32(v1, v2, comment="Store 32-bit"))

        dump = module.dump()
        assert "DSLoadB32" in dump
        assert "Load 32-bit" in dump

    def test_scalar_instructions(self):
        s0 = stinkytofu.sgpr(0)
        s1 = stinkytofu.sgpr(1)
        s2 = stinkytofu.sgpr(2)

        module = stinkytofu.LogicalModule("scalar_test")

        module.add(stinkytofu.SAddU32(s0, s1, s2))
        module.add(stinkytofu.SSubU32(s0, s1, s2))
        module.add(stinkytofu.SMulI32(s0, s1, s2))

        dump = module.dump()
        assert "SAddU32" in dump


class TestSpecialInstructions:
    """Test special instruction classes (MFMA, Label, etc.)"""

    def test_mfma(self):
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)
        acc = stinkytofu.accvgpr(0, 4)

        module = stinkytofu.LogicalModule("mfma_test")

        mfma = stinkytofu.MFMA(
            instType="f32",
            accType="f32",
            m=16,
            n=16,
            k=16,
            blocks=1,
            mfma1k=False,
            acc=acc,
            a=v0,
            b=v1,
            comment="16x16x16 MFMA",
        )
        module.add(mfma)

        dump = module.dump()
        assert "MFMA" in dump
        assert "16x16x16 MFMA" in dump

    def test_label(self):
        module = stinkytofu.LogicalModule("label_test")

        label = stinkytofu.Label("loop_start")
        module.add(label)

        dump = module.dump()
        assert "loop_start:" in dump

    def test_tensor_load_to_lds(self):
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)

        module = stinkytofu.LogicalModule("tensor_test")

        tensor_load = stinkytofu.TensorLoadToLds(v0, v1, comment="Load tensor")
        module.add(tensor_load)

        dump = module.dump()
        assert "TensorLoadToLds" in dump
        assert "Load tensor" in dump


class TestMemoryManagement:
    """Test that shared_ptr memory management works correctly"""

    def test_module_destruction(self):
        """Test that modules can be created and destroyed without memory leaks"""
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)
        v2 = stinkytofu.vgpr(2)

        # Create and destroy multiple modules
        for i in range(10):
            module = stinkytofu.LogicalModule(f"test_{i}")
            for j in range(10):
                module.add(stinkytofu.VAddF32(v0, v1, v2))
            del module

        # If we get here without segfault, memory management works!
        assert True

    def test_instruction_reuse(self):
        """Test that instructions can be shared across modules"""
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)
        v2 = stinkytofu.vgpr(2)

        # Create instruction
        inst = stinkytofu.VAddF32(v0, v1, v2, comment="shared instruction")

        # Add to multiple modules
        module1 = stinkytofu.LogicalModule("module1")
        module2 = stinkytofu.LogicalModule("module2")

        module1.add(inst)
        module2.add(inst)

        # Both should contain the instruction
        assert "shared instruction" in module1.dump()
        assert "shared instruction" in module2.dump()


class TestComplexKernel:
    """Test building a more complex kernel"""

    def test_gemm_like_kernel(self):
        """Build a simple GEMM-like kernel structure"""
        v0 = stinkytofu.vgpr(0)
        v1 = stinkytofu.vgpr(1)
        v2 = stinkytofu.vgpr(2)
        v3 = stinkytofu.vgpr(3)
        acc = stinkytofu.accvgpr(0, 4)

        module = stinkytofu.LogicalModule("simple_gemm")

        # Load data
        module.add(stinkytofu.Label("load_data"))
        module.add(stinkytofu.DSLoadB32(v0, v1, comment="Load A"))
        module.add(stinkytofu.DSLoadB32(v2, v3, comment="Load B"))

        # Compute loop
        module.add(stinkytofu.Label("compute_loop"))
        module.add(
            stinkytofu.MFMA(
                "f32",
                "f32",
                16,
                16,
                16,
                1,
                False,
                acc,
                v0,
                v2,
                comment="Matrix multiply",
            )
        )
        module.add(stinkytofu.VAddF32(v0, v0, v1, comment="Update A ptr"))
        module.add(stinkytofu.VAddF32(v2, v2, v3, comment="Update B ptr"))

        # Store result
        module.add(stinkytofu.Label("store_result"))
        module.add(stinkytofu.VMovB32(v0, acc, comment="Move result"))
        module.add(stinkytofu.DSStoreB32(v1, v0, comment="Store C"))

        dump = module.dump()
        assert "simple_gemm" in dump
        assert "load_data:" in dump
        assert "compute_loop:" in dump
        assert "store_result:" in dump
        assert "Instructions: 10" in dump


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
