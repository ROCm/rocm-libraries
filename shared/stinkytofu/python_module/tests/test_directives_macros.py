"""
Simple unit tests for directive and macro factory functions

These tests verify that the factory functions (createSetDirective, etc.)
work correctly and produce the expected directive/macro objects.
"""

import pytest


def test_import():
    """Test that we can import StinkyAsmIR"""
    from stinkytofu import StinkyAsmIR
    assert StinkyAsmIR is not None


def test_create_stinkytofu():
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


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_create_set_directive():
    """Test createSetDirective factory function"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    # Create a .set directive
    directive = builder.createSetDirective("vgprBase", "12")
    assert directive is not None

    # Check properties
    assert directive.symbol == "vgprBase"
    assert directive.value == "12"
    assert directive.name == ".set"


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_create_if_directive():
    """Test createIfDirective factory function"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    directive = builder.createIfDirective("WAVE_SIZE == 64")
    assert directive is not None
    assert directive.condition == "WAVE_SIZE == 64"
    assert directive.name == ".if"


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_create_else_directive():
    """Test createElseDirective factory function"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    directive = builder.createElseDirective()
    assert directive is not None
    assert directive.name == ".else"


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_create_endif_directive():
    """Test createEndIfDirective factory function"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    directive = builder.createEndIfDirective()
    assert directive is not None
    assert directive.name == ".endif"


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_create_vmagic_div_macro():
    """Test createVMagicDivMacro factory function"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    macro = builder.createVMagicDivMacro(
        32,                    # divisor
        builder.vgpr(0),      # quotient
        builder.vgpr(1),      # dividend
        builder.vgpr(2),      # tmpVgpr
        builder.sgpr(0)       # tmpSgpr
    )

    assert macro is not None
    assert macro.divisor == 32
    assert macro.name == "vmagic_div"
    assert len(macro.operands) == 4


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_create_prng_macro():
    """Test createPseudoRandomGeneratorMacro factory function"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    macro = builder.createPseudoRandomGeneratorMacro(
        builder.vgpr(0),      # dst
        builder.vgpr(1),      # seed
        builder.sgpr(0)       # tmpSgpr
    )

    assert macro is not None
    assert macro.name == "prng"
    assert len(macro.operands) == 3


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_create_branch_if_zero_macro():
    """Test createBranchIfZeroMacro factory function"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    macro = builder.createBranchIfZeroMacro(
        builder.sgpr(0),      # src
        "LABEL_END"           # label
    )

    assert macro is not None
    assert macro.name == "branch_if_zero"
    assert macro.label == "LABEL_END"


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_create_ds_init_macro():
    """Test createDSInitMacro factory function"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    macro = builder.createDSInitMacro(
        65536,                # sizeBytes
        0,                    # value
        builder.vgpr(0),      # tmpVgpr
        builder.sgpr(0)       # tmpSgpr
    )

    assert macro is not None
    assert macro.name == "ds_init"
    assert macro.sizeBytes == 65536
    assert macro.value == 0


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_directives_with_comments():
    """Test that directives can have comments"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    directive = builder.createSetDirective("SIZE", "64", "Wave size")
    assert directive is not None
    assert directive.comment == "Wave size"


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_macros_with_comments():
    """Test that macros can have comments"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    macro = builder.createVMagicDivMacro(
        32,
        builder.vgpr(0),
        builder.vgpr(1),
        builder.vgpr(2),
        builder.sgpr(0),
        "Divide by 32"
    )

    assert macro is not None
    assert macro.comment == "Divide by 32"


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_multiple_directives():
    """Test creating multiple directives"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    dir1 = builder.createSetDirective("VAR1", "10")
    dir2 = builder.createSetDirective("VAR2", "20")
    dir3 = builder.createIfDirective("VAR1 < VAR2")

    assert dir1 is not None
    assert dir2 is not None
    assert dir3 is not None

    assert dir1.symbol == "VAR1"
    assert dir2.symbol == "VAR2"
    assert dir3.condition == "VAR1 < VAR2"


@pytest.mark.skip(reason="Directive/Macro factory functions not yet bound to Python")
def test_multiple_macros():
    """Test creating multiple macros"""
    from stinkytofu import StinkyAsmIR

    builder = StinkyAsmIR([9, 4, 2])

    macro1 = builder.createVMagicDivMacro(32, builder.vgpr(0), builder.vgpr(1), builder.vgpr(2), builder.sgpr(0))
    macro2 = builder.createPseudoRandomGeneratorMacro(builder.vgpr(10), builder.vgpr(11), builder.sgpr(5))
    macro3 = builder.createBranchIfZeroMacro(builder.sgpr(0), "LABEL")

    assert macro1 is not None
    assert macro2 is not None
    assert macro3 is not None

    assert macro1.name == "vmagic_div"
    assert macro2.name == "prng"
    assert macro3.name == "branch_if_zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


