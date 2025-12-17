# StinkyTofu Testing Framework

This directory contains the testing framework for StinkyTofu's Python module, inspired by MLIR's testing methodology.

## Test Files

- **`test_assembly_validation.py`** - **Main comprehensive test suite** (MLIR-style with FileCheck patterns)
  - Basic instruction tests (VALU, scalar, register ranges)
  - MFMA/WMMA/SMFMA instruction tests
  - Composite instruction tests
  - gfx1250-specific instruction tests
  - Multi-architecture tests
  - Error handling tests
  - Regression tests with golden outputs
- **`test_filecheck.py`** - FileCheck-style pattern matching utility (MLIR-inspired)
- **`mlir-style-testing.md`** - Comprehensive guide to MLIR-style testing methodology
- **`testing.md`** - General testing guide

## Quick Start

```bash
# Assume you have build stinkytofu_python in shared/stinkytofu/build
# See: python_module/README.md

# Run all tests
cd /data0/yangwen/rocm-libraries/shared/stinkytofu
PYTHONPATH=build/python_module:$PYTHONPATH python3 -m pytest python_module/tests/ -v

# Run main test suite (all comprehensive tests)
pytest python_module/tests/test_assembly_validation.py -v

# Run specific test categories
pytest python_module/tests/test_assembly_validation.py::TestBasicInstructions -v
pytest python_module/tests/test_assembly_validation.py::TestBasicMFMA -v
pytest python_module/tests/test_assembly_validation.py::TestFileCheckPatterns -v

# Run tests by marker
pytest python_module/tests/ -v -m mfma  # MFMA tests only
pytest python_module/tests/ -v -m sparse  # Sparse MFMA tests
pytest python_module/tests/ -v -m architecture  # Architecture tests

# Run with coverage
pytest python_module/tests/ --cov=stinkytofu --cov-report=html
```

## MLIR-Inspired Testing

We follow MLIR's robust testing practices:

### 1. FileCheck Pattern Matching

```python
from test_filecheck import FileCheck

asm = module.emitAssembly()
checker = FileCheck(asm)

# CHECK: v_add_u32
checker.check("v_add_u32")

# CHECK-SAME: v[0]
checker.check_same("v[0]")

# CHECK-NOT: invalid
checker.check_not("invalid")
```

### 2. Instruction Validation

- Verify instruction mnemonics
- Check operand order and types
- Validate register allocation
- Test architecture-specific behavior

### 3. Regression Testing

- Compare outputs against golden files
- Catch unintended changes
- Document intentional modifications

## Test Organization

```
tests/
├── test_assembly_validation.py  # Main comprehensive test suite (MLIR-style)
│   ├── TestFileCheckPatterns    # FileCheck pattern matching tests (8 tests)
│   ├── TestBasicInstructions    # Basic VALU/scalar instructions (6 tests)
│   ├── TestBasicMFMA            # MFMA instruction tests (4 tests)
│   ├── TestMFMAWithBlocks       # Multi-block MFMA tests (2 tests)
│   ├── TestWMMA                 # WMMA (RDNA) tests (1 test)
│   ├── TestSparseMFMA           # Sparse MFMA tests (3 tests)
│   ├── TestMFMAKernel           # Complete kernel tests (2 tests)
│   ├── TestMFMAErrorHandling    # Error handling tests (2 tests)
│   ├── TestGfx1250ScalarInstructions    # gfx1250 scalar tests (4 tests)
│   ├── TestGfx1250VectorInstructions    # gfx1250 vector tests (2 tests)
│   ├── TestGfx1250InstructionsOnOlderArchitectures  # gfx1250 error tests (2 tests)
│   ├── TestGfx1250AllInstructions       # gfx1250 integration tests (2 tests)
│   ├── TestAssemblyValidity     # llvm-mc validation (2 tests, skipped)
│   ├── TestInstructionExtraction # Parsing utilities (4 tests)
│   ├── TestRegressionGoldenOutputs # Regression tests (2 tests)
│   └── Parametrized tests       # Architecture/instruction variants (11 tests)
├── test_filecheck.py            # FileCheck utility implementation
├── mlir-style-testing.md        # Detailed methodology guide
├── testing.md                   # General testing guide
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Test dependencies
├── conftest.py                  # Shared fixtures
└── README.md                    # This file

Total: 95 tests in one comprehensive file
- 54 original tests (basic, MFMA, gfx1250)
- 20 branch instruction tests
- 21 compare instruction tests (scalar, vector, cmpx)
```

## Test Markers

```bash
# Architecture-specific tests
pytest -m architecture

# Composite instruction tests
pytest -m composite

# MFMA tests
pytest -m mfma

# Sparse operation tests
pytest -m sparse
```

## Documentation

- **[mlir-style-testing.md](mlir-style-testing.md)** - Complete MLIR-style testing guide
- **[testing.md](testing.md)** - General testing guide with examples
- **[../README.md](../README.md)** - Python module README

## Comparison with MLIR

| MLIR | StinkyTofu | Purpose |
|------|------------|---------|
| `lit` | `pytest` | Test runner |
| `FileCheck` | `test_filecheck.FileCheck` | Pattern matching |
| `mlir-opt` | `StinkyTofu.emitAssembly()` | Code generation |
| `.mlir` files | `.py` test files | Test format |
| `// CHECK:` | `checker.check()` | Verification |

## Contributing Tests

When adding new features:

1. **Add FileCheck tests** for output verification
2. **Test across architectures** using parametrized tests
3. **Test error cases** for invalid inputs
4. **Update golden files** when behavior changes intentionally
5. **Document** what you're testing in the test name and docstring

Example:

```python
def test_new_instruction_format():
    """
    Test NEW_INSTRUCTION generates correct assembly.

    Verifies:
    - Correct mnemonic
    - Correct operand order
    - Correct register encoding
    """
    st = StinkyAsmIR([9, 4, 2])
    module = st.createIRList("test")

    module.add(st.NewInstruction(...))
    asm = module.emitAssembly(emit_comments=True)

    checker = FileCheck(asm)
    checker.check("expected_mnemonic")
    checker.check_same("expected_operands")
```

## Resources

- [MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/)
- [LLVM FileCheck Documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
- [pytest Documentation](https://docs.pytest.org/)

