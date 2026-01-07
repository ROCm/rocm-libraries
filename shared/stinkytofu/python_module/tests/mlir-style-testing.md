# MLIR-Style Testing for StinkyTofu

This document describes the MLIR-inspired testing methodology for StinkyTofu's Python module.

## Overview

Following MLIR's testing practices, we implement three levels of testing:

1. **Pattern Matching Tests** (FileCheck-style)
2. **Assembly Validation Tests** (llvm-mc integration)
3. **Regression Tests** (Golden outputs)

## 1. FileCheck-Style Pattern Matching

### What is FileCheck?

FileCheck is LLVM's tool for verifying that program output matches expected patterns. It's heavily used in MLIR for testing transformations and code generation.

### Our Implementation

We provide a Python `FileCheck` class that supports:

- `CHECK`: Pattern must appear in output
- `CHECK-NEXT`: Pattern must appear on next non-empty line
- `CHECK-SAME`: Pattern must appear on same line as previous match
- `CHECK-NOT`: Pattern must NOT appear
- `CHECK-DAG`: Patterns can appear in any order
- `CHECK-COUNT`: Pattern must appear exactly N times
- `CHECK-LABEL`: Find labels/markers

### Example Test

```python
from test_filecheck import FileCheck
from stinkytofu import StinkyAsmIR, vgpr

def test_instruction_pattern():
    st = StinkyAsmIR([9, 4, 2])
    module = st.createIRList("test")

    module.add(st.VAddU32(vgpr(0), vgpr(1), vgpr(2), "add"))
    module.add(st.VMulF32(vgpr(3), vgpr(0), vgpr(4), "mul"))

    asm = module.emitAssembly(emit_comments=True)

    # Verify instruction sequence
    checker = FileCheck(asm)
    checker.check("v_add_u32")      # CHECK: v_add_u32
    checker.check_same("v0")        # CHECK-SAME: v0
    checker.check("v_mul_f32")      # CHECK: v_mul_f32
    checker.check_not("v_invalid")  # CHECK-NOT: v_invalid
```

### Writing FileCheck Tests

**Best Practices:**

1. **Be Specific**: Check for complete patterns, not just substrings
   ```python
   # Good
   checker.check_regex(r'v_add_u32\s+v0,\s+v1,\s+v2')

   # Less specific (but sometimes useful)
   checker.check("v_add_u32")
   ```

2. **Check Sequences**: Use `check()` followed by `check()` for ordered patterns
   ```python
   checker.check("v_add_u32")
   checker.check("v_mul_f32")  # Must appear after add
   ```

3. **Check Same Line**: Use `check_same()` for patterns on same line
   ```python
   checker.check("v_add_u32")
   checker.check_same("v0")    # v0 must be on the v_add_u32 line
   checker.check_same("v1")
   ```

4. **Verify Absence**: Use `check_not()` for patterns that shouldn't appear
   ```python
   checker.check_not("error")
   checker.check_not("v_invalid_inst")
   ```

5. **Unordered Patterns**: Use `check_dag()` when order doesn't matter
   ```python
   # These can appear in any order
   checker.check_dag(["v_add_u32", "v_mul_f32", "s_barrier"])
   ```

## 2. Assembly Validation with llvm-mc

### Purpose

Verify that generated assembly is syntactically valid and can be assembled by the actual AMD GPU assembler.

### How It Works

We use `llvm-mc` (LLVM's machine code assembler) to:
1. Parse generated assembly
2. Assemble it to object code
3. Report any errors

### Example Test

```python
import pytest
from pathlib import Path

class TestAssemblyValidity:
    @pytest.mark.skipif(
        not Path("/usr/bin/llvm-mc").exists(),
        reason="llvm-mc not available"
    )
    def test_mfma_assembles(self):
        """Verify MFMA instruction can be assembled."""
        st = StinkyAsmIR([9, 4, 2])
        module = st.createIRList("test")

        module.add(st.createMFMA(
            instType="f32",
            accType="f32",
            m=16, n=16, k=4,
            blocks=1,
            mfma1k=False,
            acc=acc(0, 4),
            a=vgpr(0, 4),
            b=vgpr(4, 4),
            acc2=acc(0, 4),
            comment=""
        ))

        asm = module.emitAssembly()

        # Verify it assembles
        assert assemble_with_llvm_mc(asm)
```

### Running Assembly Validation

```bash
# Run only assembly validation tests
pytest tests/test_assembly_validation.py::TestAssemblyValidity -v

# Skip if llvm-mc not available
pytest tests/ -v -k "not AssemblyValidity"
```

## 3. Regression Testing (Golden Outputs)

### Purpose

Ensure assembly generation doesn't change unexpectedly between versions.

### Strategy

1. **Golden Files**: Save expected assembly outputs
2. **Comparison**: Compare new outputs against golden files
3. **Update**: Explicitly update golden files when changes are intentional

### Example Test

```python
def test_basic_kernel_regression():
    """Compare against saved golden output."""
    st = StinkyAsmIR([9, 4, 2])
    module = st.createIRList("kernel")

    # Build kernel
    module.add(st.VAddU32(vgpr(0), vgpr(1), vgpr(2), ""))
    module.add(st.VMulF32(vgpr(3), vgpr(0), vgpr(4), ""))

    asm = module.emitAssembly()

    # Compare against golden file
    golden_path = Path("golden/basic_kernel.s")
    if golden_path.exists():
        golden = golden_path.read_text()
        assert asm == golden, "Assembly changed from golden output"
    else:
        # Create golden file if it doesn't exist
        golden_path.write_text(asm)
```

### Managing Golden Files

```bash
# Run tests and save golden outputs
STINKYTOFU_UPDATE_GOLDEN=1 pytest tests/

# Compare against golden outputs
pytest tests/test_assembly_validation.py::TestRegressionGoldenOutputs -v
```

## 4. Instruction-Level Testing

### Testing Individual Instructions

For each instruction, verify:
1. Correct mnemonic
2. Correct operand order
3. Correct register encoding
4. Correct modifiers/flags

### Example

```python
def test_valu_instruction_format():
    """Verify VALU instruction format."""
    st = StinkyAsmIR([9, 4, 2])
    module = st.createIRList("test")

    module.add(st.VAddU32(vgpr(10), vgpr(20), vgpr(30), "test"))
    asm = module.emitAssembly(emit_comments=True)

    # Extract and verify instruction
    checker = FileCheck(asm)
    match = checker.check_regex(r'v_add_u32\s+(v\d+),\s+(v\d+),\s+(v\d+)')

    # Verify register numbers
    assert match.group(1) == "v10"
    assert match.group(2) == "v20"
    assert match.group(3) == "v30"

    # Verify comment
    checker.reset()
    checker.check("test")
```

## 5. Architecture-Specific Testing

### Testing Across Architectures

Verify that:
1. Instructions work on supported architectures
2. Instructions fail gracefully on unsupported architectures
3. Composite instructions lower correctly

### Example

```python
@pytest.mark.parametrize("arch,arch_name,supported", [
    ([9, 4, 2], "gfx942", True),
    ([9, 5, 0], "gfx950", True),
    ([12, 5, 0], "gfx1250", True),
])
def test_instruction_architecture_support(arch, arch_name, supported):
    """Test instruction across architectures."""
    st = StinkyAsmIR(arch)
    module = st.createIRList(f"test_{arch_name}")

    if supported:
        # Should work
        module.add(st.VAddU32(vgpr(0), vgpr(1), vgpr(2), ""))
        asm = module.emitAssembly()
        assert "v_add_u32" in asm
    else:
        # Should fail gracefully
        with pytest.raises(RuntimeError):
            module.add(st.VAddU32(vgpr(0), vgpr(1), vgpr(2), ""))
```

## 6. Composite Instruction Testing

### Testing Instruction Lowering

Composite instructions may expand to multiple instructions depending on architecture support.

### Example

```python
def test_composite_lowering():
    """Test composite instruction lowering."""
    st = StinkyAsmIR([9, 4, 2])
    module = st.createIRList("test")

    # VAddPKF32 might lower to 1 or 2 instructions
    insts = st.VAddPKF32(vgpr(0), vgpr(1), vgpr(2), "packed add")
    module.add(insts)

    asm = module.emitAssembly(emit_comments=True)

    checker = FileCheck(asm)

    # Either has packed instruction or two regular adds
    try:
        checker.check("v_pk_add_f32")
        checker.check_count("v_pk_add_f32", 1)
    except:
        # Fallback to two v_add_f32
        checker.reset()
        checker.check_count("v_add_f32", 2)
```

## 7. MFMA Testing

### Testing Matrix Instructions

MFMA instructions are complex and architecture-specific. Test:
1. Correct mnemonic generation
2. Correct register allocation
3. Correct dimension encoding
4. Correct data type handling

### Example

```python
def test_mfma_instruction_format():
    """Test MFMA instruction format."""
    st = StinkyAsmIR([9, 4, 2])
    module = st.createIRList("test")

    module.add(st.createMFMA(
        instType="bf16",
        accType="f32",
        m=16, n=16, k=16,
        blocks=1,
        mfma1k=False,
        acc=acc(0, 4),
        a=vgpr(0, 4),
        b=vgpr(4, 4),
        acc2=acc(0, 4),
        comment="mfma"
    ))

    asm = module.emitAssembly(emit_comments=True)

    checker = FileCheck(asm)

    # CHECK: v_mfma_{{bf16|f32}}_16x16x16
    checker.check_regex(r'v_mfma_\w+_16x16x16')

    # CHECK-SAME: a[0:3], v[0:3], v[4:7], a[0:3]
    checker.check_same("a[0:3]")
    checker.check_same("v[0:3]")
    checker.check_same("v[4:7]")
    checker.check_same("a[0:3]")

    # CHECK: mfma
    checker.check("mfma")
```

## 8. Error Handling Testing

### Testing Error Cases

Verify that invalid inputs produce appropriate errors.

### Example

```python
def test_invalid_mfma_dimensions():
    """Test MFMA with invalid dimensions."""
    st = StinkyAsmIR([9, 4, 2])
    module = st.createIRList("test")

    with pytest.raises(RuntimeError, match="Invalid MFMA dimensions"):
        module.add(st.createMFMA(
            instType="f32",
            accType="f32",
            m=99, n=99, k=99,  # Invalid
            blocks=1,
            mfma1k=False,
            acc=acc(0, 4),
            a=vgpr(0, 4),
            b=vgpr(4, 4),
            acc2=acc(0, 4),
            comment=""
        ))
```

## Running Tests

### Run All Tests

```bash
cd /data0/yangwen/rocm-libraries/shared/stinkytofu
PYTHONPATH=build/lib:$PYTHONPATH python3 -m pytest python_module/tests/ -v
```

### Run FileCheck Tests Only

```bash
pytest python_module/tests/test_assembly_validation.py::TestFileCheckPatterns -v
```

### Run Assembly Validation Tests

```bash
pytest python_module/tests/test_assembly_validation.py::TestAssemblyValidity -v
```

### Run Regression Tests

```bash
pytest python_module/tests/test_assembly_validation.py::TestRegressionGoldenOutputs -v
```

### Run with Coverage

```bash
pytest python_module/tests/ --cov=stinkytofu --cov-report=html
```

## Comparison with MLIR Testing

| MLIR | StinkyTofu | Purpose |
|------|------------|---------|
| `lit` | `pytest` | Test runner |
| `FileCheck` | `test_filecheck.FileCheck` | Pattern matching |
| `mlir-opt` | `StinkyTofu.emitAssembly()` | IR transformation |
| `llvm-mc` | `llvm-mc` (same) | Assembly validation |
| `.mlir` test files | `.py` test files | Test format |
| `// RUN:` directives | `pytest` functions | Test execution |
| `// CHECK:` comments | `checker.check()` | Verification |

## Best Practices

1. **Test at Multiple Levels**
   - Unit tests for individual instructions
   - Integration tests for instruction sequences
   - End-to-end tests for complete kernels

2. **Use Appropriate Assertions**
   - FileCheck for pattern matching
   - Direct assertions for exact matches
   - Regex for flexible patterns

3. **Test Error Cases**
   - Invalid operands
   - Unsupported architectures
   - Malformed instructions

4. **Test Across Architectures**
   - Use parametrized tests
   - Verify architecture-specific behavior
   - Test graceful degradation

5. **Maintain Golden Files**
   - Keep them in version control
   - Update explicitly when behavior changes
   - Document why golden output changed

6. **Use Markers**
   ```python
   @pytest.mark.slow
   @pytest.mark.requires_llvm_mc
   @pytest.mark.architecture("gfx942")
   ```

## Continuous Integration

### Example GitHub Actions Workflow

```yaml
name: StinkyTofu Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Install dependencies
      run: |
        apt-get update
        apt-get install -y llvm-15 python3-pytest

    - name: Build StinkyTofu
      run: |
        mkdir build && cd build
        cmake .. -DSTINKYTOFU_BUILD_PYTHON=ON
        make -j

    - name: Run tests
      run: |
        cd shared/stinkytofu
        PYTHONPATH=build/lib:$PYTHONPATH \
          python3 -m pytest python_module/tests/ -v \
          --cov=stinkytofu --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Resources

- [MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/)
- [LLVM FileCheck Documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
- [pytest Documentation](https://docs.pytest.org/)
- [llvm-mc Documentation](https://llvm.org/docs/CommandGuide/llvm-mc.html)

