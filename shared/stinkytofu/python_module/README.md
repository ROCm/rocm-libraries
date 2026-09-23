# StinkyTofu Python Module

A Python interface for generating GPU assembly code for AMD GPUs using the StinkyTofu IR.

## Features

- **Instance-based design** - No global state, each builder is independent
- **Explicit module management** - Create named IR modules and add instructions explicitly
- **Architecture-specific** - Target different GPU architectures (e.g., gfx1250)
- **Thread-safe** - Multiple instances can be used concurrently

## Building

### Prerequisites

- CMake 3.16+
- Python 3.6+
- nanobind: `pip install nanobind`

### Build Instructions

From the repository root:

```bash
cd shared/stinkytofu
mkdir -p build && cd build
cmake .. -DSTINKYTOFU_BUILD_PYTHON=ON
cmake --build . --target stinkytofu_python -j12
```

The Python module will be built to `build/lib/stinkytofu.*.so`.

### Add to Python Path

```bash
export PYTHONPATH=/path/to/build/lib:$PYTHONPATH
```

## Examples

### Basic Usage

```python
from stinkytofu import StinkyAsmIR, vgpr, sgpr

# Create builder for gfx1250
st = StinkyAsmIR([12, 5, 0])

# Create a module to hold instructions
module = st.createIRList("my_kernel")

# Create instructions and add to module
module.add(st.VAddU32(vgpr(0), vgpr(1), vgpr(2), "add"))
module.add(st.VMulF32(vgpr(3), vgpr(0), vgpr(4), "multiply"))
module.add(st.SBarrier("sync"))

# Emit assembly
print(module.emitAssembly())
```

### Register Ranges

```python
from stinkytofu import StinkyAsmIR, vgpr, acc

st = StinkyAsmIR([12, 5, 0])
module = st.createIRList("mfma_kernel")

# Use count parameter for register ranges
# createMFMA for Matrix-FMA instructions
module.add(st.createMFMA(
    instType="f32",  # input type
    accType="f32",   # accumulator type
    m=16, n=16, k=4, # dimensions
    blocks=1,        # number of blocks
    mfma1k=False,    # not 1k variant
    acc=acc(0, 4),   # dst: a[0:3] - 4 registers
    a=vgpr(0, 4),    # matrix A: v[0:3] - 4 registers
    b=vgpr(4, 4),    # matrix B: v[4:7] - 4 registers
    acc2=acc(0, 4),  # input acc: a[0:3] - 4 registers
    comment="MFMA"
))

print(module.emitAssembly())
```

### Composite Instructions (Architecture-Aware Lowering)

Some instructions may not be supported on all architectures. StinkyTofu automatically handles this by lowering unsupported instructions into equivalent sequences:

```python
from stinkytofu import StinkyAsmIR, vgpr

st = StinkyAsmIR([12, 5, 0])  # gfx1250 supports v_pk_add_f32
module = st.createIRList("composite_example")

# VAddPKF32 demonstrates the composite instruction pattern:
# - On architectures with v_pk_add_f32: returns 1 instruction
# - On older architectures: returns 2 v_add_f32 instructions (lowered)
insts = st.VAddPKF32(vgpr(0, 2), vgpr(2, 2), vgpr(4, 2), "packed add")
print(f"Generated {len(insts)} instruction(s)")  # Prints "1" on gfx1250

# module.add() accepts lists of instructions
module.add(insts)

print(module.emitAssembly())
# Output on gfx1250: v_pk_add_f32 v[0:1], v[2:3], v[4:5] // packed add
```

**Note:** All instruction creation methods return a list of instructions. Most of the time this list contains a single instruction, but for composite/lowered instructions, it may contain multiple instructions depending on the target architecture. The `module.add()` method accepts these lists directly.

### Multiple Modules

```python
from stinkytofu import StinkyAsmIR, vgpr

st = StinkyAsmIR([12, 5, 0])

# Create multiple kernels
kernel_a = st.createIRList("kernel_a")
kernel_b = st.createIRList("kernel_b")

# Add different instructions to each
kernel_a.add(st.VAddU32(vgpr(0), vgpr(1), vgpr(2)))
kernel_b.add(st.VMulF32(vgpr(0), vgpr(1), vgpr(2)))

# Emit separately
print("Kernel A:")
print(kernel_a.emitAssembly())

print("\nKernel B:")
print(kernel_b.emitAssembly())
```

### Conditional Assembly

```python
from stinkytofu import StinkyAsmIR, vgpr

st = StinkyAsmIR([12, 5, 0])
module = st.createIRList("conditional_kernel")

# Create instruction but don't add yet
inst = st.VAddU32(vgpr(0), vgpr(1), vgpr(2), "conditional add")

# Add conditionally
use_optimization = True
if use_optimization:
    module.add(inst)

module.add(st.SBarrier("sync"))
print(module.emitAssembly())
```

More examples in `examples/basic_usage.py`.

## Testing

The test suite uses **pytest** for better test organization and reporting.

### Quick Start

```bash
# Install dependencies
pip install -r python_module/tests/requirements.txt

# Run all tests
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -v
```

### Detailed Testing Guide

For comprehensive testing documentation including:
- Running specific tests and test categories
- Using fixtures and parametrized tests
- Coverage reports and parallel execution
- Debugging and CI/CD integration

See **[tests/testing.md](tests/testing.md)** for the complete testing guide.

## API Reference

### Main Classes

- **`StinkyAsmIR(arch)`** - Create builder for target architecture
  - `arch`: List `[major, minor, stepping]`, e.g., `[12, 5, 0]` for gfx1250

- **`IRListModule`** - Container for instructions
  - `add(inst)` - Add instruction to module (returns inst for chaining)
  - `emitAssembly(emit_cycle_info=False, emit_user_comments=True)` - Emit assembly code
  - `getName()` - Get module name

### Register Helpers

- **`vgpr(idx, count=1)`** - Vector general purpose register
- **`sgpr(idx, count=1)`** - Scalar general purpose register
- **`acc(idx, count=1)`** - Accumulator register

### Instruction Methods (on StinkyAsmIR)

Vector ALU:
- `VAddU32(dst, src0, src1, comment="")`
- `VMulF32(dst, src0, src1, comment="")`

Scalar:
- `SAbsI32(dst, src, comment="")`
- `SBarrier(comment="")`

Matrix (Generic Creation):
- `createMFMA(output_type, input_type, M, N, K, dst, src0, src1, src2, comment="")`
- `createMXMFMA(output_type, input_type, M, N, K, dst, src0, src1, src2, comment="")`
- `createSMFMA(output_type, input_type, M, N, K, dst, src0, src1, src2, index, comment="")`

See [docs/mfma-instructions.md](docs/mfma-instructions.md) for detailed MFMA documentation.

*All instruction methods return a list of `StinkyInstruction*` pointers.*

## License

MIT License - See LICENSE file for details.

Copyright (C) 2025 Advanced Micro Devices, Inc.
