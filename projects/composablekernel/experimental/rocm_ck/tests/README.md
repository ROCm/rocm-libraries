# rocm_ck Tests

## Testing Strategy

Tests are organized in two tiers, each with a different purpose.

### Tier 1: Host-side unit tests

Compiled with g++ (no HIP). Tests all schema types: Signature, resolve(), GemmSpec, Args, DataType, Layout.
Runs in seconds, no GPU needed. **This is where coverage lives.**

```bash
cd tests/
cmake -B build -S . -G Ninja
ninja -C build
ctest --test-dir build --output-on-failure
```

### Tier 2: GPU runtime tests (examples)

Examples serve as integration tests: they launch kernels, compile device code, and verify numerical
correctness on real hardware. Device compilation validation (type maps, CK Tile template
instantiation) is covered here since device compilation is slow (10-30s per variant).
GPU tests are **not** part of this test suite. They live in `examples/`.

## Test Files

| File | Tier | What it tests |
|------|------|---------------|
| `test_datatype.cpp` | 1 | DataType enum, `data_type_bits()`, `data_type_name()` |
| `test_layout.cpp` | 1 | Layout enum, `layout_name()`, `is_valid_layout_for_rank()` |
| `test_signature.cpp` | 1 | Signature construction, Tensor/Scalar/Op fields |
| `test_resolve.cpp` | 1 | `resolve()` — dtype cascade, rank/layout propagation, FMHA pattern, scalars |
| `test_gemm_spec.cpp` | 1 | `is_valid_mfma()`, `make_spec()`, physical tensor table, epilogue ops |
| `test_args.cpp` | 1 | Args ABI: sizes, offsets, alignment, trivially-copyable |
| `test_physical_tensor.cpp` | 1 | PhysicalTensor, TensorName NTTP string |
| `test_compile_errors.cpp` | — | Placeholder: expected-failure cases (see below) |

## Static Assert Policy

**Unit tests verify behavior; static_asserts protect users.**

Most static_asserts have been migrated out of headers into this test suite. Headers retain
only critical guardrails that produce clear compiler errors when a user misconfigures a kernel:

- ABI size/alignment assertions in `args.hpp` (prevent silent memory corruption)
- `is_valid_mfma()` check inside `make_spec()` (catch invalid tile configs at compile time)
- Device-side assertions in `gemm_dev.hpp` (catch invalid GemmSpec at instantiation)

## Compile-Error Tests (Future)

Some important behaviors are "this must fail to compile" — SSA violations, missing dtype,
invalid MFMA tiles. Testing expected compilation failures is a known gap. Options include:

- CMake `try_compile()` with expected failure
- `#ifdef` guarded `static_assert(false)` patterns
- Compiler-specific `__attribute__((diagnose_if))`

The commented-out cases in `test_compile_errors.cpp` document what should fail and why.
When we implement compile-error testing, that's where to look.

## Running Tests

```bash
# From the tests/ directory:
cd tests/
cmake -B build -S . -G Ninja
ninja -C build
ctest --test-dir build --output-on-failure

# Run a specific test
ctest --test-dir build -R test_resolve --output-on-failure
```
