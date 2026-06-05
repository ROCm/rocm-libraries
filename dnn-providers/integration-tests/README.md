# Integration Tests

Integration tests validate hipDNN provider plugins (engine libraries such as
`libmiopen_plugin.so` or `libhipblaslt_plugin.so`) by running graphs through
the plugin and comparing results against a reference executor.

## Quick Start

```bash
# Standalone build — point to the plugin explicitly
./bin/hipdnn_integration_tests \
  --test-article /path/to/libmiopen_plugin.so

# Superbuild — plugin discovery is automatic
./bin/hipdnn_integration_tests
```

## Test Tiers

| Tier | GTest prefix | Shape catalog | CI cadence | Timeout |
|------|-------------|---------------|------------|---------|
| Smoke | `Smoke` *(or no prefix)* — **catch-all** | `getSmall*()` + standalone tests | Every commit / PR | 600s (10 min) |
| Standard | `Standard` | `getMedium*()` | PR gate | 1800s (30 min) |
| Comprehensive | `Comprehensive` | `getLargeEdge*()` | Nightly | 3600s (60 min) |
| Full | `Full` | `getLargeStress*()` | Weekly | 7200s (120 min) |

Timeouts can be overridden per binary via `SMOKE_TIMEOUT`, `STANDARD_TIMEOUT`,
`COMPREHENSIVE_TIMEOUT`, and `FULL_TIMEOUT` arguments to
`add_tiered_test_target()`.

### Smoke is a catch-all

The smoke ctest entry uses an exclusion filter
(`-Standard*:Comprehensive*:Full*`). Every test that does **not** start with
`Standard`, `Comprehensive`, or `Full` runs in smoke automatically:

```cpp
// Runs in smoke — has Smoke prefix
INSTANTIATE_TEST_SUITE_P(Smoke, MyFixture, ...);

// Also runs in smoke — no tier prefix, caught by the exclusion filter
TEST(MyFeature, BasicBehavior) { ... }
TEST_F(MyFixture, EdgeCase) { ... }
```

If smoke starts timing out, a large shape is missing its tier prefix.

### How tiers cascade

Each higher ctest label includes all lower tiers:

```
ctest -L quick           →  [smoke]
ctest -L standard        →  [smoke + standard]
ctest -L comprehensive   →  [smoke + standard + comprehensive]
ctest -L full            →  [smoke + standard + comprehensive + full]
```

> **Note:** The ctest label uses `quick` for the smoke tier
> (backlog: rename to `smoke` for consistency).

## Running Tests

| Method | Command | Use case |
|--------|---------|----------|
| ctest | `ctest -L quick` | CI and local tier runs |
| ninja | `ninja unit-check` / `ninja check` | Local shortcut (smoke / all) |
| Direct | `./bin/hipdnn_gpu_ref_tests --gtest_filter="Smoke*"` | Debugging a specific test |

> **GTest filter syntax:** `-Standard*:Comprehensive*:Full*` uses a single
> leading dash. In GTest, only the first `-` starts the negative section.
> Using `:-` between patterns does **not** negate — the dash becomes literal.

## Adding a New Operation

### Directory layout

```
tests/
  gpu_ref/
    ConvShapeCase.hpp              # Shape struct + byTag()
    ConvShapeCatalog.hpp           # getSmall/getMedium/getLargeEdge/getLargeStress
    TestGpuFpReferenceConvolution.cpp
  my_new_op/
    MyNewOpShapeCase.hpp
    MyNewOpShapeCatalog.hpp
    TestMyNewOp.cpp
```

### Step 1 — CMake registration

Register the test binary in `tests/CMakeLists.txt`:

```cmake
add_tiered_test_target(hipdnn_my_new_op_tests ${CMAKE_CURRENT_BINARY_DIR})
```

### Step 2 — Shape catalog

Create a shape catalog following the tier pattern in
[`tests/gpu_ref/ConvShapeCatalog.hpp`](tests/gpu_ref/ConvShapeCatalog.hpp).

### Step 3 — C++ test tiers

New parameterized test suites **must** define all four tiers:

```cpp
INSTANTIATE_TEST_SUITE_P(Smoke,         MyNewOp2dTestFp32, ::testing::ValuesIn(getSmallCases()),     byTag());
INSTANTIATE_TEST_SUITE_P(Standard,      MyNewOp2dTestFp32, ::testing::ValuesIn(getMediumCases()),    byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive, MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeEdgeCases()), byTag());
INSTANTIATE_TEST_SUITE_P(Full,          MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeStressCases()), byTag());
```

`byTag()` uses the shape's `tag` field as the test name so failures show
`Smoke/MyOp2dTestFp32.Runs/n8c64k32_f3x3_s1_p1` instead of `.../7`.

### Adding a new convolution shape

Add to the appropriate function in
[`tests/gpu_ref/ConvShapeCatalog.hpp`](tests/gpu_ref/ConvShapeCatalog.hpp).
Existing `INSTANTIATE_TEST_SUITE_P` calls pick up new shapes automatically.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Engine 'X' is not loaded` | Pass `--test-article /path/to/plugin.so`, or run from a superbuild |
| Smoke tier timing out | A shape is missing its tier prefix — check `INSTANTIATE_TEST_SUITE_P` prefixes |
| `No tests matched the filter` | Use a single `-` for negative filters: `-Standard*:Comprehensive*:Full*` |
