# RPP Test Suite

A standalone [GoogleTest](https://github.com/google/googletest) correctness suite for RPP tensor operations. Each op is checked against an independent, host-computed reference model across data types, layouts, ROIs, and both backends (HOST + HIP). It builds standalone against an *installed* RPP, or in-tree against an RPP build.

## Requirements
- An RPP install discoverable under `ROCM_PATH` (not needed for an in-tree build).
- `ROCM_PATH` pointing at your ROCm install (defaults to `/opt/rocm` if unset).

GoogleTest is fetched automatically at configure time (pinned to a tag; see `cmake/gtest.cmake`). Configure with `-DRPP_TEST_USE_SYSTEM_GTEST=ON` to use an installed GoogleTest instead, for builds with no network access. HIP tests are compiled only when the RPP being built against has the HIP backend.

## Building
```shell
mkdir build && cd build
cmake ..
cmake --build . --parallel
```

Configured in-tree (as a subdirectory of an RPP build) the suite reuses that build's `rpp` target instead of finding an installed one; no other setup differs.

## Running
```shell
./rpp_tests                        # run every test
./rpp_tests --gtest_list_tests     # list tests without running
./rpp_tests --help                 # full GTest options
ctest                              # run via CTest (each case registered individually)
```

> [!NOTE]
> Running tests via `ctest` ensures each unit is isolated to its own process. While this increases the runtime of the test suite significantly (due to per-unit setup overhead), it is preferred for isolating segfaults/GPU crashes to a single test case, rather than taking down other unit tests with it.

## Test names & filtering

Every case has a structured, greppable name:

```
{Domain}_{Category}/{Op}Test.{Intent}/{Backend}_{DType}_{Layout}_{Roi}_{Size}
```

e.g. `Image_Color/BrightnessTest.Correctness/HIP_U8toU8_PKD3_FullRoi_2x36x48`.

Select subsets with `--gtest_filter` (wildcard `*`, `:`-separated patterns):

| Goal | Filter |
|------|--------|
| One operation | `--gtest_filter='*BrightnessTest*'` |
| One category | `--gtest_filter='Image_Color/*'` |
| HIP cases only | `--gtest_filter='*/HIP_*'` |
| All F32 PKD3, any op | `--gtest_filter='*F32*PKD3*'` |

## Known-defect skip list

Cases that fail against a documented kernel defect (or a non-reproducible result) are skipped rather
than left red, so a normal run stays green. The list lives in `src/framework/skip_list.hpp` as
GTest-filter-glob patterns; fixing a kernel means deleting the matching entry in the same change.
Set `RPP_TEST_NO_SKIP_LIST=1` to run the listed cases anyway and check whether any have been fixed.

## Layout
```
src/
  main.cpp     entry point; installs the suite's console reporter
  framework/   shared harness. config grid, backend memory, comparators
  reference/   independent golden models for comparison
  tests/       the tests themselves
    core/      core RPP tests unrelated to op correctness
    image/     image ops, grouped by category (color, geometric, ...)
    misc/      misc ops, grouped by category
    voxel/     voxel ops, grouped by category
```

Adding an op = a `reference/<op>_ref.hpp` golden model plus a short `TEST_P` under `tests/`.
