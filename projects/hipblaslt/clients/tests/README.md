# Gtest

hipblaslt-test is the main regression gtest for hipBLASLt. All test items should pass.

```shell
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# Run full gtest tests
./clients/hipblaslt-test

# Run gtest tests with filter
./clients/hipblaslt-test --gtest_filter=<test pattern>

# Demo: gtest tests with filter
./clients/hipblaslt-test --gtest_filter=*quick*

# Demo: DRelu gradient matmul tests
./clients/hipblaslt-test --gtest_filter=*drelu*
```

## Bespoke matched-library GPU tests

Configure the normal client/test build with `HIPBLASLT_ENABLE_CLIENT=ON`,
`HIPBLASLT_BUILD_TESTING=ON`, and `GPU_TARGETS=gfx950` or `gfx942` (both may be
specified, including feature suffixes such as `gfx942:xnack+`). Building
`hipblaslt-test` also builds the tiny logic fixtures in
`data/bespoke/<arch>` through the normal `create_device_library` /
`TensileCreateLibrary` CMake path. No downloaded or workstation-specific
library is required. The fixtures share a general Stream-K recipe and contain
bespoke beta-off, GELU, sigmoid, exact-size, full-tile edge and fixed-GSU1
loop solutions, plus small regular/grouped bias-present recipes and their general
parents: thirteen solutions per architecture. gfx950 covers a 160-wide tile; gfx942 uses
its native MT32x64x128 recipe. These are test-only synthetic boundary recipes,
not tuned performance cases.

The build writes `clients/bespoke-fixture/<arch>/library/<arch>` and a checked
build-local `indices.txt` mapping. Generation checks preserve narrowed facts,
distinct general/bespoke kernel identities, removed full-tile edge paths and
correct 2U loop decrements before the test executable is built. The fixtures
follow the build's lazy-loading, YAML/msgpack metadata, compression,
sanitizer and toolchain settings. Test installation includes the library and
index map in `bin/bespoke-fixture/<arch>` next to `bin/hipblaslt-test`, matching
the normal test executable/data installation layout.

CI runs the following after building the target, on one gfx950/gfx942 device:

```shell
./clients/hipblaslt-test --gtest_filter='Bespoke/BespokeFixture.*'
```

Each case execs a fresh copy of `hipblaslt-test`, sets
`HIPBLASLT_TENSILE_LIBPATH` internally to its adjacent fixture library **before
creating a handle**, and leaves the parent's library environment untouched.
Tests are gated to gfx950/gfx942. Numerical cases pin the generated index,
check every output against an independent full scalar reference and require
a finite `norm_error` within the client's `norm_check` tolerance. The sigmoid
case also reuses the real matmul client with `norm_check=1` and
`norm_check_assert=true`, and observes the real descriptor epilogue. Each
refusal runs an independently norm-checked compatible call first, then
`getAlgosFromIndex` → `hipblasLtMatmul` without any support check, and requires
`HIPBLAS_STATUS_INVALID_VALUE` for beta, activation, bias, alpha-vector or
exact-size contradictions.

The three bias-presence cases use the same 64×64×128 bf16 problem and a
`UseBias: 1` bespoke index. A pinned no-bias launch must return
`HIPBLAS_STATUS_INVALID_VALUE`; a no-bias heuristic query must select the
build-local general parent; a bias-present query must select the bespoke
index. Every case first launches the compatible pinned bespoke index as a
positive control, and successful launches require an independently computed
finite `norm_error` within the client's tolerance. These CI cases need no
external harness, traced call files or prepared workstation artifacts. The
host predicate behind them (a bespoke solution with nonzero `UseBias` requires
a bias-present problem; stock solutions get no narrowed-fact guard) is covered
without a GPU by `BiasPresentDerivedFacts.*` in `tensilelite-tests`.

The additional bias cases exercise pinned extension, grouped and device-argument
launches, all three support-query APIs, descriptor reuse, cached selection,
tuning-cache/override replay, mixed-bias groups in both orders, a null bias
pointer, and a mismatched bias type. Refusals require unchanged output buffers;
replay must launch the general parent. Device-argument controls verify a real
grouped solution before using that overload. Stock controls also include metadata
that shares the bespoke code object/name but has no bespoke guard marker.

The tuning-cache and override cases first launch the compatible call with no file,
then use a file entry selecting the general index for that same call and require
the launched index to change. After restoring a bias-only entry, they retain the
no-bias fallback probe. The cache case also checks its process-exit summary: two
distinct identities, one matched and one falling back. Every successful phase
still enforces the numerical norm-error verdict.

`bias_initialized_ext_result` separately verifies that extension `run()` executes
the last initialized arguments: changing `setProblem()` alone does not rebind
them. Its result must still match the initialized bias-present reference.

The portable `geko_bias_contract.dat` fixture is generated by the real geko
writer and normal TensileLite serialization for bias-off, narrowed bias-type,
and retained general-bias cases. geko's `test_specialize_host_predicate_unit.py`
requires its complete metadata to match freshly generated output, without a GPU.
`GPUBiasPresentContract.geko_output_contract` in `tensilelite-tests` loads those same
bytes and evaluates the real `collectNarrowedFactPredicate()` for matching and
incompatible bias facts. This native library load constructs device-dependent
predicates, so the case requires a visible AMD GPU and skips when none is available.
Select it with `GPUBiasPresentContract.*`; CPU-only filters can exclude `*GPU*` while
retaining the pure `BiasPresentDerivedFacts.*` cases.

