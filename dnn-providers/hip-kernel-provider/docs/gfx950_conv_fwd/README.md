# Packaged gfx950 forward convolution

`hipkernel:Gfx950ConvFwd` runs rocKE's implicit-GEMM forward convolution through
the HIP Kernel Provider. It accepts plain 2D cross-correlation with dense
channels-last storage, groups=1, FP16 or BF16 storage, FP32 accumulation, and
symmetric padding. See [the graph contract](graph_contract.md) for the exact
dimension/stride mapping and rejection rules, and [kernel mining](mining.md)
for the ABI and compile-time constraints.

The catalog contains 71 compiled variants for 61 distinct requests. Ten small
requests cover both storage types for padded 3x3, pointwise 1x1, strided,
dilated, and non-square convolution, with `tile_k=64` and `tile_k=128` for each.
The remaining 51 requests are the applicable headline convolution workloads,
using the dispatcher's default tuning. This is a specialized catalog, not a
general convolution engine. An otherwise valid request needs a matching
compiled entry. Grouped, 3D, fused, channels-first, and other dtypes are
integration gaps; they are not claims about rocKE's broader capabilities.
The [verification and coverage snapshot](coverage.md) records actual dispatched
counts, independent correctness results, and the remaining catalog gaps.

## Build and packaging

The integration is based on ingestor tooling revision
`9285929e1de2ada46cc01e9054ff3ed5fa6c0456`. Use a checkout containing hipDNN,
the HIP Kernel Provider, the shared integration tests, and their dependencies.
The build requires ROCm with gfx950 support, the rocm-kpack C++ package and
its Python sources, and the Python build requirements documented in
[descriptor packaging](../../descriptor-packaging/README.md).

From the repository root, choose separate build and install directories:

```bash
export CONV_BUILD="$PWD/build-conv"
export CONV_INSTALL="$PWD/install-conv"
export ROCKE_LLVM_FLAVOR=llvm22
export ROCKE_COMGR_LIB=/opt/rocm/lib/libamd_comgr.so
export AMD_COMGR_CACHE_DIR="$CONV_BUILD/comgr-cache"

cmake --preset hip-kernel-provider -B "$CONV_BUILD" \
    -DCMAKE_INSTALL_PREFIX="$CONV_INSTALL" \
    '-DROCM_LIBS_ENABLE_COMPONENTS=hipdnn;hipdnn-python;hipdnn-integration-tests;hip-kernel-provider' \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON \
    -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON \
    -DHIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE=ON \
    -DHIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP=ON \
    -DHIPKERNELPROVIDER_ROCKE_COMGR_LIB="$ROCKE_COMGR_LIB" \
    -DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT="$PWD/dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors" \
    -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR="<rocm-kpack Python source directory>" \
    '-DCMAKE_PREFIX_PATH=<rocm-kpack install prefix>;/opt/rocm' \
    -DGPU_TARGETS=gfx950 -DAMDGPU_TARGETS=gfx950 \
    -DENABLE_ASM_SDPA_ENGINE=OFF
cmake --build "$CONV_BUILD" --parallel
cmake --install "$CONV_BUILD"
```

Select the LLVM flavor matching the installed compiler; the example uses
LLVM 22. Keep the same flavor and comgr library for direct comparisons.

The authored descriptors live under
`descriptor-packaging/examples/descriptors/rocKE/gfx950_conv_fwd`. `hkp_pack`
compiles their rocKE builders into GPU code objects, archives those in kpack,
and rewrites the shipped descriptors to `kind: kpack`. The installed engine
loads that archive without Python. These descriptors do not belong in
`HIPDNN_DESCRIPTOR_FILES` or the embedded kernel lists.

Generator configuration and the tooling profile are under
`projects/hipdnn/tools/IngestorGenerator/configs/gfx950_conv_fwd*.yaml`.
Regenerate descriptors with the generator before changing the shipped catalog;
retain existing UUIDs when extending it. The adapter derives tuning defaults
from the actual convolution dispatcher and delegates support checks and IR
construction to the original rocKE builder.

## Correctness and engine attribution

Run `hipdnn_list_engines --plugin-dir
"$CONV_INSTALL/lib/hipdnn_plugins/engines"` and require the exact name
`hipkernel:Gfx950ConvFwd`. Enumeration alone does not prove dispatch.

The provider's `*Gfx950ConvFwd*` unit tests cover graph refusal, every baked
constraint, geometry/overflow, output validation, and buffer nonaliasing.
Its GPU tests force both `tile_k` values for FP16 and BF16, compare with the CPU
reference, and verify benchmarking and plan recreation. Use their installed
binaries with `--gtest_filter='*Gfx950ConvFwd*'`.

The shared bundles are `quick/Gfx950ConvFwd/Smoke` (two cases) and
`standard/Gfx950ConvFwd/Spatial` (eight spatial cases plus a representative
56x56 BF16 headline workload). The registered target pins the engine by name:

```bash
ctest --test-dir "$CONV_INSTALL/bin/hip_kernel_provider" \
    -R '^hip_kernel_provider_gfx950_conv_fwd_external_integration_tests$' -V

"$CONV_INSTALL/bin/hipdnn_integration_tests" \
    --test-article "$CONV_INSTALL/lib/hipdnn_plugins/engines/libhip_kernel_provider.so" \
    --test-engine hipkernel:Gfx950ConvFwd \
    --reference-executor gpu --fail-on-unsupported \
    --gtest_filter='quick_Gfx950ConvFwd*:standard_Gfx950ConvFwd*'
```

Count dispatched and skipped cases in the output. The direct invocation fails
on an unsupported graph instead of allowing a suite containing only skips to
pass. The external target is enabled when both the ingestor and production
rocKE packaging are enabled.

Reconfigure CMake after editing bundle JSON: the shared harness copies its
bundle data into the build tree during configuration, then installs that copy.

## Selection, cache, and timing

`tile_k` is an integer engine knob with values 64 and 128. Fallback ranking
prefers 64, matching the dispatcher. With `HIPDNN_FORCE_BENCHMARKING=1`, the
existing ingestor benchmarks all applicable candidates on first execution and
persists the measured ranking. A complete cached ranking suppresses a new
search, including after process restart. Use a fresh `HIPDNN_CACHE_DIR` for
each first-search experiment.

The pinned runtime uses one warmup and seven timed executions per candidate,
reduces those samples with `robustMean`, and delegates subsequent executions to
the fastest measured candidate. This integration uses that implementation and
its persistent winner cache without changing either.

The integration probe at
`projects/hipdnn/tools/IngestorGenerator/tools/benchmark_conv_integration.py`
checks the exact engine, validates each forced variant against PyTorch,
records the first search separately, and compares steady-state execution with
direct rocKE using the same spec and compiler. Run its `--help` for artifact
paths and forced/automatic/reuse modes. Execute reuse in a separate process
with the same cache. The fastest result means fastest among the valid variants
actually packaged and measured; first-search time is excluded from reported
steady-state timing. Keep raw measurements in private evidence outside Git.

To compare another shipped workload, save one object from
`configs/gfx950_conv_fwd.requests.json` to a separate request file and pass
`--request-file <file> --tile-k 64`. The probe derives the dimensions, strides,
and convolution attributes from that request, verifies the exact packaged UUID,
and records the source provenance. Custom requests use an independent PyTorch
FP32 GPU reference on the quantized inputs with TF32 disabled; the small smoke
tests use a CPU reference. The probe checks both results before timing.

Default timing measures HIP graph replay with GPU events. Use
`--timing-mode events` to include ordinary Python submission gaps in a separate
measurement; do not interpret that result as kernel execution time alone.
The probe suspends its synchronous selection-log callback and backend/plugin
logging during warmup, capture, and timing, then restores them for cache checks.
Ordinary submission results include Python bindings and frontend variant-pack
construction on each call; the report also records the frontend logging state.

## Coverage audit

`mine_conv_shapes.py` reads rocKE's canonical convolution cases and
dnn-benchmarking graph JSON directories or workload tarballs. It preserves
source hashes/URIs, attributes, every excluded record, and deduplicated requests.
Unknown categorical values fail rather than acquiring guessed defaults.

```bash
python projects/hipdnn/tools/IngestorGenerator/tools/mine_conv_shapes.py \
    --rocke-cases dnn-providers/hip-kernel-provider/rocke/library/benchmarks/common/grouped_conv/bench_cases_conv.json \
    --graphs '<headline convolution archive>' \
    --graphs '<convolution sweep archive>' \
    --arch gfx950 --out '<private evidence>/conv_requests.json'
```

Run the actual adapter predicate on the resulting requests, then validate the
graphs through the installed engine using dnn-benchmarking's explicit
`--engine` selection and `--validate pytorch`. Report actual dispatched/declined
counts separately from offline support predictions. A predicate-accepted
request absent from the catalog is missing integration coverage. A family
rejection needs its concrete predicate reason. Neither bucket proves GPU
correctness without an execution against a reference.
