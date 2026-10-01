# Packaged gfx950 forward convolution

`hipkernel:Gfx950ConvFwd` runs rocKE's implicit-GEMM forward convolution through
the HIP Kernel Provider. It accepts plain 2D cross-correlation with dense
channels-last storage, FP16 or BF16 storage, FP32 accumulation, and symmetric
padding. Grouped convolution, including depthwise and channel-multiplier
depthwise, is supported: the group count is inferred as X channels divided by
W dimension 1, and each group runs on grid z. Grouped pointwise convolution
(1x1 filter, stride 1, no padding, groups > 1) is declined because the
kernel's flat pointwise path does not select the group's channel slabs. See [the graph contract](graph_contract.md) for the exact
dimension/stride mapping and rejection rules, and [kernel mining](mining.md)
for the ABI and compile-time constraints.

The shipped catalog contains 48 compiled variants for 24 synthetic verification
requests, each request with `tile_k=64` and `tile_k=128` and otherwise the
dispatcher's default tuning:

| Requests | Source |
|---:|---|
| 10 | Synthetic verification shapes, FP16 and BF16: padded 3x3, pointwise 1x1, strided, dilated and non-square. |
| 14 | Synthetic grouped verification shapes, FP16 and BF16: two grouped 3x3 cases, depthwise 3x3 at stride 1 and 2, odd-channel depthwise 7x7 (default epilogue), a channel-multiplier depthwise 3x3 and a dilated grouped 5x5. |

Workload shapes are never committed. The shipped catalog exists to prove the
integration end to end; catalogs for real workloads are generated and packed
outside the source tree, as described in
[private workload catalogs](#private-workload-catalogs).

This is a specialized catalog, not a general convolution engine. An otherwise
valid request needs a matching compiled entry. Grouped pointwise, 3D, fused,
channels-first, and other dtypes are integration gaps; they are not claims
about rocKE's broader capabilities.
[Verification and coverage](coverage.md) records what the shipped catalog is
verified against.

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
    -DHIPKERNELPROVIDER_ROCKE_COMGR_LIB="$ROCKE_COMGR_LIB" \
    -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR="<rocm-kpack Python source directory>" \
    '-DCMAKE_PREFIX_PATH=<rocm-kpack install prefix>;/opt/rocm' \
    -DGPU_TARGETS=gfx950 -DAMDGPU_TARGETS=gfx950 \
    -DENABLE_ASM_SDPA_ENGINE=OFF
cmake --build "$CONV_BUILD" --parallel
cmake --install "$CONV_BUILD"
```

Select the LLVM flavor matching the installed compiler; the example uses
LLVM 22. Keep the same flavor and comgr library for direct comparisons.
Producer selection is per descriptor `kernel_source.kind`, so no
producer-specific packaging flag is needed.

The authored descriptors live in the provider's shipped descriptor root, under
`src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_conv_fwd`, beside
`rocKE/gfx950_attention_dense`. Production packaging is on by default for any
ingestor build whose GPU targets include gfx950. `hkp_pack` compiles their
rocKE builders into GPU code objects, archives those in kpack, and rewrites the
shipped descriptors to `kind: kpack`. The installed engine
loads that archive without Python. These descriptors do not belong in
`HIPDNN_DESCRIPTOR_FILES` or the embedded kernel lists.

The generator configuration is
`projects/hipdnn/tools/IngestorGenerator/configs/gfx950_conv_fwd.yaml`, its
request inventory and provenance `gfx950_conv_fwd.requests.json`, and the
tooling profile `gfx950_conv_fwd.profile.yaml`, all in the same directory.
Regenerate descriptors with the generator before changing the shipped catalog;
retain existing UUIDs when extending it. Only synthetic verification shapes
belong in these files. The adapter derives tuning defaults
from the actual convolution dispatcher and delegates support checks and IR
construction to the original rocKE builder.

## Correctness and engine attribution

Run `hipdnn_list_engines --plugin-dir
"$CONV_INSTALL/lib/hipdnn_plugins/engines"` and require the exact name
`hipkernel:Gfx950ConvFwd`. Enumeration alone does not prove dispatch.

The provider's `*Gfx950ConvFwd*` unit tests cover graph refusal, every baked
constraint, geometry/overflow, output validation, and buffer nonaliasing.
Its GPU tests force both `tile_k` values for FP16 and BF16 on every synthetic
verification shape, grouped and depthwise included, compare with the CPU
reference, and verify benchmarking and plan recreation. Use their installed
binaries with `--gtest_filter='*Gfx950ConvFwd*'`.

The shared bundles are `quick/Gfx950ConvFwd/Smoke` (the padded 3x3 smoke
shape and a depthwise 3x3, each in FP16 and BF16) and
`standard/Gfx950ConvFwd/Spatial` (eight spatial cases, twelve grouped and
depthwise cases). The registered target pins the engine by name:

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
pass. The external target, and the `TestGfx950ConvFwdPacks` census, are
registered only when the build's production descriptor root actually ships
`hipkernel:Gfx950ConvFwd` for gfx950 (`hkp_gfx950_conv_fwd_available()`).

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

To compare a workload, write one request object in the format of
`configs/gfx950_conv_fwd.requests.json` to a private request file outside the
source tree and pass `--request-file <file> --tile-k 64`; the request must be
packaged in the installed catalog. The probe derives the dimensions, strides,
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

## Private workload catalogs

Workload shapes, whether from customers or from benchmark corpora, stay out of
the source tree, out of commits and out of pull requests. To serve them:

1. Mine or write the requests into a private file outside the source tree.
2. Copy `configs/gfx950_conv_fwd.yaml` outside the source tree and replace or
   extend its `shapes` lists with the private requests. Grouped requests go in
   the `_G{groups}` variants group so names cannot collide.
3. Run `generate.py --config <private config> --output-dir <private dir>`.
4. Copy the shipped descriptor root
   (`src/engines/kernel_ingestor_engine/descriptors`) to a private directory and
   replace its `rocKE/gfx950_conv_fwd` bundle with the generated one, so the
   other shipped bundles keep packaging.
5. Configure with `-DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT=<private root>`,
   then build and install as above. The private catalog is compiled, packed
   and installed exactly like the shipped one, and only exists in that install.

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

Mined requests contain workload shapes: keep the output, its report and every
catalog generated from it outside the source tree.

Run the actual adapter predicate on the resulting requests, then validate the
graphs through the installed engine using dnn-benchmarking's explicit
`--engine` selection and `--validate pytorch`. Report actual dispatched/declined
counts separately from offline support predictions. A predicate-accepted
request absent from the catalog is missing integration coverage. A family
rejection needs its concrete predicate reason. Neither bucket proves GPU
correctness without an execution against a reference.
