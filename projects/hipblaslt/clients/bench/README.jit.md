# Generate and benchmark one GEMM solution

`hipblaslt-bench --jit-gemm` uses Origami to rank a small set of kernel parameter
recipes. `Tensile.JitGemm` validates them in ranked order, fills the remaining
parameters from Tensile defaults, and uses the shared `Tensile.SingleSolution`
builder to compile the first valid solution. It then passes that solution through the
normal hipBLASLt algorithm support, workspace, and execution paths. Generation
finishes before correctness checks, warmup, and timing. This experimental feature
targets functional coverage; the predicted solution is not guaranteed to be the
fastest available kernel.

Build the local host library and benchmark with
`HIPBLASLT_ENABLE_JIT_GEMM=ON` and `HIPBLASLT_ENABLE_CLIENT=ON`. The benchmark uses
the Python interpreter, Tensile source, local rocisa module, compiler, and offload
bundler selected by CMake. No project installation or prebuilt device library is
needed for JIT execution. The option remains visible in feature-disabled builds
and reports that JIT must be enabled when used.

```bash
hipblaslt-bench --jit-gemm -m 256 -n 128 -k 512 \
  -r f16_r --compute_type f32_r --alpha 1.25 --beta 0.5 \
  --verify --iters 3 --cold_iters 1 --print_kernel_info
```

The benchmark defaults to FP16 matrices with FP32 computation and the C API.
FP32 matrices are selected with `-r f32_r`. `--api_method mix` and
`--api_method cpp` use the existing extension preparation and execution paths.
Automatic prediction supports gfx90a, gfx942, and gfx950 with matching FP16 or
FP32 matrix types, FP32 computation, N/T transposes, and strided batches.
Output-amax is optional and currently requires one
batch. Other epilogues, datatype combinations, and unsupported descriptor
attributes fail explicitly. Leading dimensions, strides, alpha, beta, transpose,
and output-amax requests are carried by the normal descriptors. Automatic
prediction does not support C/D scaling.

The actual device selects the Origami hardware model and resource limits. FP16
recipes use `16x16x16` matrix instructions on gfx90a/gfx942 and may also use
`16x16x32` on gfx950; FP32 uses `16x16x4`. gfx90a recipes use zero A/B cache hints
because that architecture has no non-temporal modifier. Tensile validates the
chosen parameters against the target ISA before compilation. Origami's latency
estimate does not model the extra output-amax work.

JIT selects exactly one generated algorithm. It cannot be combined with
`--algo_method all`, `--algo_method index`, an explicit `--solution_index`,
`--requested_solution` other than 1, nonzero `--splitk` or `--wgm`, datafile mode,
grouped GEMM, pointer-array batches, or `HIPBLASLT_TUNING_FILE`. No tuning file is
created when rejecting a JIT/tuning conflict.

Artifacts are retained in a fresh directory under the system temporary directory.
Use `--jit-output-dir /path/to/artifact-parent` to choose its parent. The recipe,
manifest, and prediction summary are printed to stderr; benchmark CSV remains on
stdout. The manifest records the Origami ranking, earlier rejected candidates,
selected parameters, defaults, actual problem and hardware, and generated code
objects. Only `MatrixInstruction`, `DepthU`, and `NonTemporalA/B` are selected
by prediction. Other tuning fields start at
`Tensile/Common/GlobalParameters.py:defaultBenchmarkCommonParameters`; the
manifest distinguishes those defaults from values derived by Tensile. Invalid
recipes are skipped before compilation; compiler or resource failures stop the
request. Automatic recipes retain the defaults `GlobalSplitU: 1` and `StreamK: 0`.
Explicit YAML can select Stream-K or split-K configurations accepted by the
normal generator and runtime. Output-amax currently requires `GlobalSplitU: 1`,
`StreamK: 0`, and one batch in both the normal APIs and the standalone owner.
Generated algorithms and their modules remain registered until process exit;
there is no eviction. They are valid only on their original device in that
process and have no reusable prebuilt solution index. Replay the YAML rather
than persisting the opaque algorithm value.

For an exact recipe, keep using the standalone
[`hipblaslt-jit-gemm` sample](../samples/29_hipblaslt_jit_gemm/README.md) or
`python -m Tensile.SingleSolution`. Explicit YAML bypasses prediction. The same
bundle representation retains any helpers required by that solution.

The optional `HIPBLASLT_JIT_PYTHON`, `HIPBLASLT_JIT_TENSILE_SOURCE`,
`HIPBLASLT_JIT_PYTHONPATH`, `HIPBLASLT_JIT_CXX`, and
`HIPBLASLT_JIT_OFFLOAD_BUNDLER` environment variables override build-configured
tool paths for local development. `HIPBLASLT_JIT_PYTHONPATH` contains additional
Python import directories separated by colons. These are source/build paths;
the experimental API header is not installed.

The benchmark's current `--verify` reports numerical errors without necessarily
returning a failing exit status. Check `norm_error` and the reported `atol`/`rtol`;
`failed` denotes an allclose failure. The accompanying `test_jit_gemm.py` checks
those CSV fields, retained candidate provenance, local linkage, one generation
across repeated runs, compilation outside timing, and option conflicts. Run it
with the existing configured virtual environment and local build:

```bash
.venv/bin/python projects/hipblaslt/clients/bench/test_jit_gemm.py \
  --bench projects/hipblaslt/build/release/clients/hipblaslt-bench \
  --build-root projects/hipblaslt/build/release \
  --python .venv/bin/python --architecture gfx950 --output /tmp/hipblaslt-jit-checks
```

Use a fresh `--output` path for each run. `--case half-c-default` limits numerical
coverage to a smoke case; `--negative-only` checks parser conflicts without GPU
execution. After rebuilding the same build with JIT disabled, `--feature-off`
checks the unavailable-feature diagnostic. The full numerical matrix includes
FP16/FP32, all three API modes, odd dimensions, transpose variants, a padded
strided batch, and single-batch output-amax. Pass `--architecture gfx90a`,
`gfx942`, or `gfx950` to match the executing device; the test verifies that the
recorded instruction and cache hints are legal for that architecture. Run it
on each target GPU in shared CI. Cross-compiling an architecture's kernels does
not establish numerical correctness on that GPU. The two-second compiler-delay
check is a timing-boundary test, not a kernel performance requirement.

The `hipblaslt-jit-gemm-ci.yml` workflow builds this checkout and runs the full
benchmark matrix plus standalone and normal-API Stream-K/amax fixtures on native
gfx90a and gfx942 runners. It installs only SDK dependencies, stages the checkout's
header-only hipblas-common, and uses an empty prebuilt device-library directory.
The shared driver `.github/scripts/test_hipblaslt_jit.py` accepts `--build`,
`--architecture`, and a fresh `--output` directory; `--case` selects a single route
for local reproduction.
