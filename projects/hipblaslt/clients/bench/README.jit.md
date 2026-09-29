# Request a JIT GEMM algorithm with hipblaslt-bench

`hipblaslt-bench --jit-gemm` generates a kernel for the requested matrix
multiplication, then checks and times it through hipBLASLt. For problems covered
by its GPU performance model, Origami ranks kernel parameter recipes.
`Tensile.JitGemm` validates those recipes in order and asks `Tensile.SingleSolution` to generate the
first valid solution, including any helper kernels it needs, and hipBLASLt
compiles it with comgr.

Generation finishes before correctness checks, warmup, and timing. CPU timing
still includes host dispatch for each GEMM. This feature targets functional
coverage; the predicted solution is not guaranteed to be the fastest available
kernel. Incorporating tuning knowledge into prediction is planned work in the
[JIT roadmap](../../JIT.md#roadmap).

The components interact in this order:

1. The benchmark creates ordinary matrix descriptors and a generic JIT request
   through the internal JIT entry points.
2. The library-owned TensileLite backend translates that request and asks Origami
   for ranked tuning parameters. Its private selection plan names the ranked
   request consumed by the builder.
3. `Tensile.JitGemm` validates the supplied candidates, and `SingleSolution`
   writes the first valid one as a source bundle: main kernel assembly, helper
   source and library entry.
4. hipBLASLt builds the bundle into one code object with comgr and loads it.
5. The generic result is adapted to a GEMM algorithm, then the benchmark's
   existing C or C++ execution path checks and times it.

The benchmark does not own prediction policy or launch generated code directly.
The request, backend, solution, and operation adapter are internal: their
headers are not installed, and the benchmark includes them from
`library/src/amd_detail`. The [JIT guide](../../JIT.md#entry-points) describes
them.

## Build and run

JIT GEMM is a build-time opt-in feature enabled by
`HIPBLASLT_ENABLE_JIT=ON`. Building `hipblaslt-bench` also requires
`HIPBLASLT_ENABLE_CLIENT=ON`. The [JIT build instructions](../../JIT.md#build)
describe the required host library, comgr, Python dependencies, and compiler
setup.
The JIT path does not require a prebuilt hipBLASLt device library.

```bash
hipblaslt-bench --jit-gemm -m 256 -n 128 -k 512 \
  --alpha 1.25 --beta 0.5 \
  --verify --iters 3 --cold_iters 1 --print_kernel_info
```

In JIT mode, `hipblaslt-bench` defaults to FP16 A/B inputs and FP32 C/D matrices
and accumulation. Explicit datatype options override those defaults;
`-r f32_r` selects FP32 matrices. The `--api_method` option
chooses how hipBLASLt prepares and executes the generated algorithm:

| Value | API calls |
| --- | --- |
| `c` (default) | C descriptors and `hipblasLtMatmul` |
| `mix` | C descriptors passed to the C++ extension `Gemm`, followed by extension initialization and execution |
| `cpp` | C++ extension problem setup, initialization, and execution |

## Prediction and supported problems

The prediction path supports gfx90a, gfx942, gfx950, and gfx1250. It translates
the matmul descriptors into a TensileLite problem, including datatypes, matrix
layouts, scaling, bias, activation, auxiliary output, and output-amax. Candidate
recipes must satisfy the generator and runtime checks for that problem and
device. The hipBLASLt API must also accept the requested datatypes; successful
standalone TensileLite compilation alone does not establish API support.
A datatype or feature combination without a legal solution reports
that failure instead of dropping the requested operation. The public hipBLASLt API requires C and D to share a storage datatype.
TensileLite also represents both with one `DestDataType` field, so supporting
separate types would require changes to both contracts.

The executing device supplies the GPU model and resource limits used for
prediction. TensileLite validates each proposed recipe against that GPU's
instruction set before compilation. Origami ranks `MatrixInstruction`, macro tiles,
`DepthU`, and `NonTemporalA/B`, and supplies all workgroup-mapping and stagger
outputs applicable to the data-parallel candidate domain. The
[modeled-input inventory](../../JIT.md#origami-modeled-inputs) records translations
and mode constraints. Unsupported translations or changed modeled values reject
the candidate. Only unpredicted parameters begin with TensileLite defaults and
are then derived or validated by its solution builder.
The manifest records which values came from prediction, defaults, or derivation.
If Origami returns no finite positive-latency ranking, the request fails before
invoking the generator. If every ranked recipe is invalid, the request reports
all candidate rejection reasons. Neither path adds a default or native recipe.
The provider records descriptor scale modes separately; its Python translation
chooses the corresponding physical layout and shared TensileLite validators
check it.

On gfx950, MX inputs require the pre-swizzled block32 UE8M0 scale mode
(`--scaleA 1001 --scaleB 1001` for two MX operands). The shared subtile generator
uses that layout for scale loads and does not implement natural scale mode 3.
The initial predictor does not supply the required subtile parameters, so its
current gfx950 MX rankings are rejected. Explicit recipes remain available to
request supported MX kernels.
On gfx1250, ordinary block-scale modes use `InMemorySwizzle`; mode 1001 is not
accepted. Generation rejects an incompatible layout, and execution checks that
the generated solution's layout matches the supplied descriptors.

Origami's gfx1250 memory model currently combines provisional gfx950 values
with gfx1250 overrides, so its latency estimates are not calibrated for gfx1250.
The estimates omit bias, activation, scaling, auxiliary-output, and output-amax overhead.
These affect ranking; TensileLite still checks whether the chosen recipe is legal for the target.

Predicted recipes keep `GlobalSplitU: 1` and `StreamK: 0`, so they do not split
the K reduction across workgroups. Exploring split-K and Stream-K during
prediction is outside this initial policy. For an exact recipe, use the
`direct-gemm` case in the [JIT tests](../tests/jit/README.md) or
`python -m Tensile.SingleSolution`. Explicit YAML bypasses prediction and can
select split-K or Stream-K recipes accepted by the generator and runtime.

Output-amax requires `GlobalSplitU: 1`, `StreamK: 0`, and one batch for either
route. Its current reduction needs final output and does not combine batch
offsets. Supporting those combinations requires changes to the reduction and
its runtime predicates.

`--jit-gemm` selects one generated algorithm. Multi-algorithm selection and
tuning are outside this initial integration, so it rejects `--algo_method all`,
`--algo_method index`, an explicit `--solution_index`, `--requested_solution`
other than 1, nonzero `--splitk` or `--wgm`, and `HIPBLASLT_TUNING_FILE`.
Datafile mode, grouped GEMM, and pointer-array batches also need additional JIT
request and lifetime handling and are not supported. A rejected tuning-file
combination does not create a tuning file.

## Artifacts and reuse

Each request retains its generated files in a fresh temporary directory.
`--jit-output-dir /path/to/artifact-parent` chooses the parent directory.
The recipe, manifest path, and prediction summary are printed to stderr;
result CSV is printed to stdout. Generator diagnostics are kept in the
generator log. The artifact directory holds the chosen solution's source
bundle: its library entry, main kernel assembly and helper source, which
hipBLASLt builds into one code object with comgr. The manifest also records
rejected recipes and the sources belonging to the chosen solution. Invalid
recipes are skipped before compilation; a build or resource failure ends the
request, and a comgr failure names its retained log.

Repeated GEMM calls reuse the generated algorithm without compiling again.
The algorithm and its loaded modules remain registered until process exit and
are valid only on the device that created them. Loading a retained bundle in
a later process is not exposed by this initial API. Retain the YAML to
reproduce a recipe; the opaque algorithm value and its index cannot be saved
as a reusable library entry.

CMake selects the Python interpreter, TensileLite source and rocisa import
paths, and the compiler that TensileLite uses to probe assembler capabilities.
The environment variables `HIPBLASLT_JIT_PYTHON`,
`HIPBLASLT_JIT_TENSILE_SOURCE`, `HIPBLASLT_JIT_PYTHONPATH`, and
`HIPBLASLT_JIT_CXX` override those paths for local development.
`HIPBLASLT_JIT_PYTHONPATH` uses the platform path separator (`:` on Linux,
`;` on Windows) between additional Python import directories. Using `--jit-gemm` in a build without the feature reports
that `HIPBLASLT_ENABLE_JIT` must be enabled.

## Check the result

`hipblaslt-bench --verify` reports numerical errors without necessarily
returning a failing exit status. Inspect `norm_error` and the reported
`atol`/`rtol`; `failed` denotes an allclose failure.

The accompanying `test_jit_gemm.py` checks numerical results, recipe provenance,
generation before timing, repeated execution, and incompatible options. For
example, with a configured Python environment and local build:

```bash
python projects/hipblaslt/clients/bench/test_jit_gemm.py \
  --bench projects/hipblaslt/build/release/clients/hipblaslt-bench \
  --build-root projects/hipblaslt/build/release \
  --python /path/to/venv/bin/python --architecture gfx950 \
  --output "$(mktemp -d)/hipblaslt-jit-checks"
```

The output path must be new. `--case half-c-default` selects a smoke case;
`--negative-only` checks option conflicts without GPU execution. With JIT
disabled in the build, `--feature-off` checks its diagnostic. Architecture
checks verify that the recorded instructions and cache hints are legal for
the executing GPU. Cross-compilation checks establish generation and compiler
support; numerical correctness also requires execution on that GPU.

## Next integration steps

The [target design](../../JIT.md#target-design) moves generation behind the
heuristic query, controlled by `HIPBLASLT_JIT`, with a persistent JIT solution
library. Roadmap steps 4 and 5 in the [JIT guide](../../JIT.md#roadmap) add that
library and the heuristic integration, and step 5 removes `--jit-gemm`.
