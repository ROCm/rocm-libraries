# Benchmark JIT GEMM solutions with hipblaslt-bench

With `HIPBLASLT_JIT=2`, `hipblaslt-bench` benchmarks just-in-time (JIT)
generated kernels through its ordinary heuristic query. Tuned parameter sets
from the knowledge file come first, and for problems covered by its GPU
performance model, Origami ranks the other kernel parameter recipes.
`Tensile.JitGemm` validates those recipes in order and asks
`Tensile.SingleSolution` to generate the first valid solutions, including any
helper kernels they need, and hipBLASLt compiles them with comgr and publishes
them into the JIT solution library.

Generation finishes inside the heuristic query, before correctness checks,
warmup, and timing. CPU timing still includes host dispatch for each GEMM. This
feature targets functional coverage; the predicted solution is not guaranteed to
be the fastest available kernel. The [JIT guide](../../JIT.md#predictor-and-tuningknowledge)
describes the tuning knowledge, and the [JIT roadmap](../../JIT_ROADMAP.md#roadmap)
the remaining calibration and measurement work.

The components interact in this order:

1. The benchmark creates ordinary matrix descriptors and calls
   `hipblasLtMatmulAlgoGetHeuristic` or `GemmInstance::algoGetHeuristic`.
2. hipBLASLt looks the problem up in the JIT solution library. For the solutions
   it lacks, the library-owned TensileLite backend takes the nearest tuned sets
   from the knowledge file and asks Origami for ranked tuning parameters.
3. `Tensile.JitGemm` validates the supplied candidates, and `SingleSolution`
   writes the first valid ones as source bundles: main kernel assembly, helper
   source and library entry.
4. hipBLASLt builds each bundle into one code object with comgr, publishes it
   into the JIT solution library, and returns its solution index.
5. The benchmark's existing C or C++ execution path checks and times the
   returned algorithms.

The benchmark has no JIT-specific code. The [JIT guide](../../JIT.md#heuristic-integration)
describes the modes, the order of the lookup and the failure reports.

## Build and run

JIT GEMM is a build-time opt-in feature enabled by
`HIPBLASLT_ENABLE_JIT=ON`. Building `hipblaslt-bench` also requires
`HIPBLASLT_ENABLE_CLIENT=ON`. The [JIT build instructions](../../JIT_TENSILELITE.md#build)
describe the required host library, comgr, Python dependencies, and compiler
setup. With `HIPBLASLT_JIT=2`, the benchmark does not require a prebuilt
hipBLASLt device library.

```bash
HIPBLASLT_JIT=2 HIPBLASLT_JIT_LIBRARY_PATH=/path/to/jit-library \
  hipblaslt-bench -m 256 -n 128 -k 512 \
  --alpha 1.25 --beta 0.5 \
  --verify --iters 3 --cold_iters 1 --print_kernel_info
```

`HIPBLASLT_JIT=1` benchmarks the pre-tuned Equality solutions first, then JIT
solutions for the part of `--requested_solution` that they leave unfilled, with
kernels that the Equality solutions do not already use, then the other
pre-tuned solutions for what is still missing. Datatype
options keep their usual defaults and meaning. The `--api_method` option chooses
how hipBLASLt prepares and executes the algorithms:

| Value | API calls |
| --- | --- |
| `c` (default) | C descriptors, `hipblasLtMatmulAlgoGetHeuristic` and `hipblasLtMatmul` |
| `mix` | C descriptors passed to the C++ extension `Gemm`, followed by its heuristic query, initialization and execution |
| `cpp` | C++ extension problem setup, heuristic query, initialization, and execution |

`--requested_solution N` returns up to N generated solutions, each with a
different kernel. `--algo_method all`
and `--algo_method index` list or select pre-tuned solutions and do not
generate. Grouped GEMM is not supported: its heuristic query reports an error
and returns no solutions.

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
the candidate. Only parameters that neither Origami nor a tuned seed supplies
begin with TensileLite defaults and are then derived or validated by its
solution builder.
The manifest records which values came from prediction, defaults, or derivation.
If Origami returns no finite positive-latency ranking and the knowledge supplies
no tuned seed, the request fails before invoking the generator. If every ranked recipe is invalid, the request reports
all candidate rejection reasons. Neither path adds a default or native recipe.
In both cases the heuristic query returns no solution, stderr has one
`hipblaslt error: JIT ...` line naming the cause, and the benchmark reports
that no solution was found.
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

Tuned seeds (`tensilelite.tuned.v1`) keep their tuned split-K (`GlobalSplitU`)
and execution policy (`TileProcessingStrategy`, `WorkAssignment`). The catalog's
data-parallel recipes (`origami.gemm.dp.v1`) keep `GlobalSplitU: 1` and
`TileProcessingStrategy: None`. On gfx942, gfx950 and gfx1250, when the request
allows workspace and has no auxiliary output, the catalog also offers Hybrid
Stream-K candidates (`origami.gemm.persistent.v1`), whose grid, reduction and
mapping the runtime chooses at each launch. Without workspace, the predictor
skips Stream-K seeds and seeds with a fixed `GlobalSplitU` above 1. The [JIT guide](../../JIT.md#predictor-and-tuningknowledge)
describes both. For an exact recipe, use the
`direct-gemm` case in the [JIT tests](../tests/jit/README.md) or
`python -m Tensile.SingleSolution`. Explicit YAML bypasses prediction and can
select split-K or Stream-K recipes accepted by the generator and runtime.

Output-amax requires `GlobalSplitU: 1`, `TileProcessingStrategy: None`, and one batch for either
route. Its current reduction needs final output and does not combine batch
offsets. Supporting those combinations requires changes to the reduction and
its runtime predicates.

## Artifacts and reuse

Published solutions stay in the JIT solution library, so a later run of the same
problem with the same library, tools and device reuses them without generating.
hipBLASLt never deletes them; to generate again, delete the library directory
while no process uses it, as the [JIT guide](../../JIT.md#persistent-solution-library)
describes. `--print_kernel_info` prints the kernel name. Generation runs in a new
directory under the system temporary directory, which hipBLASLt removes after
success and keeps after a failure; the failure report names the generator log
inside it. To keep the recipe, prediction and bundles of a successful
generation, point `HIPBLASLT_JIT_PYTHON` at a wrapper that runs the real
interpreter and then copies them, as `test_jit_gemm.py` does.

The library compiles in the Python interpreter, TensileLite source and rocisa
import paths, and the compiler that TensileLite uses to probe assembler
capabilities. The environment variables `HIPBLASLT_JIT_PYTHON`,
`HIPBLASLT_JIT_TENSILE_SOURCE`, `HIPBLASLT_JIT_PYTHONPATH`, and
`HIPBLASLT_JIT_CXX` override those paths.
`HIPBLASLT_JIT_PYTHONPATH` uses the platform path separator (`:` on Linux,
`;` on Windows) between additional Python import directories. A build without
the feature ignores `HIPBLASLT_JIT` and prints a warning once.

## Check the result

`hipblaslt-bench --verify` reports numerical errors without necessarily
returning a failing exit status. Inspect `norm_error` and the reported
`atol`/`rtol`; `failed` denotes an allclose failure.

The accompanying `test_jit_gemm.py` runs the benchmark with `HIPBLASLT_JIT=2`
and a fresh JIT solution library for each case. It checks numerical results,
recipe provenance, one generation per case before timing, publication, scratch
cleanup, reuse of a published solution without generation, several generated
solutions, and one error report for each request that cannot produce a recipe.
For example, with a configured Python environment and local build:

```bash
python projects/hipblaslt/clients/bench/test_jit_gemm.py \
  --bench projects/hipblaslt/build/release/clients/hipblaslt-bench \
  --build-root projects/hipblaslt/build/release \
  --python /path/to/venv/bin/python --architecture gfx950 \
  --output "$(mktemp -d)/hipblaslt-jit-checks"
```

The output path must be new. `--case half-c-default` selects a smoke case;
`--negative-only` checks only the grouped-GEMM rejection. With JIT disabled in
the build, `--feature-off` checks that `HIPBLASLT_JIT` is ignored with one
warning. A failing case does not stop the others; the script lists the failed
cases and returns a failing status. Architecture checks verify that the
recorded instructions and cache hints are legal for the executing GPU.
Cross-compilation checks establish generation and compiler support; numerical
correctness also requires execution on that GPU.
