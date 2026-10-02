# Generate and build one GEMM solution

This source guide is for TensileLite contributors and hipBLASLt integration
developers. Its ownership and publication scope follow the
[JIT guide](../JIT.md). The builder supplies the generation
step used by hipBLASLt's TensileLite JIT backend; it does not make ordinary
matmul calls compile code.

`Tensile.SingleSolution.generateAndBuildSingleSolution` generates one GEMM
solution from a TensileLite YAML recipe. A solution includes the main assembly
kernel, any helper kernels and support code, and the metadata needed by the
TensileLite runtime. By default the builder also builds the code objects. With
`sourceOnly` it emits only the sources and metadata, which hipBLASLt builds in
process with comgr. This entry point does not run GPU work or measure kernel
performance.

The YAML supplies an exact recipe using the existing problem and parameter
schema. `SingleSolution` uses TensileLite's full target and solution validators.

[Select a recipe for a GEMM problem](#select-a-recipe-for-a-gemm-problem)
covers ranked selection. [hipBLASLt provider integration](#hipblaslt-provider-integration)
and the [JIT guide](../JIT.md#heuristic-integration) cover how hipBLASLt builds,
publishes and runs the results.

## Python and command-line use

```python
from Tensile.SingleSolution import generateAndBuildSingleSolution

result = generateAndBuildSingleSolution(
    "problem.yaml",
    "new-request-output",
    architecture="gfx950",
    libraryFormat="msgpack",
)
print(result.manifestPath)
```

The equivalent command is:

```bash
python -m Tensile.SingleSolution problem.yaml new-request-output --architecture gfx950
```

The output directory must not exist. The selected Python environment must
contain TensileLite's dependencies and be able to import the checkout and its
built rocisa extension. The optional `TensileGenerateSingleSolution` script
calls the same entry point.

The Python function returns a frozen `SingleSolutionBuildResult` containing
absolute bundle, manifest, and library paths, a tuple of all code-object paths,
the main code-object path and kernel name, the solution name and local index,
the architecture, and a tuple of source paths. A source-only result has no
code-object paths and a main code-object path of `None`; a full build lists no
sources. It raises `SingleSolutionError`, or its configuration or build
subclass, on failure.

The CLI prints progress followed by the manifest path. It reports failures on
stderr and returns a nonzero exit status. Programs invoking it should check
that status and then read `new-request-output/bundle/manifest.json`; stdout is
human-readable progress, not a JSON response.

`Tensile.SingleSolution` and
[`Tensile.JitGemm`](#select-a-recipe-for-a-gemm-problem) take two private
options that `--help` does not list. `--debug CATEGORIES` (`timing`, `progress`
or `all`, comma-separated) records how long each stage takes and what it does:
progress lines starting with `progress:` go to stderr as they happen, and a
timing table starting with `timing:` follows at the end. `--debug-dir DIR`,
which requires `--debug`, writes them as files instead: each progress event is
appended to `DIR/events.jsonl` as one JSON line, and the stage times are written
to `DIR/timing.json` when the command finishes. Nothing else the command writes
changes, and a failure to record never fails it. hipBLASLt passes both when
[`HIPBLASLT_JIT_DEBUG`](../JIT.md#diagnostics-with-hipblaslt_jit_debug) asks
for them.

## Describe one solution

Supply one `BenchmarkProblems` entry with one problem type and one parameter
group. `BenchmarkCommonParameters`, `ForkParameters`, singleton `Groups`, and
`BenchmarkFinalParameters` keep their existing meanings. The parameter lists
must describe exactly one combination. An empty candidate list or a combination
that expands to several recipes is rejected before solution filtering.
`MatrixInstruction: [[]]` is one empty-valued candidate, not an empty list of
candidates.

TensileLite checks parameter names, types, allowed values, matrix instructions,
and derived solution properties using its existing validators. Multiple
problem sizes are parsed but never timed or used to choose a recipe. Custom
kernels, solution pools, and alternate tuning modes are outside this entry
point. Custom kernels are served through the KFA custom-kernel path
([CustomKernels/README.md](Tensile/CustomKernels/README.md)), not through this
entry point or the JIT. Non-null obsolete benchmark steps and parameter-level
`CustomKernel` are rejected. `NoReject` cannot disable validation.

The derived solution determines which helpers are built. For example, split-K
may divide the reduction across workgroups and require an output-conversion
kernel to combine partial results. Beta initialization and bias-gradient
reduction can also require helpers. One helper generator may emit several
entrypoints, while activation headers and inline device functions provide
support code without adding a GPU launch.

The fixtures in `Tensile/Tests/unit/test_data/` illustrate these choices:

| Recipe | Behavior |
| --- | --- |
| `single_solution.yaml` | FP16 with FP32 accumulation; no split-K or helpers |
| `single_solution_splitk.yaml` | Split-K across four workgroups, with beta and output-conversion helper families |
| `single_solution_adaptive.yaml` | Runtime-selected split count and accumulation mode |
| `single_solution_streamk.yaml` | Stream-K with partial-tile reduction |
| `single_solution_amax.yaml` | Unscaled output-amax with packed stores |

`GlobalSplitU` selects the split-K count. Values greater than one and the
runtime-selected value `-1` are accepted when the solution validators allow
them. `AdaptiveGemmGSUA: 1` lets the runtime choose between a separate output
reduction and reduction inside the main kernel. The unrelated `AdaptiveGemm`
parameter chooses store-width paths. This builder does not force these
parameters off.

Output-amax currently requires `GlobalSplitU: 1`, `StreamK: 0`, and one batch.
Its reduction needs final output and does not combine batch offsets. Supporting
those combinations requires changes to the reduction and runtime predicates.
A consuming runtime must honor the same restrictions.

## Target and toolchain

The architecture is explicit, so generation does not need GPU enumeration.
Compiler, offload bundler, code-object version (`4`, `5`, or `6`), library format
(`msgpack` or `yaml`), source-only output (`--source-only`), and temporary-file
retention are Python and CLI options. Invocation options select the format and
toolchain; contradictory YAML target settings are rejected. The selected library
format must be enabled in the consuming host library.

With `sourceOnly`, nothing is assembled, compiled or bundled. The compiler only
probes assembler capabilities, and no offload bundler is needed. The
code-object version still shapes the assembly, whose kernel metadata version
depends on it, so the consumer must build the sources with the same version.
Without `sourceOnly`, main and helper code objects use the requested
code-object version, and helper compilation bypasses the shared helper cache so
the bundle reflects the selected sources and toolchain.

The YAML settings `GenerateSourcesAndExit`, `ForceGenerateKernel` and
`PythonProfile` conflict with this entry point and are rejected. Benchmark
timing, result export, clock control, monitoring, library-logic analysis, and
client generation do not run. Generation uses one worker.

Generation uses shared Python configuration and rocisa capability caches.
Calls must not overlap other TensileLite generation in the same process, and
the compiler identity must remain fixed between calls. This entry point rejects
overlapping calls to itself and changes of compiler identity. A fresh subprocess
with a private working directory for each request isolates that shared state
and tool temporary files.

## Bundle contents and publication

Each request creates its output directory exclusively and builds in a private
`.staging` directory. After generation and metadata serialization succeed,
that directory is atomically renamed to `bundle`. A failure leaves staging
files for diagnosis and publishes no bundle. Retrying requires a new output
path, so an existing result is never overwritten.

`keepBuildTmp=True` retains intermediate assembly, and in a full build the
object files, in the published bundle.

A source-only bundle is laid out by convention, so a consumer can build it
without reading the manifest:

| Path | Contents |
| --- | --- |
| `library/TensileLibrary.dat.zlib` (MsgPack) or `library/TensileLibrary.yaml` | The one-solution library entry |
| `sources/<kernel>.s` | Main kernel assembly |
| `sources/Kernels.cpp`, `sources/Kernels.h` | Helper kernel source, present only when the solution needs helpers |
| Other `sources/*.h` files | Static headers that `Kernels.cpp` includes |
| `manifest.json` | Provenance |

A full build keeps the code objects next to the library under `library/` and
lists generated helper sources and required headers in `support_files`
whenever the solution needs them.

The manifest is a provenance record. A source-only manifest has
`schema_version: 3` and `mode: "source"`; a full-build manifest has
`schema_version: 2`. Artifact paths are relative to the bundle directory:

| Field | Contents |
| --- | --- |
| `architecture` | Requested and resolved architecture, plus compiler target |
| `main_kernel` | Main entrypoint name, plus its code-object path in a full build |
| `sources` | Source-only: the main `.s` file, the headers, then `Kernels.cpp` and `Kernels.h` |
| `code_objects` | Full build: all main `.co` and helper `.hsaco` files |
| `helpers` | Generator kind (`kernel_family`, `header`, or `device_function`), class, and family/support name |
| `support_files` | Full build: generated helper sources and required headers |
| `solution` | Local index, solution name, and kernel name |
| `library` | Format, logical path, and physical path |
| `counts` | One solution, one main kernel, and helper/support generator counts |
| `provenance` | YAML hash, generator version/revision, compiler identity, code-object and kernel-argument versions, and temporary-retention choice |

For MsgPack, `library.path` names the physical `.dat.zlib` file and
`library.logical_path` names `.dat`; the runtime loader understands both.

The serialized solution library describes runtime support predicates,
workspace, and kernel arguments. A host runtime loads the solution's code
objects, checks the requested problem, allocates workspace, and uses the ordered
invocation sequence from `ContractionSolution::solve`. Helper-generator counts
and code-object counts do not determine the number of launches; the runtime
problem and selected accumulation path determine that sequence.

## Select a recipe for a GEMM problem

`Tensile.JitGemm` accepts a JSON request containing a Tensile `problem_type`
mapping, which specifies storage and arithmetic datatypes and operations.
The request also contains logical dimensions and physical tensor extents in `problem`,
an explicit architecture, and ranked `candidates`. Each candidate contains an
integer ID, its predicted cost or null, and Tensile tuning parameters.

```bash
python -m Tensile.JitGemm request.json new-request-output --architecture gfx950
```

The generic TensileLite provider additionally supplies `modeled_contract: origami.gemm.dp.v1`
and each candidate’s raw `modeled` outputs. The [Origami capability inventory](../JIT.md#origami-modeled-inputs)
defines this data-parallel contract, its unit translations and explicit unsupported cases.
Selection rejects missing or changed modeled outputs; ordinary caller recipes without
this marker retain the existing sentinel/default behavior. Explicit YAML bypasses this selector.

The request uses `schema_version: 1`. `model` identifies the caller's prediction
method, such as `origami.gemm.estimation`. Candidates are tried in the supplied
order; `predicted_cycles` records a positive estimate or null when none is available.
Each candidate must supply tuning parameters. Tensile's existing parameter and
solution validators determine which parameters and values are legal; the interface
does not restrict a predictor to a fixed list of four parameters.

The module builds the first candidate accepted by solution validation and the
static size/stride predicates. The optional `requested_solutions` (default 1,
at most the candidate count) asks for that many accepted candidates in ranked
order, and the optional `exclude_kernel_names` skips candidates whose kernel
name it lists. Both compare the kernel name that the published library gives
the solution, so a caller excludes the kernels its earlier results hold by
passing their names. A candidate whose kernel repeats one already accepted for
the same request is skipped with the reason `Same kernel as candidate <id>`:
candidates that differ only in fields the kernel name does not encode, such as
the matrix instruction's K, build the same kernel.
`--source-only` applies as for `Tensile.SingleSolution`. The
shared predicate definitions also control
early rejection for vector widths, buffer offsets and workgroup counts. Checks
requiring workspace, scalar values or device state remain with the host runtime,
which evaluates the complete predicates before execution. The module does not
run Origami or measure kernel latency. If all supplied candidates are rejected,
the request fails with their IDs and rejection reasons. It does not add default
or native-instruction candidates, and an empty parameter recipe is rejected.

Scale participation, block size, datatypes and physical layout affect performance
and are inputs to prediction. They describe the supplied problem and buffers, so
they remain fixed while this selector tries the ranked candidates. Named
`scale_mode_a` and `scale_mode_b` preserve the descriptor modes; the provider maps
them to Tensile's `MXScaleFormat`. A supplied `problem.mx_scale_format` must agree.
The gfx950 subtile backend requires `HostPreSwizzle` with
`Block_32_UE8M0_32_8_EXT`; natural scales are not implemented by that backend.
The gfx1250 TDM backend requires `InMemorySwizzle` with its ordinary block-scale
modes. The selector does not rearrange scale buffers. A candidate may supply a
matching `MXScaleFormat`; an omitted value or `Auto` binds to the descriptor
layout. An explicit conflicting layout rejects that candidate before solution
derivation, and selection continues in the supplied order. It is never silently
replaced with another layout.

The `implementation_parameters` report records these descriptor-bound values;
its name does not mean they are absent from the performance model. This module
consumes supplied rankings and preserves `model_assumptions`. A caller whose
model omits scale-handling costs must disclose that limitation there.

Datatype recipe reuse is limited to equivalent input and scale type families.
Normalize each type to its bit width and number kind (integer or floating point).
The A/B input pair and the A/B scale pair must each preserve their normalized
types, allowing independent permutations within each pair. FP8 and BF8 share
one floating-point class, as do FP6 and BF6; FP8×FP6 and FP4×FP4 are different
families. Other problem facts, including dimensions, scaling participation,
block sizes and physical layouts, must remain compatible with the prediction.
Tensile still validates every concrete datatype combination.

The MX tests in `Tensile/Tests/unit/test_JitGemm.py` group reuse checks by those
families, including operand permutations and equal-width floating-point formats.
They check generation and compilation, not equal kernel latency or numerical
correctness. Predicted costs belong to the supplied problem; successful compilation
does not establish that a ranking or measured performance transfers to another one.

This entry point publishes each accepted candidate as its own one-solution
bundle, `bundle-0`, `bundle-1` and so on in ranked order, all at once, and
`bundle` links to `bundle-0`. When the candidates run out after at least one
was accepted, it publishes the bundles it has. Each bundle additionally
contains `jit_prediction`. The record includes selected tuning parameters,
descriptor `implementation_parameters`, defaults, resolved values, candidate
rejections, and either a modeled cost or null. The selected YAML of rank 0 is
retained as `<output>.yaml` next to `<output>.prediction.json`, and that of rank
r as `<output>.<r>.yaml`. Generation never benchmarks candidates.

## hipBLASLt provider integration

The library's TensileLite backend creates the ranked request internally
from a generic operation request. Its data-parallel GEMM contract supplies
`MatrixInstruction`, macro tile/`DepthU`, `NonTemporalA/B`, workgroup mapping,
stagger and launch outputs. This module retains the supplied order, translates
model units, and rejects unsupported or changed predictions before compilation. Descriptor
scale modes are translated here, separately from tuning parameters, so the C++
caller does not repeat target-dependent layout rules.

The backend runs this module and `Tensile.SingleSolution` with `--source-only`
and the code-object version it builds with, and adds `requested_solutions` when
it needs more than one solution. It reads each published bundle by the layout
above and builds one raw executable code object per solution in process through
comgr, linking the main kernel assembly and the helper source together. The
[JIT guide](../JIT.md#code-object-construction-with-comgr) describes that build.

No ranking means no generation. Exhausted rankings report each rejection;
neither case substitutes an unranked recipe. The [benchmark guide](../clients/bench/README.jit.md)
shows this integration through public hipBLASLt execution APIs.
