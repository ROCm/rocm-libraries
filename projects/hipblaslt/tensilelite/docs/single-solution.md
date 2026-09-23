# Generate and build one GEMM solution

`Tensile.SingleSolution.generateAndBuildSingleSolution` compiles one GEMM
solution from a TensileLite YAML recipe. A solution includes the main assembly
kernel, any helper kernels and support code, and the metadata needed by the
TensileLite runtime. This entry point does not run GPU work or measure kernel
performance.

The YAML supplies an exact recipe using the existing problem and parameter
schema. `SingleSolution` uses TensileLite's full target and solution validators.

The [component roadmap](jit-roadmap.md) explains how this builder connects to
ranked selection, a generic JIT API, and application execution.

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
and the architecture. It raises `SingleSolutionError`, or its configuration or
build subclass, on failure.

The CLI prints progress followed by the manifest path. It reports failures on
stderr and returns a nonzero exit status. Programs invoking it should check
that status and then read `new-request-output/bundle/manifest.json`; stdout is
human-readable progress, not a JSON response.

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
point. Non-null obsolete benchmark steps and parameter-level `CustomKernel`
are rejected. `NoReject` cannot disable validation.

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
(`msgpack` or `yaml`), and temporary-file retention are Python and CLI options.
Invocation options select the format and toolchain; contradictory YAML target
settings are rejected. The selected library format must be enabled in the
consuming host library.

Both main and helper code objects use the requested code-object version.
Helper compilation bypasses the shared helper cache so the bundle reflects
the selected sources and toolchain. Source-only output, forced invalid-kernel
generation, and profiling conflict with this entry point and are rejected.
Benchmark timing, result export, clock control, monitoring, library-logic
analysis, and client generation do not run. Generation uses one worker.

Generation uses shared Python configuration and rocisa capability caches.
Calls must not overlap other TensileLite generation in the same process, and
the compiler identity must remain fixed between calls. This entry point rejects
overlapping calls to itself and changes of compiler identity. A fresh subprocess
with a private working directory for each request isolates that shared state
and tool temporary files.

## Bundle contents and publication

Each request creates its output directory exclusively and builds in a private
`.staging` directory. After compilation and metadata serialization succeed,
that directory is atomically renamed to `bundle`. A failure leaves staging
files for diagnosis and publishes no bundle. Retrying requires a new output
path, so an existing result is never overwritten.

`keepBuildTmp=True` retains intermediate assembly and object files in the
published bundle. Generated helper sources and required headers are retained
in `support_files` whenever the solution needs them.

The manifest uses `schema_version: 2`. Artifact paths are relative to the
bundle directory:

| Field | Contents |
| --- | --- |
| `architecture` | Requested and resolved architecture, plus compiler target |
| `main_kernel` | Main entrypoint name and code-object path |
| `code_objects` | All main `.co` and helper `.hsaco` files |
| `helpers` | Generator kind (`kernel_family`, `header`, or `device_function`), class, and family/support name |
| `support_files` | Generated helper sources and required headers |
| `solution` | Local index, solution name, and kernel name |
| `library` | Format, logical path, and physical path |
| `counts` | One solution, one main kernel, and helper/support generator counts |
| `provenance` | YAML hash, generator version/revision, compiler identity, code-object and kernel-argument versions, and temporary-retention choice |

For MsgPack, `library.path` names the physical `.dat.zlib` file and
`library.logical_path` names `.dat`; the runtime loader understands both.

The serialized solution library describes runtime support predicates,
workspace, and kernel arguments. A host runtime loads every listed code object,
checks the requested problem, allocates workspace, and uses the ordered
invocation sequence from `ContractionSolution::solve`. Helper-generator counts
and code-object counts do not determine the number of launches; the runtime
problem and selected accumulation path determine that sequence.
