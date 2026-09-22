# Generate and build one GEMM solution

`Tensile.SingleSolution.generateAndBuildSingleSolution` reads the existing Tensile
YAML problem and parameter schema and builds exactly one valid GEMM solution:
its main assembly kernel, all helper kernel families, required support code and
runtime solution metadata. It does not run benchmarks, a benchmark client,
library-logic analysis or GPU work.

Automatic callers use `Tensile.JitGemm` to validate Origami-ranked parameter
recipes before invoking the same builder once. The explicit YAML entry point
described here does not perform prediction and retains the normal parameter
schema. Automatic bundles additionally record ranking, rejected candidates,
selected parameters, defaults and derived values in `jit_prediction`. Automatic
prediction supports gfx90a, gfx942, and gfx950 with target-specific matrix
instructions and cache hints. The explicit YAML entry point keeps the normal
Tensile ISA support surface; it is not restricted to those three targets.

```python
from Tensile.SingleSolution import generateAndBuildSingleSolution

result = generateAndBuildSingleSolution(
    "problem.yaml",
    "new-request-output",  # Must not already exist.
    architecture="gfx950",
    libraryFormat="msgpack",
)
print(result.manifestPath)
```

The equivalent CLI is:

```text
python -m Tensile.SingleSolution problem.yaml new-request-output --architecture gfx950
```

Use the interpreter and module paths configured for your locally built checkout.
The optional `TensileGenerateSingleSolution` script calls the same entry point.
The CLI prints progress followed by the manifest path; its stdout is not a JSON
protocol. A process caller checks its exit status, then reads the known
`new-request-output/bundle/manifest.json` path.

The callable returns a frozen `SingleSolutionBuildResult` with absolute bundle,
manifest, physical library and logical library paths, `codeObjectPaths` (a tuple
of all built objects), `mainCodeObjectPath`, main `kernelName`, solution name,
local solution index and architecture. It raises
`SingleSolutionError` (with configuration/build subclasses) on failure instead of
exiting the caller. The CLI reports failure on stderr and returns a nonzero
status.

## Input restrictions

Supply exactly one `BenchmarkProblems` entry containing a problem type and one
parameter group. `BenchmarkCommonParameters`, `ForkParameters`, singleton
`Groups`, and `BenchmarkFinalParameters` retain their existing meanings. The YAML
supplies the concrete solution recipe: the API does not tune or choose among
candidates. Empty candidate lists and requests expanding to several parameter
combinations are rejected before solution filtering. `MatrixInstruction: [[]]`
is one empty-valued candidate, distinct from an empty candidate list.

Parameter types, names, allowed values, matrix instructions and derived solution
properties use the ordinary generator validators. Multiple problem sizes do not
request multiple solutions; they are parsed but never timed or used to pick a
configuration. Custom kernels, solution pools and alternate tuning modes are
outside this entry point. Non-null obsolete benchmark-step sections are rejected.
Parameter-level `CustomKernel` is also rejected, and `NoReject` cannot be enabled
to suppress solution validation, including in common, fork or grouped parameters.

The fully derived solution determines which helpers are built. Beta-only
initialization, output conversion and bias-gradient reduction can require extra
GPU entrypoints. A helper generator can emit several variants, such as output
conversion at different vector widths and GSU values; those variants belong to
the same solution. Activation enum headers and inline device functions are
support code, not independent kernel launches. The normal generator's helper
enumeration may include variants that a particular runtime problem never uses.

Fixed `GlobalSplitU` values greater than one, automatic `GlobalSplitU: [-1]`, and
`AdaptiveGemmGSUA: [1]` are accepted when the ordinary solution validators allow
them. GSU splits the K reduction across workgroups. `GlobalSplitU: -1` chooses
the split count at runtime; `AdaptiveGemmGSUA` chooses between MultipleBuffer
(separate output reduction) and MultipleBufferSingleKernel (reduction inside the
main kernel). The unrelated `AdaptiveGemm` parameter selects store-width paths.
The API does not force these parameters off.

Examples under `Tensile/Tests/unit/test_data/`:

| Recipe | Configuration |
| --- | --- |
| `single_solution.yaml` | FP16/HPA, GSU1, MultipleBufferSingleKernel, no helpers |
| `single_solution_splitk.yaml` | FP16/HPA, GSU4, MultipleBuffer, conversion and beta helper families |
| `single_solution_adaptive.yaml` | FP16/HPA, automatic GSU and adaptive accumulation, conversion and beta helper families |
| `single_solution_streamk.yaml` | FP16/HPA, non-atomic StreamK3 with partial-tile reduction |
| `single_solution_amax.yaml` | FP16/HPA, GSU1, unscaled output-amax with packed stores |

Explicit Stream-K configurations use normal
solution validation. Output-amax currently requires `GlobalSplitU: 1` and
`StreamK: 0`; its serialized solution also requires batch size one. Its current
reduction needs final output and does not reduce across batch offsets. These
limits apply through the normal validators and runtime predicates, including
when consuming the bundle through either hipBLASLt JIT API. The experimental hipBLASLt JIT GEMM route is
documented with `clients/samples/29_hipblaslt_jit_gemm/` in the parent project.

The target is explicit; GPU enumeration is unnecessary. Compiler, bundler,
code-object version (`4`, `5`, or `6`), library format (`msgpack` or `yaml`), and
temporary retention are keyword/CLI options. Contradictory YAML target identity
is rejected. Invocation options govern format and toolchain. Source-only exit,
forced invalid-kernel generation and profiling conflict with this API and are
rejected. Benchmark orchestration settings (timing, iteration counts, result
export, clocks, monitoring, caches, LibraryLogic and LibraryClient) do not cause
those workflows to run. Generation uses one worker and disables monitoring and
clock control. Main and helper code objects use the requested code-object
version. Helper compilation bypasses the shared helper cache so every bundle
is built from the current sources and configured toolchain.

## Output and process lifetime

Each request exclusively creates its output directory and builds in private
`.staging`. After generation, assembly, linking and metadata serialization
succeed, the complete directory is atomically renamed to `bundle`. Failure
leaves staging for diagnostics and never publishes a bundle. Retries require a
new output path; existing results are never overwritten. `keepBuildTmp=True`
retains intermediate assembly/object files inside the published bundle. Helper
source and required headers are retained and listed in `support_files` whenever
helper/support generators are present.

The JSON manifest is versioned with `schema_version: 2`. All artifact paths in
it are relative to the bundle directory:

| Field | Contents |
| --- | --- |
| `architecture` | `requested`, `resolved`, and `compiler_target` |
| `main_kernel` | Main entrypoint `name` and its `code_object` |
| `code_objects` | Exhaustive list of relative paths to main `.co` and helper `.hsaco` files |
| `helpers` | Generator descriptions: `kind` (`kernel_family`, `header`, `device_function`), `generator` class and family/support `name` |
| `support_files` | Relative paths to emitted helper sources and required headers |
| `solution` | Local `index`, `name`, and `kernel_name` |
| `library` | `format`, `logical_path`, and physical `path` |
| `counts` | `solutions: 1`, `main_kernels: 1`, `helper_generators` and `support_generators` |
| `provenance` | YAML SHA256, generator version/revision (null without Git metadata), compiler identity, code-object and kernarg versions, temporary-retention choice |

For MsgPack, `library.path` names the actual `.dat.zlib` file while
`library.logical_path` names `.dat`; the production loader understands both.
Select the format compiled into the consuming host library. The serialized
solution library is authoritative for predicates, size mapping and launch ABI;
the manifest does not replace it. Loading code objects, checking runtime problem
support, allocating workspace, packing arguments and launching remain the host
runtime's responsibility. Load every listed code object into the solution's
adapter and use the production `ContractionSolution::solve` invocation sequence.
`helpers[*].name` describes a generator family or support object; it is not an
exhaustive symbol table. Neither helper generator counts nor code-object counts
predict the number of launches. Runtime problem size, selected accumulation
path and epilogue determine the actual sequence.

Generation uses shared Python configuration and rocisa capability caches. Calls
must not overlap any other Tensile generation in the same process; this API
rejects overlapping calls to itself. It also rejects changing compiler identity
between its calls. Prefer one fresh subprocess per request, with a private
working directory. This isolates capability state and tool temporary files.
