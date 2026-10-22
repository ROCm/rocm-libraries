# Choose one GEMM problem from a solution index

A hipBLASLt solution describes a GPU function and the conditions under which it
can multiply matrices. A solution index selects that description within one
built device library. To exercise a particular function, start with its index
and construct an input using `load_library_metadata()` and `get_problem()`.

This guide is the entry point for that flow. The helper chooses one small,
packed matrix multiplication for an ordinary fixed-split solution. It reads
metadata without a GPU or the kernel generator. Native support checks and
execution happen later; construction alone does not establish that the chosen
function will accept the input.

## Load once, then choose problems

Use Python 3.10 or newer with the `tensilelite` package, MessagePack (`msgpack`)
and PyYAML available. In a source checkout, the package root is
`projects/hipblaslt/tensilelite`; put it on `PYTHONPATH` when running the example.
Use a device library produced by a matching hipBLASLt build.

```python
from tensilelite.Utilities.gemm_problem_for_solution import load_library_metadata

library = load_library_metadata("/path/to/library", target="gfx950")
solution_index = 17  # Replace with an index from this library build.
problem = library.get_problem(solution_index)
print(problem.m, problem.n, problem.k)
```

Keep `library` and call `get_problem()` for each index you want to exercise.
`library.indices` contains the loaded indices in sorted order, including entries
whose features this generator does not yet implement. Repeating a call with the
same index and arguments returns an equal descriptor. It does not read files,
search sizes, allocate matrix buffers or execute a GPU function.

If the indices are already known, pass them together when loading:

```python
library = load_library_metadata(
    "/path/to/library", target="gfx950", solution_indices=[17, 23]
)
first = library.get_problem(17)
second = library.get_problem(23, reduction_blocks=12)
```

Selective loading retains only those records. An index that was not loaded
raises `KeyError` when requested later. Indices are local to a library build;
reuse the same library directory and target when executing its problems.

## How the solution determines the problem

The generator reads the selected record's problem type and size mapping. It
first checks that the scheduling mode and required features fit the supported
baseline, then constructs the descriptor using the following rules.

| Input from the solution | Result in the GEMM problem |
| --- | --- |
| `sizeMapping.macroTile[0:2]` | M and N, the rows and columns of one full output tile |
| `sizeMapping.depthU` | Number of K elements in one reduction block |
| `sizeMapping.globalSplitU` | Fixed number of workgroups sharing the K reduction |
| `reduction_blocks`, a positive integer argument defaulting to 8 | K = `reduction_blocks * depthU * globalSplitU` |
| `problemType` storage and compute types | A/B/C/D storage types and computation types |
| A/B transpose flags and `operationIdentifier` | Transposes and complex conjugation |
| Solution index, names, directory and GPU target | Information used to select and identify the function in this build |

For an output tile of 128 by 96 elements, `depthU=32` and fixed split factor 4,
the default problem is **M=128, N=96, K=1024**. Four workgroups share that output
tile, each processing eight complete blocks of 32 reduction elements. The
blocks include prefetch and final stages; eight blocks does not mean eight
steady-state loop backedges.

The baseline uses one batch, alpha 1, beta 0, and no bias, activation or optional
scales. These settings are fixed by this descriptor's contract and are written
by the benchmark adapter. Storage is packed column-major: element `(row, col)`
has offset `row + col * leading_dimension`. Leading dimensions equal the
stored row counts; transposing A or B changes those counts. C and D are M-by-N
and have no transpose operation in the hipBLASLt matmul descriptor.

The generator uses the shared datatype definitions and binary metadata reader.
It preserves datatype names independently of what a particular execution
client can express. There is no architecture-specific size search or execution
budget in this operation.

## Execute the selected problem

The separate `bench_yaml()` adapter writes `hipblaslt-bench` input:

```python
from pathlib import Path
from tensilelite.Utilities.gemm_problem_bench import bench_yaml

Path("problem.yaml").write_text(bench_yaml([problem]))
print(library.directory)
```

Set the library path to the directory printed above and use the matching
benchmark executable:

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx950 hipblaslt-bench --yaml problem.yaml
```

The adapter selects the explicit solution index and requests numerical
validation. With `cold_iters=0`, the benchmark still runs once to obtain the
result being checked. `iters=0` disables additional fixed-count timed launches,
and `adaptive=False` disables adaptive timing. Initialization and split-reduction
helpers can launch separately.

Use a benchmark containing this zero-iteration behavior. Successful numerical
validation plus the reported solution index and kernel name establish which
problem was executed. A dispatch log can additionally verify that the target
function ran exactly once. A zero process exit code alone is insufficient: a
native support rejection can produce no result and no target launch. Race
reports, numerical failures, native rejections and emulator errors need their
own outcomes in a test campaign.

## Unsupported features and errors

| Result | Meaning and caller action |
| --- | --- |
| `GemmProblem` | A constructed baseline input; native solution support still needs checking |
| `UnsupportedSolution` | An explicitly unimplemented generator feature; record its index and `reasons` if continuing a sweep |
| `UnsupportedBenchProblem` | A descriptor the benchmark adapter cannot express; its index and `reason` describe that adapter limitation |
| `KeyError` | A requested index or required record field is missing; investigate the input |
| `ValueError` | An invalid argument, datatype, scheduling value, duplicate index or corrupt metadata; investigate the input |

Automatic split-K and persistent scheduling, including Stream-K, are deferred.
So are block-scaled MX inputs, grouped or sparse multiplication, required output
operations, and other layouts outside the packed GEMM baseline. The exception
names the excluded features. Catch the specific exclusion when scanning indices:

```python
from tensilelite.Utilities.gemm_problem_for_solution import UnsupportedSolution

gaps = {}
for index in library.indices:
    try:
        problem = library.get_problem(index)
    except UnsupportedSolution as error:
        gaps[index] = error.reasons
        continue
    # Pass this descriptor to the execution step in the campaign.
```

Unexpected errors should escape that handler. The formula does not solve
arbitrary native predicates, such as a required size or hardware configuration.
Matching the GPU target alone does not guarantee matching a required device ID
or compute-unit count. This one problem also leaves output edges, reduction
tails and varied scalar or output-operation settings untested.

## Loading cost and reuse

Loading accepts either the directory containing the metadata or its parent
with a target subdirectory. It reads each matching `.dat` or `.dat.zlib` shard
once; a compressed copy takes precedence over its uncompressed sibling. The
binary files contain solution descriptions and selection tables. The older
MessagePack format stores descriptions in an array; the indexed format adds a
lookup table for individual records.

Both formats require reading and decompressing entire matching shards, even
when selecting one index. Indexed metadata decodes only selected records;
the older format decodes every record. Selection tables are skipped, and the
catalog retains only the metadata used to construct problems.

Load once per library build and target, outside the loop over solution indices.
Selecting known indices together reduces retained memory. Each subsequent
`get_problem()` call uses a dictionary lookup and work proportional to the
selected record, independent of the total library size. The catalog does not
refresh itself if the files change; load a new catalog for a new build.

The [integration test](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/tensilelite/tensilelite/Tests/common/test_gemm_problem_for_solution_integration.py)
builds eight small gfx942 libraries covering FP32, FP64, FP16 and BF16 with split
factors 1 and 4. It checks geometry, numerical validation and function identity,
and verifies one target dispatch when rocJITsu logging is active. The test also
supports the common suite's build-only and cached-artifact flow.
