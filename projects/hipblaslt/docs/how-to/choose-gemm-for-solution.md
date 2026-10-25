# Choose one GEMM problem from a solution index

A hipBLASLt solution describes a GPU function and the conditions under which it
can multiply matrices. One solution can accept many matrix sizes. Given its
index in a built device library, `choose_single_problem()` chooses one small
matrix multiplication to try. It reads metadata without running the function;
native support checks and execution happen afterward.

## Load once, then choose problems

Use the TensileLite environment and device library from your hipBLASLt build.

```python
from tensilelite.Utilities.gemm_problem_for_solution import load_library_metadata

library = load_library_metadata("/path/to/library", target="gfx950")
solution_index = 17  # Replace with an index from this library build.
problem = library.choose_single_problem(solution_index)
print(problem.m, problem.n, problem.k)
```

Keep `library` and reuse it for subsequent indices. `library.indices` lists the
loaded indices, including features not implemented yet, such as grouped GEMM
and required gradient operations. Repeating a choice with the same
arguments returns an equal problem description, without more file reads or GPU
work. `reduction_blocks` requests a number of complete K blocks;
changing it changes the chosen K size as described below.

If the indices are already known, load them together:

```python
library = load_library_metadata(
    "/path/to/library", target="gfx950", solution_indices=[17, 23]
)
first = library.choose_single_problem(17)
# Increase the amount of reduction work from the default 8 blocks.
second = library.choose_single_problem(23, reduction_blocks=12)
```

Only those indices are retained; choosing an unloaded index raises `KeyError`.
Indices belong to a particular build, so execution must use the same library.

## How the dimensions are chosen

The helper first checks that it can handle the solution's scheduling and matrix
options. It starts with one full output tile, a rectangular chunk of the
result matrix, with these dimensions:

| Where the value comes from | Value | Use |
| --- | --- | --- |
| Solution metadata | `sizeMapping.macroTile[0:2]` | M and N, the output tile's rows and columns |
| Solution metadata | `sizeMapping.depthU` | Number of K elements in one reduction block |
| Solution metadata | `sizeMapping.globalSplitU` | Fixed number of workgroups sharing the K reduction |
| Caller argument, default 8 | `reduction_blocks` | Complete reduction blocks per fixed split, or per output tile with persistent scheduling |

For ordinary fixed-split solutions, K is `reduction_blocks * depthU * globalSplitU`. For a 128-by-96 output tile,
`depthU=32` and split factor 4, the default problem is **M=128, N=96, K=1024**.
Four workgroups share the output tile, each processing eight blocks of 32
reduction elements. These include prefetch and final stages; eight blocks does
not necessarily mean eight steady-state loop backedges.

For Stream-K and persistent DataParallel solutions, K is
`reduction_blocks * depthU`. Their `globalSplitU` does not multiply the workload
and can be zero. The native runtime determines the persistent grid and assigns
the reduction work. For the same tile and `depthU=32`, the default is
**M=128, N=96, K=256**. Canonical scheduling fields and their legacy StreamK
equivalents are normalized and validated before construction.

Serialized size predicates can require a different shape. The helper honors
size bounds, equalities and divisibility constraints, including minimum K for
fixed split-K. It keeps the requested dimensions when they fit, otherwise
chooses dimensions within the bounds. K remains a multiple of the reduction
block and fixed split; required MX block sizes also constrain K. For example,
a 32-by-32 tile with `depthU=64` would normally give K=512. If its predicates
require M/N from 64 through 128 and K=128, the default becomes **64-by-64-by-128**.
Thus, `reduction_blocks` is a target that explicit size bounds can override.

Contradictory or malformed size predicates raise `ValueError`. Consistent
constraints that require several batches or exclude complete reduction blocks
raise `UnsupportedSolution`. Dimension constraints under disjunction or negation
are also unimplemented. Other native predicates, including hardware and
buffer-offset checks, remain the runtime's responsibility.

This small shape runs the original indexed kernel. It does not guarantee that
every Stream-K work-distribution or cross-workgroup reduction path executes;
that needs additional shapes and runtime settings.

Storage and compute types, transposes and conjugation come from the solution.
The chosen problem uses one batch, alpha 1, beta 0 and no bias, activation or
optional scales. Required MX operand scales are preserved. Storage is packed column-major: element `(row, col)` has offset
`row + col * leading_dimension`. Each leading dimension is the stored row
count; transposing A or B changes that count. C and D are M-by-N and have no
transpose operation in the hipBLASLt matmul descriptor.

### MX block-scaled operands

Each operand retains its `mxBlockA/B` block size and `mxTypeA/B` scale datatype.
The descriptor also retains `mxScaleFormat`: 0 for ordinary scales, 1 for
gfx950 host-preswizzled scales, and 2 for gfx1250 TDM scales. A zero block size
means the operand is unscaled. These scales are part of the multiplication,
so they cannot be disabled like an optional epilogue.

The bench adapter uses the native client's scale-buffer allocation, MX input
generation and reference calculation. For gfx950 MXFP4/MXFP8 solutions with
32-element E8 blocks and format 1, it emits `scaleA: 1001` and `scaleB: 1001`,
preserving the required TN orientation. These numbers are client scaling modes,
not public hipBLASLt API enum values. Ordinary block scales use client modes
3 through 8 according to their block size and scale type; gfx1250 scale layout
preparation is handled by the client.

MX inputs use bounded `hpl` initialization to avoid FP16 output overflow in
long reductions. Other inputs keep `rand_int`. Numerical assertions and
allclose checks remain enabled. The client must be built with
`HIPBLASLT_ENABLE_MXDATAGENERATOR=ON`.

## What selective loading saves

Loading reads every matching `.dat` or `.dat.zlib` file, even for one index.
Those files contain solution descriptions and selection tables; `.zlib` means
the file is compressed. Loading accepts the target directory or its parent,
and a compressed file takes precedence over an uncompressed copy.

The indexed format has a lookup table, so only selected solution records need
decoding. The older format stores an array and requires decoding every record.
Both still pay for reading and decompressing all matching files. Selecting
fewer indices reduces retained data; with the older format it mainly
saves the cost of keeping those decoded objects alive.

For example, measurements on one 84,498-entry gfx950 library in the older
format gave these median load times over three fresh processes:

| Requested indices | Load time |
| --- | --- |
| 1 | About 1.8 seconds |
| 1,000 | About 1.8 seconds |
| All 84,498 | About 3.6 seconds |

All three read the same 388 files. Thus, a two-index selection does not take
twice as long as a one-index selection: for small selections, the file scan
dominates. Indexed libraries can save more decoding work, and record sizes also
affect the cost. Load once outside the loop over indices; subsequent choices
take about 5 microseconds in this example. Load a new catalog if the library
files change.

## What calls return or raise

Choosing a problem returns a `GemmProblem` object describing the input.
Loading, choosing or formatting can instead raise an exception. This table explains
what the caller receives before any GPU execution:

| Returned object or raised exception | What the caller does next |
| --- | --- |
| Returns `GemmProblem` | Format and execute the chosen input; native support is still unchecked |
| Raises `UnsupportedSolution` | Record the index and `reasons` as features the generator has not implemented |
| Raises `UnsupportedBenchProblem` | Record the index and `reason` as a limitation of the benchmark adapter |
| Raises `KeyError` | Investigate a missing index or required metadata field |
| Raises `ValueError` | Investigate an invalid argument, datatype, scheduling value, duplicate index or corrupt metadata |

## Examples from selection to outcome

### The selected function runs successfully

This is the complete path from a built library to a benchmark input. Replace
the path, target and index with your library and a supported solution:

```python
from pathlib import Path
from tensilelite.Utilities.gemm_problem_for_solution import load_library_metadata
from tensilelite.Utilities.gemm_problem_bench import bench_yaml

library = load_library_metadata("/path/to/library", target="gfx950")
problem = library.choose_single_problem(17)
Path("problem.yaml").write_text(bench_yaml([problem]))
print("Library:", library.directory)
print("Expected index:", problem.solution_index)
print("Expected kernel:", problem.kernel_name)
print("Shape:", problem.m, problem.n, problem.k)
```

Use the printed library directory and the matching benchmark:

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx950 \
  hipblaslt-bench --yaml problem.yaml > bench.log 2>&1
echo "Exit status: $?"
cat bench.log
```

A successful run exits with status zero and prints one result row with numerical
checks and the expected solution index and kernel name. A tested gfx950 BF16
example chose 16-by-16-by-512 and reported `norm_error` of zero. Your dimensions
and numerical error depend on the chosen solution.

The generated YAML enables numerical validation, sets `cold_iters=0` and
`iters=0`, and disables adaptive timing. Validation performs the one target
execution; initialization and split-reduction helpers can launch separately.
Use a benchmark containing this zero-timed-iteration behavior.

### The function runs but fails numerical validation

Consider a hypothetical FP32 run of the same commands where the computed result
has relative norm error 0.001. This exceeds the FP32 check's tolerance of
0.00001, so the enabled `norm_check_assert` makes the benchmark exit with status
1, possibly before printing a result row. Retain the input and log as a failed
execution, not an unsupported solution.

To inspect the numerical error in that case, make a diagnostic copy:

```python
from pathlib import Path
import yaml

rows = yaml.safe_load(Path("problem.yaml").read_text())
rows[0]["norm_check_assert"] = False
Path("diagnostic.yaml").write_text(yaml.safe_dump(rows))
```

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx950 \
  hipblaslt-bench --yaml diagnostic.yaml
```

The diagnostic row exposes `norm_error`; disabling the assertion does not make
the original case a pass. A nonzero exit can also have other causes, so inspect
the logs rather than classifying every failure as a numerical error.

### The function runs and a race is reported

Run the same `problem.yaml` under rocJITsu, using a matching device configuration
with the race and logging plugins enabled:

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx950 \
  rocjitsu --config /path/to/race-and-logging.json -- \
  hipblaslt-bench --yaml problem.yaml > race-run.log 2>&1
echo "Exit status: $?"
cat race-run.log
```

In a hypothetical failing run, the race report contains a `RACE` entry even
though the numerical result might match and the process might exit zero.
Record that case as a race failure and retain the report. With plugins writing
to files, inspect their output files too. Use the logging plugin's
`Kernel dispatch:` entries to check that the expected target ran exactly once;
helper kernels have their own entries. A run with matching numerics, the
expected dispatch and no race or emulator errors passes these checks.

### An index cannot produce a problem yet

This complete example finds and reports the first unimplemented entry in a
library, without executing anything:

```python
from tensilelite.Utilities.gemm_problem_for_solution import (
    UnsupportedSolution,
    load_library_metadata,
)

library = load_library_metadata("/path/to/library", target="gfx950")
for index in library.indices:
    try:
        library.choose_single_problem(index)
    except UnsupportedSolution as error:
        print(error.solution_index, error.reasons)
        break
else:
    print("Every loaded index produced a problem")
```

Current exclusions include automatic split-K, grouped or sparse GEMM,
required output operations, unusual
layouts and contractions, and different C/D storage types. Record the reason
and continue a sweep if desired. Catch only `UnsupportedSolution`; malformed
metadata and unexpected errors must remain visible.

### Formatting or input validation fails

The adapter can reject a constructed description. For example, this synthetic
variation on the successfully chosen problem exercises its unsupported-compute
path without depending on another library index:

```python
from dataclasses import replace
from tensilelite.Utilities.gemm_problem_bench import bench_yaml

bench_yaml([replace(problem, compute_type="Half")])
# Raises UnsupportedBenchProblem: the adapter cannot express Half computation.
```

After loading a library, each of these separate calls exercises an input error:

| Call | Exception |
| --- | --- |
| `library.choose_single_problem(-1)` | `ValueError`: indices must be nonnegative integers |
| `library.choose_single_problem(17, reduction_blocks=0)` | `ValueError`: block count must be positive |
| `library.choose_single_problem(max(library.indices) + 1)` | `KeyError`: that index was not loaded |

Finally, successful construction and formatting can still end in native
rejection, for example when the solution requires a different compute-unit
count. The benchmark can then exit zero with no result and no target dispatch.
Record that as a native rejection; it provides no execution coverage.
