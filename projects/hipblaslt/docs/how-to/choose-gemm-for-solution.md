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
loaded indices, including features not implemented yet, such as sparse GEMM. Repeating a choice with the same
arguments returns an equal problem description, without more file reads or GPU
work. `reduction_blocks` requests a number of complete K blocks;
changing it changes the chosen K size as described below. The optional `alpha`
and `beta` arguments select real scalar values; they default to 1 and 0.

For a grouped solution, the descriptor includes two `(M, N, K)` members in
`grouped_members`. Both have the chosen shape. A consumer supplies independent
buffers for the two members and uses the descriptor's required bias and scalar
input scaling. Constructing a descriptor does not allocate those buffers.

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
| Solution metadata | `sizeMapping.globalSplitU` | Fixed number of workgroups sharing the K reduction, or -1 when the runtime chooses |
| Caller argument, default 8 | `reduction_blocks` | Complete reduction blocks per fixed split, or per output tile with persistent scheduling |

For ordinary fixed-split solutions, K is `reduction_blocks * depthU * globalSplitU`. For a 128-by-96 output tile,
`depthU=32` and split factor 4, the default problem is **M=128, N=96, K=1024**.
Four workgroups share the output tile, each processing eight blocks of 32
reduction elements. These include prefetch and final stages; eight blocks does
not necessarily mean eight steady-state loop backedges.

Automatic split-K solutions (`globalSplitU=-1`) use K =
`2 * reduction_blocks * depthU`. For one output tile, the runtime splits K into
floor(K / depthU / 3) parts, at least one and at most the device CU count. With
`reduction_blocks=3`, two workgroups each process three blocks; the default of
eight gives five workgroups sharing 16 blocks.

For Stream-K and persistent DataParallel solutions, K is
`reduction_blocks * depthU`. Their `globalSplitU` does not multiply the workload
and can be zero. The native runtime determines the persistent grid and assigns
the reduction work. For the same tile and `depthU=32`, the default is
**M=128, N=96, K=256**. Canonical scheduling fields and their legacy StreamK
equivalents are normalized and validated before construction.

Serialized size predicates can require a different shape. The helper honors
size bounds, equalities and divisibility constraints, including minimum K for
fixed split-K. It also applies the runtime's split-K synchronizer size check to
the chosen tile. It keeps the requested dimensions when they fit, otherwise
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

When executed by original index, this small shape targets the indexed kernel.
It does not guarantee that every Stream-K work-distribution or cross-workgroup reduction path executes;
that needs additional shapes and runtime settings.

Storage and compute types, transposes and conjugation come from the solution.
The chosen problem uses one batch, real alpha 1 and beta 0 by default, and no
optional bias, activation or scales. A nonzero beta requires a solution with
beta support; integer computation requires integral scalars. Complex scalars
have zero imaginary part. Required MX operand scales are preserved. Storage is packed column-major: element `(row, col)` has offset
`row + col * leading_dimension`. Each leading dimension is the stored row
count; transposing A or B changes that count. C and D are M-by-N and have no
transpose operation in the hipBLASLt matmul descriptor.

### Gradient, E and amaxD outputs

Some solutions compute extra outputs, and the problem enables them. A gradient
solution (`gradient`) computes a bias gradient, reduced from the first allowed
bias source (A, B or D). A solution with an E tensor (`aux_type` is its type)
is given ReLU: forward solutions write E, and gradient solutions read it and
apply the ReLU derivative. Both also get a `bias`, of type Float if the
solution allows it, otherwise its first allowed type. An amaxD solution
(`amax_d`) also returns the absolute maximum of D. A solution whose allowed
bias sources or activations exclude these choices is unsupported.

### Swizzled operands

Solutions can require A or B to use a swizzled layout, which rearranges small
blocks of input elements in device memory. The descriptor preserves this
requirement as `swizzle_a` and `swizzle_b`. Its dimensions and strides describe
the logical column-major inputs. The execution adapter is responsible for
packing those inputs into the required device layout.

### MX block-scaled operands

Each operand retains its `mxBlockA/B` block size and `mxTypeA/B` scale datatype.
The descriptor also retains `mxScaleFormat`: 0 for ordinary scales, 1 for
gfx950 host-preswizzled scales, and 2 for gfx1250 TDM scales. A zero block size
means the operand is unscaled. These scales are part of the multiplication,
so they cannot be disabled like an optional epilogue.

These fields describe the required scale layout; constructing a descriptor does
not qualify the target or establish that a particular client supports it.

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
Loading or choosing can instead raise an exception. This table explains
what the caller receives before any GPU execution:

| Returned object or raised exception | What the caller does next |
| --- | --- |
| Returns `GemmProblem` | Format and execute the chosen input; native support is still unchecked |
| Raises `UnsupportedSolution` | Record the index and `reasons` as features the generator has not implemented |
| Raises `KeyError` | Investigate a missing index or required metadata field |
| Raises `ValueError` | Investigate an invalid argument, datatype, scheduling value, duplicate index or corrupt metadata |

## Account for unsupported indices

A caller can keep construction failures distinct from malformed metadata:

```python
from tensilelite.Utilities.gemm_problem_for_solution import UnsupportedSolution

for index in library.indices:
    try:
        problem = library.choose_single_problem(index)
    except UnsupportedSolution as error:
        print(error.solution_index, error.reasons)
        continue
    print(problem.solution_index, problem.m, problem.n, problem.k)
```

An unsupported index remains a coverage gap. A missing required field or corrupt
metadata raises a separate error and escapes this handler. A constructed problem
still needs native support checks, numerical validation and any requested race
checks before it can count as a validated solution.
