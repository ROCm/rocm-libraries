# One regular GEMM per solution

`gemm_problem_for_solution` constructs one packed GEMM descriptor from a solution's library
metadata. It is a starting workload for race detection: one full output tile and
eight full reduction blocks per split workgroup. It performs no size search and
takes no execution budget. The `reduction_blocks` argument defaults to eight and
controls the reduction work in this one problem.

```python
from tensilelite.Utilities.gemm_problem_bench import bench_yaml
from tensilelite.Utilities.gemm_problem_for_solution import UnsupportedSolution, load_library_metadata

library = load_library_metadata("/path/to/library", target="gfx942",
                                solution_indices=[17])
problem = library.get_problem(17, reduction_blocks=8)
print(problem.m, problem.n, problem.k)
with open("regular.yaml", "w") as output:
    output.write(bench_yaml([problem]))
```

For macro tile `[128, 96]`, `DepthU=32` and fixed split factor `4`, the problem is
`M=128, N=96, K=1024`. Each of the four split workgroups processes eight blocks
of 32 reduction elements. These blocks include prefetch and final-loop stages;
the number of steady-state loop backedges can differ. There is one batch,
`alpha=1`, `beta=0`, packed column-major strides, and no bias, activation or scales.
Storage and compute types and transposes come from the solution.

Run the YAML with the matching client and device library:

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/library/gfx942 hipblaslt-bench --yaml regular.yaml
```

The formatter selects the explicit solution index and enables numerical checks.
It requests zero warmup and timed iterations. Validation performs the one GEMM
execution; split-K may also launch reduction helpers. This requires a client
that honors zero timed iterations. Race reports, client rejection, numerical
failure and emulator failure are separate results to retain in a campaign.
A rejection does not provide kernel coverage.

Automatic split-K, persistent/Stream-K scheduling, MX inputs, grouped or sparse
GEMM, and mandatory epilogue features are unsupported and raise
`UnsupportedSolution`. This exception includes the index and feature names, so a
sweep can record the untested solutions and continue:

```python
problems, gaps = [], {}
for index in library.indices:
    try:
        problems.append(library.get_problem(index))
    except UnsupportedSolution as error:
        gaps[index] = error.reasons
```

Invalid arguments, malformed metadata and unexpected exceptions are errors to
investigate; do not catch all exceptions and turn them into coverage gaps.
Missing requested indices and missing record fields raise `KeyError`.
Unknown datatype names and invalid tile/reduction sizes raise `ValueError`.
The formula does not solve arbitrary solution predicates. Native support checks
still decide whether a solution accepts the descriptor; this initial workload
does not cover reduction tails, output edges or varied scalar/epilogue settings.

Datatype descriptions are preserved independently of bench's supported inputs,
across GPU targets. The `gemm_problem_bench` adapter translates datatype names
into the strings accepted by hipblaslt-bench. It raises `UnsupportedBenchProblem`
when it cannot express a descriptor, distinct from generator exclusions. Neither
metadata generation nor successful formatting proves native kernel support.

Indices identify solutions only within a particular library build. Loading reads
every matching `.dat` or `.dat.zlib` shard once, supporting legacy and indexed
MsgPack formats through the shared metadata reader. These binary files contain
kernel descriptions and the tables used to select them; `.zlib` marks a compressed
copy. The legacy format stores records in an array, while the indexed format adds
a lookup table for individual records. Compressed copies take
precedence. Missing requested indices and duplicate indices are errors.
Subsequent `get_problem` calls reuse the retained records without I/O, generator
imports or GPU calls. Selective loading limits retained records, while compressed
shards still require full decompression.

`Tests/common/test_gemm_problem_for_solution.py` builds eight small gfx942 libraries for FP32,
FP64, FP16 and BF16 with split factors 1 and 4. It verifies dimensions, numerical
results and solution/kernel identity through `hipblaslt-bench`. With rocJITsu's
logging plugin, it also checks that the target kernel dispatched exactly once.
The test accepts `--hipblaslt-bench` and supports the common test suite's
`--build-only` / `--use-cache` artifact flow.
