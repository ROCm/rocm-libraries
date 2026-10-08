# GEMM candidates for a solution index

`candidate_gemms` constructs a small, deterministic list of shapes and types to
try with one packaged TensileLite solution. It uses the solution's dimensional
constraints and tile sizes to target different reduction lengths and matrix
boundaries. It needs neither a GPU nor the kernel generator.

```python
from pathlib import Path
from tensilelite.Utilities.gemm_candidates import (
    bench_yaml,
    candidate_gemms,
    load_library_metadata,
    tensile_ini,
)

index = 17  # Substitute a solution index from your library.
library = load_library_metadata(
    "/path/to/hipblaslt/library", target="gfx942", solution_indices=[index]
)
cases = candidate_gemms(library, index, max_cases=10)

for case in cases:
    print(case.id, case.shape, case.profile.a_type, case.reason)

Path("cases.yaml").write_text(bench_yaml(cases))
Path("cases.ini").write_text(tensile_ini(cases, results_file="results.csv"))
```

Use Python 3.10 or later with `msgpack` and `PyYAML`. From a source checkout,
put `projects/hipblaslt/tensilelite` on `PYTHONPATH`. An installation with
TensileLite test artifacts provides the package under
`share/hipblaslt/tensilelite`.

## What a case contains

Each immutable `GemmCase` contains:

- A `GemmShape` with named `m`, `n`, `k` and `batch_count` fields.
- A `GemmProfile` with A/B/C/D types, computation types, transposes and the
  selected bias/scaling settings. Type strings use Tensile names, such as
  `Half`, `BFloat16` and `Float`.
- Solution index, solution name, kernel name, metadata file and code objects.
- An `id` and a readable `reason` explaining why the shape was selected.

`leading_dimensions` and `batch_strides` give packed column-major layouts in
A/B/C/D order. `shape.tensile_sizes` gives the Tensile client's **M,N,batch,K**
order. The adapters handle that distinction.

Indices identify records in a particular library build. Distinct indices with
the same solution or kernel name remain distinct entries. Do not reuse an index
against another build without checking its identity.

## Candidate policy

The generator first extracts necessary dimensional bounds from conjunctive
predicates: minimum/maximum sizes, equalities, divisibility and fixed split-K
minimum K. It intersects these bounds and rounds up to legal multiples.
For example, `M >= 4` allows `M = 5`; it does not imply `M % 4 == 0`.

It then tries the following proposals in order, keeping distinct shapes that
fit the limits:

| Candidate | Selection intent |
| --- | --- |
| `minimum` | Smallest positive shape satisfying the extracted dimensional bounds. |
| `k_1` through `k_3` | First three legal full reduction lengths starting at the minimum. The reduction scale is `DepthU × GlobalSplitU` for fixed split-K, and `DepthU` otherwise. |
| `k_long` | Eight times the first full reduction length, falling back to four times if necessary. |
| `k_tail` | A legal K with a nonzero remainder modulo `DepthU`, near two reduction scales or the minimum. |
| `split_remainder` | For fixed split-K, full `DepthU` blocks that divide unevenly among split groups. |
| `tile` | M and N aligned to full macro tiles, the output blocks computed by workgroups. |
| `mn_edge` | Partial output tiles, combined with a K tail when permitted. |
| `multi_tile` | At least two macro tiles in M and three in N. |
| `batch` | Multiple strided batches. |

`DepthU` is the kernel's reduction block length. Fixed split-K divides that
reduction among a configured number of workgroups. Automatic split-K and
Stream-K decide the work distribution at runtime; their candidates use DepthU
as a size reference, without predicting per-workgroup loop counts.

The defaults are at most 10 cases, dimensions at most 8192, and at most 128 MiB
of combined A/B/C/D storage per case. Workspace and bias/scaling buffers are
separate. `max_cases`, `max_dimension` and `max_data_bytes` are keyword arguments.
Reducing `max_cases` takes a prefix of the same ordered list. Restrictions and
deduplication can produce fewer cases; raising the cap does not enumerate the
GEMM space. An empty feasible set raises `ValueError`.

## Reuse and batching

Load metadata once, then call the function for each index:

```python
indices = [17, 42]  # Indices from the selected library.
library = load_library_metadata(
    "/path/to/hipblaslt/library", target="gfx942", solution_indices=indices
)
by_solution = {i: candidate_gemms(library, i) for i in indices}
```

The optional `solution_indices` filter reduces retained memory. Loading scans
the matching `.dat`/`.dat.zlib` shards once, skips their heuristic tables without
constructing Python objects, and shares repeated problem/predicate records.
Omit the filter to load every index into `library.indices`. Compressed
copies take precedence. The path may name an architecture directory or its
parent. Missing requested indices raise `KeyError`; duplicate indices across
shards raise `ValueError`.

Candidate generation performs no further file reads or native calls. It has a
bounded number of proposals per solution and is independent of call order.
A separate bulk-generation API is unnecessary.

## Run with either client

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/hipblaslt/library/gfx942 \
    hipblaslt-bench --yaml cases.yaml
tensilelite-client --config-file cases.ini
```

Use clients and code objects compatible with the metadata and GPU. Both
adapters enable numerical validation, set alpha=1/beta=0, disable activation,
and retain the supported per-solution bias/scaling defaults. They select one
bias type/source rather than sweeping epilogues. Tensile uses the fixed input
initialization seed `1`; its seed `0` would select the current time.

`bench_yaml` can combine rows for several solutions from the same installation;
each row carries its explicit index. `tensile_ini` accepts cases for **one
solution and profile** and rejects mixed inputs, because the client otherwise
applies all shapes to all selected solutions. Both accept `workspace_bytes` to
override the 128 MiB workspace limit. The formatters return strings and do not
execute clients.

## Limits of the result

A candidate is not a proven supported minimum or a measurement of code-path
coverage. Coupled, disjunctive and hardware predicates remain native checks.
Native clients may reject a candidate; callers must inspect actual execution,
solution identity and numerical results rather than treating generation or a
zero exit code as proof of support.

This initial implementation covers ordinary positive strided GEMMs with
FP32/FP16/BF16/FP8/INT8/INT32 storage and FP32 or INT32 computation. It supports
the standard M-axis bias/alpha-vector profile and scalar/vector A/B scales.
Grouped, sparse, MX, gradient, auxiliary-output, swizzled and other unsupported
profiles raise `ValueError`. Null dimensions, arbitrary layouts, heuristic
default selection, exhaustive epilogues and runtime coverage measurement are
outside this function's scope.
