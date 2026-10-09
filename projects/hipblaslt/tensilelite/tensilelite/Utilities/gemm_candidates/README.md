# GEMM candidates for a solution index

`library.get_problems(index, count)` returns a deterministic `list[GemmCase]` to
try with one packaged TensileLite solution. It uses the solution's dimensional
constraints and tile sizes to target different reduction lengths and matrix
boundaries. It needs neither a GPU nor the kernel generator.

```python
from pathlib import Path
from tensilelite.Utilities.gemm_candidates import (
    bench_yaml,
    load_library_metadata,
    tensile_ini,
)

index = 17  # Substitute a solution index from your library.
library = load_library_metadata(
    "/path/to/hipblaslt/library", target="gfx942", solution_indices=[index]
)
cases = library.get_problems(index, 10)

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
  `Half`, `BFloat16` and `Float`. MX profiles include each operand's scale
  block/type and the solution's scale layout.
- Selected `alpha`, `beta` and `activation` values. The initial policy chooses
  1, 0 and `None`; both adapters read these from the case. Future policies can
  vary runtime settings without changing the list-of-cases interface.
- Solution index, solution name, kernel name, target, metadata file and code objects.
- An `id` and a readable `reason` explaining why the shape was selected.

`leading_dimensions` and `batch_strides` give packed column-major layouts in
A/B/C/D order. `shape.tensile_sizes` gives the Tensile client's **M,N,batch,K**
order. The adapters handle that distinction.

The return type is independent of the runner. `bench_yaml(cases)` and
`tensile_ini(cases)` translate the same logical cases into the two clients'
input formats. They use existing client data initialization and MX packing;
the Python utility does not allocate GPU buffers or implement numerical GEMM.

Indices identify records in a particular library build. Distinct indices with
the same solution or kernel name remain distinct entries. Do not reuse an index
against another build without checking its identity.

## Candidate policy

The generator first extracts necessary dimensional bounds from conjunctive
predicates: minimum/maximum sizes, equalities, divisibility and fixed split-K
minimum K. It intersects these bounds and rounds up to legal multiples.
For example, `M >= 4` allows `M = 5`; it does not imply `M % 4 == 0`.
For sub-byte inputs, stored columns and batch starts are aligned to whole
native packed containers: 2 elements for FP4 and 32 for FP6/BF6.

It then tries the following proposals in order, keeping distinct shapes that
fit the limits:

| Candidate | Selection intent |
| --- | --- |
| `minimum` | Smallest positive shape satisfying the extracted dimensional bounds and packed-layout requirements. |
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

Larger requests pair more reduction boundaries with full/partial output tiles
and batches. They also vary M and N separately near 1, 2, 3, 4, 8, ... times
the first legal full-tile size. After those boundary cases, the function
enumerates additional shapes in increasing GEMM volume until it fills the
request. A frontier of neighboring shapes avoids constructing a dense
Cartesian product.

The default limits are dimensions at most 8192 and at most 128 MiB of combined
A/B/C/D storage per case, accounting for packed FP4/FP6 storage. Workspace,
bias and scalar/MX scale buffers are separate. Pass `max_dimension` and
`max_data_bytes` to change these limits. Requests are deterministic prefixes:
asking for 10 returns the first 10 of the 100-case result with the same limits.

A shorter list means every distinct shape in this **bounded dimensional
domain** has been returned. It does not establish exhaustive kernel-path
coverage, or exhaust all GEMMs beyond the configured limits, layouts and
runtime settings. An empty feasible set raises `ValueError`. The equivalent
free-function form is `candidate_gemms(library, index, max_cases=count)`.

## Reuse and batching

Load metadata once, then call the function for each index:

```python
indices = [17, 42]  # Indices from the selected library.
library = load_library_metadata(
    "/path/to/hipblaslt/library", target="gfx942", solution_indices=indices
)
by_solution = {i: library.get_problems(i, 10) for i in indices}
```

Loading supports both ordinary MessagePack (`solutions` arrays) and indexed
MessagePack (`format_version=2`, `solutions_index` and `solutions_blob`), in
`.dat` or `.dat.zlib` files. The optional `solution_indices` filter reduces
retained memory. For indexed files, it also avoids decoding unrequested
solution records. Compressed shards still need to be decompressed in full;
ordinary arrays require decoding each record to find its index.

Loading scans the matching shards once, skips their heuristic tables without
constructing Python objects, and shares repeated problem/predicate records.
Omit the filter to load every index into `library.indices`. Compressed copies
take precedence. The path may name an architecture directory or its parent.
Missing requested indices raise `KeyError`; malformed containers, unsupported
format versions and duplicate indices raise `ValueError`. Duplicate indices
are checked even outside the requested selection, including across shards.

Candidate generation performs no further file reads or native calls. Its
work grows with the requested count and is independent of call order.
A separate bulk-generation API is unnecessary.

## Implementation boundaries

The package shares binary I/O with the generator through
[`Utilities/LibraryIO.py`](../LibraryIO.py), and Tensile client problem options
through [`Utilities/ClientConfig.py`](../ClientConfig.py). The existing
`LibraryIO` writers and `ClientWriter` call those same helpers. Neither helper
imports generator classes or rocisa.

Within this package, `metadata.py` discovers library files and builds the
reusable lookup; `models.py` holds immutable cases, profile normalization and
packed layouts; `candidates.py` owns the shape-selection policy; and
`clients.py` formats inputs. The small table of supported storage types and
bench spellings lives with the models and is checked against the generator's
`DataType` registry in tests. This keeps the installed utility independent of
the generator while sharing serialization rules with its existing clients.

## Run with either client

```sh
HIPBLASLT_TENSILE_LIBPATH=/path/to/hipblaslt/library/gfx942 \
    hipblaslt-bench --yaml cases.yaml
tensilelite-client --config-file cases.ini
```

Use clients and code objects compatible with the metadata and GPU. Both
adapters enable numerical validation and apply each case's runtime settings.
The initial policy uses alpha=1/beta=0, no activation, and supported
per-solution bias/scaling defaults. It selects one bias type/source rather
than sweeping epilogues. Tensile uses the fixed input
initialization seed `1`; its seed `0` would select the current time.

MX cases support FP8/BF8, packed FP4/FP6/BF6 inputs, and mixed MX/ordinary
operands. Scale blocks of 16 or 32 use E8M0, E4M3 or E5M3 scale types from the
metadata. Optional scalar/vector A/B scales are disabled for MX so both clients
select the same block-scaling operation. Native predicates still determine
which type combinations a particular solution accepts.

The adapters distinguish canonical scales, gfx950 host-preswizzled scales,
and gfx1250 swizzled scales. Bench's host-preswizzled mode supports gfx950
block-32 E8M0 only. Bench always swizzles MX scales on gfx1250, so its formatter
rejects a gfx1250 `NoSwizzle` solution and directs callers to `tensile_ini`.
Unsupported client representations raise an error rather than emit a different
scale layout. Native MX execution requires a compatible GPU; host-side
generation and formatting do not establish numerical or race correctness.

`bench_yaml` can combine rows for several solutions for one target in the same
installation; each row carries its explicit index. `tensile_ini` accepts cases for **one
solution and runtime profile** and rejects mixed inputs, because the client
otherwise applies all shapes to all selected solutions. Both accept
`workspace_bytes` to override the 128 MiB workspace limit. The formatters return
strings and do not execute clients.

## Testing MX candidates

[`test_gemm_candidates_mx.py`](../../Tests/common/test_gemm_candidates_mx.py)
builds one small gfx950 MXFP4 solution and one MXFP8 solution, requests ten
candidates for each, and runs them through the Tensile client. It checks every
shape's solution index/name and numerical result against the CPU reference.
The test needs a prebuilt client and a compiler that supports gfx950; it builds
its own device libraries.

From the TensileLite project directory, run on gfx950 hardware with:

```sh
python -m pytest -q -n 0 tensilelite/Tests/common/test_gemm_candidates_mx.py \
    --gpu-targets gfx950 --prebuilt-client /path/to/tensilelite-client \
    --tensile-options=--cxx-compiler,/path/to/amdclang++ \
    --basetemp /path/to/test-work
```

For CPU emulation, prefix that command with
`rocjitsu --config /path/to/gfx950-with-race.json --`. Start from rocJITsu's
gfx950 configuration and enable the `race` plugin and a `stderr` sink: the
test also fails on any race report in the client output. The numerical checks
run in both modes.

For separate CI build and test stages, add
`--build-only --artifact-dir /path/to/artifacts` on the build machine, then
`--use-cache --artifact-dir /path/to/artifacts` on the execution machine.
The build stage needs no GPU or emulator. The execution stage loads the
packaged metadata and code objects and regenerates the candidate inputs.

The test belongs to the gfx950 `common` suite and is marked `emu_fast`.
[CI scheduling](../../../TESTING.md#pre-submit--ci-gates) determines where
those suites run. The current GitHub rocJITsu job targets gfx942; it needs
separate gfx950 job wiring to execute this MX test on pull requests.

## Limits of the result

A candidate is not a proven supported minimum or a measurement of code-path
coverage. Coupled, disjunctive and hardware predicates remain native checks.
Native clients may reject a candidate; callers must inspect actual execution,
solution identity and numerical results rather than treating generation or a
zero exit code as proof of support.

This implementation covers positive strided GEMMs, including MX, with
FP32/FP16/BF16/FP8/FP6/FP4/INT8/INT32 storage and FP32 or INT32 computation.
Packed FP4/FP6 are input types. It supports the standard M-axis
bias/alpha-vector profile and scalar/vector A/B scales.
Grouped, sparse, gradient, auxiliary-output, swizzled A/B and other unsupported
profiles raise `ValueError`. Null dimensions, arbitrary layouts, heuristic
default selection, exhaustive epilogues and runtime coverage measurement are
outside this function's scope.
