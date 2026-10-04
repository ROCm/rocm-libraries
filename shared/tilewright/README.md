# tilewright

tilewright is a small CPU library that ranks candidate GEMM kernels for a
problem with a trained model. A caller hands it a GEMM problem, the device's
properties and its pool of candidate kernels; tilewright filters the pool,
scores the survivors with the model trained for the problem's region of the
problem space, and returns the pool best-first.

hipBLASLt's TensileLite uses it, when `TENSILE_USE_TILEWRIGHT` is set, to order
the kernels of a library's Prediction rows, falling back to its analytical
(Origami) ranking when tilewright has no model for the library or scores none
of its kernels. tilewright itself
depends on no GEMM framework: it does not include Tensile, TensileLite or
Origami headers, and callers convert their own types into tilewright's.

The models are produced by the training pipeline in
[`training_pipeline/`](training_pipeline/README.md).

## Layout

| Path | Contents |
|------|----------|
| `include/tilewright/` | Public API: `types.hpp` (problem, kernel and device types), `model.hpp` (loading, routing, ranking) |
| `src/tilewright/` | The engine: `format.cpp` (MLREC_v2 loader, CRC-32, lazy dequantization), `features.cpp` (feature catalog, LDS gate, feasibility rules), `kernels.cpp` (fp32 inner-product kernels with run-time ISA selection), `rank.cpp` (routing, filtering, scoring, `CandidateSet`), `registry.cpp` (file loading, path deduplication, `tilewright_index`, environment knobs) |
| `python/` | nanobind bindings (`import tilewright`) and their pytest suite |
| `tests/` | Catch2 suite; `model_writer.*` writes synthetic models so the suite needs no shipped weights |
| `weights/hipblaslt/<arch>/<arch>/` | Shipped models (`*.tilewright.bin`) and their `tilewright_index` |
| `cmake/` | Package configuration template for `find_package(tilewright)` |
| `training_pipeline/` | Training pipeline that produces the models |

## How a problem is ranked

**Routing.** A problem maps to one of 96 base cells
`<M tier>|<N tier>|<K tier>|<batch tier>`: M and N are `Tiny` (≤ 32), `Small`
(≤ 128), `Mid` (≤ 512) or `Large`; K is `TinyK` (≤ 32), `MidK` (≤ 512) or
`LargeK`; batch is `Bnone` (batch == 1) or `Bany`. A model's split tree refines
a base cell along one axis at a time (`M`, `N`, `K` or `B`; values at or below
the threshold go to the `lo` child), giving labels such as
`Large|Large|LargeK|Bnone#N<=10092`. The problem is scored by the leaf's cell,
or by its nearest trained ancestor (the label with trailing `#...` segments
removed) when the leaf has no trained cell. With no trained cell at all, every
kernel is returned unscored.

**Filtering.** Each kernel of the pool then passes, in order:

1. an LDS-capacity gate: `mt_m*mt_k*bytes(A) + mt_n*mt_k*bytes(B)` must fit in
   the device's LDS;
2. a feasibility filter: a small batched problem (M, N ≤ 256, K < 1024,
   batch ≠ 1) must fit in one macro tile; Dot2 kernels (MI 1x1x64) only serve
   M < 3; and hinted kernels (`cache_hints != 0`) are only used when K and the
   kernel's `mt_k` are both 128-byte aligned in A's element size, where a skinny
   problem requires the non-temporal variant of its large operand
   (`cache_hints == 4`) when the pool contains such a variant for that operand,
   while every other problem rejects hinted kernels;
3. the cell's whitelist of kernel signatures
   `(mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, cache_hints_a, cache_hints_b)`.

The feasible whitelisted kernels form tier 1. If the whitelist rejects every
feasible kernel, all feasible kernels form tier 1. When the caller asks for
more results than tier 1 holds (`min_scored`), the other feasible kernels are
scored as tier 2.

**Scoring.** Each cell is a two-tower model. Features are computed in double
precision and rounded once to float, as in the training pipeline, and are
whitened with the cell's means and standard deviations (a standard deviation
below 1e-6 counts as 1):

- query features (55) describe the problem and the device;
- item features (12) describe the kernel;
- interaction features (37) describe the problem and kernel together, including
  the matrix-instruction latency, which uses the MI-latency table, the
  `parallel_mi_cu` and bandwidth constants stored in the model.

```
score = dot(query_mlp(q), item_mlp(i)) / max(|temperature|, 0.1) + interaction_mlp(x)
```

Higher is better. The feature catalog is identified by
`feature_catalog_hash()`; models trained on another catalog are rejected.

**Result.** `rank` returns one `Result {config_index, score, scored}` per
input kernel (`config_index` is the kernel's position in the pool): tier 1
best-first, then tier 2 best-first, then the unscored kernels in input order.
Equal scores keep input order. A kernel whose score is not finite is unscored,
and invalid hardware (`N_CU`, `lds_capacity` or `L2_capacity` equal to 0)
leaves every kernel unscored.

**Numerics.** Scoring is fp32. Every inner product uses one fixed summation
order: 16 interleaved lanes accumulated with fused multiply-adds, a fixed
pairwise reduction of the lanes, then the remaining elements added in order.
The scalar, AVX2 and AVX-512 kernels (chosen at run time from the CPU's
features) implement that order exactly, and the library is built without
fast-math or floating-point contraction, so rankings do not depend on the
instruction set the CPU offers or on the compiler (GCC or Clang). Features are
computed in double with the C library's `log2` and rounded once to float. Weights stay
packed in memory and a cell's weights are converted to fp32 on the first
ranking that reaches the cell.

## Model files (MLREC_v2)

A model file is little-endian:

| Field | Type | Notes |
|-------|------|-------|
| magic | `char[8]` | `MLREC_v2` |
| endian marker | `u32` | `0x01020304` |
| header_size | `u32` | bytes from offset 0 to the first payload byte |
| payload_size | `u64` | payload bytes (the trailer excluded) |
| payload_crc32 | `u32` | CRC-32/ISO-HDLC of the payload |
| weight_dtype | `u8` | 0 fp32, 1 bf16, 2 int8, 3 int4 |
| reserved | `u8[3]` | 0 |
| q_dim, i_dim, x_dim | `u32` each | 55, 12, 37 |
| n_cells, n_splits | `u32` each | |
| feature_catalog_hash | `u16` length + ASCII | must equal `feature_catalog_hash()` |
| arch | `u16` length + ASCII | for example `gfx950`, `gfx1250v0` |
| parallel_mi_cu | `f64` | > 0 |
| bw_c0, bw_c1, bw_c2 | `f64` each | `bw_occ = min(1, c0*a*a + c1*a + c2)`, `a = min(total tiles, N_CU)` |
| mi_default_cycles | `f64` | cycles for an MI shape missing from the table |
| n_mi | `u32` | at most 1024 |
| MI table | `n_mi` x {`u32` m, n, k; `i32` dtype; `f64` cycles} | dtype is a `tilewright::DataType` value |

The payload follows: `n_splits` split records
(`u16`+parent label, axis char and NUL, `i32` threshold, `u16`+lo label,
`u16`+hi label), then `n_cells` cell records (`u16`+label, `u32` embed, hidden
and interaction widths, `f32` temperature, the six whitening vectors, `u32`
signature count and `i32[8]` signatures, then the weight matrices and biases of
the three towers). Matrices use `weight_dtype` (bf16: `bits << 16`; int8: `f32`
scale then `i8` values; int4: `f32` scale then two values per byte, low nibble
first, value = scale * (nibble - 8)); biases, whitening vectors and
temperatures are always fp32. The file ends with `MLRECEND`, and its size is
exactly `header_size + payload_size + 8`.

The MI table is looked up by `(mi_m, mi_n, mi_k, dtype)` with the fnuz FP8
types mapped to their non-fnuz counterparts; the latency feature is
`cycles / max(parallel_mi_cu, 1)`.

The loader validates everything before using it: sizes and counts against the
bytes that remain, the CRC, the feature dims and hash, finite floating-point
values, split axes, labels (non-empty printable ASCII without spaces), duplicate cell labels
and split parents, an acyclic split tree, the MI table, and the trailer. MLREC_v1
files are rejected; convert them with
`training_pipeline/scripts/convert_mlrec_v1_to_v2.py`. The training pipeline
writes MLREC_v2 with `training_pipeline/lib/mlrec.py`.

## C++ API

```cpp
#include "tilewright/model.hpp"

std::string error;
tilewright::ModelPtr model = tilewright::load_model(path, &error);  // nullptr + error on failure
if (!model) { /* report error */ }

std::vector<tilewright::Config> pool = ...;               // the caller's kernels
tilewright::CandidateSet candidates(model, std::move(pool));  // reuse for every problem

tilewright::Hardware hw{n_cu, lds_bytes, l2_bytes};       // queried from the device
std::vector<tilewright::Result> ranked = candidates.rank(problem, hw, /*min_scored=*/0);
```

- `load_model(path)`, `load_model_from_memory(data, size)` and
  `load_model_by_index(logic_stem, dir)` never throw; they return `nullptr` and
  set the error message on failure. `load_model` returns the already loaded
  model for a path that resolves to the same file. `load_model_by_index` reads
  `<dir>/tilewright_index` and returns `nullptr` with an empty error when the
  index or the stem is absent.
- `CandidateSet` holds a fixed pool. The item embeddings and whitelist
  membership of its kernels are computed the first time a problem reaches a
  cell and reused afterwards; it is safe to share between threads.
  `rank_configs(model, problem, hardware, configs, min_scored)` returns the
  same result for an ad-hoc list without caching.
- `route(model, problem)` and `cell_label(model, cell)` expose routing;
  `describe(model)` returns the arch, hash, weight type and sizes;
  `compute_features(model, problem, config, hardware)` returns the raw
  (unwhitened) feature vectors.

`DataType` values are part of the model format (MI-table keys). They follow
Origami's enumeration, which has an `Int4` entry that Tensile's does not, so
convert Tensile data types by name rather than by value.

## Python

```bash
pip install shared/tilewright/python
```

```python
import tilewright as tw

model = tw.load_model("weights/hipblaslt/gfx950/gfx950/<name>.tilewright.bin")
pool = tw.CandidateSet(model, [tw.Config(mt=tw.Dim3(256, 256, 64), mi=tw.Dim3(16, 16, 32), occupancy=1)])
problem = tw.Problem(size=tw.Dim3(4096, 4096, 4096), a_transpose=tw.Transpose.T,
                     a_dtype=tw.DataType.BFloat16, b_dtype=tw.DataType.BFloat16,
                     c_dtype=tw.DataType.BFloat16, d_dtype=tw.DataType.BFloat16,
                     mi_dtype=tw.DataType.BFloat16)
ranked = pool.rank(problem, tw.Hardware(N_CU=64, lds_capacity=65536, L2_capacity=4 << 20))
```

The module mirrors the C++ API. Enumerations are integer enums (`DataType.None`
is spelled `DataType.None_`), structures accept keyword arguments, and the GIL
is released while models load and pools rank. `load_model` and
`load_model_from_memory` raise `ValueError` with the loader's message when the
data is not a valid model; `load_model_by_index` returns `None` when the index
or stem is absent.

## Environment variables

Read once per process. For `TILEWRIGHT_DIAG` and `TILEWRIGHT_PICK_LOG`, a value
of `0` or an empty value counts as unset.

| Variable | Effect |
|----------|--------|
| `TILEWRIGHT_DIAG` | One stderr line per loaded model: `[TILEWRIGHT_DIAG FILE] path=... arch=... qhash=... qdim=55 idim=12 xdim=37 n_cells=N n_splits=N weights=int4`, and `[TILEWRIGHT_DIAG FAIL] <reason>` for a load that fails |
| `TILEWRIGHT_PICK_LOG` | One stderr line per ranking that scores a kernel: `[TILEWRIGHT_PICK] m=.. n=.. k=.. b=.. tA=.. tB=.. leaf=<label> top1_sig=(mt_m=..,mt_n=..,mt_k=..,mi_m=..,mi_n=..,mi_k=..,cha=..,chb=..) top1_score=.. n_configs=..` |
| `TILEWRIGHT_FORCE_CELL` | Score every problem with this cell index, for debugging; ignored when it is not a valid index of the model |

tilewright writes nothing else to stderr. Whether hipBLASLt uses tilewright at
all is controlled by hipBLASLt (`TENSILE_USE_TILEWRIGHT`).

## Weights

Shipped models live in `weights/hipblaslt/<arch>/<arch>/`, mirroring the
Tensile library tree: one `.tilewright.bin` per trained library, named after the
logic file it was trained on (so a model shared by several trees keeps the name
of the tree it was trained for), plus a `tilewright_index` that maps Tensile
library stems to model files:

```
# comment
<TensileLibrary stem>	<model file>
```

The model files are ordinary binary files in the repository (marked `binary` in
`.gitattributes`), so any checkout that includes `shared/tilewright` has them.
hipBLASLt copies each tree's index and models next to its Tensile library
files; TensileLite finds them only for libraries loaded from per-logic files,
which is hipBLASLt's default layout (`HIPBLASLT_ENABLE_LAZY_LOAD=ON`).

## Building and testing

tilewright needs CMake 3.25.2 and a C++17 compiler. Standalone:

```bash
cd shared/tilewright
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

| Option | Default | Effect |
|--------|---------|--------|
| `TILEWRIGHT_BUILD_TESTING` | ON standalone, OFF as a subproject | Build the Catch2 suite (and the pytest suite with the bindings) |
| `TILEWRIGHT_ENABLE_PYTHON` | OFF | Build the Python module |
| `TILEWRIGHT_ENABLE_INSTALL` | ON standalone, OFF as a subproject | Install the library, headers and CMake package |
| `TILEWRIGHT_ENABLE_FETCH` | ON | Fetch Catch2 (and nanobind for CMake builds of the bindings) when they are not installed |

The tests use `find_package(Catch2 3)` and fetch Catch2 only when it is missing
and fetching is enabled. They write their own synthetic models; the real-model
tests additionally load every shipped model. The environment-variable tests run
as separate ctest entries because the knobs are read once per process.

As a subproject, `add_subdirectory(shared/tilewright)` provides the static
library `roc::tilewright`. An installed tilewright is found with
`find_package(tilewright)`, which provides the same target.
