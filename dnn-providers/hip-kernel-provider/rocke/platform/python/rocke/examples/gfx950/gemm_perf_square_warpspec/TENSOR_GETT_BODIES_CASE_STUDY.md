# Tensor contraction (GETT) on the dispatcher's GEMM bodies, gfx950 (case study)

Measured scope: f16 and bf16 tensor contractions `D = alpha * A B (+ beta * C)` (f32 accumulation, one output
rounding) over a hipTensor contraction workload set: the patterns P1 (both operands and D
permuted), P2 (D in GEMM order), P3 (operands in GEMM order, D interleaved), P4 / P5 (one operand permuted) and the
canonical (column-major) layout, at ranks 2-6 and small / medium / very large sizes, each in scale and bilinear
form, on gfx950. Compared arms, in one process, interleaved in seeded random order per round (5-7 rounds, median;
a 1 GiB read flush before every timed call): the product route `plan_tensor_gett` (the dispatcher's choice), the
earlier product route (the fixed ring-body GETT), and the vendor composition (operand / output permutes by the
framework's copy, then the vendor BLAS GEMM chosen by the framework's tunable-op search over every hipBLASLt and
rocBLAS solution, with alpha / beta in the GEMM call). Every arm of every cell passed an on-device check against
an f64 reference before and after its timing (full output up to 4e10 multiply-adds and for every output larger
than 2 GiB; a 1024-row sample otherwise). Relative results only; ratios within about 4 % of each other are
noise in these jobs (several jobs shared a node).

## What changed

| piece | what it does |
|---|---|
| MN-layout bodies (`TensorGettSpec.mn_tile` / `mn_trait`) | the contraction runs on the GEMM dispatcher's both-MN-contiguous (`CRR`) pick for the same M x N x K (or the 256x256x32 AGPR interleave ring): `gemm_universal`'s MN operand path owns the LDS stages and the `ds_read_b64_tr_b16` fragments; the GETT hook replaces only the per-lane DTL offsets (mixed-radix mode offsets over the body's own lane geometry) and the epilogue |
| `GettKnobs.epi_v16` | A's transpose reads in `QUAD_PERM` order make lanes L and L + 32 hold 8 consecutive output rows; one `v_permlane32_swap` per dword then gives every lane 8 contiguous D elements: 16-byte stores (and C loads) on permuted offsets |
| K-contiguous bodies (`kc_tile` / `kc_trait`) | the contraction runs on the dispatcher's `RCR` pick (the row-interleave / hb_sched / serpentine bodies with the direct-to-LDS ring) reading K-contiguous packed operand copies; the GETT epilogue rides the row-interleave vector epilogue's hook (8 columns per lane, 16-byte stores along D's contiguous mode) |
| `tensor/pack.py` | the operand pack: a 4-byte-gather form that keeps the source's contiguous mode innermost and a 64x64 LDS-tiled transposing form (4-byte coalesced loads along the source's runs, 16-byte stores) |
| `plan_tensor_gett` | single launch, pack hybrid (MN body on packed copies of operands whose runs are not 16-byte loads) or the K-contiguous route, chosen by the GEMM dispatcher's time model of each route's body plus a pack cost model; persistent tile walk on big grids; C loads two rows ahead for bilinear |
| K padding | K modes that do not split into the K tile pad one in-tile mode (padded rows read zeros); contractions with such K extents were refused before |
| wide D | outputs larger than 2 GiB mask the summed 32-bit offset on every route (no row / column sentinel inside D) |

## Results (relative, product route vs the vendor composition in the same job)

Geomean speedup of the product route over the vendor composition measured in the same job (> 1: rocKE faster),
208 cells (52 rows x f16 / bf16 x scale / bilinear):

| pattern | cells | f16 | bf16 |
|---|---|---|---|
| P1 (operands and D permuted) | 68 | 2.17x | 2.26x |
| P2 (D in GEMM order) | 20 | 1.42x | 1.42x |
| P3 (D interleaved) | 20 | 2.31x | 2.14x |
| P4 (one operand permuted, reversed K) | 20 | 1.40x | 1.32x |
| P5 (one operand permuted) | 20 | 1.35x | 1.28x |
| canonical (no permute) | 60 | 1.30x | 1.13x |

* Every P1-P5 cell with a same-job vendor number is at least 1.10x faster than it. For the bf16 cells at K = 16384
  the vendor GEMM alone (plus a contiguous copy of the permute bytes) is within 1-8 % of the product route; some
  canonical small shapes (a few dozen output tiles; ranks 4-6) are at 0.90-0.99x of the vendor in scale form and
  the bf16 canonical medium / large shapes at 0.93-1.05x in scale form: there the GEMM body itself, not the
  contraction machinery, sets the time.
* Against the previous product route (the fixed ring-body GETT) in the same jobs the product route is 1.3-2.0x
  faster geomean on P1, P2, P4, P5 and the canonical rows and 1.08x on P3 (the ring body already read P3's
  operands with 16-byte loads); on the rows whose operand runs are not multiples of 8 (P2 / P5 at ranks 5-6) the
  previous route was 2-3x slower than now.
* Two problem classes the previous route could not serve correctly or at all are served: a ragged output larger
  than 2 GiB (every output now matches the f64 reference), and K extents that do not split into the K tile.
* Route mix chosen by the dispatcher on these rows: the MN-layout body for most permuted-output and long-K rows,
  the K-contiguous route (RCR body on packed copies) for output-in-GEMM-order rows with short K and for the very
  large rows, the pack hybrid for permuted-output rows with short operand runs.

## Notes and caveats

* The vendor composition uses the framework's permute copy; a contiguous copy of the same bytes (a lower bound
  for any permute kernel) added to the vendor GEMM alone gives a stricter reference; the product route also beats
  that reference on most rows (geomean 1.02-1.54x per pattern and dtype); the exceptions are canonical small
  shapes and bilinear rows with the output in GEMM order (down to about 0.8-0.95x), whose C read is not
  overlapped with the main loop.
* The both-MN-contiguous row-interleave / hb_sched body (no pack) spills registers in every scheduler strategy and
  stays refused; those bodies are reached through K-contiguous packed copies.
* The K-contiguous route needs D's contiguous run on the column side to be at least 128 elements (or K >= 4096):
  shorter runs scatter the vector epilogue's stores.
