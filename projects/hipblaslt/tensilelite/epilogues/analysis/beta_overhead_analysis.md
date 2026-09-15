# How MatrixInstruction shape drives beta*C overhead on gfx950

**Problem:** GEMM `8192 x 8192 x 8192`, bf16 in / bf16 out, HPA, TN, batched, gfx950
wave64. `epilogues/YAMLs/benchmark_beta.yaml` has two `BenchmarkProblems` groups with
identical fork space, differing only by `UseBeta` (group 1 `False` -> `BBS_H`,
group 2 `True` -> `BBS_BH`). `DataInitTypeBeta: 1`, so the beta group runs the
`Beta != 0` (`label_GW_B1`) path. Question: **which `MatrixInstruction` shapes make the
beta*C path cheap?**

## Methodology (corrected)

- The YAML was run **unmodified** — no `NumWarmups` / `EnqueuesPerSync` /
  `SyncsPerBenchmark` knobs. Those perturb the measured kernel time and inflate the
  numbers, so they are not used.
- `NumBenchmarks=1` (the shipped setting) gives one data row per solution, which is
  noisy single-shot. To damp that without changing per-kernel methodology, the whole
  benchmark was **repeated 3 times** (`/tmp/beta-sweep2`, `/tmp/beta-sweep3`,
  `/tmp/beta-sweep4`) and we take the **median delta per config across runs**.
- Times come only from the CSV: `time_us = TotalFlops / GFlops / 1000`
  (col 9 `TotalFlops` = 1099511627776, cols 11+ = per-solution GFlops). The
  LibraryLogic "winner" number is a **model prediction and is never used**.
- `delta_us = time(with_beta) - time(without_beta)`. 204 / 204 configs paired, 0
  unmatched. Median single-shot spread across the 3 runs is ~15 us, so per-config
  deltas below ~15 us are within noise; the group-level trends below (n = 36-116) are
  robust.

Parser: `epilogues/analysis/parse_beta_sweep.py` -> `epilogues/analysis/parse_beta_sweep.out`.
Run logs: `epilogues/analysis/beta_sweep_run{2,3,4}.log`.

## Headline: beta overhead tracks tile size, NOT runtime

`corr(delta, no-beta time) = +0.06` — **essentially zero**. Beta overhead is *not* a
bandwidth-headroom / "slow kernels hide it" effect. It is governed by the
`MatrixInstruction` output-tile geometry:

| MI-derived quantity | Pearson corr with median delta |
|---------------------|:---:|
| per-workgroup **tile area** (M*N) | **+0.40** |
| **wave-tile area** (wtM*wtN) = elems/thread | **+0.40** |
| number of workgroups (grid size) | **-0.36** |
| aspect M/N | -0.28 |
| waveGroupM (waves in M) | -0.26 |
| waveGroupN (waves in N) | +0.22 |
| no-beta time | +0.06 |

Tile area, wave-tile area, and elements-per-thread give the *identical* correlation
because **every surviving config runs 4 waves / 256 threads** (waveGroupM*waveGroupN = 4
for all of `[.,.,2,2]`, `[.,.,4,1]`, `[.,.,1,4]`), so
`tileArea = 1024 * wtM * wtN` and `numWorkgroups ~ 1/tileArea`. There is effectively one
dominant variable: **how big a tile each workgroup writes.** (`DepthU` is not a lever —
all 204 survivors are `DepthU=64`; the 128/192 candidates were rejected for occupancy.)

### Secondary effects (weaker, noisier)

| wave-group `[wgM,wgN]` | n | mean delta | | PGR | n | mean delta |
|---|---:|---:|---|---|---:|---:|
| **(4,1) M-heavy** | 36 | **27.4** | | **PGR2** | 67 | **30.5** |
| (2,2) | 116 | 40.8 | | PGR1 | 67 | 38.6 |
| (1,4) N-heavy | 52 | 45.6 | | PGR0 | 70 | 49.5 |

M-heavy wave layouts and higher global prefetch shave a further ~15-20 us on top of the
tile-size effect, but the single-tile control (MT512x128 in all three layouts) is too
noisy to rank the layouts cleanly, so treat these as second-order.

### Extremes (median delta over 3 runs)

Beta-resistant (small/short-tail): `MT512x128 wt8x8 (4,1)` -5, `MT384x192 wt12x6 (2,2)`
+13, `MT448x128 wt14x4 (2,2)` +12, `MT576x64 wt9x4 (4,1)` ~0.
Beta-costly (large square tiles): `MT320x320 wt10x10 (2,2)` **+105**,
`MT336x256 wt21x4 (1,4)` +116, `MT160x416 wt5x13 (2,2)` +95-111,
`MT272x320 wt17x5 (1,4)` +93. Full spread in `parse_beta_sweep.out`.

## Assembly mechanism: the beta*C epilogue is a serial post-loop tail whose length scales with wave-tile area

Disassembled beta kernels (PGR2/SK0) from
`/tmp/beta-sweep2/.../build_tmp/SOURCE/assembly/`. For each I located the
`Global Write Elements` marker (epilogue start), counted the `// load C`
`buffer_load_dwordx2` in the epilogue, and checked the last `v_mfma`:

| kernel | wave-tile area wtM*wtN | epilogue `load C` count | last `v_mfma` vs epilogue |
|--------|:---:|:---:|:---:|
| MT576x64  wt9x4   | 36 | 216 | MFMA before epilogue |
| MT512x128 wt8x8   | 64 | 384 | MFMA before epilogue |
| MT384x192 wt12x6  | 72 | 432 | MFMA before epilogue |
| MT320x320 wt10x10 | 100 | 600 | MFMA before epilogue |

Two facts, both from the disassembly:

1. **`epilogue C-loads = 6 x waveTileArea`, exactly** (216=6*36, 384=6*64, 432=6*72,
   600=6*100). The beta path issues a `buffer_load_dwordx2` C read per output sub-element
   per thread, so its work is strictly linear in the per-workgroup tile.
2. **The beta*C load/combine/store runs entirely after the MFMA main loop** — in every
   kernel the last `v_mfma` precedes the `Global Write Elements` marker, and the C loads
   sit behind the `Beta == 0` branch (`s_cbranch_scc0 label_GW_B1_GSU1`). So the beta
   tail is a *serial addition after compute*, not overlapped with the workgroup's own
   MFMAs.

The store idiom itself is already the light one and is **identical across all tiles**
(established previously): `buffer_load_dwordx2` C on a decreasing-`vmcnt` drain,
`ds_bpermute`+`v_permlane` cooperative pack, `v_fmac_f32 ..., s[sgprBeta]` combine, a
reused scalar exec mask via `s_mov_b64 exec` (zero `s_and_saveexec`, zero per-element
`v_cmp` in the NonEdge body), cooperative `buffer_store_dwordx4` (mixed with
`dwordx2`/`b64` orphan-subtile stores). Only the **count** differs.

### Why this makes big tiles pay and small tiles resist

Because the beta tail is serial and its length scales with wave-tile area, a config with
a large per-workgroup tile (e.g. MT320x320, 600 C-loads) appends a long exposed beta tail
to each workgroup, and there are fewer workgroups to overlap those tails against. A config
with a smaller per-workgroup tile (MT576x64/MT512x128, 216-384 C-loads) has a shorter tail
and launches more, smaller workgroups, so the tails hide better across the grid — hence
`corr(delta, area) = +0.40`, `corr(delta, numWorkgroups) = -0.36`, and
`corr(delta, runtime) ~ 0`. The M-heavy `(4,1)` layout and PGR2 help further by improving
C-load coalescing/overlap within the tail.

## Recommendation

To minimise beta*C overhead through `MatrixInstruction` choice on this shape:

1. **Prefer smaller per-workgroup output tiles (smaller `wtM*wtN`) that launch more
   workgroups.** This is the dominant lever: it directly shortens the serial beta
   epilogue tail. Avoid large square tiles like MT320x320 (`wt10x10`) — they carry the
   longest tail (600 C-loads) and the worst overhead (~+100 us).
2. **Within a tile-size budget, favor M-heavy wave-group layouts `[.,.,4,1]` and
   `PrefetchGlobalRead: 2`** — worth another ~15-20 us on average.
3. **The store idiom is not the bottleneck** and should not be rewritten — it is already
   the optimal cooperative-dwordx4 / reused-scalar-mask GWB path. The lever is the
   *amount* of serial epilogue work (tile size), not its *structure*.
4. **To attack the tail directly (orthogonal to MI choice):** overlap the C read with the
   final K iterations of the main loop instead of deferring the whole `beta*C` load to a
   post-MFMA tail, so the `buffer_load_dwordx2 C` stream hides under MFMA rather than
   adding `6 x waveTileArea` exposed loads after it.
