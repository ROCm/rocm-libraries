<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# Kernel generators

`SkinnyGEMM.py` describes the skinny MXFP8 GEMM kernels emitted by `tensilelite_codegen` as custom kernels.
`SkinnyGEMMLogic.py` writes the gfx950 logic file that lists the pool of knob sets and the routing rows.

## Pool and routing

The pool holds every knob set in both `useBeta` variants. `useBeta=false` ignores C and beta and asserts
`AssertBetaZero`; `useBeta=true` prefetches C, computes `alpha * acc + beta * C` and asserts the C strides that match
its vector widths. Every pool entry has an enumeration-only row (an empty range), so explicit selection and
`--requested_solution -1` list both variants.

Routing rows (`data/SkinnyGEMMRows.yaml`, params without `useBeta`) target the `useBeta=false` variant only. A call
with beta != 0 fails its `AssertBetaZero`, the Range table returns no solution (it returns on the first row whose
range matches, with no fall-through to other rows), and `ExactLogicLibrary` (`include/Tensile/ExactLogicLibrary.hpp`,
`findBestSolution`) falls through to the next library (Origami). The `useBeta=true` variants are reachable only through
the enumeration or explicit selection. Tuning therefore ignores the `useBeta=true` results of the bench.

## Re-tuning the skinny GEMM routing

The routing rows (`data/SkinnyGEMMRows.yaml`) are measured, not hand-written. Everything below runs from
`projects/hipblaslt/tensilelite` with the TensileLite venv, on a machine with the GPUs to benchmark and a hipBLASLt
build that contains the skinny pool (`hipblaslt-bench` from that build).

```bash
python -m tensilelite.KernelGenerators.SkinnyGEMMTune tune \
    --bench ../build/clients/hipblaslt-bench --results /path/to/campaign --gpus 0-7
```

What `tune` does:

1. Benchmarks every point of `data/SkinnyGEMMGrid.yaml` (both orientations, skinny extent 1..16, the v_llm wide
   extents and K values, plus its `extra_points`) with `--requested_solution -1`, which times Origami's picks and every
   generated kernel in one process. Each point is run `repeats` times (spread over the GPUs, interleaved with the
   other points) and the minimum is used, since contention only adds time. Points whose result is still undecided
   within the noise get `refine_repeats` runs. The benches set `TENSILE_PREDICTION_LIB=1`, which skips Equality and
   Range rows, so the first listed solution is Origami's own default even when routing rows are installed.
2. Cuts the lattice into boxes and merges them into disjoint cells. A pool entry serves a cell only where it is at
   least `--margin` (default 1.25) faster than Origami's default at every corner, within `--tolerance` (1.10) of the
   best generated kernel there, and never `--regression` (1.05) slower than Origami at any other measured point
   inside the cell. Cells that no entry qualifies for are not routed and fall through to Origami.
3. Draws `held_out.count` points inside the cells (not on the lattice), benchmarks them the same way and checks the
   cells offline against them. The final cells are rebuilt from all measurements, so the held-out points also veto
   entries that regress.
4. Writes `data/SkinnyGEMMRows.yaml` and regenerates the logic file (`SkinnyGEMMLogic`) from it; writes `cells.md`,
   `points.md`/`points.csv` and `summary.md` into the results directory.

The run is resumable: interrupt it (Ctrl-C) and run the same command again; only missing runs are measured. The
results directory holds `results.jsonl` (one line per run), the raw bench output of every run and the reports.

Other commands:

- `cells --results DIR [--margin ...]` repeats steps 2-4 offline from a results directory (no GPU), e.g. to try
  other thresholds. On quiet GPUs a smaller margin is justified; `tune` with the new `--margin` is all that is needed.
- `validate --bench ... --results DIR` runs after the rebuild below. It benchmarks the library's own default pick (the
  routing rows active) on every held-out point, on up to 200 routed and 50 non-routed grid points, and compares it
  with the old default timed in the same process. It writes `validate.md`/`validate.csv` and exits with status 1 on a
  regression or an unexpected pick; add such points to `extra_points` in the grid file and run `tune` again.

The one manual step: rebuild hipBLASLt so the library contains the regenerated logic (the library build does not
remove stale generated code objects, so delete them first):

```bash
cd projects/hipblaslt
rm -f build/Tensile/library/gfx950/*_Generated* build/install/lib/hipblaslt/library/gfx950/*_Generated*
cmake --build build -j 200 && cmake --install build
```

Then run `validate`, and commit `data/SkinnyGEMMRows.yaml`, `data/SkinnyGEMMGrid.yaml` (if changed) and the
regenerated logic file. The plain command `python -m tensilelite.KernelGenerators.SkinnyGEMMLogic` reproduces the
committed logic file from the committed rows byte for byte.
