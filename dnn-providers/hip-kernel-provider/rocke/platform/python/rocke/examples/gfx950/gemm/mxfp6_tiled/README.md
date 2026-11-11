<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Tiled gfx950 MXFP6 GEMM: reproduction guide

This folder holds the bench worker ([`bench.py`](bench.py)), the knob tuner
([`tune.py`](tune.py)), the [plan](PLAN.md), the
[case study](mxfp6_gemm_case_study.md) and the best configuration the case
study reports ([`best_config.json`](best_config.json), on the 16x16x128
scaled MFMA) for the builder
[`instances/gfx950/block_scaled_gemm_tiled.py`](../../../../instances/gfx950/block_scaled_gemm_tiled.py).
This guide rebuilds the pieces and re-runs the correctness gates and the
timing. It records no measurements: `bench.py` and `tune.py` write theirs
to files you choose, which should live outside the repository.

## Requirements

- A gfx950 GPU with ROCm (HIP runtime and `libamd_comgr`), for the gates
  and the timing. The CPU tests need neither.
- Python 3 with `numpy`, `ml_dtypes` (the exact gate rounds with it) and
  `pytest`. torch is not needed.
- A Rust toolchain (`cargo`), for the direct code-object compiler that
  `best_config.json` uses (`"backend": "direct"`).
- Optionally `rocprofv3`, for the tuner's counter readings, and the LLVM
  tools `llvm-readelf` / `llvm-objdump`, for the register counts and the
  instruction hash `bench.py` records.

All commands below run from `dnn-providers/hip-kernel-provider/rocke/platform`
unless they say otherwise. `MXFP6_CACHE`, `MXFP6_OUT` and `MXFP6_LEDGER` are
directories outside the repository, for example made with `mktemp -d`.

```bash
export PYTHONPATH=python ROCKE_BACKEND=python
export MXFP6_CACHE=$(mktemp -d) MXFP6_OUT=$(mktemp -d) MXFP6_LEDGER=$(mktemp -d)
EX=python/rocke/examples/gfx950/gemm/mxfp6_tiled
BEST=$(cat $EX/best_config.json)
```

`ROCKE_BACKEND=python` keeps the lowering in the Python engine: the C++
engine does not read the LDS-DMA instruction offset yet.

## 1. Build the direct compiler

The compiler is the in-tree crate [`rust/rocke-codeobj`](../../../../../../rust/rocke-codeobj/README.md).
Its `target_compile` example turns serialized rocKE IR into a code object.
Build out of tree (`CARGO_TARGET_DIR` outside the repository), run its unit
tests, and point `bench.py` at the tool and the compiler at its levers:

```bash
export CARGO_TARGET_DIR=$(mktemp -d)
(cd rust/rocke-codeobj && cargo test --release --lib && cargo build --release --example target_compile)
export ROCKE_DIRECT_TARGET_COMPILE=$CARGO_TARGET_DIR/release/examples/target_compile
export ROCKE_DIRECT_LEVERS=fp6pass,scalesep
```

The direct-backend configurations in this folder were tuned and measured
with two compiler behaviours that the crate keeps behind levers: a
block-scaled MFMA reads its A and B scales where they are instead of
copying them into a fresh register pair (`scalesep`), and FP6/FP4 MFMAs are
timed at their own pass count in the hazard waits and the schedule
(`fp6pass`). Set `ROCKE_DIRECT_LEVERS=fp6pass,scalesep` for every
direct-backend run; without it the same IR compiles to different code. The
variable applies to every kernel the tool compiles; a kernel attribute that
carries the levers in the IR is a pending compiler follow-up. `bench.py`'s
compile cache keys on the IR and the compiler binary, not on the levers, so
use a fresh `MXFP6_CACHE` after changing them.

`best_config.json` also turns on the compiler's optional passes through
kernel attributes (`cc_passes`, `cc_extra`) and needs its scaled 16x16x128
MFMA; a compiler built from another source may lack them.

## 2. CPU tests

Every knob value builds, validates and lowers; the barrier waits, the DMA
placement and the operand packing are checked against host models,
including the 16x16x128 path's lane maps, packers and VMEM stream; the
tuner's search is checked on synthetic scores:

```bash
python -m pytest tests/instances/test_gfx950_block_scaled_gemm_tiled.py \
  tests/instances/test_gfx950_block_scaled_gemm_tiled_a16.py -q
```

## 3. Correctness gates and timing of the best configuration

One `bench.py --mode eval` process compiles a configuration (cached by its
IR and, for the direct backend, the compiler binary), runs both gates and
times it:

- the **exact gate** runs the same knobs on a small multi-tile shape, two
  tiles by two (`xcd_remap="none"` when the remap needs a full grid), with
  dyadic inputs and demands exact equality with a float64 reference after
  output rounding. Its K is 1152 for 32x32x64 configurations. The
  16x16x128 path runs whole trips of eight K128 stages, so its K must be
  1024 x 2^n; `bench.py` checks one trip, K = 1024;
- the **sampled gate** checks a sampled block of the target-shape output
  (edge rows and columns plus random ones) against the float64 reference
  within a BF16 tolerance; it runs with `--data random` only;
- **timing** runs back-to-back launches for `--sustain-ms` first, so the
  clock has settled, then takes the median of `--repeats` timed loops.

The record's `status` is `ok` when no gate failed and `wrong` otherwise;
with `--data zero` the sampled gate is `skipped`. `--skip-checks` times
without the gates (`status` `timing`). The default shape is
`4096,4096,65536` (`--shape M,N,K`).

```bash
E3M2=$(python -c "import json,sys; c=json.load(open(sys.argv[1])); c.update(dtype_a='fp6e3m2', dtype_b='fp6e3m2'); print(json.dumps(c))" $EX/best_config.json)
# random data: exact gate (one trip), sampled gate at the target shape, sustained timing
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" \
  --config "$BEST" --sustain-ms 1500 --repeats 5 --out "$MXFP6_OUT/best_random.json"
# the other FP6 format (E3M2) for both operands
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" \
  --config "$E3M2" --sustain-ms 1500 --repeats 5 --out "$MXFP6_OUT/best_e3m2.json"
# all-zero operand data (exact gate and timing; the sampled gate needs random data)
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" --data zero \
  --config "$BEST" --sustain-ms 1500 --repeats 5 --out "$MXFP6_OUT/best_zero.json"
```

The 16x16x128 path's exact gate covers one and two trips, 512x512x1024 and
512x512x2048, in both formats. `bench.py` runs the first; this runs both
with its helpers and prints one line per shape and format:

```bash
python - "$MXFP6_CACHE" $EX/best_config.json <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
from rocke.examples.gfx950.gemm.mxfp6_tiled import bench

cache, cfg = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
for fmt in ("fp6e2m3", "fp6e3m2"):
    for k in (1024, 2048):
        spec = bench._spec((512, 512, k), {**cfg, "dtype_a": fmt, "dtype_b": fmt, "xcd_remap": "none"})
        hsaco, _ = bench.compile_cached(spec, cache)
        inputs = bench.dyadic_inputs(spec)
        dev = bench.Device()
        try:
            module, launch, c_ptr = bench.run_case(dev, hsaco, spec, inputs)
            launch()
            dev.rt.sync()
            got = dev.download_16bit(c_ptr, (spec.M, spec.N), spec.dtype_c)
            module.unload()
        finally:
            dev.close()
        ref = bench._round_out(bench.reference(spec, *inputs)[0], spec.dtype_c)
        bad = int(np.count_nonzero(got != ref))
        print(fmt, f"512x512x{k}", "pass" if bad == 0 else f"FAIL: {bad} of {ref.size} differ")
EOF
```

`bench.py` draws uniform random (`--data random`) or all-zero operand data.
Some decisions in the case study, among them the move to the 16x16x128
path, were taken with Gaussian-distributed operand data, which `bench.py`
does not generate.

## 4. Baselines

`best_config.json` uses the canonical-A input contract: A and its scales
row-major, as a caller holds them; B and its scales pre-packed, here in
the order the 16x16x128 path reads them (`bench.b_c0c1p`,
`bench.scales16_pairs`). `bench.py` packs the inputs for every contract and
checks against the same row-major reference. Time each baseline the same
way as the best configuration, with random and with zero data, in the same
session and on the same GPU, and compare only runs that passed both gates:

```bash
for name in default row_major kblock canonical_a_llvm direct_32x32x64; do
  case $name in
    default)          CFG='{}' ;;
    row_major)        CFG='{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave"}' ;;
    kblock)           CFG='{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave", "a_layout": "kblock", "b_layout": "kblock", "a_scale_layout": "kblock", "b_scale_layout": "kblock"}' ;;
    canonical_a_llvm) CFG='{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave", "b_layout": "kblock", "b_scale_layout": "kblock", "a_scale_fetch": "iter"}' ;;
    direct_32x32x64)  CFG='{"stages": 2, "a_ring": "line", "xcd_remap": "xcd_block", "mfma_form": "intrinsic", "agpr_alloc": "auto", "backend": "direct", "b_layout": "kblock", "b_scale_layout": "kblock4", "a_scale_fetch": "iter2", "mfma_group": 1, "setprio": true, "fill_split": "line1", "scale_opsel": true, "frag_pipeline": "step", "b_release": "half", "a_scale_path": "lds2", "mfma_order": "serp", "dma_place": "even_bfirst", "cc_passes": "all", "b_scale_fetch": "pair", "prologue_wait": "stage0", "k_order": "t10", "cc_extra": "lds_waw+u16_zext", "as_dma_place": "split_light"}' ;;
  esac
  for data in random zero; do
    python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" --data $data \
      --config "$CFG" --sustain-ms 1500 --repeats 5 --out "$MXFP6_OUT/${name}_${data}.json"
  done
done
```

- `default`: every knob at its default (row-major contract, LLVM backend);
- `row_major` and `kblock`: the case study's best configurations of the
  row-major and the fully pre-packed contracts, on the LLVM backend;
- `canonical_a_llvm`: the canonical-A contract on the LLVM backend;
- `direct_32x32x64`: the case study's best configuration on the 32x32x64
  MFMA (canonical-A contract, direct backend, so it also needs
  `ROCKE_DIRECT_LEVERS`), the one the 16x16x128 path was compared against.

`canonical_a_llvm` and `direct_32x32x64` do the same work under the same
input contract as `best_config.json` (each kernel reads B pre-packed in its
own order); the other contracts move different layouts and are labelled as
such when compared. With random operand data the clock settles at the
board power limit, so compare random-data and zero-data results only
within their own data set.

## 5. The tuner

`tune.py` hill-climbs over the knobs in its `KNOBS` table with look-back
over pairs of failed moves, one `bench.py` process per evaluation (random
data unless `--data zero`); it needs the GPU, and with `--pmc best` also
`rocprofv3`. It appends every evaluation to `ledger.jsonl` under `--ledger`
and rewrites `report.md` there; a resumed run re-reads the ledger. To climb
from the best configuration:

```bash
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.tune --ledger "$MXFP6_LEDGER" \
  --cache "$MXFP6_CACHE" --pmc best --start "$BEST" --max-evals 60
```

`--threshold` sets the relative gain a move needs, and `--confirm` the
number of alternating re-timings of winner and incumbent before a move is
accepted; a move that compiles to the incumbent's code counts as noise.
From `best_config.json` a move of a 32x32x64 schedule field either builds
the same kernel (the 16x16x128 path ignores the field) or is refused, and
`mfma_atom`, `sched16` and `epi16` are not tuner knobs. The case study
describes the method and every knob's decision.
