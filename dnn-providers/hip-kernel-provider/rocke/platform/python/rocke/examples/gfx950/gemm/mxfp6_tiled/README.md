<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Tiled gfx950 MXFP6 GEMM: reproduction guide

This folder holds the bench worker ([`bench.py`](bench.py)), the knob tuner
([`tune.py`](tune.py)), the [plan](PLAN.md), the
[case study](mxfp6_gemm_case_study.md) and the best configuration the case
study reports ([`best_config.json`](best_config.json)) for the builder
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
```

`ROCKE_BACKEND=python` keeps the lowering in the Python engine: the C++
engine does not read the LDS-DMA instruction offset yet.

## 1. Build the direct compiler

The compiler is the in-tree crate [`rust/rocke-codeobj`](../../../../../../rust/rocke-codeobj/README.md).
Its `target_compile` example turns serialized rocKE IR into a code object.
Build out of tree (`CARGO_TARGET_DIR` outside the repository), run its unit
tests, and point `bench.py` at the tool:

```bash
export CARGO_TARGET_DIR=$(mktemp -d)
(cd rust/rocke-codeobj && cargo test --release --lib && cargo build --release --example target_compile)
export ROCKE_DIRECT_TARGET_COMPILE=$CARGO_TARGET_DIR/release/examples/target_compile
```

`best_config.json` sets the compiler's optional passes through kernel
attributes (`cc_passes`, `cc_extra`); a compiler built from another source
does not have them.

## 2. CPU tests

Every knob value builds, validates and lowers; the barrier waits, the DMA
placement and the operand packing are checked against host models; the
tuner's search is checked on synthetic scores:

```bash
python -m pytest tests/instances/test_gfx950_block_scaled_gemm_tiled.py -q
```

## 3. Correctness gates and timing of the best configuration

One `bench.py --mode eval` process compiles a configuration (cached by its
IR and, for the direct backend, the compiler binary), runs both gates and
times it:

- the **exact gate** runs the same knobs on a small multi-tile shape with
  dyadic inputs and demands exact equality with a float64 reference after
  output rounding;
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
BEST=$(cat $EX/best_config.json)
# random data: exact gate, sampled gate, sustained timing
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" \
  --config "$BEST" --sustain-ms 1500 --repeats 5 --out "$MXFP6_OUT/best_random.json"
# all-zero operand data (exact gate and timing; the sampled gate needs random data)
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" --data zero \
  --config "$BEST" --sustain-ms 1500 --repeats 5 --out "$MXFP6_OUT/best_zero.json"
# the other FP6 format (E3M2) for both operands
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" \
  --config "$(python -c "import json,sys; c=json.load(open(sys.argv[1])); c.update(dtype_a='fp6e3m2', dtype_b='fp6e3m2'); print(json.dumps(c))" $EX/best_config.json)" \
  --out "$MXFP6_OUT/best_e3m2.json"
```

## 4. Baselines

`best_config.json` uses the canonical-A input contract: A and its scales
row-major, as a caller holds them; B and its scales pre-packed
(`b_layout="kblock"`, `b_scale_layout="kblock4"`, B in two K halves for
`b_release="half"`). `bench.py` packs the inputs for every contract and
checks against the same row-major reference. Time each baseline the same
way as the best configuration, with random and with zero data, in the same
session and on the same GPU, and compare only runs that passed both gates:

```bash
for name in default row_major kblock canonical_a_llvm; do
  case $name in
    default)          CFG='{}' ;;
    row_major)        CFG='{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave"}' ;;
    kblock)           CFG='{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave", "a_layout": "kblock", "b_layout": "kblock", "a_scale_layout": "kblock", "b_scale_layout": "kblock"}' ;;
    canonical_a_llvm) CFG='{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave", "b_layout": "kblock", "b_scale_layout": "kblock", "a_scale_fetch": "iter"}' ;;
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
- `canonical_a_llvm`: the canonical-A contract on the LLVM backend, the
  same work and input contract as `best_config.json`.

Only `canonical_a_llvm` takes the same inputs as `best_config.json`; the
other contracts move different layouts and are labelled as such when
compared. With random operand data the clock settles at the board power
limit, so compare random-data and zero-data results only within their own
data set.

## 5. The tuner

`tune.py` hill-climbs over the knobs in its `KNOBS` table with look-back
over pairs of failed moves, one `bench.py` process per evaluation; it needs
the GPU, and with `--pmc best` also `rocprofv3`. It appends every
evaluation to `ledger.jsonl` under `--ledger` and rewrites `report.md`
there; a resumed run re-reads the ledger. To climb from the best
configuration:

```bash
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.tune --ledger "$MXFP6_LEDGER" \
  --cache "$MXFP6_CACHE" --pmc best --start "$BEST" --max-evals 60
```

`--threshold` sets the relative gain a move needs, and `--confirm` the
number of alternating re-timings of winner and incumbent before a move is
accepted. The case study describes the method and every knob's decision.
