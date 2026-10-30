# sdpa_conformance

Runs AOTriton's own SDPA pytest suite with hipDNN as the operation under test.
Every case ends as `pass`, `fail`, `crash`, `not-run`, `declined` (hipDNN has no engine, with the reason),
`native-skip` (AOTriton's own skip) or `no-hipdnn` (the test passed but hipDNN computed nothing). Layout, vendoring and the design contract:
[`aotriton/VENDOR.md`](aotriton/VENDOR.md) and `Plans/sdpa-aotriton-conformance.md`.

| Path | Role |
|---|---|
| `hipdnn_sdpa/` | adapter from torch tensors to hipDNN SDPA graphs |
| `aotriton/vendor/` | AOTriton's tests, byte-identical at a pinned commit |
| `aotriton/shim/` | `aotriton_flash.py` and `pyaotriton` replacements |
| `aotriton/hipdnn_aotriton_plugin.py` | outcome table, mode record, CSV, `--shard`, `--fake` |
| `aotriton/run_aotriton.py`, `reconcile.py` | GPU runner; offline `not-run` reconciliation |

## CPU tests

These tests are not registered with CTest and are not part of hipDNN's test categories or CI: they
need torch, pytest-xdist and numpy (`aotriton/requirements.txt`), which hipDNN's builds do not
require. Run them by hand from `tools/sdpa_conformance/`:

```
python -m pytest aotriton/tests          # plugin, shim, reference forward, reconcile, runner
python -m pytest tests -m 'not gpu'      # hipdnn_sdpa (graph tests need hipdnn_frontend importable)
python aotriton/sync.py --check --no-collect   # vendored files unchanged (hashes)
python aotriton/sync.py --check          # plus collection counts per FOR_RELEASE level (~5 min)
```

## GPU run

Needs ROCm torch with hipDNN, its providers and Python bindings installed into torch's ROCm SDK
prefix. Use dnn-benchmarking's `setup_env.py` (`--torch-mode rocm`) with its `rocm-libraries`
checkout at the commit under test, then activate its venv, which sets `ROCM_PATH`. Every pytest
process imports `hipdnn_frontend` first and fails unless the `libhipdnn_backend` mapped into it
(`/proc/self/maps`) is under `--hipdnn-prefix` (default `$ROCM_PATH`): the ROCm devel wheel
ships an older copy of that library (dnn-benchmarking `docs/troubleshooting.md`, "Mixed ROCm torch
and hipDNN prefixes").

```
python aotriton/run_aotriton.py --arch gfx950 --run <name>                 # test_backward + test_varlen
python aotriton/run_aotriton.py --arch gfx950 --run <name> --suite forward # test_forward.py
python aotriton/run_aotriton.py --out <dir> --shard 2/4 --release 1        # one shard, FOR_RELEASE=1
python aotriton/run_aotriton.py --out <dir> --fake always-decline          # harness check: no passes
```

Other options: `--packed-gpu` (adds `tests/`' `gpu` cases), `--workers N` (default one per GPU),
`-- <pytest args>`, `--hipdnn-nodeids-file PATH` (plugin option: run only the listed node ids). Pin engines with `HIPDNN_FWD_ENGINE` / `HIPDNN_BWD_ENGINE`.
`--layout-mode {pytorch,efficient,aotriton}` (default `pytorch`) chooses the memory layout every SDPA
tensor is allocated in, with no copies: `pytorch` stores O, dO and the gradients like Q (dQ/dK/dV/dB
like their inputs), `efficient` stores O, dO, dQ, dK and dV BSHD (dB like the bias), `aotriton` keeps
the vendored suite's own layouts (O and dO BHSD). q/k/v/b keep the vendored cases' own layouts in every
mode. The mode reaches the pytest workers as `HIPDNN_SDPA_CONFORMANCE_LAYOUT_MODE`, appears in the pytest
header, and `--all-engines` pass 2 inherits it. Varlen cases are unaffected.
`--fwd-no-stats` (with `--suite forward`) runs the forward without Stats, so engines that decline
when Stats are requested can be exercised; AOTriton's M buffer is filled with zeros (its wrapper rejects NaN in M) and
`test_forward.py` compares only O. The runner refuses it for any collection that includes
backward or varlen cases, whose backward consumes the forward's Stats.
`--all-engines` tests every applicable engine, not only the heuristic winner: after the normal
run it pins each other engine that is applicable to a case (`fwd_engines` / `bwd_engines` in
`results.csv`) and re-runs exactly those cases, `test_forward.py` with `HIPDNN_FWD_ENGINE`, every
other case with `HIPDNN_BWD_ENGINE`. It composes with `--suite`, `--shard`, `--release`,
`--workers` and (forward sweep) `--fwd-no-stats`; it refuses `--fake` and pre-set engine pins.
AOTriton's calibration and backend knobs (`USE_ADIFFS_TXT`, `RECORD_ADIFFS_TO`, `SKIP_BWD`,
`FWD_IMPL`, `BWD_IMPL`, ...) are refused; `FOR_RELEASE`, `SMALL_VRAM` and
`AOTRITON_REF_DEVICE_OPTION` pass through.

Outputs, in `<out>/` (default `./sdpa_conformance_results/<arch>/<run>/shard-<I>of<N>/`):
`collected.txt` (every collected node id) and `results.csv` (`nodeid,outcome,mode,fwd,bwd,reason`,
flushed per row; `fwd_engines` / `bwd_engines` list the applicable engines in heuristic rank
order, `;`-joined, for a direction hipDNN planned unpinned). With `--all-engines`, each sweep
writes the same files under `all-engines/<fwd|bwd>/<engine-slug>/`. After a run, or a killed one:

```
python aotriton/reconcile.py <arch>/<run> --out <arch>/<run>/reconciled.csv
```

`mode` is `hipdnn` or `ref-fwd` (hipDNN forward declined; a reference forward fed the hipDNN
backward). A `ref-fwd` pass proves the backward only.

For an `--all-engines` run, `reconcile.py <arch>/<run> --engines <out>` writes `<out>/engines.csv`
(`nodeid,direction,engine,source,outcome,reason`; `source` is `heuristic` for the engine that
served the case in the normal run, `pinned` for a sweep) and prints pass/fail/declined counts per
direction and engine. A pinned case served by another engine is recorded as `declined`, never as
that engine's pass.
