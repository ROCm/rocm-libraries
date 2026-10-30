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
`-- <pytest args>`. Pin engines with `HIPDNN_FWD_ENGINE` / `HIPDNN_BWD_ENGINE`.
`--fwd-no-stats` (with `--suite forward`) runs the forward without Stats, so engines that decline
when Stats are requested can be exercised; AOTriton's M buffer is filled with zeros (its wrapper rejects NaN in M) and
`test_forward.py` compares only O. The runner refuses it for any collection that includes
backward or varlen cases, whose backward consumes the forward's Stats.
AOTriton's calibration and backend knobs (`USE_ADIFFS_TXT`, `RECORD_ADIFFS_TO`, `SKIP_BWD`,
`FWD_IMPL`, `BWD_IMPL`, ...) are refused; `FOR_RELEASE`, `SMALL_VRAM` and
`AOTRITON_REF_DEVICE_OPTION` pass through.

Outputs, in `<out>/` (default `./sdpa_conformance_results/<arch>/<run>/shard-<I>of<N>/`):
`collected.txt` (every collected node id) and `results.csv` (`nodeid,outcome,mode,fwd,bwd,reason`,
flushed per row). After a run, or a killed one:

```
python aotriton/reconcile.py <arch>/<run> --out <arch>/<run>/reconciled.csv
```

`mode` is `hipdnn` or `ref-fwd` (hipDNN forward declined; a reference forward fed the hipDNN
backward). A `ref-fwd` pass proves the backward only.
