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
| `aotriton/run_aotriton.py`, `reconcile.py`, `completion.py` | GPU runner and its passes; offline `not-run` reconciliation, engine tables and run report; the shards a resumed link must finish |
| `aotriton/conformance_contract.py` | names and formats shared by the runner, plugin, shim and `hipdnn_sdpa` |

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
`--all-engines` tests every applicable engine, not only the heuristic winner. After the normal run
(pass 1) the runner adds, under `<out>/all-engines/`, one pytest run per pass, each with its own
`results.csv`, `nodeids.txt` (the cases it must run) and `pass.json` (kind, direction, pins):

| Pass directory | Pins | Cases |
|---|---|---|
| `fwd/<engine>/`, `bwd/<engine>/` | one engine | each applicable engine that did not serve the case (`test_forward.py` pins `HIPDNN_FWD_ENGINE`, every other case `HIPDNN_BWD_ENGINE`) |
| `pair/<fwd>__<bwd>/` (only with `--pairs`) | both | backward cases: every stats-capable forward that did not serve the case against every backward engine |
| `refwd/discovery/`, `refwd/<bwd>/` | `HIPDNN_FWD_ENGINE=reference` | backward cases whose forward failed or crashed in any pass: the backward unpinned, then each backward engine; with `--pairs` the pair passes then cover them too |

The forward x backward pair passes are opt-in (`--pairs`): by default every pass pins one direction, so a
failure names the engine of that direction. The reference-forward passes always run.

Kernel ids are opt-in too. `--kernel-ids` exports `HIPDNN_SDPA_KERNEL_LOG=1` to every pass; the plugin then
installs `hipdnn_sdpa`'s backend log callback and fills `fwd_kernel` / `bwd_kernel`. The callback raises the
backend log level to INFO and forces ingestor applicability inline (`HIPDNN_APPLICABILITY_MIN_OFFLOAD_VARIANTS`),
which changes how the product runs, so by default it is not installed and the kernel columns stay empty (nothing
downstream needs them; the engine that faulted is recorded regardless). `--exhaustive` implies `--kernel-ids`.

`--exhaustive` tests every kernel (candidate) of an engine, not only the one it picks. Candidates come from
`hipdnn_bench enumerate --graph <json> --engine-name <engine> --plugin-dir <prefix>/lib/hipdnn_plugins/engines`
(`<prefix>/bin/hipdnn_bench`, paged, cached per problem and engine; no hipDNN API is added). Pass 1 records,
per case and engine, the candidate count in `fwd_candidate` / `bwd_candidate` (a JSON object `{engine: count |
reason}`; the reason is hipDNN's message for an engine that cannot enumerate: ASM and any other non-ingestor
engine, or an ingestor engine whose catalog has ambiguous knob tuples). Under `<out>/cand/`:

| Pass directory | Pins | Cases |
|---|---|---|
| `<fwd\|bwd>/<engine>/<k>/` | the engine and `HIPDNN_FWD_CANDIDATE` / `HIPDNN_BWD_CANDIDATE` = `k` | every case whose count exceeds `k`; one pass per `k` up to the largest count any case reported |
| `<fwd\|bwd>/<engine>/not-enumerable/` | none (no pytest run) | each case where the engine could not enumerate: a `not-enumerable` row with the message |

The runtime builds candidate `k` with `create_execution_plan_ext(engine_id, knob_settings)`; the kernel id the
backend logs must be the candidate's (a slotted engine's `prep=..|dkdv=..|dq=..` compares by slot), otherwise
the case is `declined` with `candidate_not_honored`. In a candidate pass the candidate columns hold the pinned
index. A case whose worker died in pass 1 has no counts and gets no candidate pass; the backward on the
reference forward is not enumerated.
With `--all-engines` too (EXTREME) the engine sweeps run beside the candidate passes; with `--pairs` as well
the runner also adds `cand/pair/<fwd>__<bwd>/<i>x<j>/` for backward cases: every forward engine against every backward
engine of the case (winners included), over every pair of candidate indices, and
`cand/pair/<fwd>__<bwd>/not-enumerable/` when either engine could not enumerate. That costs the product of the
two candidate counts per pair; without `--pairs` each candidate pass pins one direction and the cost is their
sum. `--exhaustive` refuses `--fake` and pre-set pins;
`reconcile.py --engines` reads `cand/` too (`source` `candidate` or `candidate-pair`, plus `fwd_candidate`,
`bwd_candidate`).

It composes with `--suite`, `--shard`, `--release`, `--workers` and (forward sweep)
`--fwd-no-stats`; it refuses `--fake` and pre-set engine pins.
`--nodeids-file PATH` limits pass 1 to the listed node ids; the pinned passes keep their own listings.
`--resume` accepts a non-empty `--out` and runs, per pass directory, only the node ids that have no row,
from a node-id file and without `--shard`; a `crash` row counts as a row, so a crash is never run again.
`--hang-timeout S` (default 1800, 0 disables) starts `pytest_gpu_lease.watchdog` beside every pytest run
on a node-local lock file (`--lock-dir`, `$HIPDNN_SDPA_LOCK_DIR`, else a temporary directory) with its
stderr in `<pass dir>/watchdog.log`; a worker it kills is a `crash` row with reason `hang`, and the
relayed stack goes to the case's repro directory.

Every pass exports `HIPDNN_SDPA_PASS_SLUG` (`main`, `fwd-<engine>`, `pair-<fwd>__<bwd>`,
`refwd-discovery`, ...) and uses its own subdirectory of `HIPDNN_SDPA_INFLIGHT_DIR` (node-local; default
a temporary directory), where `hipdnn_sdpa` leaves a marker per worker. A worker that dies leaves its
marker; the controller turns it into a `crash` row naming the engine (`fwd`/`bwd` hold
`hipdnn:<engine>`), the direction (`fail_side`), the stage (`build` or `execute`), the engine lists and
the kernel, then deletes the marker. After each pass the runner re-runs every new `crash` case once with
`HIPDNN_SDPA_REPRO_CAPTURE=<out>/repro` into `<pass dir>/capture/` (excluded from `engines.csv` and the
accounting) so a crash leaves a replayable repro directory too. A `fail` row saves its repro directly.
Repro directories are `<out>/repro/<sha1(nodeid)[:12]>-<pass slug>/`, capped at 3 per (engine, test
function) and 2 GiB per shard.

AOTriton's calibration and backend knobs (`USE_ADIFFS_TXT`, `RECORD_ADIFFS_TO`, `SKIP_BWD`,
`FWD_IMPL`, `BWD_IMPL`, ...) are refused; `FOR_RELEASE`, `SMALL_VRAM` and
`AOTRITON_REF_DEVICE_OPTION` pass through.

Outputs, in `<out>/` (default `./sdpa_conformance_results/<arch>/<run>/shard-<I>of<N>/`):
`collected.txt` (every collected node id) and `results.csv` (`nodeid,outcome,mode,fwd,bwd,reason,
fwd_engines,bwd_engines,fwd_kernel,bwd_kernel,fail_side,stage,repro,fwd_candidate,bwd_candidate`, flushed per row; the engine
lists hold the applicable engines in heuristic rank order, `;`-joined, for a direction hipDNN planned
unpinned; `*_kernel` is the ingestor kernel id; `fail_side` is `fwd` or `bwd` for a `fail` (the vendored
forward comparison, a forward case or a backward that never ran count as `fwd`) and the direction in
flight for a `crash`; `stage` is `build` or `execute`; `repro` is the repro directory under `<out>`).
With `--all-engines`, each pass writes the same files under `all-engines/`. After a run, or a killed one:

```
python aotriton/reconcile.py <arch>/<run> --out <arch>/<run>/reconciled.csv
```

`<arch>/<run>` holds one `<suite>-<k>of<N>/` directory per shard (`default-3of16`; `shard-<k>of<N>` too).
`reconcile.py <results>/<run> --report [--shards-file FILE] [--baseline FILE]` prints the whole run: per-shard
accounting (`FILE`: `<suite> <k> <N>` per line, so a shard without a directory shows as missing), outcome totals,
passes and per-(direction, engine) counts over every shard, the engines no case passed with their first decline
reasons, every `fail` / `crash` with engine, side, stage and repro path, and the failing node ids new to the run or
fixed against `--baseline` (one node id per line). `completion.py <results>/<run> [--shards-file FILE]` lists the
shards whose `--resume` still has work (`<suite> <k> <N>` lines on stdout, reasons on stderr).

`mode` is `hipdnn` or `ref-fwd` (hipDNN forward declined; a reference forward fed the hipDNN
backward). A `ref-fwd` pass proves the backward only.

For an `--all-engines` run, `reconcile.py <arch>/<run> --engines <out>` writes `<out>/engines.csv`
(`nodeid,direction,engine,source,outcome,reason,pin_fwd,pin_bwd,kernel,stage,fwd_candidate,bwd_candidate`; `source` is `heuristic`
for the engine that served (or, for a crash, was in flight in) the case in the normal run, then `pinned`,
`pair`, `refwd`, `discovery`, `candidate` or `candidate-pair` per pass). A case a pass listed but never recorded is a `not-run` row of
that pass. It prints counts per direction and engine, per pass the cases expected against recorded, and
every engine of `<out>/engine-inventory.txt` (written by the job, one name per line) with its counts,
zeros included, and for an engine no case reached the first decline reasons the workers logged under
`<out>/declines/` (an engine matches a logged name when the names are equal or one is a `:`-suffix of
the other, or when the logged name is a `...PlanBuilder` class and the inventory name is a built-in
SDPA engine such as `ASM_SDPA_ENGINE`). A pinned case served by another engine is recorded as
`declined`, never as that engine's pass.
