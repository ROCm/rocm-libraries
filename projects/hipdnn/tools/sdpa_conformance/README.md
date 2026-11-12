# sdpa_conformance

Runs AOTriton's own SDPA pytest suite with hipDNN as the operation under test.
Every case ends as `pass`, `fail`, `crash`, `not-run`, `declined` (hipDNN has no engine, with the reason),
`native-skip` (AOTriton's own skip) or `no-hipdnn` (the test passed but hipDNN computed nothing). Layout and vendoring:
[`aotriton/VENDOR.md`](aotriton/VENDOR.md); the names and formats the parts share:
[`aotriton/conformance_contract.py`](aotriton/conformance_contract.py).

| Path | Role |
|---|---|
| `hipdnn_sdpa/` | adapter from torch tensors to hipDNN SDPA graphs |
| `aotriton/vendor/` | AOTriton's tests, byte-identical at a pinned commit |
| `aotriton/shim/` | `aotriton_flash.py` and `pyaotriton` replacements |
| `aotriton/hipdnn_aotriton_plugin.py` | outcome table, mode record, CSV, `--shard`, `--fake` |
| `aotriton/run_aotriton.py`, `reconcile.py`, `completion.py` | GPU runner and its passes; offline `not-run` reconciliation, engine tables and run report; the shards a resumed link must finish |
| `aotriton/replay.py` | re-executes a saved repro directory: `python aotriton/replay.py <repro-dir> [--device cuda\|cpu]` |
| `aotriton/conformance_contract.py` | names and formats shared by the runner, plugin, shim and `hipdnn_sdpa` |

## CPU tests

These tests are not registered with CTest and are not part of hipDNN's test categories or CI: they
need torch, pytest-xdist and numpy (`aotriton/requirements.txt`: the versions they were tested with;
torch is the one already installed), which hipDNN's builds do not require. Run them by hand from `tools/sdpa_conformance/`:

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
(`/proc/self/maps`) is under `--hipdnn-prefix` (default `$ROCM_PATH`), see "Mixed ROCm torch and
hipDNN prefixes" below.

The `hipdnn_frontend` in that venv must carry every binding `hipdnn_sdpa` calls on it (for example
`Tensor.set_ragged_offset`, `Graph.sdpa`, `Graph.sdpa_backward`). Each pytest process checks them
after the import (`hipdnn_sdpa.missing_bindings()`) and stops with a pytest usage error (exit 4), naming the missing
`Class.attr` names, when the frontend is older; there is no version check, only this one.

### Mixed ROCm torch and hipDNN prefixes

A ROCm torch wheel can carry its own `libhipdnn_backend.so` next to HIP and COMGR, older than the hipDNN
install under test. Whichever copy the process maps first stays loaded, so the conformance run imports
`hipdnn_frontend` before torch and checks `/proc/self/maps`: the mapped `libhipdnn_backend` must be under
`--hipdnn-prefix`, or the run stops. Import order does not make different HIP or COMGR versions
compatible. To see what a process loaded, read `/proc/self/maps` for `libhipdnn_backend`,
`libamdhip64` and `libamd_comgr`; a package on disk says nothing about the library mapped. Use a ROCm
torch wheel and a hipDNN install of the same ROCm version,, with the selected prefix's `lib` first on
`LD_LIBRARY_PATH`. Do not use `LD_PRELOAD` to force another backend: two LLVM
runtimes in one process can abort it with `LLVM ERROR: support is already registered`. The recipe
is dnn-benchmarking's `docs/setup.md`, section of the same name.

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
Layout coverage differs by mode (apart from `test_memory_layouts`, which sets every layout itself in every mode): the default `pytorch` never runs AOTriton's own mix (BSHD Q/K/V with BHSD
O/dO), because O and dO follow Q; `aotriton` runs that mix but never BSHD O/dO; `efficient` always runs BSHD
O/dO and never BHSD O/dO. A full sweep runs both `pytorch` and `aotriton`.
A requested layout is replaced by its equivalent at the case's shape (`canonical_perm`): an axis of extent 1 (one
head, unit seqlen, batch 1) has no place in memory, so AOTriton's own layout check would read the allocation back as
another permutation; for extents above 1 `efficient` still never runs BHSD O/dO in memory order (at extent 1 the two are one allocation), and `aotriton` mode is unchanged.
The layout wrappers find the `storage_flip` argument of the vendored `_do_test_op_bwd` by name (`unpack_names`); a
vendored refresh that drops or renames it fails with a `RuntimeError` naming the function instead of changing a
wrong argument.
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

The forward x backward pair passes are opt-in (`--pairs`, only together with `--all-engines`; without it
the runner exits 2 with a usage error): by default every pass pins one direction, so a
failure names the engine of that direction. The reference-forward passes always run.

A `ref-fwd` case, and every `refwd/` pass, never tests the hand-off from hipDNN's forward to its backward: the
backward consumes the O and Stats of a reference forward. Only `--pairs` runs a forward engine other than the heuristic's pick into a
backward engine. The recommended run is `--all-engines --pairs`.

Kernel ids are opt-in too. `--kernel-ids` exports `HIPDNN_SDPA_KERNEL_LOG=1` to every pass; the plugin then
installs `hipdnn_sdpa`'s backend log callback and fills `fwd_kernel` / `bwd_kernel`. The callback raises the
backend log level to INFO and sets `HIPDNN_APPLICABILITY_MIN_OFFLOAD_VARIANTS` to `1000000000`, which moves
hip-kernel-provider's applicability check from its worker threads into the calling thread (the callback needs
the GIL the calling thread holds while it waits for them). Engine choice and results are unchanged. By default
the callback is not installed and the kernel columns stay empty (nothing downstream needs them; the engine that
faulted is recorded regardless). `--exhaustive` implies `--kernel-ids`.

`--exhaustive` tests every kernel (candidate) of an engine, not only the one it picks. Candidates come from
`hipdnn_bench enumerate --graph <json> --engine-name <engine> --plugin-dir <prefix>/lib/hipdnn_plugins/engines`
(`<prefix>/bin/hipdnn_bench`, paged, cached per problem and engine; no hipDNN API is added). The run
depends on that executable: the subcommand `enumerate` with `--graph`, `--engine-name`, `--plugin-dir`,
`--offset` and `--limit`, printing a JSON page with `candidates` (each `id` and `knob_settings`),
`total_count` and `next_offset` (null on the last page). `--exhaustive` is refused at start-up when
`<prefix>/bin/hipdnn_bench` is not an executable file; a bench that fails to launch, or one that was
killed, ran out of its 600 s or exited without a message is an error, not a `not-enumerable` row. Pass 1 records,
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
sum. `--exhaustive` composes with `--suite`, `--shard`, `--release`, `--workers` and (forward sweep)
`--fwd-no-stats`, and refuses `--fake` and pre-set engine pins (as does `--all-engines`).
`reconcile.py --engines` reads `cand/` too (`source` `candidate` or `candidate-pair`, plus `fwd_candidate`,
`bwd_candidate`).

`--nodeids-file PATH` limits pass 1 to the listed node ids; the pinned passes keep their own listings.
`--resume` accepts a non-empty `--out` and runs, per pass directory, only the node ids that have no row,
from a node-id file and without `--shard`; a `crash` row counts as a row, so a crash is never run again.
Every link writes its options to `<out>/run-options.json` (`schema`, `hipdnn_prefix`, `all_engines`, `pairs`,
`exhaustive`, `kernel_ids`, `layout_mode`, `fwd_no_stats`, `suite`, `release`, `shard`, `packed_gpu`), and
`--resume` needs the same flags on every link: when they differ from that file in any key but `schema` and `hipdnn_prefix` the runner
exits 2 and lists each differing key as recorded against requested. A shard written before the file existed is
compared on `all_engines`, `pairs` and `exhaustive` only, against the defaults those runs used.
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
`python aotriton/replay.py <repro-dir>` replays one: a repro from a candidate pass rebuilds that pinned candidate (its
`meta.json` entry; a capture-only repro holds the index and needs `HIPDNN_SDPA_PREFIX` to resolve it), not the engine default.
A pinned candidate whose build logged no kernel selection declines naming the kernel log (set `HIPDNN_SDPA_KERNEL_LOG=1`).

Exit codes. `run_aotriton.py`: 0 when every pytest run exited 0; the first nonzero pytest exit status
otherwise (1: tests failed; a crashed worker is a `crash` row, not a runner failure); 2 for a failed
collection, `--fwd-no-stats` meeting non-forward cases, a usage error or a `--resume` refusal; 1 for the other refusals (no hipDNN prefix, no GPU, a
non-empty `--out` without `--resume`, pre-set pins, `--exhaustive` without `hipdnn_bench`).
`completion.py`: 0 once it has listed the shards (an empty list means nothing left to resume), 2 for a missing results
directory. `replay.py`: 0 the child completed; the errors it prints (against the saved hipDNN output and against the FP32 reference) are the verdict (a wrong-result fault such as a NaN output reproduces with exit 0). 1 reproduced: the child died with a signal at any point, including during `Handle()` or setup, or failed (an exception or a nonzero exit) after the failing direction's graph began executing (bad execute status, HIP error at synchronise, the CPU copy); the parent prints `reproduced: exited with status N` or `reproduced: died with signal N`, and a signal death is exit 1 whatever the marker. 70 a setup error: an exception or nonzero exit before the failing direction's graph began executing (`from_json`, engine or candidate selection, no HIP device (checked before any child starts with `--device cuda`), a device or handle initialisation exception), including a setup failure (`from_json`, engine or candidate selection) of the second direction of a fwd+bwd repro after the first executed fine.
`reconcile.py`: 0, or 2 for a missing run directory. `sync.py --check`: 1 when it reports a problem.

AOTriton's calibration, backend and suite-narrowing knobs (`USE_ADIFFS_TXT`, `RECORD_ADIFFS_TO`, `SKIP_BWD`,
`FWD_IMPL`, `BWD_IMPL`, `AOTRITON_UT_ARCH`, ...) are refused; `FOR_RELEASE`, `SMALL_VRAM` and
`AOTRITON_REF_DEVICE_OPTION` pass through.

Outputs, in `<out>/` (default `./sdpa_conformance_results/<arch>/<run>/shard-<I>of<N>/`):
`collected.txt` (every collected node id) and `results.csv` (`nodeid,outcome,mode,fwd,bwd,reason,
fwd_engines,bwd_engines,fwd_kernel,bwd_kernel,fail_side,stage,repro,fwd_candidate,bwd_candidate`, flushed per row; the engine
lists hold the applicable engines in heuristic rank order, `;`-joined, for a direction hipDNN planned
unpinned; `*_kernel` is the ingestor kernel id; `fail_side` is `fwd` or `bwd` for a `fail` (the direction whose
hipDNN call raised; else, for a backward case whose backward ran on hipDNN while its forward did not (reference
forward, or none), `bwd` for any failure, the vendored forward comparison included, since it runs after the
backward and a hipDNN backward can corrupt O; else the vendored forward comparison, a forward case or a backward
that never ran count as `fwd`) and the direction in flight for a `crash`; `stage` is `build` or `execute`; `repro`
is the repro directory under `<out>`).
With `--all-engines`, each pass writes the same files under `all-engines/`. After a run, or a killed one:

```
python aotriton/reconcile.py <arch>/<run> --out <arch>/<run>/reconciled.csv
```

`<arch>/<run>` holds one `<suite>-<k>of<N>/` directory per shard (`default-3of16`; `shard-<k>of<N>` too).
`reconcile.py <results>/<run> --report [--shards-file FILE] [--baseline FILE]` prints the whole run: per-shard
accounting (`FILE`: `<suite> <k> <N>` per line, so a shard without a directory shows as missing), outcome totals
(with `native-skip by class:` `kernel-limit` (AOTriton kernel limits such as hdim or register pressure, a missing
AITER ASM kernel), `api-rule` (an explicit attn_mask with is_causal), `resource` (low-VRAM and OOM guards) and
`other`, from each skip's reason), passes and per-(direction, engine) counts over every shard (a passing row of pass 1
credits the engines of its `fwd` and `bwd` columns and a passing row of a pair pass both engines it pins; the pinned
and reference-forward passes credit the engine they pin and the discovery pass the backward engine that served; a
failing or crashed row only the failing side; `== engines by mode and
test file` splits them by `mode`, `hipdnn` against `ref-fwd`, and by test file, and `--engines` prints it too), the
engines no case passed with their first decline reasons, every `fail` / `crash` row of every pass with its engine
(the one served on `fail_side`, else the other side's, else `-`), side, stage and repro path, and the failing node
ids new to the run or fixed against `--baseline` (one node id per line): "no longer failing" counts only the
baseline ids whose outcome here is `pass`; every other id (`declined`, `native-skip`, `no-hipdnn`, `not-run`, not in
this run) is listed under "not confirmed fixed" with its outcome. `reconcile.py <run> --compare <other run>` (the
earlier run) compares the two runs case by case: the cases in both and only in each, every outcome transition with its
count and example node ids (`pass -> declined` and `fail -> declined` included), and the cases whose served engine
changed with the same outcome. A `== duplicate node ids` section follows the shard accounting:
every node id recorded by more than one shard directory, with the directories (the first row is kept; a run directory
holds each case once), or `none`. `completion.py <results>/<run> [--shards-file FILE]` lists the
shards whose `--resume` still has work (`<suite> <k> <N>` lines on stdout, reasons on stderr).

`mode` is `hipdnn` or `ref-fwd` (hipDNN forward declined; a reference forward fed the hipDNN
backward). A `ref-fwd` pass proves the backward only.

A case that fails with an out-of-memory error (`torch.OutOfMemoryError`, or a `RuntimeError` whose text contains
`out of memory` or `hipErrorOutOfMemory`) before any hipDNN call ran is recorded as `not-run` with reason
`out of memory: <message>`, not `fail`; pytest still reports the failure. An out-of-memory error after a hipDNN call
ran stays what the case would have been otherwise (a declined case stays declined; a hipDNN call that ran and failed stays `fail`). `--resume` does not rerun a `not-run` row.

Two ways the shim differs from AOTriton's own `aotriton_flash.py`: with dropout, `encoded_softmax` comes from this
tool's own Philox mask (`hipdnn_sdpa.philox_mask`), not from an AOTriton kernel; and the stubbed
`pyaotriton.hipGetLastError` always returns `hipSuccess`, so the vendored post-call error check and its
illegal-address handlers never fire.

When a hipDNN call raises anything but a decline, the row is a `fail` (never `declined`; mode `hipdnn`, or `ref-fwd`
behind a reference forward), `fail_side` is the direction that raised, its `fwd`/`bwd` column reads `error:<engine>`
(counted as that engine's failure like `hipdnn:<engine>`), and the failing call's repro is saved. A repro that cannot be
saved leaves `repro` empty and appends `; repro not saved: <error>` to `reason`; the report goes on.
`HIPDNN_SDPA_KERNEL_LOG=1` (set for `--kernel-ids` and `--exhaustive`) is a usage error (exit 4) when the backend lacks the
log exports.

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
