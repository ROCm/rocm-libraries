# In-kernel collectives over IPC peer mailboxes (gfx950)

A multi-process (one process per GPU) test suite and microbenchmark of the
rocKE in-kernel CCL: tensor-parallel all-reduce, all-gather and the
vocab-parallel argmax merge, implemented as rocKE kernels that push into
every rank's receive arena through 64-bit global addresses and poll only
local memory.

| piece | where |
|---|---|
| host runtime (IPC arenas, peer table, reset, rendezvous, error record) | `rocke.runtime.peer` (`PeerGroup`, `P2PGroup`, `topology`) |
| device emitters, kernel builders, CPU references, slot-reuse checker, peer-address lint | `library/kernels/gfx950/qwen38_mono/ccl/` |
| per-rank driver | `harness.py` |
| suite + benchmark CLI | `bench_ccl.py` (suite: 17 tests + 3 CCL-LM tests), `bench_ccl_lm.py` (CCL-LM bench / diag / link probe) |
| measured scope, method, ratios | `ccl_case_study.md` |
| CPU tests | `platform/tests/test_peer_runtime.py`, `library/tests/test_qwen38_ccl.py` |

```bash
export PYTHONPATH=<rocke>/platform/python:<rocke>/library
export ROCKE_HIP_LIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so"))')
python -m torch.distributed.run --standalone --nproc_per_node 4 -m rocke.examples.gfx950.ccl.bench_ccl test  --out DIR
python -m torch.distributed.run --standalone --nproc_per_node 4 -m rocke.examples.gfx950.ccl.bench_ccl bench --out DIR
```

`CCL_MEM=uncached|fine|coarse` picks the arena memory, `CCL_DEVICE_SHIFT=k`
maps rank r to GPU (r + k) mod N (the permutation run). Run only on GPUs no
other process uses: a persistent or polling kernel that shares its GPU can
stall until its deadline.

## Protocol in one paragraph

Unit `u` (1 KiB: one 16-byte store per lane of a wave) of an exchange belongs
to global wave `u mod (G W)` on every rank. That wave pushes its unit into
every rank's arena (`global_store_dwordx4 ... sc0 sc1`, `saddr` = the peer
base), then, in the `line` form, drains and stores the use's tag into the
unit's flag on every rank (one per-lane 64-bit address store: lane k writes
rank k mod N); in the `pair` form the payload carries the tag in every 8-byte
pair. It then polls its own arena (one lane per source for flags), loads the
N parts and adds them in f32 in rank order, so every rank holds the same bits
and a numpy loop reproduces them. Tags are `epoch (+) use` in 1 .. 2^32 - 1,
slots `(slotbase + use) mod S`; the last CTA of a launch advances the epoch
and slot base in the arena, so launches need no host reset and replay under
HIP graphs. Every poll has a deadline (`s_memrealtime`) and loads the local
abort word each round; the first failing CTA writes a host-mapped error
record and pushes ABORT to every rank. A wave that records an error marks
itself dead (one LDS word per wave) and never pushes or publishes again in
that launch, so any valid-tagged word a receiver accepts was pushed by a wave
whose inputs were good; the abort word only makes peers fail fast. Two-shot:
the partial of unit `u` goes to its owner `u mod N` only, the owner sums in
rank order, rounds to the wire dtype and pushes the result to every rank (an
owner whose reduce-scatter poll failed does not push). With the launch header
(on by default) use 0 of every launch exchanges (rows, cols, dtype, uses,
schedule hash): ranks launched with different shapes record `CCL_MISMATCH`
instead of finishing with wrong data.

## Knobs (`CclKnobs`, all built; the kernel name names non-defaults)

`form` (`line` default, `pair`), `ar` (`oneshot` default, `twoshot`),
`slots` (2 default; 2..8), `homing` (`single` default, `dual`: two flag
copies 8 KiB apart, poll the IOD-local one; `flips` kernarg from a
calibration), `push_cpol` (`sc0sc1` default, `sc0sc1nt`), `pause` (`s_nop`
default, `s_sleep1`, `none`), `header` (on by default: launch-header
exchange, a mismatched launch becomes `CCL_MISMATCH`; off only in
microbenchmarks, named `nohdr`), `hooks` (test build: drop a use, abort at a
use, delay injection), `mutant` (must-fail mutants, tests only: `ge`, `any`
accept wrong tags, `slot`, `revsum`, `deadpush` lets a failed wave keep
publishing), `waves` per CTA, `carry` (`auto` default: the all-reduce chain
carries its next input in registers when every wave owns one unit; `mem`),
`poll_cpol` (`sc0sc1` default, `sc1`, `none` = the coherence negative
control; both non-default values fail the lint unless waived), `sync`
(`none` default, `grid`: a grid-wide barrier after every collective of the
chain, the shape of a layer that needs the whole vector).
Large-message (CCL-LM) knobs, default off (`ccl_case_study.md`, "CCL-LM"):
`form="lamport"` (payload-only lines: the consumer polls the data itself,
empty = zero, the producer stores the complement of its canonicalised
data, the consumer zeroes what it read; mutants `noreset`, `nocanon`),
`form="pull"` (stage locally + flag, consumers read every rank's partial
through the peer bases; mutant `slot`), `unit_kib` (1 / 2 / 4 KiB units),
`ctas` (grid cap), `publish` (`counted`, `comm_wave`), `self_local`,
`ar_dtype="bf16"` (bf16 partials of f32 data), `gsync` (`stripe`: 64
agent-scope stripes in device memory, kernarg `gsb`; mutant `gsearly`),
`place="table"` (per-XCD unit table, kernarg `utab`), `pause` (+ `s_sleep4`,
`s_sleep16`), `diag` (per-CTA phase stamps). `layout.lm_select(ranks,
payload_bytes, dtype)` / `auto_allreduce_spec(..., lm=True)` return the
measured CCL-LM choice for TP2 from 80 KiB per rank (W3's selection
elsewhere); `lm_select(..., layer=True)` returns the layer shape with each
band's measured grid sync. `bench_ccl_lm.py`: `bench` (levers vs W3 forms,
RCCL, external arms), `diag` (phase stamps), `linkbw` (push / pull probe per
XCD subset).
- **Numerics gate.** The bench gate runs the timed chain with non-contracting
  scales (`--chain-growth`, default `ref.LM_GROWTH`; 1 = W3's chain, which
  reaches a bitwise fixed point and cannot see a stale-slot read).
- **Must-fail twins.** Each size also runs the must-fail stale-read twins
  (`MUSTFAIL`, `--no-mustfail`); a missed twin exits with rc 4.
- **Seed and per-round data.** `--seed` sets the arm-order seed (fresh per run
  by default). Each row records the seed, the round times (`round_t`) and every
  rank's error record for a dropped arm.
`layout.select_knobs(ranks, payload_bytes, calibrated=..., shape=...)` returns
the measured best form / ar / homing (`ccl_case_study.md`): `shape="layer"`
(default) from the grid-synced chain, `shape="chain"` from the best-case chain. Public builders take
`(spec, *, arch)`; `*_kernel_in(spec, lay, ex_index)` places a collective in a
shared multi-collective arena map (`layout.make_layout([...])`).

## Known limits

* A collective's layout owns the arena regions it uses; switching between
  collectives with different layouts on one arena needs a quiescent point
  (synchronize + barrier on every rank), as the suite and the benchmark do
  between arms.
* `TAG_AHEAD` detection uses modular u32 differences, so a tag that is ahead
  across the 2^32 wrap is reported as a timeout instead; acceptance is exact
  match and is not affected.
* The NUMA `dual` homing knob needs a per-page calibration of remotely
  written arenas before it can pick the IOD-local copy (`flips`); without it
  copy 0 is polled (correct, not necessarily local).
* Compiled with the Python engine (`collectives.backend_for`); the builds use
  inline asm and plain IR only.
* The lint (`ccl/lint.py`, run on every harness build) rejects peer accesses
  without system scope, peer atomics / buffer ops / scalar loads, and
  receive-arena (`arena`) loads or stores without `sc0 sc1`; the harness
  waives only the last rule, and only for the declared `poll_cpol` lever
  builds.
* The benchmark's in-kernel chain without `sync="grid"` is a best case (each
  wave waits only for its own unit); read the grid-synced and the 32-CTA
  arms for layer-shaped costs (`ccl_case_study.md`).
