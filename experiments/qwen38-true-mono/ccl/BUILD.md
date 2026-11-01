# CCL build record: IPC peer mailboxes + in-kernel collectives (WP-C0/C1/C2/C3, WP-C4 part)

W3, 2026-10-30. Design: `../DESIGN-CCL.md`. Ratios only; absolute device times are in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/ccl-results.md` (+ raw JSON under `ccl-raw/`).
Abbreviations as the design: `R` = `copy-rocm-lib/dnn-providers/hip-kernel-provider/rocke`, `P` =
`R/platform/python/rocke`, `Q` = `R/library/kernels/gfx950/qwen38_mono`.

## What was built

| piece | path | notes |
|---|---|---|
| host runtime | `P/runtime/peer.py` | `PeerGroup` (one process per GPU: HIP IPC of an `uncached` / `fine` / `coarse` arena, 2 MiB pages; peer bases; host-mapped error record; collective `reset(epoch)` / `close`; `file:` or gloo rendezvous; fake-allocator mode for CPU tests), `P2PGroup` (one process, several devices, `hipDeviceEnablePeerAccess`), `topology()`, `tag_add` host twin. Host-only ctypes, reuses `hip_module`'s bound functions; nothing imports it. |
| arena map, knobs, specs, errors | `Q/ccl/layout.py` | `CclKnobs` (form, ar, slots, homing, push_cpol, poll_cpol, pause, header, hooks, mutant, waves, carry), `AllReduceSpec` / `AllGatherSpec` / `ArgmaxSpec`, `make_layout` (multi-collective arena map, flags in one 2 MiB page, dual copies 8 KiB apart), `select_knobs` / `auto_allreduce_spec` (the measured crossover). |
| device emitters | `Q/ccl/wire.py` | asm only: `saddr` 16-byte pushes to peer bases (`sc0 sc1`, `nt` knob) ending in the store-data hazard pad, per-lane 64-bit flag stores (lane k -> rank k mod N, copy k div N), bounded flag poll and tagged-pair poll (deadline every 16 rounds, abort word every round, TAG_AHEAD detection), tag arithmetic with the 0-skip at the u32 wrap. |
| collectives | `Q/ccl/collectives.py` | one-shot and two-shot all-reduce (line / pair), all-gather (line / pair), vocab-parallel argmax merge (lower global id wins ties); rank-ordered f32 sums; register- or memory-carried chains; epoch + slot base + launch count advanced by the last CTA (no host reset between launches, graph-replay safe); launch header (CCL_MISMATCH); first-error record + ABORT push; test hooks (drop a use, abort at a use, delay injection) and must-fail mutants. Builders `build_*_kernel(spec, *, arch)` (contract-clean); `*_kernel_in(spec, lay, ex_index)` for shared arena maps. |
| CPU references | `Q/ccl/ref.py` | bit-exact rank-order sums, bf16 RNE, all-gather rotation, argmax / LSE / top-k merges, delay-hash twin. |
| slot-reuse checker | `Q/ccl/war.py` | happens-before over program order, launch boundaries and the exchange edges; schedules for one-shot / two-shot / any spec; mutants (consumer off-chain, S=1, slots restarting per launch) fail. |
| peer-address lint | `Q/ccl/lint.py` | taints `peer*` / `ctrl`; rejects atomics (IR or asm), buffer descriptors / ops, scalar loads, plain IR accesses and asm loads/stores without `sc0 sc1`. Every build passes; synthetic violations are caught. Run on every build by the harness. |
| suite + microbenchmark | `P/examples/gfx950/ccl/` (`harness.py`, `bench_ccl.py`, `README.md`, `ccl_case_study.md`) | torchrun, one process per GPU, bitwise checks against `ref`, interleaved shuffled timing vs RCCL. |
| CPU tests | `R/library/tests/test_qwen38_ccl.py` (133), `R/platform/tests/test_peer_runtime.py` (8) | layout, names, WAR + mutants, lint (clean + 6 violation kinds), references, LLVM text of every knob, kernarg packing, auto selection, helper lowering unchanged by the import; rendezvous with two local processes. |
| docs | `R/platform/dsl_docs/runtime/limitations.md` ("Multi-GPU and RCCL" update), `Q/README.md` (`ccl/` row), `R/library/tests/test_builder_signature_contract.py` (the four CCL spec classes listed) | |
| harness (jobs) | `hackathon-prep/qwen38/truemono-lb/ccl-h/` (`job.sh`, `job_ft.sh`, `diag_ts.py`), `.../ccl/` (`compile_check.py`, `mk_tables.py`) | |

Byte identity: no tracked rocKE source file was modified (only new files, plus the two docs and
the contract test's spec list), `rocke.helpers` and the lowering are untouched, and the targeted
tests (`test_mailbox_ir`, `test_qwen38_layer_framework`, `test_mailbox_helpers`) pass in every job;
their C++-engine byte-identity cases skip there (no `rocke_engine` in the venv) and are unaffected by
construction.

## Tests (all on gfx950, `rocke` partition)

GPU suite (`bench_ccl.py test`), 16 tests, final runs: **16/16 at TP2, TP3, TP4 (uncached), TP4
coarse, TP2 fine-grained and coarse** (jobs 68218458, 68218774, 68218863 on the final code); the 15-test form also passed with fine,
coarse and the GPU permutation (rank r on GPU r+1) (job 68218378).

| test | what | result |
|---|---|---|
| single_rank | N=1 group: push to self, poll, sum of one == input, 4 forms x 2 dtypes | pass |
| bitwise_allreduce | every rank bitwise == numpy rank-order reference AND all ranks equal; one-/two-shot x line/pair x {2560, 10240} x {bf16, f32} x rows {1, 3, 5, 17, 127, 128}, 3 chained collectives x 2 launches without reset (96 builds per N) | pass |
| bitwise_allreduce_knobs | dual homing, slots=3, nt + s_sleep pause, header, carry=mem (3 forms), waves=4 | pass |
| bitwise_allgather | line / pair, 4 shapes | pass |
| argmax_merge_ties | planted cross-rank ties, lower global id wins, 4 row counts x 5 collectives | pass |
| consecutive_launches_skewed | 1,000 launches, random 0-20 ms host sleep per rank before each launch, no reset | pass |
| tag_wrap | epoch 2^32-301, 50 launches x 8 uses, line / pair / two-shot: all cross the wrap | pass |
| delay_injection_stress | 2,000 + 1,000 launches with random per-wave `s_sleep` (up to 400 x 64 clocks) at every push and poll | pass |
| dropped_peer_times_out | last rank joins the reset, never launches: others record PEER_TIMEOUT / PEER_ABORT within the deadline; reset + next launch correct | pass |
| dropped_use_times_out | rank 1 skips its pushes of use 2: PEER_TIMEOUT at use 2 | pass |
| aborting_peer_propagates | rank 0 aborts at use 1: peers record PEER_ABORT (not timeout), well inside the 5 s deadline | pass |
| launch_mismatch_detected | header build, rank 0 rows 4 vs others rows 5: CCL_MISMATCH on every rank | pass |
| stale_tags | receiver slot pre-filled with tag u-1 / u-S (never accepted: correct result), u+S (TAG_AHEAD); `ge` mutant accepts u+S garbage (must-fail detected), line and pair | pass |
| mutant_must_fail | rank-0 reversed sum (detected at N>=3, non-discriminating at N=2 as expected), wrong payload slot | pass |
| coherence_cache_bits | `poll_cpol=sc1` correct on all memory types; `poll_cpol=none` (plain loads) fails on uncached, fine and coarse arenas: the negative control discriminates | pass |

Bugs found by the suite and fixed:
1. **Inline-asm store-data hazard (gfx950).** A `global_store_dwordx4` inside inline asm followed by
   a compiler VALU write of its data VGPRs (`v_mov_b64` right after the asm) corrupted the high 8
   bytes of the line in flight (two-shot pair form: valid low pair, data in the high tag). LLVM pads
   only stores it can see. Fix: `s_nop 1` after the asm stores. **The shared
   `rocke.helpers.mailbox` asm put of 4 dwords (`put2`, `global_store_dwordx4`) has the same latent
   pattern; not changed here (helpers are byte-identity-gated) -- flagged for its owner.**
2. Layout switch race: two collectives with different arena maps on one arena need a quiescent point;
   the harness now synchronizes + barriers when the layout changes (documented in the README).
3. A failed rank kept pushing valid-tagged garbage: polls now load the abort word every round and
   abort wins over ready data.
4. Windows file-rendezvous race (`os.replace` while a reader holds the file): retry.

## Benchmarks (WP-C4, ratios; < 1 = rocKE faster)

> **Superseded by the Fix round below.** These ratios used RCCL's default protocol only (inflated
> about 3x at 40-160 KiB per rank) and only the best-case chain; the "faster at every size" claim
> below does not hold for the host-launched or the grid-synced arms, nor against aiter.

Method: one process per GPU, rocKE and RCCL (`torch.distributed` nccl) on the same GPUs and streams,
12 shuffled rounds, numerics first (rocKE bitwise vs the reference on every rank, RCCL vs a float64
sum), A/A control 0.95-1.03 in the reported job (median about 1.00). rocKE "chain" = per-collective latency inside one
persistent launch of 96 dependent all-reduces (device stamps); RCCL best = fastest of a HIP graph of
96 calls, the same with a scaling kernel between calls, and eager calls. Full tables:
`P/examples/gfx950/ccl/ccl_case_study.md` (job 68218604; replicated in 68218424 and 68218472).

| TP | rocKE chain / RCCL best (all 32 sizes) | rows 1-16 x 2560 | 128 x 10240 f32 | rocKE one launch per collective / RCCL graph |
|---|---|---|---|---|
| 2 | 0.09-0.95 | 0.09-0.32 | 0.95 | 0.27-1.04 |
| 3 (8 sizes) | 0.09-0.48 | 0.09-0.23 | - | 0.23-0.73 |
| 4 | 0.09-0.84 | 0.09-0.22 | 0.84 | 0.25-1.00 |

* rocKE's in-kernel exchange is faster than RCCL's best host-launched form at every measured size;
  the gap closes at the largest payloads, where both are bound by one xGMI link per GPU pair.
* Measured crossover (encoded in `select_knobs`): TP2 pair one-shot up to 80 KiB, line one-shot
  above (two-shot never wins at TP2); TP3 pair one-shot up to 80 KiB, line one-shot up to 160 KiB,
  line two-shot above; TP4 pair two-shot up to 160 KiB, line two-shot above.
* Levers (all built, default-off; `ccl_case_study.md`): calibrated dual flag homing 0.70-0.92 for
  line forms up to 160 KiB (1.04-1.08 above; turned on by `select_knobs(..., calibrated=True)` there);
  memory-carried chain wins for the line chain in the microbench (0.87-1.00) but loses for pairs;
  `poll_cpol=sc1` neutral; dual + memory carry loses.
* Absolute small-message latency moved by up to about 1.3x between jobs on different GPU pairs
  (same node type); all decisions use same-job ratios.

FlyTape exchange (same job, same node and GPUs, container: FlyTape `scripts/bench_tp.py` toy at
hidden 2560 f32, 200 steps; its `dense.reduce` phase includes the next layer's grid sync), job
68218583:

| | TP2 B1 | TP2 B4 | TP4 B1 | TP4 B4 |
|---|---:|---:|---:|---:|
| rocKE best chain per all-reduce (2560 f32) / FlyTape reduce phase | 0.64 | 0.72 | 0.91 | 0.94 |

Caveat: not the same quantity. FlyTape's phase is the wait + rank-order sum + the next grid sync,
with its pushes overlapped into the preceding GEMV phase; rocKE's number is a full dependent round
trip (push, flight, poll, sum). The in-situ GEMM + all-reduce comparison that decides the large-batch
budgets is WP-C5 (not run yet).

## Topology facts (WP-C0, also in the alola-cluster skill, rule 135)

Every GPU pair: XGMI, 1 hop, weight 15, coherent, 64/32-bit atomics (`amd-smi topology`);
`hipDeviceCanAccessPeer` all 1; IPC works across a job's GPUs in venv mode (torch 2.12 + ROCm 7.2) and in
the q38 container (torch 2.13 + ROCm 10); both nodes. Dual-homing calibration found per-rank page flips
of both values. H7 (buffer descriptors on peer pointers) was not tested: the CCL never uses them.

## Not built / deviations from DESIGN-CCL (honest list)

> Updated in the Fix round below (launch header now on by default; aiter arm measured).

* Wire form `lamport` (sentinel payloads) and publish variants `comm_wave` / `counted`: not built
  (only `unit_end`-style publish: drain, then flag).
* `ar_dtype` (bf16 partials of f32 data, fp8): the wire dtype is the data dtype (bf16 or f32);
  `self=local`, `poll_width=16`, `poll_wave=one`: not built.
* LSE and top-k merges: CPU references only (`ref.py`); the device merge is argmax.
* Cancel word, torn-pair checksum detector, 10k-launch stress (3k run), TP8 (the job wrapper caps at 7
  GPUs), aiter custom all-reduce arm, `P2PGroup` on hardware (CPU-tested API only), a watchdog thread
  (the harness polls a completion event instead): not done.
* The WAR checker runs on the microkernel schedules; the LB `TaskGraph` hookup is WP-C7.
* WP-C5 (in-situ GEMM + all-reduce) and WP-C6 timing of memory types / `nt` / pause / slots: not run.
* Default `homing` stays `single` (dual needs the arena calibration; `select_knobs(calibrated=True)`).
* The persistent-mailbox harness's `calibrate_iod_home` is reused for the receive arena (it overwrites
  arena words; `Rank.calibrate_flips` resets the group afterwards).

## Jobs

smoke 68218023; suites 68218119, 68218194, 68218261, 68218378, 68218458, 68218774, 68218863 (final code); benchmarks 68218133,
68218217, 68218424, 68218472, 68218604; container (FlyTape + rocKE) 68218463, 68218583. All on the `rocke`
partition (`smci355-ccs-aus-n01-09`, `n07-17`); per-row node and bus ids are in the raw JSON.

## Verification (independent W3 verifier, 2026-10-30)

Verdict: **partial**. Correctness, multi-GPU operation, byte identity and the protocol hold up. The RCCL speed-up figures
for small messages are overstated, because the RCCL baseline ran with RCCL's default protocol choice, which is slow
at 40-160 KiB. Raw JSON: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/ccl-raw/68219011-verify/`.

My re-run, job 68219011 (n01-09, 2 GPUs, buses 15 / 65, `amd-smi process` empty on both). It used stages `w3-vccl-rocke`
(the current tree, synced after the last edit) and `w3-vccl-h` (`hackathon-prep/qwen38/truemono-lb/ccl-v/`):
- CPU tests: 133 + 8 pass.
- `bench_ccl test`, 16/16 at TP2 with the GPU permutation (`CCL_DEVICE_SHIFT=1`, uncached) and 16/16 at TP2
  fine-grained, both on the final code.
- `vcheck.py`, an independent reference written from the spec rather than from `ccl/ref.py`. It covers one-shot and
  two-shot, line and pair, memory carry and the header build, at rows 1 / 37 / 128 x 2560 / 10240 x bf16 / f32, with 1
  or 5 chained collectives and 3 launches each without a reset. `dst` is zeroed before every launch, and a perturbed
  reference must mismatch. Result: 60/60 cases bitwise on both ranks, on uncached memory and on coarse memory with the
  GPUs permuted. Ranks hold identical bits. The device epoch ends at exactly 1 + the uses (571).
- RCCL protocol probe: the same bench at TP2, rows 2-64 x 2560 bf16, under the default protocol and with
  `NCCL_PROTO=Simple / LL / LL128`.

What I confirmed:
- **Separate GPUs.** Every suite log (68218774, 68218863, 68219011) prints a distinct PCI bus per rank, and torch maps
  each rank to its own device.
- **The negative controls really fail** (checked in the per-rank JSON of 68218458 and 68219011, not just the PASS
  lines):
  - The dropped peer records PEER_TIMEOUT at the 0.5 s deadline.
  - Peers of the aborting rank record PEER_ABORT well inside the 5 s deadline (under 0.1 % of it).
  - The launch mismatch records CCL_MISMATCH on every rank.
  - A tag S ahead raises TAG_AHEAD.
  - The slot mutant is detected on every rank. The reversed-sum mutant is detected at N >= 3; at N = 2 it cannot be
    detected, and the suite says so.
  - `poll_cpol=none` gives 0/5 correct runs on uncached, fine and coarse memory.
- **Byte identity.** `git diff --stat` (read-only) shows 3 tracked files changed with 16 added lines: the README row,
  `limitations.md` and the contract-test spec list. Every other rocKE file is new and imported by nothing existing.
- **Memory ordering**, checked in the gfx950 ISA of `q38ccl_ar_n2_r4_c2560_bf16{,_pair_ts}`:
  - Each push is a `global_store_dwordx4 ... sc0 sc1` to every peer base, then `s_nop 1` (2 wait states, the gfx950
    VMEM-store-data hazard), then `s_waitcnt vmcnt(0)`, and only then the `global_store_dword ... off sc0 sc1` flag
    stores.
  - Polls load `sc0 sc1`, check the abort word every round and the deadline every 16 rounds. Payload loads issue only
    after the poll's `vmcnt(0)`.
  - The tag arithmetic is one's-complement addition (associative, never 0), so tags across launches and across the
    wrap equal the host twin.
  - The last CTA advances the epoch and slot base only after every CTA has read them.
  - The S = 2 write-after-read argument holds for one-shot and two-shot, across launches too, because the wave that
    owns a unit is the same on every rank.
  - `reset()` orders synchronize, barrier, memset, synchronize, barrier.
- **64-bit global peer pointers.** The ISA has no `buffer_*` instructions. Peers are reached as `saddr` SGPR pairs
  (`s[8:11]` = `peer0..1`), and atomics touch only the own `arena` kernarg (`s[14:15]`). The lint runs on every
  harness build.
- **The `rocke.helpers.mailbox` hazard report is plausible.** Its asm put is `global_store_dwordx{n}` with no pad (and
  no memory clobber).

Refuted or weakened:
1. **The small-message RCCL ratios are inflated.** The torch 2.12 RCCL default switches TP2 from the LL protocol to
   Simple at 40 KiB and up, and Simple is about 3.4x slower than LL at 40-80 KiB. With `NCCL_PROTO=LL`, rocKE's
   chain / RCCL-best-over-protocols for rows 2-64 x 2560 bf16 is **0.21-0.35** (same job, same GPUs). Under the default
   it is 0.07-0.35. So:
   - "0.09-0.32 for rows 1-16 x 2560" should read about 0.21-0.28.
   - The like-for-like "one launch / RCCL graph" 0.27 values become 0.83-0.88 against RCCL LL (0.72-0.92 over rows
     2-64).
   - TP3 and TP4 have not been re-measured with LL. They should be, and every rocKE-vs-RCCL table should carry an
     `NCCL_PROTO=LL` arm and use the best protocol per size.
   - "Faster than RCCL at every size" holds only for the in-kernel chain metric. The host-launched arm is slower than
     the RCCL graph at 128 x 10240 bf16 / f32 (1.02-1.04).
2. **The chain metric is a best case.** Wave w's next collective waits only for its own unit from every rank; it has
   no grid-wide dependency like a layer that needs the whole vector. It also uses up to 256 CTAs, the whole GPU.
   Together with the missing aiter custom all-reduce arm (ATOM's real TP path), this means WP-C5 still decides whether
   rocKE wins.
3. **`coherence_cache_bits` always logs ok=True** (`bench_ccl.py`), so its PASS line is vacuous; the discrimination is
   visible only in the JSON fields. It should assert `sc0sc1` and `sc1` correct and `none` discriminating.
4. **The stale-tag must-fail is weak.** "ge plusS" passes whether the mutant accepts garbage or TAG_AHEAD fires (both
   happened). No accept-any-tag mutant proves that the u-1 / u-S cases would catch an acceptance bug.
5. **Provenance.** 68218774 (TP2 / TP3 uncached) ran a stage synced at 09:16, before the 09:17 rename patch.
   68218458 (TP2 fine, TP4 coarse) ran even earlier code. The final code is covered by 68218863 (TP4 uncached, TP2
   coarse) and 68219011 (above). TP3 has not run on the final code; the diff is rename-only.
6. **`CclKnobs.header` defaults to False**, against DESIGN-CCL 4.10 ("off only for microbenchmarks"), and the
   deviation list does not say so. Without the header, ranks launched with different `rows` but the same `uses` can
   finish silently with wrong data on the rank that polls fewer units; only the other rank times out. WP-C7 should
   make the header the default for model builds.
7. **Lint gap.** The lint taints only `peer*` / `ctrl`, so local polls through `arena` are not checked. A
   `poll_cpol=none` build, which the coherence control shows is broken, passes the lint.
8. **Theoretical abort-vs-data race.** The abort-word load and the data load of one poll round are unordered, so a
   peer could in principle accept a failed rank's valid-tagged data in the same round as a stale abort read. The host
   still sees the failed rank's error record.

## Fix round (W3, 2026-10-30)

Every verifier finding is fixed in code, re-run on the cluster on code synced after the last edit,
and recorded here (ratios only; absolute times in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/ccl-results.md`, raw JSON under
`ccl-raw/{68219496-fix-suite,68219499-fix-bench,68219502-fix-ctr,68220407-fix-suite-final}/`).
Every suite line, benchmark line and JSON row now carries a code digest (sha256 of `ccl/*.py`,
`rocke.runtime.peer`, the harness and the bench), the node and the PCI bus of every rank.

### Jobs (all `rocke` partition, `smci355-ccs-aus-n01-09`, our GPUs idle)

| job | mode | what | code digest | GPUs (bus) |
|---|---|---|---|---|
| 68219496 | venv | CPU tests; 17-test suite at TP2 / TP3 / TP4 uncached, TP4 coarse, TP2 fine, TP2 coarse + permutation, TP3 coarse + permutation | `6f06506ac771` | 65, 15, f5, 95 |
| 68219499 | venv | benchmark TP2 (32 sizes), TP3 (20), TP4 (32) vs RCCL, all four protocols | `6f06506ac771` | 15, f5, 85, e5 |
| 68219502 | container | FlyTape toy; benchmark TP2 / TP4 (bf16, 16 sizes; f32 toy shapes) vs RCCL (four protocols) and aiter custom all-reduce | `dc5fedfab364` | 15, f5, 85, e5 |
| 68220407 | venv | CPU tests; suite at TP2 / TP3 / TP4 uncached, TP4 coarse + permutation, final code | `dc5fedfab364` | 75, 05, 65, 15 |

`6f06506ac771` -> `dc5fedfab364` changed only `layout.select_knobs` (the `shape` argument below)
and its CPU test; the final code passed the full suite at TP2, TP3 and TP4 (68220407).

### Findings and fixes

| # | finding | fix | evidence |
|---|---|---|---|
| 1 | RCCL baseline used the default protocol only | `bench` builds one RCCL communicator per `NCCL_PROTO` (default, LL, LL128, Simple; the variable is set while the communicator is created), each with a captured 96-call graph and eager calls; "RCCL best" = the fastest of all of them; the default-protocol-only ratio stays as a separate column | RCCL logged "NCCL_PROTO set by environment" once per rank per protocol (27 lines at TP2+3+4); the default protocol's cliff shows at TP2, TP3 and TP4 (chain / RCCL default protocol 0.08-0.10 where the best-protocol ratio is 0.19-0.33) |
| 2 | "faster than RCCL at every size" held only for the chain metric | claims restated per arm (below); the one-launch arm and the new layer-shaped arms are reported with their losing sizes | tables below and in `ccl_case_study.md` |
| 3 | the chain is a best case; aiter (ATOM's path) not measured | new knob `sync="grid"` (device code: a grid-wide barrier through an arena counter after every chained collective, GRID_SYNC_TIMEOUT on its deadline, counter cleared by the last CTA; bitwise-tested in the suite on uncached and coarse arenas); benchmark arms: grid-synced chain on all CTAs and on 32 CTAs, the default model build (auto form + header) one launch per collective; aiter `CustomAllreduce` arm (graph and eager) as an external-arm plugin (`--extra-arms`, `hackathon-prep/qwen38/truemono-lb/ccl-f/aiter_arm.py`) in the q38 container | 68219499, 68219502; WP-C5 (in-situ GEMM + all-reduce) still not run |
| 4 | coherence test logged `ok=True` always | the test now asserts: `sc0sc1` and `sc1` correct in 5 of 5 launches on every rank, `none` fails on some rank, and the lint rejects the `sc1` / `none` builds and accepts `sc0sc1` | every run: `none` 0 of 5 correct (timeouts / aborts) on uncached, fine and coarse arenas; `sc1`, `sc0sc1` 5 of 5 |
| 5 | stale-tag must-fail passed on either outcome | the `ge` mutant (accept tag >= expected) has no tag-ahead check, so u + S garbage is accepted with no error, the one allowed outcome; new mutant `any` (accept any non-zero tag) must accept garbage for u - 1, u - S and u + S, proving each stale case discriminates; the real build must give correct data for u - 1 / u - S and TAG_AHEAD for u + S | 18 of 18 rank-0 checks per run with exactly the expected outcome (line and pair) |
| 6 | provenance (TP3 not on the final code; mixed stages) | code digest + node + buses in every line; all runs on stages synced after the last edit | 68220407 (final code) passed TP2, TP3, TP4 and TP4 coarse + permutation |
| 7 | `CclKnobs.header` off by default against DESIGN-CCL 4.10 | default `header=True`; implemented for the all-gather and the argmax merge too (it was all-reduce only); header-off builds are named `nohdr` and used only by the microbenchmark arms; `select_knobs` returns header-on knobs | `launch_mismatch_detected` now covers all three collectives: CCL_MISMATCH on every rank at TP2, TP3, TP4; the four default forms of the bitwise test run with the header |
| 8 | lint did not check local polls through `arena` | receive-arena taint: addresses derived from the `arena` kernarg through address arithmetic (loaded data does not taint); asm loads / stores without `sc0 sc1` and plain IR loads / stores of the arena are rejected, local atomics allowed; `allow_local_scope=True` waives only the scope rule and the harness passes it only for the declared `poll_cpol` lever builds | CPU tests: 5 synthetic arena cases, 4 real lever builds (`sc1` / `none` x line / pair) rejected, every default build clean; the suite asserts the lint verdict on hardware-tested builds |
| 9 | abort-word and data loads of one poll round unordered | closed at the source instead of by ordering loads: a wave that records an error (failed poll or abort hook) sets its own LDS word and never pushes or publishes again in that launch; every push and flag store is guarded (one-shot, two-shot partial and the owner's reduced push, all-gather, argmax). A wave's later inputs come only from units it consumed itself, so any valid-tagged word a receiver accepts was pushed by a wave whose inputs were good; the per-round abort check only makes peers fail fast | new suite test `failed_wave_goes_silent` (one-shot: abort hook at use 1; two-shot: the owner of unit 0 times out because rank 1 drops its partials), line and pair: no rank's arena holds the failed wave's tag of use 1 or 2; must-fail mutant `deadpush` (guard off) publishes on every rank (2 / 3 / 4); ISA: `ds_read_b32` + `v_readfirstlane` before every guarded push |

Also found and fixed in this round: with the launch header off, a grid-sync build whose barrier
target was computed inside the one-lane region did not compile (LLVM moved the loop counter's
chain to VALU, so SGPR operands of the chain's inline asm became VGPRs); the uniform operands are
now computed before the region.

`select_knobs` gained `shape="layer"` (the default; crossover measured on the grid-synced chain)
next to `shape="chain"` (the earlier best-case crossover, unchanged): TP2 pair one-shot up to
80 KiB, line one-shot up to 320 KiB, line two-shot above; TP3 pair one-shot up to 80 KiB, pair
two-shot up to 160 KiB, line two-shot above; TP4 pair one-shot up to 20 KiB, pair two-shot up to
160 KiB, line two-shot above. The container job replicated the TP2 / TP4 layer crossover.

### Tests

* CPU: `test_qwen38_ccl.py` 150 (was 133: arena lint, lever-build lint, header default for all
  three collectives, mutant asm, grid-sync build, layer / chain selection), `test_peer_runtime.py`
  8; in the jobs 330 passed / 10 skipped (CCL + mailbox IR + layer framework) and 69 / 11
  (platform); `test_builder_signature_contract.py` keeps its 3 pre-existing failures (qwen38_mono
  framework / probes, `mx_decode_gemv`), none from the CCL.
* GPU suite, 17 tests (new: `failed_wave_goes_silent`; `launch_mismatch_detected` over three
  collectives; `coherence_cache_bits` and `stale_tags` with single-outcome assertions;
  `bitwise_allreduce_knobs` adds three grid-sync builds and a header-off build; `bitwise_allgather`
  and `argmax_merge_ties` add header-off builds): **17/17 in all 11 runs**: TP2 / TP3 / TP4
  uncached, TP4 coarse, TP2 fine, TP2 coarse + permutation, TP3 coarse + permutation (68219496;
  339 / 500 / 661 rank-checks per TP2 / TP3 / TP4 run), and TP2 / TP3 / TP4 uncached, TP4 coarse
  + permutation on the final code (68220407).

### Benchmarks (ratios; < 1 = rocKE faster; RCCL best = fastest protocol and form per size)

Full tables: `R/platform/python/rocke/examples/gfx950/ccl/ccl_case_study.md`.

| TP | chain / RCCL best (rows 1-16 x 2560) | grid-synced chain / RCCL best | grid-synced, 32 CTAs / RCCL best | 1 launch / RCCL graph best | default model build, 1 launch / RCCL graph best |
|---|---|---|---|---|---|
| 2 (32 sizes) | 0.28-0.95 (0.28-0.33) | 0.27-1.04 | 0.27-1.09 | 0.75-1.04 | 0.81-1.09 |
| 3 (20 sizes) | 0.20-0.71 (0.20-0.22) | 0.20-0.79 | 0.21-0.85 | 0.53-0.79 | 0.59-0.81 |
| 4 (32 sizes) | 0.17-0.85 (0.17-0.23) | 0.21-1.00 | 0.21-1.12 | 0.51-1.01 | 0.51-1.04 |

Against the aiter custom all-reduce (captured graph, ATOM's TP path; container job 68219502;
bf16 rows 1-128 x 2560 / 10240 plus the f32 toy shapes):

| TP | chain / aiter | grid-synced / aiter, up to 80 KiB per rank | grid-synced / aiter, 160 KiB and up | 1 launch / aiter graph | aiter graph / RCCL graph best |
|---|---|---|---|---|---|
| 2 | 0.41-0.98 | 0.51-0.98 | 1.13-1.21 | 1.13-3.20 | 0.36-0.87 |
| 4 | 0.47-0.96 | 0.65-0.99 | 1.13-1.35 (32 CTAs: up to 1.46) | 1.36-2.54 | 0.25-0.66 |

FlyTape toy (same job and GPUs; FlyTape's reduce phase includes the next grid sync, its pushes
overlap the preceding GEMV): rocKE grid-synced chain / FlyTape reduce phase 0.71 (TP2 B1), 1.11
(TP2 B4), 1.11 (TP4 B1), 1.29 (TP4 B4); the best-case chain 0.61 / 0.67 / 0.97 / 0.93.

### Corrected claims

* The in-kernel chain (each wave waits for its own unit) is faster than RCCL's best form at every
  measured size (0.17-0.95), and faster than aiter's custom all-reduce graph at every measured
  size (0.41-0.98).
* With a grid-wide barrier per collective (the shape of a layer that needs the whole vector) it is
  faster than RCCL's best form up to 1.25 MiB per rank (TP3 and TP4: up to 2.5 MiB) and ties or
  loses at 2.5-5 MiB at TP2 (1.00-1.04) and at 5 MiB at TP4 (1.00); against aiter it is faster up to 80 KiB per rank and **slower from 160 KiB
  (1.13-1.35)**, which covers the large-batch decode all-reduces (rows 64-128 x 2560 bf16 =
  320-640 KiB per rank). The large-batch budgets in DESIGN-LB cannot be claimed from these numbers;
  the large-message exchange is the next optimization target, and WP-C5 still decides the in-situ
  cost.
* One launch per collective (host-launched, no graph): 0.51-1.04 of RCCL's best graph (it only
  ties RCCL's LL graph up to 20 KiB at TP2), and 1.13-3.20 of aiter's graph: the host-launched
  path is not competitive with aiter; the CCL is meant to run inside persistent launches.
* The default protocol's cliff (LL -> Simple from 40 KiB at TP2) exists at TP3 and TP4 as well;
  the earlier "0.09-0.32" for rows 1-16 x 2560 is 0.28-0.33 against the best protocol.

### Not built / open (updated)

* Not built: `lamport` wire form, `comm_wave` / `counted` publish, `ar_dtype` narrower than the data,
  `self=local`, `poll_width=16`, `poll_wave=one`, device LSE / top-k merges (CPU references only),
  the cancel word, TP8 (the job wrapper caps at 7 GPUs), `P2PGroup` on hardware.
* Not run: WP-C5 (in-situ GEMM + all-reduce); timing of the memory types, `nt`, pause and slot
  levers; dual homing under the grid sync.
* The large-message layer-shaped all-reduce is slower than aiter's custom all-reduce from 160 KiB
  per rank (above): the first optimization target before WP-C7 wires the CCL into the large-batch
  builds.
* `rocke.helpers.mailbox`'s 4-dword asm put still has the store-data hazard pattern (owner's
  change; helpers are byte-identity-gated).

### Byte identity

Unchanged: no tracked rocKE source file is modified. The tracked diff is 3 files, 21 added lines
(the `ccl/` README row, the `limitations.md` "Multi-GPU and RCCL" update, the four CCL spec
classes in the contract test's list); everything else is new and imported by nothing existing.
`rocke.helpers` and the lowering are untouched.

## CCL-LM follow-up (W4, 2026-10-31)

The large-message loss above (grid-synced, from 160 KiB per rank) is fixed in `../large-batch/CCL-LM.md`: the lamport
and pull forms, wider units, a grid cap and the striped grid sync (all default-off knobs; every build above
byte-identical), with the selection `layout.lm_select` beating the aiter graph at every measured size from 80 KiB to
5 MiB per rank at TP2 (DESIGN 9.1 win rule); the suite grew to 20 tests (17 above + `lm_bitwise`, `lm_canon`,
`lm_mutants`).

**Fix round** (the "Fix round" section of `CCL-LM.md`):
- **Gate.** A stricter numerics gate: a non-contracting chain plus must-fail stale-read twins in every benchmark.
- **Selection.** Three bands changed: 80 KiB uses 4 waves; 1.25 MiB f32 uses the counted publish; above 1.25 MiB, pull
  on 64 CTAs with the word grid sync.
- **Confirmation.** Re-confirmed on both rocke nodes, on the final code and against the per-round best aiter capture.
  The selection wins from 80 KiB to 2.5 MiB. 5 MiB is a link-bound tie.
- **W3 code.** The 2284 W3 builds are still byte-identical. The W3 tests keep W3's contracting chain
  (`Rank.chain_growth = 1`).
