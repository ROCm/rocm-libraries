# Case study: mailbox hop latency on gfx950 (relative results)

Scope: the cross-CTA hand-off of a resident-grid kernel on one gfx950 GPU,
measured with the microkernels in this folder: a 1-hop ping-pong between two
CTAs, a 160-producer all-to-all round on a full 256 x 512 grid, and the
launch gap of a 256 x 512 kernel. Baseline: the tagged-pair mailbox of the
public FlyDSL-based ATOM decode monokernel (`put`, `poll_raw`: sc1 stores and
loads, `s_nop` pause; the FlyDSL wheel of the serving container), built as
the same microkernels and run in the same process.

Method: device-side `s_memrealtime` tick differences (one clock for every
XCD); ping-pong 20000 round trips per launch; all-to-all 300-400 rounds per
launch; arms interleaved in rotating order within one job (ABAB), medians of
4-6 rounds; every cell passes its numerics gate first (payloads of every
hand-off, stamp order across XCDs, all-to-all per-thread payload sums against
a host model). Diagnostic microbenchmarks, not a serving result. Absolute
values are kept in an access-controlled internal record.

## Results (ratios; < 1 is faster)

| measurement | ratio |
|---|---|
| ping-pong, rocKE `ir` impl vs FlyDSL, same mailbox address, per CTA distance | 0.99 to 1.10 (geomean per distance) |
| ping-pong, rocKE `asm` impl vs FlyDSL | 1.01 to 1.19 (same-XCD pairs worst) |
| ping-pong, mailbox homed at the pair's IOD vs the other IOD (same-IOD pairs) | about 0.6 to 0.75 |
| ping-pong, cross-IOD pairs, either home | no material difference |
| all-to-all `wave0` (FlyDSL shape), rocKE `asm` / `ir` (bare barrier) vs FlyDSL (`gpu.barrier()`) | 0.94 / 0.98 |
| all-to-all `waves` (one wave per 64 producers), `asm` vs FlyDSL | 0.77 to 0.80 |
| all-to-all NUMA dual copy, uncalibrated, best cell vs FlyDSL single copy (layout change, see below) | 0.53 to 0.61 |
| all-to-all NUMA dual copy, calibrated, best cell (`ir`, `wave0`) vs FlyDSL single copy (layout change, see below) | 0.44 to 0.45 |
| all-to-all `put_nt` (nt bit on mailbox stores) vs default | 1.4 to 1.75 (losing lever, kept default-off) |
| all-to-all `scope=system` (sc0 sc1) vs `agent` | 0.98 to 1.04 |
| all-to-all `s_sleep 1` pause vs `s_nop` | 1.00 to 1.04 |
| all-to-all unthrottled (`all` waves poll) vs `wave0` | 1.37 to 1.56 |
| FlyDSL all-to-all round vs hipGraph launch gap of a 256 x 512 kernel | 2.0 |
| rocKE calibrated dual-copy all-to-all round vs the same launch gap | 0.88 to 0.90 |
| hipGraph launch gap vs plain-stream launch gap | about 0.24 |

### Like-for-like re-measurement (review fix)

The first head-to-head compared rocKE's all-to-all (bare `s_barrier` per
round) with a FlyDSL arm that uses `gpu.barrier()`, and credited the NUMA
dual-copy layout, which FlyDSL could use as well, to rocKE. A second job
adds a `barrier="sync"` rocKE arm (`b.sync()`, the `gpu.barrier()`
analogue) and a FlyDSL arm with the same dual-copy layout and calibrated
flips (`fly2cal`); every arm, FlyDSL included, passes a per-lane payload-sum
check first. Ratios, medians of 6 interleaved rounds:

| measurement | ratio |
|---|---|
| FlyDSL dual copy (calibrated) vs FlyDSL single copy | 0.53 |
| rocKE dual copy (calibrated), best cell (`ir`, `wave0`, `sync`) vs FlyDSL dual copy | 0.875 |
| rocKE dual copy (calibrated), other cells vs FlyDSL dual copy | 0.90 to 0.99 |
| rocKE single copy `asm` `wave0`: `sync` barrier vs bare barrier | 0.85 |
| rocKE single copy vs FlyDSL single copy, all cells | 0.83 to 1.05 |

* The layout is the large lever (about 0.53 for either code base); rocKE's
  code is worth 0.875 to 0.99 on top of the same layout. ATOM can adopt the
  dual-copy layout directly.
* In this idle-GPU microbenchmark the per-round `b.sync()` is faster than the
  bare barrier for the single-copy `wave0` shape. A real layer kernel cannot
  drain `vmcnt` there (in-flight weight prefetch), so the bare barrier stays
  the default and `sync` is a kept, default-off knob of the microkernel.
* Single-copy ratios depend on where the mailbox lands (the IOD home of its
  8 KiB block): the same rocKE cell measured 0.93 and 1.01 against FlyDSL in
  two jobs with different allocations. Only home-normalized (dual-copy,
  calibrated) comparisons are like-for-like.

## Findings

* The compiler-visible poll (`tile.tagged_poll`, Python engine only) matches
  the FlyDSL poll loop; the self-contained asm poll (both engines) costs up
  to about a fifth more on same-XCD hand-offs (opaque loop, 64-bit shift +
  compare per pair).
* Memory home matters more than code: gfx950 homes each 8 KiB block at one
  of the two IODs (XCDs 0-3 / 4-7) by address bit 13 XOR a constant per
  2 MiB page (one constant fits 97.6% of 384 sampled blocks; the rest is
  measurement noise at the threshold). A hand-off between CTAs of one IOD is
  markedly faster when its mailbox is homed there.
* Dual copies exploit it for fan-in: every producer stores its pair twice
  (8 KiB apart, so the copies have different homes) and every consumer polls
  the copy homed at its IOD. A per-page `flip` bit, calibrated once on the
  host with a same-XCD ping-pong, tells consumers which copy is local.
* With dual copies the 160-producer all-to-all round drops to about the
  hipGraph launch gap (rocKE slightly below it, FlyDSL with the same layout
  about at it), so a multi-layer persistent launch saves time per layer
  boundary directly only with this layout; with the single-copy layout it
  costs about twice the gap.

## Reproduce

```bash
python -m rocke.examples.gfx950.persistent_mailbox.bench_mailbox_rtt all --out <dir>
```

The FlyDSL arms and the interleaved head-to-head drivers live outside this
repository (they import FlyDSL and ATOM).

## Correctness evidence

* Existing emitted IR is unchanged: `tests/core/test_ir_golden.py` (IR
  sha256 goldens) passes on the tree with these helpers. The new ops are
  additive (new opcodes and new helper modules only).
* `tools/check_byte_identity.py` (GREEN at llvm22 and llvm23) proves the two
  engines agree; the helper-built mailbox kernels are additionally lowered
  with `backend="both"` in `tests/test_mailbox_helpers.py`.

## Caveats

* Microbenchmarks on an otherwise idle GPU; a real layer kernel adds weight
  traffic that competes with mailbox traffic.
* The address-home rule and the calibration are gfx950 measurements on one
  node; re-calibrate per allocation (the per-page constant depends on the
  physical pages).
* The launch gap is that of a near-empty kernel; a kernel with a large
  argument block may launch more slowly.
