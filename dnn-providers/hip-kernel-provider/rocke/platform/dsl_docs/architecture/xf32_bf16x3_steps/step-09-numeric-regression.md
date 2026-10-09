# Step 9 — what the recipe actually computes

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
step 9). Ticket AICK-2208.

## Contents

- [The gap steps 5–8 leave](#the-gap-steps-58-leave)
- [Two halves, split by what each can prove](#two-halves-split-by-what-each-can-prove)
- [The host half: the arithmetic](#the-host-half-the-arithmetic)
- [The device half: layout-free by construction](#the-device-half-layout-free-by-construction)
- [The on-device run](#the-on-device-run)
- [Verification](#verification)

## The gap steps 5–8 leave

[Steps 5+6](step-05-bf16x3-lowering.md) pin the IR the expansion emits.
[Step 8](step-08-comgr-compile-gate.md) pins that the IR compiles and that the
right three MFMAs and the right convert reach the ISA. Between them they
establish that the recipe is *written* and *buildable* — and nothing at all
about whether it is *worth building*. A recipe with the two residual terms
swapped, or with a truncating split, or with only the `big*big` term surviving
a dead-code pass, would clear every gate so far.

So this step gates the numbers.

## Two halves, split by what each can prove

The two emulated rows are gfx950-only, and the accuracy claims are claims about
arithmetic, not about silicon. Those are different questions with different
reach, so they are two artifacts rather than one:

| question | where | needs a GPU |
|---|---|---|
| Is the arithmetic worth it — vs fp32, vs real TF32, vs plain bf16? | [`tests/core/test_xf32_bf16x3_numerics.py`](../../../tests/core/test_xf32_bf16x3_numerics.py) | no |
| Does the lowering compute that arithmetic on hardware? | [`examples/gfx950/xf32/bf16x3_parity.py`](../../../python/rocke/examples/gfx950/xf32/bf16x3_parity.py), run by [`tests/core/test_xf32_bf16x3_device.py`](../../../tests/core/test_xf32_bf16x3_device.py) | yes, gfx950 |

Putting the accuracy claims on the host side is deliberate: they are the part
of this feature most likely to be quietly wrong, and a check that only runs on
one arch of one vendor's validation node is a check that mostly does not run.

## The host half: the arithmetic

A model of the recipe in plain numpy — no torch, no `ml_dtypes`, RNE bf16 done
by hand in `uint32` so the narrowing matches the packed convert
[step 8](step-08-comgr-compile-gate.md#what-the-disassembly-is-read-for)
pinned rather than the truncating scalar fallback.

Error is normalised by `sum |a_i b_i|`, not by the result. A random length-256
dot product cancels down to a fraction of the terms that went into it, so
dividing by the result reports the luck of that cancellation instead of the
arithmetic under test — an early draft of this test failed for exactly that
reason.

What it pins:

- **The split is an error-free transform.** `big + small` recovers about
  sixteen significand bits against bf16's own eight, and both halves are
  themselves exactly representable in bf16 — they have to be, since the MMA
  will narrow them again.
- **Three terms beat one by two orders of magnitude**, and beat real TF32 by
  one. The emulation is *more* accurate than the thing it substitutes for, so
  a caller swapping TF32 for xf32 never loses by it.
- **It lands near the fp32 accumulation noise floor.** An fp32 MMA is not
  exact either; over K=256 its own rounding is within about one order of
  magnitude of where the emulation sits. That ceiling is why the fourth term
  is not issued — `small*small` is measurably below where a fp32 accumulator
  can carry it.
- **The term order is load-bearing.** fp32 addition is not associative, so a
  permutation is observable; summing smallest-magnitude first is not worse
  than the reverse. Without this, the order in `_expand_emulated_mma` would be
  arbitrary and free to drift.
- **The model's own bf16 is RNE**, tie-to-even both directions, and not
  truncation. If that were wrong every number above would be wrong with it.

This is a model of the arithmetic, not of the hardware: it accumulates over K
in sequential fp32, which is not necessarily an MFMA's internal order. So the
assertions are relative-accuracy properties and the tolerances are set an order
of magnitude off the measured values. Nothing here claims bit equality with a
device — that claim belongs to the device half, below.

## The device half: layout-free by construction

The obvious on-device test — fill A and B, compute a reference GEMM, compare —
needs a host-side model of the MFMA fragment layout. Hand-deriving one and
then testing against it means testing two derivations of the same thing and
believing the agreement. Both checks here avoid needing the layout at all:

1. **Faithfulness.** The emulated `tile.mma` runs against a hand-written
   kernel issuing the same three BF16 MFMAs, in the same order, over operands
   split on the host. Both kernels are fed *identical flat fragment buffers*
   and share the same physical atom, so whatever the lane layout is, both see
   it. The comparison is bitwise — not tolerant — because there is nothing for
   the two to disagree about unless the lowering built something else.

2. **Accuracy.** Each operand is filled with a single constant. Every logical
   element of A is then `alpha` and every logical element of B is `beta`, so
   `D = K * alpha * beta` at every output position *for any layout*. `alpha`
   and `beta` are chosen to round to exactly 1.0 in bf16, which puts the whole
   answer in the two residual terms — the part of the recipe worth gating. The
   same kernel with only `big*big` serves as a negative control: it must be
   badly wrong here, or the tolerance check proves nothing.

The second kernel is reused for both the reference and the control, differing
only in how many terms it issues, so the two cannot drift apart.

## The on-device run

**Measured on a gfx950 device, 2026-10-08.** An earlier revision of this
doc recorded the run as not performed: the development host at the time saw
only gfx942, and the emulated rows are gfx950-only. Both checks now run on
silicon, with `--require-device` so a missing device cannot pass as green:

| atom | expansion vs hand-written BF16x3 | emulated rel err | bf16-only rel err |
|---|---|---|---|
| `xf32_bf16x3_f32_16x16x32` | 0 of 256 slots differ | 1.188e-07 | 2.424e-04 |
| `xf32_bf16x3_f32_32x32x16` | 0 of 1024 slots differ | 1.188e-07 | 2.424e-04 |

Both land three orders inside the `2^-16` tolerance, and the single-term
control misses by more than its `2^-13` floor — so the tolerance check is
being met by the residual terms rather than by bf16 happening to be good
enough. The two atoms agree to the digit because the operands are constant and
the accumulate is exact at both K.

### That the device checks bite

Green on a check that cannot fail is worth nothing, so the lowering was broken
twice — in the Python engine only, in a throwaway copy of the tree — and the
harness re-run on the same device:

| injected fault | result |
|---|---|
| term order rotated to `big*big, small*big, big*small` | 170 / 531 slots differ; exit 2 |
| residual terms dropped, leaving `big*big` | 256 / 1024 slots differ, emulated err rises to the bf16 floor 2.424e-04; exit 2 |

The first is caught by the differential alone and the second by both checks,
which is the split the two were written for: the differential sees *order*, the
absolute check sees *missing terms*. Note the first fault leaves the accuracy
figure untouched — a reordered sum of the same three terms is still accurate —
which is precisely why the bitwise differential exists.

### How the suite reaches it

The harness is an examples-only `__main__` script, so as first written neither
pytest nor `tests/run_all.py` invoked it, and its assertions would not have run
even on a gfx950 host. After review it is run by
[`tests/core/test_xf32_bf16x3_device.py`](../../../tests/core/test_xf32_bf16x3_device.py),
which pytest — and so `run_all.py` and `run_checks.py` — discovers like any
other test:

- every visible device is probed in a subprocess, keeping HIP out of the
  pytest process, and the first gfx950 among them is handed to the harness
  through the visibility env var. Probing device 0 alone would skip a node
  whose gfx950 sits behind another card — which is a silent hole in the one
  gate that touches silicon, not a conservative default;
- on gfx950 the harness runs with `--require-device`, and anything but exit 0
  with `PASSED` fails the test;
- it skips only on a *confirmed* absence, which takes three shapes: the HIP
  library will not load; `hipGetDeviceCount` answers `hipErrorNoDevice`, which
  is a real answer and what a host with ROCm but an unusable `/dev/kfd`
  reports; or the count succeeded and none of the devices it counted is a
  gfx950. Anything else fails — another error code, an import error, or (only
  when no gfx950 was found) a device whose architecture HIP would not name,
  since that device could be the one the gate exists for. Device absence is now
  the only reason it skips at all, the harness needing nothing beyond numpy;
- the probe does **not** go through `get_device_count()` / `get_device_arch()`.
  Both are deliberately forgiving — every failure becomes the `0` / `None` a
  healthy query gives for an empty machine — so routing through them merges two
  of those confirmed absences with every case that must fail, and
  `get_device_count` raises `HipError` for a missing library, the one case that
  should skip quietly. The probe resolves the runtime itself and reads
  `hipGetDeviceCount`'s return code;
- `ROCKE_REQUIRE_GFX950=1` turns even that skip into a failure. That is the
  spelling for a gfx950 validation node, mirroring `ROCKE_REQUIRE_GFX1250`.

The harness originally allocated through torch, which would have skipped the
gate on the gfx950 node it was written for: neither that node nor the ROCm
images on it carry torch. It now uses `DeviceMem` + `Runtime.memcpy_*` over
numpy host arrays — rocke's only hard dependency — with the fp32→bf16 RNE
narrowing hand-rolled in `uint32`, the same way
[`test_xf32_bf16x3_numerics.py`](../../../tests/core/test_xf32_bf16x3_numerics.py)
does it. That the host split matches the ISA's own convert is not assumed: it
is exactly what the bitwise differential above measures.

## Verification

- `tests/core/test_xf32_bf16x3_numerics.py` — 13 tests, green.
- `tests/core` on the gfx950 node — 440 passed, 7 skipped, 799 subtests with
  the `rocke_engine` extension built; 428 / 19 without it (the extra skips are
  all engine- or `ml_dtypes`-gated).
- `examples/gfx950/xf32/bf16x3_parity.py --require-device` on gfx950 —
  `PASSED`, with the figures tabulated above, in 0.7 s.
- `tests/core/test_xf32_bf16x3_device.py` under `ROCKE_REQUIRE_GFX950=1` on
  the same node — 1 passed.
- Both injected faults rejected by the harness on-device (table above).
- Earlier, on a gfx942 host — the harness prints `SKIPPED` (exit 0) and
  `FAILED` (exit 2) under `--require-device`, and the test skips, erroring
  under `ROCKE_REQUIRE_GFX950=1`.
- The probe's own branches, driven on a host with no device — some with faked
  probe output, and the load/count ones by running the real probe source
  against a stubbed `_resolve_hip` / `_hipGetDeviceCount`: an unloadable HIP
  library and `hipErrorNoDevice` both skip; `hipError(101)` fails; a device
  whose architecture reads back as unknown fails when no gfx950 was found and
  is ignored when one was; unstructured probe output fails rather than reading
  as "no devices"; a gfx950 at index 2 behind a gfx942 and a gfx1100 is found
  and selected; an existing `HIP_VISIBLE_DEVICES=4,5,6` resolves to `6` rather
  than being re-indexed to `2`; and `ROCKE_REQUIRE_GFX950=1` turns each skip
  into a failure.
- On the development host as it stands — ROCm present, `hipGetDeviceCount`
  answering `hipError(100)` — `tests/core` is 427 passed / 20 skipped, this
  test among the skips, and 1 error under `ROCKE_REQUIRE_GFX950=1`.
- No emission changed in this step, so the byte-identity result from
  [steps 5+6](step-05-bf16x3-lowering.md) still stands.
