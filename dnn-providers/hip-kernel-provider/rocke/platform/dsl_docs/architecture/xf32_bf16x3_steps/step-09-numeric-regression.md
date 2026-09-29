# Step 9 — what the recipe actually computes

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
step 9). Ticket AICK-2208.

## Contents

- [The gap steps 5–8 leave](#the-gap-steps-58-leave)
- [Two halves, split by what each can prove](#two-halves-split-by-what-each-can-prove)
- [The host half: the arithmetic](#the-host-half-the-arithmetic)
- [The device half: layout-free by construction](#the-device-half-layout-free-by-construction)
- [Status of the on-device run](#status-of-the-on-device-run)
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
| Does the lowering compute that arithmetic on hardware? | [`examples/gfx950/xf32/bf16x3_parity.py`](../../../python/rocke/examples/gfx950/xf32/bf16x3_parity.py) | yes, gfx950 |

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

## Status of the on-device run

**The gfx950 run has not been performed.** The only device visible from this
development host is gfx942 (`rocminfo` reports
`amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-`), and the emulated rows are
gfx950-only, so there is nothing here to run them on. The harness detects this
and prints `SKIPPED`, exiting 0; `--require-device` turns the same condition
into a failure, which is the spelling a gfx950 validation node should use so a
missing device cannot pass as a green run.

What *was* verified on this host, without a device:

- both kernels build and lower for gfx950 through the Python engine;
- the emulated kernel emits three MFMA calls and the single-term control emits
  one, as the recipe requires;
- the skip path and the `--require-device` path both behave as documented.

The numeric assertions inside the harness are therefore **unexercised**. They
are not reported as passing anywhere, and this note is the record of that.

## Verification

- `tests/core/test_xf32_bf16x3_numerics.py` — 13 tests, green.
- `tests/core` — 425 passed, 19 skipped, 757 subtests.
- `examples/gfx950/xf32/bf16x3_parity.py` — `SKIPPED` (exit 0) and, with
  `--require-device`, `FAILED` (exit 2) on this gfx942 host. Numeric result:
  **not measured**.
- No emission changed in this step, so the byte-identity result from
  [steps 5+6](step-05-bf16x3-lowering.md) still stands.
