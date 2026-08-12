# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# derive_guards.py -- derive admission guards for real kernel families and show
# what they cost and what they buy.
#
#   python3 -m rocke.portable_ir.drivers.derive_guards
#   python3 -m rocke.portable_ir.drivers.derive_guards --family attention_tiled_2d \
#           --axes num_query_heads,num_kv_heads
#   python3 -m rocke.portable_ir.drivers.derive_guards --roll --bundle /tmp/guarded.cbor
#
# Two things worth reading in the output.
#
# UNSOUND must be zero everywhere. It counts points the derived guard admits and
# the family's own gate rejects, which is precisely the failure a guard exists to
# prevent -- hipDNN handed a configuration the kernel never supported. derivation
# already refuses to return such a guard, so a non-zero column here would mean
# the oracle itself is being fooled, and this driver exits non-zero.
#
# OVER-STRICT is a coverage number, not a bug. It counts shapes the kernel would
# have accepted that the guard will refuse, sending hipDNN to another provider.
# Zero is ideal, small is fine, and large means the derivation could not find the
# structure of that family's gate -- which is a finding about the gate as much as
# about the fitting, and usually points at a constraint the kernel enforces deep
# in a builder rather than declaring up front.
#
# The measurement reuses the gfx950 sweep's family table and candidate domains,
# so the axes and values here are the same ones the rolling numbers are quoted
# against.

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Sequence

from rocke.portable_ir.drivers import roll_gfx950_sweep as sweep
from rocke.portable_ir.src.guard import (
    GuardDerivationError,
    attach_guard,
    derive_guard,
    gate_from_spec,
    verify_guard,
)
from rocke.portable_ir.src.recipe_bundle import cbor_encode

ARCH = "gfx950"
_PREFERRED = ["head_size", "block_size"]


def _axes_for(fam, requested: Sequence[str]) -> List[str]:
    if requested:
        return [a for a in requested if a in fam.axes]
    return [a for a in _PREFERRED if a in fam.axes] or fam.axes[:2]


def _candidates(axes: Sequence[str]) -> Dict[str, List[int]]:
    return {a: sweep.CANDIDATES.get(a, sweep._DEFAULT_CANDIDATES) for a in axes}


def run(
    families: Sequence[str],
    axes_req: Sequence[str],
    *,
    probe: bool,
    pool_cap: int,
    samples: int,
) -> int:
    """Derive and verify a guard per family. Returns a process exit code."""
    unsound_total = 0
    for fam in sweep._families():
        if families and fam.label not in families:
            continue
        axes = _axes_for(fam, axes_req)
        if len(axes) < 1:
            continue
        cands = _candidates(axes)

        # The build probe is the most truthful layer and by far the slowest: it
        # compiles a kernel per gate call, where the declarative layers are a
        # few comparisons. Off by default -- a guard derived without it is still
        # sound with respect to the gate it measured, and §4 of the integration
        # doc argues the builder-deep constraints belong in the gate anyway.
        gate = gate_from_spec(
            fam.make_spec,
            admits=fam.admits,
            probe=fam.build if probe else None,
            coherent=fam.coherent,
        )

        print(f"{fam.label}  [{', '.join(axes)}]")
        t0 = time.time()
        try:
            guard = derive_guard(
                gate,
                cands,
                gate_name=fam.label,
                arch=ARCH,
                pool_cap=pool_cap,
            )
        except GuardDerivationError as e:
            print(f"    no guard: {e}\n")
            continue
        dt = time.time() - t0

        report = verify_guard(guard, gate, cands, samples=samples, seed=1337)
        unsound_total += len(report.unsound)
        space = 1
        for a in axes:
            space *= len(cands[a])
        print(f"    method     {guard['derivation']['method']}")
        for rule in guard["rules"]:
            print(f"      - {rule['reason'][:96]}")
        print(
            f"    oracle     {report.agreed}/{report.checked} agreed, "
            f"{len(report.unsound)} UNSOUND, {len(report.strict)} over-strict"
        )
        if report.unsound:
            print(f"      !! admits {report.unsound[0]}, which the gate rejects")
        print(
            f"    cost       {dt:.2f}s to derive, {len(cbor_encode(guard))} B of CBOR, "
            f"over a {space}-point candidate space\n"
        )
    return 1 if unsound_total else 0


def roll_and_bundle(family: str, axes_req: Sequence[str], out: str) -> int:
    """The whole chain on a real kernel: roll a recipe, guard it, ship it, and
    make the C engine enforce it.

    Rolling is what creates the need for the guard, so a demonstration that
    skips it is not demonstrating much. This records one parametric recipe over
    the family's axes, attaches a guard derived from that family's own gate,
    encodes the bundle, and then asks the C API about points on both sides of
    the guard -- the same call hipDNN would make."""
    from rocke.portable_ir.src.roll_nd import roll_nd

    fam = next((f for f in sweep._families() if f.label == family), None)
    if fam is None:
        print(f"no such family '{family}'", file=sys.stderr)
        return 2
    gate = gate_from_spec(fam.make_spec, admits=fam.admits, coherent=fam.coherent)

    # The guard has to cover exactly the recipe's free axes, so which axis to
    # guard is decided by which one ROLLS, not by which one is interesting.
    # Several do not: head_size changes how much code the tiled kernels emit, so
    # one recipe cannot cover it and there is nothing to guard. Search rather
    # than assume, and say which axis was chosen.
    axis, recipe = "", None
    for cand_axis in axes_req or fam.axes:
        if cand_axis not in fam.axes:
            continue
        legal = [v for v in _candidates([cand_axis])[cand_axis] if gate({cand_axis: v})]
        if len(legal) < 3:
            continue
        try:
            r = roll_nd(fam.build, axes={cand_axis: legal[:3]})
        except Exception as e:
            print(f"  {cand_axis}: {type(e).__name__}: {str(e)[:70]}")
            continue
        if r.ok:
            axis, recipe = cand_axis, r.recipe
            print(f"rolled {fam.label} over {cand_axis} = {legal[:3]}")
            break
        print(f"  {cand_axis}: does not roll ({r.reason[:70]})")
    if recipe is None:
        print(f"no axis of {fam.label} rolls; nothing to guard", file=sys.stderr)
        return 1

    cands = _candidates([axis])
    guard = derive_guard(gate, cands, gate_name=fam.label, arch=ARCH)
    ref = dict(guard["derivation"]["reference"])
    legal = [v for v in cands[axis] if gate({**ref, axis: v})]
    recipe = attach_guard(recipe, guard)
    blob = cbor_encode(
        {
            "schema": "rocke.bundle/v1",
            "entries": [{"key": fam.label, "arch": ARCH, "recipe": recipe}],
        }
    )
    with open(out, "wb") as fh:
        fh.write(blob)
    print(f"wrote {out} ({len(blob)} B, guard {len(cbor_encode(guard))} B)")

    try:
        from rocke.portable_ir.src import online

        illegal = next(
            (v for v in range(1, max(legal) + 2) if v not in legal), legal[0] + 1
        )
        for value, expect in ((legal[0], "admitted"), (illegal, "refused")):
            verdict, why = online.check_bundle_guard(
                blob, fam.label, arch=ARCH, ints={axis: value}
            )
            mark = "ok" if verdict == expect else "MISMATCH"
            print(f"  C API {axis}={value}: {verdict} {why} [{mark}]")
    except Exception as e:  # no shared librocke here is not a failure of the driver
        print(f"  (skipped C check: {type(e).__name__}: {str(e)[:80]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="", help="only this family")
    ap.add_argument("--axes", default="", help="comma-separated free axes")
    ap.add_argument(
        "--probe",
        action="store_true",
        help="include an actual build in the gate (truthful, slow)",
    )
    ap.add_argument("--pool-cap", type=int, default=32)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--roll", action="store_true", help="roll + bundle + check in C")
    ap.add_argument("--bundle", default="/tmp/guarded_bundle.cbor")
    args = ap.parse_args()

    axes = [a for a in args.axes.split(",") if a]
    if args.roll:
        return roll_and_bundle(args.family or "attention_tiled_2d", axes, args.bundle)
    return run(
        [args.family] if args.family else [],
        axes,
        probe=args.probe,
        pool_cap=args.pool_cap,
        samples=args.samples,
    )


if __name__ == "__main__":
    raise SystemExit(main())
