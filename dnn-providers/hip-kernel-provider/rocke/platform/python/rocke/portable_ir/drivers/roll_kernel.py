# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Record, roll and ship ONE named kernel — the generic form of the gates.

Every other roll driver here carries a hard-coded family list pinned to
``kernels/gfx950``, because each is a gate defending a fixed claim. This one is
the developer-facing tool: you name a kernel module and its axes on the command
line, and it runs the same pipeline the gates run.

    record -> roll (N axes) -> verify .ll/HSACO against the Python oracle
           -> derive guard -> stamp ABI -> write a CBOR bundle

Start with ``--probe``, which is the step most people skip and then misread.
For each axis it answers two separate questions:

  * **does it roll?** — a refusal is a normal, safe outcome, and the reason
    tells you whether it is a modelling gap or a real structural change.
  * **does it matter?** — an axis the emitted program does not depend on
    "rolls" trivially. That is a vacuous pass: the recipe covers the axis
    because nothing varies with it, so the coverage it appears to buy is not
    real. Only a probe that compares recorded programs can tell you this, which
    is why it runs before the roll and not after.

Examples::

    # triage: which axes are worth rolling?
    python3 -m rocke.portable_ir.drivers.roll_kernel \\
        --kernel kernels.gfx1151.wmma_fmha_fwd --arch gfx1151 \\
        --fixed head_size=64 --fixed mask_mode=causal \\
        --axis num_query_heads=8,16 --axis sliding_window=64,128 --probe

    # roll, verify byte-identity, and write a shippable bundle
    python3 -m rocke.portable_ir.drivers.roll_kernel \\
        --kernel kernels.gfx1151.wmma_fmha_fwd --arch gfx1151 \\
        --fixed head_size=64 --fixed mask_mode=causal \\
        --axis num_query_heads=8,16 --holdout num_query_heads=32 \\
        --verify --hsaco --guard --out /tmp/wmma_fmha.cbor

Exit status is 0 only if every requested stage passed, so it drops straight
into CI. ``--verify`` needs a shared ``librocke`` (``ROCKE_ONLINE_LIB``, or
``online.build_lib()``); ``--hsaco`` additionally needs comgr.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

Point = Dict[str, Any]


# --------------------------------------------------------------------------
# resolving the kernel module
# --------------------------------------------------------------------------
def _public(mod: Any) -> List[str]:
    """Names the module owns, preferring __all__ over whatever it imported."""
    names = getattr(mod, "__all__", None)
    if names:
        return list(names)
    return [
        n
        for n in dir(mod)
        if not n.startswith("_")
        and getattr(getattr(mod, n), "__module__", None) == mod.__name__
    ]


def _pick(mod: Any, kind: str, want: Optional[str], match: Callable[[Any, str], bool]):
    """The one name of ``kind`` in ``mod``, or a hard error naming the choices."""
    if want:
        if not hasattr(mod, want):
            raise SystemExit(f"{mod.__name__} has no {kind} named {want!r}")
        return getattr(mod, want)
    found = [n for n in _public(mod) if match(getattr(mod, n), n)]
    if len(found) == 1:
        return getattr(mod, found[0])
    if not found:
        raise SystemExit(f"no {kind} found in {mod.__name__}; pass it explicitly")
    raise SystemExit(
        f"{mod.__name__} exposes several {kind}s ({', '.join(sorted(found))}); "
        f"pick one explicitly"
    )


def resolve(module: str, build: Optional[str], spec: Optional[str]):
    """(Spec dataclass, build fn, is_valid_spec or None) for a kernel module."""
    mod = importlib.import_module(module)
    spec_cls = _pick(
        mod, "spec dataclass", spec, lambda o, n: dataclasses.is_dataclass(o)
    )
    build_fn = _pick(
        mod, "build function", build, lambda o, n: callable(o) and n.startswith("build")
    )
    gate = getattr(mod, "is_valid_spec", None)
    if gate is None:  # the other convention in this tree
        cands = [n for n in _public(mod) if n.startswith("supports_")]
        gate = getattr(mod, cands[0]) if len(cands) == 1 else None
    return spec_cls, build_fn, gate


def _coerce(text: str) -> Any:
    for cast in (int,):
        try:
            return cast(text)
        except ValueError:
            pass
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    return text


def _kv(items: List[str], *, many: bool) -> Dict[str, Any]:
    """``name=v`` (or ``name=v1,v2`` when ``many``) pairs into a dict."""
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected name=value, got {item!r}")
        name, _, rhs = item.partition("=")
        vals = [_coerce(v) for v in rhs.split(",") if v != ""]
        if not vals:
            raise SystemExit(f"no value given for {name!r}")
        out[name.strip()] = vals if many else vals[0]
    return out


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def probe_axes(build_at, axes: Dict[str, List[Any]], structural: Optional[str]) -> bool:
    """Per-axis triage. Returns False only if an axis is VACUOUS.

    A refusal is a normal outcome everywhere else in this tree and it is one
    here: an axis that declines costs coverage, not correctness, and some will
    decline until the roller grows. Failing on that would leave a CI job
    permanently red for a known gap. A vacuous axis is different — it is an
    authoring mistake that quietly claims coverage it does not have."""
    from rocke.portable_ir.src import recipe_bundle
    from rocke.portable_ir.src.recording_builder import record_kernel
    from rocke.portable_ir.src.roll_nd import roll_nd

    base = {a: v[0] for a, v in axes.items()}
    print("-- per-axis probe: does it roll, and does it change the program? --\n")
    print("%-22s %-10s %s" % ("axis", "verdict", "detail"))
    clean = True

    for axis, values in axes.items():
        if len(values) < 2:
            print("%-22s %-10s %s" % (axis, "skipped", "needs >= 2 sample values"))
            continue

        # Does the recorded program actually depend on this axis?
        traces = []
        for v in values[:2]:
            _, rec = record_kernel(lambda p={**base, axis: v}: build_at(**p))
            traces.append(recipe_bundle.cbor_encode(rec))
        if traces[0] == traces[1]:
            clean = False
            print(
                "%-22s %-10s %s"
                % (
                    axis,
                    "VACUOUS",
                    f"identical program at {values[0]} and {values[1]} — "
                    f"rolling it proves nothing",
                )
            )
            continue

        r = roll_nd(
            lambda **p: build_at(**{**base, **p}),
            axes={axis: list(values)},
            structural_axis=axis if structural == axis else None,
            extra_spec={},
        )
        if r.ok:
            print(
                "%-22s %-10s %s" % (axis, "rolls", f"{len(r.points)} points verified")
            )
        else:
            print("%-22s %-10s %s" % (axis, "declines", r.reason[:96]))
    return clean


def verify(build_at, recipe, points: List[Point], arch: str, hsaco: bool) -> bool:
    """Replay each point through the C engine and diff against the oracle."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.portable_ir.src import online, recipe_bundle

    cbor = recipe_bundle.cbor_encode(recipe)
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR", "")
    print("\n-- verify: Python oracle vs C replay of the rolled recipe --\n")
    print("%-38s %-8s %-14s %s" % ("point", ".ll", "ll sha", "HSACO"))
    ok = True
    for p in points:
        py_ll = lower_kernel_to_llvm(
            build_at(**p), arch=arch, **({"llvm_flavor": flavor} if flavor else {})
        )
        vm_ll, _ = online.recipe_cbor_to_llvm(cbor, arch=arch, ints=dict(p))
        same = py_ll == vm_ll
        ok &= same
        cell = "-"
        if hsaco:
            from rocke.core.arch import ArchTarget
            from rocke.runtime.comgr import build_hsaco_from_llvm_ir

            isa = ArchTarget.from_gfx(arch).isa_triple
            py_h, _ = build_hsaco_from_llvm_ir(py_ll, isa=isa, options=["-O3"])
            vm_h, _ = build_hsaco_from_llvm_ir(vm_ll, isa=isa, options=["-O3"])
            ok &= py_h == vm_h
            cell = f"{_sha(py_h)} ({len(py_h)} B)" if py_h == vm_h else "DIFFER"
        label = " ".join(f"{k}={v}" for k, v in sorted(p.items()))
        print(
            "%-38s %-8s %-14s %s"
            % (label[:38], "EXACT" if same else "DIFFER", _sha(py_ll.encode()), cell)
        )
    return ok


def ship(recipe, spec_cls, gate_fn, domain, fixed, arch, out: str, want_guard: bool):
    """Attach a guard, stamp the wire ABI, and write a one-entry bundle.

    ``domain`` is the set of values the bundle is meant to *serve*, which is not
    the set it was fitted from. Deriving a guard from the two sample points
    produces a rule admitting exactly those two, so the recipe refuses shapes it
    replays byte-identically — over-strict, and silently so."""
    from rocke.portable_ir.src import abi as _abi
    from rocke.portable_ir.src import recipe_bundle

    print("\n-- ship --\n")
    if want_guard:
        from rocke.portable_ir.src.guard import derive_guard, gate_from_spec

        if gate_fn is None:
            print("   guard   : skipped (module exposes no is_valid_spec/supports_*)")
        else:
            gate = gate_from_spec(
                lambda **p: spec_cls(**{**fixed, **p}),
                admits=lambda s: gate_fn(s, arch),
            )
            t0 = time.perf_counter()
            guard = derive_guard(
                gate, {a: list(v) for a, v in domain.items()}, arch=arch
            )
            recipe = {**recipe, "guard": guard}
            print(
                "   guard   : %d rule(s) over %s, derived in %.0f ms"
                % (
                    len(guard.get("rules", [])),
                    ", ".join(f"{a}[{len(v)}]" for a, v in domain.items()),
                    (time.perf_counter() - t0) * 1e3,
                )
            )
            for rule in guard.get("rules", []):
                print(f"             - {rule.get('reason', '')}")

    key = recipe.get("kernel_name_fmt") or recipe.get("kernel_name") or "kernel"
    bundle = recipe_bundle.build_bundle([{"key": key, "arch": arch, "recipe": recipe}])
    blob = recipe_bundle.cbor_encode(bundle)
    with open(out, "wb") as fh:
        fh.write(blob)
    print("   abi     :", _abi.describe(bundle))
    print("   key     :", key)
    print("   wrote   : %s (%.1f KiB)" % (out, len(blob) / 1024.0))


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kernel", required=True, help="module path, e.g. kernels.gfx.foo")
    ap.add_argument("--arch", required=True)
    ap.add_argument(
        "--build", default=None, help="build fn name (default: the only one)"
    )
    ap.add_argument(
        "--spec", default=None, help="spec class name (default: the only one)"
    )
    ap.add_argument(
        "--axis",
        action="append",
        default=[],
        metavar="NAME=V1,V2",
        help="a free axis and its >=2 sample values; repeatable",
    )
    ap.add_argument(
        "--holdout",
        action="append",
        default=[],
        metavar="NAME=V",
        help="values never used for fitting, verified after; repeatable",
    )
    ap.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="NAME=V",
        help="spec fields held constant (baked into the recipe); repeatable",
    )
    ap.add_argument(
        "--domain",
        action="append",
        default=[],
        metavar="NAME=V1,V2,..",
        help="every value the guard should ADMIT for an axis — the shapes you "
        "intend to serve, not the ones you fitted from. Defaults to the sample "
        "values plus the holdouts, which is almost certainly too narrow",
    )
    ap.add_argument("--structural", default=None, help="the one axis that may reshape")
    ap.add_argument("--probe", action="store_true", help="per-axis triage, then stop")
    ap.add_argument("--verify", action="store_true", help=".ll parity vs the oracle")
    ap.add_argument("--hsaco", action="store_true", help="and compare HSACO (comgr)")
    ap.add_argument("--guard", action="store_true", help="derive an admission guard")
    ap.add_argument("--out", default="", help="write a CBOR bundle here")
    args = ap.parse_args(argv)

    os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")
    axes = {k: v for k, v in _kv(args.axis, many=True).items()}
    fixed = _kv(args.fixed, many=False)
    holds = _kv(args.holdout, many=True)
    if not axes:
        raise SystemExit("need at least one --axis")
    if args.structural and args.structural not in axes:
        raise SystemExit(f"--structural {args.structural!r} is not one of the axes")

    spec_cls, build_fn, gate_fn = resolve(args.kernel, args.build, args.spec)

    def build_at(**point: Any):
        return build_fn(spec_cls(**{**fixed, **point}), arch=args.arch)

    print(f"== {args.kernel} on {args.arch} ==")
    print(f"   spec  : {spec_cls.__name__}   build: {build_fn.__name__}")
    print(f"   fixed : {fixed or '(none)'}")
    print(f"   axes  : {axes}")
    print(f"   holdout: {holds or '(none)'}\n")

    # Build once at the base point before anything else. Without this a missing
    # spec field surfaces as a TypeError from inside the roller's recording
    # lambda, several frames from the thing the user got wrong.
    base = {a: v[0] for a, v in axes.items()}
    try:
        build_at(**base)
    except TypeError as e:
        need = [
            f.name
            for f in dataclasses.fields(spec_cls)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
            and f.name not in fixed
            and f.name not in axes
        ]
        print(f"cannot build at the base point {base}: {e}")
        if need:
            print(f"   {spec_cls.__name__} still needs: {', '.join(need)}")
            print("   pass each as --fixed NAME=V or --axis NAME=V1,V2")
        return 2
    except Exception as e:  # noqa: BLE001 - a rejected base point is a real answer
        print(f"cannot build at the base point {base}:\n   {type(e).__name__}: {e}")
        return 2

    if args.probe:
        return 0 if probe_axes(build_at, axes, args.structural) else 1

    if args.verify or args.hsaco:
        from rocke.portable_ir.src import online

        online.load()

    from rocke.portable_ir.src.roll_nd import roll_nd

    # A holdout must name every axis, so a per-axis list becomes a point list by
    # position, with unnamed axes held at their base value.
    n_hold = max((len(v) for v in holds.values()), default=0)
    hold_points = [
        {
            a: (holds[a][i] if a in holds and i < len(holds[a]) else axes[a][0])
            for a in axes
        }
        for i in range(n_hold)
    ]

    t0 = time.perf_counter()
    r = roll_nd(
        build_at,
        axes=axes,
        structural_axis=args.structural,
        holdout_points=hold_points,
        extra_spec={},
    )
    if not r.ok:
        print(f"DECLINED after {(time.perf_counter() - t0) * 1e3:.0f} ms")
        print(f"   {r.reason}")
        print("\nThe concrete path still works: ship r.traces as per-point recipes.")
        print("Run with --probe to see which single axis is responsible.")
        return 1

    from rocke.portable_ir.src import recipe_bundle

    cbor = recipe_bundle.cbor_encode(r.recipe)
    concrete = sum(len(recipe_bundle.cbor_encode(t)) for t in r.traces.values())
    print("ROLLED in %.0f ms" % ((time.perf_counter() - t0) * 1e3))
    print(f"   recorded {r.n_recorded} trace(s), verified {len(r.points)} point(s)")
    print(f"   name_fmt : {r.recipe.get('kernel_name_fmt')}")
    print(
        "   CBOR     : %.1f KiB parametric vs %.1f KiB for the same points concrete"
        % (len(cbor) / 1024.0, concrete / 1024.0)
    )

    ok = True
    if args.verify or args.hsaco:
        ok = verify(build_at, r.recipe, r.points, args.arch, args.hsaco)
        print("\n  ", "all points byte-identical" if ok else "PARITY FAILED")

    if args.out or args.guard:
        declared = _kv(args.domain, many=True)
        domain = {
            a: declared.get(a, sorted(set(axes[a]) | set(holds.get(a, []))))
            for a in axes
        }
        if args.guard and not declared:
            print(
                "\n   note: no --domain given, so the guard can only admit the "
                "values seen here.\n         Pass the real serving domain or the "
                "bundle will refuse shapes it can build."
            )
        ship(
            r.recipe,
            spec_cls,
            gate_fn,
            domain,
            fixed,
            args.arch,
            args.out or "/dev/null",
            args.guard,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
