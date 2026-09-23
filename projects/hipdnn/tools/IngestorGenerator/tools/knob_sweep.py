"""Plan a knob sweep: isolate each knob, pair only the survivors, ship what moved.

A two-arm set holds everything else at parity, so an effect is attributable to
one knob, and restricting the pairwise pass to survivors keeps it from becoming
a cross-product. A knob the dispatcher already varies per shape, or one
carrying a measured verdict in the kernel's source, is excluded before any GPU
time; `--plan` prints that partition with reasons.

    knob_sweep.py --profile <p.yaml> --shapes <corpus.json> --plan
    knob_sweep.py --profile <p.yaml> --shapes <corpus.json> --isolate --out-dir <d>
    knob_sweep.py --profile <p.yaml> --shapes <corpus.json> \\
                  --pairwise waves_per_eu,block_n --out-dir <d>

Emits one generator config per arm and never measures.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dispatch_parity import (  # noqa: E402
    ParityError,
    _bind_provider,
    _import,
    _load_profile,
    _predicate_result,
    _required,
    build_config,
    knob_partition,
    resolve_shapes,
)


class Knob:
    """One sweep candidate, with the reason it is or is not one."""

    def __init__(self, name: str, decl: dict):
        self.name = name
        self.values = list(decl.get("values") or [])
        self.hazard = decl.get("hazard") or ""
        self.verdict = decl.get("verdict") or ""

    @property
    def settled(self) -> bool:
        """A knob with a measured verdict is not an open question."""
        return bool(self.verdict)


def load_knobs(profile: dict) -> list[Knob]:
    return [Knob(name, decl) for name, decl in (profile.get("sweep") or {}).items()]


def _promote(spec, arch_spec_cls, overrides: dict):
    """Re-express `spec` as the arch subclass so a private knob can be set,
    carrying every shared field across unchanged. The dispatcher returns the
    shared spec and touches no arch-private codegen knob, so the knobs most
    worth sweeping are exactly the ones absent from it."""
    shared = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
    allowed = {f.name for f in dataclasses.fields(arch_spec_cls)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ParityError(
            f"{unknown} are not fields of {arch_spec_cls.__name__}; a sweep candidate "
            f"must name a real spec field, or the arm silently equals parity."
        )
    return arch_spec_cls(**{**shared, **overrides})


def _support_predicate(profile: dict):
    """The engine's own eligibility predicate, or None when the profile
    declares none. Bound once per arm rather than per shape: a predicate that
    will not import is an operational failure of the whole run."""
    declaration = profile.get("predicate") or {}
    if not declaration:
        return None
    return _import(*_required(declaration, "predicate", "module", "function"))


def _arm(
    resolutions, profile: dict, overrides: dict, arch_spec_cls=None
) -> tuple[dict, list[tuple[int, str]]]:
    """One arm: the parity set with `overrides` forced onto every served spec.

    Built by mutating the dispatcher's own resolution, so the arm differs from
    parity only in the named knobs. Every arm is promoted to the arch spec when
    one is declared, including the baseline, which would otherwise also differ
    in the subclass's private defaults.

    Returns `(config, unbuildable)`, where `unbuildable` lists
    `(shape_index, reason)` for every served shape whose spec refuses this
    arm's value -- a property of the set that the caller must report, since a
    ratio over a different shape population is not a comparison.

    A shape is refused either by the spec constructor or by the eligibility
    predicate, which is asked about the final spec, after promotion and this
    arm's overrides.
    """
    mutated = []
    unbuildable: list[tuple[int, str]] = []
    predicate = _support_predicate(profile)
    arch = profile.get("arch")
    for index, resolution in enumerate(resolutions):
        if resolution.spec is None:
            mutated.append(resolution)
            continue
        clone = copy.copy(resolution)
        private = [k for k in overrides if not hasattr(resolution.spec, k)]
        if private and arch_spec_cls is None:
            raise ParityError(
                f"{private} are not on the spec the dispatcher returns and no "
                f"arch_spec is declared in the profile. Arch-private codegen knobs "
                f"are exactly the ones the dispatcher leaves alone, so a sweep needs "
                f"the builder's own spec class to reach them."
            )
        # A knob value can be illegal for some shapes and legal for others, and
        # that is a property of the set: `wide_lds_dma` requires `block_n=64`,
        # so a `block_n=128` arm cannot express a shape the dispatcher resolved
        # wide-DMA for. Drop those shapes and count them rather than aborting.
        try:
            if arch_spec_cls is not None:
                clone.spec = _promote(resolution.spec, arch_spec_cls, overrides)
            else:
                # The spec is a frozen dataclass, so replace() rather than
                # setattr: mutating a shared object in place is how one arm's
                # override leaks into the next one's baseline.
                clone.spec = dataclasses.replace(resolution.spec, **overrides)
        except ParityError:
            raise
        except Exception as exc:
            unbuildable.append((index, f"{type(exc).__name__}: {exc}"))
            continue
        if predicate is not None:
            # The final spec, promoted to the builder's class and carrying this
            # arm's overrides: a perturbation that constructs and is still
            # declined is invisible to the constructor above.
            supported, why = _predicate_result(
                predicate, clone.spec, **({"arch": arch} if arch else {})
            )
            if not supported:
                unbuildable.append((index, f"unsupported: {why}"))
                continue
        mutated.append(clone)
    return build_config(mutated, profile), unbuildable


def _write(config: dict, path: Path) -> int:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return len(config["packs"][0]["kernels"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan and emit a knob sweep: isolation first, then pairwise.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--shapes", required=True)
    parser.add_argument("--out-dir", help="Where to write the arm configs.")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the candidate/excluded partition and stop.",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Emit the two-arm isolation set, one arm per knob.",
    )
    parser.add_argument("--pairwise", help="Comma-separated SURVIVORS to pair.")
    parser.add_argument(
        "--include-settled",
        action="store_true",
        help="Sweep knobs that already carry a measured verdict. "
        "Off by default: re-measuring a settled knob spends GPU "
        "time to rediscover something the source already says.",
    )
    args = parser.parse_args(argv)

    try:
        profile = _load_profile(args.profile)
        _bind_provider(profile.get("provider_root"))
        shapes = json.loads(Path(args.shapes).read_text())
        resolutions = resolve_shapes(shapes, profile)
    except ParityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    served = [r for r in resolutions if r.spec is not None]
    if not served:
        print("FAIL: no shape resolved; there is nothing to sweep.", file=sys.stderr)
        return 1

    varies, constant = knob_partition(resolutions)
    knobs = load_knobs(profile)
    arch_decl = profile.get("arch_spec") or {}
    arch_spec_cls = (
        _import(arch_decl["module"], arch_decl["class"]) if arch_decl else None
    )

    candidates, excluded = [], []
    for knob in knobs:
        if knob.settled and not args.include_settled:
            excluded.append((knob.name, f"settled: {knob.verdict}"))
        elif knob.name in varies:
            # The dispatcher already moves it per shape, so it is a production axis
            # rather than a sweep candidate -- perturbing it fights the policy.
            excluded.append((knob.name, "the dispatcher varies it per shape"))
        elif len(knob.values) < 2:
            excluded.append((knob.name, "fewer than two values declared"))
        else:
            candidates.append(knob)

    print("knob sweep")
    print(f"  shapes served     {len(served)}")
    print(f"  dispatcher varies {', '.join(varies) or '(none)'}")
    print(f"  candidates        {len(candidates)}")
    for knob in candidates:
        hazard = f"  -- {knob.hazard}" if knob.hazard else ""
        print(f"      {knob.name} = {knob.values}{hazard}")
    print(f"  excluded          {len(excluded)}")
    for name, why in excluded:
        print(f"      {name}: {why}")

    if args.plan:
        print(
            "\n  Isolate before pairing. A knob that does nothing alone is not worth a "
            "cross-product\n  slot, and the last sweep's wide arm bought nothing "
            "measurable over the condensed one."
        )
        return 0

    if not args.out_dir:
        parser.error("--isolate and --pairwise need --out-dir")
    out = Path(args.out_dir)

    if args.isolate:
        if not candidates:
            print("\nFAIL: no candidate knobs to isolate.", file=sys.stderr)
            return 1
        baseline, base_unbuildable = _arm(resolutions, profile, {}, arch_spec_cls)
        assert not base_unbuildable, (
            "the PARITY baseline itself has unbuildable shapes, which is impossible "
            "by construction -- the dispatcher resolved these specs, so they build. "
            f"Got: {base_unbuildable[:3]}"
        )
        count = _write(baseline, out / "arm_parity.yaml")
        served_total = count
        print(f"\n  arm_parity.yaml               {count:5d} kernels  (the baseline)")
        base_specs = [
            k["kernel_source"]["spec"] for k in baseline["packs"][0]["kernels"]
        ]
        for knob in candidates:
            for value in knob.values:
                arm, unbuildable = _arm(
                    resolutions, profile, {knob.name: value}, arch_spec_cls
                )
                name = f"arm_{knob.name}_{value}.yaml"
                count = _write(arm, out / name)
                # An arm whose value is what the dispatcher already resolves is
                # the baseline under another name: it measures 1.000x and
                # reports "no effect" having never tried the other side.
                arm_specs = [
                    k["kernel_source"]["spec"] for k in arm["packs"][0]["kernels"]
                ]
                identical = arm_specs and all(
                    b.get(knob.name) == a.get(knob.name)
                    for b, a in zip(base_specs, arm_specs)
                )
                note = "  == parity, measures nothing" if identical else ""
                print(f"  {name:<30}{count:5d} kernels{note}")
                # A narrowed arm is not a full comparison. Report the fraction
                # and the distinct reasons so the arm's coverage is a number
                # the reader checks rather than assumes.
                if unbuildable:
                    reasons: dict[str, int] = {}
                    for _, why in unbuildable:
                        reasons[why] = reasons.get(why, 0) + 1
                    print(
                        f"  {'':30}{'':5} NARROWED: covers {count} of "
                        f"{served_total} served shapes; "
                        f"{len(unbuildable)} cannot express this value"
                    )
                    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
                        print(f"  {'':30}{'':5}   {n:4d} x {why}")
        print(
            "\n  Two arms per knob, everything else at parity, so an effect is "
            "attributable\n  to that knob alone. Measure these before pairing anything."
            "\n  A NARROWED arm is measurable but is NOT a comparison over the whole"
            "\n  set -- report its fraction, never a bare ratio against parity."
        )
        return 0

    if args.pairwise:
        names = [n.strip() for n in args.pairwise.split(",") if n.strip()]
        by_name = {k.name: k for k in candidates}
        missing = [n for n in names if n not in by_name]
        if missing:
            print(
                f"FAIL: {missing} are not sweep candidates. Pair only knobs that "
                f"survived isolation -- pairing everything is the cross-product this "
                f"tool exists to avoid.",
                file=sys.stderr,
            )
            return 2
        chosen = [by_name[n] for n in names]
        combos = list(itertools.product(*[k.values for k in chosen]))
        print(f"\n  pairwise over {names}: {len(combos)} arms")
        for combo in combos:
            overrides = dict(zip(names, combo))
            label = "_".join(f"{k}{v}" for k, v in overrides.items())
            arm, unbuildable = _arm(resolutions, profile, overrides, arch_spec_cls)
            count = _write(arm, out / f"pair_{label}.yaml")
            note = (
                f"  NARROWED: {len(unbuildable)} shapes cannot express this combination"
                if unbuildable
                else ""
            )
            print(f"  pair_{label:<26}{count:5d} kernels{note}")
        return 0

    parser.error("choose --plan, --isolate or --pairwise")


if __name__ == "__main__":
    sys.exit(main())
