"""Generate the variant set rocKE's own dispatcher would resolve.

Stage 1 of an integration, and the only configuration argued from rocKE's
behaviour rather than from measurement: the validators, the spec dataclass and
the ``supports_*`` predicate answer what is legal, the dispatcher answers what
the library ships. Calling the factory applies rules the dispatcher derives
from the request (``persistent = work >= num_persistent``).

Emits one variant per servable shape, at the dispatcher's resolved spec, as a
generator config ready for ``generate.py``. Not a cross-product: one spec per
shape.

    dispatch_parity.py --profile <profile.yaml> --shapes <corpus.json> \\
                       --out configs/<slug>_A.yaml

A request-construction failure means the corpus and the request class disagree
about what a shape is, so it aborts (``ParityError``, ``FAIL`` on stderr, exit
2). A decline -- the eligibility predicate returning false with a reason -- is
a per-shape outcome, counted and printed by ``--report-gaps``.

This tool does not sweep. ``--report-knobs`` partitions the spec fields into
those that vary across dispatch decisions and those the library ships (see
``knob_partition``).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import itertools
import json
import math
import os
import sys
from pathlib import Path


class ParityError(RuntimeError):
    """The dispatcher could not be reached or asked. Never a shape-level decline."""


def _predicate_result(predicate, *args, **kwargs) -> tuple[bool, str]:
    """Invoke an eligibility API without converting operational errors to declines."""
    if not callable(predicate):
        raise ParityError("eligibility predicate is not callable")
    try:
        inspect.signature(predicate).bind(*args, **kwargs)
        result = predicate(*args, **kwargs)
    except Exception as exc:
        raise ParityError(f"eligibility predicate failed: {exc}") from exc
    if (
        type(result) is not tuple
        or len(result) != 2
        or type(result[0]) is not bool
        or type(result[1]) is not str
        or (not result[0] and not result[1].strip())
    ):
        raise ParityError(
            "eligibility predicate must return (bool, str), with a decline reason"
        )
    return result


def _eligible(candidate, request) -> tuple[bool, str]:
    """Ask one candidate the complete eligibility question. `admits` is the
    only eligibility API, so a candidate not exposing it raises rather than
    reading as a decline."""
    try:
        # Static lookup distinguishes absence from a descriptor that raises on access.
        inspect.getattr_static(candidate, "admits")
        predicate = getattr(candidate, "admits")
    except Exception as exc:
        raise ParityError(f"cannot look up eligibility API 'admits': {exc}") from exc
    return _predicate_result(predicate, request)


def _load_profile(path: str) -> dict:
    """Parse a profile as JSON, falling back to YAML. The mapping check covers
    both paths: valid JSON that is not an object would otherwise crash several
    frames later naming neither the file nor the problem."""
    text = Path(path).read_text()
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError:  # pragma: no cover - environment-dependent
            raise ParityError(f"{path} is not JSON and PyYAML is not installed.")
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ParityError(
            f"profile {path} must be a mapping; got {type(loaded).__name__}."
        )
    return loaded


def _bind_provider(provider_root: str | None) -> None:
    root = provider_root or os.environ.get("ROCKE_PROVIDER_ROOT")
    if not root:
        raise ParityError(
            "no provider_root in the profile and no ROCKE_PROVIDER_ROOT set; the "
            "dispatcher cannot be imported without the rocKE library."
        )
    root = os.path.abspath(os.path.expanduser(root))
    for sub in ("rocke/library", "rocke/platform/python"):
        candidate = os.path.join(root, sub)
        if not os.path.isdir(candidate):
            raise ParityError(f"{candidate} does not exist; is {root} a provider root?")
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _import(dotted: str, symbol: str):
    import importlib

    try:
        module = importlib.import_module(dotted)
    except ImportError as exc:
        raise ParityError(f"cannot import '{dotted}': {exc}")
    attribute = getattr(module, symbol, None)
    if attribute is None:
        raise ParityError(f"'{dotted}' does not define '{symbol}'")
    return attribute


@dataclasses.dataclass
class Resolution:
    """One shape's outcome. Exactly one of `spec` / `reason` is set."""

    shape: dict
    spec: object | None = None
    reason: str | None = None
    #: "constructed" (spec built and predicate accepted) or "declined"
    #: (the predicate returned a validated false result).
    kind: str = "constructed"


def _required(decl: dict, scope: str, *keys: str) -> list:
    """Pull `keys` out of a profile block, naming the block when one is
    missing: a bare `decl["module"]` raises `KeyError: 'module'` without
    saying which of the profile's several blocks was incomplete.
    """
    missing = [k for k in keys if k not in decl]
    if missing:
        raise ParityError(
            f"the profile's '{scope}' block is missing {missing}. It needs "
            f"{list(keys)} so the tool knows what to import."
        )
    return [decl[k] for k in keys]


def resolve_shapes(shapes: list[dict], profile: dict) -> list[Resolution]:
    """Ask the dispatcher for every shape; API errors are operational failures."""
    dispatch = profile.get("dispatch") or {}
    request_decl = profile.get("request") or {}
    predicate_decl = profile.get("predicate") or {}

    factory = _import(*_required(dispatch, "dispatch", "module", "function"))
    request_cls = _import(*_required(request_decl, "request", "module", "class"))
    predicate = (
        _import(*_required(predicate_decl, "predicate", "module", "function"))
        if predicate_decl
        else None
    )
    arch = profile.get("arch")
    defaults = dict(request_decl.get("defaults") or {})

    out: list[Resolution] = []
    for shape in shapes:
        # Keys prefixed `_` are carried provenance, not request fields: they
        # travel with a shape so results can be split by origin, but the
        # request class would reject the key.
        fields = {
            **defaults,
            **{k: v for k, v in shape.items() if not k.startswith("_")},
        }
        if arch and "arch" not in fields:
            fields["arch"] = arch
        try:
            request = request_cls(**fields)
            spec = factory(request)
        except Exception as exc:
            raise ParityError(f"request/spec construction failed: {exc}") from exc
        if predicate is not None:
            supported, why = _predicate_result(
                predicate, spec, **({"arch": arch} if arch else {})
            )
            if not supported:
                out.append(Resolution(shape, reason=why, kind="declined"))
                continue
        out.append(Resolution(shape, spec=spec))
    return out


def knob_partition(resolutions: list[Resolution]) -> tuple[list[str], list[str]]:
    """(varies, constant) across the dispatcher's own decisions.

    A field the dispatcher resolves identically for every shape it serves is not a
    tuning axis -- it is a value the library ships.
    """
    served = [r.spec for r in resolutions if r.spec is not None]
    if not served:
        return [], []
    names = [f.name for f in dataclasses.fields(served[0])]
    varies, constant = [], []
    for name in names:
        values = {repr(getattr(spec, name)) for spec in served}
        (varies if len(values) > 1 else constant).append(name)
    return varies, constant


def _kernel_name(slug: str, spec, index: int) -> str:
    """A kernel name derived from the spec's own fields, whatever op this is.

    Hand-listing one op's field names collapses other ops' variants onto the
    same string, and nothing downstream catches it. Scalars only, with the
    index appended unconditionally.
    """
    parts = [slug]
    try:
        fields = [f.name for f in dataclasses.fields(spec)]
    except TypeError:  # not a dataclass; fall back to the index alone
        fields = []
    for name in fields:
        value = getattr(spec, name, None)
        if value is None or isinstance(value, (list, tuple, dict, set)):
            continue
        if isinstance(value, bool):
            # A bare 0/1 reads as a magnitude; the field name alone reads as a flag.
            if value:
                parts.append(_abbrev(name))
            continue
        parts.append(f"{_abbrev(name)}{value}")
    parts.append(f"v{index}")
    return "_".join(str(p) for p in parts)


def _abbrev(field: str) -> str:
    """`num_query_heads` -> `nqh`, `dtype` -> `dt`. Short, and stable per field."""
    words = [w for w in field.split("_") if w]
    if len(words) == 1:
        return words[0][:2]
    return "".join(w[0] for w in words)


def _policy_resolvers(profile: dict) -> dict:
    """Bind each policy-owned knob's resolver, once."""
    resolvers = {}
    for knob, decl in (profile.get("policies") or {}).items():
        resolvers[knob] = (
            _import(*_required(decl, f"policies.{knob}", "module", "function")),
            list(decl.get("args") or []),
        )
    return resolvers


def _specialization(profile: dict) -> dict:
    """The profile's `specialization` block, carried through to the emitted
    config so a machine without rocKE can say which metadata fields the
    producing compiler specialized on and how each is read off the builder.

    Without it the descriptors' `provenance.specialization_contract` cannot be
    written, so a knob resolved by a standalone policy callback needs an
    explicit builder-owned readout.
    """
    declaration = profile.get("specialization")
    if not isinstance(declaration, dict) or not declaration:
        raise ParityError(
            "the profile declares no 'specialization' block, so the emitted config "
            "cannot state which metadata fields the producing compiler specializes "
            "on. Declare 'metadata_fields' with a 'bindings' entry each, and "
            "'matcher_only_fields' for the rest."
        )
    bindings = declaration.get("bindings") or {}
    for knob in profile.get("policies") or {}:
        binding = bindings.get(knob)
        if not isinstance(binding, dict) or set(binding) not in ({"field"}, {"method"}):
            raise ParityError(
                f"'{knob}' is resolved here by a standalone policy callback but the "
                f"specialization block binds no builder-owned readout for it. Add "
                f"bindings.{knob} naming the attribute or zero-argument accessor on "
                f"the builder's own spec object, as {{'method': '<accessor>'}} or "
                f"{{'field': '<attr>'}}. There is no convention to infer one from, "
                f"and moving it to matcher_only_fields would claim the compiler does "
                f"not specialize on a field it does."
            )
    return copy.deepcopy(declaration)


def build_config(
    resolutions: list[Resolution], profile: dict, knobs: dict | None = None
) -> dict:
    """A generator config carrying one kernel per served shape, with every spec
    field the dispatcher set written out verbatim rather than transcribed.

    Policy-owned knobs need one more step: the dispatcher returns the shared
    spec and leaves arch-private knobs absent, meaning the kernel's policy
    decides at build time. The matcher compares metadata, where an absent knob
    falls back to the KMD default and names a different kernel, so the knob is
    resolved here rather than in the generator, which must not import rocKE.
    """
    slug = profile["slug"]
    metadata_fields = list(profile.get("metadata_fields") or [])
    vocabulary = dict(profile.get("vocabulary") or {})
    resolvers = _policy_resolvers(profile)
    # Resolved before any kernel is built, so a knob with no declared readout
    # is refused while the message can still name it.
    specialization = _specialization(profile)
    # Arch-private fields are absent from the shared spec the dispatcher
    # returns, but the engine may still read them from the catalog: the gfx942
    # matcher checks `seqlen_q % block_m == 0`, so a descriptor omitting it
    # states no tile at all. The builder's spec class carries the value the
    # binary is built with, so ask it.
    arch_decl = profile.get("arch_spec") or {}
    arch_defaults: dict = {}
    # Every field the builder's spec accepts, defaulted or not. Distinct from
    # arch_defaults: a pinned knob is written into the spec whenever the
    # builder would accept it, including fields whose default is MISSING.
    arch_field_names: set = set()
    if arch_decl:
        arch_cls = _import(*_required(arch_decl, "arch_spec", "module", "class"))
        for field in dataclasses.fields(arch_cls):
            arch_field_names.add(field.name)
            if field.default is not dataclasses.MISSING:
                arch_defaults[field.name] = field.default
    knobs = knobs or {}
    for knob, values in knobs.items():
        if knob not in metadata_fields:
            raise ParityError(
                f"--knobs names '{knob}', which this profile's metadata_fields does "
                f"not declare. An undeclared metadata field drops the WHOLE pack at "
                f"resolveDescriptorSets(), so crossing on one would emit a package "
                f"that cannot load. Declared: {', '.join(metadata_fields) or '(none)'}."
            )
        if not isinstance(values, list) or not values:
            raise ParityError(
                f"--knobs entry '{knob}' must be a non-empty list of values, got "
                f"{values!r}. An empty list's cross-product is empty, which would "
                f"silently emit ZERO kernels instead of failing here."
            )
        # A knob the builder's spec does not accept can only be written to
        # metadata, which makes both arms name the same binary under two
        # catalog entries; the sweep then reports "no effect" for a knob whose
        # other side was never compiled.
        if arch_field_names and knob not in arch_field_names:
            raise ParityError(
                f"--knobs names '{knob}', which the builder's spec class "
                f"({arch_decl.get('module')}.{arch_decl.get('class')}) does not "
                f"accept, so pinning it would change the catalog entry without "
                f"changing the compiled binary -- both arms would be the same "
                f"kernel and the sweep would measure nothing. Either the name is "
                f"wrong, or the field is not a build-time knob of this kernel."
            )
    kernels = []
    for index, resolution in enumerate(resolutions):
        if resolution.spec is None:
            continue
        spec = {
            f.name: getattr(resolution.spec, f.name)
            for f in dataclasses.fields(resolution.spec)
        }
        metadata = {}
        for name in metadata_fields:
            if name in resolvers and spec.get(name) is None:
                func, argnames = resolvers[name]
                try:
                    value = func(*[spec[a] for a in argnames])
                except KeyError as exc:
                    raise ParityError(
                        f"policy for '{name}' needs spec key {exc}, which the "
                        f"dispatcher's resolved spec does not carry."
                    )
            elif spec.get(name) is None and name in arch_defaults:
                value = arch_defaults[name]
            else:
                value = spec.get(name)
            if isinstance(value, bool):
                value = int(value)
            if name in vocabulary and isinstance(value, str):
                # The matcher compares the hipDNN spelling; the spec carries
                # the builder's. Copying one over the other declines every
                # graph while the engine still loads.
                mapping = vocabulary[name]
                if isinstance(mapping, dict):
                    value = mapping.get(value, value)
            metadata[name] = value
        # The shipping cross-product: dispatcher-resolved shapes crossed with
        # the knobs that earned a slot. Not a pack `axes:` block, since axes
        # cross one kernel_template and here every shape carries its own spec.
        # Knobs absent from --knobs keep their policy-resolved value; with no
        # --knobs this is a single empty combination, the parity set.
        axis_names = sorted(knobs)
        for combo in itertools.product(*[knobs[k] for k in axis_names]):
            pinned = dict(zip(axis_names, combo))
            variant_metadata = dict(metadata)
            variant_spec = dict(spec)
            for knob, value in pinned.items():
                variant_metadata[knob] = (
                    int(value) if isinstance(value, bool) else value
                )
                # Write it into the spec, not only the metadata: the spec is
                # what the builder compiles, and an arch-private knob is absent
                # from the shared spec, so a `knob in variant_spec` guard would
                # skip exactly the knobs worth sweeping.
                variant_spec[knob] = value
            name = _kernel_name(slug, resolution.spec, index)
            if pinned:
                name += "." + "_".join(f"{k}{pinned[k]}" for k in axis_names)
            kernels.append(
                {
                    "name": name,
                    "kernel_source": {
                        "kind": "rocke",
                        "source": profile["source"],
                        "builder": profile["builder"],
                        "spec": variant_spec,
                    },
                    "metadata": variant_metadata,
                }
            )
    # `dialect: packaged` is stated rather than guessed: a rocKE builder can
    # only be authored packaged, and the loader rejects any other pairing.
    return {
        "dialect": profile.get("dialect", "packaged"),
        "authored_subpath": profile.get("authored_subpath", f"rocKE/{slug}"),
        "engine": profile["engine"],
        "kmd_fields": profile["kmd_fields"],
        "specialization": specialization,
        "kernel_source_kind": profile.get("kernel_source_kind", "rocke"),
        "workspace_policy": profile.get("workspace_policy", "none"),
        "packs": [
            {
                "name": profile.get("pack", slug),
                "arch": [profile["arch"]],
                "kernels": kernels,
            }
        ],
    }


def _compact(config: dict, knob_fields: list, profile: dict) -> str:
    """The enumerated config, rewritten as `variants` and rendered.

    Shared with the retrofit path (`factorise_config.py`) so the two cannot
    disagree about what a compact config means.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from factorise_config import FactoriseError, _round_trip, dump, factorise

    try:
        compact = factorise(config, list(knob_fields), profile.get("vocabulary") or {})
        # Not optional: the compact form is what ships, so it is checked
        # against the enumeration it stands for. That check also catches a
        # kernel-name collision before the loader's pack-level check does.
        _round_trip(config, compact)
    except FactoriseError as exc:
        raise ParityError(
            f"the emitted set could not be written in the compact `variants` form: "
            f"{exc}"
        )
    return dump(compact)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the dispatcher-resolved variant set (stage 1 parity).",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Kernel profile naming the dispatcher, request class, "
        "predicate and descriptor skeleton.",
    )
    parser.add_argument(
        "--shapes", required=True, help="JSON list of request-field mappings."
    )
    parser.add_argument("--out", help="Write the generator config here.")
    parser.add_argument(
        "--report-knobs",
        action="store_true",
        help="Print which spec fields vary across the dispatcher's "
        "decisions and which are constant.",
    )
    parser.add_argument(
        "--report-gaps",
        action="store_true",
        help="Print every shape the dispatcher would not serve, with "
        "its reason and which layer refused.",
    )
    parser.add_argument(
        "--knobs",
        help="JSON mapping of knob name to the list of values that SURVIVED the "
        "sweep, e.g. '{\"use_exp2_fast\": [0, 1]}'. The dispatcher-resolved set is "
        "crossed with it to build the shipping package (RUNBOOK §6). Omit it and "
        "you get the parity set: one kernel per servable shape.",
    )
    args = parser.parse_args(argv)

    try:
        profile = _load_profile(args.profile)
        _bind_provider(profile.get("provider_root"))
        shapes = json.loads(Path(args.shapes).read_text())
        if not isinstance(shapes, list):
            raise ParityError("--shapes must be a JSON list of field mappings.")
        resolutions = resolve_shapes(shapes, profile)
    except ParityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    served = [r for r in resolutions if r.spec is not None]
    declined = [r for r in resolutions if r.kind == "declined"]

    print("dispatcher parity")
    print(f"  shapes in         {len(resolutions)}")
    print(f"  servable          {len(served)}")
    print(f"  declined          {len(declined)}  (predicate said no)")

    if args.report_gaps:
        for resolution in declined:
            print(f"    [{resolution.kind}] {resolution.shape} -- {resolution.reason}")

    if args.report_knobs:
        varies, constant = knob_partition(resolutions)
        print("\n  VARIES across dispatch decisions -- the tuning surface:")
        print(f"    {', '.join(varies) or '(none)'}")
        print("\n  CONSTANT -- shipped values, NOT tuning axes:")
        print(f"    {', '.join(constant) or '(none)'}")
        print(
            "\n  A knob the dispatcher fixes is not an axis. Sweeping one measures a\n"
            "  configuration rocKE would never resolve to."
        )

    if not served:
        print(
            "\nFAIL: no shape resolved; there is nothing to generate.", file=sys.stderr
        )
        return 1

    if args.out:
        try:
            knobs = json.loads(args.knobs) if args.knobs else {}
            if not isinstance(knobs, dict):
                raise ParityError(
                    f"--knobs must be a JSON mapping of knob name to a list of "
                    f"values, got {type(knobs).__name__}."
                )
            config = build_config(resolutions, profile, knobs)
            # Emit the compact form; build_config stays the source of truth and
            # `_compact` refuses anything that does not re-expand
            # kernel-for-kernel. The knob axes are exactly --knobs, since the
            # dispatcher returns one spec per shape.
            text = _compact(config, sorted(knobs), profile)
        except json.JSONDecodeError as exc:
            print(f"FAIL: --knobs is not valid JSON: {exc}", file=sys.stderr)
            return 2
        except ParityError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        Path(args.out).write_text(text)
        count = len(config["packs"][0]["kernels"])
        if knobs:
            arms = math.prod(len(v) for v in knobs.values())
            print(
                f"\n  wrote {args.out}: {count} kernels "
                f"= {len(served)} servable shapes x {arms} surviving knob "
                f"combination(s) ({', '.join(sorted(knobs))})"
            )
            # The cap the runbook states, enforced where the number is known:
            # past the low thousands the pack time, the archive and the catalog
            # all stop being reasonable.
            if count > 4000:
                print(
                    f"  WARNING: {count} descriptors is past the low-thousands cap. "
                    f"Cut axes, not shapes -- a knob that did not earn its slot in "
                    f"isolation will not earn it in the cross-product.",
                    file=sys.stderr,
                )
        else:
            print(f"\n  wrote {args.out}: {count} kernels, one per servable shape")
            if count != len(served):
                print(
                    f"  NOTE: {len(served)} shapes resolved but {count} kernels "
                    f"emitted -- distinct shapes sharing one resolved spec are one "
                    f"variant."
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
