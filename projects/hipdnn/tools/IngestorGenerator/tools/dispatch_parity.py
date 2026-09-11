"""Generate the variant set rocKE's own dispatcher would resolve -- one command.

STAGE 1 OF AN INTEGRATION, and the only configuration whose correctness argues from
rocKE's behaviour rather than from measurement. Everything after it is a deviation
that has to be justified; this is the thing deviations are measured against.

It answers the question the mining sources do not. The validators, the spec
dataclass and the ``supports_*`` predicate all answer "what is LEGAL?". The
dispatcher is the only source that says "what does the library itself SHIP for this
request?", and the difference is not academic: a field the dispatcher DERIVES from
the request (``persistent = work >= num_persistent``) reads as an ordinary local
variable, gets missed by a human transcribing constants, and silently takes the
dataclass default instead -- which was the opposite of the dispatcher's answer on
62 of 145 variants, for a lever the kernel's own notes mark "KEEP everywhere". No
gate caught it. Descriptors validated, the desk check was clean, correctness passed
on device; the only symptom was a performance number, misattributed three times.

Calling the factory cannot make that mistake. A rule is applied rather than read.

WHAT IT EMITS. One variant per servable shape, at the dispatcher's own resolved
spec, as a generator config ready for ``generate.py``. Not a cross-product: the
dispatcher returns exactly one spec per shape, and that IS the tuning surface rocKE
exposes for the op.

    dispatch_parity.py --profile <profile.yaml> --shapes <corpus.json> \\
                       --out configs/<slug>_A.yaml

THE TWO DENOMINATORS. "Does the kernel support this shape?" has two answers and
they differ by 59 on the published corpus. A support check that only calls the
predicate misses every shape rejected at SPEC CONSTRUCTION -- those raise
``ValueError`` before a predicate ever runs, so the constructor must be inside the
try. Both rejection kinds are reported here, separately and by reason, because
"uncovered and unexplained" is the state that hides a defect: an uncovered servable
shape is a defect until proven otherwise, and the proof is this cheap.

WHAT IT DELIBERATELY DOES NOT DO. It does not sweep. A knob that is CONSTANT across
every dispatch decision is not a tuning axis -- it is a value rocKE ships -- and
--report-knobs prints that partition so a sweep can start from what actually varies
instead of from a cross-product of everything nameable. On gfx942 attention_dense
the surface is 8 shape fields plus waves_per_eu and persistent; the other twelve
knobs are dispatch-invisible, and the ones a commit message says were "swept" were
explored, not shipped.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import sys
from pathlib import Path


class ParityError(RuntimeError):
    """The dispatcher could not be reached or asked. Never a shape-level decline."""


def _load_profile(path: str) -> dict:
    """Parse a profile as JSON, falling back to YAML.

    The mapping check covers BOTH paths. It used to sit only on the YAML branch, so a
    file that parsed as valid JSON but was not an object -- a bare list, say -- sailed
    through and crashed several frames later with `AttributeError: 'list' object has
    no attribute 'get'`, which names neither the file nor the problem. Sibling tools
    (verify_variant_sets, variant_reachability) already had it outside the try.
    """
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
    #: "constructed" (spec built and predicate accepted), "declined" (predicate said
    #: no), or "rejected" (spec construction raised -- the answer the predicate alone
    #: never sees).
    kind: str = "constructed"


def _required(decl: dict, scope: str, *keys: str) -> list:
    """Pull `keys` out of a profile block, naming the block when one is missing.

    A bare `decl["module"]` raises `KeyError: 'module'` with no indication of WHICH
    profile block was incomplete -- and a profile has several blocks that all take a
    `module`. The tool has a named error type for exactly this kind of thing; this
    makes the profile-shape failures use it too.
    """
    missing = [k for k in keys if k not in decl]
    if missing:
        raise ParityError(
            f"the profile's '{scope}' block is missing {missing}. It needs "
            f"{list(keys)} so the tool knows what to import."
        )
    return [decl[k] for k in keys]


def resolve_shapes(shapes: list[dict], profile: dict) -> list[Resolution]:
    """Ask the dispatcher for every shape, keeping both kinds of refusal apart."""
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
        # Keys prefixed `_` are carried metadata, not request fields. Provenance
        # travels with a shape so a result can be split by where the shape came
        # from -- the split that turned "the win is synthetic" from a suspicion
        # into a measurement -- but the request class would reject the key.
        fields = {
            **defaults,
            **{k: v for k, v in shape.items() if not k.startswith("_")},
        }
        if arch and "arch" not in fields:
            fields["arch"] = arch
        try:
            # The constructor is INSIDE the try on purpose. Structural rejections --
            # a decode Sq the block size cannot divide, an unsupported head_size --
            # raise here, before any predicate runs. Calling only the predicate
            # reports those shapes as supported and ships a wrong denominator.
            request = request_cls(**fields)
            spec = factory(request)
        except Exception as exc:
            out.append(
                Resolution(
                    shape, reason=f"{type(exc).__name__}: {exc}", kind="rejected"
                )
            )
            continue
        if predicate is not None:
            supported, why = predicate(spec, arch=arch) if arch else predicate(spec)
            if not supported:
                out.append(
                    Resolution(
                        shape, reason=str(why) or "predicate declined", kind="declined"
                    )
                )
                continue
        out.append(Resolution(shape, spec=spec))
    return out


def knob_partition(resolutions: list[Resolution]) -> tuple[list[str], list[str]]:
    """(varies, constant) across the dispatcher's own decisions.

    The mechanical form of "which knobs may be exposed". A field the dispatcher
    resolves identically for every shape it serves is not an axis -- it is a value
    the library ships. Sweeping it measures a configuration rocKE would never pick.
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
    """A name derived from the spec's OWN fields, whatever op this is.

    The first version listed attention's field names and abbreviated those it found.
    On any other op it found none of them and every variant collapsed onto the same
    string -- two distinct conv variants both named `conv_fwd_dtfp16`. Nothing
    downstream catches that: the config loader checks PACK name uniqueness, not
    kernel names, and de-duplication keys on metadata rather than name, so the
    colliding entries ship as separate descriptors that are impossible to tell apart
    in a log, a winner record or a failure message.

    So the fields come from the dataclass. Scalars only -- a name is an identifier,
    not a serialisation -- and the index is appended unconditionally, because a name
    built from a subset of fields is only unique by luck and this tool cannot know
    which subset a given kernel varies over.
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


def build_config(
    resolutions: list[Resolution], profile: dict, knobs: dict | None = None
) -> dict:
    """A generator config carrying one kernel per served shape.

    Every spec field the dispatcher set is written out verbatim. That is the point:
    a derived field is indistinguishable from a constant once it is a value, and the
    only way to be sure one was not missed is to never transcribe any of them.

    POLICY-OWNED KNOBS need one more step. The dispatcher returns the SHARED spec
    and deliberately leaves gfx942-private knobs alone -- `use_exp2_fast` is absent
    from it entirely, meaning "the kernel's policy decides at build time". The
    binary is still definite, so the descriptor must SAY which one it is: the
    matcher compares metadata, and a knob absent there resolves to the KMD default,
    which is a different kernel. Resolving it here is right and resolving it in the
    generator would be wrong -- the generator must not import rocKE, and guessing a
    policy without asking it is what shipped an explicit `false` over a policy that
    answers True above a sequence-length threshold, throwing away the win it was
    measured to give.
    """
    slug = profile["slug"]
    metadata_fields = list(profile.get("metadata_fields") or [])
    vocabulary = dict(profile.get("vocabulary") or {})
    resolvers = _policy_resolvers(profile)
    # Arch-PRIVATE fields are absent from the shared spec the dispatcher returns, but
    # the engine may still read them from the catalog: the gfx942 matcher checks
    # `seqlen_q % block_m == 0` and prepare() passes block_m to the grid helper, so a
    # descriptor omitting it states no tile at all -- while the C++ compiles fine and
    # the omission is invisible. The builder's own spec class carries the value the
    # binary is actually built with, so ask it rather than leaving a hole.
    arch_decl = profile.get("arch_spec") or {}
    arch_defaults: dict = {}
    # Every field the BUILDER's spec accepts, defaulted or not. Distinct from
    # arch_defaults: a pinned knob must be written into the spec whenever the
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
        # A knob the BUILDER's spec does not accept can only ever be written to
        # metadata, which makes both arms name the SAME binary under two catalog
        # entries: identical kernels, one of which the matcher will prefer for
        # reasons that have nothing to do with the knob. The sweep then measures
        # 1.000x and reports "no effect" for a knob whose other side was never
        # compiled. Refuse rather than emit that, and say which of the two real
        # cases the author is in -- a typo, or a knob that needs the arch spec
        # promoted the way the builder does it.
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
                # The matcher compares the hipDNN spelling; the spec carries the
                # builder's. Copying one over the other declines every graph while
                # the engine still loads and every count reconciles.
                mapping = vocabulary[name]
                if isinstance(mapping, dict):
                    value = mapping.get(value, value)
            metadata[name] = value
        # The SHIPPING cross-product. Stage 4a-3 builds the package from the knobs
        # that measurably earned a slot, over the shapes the dispatcher resolves --
        # so the base set stays dispatcher-derived (never hand-transcribed) and only
        # the surviving knobs multiply it.
        #
        # This cannot be expressed as a pack `axes:` block: axes cross ONE
        # kernel_template, and here every shape carries its own resolved spec. The
        # cross-product therefore has to happen where the specs are, which is here.
        #
        # A pinned knob overrides what the policy resolved above, and that is the
        # point of sweeping it -- but only for knobs the author listed. Everything
        # absent from --knobs keeps its policy-resolved value, because pinning a knob
        # the kernel resolves by policy DISCARDS the policy, and a generated set can
        # be strictly worse than a smaller one exactly that way.
        # With no --knobs this is a single empty combination: the parity set, one
        # kernel per servable shape, name and payload byte-identical to what this
        # tool emitted before --knobs existed.
        axis_names = sorted(knobs)
        for combo in itertools.product(*[knobs[k] for k in axis_names]):
            pinned = dict(zip(axis_names, combo))
            variant_metadata = dict(metadata)
            variant_spec = dict(spec)
            for knob, value in pinned.items():
                variant_metadata[knob] = (
                    int(value) if isinstance(value, bool) else value
                )
                # Write it into the SPEC, not only the metadata. The spec is what
                # the builder compiles, and an arch-private knob is absent from the
                # shared spec the dispatcher returns -- so a `knob in variant_spec`
                # guard silently skips exactly the knobs most worth sweeping, and
                # both arms build one binary. The refusal above guarantees the
                # builder accepts this field, so setting it is always legal.
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
    # `dialect: packaged` is not a default worth guessing: a rocKE builder can only
    # be authored packaged (hkp_pack lowers it through comgr at build time and
    # rewrites the descriptor to kind: kpack), and the loader rejects the pairing
    # outright otherwise. Stating it here means the emitted config is directly
    # generatable rather than a skeleton someone has to finish.
    return {
        "dialect": profile.get("dialect", "packaged"),
        "authored_subpath": profile.get("authored_subpath", f"rocKE/{slug}"),
        "engine": profile["engine"],
        "kmd_fields": profile["kmd_fields"],
        "kernel_source_kind": profile.get("kernel_source_kind", "rocke"),
        "workspace_policy": profile.get("workspace_policy", "none"),
        "delegates_to_existing_plan": profile.get("delegates_to_existing_plan", False),
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

    A fully-enumerated set is one YAML block per kernel: the largest shipped gfx942
    attention_dense config was 89,265 lines for 2,710 kernels, and a config nobody
    can read is a config nobody checks -- while being the ONE file in a descriptor
    PR worth reviewing, since the descriptors are its deterministic output.

    Shared with the retrofit path (`factorise_config.py`) rather than reimplemented,
    so the two cannot disagree about what a compact config means. The factoriser
    verifies its own output by re-expanding it, so a set that cannot be compacted
    losslessly fails here instead of shipping a config that generates something else.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from factorise_config import FactoriseError, _round_trip, dump, factorise

    try:
        compact = factorise(config, list(knob_fields), profile.get("vocabulary") or {})
        # Not optional. The compact form is what ships, so it has to be checked
        # against the enumeration it stands for -- and that check is also what
        # catches a kernel-name collision before the loader's own uniqueness check
        # rejects the whole pack -- and it names the inference that caused it.
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
        "crossed with it to build the shipping package (stage 4a-3). Omit it and "
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
    rejected = [r for r in resolutions if r.kind == "rejected"]

    print("dispatcher parity")
    print(f"  shapes in         {len(resolutions)}")
    print(f"  servable          {len(served)}")
    print(f"  declined          {len(declined)}  (predicate said no)")
    print(f"  rejected          {len(rejected)}  (spec construction raised)")

    if args.report_gaps:
        for resolution in declined + rejected:
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
            # Emit the COMPACT form. build_config stays the source of truth --
            # every spec field the dispatcher set, written out, never transcribed --
            # and factorise_config collapses the result into the shape x knob-set
            # form a reviewer can read. One mechanism, not two: the factoriser
            # re-expands what it wrote and refuses to emit anything that does not
            # reproduce the enumeration kernel-for-kernel, so the compact config and
            # the longhand one it stands for cannot drift.
            #
            # The knob axes are exactly --knobs: the dispatcher returns ONE spec per
            # shape, so those are the only fields that vary within a shape.
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
            # The cap the runbook's 4a-3 gate states, enforced where the number is
            # actually known. Past the low thousands the pack time, the archive and
            # the catalog all stop being reasonable, and the marginal variant is
            # almost never the one that wins.
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
