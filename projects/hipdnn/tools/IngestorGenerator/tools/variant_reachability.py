"""The converse of the desk check: can any graph select this variant at all?

Applicability and ranking are declared (`--divides`, `--score-field`), never
inferred from field names: the scorer is native C++ per engine and this tool
cannot call it. So a corpus field no rule names is invisible to applicability,
and without a declared ranking every applicable variant is reported reachable
-- the output says so rather than implying the native scorer was checked.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import solely for the shared descriptor package's path bootstrap.
import verify_variant_sets  # noqa: E402, F401

from hkp_pack import descriptor_context  # noqa: E402


class ReachabilityError(RuntimeError):
    """A problem with the inputs (bundle, corpus, or ranking declaration) --
    never a finding about the variants themselves. Findings are reported, not
    raised."""


def _load_profile(path: str) -> dict:
    """JSON or YAML, like `dispatch_parity.py`'s loader -- one profile can serve
    both tools without restating itself in two dialects."""
    text = Path(path).read_text()
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError:  # pragma: no cover - environment-dependent
            raise ReachabilityError(f"{path} is not JSON and PyYAML is not installed.")
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ReachabilityError(f"profile {path} must be a mapping.")
    return loaded


def load_bundle(kdp_path: str, tree: str | None = None) -> tuple[dict, list[dict]]:
    """(name -> KMD default_value, kernelDescriptors) for one *.kdp.json.

    The schema is reached by reference through the id chain the documents
    declare, resolved across `tree` by `hkp_pack.descriptor_context`. Its
    defaults make "wrote the default explicitly" and "left it absent" the same
    variant at runtime -- see `_resolved_metadata`.
    """
    kdp = Path(kdp_path)
    if not kdp.name.endswith(".kdp.json"):
        raise ReachabilityError(f"{kdp} does not look like a *.kdp.json")
    root = Path(tree) if tree else kdp.parent
    try:
        bundles = descriptor_context.resolve_bundles(
            descriptor_context.Index(str(root))
        )
    except descriptor_context.DescriptorContextError as exc:
        raise ReachabilityError(str(exc))
    matching = [b for b in bundles if Path(b.kdp_path).resolve() == kdp.resolve()]
    if not matching:
        raise ReachabilityError(
            f"{kdp} is not among the KDPs under {root}. Pass --tree naming the root "
            f"the bundle's generics live under, so its engine and schema references "
            f"can be resolved."
        )
    bundle = matching[0]
    defaults = {f["name"]: f.get("default_value") for f in bundle.kmd["fields"]}
    return defaults, [entry.ukd for entry in bundle.entries]


def _resolved_metadata(descriptor: dict, defaults: dict) -> dict:
    """`metadata` with absent KMD fields filled from their default -- the tuple
    the loader itself would compare, per `verify_variant_sets.py`'s own rule."""
    meta = dict(descriptor.get("metadata", {}))
    for name, default in defaults.items():
        meta.setdefault(name, default)
    return meta


def _remap(shape: dict, field_map: dict) -> dict:
    """Rename a corpus field to the metadata name it corresponds to. The corpus
    speaks its producer's vocabulary and KMD metadata the matcher's; where they
    differ, --field-map declares the mapping rather than this tool guessing."""
    out = dict(shape)
    for old, new in field_map.items():
        if old in out:
            out[new] = out.pop(old)
    return out


def _same_value(shape_value, metadata_value) -> bool:
    """Are these the same value, allowing for the two vocabularies? Numbers
    compare numerically (a bool is an int, so metadata 1 and corpus True are
    one graph); strings compare case-insensitively, for the reason `applicable`
    gives."""
    if isinstance(shape_value, str) and isinstance(metadata_value, str):
        return shape_value.strip().lower() == metadata_value.strip().lower()
    if isinstance(shape_value, (int, float)) and isinstance(
        metadata_value, (int, float)
    ):
        return float(shape_value) == float(metadata_value)
    return shape_value == metadata_value


def applicable(metadata: dict, shape: dict, divides: dict) -> bool:
    """True when `metadata` (a variant, defaults resolved) is legal for `shape`.

    Two kinds of shape-valued field:

      * a metadata field sharing a name with a shape field must be equal to it,
        unless that field is declared as a divisor;
      * a declared divisor field must evenly divide the shape field it is
        declared against, and non-positive tiles never divide anything.

    A field the caller never declared and sharing no name with any shape key is
    invisible here. String comparison is case-insensitive: metadata carries the
    matcher's spelling (``"BF16"``) and a corpus the builder's (``"bf16"``).
    """
    for field, value in metadata.items():
        if field in divides:
            continue
        if field not in shape:
            continue
        if not _same_value(shape[field], value):
            return False
    for field, of in divides.items():
        if field not in metadata or of not in shape:
            continue  # nothing this corpus can test this rule against
        value = metadata[field]
        target = shape[of]
        if not isinstance(value, (int, float)) or not isinstance(target, (int, float)):
            raise ReachabilityError(
                f"--divides {field}={of} needs numeric values; got "
                f"{value!r} / {target!r}"
            )
        if value <= 0 or target % value != 0:
            return False
    return True


@dataclasses.dataclass
class Verdict:
    name: str
    bucket: str  # "SELECTED" | "APPLICABLE-BUT-NEVER-WINS" | "UNREACHABLE"
    applicable_shapes: list[int]
    won_shapes: list[int]
    #: names of variants that outranked this one at every shape it could have
    #: won, for the APPLICABLE-BUT-NEVER-WINS diagnostic. Empty otherwise.
    always_beaten_by: list[str]


def classify(
    defaults: dict,
    descriptors: list[dict],
    shapes: list[dict],
    divides: dict,
    field_map: dict,
    score: dict | None,
) -> list[Verdict]:
    """One Verdict per descriptor, against the whole corpus. `score` is
    `{"field": ..., "prefer": "max" | "min"}` or None; None means no ranking
    was declared, so every applicable variant wins and UNREACHABLE is the only
    finding left."""
    if score is not None and score.get("prefer") not in ("max", "min"):
        raise ReachabilityError("score.prefer must be 'max' or 'min'")

    remapped = [_remap(s, field_map) for s in shapes]
    metas = {d["name"]: _resolved_metadata(d, defaults) for d in descriptors}

    # shape index -> [variant names applicable there]
    applicable_at: list[list[str]] = [
        [name for name, meta in metas.items() if applicable(meta, s, divides)]
        for s in remapped
    ]

    # shape index -> [variant names that win there] (ties all win: without the
    # real scorer's tie-break this tool cannot rule either side out, and
    # picking one would manufacture a false APPLICABLE-BUT-NEVER-WINS).
    winners_at: list[list[str]] = []
    for idx, names in enumerate(applicable_at):
        if not names:
            winners_at.append([])
            continue
        if score is None:
            winners_at.append(list(names))
            continue
        field, prefer = score["field"], score["prefer"]
        values = {}
        for name in names:
            if field not in metas[name]:
                raise ReachabilityError(
                    f"variant '{name}' is applicable to shape {idx} but has no "
                    f"'{field}' to score it by -- the declared ranking does not "
                    f"cover every applicable variant."
                )
            values[name] = metas[name][field]
        best = max(values.values()) if prefer == "max" else min(values.values())
        winners_at.append([n for n, v in values.items() if v == best])

    verdicts = []
    for name in metas:
        applicable_shapes = [
            i for i, names in enumerate(applicable_at) if name in names
        ]
        won_shapes = [i for i in applicable_shapes if name in winners_at[i]]
        if not applicable_shapes:
            verdicts.append(Verdict(name, "UNREACHABLE", [], [], []))
        elif won_shapes:
            verdicts.append(
                Verdict(name, "SELECTED", applicable_shapes, won_shapes, [])
            )
        else:
            rivals: set[str] = set()
            for i in applicable_shapes:
                rivals.update(winners_at[i])
            verdicts.append(
                Verdict(
                    name,
                    "APPLICABLE-BUT-NEVER-WINS",
                    applicable_shapes,
                    [],
                    sorted(rivals),
                )
            )
    return verdicts


def _parse_kv(pairs: list[str], sep: str = "=") -> dict:
    out = {}
    for pair in pairs:
        if sep not in pair:
            raise ReachabilityError(f"expected KEY{sep}VALUE, got {pair!r}")
        k, v = pair.split(sep, 1)
        out[k] = v
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="For every shipped variant: is there a graph that could "
        "select it? (the converse of the desk check)",
    )
    parser.add_argument("--kdp", required=True, help="Path to a *.kdp.json bundle.")
    parser.add_argument(
        "--tree",
        help="Descriptor root the bundle's engine and schema references resolve "
        "across. Defaults to the --kdp file's own directory; name the root "
        "explicitly when the generics live above it.",
    )
    parser.add_argument(
        "--shapes",
        required=True,
        help="JSON list of request-field mappings, the same corpus format "
        "dispatch_parity.py --shapes consumes.",
    )
    parser.add_argument("--profile", help="JSON/YAML with field_map/divides/score.")
    parser.add_argument(
        "--divides",
        action="append",
        default=[],
        metavar="METADATA_FIELD=SHAPE_FIELD",
        help="A tile-style knob: applicable when it evenly divides the named "
        "shape field, not when it equals one. Repeatable.",
    )
    parser.add_argument(
        "--field-map",
        action="append",
        default=[],
        metavar="SHAPE_FIELD=METADATA_FIELD",
        help="Rename a corpus field before comparing, when the corpus and the "
        "metadata spell the same axis differently. Repeatable.",
    )
    parser.add_argument("--score-field", help="Metadata field the scorer ranks by.")
    parser.add_argument(
        "--score-prefer",
        choices=("max", "min"),
        help="Which end of --score-field wins. Required if --score-field is given.",
    )
    parser.add_argument(
        "--allow-unreachable",
        action="store_true",
        help="Do not fail the exit code on APPLICABLE-BUT-NEVER-WINS or "
        "UNREACHABLE findings; still reports them.",
    )
    args = parser.parse_args(argv)

    try:
        profile = _load_profile(args.profile) if args.profile else {}
        divides = dict(profile.get("divides") or {})
        divides.update(_parse_kv(args.divides))
        field_map = dict(profile.get("field_map") or {})
        field_map.update(_parse_kv(args.field_map))

        score = profile.get("score")
        if args.score_field:
            if not args.score_prefer:
                raise ReachabilityError("--score-field needs --score-prefer")
            score = {"field": args.score_field, "prefer": args.score_prefer}
        if score is not None and ("field" not in score or "prefer" not in score):
            raise ReachabilityError("score needs both 'field' and 'prefer'")

        defaults, descriptors = load_bundle(args.kdp, args.tree)
        shapes = json.loads(Path(args.shapes).read_text())
        if not isinstance(shapes, list):
            raise ReachabilityError("--shapes must be a JSON list of field mappings.")

        verdicts = classify(defaults, descriptors, shapes, divides, field_map, score)
    except ReachabilityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    by_bucket: dict[str, list[Verdict]] = {
        "SELECTED": [],
        "APPLICABLE-BUT-NEVER-WINS": [],
        "UNREACHABLE": [],
    }
    for v in verdicts:
        by_bucket[v.bucket].append(v)

    print("variant reachability (the converse of the desk check)")
    print(f"  variants          {len(verdicts)}")
    print(f"  shapes in corpus  {len(shapes)}")
    if score is None:
        print(
            "  NO RANKING DECLARED (no --score-field/--profile score) -- every "
            "applicable variant is reported reachable. This tool did NOT verify "
            "which one the native scorer would actually pick."
        )
    else:
        print(f"  ranking declared  {score['field']} ({score['prefer']} wins)")
    print()

    print(f"  SELECTED                    {len(by_bucket['SELECTED'])}")
    print(
        f"  APPLICABLE-BUT-NEVER-WINS   {len(by_bucket['APPLICABLE-BUT-NEVER-WINS'])}"
    )
    print(f"  UNREACHABLE                 {len(by_bucket['UNREACHABLE'])}")

    if by_bucket["APPLICABLE-BUT-NEVER-WINS"]:
        print(
            "\n  APPLICABLE-BUT-NEVER-WINS: legal for at least one shape, and always "
            "outranked there.\n  The fix is a shape where the rival below is ILLEGAL, "
            "not another variant --\n  this axis already has one, and it cannot be "
            "measured until something can pick it."
        )
        for v in by_bucket["APPLICABLE-BUT-NEVER-WINS"]:
            beaten_by = ", ".join(v.always_beaten_by) or "(nothing recorded)"
            print(
                f"    {v.name}: applicable to {len(v.applicable_shapes)} shape(s), "
                f"always beaten by: {beaten_by}"
            )

    if by_bucket["UNREACHABLE"]:
        print(
            "\n  UNREACHABLE: applicable to no corpus shape at all. Either the "
            "corpus is\n  missing a shape family, or this variant should not have "
            "been built."
        )
        for v in by_bucket["UNREACHABLE"]:
            print(f"    {v.name}")

    dead_weight = by_bucket["APPLICABLE-BUT-NEVER-WINS"] or by_bucket["UNREACHABLE"]
    print()
    if dead_weight and not args.allow_unreachable:
        print(
            f"FAIL: {len(dead_weight)} variant(s) no graph in this corpus can select."
        )
        return 1
    if dead_weight:
        print(
            f"PASSED with --allow-unreachable: {len(dead_weight)} variant(s) still "
            f"unreachable, exit code suppressed."
        )
        return 0
    print("PASSED: every variant wins somewhere in this corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
