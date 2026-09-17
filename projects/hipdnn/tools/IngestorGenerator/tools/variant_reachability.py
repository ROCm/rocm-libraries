"""The converse of the desk check: can any graph SELECT this variant at all?

Every existing check asks "does a shipped variant match this graph?" (the desk
check, `hkp_desk_check.py`) or "is the set internally consistent?"
(`verify_variant_sets.py`). Neither one is ever asked backwards, and the backwards
question is where dead weight hides: a real integration shipped 48 variants of
which 24 could not be selected by ANY graph the author could write. The cause was
not laziness -- every shipped shape had a sequence length divisible by the wider
tile, so both tiles were always APPLICABLE and the scorer, which ranks the wider
tile higher, chose it every single time. Half the tuning axis was unreachable and
the suite was green throughout, because nothing had ever asked "for the narrow
tile, is there a shape where it wins?"

A variant no graph can select is not neutral. It still costs a compile, a slot in
the catalog, and a benchmark run to advertise a choice that does not exist.

THE MODEL, and its two honest gaps.

Selection happens in two stages and this tool models both, badly on purpose where
it must be honest about it:

  1. APPLICABILITY. A variant's shape-valued metadata (KMD default substituted for
     anything absent, exactly as the loader substitutes it) must be consistent
     with the graph's shape. Most fields compare by EQUALITY on a shared name
     (dtype, head counts, ...). Tile-style knobs compare by DIVISIBILITY instead
     -- `block_n` does not equal a shape field, it must evenly divide one
     (`seqlen_kv % block_n == 0`), and a corpus where every shape is divisible by
     every shipped tile is exactly the historical failure. Divisibility rules are
     therefore DECLARED (`--divides block_n=seqlen_kv`), never guessed: guessing
     which fields are tiles from their names would be exactly the kind of silent
     assumption this tool exists to refuse to make.

  2. SCORING. Among the applicable variants for one shape, something ranks them
     and picks a winner. That something is native C++ per engine -- this tool
     cannot call it and does not pretend to. The ranking is instead DECLARED
     (`--score-field block_n --score-prefer max`, or the same under `score:` in a
     --profile), same spirit as `verify_variant_sets.py`'s policy resolvers: a
     fact only the kernel knows, supplied rather than invented. WITHOUT a
     declared ranking every applicable variant is reported reachable, and the
     output SAYS the ranking was not declared -- a gate that quietly stops
     checking a property is worse than one that admits it never checked it.

THREE BUCKETS, not two, because "applicable" and "selected" are different claims:

  * SELECTED -- wins outright for at least one corpus shape (or a ranking was
    never declared, in which case "applicable" and "wins" are the same claim by
    construction, and the output says so).
  * APPLICABLE-BUT-NEVER-WINS -- the dangerous one. The variant is legal for at
    least one shape, and something else always outranks it there. This is the
    24-of-48 case exactly: the fix is a shape where the RIVAL is illegal, not
    another variant, and the diagnostic says that rather than "add coverage".
  * UNREACHABLE -- applicable to nothing in the corpus at all. Either the corpus
    is missing a shape family or the variant should never have been built.

WHAT THIS CANNOT KNOW. A shape corpus field this tool was not told to compare
(no shared name, not named in --divides) is invisible to applicability -- the
corpus and the declared rules are the only inputs, and a rule nobody declared is
a rule this tool cannot enforce. A declared ranking field absent from an
applicable variant's metadata is a configuration error, not a silent "call it a
tie": scoring on a value that is not there is not scoring, it is guessing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


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


def load_bundle(kdp_path: str) -> tuple[dict, list[dict]]:
    """(name -> KMD default_value, kernelDescriptors) for one *.kdp.json.

    The KMD is read for its defaults, not its schema in the abstract: the loader
    substitutes `default_value` for any field a descriptor's metadata omits, and
    two descriptors that differ only in "wrote the default explicitly" vs. "left
    it absent" are the SAME variant at runtime. Comparing raw `metadata` dicts
    instead would report them as differently-shaped and could hide or invent an
    applicability difference that is not real.
    """
    kdp = Path(kdp_path)
    if not kdp.name.endswith(".kdp.json"):
        raise ReachabilityError(f"{kdp} does not look like a *.kdp.json")
    kmd = kdp.with_name(kdp.name[: -len(".kdp.json")] + ".kmd.json")
    if not kmd.exists():
        raise ReachabilityError(
            f"{kdp} has no sibling {kmd.name}; cannot read the schema."
        )
    kmd_doc = json.loads(kmd.read_text())
    defaults = {f["name"]: f.get("default_value") for f in kmd_doc["fields"]}
    descriptors = json.loads(kdp.read_text())["kernelDescriptors"]
    return defaults, descriptors


def _resolved_metadata(descriptor: dict, defaults: dict) -> dict:
    """`metadata` with absent KMD fields filled from their default -- the tuple
    the loader itself would compare, per `verify_variant_sets.py`'s own rule."""
    meta = dict(descriptor.get("metadata", {}))
    for name, default in defaults.items():
        meta.setdefault(name, default)
    return meta


def _remap(shape: dict, field_map: dict) -> dict:
    """Rename a corpus field to the metadata name it corresponds to.

    The shape corpus speaks whatever vocabulary its own producer chose
    (`dispatch_parity.py --shapes` uses `nhead_q`/`hdim_q`/...); KMD metadata
    speaks the matcher's. Where the two differ, --field-map says so explicitly
    rather than this tool guessing a mapping from field names alone.
    """
    out = dict(shape)
    for old, new in field_map.items():
        if old in out:
            out[new] = out.pop(old)
    return out


def _same_value(shape_value, metadata_value) -> bool:
    """Are these the same value, allowing for the two vocabularies?

    Numbers compare numerically (a bool is an int here: a `causal` metadata 1 and a
    corpus `True` are the same graph). Strings compare case-insensitively, because
    metadata carries the matcher's spelling and a request corpus carries the
    builder's -- see `applicable`.
    """
    if isinstance(shape_value, str) and isinstance(metadata_value, str):
        return shape_value.strip().lower() == metadata_value.strip().lower()
    if isinstance(shape_value, (int, float)) and isinstance(
        metadata_value, (int, float)
    ):
        return float(shape_value) == float(metadata_value)
    return shape_value == metadata_value


def applicable(metadata: dict, shape: dict, divides: dict) -> bool:
    """True when `metadata` (a variant, defaults resolved) is legal for `shape`.

    Two kinds of shape-valued field, matching the two kinds the real matcher
    tests exercise (see TestGfx942AttentionDenseMatchers.cpp's
    RefusesATileThatDoesNotDivideTheSequence / AcceptsEitherShippedTileForA...):

      * a metadata field sharing a name with a shape field must be EQUAL to it,
        unless that field is declared as a divisor (divisor fields never compare
        by equality -- a shape rarely carries a field literally named `block_n`);
      * a declared divisor field must evenly divide the shape field it is
        declared against, and non-positive tiles never divide anything.

    A field the caller never declared and that shares no name with any shape key
    is invisible here -- see the module docstring's "what this cannot know".

    STRING COMPARISON IS CASE-INSENSITIVE, and that is not laziness. Metadata
    carries the hipDNN spelling the matcher compares (`"BF16"`); a request corpus
    carries the builder's (`"bf16"`). They are the same value in two vocabularies,
    and the whole pipeline elsewhere translates between them deliberately. Comparing
    them raw makes EVERY variant unreachable -- observed: 91 of 91 on a set generated
    from the very corpus it was tested against, which is a false alarm so total it
    would train an author to pass --allow-unreachable and stop reading.
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
    """One Verdict per descriptor, against the whole corpus.

    `score` is `{"field": ..., "prefer": "max" | "min"}` or None. None means no
    ranking was declared: every applicable variant is, by construction, a winner
    everywhere it applies (there is nothing here that could rank it out), so the
    only real finding left is UNREACHABLE.
    """
    if score is not None and score.get("prefer") not in ("max", "min"):
        raise ReachabilityError("score.prefer must be 'max' or 'min'")

    remapped = [_remap(s, field_map) for s in shapes]
    metas = {d["name"]: _resolved_metadata(d, defaults) for d in descriptors}

    # shape index -> [variant names applicable there]
    applicable_at: list[list[str]] = [
        [name for name, meta in metas.items() if applicable(meta, s, divides)]
        for s in remapped
    ]

    # shape index -> [variant names that WIN there] (ties all win: without the
    # real scorer's own tie-break this tool cannot rule either side out, and
    # picking one arbitrarily would manufacture a false APPLICABLE-BUT-NEVER-WINS).
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

        defaults, descriptors = load_bundle(args.kdp)
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
