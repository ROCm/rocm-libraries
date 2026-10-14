"""Reconcile every decline against the reference, within the same kernel family.

If the reference's implementation of the kernel being integrated serves an
equivalent request and its result validates, hipDNN must serve it too: a
decline that kernel does not share is missing coverage or wrong applicability
logic, not a scope decision.

Scoped by `family` from the profile, never library-wide, since a shape only a
sibling candidate serves is that sibling's job. The profile also names the
attribute to match on, because a library may give every candidate the same
`family` while `algorithm` is the real discriminator.

Only applicability is asked about, never numerics; correctness comes from the
benchmark sweep with `--validate`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dispatch_parity import (  # noqa: E402
    ParityError,
    _bind_provider,
    _eligible,
    _import,
    _load_profile,
    _required,
    resolve_shapes,
)


def reference_serves(shapes: list[dict], profile: dict) -> dict:
    """For each shape: does the reference kernel FAMILY serve it, and which candidate?

    Request construction and reference API failures are operational errors; only a
    validated false eligibility result is an ordinary decline.
    """
    entry = profile.get("reference_candidates") or {}
    if not entry:
        raise ParityError(
            "the profile declares no 'reference_candidates' block, so there is no "
            "oracle to reconcile against. It needs module/function naming the "
            "library's candidate registry, plus `match` (the attribute to scope on, "
            "e.g. algorithm) and `family` (its value for the kernel you are "
            "integrating)."
        )
    registry = _import(*_required(entry, "reference_candidates", "module", "function"))
    attribute = entry.get("match", "algorithm")
    family = entry.get("family")
    if not family:
        raise ParityError(
            "'reference_candidates.family' is required: it names WHICH kernel this "
            "integration is a port of. Without it the oracle would compare against "
            "every candidate the library registers, and report a sibling kernel's "
            "coverage as this integration's gap."
        )

    try:
        if not callable(registry):
            raise TypeError("candidate registry is not callable")
        inspect.signature(registry).bind()
        registered = list(registry())
        candidates = [c for c in registered if getattr(c, attribute, None) == family]
    except Exception as exc:
        raise ParityError(f"reference candidate registry failed: {exc}") from exc
    if not candidates:
        available = sorted({str(getattr(c, attribute, "?")) for c in registered})
        raise ParityError(
            f"no registered candidate has {attribute}={family!r}. Available "
            f"{attribute} values: {available}. A profile naming a family that does not "
            f"exist would report EVERY shape as unreconciled."
        )

    # The reference's request class; `reference_request:` overrides `request:`
    # when present. An adapter class cannot be reused, because candidates
    # `isinstance`-check their argument; that refusal is per-shape and looks
    # exactly like a decline, so without the override every decline would
    # reconcile against a reference never consulted.
    reference_decl = profile.get("reference_request") or profile.get("request") or {}
    request_cls = _import(
        *_required(reference_decl, "reference_request", "module", "class")
    )
    arch = profile.get("arch")
    # Opt-in defaults ARE inherited here: without its selector, a candidate that only
    # matches when the request names it declines everything and reconciles trivially.
    defaults = dict(reference_decl.get("defaults") or {})

    # An optional translator (`reference_request.via: {module, function}`)
    # taking the shape dict to a reference request, for a corpus written in the
    # generator side's vocabulary. The mapping lives in the adapter.
    via = None
    via_decl = reference_decl.get("via") or {}
    if via_decl:
        via = _import(
            *_required(via_decl, "reference_request.via", "module", "function")
        )
        # Declared and unusable is an error, never a fallback: a profile naming
        # a translator has said its corpus is in the wrong vocabulary, so every
        # type rejection would otherwise read as a decline.
        if not callable(via):
            raise ParityError("reference_request.via must name a callable")
    if not isinstance(request_cls, type):
        raise ParityError("reference_request.class must name a request class")

    out = {}
    for index, shape in enumerate(shapes):
        fields = {
            **defaults,
            **{k: v for k, v in shape.items() if not k.startswith("_")},
        }
        if arch and "arch" not in fields:
            fields["arch"] = arch
        try:
            if via is not None:
                inspect.signature(via).bind(fields)
                request = via(fields)
            else:
                inspect.signature(request_cls).bind(**fields)
                request = request_cls(**fields)
            if not isinstance(request, request_cls):
                raise TypeError("reference translator returned the wrong request type")
        except Exception as exc:
            raise ParityError(
                f"reference request {index} construction failed: {exc}"
            ) from exc
        served, why = False, None
        # Collect every candidate's verdict, then choose the reason
        # deliberately: on a multi-member family a sibling's capability gate
        # rejects on arch before the shared predicate runs, masking the
        # kernel-specific reason. See `_decline_reason`.
        reasons = []
        for candidate in candidates:
            # Every candidate is consulted, even after an acceptance: a later broken
            # API must not be hidden by an earlier successful predicate. `_eligible`
            # raises ParityError rather than returning an unvalidated verdict.
            ok, reason = _eligible(candidate, request)
            if ok and not served:
                served = True
                why = str(getattr(candidate, "spec_id", family))
            if not ok:
                reasons.append((getattr(candidate, "spec_id", "?"), reason))
        if not served:
            why = _decline_reason(reasons)
        out[index] = (served, why)
    return out


def _decline_reason(reasons: list[tuple[str, str]]) -> str:
    """Pick the most informative decline from a family's candidates: a
    capability rejection ("wrong arch") says only that this member is not the
    one for this target, so prefer a non-capability reason, falling back to the
    first, and name the candidate either way."""
    if not reasons:
        return "no candidate in this family accepted it"
    substantive = [
        (s, r) for s, r in reasons if not r.lower().startswith("capability:")
    ]
    spec_id, reason = (substantive or reasons)[0]
    return f"{reason} [{spec_id}]"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile this integration's declines against the reference library.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--shapes", required=True)
    parser.add_argument(
        "--declines",
        help="JSON mapping of shape index (or graph name) -> the reason THIS "
        "integration declined it at runtime, which is the form RUNBOOK §7 "
        "requires. Omit to reconcile against the dispatcher-resolved parity set "
        "instead, which is the offline form.",
    )
    parser.add_argument(
        "--allow-unreconciled",
        action="store_true",
        help="Exit 0 even with unreconciled declines. Use only when every one has a "
        "written justification a human has accepted.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Permit a run in which neither side serves any shape. Normally that "
        "means the profile's opt-in selector is missing and the comparison is "
        "vacuous, so it is a hard failure rather than a pass.",
    )
    args = parser.parse_args(argv)

    try:
        profile = _load_profile(args.profile)
        _bind_provider(profile.get("provider_root"))
        shapes = json.loads(Path(args.shapes).read_text())
        if (
            not isinstance(shapes, list)
            or not shapes
            or any(not isinstance(s, dict) for s in shapes)
        ):
            raise ParityError(
                "--shapes must contain a nonempty list of request mappings"
            )
        graph_owners = {}
        shape_names = []
        for index, shape in enumerate(shapes):
            provenance = shape.get("_provenance") or {}
            occurrences = shape.get("_provenance_occurrences") or [provenance]
            if not isinstance(occurrences, list) or any(
                not isinstance(p, dict) for p in occurrences
            ):
                raise ParityError("provenance occurrences must be a list of mappings")
            names = {p["graph"] for p in occurrences if p.get("graph")}
            for name in names:
                if (
                    not isinstance(name, str)
                    or (name in graph_owners and graph_owners[name] != index)
                    or (name.isdigit() and int(name) != index)
                ):
                    raise ParityError("ambiguous graph name in shape corpus")
                graph_owners[name] = index
            shape_names.append(names)
        ours = resolve_shapes(shapes, profile)
        theirs = reference_serves(shapes, profile)
    # Every failure above is operational -- the comparison did not happen -- so
    # it exits 2 whatever it was raised as; an enumerated exception list would
    # let an unexpected type escape as exit 1, an unreconciled-decline result.
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    declines = {}
    if args.declines:
        try:
            declines = json.loads(Path(args.declines).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL: --declines {args.declines}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(declines, dict):
            print(
                f"FAIL: --declines must be a JSON mapping of shape index or graph "
                f"name to the reason this integration declined it, got "
                f"{type(declines).__name__}.",
                file=sys.stderr,
            )
            return 2
        if any(
            not isinstance(reason, str) or not reason.strip()
            for reason in declines.values()
        ):
            print("FAIL: every decline must carry a nonempty reason", file=sys.stderr)
            return 2

    both_serve, both_decline, only_reference, only_ours = [], [], [], []
    matched_keys = set()
    for index, resolution in enumerate(ours):
        we_serve = resolution.spec is not None
        runtime_reasons = []
        if args.declines:
            # A runtime declines file overrides the offline answer: what the engine
            # ACTUALLY did beats what the dispatcher says it could do.
            key = str(index)
            for candidate_key in {key, *shape_names[index]}:
                if candidate_key in declines:
                    matched_keys.add(candidate_key)
                    runtime_reasons.append(declines[candidate_key])
                    we_serve = False
        they_serve, why = theirs[index]
        if we_serve and they_serve:
            both_serve.append(index)
        elif they_serve:
            only_reference.append(
                (
                    index,
                    why,
                    "; ".join(runtime_reasons)
                    or resolution.reason
                    or "no variant matched",
                )
            )
        elif we_serve:
            # We serve a shape the reference declines: not a coverage gap, but
            # not agreement either. Either this integration serves something
            # the reference computes wrongly, or the reference is missing a
            # capability.
            only_ours.append((index, why))
        else:
            both_decline.append((index, why))

    if args.declines:
        unmatched = sorted(set(declines) - matched_keys)
        if unmatched:
            print(
                f"FAIL: {len(unmatched)} --declines key(s) matched no shape in this "
                f"corpus: {', '.join(unmatched[:8])}"
                + (" ..." if len(unmatched) > 8 else ""),
                file=sys.stderr,
            )
            print(
                "  A key that matches nothing is silently ignored, and the shape it "
                "was meant to\n  mark stays counted as served. Index keys shift when "
                "the corpus is re-mined with\n  different flags; prefer graph names "
                "where the corpus carries them.",
                file=sys.stderr,
            )
            return 2

    print("applicability reconciliation (reference = the same kernel family, scoped)")
    print(f"  shapes                  {len(shapes)}")
    print(f"  both serve              {len(both_serve)}")
    print(f"  both decline            {len(both_decline)}")
    print(f"  ONLY THE REFERENCE      {len(only_reference)}")
    if only_ours:
        print(f"  only this integration   {len(only_ours)}")

    # The signature of a misconfigured scope, and both conditions are required:
    # a corpus where nothing is served can be legitimate, so an empty serve
    # count alone proves nothing. A shape either side serves proves the
    # comparison is live.
    match_key = ((profile.get("reference_candidates") or {}).get("match")) or ""
    request_defaults = (profile.get("request") or {}).get("defaults") or {}
    nothing_served = not both_serve and not only_reference and not only_ours
    if shapes and nothing_served and match_key and match_key not in request_defaults:
        print(
            f"\nFAIL: nothing is served by EITHER side, and this profile scopes the "
            f"reference\n  on '{match_key}' while `request.defaults` never sets it. An "
            f"opt-in kernel only\n  admits a request that names it, so the reference "
            f"declines every shape and this\n  integration declines them for the same "
            f"reason -- agreement about nothing.\n  Add '{match_key}' to "
            f"request.defaults, or pass --allow-empty if a corpus that\n  nothing "
            f"serves is genuinely what you meant to reconcile.",
            file=sys.stderr,
        )
        if not args.allow_empty:
            return 2

    if both_decline:
        print(
            "\n  Both decline -- record the reference's reason, it is independent evidence:"
        )
        for index, why in both_decline[:10]:
            print(f"    [{index}] {why[:96]}")
        if len(both_decline) > 10:
            print(f"    ... and {len(both_decline) - 10} more")

    if only_ours:
        print(
            "\n  THIS INTEGRATION SERVES WHAT THE REFERENCE DECLINES. Not a coverage\n"
            "  gap, and not agreement either -- check the reference's reason before\n"
            "  claiming the shape. If it declines because the kernel computes the\n"
            "  wrong answer there, serving it is a correctness bug, not extra reach:"
        )
        for index, why in only_ours[:10]:
            print(f"    [{index}] reference declines: {why[:80]}")
        if len(only_ours) > 10:
            print(f"    ... and {len(only_ours) - 10} more")

    if only_reference:
        print(
            "\n  UNRECONCILED DECLINES. The reference serves these and this integration\n"
            "  does not. Each is a defect until shown otherwise -- missing coverage, or\n"
            "  a matcher rejecting what it should accept:"
        )
        for index, served_by, ours_reason in only_reference[:20]:
            print(f"    [{index}] reference: {served_by}")
            print(f"          ours: {ours_reason[:88]}")
        if len(only_reference) > 20:
            print(f"    ... and {len(only_reference) - 20} more")
        print(
            "\n  For each: add the variant, fix the matcher, or -- if you believe the\n"
            "  reference is wrong -- show that it computes an INCORRECT result for that\n"
            "  shape and report it as a reference defect. 'We chose not to' is not one\n"
            "  of the three."
        )

    print()
    if only_reference and not args.allow_unreconciled:
        print(
            f"RECONCILIATION FAILED ({len(only_reference)} decline(s) the reference does not share)"
        )
        return 1
    if only_reference:
        print(
            f"RECONCILIATION ACCEPTED UNDER PROTEST: {len(only_reference)} unreconciled "
            f"decline(s), each of which needs a written justification at RUNBOOK §7."
        )
        return 0
    print("RECONCILED: every decline is one the reference makes too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
