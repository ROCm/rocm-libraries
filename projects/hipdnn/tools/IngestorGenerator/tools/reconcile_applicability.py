"""Every decline must be reconciled against the reference, WITHIN THE SAME KERNEL FAMILY.

THE RULE, and it is not negotiable:

    If rocKE's implementation OF THE KERNEL YOU ARE INTEGRATING serves an equivalent
    request and its result validates, hipDNN must serve it too. A hipDNN decline that
    that kernel does not share is a DEFECT in this integration -- missing coverage, or
    applicability logic that is wrong -- not a scope decision.

An engine's own bundles cannot show this. They test the graphs the author thought of,
against a reference the author chose, and they are green precisely when the author's
model of "what we support" is self-consistent. This tool compares that model against
the kernel's own, which is the only comparison that can find a gap the author does not
already know about.

SCOPE IS THE WHOLE DESIGN, AND GETTING IT WRONG INVENTS WORK. A library typically
registers several candidates for one operation -- for attention: a dense kernel, a
couple of unified tiled paths, decode specialists. They are DIFFERENT KERNELS with
different capabilities. If you are integrating the dense kernel, a shape that only the
tiled path serves is NOT your coverage gap: it is a different kernel's job, and
"integrate the tiled path too" is a separate piece of work with its own variant set.

Comparing against every registered candidate was tried and is wrong. On one real
corpus it reported 51 shapes -- decode and large head sizes -- as gaps in a dense
integration, when the dense kernel declines every one of them for exactly the reasons
hipDNN does, and the shapes were being served by sibling candidates. That is a false
alarm with a plausible story attached, which is the expensive kind.

So the oracle is scoped by `family` from the profile: only candidates whose algorithm
(or family, or spec_id -- whichever the library keys on) matches the kernel under
integration. Note that a library may give every candidate the same `family` string
while the real discriminator is `algorithm`; the profile says which attribute to match
on, because this tool cannot know.

OPT-IN KERNELS NEED THEIR SELECTOR SET. Where a candidate only matches when the
request names it, the oracle must set that selector or the kernel declines everything
and every decline reconciles trivially -- a gate that passes by asking nothing. That
is what `request.defaults` is for here, and it is the one place those defaults SHOULD
be inherited.

THREE OUTCOMES, and only the first two are a pass:

  * BOTH SERVE   -- fine, nothing owed.
  * BOTH DECLINE -- fine, and the reasons should agree. Record the reference's reason;
                    it is better evidence than your own matcher's, because it is
                    independent.
  * ONLY THE REFERENCE SERVES -- **FAIL**. Either the variant set is missing this
                    shape, or the matcher rejects something it should accept.

THE FOURTH CASE, which is real and must not be silently swallowed: the reference
accepts a request it then computes WRONGLY. That is a reference defect, and a finding
to report -- not licence for hipDNN to decline quietly. This tool cannot detect it,
because it asks only about applicability, never about numerics. Correctness comes from
the benchmark sweep with `--validate`.

    reconcile_applicability.py --profile <p.yaml> --shapes <corpus.json> \\
                               [--declines <hipdnn-declines.json>]

Without `--declines` it reports what the reference kernel serves against what THIS
integration's dispatcher-resolved parity set would serve, which is the cheap offline
form. With a declines file -- graph names and reasons harvested from a real run -- it
reconciles actual runtime behaviour, which is the form step 9 requires.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dispatch_parity import (  # noqa: E402
    ParityError,
    _bind_provider,
    _import,
    _load_profile,
    _required,
    resolve_shapes,
)


def _eligible(candidate, request):
    """Ask the candidate the COMPLETE eligibility question.

    Returns ``(ok, reason, degraded)``. ``degraded`` is True when only the residual
    predicate could be consulted, and the caller MUST carry that into the report: a
    served=True obtained without the capability gate is an unverified answer, and one
    that silently looks identical to a verified one is worse than no answer at all.
    """
    admits = getattr(candidate, "admits", None)
    if callable(admits):
        ok, reason = admits(request)
        return ok, reason, False
    supports = getattr(candidate, "_supports", None)
    if callable(supports):
        ok, reason = supports(request)
        return ok, f"{reason} (predicate only: no public eligibility accessor)", True
    raise ParityError(
        f"candidate {getattr(candidate, 'spec_id', candidate)!r} exposes neither "
        f"`admits` nor `_supports`; this tool cannot ask it about eligibility."
    )


def reference_serves(shapes: list[dict], profile: dict) -> dict:
    """For each shape: does the reference kernel FAMILY serve it, and which candidate?

    Scoped, never library-wide. A sibling candidate serving a shape this kernel refuses
    is a different kernel's job, not this integration's gap.

    The request is constructed inside the try for the same reason it is in
    `dispatch_parity`: a structurally-invalid request raises at CONSTRUCTION, before any
    predicate runs, so a check that only calls the predicate reports those shapes as
    supported and ships a wrong denominator.
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

    candidates = [c for c in registry() if getattr(c, attribute, None) == family]
    if not candidates:
        available = sorted({str(getattr(c, attribute, "?")) for c in registry()})
        raise ParityError(
            f"no registered candidate has {attribute}={family!r}. Available "
            f"{attribute} values: {available}. A profile naming a family that does not "
            f"exist would report EVERY shape as unreconciled."
        )

    # The REFERENCE's request class, which is not always the one the generator side
    # uses. `reference_request:` overrides `request:` when present.
    #
    # Why the override exists. A profile whose `request.class` is an ADAPTER -- because
    # the kernel's own dispatch entry point needs a different vocabulary than the
    # library's registry does -- cannot use that adapter here: rocKE's candidates
    # `isinstance`-check their argument and refuse anything else with
    # "expected AttentionRequest, got X". Duck-typing does not satisfy a type check.
    #
    # That refusal is per-shape and looks exactly like a decline, so WITHOUT this the
    # tool reports `RECONCILED: every decline is one the reference makes too` having
    # never consulted the reference at all -- the "gate that passes by asking nothing"
    # this file's own header warns about, and indistinguishable from a real pass.
    reference_decl = profile.get("reference_request") or profile.get("request") or {}
    request_cls = _import(
        *_required(reference_decl, "reference_request", "module", "class")
    )
    arch = profile.get("arch")
    # Opt-in defaults ARE inherited here, unlike a library-wide oracle. A candidate that
    # only matches when the request names it declines everything without its selector,
    # and every decline would then reconcile trivially -- a gate that passes by asking
    # nothing at all.
    defaults = dict(reference_decl.get("defaults") or {})

    # An optional TRANSLATOR, for the case where the corpus is written in the
    # generator side's vocabulary and the reference wants its own. Declared as
    # `reference_request.via: {module, function}`; it takes the shape dict and returns
    # the reference request object, so the mapping lives in the integration's own
    # adapter rather than being guessed here.
    via = None
    via_decl = reference_decl.get("via") or {}
    if via_decl:
        via = _import(
            *_required(via_decl, "reference_request.via", "module", "function")
        )

    out = {}
    for index, shape in enumerate(shapes):
        fields = {
            **defaults,
            **{k: v for k, v in shape.items() if not k.startswith("_")},
        }
        if arch and "arch" not in fields:
            fields["arch"] = arch
        try:
            request = via(fields) if via is not None else request_cls(**fields)
        except Exception as exc:
            out[index] = (False, f"{type(exc).__name__}: {exc}")
            continue
        served, why, degraded = False, None, False
        # Collect EVERY candidate's verdict, then choose the reason deliberately. The
        # naive loop kept whichever decline came last, and on a family with more than
        # one member that is arbitrary: scoping on a shared `algorithm` matches both the
        # gfx942 and gfx950 dense candidates, and gfx950's capability gate rejects a
        # gfx942 request BEFORE reaching the shared predicate -- so every recorded reason
        # became "arch not in (gfx950,)" and the real, kernel-specific reason
        # (seqlen_q % 256, head_size, ...) was masked on every shape. The verdict was
        # still right; the evidence step 9 exists to collect was uniformly wrong.
        reasons = []
        for candidate in candidates:
            try:
                # `admits`, NOT the underscore predicate. The library's own docstring
                # is explicit: admits is "the only eligibility question a caller should
                # ask", because registered candidates keep their arch and dtype gates in
                # `capability` and the predicate carries only the residual checks. Its
                # worked example is a candidate whose predicate "happily accepts" a
                # target its capability block forbids -- so calling the predicate alone
                # would report the reference as serving a shape it cannot, and turn a
                # correct hipDNN decline into a phantom coverage gap.
                ok, reason, partial = _eligible(candidate, request)
            except Exception as exc:  # a predicate that raises is a decline, loudly
                ok, reason, partial = False, f"{type(exc).__name__}: {exc}", False
            if ok:
                served, degraded = True, partial
                why = str(getattr(candidate, "spec_id", family))
                break
            reasons.append((getattr(candidate, "spec_id", "?"), str(reason)))
        if not served:
            why = _decline_reason(reasons)
        # A served=True reached through the degraded path is an UNVERIFIED answer, and
        # the caveat has to survive into the report rather than being thrown away with
        # the string it arrived in.
        out[index] = (
            served,
            why + (" [UNVERIFIED: predicate only]" if degraded else ""),
        )
    return out


def _decline_reason(reasons: list[tuple[str, str]]) -> str:
    """Pick the most informative decline from a family's candidates.

    A capability rejection ("wrong arch", "wrong dtype") says only that THIS member of
    the family is not the one for this target -- true, and useless as evidence. The
    kernel-specific reason from the member that actually targets this request is what a
    write-up needs. Prefer a non-capability reason; fall back to the first if every
    member rejected on capability, and name the candidate either way so a reader can
    tell which sibling spoke.
    """
    if not reasons:
        return "no candidate in this family accepted it"
    substantive = [
        (s, r) for s, r in reasons if not r.lower().startswith("capability:")
    ]
    spec_id, reason = (substantive or reasons)[0]
    return f"{reason} [{spec_id}]"
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile this integration's declines against the reference library.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--shapes", required=True)
    parser.add_argument(
        "--declines",
        help="JSON mapping of shape index (or graph name) -> the reason THIS "
        "integration declined it at runtime. Omit to reconcile against the "
        "dispatcher-resolved parity set instead, which is the offline form.",
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
        ours = resolve_shapes(shapes, profile)
        theirs = reference_serves(shapes, profile)
    except ParityError as exc:
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

    both_serve, both_decline, only_reference, only_ours = [], [], [], []
    matched_keys = set()
    for index, resolution in enumerate(ours):
        we_serve = resolution.spec is not None
        name = (shapes[index].get("_provenance") or {}).get("graph")
        if args.declines:
            # A runtime declines file overrides the offline answer: what the engine
            # ACTUALLY did beats what the dispatcher says it could do.
            key = str(index)
            for candidate_key in (key, name):
                if candidate_key and candidate_key in declines:
                    matched_keys.add(candidate_key)
                    we_serve = False
        they_serve, why = theirs[index]
        if we_serve and they_serve:
            both_serve.append(index)
        elif they_serve:
            only_reference.append(
                (index, why, resolution.reason or "no variant matched")
            )
        elif we_serve:
            # WE serve a shape the reference declines. Not a coverage gap -- the
            # opposite -- but it must not be filed under "both decline", which is
            # what a branch keying only on `they_serve` did. Either this integration
            # is serving something the reference knows it cannot compute correctly,
            # or the reference is missing a capability. Both are worth a look, and
            # neither is "agreed".
            only_ours.append((index, why))
        else:
            both_decline.append((index, why))

    # A declines key that matches nothing is a gate that passed because the question
    # was never asked. Index keys are the only option for a published-CSV corpus (no
    # graph names), and they are not stable across a re-mine with different flags --
    # so the same file silently attributes a decline to a different shape.
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

    # The signature of a misconfigured scope. An opt-in kernel only admits a request
    # that NAMES it, so if the profile scopes the reference on `algorithm: dense` but
    # `request.defaults` never sets `algorithm`, the reference declines every shape --
    # and this integration declines them for the same reason. The tool then printed
    # "RECONCILED" over a comparison that agreed about nothing, which is the gate
    # passing by asking nothing this tool's own docstring warns against.
    #
    # Both conditions are required. A corpus where nothing is served can be perfectly
    # legitimate (every shape genuinely out of scope, correctly declined by both
    # sides), so an empty-serve count alone is not evidence of anything.
    # A shape either side serves proves the comparison is live, so only a run where
    # NOTHING was served by anyone can be vacuous.
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
            f"decline(s), each of which needs a written justification at step 10."
        )
        return 0
    print("RECONCILED: every decline is one the reference makes too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
