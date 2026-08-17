#!/usr/bin/env python3
"""Resolve label-gated TheRock cmake flags for a multi-arch CI run.

The multi-arch workflows hand the whole build to TheRock's reusable
``setup_multi_arch.yml``, so there is no matrix row in this repository to
attach per-PR cmake options to. What there *is* is the ``external_repo`` JSON
string those workflows already pass: TheRock reads an optional
``extra_cmake_options`` key out of it and splices the value onto the top-level
cmake command line for every build stage. This script builds that JSON instead
of the workflows hardcoding it, adding ``extra_cmake_options`` when the pull
request carries a label listed in ``LABEL_GATED_THEROCK_FLAGS``.

Two independent questions are answered here, and conflating them is the easiest
way to get this wrong:

- *Does a label inject a flag?* Membership in ``LABEL_GATED_THEROCK_FLAGS``,
  and nothing else. The full label set of the pull request is consulted, so a
  label applied earlier keeps taking effect on later pushes.
- *Does a label change justify a build at all?* Only for ``labeled`` /
  ``unlabeled`` events, and only when the single label that changed is a key in
  the map. That answer is exported as ``label_relevant`` and the workflows use
  it to skip the expensive jobs when someone adds an unrelated label. Events
  that are not label events ignore it entirely.

Labels are read from the event payload (``PR_LABELS_JSON``) rather than a live
``gh pr view``: the payload is the label state the run was triggered for, so
logs and artifacts cannot disagree with it, it needs no token or network, and
it works unchanged for pull requests from forks.

Environment inputs:

``GITHUB_EVENT_NAME``
    Labels are only honored for ``pull_request``. A ``workflow_dispatch`` run
    therefore always builds flag-off.
``PR_LABELS_JSON``
    ``toJSON(github.event.pull_request.labels.*.name)``. Junk is tolerated.
``CHANGED_LABEL``
    ``github.event.label.name``; only set on ``labeled`` / ``unlabeled``.
``EXTERNAL_REPO_REPOSITORY`` / ``EXTERNAL_REPO_REF``
    The repository and commit the external-repo payload points at.
``PREBUILT_STAGES`` / ``BASELINE_RUN_ID``
    Seeding a run from another run's artifacts can never be correct with a
    non-default cmake configuration, so that combination is a hard error.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Iterable, Mapping, Optional

from ci_utils import append_step_summary, set_github_output

# Labels that turn TheRock cmake flags on for a single pull request.
#
# Shipped empty: an entry here changes what CI builds, so adding one is a
# deliberate act. A label has an effect if and only if it is a key in this map;
# no naming convention is enforced. `ci:<project>-<feature>` is the suggested
# form for new labels.
#
# Only `-DTHEROCK_FLAG_*` options belong here (see FLAG_RE below), and the flag
# must already be declared with `therock_declare_flag(... SUB_PROJECTS <proj>)`
# in TheRock's FLAGS.cmake or it will not reach the subproject.
#
# The single-arch workflow has no equivalent map in this repository today; if
# one is reintroduced, the two are independent and a label meant to drive both
# workflows needs an entry in each.
#
# Example:
#     "ci:miopen-hipdnn-wrapper": [
#         "-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON",
#     ],
LABEL_GATED_THEROCK_FLAGS: dict[str, list[str]] = {}

# Deliberately narrow, and it carries the whole safety argument for splicing
# these tokens onto a shell command line in another repository:
#
# - Anchored, with no `.` and no whitespace class, so a token cannot contain a
#   space, `;`, `&`, `|`, `$`, backtick, newline, or a quote character. The
#   quote case is load-bearing: the consuming workflow passes the payload as
#   `--external-repo-json='<json>'`, which a single quote would break.
# - `THEROCK_FLAG_` is the correct namespace, not a multi-arch quirk.
#   `therock_declare_flag` adds that prefix, so `NAME FOO` creates the cache
#   variable `THEROCK_FLAG_FOO`; that prefixed name is the superbuild knob, and
#   the flag machinery is what forwards the unprefixed name into the
#   subprojects listed in SUB_PROJECTS. Setting the unprefixed name at top
#   level does nothing, because subproject arguments are an explicit allowlist.
# - The prefix is also what makes the splice ordering safe. These options land
#   *before* TheRock's own generated cmake args and cmake takes the last `-D`
#   wins. TheRock generates `THEROCK_AMDGPU_FAMILIES`, `THEROCK_DIST_*`,
#   `THEROCK_ENABLE_ALL` and `THEROCK_ENABLE_<FEATURE>` — never
#   `THEROCK_FLAG_*`. Admitting anything from a clobbered namespace would ship
#   a flag that silently does nothing, so reject it instead.
FLAG_RE = re.compile(r"^-DTHEROCK_FLAG_[A-Z][A-Z0-9_]*=(ON|OFF)$")


def validate_label_gated_therock_flags(
    mapping: Optional[Mapping[str, object]] = None,
) -> None:
    """Raise ``ValueError`` if the label map is malformed."""
    if mapping is None:
        mapping = LABEL_GATED_THEROCK_FLAGS
    if not isinstance(mapping, dict):
        raise ValueError("LABEL_GATED_THEROCK_FLAGS must be a dict")

    for label, flags in mapping.items():
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"LABEL_GATED_THEROCK_FLAGS keys must be non-empty strings, got {label!r}"
            )
        # A bare string here would iterate character by character and produce
        # nonsense flags, so require a list explicitly.
        if not isinstance(flags, list):
            raise ValueError(
                f"LABEL_GATED_THEROCK_FLAGS['{label}'] must be a list of strings, "
                f"got {type(flags).__name__}"
            )
        if not flags:
            raise ValueError(f"LABEL_GATED_THEROCK_FLAGS['{label}'] must not be empty")
        for flag in flags:
            if not isinstance(flag, str) or not FLAG_RE.fullmatch(flag):
                raise ValueError(
                    f"LABEL_GATED_THEROCK_FLAGS['{label}'] contains {flag!r}; "
                    "entries must match -DTHEROCK_FLAG_<NAME>=ON|OFF"
                )


def parse_labels(raw: str) -> list[str]:
    """Parse ``toJSON(...labels.*.name)`` into a list of label names.

    Tolerates every shape GitHub can hand us for a non-pull-request event: an
    empty string, the literal ``null``, a non-list, or a list with non-string
    elements.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str) and item]


def collect_flags(
    labels: Iterable[str],
    mapping: Optional[Mapping[str, list[str]]] = None,
) -> tuple[list[str], list[str]]:
    """Return ``(matched_labels, flags)`` for the labels present on the PR.

    Iterates the map rather than the labels so the flag order is a property of
    the map and not of however GitHub happened to order the payload.
    """
    if mapping is None:
        mapping = LABEL_GATED_THEROCK_FLAGS
    label_set = set(labels)
    matched: list[str] = []
    flags: list[str] = []
    # Flag name -> (value, label that set it), to catch two labels asking for
    # opposite values instead of letting map order silently decide.
    seen: dict[str, tuple[str, str]] = {}

    for label, label_flags in mapping.items():
        if label not in label_set:
            continue
        matched.append(label)
        for flag in label_flags:
            name, _, value = flag.partition("=")
            previous = seen.get(name)
            if previous is None:
                seen[name] = (value, label)
                flags.append(flag)
            elif previous[0] != value:
                raise ValueError(
                    f"Labels '{previous[1]}' and '{label}' set {name} to "
                    f"conflicting values ({previous[0]} vs {value}); "
                    "remove one of the labels."
                )

    return matched, flags


def build_external_repo(repository: str, ref: str, flags: list[str]) -> str:
    """Render the ``external_repo`` JSON string the workflows pass to TheRock.

    With no flags this is byte-identical to the literal the workflows used
    before this script existed, so unlabeled runs are unchanged.
    """
    payload: dict[str, str] = {"repository": repository, "ref": ref}
    if flags:
        payload["extra_cmake_options"] = " ".join(flags)
    rendered = json.dumps(payload, separators=(",", ":"))
    # The consumer interpolates this into a single-quoted shell argument.
    # Nothing that reaches here should be able to break that, but the cost of
    # being sure is a single check.
    if "'" in rendered or "\n" in rendered:
        raise ValueError(f"external_repo payload is not shell-safe: {rendered!r}")
    return rendered


def build_summary(
    *,
    event_name: str,
    labels: list[str],
    matched: list[str],
    flags: list[str],
    external_repo: str,
    changed_label: str,
    label_relevant: bool,
) -> str:
    """Render the step summary. Names the exact labels it acted on."""
    lines: list[str] = []
    lines.append("### Label-gated cmake flags")
    lines.append("")

    if event_name != "pull_request":
        lines.append(
            f"Event is `{event_name}`, not `pull_request`; labels are not read and "
            "the build uses the default configuration."
        )
        lines.append("")
    else:
        label_list = ", ".join(f"`{label}`" for label in labels) if labels else "_none_"
        lines.append(f"- **Labels on this pull request:** {label_list}")
        if changed_label:
            lines.append(
                f"- **Label just added/removed:** `{changed_label}` "
                f"({'gated, so this run proceeds' if label_relevant else 'not gated, so the expensive jobs are skipped'})"
            )
        matched_list = (
            ", ".join(f"`{label}`" for label in matched) if matched else "_none_"
        )
        lines.append(f"- **Matched gating labels:** {matched_list}")
        flag_list = " ".join(f"`{flag}`" for flag in flags) if flags else "_none_"
        lines.append(f"- **Injected cmake options:** {flag_list}")
        lines.append("")

    lines.append("External repo payload handed to TheRock:")
    lines.append("")
    lines.append("```json")
    lines.append(external_repo)
    lines.append("```")

    if flags:
        lines.append("")
        lines.append("> [!IMPORTANT]")
        lines.append(
            "> The flag-on build **replaces** the normal build; this run produces no "
            "flag-off signal. Stage reuse is forced off, so every stage is rebuilt."
        )
        lines.append(">")
        lines.append(
            "> Re-running an older run replays the label set from when that run was "
            "first triggered. Push a commit or re-apply the label instead of "
            "re-running a stale run."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    # Unconditional: a malformed map should fail every run loudly, not only the
    # runs that happen to carry the offending label.
    validate_label_gated_therock_flags()

    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    repository = os.environ.get("EXTERNAL_REPO_REPOSITORY", "")
    ref = os.environ.get("EXTERNAL_REPO_REF", "")
    # Only meaningful on labeled/unlabeled; empty for every other event.
    changed_label = os.environ.get("CHANGED_LABEL", "")

    # Label behavior must be unreachable from dispatch, push and nightlies.
    labels = (
        parse_labels(os.environ.get("PR_LABELS_JSON", ""))
        if event_name == "pull_request"
        else []
    )

    matched, flags = collect_flags(labels)
    label_relevant = bool(changed_label) and changed_label in LABEL_GATED_THEROCK_FLAGS
    external_repo = build_external_repo(repository, ref, flags)

    if flags:
        prebuilt_stages = os.environ.get("PREBUILT_STAGES", "").strip()
        baseline_run_id = os.environ.get("BASELINE_RUN_ID", "").strip()
        if prebuilt_stages or baseline_run_id:
            raise SystemExit(
                "Label-gated cmake flags are active "
                f"({' '.join(flags)}), but this run was asked to seed itself from "
                f"another run's artifacts (prebuilt_stages='{prebuilt_stages}', "
                f"baseline_run_id='{baseline_run_id}'). Those artifacts were built "
                "with a different cmake configuration. Remove the label or drop the "
                "prebuilt-stage inputs."
            )

    set_github_output(
        {
            "external_repo": external_repo,
            "flags_active": "true" if flags else "false",
            "label_relevant": "true" if label_relevant else "false",
            "flags": " ".join(flags),
            "matched_labels": ",".join(matched),
        }
    )
    append_step_summary(
        build_summary(
            event_name=event_name,
            labels=labels,
            matched=matched,
            flags=flags,
            external_repo=external_repo,
            changed_label=changed_label,
            label_relevant=label_relevant,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
