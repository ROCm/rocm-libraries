# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The review offload (config `offload`): the reviewer model does each review's
substance, first rounds included, and the orchestrator (Claude) audits it.

- off (the default): the reviewer model drafts (worker/review_agent.py); the orchestrator verifies every claim and runs
  the code quality pass (the verifier and quality subagents).
- shadow (the trial): as off, and the offload passes also run for each review, into workers/pr<N>/offload/, posting
  nothing; `rk shadow-report` compares them with the review we posted (rkmon_lib/shadow.py).
- on: the offload passes do the review (worker/offload_agent.py: draft, quality and regrade in the worker job, the
  assembly in the finish job after the checks); `rk evidence-check --offload` re-checks their evidence
  (rkmon_lib/offload_evidence.py); the orchestrator audits it (the agentic offload.md) and posts. A review goes deep, the full Claude review, when its
  evidence check fails, a pass wrote nothing, or a first round is large or touches the risky paths (`deep_reasons`);
  the auditor sends one deep too when the claims contradict each other or the draft does not fit the diff.

Every run of the passes is kept with its evidence report in <paths.lessons>/offload/pr<N>_<round>/ (`archive_dir`).
"""

from __future__ import annotations

import json
import pathlib
import re

from . import config, paths, registry, verifier_file

MODES = ("off", "shadow", "on")
ROLES = ("draft", "quality", "regrade")
# Every setting with its default: the package's (config.defaults.json's `offload`).
DEFAULTS = registry.package_section("offload")
WORKER_TIMEOUT_MIN = (
    120  # the worker job of a review with the offload passes (60 without them)
)
FINISH_TIMEOUT_MIN = 60


def settings(cfg: dict) -> dict:
    """config `offload` over DEFAULTS (a blank value keeps the default)."""
    raw = cfg.get("offload") or {}
    out = {
        k: (raw[k] if k in raw and raw[k] not in (None, "") else v)
        for k, v in DEFAULTS.items()
    }
    out["budget_shares"] = dict(out["budget_shares"] or {})
    return out


def mode(cfg: dict) -> str:
    m = str(settings(cfg)["mode"]).strip().lower()
    return m if m in MODES else DEFAULTS["mode"]


def problems(cfg: dict) -> list[str]:
    """What is wrong with config `offload` (the configuration checks)."""
    raw, s, out = cfg.get("offload") or {}, settings(cfg), []
    for key in raw:
        if key not in DEFAULTS:
            out.append(
                f"offload.{key} is not a setting; the settings are {', '.join(DEFAULTS)}"
            )
    if str(s["mode"]).strip().lower() not in MODES:
        out.append(
            f"offload.mode is {s['mode']!r}; it must be one of {', '.join(MODES)}"
        )
    try:
        ok = 0 <= float(s["spot_check"]) <= 1
    except (TypeError, ValueError):
        ok = False
    if not ok:
        out.append(
            f"offload.spot_check is {s['spot_check']!r}; it must be a fraction from 0 to 1 (0.25: a quarter)"
        )
    for key in (
        "spot_check_min",
        "deep_lines",
        "deep_files",
        "max_steps",
        "probe_timeout_s",
        "check_rerun_max_s",
        "check_total_s",
        "trial_prs",
    ):
        v = s[key]
        if not (
            isinstance(v, int)
            and not isinstance(v, bool)
            and v >= (1 if key in ("max_steps", "trial_prs") else 0)
        ):
            out.append(f"offload.{key} is {v!r}; it must be a whole number")
    shares = s["budget_shares"]
    if (
        not isinstance(shares, dict)
        or set(shares) - set(ROLES)
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
            for v in shares.values()
        )
        or not sum(shares.values())
    ):
        out.append(
            f"offload.budget_shares must give {', '.join(ROLES)} each a share, 0 or more, not all 0 (e.g. "
            f"{json.dumps(DEFAULTS['budget_shares'])})"
        )
    if not isinstance(s["deep_hot_spots"], list) or not all(
        isinstance(k, str) for k in s["deep_hot_spots"]
    ):
        out.append(
            "offload.deep_hot_spots must list hot spots by their names (hot_spots' keys, such as ABI)"
        )
    for rx in (
        s["deep_paths"] if isinstance(s["deep_paths"], list) else [s["deep_paths"]]
    ):
        try:
            re.compile(rx)
        except (re.error, TypeError) as exc:
            out.append(
                f"offload.deep_paths: {rx!r} is not a regular expression ({exc})"
            )
    return out


def shares_line(s: dict) -> str:
    total = sum((s.get("budget_shares") or {}).values()) or 1
    return ", ".join(
        f"{r} {100 * float((s['budget_shares'] or {}).get(r, 0)) / total:.0f}%"
        for r in ROLES
    )


def job_env(cfg: dict) -> dict[str, str]:
    """What the worker tools read (config.env adds it to every job): the mode, and the passes' settings."""
    s = settings(cfg)
    m = mode(cfg)
    return {
        "RKMON_OFFLOAD_MODE": m,
        "RKMON_OFFLOAD_JSON": json.dumps(
            {
                "mode": m,
                "budget_shares": s["budget_shares"],
                "max_steps": s["max_steps"],
                "probe_timeout_s": s["probe_timeout_s"],
                "line1": cfg.get("review_first_line") or config.first_line(cfg),
                "audit_name": config.claude_name(cfg, "audit"),
            },
            sort_keys=True,
        ),
    }


def review_dir(cfg: dict, n: int, m: str | None = None) -> pathlib.Path:
    """Where PR n's offload passes write: its review folder (on), or offload/ in it (shadow)."""
    ws = pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}"
    return ws / "offload" if (m or mode(cfg)) == "shadow" else ws


def archive_root(cfg: dict) -> pathlib.Path:
    return pathlib.Path(paths.get(cfg, "lessons")) / "offload"


def archive_dir(cfg: dict, n: int, rnd: str) -> pathlib.Path:
    """Where a run of the passes is kept with its evidence report: <lessons>/offload/pr<N>_<round>."""
    return archive_root(cfg) / f"pr{n}_{rnd}"


def finish_script(cfg: dict, n: int, rnd: str, m: str | None = None) -> str:
    """The finish job, after the checks: the assembly pass, then `rk evidence-check --offload` (which keeps the run in
    the archive)."""
    m = m or mode(cfg)
    out = review_dir(cfg, n, m)
    pr_dir = f" --pr-dir {out.parent}/pr" if m == "shadow" else ""
    return (
        f"{paths.shell_vars(cfg)}; N={n}\n{config.export_line(cfg)}\n"
        f"export RKMON_OFFLOAD_ROUND={rnd} RKMON_PIPE=pr{n}\n"
        f"python3 -B $T/offload_agent.py {out} $N assemble{pr_dir} || echo '(the assembly pass failed)'\n"
        "$M/rk evidence-check $N --offload\n"
    )


def outputs(folder: pathlib.Path) -> dict[str, bool]:
    """Which of a run's outputs exist (non-empty)."""
    return {
        name: (folder / name).is_file() and (folder / name).stat().st_size > 0
        for name in (
            "draft.md",
            "quality.md",
            "regrade.md",
            "post_draft.md",
            "evidence.jsonl",
            "evidence_check.json",
        )
    }


def read_report(folder: pathlib.Path) -> dict:
    try:
        return json.loads((folder / "evidence_check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def deep_reasons(
    cfg: dict, rec: dict, kind: str, files: list[str], folder: pathlib.Path
) -> list[str]:
    """Why a review goes deep (the full Claude review), the reasons the code can see: its evidence check failed, a
    pass wrote nothing (the reviewer model went down mid-review), or a first round is large or touches the risky paths
    (offload.deep_lines, deep_files, deep_hot_spots, deep_paths). Empty: an audit."""
    s, out = settings(cfg), []
    have = outputs(folder)
    missing = [
        name
        for name in ("draft.md", "quality.md", "regrade.md", "post_draft.md")
        if not have[name]
    ]
    if missing:
        out.append(f"the reviewer model wrote no {', '.join(missing)}")
    report = read_report(folder)
    if not report:
        out.append("no evidence check")
    elif report.get("result") != "PASS":
        out.append(
            f"evidence check {report.get('result', '?')}: "
            + (report.get("summary") or "")[:160]
        )
    if kind == "first-round":
        size = int(rec.get("size") or 0)
        if size > int(s["deep_lines"]):
            out.append(
                f"a first round of {size} changed lines (offload.deep_lines {s['deep_lines']})"
            )
        if len(files) > int(s["deep_files"]):
            out.append(
                f"a first round of {len(files)} files (offload.deep_files {s['deep_files']})"
            )
        from .queue import hot_paths, path_hits

        hot = sorted(
            k
            for k in s["deep_hot_spots"] or []
            if k in (cfg.get("hot_spots") or {})
            and any(path_hits(hot_paths(cfg, k), f) for f in files)
        )
        if hot:
            out.append(
                f"a first round touching the hot spots {', '.join(hot)} (offload.deep_hot_spots)"
            )
        rx = [re.compile(p) for p in s["deep_paths"] or []]
        risky = sorted({f for f in files if any(r.search(f) for r in rx)})
        if risky:
            out.append(
                f"a first round touching {', '.join(risky[:3])}{' …' if len(risky) > 3 else ''} "
                "(offload.deep_paths)"
            )
    return out


def ready_line(
    cfg: dict, n: int, kind: str, ws: str, folder: pathlib.Path, reasons: list[str]
) -> str:
    """REVIEW_READY for an offload review: audit or deep, the evidence check to read first, and the inputs."""
    rep = read_report(folder)
    head = (
        f"REVIEW_READY #{n} ({kind}, deep: {'; '.join(reasons)})"
        if reasons
        else f"REVIEW_READY #{n} ({kind}, audit)"
    )
    return (
        f"{head}: offload review by the reviewer model. Read first: {folder}/evidence_check.md "
        f"({rep.get('summary') or 'no evidence check'}). Inputs: {folder}/post_draft.md, notes_draft.md, "
        f"draft.md, quality.md, regrade.md, evidence.jsonl; {ws}/checks/checks.md, {ws}/pr/meta.md. "
        + (
            "Deep review: the full verification and code quality pass, these as extra inputs (offload.md)"
            if reasons
            else "Audit by offload.md, then `rk post N FILE --premises SPEC`"
        )
    )


# --- rk doctor --------------------------------------------------------------------------------------------------------


def settings_check(cfg: dict):
    """The row of the offload's settings (a checks row): FAIL with what is wrong with them (problems), OFF while
    offload.mode is off, else PASS with the mode and the budget shares."""
    from . import doctor
    from .checks import Check, Steps

    bad = problems(cfg)
    if bad:
        return Check(
            "review offload",
            "FAIL",
            "; ".join(bad),
            Steps(("fix them in", *doctor.settings_file())),
        )
    m = mode(cfg)
    if m == "off":
        return Check("review offload", "OFF", "off (offload.mode)")
    return Check("review offload", "PASS", f"{m}; {shares_line(settings(cfg))}")


def doctor_checks(cfg: dict) -> list:
    """rk doctor's rows (checks rows) for the offload: the settings (settings_check), the rendered team member files
    the passes read, and the shadow trial's progress.
    """
    from . import doctor
    from .checks import Check, Steps

    first = settings_check(cfg)
    if first.status != "PASS":
        return [first]
    m = mode(cfg)
    out = [first]
    agentic = pathlib.Path(paths.resolve(cfg)["agentic"] or ".")
    files = (paths.QUALITY_FILE, verifier_file.FILE)
    gone = [f for f in files if not (agentic / f).is_file()]
    out.append(
        Check(
            "offload instructions",
            "WARN" if gone else "PASS",
            (
                doctor.blocking(
                    f"missing {', '.join(gone)} in {agentic}",
                    "the quality, re-grade and assembly passes",
                )
                if gone
                else f"{agentic}: the quality and verifier instructions the passes read"
            ),
            Steps((*doctor.section_parts("agentic"), "writes them")) if gone else None,
        )
    )
    w = config.worker(cfg)
    if int(w.get("context") or 0) < 131072:
        out.append(
            Check(
                "offload context",
                "WARN",
                f"worker.context {w.get('context')}: a pass sends the whole diff, "
                "the instructions and its tool rounds",
                Steps(
                    (
                        "choose a reviewer model with 128k tokens of context or more:",
                        *doctor.section_parts("model"),
                    )
                ),
            )
        )
    if m == "shadow":
        root = archive_root(cfg)
        runs = sorted(root.glob("pr*_*")) if root.is_dir() else []
        out.append(
            Check(
                "shadow trial",
                "PASS",
                f"{len(runs)} offload run(s) kept in {root}; the report covers the "
                f"last {settings(cfg)['trial_prs']} PRs (rk shadow-report)",
            )
        )
    return out
