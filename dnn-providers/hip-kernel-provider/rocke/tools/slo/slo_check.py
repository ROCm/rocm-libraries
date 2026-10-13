# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare measured numbers with the baseline and produce findings.

A finding is a plain dict:
  check   SLO1xx id
  kind    "worse"        old file got worse than its baseline
          "over_target"  new file is over a target
          "better"       old file got better (progress, never a warning)
          "error"        the tool could not measure something (exit code 2)
  path, line, metric, old, new, target, message
  enforce False for report-only metrics; those never fail, even with --fail-on
"""
from __future__ import annotations

import json
from pathlib import Path

import slo_config as C

CHECK_ID = {
    "sloc": "SLO101",
    "longest_fn": "SLO102",
    "raw_per_100": "SLO103",
    "hcr": "SLO104",
    "spec_bools_max": "SLO105",
    "overlap_max": "SLO106",
}
CHECK_TITLE = {
    "SLO101": "File SLOC",
    "SLO102": "Longest function",
    "SLO103": "Raw IR ops per 100 SLOC",
    "SLO104": "Helper-call ratio (report only)",
    "SLO105": "Bool fields per frozen spec",
    "SLO106": "Copy-paste overlap",
    "SLO107": "Repeated low-level patterns",
}


def check_of(metric: str) -> str:
    return "SLO107" if metric.startswith("pat.") else CHECK_ID[metric]


def tracked(r: dict) -> dict:
    """The numbers the baseline keeps for one file."""
    out = {k: r.get(k) for k in C.WORSE_IF if k in r}
    out.update({f"pat.{k}": v for k, v in r["patterns"].items()})
    return out


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {"meta": {}, "files": {}}
    return json.loads(path.read_text())


def save_baseline(path: Path, current: dict, meta: dict) -> None:
    doc = {
        "meta": meta,
        "files": {p: tracked(r) for p, r in sorted(current["files"].items())},
    }
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _line_for(metric: str, r: dict) -> int:
    if metric == "longest_fn":
        return r["longest_fn_line"]
    if metric == "spec_bools_max":
        return r["spec_bools_line"]
    return 1


def _is_worse(metric: str, old, new) -> bool | None:
    """True worse, False better, None same (within tolerance)."""
    direction = "up" if metric.startswith("pat.") else C.WORSE_IF[metric]
    tol = C.TOLERANCE.get(metric, 0)
    if abs(new - old) <= tol:
        return None
    return (new > old) if direction == "up" else (new < old)


def _describe(metric: str, r: dict) -> str:
    if metric == "longest_fn":
        return f" ({r['longest_fn_name']})"
    if metric == "spec_bools_max":
        return f" ({r['spec_bools_class']})"
    if metric == "overlap_max" and r.get("overlap_with"):
        return f" (with {r['overlap_with']})"
    return ""


def compare(
    current: dict,
    baseline: dict,
    only: set[str] | None = None,
    renamed: dict[str, str] | None = None,
) -> list[dict]:
    """Findings for files in `only` (all measured files when None)."""
    renamed = renamed or {}
    base_files = baseline.get("files", {})
    findings = list(current["errors"])
    for rel, r in sorted(current["files"].items()):
        if only is not None and rel not in only:
            continue
        key = rel if rel in base_files else renamed.get(rel)
        now = tracked(r)
        if key in base_files:
            old_vals = base_files[key]
            for metric, new in now.items():
                old = old_vals.get(metric)
                if old is None or new is None:
                    continue
                verdict = _is_worse(metric, old, new)
                if verdict is None:
                    continue
                cid = check_of(metric)
                was = f" (baseline: {key})" if key != rel else ""
                findings.append(
                    {
                        "check": cid,
                        "kind": "worse" if verdict else "better",
                        "path": rel,
                        "line": _line_for(metric, r),
                        "metric": metric,
                        "old": old,
                        "new": new,
                        "target": None,
                        "enforce": metric not in C.REPORT_ONLY,
                        "message": f"{metric} {old} -> {new}{_describe(metric, r)}{was}",
                    }
                )
        else:
            for metric, (limit, direction) in C.TARGETS.items():
                new = now.get(metric)
                if new is None:
                    continue
                bad = new > limit if direction == "max" else new < limit
                if bad:
                    findings.append(
                        {
                            "check": check_of(metric),
                            "kind": "over_target",
                            "path": rel,
                            "line": _line_for(metric, r),
                            "metric": metric,
                            "old": None,
                            "new": new,
                            "target": limit,
                            "enforce": metric not in C.REPORT_ONLY,
                            "message": f"new file: {metric} {new}{_describe(metric, r)}, target "
                            f"{'<=' if direction == 'max' else '>='} {limit}",
                        }
                    )
    return findings
