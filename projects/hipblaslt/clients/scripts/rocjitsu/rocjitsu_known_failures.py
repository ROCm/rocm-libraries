# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Apply reviewed kernel-name skips after the sweep has selected its cases."""

from datetime import date
import hashlib
import re

import yaml


def known_failures(text):
    """Reject malformed entries instead of silently dropping intended coverage."""
    data = yaml.safe_load(text)

    def require(condition, message):
        if not condition:
            raise ValueError(f"Invalid known-failure list: {message}")

    def keys(value, expected):
        require(
            isinstance(value, dict) and set(value) == set(expected.split()), expected
        )

    def strings(value, allowed=None):
        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(v, str) and v.strip() for v in value)
            and len(set(value)) == len(value)
            and (allowed is None or set(value) <= allowed)
        )

    def digest(value, size):
        return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{size}}}", value)

    keys(data, "version rocjitsu_revision case_policy failures")
    require(type(data["version"]) is int and data["version"] == 3, "version must be 3")
    require(digest(data["rocjitsu_revision"], 40), "expected a full rocJITsu revision")
    require(isinstance(data["case_policy"], str) and data["case_policy"], "case_policy")
    require(isinstance(data["failures"], list), "failures must be a list")
    ids = set()
    for entry in data["failures"]:
        keys(entry, "id reason tracking review_by target backends cases kernel_names")
        for name in ("id", "reason"):
            require(isinstance(entry[name], str) and entry[name].strip(), name)
        require(entry["id"] not in ids, f"duplicate id: {entry['id']}")
        ids.add(entry["id"])
        require(strings(entry["tracking"]), f"{entry['id']}: tracking links required")
        require(
            all(re.fullmatch(r"https://\S+", link) for link in entry["tracking"]),
            f"{entry['id']}: expected HTTPS tracking links",
        )
        entry["review_by"] = str(entry["review_by"])
        date.fromisoformat(entry["review_by"])
        require(entry["target"] in ("gfx942", "gfx950"), "unsupported target")
        require(strings(entry["backends"], {"tensile", "bench"}), "backends")
        require(
            strings(entry["cases"], {"tile", "multi_tile", "mn_edge", "k_remainder"}),
            "cases must name individual cases; list all four to skip a solution",
        )
        require(strings(entry["kernel_names"]), "kernel_names")
    return data


def apply_known_failures(manifest, text, revision, run_known_failures=False):
    """Match kernels across launch configurations, solution names and indices."""
    data = known_failures(text)
    if not run_known_failures:
        for name, actual in (
            ("rocjitsu_revision", revision),
            ("case_policy", manifest["policy"]),
        ):
            if data[name] != actual:
                raise ValueError(
                    f"Known-failure {name} {data[name]} does not match {actual}; "
                    "use --run-known-failures to recheck before updating the list"
                )
    matches = {}
    for entry in data["failures"]:
        if (
            entry["target"] != manifest["target"]
            or manifest["backend"] not in entry["backends"]
        ):
            continue
        for kernel_name in entry["kernel_names"]:
            for case_id in entry["cases"]:
                key = (kernel_name, case_id)
                if key in matches:
                    raise ValueError(
                        f"Overlapping known failures: {entry['id']}, {matches[key]['id']}"
                    )
                matches[key] = entry
    matched = 0
    for job in manifest["jobs"]:
        # Skips describe runnable native cases, never missing inputs or planner errors.
        if "planning_error" in job:
            continue
        kernel_name = job["solutions"][0]["kernel"]
        for case in job["cases"]:
            entry = matches.get((kernel_name, case["id"]))
            if entry is None:
                continue
            matched += 1
            if run_known_failures:
                continue
            case["skip"] = {
                k: entry[k] for k in ("id", "reason", "tracking", "review_by")
            }
    manifest["rocjitsu_revision"] = revision
    manifest["known_failures"] = {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "verified_revision": data["rocjitsu_revision"],
        "run_known_failures": run_known_failures,
        "matched_cases": matched,
        "skipped_cases": 0 if run_known_failures else matched,
    }
