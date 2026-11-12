# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Merge completed GTest reports without inventing missing verdicts."""

from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET


_COUNT_ATTRIBUTES = {
    "passed": "passed",
    "failed": "failures",
    "skipped": "skipped",
    "disabled": "disabled",
    "total": "tests",
}


def _disabled(suite: str, test: str) -> bool:
    return any(
        part.startswith("DISABLED_")
        for name in (suite, test)
        for part in name.split("/")
    )


def _duration(value: str | None, context: str) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid time for {context}: {value!r}") from exc
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"Invalid time for {context}: {value!r}")
    return duration


def _outcome(case: ET.Element, context: str) -> str:
    status, result = case.get("status"), case.get("result")
    if (status, result) not in {
        ("run", "completed"),
        ("run", "skipped"),
        ("notrun", "suppressed"),
    }:
        raise ValueError(
            f"Not a completed GTest execution row for {context}: "
            f"status={status!r}, result={result!r} (listing XML is not a report)"
        )
    _duration(case.get("time"), context)
    if case.find("error") is not None:
        raise ValueError(f"Unexpected GTest error element for {context}")
    failed = case.find("failure") is not None
    skipped = case.find("skipped") is not None
    if status == "notrun":
        if failed or skipped:
            raise ValueError(
                f"Suppressed row contains execution diagnostics: {context}"
            )
        return "disabled"
    # Skip diagnostics after earlier assertion failures must not hide failures.
    if failed:
        return "failed"
    if (result == "skipped") != skipped:
        raise ValueError(f"Inconsistent skipped result and diagnostics: {context}")
    return "skipped" if skipped else "passed"


def _check_counts(element: ET.Element, counts: dict[str, int], context: str) -> None:
    for key in ("total", "failed", "skipped"):
        attribute = _COUNT_ATTRIBUTES[key]
        value = element.get(attribute)
        if value is None and key == "skipped" and element.tag == "testsuites":
            continue  # GTest emits skipped on suites, but not the root.
        try:
            declared = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid {attribute} for {context}") from exc
        if declared != counts[key]:
            raise ValueError(
                f"Inconsistent {attribute} for {context}: "
                f"declared {declared}, found {counts[key]}"
            )
    if element.get("errors") != "0":
        raise ValueError(f"Missing or nonzero GTest errors for {context}")


def _counts(cases: list[ET.Element]) -> dict[str, int]:
    counts = dict.fromkeys(_COUNT_ATTRIBUTES, 0)
    for case in cases:
        counts[_outcome(case, case.get("name", "unnamed testcase"))] += 1
        counts["total"] += 1
    return counts


def _set_counts(element: ET.Element, counts: dict[str, int]) -> None:
    for key, attribute in _COUNT_ATTRIBUTES.items():
        element.set(attribute, str(counts[key]))
    element.set("errors", "0")


def merge_reports(
    report_paths: list[Path],
    expected_names: list[str],
    output: Path,
    elapsed_seconds: float,
) -> dict[str, int]:
    """Write a complete report and return exclusive testcase outcome counts.

    Suppressed shard copies lose to actual executions. Only canonically
    DISABLED_ suite/test components may stay suppressed. Invalid, partial,
    missing or duplicate executions raise before writing output. Real failures
    are retained and returned. Root time is phase wall time, not shard-time sum.
    """
    elapsed = _duration(str(elapsed_seconds), "merged phase")
    expected = set(expected_names)
    if len(expected) != len(expected_names):
        raise ValueError("Duplicate expected GTest identities")
    if not report_paths:
        raise ValueError("No GTest execution reports supplied")

    selected: dict[str, ET.Element] = {}
    suite_attributes: dict[str, dict[str, str]] = {}
    suite_properties: dict[str, list[ET.Element]] = {}
    root_properties: list[ET.Element] = []
    timestamps: list[str] = []
    for path in report_paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise ValueError(
                f"Cannot read GTest execution report {path}: {exc}"
            ) from exc
        if root.tag != "testsuites":
            raise ValueError(f"Invalid GTest report root in {path}: {root.tag}")
        if any(child.tag not in {"testsuite", "properties"} for child in root):
            raise ValueError(
                f"Unexpected report-level diagnostics or elements in {path}"
            )
        root_properties.extend(root.findall("properties"))
        if root.get("timestamp"):
            timestamps.append(root.get("timestamp"))
        report_cases = []
        for suite in root.findall("testsuite"):
            suite_name = suite.get("name")
            if not suite_name:
                raise ValueError(f"Missing GTest suite name in {path}")
            if any(child.tag not in {"testcase", "properties"} for child in suite):
                raise ValueError(
                    f"Unexpected suite-level diagnostics in {path}: {suite_name}"
                )
            suite_attributes.setdefault(suite_name, dict(suite.attrib))
            suite_properties.setdefault(suite_name, suite.findall("properties"))
            cases = suite.findall("testcase")
            for case in cases:
                test_name = case.get("name")
                if not test_name:
                    raise ValueError(
                        f"Missing GTest testcase name in {path}: {suite_name}"
                    )
                name = f"{suite_name}.{test_name}"
                if name not in expected:
                    raise ValueError(f"Unexpected GTest identity in {path}: {name}")
                _outcome(case, f"{path}: {name}")
                if case.get("classname") != suite_name:
                    raise ValueError(f"Inconsistent GTest classname in {path}: {name}")
                previous = selected.get(name)
                if previous is not None and previous.get("status") == "run":
                    if case.get("status") == "run":
                        raise ValueError(
                            f"Duplicate actual GTest execution: {name} ({path})"
                        )
                elif previous is None or case.get("status") == "run":
                    selected[name] = case
            _check_counts(suite, _counts(cases), f"{path}: {suite_name}")
            report_cases.extend(cases)
        _check_counts(root, _counts(report_cases), str(path))

    missing = sorted(expected - selected.keys())
    if missing:
        raise ValueError(
            f"Missing expected GTest identities ({len(missing)}): {', '.join(missing)}"
        )
    not_run = [
        name
        for name, case in selected.items()
        if case.get("status") != "run"
        and not _disabled(case.get("classname"), case.get("name"))
    ]
    if not_run:
        raise ValueError(
            f"Expected runnable GTest cases did not run: {', '.join(sorted(not_run))}"
        )

    merged = ET.Element("testsuites", name="AllTests", time=str(elapsed))
    if timestamps:
        merged.set("timestamp", min(timestamps))
    for properties in root_properties:
        merged.append(deepcopy(properties))
    suites: dict[str, ET.Element] = {}
    for name in sorted(selected):
        case = selected[name]
        suite_name = case.get("classname")
        if suite_name not in suites:
            suite = ET.SubElement(merged, "testsuite", suite_attributes[suite_name])
            for properties in suite_properties[suite_name]:
                suite.append(deepcopy(properties))
            suites[suite_name] = suite
        suites[suite_name].append(deepcopy(case))
    for suite in suites.values():
        cases = suite.findall("testcase")
        _set_counts(suite, _counts(cases))
        suite.set(
            "time",
            str(sum(_duration(case.get("time"), suite.get("name")) for case in cases)),
        )
    counts = _counts(list(selected.values()))
    _set_counts(merged, counts)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(merged)
    ET.ElementTree(merged).write(output, encoding="utf-8", xml_declaration=True)
    return counts
