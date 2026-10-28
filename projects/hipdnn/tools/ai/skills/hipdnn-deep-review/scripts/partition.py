#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Partition helper for the hipdnn-deep-review RUNBOOK (Orient area table, Partition, Gate).

Every subcommand works on one round directory ($EVIDENCE/r<N>, --round-dir, default `.`)
holding files.txt and numstat.txt, and exits non-zero with a one-line reason on failure.

  areas    files.txt -> areas.tsv (`path<TAB>slug`) by the RUNBOOK Orient table
  plan     areas.tsv + numstat.txt -> owners.tsv (`owner<TAB>path`) and a Lanes summary
  packets  owners.tsv + numstat.txt + areas.tsv -> packets/<LANE>.md
  gate     numstat.txt vs owners.tsv: owned once, in numstat, owned lines sum to the total

A `--sampled` file has one `path<TAB>generator[<TAB>sample]` row per generated, golden or
bundle file: `generator` is the generator or source of truth, a third column `sample` marks
the files the lane reads.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reviewlib as rl  # noqa: E402

TARGET_LINES = 2000

# First matching row wins; a prefix ending in "/" matches a directory, else the exact path.
AREA_ROWS = [
    ("frontend", ("projects/hipdnn/frontend/", "projects/hipdnn/python/")),
    ("backend", ("projects/hipdnn/backend/",)),
    ("data-sdk", ("projects/hipdnn/data_sdk/", "projects/hipdnn/flatbuffers_sdk/")),
    ("plugin-sdk", ("projects/hipdnn/plugin_sdk/",)),
    ("test-sdk", ("projects/hipdnn/test_sdk/",)),
    ("integration-tests", ("dnn-providers/integration-tests/",)),
    ("provider-*", ("dnn-providers/*/",)),
    ("core-tests", ("projects/hipdnn/tests/",)),
    (
        "core-build",
        (
            "projects/hipdnn/cmake/",
            "projects/hipdnn/CMakeLists.txt",
            "projects/hipdnn/scripts/",
        ),
    ),
    ("build-ci", ("CMakeLists.txt", "cmake/", "CMakePresets.json", ".github/")),
    ("shared-*", ("shared/*/",)),
    ("samples", ("projects/hipdnn/samples/",)),
    ("tooling", ("projects/hipdnn/tools/", "tools/")),
]

PROJECT_ROOTS = ("projects", "dnn-providers", "shared")
LICENSE_STEMS = {"LICENSE", "CONTRIBUTING", "SECURITY", "README"}
CONCERN_LANES = {
    "TESTQ",
    "TESTB",
    "AUTH",
    "BUILD",
    "DEPS",
    "API",
    "ROCK",
    "LEAK",
    "DUP",
    "DOCS",
    "RUN",
    "DRIFT",
}


def fail(msg: str):
    sys.exit(f"partition: {msg}")


def is_docs(path: str) -> bool:
    """Prose no tool reads (policy.md *Review rules*): markdown, reStructuredText, license
    files, and plain text under a docs directory. Config files and CMake/requirements
    lists are not documentation."""
    parts = path.split("/")
    name = parts[-1]
    if name.split(".")[0].upper() in LICENSE_STEMS:
        return True
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in ("md", "markdown", "rst"):
        return True
    return ext == "txt" and any(p in ("docs", "doc") for p in parts[:-1])


def _matches(path: str, prefix: str) -> str | None:
    """Return the `*` capture (empty when none) if path matches the prefix, else None."""
    if "*" in prefix:
        head, _ = prefix.split("*")
        if path.startswith(head):
            rest = path[len(head) :]
            seg, slash, _ = rest.partition("/")
            if seg and slash:
                return seg
        return None
    if prefix.endswith("/"):
        return "" if path.startswith(prefix) else None
    return "" if path == prefix else None


def area_of(path: str) -> str:
    for slug, prefixes in AREA_ROWS:
        for prefix in prefixes:
            cap = _matches(path, prefix)
            if cap is not None:
                return slug.replace("*", cap)
    if is_docs(path):
        return "docs"
    parts = path.split("/")
    if parts[0] in PROJECT_ROOTS and len(parts) > 2:
        project, below = parts[1], parts[2:]
        return f"{project}-{below[0]}" if len(below) > 1 else project
    if len(parts) > 2:
        return f"{parts[0]}-{parts[1]}"
    return parts[0] if len(parts) == 2 else "root"


def read_lines(path: Path, what: str) -> list[str]:
    if not path.is_file():
        fail(f"{what} not found: {path}")
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]


def read_numstat(round_dir: Path) -> list[tuple[str, int, str]]:
    """Rows of (path, lines, raw row); a binary `-` row is zero lines."""
    rows = []
    for ln in read_lines(round_dir / "numstat.txt", "numstat.txt"):
        f = ln.split("\t", 2)
        if len(f) != 3:
            fail(f"malformed numstat row: {ln!r}")
        added, deleted, path = f
        lines = 0 if added == "-" else int(added) + int(deleted)
        rows.append((path, lines, ln))
    return rows


def read_areas(round_dir: Path) -> dict[str, str]:
    areas = {}
    for ln in read_lines(round_dir / "areas.tsv", "areas.tsv (run `areas` first)"):
        path, _, slug = ln.partition("\t")
        areas[path] = slug
    return areas


def read_owners(round_dir: Path) -> list[tuple[str, str]]:
    """(owner, path) rows of owners.tsv; a row without a tab or with an owner that is not
    `CORR<k>` or `DOCS` is a failure."""
    rows = []
    for ln in read_lines(round_dir / "owners.tsv", "owners.tsv"):
        owner, sep, path = ln.partition("\t")
        if not sep:
            fail(f"malformed owners.tsv row (no tab): {ln!r}")
        if not re.fullmatch(r"CORR\d+|DOCS", owner):
            fail(f"owners.tsv owner is not CORR<k> or DOCS: {owner!r}")
        rows.append((owner, path))
    return rows


def read_sampled(sampled: str | None) -> dict[str, tuple[str, bool]]:
    """path -> (generator, in_sample)."""
    out = {}
    if sampled:
        for ln in read_lines(Path(sampled), "--sampled file"):
            f = ln.split("\t")
            if len(f) < 2 or not f[1]:
                fail(f"--sampled row needs path<TAB>generator: {ln!r}")
            out[f[0]] = (f[1], len(f) > 2 and f[2] == "sample")
    return out


def cmd_areas(args) -> None:
    """Assign each path of files.txt to an area slug; write areas.tsv."""
    rd = Path(args.round_dir)
    files = read_lines(rd / "files.txt", "files.txt")
    rl.write(rd / "areas.tsv", "".join(f"{p}\t{area_of(p)}\n" for p in files))
    print(f"areas.tsv: {len(files)} files, {len({area_of(p) for p in files})} areas")


def test_stem(path: str) -> str | None:
    """Lowercased stem of a test file with its `test_` or `Test` prefix stripped, else None."""
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem.startswith("test_"):
        return stem[5:].lower()
    if stem.startswith("Test") and len(stem) > 4:
        return stem[4:].lower()
    return None


def pair_tests(paths: list[str]) -> dict[str, str]:
    """path -> unit key for each changed test and its sources. A test pairs with every
    changed non-test file of its stem when they all sit in one directory (`X.cpp` and
    `X.hpp`); with no such file, or same-stem files in several directories, it is left out.
    The key is the first source."""
    sources: dict[str, list[str]] = {}
    for p in paths:
        if test_stem(p) is None:
            sources.setdefault(
                p.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower(), []
            ).append(p)
    unit = {}
    for p in paths:
        stem = test_stem(p)
        found = sources.get(stem, []) if stem is not None else []
        if found and len({q.rpartition("/")[0] for q in found}) == 1:
            unit[p] = found[0]
            unit.update({q: found[0] for q in found})
    return unit


def plan_lanes(
    files: list[tuple[str, int, str]], areas: dict[str, str]
) -> list[list[tuple[str, int, str]]]:
    """Pack area groups (first-appearance order) into lanes of ~TARGET_LINES lines.

    A test file travels with its source (pair_tests) in the source's area group. A total
    within target is one lane; an area over target splits by unit; a unit over target gets
    its own lane; no file is split from its unit."""
    if sum(f[1] for f in files) <= TARGET_LINES:
        return [files] if files else []
    pairs = pair_tests([f[0] for f in files])
    units: dict[str, list] = {}
    for f in files:
        units.setdefault(pairs.get(f[0], f[0]), []).append(f)
    groups: dict[str, list[list]] = {}
    for key, unit in units.items():
        groups.setdefault(areas[key], []).append(unit)
    lanes: list[list] = []
    cur: list = []
    cur_lines = 0

    def flush():
        nonlocal cur, cur_lines
        if cur:
            lanes.append(cur)
        cur, cur_lines = [], 0

    for group in groups.values():
        g_lines = sum(f[1] for unit in group for f in unit)
        if g_lines <= TARGET_LINES:
            if cur_lines + g_lines > TARGET_LINES:
                flush()
            cur += [f for unit in group for f in unit]
            cur_lines += g_lines
            continue
        flush()
        for unit in group:
            u_lines = sum(f[1] for f in unit)
            if u_lines > TARGET_LINES:
                flush()
                lanes.append(unit)
            else:
                if cur_lines + u_lines > TARGET_LINES:
                    flush()
                cur += unit
                cur_lines += u_lines
        flush()
    flush()
    return lanes


def cmd_plan(args) -> None:
    """Propose owners.tsv: CORR lanes of ~2000 lines grouped by area, sampled files in
    their own lane, an all-docs change owned by DOCS alone. Prints the Lanes summary."""
    rd = Path(args.round_dir)
    rows = read_numstat(rd)
    areas = read_areas(rd)
    sampled = read_sampled(args.sampled)
    for path, _, _ in rows:
        if path not in areas:
            fail(f"{path} is in numstat.txt but not in areas.tsv")

    owners: list[tuple[str, list]] = []
    if all(is_docs(p) for p, _, _ in rows):
        owners.append(("DOCS", rows))
    else:
        normal = [r for r in rows if r[0] not in sampled]
        lanes = plan_lanes(normal, areas)
        extra = [r for r in rows if r[0] in sampled]
        if extra:
            lanes.append(extra)
        owners = [(f"CORR{i}", lane) for i, lane in enumerate(lanes, 1)]
    if not owners:
        fail("numstat.txt has no rows")

    rl.write(
        rd / "owners.tsv",
        "".join(f"{o}\t{p}\n" for o, lane in owners for p, _, _ in lane),
    )
    for o, lane in owners:
        slugs = sorted({areas[p] for p, _, _ in lane})
        tag = " (sampled)" if all(p in sampled for p, _, _ in lane) else ""
        print(
            f"{o}: {len(lane)} files, {sum(n for _, n, _ in lane)} lines, areas {','.join(slugs)}{tag}"
        )


def orient_paths(slugs, rd: Path) -> list[str]:
    out = []
    for slug in sorted(slugs):
        if slug == "docs":
            continue
        rel = f"orient/{slug}.md"
        out.append(
            rel if (rd / rel).is_file() else f"{rel} (absent: scout failed or none)"
        )
    return out


def concern_scope(lane: str, rows):
    cmake = (
        lambda p: p.rsplit("/", 1)[-1] == "CMakeLists.txt"
        or p.endswith(".cmake")
        or p.endswith(".cmake.in")
        or p.rsplit("/", 1)[-1] == "CMakePresets.json"
        or p.startswith(".github/")
    )
    name = lambda p: p.rsplit("/", 1)[-1]
    if lane == "BUILD":
        return [
            r
            for r in rows
            if cmake(r[0])
            or name(r[0]) == "version.json"
            or re.fullmatch(r".*test_categories.*\.yaml", name(r[0]))
        ]
    if lane == "DEPS":
        return [
            r
            for r in rows
            if cmake(r[0])
            or name(r[0]) == "pyproject.toml"
            or re.fullmatch(r"requirements.*\.txt", name(r[0]))
            or re.search(r"\.(c|cc|cpp|cxx|h|hh|hpp|hxx|hip|cu|cuh)$", name(r[0]))
        ]
    return rows


def drift_trigger(rd: Path) -> str:
    """The value of the `Trigger:` line of drift.txt (`yes` or `no (<reason>)`)."""
    path = rd / "drift.txt"
    if not path.is_file():
        fail(f"drift.txt not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Trigger:"):
            return line.partition(":")[2].strip()
    fail("drift.txt has no Trigger line")


def write_packet(rd: Path, lane: str, kind: str, rows, areas, sampled, parent) -> None:
    out = [f"# Packet: {lane}", "", f"Lane: {lane} ({kind})", ""]
    out += [
        "## Files" if kind == "correctness" else "## Changed files in scope",
        "",
        "| path | added | deleted | area |",
        "|---|---|---|---|",
    ]
    for path, _, raw in rows:
        added, deleted, _ = raw.split("\t", 2)
        out.append(f"| `{path}` | {added} | {deleted} | {areas[path]} |")
    out += ["", "## Orient files", ""]
    out += [f"- {p}" for p in orient_paths({areas[r[0]] for r in rows}, rd)] or [
        "- none"
    ]
    if kind == "concern":
        out += ["", f"Stack parent: {parent or 'none'}"]
    if lane == "DRIFT":
        out += ["", "## Target drift", "", "- drift.txt"]
    marked = [r[0] for r in rows if r[0] in sampled]
    if marked and kind == "correctness":
        gens = sorted({sampled[p][0] for p in marked})
        sample = [p for p in marked if sampled[p][1]]
        if not sample:
            fail(f"{lane} owns sampled files but --sampled marks no sample file for it")
        out += ["", f"Sampled: {', '.join(gens)}; sample {', '.join(sample)}"]
    if (rd / "range-diff.txt").is_file():
        out += [
            "",
            "## Round delta",
            "",
            "- range-diff.txt",
            "- affected.txt",
            "- ledger rows: ../LEDGER.md rows in scope",
        ]
    packets = rd / "packets"
    packets.mkdir(exist_ok=True)
    rl.write(packets / f"{lane}.md", "\n".join(out) + "\n")


def cmd_packets(args) -> None:
    """Write packets/<LANE>.md for each lane of owners.tsv (files with numstat rows, areas,
    Orient paths, `Sampled:`) and each concern lane named by --lanes (changed files in
    scope). BUILD and DEPS scope to their trigger files, other concern lanes take every
    changed file; DRIFT also lists drift.txt and is N/A when its Trigger is `no`. The diff
    is not read."""
    rd = Path(args.round_dir)
    rows = {r[0]: r for r in read_numstat(rd)}
    areas = read_areas(rd)
    sampled = read_sampled(args.sampled)
    owned: dict[str, list] = {}
    for owner, path in read_owners(rd):
        if path not in rows:
            fail(f"owners.tsv path not in numstat.txt: {path}")
        owned.setdefault(owner, []).append(rows[path])
    for path in rows:
        if path not in areas:
            fail(f"{path} is in numstat.txt but not in areas.tsv")
    for lane, lane_rows in owned.items():
        write_packet(rd, lane, "correctness", lane_rows, areas, sampled, None)
        print(f"packets/{lane}.md: {len(lane_rows)} files")
    for lane in [x.strip() for x in args.lanes.split(",")] if args.lanes else []:
        if lane not in CONCERN_LANES:
            fail(f"unknown concern lane: {lane}")
        if lane in owned:
            continue
        if lane == "DRIFT":
            trigger = drift_trigger(rd)
            if trigger.startswith("no"):
                print(f"DRIFT: N/A {trigger[2:].strip()}")
                continue
        scope = concern_scope(lane, list(rows.values()))
        if not scope:
            print(f"{lane}: no changed file in scope, N/A")
            continue
        write_packet(rd, lane, "concern", scope, areas, sampled, args.parent)
        print(f"packets/{lane}.md: {len(scope)} files")


def cmd_gate(args) -> None:
    """Partition Gate: every numstat path owned exactly once, every owned path in
    numstat, owned lines equal the total. Exit 0 only then."""
    rd = Path(args.round_dir)
    if not (rd / "owners.tsv").is_file() or not (rd / "owners.tsv").stat().st_size:
        fail("owners.tsv empty")
    own: dict[str, int] = {}
    for _, path in read_owners(rd):
        own[path] = own.get(path, 0) + 1
    rows = read_numstat(rd)
    seen = {p for p, _, _ in rows}
    total = owned = 0
    bad = False
    for path, lines, _ in rows:
        total += lines
        if path in own:
            owned += lines
        else:
            print(f"unowned: {path}")
            bad = True
    for path in sorted(own):
        if own[path] > 1:
            print(f"owned twice: {path}")
            bad = True
        if path not in seen:
            print(f"not in numstat: {path}")
            bad = True
    if owned != total:
        print(f"owned {owned} of {total} lines")
        bad = True
    print(f"{len(rows)} files, {total} lines")
    if bad:
        sys.exit(1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("areas", cmd_areas),
        ("plan", cmd_plan),
        ("packets", cmd_packets),
        ("gate", cmd_gate),
    ):
        p = sub.add_parser(
            name, help=fn.__doc__.splitlines()[0], description=fn.__doc__
        )
        p.add_argument(
            "--round-dir",
            default=".",
            help="$EVIDENCE/r<N> (default: current directory)",
        )
        p.set_defaults(fn=fn)
        if name in ("plan", "packets"):
            p.add_argument(
                "--sampled", help="file of generated/golden paths (see module docs)"
            )
        if name == "packets":
            p.add_argument(
                "--lanes",
                help="comma-separated concern lanes to write, e.g. TESTQ,BUILD",
            )
            p.add_argument("--parent", help="stack parent for concern-lane packets")
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
