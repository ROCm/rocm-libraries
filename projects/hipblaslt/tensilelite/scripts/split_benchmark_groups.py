#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Plan, render, and verify complete-group TensileLite YAML partitions.

Scan any repository directories containing unsplit YAMLs, optionally reading them
from a Git ref. In-place writes require an unsplit Git ref; scan records named
refs as commit IDs so committing the outputs does not change the inputs.
A timing JSON maps repository-relative filenames to measured
seconds. The planner repeatedly bisects the estimated longest part, stopping at
the target duration or an added-startup budget. Its FLOP, parameter-count, and
text-size weights are only relative costs; projected times are not measurements.
Untimed files stay whole unless --untimed-max-parts is requested explicitly.

A reviewed plan records group counts in source order, non-group settings hashes,
the scanned directories, and occasional exceptions. Render reads the unsplit Git
ref even after the worktree is split. --verify --strict-bytes is read-only;
--write is the explicit update. Unexpected YAML files in the recorded directories
are rejected, including obsolete partitions. Rebuild a changed plan in a clean
unsplit checkout; the renderer does not delete files. See split_benchmark_groups.md.
By default, the first part keeps the original filename and later names are
automatic; a reviewed plan can override names or permute output slots.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import yaml


REPO = Path(__file__).resolve().parents[4]
HEADER = "# Copyright Advanced Micro Devices, Inc., or its affiliates.\n# SPDX-License-Identifier: MIT\n\n"
LOADER = yaml.CSafeLoader if hasattr(yaml, "CSafeLoader") else yaml.SafeLoader


class SplitError(ValueError):
    pass


def load_yaml(data):
    if hasattr(yaml, "CSafeLoader"):
        return yaml.load(data, Loader=yaml.CSafeLoader)
    return yaml.safe_load(data)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def relative_path(repo, path):
    path = Path(path)
    absolute = path.resolve() if path.is_absolute() else (repo / path).resolve()
    try:
        return absolute.relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise SplitError(f"path is outside the repository: {path}") from error


def source_paths(repo, directories, source_ref=None):
    dirs = [relative_path(repo, directory) for directory in directories]
    if source_ref:
        names = (
            git(repo, "ls-tree", "-r", "--name-only", source_ref, "--", *dirs).decode().splitlines()
        )
        return sorted(name for name in names if name.endswith(".yaml"))
    return sorted(
        {
            path.relative_to(repo).as_posix()
            for directory in dirs
            for path in (repo / directory).rglob("*.yaml")
        }
    )


def source_bytes(repo, path, source_ref=None):
    if source_ref:
        return git(repo, "show", f"{source_ref}:{path}")
    return (repo / path).read_bytes()


def source_mode(repo, path, source_ref=None):
    if source_ref:
        record = git(repo, "ls-tree", source_ref, "--", path).decode().strip()
        if not record:
            raise SplitError(f"missing source in {source_ref}: {path}")
        return int(record.split()[0], 8) & 0o777
    return (repo / path).stat().st_mode & 0o777


def load_config(text):
    doc = load_yaml(text)
    if not isinstance(doc, dict) or not isinstance(doc.get("BenchmarkProblems"), list):
        raise SplitError("not a BenchmarkProblems configuration")
    for entry in doc["BenchmarkProblems"]:
        if not isinstance(entry, list) or len(entry) < 2 or not isinstance(entry[0], dict):
            raise SplitError("every BenchmarkProblems entry needs a problem type and a group")
        if any(not isinstance(group, dict) for group in entry[1:]):
            raise SplitError("every benchmark group must be a mapping")
    return doc


def groups(doc):
    return [(entry[0], group) for entry in doc["BenchmarkProblems"] for group in entry[1:]]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def settings_digest(doc):
    settings = {key: value for key, value in doc.items() if key != "BenchmarkProblems"}
    return hashlib.sha256(canonical(settings).encode()).hexdigest()


def blockers(doc, path):
    reasons = []
    if list(doc)[-1] != "BenchmarkProblems":
        reasons.append("sections follow BenchmarkProblems")
    if any(word in path.lower() for word in ("xfail", "wip", "disabled")):
        reasons.append("path encodes a whole-file outcome")
    marks = (doc.get("TestParameters") or {}).get("marks") or []
    if any("xfail" in str(mark) or str(mark) == "ffm_fail" for mark in marks):
        reasons.append("whole-file expected-failure mark")
    if (doc.get("TestParameters") or {}).get("streamk_schedules"):
        reasons.append("schedule assertions need per-output review")
    return reasons


def group_weight(problem_type, group):
    variants = 1
    for item in group.get("ForkParameters") or []:
        for choices in item.values():
            if isinstance(choices, list):
                variants *= max(1, len(choices))
    flops = 0
    unknown = 0
    for final in group.get("BenchmarkFinalParameters") or []:
        for size in final.get("ProblemSizes") or []:
            exact = size.get("Exact") if isinstance(size, dict) else None
            if (
                problem_type.get("OperationType") == "GEMM"
                and isinstance(exact, list)
                and len(exact) in (3, 4)
                and all(type(x) is int and x >= 0 for x in exact)
            ):
                m, n = exact[:2]
                batch, k = (exact[2], exact[3]) if len(exact) == 4 else (1, exact[2])
                flops += 2 * m * n * batch * k
            else:
                unknown += 1
    return variants * (1 + flops / 1_000_000_000 + unknown + len(str(group)) / 1000)


def propose_sizes(
    weights,
    seconds=None,
    target_seconds=60,
    startup_seconds=3.5,
    max_overhead=0.05,
    max_parts=16,
    untimed_max_parts=1,
):
    """Return group counts for contiguous partitions of one source YAML.

    Args:
        weights: Positive relative costs, one per complete group in source
            order. Groups are never reordered or divided.
        seconds: Measured whole-file elapsed seconds, or None for an untimed
            file. Timed parts are estimated as seconds * part_weight / total.
        target_seconds: Stop when every splittable part is estimated at or
            below this duration, excluding added startup. An indivisible group
            can still exceed the target.
        startup_seconds: Estimated extra serial cost for each additional file.
        max_overhead: Maximum added startup as a fraction of seconds, not a
            percentage; 0.05 allows (parts - 1) * startup_seconds <= 0.05 * seconds.
        max_parts: Maximum output count for a timed file.
        untimed_max_parts: Output limit used instead of max_parts when seconds
            is None; duration and overhead checks then do not apply.

    Each round bisects the highest-weight part that still contains multiple
    groups, at the boundary that most evenly divides its weight. Ties choose
    the earliest part or boundary. The result contains group counts, not
    weights or times: [1, 2, 1] assigns four groups to three output files.
    """
    if len(weights) < 2:
        return [len(weights)]
    intervals = [(0, len(weights))]
    total = sum(weights)
    limit = min(len(weights), max_parts if seconds is not None else untimed_max_parts)
    while len(intervals) < limit:
        eligible = [
            (sum(weights[start:end]), index, start, end)
            for index, (start, end) in enumerate(intervals)
            if end - start > 1
        ]
        if not eligible:
            break
        cost, index, start, end = max(eligible, key=lambda item: (item[0], -item[1]))
        if seconds is not None:
            if seconds * cost / total <= target_seconds:
                break
            if len(intervals) * startup_seconds > seconds * max_overhead:
                break
        cut = min(
            range(start + 1, end),
            key=lambda i: (abs(sum(weights[start:i]) - sum(weights[i:end])), i),
        )
        intervals[index : index + 1] = [(start, cut), (cut, end)]
    return [end - start for start, end in intervals]


def anchor_values(text):
    events = list(yaml.parse(text))
    definitions = {}
    for index, event in enumerate(events):
        anchor = getattr(event, "anchor", None)
        if not anchor or isinstance(event, yaml.events.AliasEvent):
            continue
        end = event.end_mark.index
        if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
            depth = 1
            for following in events[index + 1 :]:
                if isinstance(
                    following, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)
                ):
                    depth += 1
                elif isinstance(
                    following, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)
                ):
                    depth -= 1
                if depth == 0:
                    end = following.end_mark.index
                    break
        line_start = text.rfind("\n", 0, event.start_mark.index) + 1
        indent = len(text[line_start:]) - len(text[line_start:].lstrip(" "))
        value_lines = text[event.start_mark.index : end].splitlines(keepends=True)
        # YAML end marks can include comments that introduce the next group.
        while value_lines and (
            not value_lines[-1].strip() or value_lines[-1].lstrip().startswith("#")
        ):
            value_lines.pop()
        definitions[anchor] = (
            "".join(value_lines).rstrip(),
            indent,
            getattr(event, "flow_style", None),
        )
    return definitions


def localize_anchors(part, original, expand_all=False):
    definitions = anchor_values(original)
    replacements = 0
    while True:
        known = set()
        alias = None
        for token in yaml.scan(part):
            if isinstance(token, yaml.tokens.AnchorToken):
                known.add(token.value)
            elif isinstance(token, yaml.tokens.AliasToken) and (
                expand_all or token.value not in known
            ):
                alias = token
                break
        if alias is None:
            if expand_all:
                anchors = [
                    token for token in yaml.scan(part) if isinstance(token, yaml.tokens.AnchorToken)
                ]
                for token in reversed(anchors):
                    part = part[: token.start_mark.index] + part[token.end_mark.index :]
            return part
        if alias.value not in definitions or replacements >= 100:
            raise SplitError(f"cannot expand YAML anchor {alias.value}")
        definition, source_indent, flow_style = definitions[alias.value]
        line_start = part.rfind("\n", 0, alias.start_mark.index) + 1
        line = part[line_start:]
        target_indent = len(line) - len(line.lstrip(" "))
        delta = target_indent - source_indent
        lines = definition.splitlines(keepends=True)
        if flow_style is False and len(lines) > 1:
            content_indent = min(
                len(x) - len(x.lstrip(" "))
                for x in lines[1:]
                if x.strip() and not x.lstrip().startswith("#")
            )
            before_alias = part[line_start : alias.start_mark.index].rstrip()
            key_indent = target_indent + (2 if before_alias.lstrip().startswith("- ") else 0)
            target_content_indent = key_indent + 2 if ":" in before_alias else target_indent + 2
            delta = target_content_indent - content_indent
        for index in range(1, len(lines)):
            if lines[index].strip():
                lines[index] = " " * delta + lines[index] if delta >= 0 else lines[index][-delta:]
        line_end = part.find("\n", alias.end_mark.index)
        if line_end < 0:
            line_end = len(part)
        trailing = part[alias.end_mark.index : line_end]
        if len(lines) > 1 and trailing.lstrip().startswith("#"):
            comment = " " * target_indent + trailing.lstrip() + "\n"
            part = (
                part[:line_start]
                + comment
                + part[line_start : alias.start_mark.index]
                + "".join(lines)
                + part[line_end:]
            )
        else:
            part = part[: alias.start_mark.index] + "".join(lines) + part[alias.end_mark.index :]
        replacements += 1


def partition_text(text, sizes, expand_all_anchors=False):
    doc = load_config(text)
    units = groups(doc)
    if any(size < 1 for size in sizes) or sum(sizes) != len(units):
        raise SplitError(f"part sizes {sizes} do not cover {len(units)} groups")
    lines = text.splitlines(keepends=True)
    tree = yaml.compose(text, Loader=LOADER)
    key, problems = next(
        (key, value) for key, value in tree.value if key.value == "BenchmarkProblems"
    )
    if tree.value[-1][0].value != "BenchmarkProblems":
        raise SplitError("sections follow BenchmarkProblems")
    entries = [token for token in yaml.scan(text) if isinstance(token, yaml.tokens.BlockEntryToken)]

    def comments_before(line, lower):
        # Keep leading comments with the group they describe.
        while line > lower and (
            not lines[line - 1].strip() or lines[line - 1].lstrip().startswith("#")
        ):
            line -= 1
        return line

    outer_tokens = [
        token
        for token in entries
        if token.start_mark.column == problems.start_mark.column
        and problems.start_mark.index <= token.start_mark.index < problems.end_mark.index
    ]
    if len(outer_tokens) != len(doc["BenchmarkProblems"]):
        raise SplitError("unsupported BenchmarkProblems layout")
    outer_starts = [
        comments_before(token.start_mark.line, key.start_mark.line + 1) for token in outer_tokens
    ]
    outer_ends = outer_starts[1:] + [len(lines)]
    pieces = []
    for outer, (token, start, end) in enumerate(zip(outer_tokens, outer_starts, outer_ends)):
        child_tokens = [
            item
            for item in entries
            if item.start_mark.column == token.start_mark.column + 2
            and token.start_mark.line <= item.start_mark.line < end
        ]
        if len(child_tokens) != len(doc["BenchmarkProblems"][outer]):
            raise SplitError("unsupported benchmark-group layout")
        starts = [
            (
                comments_before(item.start_mark.line, child_tokens[index - 1].start_mark.line + 1)
                if index
                else item.start_mark.line
            )
            for index, item in enumerate(child_tokens)
        ]
        prefix = "".join(lines[start : starts[1]])
        children = ["".join(lines[a:b]) for a, b in zip(starts[1:], starts[2:] + [end])]
        pieces.append((prefix, children))
    prelude = "".join(lines[: outer_starts[0]])
    if not prelude.startswith(HEADER.rstrip()):
        prelude = HEADER + prelude
    rendered = []
    cursor = 0
    for size in sizes:
        chosen = set(range(cursor, cursor + size))
        cursor += size
        result = prelude
        offset = 0
        for prefix, children in pieces:
            selected = [chunk for index, chunk in enumerate(children) if offset + index in chosen]
            if selected:
                result += prefix + "".join(selected)
            offset += len(children)
        result = localize_anchors(result, text, expand_all_anchors)
        result = "\n".join(line.rstrip() for line in result.rstrip().splitlines()) + "\n"
        rendered.append(result)
    parsed = [load_config(part) for part in rendered]
    expected = Counter(canonical(unit) for unit in units)
    actual = Counter(canonical(unit) for part in parsed for unit in groups(part))
    if actual != expected or any(settings_digest(part) != settings_digest(doc) for part in parsed):
        raise SplitError("rendered groups or non-group settings changed")
    return rendered


def path_policy(path):
    lower = path.lower()
    arch = re.search(r"gfx\d+(?:-strict)?", Path(path).name)
    return (
        tuple(word in lower for word in ("xfail", "wip", "disabled")),
        arch.group() if arch else None,
    )


def planned_outputs(repo, records, source_ref):
    directories = sorted({str(Path(record["path"]).parent) for record in records})
    originals = set(source_paths(repo, directories, source_ref))
    reserved = {record["path"] for record in records}
    outputs = {}
    for record in records:
        source = record["path"]
        sizes = record["sizes"]
        parent = Path(source).parent
        if "names" in record and "slots" in record:
            raise SplitError(f"choose names or slots for {source}")
        if "names" in record:
            if len(record["names"]) != len(sizes):
                raise SplitError(f"wrong number of names for {source}")
            names = record["names"]
            if any(Path(name).name != name or not name.endswith(".yaml") for name in names):
                raise SplitError(f"output names must be YAML basenames: {source}")
            paths = [(parent / name).as_posix() for name in names]
        else:
            paths = [source]
            suffix = 2
            while len(paths) < len(sizes):
                candidate = (parent / f"{Path(source).stem}_part{suffix}.yaml").as_posix()
                if candidate not in originals and candidate not in reserved:
                    paths.append(candidate)
                suffix += 1
            slots = record.get("slots", list(range(1, len(paths) + 1)))
            if sorted(slots) != list(range(1, len(paths) + 1)):
                raise SplitError(f"slots must permute all output files: {source}")
            paths = [paths[slot - 1] for slot in slots]
        if len(set(paths)) != len(paths):
            raise SplitError(f"duplicate output name for {source}")
        if source not in paths:
            raise SplitError(f"outputs must retain the source filename: {source}")
        for path in paths:
            if path in reserved and path != source:
                raise SplitError(f"output collides with another source or output: {path}")
            if path in originals and path != source:
                raise SplitError(f"output collides with an original YAML: {path}")
            if path_policy(path) != path_policy(source):
                raise SplitError(f"output changes path-derived test policy: {path}")
            reserved.add(path)
        outputs[source] = paths
    all_yaml = set(source_paths(repo, ["."], source_ref))
    all_yaml.update(path for paths in outputs.values() for path in paths)
    artifacts = {}
    for path in all_yaml:
        name = Path(path).with_suffix("").as_posix().replace("/", "__")
        if name in artifacts and artifacts[name] != path:
            raise SplitError(f"artifact names collide: {artifacts[name]} and {path}")
        artifacts[name] = path
    return outputs


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("version") != 1 or not isinstance(plan.get("splits"), list):
        raise SplitError("unsupported split plan")
    directories = plan.get("directories")
    if (
        not isinstance(directories, list)
        or not directories
        or any(not isinstance(directory, str) or not directory for directory in directories)
    ):
        raise SplitError("split plan must record its scanned directories")
    records = plan["splits"]
    if len({record["path"] for record in records}) != len(records):
        raise SplitError("duplicate source in split plan")
    for record in records:
        if not isinstance(record.get("sizes"), list) or len(record["sizes"]) < 2:
            raise SplitError(f"invalid split sizes for {record['path']}")
        if any(type(size) is not int or size < 1 for size in record["sizes"]):
            raise SplitError(f"invalid split sizes for {record['path']}")
    return plan


def load_timings(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        data = {row["path"]: row["seconds"] for row in data}
    timings = {key: float(value) for key, value in data.items()}
    if any(not math.isfinite(value) or value <= 0 for value in timings.values()):
        raise SplitError("timings must contain positive finite seconds")
    return timings


def scan(args):
    repo = args.repo.resolve()
    if (
        args.target_seconds <= 0
        or args.startup_seconds < 0
        or args.max_overhead < 0
        or args.max_parts < 1
        or args.untimed_max_parts < 1
    ):
        raise SplitError("invalid planning budget")
    source_ref = (
        git(repo, "rev-parse", "--verify", f"{args.source_ref}^{{commit}}").decode().strip()
        if args.source_ref
        else None
    )
    paths = source_paths(repo, args.directories, source_ref)
    timings = load_timings(args.timings)
    plan = {
        "version": 1,
        "source_ref": source_ref,
        "directories": [relative_path(repo, directory) for directory in args.directories],
        "splits": [],
    }
    counts = Counter()
    for path in paths:
        try:
            text = source_bytes(repo, path, source_ref).decode()
            doc = load_config(text)
        except (SplitError, UnicodeError, yaml.YAMLError):
            counts["not_config"] += 1
            continue
        units = groups(doc)
        reasons = blockers(doc, path)
        if len(units) < 2:
            counts["single_group"] += 1
            continue
        if reasons:
            counts["blocked"] += 1
            print(json.dumps({"path": path, "groups": len(units), "blocked": reasons}))
            continue
        weights = [group_weight(*unit) for unit in units]
        sizes = propose_sizes(
            weights,
            timings.get(path),
            args.target_seconds,
            args.startup_seconds,
            args.max_overhead,
            args.max_parts,
            args.untimed_max_parts,
        )
        seconds = timings.get(path)
        end = 0
        part_weights = []
        for size in sizes:
            part_weights.append(sum(weights[end : end + size]))
            end += size
        counts["candidate"] += 1
        print(
            json.dumps(
                {
                    "path": path,
                    "groups": len(units),
                    "lines": len(text.splitlines()),
                    "seconds": seconds,
                    "sizes": sizes,
                    "cost_proxy": round(sum(weights), 2),
                    "estimated_slowest_part_seconds": (
                        seconds * max(part_weights) / sum(weights) if seconds is not None else None
                    ),
                    "estimated_added_startup_fraction": (
                        (len(sizes) - 1) * args.startup_seconds / seconds
                        if seconds is not None
                        else None
                    ),
                    "needs_timing": seconds is None,
                }
            )
        )
        if len(sizes) > 1:
            counts["proposed"] += 1
            plan["splits"].append(
                {"path": path, "sizes": sizes, "settings_sha256": settings_digest(doc)}
            )
    print(json.dumps({"summary": dict(counts), "proposed_sources": len(plan["splits"])}))
    if args.plan_out:
        Path(args.plan_out).write_text(json.dumps(plan, indent=2) + "\n")


def render(args):
    repo = args.repo.resolve()
    plan = load_plan(args.plan)
    source_ref = args.source_ref or plan.get("source_ref")
    if args.write and not source_ref:
        raise SplitError("--write requires an unsplit --source-ref or a source_ref in the plan")
    records = plan["splits"]
    directories = [relative_path(repo, directory) for directory in plan["directories"]]
    for record in records:
        source = relative_path(repo, record["path"])
        if source != record["path"] or not any(
            Path(source).is_relative_to(directory) for directory in directories
        ):
            raise SplitError(f"source must be a relative path in a recorded directory: {source}")
    outputs = planned_outputs(repo, records, source_ref)
    if source_ref:
        originals = set(source_paths(repo, directories, source_ref))
        expected = {output for paths in outputs.values() for output in paths}
        unexpected = set(source_paths(repo, directories)) - originals - expected
        if unexpected:
            raise SplitError(
                "unexpected YAML outputs; rebuild the plan in a clean unsplit checkout: "
                + ", ".join(sorted(unexpected))
            )
    pending = []
    counts = Counter()
    for record in records:
        path = record["path"]
        text = source_bytes(repo, path, source_ref).decode()
        doc = load_config(text)
        reasons = blockers(doc, path)
        if record.get("allow_streamk_schedules"):
            reasons = [
                reason
                for reason in reasons
                if reason != "schedule assertions need per-output review"
            ]
        if reasons:
            raise SplitError(f"{path}: {', '.join(reasons)}")
        if record.get("settings_sha256") != settings_digest(doc):
            raise SplitError(f"non-group settings changed; review and update plan: {path}")
        parts = partition_text(text, record["sizes"], record.get("expand_all_anchors", False))
        mode = source_mode(repo, path, source_ref)
        for output, part in zip(outputs[path], parts):
            target = repo / output
            data = part.encode()
            if target.exists():
                current = target.read_bytes()
                if load_yaml(current) != load_yaml(data):
                    counts["semantic_differences"] += 1
                if current != data:
                    counts["text_differences"] += 1
                if target.stat().st_mode & 0o111 != mode & 0o111:
                    counts["mode_differences"] += 1
            else:
                counts["missing"] += 1
            pending.append((output, target, data, mode))
        counts["sources"] += 1
    if args.write:
        changed = [
            (output, target, data, mode)
            for output, target, data, mode in pending
            if not target.exists()
            or target.read_bytes() != data
            or target.stat().st_mode & 0o111 != mode & 0o111
        ]
        for output, target, _, _ in changed:
            if git(repo, "status", "--porcelain", "--", output).strip():
                raise SplitError(f"refusing to overwrite a locally modified output: {output}")
        for _, target, data, mode in changed:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            os.chmod(target, mode)
        counts["written"] = len(changed)
    elif args.strict_bytes and (
        counts["text_differences"] or counts["mode_differences"] or counts["missing"]
    ):
        raise SplitError(f"outputs differ from generated bytes: {dict(counts)}")
    if args.verify:
        for output, target, data, mode in pending:
            if not target.is_file():
                raise SplitError(f"missing output: {output}")
            if load_yaml(target.read_bytes()) != load_yaml(data):
                raise SplitError(f"output differs semantically: {output}")
            if target.stat().st_mode & 0o111 != mode & 0o111:
                raise SplitError(f"output mode differs: {output}")
        counts["verified_outputs"] = len(pending)
    print(json.dumps(dict(counts), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan_parser = subparsers.add_parser(
        "scan", help="report candidates and optionally write a plan"
    )
    scan_parser.add_argument("directories", nargs="+", help="directories under the repository")
    scan_parser.add_argument("--source-ref", help="read unsplit inputs from this Git ref")
    scan_parser.add_argument("--timings", type=Path, help="JSON mapping paths to measured seconds")
    scan_parser.add_argument("--target-seconds", type=float, default=60)
    scan_parser.add_argument("--startup-seconds", type=float, default=3.5)
    scan_parser.add_argument("--max-overhead", type=float, default=0.05)
    scan_parser.add_argument("--max-parts", type=int, default=16)
    scan_parser.add_argument("--untimed-max-parts", type=int, default=1)
    scan_parser.add_argument("--plan-out", type=Path)
    scan_parser.set_defaults(func=scan)
    render_parser = subparsers.add_parser("render", help="compare or write planned outputs")
    render_parser.add_argument("plan", type=Path)
    render_parser.add_argument("--source-ref", help="override the plan's unsplit Git ref")
    render_parser.add_argument(
        "--verify", action="store_true", help="require every output to exist and match"
    )
    render_parser.add_argument(
        "--strict-bytes", action="store_true", help="also require exact generated bytes"
    )
    render_parser.add_argument(
        "--write", action="store_true", help="write changed outputs after validation"
    )
    render_parser.set_defaults(func=render)
    args = parser.parse_args()
    try:
        args.func(args)
    except (SplitError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
