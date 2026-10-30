#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Replace a ForkParameters cross product with a pairwise covering set of rows.

A benchmark group's ForkParameters multiplies every multi-valued parameter with
every other. Most parameters do not interact, so a much smaller set of
combinations exercises every pair of values that can appear together.

The tool asks TensileLite which combinations in the full product are accepted
(not rejected, not a duplicate solution), then greedily keeps accepted
combinations until every pair of parameter values found together in some
accepted combination appears in a kept one. Selection is deterministic: ties
go to the combination that comes first in the full product.

The chosen rows replace the multi-valued entries of ForkParameters with one
``Groups`` entry, leaving the rest of the file text untouched. After writing,
the tool re-reads the file through TensileLite and checks that it expands to
exactly the chosen rows.

    sparsify_fork_grid.py plan  FILE.yaml --group N [--keep PARAM ...]
    sparsify_fork_grid.py apply FILE.yaml --group N [--keep PARAM ...]

``--group`` counts every ``ForkParameters`` block in the file from zero.
``--keep`` leaves a parameter as an ordinary multi-valued entry; it still
multiplies the chosen rows. The ``plan`` and ``apply`` commands need an
amdclang++ on PATH and a built rocisa; no device is used.
"""

import argparse
import heapq
import itertools
import json
from pathlib import Path
import re
import sys

import yaml


def canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def cover_pairs(rows):
    """Return indices of rows (dicts of axis -> value) covering every pair seen in rows.

    Greedy lazy selection: a row's gain only shrinks as pairs get covered, so a
    stale heap entry is re-scored when it reaches the top.
    """
    axes = sorted({axis for row in rows for axis in row})
    keys = [canonical_pairs(row, axes) for row in rows]
    uncovered = set().union(*keys) if keys else set()
    heap = [(-len(k), i) for i, k in enumerate(keys)]
    heapq.heapify(heap)
    chosen = []
    while uncovered:
        _, i = heapq.heappop(heap)
        gain = len(keys[i] & uncovered)
        if heap and gain < -heap[0][0]:
            heapq.heappush(heap, (-gain, i))
            continue
        chosen.append(i)
        uncovered -= keys[i]
    return sorted(chosen)


def canonical_pairs(row, axes):
    items = [(axis, canonical(row[axis])) for axis in axes if axis in row]
    return {pair for pair in itertools.combinations(items, 2)}


def fork_blocks(lines):
    """Yield (index of the ForkParameters line, indent, list of (start, end) item spans)."""
    for n, line in enumerate(lines):
        match = re.match(r"^(\s*)ForkParameters:\s*(#.*)?$", line)
        if not match:
            continue
        key_indent = len(match.group(1))
        spans, item_indent, start, end = [], None, None, n + 1
        for m in range(n + 1, len(lines)):
            text = lines[m]
            if not text.strip() or text.lstrip().startswith("#"):
                continue
            indent = len(text) - len(text.lstrip())
            if indent <= key_indent and not text.lstrip().startswith("- "):
                break
            if indent < key_indent or (indent == key_indent and not text.lstrip().startswith("- ")):
                break
            if item_indent is None:
                item_indent = indent
            if indent == item_indent and text.lstrip().startswith("- "):
                if start is not None:
                    spans.append((start, end))
                start = m
            elif indent < item_indent:
                break
            end = m + 1
        if start is not None:
            spans.append((start, end))
        yield n, item_indent, spans


def format_value(value):
    # Lists get one extra level: TensileLite expands a list inside Groups into
    # alternatives, so a list-valued parameter must be written as a list of one.
    if isinstance(value, list):
        value = [value]
    return (
        yaml.safe_dump(value, default_flow_style=True, width=10**9)
        .strip()
        .removesuffix("...")
        .strip()
    )


def render(text, group, axes, rows, keep=()):
    """Return text with the multi-valued ``axes`` of block ``group`` replaced by one Groups entry."""
    lines = text.splitlines(keepends=True)
    blocks = list(fork_blocks(lines))
    _, indent, spans = blocks[group]
    first = None
    out = []
    removed = set()
    for start, end in spans:
        head = re.match(r"^\s*-\s*([^:\s]+):", lines[start])
        if head and head.group(1) in axes and head.group(1) not in keep:
            removed.update(range(start, end))
            first = start if first is None else first
    pad = " " * indent
    entry = [
        f"{pad}# Pairwise cover of the full parameter grid, written by scripts/sparsify_fork_grid.py.\n",
        f"{pad}# Every pair of values that is valid together appears in at least one row.\n",
        f"{pad}- Groups:\n",
        f"{pad}  -\n",
    ]
    entry += [
        f"{pad}    - {{"
        + ", ".join(f"{a}: {format_value(r[a])}" for a in sorted(axes - set(keep)))
        + "}\n"
        for r in rows
    ]
    for n, line in enumerate(lines):
        if n == first:
            out.extend(entry)
        if n not in removed:
            out.append(line)
    return "".join(out)


def multi_valued_axes(forks):
    return {k for k, v in forks.items() if isinstance(v, list) and len(v) > 1 and k != "Groups"}


def group_config(doc, group):
    """Return (problem type config, size-group config) for the group-th ForkParameters block."""
    seen = -1
    for entry in doc["BenchmarkProblems"]:
        for size_group in entry[1:]:
            if "ForkParameters" in size_group:
                seen += 1
                if seen == group:
                    return entry[0], size_group
    raise SystemExit(f"no ForkParameters block {group}")


def _expand(problem_type, size_group, arch):
    """Full product as (perm, accepted) pairs plus the constant parameters, via TensileLite."""
    from tensilelite.BenchmarkProblems import _generate_single_solution
    from tensilelite.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Common.GlobalParameters import assignGlobalParameters
    from tensilelite.Common.Parallel import ParallelMap2
    from tensilelite.Common.Types import makeDebugConfig
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain
    import rocisa

    cxx = validateToolchain("amdclang++")
    isa_info = makeIsaInfoMap([gfxToIsa(arch)], cxx)
    assembler = makeAssemblyToolchain(
        cxx, validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER), "default"
    ).assembler
    assignGlobalParameters({}, isa_info)
    process = BenchmarkProcess(problem_type, size_group, False)
    step = process[0]
    perms = list(constructForkPermutations(step.forkParams, step.paramGroups))
    debug = makeDebugConfig({})
    raw = ParallelMap2(
        _generate_single_solution,
        zip(
            perms,
            itertools.repeat(process.problemType),
            itertools.repeat(step.constantParams),
            itertools.repeat(assembler),
            itertools.repeat(debug),
            itertools.repeat(isa_info),
            itertools.repeat(rocisa.rocIsa.getInstance().getData()),
        ),
        "fork solutions",
        return_as="list",
    )
    first, accepted = {}, []
    for solution in raw:
        accepted.append(solution is not None and first.setdefault(solution, solution) is solution)
    return perms, accepted, step.constantParams


def plan(path, group, keep, arch):
    text = Path(path).read_text()
    problem_type, size_group = group_config(yaml.safe_load(text), group)
    forks = {k: v for item in size_group["ForkParameters"] for k, v in item.items()}
    if "Groups" in forks:
        raise SystemExit("block already uses Groups")
    axes = multi_valued_axes(forks) - set(keep)
    perms, accepted, constants = _expand(problem_type, size_group, arch)
    rows = [{a: p[a] for a in axes} for p, ok in zip(perms, accepted) if ok]
    # Rows that differ only in kept axes count once; the kept axes multiply the result.
    unique = list({canonical(r): r for r in rows}.values())
    chosen = [unique[i] for i in cover_pairs(unique)]
    return (
        text,
        axes,
        chosen,
        {
            "full": len(perms),
            "accepted": sum(accepted),
            "distinct_rows": len(unique),
            "chosen": len(chosen),
            "axes": sorted(axes),
        },
    )


def verify(path, group, axes, chosen, arch):
    text = Path(path).read_text()
    problem_type, size_group = group_config(yaml.safe_load(text), group)
    perms, accepted, _ = _expand(problem_type, size_group, arch)
    got = {canonical({a: p[a] for a in axes}) for p in perms}
    want = {canonical(r) for r in chosen}
    if got != want or not all(accepted):
        raise SystemExit(
            f"rewritten block expands to {len(got)} rows ({sum(accepted)} accepted); expected {len(want)}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("plan", "apply"))
    parser.add_argument("file", type=Path)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--keep", nargs="*", default=[])
    parser.add_argument("--arch", default="gfx1250")
    args = parser.parse_args(argv)
    text, axes, chosen, stats = plan(args.file, args.group, args.keep, args.arch)
    print(json.dumps(stats))
    if args.command == "apply":
        args.file.write_text(render(text, args.group, axes, chosen, args.keep))
        verify(args.file, args.group, axes, chosen, args.arch)
        print("verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
