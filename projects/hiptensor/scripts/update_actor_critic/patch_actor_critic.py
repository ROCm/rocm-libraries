#!/usr/bin/env python3
"""
patch_actor_critic.py - Merge probe results from one or more ASICs and patch
the actor-critic unique_id strings in contraction_selection.cpp.

Different GPU ASICs support different data-type combinations, so you will
typically have separate JSON files from at least two probe runs (e.g. one for
gfx90a which supports F64, one for gfx942 which has additional types).  This
script merges them and writes a single patched source file.

Usage
=====
    # Basic: single ASIC
    python3 scripts/update_actor_critic/patch_actor_critic.py results_gfx90a.json

    # Multiple ASICs: list them in priority order (first file wins on conflict)
    python3 scripts/update_actor_critic/patch_actor_critic.py \
        results_gfx942.json results_gfx90a.json

    # Preview only — do not write anything
    python3 scripts/update_actor_critic/patch_actor_critic.py results_*.json --dry-run
Options
=======
  JSON_FILES        One or more probe-result JSON files produced by
                    probe_actor_critic.py.  When multiple files provide a
                    winner for the same (combo, layout, rank) slot, the value
                    from the first file that has a non-null winner is used.
                    Specify higher-priority ASICs first.
  --src FILE        contraction_selection.cpp to patch
                    (default: <repo_root>/library/src/contraction/
                              contraction_selection.cpp)
  --dry-run         Show what would change without writing anything.
  --show-conflicts  Print every slot where two files disagree (implies a
                    warning even without this flag; this makes them verbose).

Merge semantics
===============
For each (combo_key, layout, rank) slot:

  1. Collect all non-null winners across all input files in argument order.
  2. If exactly one unique winner exists → use it.
  3. If multiple files have the SAME winner → use it (harmless agreement).
  4. If multiple files have DIFFERENT winners → warn (the slot is genuinely
     ambiguous across ASICs).  The winner from the first file that supplied a
     non-null value is used; pass --show-conflicts for full detail.
  5. If no file has a winner → leave the existing string unchanged.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _actor_critic_combos import TYPE_COMBINATIONS  # noqa: E402


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, dict]:
    """Load a probe JSON and normalise rank keys to int."""
    raw = json.loads(path.read_text())
    result: dict[str, dict] = {}
    for combo_key, layouts in raw.items():
        result[combo_key] = {}
        for lname, rank_map in layouts.items():
            result[combo_key][lname] = {
                int(k): v for k, v in rank_map.items()
            }
    return result


def _winner(entry: dict | None) -> str | None:
    """Extract the winner string from a probe entry (or None)."""
    if entry is None:
        return None
    return entry.get("winner")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_results(
    datasets: list[tuple[str, dict[str, dict]]],
    show_conflicts: bool,
) -> dict[str, dict[str, dict[int, str | None]]]:
    """
    Merge probe datasets from multiple files.

    datasets: [(label, data), ...] in priority order (first wins on conflict).

    Returns:
      merged[combo_key][layout_name][rank] = winner_kernel_name | None
    """
    merged: dict[str, dict[str, dict[int, str | None]]] = {}
    conflict_count = 0

    # Collect all combo keys across all datasets
    all_keys = {k for _, d in datasets for k in d}

    for key in sorted(all_keys):
        merged[key] = {}
        combo_layouts = {"col_major", "row_major"}
        for lname in combo_layouts:
            merged[key][lname] = {}
            # Collect all ranks across all datasets for this slot
            all_ranks = {
                rank
                for _, d in datasets
                if key in d and lname in d[key]
                for rank in d[key][lname]
            }
            for rank in sorted(all_ranks):
                # Gather (label, winner) for every file that has a non-null entry
                candidates: list[tuple[str, str]] = []
                for label, data in datasets:
                    w = _winner(data.get(key, {}).get(lname, {}).get(rank))
                    if w is not None:
                        candidates.append((label, w))

                if not candidates:
                    merged[key][lname][rank] = None
                    continue

                # Check for conflicts (same slot, different kernel names)
                unique_winners = {w for _, w in candidates}
                if len(unique_winners) > 1:
                    conflict_count += 1
                    msg = (
                        f"  CONFLICT [{key}][{lname}][rank={rank}]:\n"
                        + "\n".join(
                            f"    {label}: {w}" for label, w in candidates
                        )
                        + f"\n    → using value from '{candidates[0][0]}'"
                    )
                    print(f"WARNING: {msg}", file=sys.stderr)
                    if show_conflicts:
                        print(msg)

                # First-file-wins
                merged[key][lname][rank] = candidates[0][1]

    if conflict_count:
        print(
            f"\nWARNING: {conflict_count} conflict(s) found. "
            "The value from the first JSON file was used in each case. "
            "Use --show-conflicts for details.",
            file=sys.stderr,
        )

    return merged


# ---------------------------------------------------------------------------
# Source patching
# ---------------------------------------------------------------------------

_UNIQUE_ID_ASSIGN_RE = re.compile(
    r'(unique_id\s*=\s*)'           # group 1: "unique_id ="
    r'("(?:[^"\\]|\\.)*"'           # first string literal
    r'(?:\s*"(?:[^"\\]|\\.)*")*)'   # optional continuation string literals
    r'(\s*;)',                      # group 3: semicolon
)

_COL_MAJOR_IF_RE = re.compile(
    r'if\s*\(\s*options\s*->\s*isColMajorStrides\s*\(\s*\)\s*\)'
)

_RANK_COMMENT_RE = re.compile(r'//\s*m(\d+)n\d+k\d+')


def _find_block(src: str, pattern: str) -> tuple[int, int] | None:
    """Return (start, end) offsets of the struct body for the given pattern."""
    m = re.search(pattern, src)
    if m is None:
        return None
    i, depth, block_start = m.end(), 0, None
    while i < len(src):
        if src[i] == '{':
            depth += 1
            if depth == 1:
                block_start = i
        elif src[i] == '}':
            depth -= 1
            if depth == 0 and block_start is not None:
                return (m.start(), i + 1)
        i += 1
    return None


def _split_col_row(block_src: str) -> tuple[str, str, int] | None:
    """
    Split a selectWinner block into the col-major and row-major sub-strings.
    Returns (col_half, row_half, col_if_start) or None on parse failure.
    col_if_start is the offset of the 'if(isColMajorStrides())' within block_src.
    """
    m = _COL_MAJOR_IF_RE.search(block_src)
    if m is None:
        return None
    depth, split_pos = 0, None
    for idx in range(m.end(), len(block_src)):
        if block_src[idx] == '{':
            depth += 1
        elif block_src[idx] == '}':
            depth -= 1
            if depth == 0:
                split_pos = idx + 1
                break
    if split_pos is None:
        return None
    return block_src[m.start():split_pos], block_src[split_pos:], m.start()


def _patch_half(half_src: str, rank_map: dict[int, str | None],
                key: str, lname: str,
                dry_run: bool) -> tuple[str, int]:
    """Replace unique_id strings inside one layout half."""
    # Collect all replacements using positions on the original (unmodified) string,
    # then apply them in reverse order so earlier edits don't shift later positions.
    replacements: list[tuple[int, int, str, str, int]] = []  # (start, end, old, new, rank)

    rank_comments = list(_RANK_COMMENT_RE.finditer(half_src))
    for i, rm in enumerate(rank_comments):
        rank = int(rm.group(1))
        winner = rank_map.get(rank)
        if winner is None:
            continue

        uid_m = _UNIQUE_ID_ASSIGN_RE.search(half_src, rm.end())
        if uid_m is None:
            continue
        # Don't jump past the next rank comment
        if i + 1 < len(rank_comments) and uid_m.start() > rank_comments[i + 1].start():
            continue

        old = uid_m.group(0)
        new = uid_m.group(1) + f'"{winner}"' + uid_m.group(3)
        if old == new:
            continue

        replacements.append((uid_m.start(), uid_m.end(), old, new, rank))

    if not replacements:
        return half_src, 0

    for _, _, old, new, rank in replacements:
        print(f"  [{key}][{lname}][rank={rank}]")
        print(f"    - {old.strip()}")
        print(f"    + {new.strip()}")

    if dry_run:
        return half_src, len(replacements)

    # Apply in reverse order to preserve earlier positions
    out = half_src
    for start, end, _, new, _ in sorted(replacements, key=lambda r: r[0], reverse=True):
        out = out[:start] + new + out[end:]

    return out, len(replacements)


def patch_source(
    src: str,
    merged: dict[str, dict[str, dict[int, str | None]]],
    dry_run: bool,
) -> tuple[str, int]:
    """
    Apply merged results to src.  Returns (patched_src, total_changes).
    Processes specializations in order; recalculates block offsets after each
    write so that earlier edits do not corrupt later ones.
    """
    new_src = src
    total = 0

    for combo in TYPE_COMBINATIONS:
        key = combo["key"]
        if key not in merged:
            print(f"  [SKIP] {key}: no data in any input file")
            continue

        block = _find_block(new_src, combo["cpp_match"])
        if block is None:
            print(f"  [WARN] {key}: C++ specialisation not found in source")
            continue

        bs, be = block
        block_src = new_src[bs:be]

        parts = _split_col_row(block_src)
        if parts is None:
            print(f"  [WARN] {key}: could not split col/row branches")
            continue
        col_half, row_half, col_if_start = parts

        col_map = merged[key].get("col_major", {})
        row_map = merged[key].get("row_major", {})

        new_col, c1 = _patch_half(col_half, col_map, key, "col_major", dry_run)
        new_row, c2 = _patch_half(row_half, row_map, key, "row_major", dry_run)
        total += c1 + c2

        if not dry_run and (c1 + c2) > 0:
            new_block = block_src[:col_if_start] + new_col + new_row
            # Recalculate after earlier edits may have shifted offsets
            block = _find_block(new_src, combo["cpp_match"])
            if block:
                bs, be = block
                new_src = new_src[:bs] + new_block + new_src[be:]

    return new_src, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_src = (
        repo_root / "library/src/contraction/contraction_selection.cpp"
    )

    parser = argparse.ArgumentParser(
        description="Merge probe JSONs and patch actor-critic unique_id strings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        metavar="JSON_FILE",
        help="Probe result JSON(s) from probe_actor_critic.py, in priority order",
    )
    parser.add_argument(
        "--src",
        default=str(default_src),
        metavar="FILE",
        help="contraction_selection.cpp to patch (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    parser.add_argument(
        "--show-conflicts",
        action="store_true",
        help="Print every slot where input files disagree (always warned, "
             "this flag also prints to stdout)",
    )
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        print(f"ERROR: source not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    # Load all JSON files
    datasets: list[tuple[str, dict[str, dict]]] = []
    for p in args.json_files:
        path = Path(p)
        if not path.exists():
            print(f"ERROR: JSON file not found: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {path}")
        datasets.append((path.name, _load_json(path)))

    # Merge
    print(f"\nMerging {len(datasets)} dataset(s)…")
    merged = merge_results(datasets, args.show_conflicts)

    # Coverage summary — denominator is the full expected slot count derived
    # from TYPE_COMBINATIONS × 2 layouts × 6 ranks, not just what appeared in
    # the JSON data (which would undercount if any input file is missing ranks).
    from _actor_critic_combos import ALL_RANKS, LAYOUTS  # noqa: E402
    expected_total = len(TYPE_COMBINATIONS) * len(LAYOUTS) * len(ALL_RANKS)
    filled_slots = 0
    for key in merged:
        for lname in merged[key]:
            for winner in merged[key][lname].values():
                if winner is not None:
                    filled_slots += 1

    print(f"Coverage after merge: {filled_slots}/{expected_total} slots filled")
    if filled_slots < expected_total:
        missing = expected_total - filled_slots
        print(
            f"  {missing} slot(s) have no winner — "
            "those unique_id strings will not be changed.",
        )

    # Patch
    print(f"\nPatching {src_path}" + (" [DRY RUN]" if args.dry_run else "") + "…")
    original = src_path.read_text()
    new_src, num_changes = patch_source(original, merged, args.dry_run)

    if num_changes == 0:
        print("No changes needed — all unique_id strings are already current.")
    elif args.dry_run:
        print(f"\n{num_changes} change(s) would be made (nothing written).")
    else:
        src_path.write_text(new_src)
        print(f"\n{num_changes} change(s) written to {src_path}.")
        print("Rebuild the library to pick up the updated kernel names.")


if __name__ == "__main__":
    main()
