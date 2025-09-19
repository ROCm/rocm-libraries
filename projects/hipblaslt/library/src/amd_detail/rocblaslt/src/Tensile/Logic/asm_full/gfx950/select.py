#!/usr/bin/env python3
import argparse
import copy
import random
import sys
from pathlib import Path

import pandas as pd
import yaml


TILE_COLS = [
    "MacroTile0",
    "MacroTile1",
    "DepthU",
    "MatrixInstM",
    "MatrixInstN",
    "MatrixInstK",
    "MatrixInstB",
]

YAML_ARGS = {"default_flow_style": None, "sort_keys": False}


def load_library(path):
    with open(path, "rt") as fp:
        lib = yaml.load(fp, Loader=yaml.SafeLoader)
    if not isinstance(lib, list) or len(lib) <= 5 or not isinstance(lib[5], list):
        raise ValueError(f"{path} does not look like a valid Tensile logic YAML (missing index 5 Solutions).")
    return lib


def solutions_df(lib):
    """Return a DataFrame over lib[5] (Solutions). Missing columns are filled with None."""
    df = pd.DataFrame(lib[5])
    for c in TILE_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def macro_key_row(row):
    return tuple(row[c] for c in TILE_COLS)


def pick_random_per_tile(aug_df, candidate_tiles, rng):
    """For each tile key in candidate_tiles, pick one random row index from aug_df."""
    selected_rows = []
    # Group once by tile to avoid repeated filtering
    aug_df = aug_df.copy()
    aug_df["_tilekey"] = aug_df.apply(macro_key_row, axis=1)
    grouped = aug_df.groupby("_tilekey", dropna=False)

    for tile in candidate_tiles:
        if tile not in grouped.groups:
            # Shouldn't happen, but be robust
            continue
        grp = aug_df.loc[grouped.groups[tile]]
        # Pick a random row within this tile group
        chosen = grp.sample(n=1, random_state=rng.randrange(1 << 30)).iloc[0]
        selected_rows.append(chosen)

    if not selected_rows:
        return pd.DataFrame(columns=aug_df.columns)
    sel_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    return sel_df.drop(columns=["_tilekey"])


def build_selected_library(base_lib, aug_lib, seed=None):
    base_df = solutions_df(base_lib)
    aug_df = solutions_df(aug_lib)

    # Build sets of macro tiles present in base and augmented
    base_tiles = set(base_df.apply(macro_key_row, axis=1).tolist())
    aug_tiles = set(aug_df.apply(macro_key_row, axis=1).tolist())

    # Tiles that are in augmented but not in base
    candidate_tiles = sorted(aug_tiles - base_tiles)

    rng = random.Random(seed)

    # Pick one random solution per candidate tile
    selected_df = pick_random_per_tile(aug_df, candidate_tiles, rng)

    # Map selected rows back to full solution dicts (from aug_lib[5]) and reindex SolutionIndex
    selected_solutions = []
    for i, row in selected_df.iterrows():
        # Find the corresponding solution dict in aug_lib[5].
        # Use SolutionIndex if present and unique; otherwise match by all TILE_COLS.
        sol_dict = None

        if "SolutionIndex" in row and pd.notna(row["SolutionIndex"]):
            idx = int(row["SolutionIndex"])
            if 0 <= idx < len(aug_lib[5]):
                sol_dict = copy.deepcopy(aug_lib[5][idx])

        if sol_dict is None:
            # Fallback: first match by all tile columns + KernelNameMin if available
            for cand in aug_lib[5]:
                if all(cand.get(c, None) == row.get(c, None) for c in TILE_COLS):
                    sol_dict = copy.deepcopy(cand)
                    break

        if sol_dict is None:
            # As a last resort, construct from the row dict
            sol_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}

        sol_dict["SolutionIndex"] = i  # re-number
        selected_solutions.append(sol_dict)

    # Compose the output library:
    # Start from augmented (has compatible headers) and replace Solutions list at [5]
    out_lib = copy.deepcopy(aug_lib)
    out_lib[5] = selected_solutions

    # If the YAML has grids/mappings (often at [7]) that depend on indices, they are now stale.
    # It's safer to drop them if present.
    if len(out_lib) > 7 and isinstance(out_lib[7], list):
        out_lib[7] = []
    print("Tiles present in augmented but not in base:")
    for tile in candidate_tiles:
        print(dict(zip(TILE_COLS, tile)))
    return out_lib, len(candidate_tiles)


def main():
    p = argparse.ArgumentParser(
        description="Select macro tiles absent in base but present in augmented, choosing 1 random solution per tile."
    )
    p.add_argument("--base_library", type=Path, help="Path to base library YAML")
    p.add_argument("--augmented_library", type=Path, help="Path to augmented library YAML")
    p.add_argument("-o", "--output", type=Path, required=True, help='Path to write the "selected" library YAML')
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible selection")
    args = p.parse_args()

    base_lib = load_library(args.base_library)
    aug_lib = load_library(args.augmented_library)

    selected_lib, n_tiles = build_selected_library(base_lib, aug_lib, seed=args.seed)

    with open(args.output, "w") as f_out:
        yaml.dump(selected_lib, f_out, **YAML_ARGS)

    print(f"Wrote {n_tiles} macro tiles to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
