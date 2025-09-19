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

# ---- core functions copied so this script is standalone ----
def load_library(path):
    with open(path, "rt") as fp:
        lib = yaml.load(fp, Loader=yaml.SafeLoader)
    if not isinstance(lib, list) or len(lib) <= 5 or not isinstance(lib[5], list):
        raise ValueError(f"{path} does not look like a valid Tensile logic YAML (missing index 5 Solutions).")
    return lib

def solutions_df(lib):
    df = pd.DataFrame(lib[5])
    for c in TILE_COLS:
        if c not in df.columns:
            df[c] = None
    return df

def macro_key_row(row):
    return tuple(row[c] for c in TILE_COLS)

def pick_random_per_tile(aug_df, candidate_tiles, rng):
    selected_rows = []
    aug_df = aug_df.copy()
    aug_df["_tilekey"] = aug_df.apply(macro_key_row, axis=1)
    grouped = aug_df.groupby("_tilekey", dropna=False)

    for tile in candidate_tiles:
        if tile not in grouped.groups:
            continue
        grp = aug_df.loc[grouped.groups[tile]]
        chosen = grp.sample(n=1, random_state=rng.randrange(1 << 30)).iloc[0]
        selected_rows.append(chosen)

    if not selected_rows:
        return pd.DataFrame(columns=aug_df.columns)
    sel_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    return sel_df.drop(columns=["_tilekey"])

def build_selected_library(base_lib, aug_lib, seed=None):
    base_df = solutions_df(base_lib)
    aug_df = solutions_df(aug_lib)

    base_tiles = set(base_df.apply(macro_key_row, axis=1).tolist())
    aug_tiles = set(aug_df.apply(macro_key_row, axis=1).tolist())

    candidate_tiles = sorted(aug_tiles - base_tiles)
    rng = random.Random(seed)
    selected_df = pick_random_per_tile(aug_df, candidate_tiles, rng)

    selected_solutions = []
    for i, row in selected_df.iterrows():
        sol_dict = None
        if "SolutionIndex" in row and pd.notna(row["SolutionIndex"]):
            idx = int(row["SolutionIndex"])
            if 0 <= idx < len(aug_lib[5]):
                sol_dict = copy.deepcopy(aug_lib[5][idx])

        if sol_dict is None:
            for cand in aug_lib[5]:
                if all(cand.get(c, None) == row.get(c, None) for c in TILE_COLS):
                    sol_dict = copy.deepcopy(cand)
                    break

        if sol_dict is None:
            sol_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}

        sol_dict["SolutionIndex"] = i
        selected_solutions.append(sol_dict)

    out_lib = copy.deepcopy(aug_lib)
    out_lib[5] = selected_solutions

    if len(out_lib) > 7 and isinstance(out_lib[7], list):
        out_lib[7] = []

    # keep your mod: print the differing tiles
    print("Tiles present in augmented but not in base:")
    for tile in candidate_tiles:
        print(dict(zip(TILE_COLS, tile)))

    return out_lib, len(candidate_tiles)
# ---- end core functions ----

# --- filename pairing helpers ---
# Tokens we ignore when matching filenames (noisy config bits)
IGNORE_TOKENS = {
    "BH", "HAS", "SAB", "SABV", "SCD", "SAV",
    "Bias", "BiasS", "BiasSB", "BiasSH", "BiasSHB",
    "UserArgs", "UserArgs.yaml"
}

def name_key(p: Path) -> tuple:
    """
    Normalize a filename into a tuple key to match base<->augmented.
    Strategy:
      - strip extension
      - split by '_' and remove ignored tokens
      - keep order of meaningful tokens (e.g., Ailk/Alik, Bljk/Bjlk, datatype blocks)
    """
    stem = p.name.replace(".yaml", "")
    toks = [t for t in stem.split('_') if t and t not in IGNORE_TOKENS]
    # also drop a leading 'gfx...' token, but keep the arch (e.g., gfx950) as first element
    # (we actually KEEP gfx950 so arch must match)
    return tuple(toks)

def best_base_for_aug(aug_path: Path, base_index: dict) -> Path | None:
    k = name_key(aug_path)
    # exact key match first
    if k in base_index:
        return base_index[k]
    # fallback: longest common prefix match
    best = None
    best_score = -1
    for bk, bpath in base_index.items():
        # score = number of equal tokens in order from start
        prefix = 0
        for a_tok, b_tok in zip(k, bk):
            if a_tok == b_tok:
                prefix += 1
            else:
                break
        if prefix > best_score:
            best_score = prefix
            best = bpath
    return best

def main():
    ap = argparse.ArgumentParser(
        description="Batch-select tiles for all augmented libraries by comparing to matching base libraries."
    )
    ap.add_argument("--base-dir", type=Path, required=True, help="Directory containing base YAML libraries")
    ap.add_argument("--aug-dir", type=Path, required=True, help="Directory containing augmented YAML libraries")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write selected YAML libraries")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible selection")
    ap.add_argument("--suffix", type=str, default="_selected.yaml",
                    help="Output file suffix (default: _selected.yaml). If ends with .yaml it's used as full suffix; "
                         "otherwise .yaml is appended.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # index base files by normalized key
    base_files = [p for p in args.base_dir.glob("*.yaml") if p.is_file()]
    if not base_files:
        print(f"ERROR: no base YAML files found in {args.base_dir}", file=sys.stderr)
        sys.exit(2)

    base_index: dict[tuple, Path] = {name_key(p): p for p in base_files}

    aug_files = [p for p in args.aug_dir.glob("*.yaml") if p.is_file()]
    if not aug_files:
        print(f"ERROR: no augmented YAML files found in {args.aug_dir}", file=sys.stderr)
        sys.exit(2)

    # normalize suffix
    suf = args.suffix if args.suffix.endswith(".yaml") else (args.suffix + ".yaml")

    total = 0
    wrote = 0
    skipped = 0

    for aug in sorted(aug_files):
        total += 1
        base = best_base_for_aug(aug, base_index)
        if base is None:
            print(f"[SKIP] No matching base for augmented: {aug.name}")
            skipped += 1
            continue

        print(f"\n=== Processing ===")
        print(f"Base:       {base.name}")
        print(f"Augmented:  {aug.name}")

        try:
            base_lib = load_library(base)
            aug_lib = load_library(aug)
            selected_lib, n_tiles = build_selected_library(base_lib, aug_lib, seed=args.seed)

            if n_tiles == 0:
                print(f"[SKIP] No differing tiles for {aug.name}")
                skipped += 1
                continue

            out_name = aug.stem + suf if suf.startswith("_") else aug.stem + "_" + suf
            out_path = out_dir / out_name
            with open(out_path, "w") as f_out:
                yaml.dump(selected_lib, f_out, **YAML_ARGS)

            print(f"[OK] Wrote {n_tiles} macro tiles -> {out_path}")
            wrote += 1
        except Exception as e:
            print(f"[ERROR] {aug.name}: {e}", file=sys.stderr)
            continue

    print(f"\nDone. Total augmented: {total}, wrote: {wrote}, skipped: {skipped}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
