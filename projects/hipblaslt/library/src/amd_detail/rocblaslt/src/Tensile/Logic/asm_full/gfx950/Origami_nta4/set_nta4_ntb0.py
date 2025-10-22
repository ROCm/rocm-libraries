#!/usr/bin/env python3
import os
import glob
import shutil
import yaml

def patch_solutions(logic):
    """
    For each solution in logic[5], if NonTemporalA==0 and NonTemporalB==0,
    set NonTemporalB to 1.
    Returns (changed, new_logic)
    """
    if not isinstance(logic, list) or len(logic) <= 5 or not isinstance(logic[5], list):
        return False, logic  # not a recognized Tensile logic format

    changed = False
    for sol in logic[5]:
        if not isinstance(sol, dict):
            continue

        sol["NonTemporalA"] = 4
        sol["NonTemporalB"] = 0
        changed = True

    return changed, logic

def process_file(path):
    yaml_args = {"default_flow_style": None, "sort_keys": False}
    with open(path, "rt") as fp:
        logic = yaml.load(fp, Loader=yaml.SafeLoader)

    changed, new_logic = patch_solutions(logic)
    if not changed:
        print(f"[SKIP] {os.path.basename(path)} (no matching kernels found or unrecognized layout)")
        return

    # Make a backup first
    backup = path + ".bak"
    shutil.copy2(path, backup)

    # Write changes atomically
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f_out:
        yaml.dump(new_logic, f_out, **yaml_args)
    os.replace(tmp_path, path)

    print(f"[OK]   {os.path.basename(path)} — patched kernels (backup: {backup})")

def main():
    paths = sorted(set(glob.glob("*.yaml") + glob.glob("*.yml")))
    if not paths:
        print("No .yaml or .yml files found in the current directory.")
        return

    for p in paths:
        try:
            process_file(p)
        except Exception as e:
            print(f"[ERR]  {p}: {e}")

if __name__ == "__main__":
    main()
 