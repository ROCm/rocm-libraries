#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Regenerate every conv engine family's descriptors, and record its catalogs in DVC.

Each family's generator (gen_<...>.py, table FAMILIES below) writes the family's complete
descriptor set: the engine descriptors (`*.ued.json`, `*.udd.json`, `*.kmd.json`,
`*.umd.json`), which git tracks, and the instance catalogs (`*.kdp.json`), which DVC
tracks, one output per catalog with a `<catalog>.dvc` pointer in git. See ../REGEN.md.

Every generator is deterministic (ids are uuid5 of the config content, files are written
with fixed formatting), so a regeneration reproduces the tree byte for byte, and with it
the md5 every pointer records.

Modes:
  (default)   regenerate into the tree. Each generator runs into a fresh scratch directory
              and its files are copied over the family's. A catalog the tree holds that the
              generator no longer writes is reported and fails the run; --prune deletes it
              with its pointer.
  --check     write nothing: regenerate into scratch and require every generated file to
              equal the tree's byte for byte, and every catalog's pointer to record that
              catalog's md5 and size and the `ingestor` remote. Needs the catalogs pulled.
  --dvc-add   after regenerating: take the catalogs out of the git index if a merge put them
              back, `dvc add` them (one pointer each), give each pointer `remote: ingestor`,
              and `git add` the pointers, the .gitignore entries and the git-tracked
              descriptors. Uploading (`dvc push`) stays a separate, credentialed step.

Python 3, standard library only; `dvc` and `git` on PATH for --dvc-add and --prune.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GENERATORS_DIR = Path(__file__).resolve().parent
DESCRIPTORS_DIR = (
    GENERATORS_DIR.parent.parent
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / "conv"
)
REMOTE = "ingestor"

# generator script -> family folder under descriptors/conv/. naive/ has no generator: its
# three small catalogs are written by hand and stay in git.
FAMILIES = {
    "gen_depthwise_mfma.py": "depthwise_mfma",
    "gen_depthwise_rows.py": "depthwise_rows",
    "gen_depthwise_valu.py": "depthwise_valu",
    "gen_depthwise_valu_wrw.py": "depthwise_valu_wrw",
    "gen_direct.py": "direct",
    "gen_fft.py": "fft",
    "gen_group.py": "group",
    "gen_igemm.py": "igemm",
    "gen_igemm_nchw.py": "igemm_nchw",
    "gen_smallc.py": "smallc",
    "gen_conv_stem_bwd.py": "stem_bwd",
    "gen_winograd.py": "winograd",
    "gen_conv_wrw.py": "wrw",
}

CATALOG_SUFFIX = ".kdp.json"


def is_catalog(path: Path) -> bool:
    return path.name.endswith(CATALOG_SUFFIX)


def pointer_of(catalog: Path) -> Path:
    return catalog.with_name(catalog.name + ".dvc")


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=GENERATORS_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def generate(script: str, out_dir: Path) -> list[Path]:
    """Run one generator into an empty `out_dir`; return the files it wrote, sorted."""
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [sys.executable, str(GENERATORS_DIR / script), str(out_dir)],
        cwd=GENERATORS_DIR,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    written = sorted(p for p in out_dir.rglob("*") if p.is_file())
    if not written:
        raise SystemExit(f"{script} wrote nothing into {out_dir}")
    return written


def read_pointer(pointer: Path) -> dict[str, str]:
    """The scalar fields of a single-output .dvc pointer (md5, size, path, remote, ...)."""
    fields: dict[str, str] = {}
    for line in pointer.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().removeprefix("- ")
        key, sep, value = stripped.partition(":")
        if sep and value.strip():
            fields[key.strip()] = value.strip()
    return fields


def ensure_remote(pointer: Path) -> None:
    """Give a pointer `remote: ingestor`, so a bare `dvc pull` finds its content."""
    fields = read_pointer(pointer)
    if fields.get("remote") == REMOTE:
        return
    if "remote" in fields:
        raise SystemExit(
            f"{pointer}: names remote '{fields['remote']}', expected '{REMOTE}'"
        )
    text = pointer.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    pointer.write_text(text + f"  remote: {REMOTE}\n", encoding="utf-8")


def check_family(script: str, family: str, scratch: Path) -> list[str]:
    problems = []
    tree = DESCRIPTORS_DIR / family
    written = generate(script, scratch)
    produced = {p.relative_to(scratch) for p in written}
    for rel in sorted(produced):
        mine = tree / rel
        if not mine.is_file():
            hint = (
                " (not pulled? dvc pull -r ingestor <pointer>)"
                if is_catalog(mine)
                else ""
            )
            problems.append(f"{family}/{rel}: missing from the tree{hint}")
        elif not filecmp.cmp(scratch / rel, mine, shallow=False):
            problems.append(f"{family}/{rel}: differs from what {script} generates")
        if is_catalog(mine):
            pointer = pointer_of(mine)
            if not pointer.is_file():
                problems.append(f"{family}/{rel}: no DVC pointer {pointer.name}")
                continue
            fields = read_pointer(pointer)
            data = (scratch / rel).read_bytes()
            if fields.get("md5") != hashlib.md5(data).hexdigest():
                problems.append(
                    f"{family}/{pointer.name}: md5 does not match the generated catalog"
                )
            if fields.get("size") != str(len(data)):
                problems.append(
                    f"{family}/{pointer.name}: size does not match the generated catalog"
                )
            if fields.get("remote") != REMOTE:
                problems.append(f"{family}/{pointer.name}: remote is not '{REMOTE}'")
    for catalog in sorted(tree.glob("*" + CATALOG_SUFFIX)):
        if catalog.relative_to(tree) not in produced:
            problems.append(
                f"{family}/{catalog.name}: stale, {script} no longer writes it"
            )
    for pointer in sorted(tree.glob("*" + CATALOG_SUFFIX + ".dvc")):
        if Path(pointer.name.removesuffix(".dvc")) not in produced:
            problems.append(
                f"{family}/{pointer.name}: stale pointer, {script} no longer writes its catalog"
            )
    return problems


def regen_family(
    script: str, family: str, scratch: Path, prune: bool, root: Path
) -> list[str]:
    tree = DESCRIPTORS_DIR / family
    written = generate(script, scratch)
    produced = {p.relative_to(scratch) for p in written}
    for src in written:
        dst = tree / src.relative_to(scratch)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or not filecmp.cmp(src, dst, shallow=False):
            shutil.copyfile(src, dst)
    stale = []
    names = {c.name for c in tree.glob("*" + CATALOG_SUFFIX)}
    names |= {
        p.name.removesuffix(".dvc") for p in tree.glob("*" + CATALOG_SUFFIX + ".dvc")
    }
    for name in sorted(names):
        if Path(name) in produced:
            continue
        catalog = tree / name
        if not prune:
            stale.append(
                f"{family}/{name}: stale, {script} no longer writes it (rerun with --prune)"
            )
            continue
        pointer = pointer_of(catalog)
        if pointer.is_file():
            run(["dvc", "remove", str(pointer.relative_to(root))], cwd=root)
            run(
                [
                    "git",
                    "rm",
                    "-q",
                    "--cached",
                    "--ignore-unmatch",
                    str(pointer.relative_to(root)),
                ],
                cwd=root,
            )
        run(
            [
                "git",
                "rm",
                "-q",
                "--cached",
                "--ignore-unmatch",
                str(catalog.relative_to(root)),
            ],
            cwd=root,
        )
        if catalog.is_file():
            catalog.unlink()
        print(f"pruned {family}/{name}")
    return stale


def dvc_add(families: list[str], root: Path) -> None:
    catalogs = []
    for family in families:
        catalogs += sorted((DESCRIPTORS_DIR / family).glob("*" + CATALOG_SUFFIX))
    rel = [str(c.relative_to(root)) for c in catalogs]
    # A merge that brought a catalog back into git leaves it in the index; DVC refuses an
    # output git already tracks.
    run(["git", "rm", "-q", "--cached", "--ignore-unmatch", "--", *rel], cwd=root)
    run(["dvc", "add", "-q", *rel], cwd=root)
    for catalog in catalogs:
        ensure_remote(pointer_of(catalog))
    staged = [f"{r}.dvc" for r in rel]
    for family in families:
        tree = DESCRIPTORS_DIR / family
        staged.append(str((tree / ".gitignore").relative_to(root)))
        staged += [
            str(p.relative_to(root))
            for p in sorted(tree.glob("*.json"))
            if not is_catalog(p)
        ]
    run(["git", "add", "--", *staged], cwd=root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--families", help="comma-separated subset of families (default: all)"
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="compare only; write nothing"
    )
    mode.add_argument(
        "--dvc-add", action="store_true", help="regenerate, then dvc add and git add"
    )
    ap.add_argument(
        "--prune", action="store_true", help="delete catalogs no generator writes"
    )
    args = ap.parse_args(argv)

    selected = dict(FAMILIES)
    if args.families:
        wanted = [f.strip() for f in args.families.split(",") if f.strip()]
        unknown = sorted(set(wanted) - set(FAMILIES.values()))
        if unknown:
            raise SystemExit(
                f"unknown families {unknown}; known: {sorted(FAMILIES.values())}"
            )
        selected = {s: f for s, f in FAMILIES.items() if f in wanted}

    root = repo_root() if (args.dvc_add or args.prune) else Path()
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="conv-regen-") as tmp:
        for script, family in sorted(selected.items(), key=lambda item: item[1]):
            scratch = Path(tmp) / family
            scratch.mkdir()
            if args.check:
                found = check_family(script, family, scratch)
            else:
                found = regen_family(script, family, scratch, args.prune, root)
            problems += found
            print(
                f"{family}: {'ok' if not found else f'{len(found)} problem(s)'}",
                flush=True,
            )

    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    if args.dvc_add:
        dvc_add(sorted(selected.values()), root)
        print(
            "\nRecorded. Upload before the PR goes up (needs write credentials):\n"
            + "\n".join(
                f"  dvc push -r {REMOTE} {(DESCRIPTORS_DIR / f).relative_to(root)}/*{CATALOG_SUFFIX}.dvc"
                for f in sorted(selected.values())
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
