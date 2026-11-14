# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regenerate C++ twins of rocKE Python modules and write them into a rocKE tree.

    python -m tr.regen --rocke <root> <module>...

Steps (the generated ABI; --hand-abi translates against the tree's headers):

0. The code maps (sigmap.json, twin_map.json): data/maps, or $TR_MAPS. A
   copy of the translator without them mines them from the --rocke tree
   (tools/mine_maps.py) into a temporary directory.
1. tr.abi: the generated header of every module, into a staging directory
   (--abi-dir, else a temporary one). Skipped with --hand-abi: the C ABI is
   the hand-written headers already in the tree.
2. tr.translate: each module's .cpp against those headers, clang-formatted.
3. If any function is still a hole (no rule and no hand fill), list them and
   write nothing. Otherwise write <m>.cpp into
   platform/cpp/instances/common/ and the generated headers into
   platform/cpp/include/rocke/ of --into (default: the --rocke tree itself),
   plus the prelude tr/prelude/rocke_py*.hpp when this copy has one (a copy
   inside a rocKE tree has none: there platform/cpp/include/rocke/rocke_py*.hpp
   is the only copy, written by hand).

--check writes into a temporary tree instead and compares every file it
would write with the one in --into: exit 1 and the list of files that
differ when the checked-in generated files are not current. Temporary
directories (maps, headers, the --check tree) are removed on exit.

Name every module whose generated header the others use in the same command
(addr needs valu, counted and aux: _attention_bwd_valu _attention_bwd_counted
attention_bwd_aux).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "modules",
        nargs="+",
        metavar="module",
        help="Python module under library/kernels/common (e.g. _attention_bwd_addr)",
    )
    ap.add_argument(
        "--rocke",
        help="rocKE root to read (default: $ROCKE_ROOT, else the tree tr/ sits in)",
    )
    ap.add_argument(
        "--into", metavar="ROOT", help="rocKE root to write into (default: --rocke)"
    )
    ap.add_argument(
        "--abi-dir",
        metavar="DIR",
        help="keep the generated headers here (default: a temporary directory)",
    )
    ap.add_argument(
        "--hand-abi",
        action="store_true",
        help="translate against the tree's hand-written headers; generate none",
    )
    ap.add_argument(
        "--package",
        help="naming package (tr/package.py; default: $TR_PACKAGE, else attention_bwd)",
    )
    ap.add_argument(
        "--clang-format",
        metavar="EXE",
        help="clang-format to run (default: $CLANG_FORMAT, the venv's, or PATH)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if the generated files in --into are not current",
    )
    a = ap.parse_args(argv)
    scratch = []
    maps_env = os.environ.get("TR_MAPS")
    try:
        return regen(a, scratch)
    finally:
        for d in scratch:
            shutil.rmtree(d, ignore_errors=True)
        if maps_env is None:
            os.environ.pop("TR_MAPS", None)
        else:
            os.environ["TR_MAPS"] = maps_env


def regen(a, scratch: list) -> int:
    """main() after parsing; every temporary directory made goes into scratch."""
    if a.package:
        os.environ["TR_PACKAGE"] = (
            a.package
        )  # read when tr.package is first imported, just below

    from . import abi as A
    from .env import maps_dir, rocke_root
    from .package import PKG
    from .translate import clang_format, set_clang_format, translate

    if a.package and PKG.name != a.package:
        sys.exit(f"error: package {PKG.name} was already selected in this process")
    rocke = rocke_root(a.rocke)
    into = a.into or rocke
    set_clang_format(a.clang_format)
    if not os.path.exists(os.path.join(maps_dir(), "sigmap.json")):
        mine_maps(rocke, scratch)

    abi_dir = None
    generated = []
    if not a.hand_abi:
        abi_dir = a.abi_dir
        if not abi_dir:
            abi_dir = tempfile.mkdtemp(prefix="tr_abi_")
            scratch.append(abi_dir)
        os.makedirs(abi_dir, exist_ok=True)
        for m in a.modules:
            abi, env, _mod = A.build(m, rocke)
            path = os.path.join(abi_dir, A.header_name(env, m))
            open(path, "w").write(clang_format(A.header_text(abi, env, m), rocke))
            generated.append(path)
            print(f"header  {path}")

    out = {}
    holes = []
    for m in a.modules:
        res = translate(m, rocke, abi_dir)
        n = len(res.functions)
        mech = sum(f["status"] not in ("hole", "filled") for f in res.functions)
        filled = sum(f["status"] == "filled" for f in res.functions)
        left = [f for f in res.functions if f["status"] == "hole"]
        print(f"{m}: {mech}/{n} mechanical, {filled} hand fills, {len(left)} holes")
        holes += [f"  {m}: {f['name']} ({f['reason']} at line {f['at']})" for f in left]
        out[m] = clang_format(res.code, rocke)
    if holes:
        print("holes left (fill them, see HOWTO.md); nothing written:")
        print("\n".join(holes))
        return 1

    dest = into
    if a.check:
        dest = tempfile.mkdtemp(prefix="tr_check_")
        scratch.append(dest)
    SRC = os.path.join("platform", "cpp", "instances", "common")
    INC = os.path.join("platform", "cpp", "include", "rocke")
    os.makedirs(os.path.join(dest, SRC), exist_ok=True)
    os.makedirs(os.path.join(dest, INC), exist_ok=True)
    written = []
    for m, code in out.items():
        open(os.path.join(dest, SRC, f"{m}.cpp"), "w").write(code)
        written.append(os.path.join(SRC, f"{m}.cpp"))
    headers = sorted(glob.glob(os.path.join(HERE, "prelude", "rocke_py*.hpp")))
    headers += generated
    for h in headers:
        shutil.copy(h, os.path.join(dest, INC))
        written.append(os.path.join(INC, os.path.basename(h)))
    if not a.check:
        for w in written:
            print(f"wrote   {os.path.join(into, w)}")
        return 0
    stale = [
        w
        for w in written
        if not same_file(os.path.join(dest, w), os.path.join(into, w))
    ]
    for w in stale:
        print(f"stale   {os.path.join(into, w)}")
    if stale:
        print(
            f"{len(stale)} of {len(written)} generated files are not current; run tr.regen without --check"
        )
        return 1
    print(f"current: all {len(written)} generated files")
    return 0


def same_file(a: str, b: str) -> bool:
    if not os.path.exists(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def mine_maps(rocke: str, scratch: list) -> None:
    """No code maps (a copy without data/maps): mine sigmap.json and
    twin_map.json from the tree being read (tools/mine_maps.py) into a
    temporary directory (added to scratch), and point $TR_MAPS at it."""
    out = tempfile.mkdtemp(prefix="tr_maps_")
    scratch.append(out)
    tool = os.path.join(os.path.dirname(HERE), "tools", "mine_maps.py")
    cmd = [
        sys.executable,
        tool,
        "--rocke",
        rocke,
        "--out",
        out,
        "--only",
        "sigmap",
        "--only",
        "twin_map",
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if r.returncode:
        sys.exit(
            f"error: mining the code maps failed (exit {r.returncode}): {' '.join(cmd)}"
        )
    os.environ["TR_MAPS"] = out
    print(f"maps    {out}")


if __name__ == "__main__":
    sys.exit(main())
