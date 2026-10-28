# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Translate rocKE Python modules to a first draft of their C++ twins and write
them into a rocKE tree.

    python -m tr.regen --rocke <root> <module>...

What it writes is a first draft: once a file is in the tree it is ordinary
hand-maintained code, and tr.regen never writes over it unless told to.

Steps:

0. The code maps (sigmap.json, twin_map.json): data/maps, or $TR_MAPS. A
   copy of the translator without them mines them from the --rocke tree
   (tools/mine_maps.py) into a temporary directory.
1. The ABI. When every module named already has its header in the --rocke
   tree (the package's header name, e.g. helper_rocke.instances.common.<m>.h,
   or the other rocKE convention), the sources are translated
   against those headers (--hand-abi) and no header is generated;
   --generated-abi overrides that. Otherwise tr.abi writes the generated
   header of every module into a staging directory (--abi-dir, else a
   temporary one).
2. tr.translate: each module's .cpp against those headers, clang-formatted.
3. If any function is still a hole (no rule and no hand fill), list them and
   write nothing. Otherwise write <m>.cpp into the package's C++ directory
   (platform/cpp/instances/common/ for attention backward, tr/package.py)
   and the generated headers into
   platform/cpp/include/rocke/ of --into (default: the --rocke tree itself).
4. Support headers: every quoted same-directory include of a written source
   (#include "rocke_py.hpp", "rocke_py_adapters.hpp", and the support
   headers those include in turn) is copied from its master copy (tr/prelude/, else tr/packages/<package>/) into the same
   C++ directory. It is not installed.

Every target that already exists in --into (source, header or support
header) is left as it is: "skip <path> (exists)", exit 0. When every
module's source, its header and the support headers those sources include
already exist in --into, nothing is translated (so the ABI choice of step 1
does not matter): the skips are listed and the exit code is 0. --overwrite
writes over existing files; --pretend lists what would be written or
skipped and writes nothing. Temporary directories (maps, headers) are
removed on exit.

Name every module whose generated header the others use in the same command
(addr needs valu, counted and aux: _attention_bwd_valu _attention_bwd_counted
attention_bwd_aux).
"""

from __future__ import annotations

import argparse
import os
import re
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
        help="Python module of the package (e.g. _attention_bwd_addr under library/kernels/common)",
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
    which = ap.add_mutually_exclusive_group()
    which.add_argument(
        "--hand-abi",
        action="store_true",
        help="translate against the tree's existing headers; generate none"
        " (the default when every module named has one)",
    )
    which.add_argument(
        "--generated-abi",
        action="store_true",
        help="generate the headers even when the tree has them",
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
        "--overwrite",
        action="store_true",
        help="write over files that already exist in --into",
    )
    ap.add_argument(
        "--pretend",
        action="store_true",
        help="list what would be written or skipped; write nothing",
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
    from .package import PKG, support_file
    from .translate import clang_format, set_clang_format, translate

    if a.package and PKG.name != a.package:
        sys.exit(f"error: package {PKG.name} was already selected in this process")
    rocke = rocke_root(a.rocke)
    into = a.into or rocke
    set_clang_format(a.clang_format)
    if not a.overwrite:
        present = existing_targets(into, a.modules, support_file)
        if present:
            for path in present:
                print(f"skip    {path} (exists)")
            print("every target exists; nothing translated (--overwrite to replace)")
            return 0
    if not os.path.exists(os.path.join(maps_dir(), "sigmap.json")):
        mine_maps(rocke, scratch)

    hand_abi = a.hand_abi or (not a.generated_abi and existing_abi(rocke, a.modules))
    abi_dir = None
    generated = []
    if not hand_abi:
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
        if hand_abi and not a.hand_abi:
            print("headers were picked automatically; try --generated-abi")
        return 1

    # support headers the sources include from their own directory
    support = {}
    for m, code in out.items():
        for name in re.findall(r'^#include "([^"/]+)"', code, re.M):
            path = support_file(name)
            if path is None:
                sys.exit(
                    f"error: {m}.cpp includes {name!r}, which is in neither tr/prelude/ nor {PKG.data_dir}"
                )
            support[name] = path
    # and the support headers those include (a package's adapters include the generic ones)
    todo = list(support.values())
    while todo:
        text = open(todo.pop()).read()
        for name in re.findall(r'^#include "([^"/]+)"', text, re.M):
            if name not in support and (path := support_file(name)) is not None:
                support[name] = path
                todo.append(path)

    SRC = os.path.join(*PKG.cpp_dir.split("/"))
    INC = os.path.join("platform", "cpp", "include", "rocke")
    targets = [(os.path.join(SRC, f"{m}.cpp"), code, None) for m, code in out.items()]
    targets += [
        (os.path.join(SRC, name), None, path) for name, path in sorted(support.items())
    ]
    targets += [(os.path.join(INC, os.path.basename(h)), None, h) for h in generated]
    for rel, code, src in targets:
        path = os.path.join(into, rel)
        if os.path.exists(path) and not a.overwrite:
            print(f"skip    {path} (exists)")
            continue
        if a.pretend:
            print(f"write   {path} (pretend)")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if src is None:
            open(path, "w").write(code)
        else:
            shutil.copy(src, path)
        print(f"wrote   {path}")
    return 0


def tree_header(root: str, m: str) -> str | None:
    """Module m's header in the tree at root (any of its names), or None."""
    from .package import PKG

    inc = os.path.join(root, "platform", "cpp", "include", "rocke")
    for cand in PKG.header_names(m):
        if os.path.exists(os.path.join(inc, cand)):
            return os.path.join(inc, cand)
    return None


def existing_targets(into: str, modules: list, support_file) -> list | None:
    """Every target in --into, in write order, when all of them already
    exist: each module's .cpp and header, and every support header those
    sources include. None if any of them is missing (then the normal path
    runs and writes only what is missing)."""
    from .package import PKG

    src = os.path.join(into, *PKG.cpp_dir.split("/"))
    sources, headers, support = [], [], set()
    for m in modules:
        cpp = os.path.join(src, f"{m}.cpp")
        h = tree_header(into, m)
        if not os.path.exists(cpp) or h is None:
            return None
        sources.append(cpp)
        headers.append(h)
        todo = [open(cpp).read()]
        while todo:
            for name in re.findall(r'^#include "([^"/]+)"', todo.pop(), re.M):
                if name not in support and (path := support_file(name)) is not None:
                    support.add(name)
                    todo.append(open(path).read())
    support_paths = [os.path.join(src, n) for n in sorted(support)]
    if not all(os.path.exists(p) for p in support_paths):
        return None
    return sources + support_paths + headers


def existing_abi(rocke: str, modules: list) -> bool:
    """True when every module has its header in the tree (hand-written or an
    earlier first draft: the two share one name), so the sources are
    translated against them. Some but not all is an error: name the ABI."""
    have = {m: h for m in modules if (h := tree_header(rocke, m)) is not None}
    if not have:
        return False
    if len(have) < len(modules):
        missing = " ".join(m for m in modules if m not in have)
        sys.exit(
            f"error: some modules have a header in the tree and {missing} not;"
            " pass --hand-abi or --generated-abi"
        )
    for m in modules:
        print(
            f"{m}: header {have[m]} exists; using its ABI (--generated-abi to override)"
        )
    return True


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
