# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Helpers of the translator self-test: run tr.<tool> against a fixture root."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

TR = Path(__file__).resolve().parents[1]
ROCKE = TR.parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIXTURES / "golden"
CLANG_FORMAT_VERSION = "18.1.4"  # the version the goldens were formatted with

# the generated headers (addr is needed by valu and counted) and the sources
ABI_MODULES = ("_attention_bwd_addr", "_attention_bwd_valu", "_attention_bwd_counted")
CPP_MODULES = ("_attention_bwd_valu", "_attention_bwd_counted")


class Tr:
    """Runs tr.<tool> against a fixture root."""

    def __init__(self, library: Path, data: Path = FIXTURES / "data"):
        self.library = library
        self.data = data

    def env(self) -> dict:
        env = dict(os.environ)
        env.pop("TR_MAPS", None)
        env.update(
            TR_DATA=str(self.data),
            TR_HOLES=str(FIXTURES / "holes"),
            TR_LIBRARY=str(self.library),
            PYTHONDONTWRITEBYTECODE="1",
        )
        return env

    def run(self, tool: str, *args: str) -> subprocess.CompletedProcess:
        p = subprocess.run(
            [sys.executable, "-m", f"tr.{tool}", *args, "--rocke", str(ROCKE)],
            cwd=TR,
            env=self.env(),
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, f"tr.{tool} {' '.join(args)}:\n{p.stdout}{p.stderr}"
        return p

    def generate(self, out: Path) -> dict[str, subprocess.CompletedProcess]:
        """Headers for ABI_MODULES, then the .cpp of CPP_MODULES, into out.
        Returns the translate runs by module."""
        abi = out / "abi"
        abi.mkdir(parents=True, exist_ok=True)
        for m in ABI_MODULES:
            self.run("abi", m, "--header", str(abi))
        return {
            m: self.run(
                "translate", m, "--abi", str(abi), "--out", str(out / f"{m}.cpp")
            )
            for m in CPP_MODULES
        }


def outputs(out: Path) -> dict[str, Path]:
    """golden file name -> generated file."""
    files = {f"{m}.cpp.golden": out / f"{m}.cpp" for m in CPP_MODULES}
    for m in ABI_MODULES:
        h = f"helper_rocke.instances.common.{m}.h"
        files[f"{h}.golden"] = out / "abi" / h
    return files


def normalize(text: str) -> str:
    """Drop the source hash the banners carry (it changes with any Python edit)."""
    text = re.sub(r"(\.py):[0-9a-f]{12}\b", r"\1:<sha>", text)
    return re.sub(r"\(sha256 [0-9a-f]{12}\)", "(sha256 <sha>)", text)


def platform_probe(
    root: Path,
    extra: str,
    fills: dict[str, str] | None = None,
    package: str = "batched_transpose",
    arch: str = "common",
    edits: list[tuple[str, str]] | None = None,
) -> dict:
    """The environment of a platform-instance probe under root: the package's
    real module with `extra` appended (its names added to __all__), imported
    from a symlink overlay of platform/python (TR_LIBRARY), the code maps mined
    from the tree, no facts, and `fills` (function -> C++) as the hand fills.
    `edits` are (old, new) text replacements made in the real module first."""
    real = ROCKE / "platform" / "python"
    lib = root / "python"
    path = ["rocke", "instances", arch]
    src, dst = real, lib
    for part in path:
        dst.mkdir(parents=True)
        for f in (src).iterdir():
            if f.name != part and f.name != "__pycache__":
                (dst / f.name).symlink_to(f)
        src, dst = src / part, dst / part
    dst.mkdir()
    for f in src.iterdir():
        if f.name != f"{package}.py" and f.name != "__pycache__":
            (dst / f.name).symlink_to(f)
    text = (src / f"{package}.py").read_text()
    for old, new in edits or []:
        assert old in text, old
        text = text.replace(old, new, 1)
    names = re.findall(r"^def (\w+)", extra, re.M)
    text = text.replace(
        "__all__ = [", "__all__ = [\n" + "".join(f'    "{n}",\n' for n in names), 1
    )
    (dst / f"{package}.py").write_text(text + "\n\n" + extra)
    holes = root / "holes" / package
    holes.mkdir(parents=True)
    for name, body in (fills or {}).items():
        (holes / f"{name}.cpp").write_text(body)
    (root / "data").mkdir()
    env = dict(os.environ)
    env.update(
        TR_PACKAGE=package,
        TR_LIBRARY=str(lib),
        TR_DATA=str(root / "data"),
        TR_MAPS=str(root / "maps"),
        TR_HOLES=str(root / "holes"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    p = subprocess.run(
        [sys.executable, "tools/mine_maps.py", "--rocke", str(ROCKE)]
        + ["--out", str(root / "maps"), "--only", "sigmap", "--only", "twin_map"],
        cwd=TR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    return env


def strip_holes(cpp: str) -> str:
    """The source without the #error line of each hole (it may be wrapped)."""
    return re.sub(r'#error (?:\\\n\s*)?".*"\n', "", cpp)
