# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fixed-recipe resolution for ``--specialize``.

A traced call names the solution hipBLASLt ran by its build-global
``solution_index``. The baseline build's ``device-library/MatchTable.yaml`` maps
that index to (stock logic file, ``SolutionIndex`` of a solution in it). The
solution found there is the call's recipe only if TensileLite's own naming of it
gives the logged solution and kernel names; a kernel name shared with another
recipe is no ambiguity, because the index already selects one solution.

Baseline root (``--hipblaslt``): the library that served the traces and its own
MatchTable, in the first of these layouts that exists:
``<root>/device-library/MatchTable.yaml`` + ``<root>/Tensile/library/<arch>/`` (a
cmake build tree), ``<root>/library/MatchTable.yaml`` + ``<root>/library/<arch>/``
(``geko --specialize --build-library`` output) or
``<root>/lib/hipblaslt/library/`` holding ``MatchTable.yaml`` (an install). The
stock logic root is ``--logic-root``, else ``logic=`` on the first line of
``<root>/params.txt``, else the hipBLASLt source checkout directory
``.../library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full`` that holds
every source logic file the MatchTable names.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from geko.specialize.calls import UNKNOWN

# Stock logic directory of a hipBLASLt source checkout, below projects/hipblaslt.
CHECKOUT_LOGIC = ("library", "src", "amd_detail", "rocblaslt", "src", "Tensile", "Logic", "asm_full")

ARCH_ISA = {"gfx950": (9, 5, 0), "gfx942": (9, 4, 2)}
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class UsageError(Exception):
    """Invalid command-line input; ``--specialize`` exits 2 without generating."""


class Refused(Exception):
    """A call that is not generated; ``code`` is stable, ``reason`` is for people."""

    def __init__(self, code: str, reason: str):
        super().__init__(f"{code}: {reason}")
        self.code, self.reason = code, reason


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def library_fp(library: Path) -> str:
    """Content-tree hash of a library directory (the ``library_fp`` a model-run harness records):
    sorted walk; per regular file ``rel\\0size\\0sha256\\0``, per symlink ``rel\\0L\\0target\\0``."""
    digest = hashlib.sha256()
    library = Path(library)
    for directory, dirs, names in os.walk(library):
        dirs.sort()
        for name in sorted(names):
            path = Path(directory, name)
            rel = str(path.relative_to(library))
            if path.is_symlink():
                digest.update(f"{rel}\0L\0{os.readlink(path)}\0".encode())
            else:
                digest.update(f"{rel}\0{path.stat().st_size}\0{sha256_bytes(path.read_bytes())}\0".encode())
    return digest.hexdigest()


def single_solution_logic(stock: Dict[str, Any], solution: Dict[str, Any], sizes) -> Dict[str, Any]:
    """One-solution Equality logic for (M, N, batch, K)."""
    keep = ("MinimumRequiredVersion", "ScheduleName", "ArchitectureName", "CUCount",
            "DeviceNames", "ProblemType", "DefaultSolution")
    logic = {k: copy.deepcopy(stock[k]) for k in keep if k in stock}
    logic["Solutions"] = [copy.deepcopy(solution)]
    # IndexOrder [2, 3, 0, 1] keys Equality rows as [M, N, batch, K].
    logic.update(IndexOrder=[2, 3, 0, 1], ExactLogic=[[list(sizes), [0, 0.0]]],
                 RangeLogic=None, TileSelectionIndices=None,
                 PerfMetric=stock.get("PerfMetric", "DeviceEfficiency"), LibraryType="Equality")
    return logic


def write_logic(path: Path, logic: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(logic, f, sort_keys=False, default_flow_style=None, width=1 << 16)


class Deriver:
    """Derives solutions with the TensileLite tree's LibraryIO, as TensileCreateLibrary does."""

    def __init__(self, arch: str):
        from tensilelite.Common.Architectures import IsaVersion
        from tensilelite.Common.Capabilities import makeIsaInfoMap
        from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
        from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain
        import tensilelite.LibraryIO as LibraryIO

        cxx = validateToolchain("amdclang++")
        self.isaInfoMap = makeIsaInfoMap([IsaVersion(*ARCH_ISA[arch])], cxx)
        bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
        self.assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
        self.LibraryIO = LibraryIO

    def solution(self, path: Path):
        try:
            lib = self.LibraryIO.parseLibraryLogicFile(str(path), self.assembler, False, False, False,
                                                       self.isaInfoMap, False)
        except RuntimeError as e:
            if str(e).startswith("AssertSizeEqual contradicts"):
                raise Refused("contradiction", f"{path.name}: {e}") from e
            # A TensileLite error on this logic refuses its call only; the run continues.
            raise Refused("derivation-error", f"derivation failed: {path.name}: TensileLite error: {e}") from e
        except SystemExit as e:  # TensileLite printExit: a fatal for this logic, not for the run
            raise Refused("derivation", f"derivation failed: {path.name}: the target tree exited ({e.code})") from None
        sols = list(lib.solutions.values()) if isinstance(lib.solutions, dict) else list(lib.solutions)
        if len(sols) != 1 or not sols[0]["Valid"]:
            raise Refused("derivation", f"derivation failed: {path.name}: the target tree derives no valid solution")
        return sols[0]

    def derive(self, logic: Dict[str, Any], scratch: Path, name: str):
        path = Path(scratch) / name
        write_logic(path, logic)
        return self.solution(path)

    def derive_named(self, logic: Dict[str, Any], scratch: Path, name: str, source: str):
        """derive(), with ``source`` (the stock recipe or a call's bespoke logic) named in a
        refusal instead of the scratch file ``name``."""
        try:
            return self.derive(logic, scratch, name)
        except Refused as e:
            raise Refused(e.code, e.reason.replace(name, source, 1)) from e


@lru_cache(maxsize=None)
def deriver(arch: str) -> Deriver:
    return Deriver(arch)


def names(solution) -> Tuple[str, str]:
    """(solution name, kernel name) as hipBLASLt logs them (TensileLite Min names, splitGSU off)."""
    from tensilelite.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin
    return getSolutionNameMin(solution, False), getKernelNameMin(solution, False)


def custom_kernel_name(solution: Dict[str, Any]) -> str | None:
    """Name of a hand-written custom kernel recipe (not specializable), else None."""
    ck = solution.get("CustomKernel")
    if isinstance(ck, dict) and ck.get("name") and not ck.get("generated", False):
        return ck["name"]
    return solution.get("CustomKernelName") or None


@dataclass(frozen=True)
class Recipe:
    """Recipe identity (source_file, sha256, in_file_index) plus what generation needs."""

    source_file: str
    sha256: str
    in_file_index: int
    baseline_index: int
    path: Path
    solution_name: str
    kernel_name: str
    custom: bool

    def key(self) -> Tuple[str, str, int]:
        return (self.source_file, self.sha256, self.in_file_index)

    def manifest(self) -> Dict[str, Any]:
        return {"source_file": self.source_file, "sha256": self.sha256,
                "in_file_index": self.in_file_index, "baseline_index": self.baseline_index}


class BaselineBuild:
    """The ``--hipblaslt`` baseline build root for one architecture."""

    def __init__(self, root: Path, arch: str, scratch: Path | None, logic_root: Path | str | None = None):
        self.root, self.arch = Path(root), arch
        self.scratch = Path(scratch) if scratch else None  # derivation scratch, set before resolving
        self.matchtable_path, self.library = self._layout()
        self.table = self._load_table()
        self.logic_root = self._logic_root(logic_root)
        self.fingerprint = library_fp(self.library)
        self._stock: Dict[Path, Tuple[Dict[str, Any], str]] = {}
        self._resolved: Dict[Tuple[int, str, str], Recipe] = {}
        self._derived: Dict[Tuple[Path, int], Any] = {}

    def _layout(self) -> Tuple[Path, Path]:
        """(MatchTable, library directory) of the first baseline layout present under the root."""
        layouts = [(self.root / "device-library" / "MatchTable.yaml", self.root / "Tensile" / "library" / self.arch),
                   (self.root / "library" / "MatchTable.yaml", self.root / "library" / self.arch),
                   (self.root / "lib" / "hipblaslt" / "library" / "MatchTable.yaml",
                    self.root / "lib" / "hipblaslt" / "library")]
        for table, library in layouts:
            if table.is_file() and library.is_dir():
                return table, library
        expected = "; ".join(f"{t} with {l}" for t, l in layouts)
        raise UsageError(f"--hipblaslt {self.root} is not a baseline root: the library that served the traces "
                         f"must come with that build's own MatchTable.yaml (the recipe map; another build's "
                         f"map is never used). Expected one of: {expected}")

    def _logic_root(self, explicit: Path | str | None) -> Path:
        """Stock logic root: --logic-root, else params.txt logic=, else the source checkout of the
        sources, else (geko --build-library output) OUT/logic. A given root must hold every source
        logic file the MatchTable names, so the recipes and a merged build's stock copy agree."""
        if explicit is not None:
            return self._bound(Path(explicit), "--logic-root")
        params = self.root / "params.txt"
        lines = params.read_text().splitlines() if params.is_file() else []
        match = re.search(r"(?:^|\s)logic=(\S+)", lines[0] if lines else "")
        if match:
            return self._bound(Path(match.group(1)), f"{params} logic=")
        roots = {self._checkout_root(Path(source)) for source, _ in self.table.values()}
        if len(roots) == 1 and None not in roots:
            return roots.pop()
        own = self.root / "logic"
        if self.matchtable_path == self.root / "library" / "MatchTable.yaml" and own.is_dir() and not self._outside(own):
            return own
        raise UsageError(f"--hipblaslt {self.root}: stock logic root not found (no --logic-root, no params.txt "
                         f"logic=, and the MatchTable's source logic files are not all in one hipBLASLt source "
                         f"checkout .../{'/'.join(CHECKOUT_LOGIC)}); pass --logic-root <the stock logic directory "
                         f"the library was built from>")

    def _outside(self, root: Path) -> list:
        """MatchTable source logic files not under root (also compared with symlinks resolved)."""
        resolved = root.resolve()
        return sorted(s for s, _ in self.table.values()
                      if not (Path(s).is_relative_to(root) or Path(s).resolve().is_relative_to(resolved)))

    def _bound(self, root: Path, what: str) -> Path:
        if not root.is_dir():
            raise UsageError(f"{what} {root} is not a directory; pass --logic-root <dir>")
        outside = self._outside(root)
        if outside:
            raise UsageError(f"{what} {root} does not hold the stock logic this library was built from: "
                             f"{len(outside)} source logic file(s) named by the MatchTable "
                             f"{self.matchtable_path} are outside it, e.g. {outside[0]}")
        return root

    def relative(self, path: Path) -> str:
        """A source logic file relative to the stock logic root, which holds every MatchTable
        source (checked when the root is chosen; symlinks resolved if needed)."""
        try:
            return str(path.relative_to(self.logic_root))
        except ValueError:
            return str(path.resolve().relative_to(self.logic_root.resolve()))

    @staticmethod
    def _checkout_root(source: Path) -> Path | None:
        for parent in source.parents:
            if parent.parts[-len(CHECKOUT_LOGIC):] == CHECKOUT_LOGIC:
                return parent
        return None

    def _load_table(self) -> Dict[int, Any]:
        """MatchTable {index: [source logic file, SolutionIndex]}; every entry and file checked up front."""
        with open(self.matchtable_path) as f:
            table = yaml.load(f, Loader=_LOADER) or {}
        if not isinstance(table, dict):
            raise UsageError(f"{self.matchtable_path}: not a MatchTable mapping")
        sources = set()
        for index, entry in table.items():
            if (not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[0], str)
                    or type(entry[1]) is not int):
                raise UsageError(f"{self.matchtable_path}: MatchTable entry {index} is not "
                                 f"[source logic file, SolutionIndex]: {entry!r}")
            sources.add(entry[0])
        missing = sorted(s for s in sources if not Path(s).is_file())
        if missing:
            raise UsageError(f"{self.matchtable_path}: {len(missing)} source logic file(s) named by the "
                             f"MatchTable do not exist, e.g. {missing[0]}")
        return table

    def stock(self, path: Path) -> Tuple[Dict[str, Any], str]:
        """(parsed stock logic, sha256 of its bytes)."""
        if path not in self._stock:
            data = path.read_bytes()
            self._stock[path] = (yaml.load(data, Loader=_LOADER), sha256_bytes(data))
        return self._stock[path]

    def resolve(self, observation: Tuple[Any, Any, Any]) -> Recipe:
        """Recipe of one (solution_index, solution_Name, kernel_name) observation, or Refused."""
        index, solution_name, kernel_name = observation
        if solution_name is UNKNOWN or kernel_name is UNKNOWN:
            raise Refused("rocroller", "non-TensileLite route (RocRoller): the record logs no "
                          "solution_Name/kernel_name, so no TensileLite recipe exists")
        if type(index) is not int or index not in self.table:
            raise Refused("index", f"index not in baseline MatchTable: solution_index {index}")
        key = (index, str(solution_name), str(kernel_name))
        if key not in self._resolved:
            self._resolved[key] = self._confirm(index, str(solution_name), str(kernel_name))
        return self._resolved[key]

    def _confirm(self, index: int, solution_name: str, kernel_name: str) -> Recipe:
        """The solution the MatchTable entry names, confirmed by TensileLite's own naming.

        TensileCreateLibrary writes [source logic file, SolutionIndex] (Run.py, libraryLogicIndex).
        That SolutionIndex is authoritative: it must name exactly one solution of the file, whose
        real position is recorded. Names only confirm that solution; a mismatch is refused, never
        resolved to another solution."""
        source, value = self.table[index]
        path = Path(source)
        stock, digest = self.stock(path)
        solutions = stock.get("Solutions") or []
        if stock.get("ArchitectureName") != self.arch:
            raise Refused("arch", f"index {index} resolves to {stock.get('ArchitectureName')} logic, not --arch {self.arch}")
        matches = [i for i, s in enumerate(solutions) if s.get("SolutionIndex") == value]
        if len(matches) != 1:
            raise Refused("recipe-name-mismatch", f"recipe name mismatch: index {index} names SolutionIndex {value} "
                          f"of {path}, which {len(matches)} of its {len(solutions)} solutions carry (exactly one "
                          "must)")
        local = matches[0]
        custom = custom_kernel_name(solutions[local])
        derived = (custom, custom) if custom else names(self._derive_stock(path, local))
        if derived != (solution_name, kernel_name):
            raise Refused("recipe-name-mismatch",
                          f"recipe name mismatch: index {index} resolves to {path.name}[{local}] (SolutionIndex "
                          f"{value}) named {derived[0]} / {derived[1]}, the trace logs {solution_name} / {kernel_name}")
        return Recipe(self.relative(path), digest, local, index, path, solution_name, kernel_name, bool(custom))

    def general_solution(self, recipe: Recipe):
        """The stock solution as the target tree derives it (cached per recipe)."""
        return self._derive_stock(recipe.path, recipe.in_file_index)

    def _derive_stock(self, path: Path, local: int):
        # The derived solution does not depend on the Equality row, so one
        # derivation per recipe serves name confirmation and every call.
        if (path, local) not in self._derived:
            stock, _ = self.stock(path)
            general = single_solution_logic(stock, stock["Solutions"][local], (1, 1, 1, 1))
            self._derived[(path, local)] = deriver(self.arch).derive_named(
                general, self.scratch, f"stock_{len(self._derived)}.yaml", f"{path.name}[{local}]")
        return self._derived[(path, local)]
