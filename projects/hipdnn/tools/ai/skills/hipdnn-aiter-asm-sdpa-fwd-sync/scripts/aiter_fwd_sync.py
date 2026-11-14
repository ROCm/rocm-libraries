#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sync the ASM SDPA forward kernels of hip-kernel-provider from AITER.

Ports only AITER's prebuilt ASM forward kernels: the .co objects and their CSV
tables under hsa/<arch>/<module>/. CK kernels, CK host code, FlyDSL and Triton
are never ported. Upstream host sources are read to detect kernel ABI and
dispatch changes and are never copied.

Subcommands:
    latest   print the newest AITER release tag and its commit
    archs    list the GPU archs that ship tracked forward ASM modules at a ref
    plan     compare the tree with a ref for the chosen archs and write a
             tiered report; changes nothing in the tree
    apply    apply a reviewed plan report to the tree
    mark-ported
             record that the engine now mirrors the upstream code (argument
             structs, packing, launch, dispatch, codegen) of a module's pinned commit
    verify   check the tree: pins, manifests, CSV rows against objects, symbols

Exit codes: 0 success or clean, 1 verify found problems, 2 cannot run.

Examples:
    python3 <skill>/scripts/aiter_fwd_sync.py latest
    python3 <skill>/scripts/aiter_fwd_sync.py archs --ref v0.1.27
    python3 <skill>/scripts/aiter_fwd_sync.py plan --ref v0.1.27 \\
        --arch gfx942,gfx950 --baseline 17d4a33b6f9535e820353ebc6217769efc3766d6 \\
        --report /tmp/aiter-plan.json
    python3 <skill>/scripts/aiter_fwd_sync.py apply --report /tmp/aiter-plan.json
    python3 <skill>/scripts/aiter_fwd_sync.py verify
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SCHEMA = 1
UPSTREAM_URL = "https://github.com/ROCm/aiter.git"
UPSTREAM_WEB = "https://github.com/ROCm/aiter"
ENGINE_DIR = PurePosixPath(
    "dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine"
)
KERNELS_DIR = ENGINE_DIR / "asm" / "asm_kernels"
ENGINE_CMAKE = ENGINE_DIR / "CMakeLists.txt"
PINS_NAME = "aiter_fwd_pins.json"
SOURCE_MD = "SOURCE.md"
SKILL_NAME = "hipdnn-aiter-asm-sdpa-fwd-sync"

# Forward ASM modules this tool ports. Each is a directory of prebuilt objects
# and CSV tables under AITER's hsa/<arch>/. Adding one is a deliberate change.
TRACKED_MODULES = ("fmha_v3_fwd", "fmha_fwd_bf16", "fmha_fwd_bf16_varlen")

# Forward ASM modules AITER ships that are deliberately not ported.
UNTRACKED_MODULES = {
    "fmha_fwd_mxfp8": "needs MX block-scale descales, which the hipDNN SDPA graph does not model",
    "fmha_v4_fwd": "not consumed by the ASM SDPA engine",
}

# CSV columns that are names, and tile-size columns the engine reads as data.
NAME_COLUMNS = frozenset({"knl_name", "co_name"})
DATA_COLUMNS = frozenset({"ts_qo", "ts_kv", "ts"})

# A symbol from Composable Kernel. Any one of these in an object means it is
# not one of AITER's ASM kernels. "ck_tile" anywhere also covers _ZN7ck_tile.
CK_SYMBOL = re.compile(r"ck_tile|^_ZN2ck|^_ZNK2ck|kentry")

COMMENT_PREFIXES = ("#", "//", ";")

# Findings that make a plan impossible to apply.
BLOCKING_KINDS = frozenset(
    {
        "ck_symbol",
        "not_elf",
        "object_missing_upstream",
        "symbol_missing",
        "unexpected_files",
    }
)

# Findings that mean the mirrored host code must change: apply then keeps the
# module's host-code commit, and plan names the other archs that share that code.
HOST_CODE_KINDS = frozenset({"reference_changed", "reference_missing"})


@dataclass(frozen=True)
class Ref:
    """A region of upstream host source that the engine mirrors by hand."""

    path: str
    kind: str  # "file", "struct", "function" (C++) or "pyfunction" (Python)
    # The function or struct name; for "pyfunction" the dotted name of a function
    # nested in others, such as "outer.inner".
    name: str | None
    tier: int
    what: str


# The SGPR padding structs every ASM kernel argument struct is built from, and
# the codegen that turns every module's CSVs into a registry. The vendored
# asm_kernels/codegen.py is a derived copy, so an upstream change is reviewed
# and ported like the host code.
_SHARED_REFS = tuple(
    Ref("csrc/include/aiter_hip_common.h", "struct", name, 3, "kernel argument padding")
    for name in ("p1", "p2", "p3")
) + (
    Ref(
        "hsa/codegen.py",
        "file",
        None,
        2,
        "registry codegen; the vendored asm_kernels/codegen.py is a derived copy",
    ),
)

# Upstream code read to detect ABI, dispatch and codegen changes; never copied.
# In csrc/cpp_itfs/mha_fwd.cu only the ASM functions are listed: fmha_fwd_ck()
# and the CK fallback in mha_fwd() are out of scope. aiter/ops/mha.py decides in
# Python, before csrc/ is reached, which cases may use the ASM v3 kernels; a
# "pyfunction" region also holds the helpers the function calls, such as
# is_fmha_v3_fp8() and _fmha_kv_byte_extent_ge_u32(), so they need no entry.
REFERENCES: dict[str, tuple[Ref, ...]] = {
    "fmha_v3_fwd": _SHARED_REFS
    + (
        Ref(
            "csrc/include/mha_fwd.h",
            "struct",
            "fmha_fwd_v3_args",
            3,
            "kernel argument struct",
        ),
        Ref(
            "csrc/cpp_itfs/mha_fwd.cu",
            "function",
            "init_fmha_fwd_v3_args",
            3,
            "kernel argument packing",
        ),
        Ref("csrc/cpp_itfs/mha_fwd.cu", "function", "get_grid_dim", 3, "launch grid"),
        Ref(
            "csrc/cpp_itfs/mha_fwd.cu",
            "function",
            "get_kernel_co_name",
            3,
            "object path per device (MI300/MI308)",
        ),
        Ref(
            "csrc/include/aiter_hip_common.h",
            "function",
            "is_mi308_device",
            3,
            "MI308 detection, which picks the gfx942 MI300/ or MI308/ objects",
        ),
        Ref(
            "csrc/cpp_itfs/mha_fwd.cu",
            "function",
            "get_kernel_name_key",
            2,
            "kernel selection key",
        ),
        Ref(
            "csrc/cpp_itfs/mha_fwd.cu",
            "function",
            "get_cfg_mask_type",
            2,
            "mask mapping",
        ),
        Ref(
            "csrc/cpp_itfs/mha_fwd.cu",
            "function",
            "fmha_fwd_v3",
            3,
            "ASM dispatch conditions and block size; each early 'return -1' hands "
            "the case to CK, so the engine must decline it",
        ),
        *(
            Ref(
                "aiter/ops/mha.py",
                "pyfunction",
                f"{caller}.can_impl_fmha_v3_fwd",
                3,
                f"Python ASM v3 gate for {mode}; a case it rejects never reaches the "
                "ASM kernels, so the engine must decline it if it is a kernel limit",
            )
            for caller, mode in (
                ("_flash_attn_forward", "batch mode"),
                ("_flash_attn_varlen_forward", "group (varlen) mode"),
            )
        ),
    ),
    "fmha_fwd_bf16": _SHARED_REFS
    + (
        Ref(
            "csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu",
            "file",
            None,
            3,
            "host interface: argument struct, checks and launch",
        ),
    ),
    "fmha_fwd_bf16_varlen": _SHARED_REFS
    + (
        Ref(
            "csrc/py_itfs_cu/asm_fmha_fwd_with_sink_varlen.cu",
            "file",
            None,
            3,
            "host interface: argument struct, checks and launch",
        ),
    ),
}


class SyncError(Exception):
    """The tool cannot run; exit code 2."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(repo: Path | None, *args: str) -> str:
    command = ["git"] + (["-C", str(repo)] if repo else []) + list(args)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SyncError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout


def try_git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def module_key(arch: str, module: str) -> str:
    return f"{arch}/{module}"


def split_key(key: str) -> tuple[str, str]:
    arch, _, module = key.partition("/")
    if not arch or not module:
        raise SyncError(f"expected <arch>/<module>, got '{key}'")
    return arch, module


def version_key(tag: str) -> tuple[int, int, int, int] | None:
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?", tag)
    if not match:
        return None
    major, minor, patch, post = match.groups()
    return int(major), int(minor), int(patch), int(post) if post is not None else -1


# --------------------------------------------------------------------------- #
# ELF symbols
# --------------------------------------------------------------------------- #


class NotElf(Exception):
    pass


def elf_symbols(data: bytes) -> set[str]:
    """Names in the .symtab and .dynsym of a 64-bit little-endian ELF object."""
    if data[:4] != b"\x7fELF" or len(data) < 64:
        raise NotElf("no ELF magic")
    if data[4] != 2 or data[5] != 1:
        raise NotElf("not a 64-bit little-endian ELF")
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    if shentsize != 64 or shoff + shnum * 64 > len(data):
        raise NotElf("bad section header table")
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * 64) for i in range(shnum)
    ]
    names: set[str] = set()
    for _, sh_type, _, _, offset, size, link, _, _, entsize in sections:
        if sh_type not in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
            continue
        if link >= len(sections):
            raise NotElf("symbol table links a missing string table")
        str_offset, str_size = sections[link][4], sections[link][5]
        if offset + size > len(data) or str_offset + str_size > len(data):
            raise NotElf("symbol or string table runs past the end of the file")
        strtab = data[str_offset : str_offset + str_size]
        step = entsize or 24
        if step < 4:
            raise NotElf("bad symbol entry size")
        for entry in range(offset, offset + size - step + 1, step):
            st_name = struct.unpack_from("<I", data, entry)[0]
            end = strtab.find(b"\0", st_name)
            if st_name and end > st_name:
                names.add(strtab[st_name:end].decode("utf-8", "replace"))
    return names


def object_problems(data: bytes, knl: str) -> list[tuple[str, str]]:
    """Why an object is not the ASM kernel knl, as (finding kind, message) pairs.
    plan blocks apply on any of them and verify reports them, so both use this."""
    try:
        symbols = elf_symbols(data)
    except NotElf as error:
        return [("not_elf", str(error))]
    problems = []
    if knl not in symbols and f"{knl}.kd" not in symbols:
        problems.append(("symbol_missing", f"does not define {knl}"))
    ck = sorted(s for s in symbols if CK_SYMBOL.search(s))
    if ck:
        problems.append(
            ("ck_symbol", f"CK symbols {ck[:3]}; only ASM kernels are ported")
        )
    return problems


# --------------------------------------------------------------------------- #
# Module contents
# --------------------------------------------------------------------------- #


@dataclass
class CsvTable:
    header: list[str]
    rows: list[dict[str, str]]
    lines: dict[str, str]  # knl_name -> the row's text upstream, comments removed
    header_line: str


def _strip_line_end_comment(line: str) -> str:
    """Cut a line at its first comment outside quotes, as AITER's hsa/codegen.py does:
    '#' anywhere, '//' unless it follows ':', and ';' after a space or tab."""
    in_quotes = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == '"':
            if in_quotes and line[index + 1 : index + 2] == '"':
                index += 2
                continue
            in_quotes = not in_quotes
        elif not in_quotes:
            if char == "#" or (char == ";" and index > 0 and line[index - 1] in " \t"):
                return line[:index].rstrip()
            if line.startswith("//", index):
                if index > 0 and line[index - 1] == ":":
                    index += 2
                    continue
                return line[:index].rstrip()
        index += 1
    return line


def csv_content_lines(text: str) -> list[str]:
    """The non-empty lines of a CSV without the comments AITER's codegen accepts:
    /* block */ comments, whole-line comments and line-end comments."""
    while (start := text.find("/*")) >= 0:
        end = text.find("*/", start + 2)
        text = text[:start] if end < 0 else text[:start] + text[end + 2 :]
    content = []
    for line in text.split("\n"):
        if line.lstrip().startswith(COMMENT_PREFIXES):
            continue
        line = _strip_line_end_comment(line).rstrip()
        if line.strip():
            content.append(line)
    return content


def parse_csv(data: bytes, where: str) -> CsvTable:
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    content = csv_content_lines(text)
    if not content:
        raise SyncError(f"{where}: empty CSV")
    parsed = list(csv.reader(content))
    header = [column.strip() for column in parsed[0]]
    if not NAME_COLUMNS.issubset(header):
        raise SyncError(f"{where}: CSV lacks knl_name or co_name")
    rows: list[dict[str, str]] = []
    lines: dict[str, str] = {}
    for line, values in zip(content[1:], parsed[1:]):
        if len(values) != len(header):
            raise SyncError(
                f"{where}: row has {len(values)} fields, header has {len(header)}"
            )
        row = {column: value.strip() for column, value in zip(header, values)}
        if row["knl_name"] in lines:
            raise SyncError(f"{where}: kernel {row['knl_name']} appears twice")
        rows.append(row)
        lines[row["knl_name"]] = line
    return CsvTable(header, rows, lines, content[0])


@dataclass
class ModuleTree:
    csvs: dict[str, CsvTable] = field(default_factory=dict)
    objects: dict[str, bytes] = field(default_factory=dict)
    other: dict[str, bytes] = field(default_factory=dict)

    def object_roots(self) -> set[str]:
        return {str(PurePosixPath(path).parent) for path in self.objects}


def module_tree(files: dict[str, bytes], where: str) -> ModuleTree:
    tree = ModuleTree()
    for path, data in files.items():
        if path.endswith(".csv"):
            tree.csvs[path] = parse_csv(data, f"{where}/{path}")
        elif path.endswith(".co"):
            tree.objects[path] = data
        else:
            tree.other[path] = data
    return tree


def read_dir(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --------------------------------------------------------------------------- #
# Upstream access
# --------------------------------------------------------------------------- #


class Upstream:
    """A blobless partial clone of AITER; files are read through sparse checkouts."""

    def __init__(self, url: str, cache: Path):
        self.url = url
        self.repo = cache / "aiter"
        self._fetched = False

    def ensure(self) -> None:
        if not (self.repo / ".git").is_dir():
            self.repo.parent.mkdir(parents=True, exist_ok=True)
            run_git(
                None,
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                self.url,
                str(self.repo),
            )
            run_git(self.repo, "sparse-checkout", "set", "--no-cone", "/.gitignore")
            self._fetched = True
            return
        origin = run_git(self.repo, "remote", "get-url", "origin").strip()
        if origin != self.url:
            raise SyncError(
                f"cache {self.repo} tracks {origin}, not {self.url}; pass another --cache"
            )
        if not self._fetched:
            run_git(
                self.repo,
                "fetch",
                "--quiet",
                "--force",
                "--tags",
                "--filter=blob:none",
                "origin",
            )
            self._fetched = True

    def resolve(self, ref: str) -> tuple[str, str | None]:
        self.ensure()
        commit = try_git(
            self.repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"
        )
        if commit is None:
            try_git(self.repo, "fetch", "--quiet", "--filter=blob:none", "origin", ref)
            commit = try_git(
                self.repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"
            )
        if commit is None:
            raise SyncError(f"AITER ref '{ref}' not found in {self.url}")
        tag = (
            ref
            if try_git(
                self.repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{ref}"
            )
            else None
        )
        return commit.strip(), tag

    def latest_tag(self) -> tuple[str, str]:
        """The newest release tag and where it was learned.

        For a GitHub upstream this is the release GitHub marks latest, which
        skips tags that have no published release yet. Otherwise, or when the
        API cannot be reached, it is the highest vX.Y.Z[.postN] tag.
        """
        self.ensure()
        match = re.fullmatch(
            r"https://github\.com/([^/]+/[^/]+?)(?:\.git)?/?", self.url
        )
        if match:
            api = f"https://api.github.com/repos/{match.group(1)}/releases/latest"
            try:
                with urllib.request.urlopen(api, timeout=20) as response:
                    return json.load(response)["tag_name"], "GitHub latest release"
            except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
                pass
        tags = [
            t
            for t in run_git(self.repo, "tag", "--list", "v*").split()
            if version_key(t)
        ]
        if not tags:
            raise SyncError(f"no vX.Y.Z release tags in {self.url}")
        return max(tags, key=version_key), "highest release tag"

    def subdirs(self, commit: str, path: str) -> set[str]:
        """The names of the directories directly under path at a commit."""
        out = run_git(self.repo, "ls-tree", commit, "--", f"{path.rstrip('/')}/")
        names = set()
        for line in out.splitlines():
            meta, _, name = line.partition("\t")
            if meta.split()[1] == "tree":
                names.add(PurePosixPath(name).name)
        return names

    def materialize(
        self, commit: str, dirs: list[str], files: list[str]
    ) -> dict[str, bytes]:
        """Check out only the given paths at a commit and return their bytes."""
        patterns = [f"/{d.strip('/')}/" for d in dirs] + [f"/{f}" for f in files]
        run_git(self.repo, "sparse-checkout", "set", "--no-cone", *patterns)
        run_git(self.repo, "checkout", "--quiet", "--force", "--detach", commit)
        result: dict[str, bytes] = {}
        for d in dirs:
            root = self.repo / d
            for rel, data in read_dir(root).items():
                result[f"{d.strip('/')}/{rel}"] = data
        for f in files:
            path = self.repo / f
            if path.is_file():
                result[f] = path.read_bytes()
        return result


def subtree(files: dict[str, bytes], prefix: str) -> dict[str, bytes]:
    prefix = prefix.rstrip("/") + "/"
    return {
        path[len(prefix) :]: data
        for path, data in files.items()
        if path.startswith(prefix)
    }


# --------------------------------------------------------------------------- #
# Source regions
# --------------------------------------------------------------------------- #


def _brace_block(text: str, start: int) -> int | None:
    """Index just past the brace block that opens at or after start."""
    open_at = text.find("{", start)
    if open_at < 0:
        return None
    depth = 0
    for index in range(open_at, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


_PY_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _python_def(node: ast.AST, name: str) -> ast.AST | None:
    """The first function or class called name in the body of node, including the
    blocks (if, try, with, ...) it opens; never inside another function or class."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _PY_DEFS):
            if child.name == name:
                return child
        elif (found := _python_def(child, name)) is not None:
            return found
    return None


_PY_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _own_nodes(node: ast.AST):
    """node and the nodes below it in its own scope: the functions, classes and
    lambdas it defines are yielded, but their bodies are other scopes."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*_PY_DEFS, ast.Lambda)):
            yield child
        else:
            yield from _own_nodes(child)


def _parameters(function: ast.AST) -> set[str]:
    return {a.arg for a in ast.walk(function.args) if isinstance(a, ast.arg)}


def _bound_names(statement: ast.AST) -> set[str]:
    """The names a statement binds in the scope that holds it."""
    if isinstance(statement, _PY_DEFS):
        return {statement.name}
    names: set[str] = set()
    for node in _own_nodes(statement):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, _PY_DEFS):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def _read_names(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _bound_names_in(function: ast.AST) -> set[str]:
    """The names a function binds in its own body."""
    names: set[str] = set()
    for statement in function.body:
        names |= _bound_names(statement)
    return names


def _python_function(text: str, dotted: str) -> str | None:
    """The source of a Python function by its dotted name, decorators included,
    followed by the code the file defines that its result depends on, in file order:
    each name it reads from an enclosing scope resolves, as Python resolves it, to
    the functions and statements that bind the name there, or to a parameter, which
    brings in that function's signature; the same applies to what they read in turn.
    A dispatch gate nested in its caller reads the caller's locals (swa, hdim_q) and
    calls helpers (is_fmha_v3_fp8), so a change confined to one of them shows too.
    Imported names are not followed."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    node: ast.AST | None = tree
    scopes: list[ast.AST] = []
    for part in dotted.split("."):
        scopes.append(node)
        node = _python_def(node, part)
        if node is None:
            return None

    lines = text.split("\n")
    # Each part of the region by its first line: the function, then every binding
    # statement, helper and enclosing signature it depends on.
    parts: dict[int, str] = {}
    seen: set[object] = {id(node)}

    def add(unit: ast.AST, signature: bool = False) -> int:
        decorators = getattr(unit, "decorator_list", [])
        first = min([unit.lineno] + [d.lineno for d in decorators])
        last = unit.body[0].lineno - 1 if signature else unit.end_lineno
        parts[first] = "\n".join(lines[first - 1 : last])
        return first

    start = add(node)
    # (reader, the scopes enclosing it, the names it binds itself)
    pending: list[tuple[ast.AST, list[ast.AST], set[str]]] = [
        (node, scopes, _parameters(node) | _bound_names_in(node))
    ]
    while pending:
        reader, enclosing, own = pending.pop()
        for name in sorted(_read_names(reader) - own):
            # Python resolves a free name in the enclosing functions and then the
            # module; class bodies are not enclosing scopes.
            for depth in range(len(enclosing) - 1, -1, -1):
                scope = enclosing[depth]
                if isinstance(scope, ast.ClassDef):
                    continue
                binders = [
                    s for s in scope.body if s is not reader and name in _bound_names(s)
                ]
                is_parameter = isinstance(scope, _PY_FUNCTIONS) and (
                    name in _parameters(scope)
                )
                if not binders and not is_parameter:
                    continue
                if is_parameter and ("signature", id(scope)) not in seen:
                    seen.add(("signature", id(scope)))
                    add(scope, signature=True)
                for binder in binders:
                    if id(binder) in seen or isinstance(
                        binder, (ast.Import, ast.ImportFrom)
                    ):
                        continue
                    seen.add(id(binder))
                    add(binder)
                    inner = enclosing[: depth + 1]
                    if isinstance(binder, _PY_FUNCTIONS):
                        own_names = _parameters(binder) | _bound_names_in(binder)
                        pending.append((binder, inner, own_names))
                    else:
                        pending.append((binder, inner, set()))
                break

    rest = [parts[line] for line in sorted(parts) if line != start]
    return "\n\n".join([parts[start], *rest]) + "\n"


def extract_region(text: str, ref: Ref) -> str | None:
    if ref.kind == "file":
        return text
    if ref.kind == "pyfunction":
        return _python_function(text, ref.name)
    if ref.kind == "struct":
        pattern = re.compile(
            rf"^[ \t]*struct\b[^;{{\n]*\b{re.escape(ref.name)}\b[^;\n]*$", re.M
        )
        for match in pattern.finditer(text):
            end = _brace_block(text, match.start())
            if end is not None:
                semicolon = text.find(";", end)
                return text[match.start() : semicolon + 1 if semicolon >= 0 else end]
        return None
    pattern = re.compile(rf"^[A-Za-z_][^;(\n]*\b{re.escape(ref.name)}\s*\(", re.M)
    for match in pattern.finditer(text):
        brace = text.find("{", match.end())
        semicolon = text.find(";", match.end())
        if brace < 0 or (0 <= semicolon < brace):
            continue  # a declaration
        end = _brace_block(text, match.start())
        if end is not None:
            return text[match.start() : end]
    return None


# --------------------------------------------------------------------------- #
# Repository state
# --------------------------------------------------------------------------- #


def find_repo_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).resolve()
    else:
        root = Path(run_git(None, "rev-parse", "--show-toplevel").strip())
    if not (root / KERNELS_DIR).is_dir():
        raise SyncError(f"{root} has no {KERNELS_DIR}; pass --repo-root")
    return root


def load_pins(kernels: Path) -> dict:
    path = kernels / PINS_NAME
    if not path.exists():
        return {"schema": SCHEMA, "upstream": UPSTREAM_WEB, "modules": {}}
    pins = json.loads(path.read_text(encoding="utf-8"))
    if pins.get("schema") != SCHEMA:
        raise SyncError(f"{path}: unsupported schema {pins.get('schema')}")
    return pins


def write_pins(kernels: Path, pins: dict) -> None:
    pins["modules"] = dict(sorted(pins["modules"].items()))
    (kernels / PINS_NAME).write_text(
        json.dumps(pins, indent=2) + "\n", encoding="utf-8"
    )


def cmake_archs(text: str) -> set[str]:
    match = re.search(r"set\(\s*ASM_SDPA_GPU_ARCHS\s+([^)]*)\)", text)
    return set(match.group(1).split()) if match else set()


def excluded_reason(row: dict[str, str], rules: list[dict]) -> str | None:
    for rule in rules:
        if all(row.get(column) in values for column, values in rule["match"].items()):
            return rule["reason"]
    return None


def render_source_md(arch: str, module: str, entry: dict) -> str:
    tag = entry.get("tag") or "none (commit pin)"
    lines = [
        f"# Source Provenance — {arch} / {module}",
        "",
        f"Generated by `aiter_fwd_sync.py` from `../../{PINS_NAME}`. Do not edit it by",
        f"hand; use the `{SKILL_NAME}` skill to change this directory.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Repository | {UPSTREAM_WEB} |",
        f"| Tag | {tag} |",
        f"| Commit | `{entry['commit']}` |",
        f"| Upstream path | `{entry['upstream_path']}/` |",
        "| Host code the engine mirrors | "
        + (
            f"`{entry['abi_commit']}` |"
            if entry.get("abi_commit")
            else "not yet ported |"
        ),
        "",
        "Every object is one of AITER's prebuilt ASM kernels, copied byte for byte.",
        "No CK kernel and no upstream source file is vendored here. Each CSV keeps",
        "the upstream rows that no rule below excludes, as written upstream with",
        "comments removed.",
        "",
        "## Excluded upstream rows",
        "",
    ]
    if entry["exclude"]:
        lines += ["| Match | Reason |", "|---|---|"]
        for rule in entry["exclude"]:
            match = "; ".join(
                f"`{c}` in {', '.join(v)}" for c, v in rule["match"].items()
            )
            lines.append(f"| {match} | {rule['reason']} |")
    else:
        lines.append("None.")
    if entry.get("include"):
        lines += ["", "## Rows kept despite a rule", ""]
        lines += [f"- `{name}`" for name in entry["include"]]
    lines += ["", f"## SHA256 manifest ({len(entry['files'])} files)", "", "```"]
    lines += [f"{digest}  {path}" for path, digest in sorted(entry["files"].items())]
    lines += ["```", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #


class Findings:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, tier: int, kind: str, detail: str, diff: str | None = None) -> None:
        item = {"tier": tier, "kind": kind, "detail": detail}
        if diff:
            item["diff"] = diff
        self.items.append(item)

    @property
    def tier(self) -> int:
        return max((item["tier"] for item in self.items), default=0)


def parse_rule(spec: str, reason: str) -> dict:
    match: dict[str, list[str]] = {}
    for part in spec.split(";"):
        column, sep, values = part.partition("=")
        if not sep or not column.strip() or not values.strip():
            raise SyncError(
                f"bad --exclude match '{spec}'; expected column=v1,v2[;column=v]"
            )
        match[column.strip()] = [v.strip() for v in values.split(",")]
    if not reason.strip():
        raise SyncError("--exclude needs a reason")
    return {"match": match, "reason": reason.strip()}


def filtered_csv(table: CsvTable, dropped: set[str]) -> bytes:
    """The upstream table without the dropped kernels or comments; rows otherwise as
    written upstream, so the vendored codegen never meets a comment."""
    lines = [table.header_line]
    lines += [
        table.lines[row["knl_name"]]
        for row in table.rows
        if row["knl_name"] not in dropped
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def cfg_name(csv_path: str) -> str:
    """The registry a CSV feeds: the vendored codegen merges CSVs by this name."""
    return PurePosixPath(csv_path).name.split(".")[0]


def plan_module(
    arch: str,
    module: str,
    local_files: dict[str, bytes],
    arch_has_forward: bool,
    target_files: dict[str, bytes],
    target_references: dict[str, str],
    baseline: dict | None,
    abi_baseline: dict | None,
    rules: list[dict],
    includes: set[str],
    cmake_text: str,
) -> tuple[Findings, dict]:
    """Classify one module and compute the files it holds after apply."""
    findings = Findings()
    key = module_key(arch, module)
    up = module_tree(target_files, f"upstream {key}")
    local = module_tree(local_files, key) if local_files else None
    base = module_tree(baseline["files"], f"baseline {key}") if baseline else None
    rules = list(rules)

    if not arch_has_forward:
        findings.add(3, "new_arch", f"{arch} has no forward kernels in the tree yet")
    if local is None:
        findings.add(
            3,
            "new_module",
            (
                f"{key} is not vendored yet; read its reference sources in full, since "
                "there is no baseline to diff them against"
                if abi_baseline is None
                else f"{key} is not vendored yet; its host code is the engine's one copy, "
                "so the reference diffs below run from the host-code commit of the archs "
                "already vendored"
            ),
        )
    elif local.object_roots() != up.object_roots():
        findings.add(
            3,
            "layout_changed",
            f"object directories {sorted(local.object_roots())} -> {sorted(up.object_roots())}",
        )
    if arch not in cmake_archs(cmake_text):
        findings.add(
            3,
            "cmake_arch_missing",
            f"{arch} is not in ASM_SDPA_GPU_ARCHS in {ENGINE_CMAKE}",
        )
    if not re.search(rf"-m\s+{re.escape(module)}\b", cmake_text):
        findings.add(
            3,
            "cmake_module_missing",
            f"{ENGINE_CMAKE} has no codegen command for -m {module}",
        )

    kept_csvs: dict[str, bytes] = {}
    kept_objects: dict[str, str] = {}  # object path -> the kernel it must define
    excluded: list[dict] = []
    auto_rules: list[dict] = []
    for name in sorted(set(up.csvs) | set(local.csvs if local else {})):
        up_csv = up.csvs.get(name)
        local_csv = local.csvs.get(name) if local else None
        if up_csv is None:
            findings.add(3, "csv_removed_upstream", f"{name} no longer exists upstream")
            continue
        if local_csv and local_csv.header != up_csv.header:
            findings.add(
                3, "header_changed", f"{name}: {local_csv.header} -> {up_csv.header}"
            )
        local_rows = {
            row["knl_name"]: row for row in (local_csv.rows if local_csv else [])
        }
        base_csv = (
            base.csvs.get(name) if base and baseline["infer_exclusions"] else None
        )
        base_names = {row["knl_name"] for row in base_csv.rows} if base_csv else set()
        known: dict[str, set[str]] = {}
        for row in local_rows.values():
            for column, value in row.items():
                known.setdefault(column, set()).add(value)
        dispatch_columns = [
            c for c in up_csv.header if c not in NAME_COLUMNS | DATA_COLUMNS
        ]
        known_combinations = {
            tuple(row.get(c) for c in dispatch_columns) for row in local_rows.values()
        }
        dropped: set[str] = set()
        for row in up_csv.rows:
            knl = row["knl_name"]
            reason = None if knl in includes else excluded_reason(row, rules)
            if (
                reason is None
                and knl not in includes
                and local_csv is not None
                and knl not in local_rows
                and knl in base_names
            ):
                reason = f"not vendored at baseline {baseline['commit'][:12]}"
                rule = {"match": {"knl_name": [knl]}, "reason": reason}
                auto_rules.append(rule)
                rules.append(rule)
            if reason is not None:
                excluded.append({"csv": name, "knl_name": knl, "reason": reason})
                dropped.add(knl)
                if knl in local_rows:
                    findings.add(2, "row_dropped_by_rule", f"{knl}: {reason}")
                continue
            for root in sorted(up.object_roots()):
                kept_objects[str(PurePosixPath(root) / row["co_name"])] = knl
            if local_csv is None:
                continue
            if knl not in local_rows:
                new = [
                    f"{c}={row[c]}"
                    for c in dispatch_columns
                    if row[c] not in known.get(c, set())
                ]
                if new:
                    findings.add(
                        2, "row_added_new_value", f"{knl}: new {', '.join(new)}"
                    )
                elif tuple(row[c] for c in dispatch_columns) not in known_combinations:
                    combination = ", ".join(f"{c}={row[c]}" for c in dispatch_columns)
                    findings.add(
                        2,
                        "row_added_new_combination",
                        f"{knl}: every value is known, but not together: {combination}",
                    )
                else:
                    findings.add(1, "row_added", knl)
            elif local_rows[knl] != row:
                changed = sorted(c for c in row if local_rows[knl].get(c) != row[c])
                tier = 1 if set(changed) <= DATA_COLUMNS else 2
                findings.add(tier, "row_changed", f"{knl}: {', '.join(changed)}")
        upstream_names = {row["knl_name"] for row in up_csv.rows}
        for knl in sorted(set(local_rows) - upstream_names):
            findings.add(
                2,
                "row_removed_upstream",
                f"{knl}: AITER no longer ships it as an ASM kernel; the engine "
                "must decline it (never substitute a CK kernel)",
            )
        kept_csvs[name] = filtered_csv(up_csv, dropped)

    target: dict[str, bytes] = dict(kept_csvs)
    for path, knl in sorted(kept_objects.items()):
        data = up.objects.get(path)
        if data is None:
            findings.add(
                3, "object_missing_upstream", f"{path} is named by a CSV row but absent"
            )
            continue
        for kind, message in object_problems(data, knl):
            findings.add(3, kind, f"{path}: {message}")
        target[path] = data
        if local is not None:
            if path in local.objects and local.objects[path] != data:
                findings.add(1, "object_changed", path)
            if (
                base is not None
                and path in local.objects
                and path in base.objects
                and local.objects[path] != base.objects[path]
            ):
                findings.add(
                    1,
                    "local_differs_from_baseline",
                    f"{path}: the vendored bytes are not those at the baseline commit",
                )
    if local is not None:
        for path in sorted(set(local.objects) - set(target)):
            findings.add(1, "object_removed", path)
    stray = sorted(p for p in (local.other if local else {}) if p != SOURCE_MD)
    if stray:
        findings.add(
            3,
            "unexpected_files",
            f"{key} holds files this tool does not manage: {stray}",
        )

    for ref in REFERENCES[module]:
        target_text = target_references.get(ref.path)
        region = extract_region(target_text, ref) if target_text is not None else None
        if region is None:
            findings.add(
                3,
                "reference_missing",
                f"{ref.path}::{ref.name or '(file)'} ({ref.what}) not found at the target",
            )
            continue
        if abi_baseline is None:
            continue
        base_text = abi_baseline["references"].get(ref.path)
        base_region = extract_region(base_text, ref) if base_text is not None else None
        if base_region != region:
            diff = "".join(
                difflib.unified_diff(
                    (base_region or "").splitlines(keepends=True),
                    region.splitlines(keepends=True),
                    f"baseline/{ref.path}",
                    f"target/{ref.path}",
                )
            )
            findings.add(
                ref.tier,
                "reference_changed",
                f"{ref.path}::{ref.name or '(file)'}: {ref.what}",
                diff,
            )
    if abi_baseline is None and local is not None:
        findings.add(
            3,
            "baseline_unknown",
            "no host-code baseline (no arch of the module records an abi_commit, and "
            "no --baseline): upstream ABI and dispatch changes cannot be diffed; "
            "review the reference sources by hand",
        )
    elif abi_baseline is not None and baseline is not None:
        if abi_baseline["commit"] != baseline["commit"]:
            findings.add(
                3,
                "abi_unported",
                f"the engine mirrors AITER host code at {abi_baseline['commit'][:12]}, "
                f"older than the vendored objects ({baseline['commit'][:12]}); the "
                "reference diffs here include the changes still to port",
            )

    plan = {
        "upstream_path": f"hsa/{arch}/{module}",
        "baseline": baseline["commit"] if baseline else None,
        "abi_baseline": abi_baseline["commit"] if abi_baseline else None,
        "exclude": rules,
        "auto_exclude": auto_rules,
        "include": sorted(includes),
        "excluded_rows": excluded,
        "csv_headers": {name: up.csvs[name].header for name in sorted(kept_csvs)},
        "target": {path: sha256(data) for path, data in sorted(target.items())},
        "local_snapshot": {
            path: sha256(data) for path, data in sorted(local_files.items())
        },
    }
    return findings, plan


def reference_files(modules: list[str]) -> list[str]:
    return sorted({ref.path for module in modules for ref in REFERENCES[module]})


def forward_modules(entries: set[str]) -> tuple[list[str], list[str]]:
    """Of the module directories under an AITER hsa/<arch>/, the tracked forward
    ASM modules and the forward ones this tool does not port."""
    tracked = [m for m in TRACKED_MODULES if m in entries]
    untracked = sorted(
        m
        for m in entries
        if m.startswith("fmha") and "fwd" in m and m not in TRACKED_MODULES
    )
    return tracked, untracked


def cmd_plan(args: argparse.Namespace, upstream: Upstream, root: Path) -> int:
    kernels = root / KERNELS_DIR
    archs = [a.strip() for a in args.arch.split(",") if a.strip()]
    if not archs:
        raise SyncError("--arch needs at least one GPU arch")
    commit, tag = upstream.resolve(args.ref)
    available: dict[str, list[str]] = {}
    untracked: list[dict] = []
    for arch in archs:
        entries = upstream.subdirs(commit, f"hsa/{arch}")
        if not entries:
            raise SyncError(f"AITER {args.ref} has no hsa/{arch}/")
        available[arch], others = forward_modules(entries)
        if not available[arch]:
            raise SyncError(
                f"AITER {args.ref} ships no tracked forward ASM module for {arch}"
            )
        untracked += [
            {
                "arch": arch,
                "module": name,
                "reason": UNTRACKED_MODULES.get(
                    name, "unknown forward module: decide whether to track it"
                ),
            }
            for name in others
        ]
    pins = load_pins(kernels)
    modules = [module_key(a, m) for a in archs for m in available[a]]
    rules_by_key: dict[str, list[dict]] = {k: [] for k in modules}
    includes_by_key: dict[str, set[str]] = {k: set() for k in modules}
    for key in modules:
        entry = pins["modules"].get(key)
        if entry:
            rules_by_key[key] = list(entry["exclude"])
            includes_by_key[key] = set(entry.get("include", []))

    def planned(option: str, key: str) -> str:
        if key not in rules_by_key:
            raise SyncError(f"{option} {key}: not one of the planned modules {modules}")
        return key

    for key, spec in args.drop_exclude or []:
        planned("--drop-exclude", key)
        match = parse_rule(spec, "-")["match"]
        kept = [rule for rule in rules_by_key[key] if rule["match"] != match]
        if len(kept) == len(rules_by_key[key]):
            raise SyncError(f"--drop-exclude {key}: no pinned rule matches {spec}")
        rules_by_key[key] = kept
    for key, spec, reason in args.exclude or []:
        planned("--exclude", key)
        rule = parse_rule(spec, reason)
        rules_by_key[key] = [
            r for r in rules_by_key[key] if r["match"] != rule["match"]
        ]
        rules_by_key[key].append(rule)
    for key, knl in args.include or []:
        includes_by_key[planned("--include", key)].add(knl)

    module_dirs = [f"hsa/{k}" for k in modules]
    refs = reference_files([split_key(k)[1] for k in modules])
    target_all = upstream.materialize(commit, module_dirs, refs)

    # Two baselines per module. The object baseline is the commit the vendored
    # objects came from. The host-code (ABI) baseline is the commit whose
    # argument structs, packing, grid and dispatch the engine mirrors; it lags
    # the object baseline until the reference diffs are ported (mark-ported).
    baseline_commit: dict[str, str] = {}
    baseline_source: dict[str, str] = {}
    abi_commit: dict[str, str] = {}
    explicit_baseline = upstream.resolve(args.baseline)[0] if args.baseline else None
    for key in modules:
        arch, module = split_key(key)
        entry = pins["modules"].get(key)
        if entry:
            baseline_commit[key], baseline_source[key] = entry["commit"], "pin"
            if entry.get("abi_commit"):
                abi_commit[key] = entry["abi_commit"]
        elif explicit_baseline and (kernels / arch / module).is_dir():
            baseline_commit[key], baseline_source[key] = explicit_baseline, "--baseline"
            abi_commit[key] = explicit_baseline
        if key not in abi_commit:
            # The engine launches every arch of a module with one copy of the host
            # code, so a module without its own host-code commit takes the one
            # already vendored archs record; the oldest, if they differ, so that no
            # change since any of them is missed.
            shared = {
                entry["abi_commit"]
                for other, entry in pins["modules"].items()
                if split_key(other)[1] == module and entry.get("abi_commit")
            }
            if shared:
                abi_commit[key] = min(
                    shared,
                    key=lambda c: int(
                        run_git(upstream.repo, "log", "-1", "--format=%ct", c)
                    ),
                )
    baseline_files: dict[str, dict[str, bytes]] = {}
    for base in sorted(set(baseline_commit.values()) | set(abi_commit.values())):
        baseline_files[base] = upstream.materialize(base, module_dirs, refs)

    cmake_text = (root / ENGINE_CMAKE).read_text(encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "upstream": upstream.url,
        "ref": args.ref,
        "tag": tag,
        "commit": commit,
        "archs": archs,
        "untracked_upstream_modules": untracked,
        "modules": {},
    }
    planned: dict[str, tuple[Findings, dict]] = {}
    for key in modules:
        arch, module = split_key(key)
        target_files = subtree(target_all, f"hsa/{key}")
        target_references = {
            p: target_all[p].decode("utf-8", "replace") for p in refs if p in target_all
        }
        baseline = None
        if key in baseline_commit:
            files = baseline_files[baseline_commit[key]]
            baseline = {
                "commit": baseline_commit[key],
                "files": subtree(files, f"hsa/{key}"),
                # Only a module's first update infers its exclusions from what
                # the tree lacks; afterwards the pinned rules are the record.
                "infer_exclusions": baseline_source[key] == "--baseline",
            }
        abi_baseline = None
        if key in abi_commit:
            files = baseline_files[abi_commit[key]]
            abi_baseline = {
                "commit": abi_commit[key],
                "references": {
                    p: files[p].decode("utf-8", "replace") for p in refs if p in files
                },
            }
        findings, plan = plan_module(
            arch,
            module,
            read_dir(kernels / arch / module),
            any((kernels / arch / m).is_dir() for m in TRACKED_MODULES),
            target_files,
            target_references,
            baseline,
            abi_baseline,
            rules_by_key[key],
            includes_by_key[key],
            cmake_text,
        )
        plan["baseline_source"] = baseline_source.get(key)
        planned[key] = (findings, plan)

    # The vendored codegen merges same-named CSVs of every arch on disk into one
    # registry struct and stops when their headers differ, so check the headers
    # each arch will hold after apply, including archs this plan leaves alone.
    headers: dict[tuple[str, str], dict[str, list[str]]] = {}
    for key, (_, plan) in planned.items():
        arch, module = split_key(key)
        for name, header in plan["csv_headers"].items():
            headers.setdefault((module, cfg_name(name)), {})[arch] = header
    for module in {split_key(key)[1] for key in planned}:
        for arch_dir in sorted(kernels.glob(f"gfx*/{module}")):
            arch = arch_dir.parent.name
            if module_key(arch, module) in planned:
                continue
            tree = module_tree(read_dir(arch_dir), module_key(arch, module))
            for name, table in tree.csvs.items():
                headers.setdefault((module, cfg_name(name)), {})[arch] = table.header
    for key, (findings, plan) in planned.items():
        arch, module = split_key(key)
        for name in sorted(plan["csv_headers"]):
            ours = plan["csv_headers"][name]
            others = headers[(module, cfg_name(name))]
            differ = sorted(a for a, header in others.items() if header != ours)
            if differ:
                findings.add(
                    3,
                    "header_mismatch_across_archs",
                    f"{name}: header {ours} differs from {', '.join(differ)}; the "
                    "vendored codegen.py needs one header per CSV name across archs",
                )
        # The engine holds one copy of a module's mirrored host code (argument
        # struct, packing, grid, dispatch) and launches every arch's objects with
        # it, so porting a reference change re-targets archs this plan leaves
        # alone. Their objects stay built for the host code they are pinned to.
        if any(f["kind"] in HOST_CODE_KINDS for f in findings.items):
            unplanned = sorted(
                arch_dir.parent.name
                for arch_dir in kernels.glob(f"gfx*/{module}")
                if module_key(arch_dir.parent.name, module) not in planned
                and pins["modules"]
                .get(module_key(arch_dir.parent.name, module), {})
                .get("abi_commit")
                != commit
            )
            if unplanned:
                findings.add(
                    3,
                    "host_code_shared_with_unplanned_archs",
                    f"{', '.join(unplanned)} also vendor {module}, and the engine "
                    "launches them with the same host code; porting these reference "
                    "changes changes their launch too: add them to --arch so every "
                    f"arch moves to the new host code together, or leave {key} out "
                    "of this update",
                )
        plan["tier"] = findings.tier
        plan["findings"] = findings.items
        report["modules"][key] = plan
    report["tier"] = max(m["tier"] for m in report["modules"].values())

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_report(report)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"report: {out} (markdown: {out.with_suffix('.md')})")
    return 0


def render_report(report: dict) -> str:
    target = f"{report['tag'] or report['ref']} ({report['commit'][:12]})"
    lines = [
        "# AITER forward ASM sync plan",
        "",
        f"- Upstream: {report['upstream']}",
        f"- Target: {target}",
        f"- Archs: {', '.join(report['archs'])}",
        f"- Highest tier: {report['tier']}",
        "",
        "| Module | Object baseline | Host-code baseline | Tier | Kept objects | Excluded rows |",
        "|---|---|---|---|---|---|",
    ]
    for key, plan in report["modules"].items():
        base = (
            f"{plan['baseline'][:12]} ({plan['baseline_source']})"
            if plan["baseline"]
            else "unknown"
        )
        abi = plan["abi_baseline"][:12] if plan["abi_baseline"] else "unknown"
        objects = sum(1 for p in plan["target"] if p.endswith(".co"))
        lines.append(
            f"| {key} | {base} | {abi} | {plan['tier']} | {objects} "
            f"| {len(plan['excluded_rows'])} |"
        )
    for key, plan in report["modules"].items():
        lines += ["", f"## {key} (tier {plan['tier']})", ""]
        if not plan["findings"]:
            lines.append("- Up to date.")
        for item in sorted(
            plan["findings"], key=lambda f: (-f["tier"], f["kind"], f["detail"])
        ):
            lines.append(f"- [T{item['tier']}] {item['kind']}: {item['detail']}")
        if plan["auto_exclude"]:
            lines.append(
                f"- Excluded because absent locally at the baseline: "
                f"{len(plan['auto_exclude'])} rows (pass --include to keep one)"
            )
        for item in plan["findings"]:
            if item.get("diff"):
                lines += ["", f"```diff\n{item['diff'].rstrip()}\n```"]
    if report["untracked_upstream_modules"]:
        lines += ["", "## Forward modules not ported", ""]
        lines += [
            f"- {u['arch']}/{u['module']}: {u['reason']}"
            for u in report["untracked_upstream_modules"]
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #


def safe_module_dir(kernels: Path, key: str) -> Path:
    arch, module = split_key(key)
    if module not in TRACKED_MODULES or not re.fullmatch(r"gfx[0-9a-z]+", arch):
        raise SyncError(f"refusing to write {key}: not a tracked forward ASM module")
    path = (kernels / arch / module).resolve()
    if kernels.resolve() not in path.parents:
        raise SyncError(f"refusing to write outside {kernels}: {path}")
    return path


def cmd_apply(args: argparse.Namespace, upstream: Upstream, root: Path) -> int:
    kernels = root / KERNELS_DIR
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA:
        raise SyncError(f"{args.report}: unsupported schema")
    if report["upstream"] != upstream.url:
        raise SyncError(f"report is for {report['upstream']}, not {upstream.url}")
    blocking = [
        f"{key}: {f['kind']}: {f['detail']}"
        for key, plan in report["modules"].items()
        for f in plan["findings"]
        if f["kind"] in BLOCKING_KINDS
    ]
    if blocking:
        raise SyncError("the plan has blocking findings:\n  " + "\n  ".join(blocking))
    for key, plan in report["modules"].items():
        current = {
            p: sha256(d) for p, d in read_dir(safe_module_dir(kernels, key)).items()
        }
        if current != plan["local_snapshot"]:
            raise SyncError(f"{key} changed since the plan was made; run plan again")
    commit, _ = upstream.resolve(report["commit"])
    keys = list(report["modules"])
    files = upstream.materialize(commit, [f"hsa/{k}" for k in keys], [])
    pins = load_pins(kernels)
    pins["upstream"] = UPSTREAM_WEB
    for key in keys:
        plan = report["modules"][key]
        arch, module = split_key(key)
        upstream_files = subtree(files, f"hsa/{key}")
        content: dict[str, bytes] = {}
        for path, digest in plan["target"].items():
            if (
                PurePosixPath(path).suffix not in (".csv", ".co")
                or ".." in PurePosixPath(path).parts
            ):
                raise SyncError(
                    f"{key}/{path}: the plan names a file this tool does not write"
                )
            if path.endswith(".csv") and path in upstream_files:
                table = parse_csv(upstream_files[path], f"upstream {key}/{path}")
                dropped = {
                    p["knl_name"] for p in plan["excluded_rows"] if p["csv"] == path
                }
                data = filtered_csv(table, dropped)
            elif path in upstream_files:
                data = upstream_files[path]
            else:
                raise SyncError(f"{key}/{path}: missing upstream; run plan again")
            if sha256(data) != digest:
                raise SyncError(
                    f"{key}/{path}: upstream bytes differ from the plan; run plan again"
                )
            content[path] = data
        target_dir = safe_module_dir(kernels, key)
        for path in read_dir(target_dir):
            if path not in content:
                (target_dir / path).unlink()
        for path, data in content.items():
            destination = (target_dir / path).resolve()
            if target_dir not in destination.parents:
                raise SyncError(
                    f"refusing to write outside {target_dir}: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        for directory in sorted(
            (p for p in target_dir.rglob("*") if p.is_dir()), reverse=True
        ):
            if not any(directory.iterdir()):
                directory.rmdir()
        # The host-code commit advances with the objects only when nothing in the
        # mirrored upstream code changed or went missing; otherwise it stays until
        # mark-ported. A region renamed upstream is missing under its old name;
        # keeping the old commit keeps the moved code in later reports once
        # REFERENCES names it, as a whole added function to compare by hand.
        abi_changed = any(f["kind"] in HOST_CODE_KINDS for f in plan["findings"])
        abi = plan["abi_baseline"]
        entry = {
            "tag": report["tag"],
            "commit": report["commit"],
            "abi_commit": report["commit"] if abi and not abi_changed else abi,
            "upstream_path": plan["upstream_path"],
            "exclude": plan["exclude"],
            "files": plan["target"],
        }
        if plan["include"]:
            entry["include"] = plan["include"]
        pins["modules"][key] = entry
        (target_dir / SOURCE_MD).write_text(
            render_source_md(arch, module, entry), encoding="utf-8"
        )
        if entry["abi_commit"] is None:
            state = "host code not yet ported (run mark-ported once it is)"
        elif entry["abi_commit"] == entry["commit"]:
            state = "host-code mirror matches the objects"
        else:
            state = "host-code mirror lags the objects (run mark-ported once ported)"
        print(
            f"applied {key}: {len(content)} files at {report['commit'][:12]}; {state}"
        )
    write_pins(kernels, pins)
    print(f"updated {KERNELS_DIR / PINS_NAME}; run verify next")
    return 0


def cmd_mark_ported(args: argparse.Namespace, root: Path) -> int:
    """Record that the engine now mirrors the host code of the pinned commit. The
    engine launches every arch of a module with one copy of that code, so the record
    covers every arch that vendors the module, and their objects must all come from
    the same commit."""
    kernels = root / KERNELS_DIR
    pins = load_pins(kernels)
    for key in args.module:
        entry = pins["modules"].get(key)
        if entry is None:
            raise SyncError(f"{key} is not pinned; run plan and apply first")
        _, module = split_key(key)
        sharing = sorted(
            module_key(arch_dir.parent.name, module)
            for arch_dir in kernels.glob(f"gfx*/{module}")
        )
        unpinned = [k for k in sharing if k not in pins["modules"]]
        if unpinned:
            raise SyncError(
                f"{', '.join(unpinned)} share {key}'s host code but are not pinned; "
                "run plan and apply for them first"
            )
        stale = [k for k in sharing if pins["modules"][k]["commit"] != entry["commit"]]
        if stale:
            raise SyncError(
                f"{', '.join(stale)} share {key}'s host code but hold objects from "
                f"another commit; update them to {entry['commit'][:12]} (plan --arch "
                "with every arch, then apply) before marking the host code ported"
            )
        for shared_key in sharing:
            shared = pins["modules"][shared_key]
            shared["abi_commit"] = shared["commit"]
            (safe_module_dir(kernels, shared_key) / SOURCE_MD).write_text(
                render_source_md(split_key(shared_key)[0], module, shared),
                encoding="utf-8",
            )
            print(f"{shared_key}: host-code mirror recorded at {shared['commit'][:12]}")
    write_pins(kernels, pins)
    return 0


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


def verify_module(kernels: Path, key: str, entry: dict) -> list[str]:
    problems: list[str] = []
    arch, module = split_key(key)
    if module not in TRACKED_MODULES:
        problems.append(f"{key}: pinned module is not a tracked forward ASM module")
        return problems
    if entry.get("upstream_path") != f"hsa/{key}":
        problems.append(
            f"{key}: upstream_path {entry.get('upstream_path')} is not hsa/{key}"
        )
    files = read_dir(kernels / arch / module)
    on_disk = {p: sha256(d) for p, d in files.items() if p != SOURCE_MD}
    for path in sorted(set(on_disk) | set(entry["files"])):
        if path not in on_disk:
            problems.append(f"{key}/{path}: in the manifest, missing on disk")
        elif path not in entry["files"]:
            problems.append(f"{key}/{path}: on disk, not in the manifest")
        elif on_disk[path] != entry["files"][path]:
            problems.append(f"{key}/{path}: SHA256 differs from the manifest")
    source = files.get(SOURCE_MD, b"").decode("utf-8")
    if source != render_source_md(arch, module, entry):
        problems.append(
            f"{key}/{SOURCE_MD}: does not match the pin; it is generated, never edited"
        )
    tree = module_tree({p: d for p, d in files.items() if p != SOURCE_MD}, key)
    roots = tree.object_roots()
    referenced: set[str] = set()
    for name, table in tree.csvs.items():
        for row in table.rows:
            if excluded_reason(row, entry["exclude"]) and row[
                "knl_name"
            ] not in entry.get("include", []):
                problems.append(
                    f"{key}/{name}: {row['knl_name']} is kept but an exclusion matches it"
                )
            for root in roots:
                path = str(PurePosixPath(root) / row["co_name"])
                referenced.add(path)
                data = tree.objects.get(path)
                if data is None:
                    problems.append(
                        f"{key}/{name}: {row['knl_name']} names {path}, which is missing"
                    )
                    continue
                problems += [
                    f"{key}/{path}: {message}"
                    for _, message in object_problems(data, row["knl_name"])
                ]
    for path in sorted(set(tree.objects) - referenced):
        problems.append(f"{key}/{path}: no CSV row references it")
    return problems


def cmd_verify(args: argparse.Namespace, root: Path) -> int:
    kernels = root / KERNELS_DIR
    pins = load_pins(kernels)
    wanted = {a.strip() for a in args.arch.split(",")} if args.arch else None
    problems: list[str] = []
    if not (kernels / PINS_NAME).exists():
        problems.append(
            f"{KERNELS_DIR / PINS_NAME} does not exist; the first update creates it"
        )
    for arch_dir in sorted(
        p for p in kernels.iterdir() if p.is_dir() and p.name.startswith("gfx")
    ):
        if wanted and arch_dir.name not in wanted:
            continue
        for module in TRACKED_MODULES:
            key = module_key(arch_dir.name, module)
            if (arch_dir / module).is_dir() and key not in pins["modules"]:
                problems.append(f"{key}: vendored but not pinned")
    for key, entry in sorted(pins["modules"].items()):
        if wanted and split_key(key)[0] not in wanted:
            continue
        problems.extend(verify_module(kernels, key, entry))
    for problem in problems:
        print(f"FAIL {problem}")
    checked = ", ".join(
        sorted(k for k in pins["modules"] if not wanted or split_key(k)[0] in wanted)
    )
    print(
        f"verify: {len(problems)} problem(s); pinned modules checked: {checked or 'none'}"
    )
    return 1 if problems else 0


# --------------------------------------------------------------------------- #
# latest / archs
# --------------------------------------------------------------------------- #


def cmd_latest(upstream: Upstream) -> int:
    tag, source = upstream.latest_tag()
    commit, _ = upstream.resolve(tag)
    print(f"{tag} {commit} ({source})")
    return 0


def cmd_archs(args: argparse.Namespace, upstream: Upstream, root: Path) -> int:
    commit, tag = upstream.resolve(args.ref)
    kernels = root / KERNELS_DIR
    rows = []
    for arch in sorted(upstream.subdirs(commit, "hsa")):
        if not arch.startswith("gfx"):
            continue
        tracked, other = forward_modules(upstream.subdirs(commit, f"hsa/{arch}"))
        if not tracked:
            continue
        local = [m for m in TRACKED_MODULES if (kernels / arch / m).is_dir()]
        rows.append(
            {"arch": arch, "tracked": tracked, "untracked": other, "local": local}
        )
    if args.json:
        print(
            json.dumps(
                {"ref": args.ref, "tag": tag, "commit": commit, "archs": rows}, indent=2
            )
        )
        return 0
    print(
        f"AITER {args.ref} ({commit[:12]}): GPU archs with forward ASM kernels this skill ports"
    )
    for row in rows:
        print(
            f"- {row['arch']}: upstream {', '.join(row['tracked'])}; "
            f"in tree: {', '.join(row['local']) or 'none'}"
            + (
                f"; not ported: {', '.join(row['untracked'])}"
                if row["untracked"]
                else ""
            )
        )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_cache() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "hipdnn-aiter-fwd-sync"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo-root", help="rocm-libraries checkout (default: the current one)"
    )
    parser.add_argument("--upstream", default=UPSTREAM_URL, help="AITER git URL")
    parser.add_argument(
        "--cache",
        type=Path,
        default=default_cache(),
        help="where the partial AITER clone lives",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("latest", help="newest AITER release tag and commit")
    archs = sub.add_parser("archs", help="GPU archs with tracked forward ASM modules")
    archs.add_argument("--ref", required=True)
    archs.add_argument("--json", action="store_true")
    plan = sub.add_parser(
        "plan", help="compare the tree with a ref; writes a report only"
    )
    plan.add_argument("--ref", required=True, help="AITER tag or commit")
    plan.add_argument(
        "--arch", required=True, help="comma-separated GPU archs, e.g. gfx942,gfx950"
    )
    plan.add_argument(
        "--baseline", help="AITER commit the unpinned vendored modules came from"
    )
    plan.add_argument(
        "--report",
        required=True,
        help="report path (.json; a .md is written beside it)",
    )
    plan.add_argument(
        "--exclude",
        nargs=3,
        action="append",
        metavar=("MODULE", "MATCH", "REASON"),
        help="exclude rows: MODULE is arch/module, MATCH is column=v1,v2[;column=v]",
    )
    plan.add_argument(
        "--include",
        nargs=2,
        action="append",
        metavar=("MODULE", "KNL_NAME"),
        help="keep a row that a rule or the baseline would exclude",
    )
    plan.add_argument(
        "--drop-exclude",
        nargs=2,
        action="append",
        metavar=("MODULE", "MATCH"),
        help="remove the pinned exclusion rule whose match is exactly MATCH",
    )
    apply = sub.add_parser("apply", help="apply a reviewed plan report")
    apply.add_argument("--report", required=True)
    ported = sub.add_parser(
        "mark-ported",
        help="record that the engine mirrors the host code of the pinned commit",
    )
    ported.add_argument("--module", action="append", required=True, help="arch/module")
    verify = sub.add_parser("verify", help="check the vendored forward kernels")
    verify.add_argument("--arch", help="limit to these comma-separated archs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = find_repo_root(args.repo_root)
        if args.command == "verify":
            return cmd_verify(args, root)
        if args.command == "mark-ported":
            return cmd_mark_ported(args, root)
        upstream = Upstream(args.upstream, args.cache)
        if args.command == "latest":
            return cmd_latest(upstream)
        if args.command == "archs":
            return cmd_archs(args, upstream, root)
        if args.command == "plan":
            return cmd_plan(args, upstream, root)
        return cmd_apply(args, upstream, root)
    except SyncError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
