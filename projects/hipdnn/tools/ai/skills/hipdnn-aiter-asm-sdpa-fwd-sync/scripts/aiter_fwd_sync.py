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
             record that the engine now mirrors the upstream host code (argument
             structs, packing, grid, dispatch) of a module's pinned commit
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
# not one of AITER's ASM kernels.
CK_SYMBOL = re.compile(r"ck_tile|^_ZN2ck|^_ZNK2ck|^_ZN7ck_tile|kentry")

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


@dataclass(frozen=True)
class Ref:
    """A region of upstream host source that the engine mirrors by hand."""

    path: str
    kind: str  # "file", "struct" or "function"
    name: str | None
    tier: int
    what: str


# The SGPR padding structs every ASM kernel argument struct is built from.
_PADDING_REFS = tuple(
    Ref("csrc/include/aiter_hip_common.h", "struct", name, 3, "kernel argument padding")
    for name in ("p1", "p2", "p3")
)

# Upstream host code read to detect ABI and dispatch changes; never copied.
# In csrc/cpp_itfs/mha_fwd.cu only the ASM functions are listed: fmha_fwd_ck()
# and the CK fallback in mha_fwd() are out of scope.
REFERENCES: dict[str, tuple[Ref, ...]] = {
    "fmha_v3_fwd": _PADDING_REFS
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
            2,
            "ASM dispatch conditions; each early 'return -1' hands the case to CK, "
            "so the engine must decline it",
        ),
    ),
    "fmha_fwd_bf16": _PADDING_REFS
    + (
        Ref(
            "csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu",
            "file",
            None,
            3,
            "host interface: argument struct, checks and launch",
        ),
    ),
    "fmha_fwd_bf16_varlen": _PADDING_REFS
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
UPSTREAM_CODEGEN = "hsa/codegen.py"


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


# --------------------------------------------------------------------------- #
# Module contents
# --------------------------------------------------------------------------- #


@dataclass
class CsvTable:
    header: list[str]
    rows: list[dict[str, str]]
    lines: dict[str, str]  # knl_name -> the row's text, as written upstream
    header_line: str


def parse_csv(data: bytes, where: str) -> CsvTable:
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    content = [
        line
        for line in text.split("\n")
        if line.strip() and not line.lstrip().startswith(COMMENT_PREFIXES)
    ]
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
        lines[row["knl_name"]] = line.rstrip()
    return CsvTable(header, rows, lines, content[0].rstrip())


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

    def list_dir(self, commit: str, path: str) -> list[tuple[str, str]]:
        out = run_git(self.repo, "ls-tree", commit, "--", f"{path.rstrip('/')}/")
        entries = []
        for line in out.splitlines():
            meta, _, name = line.partition("\t")
            entries.append((meta.split()[1], PurePosixPath(name).name))
        return entries

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


def extract_region(text: str, ref: Ref) -> str | None:
    if ref.kind == "file":
        return text
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
        "the upstream rows that no rule below excludes, as written upstream.",
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
    """The upstream table without the dropped kernels, rows kept as written upstream."""
    lines = [table.header_line]
    lines += [
        table.lines[row["knl_name"]]
        for row in table.rows
        if row["knl_name"] not in dropped
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def plan_module(
    arch: str,
    module: str,
    local_files: dict[str, bytes],
    arch_is_local: bool,
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

    if not arch_is_local:
        findings.add(3, "new_arch", f"{arch} has no forward kernels in the tree yet")
    if local is None:
        findings.add(
            3,
            "new_module",
            f"{key} is not vendored yet; read its reference sources in full, since "
            "there is no baseline to diff them against",
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
    kept_objects: set[str] = set()
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
                kept_objects.add(str(PurePosixPath(root) / row["co_name"]))
            if local_csv is None:
                continue
            if knl not in local_rows:
                new = [
                    f"{c}={row[c]}"
                    for c in up_csv.header
                    if c not in NAME_COLUMNS | DATA_COLUMNS
                    and row[c] not in known.get(c, set())
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
    knl_by_object: dict[str, str] = {}
    for name, table in up.csvs.items():
        for row in table.rows:
            for root in up.object_roots():
                knl_by_object[str(PurePosixPath(root) / row["co_name"])] = row[
                    "knl_name"
                ]
    for path in sorted(kept_objects):
        data = up.objects.get(path)
        if data is None:
            findings.add(
                3, "object_missing_upstream", f"{path} is named by a CSV row but absent"
            )
            continue
        try:
            symbols = elf_symbols(data)
        except NotElf as error:
            findings.add(3, "not_elf", f"{path}: {error}")
            continue
        ck = sorted(s for s in symbols if CK_SYMBOL.search(s))
        if ck:
            findings.add(
                3,
                "ck_symbol",
                f"{path}: CK symbols {ck[:3]}; only ASM kernels are ported",
            )
        knl = knl_by_object[path]
        if knl not in symbols and f"{knl}.kd" not in symbols:
            findings.add(3, "symbol_missing", f"{path} does not define {knl}")
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
            "no host-code baseline (no pinned abi_commit and no --baseline): upstream ABI "
            "and dispatch changes cannot be diffed; review the reference sources by hand",
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
        "target": {path: sha256(data) for path, data in sorted(target.items())},
        "local_snapshot": {
            path: sha256(data) for path, data in sorted(local_files.items())
        },
    }
    return findings, plan


def reference_files(modules: list[str]) -> list[str]:
    return sorted({ref.path for module in modules for ref in REFERENCES[module]})


def cmd_plan(args: argparse.Namespace, upstream: Upstream, root: Path) -> int:
    kernels = root / KERNELS_DIR
    archs = [a.strip() for a in args.arch.split(",") if a.strip()]
    if not archs:
        raise SyncError("--arch needs at least one GPU arch")
    commit, tag = upstream.resolve(args.ref)
    available: dict[str, list[str]] = {}
    untracked: list[dict] = []
    for arch in archs:
        entries = {
            name
            for kind, name in upstream.list_dir(commit, f"hsa/{arch}")
            if kind == "tree"
        }
        if not entries:
            raise SyncError(f"AITER {args.ref} has no hsa/{arch}/")
        available[arch] = [m for m in TRACKED_MODULES if m in entries]
        if not available[arch]:
            raise SyncError(
                f"AITER {args.ref} ships no tracked forward ASM module for {arch}"
            )
        for name in sorted(entries):
            if (
                name.startswith("fmha")
                and "fwd" in name
                and name not in TRACKED_MODULES
            ):
                untracked.append(
                    {
                        "arch": arch,
                        "module": name,
                        "reason": UNTRACKED_MODULES.get(
                            name, "unknown forward module: decide whether to track it"
                        ),
                    }
                )
    pins = load_pins(kernels)
    modules = [module_key(a, m) for a in archs for m in available[a]]
    rules_by_key: dict[str, list[dict]] = {k: [] for k in modules}
    includes_by_key: dict[str, set[str]] = {k: set() for k in modules}
    for key in modules:
        entry = pins["modules"].get(key)
        if entry:
            rules_by_key[key] = list(entry["exclude"])
            includes_by_key[key] = set(entry.get("include", []))
    for key, spec in args.drop_exclude or []:
        if key not in rules_by_key:
            raise SyncError(
                f"--drop-exclude {key}: not one of the planned modules {modules}"
            )
        match = parse_rule(spec, "-")["match"]
        kept = [rule for rule in rules_by_key[key] if rule["match"] != match]
        if len(kept) == len(rules_by_key[key]):
            raise SyncError(f"--drop-exclude {key}: no pinned rule matches {spec}")
        rules_by_key[key] = kept
    for key, spec, reason in args.exclude or []:
        if key not in rules_by_key:
            raise SyncError(
                f"--exclude {key}: not one of the planned modules {modules}"
            )
        rule = parse_rule(spec, reason)
        rules_by_key[key] = [
            r for r in rules_by_key[key] if r["match"] != rule["match"]
        ]
        rules_by_key[key].append(rule)
    for key, knl in args.include or []:
        if key not in includes_by_key:
            raise SyncError(
                f"--include {key}: not one of the planned modules {modules}"
            )
        includes_by_key[key].add(knl)

    module_dirs = [f"hsa/{k}" for k in modules]
    refs = reference_files([split_key(k)[1] for k in modules])
    target_all = upstream.materialize(commit, module_dirs, refs + [UPSTREAM_CODEGEN])

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
    baseline_files: dict[str, dict[str, bytes]] = {}
    for base in sorted(set(baseline_commit.values()) | set(abi_commit.values())):
        baseline_files[base] = upstream.materialize(
            base, module_dirs, refs + [UPSTREAM_CODEGEN]
        )

    cmake_text = (root / ENGINE_CMAKE).read_text(encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "upstream": upstream.url,
        "ref": args.ref,
        "tag": tag,
        "commit": commit,
        "archs": archs,
        "untracked_upstream_modules": untracked,
        "global_findings": [],
        "modules": {},
    }
    global_findings = Findings()
    for base in sorted(set(baseline_commit.values())):
        if baseline_files[base].get(UPSTREAM_CODEGEN) != target_all.get(
            UPSTREAM_CODEGEN
        ):
            global_findings.add(
                2,
                "codegen_changed",
                f"{UPSTREAM_CODEGEN} changed since {base[:12]}; the vendored "
                "asm_kernels/codegen.py is a derived copy, so review the change",
            )
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
            (kernels / arch).is_dir(),
            target_files,
            target_references,
            baseline,
            abi_baseline,
            rules_by_key[key],
            includes_by_key[key],
            cmake_text,
        )
        plan["baseline_source"] = baseline_source.get(key)
        plan["tier"] = findings.tier
        plan["findings"] = findings.items
        report["modules"][key] = plan
    report["global_findings"] = global_findings.items
    report["tier"] = max(
        [global_findings.tier] + [m["tier"] for m in report["modules"].values()]
    )

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
    if report["global_findings"]:
        lines += ["", "## All modules", ""]
        lines += [
            f"- [T{f['tier']}] {f['kind']}: {f['detail']}"
            for f in report["global_findings"]
        ]
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
        # mirrored upstream code changed; otherwise it stays until mark-ported.
        abi_changed = any(f["kind"] == "reference_changed" for f in plan["findings"])
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
    """Record that the engine now mirrors the host code of the pinned commit."""
    kernels = root / KERNELS_DIR
    pins = load_pins(kernels)
    for key in args.module:
        entry = pins["modules"].get(key)
        if entry is None:
            raise SyncError(f"{key} is not pinned; run update first")
        arch, module = split_key(key)
        entry["abi_commit"] = entry["commit"]
        (safe_module_dir(kernels, key) / SOURCE_MD).write_text(
            render_source_md(arch, module, entry), encoding="utf-8"
        )
        print(f"{key}: host-code mirror recorded at {entry['commit'][:12]}")
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
                try:
                    symbols = elf_symbols(data)
                except NotElf as error:
                    problems.append(f"{key}/{path}: {error}")
                    continue
                if (
                    row["knl_name"] not in symbols
                    and f"{row['knl_name']}.kd" not in symbols
                ):
                    problems.append(f"{key}/{path}: does not define {row['knl_name']}")
                ck = sorted(s for s in symbols if CK_SYMBOL.search(s))
                if ck:
                    problems.append(
                        f"{key}/{path}: CK symbols {ck[:3]}; only ASM kernels belong here"
                    )
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
    for kind, arch in upstream.list_dir(commit, "hsa"):
        if kind != "tree" or not arch.startswith("gfx"):
            continue
        modules = {
            name for k, name in upstream.list_dir(commit, f"hsa/{arch}") if k == "tree"
        }
        tracked = [m for m in TRACKED_MODULES if m in modules]
        if not tracked:
            continue
        other = sorted(
            m
            for m in modules
            if m.startswith("fmha") and "fwd" in m and m not in TRACKED_MODULES
        )
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
