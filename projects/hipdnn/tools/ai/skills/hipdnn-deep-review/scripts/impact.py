#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Runtime-integration impact of a git diff for hipDNN's kernel ingestor (kernel ABI, dispatch).

Usage: impact.py <files|kernargs|symbols|surfaces|data|report> --repo DIR --base REV --head REV
                 [--target-tip REV] [--json]
       impact.py <subcommand> --round-dir DIR [--json]
       impact.py report ... --emit FILE.json

Emits facts, never severity beyond the row statuses and the rollup line. Reads git objects only
(cat-file, diff, ls-tree, grep, merge-tree); no ref, index or worktree is written
(`merge-tree --write-tree` writes the merged trees to the object store). Exit 0 when the analysis
ran, whatever it found; exit 1 with one line `impact: <reason>` on stderr when it could not.

Views. Each fact is computed at base -> head and, when a target tip differs from base, also at the
merged tree `git merge-tree --write-tree <tip> <head>` (before = the tip, after = the merge). A
conflict skips the merged view and says why; the skip is `changed` when the change touches the
surface or the conflict does. A row that holds only in the merged view is marked
`merged-only`: that is the race class (a parallel PR landed a copy the diff cannot see).
`--round-dir` reads PIN.md (CLONE, BASE, HEAD, TARGET_TIP); a tip equal to BASE gives no merged
view.

Row status: mismatch (producer and a copy disagree, a removed symbol still has a reader), changed
(a contract side changed, or static analysis cannot settle it; a surface file not understood is
always changed), inert (touched, provably without runtime effect), outside (not on the surface;
counted, never dropped). Test files and CMake are outside for status but are still searched as
kernarg copies and as symbol readers. CMake and `requirements*.txt` belong to the BUILD and DEPS
lanes; the rollup names how many it left to them.

Rollup (last line of `report`):
  Runtime impact: BREAKS | NEEDS-DEVICE <archs> | NONE (<n> files inspected[; not judged here: ...])
                  | N/A (no surface file[; not judged here: ...])

`report --emit FILE.json` folds in impact_emit.py output (schema 2), which must have been made for
the same base and head trees: changed params vs a static C++ list is a mismatch; changed IR is
`changed` for that arch; when every family is equal (nothing changed, nothing failed, nothing not
inspected) the rows that were `changed` only because a shipped module changed become `inert`, but
only on modules a family with built variants imports. Static kernarg mismatch rows are never
downgraded.
"""

import argparse
import ast
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reviewlib as rl  # noqa: E402

# -- Tables -------------------------------------------------------------------

HKP = "dnn-providers/hip-kernel-provider/"
ROCKE_LIB = HKP + "rocke/library/"
#: Where an import of a rocKE module may resolve.
IMPORT_ROOTS = (ROCKE_LIB, HKP + "rocke/platform/python/")
ENGINE = HKP + "src/engines/kernel_ingestor_engine/"
DESC_DIR = ENGINE + "descriptors/"
PACKER = HKP + "descriptor-packaging/"
GEN = "projects/hipdnn/tools/IngestorGenerator/"
SDK = "projects/hipdnn/plugin_sdk/"
SDK_INGESTOR = SDK + "include/hipdnn_plugin_sdk/ingestor/"
GENERIC_TARGETS_JSON = SDK + "data/gpu_generic_targets.json"
#: Pathspec of every listing and grep: where producers, copies and readers can live.
SEARCH = (HKP, GEN, SDK, "dnn-providers/integration-tests/")
#: Python files here that import rocKE modules are roots of the consumed closure.
IMPORTER_DIRS = (PACKER, ENGINE, GEN)
#: A shipped build's compile path starts here: the production descriptors and the packer.
PRODUCTION_DIRS = (DESC_DIR, PACKER + "python/", PACKER + "tools/")
DESCRIPTOR_SUFFIXES = tuple(
    f".{kind}.json" for kind in ("kdp", "ukd", "kmd", "ued", "umd", "uhd", "udd")
)
ALL_ARCHS = "all-ingestor-archs"

_CARVE = (
    ("asm_sdpa_engine", "ASM_SDPA_ENGINE"),
    ("hip_mlops_engine", "HIP_MLOPS_ENGINE"),
    ("hip_flash2_engine", "FLASH2"),
)
#: Path -> surface tag, first match wins. A tag in OUTSIDE_TAGS is outside the surface for
#: status; every other tag is a surface. Unmatched paths are `other` (outside). The ASM SDPA,
#: HIP MLOps and flash2 engines, their tests, configs and test-category lists are carved out;
#: test_sdk, frontend, data_sdk and other providers fall through to `other`.
SURFACE_TABLE = (
    *[("carve-out", HKP + f"src/*/{d}/*") for d, _ in _CARVE],
    *[("carve-out", HKP + f"src/*/*/{d}/*") for d, _ in _CARVE],
    *[("carve-out", HKP + f"config/*{k}*") for _, k in _CARVE],
    *[("carve-out", HKP + f"*{k}*_test_categories*.yaml") for _, k in _CARVE],
    ("test", SDK + "tests/*"),
    ("test", HKP + "src/tests/*"),
    ("test", HKP + "src/integration_tests/*"),
    ("test", PACKER + "tests/*"),
    ("test", PACKER + "examples/*"),
    ("test", HKP + "rocke/*/tests/*"),
    ("test", ENGINE + "test_descriptors/*"),
    ("test", GEN + "tests/*"),
    ("test", "dnn-providers/integration-tests/*"),
    ("sdk-ingestor", SDK_INGESTOR + "*"),
    ("generic-targets", GENERIC_TARGETS_JSON),
    ("generic-targets", SDK + "include/hipdnn_plugin_sdk/GpuGenericTargets*"),
    ("descriptor", DESC_DIR + "*"),
    ("native-pack", ENGINE + "*"),
    ("native-pack", HKP + "src/core/Container.cpp"),
    ("kpack-runtime", HKP + "src/compilation/Kpack*"),
    ("packer", PACKER + "*"),
    ("rocke", HKP + "rocke/*"),
    ("generator", GEN + "*"),
    ("provider", HKP + "*"),
)
OUTSIDE_TAGS = {"carve-out", "test", "cmake", "deps", "other"}
#: Outside tags another deep-review lane judges; the rollup names them so N/A is not read
#: as "no impact" for these files.
OWNER_LANES = {"cmake": "BUILD", "deps": "DEPS"}
#: Tags whose files are read as copies or readers although they are outside for status.
SEARCHED_TAGS = {"test"}

GENERATED = (
    (
        "*/rocke/*/golden/*_sha256.json",
        (HKP + "rocke/library/kernels/", HKP + "rocke/platform/"),
    ),
    ("*_generated.h", (GEN, "projects/hipdnn/flatbuffers_sdk/")),
    ("*/hkp-packed.marker", (PACKER + "python/hkp_pack/",)),
    ("*.provenance.json.gz", (PACKER + "python/hkp_pack/",)),
    ("*.kpack", (PACKER + "python/hkp_pack/",)),
)
DOC_SUFFIXES = (".md", ".rst", ".txt")
C_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inl",
    ".hip",
    ".cu",
    ".cuh",
)
COPY_SUFFIXES = (".py", ".json") + C_SUFFIXES
SEVERITY = {"outside": 0, "inert": 1, "changed": 2, "mismatch": 3}
#: Classes of change that a counterpart can follow (a comment or whitespace edit does not move it).
MOVING = {"code", "generated", "dvc-pointer", "binary"}
CONDITIONAL_NODES = (
    ast.If,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
ZERO_SHA = "0" * 40
GIT_CONFIG = (
    "-c", "core.quotepath=off",
    "-c", "core.autocrlf=false",
    "-c", "diff.noprefix=false",
    "-c", "diff.mnemonicPrefix=false",
    "-c", "grep.column=false",
)  # fmt: skip


# -- git access ----------------------------------------------------------------


def git(repo, *args, stdin=None, ok=(0,)) -> bytes:
    """Run git on `repo` immune to user config; raise Fail on a status not in `ok`."""
    done = subprocess.run(
        ["git", "-C", str(repo), *GIT_CONFIG, *args],
        input=stdin,
        capture_output=True,
        check=False,
    )
    if done.returncode not in ok:
        err = done.stderr.decode("utf-8", errors="replace").strip()
        raise rl.Fail(f"git {args[0]}: {err or 'exit ' + str(done.returncode)}")
    return done.stdout


def resolve_revs(repo, revs) -> list:
    """[(commit id, tree id)] of each rev, in one rev-parse."""
    args = [x for r in revs for x in (f"{r}^{{commit}}", f"{r}^{{tree}}")]
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", *args], capture_output=True, check=False
    )
    lines = done.stdout.decode("utf-8", errors="replace").split()
    if done.returncode != 0 or len(lines) != len(args):
        raise rl.Fail(f"cannot resolve {', '.join(revs)} to a commit")
    return list(zip(lines[0::2], lines[1::2]))


def decode(data):
    if data is None or b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


class Tree:
    """One side of a view: a tree object. Blobs, listings, parses and greps are cached."""

    def __init__(self, repo, oid):
        self.repo, self.oid = repo, oid
        self.blobs, self.asts, self.lists, self._under = {}, {}, {}, {}
        self._paths = None
        #: module_index and module_imports caches.
        self.stems, self.imports = None, {}

    @property
    def paths(self) -> set:
        if self._paths is None:
            out = git(
                self.repo, "ls-tree", "-r", "-z", "--name-only", self.oid,
                "--", *SEARCH, GENERIC_TARGETS_JSON,
            )  # fmt: skip
            self._paths = {
                p for p in out.decode("utf-8", errors="replace").split("\0") if p
            }
        return self._paths

    def under(self, prefix: str) -> list:
        """Paths below `prefix`; a narrow ls-tree unless the full listing is already cached."""
        if self._paths is not None:
            return [p for p in self._paths if p.startswith(prefix)]
        if prefix not in self._under:
            out = git(
                self.repo, "ls-tree", "-r", "-z", "--name-only", self.oid, "--", prefix
            )
            self._under[prefix] = [
                p for p in out.decode("utf-8", errors="replace").split("\0") if p
            ]
        return self._under[prefix]

    def read(self, paths) -> dict:
        """path -> bytes, None when absent or not a blob; one cat-file for the unseen paths."""
        fetch([(self, p) for p in paths])
        return {p: self.blobs[p] for p in paths}

    def text(self, path):
        return decode(self.read([path])[path])

    def ast(self, path):
        if path not in self.asts:
            text = self.text(path)
            try:
                self.asts[path] = None if text is None else ast.parse(text)
            except (SyntaxError, ValueError):
                self.asts[path] = None
        return self.asts[path]

    def grep(self, patterns, options=()) -> list:
        return grep_trees([self], patterns, options)[self.oid]


def fetch(wanted) -> None:
    """Read the unseen (tree, path) blobs into each tree's cache with one cat-file."""
    todo = [(t, p) for t, p in dict.fromkeys(wanted) if p not in t.blobs]
    if not todo:
        return
    specs = [f"{t.oid}:{p}" for t, p in todo]
    out = git(
        todo[0][0].repo, "cat-file", "--batch", stdin=("\n".join(specs) + "\n").encode()
    )
    pos = 0
    for tree, path in todo:
        end = out.index(b"\n", pos)
        header = out[pos:end].decode("utf-8", errors="replace")
        pos = end + 1
        parts = header.rsplit(" ", 2)
        if len(parts) != 3 or not parts[2].isdigit():
            tree.blobs[path] = None
            continue
        size = int(parts[2])
        tree.blobs[path] = out[pos : pos + size] if parts[1] == "blob" else None
        pos += size + 1


def grep_trees(trees, patterns, options=()) -> dict:
    """Batched `git grep` over SEARCH in all `trees` at once: {oid: [(path, line, text)]}, or
    [(path, 0, "")] entries with -l."""
    hits = {t.oid: [] for t in trees}
    listing = "-l" in options
    for i in range(0, len(patterns), 100):
        args = ["grep", "-I", "--no-color", "--no-column", *options]
        if not listing:
            args.append("-n")
        for pattern in patterns[i : i + 100]:
            args += ["-e", pattern]
        out = git(trees[0].repo, *args, *hits, "--", *SEARCH, ok=(0, 1))
        for raw in out.decode("utf-8", errors="replace").splitlines():
            oid, rest = raw.split(":", 1)
            if listing:
                hits[oid].append((rest, 0, ""))
            else:
                path, line, text = rest.split(":", 2)
                hits[oid].append((path, int(line), text))
    return hits


@dataclass
class Change:
    path: str
    blob_base: str | None
    blob_head: str | None
    submodule: bool = False
    #: (base_start, base_len, head_start, head_len), git -U0 semantics.
    hunks: list = field(default_factory=list)
    added: set = field(default_factory=set)
    base: bytes | None = None
    head: bytes | None = None
    tag: str = "other"
    cls: str = ""
    funcs: list = field(default_factory=list)


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_raw(out: str) -> list:
    """Changes from the `--raw` lines (`:modeA modeB shaA shaB S<TAB>path`) of a diff."""
    changes = []
    for line in out.splitlines():
        if not line.startswith(":") or "\t" not in line:
            continue
        meta, path = line[1:].split("\t", 1)
        mode_a, mode_b, sha_a, sha_b, _status = meta.split(" ")
        changes.append(
            Change(
                path=path,
                blob_base=None if sha_a == ZERO_SHA else sha_a,
                blob_head=None if sha_b == ZERO_SHA else sha_b,
                submodule="160000" in (mode_a, mode_b),
            )
        )
    return changes


def parse_hunks(out: bytes) -> dict:
    hunks, path, in_hunk = {}, None, False
    for line in out.decode("utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            path, in_hunk = None, False
        elif not in_hunk and line.startswith(("--- a/", "+++ b/")):
            path = line[6:].rstrip("\t")
        elif line.startswith("@@") and path is not None:
            match = _HUNK_RE.match(line)
            if match:
                a, b, c, d = match.groups()
                hunks.setdefault(path, []).append(
                    (
                        int(a),
                        1 if b is None else int(b),
                        int(c),
                        1 if d is None else int(d),
                    )
                )
                in_hunk = True
    return hunks


def collect_changes(before: Tree, after: Tree) -> list:
    """Changes before -> after with their hunks: one `git diff` and one `cat-file`."""
    out = git(
        before.repo, "diff", "-U0", "--raw", "--patch", "--no-abbrev", "--no-renames",
        "--src-prefix=a/", "--dst-prefix=b/", "--no-color", "--no-ext-diff", "--no-textconv",
        before.oid, after.oid,
    )  # fmt: skip
    text = out.decode("utf-8", errors="replace")
    changes = parse_raw(text)
    hunks = parse_hunks(out)
    for ch in changes:
        ch.tag = tag_of(ch.path)
    readable = [c.path for c in changes if not c.submodule and c.tag != "other"]
    fetch([(t, p) for p in readable for t in (before, after)])
    for ch in changes:
        ch.base, ch.head = before.blobs.get(ch.path), after.blobs.get(ch.path)
        ch.hunks = hunks.get(ch.path, [])
        ch.added = {h[2] + i for h in ch.hunks for i in range(h[3])}
        ch.cls = "unread" if ch.submodule or ch.tag == "other" else classify(ch)
        if ch.cls == "code" and ch.tag not in ("other", "carve-out"):
            ch.funcs = touched_functions(ch, before, after)
    return sorted(changes, key=lambda c: c.path)


def merged_tree(repo, tip: str, head: str):
    """(tree oid, None) of the clean merge of tip and head, or (None, reason)."""
    done = subprocess.run(
        ["git", "-C", str(repo), *GIT_CONFIG, "merge-tree", "--write-tree", "--name-only", tip, head],
        capture_output=True,
        check=False,
    )  # fmt: skip
    lines = done.stdout.decode("utf-8", errors="replace").splitlines()
    if done.returncode == 0 and lines:
        return lines[0].strip(), None
    if done.returncode == 1 and lines:
        files = []
        for line in lines[1:]:
            if not line.strip():
                break
            files.append(line.strip())
        return None, "merge conflict in " + (
            ", ".join(files) if files else "unknown files"
        )
    err = done.stderr.decode("utf-8", errors="replace").strip()
    return None, f"git merge-tree failed: {err or 'exit ' + str(done.returncode)}"


# -- Surface tags and content classes ------------------------------------------------


def is_cmake(path: str) -> bool:
    return PurePosixPath(path).name == "CMakeLists.txt" or path.endswith(".cmake")


def is_requirements(path: str) -> bool:
    return re.fullmatch(r"requirements.*\.txt", PurePosixPath(path).name) is not None


def is_doc(path: str) -> bool:
    name = PurePosixPath(path).name
    return (
        path.endswith(DOC_SUFFIXES) and not is_cmake(path) and not is_requirements(path)
    ) or name == ".gitignore"


def tag_of(path: str) -> str:
    if path.startswith(SEARCH) or path.startswith("projects/hipdnn/"):
        if is_cmake(path):
            return "cmake"
        if is_requirements(path):
            return "deps"
    for tag, pattern in SURFACE_TABLE:
        if fnmatch.fnmatchcase(path, pattern):
            return tag
    return "other"


# gfx950, gfx90a, gfx1151, gfx11-generic; in `Gfx950AttentionDense` the `A` starts a word.
_ARCH = re.compile(r"gfx(?:\d+-generic|\d+(?:[a-f](?![a-z]))?)", re.IGNORECASE)


def arch_of(*texts):
    for text in texts:
        match = _ARCH.search(text or "")
        if match:
            return match.group(0).lower()
    return None


def generated_rule(path: str):
    for pattern, sources in GENERATED:
        if fnmatch.fnmatchcase(path, pattern):
            return sources
    return None


#: A C++ raw string opener with any encoding prefix: `R"x(`, `u8R"(`, `LR"(`, `uR"(`, `UR"(`.
_RAW = re.compile(r'(?:u8|u|U|L)?R"([^()\\ ]{0,16})\(')


def raw_string_end(text: str, i: int):
    """End offset of the raw string literal starting at `i`, or None when none starts there."""
    if text[i] not in "RuUL" or (i and (text[i - 1].isalnum() or text[i - 1] == "_")):
        return None
    match = _RAW.match(text, i)
    if not match:
        return None
    close = text.find(")" + match.group(1) + '"', match.end())
    return len(text) if close < 0 else close + len(match.group(1)) + 2


def mask_cpp(text: str, keep_strings: bool = False) -> str:
    """Comments become spaces and literal contents `x`; length and newlines are preserved."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        end = raw_string_end(text, i)
        if end is not None:
            seg = text[i:end]
            out.append(seg if keep_strings else re.sub(r"[^\n]", "x", seg))
            i = end
        elif ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            seg = text[i : min(j + 1, n)]
            if keep_strings or len(seg) < 2:
                out.append(seg)
            else:
                out.append(seg[0] + re.sub(r"[^\n]", "x", seg[1:-1]) + seg[-1])
            i += len(seg)
        elif text.startswith("//", i):
            # A backslash before the newline splices the next line into the comment.
            j = text.find("\n", i)
            while j > 0 and text[i:j].rstrip("\r").endswith("\\"):
                j = text.find("\n", j + 1)
            j = n if j < 0 else j
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def strip_c_comments(text: str) -> str:
    return mask_cpp(text, keep_strings=True)


def strip_hash_comments(text: str) -> str:
    """Drop `#` comments outside double quotes, keeping the line structure."""
    lines = []
    for line in text.split("\n"):
        quoted = False
        for i, ch in enumerate(line):
            if ch == '"' and (i == 0 or line[i - 1] != "\\"):
                quoted = not quoted
            elif ch == "#" and not quoted:
                line = line[:i]
                break
        lines.append(line)
    return "\n".join(lines)


def collapse_space(text: str, quotes: str, raw: bool = False) -> str:
    """Whitespace runs outside string literals become one space, or one newline when the run
    holds one. Literals stay verbatim and tokens never join (`"attn fwd"` != `"attnfwd"`);
    `raw` keeps C++ raw strings verbatim too, across lines.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        end = raw_string_end(text, i) if raw else None
        if end is not None:
            out.append(text[i:end])
            i = end
        elif ch in quotes:
            j = i + 1
            while j < n and text[j] != ch and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            out.append("\n" if "\n" in text[i:j] else " ")
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out).strip()


def without_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if body and is_docstring(body[0]):
                node.body = body[1:] or [ast.Pass()]
    return tree


def is_docstring(node) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def python_shape(text: str):
    try:
        return ast.dump(without_docstrings(ast.parse(text)))
    except (SyntaxError, ValueError):
        return None


def load_data(path: str, text: str):
    """Canonical JSON text of the parsed JSON/YAML. `json.dumps` keeps `3` and `3.0` apart,
    and `1` and `true`; None when the text does not parse."""
    try:
        if path.endswith(".json"):
            data = json.loads(text)
        else:
            data = require_yaml().safe_load(text)
        return json.dumps(data, sort_keys=True, default=repr)
    except rl.Fail:
        raise
    except Exception:  # noqa: BLE001 - any parse failure means "not data-equal"
        return None


def require_yaml():
    try:
        import yaml  # noqa: PLC0415 - optional until a YAML file needs parsing
    except ImportError:
        raise rl.Fail(
            "PyYAML is required to read *.yaml (python -m pip install pyyaml)"
        ) from None
    return yaml


def parse_dvc_pointer(text: str) -> dict:
    """Output path -> md5 of a `.dvc` pointer; the format is small enough to read directly."""
    outs, entry, in_outs = [], None, False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace() and not line.startswith("-"):
            in_outs, entry = line.split(":", 1)[0].strip() == "outs", None
            continue
        if not in_outs:
            continue
        body = line.strip()
        if body.startswith("- "):
            entry = {}
            outs.append(entry)
            body = body[2:]
        if entry is not None and ":" in body:
            key, value = body.split(":", 1)
            entry[key.strip()] = value.strip().strip("'\"")
    return {o["path"]: o["md5"] for o in outs if o.get("md5") and o.get("path")}


def classify(ch: Change) -> str:
    """Content class: code, comment-only, whitespace-only, data-equal, docs, generated,
    dvc-pointer or binary. Decided by content, never by path (except docs and generated).
    """
    path = ch.path
    if generated_rule(path):
        return "generated"
    base = "" if ch.base is None else decode(ch.base)
    head = "" if ch.head is None else decode(ch.head)
    if base is None or head is None:
        return "binary"
    if path.endswith(".dvc"):
        return (
            "data-equal"
            if parse_dvc_pointer(base) == parse_dvc_pointer(head)
            else "dvc-pointer"
        )
    if is_doc(path):
        return "docs"
    if base.replace("\r\n", "\n") == head.replace("\r\n", "\n"):
        return "whitespace-only"
    if path.endswith(C_SUFFIXES):
        if collapse_space(base, "\"'", raw=True) == collapse_space(
            head, "\"'", raw=True
        ):
            return "whitespace-only"
        if collapse_space(strip_c_comments(base), "\"'", raw=True) == collapse_space(
            strip_c_comments(head), "\"'", raw=True
        ):
            return "comment-only"
    elif is_cmake(path):
        if collapse_space(strip_hash_comments(base), '"') == collapse_space(
            strip_hash_comments(head), '"'
        ):
            return "comment-only"
    elif path.endswith(".py"):
        shape = python_shape(base)
        if shape is not None and shape == python_shape(head):
            lines = lambda t: [
                ln.strip() for ln in t.splitlines() if ln.strip()
            ]  # noqa: E731
            return "whitespace-only" if lines(base) == lines(head) else "comment-only"
    if (
        path.endswith((".json", ".yaml", ".yml"))
        and ch.base is not None
        and ch.head is not None
    ):
        data = load_data(path, base)
        if data is not None and data == load_data(path, head):
            return "data-equal"
    return "code"


# -- Functions enclosing a hunk ----------------------------------------------------------


def py_scopes(tree) -> list:
    """[(qualname, start, end)] of every def and class."""
    found = []

    def visit(node, qual):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{qual}.{child.name}" if qual else child.name
                found.append((name, child.lineno, child.end_lineno))
                visit(child, name)
            else:
                visit(child, qual)

    if tree is not None:
        visit(tree, "")
    return found


_CTRL = {"if", "for", "while", "switch", "catch", "else", "do", "try", "return"}


def cpp_functions(text: str) -> list:
    """[(qualname, start, end)] of the function definitions of a C++ file, by brace scan: enough
    to name `kernelMatches`, `scoreKernel` or `attentionDenseKernelSignature`. Bodies are skipped,
    namespaces and classes qualify, initializer lists and lambdas at file scope are not functions.
    """
    code = mask_cpp(text)
    out, stack = [], []  # stack entries: (kind, name)
    depth, stmt, line, stmt_line = 0, 0, 1, 1
    start_line = 1
    for i, ch in enumerate(code):
        if ch == "\n":
            line += 1
        if ch == "{":
            if depth:
                depth += 1
                continue
            raw = code[stmt:i].split("\n")
            first = next(
                (
                    k
                    for k, ln in enumerate(raw)
                    if ln.strip() and not ln.lstrip().startswith("#")
                ),
                0,
            )
            header = "\n".join(
                ln for ln in raw if not ln.lstrip().startswith("#")
            ).strip()
            paren = header.find("(")
            scope = re.search(
                r"\b(namespace|class|struct|union|enum)\b\s*(?:class\b\s*)?(\w*)",
                header,
            )
            eq = header.find("=")
            m = re.search(r"([~\w][\w:~]*)\s*\(", header)
            if scope and (paren < 0 or scope.start() < paren):
                stack.append(("scope", scope.group(2)))
            elif m and not (0 <= eq < m.start()) and m.group(1) not in _CTRL:
                name = m.group(1)
                if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
                    close = header.find(")", m.end())
                    name = " ".join(header[m.start() : close + 1].split())
                scopes = [n for k, n in stack if k == "scope" and n]
                depth = 1
                start_line = stmt_line + first
                stack.append(("func", "::".join([*scopes, name])))
            else:
                stack.append(("other", ""))
            stmt, stmt_line = i + 1, line
        elif ch == "}":
            if depth > 1:
                depth -= 1
                continue
            if depth == 1:
                depth = 0
                _, name = stack.pop()
                out.append((name, start_line, line))
            elif stack:
                stack.pop()
            stmt, stmt_line = i + 1, line
        elif ch == ";" and not depth:
            stmt, stmt_line = i + 1, line
    return out


def touched_functions(ch: Change, before: Tree, after: Tree) -> list:
    """Names of the functions a change's hunks fall in, head side and base side."""
    if ch.path.endswith(".py"):
        scopes = {
            "base": py_scopes(before.ast(ch.path)),
            "head": py_scopes(after.ast(ch.path)),
        }
    elif ch.path.endswith(C_SUFFIXES):
        scopes = {
            "base": cpp_functions(decode(ch.base) or ""),
            "head": cpp_functions(decode(ch.head) or ""),
        }
    else:
        return []
    names = set()
    outside = False
    for b_start, b_len, h_start, h_len in ch.hunks:
        for side, (start, length) in (
            ("base", (b_start, b_len)),
            ("head", (h_start, h_len)),
        ):
            lo, hi = start, start + max(length, 1) - 1
            if length == 0 and side == "head":
                continue
            inside = [(q, s, e) for q, s, e in scopes[side] if s <= hi and lo <= e]
            if inside:
                # innermost scopes only: the one with the latest start that contains each line
                names.update(
                    q
                    for q, s, e in inside
                    if not any(q2 != q and q2.startswith(q + ".") for q2, *_ in inside)
                )
            elif length:
                outside = True
    out = sorted(names)
    if outside and not out:
        out = ["<file scope>"]
    return out


# -- Kernel ABI: parameter lists -----------------------------------------------------------


@dataclass
class ParamList:
    lineno: int
    names: list = field(default_factory=list)
    unconditional: list = field(default_factory=list)
    conditional: list = field(default_factory=list)
    undetermined: bool = False
    #: Source text of everything after the name (type, `noalias=`, ...), parallel to `names`.
    decls: list = field(default_factory=list)

    def marked(self) -> list:
        return [n + "?" if c else n for n, c in zip(self.names, self.conditional)]


def param_lists(tree) -> dict:
    """Function qualname -> its `b.param` names in source order, conditional ones marked."""
    result, calls = {}, {}

    def visit(node, qual, function, conditional):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{qual}.{child.name}" if qual else child.name
                result.setdefault(name, ParamList(child.lineno))
                visit(child, name, name, False)
            elif isinstance(child, ast.ClassDef):
                visit(
                    child, f"{qual}.{child.name}" if qual else child.name, None, False
                )
            else:
                if (
                    function
                    and isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "param"
                ):
                    calls.setdefault(function, []).append((child, conditional))
                visit(
                    child,
                    qual,
                    function,
                    conditional or isinstance(child, CONDITIONAL_NODES),
                )

    visit(tree, "", None, False)
    for function, found in calls.items():
        info = result[function]
        for call, conditional in sorted(
            found, key=lambda c: (c[0].lineno, c[0].col_offset)
        ):
            first = call.args[0] if call.args else None
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                info.undetermined = True
                continue
            info.names.append(first.value)
            info.conditional.append(conditional)
            info.decls.append(
                ", ".join(
                    [ast.unparse(a) for a in call.args[1:]]
                    + [
                        (
                            f"{k.arg}={ast.unparse(k.value)}"
                            if k.arg
                            else f"**{ast.unparse(k.value)}"
                        )
                        for k in call.keywords
                    ]
                )
            )
            if not conditional:
                info.unconditional.append(first.value)
    return {n: i for n, i in result.items() if i.names or i.undetermined}


def function_nodes(tree) -> dict:
    """Qualname -> FunctionDef for every function, nested ones included."""
    found = {}

    def visit(node, qual):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{qual}.{child.name}" if qual else child.name
                if not isinstance(child, ast.ClassDef):
                    found[name] = child
                visit(child, name)

    if tree is not None:
        visit(tree, "")
    return found


def string_lists(path: str, text: str, tree=None) -> list:
    """Ordered string-literal lists of 2+ items: [(names, start, end, function)]. Python by AST;
    other text by quoted runs inside `{...}` or `[...]`."""
    found = []
    if path.endswith(".py"):
        functions = sorted(
            ((f.lineno, f.end_lineno, q) for q, f in function_nodes(tree).items()),
            reverse=True,
        )
        for node in ast.walk(tree) if tree is not None else []:
            if (
                isinstance(node, (ast.List, ast.Tuple))
                and len(node.elts) > 1
                and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str)
                    for e in node.elts
                )
            ):
                owner = next(
                    (q for s, e, q in functions if s <= node.lineno <= e), None
                )
                found.append(
                    ([e.value for e in node.elts], node.lineno, node.end_lineno, owner)
                )
        return found
    for match in re.finditer(
        r'[\[{]\s*("[^"\n]*"(?:\s*,\s*"[^"\n]*")+)\s*,?\s*[\]}]', strip_c_comments(text)
    ):
        start = text.count("\n", 0, match.start()) + 1
        names = re.findall(r'"([^"\n]*)"', match.group(1))
        found.append((names, start, text.count("\n", 0, match.end()) + 1, None))
    return found


# -- Kernel ABI: static C++ `KernelArgument` lists ----------------------------------------------

_KA_FUNC = re.compile(
    r"vector\s*<\s*(?:[\w:]*::)?KernelArgument\s*>\s*(?:const\s*)?&?\s*(\w+)\s*"
    r"\((?:[^()]|\([^()]*\))*\)\s*(?:const\s*)?(?:noexcept\s*)?\{"
)
_KA_LIT = re.compile(r"KernelArgument\s*\{")
_KA_DYNAMIC = re.compile(r"\b(?:emplace_back|push_back|for|while)\b|\[[&=]\]")
_STR = re.compile(r'"((?:[^"\\]|\\.)*)"')


def split_args(text: str) -> list:
    out, cur, depth, quote = [], [], 0, None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            cur.append(ch)
        elif ch in "({<[":
            depth += 1
            cur.append(ch)
        elif ch in ")}>]":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def match_brace(code: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(code) - 1


_SIZES = {"PTR": 8, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "F64": 8}
_SIZEOF = {"void*": 8, "int32_t": 4, "uint32_t": 4, "float": 4, "int64_t": 8,
           "uint64_t": 8, "double": 8, "int": 4}  # fmt: skip


def cpp_arg_size(arg: str):
    arg = arg.strip()
    if arg in _SIZES:
        return _SIZES[arg]
    if arg.isdigit():
        return int(arg)
    m = re.search(r"sizeof\s*\(\s*([\w:]+\s*\*?)\s*\)", arg)
    return _SIZEOF.get(m.group(1).replace(" ", "")) if m else None


#: Byte size of a rocKE scalar type named in `b.param`; `PtrType(...)` is a pointer.
_BUILDER_SIZES = {"I8": 1, "U8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4,
                  "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}  # fmt: skip


def decl_size(decl: str):
    """Size of the type a `b.param` declaration names first: `ptr`, bytes, or None."""
    args = split_args(decl) if decl else []
    first = args[0].strip() if args else ""
    if first.startswith("PtrType(") or first == "PTR":
        return "ptr"
    return _BUILDER_SIZES.get(first)


def size_text(size) -> str:
    return "pointer" if size == "ptr" else f"{size}-byte"


def name_bindings(statements) -> dict:
    """Name -> source text of the simple `NAME = expr` assignments among `statements`."""
    out = {}
    for node in statements:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            out[node.targets[0].id] = ast.unparse(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out[node.target.id] = ast.unparse(node.value)
    return out


def resolved_decls(tree, qual: str, decls: list) -> list:
    """`decls` with the module's and the function's own `NAME = expr` bindings substituted
    once, so `b.param("scale", SC)` changes when `SC = F16` becomes `SC = I64`."""
    if tree is None:
        return decls
    scope = name_bindings(tree.body)
    fn = function_nodes(tree).get(qual)
    if fn is not None:
        scope.update(name_bindings(fn.body))
    if not scope:
        return decls
    word = re.compile(
        r"\b(" + "|".join(sorted(map(re.escape, scope), key=len, reverse=True)) + r")\b"
    )
    return [word.sub(lambda m: scope[m.group(1)], d) for d in decls]


def cpp_kernel_lists(text: str) -> list:
    """Every `std::vector<KernelArgument>`-returning function: one entry per static list (a
    multi-launch pack holds several, each its own consumer), or one dynamic entry.
    [{func, line, names, sizes, dynamic, unnamed}]; `names` is None unless every entry is a
    `KernelArgument{kind, size, offset, "name"}` literal with a non-empty name."""
    code = mask_cpp(text)
    out = []
    for fn in _KA_FUNC.finditer(code):
        open_at = fn.end() - 1
        close = match_brace(code, open_at)
        body = code[open_at:close]
        line = text.count("\n", 0, fn.start()) + 1
        literals = {m.start(): m for m in _KA_LIT.finditer(code, open_at, close)}
        if _KA_DYNAMIC.search(body) or not literals:
            out.append(dict(func=fn.group(1), line=line, names=None, sizes=None,
                            dynamic=True, unnamed=False))  # fmt: skip
            continue
        groups, stack = {}, []
        for i in range(open_at, close):
            if i in literals:
                end = match_brace(code, literals[i].end() - 1)
                args = split_args(text[literals[i].end() : end])
                groups.setdefault(stack[-1] if stack else open_at, []).append(
                    (args, text.count("\n", 0, i) + 1)
                )
            if code[i] == "{":
                stack.append(i)
            elif code[i] == "}" and stack:
                stack.pop()
        for _, entries in sorted(groups.items()):
            names, sizes = [], []
            for args, _ in entries:
                name = _STR.fullmatch(args[3]) if len(args) >= 4 else None
                names.append(name.group(1) if name else "")
                kind = _STR.fullmatch(args[0]) if args else None
                sizes.append(
                    "ptr" if kind and kind.group(1) == "global_buffer"
                    else cpp_arg_size(args[1]) if len(args) > 1 else None
                )  # fmt: skip
            unnamed = any(not n for n in names)
            out.append(dict(func=fn.group(1), line=entries[0][1], names=None if unnamed else names,
                            sizes=sizes, dynamic=False, unnamed=unnamed))  # fmt: skip
    return out


# -- Copies of a kernarg list ---------------------------------------------------------------


def named_lists(tree: Tree, path: str) -> list:
    """Ordered name lists of a file: [(names, start, end, owner)]."""
    if path in tree.lists:
        return tree.lists[path]
    text, found = tree.text(path), []
    if text is None:
        pass
    elif path.endswith(".py"):
        found = string_lists(path, text, tree.ast(path))
    elif path.endswith(".json"):
        try:
            doc = json.loads(text)
        except ValueError:
            doc = None
        for node in walk_json(doc):
            sig = node.get("signature")
            if isinstance(sig, list) and len(sig) > 1:
                names = [a.get("name") if isinstance(a, dict) else a for a in sig]
                if all(isinstance(n, str) for n in names):
                    line = text.count("\n", 0, max(text.find('"signature"'), 0)) + 1
                    found.append((names, line, line, None))
    elif path.endswith(C_SUFFIXES):
        found = [
            (e["names"], e["line"], e["line"], None)
            for e in cpp_kernel_lists(text)
            if e["names"]
        ]
        found += string_lists(path, text)
    tree.lists[path] = found
    return found


def walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def compatible(cpp_names: list, pl: ParamList) -> bool:
    """A static list matches a builder when it is an ordered subsequence of the builder's
    params (conditional groups may be absent) holding every unconditional one."""
    it = iter(pl.names)
    return all(n in it for n in cpp_names) and all(
        n in cpp_names for n in pl.unconditional
    )


def related(cpp_names: list, names: list) -> bool:
    shared = len(set(cpp_names) & set(names))
    return shared >= 2 and shared * 2 >= len(cpp_names)


def list_diff(old: list, new: list) -> str:
    added = [n for n in new if n not in old]
    removed = [n for n in old if n not in new]
    parts = []
    if added:
        parts.append("added " + ", ".join(added))
    if removed:
        parts.append("removed " + ", ".join(removed))
    if not parts:
        parts.append("reordered")
    elif [n for n in old if n in new] != [n for n in new if n in old]:
        parts.append("reordered")
    return "; ".join(parts)


# -- Declared pairs (engine manifests) ----------------------------------------------------------


@dataclass
class Surface:
    manifest: str
    name: str
    module: str
    function: str
    cpp_mirror: str
    test: str
    arch: str | None


def load_surfaces(tree: Tree) -> list:
    """`launch_surface` entries of every `*.engine.yaml` under the generator's configs."""
    paths = sorted(
        p for p in tree.under(GEN + "configs/") if p.endswith(".engine.yaml")
    )
    if not paths:
        return []
    yaml = require_yaml()
    out = []
    for path, data in tree.read(paths).items():
        text = decode(data)
        try:
            manifest = yaml.safe_load(text) if text else None
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("launch_surface"), list
        ):
            continue
        root = str(manifest.get("provider_root") or HKP).rstrip("/") + "/"
        for entry in manifest["launch_surface"]:
            source = entry.get("python_source") if isinstance(entry, dict) else None
            if not isinstance(source, str) or ":" not in source:
                continue
            module, function = source.rsplit(":", 1)
            out.append(
                Surface(
                    path,
                    str(entry.get("name")),
                    root + "rocke/library/" + module.strip(),
                    function.strip(),
                    str(entry.get("cpp_mirror") or "").split(":", 1)[0].strip(),
                    str(entry.get("test") or "").split(":", 1)[0].strip(),
                    str(manifest.get("arch") or "") or None,
                )
            )
    return out


# -- Consumed closure ----------------------------------------------------------------------------


def module_name(path: str):
    for root in IMPORT_ROOTS:
        if path.startswith(root) and path.endswith(".py"):
            parts = path[len(root) : -3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)
    return None


def resolve_module(parts, known):
    if not parts:
        return None
    for root in IMPORT_ROOTS:
        stem = root + "/".join(parts)
        for candidate in (stem + ".py", stem + "/__init__.py"):
            if candidate in known:
                return candidate
    return None


def resolve_path_literal(value: str, known):
    value = value.replace("\\", "/")
    for root in IMPORT_ROOTS:
        marker = root[len(HKP) :]
        if marker in value:
            candidate = root + value.split(marker, 1)[1]
            if candidate in known:
                return candidate
        if root + value in known:
            return root + value
    return None


def eager_nodes(tree):
    """Every node that runs when the module is imported: all but the bodies of functions."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def import_targets(path: str, tree, known, eager: bool = False) -> list:
    """[(target path, line)] for every rocKE module a file imports; `eager` keeps only the
    imports that run at import time."""
    targets = []
    if tree is None:
        return targets
    own = module_name(path)
    package = own.split(".") if own else []
    if own and not path.endswith("__init__.py"):
        package = package[:-1]
    for node in eager_nodes(tree) if eager else ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(
                    (resolve_module(alias.name.split("."), known), node.lineno)
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if not own:
                    continue
                base = (
                    package[: len(package) - (node.level - 1)]
                    if node.level > 1
                    else list(package)
                )
                base += node.module.split(".") if node.module else []
            else:
                base = node.module.split(".") if node.module else []
            for alias in node.names:
                target = (
                    None
                    if alias.name == "*"
                    else resolve_module(base + [alias.name], known)
                )
                targets.append((target or resolve_module(base, known), node.lineno))
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name in ("spec_from_file_location", "SourceFileLoader", "load_source"):
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.endswith(".py")
                    ):
                        targets.append(
                            (resolve_path_literal(arg.value, known), node.lineno)
                        )
    return [(t, line) for t, line in targets if t]


def ancestor_inits(path: str, known) -> list:
    """The `__init__.py` of every package above `path`: importing the module runs each."""
    for root in IMPORT_ROOTS:
        if path.startswith(root):
            parts = path[len(root) :].split("/")[:-1]
            inits = [
                root + "/".join(parts[:d]) + "/__init__.py"
                for d in range(1, len(parts) + 1)
            ]
            return [i for i in inits if i in known]
    return []


@dataclass
class Closure:
    #: rocKE module -> the root that reaches it (a descriptor, packer or tool import, or an
    #: eager `__init__` import chain; every ancestor `__init__.py` of a consumed module is in).
    consumed: dict = field(default_factory=dict)
    #: module -> archs of the production roots whose ordinary imports reach its body.
    shipped: dict = field(default_factory=dict)
    #: archs whose descriptors sit behind DVC: their kernel modules are roots by directory.
    opaque: set = field(default_factory=set)
    #: module -> [(imported module, line, eager)], ancestor `__init__.py` files included.
    graph: dict = field(default_factory=dict)

    def reach(self, start: str) -> set:
        """Every module that importing `start` loads."""
        seen, stack = set(), [start]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack += [t for t, _, _ in self.graph.get(current, ())]
        return seen


def build_closure(tree: Tree) -> Closure:
    known = {p for p in tree.paths if p.endswith(".py") and p.startswith(IMPORT_ROOTS)}
    out = Closure()
    roots = []  # (label, target, arch, production)
    descriptors = sorted(
        p for p in tree.paths if p.startswith(DESC_DIR) and p.endswith(".json")
    )
    for path, data in tree.read(descriptors).items():
        text = decode(data)
        if not text or "kernel_source" not in text:
            continue
        try:
            doc = json.loads(text)
        except ValueError:
            continue
        seen = set()
        for node in walk_json(doc):
            source = node.get("kernel_source")
            if isinstance(source, dict) and isinstance(source.get("source"), str):
                target = resolve_path_literal(source["source"], known)
                if target and source["source"] not in seen:
                    seen.add(source["source"])
                    match = re.search(
                        r'"source"\s*:\s*"' + re.escape(source["source"]) + '"', text
                    )
                    line = text.count("\n", 0, match.start()) + 1 if match else 0
                    roots.append(
                        (f"{path}:{line}", target, arch_of(path, target), True)
                    )
    for path in sorted(
        p for p in tree.paths if p.startswith(DESC_DIR) and p.endswith(".dvc")
    ):
        arch = arch_of(path)
        if arch:
            out.opaque.add(arch)
            for mod in sorted(known):
                if (
                    mod.startswith(ROCKE_LIB + "kernels/")
                    and arch in mod.split("/")
                    and not mod.endswith("__init__.py")
                ):
                    roots.append((path, mod, arch, True))
    packages = sorted(
        {
            p[len(r) :].split("/")[0].removesuffix(".py")
            for r in IMPORT_ROOTS
            for p in known
            if p.startswith(r)
        }
    )
    mention = re.compile(
        r"\b(?:"
        + "|".join(map(re.escape, packages + ["spec_from_file_location"]))
        + r")\b"
    )
    importers = sorted(
        p for p in tree.paths
        if p.endswith(".py") and p.startswith(IMPORTER_DIRS) and p not in known and tag_of(p) != "test"
    )  # fmt: skip
    for path, data in tree.read(importers).items():
        text = decode(data)
        if text and mention.search(text):
            for target, line in import_targets(path, tree.ast(path), known):
                roots.append(
                    (f"{path}:{line}", target, None, path.startswith(PRODUCTION_DIRS))
                )
    graph, body = out.graph, {}
    frontier = sorted({t for _, t, _, _ in roots})
    while frontier:
        tree.read(frontier)
        for path in frontier:
            is_init = path.endswith("__init__.py")
            imports = import_targets(path, tree.ast(path), known, eager=is_init)
            body[path] = [] if is_init else sorted({t for t, _ in imports})
            graph[path] = [(t, line, is_init) for t, line in imports] + [
                (a, 0, False) for a in ancestor_inits(path, known)
            ]
        frontier = sorted({t for p in frontier for t, _, _ in graph[p]} - graph.keys())
    for label, target, _arch, _prod in roots:
        stack = [(target, label)]
        while stack:
            current, why = stack.pop()
            if current in out.consumed:
                continue
            out.consumed[current] = why
            for nxt, at, eager in graph.get(current, ()):
                stack.append(
                    (
                        nxt,
                        (
                            f"{why}, through the eager import at {current}:{at}"
                            if eager
                            else why
                        ),
                    )
                )
    for _label, target, arch, production in roots:
        if not production:
            continue
        seen, stack = set(), [target]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            archs = out.shipped.setdefault(current, set())
            if arch:
                archs.add(arch)
            stack += body.get(current, [])
    return out


# -- Rows ------------------------------------------------------------------------------------------


def make_row(
    kind, path, line, name, status, reason, counterparts=(), arch=None, foldable=False
):
    return {
        "kind": kind,
        "path": path,
        "line": line,
        "name": name,
        "status": status,
        "reason": reason,
        "counterparts": sorted(
            counterparts, key=lambda c: (c["path"], c["line"], c["moved"])
        ),
        "arch": arch or ALL_ARCHS,
        "foldable": foldable,
        "view": "",
        "merged_only": False,
    }


def cp(path, line, moved):
    return {"path": path, "line": line, "moved": bool(moved)}


class Ctx:
    """One view: before -> after, its changes and lazily built analyses."""

    def __init__(self, before: Tree, after: Tree, label: str):
        self.before, self.after, self.label = before, after, label
        self.changes = {c.path: c for c in collect_changes(before, after)}
        self._closures, self._surfaces = {}, {}

    def closure(self, side) -> Closure:
        if side not in self._closures:
            self._closures[side] = build_closure(
                self.before if side == "before" else self.after
            )
        return self._closures[side]

    def surfaces(self, side) -> list:
        if side not in self._surfaces:
            self._surfaces[side] = load_surfaces(
                self.before if side == "before" else self.after
            )
        return self._surfaces[side]

    def tree_of(self, side) -> Tree:
        return self.before if side == "before" else self.after

    def ast(self, ch: Change, side):
        return self.tree_of(side).ast(ch.path)

    def moved(self, path: str) -> bool:
        ch = self.changes.get(path)
        return ch is not None and ch.cls in MOVING


def func_note(funcs: list) -> str:
    if not funcs:
        return ""
    shown = ", ".join(funcs[:6]) + (
        f", +{len(funcs) - 6} more" if len(funcs) > 6 else ""
    )
    return f" in {shown}"


# -- files -------------------------------------------------------------------------------------------


def surface_changes(base_tree, head_tree) -> list:
    """Defs removed or re-signed and module-level statements changed; body-only edits are []."""

    def surface(tree):
        defs, statements = {}, set()
        functions = (ast.FunctionDef, ast.AsyncFunctionDef)
        sig = lambda n: repr((ast.dump(n.args), [ast.dump(d) for d in n.decorator_list],  # noqa: E731
                              ast.dump(n.returns) if n.returns else None))  # fmt: skip
        for node in tree.body:
            if isinstance(node, functions):
                defs[f"def {node.name}"] = sig(node)
            elif isinstance(node, ast.ClassDef):
                members = [
                    n
                    for n in node.body
                    if not isinstance(n, functions) and not is_docstring(n)
                ]
                defs[f"class {node.name}"] = repr(([ast.dump(b) for b in node.bases],
                    [ast.dump(d) for d in node.decorator_list], [ast.dump(m) for m in members]))  # fmt: skip
                for item in node.body:
                    if isinstance(item, functions):
                        defs[f"def {node.name}.{item.name}"] = sig(item)
            elif not is_docstring(node):
                statements.add(ast.dump(node))
        return defs, statements

    old_defs, old_stmts = surface(base_tree)
    new_defs, new_stmts = surface(head_tree)
    out = [f"{k} removed" if k not in new_defs else f"{k} changed"
           for k, v in old_defs.items() if new_defs.get(k) != v]  # fmt: skip
    if old_stmts != new_stmts:
        out.append("a module-level statement changed")
    return out


def rocke_status(ctx: Ctx, ch: Change):
    """(status, reason, arch, foldable) of a rocKE file whose code changed."""
    path = ch.path
    if not path.endswith(".py") or not path.startswith(IMPORT_ROOTS):
        return (
            "changed",
            "rocKE file that is not an importable Python module: not understood",
            arch_of(path),
            False,
        )
    side = ctx.closure("before" if ch.head is None else "after")
    before = after = side
    archs = set(side.shipped.get(path, ()))
    if path in side.shipped:
        arch = ", ".join(sorted(archs)) if archs else None
        return (
            "changed",
            "module body is reachable from a shipped variant; emitted code or params may differ"
            + (f" ({', '.join(sorted(archs))})" if archs else ""),
            arch or arch_of(path),
            True,
        )
    why = after.consumed.get(path) or before.consumed.get(path)
    if why:
        bt, at = ctx.ast(ch, "before"), ctx.ast(ch, "after")
        if bt is None or at is None:
            return (
                "changed",
                f"imported module added, removed or unparsable ({why})",
                arch_of(path),
                True,
            )
        diff = surface_changes(bt, at)
        if diff:
            return (
                "changed",
                f"imported by {why}; module surface changed: {'; '.join(diff[:3])}",
                arch_of(path),
                True,
            )
        return (
            "inert",
            f"body-only edit; imported by {why} but no shipped body reaches it",
            None,
            False,
        )
    return "inert", "reached by no shipped variant", None, False


def file_row(ctx: Ctx, ch: Change):
    path, tag, cls = ch.path, ch.tag, ch.cls
    kind = "file"
    if tag in OUTSIDE_TAGS:
        return make_row(
            kind,
            path,
            0,
            path,
            "outside",
            (
                f"{tag}: judged by the {OWNER_LANES[tag]} lane, not by this report"
                if tag in OWNER_LANES
                else f"{tag}: not on the runtime-integration surface"
            ),
        )
    if cls == "docs":
        return make_row(kind, path, 0, path, "outside", "documentation")
    if cls in ("comment-only", "whitespace-only", "data-equal"):
        return make_row(kind, path, 0, path, "inert", cls)
    arch = arch_of(path)
    if tag == "descriptor" or path == GENERIC_TARGETS_JSON or cls == "dvc-pointer":
        return None  # data rows carry these files
    if cls == "generated":
        sources = generated_rule(path)
        other = next(
            (
                p
                for p in ctx.changes
                if p != path and p.startswith(sources) and ctx.changes[p].cls in MOVING
            ),
            None,
        )
        if other:
            return make_row(
                kind,
                path,
                0,
                path,
                "inert",
                f"generated artifact regenerated with its source ({other})",
            )
        return make_row(
            kind,
            path,
            0,
            path,
            "changed",
            "generated artifact changed without its source",
            arch=arch,
        )
    if cls == "binary":
        return make_row(
            kind,
            path,
            0,
            path,
            "changed",
            "binary or non-UTF-8 surface file: not understood",
            arch=arch,
        )
    line = ch.hunks[0][2] if ch.hunks else 0
    if tag == "rocke":
        status, reason, arch, fold = rocke_status(ctx, ch)
        return make_row(
            kind,
            path,
            line,
            path,
            status,
            reason + func_note(ch.funcs) if status != "inert" else reason,
            arch=arch,
            foldable=fold,
        )
    arch = arch_of(path, " ".join(ch.funcs))
    reason = {
        "native-pack": "native pack code changed",
        "sdk-ingestor": "ingestor SDK header changed",
        "generic-targets": "generic-target header changed",
        "kpack-runtime": "kpack runtime changed",
        "packer": "packer code changed",
        "generator": "ingestor generator changed",
    }.get(tag, "provider code changed")
    return make_row(
        kind, path, line, path, "changed", reason + func_note(ch.funcs), arch=arch
    )


# -- kernargs ---------------------------------------------------------------------------------------


@dataclass
class Work:
    change: Change
    qual: str
    old: ParamList
    new: ParamList | None
    line: int


def find_copies(ctx: Ctx, work: list) -> dict:
    """(path, qual) -> counterparts holding the old kernarg list, searched in both trees by the
    ordered-name sequence. A copy still holding the old list after is unmoved; one that held it
    before and no longer does moved with the producer."""
    targets = {}
    for w in work:
        for names in (w.old.names, w.old.unconditional):
            if len(names) > 1:
                targets.setdefault(tuple(names), set()).add((w.change.path, w.qual))
    copies = {(w.change.path, w.qual): [] for w in work}
    if not targets:
        return copies
    patterns = [f'"{n}"' for n in sorted({names[0] for names in targets})]
    hits = grep_trees([ctx.before, ctx.after], patterns, ["-F", "-l"])
    candidates = {
        p for found in hits.values() for p, _, _ in found if p.endswith(COPY_SUFFIXES)
    }
    fetch([(t, p) for p in sorted(candidates) for t in (ctx.before, ctx.after)])
    for path in sorted(candidates):
        held = {}
        for side, tree in (("before", ctx.before), ("after", ctx.after)):
            for names, start, _end, owner in named_lists(tree, path):
                held.setdefault(tuple(names), {"before": [], "after": []})[side].append(
                    (start, owner)
                )
        for names, sides in held.items():
            for target in targets.get(names, ()):
                after = [s for s, o in sides["after"] if (path, o) != target]
                before = [s for s, o in sides["before"] if (path, o) != target]
                entry = None
                if after:
                    entry = cp(path, after[0], False)
                elif before:
                    entry = cp(path, before[0], True)
                if entry and entry not in copies[target]:
                    copies[target].append(entry)
    return copies


def kernarg_rows(ctx: Ctx) -> list:
    rows, work, seen_lists = [], [], set()
    for ch in ctx.changes.values():
        if ch.tag != "rocke" or not ch.path.endswith(".py"):
            continue
        bt, at = ctx.ast(ch, "before"), ctx.ast(ch, "after")
        if bt is None:
            continue
        old_lists = param_lists(bt)
        new_lists = param_lists(at) if at is not None else {}
        new_funcs = function_nodes(at) if at is not None else {}
        old_funcs = function_nodes(bt)
        arch = arch_of(ch.path)
        for qual in sorted(old_lists):
            old, new = old_lists[qual], new_lists.get(qual)
            if new is None:
                work.append(Work(ch, qual, old, None, old_funcs[qual].lineno))
            elif old.undetermined or new.undetermined:
                if ast.dump(old_funcs[qual]) != ast.dump(new_funcs[qual]):
                    rows.append(make_row("kernargs", ch.path, new.lineno, qual, "changed",
                        f"{qual} passes a non-literal name to b.param; its kernarg list cannot be compared",
                        arch=arch, foldable=True))  # fmt: skip
            elif old.names != new.names:
                work.append(Work(ch, qual, old, new, new.lineno))
            elif old.conditional != new.conditional:
                rows.append(make_row("kernargs", ch.path, new.lineno, qual, "changed",
                    f"b.param list of {qual} keeps its names but not which are conditional: "
                    f"{old.marked()} -> {new.marked()}; a variant may now pass a different list",
                    arch=arch, foldable=True))  # fmt: skip
            else:
                before = resolved_decls(bt, qual, old.decls)
                after = resolved_decls(at, qual, new.decls)
                if before == after:
                    continue
                diff = [
                    f"{n}({b or '-'} -> {a or '-'})"
                    for n, b, a in zip(new.names, before, after)
                    if b != a
                ]
                rows.append(make_row("kernargs", ch.path, new.lineno, qual, "changed",
                    f"b.param types of {qual} changed with the same names and order: {'; '.join(diff[:4])}"
                    "; a type or attribute change moves the kernarg layout",
                    arch=arch, foldable=True))  # fmt: skip
    if work:
        copies = find_copies(ctx, work)
        for w in work:
            found = copies[(w.change.path, w.qual)]
            unmoved = [c for c in found if not c["moved"]]
            if w.new is None:
                change = "absent after (deleted or renamed)"
            else:
                change = f"{list_diff(w.old.names, w.new.names)}: {w.old.marked()} -> {w.new.marked()}"
            seen_lists |= {(c["path"], c["line"]) for c in unmoved}
            where = (
                f"{len(unmoved)} of {len(found)} copies unmoved"
                if found
                else "no copy found"
            )
            rows.append(make_row("kernargs", w.change.path, w.line, w.qual,
                "mismatch" if unmoved else "changed",
                f"b.param list of {w.qual} changed ({change}); {where}",
                found, arch=arch_of(w.change.path), foldable=not unmoved))  # fmt: skip
    rows += declared_kernarg_rows(ctx, seen_lists)
    return rows


def declared_kernarg_rows(ctx: Ctx, reported: set) -> list:
    """Compare each declared `kernargs` pair's static C++ lists with its builder, after."""
    rows = []
    for surface in ctx.surfaces("after"):
        if surface.name != "kernargs" or not surface.cpp_mirror:
            continue
        producer, mirror = ctx.changes.get(surface.module), ctx.changes.get(
            surface.cpp_mirror
        )
        if producer is None and mirror is None:
            continue
        tree = ctx.after.ast(surface.module)
        lists = param_lists(tree) if tree is not None else {}
        pl = lists.get(surface.function)
        text = ctx.after.text(surface.cpp_mirror)
        if pl is None or text is None:
            what = "builder" if pl is None else "C++ mirror"
            rows.append(make_row("kernargs", surface.cpp_mirror, 0, f"declared:{surface.function}",
                "mismatch", f"declared kernargs pair is broken at the after side: {what} of "
                f"{surface.module}:{surface.function} is missing", arch=surface.arch))  # fmt: skip
            continue
        counter = [
            cp(
                surface.module,
                pl.lineno,
                producer is not None and producer.cls in MOVING,
            )
        ]
        moved = (producer is not None and producer.cls in MOVING) or (
            mirror is not None and mirror.cls in MOVING
        )
        sizes = dict(
            zip(
                pl.names,
                map(decl_size, resolved_decls(tree, surface.function, pl.decls)),
            )
        )
        compared, closest = False, 0
        arch = surface.arch or arch_of(surface.cpp_mirror)
        for entry in cpp_kernel_lists(text):
            key = (surface.cpp_mirror, entry["line"])
            if key in reported:
                compared = True
                continue
            if entry["dynamic"] or entry["unnamed"]:
                compared = True
                if mirror is not None and mirror.cls in MOVING:
                    kind = "dynamic" if entry["dynamic"] else "static but unnamed"
                    rows.append(make_row("kernargs", surface.cpp_mirror, entry["line"], f"declared:{entry['func']}",
                        "changed", f"{entry['func']}() is a {kind} kernarg signature: it cannot be "
                        f"compared with {surface.function} statically", counter, arch=arch))  # fmt: skip
                continue
            if not related(entry["names"], pl.names) and not compatible(
                entry["names"], pl
            ):
                closest = max(closest, len(set(entry["names"]) & set(pl.names)))
                continue
            compared = True
            if not compatible(entry["names"], pl):
                rows.append(make_row("kernargs", surface.cpp_mirror, entry["line"], f"declared:{entry['func']}",
                    "mismatch", f"static list {entry['func']}() disagrees with builder {surface.function} "
                    f"({list_diff(entry['names'], pl.names)}): {entry['names']} vs {pl.marked()}",
                    counter, arch=arch))  # fmt: skip
                continue
            differ, unknown = [], []
            for name, size in zip(entry["names"], entry["sizes"]):
                want = sizes.get(name)
                if size is None or want is None:
                    unknown.append(name)
                elif size != want:
                    differ.append(
                        f"{name}: builder {size_text(want)}, C++ {size_text(size)}"
                    )
            if differ:
                rows.append(make_row("kernargs", surface.cpp_mirror, entry["line"], f"declared:{entry['func']}",
                    "mismatch", f"static list {entry['func']}() names agree with builder {surface.function} "
                    f"but types do not: {'; '.join(differ[:4])}", counter, arch=arch))  # fmt: skip
            elif moved:
                if not unknown:
                    note = "names and types agree"
                elif len(unknown) == len(entry["names"]):
                    note = "names agree; types not compared"
                else:
                    note = f"names agree; types not compared for {', '.join(unknown)}"
                rows.append(make_row("kernargs", surface.cpp_mirror, entry["line"], f"declared:{entry['func']}",
                    "changed", f"{entry['func']}() vs {surface.function}: {note}", counter, arch=arch))  # fmt: skip
        if not compared and moved:
            rows.append(make_row("kernargs", surface.cpp_mirror, 0, f"declared:{surface.function}",
                "changed", f"no static list in the C++ mirror shares two names with {surface.function} "
                f"(closest shares {closest}): cannot compare", counter, arch=arch))  # fmt: skip
    return rows


# -- symbols ----------------------------------------------------------------------------------------

SYMBOL_TAGS = {"rocke", "packer"}
DEFINITION_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def symbol_table(tree) -> dict:
    """Qualname -> (kind, line, end, is_method, contract): top-level defs and classes, their
    methods and class fields (annotated or assigned names)."""
    table = {}
    sig = lambda n: repr((ast.dump(n.args), [ast.dump(d) for d in n.decorator_list],  # noqa: E731
                          ast.dump(n.returns) if n.returns else None))  # fmt: skip
    for node in getattr(tree, "body", []):
        if isinstance(node, DEFINITION_FUNCS):
            table[node.name] = ("def", node.lineno, node.end_lineno, False, sig(node))
        elif isinstance(node, ast.ClassDef):
            table[node.name] = ("class", node.lineno, node.end_lineno, False,
                repr(([ast.dump(b) for b in node.bases], [ast.dump(d) for d in node.decorator_list])))  # fmt: skip
            for item in node.body:
                if isinstance(item, DEFINITION_FUNCS):
                    table[f"{node.name}.{item.name}"] = (
                        "def",
                        item.lineno,
                        item.end_lineno,
                        True,
                        sig(item),
                    )
                elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    table[f"{node.name}.{item.target.id}"] = (
                        "field",
                        item.lineno,
                        item.end_lineno,
                        True,
                        ast.dump(item.annotation),
                    )
                elif isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name):
                            table[f"{node.name}.{t.id}"] = (
                                "field",
                                item.lineno,
                                item.end_lineno,
                                True,
                                "assign",
                            )
    return {
        q: e
        for q, e in table.items()
        if not (q.split(".")[-1].startswith("__") and q.endswith("__"))
    }


def module_bindings(tree) -> set:
    """Names a module still binds at top level: defs, classes, imports and assignments."""
    bound = set()

    def visit(statements):
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    bound.update(
                        n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                    )
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                for attr in ("body", "orelse", "finalbody"):
                    visit(getattr(node, attr, []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)

    visit(getattr(tree, "body", []))
    return bound


def iter_imports(path: str, tree):
    """(bound name, module path stem, imported name or None for a module, line)."""
    here = PurePosixPath(path).parent
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = here
                for _ in range(node.level - 1):
                    base = base.parent
                stem = "/" + str(
                    base / node.module.replace(".", "/") if node.module else base
                )
            else:
                stem = (node.module or "").replace(".", "/")
            for alias in node.names:
                yield alias.asname or alias.name, stem, alias.name, alias.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                yield name, (alias.name if alias.asname else name).replace(
                    ".", "/"
                ), None, alias.lineno


def in_module(path: str, stem: str) -> bool:
    files = (stem + ".py", stem + "/__init__.py")
    if stem.startswith("/"):
        return path in (f[1:] for f in files)
    return any(path == f or path.endswith("/" + f) for f in files)


def module_index(tree: Tree) -> dict:
    """Import stem -> repo path of every Python file, keyed as `in_module` matches: the exact
    `/a/b/m`, and each trailing `a/b/m`, `b/m`, `m` (a package is its `__init__.py`)."""
    if tree.stems is None:
        tree.stems = {}
        for path in sorted(p for p in tree.paths if p.endswith(".py")):
            stem = path[:-3].removesuffix("/__init__")
            tree.stems.setdefault("/" + stem, path)
            parts = stem.split("/")
            for i in range(len(parts)):
                tree.stems.setdefault("/".join(parts[i:]), path)
    return tree.stems


def module_imports(tree: Tree, path: str) -> set:
    """Repo files that importing `path` loads directly: `import a.b.c` loads `a`, `a.b` and
    `a.b.c`; `from m import x` loads `m` and, when `x` is a submodule, `m/x`."""
    if path not in tree.imports:
        module, index, out = tree.ast(path), module_index(tree), set()
        for node in ast.walk(module) if module is not None else ():
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    out |= {
                        index.get("/".join(parts[:i])) for i in range(1, len(parts) + 1)
                    }
        for _, stem, imported, _ in (
            iter_imports(path, module) if module is not None else ()
        ):
            if imported is not None:
                out |= {index.get(stem), index.get(f"{stem}/{imported}")}
        tree.imports[path] = out - {None, path}
    return tree.imports[path]


def modules_reaching(tree: Tree, starts, target: str) -> set:
    """The `starts` whose imports, followed transitively, load `target`: one shared search
    with batched reads, then a reverse walk from `target`."""
    graph, frontier = {}, sorted(set(starts))
    while frontier:
        tree.read([p for p in frontier if p not in tree.imports])
        for path in frontier:
            graph[path] = module_imports(tree, path)
        frontier = sorted({t for p in frontier for t in graph[p]} - graph.keys())
    importers = {}
    for path, targets in graph.items():
        for t in targets:
            importers.setdefault(t, []).append(path)
    seen, stack = {target}, [target]
    while stack:
        for path in importers.get(stack.pop(), ()):
            if path not in seen:
                seen.add(path)
                stack.append(path)
    return set(starts) & seen


def symbol_references(tree, path: str, defining: str, name: str, is_method: bool):
    """[(line, node)] of the AST reads of `name`, a symbol of `defining`. A method or field is
    an attribute load; a function or class is an import of the name from the defining module,
    each read of what that import binds, `m.name` for an imported module alias and, in the
    defining file, a read of a name the file never binds. Keyword arguments, strings, docstrings
    and same-named bindings from elsewhere are not reads. (node None: an import line.)
    """
    if is_method:
        return [
            (n.lineno, n)
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and n.attr == name
            and isinstance(n.ctx, ast.Load)
        ]
    imports = list(iter_imports(path, tree))
    from_defining = [
        (b, line)
        for b, stem, imp, line in imports
        if imp == name and in_module(defining, stem)
    ]
    aliases = {b for b, _ in from_defining}
    modules = {
        b
        for b, stem, imp, _ in imports
        if in_module(defining, stem if imp is None else f"{stem}/{imp}")
    }
    binds_name = any(b == name for b, *_ in imports) or any(
        (isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load))
        or (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.name == name
        )
        or (isinstance(n, ast.arg) and n.arg == name)
        for n in ast.walk(tree)
    )
    reads_own = path == defining and not binds_name
    refs = [(line, None) for _, line in from_defining]
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in aliases or (reads_own and node.id == name):
                refs.append((node.lineno, node))
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr == name
            and isinstance(node.value, ast.Name)
            and node.value.id in modules
        ):
            refs.append((node.lineno, node))
    return refs


def required_missing(definition, call, skip_first=False) -> list:
    """Parameters without a default that `call` does not supply."""
    args = definition.args
    positional = [a.arg for a in args.posonlyargs + args.args]
    required = positional[: len(positional) - len(args.defaults)]
    if skip_first and positional:
        positional, required = positional[1:], required[1:]
    keywords = {k.arg for k in call.keywords}
    supplied = set(positional[: len(call.args)]) | keywords
    missing = [p for p in required if p not in supplied]
    missing += [
        a.arg
        for a, d in zip(args.kwonlyargs, args.kw_defaults)
        if d is None and a.arg not in keywords
    ]
    return missing


def call_problem(definition, call, skip_first=False):
    """Why `call` does not fit `definition`, or None. Starred arguments are not judged."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        k.arg is None for k in call.keywords
    ):
        return None
    args = definition.args
    positional = [a.arg for a in args.posonlyargs + args.args]
    if skip_first and positional:
        positional = positional[1:]
    if len(call.args) > len(positional) and not args.vararg:
        return f"{len(call.args)} positional arguments for {len(positional)} parameters"
    allowed = set(positional) | {a.arg for a in args.kwonlyargs}
    unknown = [k.arg for k in call.keywords if k.arg not in allowed]
    if unknown and not args.kwarg:
        return f"unknown keyword(s) {', '.join(unknown)}"
    missing = required_missing(definition, call, skip_first)
    return f"omits required parameter(s) {', '.join(missing)}" if missing else None


_EXTERNAL_BASES = {"object", "ABC", "Generic", "Protocol", "NamedTuple", "Enum", "IntEnum",
                   "Exception", "TypedDict", "str", "int", "dict", "list"}  # fmt: skip


def class_members(node) -> set:
    """Names a class body defines: methods, properties, class attributes, annotated fields."""
    out = set()
    for item in node.body:
        if isinstance(item, DEFINITION_FUNCS):
            out.add(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            out.add(item.target.id)
        elif isinstance(item, ast.Assign):
            out.update(t.id for t in item.targets if isinstance(t, ast.Name))
    return out


def resolve_class(tree: Tree, known: list, path: str, name: str, depth: int = 0):
    """(module path, ClassDef) that `name` denotes in module `path`, following top-level aliases
    (`A = B`) and imports into repo modules; None when it cannot be resolved."""
    module = tree.ast(path)
    if module is None or depth > 8:
        return None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return path, node
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            value = node.value
            if isinstance(value, ast.Name):
                return resolve_class(tree, known, path, value.id, depth + 1)
            return None
    for bound, stem, imported, _ in iter_imports(path, module):
        if bound != name or imported is None:
            continue
        target = next((p for p in known if in_module(p, stem)), None)
        return (
            resolve_class(tree, known, target, imported, depth + 1) if target else None
        )
    return None


def class_provides(tree: Tree, known: list, path: str, cls: str, member: str):
    """Whether a base of `cls` supplies `member` in `tree`: True, False, or the name of the
    first base that cannot be resolved."""
    found = resolve_class(tree, known, path, cls)
    if found is None:
        return cls
    unresolved = None
    for base in found[1].bases:
        while isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Attribute):
            name = base.attr
            module = next(
                (
                    p
                    for p in known
                    if isinstance(base.value, ast.Name) and in_module(p, base.value.id)
                ),
                None,
            )
            where = module or found[0]
        elif isinstance(base, ast.Name):
            name, where = base.id, found[0]
        else:
            unresolved = unresolved or ast.unparse(base)
            continue
        parent = resolve_class(tree, known, where, name)
        if parent is None:
            if name not in _EXTERNAL_BASES:
                unresolved = unresolved or name
            continue
        if member in class_members(parent[1]):
            return True
        deeper = class_provides(tree, known, parent[0], parent[1].name, member)
        if deeper is True:
            return True
        unresolved = unresolved or (deeper or None)
    return unresolved or False


def symbol_rows(ctx: Ctx) -> list:
    cands, direct = [], []  # cands: (change, qual, entry, kind, after entry)
    for ch in ctx.changes.values():
        if not ch.path.endswith(".py") or ch.tag not in SYMBOL_TAGS:
            continue
        bt, at = ctx.ast(ch, "before"), ctx.ast(ch, "after")
        if bt is None:
            continue
        old = symbol_table(bt)
        new = symbol_table(at) if at is not None else {}
        bound = module_bindings(at) if at is not None else set()
        for qual, entry in sorted(old.items()):
            name = qual.split(".")[-1]
            if qual not in new:
                if not entry[3] and name in bound:
                    continue  # still bound by its module: re-exported, not removed
                cls = qual.split(".")[0]
                if entry[3] and cls in old and cls in new:
                    # a method or field dropped from a class that still inherits the name is not removed
                    known = [p for p in ctx.after.under(HKP) if p.endswith(".py")]
                    provided = class_provides(ctx.after, known, ch.path, cls, name)
                    if provided is True:
                        continue
                    if provided:
                        direct.append(make_row("symbols", ch.path, entry[1], qual, "changed",
                            f"override removed; base {provided} not resolved", arch=arch_of(ch.path)))  # fmt: skip
                        continue
                cands.append((ch, qual, entry, "removed", None))
            elif new[qual][4] != entry[4] and entry[0] in ("def", "field"):
                cands.append((ch, qual, entry, "resigned", new[qual]))
        if at is None and not old:
            cands.append((ch, "<module>", ("module", 0, 0, False, ""), "removed", None))
    rows = direct
    if cands:
        rows += reader_rows(ctx, cands)
    rows += call_arity_rows(ctx)
    return rows


def reader_rows(ctx: Ctx, cands: list) -> list:
    names = sorted({q.split(".")[-1] for _, q, e, *_ in cands if e[0] != "module"})
    stems = sorted(
        {
            PurePosixPath(c[0].path).stem
            for c in cands
            if c[2][0] == "module" or c[0].head is None
        }
    )
    words = sorted(set(names) | set(stems) - {"__init__"})
    mentioning = (
        sorted(
            {
                p
                for p, _, _ in ctx.after.grep(words, ["-w", "-F", "-l"])
                if p.endswith(".py")
            }
        )
        if words
        else []
    )
    ctx.after.read(mentioning)
    rows = []
    for ch, qual, entry, kind, new_entry in cands:
        name, is_method = qual.split(".")[-1], entry[3]
        own_hunks = [
            (h, h + max(hl, 1) - 1)
            for b, bl, h, hl in ch.hunks
            if b <= entry[2] and entry[1] <= b + max(bl, 1) - 1
        ]
        refs, calls, loose = [], [], []
        for path in mentioning:
            tree = ctx.after.ast(path)
            text = ctx.after.text(path) or ""
            if tree is None:
                continue
            if entry[0] == "module":
                found = [
                    (line, None)
                    for b, stem, imp, line in iter_imports(path, tree)
                    if in_module(ch.path, stem)
                    or in_module(ch.path, stem if imp is None else f"{stem}/{imp}")
                ]
            else:
                if is_method and not re.search(
                    r"\b" + re.escape(qual.split(".")[0]) + r"\b", text
                ):
                    # `.name` read through a subclass or an untyped receiver: the class
                    # cannot be told, so it can only make the removal `changed`.
                    loose += [
                        cp(path, line, False)
                        for line, _ in symbol_references(
                            tree, path, ch.path, name, True
                        )
                    ]
                    continue
                # Only reads bound to the old module count (an import from it, `m.name` on
                # it, or an unbound read in the file itself), so a same-named def elsewhere
                # neither hides a stale importer nor turns an unrelated file into a reader.
                found = symbol_references(tree, path, ch.path, name, is_method)
            other = ctx.changes.get(path)
            for line, node in found:
                if path == ch.path and any(a <= line <= b for a, b in own_hunks):
                    continue
                refs.append(cp(path, line, other is not None and line in other.added))
                if node is not None:
                    calls.append((path, tree, node))
        arch = arch_of(ch.path)
        if not refs:
            # Only a method of a class that survives (a removed class has its own row; a
            # field's `.name` is too common a word to stand for a reader), read in a file
            # whose imports reach the defining module.
            survives = qual.split(".")[0] in module_bindings(ctx.after.ast(ch.path))
            if kind == "removed" and entry[0] == "def" and survives and loose:
                reaching = modules_reaching(
                    ctx.after, {c["path"] for c in loose}, ch.path
                )
                loose = [c for c in loose if c["path"] in reaching]
            if loose and kind == "removed" and entry[0] == "def" and survives:
                rows.append(make_row("symbols", ch.path, entry[1], qual, "changed",
                    f"{qual} is removed; {len(loose)} `.{name}` read(s) in files that never name "
                    f"{qual.split('.')[0]}: their receiver's class cannot be told", loose, arch=arch))  # fmt: skip
            continue
        if kind == "removed":
            what = "module" if entry[0] == "module" else qual
            rows.append(make_row("symbols", ch.path, entry[1], qual, "mismatch",
                f"{what} is removed but still referenced after", refs, arch=arch))  # fmt: skip
            continue
        problems = []
        defn = (
            next(
                (
                    n
                    for n in ast.walk(ctx.after.ast(ch.path))
                    if isinstance(n, DEFINITION_FUNCS) and n.lineno == new_entry[1]
                ),
                None,
            )
            if entry[0] == "def"
            else None
        )
        if defn is not None and not defn.decorator_list:
            for path, tree, node in calls:
                for call in ast.walk(tree):
                    if isinstance(call, ast.Call) and call.func is node:
                        why = call_problem(defn, call, skip_first=is_method)
                        if why:
                            problems.append((path, call.lineno, why))
        if problems:
            rows.append(make_row("symbols", ch.path, new_entry[1], qual, "mismatch",
                f"{qual} was re-signed and a reader no longer fits: {problems[0][2]} at {problems[0][0]}:{problems[0][1]}",
                refs, arch=arch))  # fmt: skip
        else:
            rows.append(make_row("symbols", ch.path, new_entry[1], qual, "changed",
                f"{qual} was re-signed; {len(refs)} reader(s) read it", refs, arch=arch))  # fmt: skip
    return rows


def call_arity_rows(ctx: Ctx) -> list:
    """Added calls to a module-level function with exactly one def, omitting a required parameter."""
    calls = []  # (change, call node, def name, module stem or None for "this file")
    for ch in ctx.changes.values():
        if not ch.path.endswith(".py") or not ch.added or ch.tag == "other":
            continue
        tree = ctx.after.ast(ch.path)
        if tree is None:
            continue
        bound = {n: (s, i) for n, s, i, _ in iter_imports(ch.path, tree)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or node.lineno not in ch.added:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                k.arg is None for k in node.keywords
            ):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                stem, name = bound.get(func.id, (None, func.id))
                if name is not None:
                    calls.append((ch, node, name, stem))
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in bound
            ):
                stem, name = bound[func.value.id]
                calls.append(
                    (ch, node, func.attr, stem if name is None else f"{stem}/{name}")
                )
    if not calls:
        return []
    names = sorted({c[2] for c in calls})
    patterns = [
        rf"^[[:space:]]*(async[[:space:]]+)?def[[:space:]]+({'|'.join(names[i : i + 100])})[[:space:]]*\("
        for i in range(0, len(names), 100)
    ]
    definitions = {}
    for path, line, text in ctx.after.grep(patterns, ["-E"]):
        if path.endswith(".py"):
            definitions.setdefault(re.search(r"def\s+(\w+)", text).group(1), []).append(
                (path, line)
            )
    unique = {n: d[0] for n, d in definitions.items() if len(d) == 1}
    ctx.after.read(sorted({p for p, _ in unique.values()}))
    rows = []
    for ch, call, name, stem in calls:
        if name not in unique:
            continue
        path, line = unique[name]
        if (path != ch.path) if stem is None else not in_module(path, stem):
            continue
        tree = ctx.after.ast(path)
        defn = next(
            (
                n
                for n in getattr(tree, "body", [])
                if isinstance(n, DEFINITION_FUNCS) and n.lineno == line
            ),
            None,
        )
        if defn is None or defn.decorator_list:
            continue
        missing = required_missing(defn, call)
        if missing:
            other = ctx.changes.get(path)
            rows.append(make_row("symbols", ch.path, call.lineno, f"call:{name}", "mismatch",
                f"call {name}(...) omits required parameter(s) {', '.join(missing)}; sole def at {path}:{line}",
                [cp(path, line, other is not None and line in other.added)],
                arch=arch_of(ch.path, path)))  # fmt: skip
    return rows


# -- surfaces ---------------------------------------------------------------------------------------


def surfaces_rows(ctx: Ctx) -> list:
    rows, seen = [], set()
    for surface in ctx.surfaces("before") + ctx.surfaces("after"):
        key = (
            surface.manifest,
            surface.name,
            surface.module,
            surface.function,
            surface.cpp_mirror,
        )
        if key in seen:
            continue
        seen.add(key)
        py_changed = False
        line = 0
        for side in ("before", "after"):
            tree = ctx.tree_of(side).ast(surface.module)
            node = (
                function_nodes(tree).get(surface.function) if tree is not None else None
            )
            if node is not None and side == "after":
                line = node.lineno
            dump = None if node is None else ast.dump(without_docstrings(node))
            if side == "before":
                before_dump = dump
        py_changed = before_dump != dump
        cpp_changed = ctx.moved(surface.cpp_mirror) if surface.cpp_mirror else False
        test_changed = ctx.moved(surface.test) if surface.test else False
        manifest_changed = ctx.moved(surface.manifest)
        counter = [cp(surface.cpp_mirror, 0, cpp_changed)] if surface.cpp_mirror else []
        if surface.test:
            counter.append(cp(surface.test, 0, test_changed))
        arch = surface.arch or arch_of(surface.module, surface.cpp_mirror)
        name = f"{surface.name}:{surface.function}"
        broken = [
            what
            for what, path in (
                ("python module", surface.module),
                ("C++ mirror", surface.cpp_mirror),
            )
            if path and ctx.after.read([path])[path] is None
        ]
        if not broken and dump is None and (py_changed or manifest_changed):
            broken = [f"function {surface.function}"]
        if broken:
            rows.append(make_row("surfaces", surface.manifest, 0, name, "mismatch",
                f"declared launch_surface {surface.name} points at a missing {', '.join(broken)} after the change",
                counter, arch=arch))  # fmt: skip
        elif py_changed or cpp_changed or manifest_changed:
            side = (
                ("python side" if py_changed else "")
                + (" and " if py_changed and cpp_changed else "")
                + ("C++ mirror" if cpp_changed else "")
            )
            if manifest_changed and not (py_changed or cpp_changed):
                side = "manifest entry"
            other = (
                "the counterpart did not change in code"
                if not (py_changed and cpp_changed)
                else "both sides changed"
            )
            rows.append(make_row("surfaces", surface.module if py_changed else surface.cpp_mirror or surface.manifest,
                line if py_changed else 0, name, "changed",
                f"launch_surface {surface.name}: {side} changed; {other}", counter, arch=arch))  # fmt: skip
    return rows


# -- data -------------------------------------------------------------------------------------------


def descriptor_delta(old: str, new: str) -> str:
    """What changed between two versions of a descriptor, with JSON types preserved."""

    def parse(text):
        try:
            doc = json.loads(text) if text else {}
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None

    before, after = parse(old), parse(new)
    if before is None or after is None:
        return "content is not comparable JSON"

    def entries(doc):
        items = doc.get("kernelDescriptors")
        return {
            str(i.get("name", i.get("id", n))): json.dumps(i, sort_keys=True)
            for n, i in enumerate(items if isinstance(items, list) else [])
            if isinstance(i, dict)
        }

    old_e, new_e = entries(before), entries(after)
    changed = sum(1 for k in old_e.keys() & new_e.keys() if old_e[k] != new_e[k])
    other = sorted(
        k
        for k in (before.keys() | after.keys()) - {"kernelDescriptors"}
        if json.dumps(before.get(k), sort_keys=True)
        != json.dumps(after.get(k), sort_keys=True)
    )
    return (
        f"kernel descriptors {len(old_e)} -> {len(new_e)} "
        f"(+{len(new_e.keys() - old_e.keys())}, -{len(old_e.keys() - new_e.keys())}, {changed} changed)"
        + (f"; other keys changed: {', '.join(other)}" if other else "")
    )


def generic_members(text):
    """{(generic, member)} of gpu_generic_targets.json, or None when the shape is unknown."""
    try:
        generics = json.loads(text).get("generics")
    except (ValueError, AttributeError):
        return None
    if not isinstance(generics, dict):
        return None
    out = set()
    for generic, members in generics.items():
        if not isinstance(members, list):
            return None
        out |= {(generic, str(m)) for m in members}
    return out


def data_rows(ctx: Ctx) -> list:
    rows = []
    for ch in ctx.changes.values():
        path = ch.path
        if ch.tag == "generic-targets" and path == GENERIC_TARGETS_JSON:
            old, new = generic_members(decode(ch.base) or ""), generic_members(
                decode(ch.head) or ""
            )
            if ch.cls in ("data-equal", "whitespace-only"):
                rows.append(make_row("data", path, 0, path, "inert", ch.cls))
            elif old is None or new is None:
                rows.append(
                    make_row(
                        "data",
                        path,
                        0,
                        path,
                        "changed",
                        "generic-target table not readable as generics -> members",
                    )
                )
            else:
                for generic, member in sorted(old - new):
                    rows.append(make_row("data", path, 0, f"{generic}:{member}", "changed",
                        f"generic target {generic}: member {member} removed", arch=arch_of(member)))  # fmt: skip
                for generic, member in sorted(new - old):
                    rows.append(make_row("data", path, 0, f"{generic}:{member}", "changed",
                        f"generic target {generic}: member {member} added", arch=arch_of(member)))  # fmt: skip
                if old == new:
                    before, after = json.loads(decode(ch.base)), json.loads(
                        decode(ch.head)
                    )
                    keys = sorted(k for k in before.keys() | after.keys()
                                  if k not in ("$comment", "source", "generics")
                                  and json.dumps(before.get(k), sort_keys=True) != json.dumps(after.get(k), sort_keys=True))  # fmt: skip
                    rows.append(make_row("data", path, 0, path, "changed" if keys else "inert",
                        f"membership unchanged; keys changed: {', '.join(keys)}" if keys else "membership unchanged; comment or source keys only"))  # fmt: skip
        elif ch.cls == "dvc-pointer":
            old = parse_dvc_pointer(decode(ch.base) or "")
            new = parse_dvc_pointer(decode(ch.head) or "")
            shown = ", ".join(
                f"{p}: {old.get(p, 'none')[:8]} -> {new.get(p, 'none')[:8]}"
                for p in sorted(old.keys() | new.keys())
                if old.get(p) != new.get(p)
            )
            rows.append(make_row("data", path, 0, path, "changed",
                f"catalog content changed, not inspected ({shown})", arch=arch_of(path), foldable="dvc"))  # fmt: skip
        elif (
            ch.tag == "descriptor"
            and ch.cls not in ("docs",)
            and path.endswith(DESCRIPTOR_SUFFIXES)
        ):
            if ch.cls in ("data-equal", "whitespace-only"):
                rows.append(make_row("data", path, 0, path, "inert", ch.cls))
            else:
                rows.append(make_row("data", path, 0, path, "changed",
                    "descriptor content changed: " + descriptor_delta(decode(ch.base) or "", decode(ch.head) or ""),
                    arch=arch_of(path)))  # fmt: skip
        elif ch.tag == "descriptor" and ch.cls not in ("docs",):
            rows.append(
                make_row(
                    "data",
                    path,
                    0,
                    path,
                    "changed",
                    "descriptor-tree file changed: not understood",
                    arch=arch_of(path),
                )
            )
    return rows


# -- one view -----------------------------------------------------------------------------------------


def analyse_view(before: Tree, after: Tree, label: str):
    ctx = Ctx(before, after, label)
    rows = []
    for ch in ctx.changes.values():
        row = file_row(ctx, ch)
        if row:
            rows.append(row)
    for fn in (kernarg_rows, symbol_rows, surfaces_rows, data_rows):
        rows += fn(ctx)
    for row in rows:
        row["view"] = label
    return ctx, rows


def merge_views(per_view: list) -> list:
    """Rows of the first view, plus the rows of later views that are new or stronger there."""
    rows = list(per_view[0])
    keyed = {(r["kind"], r["path"], r["name"]): r for r in rows}
    for later in per_view[1:]:
        for row in later:
            prior = keyed.get((row["kind"], row["path"], row["name"]))
            if prior is None or SEVERITY[row["status"]] > SEVERITY[prior["status"]]:
                row["merged_only"] = True
                rows.append(row)
    return rows


# -- emit fold ---------------------------------------------------------------------------------------

EMIT_SIZES = {"i8": 1, "i16": 2, "i32": 4, "float": 4, "half": 2, "i64": 8, "double": 8}


def load_emit(path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise rl.Fail(f"cannot read emit file {path}: {exc}") from None
    if (
        not isinstance(data, dict)
        or data.get("schema") != 2
        or data.get("tool") != "impact_emit"
    ):
        raise rl.Fail(f"{path} is not impact_emit schema 2 output")
    problem = emit_shape_problem(data)
    if problem:
        raise rl.Fail(f"emit file {path}: {problem}")
    return data


def emit_shape_problem(data: dict):
    """Why an impact_emit document cannot be folded, or None: every field the fold reads
    must have its type, so a malformed file fails instead of crashing or folding."""

    def strings(value):
        return isinstance(value, list) and all(isinstance(v, str) for v in value)

    def count(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def params(value):
        return value is None or (
            isinstance(value, list)
            and all(isinstance(p, list) and len(p) == 2 and strings(p) for p in value)
        )

    for key in ("base_tree", "head_tree"):
        if not (
            isinstance(data.get(key), str)
            and re.fullmatch(r"[0-9a-f]{40,64}", data[key])
        ):
            return f"`{key}` is not a tree id"
    if not isinstance(data.get("families"), list):
        return "`families` is not a list"
    skipped = data.get("not_inspected")
    if not isinstance(skipped, list) or not all(isinstance(i, dict) for i in skipped):
        return "`not_inspected` is not a list of objects"
    for n, fam in enumerate(data["families"]):
        if not isinstance(fam, dict):
            return f"family {n} is not an object"
        if not all(
            isinstance(fam.get(k), str) and fam[k]
            for k in ("arch", "source", "builder")
        ):
            return f"family {n} lacks arch, source or builder"
        variants = fam.get("variants")
        if not (
            isinstance(variants, dict)
            and count(variants.get("base"))
            and count(variants.get("head"))
        ):
            return f"family {n}: `variants` lacks base and head counts"
        lists = ("only_in_base", "only_in_head", "params_changed", "ir_changed")
        if not all(strings(fam.get(k)) for k in lists):
            return f"family {n}: {', '.join(lists)} must be lists of names"
        failures, built = fam.get("failures"), fam.get("params")
        if not (
            isinstance(failures, dict)
            and all(isinstance(failures.get(s), dict) for s in ("base", "head"))
        ):
            return f"family {n}: `failures` lacks base and head objects"
        if not (
            isinstance(built, dict)
            and all(params(built.get(s)) for s in ("base", "head"))
        ):
            return f"family {n}: `params` is not base and head [name, type] lists"
        if not isinstance(fam.get("kdp", ""), str) or not isinstance(
            fam.get("spec_origin", ""), str
        ):
            return f"family {n}: `kdp` or `spec_origin` is not a string"
    return None


def emit_agrees(params, entry):
    """None when the head params and a static C++ list agree, else why they do not."""
    names = [p[0] for p in params]
    if names != entry["names"]:
        return f"names {names} vs {entry['names']} ({list_diff(entry['names'], names)})"
    for (name, typ), size in zip(params, entry["sizes"]):
        want = "ptr" if typ.startswith("ptr") else EMIT_SIZES.get(typ)
        if size is not None and want is not None and size != want:
            return f"{name}: IR type {typ} vs C++ {size if size == 'ptr' else f'{size}-byte'}"
    return None


def emit_source(ctx: Ctx, source: str) -> str:
    """Repo path of a family's `source`: descriptors name it relative to the rocKE library
    (`kernels/gfx950/attn.py`); an unresolvable source is returned as given."""
    for tree in (ctx.after, ctx.before):
        if source in tree.paths:
            return source
        known = {
            p for p in tree.paths if p.endswith(".py") and p.startswith(IMPORT_ROOTS)
        }
        found = resolve_path_literal(source, known)
        if found:
            return found
    return source


def fold_emit(ctx: Ctx, rows: list, emit: dict) -> list:
    """Apply impact_emit output to the base..head view's rows (see the module docstring)."""
    families = emit["families"]
    out = []
    surfaces = ctx.surfaces("after")
    equal = bool(families) and not emit["not_inspected"]
    #: modules a built variant imports: only rows on these paths can be proven equal.
    covered = set()
    for fam in families:
        arch, builder = fam["arch"], fam["builder"]
        quiet = not any(
            fam[k]
            for k in ("params_changed", "ir_changed", "only_in_base", "only_in_head")
        )
        failures = fam["failures"]
        if not quiet or any(failures[side] for side in ("base", "head")):
            equal = False
        params = fam["params"]["head"]
        source = emit_source(ctx, fam["source"])
        if fam["variants"]["head"] > 0:
            for side in ("before", "after"):
                covered |= ctx.closure(side).reach(source)
        if fam.get("params_changed") and params:
            entries = []
            for s in surfaces:
                if (
                    s.name == "kernargs"
                    and s.module == source
                    and s.function == builder
                    and s.cpp_mirror
                ):
                    entries += [
                        (s.cpp_mirror, e)
                        for e in cpp_kernel_lists(ctx.after.text(s.cpp_mirror) or "")
                    ]
            if not entries:
                for p in sorted(ctx.after.paths):
                    if (
                        p.startswith(ENGINE + "packs/")
                        and p.endswith(C_SUFFIXES)
                        and arch
                        and arch in p.lower()
                    ):
                        entries += [
                            (p, e) for e in cpp_kernel_lists(ctx.after.text(p) or "")
                        ]
            names = [p[0] for p in params]
            entries = [
                (p, e) for p, e in entries if e["names"] and related(e["names"], names)
            ]
            bad = [(p, e, emit_agrees(params, e)) for p, e in entries]
            bad = [b for b in bad if b[2]]
            counter = [cp(source, 0, True)]
            if bad:
                p, e, why = bad[0]
                out.append(make_row("kernargs", p, e["line"], f"emit:{arch}:{builder}", "mismatch",
                    f"emitted params of {builder} ({arch}) disagree with static list {e['func']}(): {why}",
                    counter, arch=arch))  # fmt: skip
            else:
                note = (
                    f"consistent with {len(entries)} static C++ list(s)"
                    if entries
                    else "no static C++ list to compare"
                )
                out.append(make_row("kernargs", source, 0, f"emit:{arch}:{builder}", "changed",
                    f"emitted params of {builder} changed for {len(fam['params_changed'])} variant(s); {note}",
                    counter, arch=arch))  # fmt: skip
        elif fam.get("params_changed"):
            out.append(make_row("kernargs", source, 0, f"emit:{arch}:{builder}", "changed",
                f"emitted params of {builder} changed for {len(fam['params_changed'])} variant(s); no param list recorded", arch=arch))  # fmt: skip
        if fam.get("ir_changed"):
            out.append(make_row("kernargs", source, 0, f"emit-ir:{arch}:{builder}", "changed",
                f"emitted IR of {builder} changed for {len(fam['ir_changed'])} variant(s)", arch=arch))  # fmt: skip
        failed = {side: v for side, v in failures.items() if v}
        if failed:
            out.append(make_row("kernargs", source, 0, f"emit-fail:{arch}:{builder}", "changed",
                "variants failed to build: " + "; ".join(f"{s}: {len(v)}" for s, v in sorted(failed.items())), arch=arch))  # fmt: skip
    for item in emit.get("not_inspected") or []:
        out.append(make_row("kernargs", item.get("kdp", ""), 0, f"emit-skip:{item.get('kdp')}:{item.get('side')}",
            "changed", f"not inspected ({item.get('side')}): {item.get('reason')}"))  # fmt: skip
    if equal:
        dvc_dirs = [
            str(PurePosixPath(f["kdp"]).parent) + "/"
            for f in families
            if f.get("spec_origin") == "registry_enumerator"
            and f.get("kdp")
            and f["variants"]["head"] > 0
        ]
        for row in rows:
            if (
                row["view"] != ctx.label
                or row["status"] != "changed"
                or not row["foldable"]
            ):
                continue
            if row["foldable"] == "dvc":
                pointer_dir = str(PurePosixPath(row["path"]).parent) + "/"
                if not any(
                    d.startswith(pointer_dir) or pointer_dir.startswith(d)
                    for d in dvc_dirs
                ):
                    continue
                row.update(
                    status="inert",
                    reason="catalog regenerated: every shipped variant equal",
                )
            elif row["path"] in covered:
                row.update(
                    status="inert",
                    reason="emitted IR and params equal for every shipped variant",
                )
    for row in out:
        row["view"] = ctx.label
    return rows + out


# -- rollup and output -----------------------------------------------------------------------------


def rollup(rows: list) -> str:
    live = [r for r in rows if r["status"] != "outside"]
    if any(r["status"] == "mismatch" for r in live):
        return "Runtime impact: BREAKS"
    changed = [r for r in live if r["status"] == "changed"]
    if changed:
        archs = sorted({a.strip() for r in changed for a in r["arch"].split(",")})
        return "Runtime impact: NEEDS-DEVICE " + ", ".join(archs)
    left = {}
    for r in rows:
        lane = OWNER_LANES.get(tag_of(r["path"])) if r["kind"] == "file" else None
        if lane and r["status"] == "outside":
            left.setdefault(lane, set()).add(r["path"])
    note = (
        "; not judged here: "
        + ", ".join(f"{len(p)} file(s) for {lane}" for lane, p in sorted(left.items()))
        if left
        else ""
    )
    if live:
        return f"Runtime impact: NONE ({len({r['path'] for r in live})} files inspected{note})"
    return f"Runtime impact: N/A (no surface file{note})"


def anchor(row) -> str:
    return f"{row['path']}:{row['line']}" if row["line"] else row["path"]


def render_rows(rows: list) -> str:
    def esc(text):
        return str(text).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| Anchor | View | Status | Reason | Counterparts |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(
        rows, key=lambda r: (-SEVERITY[r["status"]], r["path"], r["line"], r["name"])
    ):
        view = r["view"] + (" (merged-only)" if r["merged_only"] else "")
        counter = "; ".join(
            f"{anchor(c)} ({'moved' if c['moved'] else 'unmoved'})"
            for c in r["counterparts"][:6]
        ) + (
            f"; +{len(r['counterparts']) - 6} more"
            if len(r["counterparts"]) > 6
            else ""
        )
        lines.append(
            f"| {esc(anchor(r))} | {view} | {r['status']} | {esc(r['reason'])} | {esc(counter)} |"
        )
    return "\n".join(lines)


def render_files(ctx_changes: list, rows: list) -> str:
    worst = {}
    for r in rows:
        if SEVERITY[r["status"]] >= SEVERITY[worst.get(r["path"], "outside")]:
            worst[r["path"]] = r["status"]
    lines = ["| Path | Surface | Class | Status | Functions |", "|---|---|---|---|---|"]
    for ch in ctx_changes:
        lines.append(
            f"| {ch.path} | {ch.tag} | {ch.cls} | {worst.get(ch.path, 'outside')} | {', '.join(ch.funcs[:6])} |"
        )
    return "\n".join(lines)


# -- command line --------------------------------------------------------------------------------------

KINDS = {
    "files": ("file",),
    "kernargs": ("kernargs",),
    "symbols": ("symbols",),
    "surfaces": ("surfaces",),
    "data": ("data",),
}


def resolve(args):
    """(repo, (base commit, tree), (head commit, tree), (tip commit, tree) or None)."""
    if args.round_dir:
        rd = args.round_dir
        repo = rl.pin_value(rd, "CLONE")
        base, head = rl.pin_value(rd, "BASE"), rl.pin_value(rd, "HEAD")
        if not (repo and base and head):
            raise rl.Fail(f"PIN.md in {rd} lacks CLONE, BASE or HEAD")
        tip = args.target_tip or rl.pin_value(rd, "TARGET_TIP")
    else:
        if not (args.repo and args.base and args.head):
            raise rl.Fail("give --repo, --base and --head, or --round-dir")
        repo, base, head, tip = args.repo, args.base, args.head, args.target_tip
    found = resolve_revs(repo, [base, head] + ([tip] if tip else []))
    tip_pair = found[2] if tip and found[2][0] != found[0][0] else None
    return repo, found[0], found[1], tip_pair


def run(args):
    repo, (base, base_tree_id), (head, head_tree_id), tip_pair = resolve(args)
    if base_tree_id == head_tree_id:
        raise rl.Fail("empty diff")
    base_tree, head_tree = Tree(repo, base_tree_id), Tree(repo, head_tree_id)
    emit = None
    if args.cmd == "report" and args.emit:
        emit = load_emit(args.emit)
        made_for = (emit["base_tree"], emit["head_tree"])
        if made_for != (base_tree_id, head_tree_id):
            raise rl.Fail(
                f"emit file {args.emit} was made for trees {made_for[0][:12]}..{made_for[1][:12]}, "
                f"not {base_tree_id[:12]}..{head_tree_id[:12]}; run impact_emit.py on this base and head"
            )
    views = [
        dict(
            label="base..head",
            before=base[:12],
            after=head[:12],
            status="ok",
            reason="",
        )
    ]
    ctx, rows = analyse_view(base_tree, head_tree, "base..head")
    if not ctx.changes:
        raise rl.Fail("empty diff")
    views[0]["files"] = len(ctx.changes)
    per_view = [rows]
    if tip_pair:
        tip, tip_tree_id = tip_pair
        tree_id, why = merged_tree(repo, tip, head)
        view = dict(
            label="merged",
            before=tip[:12],
            after=tree_id[:12] if tree_id else "",
            status="ok",
            reason="",
        )
        views.append(view)
        if tree_id is None:
            view.update(status="skipped", reason=why)
            # Skipping the merged view loses the race check for every surface file of the
            # change, wherever the conflict sits; a conflicted test is also a searched copy.
            matters = any(r["status"] != "outside" for r in rows)
            if (
                why.startswith("merge conflict in ")
                and why != "merge conflict in unknown files"
            ):
                matters = matters or any(
                    tag_of(p.strip()) not in OUTSIDE_TAGS - SEARCHED_TAGS
                    for p in why[18:].split(",")
                )
            else:
                matters = True
            per_view.append([
                dict(make_row("views", "(merged view)", 0, "merged", "changed" if matters else "outside",
                    f"merged view skipped: {why}; a race with the target cannot be excluded"),
                    view="merged", merged_only=True)
            ])  # fmt: skip
        else:
            mctx, mrows = analyse_view(
                Tree(repo, tip_tree_id), Tree(repo, tree_id), "merged"
            )
            view["files"] = len(mctx.changes)
            per_view.append(mrows)
    if emit is not None:
        per_view[0] = fold_emit(ctx, per_view[0], emit)
    return ctx, views, merge_views(per_view)


def public(row) -> dict:
    return {k: v for k, v in row.items() if k != "foldable"}


def cmd(args) -> None:
    ctx, views, rows = run(args)
    selected = (
        rows
        if args.cmd == "report"
        else [
            r
            for r in rows
            if r["kind"] in KINDS[args.cmd] or (args.cmd == "files" and False)
        ]
    )
    files = sorted(ctx.changes.values(), key=lambda c: c.path)
    worst = {}
    for r in rows:
        if SEVERITY[r["status"]] >= SEVERITY[worst.get(r["path"], "outside")]:
            worst[r["path"]] = r["status"]
    if args.cmd == "files":
        selected = [r for r in rows if r["kind"] == "file"]
    if args.json:
        payload = {"views": views, "rows": [public(r) for r in selected]}
        if args.cmd in ("files", "report"):
            payload["files"] = [
                {"path": c.path, "surface": c.tag, "class": c.cls, "functions": c.funcs,
                 "status": worst.get(c.path, "outside")} for c in files
            ]  # fmt: skip
        if args.cmd == "report":
            payload["rollup"] = rollup(rows)
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return
    out = []
    for v in views:
        tail = f"; {v['reason']}" if v["status"] != "ok" else ""
        out.append(
            f"view {v['label']}: {v['before']} -> {v['after']}, {v.get('files', 0)} files, {v['status']}{tail}"
        )
    if args.cmd == "files":
        out += ["", render_files(files, rows)]
    elif selected:
        out += ["", render_rows(selected)]
    else:
        out += ["", f"no {args.cmd} rows"]
    if args.cmd == "report":
        out += ["", rollup(rows)]
    sys.stdout.write("\n".join(out) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("files", "kernargs", "symbols", "surfaces", "data", "report"):
        p = sub.add_parser(name, help=f"{name} rows of the diff")
        p.add_argument("--repo", help="git directory (clone or worktree)")
        p.add_argument("--base", help="base revision (merge base)")
        p.add_argument("--head", help="head revision")
        p.add_argument(
            "--target-tip",
            help="target branch tip; adds the merged view when it differs from base",
        )
        p.add_argument(
            "--round-dir",
            help="round directory holding PIN.md (CLONE, BASE, HEAD, KEY)",
        )
        p.add_argument("--json", action="store_true", help="machine-readable output")
        if name == "report":
            p.add_argument("--emit", help="impact_emit.py output to fold in")
    args = parser.parse_args(argv)
    try:
        cmd(args)
    except rl.Fail as exc:
        print(f"impact: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
