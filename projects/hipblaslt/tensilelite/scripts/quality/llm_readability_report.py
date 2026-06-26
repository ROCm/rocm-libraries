#!/usr/bin/env python3
# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""LLM-readability metrics scanner — Phase 1.

Scans a Python source tree and reports 11 signals indicating how well the
codebase is structured for LLM (and human) navigation. Signals numbered per
``work/llm-readability/findings.md``:

    #1   file LOC
    #3   generic filename (utils/helpers/common/...)
    #4   typing.Any / typing.cast / # type: ignore count
    #5   Optional-param ratio
    #6   swallowed-error patterns
    #7   max conditional nesting depth per function
    #15  public surface ratio per module
    #17  __init__.py bloat (LOC + re-export count)
    #22  cross-feature imports (lateral imports across top-level feature dirs)
    #23  top-level feature count
    #24  generic feature-dir names

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

GENERIC_NAMES = {"utils", "helpers", "common", "misc", "core", "lib", "shared", "base"}
INFRA_DIR_EXCLUDE = {
    "shared",
    "core",
    "common",
    "infra",
    "_internal",
    "adapters",
    # tensilelite shared-infra dirs (cross-feature imports of these are expected,
    # not lateral coupling between features).
    "Common",
    "Utilities",
}
LOC_HIGH = 1000
LOC_WARN = 500
INIT_LOC_THRESHOLD = 50
INIT_REEXPORT_THRESHOLD = 20
NESTING_THRESHOLD = 5
OPTIONAL_RATIO_THRESHOLD = 0.5
OPTIONAL_MIN_PARAMS = 4
PUBLIC_RATIO_THRESHOLD = 0.7
PUBLIC_MIN_SYMBOLS = 5
FEATURE_COUNT_LIMIT = 15
SKIP_DIRS = {
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    "node_modules",
    "Tests",
    "CustomKernels",
}

# Phase-2 thresholds — keep in sync with metrics/BASELINE.md.
INTERFACE_FIRST_THRESHOLD = 0.3  # files below ratio = impl-heavy top of file (#25)
MODULE_DEPTH_SHALLOW = 3.0  # impl/interface ratio below = shallow module (#14)
SHALLOW_MODULE_MIN_PUBLIC = 8  # need this many public symbols to even consider (#16)
SHALLOW_MODULE_MAX_IMPL_PER_PUBLIC = 15  # mean impl LOC / public symbol below = shallow
DUPLICATE_LITERAL_MIN_CHARS = 40  # min string-literal length to track for duplication (#10)
CROSS_LAYER_MIN_FEATURES = 3  # same literal in ≥N distinct features = special-casing (#9)
TESTS_DIR_NAME = "Tests"  # test suite dir (tensilelite: Tensile/Tests) (#21, #26)
INTERNAL_PATH_MARKERS = ("._", ".internal.", ".detail.")  # private-path imports (#26)

# Phase-3 (Tier-B) thresholds.
TOKEN_CHARS_PER_TOKEN = 4  # char/4 stdlib heuristic for #13 (no tokenizer dep)
FILE_TOKEN_THRESHOLD = 5000  # files above ≈ this many tokens are split candidates (#13)
PARALLEL_IMPL_JACCARD = 0.4  # symbol-name overlap above = parallel implementations (#19)
PARALLEL_IMPL_MIN_SYMBOLS = 5  # both modules need this many public symbols to compare (#19)

# #20 adapter discipline — external deps that should only be imported from their
# designated wrapper module(s). Values are fnmatch globs on the src-relative
# path. The ratchet then keeps new I/O code flowing through these seams. Edit
# deliberately when adding a sanctioned wrapper. Empty for tensilelite v1 — no
# adapter seams have been designated yet, so this signal stays at 0.
ADAPTER_ALLOWLIST: dict[str, tuple[str, ...]] = {}

TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore")


@dataclass
class FileReport:
    path: str
    feature: str = ""
    loc: int = 0
    is_init: bool = False
    init_reexports: int = 0
    generic_name: bool = False
    any_count: int = 0
    cast_count: int = 0
    type_ignore_count: int = 0
    optional_params: int = 0
    total_params: int = 0
    swallowed_error_lines: list[int] = field(default_factory=list)
    max_nesting_depth: int = 0
    deep_nesting_functions: int = 0
    public_symbols: int = 0
    total_symbols: int = 0
    cross_feature_imports: list[str] = field(default_factory=list)
    # Phase-2 additions:
    impl_loc: int = 0  # function-body LOC (excluding signature)
    interface_loc: int = 0  # class decls + signatures + module-level type aliases
    compare_eq_literals: list[str] = field(default_factory=list)  # for #9
    long_literals: list[str] = field(default_factory=list)  # for #10
    internal_test_imports: list[str] = field(default_factory=list)  # for #26
    # Phase-3 (Tier-B) additions:
    token_estimate: int = 0  # ≈ len(text)/4 (#13)
    public_names: list[str] = field(default_factory=list)  # for #19 symbol jaccard
    imported_top_modules: list[str] = field(default_factory=list)  # for #20 adapter check

    @property
    def optional_ratio(self) -> float:
        return self.optional_params / self.total_params if self.total_params else 0.0

    @property
    def public_ratio(self) -> float:
        return self.public_symbols / self.total_symbols if self.total_symbols else 0.0

    @property
    def interface_first_ratio(self) -> float:
        """#25: interface LOC as fraction of (interface + impl). 0 if empty."""
        denom = self.interface_loc + self.impl_loc
        return self.interface_loc / denom if denom else 0.0


def _is_optional_annotation(node: ast.expr) -> bool:
    if isinstance(node, ast.Subscript):
        val = node.value
        if isinstance(val, ast.Name) and val.id == "Optional":
            return True
        if isinstance(val, ast.Attribute) and val.attr == "Optional":
            return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        for side in (node.left, node.right):
            if isinstance(side, ast.Constant) and side.value is None:
                return True
    return False


class FileAnalyser(ast.NodeVisitor):
    """Single-pass AST walk filling out a FileReport."""

    NESTING_TYPES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)

    def __init__(self, report: FileReport):
        self.r = report
        self.depth = 0
        self.func_max_depth = 0

    def visit(self, node):
        if isinstance(node, self.NESTING_TYPES):
            self.depth += 1
            if self.depth > self.r.max_nesting_depth:
                self.r.max_nesting_depth = self.depth
            if self.depth > self.func_max_depth:
                self.func_max_depth = self.depth
            super().visit(node)
            self.depth -= 1
            return
        super().visit(node)

    def visit_FunctionDef(self, node):
        self._count_params(node)
        self._tally_function_loc(node)
        saved_depth = self.depth
        saved_func_max = self.func_max_depth
        self.depth = 0
        self.func_max_depth = 0
        self.generic_visit(node)
        if self.func_max_depth >= NESTING_THRESHOLD:
            self.r.deep_nesting_functions += 1
        self.depth = saved_depth
        self.func_max_depth = saved_func_max

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        # Class declaration line(s) are interface; body methods/fields are
        # walked recursively (their signatures add more interface_loc, their
        # bodies add impl_loc via visit_FunctionDef).
        self.r.interface_loc += 1  # the `class X:` line
        self.r.interface_loc += len(node.decorator_list)
        self.generic_visit(node)

    def _tally_function_loc(self, node) -> None:
        """Split FunctionDef LOC into signature (interface) + body (impl)."""
        body = node.body
        body_start = body[0].lineno if body else node.end_lineno
        body_end = node.end_lineno or body_start
        signature_lines = max(1, body_start - node.lineno)
        # Subtract docstring lines from impl — they're interface in spirit.
        docstring_lines = 0
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_lines = (body[0].end_lineno or body[0].lineno) - body[0].lineno + 1
        impl_lines = max(0, body_end - body_start + 1 - docstring_lines)
        self.r.interface_loc += signature_lines + docstring_lines + len(node.decorator_list)
        self.r.impl_loc += impl_lines

    def visit_Compare(self, node):
        """#9: track string literals used in equality comparisons."""
        if any(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
            for cmp in [node.left, *node.comparators]:
                if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str) and cmp.value:
                    self.r.compare_eq_literals.append(cmp.value)
        self.generic_visit(node)

    def visit_Constant(self, node):
        """#10: collect long string literals for repo-wide duplication check."""
        if isinstance(node.value, str) and len(node.value) >= DUPLICATE_LITERAL_MIN_CHARS:
            self.r.long_literals.append(node.value)
        self.generic_visit(node)

    def _count_params(self, node):
        all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        annotated_optional: set[int] = set()
        for arg in all_args:
            self.r.total_params += 1
            if arg.annotation is not None and _is_optional_annotation(arg.annotation):
                self.r.optional_params += 1
                annotated_optional.add(id(arg))
        # default of None implies optional even without annotation
        defaults = list(node.args.defaults)
        positional = list(node.args.posonlyargs) + list(node.args.args)
        if defaults:
            for arg, default in zip(positional[-len(defaults) :], defaults, strict=False):
                if id(arg) in annotated_optional:
                    continue
                if isinstance(default, ast.Constant) and default.value is None:
                    self.r.optional_params += 1
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
            if default is None or id(arg) in annotated_optional:
                continue
            if isinstance(default, ast.Constant) and default.value is None:
                self.r.optional_params += 1

    def visit_Name(self, node):
        if node.id == "Any":
            self.r.any_count += 1
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in {
            "typing",
            "t",
            "typing_extensions",
        }:
            if node.attr == "Any":
                self.r.any_count += 1
            elif node.attr == "cast":
                self.r.cast_count += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "cast":
            self.r.cast_count += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        body = node.body
        swallow = False
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                swallow = True
            elif isinstance(stmt, ast.Return):
                if stmt.value is None:
                    swallow = True
                elif isinstance(stmt.value, ast.Constant) and stmt.value.value in (False, None, 0):
                    swallow = True
        if isinstance(node.type, ast.Name) and node.type.id == "BaseException":
            swallow = True
        if swallow:
            self.r.swallowed_error_lines.append(node.lineno)
        self.generic_visit(node)


def _public_symbols(tree: ast.Module) -> tuple[int, int]:
    explicit_all: set[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        explicit_all = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                explicit_all.add(elt.value)
    public = 0
    total = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            total += 1
            if explicit_all is not None:
                if node.name in explicit_all:
                    public += 1
            elif not node.name.startswith("_"):
                public += 1
    return public, total


def _public_name_set(tree: ast.Module) -> set[str]:
    """Public top-level def/class names (respecting __all__ when present)."""
    explicit_all: set[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        explicit_all = {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if explicit_all is not None:
                if node.name in explicit_all:
                    names.add(node.name)
            elif not node.name.startswith("_"):
                names.add(node.name)
    return names


def _count_init_reexports(tree: ast.Module) -> int:
    n = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            n += len(node.names)
        elif isinstance(node, ast.Import):
            n += len(node.names)
    return n


def _feature_for(path: Path, src_root: Path) -> str:
    try:
        rel = path.relative_to(src_root)
    except ValueError:
        return ""
    if len(rel.parts) < 2:
        return ""
    return rel.parts[0]


def _resolve_import_feature(
    module: str | None, level: int, src_root: Path, file_path: Path
) -> str | None:
    if level > 0:
        base = file_path.parent
        for _ in range(level - 1):
            base = base.parent
        if module:
            target = base / module.split(".")[0]
        else:
            target = base
        try:
            rel = target.relative_to(src_root)
        except ValueError:
            return None
        if not rel.parts:
            return None
        # Only treat as a feature if the top-level segment is a directory
        # under src_root. Sibling utility modules (logging_utils.py at the
        # package root) shouldn't read as a separate feature.
        if not (src_root / rel.parts[0]).is_dir():
            return None
        return rel.parts[0]
    if not module:
        return None
    first = module.split(".")[0]
    # Direct feature reference: "<feature>.x.y"
    if (src_root / first).is_dir():
        return first
    # Package-prefixed reference: "<src_root.name>.<feature>.x" — strip the prefix
    if first == src_root.name:
        rest = module[len(first) + 1 :] if len(module) > len(first) else ""
        if rest:
            second = rest.split(".")[0]
            if (src_root / second).is_dir():
                return second
        return None
    # Last resort: module already contains a known feature segment
    for child in src_root.iterdir():
        if child.is_dir() and module.startswith(child.name + "."):
            tail = module[len(child.name) + 1 :]
            return tail.split(".")[0] if tail else None
    return None


def _module_level_interface_loc(tree: ast.Module) -> int:
    """Count module-top type aliases / annotated assignments as interface.

    Treats ``X: TypeAlias = Foo`` and the legacy ``X = Foo`` (when value is a
    Subscript like ``List[int]`` or a Name like ``Foo``) as interface lines.
    """
    extra = 0
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            span = (node.end_lineno or node.lineno) - node.lineno + 1
            extra += span
        elif isinstance(node, ast.Assign):
            # Heuristic: module-level Assign whose value is a type-shaped
            # expression (Subscript, Name w/ first-cap, or BinOp w/ Name) is
            # an alias. Skip module-level data dicts/lists (they're impl).
            value = node.value
            if isinstance(value, ast.Subscript):
                extra += (node.end_lineno or node.lineno) - node.lineno + 1
            elif isinstance(value, ast.Name) and value.id[:1].isupper():
                extra += 1
    return extra


def _collect_internal_test_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(marker in f".{mod}." for marker in INTERNAL_PATH_MARKERS):
                out.append(f"{mod}:{node.lineno}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(marker in f".{alias.name}." for marker in INTERNAL_PATH_MARKERS):
                    out.append(f"{alias.name}:{node.lineno}")
    return out


def _is_test_file(path: Path) -> bool:
    return TESTS_DIR_NAME in path.parts


def _top_level_imports(tree: ast.Module) -> list[str]:
    """Top-level module names of absolute imports (relative imports excluded)."""
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            tops.add(node.module.split(".")[0])
    return sorted(tops)


def analyse_file(path: Path, src_root: Path) -> FileReport:
    text = path.read_text(encoding="utf-8", errors="replace")
    loc = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    r = FileReport(path=str(path), loc=loc, feature=_feature_for(path, src_root))
    r.is_init = path.name == "__init__.py"
    r.generic_name = path.stem in GENERIC_NAMES and not r.is_init
    r.type_ignore_count = len(TYPE_IGNORE_RE.findall(text))
    r.token_estimate = len(text) // TOKEN_CHARS_PER_TOKEN

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return r

    FileAnalyser(r).visit(tree)
    r.interface_loc += _module_level_interface_loc(tree)
    r.public_symbols, r.total_symbols = _public_symbols(tree)
    r.public_names = sorted(_public_name_set(tree))
    r.imported_top_modules = _top_level_imports(tree)
    if r.is_init:
        r.init_reexports = _count_init_reexports(tree)
    if _is_test_file(path):
        r.internal_test_imports = _collect_internal_test_imports(tree)

    if r.feature and r.feature not in INFRA_DIR_EXCLUDE:
        seen: set[tuple[str, int]] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                tgt = _resolve_import_feature(node.module, node.level, src_root, path)
                src_str = f"{'.' * node.level}{node.module or ''}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    tgt = _resolve_import_feature(alias.name, 0, src_root, path)
                    if tgt and tgt != r.feature and tgt not in INFRA_DIR_EXCLUDE:
                        key = (alias.name, node.lineno)
                        if key not in seen:
                            seen.add(key)
                            r.cross_feature_imports.append(f"{alias.name}:{node.lineno}")
                continue
            else:
                continue
            if tgt and tgt != r.feature and tgt not in INFRA_DIR_EXCLUDE:
                key = (src_str, node.lineno)
                if key not in seen:
                    seen.add(key)
                    r.cross_feature_imports.append(f"{src_str}:{node.lineno}")
    return r


def _guess_tests_root(src_root: Path) -> Path | None:
    """Locate the test suite dir for seam-coverage detection (#21).

    tensilelite nests its tests under the src root (``Tensile/Tests``) rather
    than as a sibling, so check inside first, then fall back to sibling layouts.
    """
    for candidate in (
        src_root / TESTS_DIR_NAME,
        src_root.parent / TESTS_DIR_NAME,
        src_root.parent.parent / TESTS_DIR_NAME,
    ):
        if candidate.is_dir():
            return candidate
    return None


def _module_depth_aggregates(files: list[FileReport]) -> dict[str, dict[str, float]]:
    """#14 + #16 per feature directory."""
    per_feature: dict[str, dict[str, int]] = {}
    for f in files:
        if not f.feature:
            continue
        bucket = per_feature.setdefault(
            f.feature, {"impl": 0, "interface": 0, "public": 0, "files": 0}
        )
        bucket["impl"] += f.impl_loc
        bucket["interface"] += f.interface_loc
        bucket["public"] += f.public_symbols
        bucket["files"] += 1

    out: dict[str, dict[str, float]] = {}
    for feature, b in per_feature.items():
        depth = (b["impl"] / b["interface"]) if b["interface"] else 0.0
        mean_impl_per_public = (b["impl"] / b["public"]) if b["public"] else 0.0
        shallow = (
            b["public"] >= SHALLOW_MODULE_MIN_PUBLIC
            and mean_impl_per_public < SHALLOW_MODULE_MAX_IMPL_PER_PUBLIC
        )
        out[feature] = {
            "impl_loc": b["impl"],
            "interface_loc": b["interface"],
            "public_symbols": b["public"],
            "depth_ratio": round(depth, 2),
            "mean_impl_per_public": round(mean_impl_per_public, 2),
            "shallow": shallow,
        }
    return out


def _duplicate_literal_clusters(files: list[FileReport]) -> list[dict]:
    """#10: long string literals appearing in ≥2 distinct files."""
    by_literal: dict[str, set[str]] = {}
    for f in files:
        for lit in set(f.long_literals):
            by_literal.setdefault(lit, set()).add(f.path)
    return [
        {"literal": lit[:80] + ("…" if len(lit) > 80 else ""), "file_count": len(paths)}
        for lit, paths in by_literal.items()
        if len(paths) >= 2
    ]


def _cross_layer_special_cases(files: list[FileReport]) -> list[dict]:
    """#9: same Compare-Eq literal referenced from ≥3 distinct features."""
    by_literal: dict[str, set[str]] = {}
    for f in files:
        if not f.feature:
            continue
        for lit in set(f.compare_eq_literals):
            by_literal.setdefault(lit, set()).add(f.feature)
    return [
        {"literal": lit, "feature_count": len(feats), "features": sorted(feats)}
        for lit, feats in by_literal.items()
        if len(feats) >= CROSS_LAYER_MIN_FEATURES
    ]


def _parallel_impl_pairs(files: list[FileReport]) -> list[dict]:
    """#19: module pairs in different features with high public-symbol overlap."""
    candidates = [
        f for f in files if f.feature and len(f.public_names) >= PARALLEL_IMPL_MIN_SYMBOLS
    ]
    pairs: list[dict] = []
    for i, a in enumerate(candidates):
        a_names = set(a.public_names)
        for b in candidates[i + 1 :]:
            if a.feature == b.feature:
                continue
            b_names = set(b.public_names)
            union = a_names | b_names
            jaccard = len(a_names & b_names) / len(union) if union else 0.0
            if jaccard >= PARALLEL_IMPL_JACCARD:
                pairs.append(
                    {
                        "a": Path(a.path).name,
                        "b": Path(b.path).name,
                        "features": sorted({a.feature, b.feature}),
                        "jaccard": round(jaccard, 2),
                    }
                )
    return pairs


def _adapter_violations(files: list[FileReport], src_root: Path) -> list[dict]:
    """#20: tracked external deps imported outside their sanctioned wrapper(s)."""
    out: list[dict] = []
    for f in files:
        try:
            rel = Path(f.path).resolve().relative_to(src_root.resolve()).as_posix()
        except ValueError:
            continue
        for dep in f.imported_top_modules:
            allowed = ADAPTER_ALLOWLIST.get(dep)
            if allowed is None:
                continue
            if not any(fnmatch.fnmatch(rel, pat) for pat in allowed):
                out.append({"dep": dep, "path": rel})
    return out


def _missing_seam_tests(src_root: Path, tests_root: Path | None) -> list[str]:
    """#21: public source modules without a matching test file.

    tensilelite nests tests under ``Tensile/Tests/unit/<Area>/test_*.py`` rather
    than mirroring the source tree flatly, so match by ``test_<stem>.py`` basename
    *anywhere* under the test root rather than at a computed relative path. The
    module walk is recursive so nested packages (e.g. ``Components/Subtile``) are
    covered too.
    """
    if tests_root is None:
        return []
    test_stems = {p.stem for p in tests_root.rglob("test_*.py")}
    out: list[str] = []
    for feature_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if feature_dir.name.startswith(("_", ".")) or feature_dir.name in SKIP_DIRS:
            continue
        if feature_dir.name in INFRA_DIR_EXCLUDE:
            continue
        for mod in sorted(feature_dir.rglob("*.py")):
            if any(part in SKIP_DIRS for part in mod.parts):
                continue
            if mod.name == "__init__.py" or mod.stem.startswith("_"):
                continue
            stem = mod.stem
            if f"test_{stem}" in test_stems or f"test_{feature_dir.name}_{stem}" in test_stems:
                continue
            out.append(mod.relative_to(src_root).as_posix())
    return out


def _internal_test_imports_total(tests_root: Path | None) -> int:
    """#26: count private-path imports across the test suite.

    Test files live under ``Tests/`` which ``scan`` skips, so this signal cannot
    be read off the scanned ``files`` list — walk the test root directly.
    """
    if tests_root is None:
        return 0
    total = 0
    for path in sorted(tests_root.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError:
            continue
        total += len(_collect_internal_test_imports(tree))
    return total


def scan(src_root: Path) -> tuple[list[FileReport], dict]:
    files: list[FileReport] = []
    for path in sorted(src_root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(analyse_file(path, src_root))

    feature_dirs = sorted(
        p.name
        for p in src_root.iterdir()
        if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith(".")
    )
    generic_features = [f for f in feature_dirs if f in GENERIC_NAMES]
    module_depths = _module_depth_aggregates(files)
    shallow_modules = sorted(name for name, info in module_depths.items() if info["shallow"])
    depth_violators = sorted(
        name
        for name, info in module_depths.items()
        if info["interface_loc"] and info["depth_ratio"] < MODULE_DEPTH_SHALLOW
    )
    dup_clusters = _duplicate_literal_clusters(files)
    cross_layer = _cross_layer_special_cases(files)
    tests_root = _guess_tests_root(src_root)
    missing_tests = _missing_seam_tests(src_root, tests_root)
    interface_first_violators = [
        f.path
        for f in files
        if (f.interface_loc + f.impl_loc) >= 30  # noise floor — ignore tiny files
        and f.interface_first_ratio < INTERFACE_FIRST_THRESHOLD
    ]
    internal_test_imports_total = _internal_test_imports_total(tests_root)
    token_heavy = [f.path for f in files if f.token_estimate > FILE_TOKEN_THRESHOLD]
    parallel_impls = _parallel_impl_pairs(files)
    adapter_violations = _adapter_violations(files, src_root)

    summary = {
        "src_root": str(src_root),
        "file_count": len(files),
        "top_level_features": feature_dirs,
        "top_level_feature_count": len(feature_dirs),
        "generic_feature_dirs": generic_features,
        "feature_count_over_limit": len(feature_dirs) > FEATURE_COUNT_LIMIT,
        # Phase-2 additions:
        "module_depths": module_depths,
        "shallow_modules": shallow_modules,
        "depth_below_threshold": depth_violators,
        "duplicate_literal_clusters": dup_clusters,
        "cross_layer_special_cases": cross_layer,
        "missing_seam_tests": missing_tests,
        "interface_first_violators": interface_first_violators,
        "internal_test_imports_total": internal_test_imports_total,
        "tests_root": str(tests_root) if tests_root else None,
        # Phase-3 (Tier-B) additions:
        "token_heavy_files": token_heavy,
        "parallel_impl_pairs": parallel_impls,
        "adapter_violations": adapter_violations,
    }
    return files, summary


def write_json(files: list[FileReport], summary: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "report.json"
    payload = {"summary": summary, "files": [asdict(f) for f in files]}
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


BESPOKE_MIN_TOKENS = 20  # functions smaller than this are too trivial to compare (#12)
BESPOKE_JACCARD = 0.85  # token-set overlap above = near-duplicate implementations (#12)


def _function_token_sets(src_root: Path) -> list[tuple[str, set[str]]]:
    """For #12: a structural token set per function (Name ids + node-type tags)."""
    out: list[tuple[str, set[str]]] = []
    for path in sorted(src_root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        name = path.name
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                toks: set[str] = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name):
                        toks.add(f"n:{sub.id}")
                    elif isinstance(sub, ast.Attribute):
                        toks.add(f"a:{sub.attr}")
                    else:
                        toks.add(f"t:{type(sub).__name__}")
                if len(toks) >= BESPOKE_MIN_TOKENS:
                    out.append((f"{name}:{node.name}", toks))
    return out


def bespoke_helper_pairs(src_root: Path) -> list[dict]:
    """#12 (CI-only, O(functions²)): near-duplicate function pairs by token jaccard.

    Not called by ``scan`` — it's expensive, so only ``measure.py`` (which runs
    in CI, not pre-commit) invokes it.
    """
    fns = _function_token_sets(src_root)
    pairs: list[dict] = []
    for i, (label_a, set_a) in enumerate(fns):
        for label_b, set_b in fns[i + 1 :]:
            union = set_a | set_b
            jaccard = len(set_a & set_b) / len(union) if union else 0.0
            if jaccard >= BESPOKE_JACCARD:
                pairs.append({"a": label_a, "b": label_b, "jaccard": round(jaccard, 2)})
    return pairs


def _section(buf: list[str], title: str, items: list, fmt) -> None:
    buf.append(f"\n## {title} ({len(items)})\n\n")
    if not items:
        buf.append("_none_\n")
        return
    for it in items[:50]:
        buf.append(fmt(it))
    if len(items) > 50:
        buf.append(f"\n_…{len(items) - 50} more_\n")


def write_md(files: list[FileReport], summary: dict, out_dir: Path) -> Path:
    p = out_dir / "report.md"
    buf: list[str] = []
    s = summary
    buf.append("# LLM-readability report\n\n")
    buf.append(f"Source root: `{s['src_root']}`  \n")
    buf.append(f"Files scanned: **{s['file_count']}**  \n")
    feats = ", ".join(s["top_level_features"]) or "—"
    buf.append(f"Top-level features (**#23** = {s['top_level_feature_count']}): {feats}  \n")
    if s["generic_feature_dirs"]:
        buf.append(f"\n⚠️  **#24 generic feature dirs:** {', '.join(s['generic_feature_dirs'])}\n")
    if s["feature_count_over_limit"]:
        buf.append(
            f"\n⚠️  **#23 top-level feature count > {FEATURE_COUNT_LIMIT}** "
            f"({s['top_level_feature_count']}) — exceeds 7±2 mental chunk.\n"
        )

    big = [f for f in files if f.loc >= LOC_HIGH]
    warn = [f for f in files if LOC_WARN <= f.loc < LOC_HIGH]
    generic = [f for f in files if f.generic_name]
    init_bloat = [
        f
        for f in files
        if f.is_init and (f.loc > INIT_LOC_THRESHOLD or f.init_reexports > INIT_REEXPORT_THRESHOLD)
    ]
    swallow = [f for f in files if f.swallowed_error_lines]
    deep_nest = [f for f in files if f.max_nesting_depth >= NESTING_THRESHOLD]
    cross_feat = [f for f in files if f.cross_feature_imports]

    _section(buf, f"#1 File LOC ≥ {LOC_HIGH}", big, lambda f: f"- `{f.path}` — **{f.loc} LOC**\n")
    _section(
        buf,
        f"#1 File LOC {LOC_WARN}-{LOC_HIGH - 1} (warn)",
        warn,
        lambda f: f"- `{f.path}` — {f.loc} LOC\n",
    )
    _section(
        buf,
        "#3 Generic filename",
        generic,
        lambda f: f"- `{f.path}` (basename `{Path(f.path).stem}`)\n",
    )
    _section(
        buf,
        "#17 __init__.py bloat",
        init_bloat,
        lambda f: f"- `{f.path}` — {f.loc} LOC, {f.init_reexports} re-exports\n",
    )
    _section(
        buf,
        "#6 Swallowed errors",
        swallow,
        lambda f: (
            f"- `{f.path}` lines: {', '.join(str(line) for line in f.swallowed_error_lines)}\n"
        ),
    )
    _section(
        buf,
        f"#7 Max nesting depth ≥ {NESTING_THRESHOLD}",
        deep_nest,
        lambda f: f"- `{f.path}` — depth {f.max_nesting_depth}\n",
    )
    _section(
        buf,
        "#22 Cross-feature imports",
        cross_feat,
        lambda f: (
            f"- `{f.path}` ({f.feature}) → "
            f"{', '.join(sorted({i.split(':')[0] for i in f.cross_feature_imports}))}\n"
        ),
    )

    type_hits = [f for f in files if f.any_count or f.cast_count or f.type_ignore_count]
    buf.append(f"\n## #4 `Any` / `cast` / `# type: ignore` ({len(type_hits)})\n\n")
    if not type_hits:
        buf.append("_none_\n")
    else:
        buf.append("| File | Any | cast | type:ignore |\n|---|---:|---:|---:|\n")
        for f in type_hits[:80]:
            buf.append(f"| `{f.path}` | {f.any_count} | {f.cast_count} | {f.type_ignore_count} |\n")
        if len(type_hits) > 80:
            buf.append(f"\n_…{len(type_hits) - 80} more_\n")

    opt_hits = [
        f
        for f in files
        if f.optional_ratio >= OPTIONAL_RATIO_THRESHOLD and f.total_params >= OPTIONAL_MIN_PARAMS
    ]
    buf.append(
        f"\n## #5 Optional-param ratio ≥ {OPTIONAL_RATIO_THRESHOLD} "
        f"(min {OPTIONAL_MIN_PARAMS} params) ({len(opt_hits)})\n\n"
    )
    if not opt_hits:
        buf.append("_none_\n")
    else:
        for f in opt_hits[:50]:
            buf.append(
                f"- `{f.path}` — {f.optional_params}/{f.total_params} = {f.optional_ratio:.0%}\n"
            )

    pub_hits = [
        f
        for f in files
        if f.total_symbols >= PUBLIC_MIN_SYMBOLS and f.public_ratio > PUBLIC_RATIO_THRESHOLD
    ]
    buf.append(
        f"\n## #15 Public surface ratio > {PUBLIC_RATIO_THRESHOLD} "
        f"(min {PUBLIC_MIN_SYMBOLS} symbols) ({len(pub_hits)})\n\n"
    )
    if not pub_hits:
        buf.append("_none_\n")
    else:
        for f in pub_hits[:50]:
            buf.append(
                f"- `{f.path}` — {f.public_symbols}/{f.total_symbols} = {f.public_ratio:.0%}\n"
            )

    p.write_text("".join(buf))
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-readability metrics scanner (Phase 1).")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Tensile"),
        help="Source root to scan (default: Tensile).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("work/llm-readability"),
        help="Output directory (default: work/llm-readability).",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: src root not a dir: {args.root}", file=sys.stderr)
        return 2

    files, summary = scan(args.root)
    jp = write_json(files, summary, args.out)
    mp = write_md(files, summary, args.out)
    print(f"scanned {summary['file_count']} files in {args.root}")
    print(f"  json -> {jp}")
    print(f"  md   -> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
