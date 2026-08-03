# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Report broken Sphinx Python symbol references in rocKE Markdown docs.

The checker builds a temporary Sphinx project through ``uvx``. It indexes Python
symbols with the standard library AST instead of importing rocKE, so checking
documentation does not require ROCm, a GPU, torch, or rocKE's runtime libraries.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SPHINX_VERSION = "8.2.3"
MYST_PARSER_VERSION = "4.0.1"
PYTHON_ROLES = "attr|class|const|data|exc|func|meth|mod|obj"
RST_ROLE_RE = re.compile(rf":(?:py:)?(?P<role>{PYTHON_ROLES}):`(?P<target>[^`\n]+)`")
MYST_ROLE_RE = re.compile(rf"\{{py:(?P<role>{PYTHON_ROLES})\}}`(?P<target>[^`\n]+)`")
MISSING_REFERENCE_RE = re.compile(
    r"^(?P<path>.+):(?P<line>\d+): WARNING: py:(?P<role>[a-z]+) "
    r"reference target not found: (?P<target>.+?) \[ref\.[^]]+\]$"
)
FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")


@dataclass(frozen=True)
class ImportRecord:
    module: str
    is_package: bool
    level: int
    imported_module: str | None
    name: str
    alias: str


@dataclass(frozen=True)
class BrokenReference:
    path: str
    line: int
    role: str
    target: str


@dataclass(frozen=True)
class ReferenceLocation:
    path: str
    line: int
    role: str
    target: str


def normalize_sphinx_roles(text: str) -> tuple[str, int]:
    """Convert reStructuredText Python roles to MyST roles outside code fences."""
    normalized: list[str] = []
    replacements = 0
    active_fence: str | None = None

    for line in text.splitlines(keepends=True):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group("fence")[0]
            if active_fence is None:
                active_fence = marker
            elif active_fence == marker:
                active_fence = None
            normalized.append(line)
            continue

        if active_fence is None:

            def replace(match: re.Match[str]) -> str:
                nonlocal replacements
                replacements += 1
                return f"{{py:{match.group('role')}}}`{match.group('target')}`"

            line = RST_ROLE_RE.sub(replace, line)
        normalized.append(line)

    return "".join(normalized), replacements


def find_sphinx_references(text: str, path: str) -> list[ReferenceLocation]:
    """Return MyST Python references outside fenced code blocks."""
    references: list[ReferenceLocation] = []
    active_fence: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group("fence")[0]
            if active_fence is None:
                active_fence = marker
            elif active_fence == marker:
                active_fence = None
            continue
        if active_fence is not None:
            continue
        for match in MYST_ROLE_RE.finditer(line):
            target = match.group("target")
            explicit_target = re.search(r"<([^<>]+)>$", target)
            if explicit_target:
                target = explicit_target.group(1)
            references.append(
                ReferenceLocation(
                    path=path,
                    line=line_number,
                    role=match.group("role"),
                    target=target.lstrip("~"),
                )
            )
    return references


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def _is_property(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "setter":
            return True
    return False


def _add_class_symbols(
    symbols: dict[str, str], class_node: ast.ClassDef, qualified_name: str
) -> None:
    symbols[qualified_name] = "class"
    for node in class_node.body:
        member_name = f"{qualified_name}.{getattr(node, 'name', '')}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols[member_name] = "attribute" if _is_property(node) else "method"
        elif isinstance(node, ast.ClassDef):
            _add_class_symbols(symbols, node, member_name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in _assignment_names(node):
                symbols[f"{qualified_name}.{name}"] = "attribute"


def _module_name(root: Path, source: Path) -> tuple[str, bool]:
    parts = list(source.relative_to(root).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _literal_string_list(node: ast.AST) -> tuple[str, ...]:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return ()
    if not isinstance(value, (list, tuple)):
        return ()
    if not all(isinstance(item, str) for item in value):
        return ()
    return tuple(value)


def build_symbol_inventory(python_roots: Sequence[Path]) -> dict[str, str]:
    """Return Sphinx directive kinds keyed by fully qualified Python name."""
    symbols: dict[str, str] = {}
    imports: list[ImportRecord] = []
    exports: dict[str, tuple[str, ...]] = {}

    for root in python_roots:
        for source in sorted(root.rglob("*.py")):
            module, is_package = _module_name(root, source)
            if not module:
                continue
            try:
                tree = ast.parse(
                    source.read_text(encoding="utf-8"), filename=str(source)
                )
            except (OSError, SyntaxError) as error:
                raise RuntimeError(
                    f"cannot parse Python source {source}: {error}"
                ) from error

            symbols[module] = "module"
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols[f"{module}.{node.name}"] = "function"
                elif isinstance(node, ast.ClassDef):
                    _add_class_symbols(symbols, node, f"{module}.{node.name}")
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    for name in _assignment_names(node):
                        if name == "__all__":
                            value = node.value
                            exports[module] = _literal_string_list(value)
                        else:
                            symbols[f"{module}.{name}"] = "data"
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append(
                            ImportRecord(
                                module=module,
                                is_package=is_package,
                                level=node.level,
                                imported_module=node.module,
                                name=alias.name,
                                alias=alias.asname or alias.name,
                            )
                        )

    unresolved = imports
    while unresolved:
        pending: list[ImportRecord] = []
        changed = False
        for record in unresolved:
            package = (
                record.module if record.is_package else record.module.rpartition(".")[0]
            )
            import_name = "." * record.level + (record.imported_module or "")
            try:
                imported_module = (
                    importlib.util.resolve_name(import_name, package)
                    if record.level
                    else record.imported_module
                )
            except (ImportError, ValueError):
                pending.append(record)
                continue
            if not imported_module:
                pending.append(record)
                continue

            if record.name == "*":
                names = exports.get(imported_module, ())
                candidates = [
                    (f"{imported_module}.{name}", f"{record.module}.{name}")
                    for name in names
                ]
            else:
                candidates = [
                    (
                        f"{imported_module}.{record.name}",
                        f"{record.module}.{record.alias}",
                    )
                ]

            resolved = False
            for target, alias in candidates:
                kind = symbols.get(target)
                if kind is None:
                    continue
                if alias not in symbols:
                    symbols[alias] = kind
                    changed = True
                resolved = True
            if not resolved:
                pending.append(record)
        if not changed:
            break
        unresolved = pending

    return symbols


def render_inventory(symbols: dict[str, str]) -> str:
    """Render a minimal RST object inventory for Sphinx's Python domain."""
    directives = {
        "attribute": "attribute",
        "class": "class",
        "data": "data",
        "function": "function",
        "method": "method",
        "module": "module",
    }
    lines = ["rocKE Python symbol inventory", "=============================", ""]
    ordered_symbols = sorted(
        symbols.items(), key=lambda item: (item[1] == "module", item[0])
    )
    for name, kind in ordered_symbols:
        suffix = "()" if kind in {"function", "method"} else ""
        lines.extend([f".. py:{directives[kind]}:: {name}{suffix}", ""])
    return "\n".join(lines)


def _prepare_sphinx_source(
    source_root: Path, docs_root: Path, inventory: str
) -> tuple[int, list[ReferenceLocation]]:
    copied_docs = source_root / "docs"
    shutil.copytree(docs_root, copied_docs)

    markdown_files = sorted(copied_docs.rglob("*.md"))
    references: list[ReferenceLocation] = []
    for markdown in markdown_files:
        normalized, _ = normalize_sphinx_roles(markdown.read_text(encoding="utf-8"))
        relative_path = markdown.relative_to(copied_docs).as_posix()
        references.extend(find_sphinx_references(normalized, relative_path))
        markdown.write_text(normalized, encoding="utf-8")

    entries = [
        markdown.relative_to(source_root).with_suffix("").as_posix()
        for markdown in markdown_files
    ]
    index = [
        "# rocKE documentation symbol check",
        "",
        "```{toctree}",
        ":hidden:",
        "",
        "api",
        *entries,
        "```",
        "",
    ]
    (source_root / "index.md").write_text("\n".join(index), encoding="utf-8")
    (source_root / "api.rst").write_text(inventory, encoding="utf-8")
    (source_root / "conf.py").write_text(
        "\n".join(
            [
                'project = "rocKE documentation symbol check"',
                'extensions = ["myst_parser"]',
                "nitpicky = True",
                'root_doc = "index"',
                'source_suffix = {".md": "markdown", ".rst": "restructuredtext"}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return len(markdown_files), references


def parse_missing_references(
    warnings: str,
    copied_docs: Path,
    docs_root: Path,
    platform_root: Path,
    references: Sequence[ReferenceLocation],
) -> list[BrokenReference]:
    """Extract unresolved references and map scratch paths to source paths."""
    broken: list[BrokenReference] = []
    used_references: set[int] = set()
    for warning in warnings.splitlines():
        match = MISSING_REFERENCE_RE.match(warning)
        if not match:
            continue
        warning_path = Path(match.group("path"))
        try:
            relative = warning_path.relative_to(copied_docs)
        except ValueError:
            continue
        relative_path = relative.as_posix()
        source_path = docs_root / relative
        try:
            display_path = source_path.relative_to(platform_root).as_posix()
        except ValueError:
            display_path = source_path.as_posix()
        warning_line = int(match.group("line"))
        candidates = [
            (index, reference)
            for index, reference in enumerate(references)
            if index not in used_references
            and reference.path == relative_path
            and reference.role == match.group("role")
            and reference.target == match.group("target")
        ]
        if candidates:
            index, reference = min(
                candidates, key=lambda item: abs(item[1].line - warning_line)
            )
            used_references.add(index)
            source_line = reference.line
        else:
            source_line = warning_line
        broken.append(
            BrokenReference(
                path=display_path,
                line=source_line,
                role=match.group("role"),
                target=match.group("target"),
            )
        )
    return sorted(broken, key=lambda item: (item.path, item.line, item.target))


def _text_report(
    docs_root: Path,
    markdown_files: int,
    references: int,
    broken: Sequence[BrokenReference],
) -> str:
    lines = [
        "rocKE documentation symbol report",
        f"docs root: {docs_root}",
        f"Markdown files: {markdown_files}",
        f"Sphinx symbol references: {references}",
        f"broken references: {len(broken)}",
    ]
    if broken:
        lines.append("")
        lines.extend(
            f"{item.path}:{item.line}: error: unresolved py:{item.role} target "
            f"'{item.target}'"
            for item in broken
        )
    return "\n".join(lines) + "\n"


def _json_report(
    docs_root: Path,
    markdown_files: int,
    references: int,
    broken: Sequence[BrokenReference],
) -> str:
    payload = {
        "broken_references": [asdict(item) for item in broken],
        "docs_root": str(docs_root),
        "markdown_files": markdown_files,
        "reference_count": references,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, default=Path("dsl_docs"))
    parser.add_argument(
        "--python-root", type=Path, action="append", default=None, dest="python_roots"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="scratch directory for the temporary Sphinx build",
    )
    parser.add_argument("--output", type=Path, help="write the report to this path")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--offline", action="store_true", help="require uvx to use cached packages"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    platform_root = Path(__file__).resolve().parents[1]
    docs_root = (platform_root / args.docs_root).resolve()
    python_roots = [
        (platform_root / root).resolve()
        for root in (args.python_roots or [Path("python")])
    ]

    if shutil.which("uvx") is None:
        print(
            "error: uvx is required to run the Sphinx symbol checker", file=sys.stderr
        )
        return 2
    if not docs_root.is_dir():
        print(f"error: docs root is not a directory: {docs_root}", file=sys.stderr)
        return 2
    missing_roots = [str(root) for root in python_roots if not root.is_dir()]
    if missing_roots:
        print(
            f"error: Python root is not a directory: {', '.join(missing_roots)}",
            file=sys.stderr,
        )
        return 2

    base_dir = args.work_dir.resolve() if args.work_dir else None
    if base_dir:
        base_dir.mkdir(parents=True, exist_ok=True)

    try:
        symbols = build_symbol_inventory(python_roots)
        inventory = render_inventory(symbols)
        with tempfile.TemporaryDirectory(
            prefix="rocke-doc-symbols-", dir=base_dir
        ) as temporary:
            run_root = Path(temporary)
            source_root = run_root / "source"
            source_root.mkdir()
            markdown_files, reference_locations = _prepare_sphinx_source(
                source_root, docs_root, inventory
            )
            warnings_path = run_root / "sphinx-warnings.txt"
            build_root = run_root / "build"
            command = [
                "uvx",
                "--from",
                f"sphinx=={SPHINX_VERSION}",
                "--with",
                f"myst-parser=={MYST_PARSER_VERSION}",
            ]
            if args.offline:
                command.append("--offline")
            command.extend(
                [
                    "sphinx-build",
                    "-E",
                    "-b",
                    "dummy",
                    "-n",
                    "-q",
                    "-w",
                    str(warnings_path),
                    str(source_root),
                    str(build_root),
                ]
            )
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                details = result.stderr.strip() or result.stdout.strip()
                print(f"error: Sphinx build failed\n{details}", file=sys.stderr)
                return 2
            warnings = warnings_path.read_text(encoding="utf-8")
            broken = parse_missing_references(
                warnings,
                source_root / "docs",
                docs_root,
                platform_root,
                reference_locations,
            )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    reference_count = len(reference_locations)
    try:
        report_docs_root = docs_root.relative_to(platform_root)
    except ValueError:
        report_docs_root = docs_root
    report = (
        _json_report(report_docs_root, markdown_files, reference_count, broken)
        if args.format == "json"
        else _text_report(report_docs_root, markdown_files, reference_count, broken)
    )
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"wrote {args.format} report to {output}")
        print(
            f"checked {reference_count} symbol references; found {len(broken)} broken"
        )
    else:
        print(report, end="")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
