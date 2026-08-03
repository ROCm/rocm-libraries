# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Report broken Sphinx Python symbol references in rocKE documentation.

The checker builds a temporary Sphinx project through ``uvx``. It indexes Python
symbols with the standard library AST instead of importing rocKE, so checking
documentation does not require ROCm, a GPU, torch, or rocKE's runtime libraries.
It also checks explicit Sphinx roles in Python docstrings against that index.
"""

from __future__ import annotations

import argparse
import ast
import builtins
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
FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<rest>.*)$")
Fence = tuple[str, int]


@dataclass(frozen=True)
class ImportRecord:
    module: str
    target: str
    alias: str
    star: bool = False


@dataclass(frozen=True)
class BrokenReference:
    path: str
    line: int
    role: str
    target: str
    source: str = "markdown"
    found_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceLocation:
    path: str
    line: int
    role: str
    target: str


@dataclass(frozen=True)
class DocstringReference:
    path: str
    line: int
    role: str
    target: str
    module: str
    class_name: str | None


@dataclass(frozen=True)
class PythonIndex:
    symbols: dict[str, str]
    local_aliases: dict[str, str]
    references: tuple[DocstringReference, ...]
    external_aliases: dict[str, frozenset[str]]
    package_roots: frozenset[str]
    python_files: int
    docstrings: int


def _reference_target(target: str) -> str | None:
    explicit_target = re.search(r"<([^<>]+)>$", target)
    if explicit_target:
        target = explicit_target.group(1)
    target = target.lstrip("~")
    if target.startswith("!"):
        return None
    return target.removesuffix("()")


def _fence_transition(line: str, active_fence: Fence | None) -> Fence | None:
    match = FENCE_RE.match(line)
    if not match:
        return active_fence
    fence = match.group("fence")
    marker = fence[0]
    if active_fence is None:
        return marker, len(fence)
    active_marker, active_length = active_fence
    if (
        marker == active_marker
        and len(fence) >= active_length
        and not match.group("rest").strip()
    ):
        return None
    return active_fence


def normalize_sphinx_roles(text: str) -> tuple[str, int]:
    """Convert reStructuredText Python roles to MyST roles outside code fences."""
    normalized: list[str] = []
    replacements = 0
    active_fence: Fence | None = None

    for line in text.splitlines(keepends=True):
        next_fence = _fence_transition(line, active_fence)
        if next_fence != active_fence:
            active_fence = next_fence
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
    active_fence: Fence | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        next_fence = _fence_transition(line, active_fence)
        if next_fence != active_fence:
            active_fence = next_fence
            continue
        if active_fence is not None:
            continue
        for match in MYST_ROLE_RE.finditer(line):
            target = _reference_target(match.group("target"))
            if target is None:
                continue
            references.append(
                ReferenceLocation(
                    path=path,
                    line=line_number,
                    role=match.group("role"),
                    target=target,
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


def _display_path(
    path: Path, display_root: Path, fallback_root: Path | None = None
) -> str:
    try:
        return path.relative_to(display_root).as_posix()
    except ValueError:
        if fallback_root is not None:
            try:
                relative = path.relative_to(fallback_root)
                return (Path(fallback_root.name) / relative).as_posix()
            except ValueError:
                pass
        return path.as_posix()


def _docstring_roles(
    value: ast.Constant,
    *,
    path: str,
    module: str,
    class_name: str | None,
) -> list[DocstringReference]:
    if not isinstance(value.value, str):
        return []
    matches = [*RST_ROLE_RE.finditer(value.value), *MYST_ROLE_RE.finditer(value.value)]
    references: list[DocstringReference] = []
    for match in sorted(matches, key=lambda item: item.start()):
        target = _reference_target(match.group("target"))
        if target is None:
            continue
        references.append(
            DocstringReference(
                path=path,
                line=value.lineno + value.value.count("\n", 0, match.start()),
                role=match.group("role"),
                target=target,
                module=module,
                class_name=class_name,
            )
        )
    return references


def _find_docstring_references(
    tree: ast.Module,
    *,
    path: str,
    module: str,
) -> tuple[list[DocstringReference], int]:
    references: list[DocstringReference] = []
    docstrings = 0

    def visit(
        node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        class_name: str | None,
    ) -> None:
        nonlocal docstrings
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings += 1
            references.extend(
                _docstring_roles(
                    body[0].value,
                    path=path,
                    module=module,
                    class_name=class_name,
                )
            )

        for child in body:
            if isinstance(child, ast.ClassDef):
                parent = class_name or module
                visit(child, f"{parent}.{child.name}")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, class_name)

    visit(tree, None)
    return references, docstrings


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


def _resolved_import_base(
    module: str, is_package: bool, node: ast.ImportFrom
) -> str | None:
    if not node.level:
        return node.module
    package = module if is_package else module.rpartition(".")[0]
    import_name = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(import_name, package)
    except (ImportError, ValueError):
        return None


def _copy_symbol_aliases(symbols: dict[str, str], target: str, alias: str) -> bool:
    kind = symbols.get(target)
    if kind is None or alias in symbols:
        return False
    symbols[alias] = kind
    return True


def build_python_index(python_roots: Sequence[Path], display_root: Path) -> PythonIndex:
    """Index Python definitions, imports, and explicit roles in docstrings."""
    symbols: dict[str, str] = {}
    imports: list[ImportRecord] = []
    imported_aliases: list[tuple[str, str, str]] = []
    exports: dict[str, tuple[str, ...]] = {}
    references: list[DocstringReference] = []
    module_names: set[str] = set()
    python_files = 0
    docstrings = 0

    for root in python_roots:
        for source in sorted(root.rglob("*.py")):
            module, is_package = _module_name(root, source)
            if not module:
                continue
            python_files += 1
            module_names.add(module)
            try:
                tree = ast.parse(
                    source.read_text(encoding="utf-8"), filename=str(source)
                )
            except (OSError, SyntaxError) as error:
                raise RuntimeError(
                    f"cannot parse Python source {source}: {error}"
                ) from error

            symbols[module] = "module"
            source_references, source_docstrings = _find_docstring_references(
                tree,
                path=_display_path(source, display_root, root),
                module=module,
            )
            references.extend(source_references)
            docstrings += source_docstrings
            for imported_node in ast.walk(tree):
                if isinstance(imported_node, ast.Import):
                    for alias in imported_node.names:
                        binding = alias.asname or alias.name.partition(".")[0]
                        imported_aliases.append((module, alias.name, binding))
                elif (
                    isinstance(imported_node, ast.ImportFrom)
                    and not imported_node.level
                    and imported_node.module
                ):
                    for alias in imported_node.names:
                        if alias.name != "*":
                            imported_aliases.append(
                                (
                                    module,
                                    f"{imported_node.module}.{alias.name}",
                                    alias.asname or alias.name,
                                )
                            )
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
                    imported_module = _resolved_import_base(module, is_package, node)
                    if not imported_module:
                        continue
                    for alias in node.names:
                        imports.append(
                            ImportRecord(
                                module=module,
                                target=(
                                    imported_module
                                    if alias.name == "*"
                                    else f"{imported_module}.{alias.name}"
                                ),
                                alias=alias.asname or alias.name,
                                star=alias.name == "*",
                            )
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        binding = alias.asname or alias.name.partition(".")[0]
                        target = alias.name if alias.asname else binding
                        imports.append(
                            ImportRecord(
                                module=module,
                                target=target,
                                alias=binding,
                            )
                        )

    package_roots = frozenset(name.partition(".")[0] for name in module_names)
    external_aliases: dict[str, set[str]] = {}
    local_aliases: dict[str, str] = {}
    for module, target, alias in imported_aliases:
        if target.partition(".")[0] not in package_roots:
            external_aliases.setdefault(module, set()).add(alias)
    local_imports: list[ImportRecord] = []
    for record in imports:
        if record.target.partition(".")[0] in package_roots:
            local_imports.append(record)
        elif not record.star:
            external_aliases.setdefault(record.module, set()).add(record.alias)

    unresolved = local_imports
    while unresolved:
        pending: list[ImportRecord] = []
        changed = False
        for record in unresolved:
            if record.star:
                names = exports.get(record.target, ())
                candidates = [
                    (f"{record.target}.{name}", f"{record.module}.{name}")
                    for name in names
                ]
            else:
                candidates = [
                    (
                        record.target,
                        f"{record.module}.{record.alias}",
                    )
                ]

            resolved = False
            for target, alias in candidates:
                if target not in symbols:
                    continue
                local_aliases.setdefault(alias, target)
                changed = _copy_symbol_aliases(symbols, target, alias) or changed
                resolved = True
            if not resolved:
                pending.append(record)
        if not changed:
            break
        unresolved = pending

    return PythonIndex(
        symbols=symbols,
        local_aliases=local_aliases,
        references=tuple(references),
        external_aliases={
            module: frozenset(aliases) for module, aliases in external_aliases.items()
        },
        package_roots=package_roots,
        python_files=python_files,
        docstrings=docstrings,
    )


def build_symbol_inventory(python_roots: Sequence[Path]) -> dict[str, str]:
    """Return Sphinx directive kinds keyed by fully qualified Python name."""
    display_root = Path.cwd().resolve()
    return build_python_index(python_roots, display_root).symbols


ROLE_KINDS = {
    "attr": frozenset({"attribute", "data"}),
    "class": frozenset({"class"}),
    "const": frozenset({"attribute", "data"}),
    "data": frozenset({"attribute", "data"}),
    "exc": frozenset({"class"}),
    "func": frozenset({"function"}),
    "meth": frozenset({"method"}),
    "mod": frozenset({"module"}),
    "obj": frozenset({"attribute", "class", "data", "function", "method", "module"}),
}


def _reference_candidates(reference: DocstringReference) -> list[str]:
    target = reference.target
    relative_target = target.lstrip(".")
    package = reference.module.rpartition(".")[0]
    candidates: list[str] = []
    if target.startswith(".") and package:
        candidates.append(f"{package}.{relative_target}")
    else:
        candidates.append(target)
    if reference.class_name:
        candidates.append(f"{reference.class_name}.{relative_target}")
    candidates.append(f"{reference.module}.{relative_target}")
    if package:
        candidates.append(f"{package}.{relative_target}")
    root = reference.module.partition(".")[0]
    candidates.append(f"{root}.{relative_target}")
    return list(dict.fromkeys(candidates))


def _builtin_kind(target: str) -> str | None:
    if "." in target or not hasattr(builtins, target):
        return None
    value = getattr(builtins, target)
    if isinstance(value, type):
        return "class"
    if callable(value):
        return "function"
    return "data"


def _indexed_kind(index: PythonIndex, name: str) -> str | None:
    kind = index.symbols.get(name)
    expanded = name
    for _ in range(10):
        if kind is not None:
            return kind
        aliases = [
            (alias, target)
            for alias, target in index.local_aliases.items()
            if expanded == alias or expanded.startswith(f"{alias}.")
        ]
        if not aliases:
            return None
        alias, target = max(aliases, key=lambda item: len(item[0]))
        expanded = target + expanded[len(alias) :]
        kind = index.symbols.get(expanded)
    return kind


def check_docstring_references(
    index: PythonIndex,
) -> tuple[list[BrokenReference], list[DocstringReference]]:
    """Resolve local docstring roles and separate unindexed external targets."""
    broken: list[BrokenReference] = []
    external: list[DocstringReference] = []
    for reference in index.references:
        allowed_kinds = ROLE_KINDS[reference.role]
        found_kinds = {
            kind
            for candidate in _reference_candidates(reference)
            if (kind := _indexed_kind(index, candidate)) is not None
        }
        suffix = reference.target.lstrip(".")
        found_kinds.update(
            kind for name, kind in index.symbols.items() if name.endswith(f".{suffix}")
        )
        builtin_kind = _builtin_kind(reference.target)
        if builtin_kind is not None:
            found_kinds.add(builtin_kind)
        if found_kinds & allowed_kinds:
            continue

        first_component = reference.target.lstrip(".").partition(".")[0]
        imported_external = first_component in index.external_aliases.get(
            reference.module, frozenset()
        )
        standard_library = first_component in sys.stdlib_module_names
        if not found_kinds and (imported_external or standard_library):
            external.append(reference)
            continue

        broken.append(
            BrokenReference(
                path=reference.path,
                line=reference.line,
                role=reference.role,
                target=reference.target,
                source="docstring",
                found_kinds=tuple(sorted(found_kinds)),
            )
        )
    return (
        sorted(broken, key=lambda item: (item.path, item.line, item.target)),
        sorted(external, key=lambda item: (item.path, item.line, item.target)),
    )


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
        display_path = _display_path(source_path, platform_root, docs_root)
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
    index: PythonIndex,
    markdown_files: int,
    markdown_references: int,
    external_references: Sequence[DocstringReference],
    broken: Sequence[BrokenReference],
) -> str:
    markdown_broken = sum(item.source == "markdown" for item in broken)
    docstring_broken = [item for item in broken if item.source == "docstring"]
    role_mismatches = sum(bool(item.found_kinds) for item in docstring_broken)
    lines = [
        "rocKE documentation symbol report",
        f"docs root: {docs_root}",
        f"Markdown files: {markdown_files}",
        f"Markdown symbol references: {markdown_references}",
        f"Python files: {index.python_files}",
        f"Python docstrings: {index.docstrings}",
        f"indexed Python symbols: {len(index.symbols)}",
        f"Python docstring symbol references: {len(index.references)}",
        f"external docstring references skipped: {len(external_references)}",
        f"broken Markdown references: {markdown_broken}",
        f"broken Python docstring references: {len(docstring_broken)}",
        f"docstring targets not found: {len(docstring_broken) - role_mismatches}",
        f"docstring role mismatches: {role_mismatches}",
        f"broken local references: {len(broken)}",
    ]
    if broken:
        lines.append("")
        for item in broken:
            location = f"{item.path}:{item.line}: error:"
            if item.found_kinds:
                found = ", ".join(item.found_kinds)
                lines.append(
                    f"{location} py:{item.role} target '{item.target}' has "
                    f"incompatible kind {found} ({item.source})"
                )
            else:
                lines.append(
                    f"{location} unresolved local py:{item.role} target "
                    f"'{item.target}' ({item.source})"
                )
    return "\n".join(lines) + "\n"


def _json_report(
    docs_root: Path,
    index: PythonIndex,
    markdown_files: int,
    markdown_references: int,
    external_references: Sequence[DocstringReference],
    broken: Sequence[BrokenReference],
) -> str:
    markdown_broken = sum(item.source == "markdown" for item in broken)
    docstring_broken = [item for item in broken if item.source == "docstring"]
    role_mismatches = sum(bool(item.found_kinds) for item in docstring_broken)
    payload = {
        "broken_markdown_references": markdown_broken,
        "broken_python_docstring_references": len(docstring_broken),
        "broken_references": [asdict(item) for item in broken],
        "docs_root": str(docs_root),
        "external_docstring_references_skipped": [
            asdict(item) for item in external_references
        ],
        "indexed_python_symbols": len(index.symbols),
        "markdown_files": markdown_files,
        "markdown_reference_count": markdown_references,
        "python_docstring_reference_count": len(index.references),
        "python_docstring_role_mismatches": role_mismatches,
        "python_docstring_targets_not_found": len(docstring_broken) - role_mismatches,
        "python_docstrings": index.docstrings,
        "python_files": index.python_files,
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
        index = build_python_index(python_roots, platform_root)
        inventory = render_inventory(index.symbols)
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
            markdown_broken = parse_missing_references(
                warnings,
                source_root / "docs",
                docs_root,
                platform_root,
                reference_locations,
            )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    docstring_broken, external_references = check_docstring_references(index)
    broken = sorted(
        [*markdown_broken, *docstring_broken],
        key=lambda item: (item.path, item.line, item.target),
    )
    markdown_reference_count = len(reference_locations)
    total_reference_count = markdown_reference_count + len(index.references)
    try:
        report_docs_root = docs_root.relative_to(platform_root)
    except ValueError:
        report_docs_root = docs_root
    report = (
        _json_report(
            report_docs_root,
            index,
            markdown_files,
            markdown_reference_count,
            external_references,
            broken,
        )
        if args.format == "json"
        else _text_report(
            report_docs_root,
            index,
            markdown_files,
            markdown_reference_count,
            external_references,
            broken,
        )
    )
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"wrote {args.format} report to {output}")
        print(
            f"checked {total_reference_count} symbol references; "
            f"found {len(broken)} broken local references"
        )
    else:
        print(report, end="")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
