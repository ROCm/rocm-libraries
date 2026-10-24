# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-file metrics for the SLO linter (SLO101-SLO105, SLO107). Pure stdlib `ast`.

The IR builder is found by type, not only by the name `b`: renaming the builder
or storing it on `self` must not hide raw IR calls.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import slo_config as C
from slo_patterns import count_patterns


def sloc_of(src: str, tree: ast.AST) -> int:
    """Non-blank lines that are not comment-only and not inside a docstring."""
    doc_lines: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(
            n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = n.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return sum(
        1
        for i, line in enumerate(src.splitlines(), 1)
        if line.strip() and not line.strip().startswith("#") and i not in doc_lines
    )


def _package_of(rel: str) -> list[str]:
    """Dotted package of a platform file, for resolving relative imports."""
    marker = "platform/python/"
    return rel.split(marker, 1)[1].split("/")[:-1] if marker in rel else []


def _helper_imports(tree: ast.AST, rel: str, helper_modules: set[str]):
    """Names imported from rocke.helpers: (functions/classes, module aliases)."""
    names: set[str] = set()
    mods: set[str] = set()
    pkg = _package_of(rel)
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if n.level:
                base = pkg[: len(pkg) - (n.level - 1)] if n.level > 1 else pkg
                mod = ".".join(base + ([mod] if mod else []))
            if mod == "rocke.helpers" or mod.startswith("rocke.helpers."):
                for a in n.names:
                    target = (
                        mods
                        if (mod == "rocke.helpers" and a.name in helper_modules)
                        else names
                    )
                    target.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("rocke.helpers"):
                    mods.add(a.asname or a.name.split(".")[-1])
    return names, mods


def _mentions_irbuilder(ann: ast.AST | None) -> bool:
    """True if the annotation IS an IRBuilder: `IRBuilder`, `ir.IRBuilder`,
    `"IRBuilder"`, `Optional[IRBuilder]`, `IRBuilder | None`. Not when IRBuilder
    only appears inside something else, like `Callable[[IRBuilder], Value]`."""
    if ann is None:
        return False
    if isinstance(ann, ast.Name):
        return ann.id == "IRBuilder"
    if isinstance(ann, ast.Attribute):
        return ann.attr == "IRBuilder"
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            return _mentions_irbuilder(ast.parse(ann.value, mode="eval").body)
        except SyntaxError:
            return False
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _mentions_irbuilder(ann.left) or _mentions_irbuilder(ann.right)
    if isinstance(ann, ast.Subscript) and _decorator_name(ann.value) == "Optional":
        return _mentions_irbuilder(ann.slice)
    return False


def _builder_receivers(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Variable names and `self.<attr>` names that hold an IR builder.

    Found by type, not only by the name `b`: parameters annotated IRBuilder,
    `x = IRBuilder(...)`, plain aliases `x = b`, and `self.x = <builder>`.
    """
    names = set(C.BUILDER_NAMES)
    attrs: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                if _mentions_irbuilder(arg.annotation):
                    names.add(arg.arg)
        elif (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and _mentions_irbuilder(n.annotation)
        ):
            names.add(n.target.id)
    changed = True
    while changed:
        changed = False
        for n in ast.walk(tree):
            if not isinstance(n, ast.Assign):
                continue
            v = n.value
            is_builder = (isinstance(v, ast.Call) and _mentions_irbuilder(v.func)) or (
                isinstance(v, ast.Name) and v.id in names
            )
            if not is_builder:
                continue
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id not in names:
                    names.add(t.id)
                    changed = True
                elif (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                    and t.attr not in attrs
                ):
                    attrs.add(t.attr)
                    changed = True
    return names, attrs


def _spec_bools(tree: ast.AST) -> list[dict]:
    """Bool fields per frozen dataclass (SLO105)."""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.ClassDef):
            continue
        frozen = False
        for d in n.decorator_list:
            if isinstance(d, ast.Call) and _decorator_name(d.func) == "dataclass":
                frozen = any(
                    k.arg == "frozen"
                    and isinstance(k.value, ast.Constant)
                    and k.value.value is True
                    for k in d.keywords
                )
        if not frozen:
            continue
        bools = sum(
            1
            for s in n.body
            if isinstance(s, ast.AnnAssign)
            and (
                (isinstance(s.annotation, ast.Name) and s.annotation.id == "bool")
                or (
                    isinstance(s.annotation, ast.Constant)
                    and s.annotation.value == "bool"
                )
            )
        )
        out.append({"name": n.name, "line": n.lineno, "bools": bools})
    return out


def _decorator_name(f: ast.AST) -> str:
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def analyze_source(src: str, rel: str, helper_modules: set[str]) -> dict:
    tree = ast.parse(src)
    helper_names, helper_mods = _helper_imports(tree, rel, helper_modules)
    recv_names, recv_attrs = _builder_receivers(tree)

    def builder_call(call: ast.Call) -> str | None:
        f = call.func
        if not isinstance(f, ast.Attribute):
            return None
        v = f.value
        if isinstance(v, ast.Name) and v.id in recv_names:
            return f.attr
        if (
            isinstance(v, ast.Attribute)
            and isinstance(v.value, ast.Name)
            and v.value.id == "self"
            and v.attr in recv_attrs
        ):
            return f.attr
        return None

    def is_helper_ref(node: ast.AST) -> bool:
        """`H`, `mod.H` -- a name that comes from rocke.helpers."""
        if isinstance(node, ast.Name):
            return node.id in helper_names or node.id in helper_mods
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in helper_mods
        )

    # Objects made by a helper: `x = H(...)`, `x = H.from_grid(...)`, `x = mod.H(...)`.
    helper_objs: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            if is_helper_ref(f) or (
                isinstance(f, ast.Attribute) and is_helper_ref(f.value)
            ):
                helper_objs.update(t.id for t in n.targets if isinstance(t, ast.Name))

    def helper_key(f: ast.AST) -> str:
        """Readable name of a helper call, for the per-file histogram."""
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            v = f.value
            if isinstance(v, ast.Name):
                return f"{v.id}.{f.attr}"
            if isinstance(v, ast.Call):
                return f"{helper_key(v.func)}(...).{f.attr}"
        return "?"

    raw = helper = 0
    ops: Counter = Counter()
    helpers: Counter = Counter()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        op = builder_call(n)
        if op is not None:
            raw += 1
            ops[op] += 1
        elif (
            isinstance(f, ast.Name)
            and f.id.startswith("const_")
            and f.id not in helper_names
        ):
            raw += 1
            ops[f.id] += 1
        elif is_helper_ref(f):
            helper += 1
            helpers[helper_key(f)] += 1
        elif isinstance(f, ast.Attribute) and (
            (isinstance(f.value, ast.Name) and f.value.id in helper_objs)
            or (
                isinstance(f.value, ast.Call)
                and (
                    is_helper_ref(f.value.func)
                    or (
                        isinstance(f.value.func, ast.Attribute)
                        and is_helper_ref(f.value.func.value)
                    )
                )
            )
            or is_helper_ref(f.value)
        ):
            helper += 1
            helpers[helper_key(f)] += 1

    fns = sorted(
        (
            (n.name, n.lineno, n.end_lineno - n.lineno + 1)
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        key=lambda t: -t[2],
    )
    sloc = sloc_of(src, tree)
    specs = _spec_bools(tree)
    worst_spec = max(specs, key=lambda s: s["bools"]) if specs else None

    def ratio(h, r):
        return round(h / (h + r), 4) if h + r else None

    return {
        "path": rel,
        "loc": len(src.splitlines()),
        "sloc": sloc,
        "n_funcs": len(fns),
        "longest_fn": fns[0][2] if fns else 0,
        "longest_fn_name": fns[0][0] if fns else "",
        "longest_fn_line": fns[0][1] if fns else 1,
        "long_funcs": [
            {"name": a, "line": b, "len": c}
            for a, b, c in fns
            if c > C.TARGETS["longest_fn"][0]
        ],
        "raw_ops": raw,
        "raw_per_100": round(100 * raw / sloc, 2) if sloc else 0,
        "helper_calls": helper,
        "hcr": ratio(helper, raw),
        "builder_names": sorted(recv_names - set(C.BUILDER_NAMES))
        + [f"self.{a}" for a in sorted(recv_attrs)],
        "spec_bools_max": worst_spec["bools"] if worst_spec else 0,
        "spec_bools_class": worst_spec["name"] if worst_spec else "",
        "spec_bools_line": worst_spec["line"] if worst_spec else 1,
        "patterns": dict(count_patterns(tree, builder_call)),
        "top_ops": ops.most_common(10),
        "ops": dict(ops.most_common()),
        "helpers": dict(helpers.most_common()),
        "funcs": [{"name": a, "line": b, "len": c} for a, b, c in fns if c >= 60],
    }


def helper_modules_of(root: Path) -> set[str]:
    return {p.stem for p in (root / "platform/python/rocke/helpers").glob("*.py")}


def in_scope(rel: str, scope=None) -> bool:
    parts = rel.split("/")
    if any(p in C.EXCLUDE_PARTS for p in parts) or parts[-1].startswith("test_"):
        return False
    return any(rel == s or rel.startswith(s + "/") for s in (scope or C.SCOPE))


def scope_files(root: Path, scope=None) -> list[str]:
    out = []
    for s in scope or C.SCOPE:
        for p in sorted((root / s).rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if in_scope(rel, scope):
                out.append(rel)
    return out
