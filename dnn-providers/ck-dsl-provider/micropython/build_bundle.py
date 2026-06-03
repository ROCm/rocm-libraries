#!/usr/bin/env python3
# Build a MicroPython-runnable bundle of ck_dsl (run under CPython 3.12).
#
#   1. copy ck_dsl -> spike/mp1/ckbundle/ck_dsl   (original tree untouched)
#   2. AST transform: inside every @dataclass class, give each field an explicit
#      `= field(...)` value so it lands in __dict__ (MicroPython erases bare
#      class annotations). Required for the custom dataclasses shim.
#   3. trim the heavy package __init__.py files (they eager-import the world).
#
# This mirrors what a real "MicroPython bundle" build step would do.
import ast
import os
import shutil

# Paths are supplied by the CMake freeze pipeline (build_embed.sh) via the environment:
#   CK_DSL_SRC          - source ck_dsl package (.../python/ck_dsl)
#   CK_DSL_PROVIDER_SRC - source ck_dsl_provider package (.../python/ck_dsl_provider)
#   BUNDLE_DIR          - output dir to hold the transformed ck_dsl + ck_dsl_provider
SRC = os.environ["CK_DSL_SRC"]
PROVIDER_SRC = os.environ["CK_DSL_PROVIDER_SRC"]
BUNDLE_DIR = os.environ["BUNDLE_DIR"]
DST = os.path.join(BUNDLE_DIR, "ck_dsl")
PROVIDER_DST = os.path.join(BUNDLE_DIR, "ck_dsl_provider")

# Heavy package __init__s replaced with empty stubs (eager-import roots).
TRIM_INITS = [
    "__init__.py",
    "helpers/__init__.py",
    "instances/__init__.py",
    "runtime/__init__.py",
    "analysis/__init__.py",
    "benchmark/__init__.py",
]


def _is_dataclass_decorated(node):
    for d in node.decorator_list:
        n = d.func if isinstance(d, ast.Call) else d
        if isinstance(n, ast.Name) and n.id == "dataclass":
            return True
        if isinstance(n, ast.Attribute) and n.attr == "dataclass":
            return True
    return False


def _is_classvar(ann):
    t = ann.value if isinstance(ann, ast.Subscript) else ann
    if isinstance(t, ast.Name):
        return t.id == "ClassVar"
    if isinstance(t, ast.Attribute):
        return t.attr == "ClassVar"
    return False


def _is_field_call(value):
    if not isinstance(value, ast.Call):
        return False
    f = value.func
    return (isinstance(f, ast.Name) and f.id == "field") or (
        isinstance(f, ast.Attribute) and f.attr == "field"
    )


def _is_mutable_literal(value):
    return isinstance(value, (ast.List, ast.Dict, ast.Set))


def _field_call(default=None, factory=None):
    kw = []
    if factory is not None:
        kw.append(ast.keyword(arg="default_factory", value=factory))
    elif default is not None:
        kw.append(ast.keyword(arg="default", value=default))
    return ast.Call(func=ast.Name(id="field", ctx=ast.Load()), args=[], keywords=kw)


def _name(n):
    return ast.Name(id=n, ctx=ast.Load())


def _rewrite_seq(elts, as_tuple):
    # [a, *b, c] -> [a] + list(b) + [c]   (MicroPython lacks PEP-448 displays)
    parts = []
    cur = []
    for e in elts:
        if isinstance(e, ast.Starred):
            if cur:
                parts.append(ast.List(elts=cur, ctx=ast.Load()))
                cur = []
            parts.append(ast.Call(func=_name("list"), args=[e.value], keywords=[]))
        else:
            cur.append(e)
    if cur:
        parts.append(ast.List(elts=cur, ctx=ast.Load()))
    if not parts:
        expr = ast.List(elts=[], ctx=ast.Load())
    else:
        expr = parts[0]
        for p in parts[1:]:
            expr = ast.BinOp(left=expr, op=ast.Add(), right=p)
    if as_tuple:
        expr = ast.Call(func=_name("tuple"), args=[expr], keywords=[])
    return expr


class _Xform(ast.NodeTransformer):
    def __init__(self):
        self.changed = False

    def visit_List(self, node):
        self.generic_visit(node)
        if any(isinstance(e, ast.Starred) for e in node.elts):
            self.changed = True
            return _rewrite_seq(node.elts, as_tuple=False)
        return node

    def visit_Tuple(self, node):
        self.generic_visit(node)
        # Only Load-context tuples are displays; Store-context (a, *b = x) is fine.
        if isinstance(node.ctx, ast.Load) and any(
            isinstance(e, ast.Starred) for e in node.elts
        ):
            self.changed = True
            return _rewrite_seq(node.elts, as_tuple=True)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        # MicroPython open() rejects str subclasses (our pathlib.Path) — coerce.
        if isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
            node.args[0] = ast.Call(func=_name("str"), args=[node.args[0]], keywords=[])
            self.changed = True
        # os.environ.get(...) -> os.getenv(...) (MicroPython os has no environ).
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "get"
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "environ"
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id == "os"
        ):
            node.func = ast.Attribute(value=_name("os"), attr="getenv", ctx=ast.Load())
            self.changed = True
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)  # handle nested classes
        if not _is_dataclass_decorated(node):
            return node
        for i, stmt in enumerate(node.body):
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            if _is_classvar(stmt.annotation):
                continue
            v = stmt.value
            if v is None:
                stmt.value = _field_call()
                self.changed = True
            elif _is_field_call(v):
                continue
            elif _is_mutable_literal(v):
                # bare mutable default -> default_factory=lambda: <literal>
                stmt.value = _field_call(
                    factory=ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[],
                            vararg=None,
                            kwonlyargs=[],
                            kw_defaults=[],
                            kwarg=None,
                            defaults=[],
                        ),
                        body=v,
                    )
                )
                self.changed = True
            else:
                stmt.value = _field_call(default=v)
                self.changed = True
        return node


def _ensure_field_import(tree):
    # If module already imports `field` from dataclasses, fine. Otherwise add it.
    has_field = False
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "dataclasses":
            if any(a.name == "field" for a in n.names):
                has_field = True
            else:
                n.names.append(ast.alias(name="field", asname=None))
                has_field = True
    if not has_field:
        tree.body.insert(
            0,
            ast.ImportFrom(
                module="dataclasses",
                names=[ast.alias(name="field", asname=None)],
                level=0,
            ),
        )


def _transform_file(path):
    with open(path) as f:
        src = f.read()
    # MicroPython's embed `object` has no __setattr__; ck_dsl uses it in hand-written
    # frozen __init__s. Since the shim doesn't enforce immutability, plain setattr is
    # equivalent: object.__setattr__(self, n, v) -> setattr(self, n, v).
    src2 = src.replace("object.__setattr__(", "setattr(")
    # Embed MicroPython's builtin `time` has no perf_counter (and a builtin can't be
    # shadowed by a frozen shim). compile_kernel uses it only for diagnostic timings,
    # so zero them out: time.perf_counter() -> 0.0.
    src2 = src2.replace("time.perf_counter()", "0.0")
    text_changed = src2 != src
    src = src2
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print("  SKIP (parse error):", path, e)
        return
    x = _Xform()
    x.visit(tree)
    if x.changed:
        _ensure_field_import(tree)
    if x.changed or text_changed:
        ast.fix_missing_locations(tree)
        with open(path, "w") as f:
            f.write(ast.unparse(tree))


def main():
    if os.path.exists(DST):
        shutil.rmtree(os.path.dirname(DST))
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    shutil.copytree(SRC, DST)
    # Provider package alongside ck_dsl (skip any __pycache__).
    shutil.copytree(
        PROVIDER_SRC, PROVIDER_DST, ignore=shutil.ignore_patterns("__pycache__")
    )

    n = 0
    for root in (DST, PROVIDER_DST):
        for dirpath, _, files in os.walk(root):
            if "/examples/" in dirpath + "/" or dirpath.endswith("/examples"):
                continue
            for fn in files:
                if fn.endswith(".py"):
                    _transform_file(os.path.join(dirpath, fn))
                    n += 1
    print("transformed %d .py files" % n)

    for rel in TRIM_INITS:
        p = os.path.join(DST, rel)
        if os.path.exists(p):
            with open(p, "w") as f:
                f.write(
                    "# trimmed for MicroPython bundle (eager-import root removed)\n"
                )
    print("trimmed %d package __init__s" % len(TRIM_INITS))
    print("bundle at", DST)


if __name__ == "__main__":
    main()
