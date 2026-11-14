# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Keys and source stamps of recorded facts (data/facts/<module>.json).

A function's facts are keyed by its CPython qualname, with no line number, so
an edit above a function does not lose its types. Two code objects can share
a qualname: generator expressions, lambdas and comprehensions always carry
`#<n>`, their 1-based position in source order among the same qualname;
a named function gets `#<n>` only when its qualname repeats in the module (a
property setter, a conditional redefinition).

Each entry also carries `src`, a hash of the function's AST with docstrings
dropped (line numbers are not part of it). When the Python changes after the
recording, the translator warns that the facts are stale; it does not change
its output.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import sys

# nodes that compile to their own code object, and the name CPython gives it
_ANON = {
    ast.Lambda: "<lambda>",
    ast.GeneratorExp: "<genexpr>",
    ast.ListComp: "<listcomp>",
    ast.SetComp: "<setcomp>",
    ast.DictComp: "<dictcomp>",
}
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _code_nodes(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """(co_qualname, node) of every node with its own code object, in source order."""
    out = []

    def visit(n, prefix):
        if isinstance(n, _DEFS):
            q = f"{prefix}{n.name}"
            out.append((q, n))
            # decorators, defaults and annotations belong to the enclosing scope
            for d in n.decorator_list:
                visit(d, prefix)
            visit(n.args, prefix)
            if n.returns is not None:
                visit(n.returns, prefix)
            for b in n.body:
                visit(b, q + ".<locals>.")
        elif isinstance(n, ast.ClassDef):
            q = f"{prefix}{n.name}"
            out.append((q, n))  # the class body is a code object too
            for d in n.decorator_list + n.bases + [k.value for k in n.keywords]:
                visit(d, prefix)
            for b in n.body:
                visit(b, q + ".")
        elif isinstance(n, ast.Lambda):
            q = f"{prefix}<lambda>"
            out.append((q, n))
            visit(n.args, prefix)
            visit(n.body, q + ".<locals>.")
        elif type(n) in _ANON:
            if _own_code(n):
                q = f"{prefix}{_ANON[type(n)]}"
                out.append((q, n))
                inner = q + "."
            else:
                inner = prefix  # an inlined comprehension (PEP 709)
            # the first iterable is evaluated in the enclosing scope
            visit(n.generators[0].iter, prefix)
            for i, g in enumerate(n.generators):
                if i:
                    visit(g.iter, inner)
                visit(g.target, inner)
                for c in g.ifs:
                    visit(c, inner)
            for e in (n.key, n.value) if isinstance(n, ast.DictComp) else (n.elt,):
                visit(e, inner)
        else:
            for c in ast.iter_child_nodes(n):
                visit(c, prefix)

    visit(tree, "")
    return out


def _own_code(node) -> bool:
    """A comprehension has its own code object before Python 3.12 (PEP 709);
    a generator expression always does."""
    return isinstance(node, ast.GeneratorExp) or sys.version_info < (3, 12)


def code_keys(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """(facts key, node) of every code object in a module, in source order."""
    nodes = _code_nodes(tree)
    count: dict[str, int] = {}
    for q, _ in nodes:
        count[q] = count.get(q, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for q, n in nodes:
        seen[q] = seen.get(q, 0) + 1
        anon = q.rsplit(".", 1)[-1].startswith("<")
        out.append((f"{q}#{seen[q]}" if anon or count[q] > 1 else q, n))
    return out


def first_line(node: ast.AST) -> int:
    """co_firstlineno: a decorated function's code starts at its first decorator."""
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _strip_docstrings(node: ast.AST) -> ast.AST:
    node = copy.deepcopy(node)
    for n in ast.walk(node):
        body = getattr(n, "body", None)
        if (
            isinstance(n, (*_DEFS, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            n.body = body[1:] or [ast.Pass()]
    return node


def src_hash(node: ast.AST) -> str:
    """Stamp of a function's source: its AST without docstrings or positions."""
    return hashlib.sha256(ast.dump(_strip_docstrings(node)).encode()).hexdigest()[:12]
