# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Facts keys (qualname, no line number) and the staleness stamp."""

from __future__ import annotations

import ast
import collections
import json
import re
import shutil
import sys

import pytest
from trtest import FIXTURES, GOLDEN, TR, Tr, normalize

sys.path.insert(0, str(TR))
from tr.factkeys import code_keys, first_line, src_hash  # noqa: E402
from tr.rekey import rekey  # noqa: E402

COMMON = FIXTURES / "library" / "kernels" / "common"
VALU = "_attention_bwd_valu"

SNIPPET = """
class C:
    @property
    def x(self):
        return 1

    @x.setter
    def x(self, v):
        pass

    f = lambda self: [lambda: 0 for _ in range(2)]


def f(a=(i for i in range(3))):
    g = (j for j in (k for k in a))
    h = [lambda q: (z for z in q) for _ in a]
    d = {k: (lambda: k) for k in a}

    def inner():
        return sorted(a, key=lambda t: t)

    return g, h, d, inner
"""


def _code_objects(co):
    for c in co.co_consts:
        if hasattr(c, "co_qualname"):
            yield c
            yield from _code_objects(c)


@pytest.mark.parametrize(
    "name", ["<snippet>"] + sorted(p.name for p in COMMON.glob("*.py"))
)
def test_keys_follow_cpython(name):
    """Every code object CPython compiles has exactly one key, at its line."""
    src = SNIPPET if name == "<snippet>" else (COMMON / name).read_text()
    real = collections.Counter(
        (c.co_qualname, c.co_firstlineno)
        for c in _code_objects(compile(src, name, "exec"))
    )
    keys = code_keys(ast.parse(src))
    mine = collections.Counter((k.split("#")[0], first_line(n)) for k, n in keys)
    assert real == mine
    assert len({k for k, _ in keys}) == len(keys), "keys are unique"


def test_key_ordinals():
    keys = [k for k, _ in code_keys(ast.parse(SNIPPET))]
    # a repeated named function and every anonymous code object are numbered
    assert {"C.x#1", "C.x#2", "C.<lambda>#1", "f", "f.<locals>.inner"} <= set(keys)
    # the inner generator is the outer one's first iterable: both run in f
    assert [k for k in keys if k.startswith("f.<locals>.<genexpr>")] == [
        "f.<locals>.<genexpr>#1",
        "f.<locals>.<genexpr>#2",
    ]
    assert "<genexpr>#1" in keys  # a default value: the module scope


def _hash(src: str) -> str:
    return src_hash(ast.parse(src).body[0])


def test_src_hash():
    base = _hash("def f(a):\n    return a + 1\n")
    assert _hash('\n\n\ndef f(a):\n    """Doc."""\n    return a + 1\n') == base
    assert _hash("def f(a):\n    # a comment\n    return a + 1\n") == base
    assert _hash("def f(a):\n    return a + 2\n") != base


def _old_format(facts: dict, src: str) -> dict:
    """The same facts keyed the way tr.record wrote them before: qualname@line."""
    line = {k: first_line(n) for k, n in code_keys(ast.parse(src))}
    funcs = {}
    for k, v in facts["funcs"].items():
        v = {f: x for f, x in v.items() if f != "src"}
        funcs[f"{k.split('#')[0]}@{line[k]}"] = v
    return dict(facts, funcs=funcs)


@pytest.mark.parametrize("module", [VALU, "_attention_bwd_counted"])
def test_rekey_converts_old_keys(module):
    facts = json.loads((FIXTURES / "data" / "facts" / f"{module}.json").read_text())
    src = (COMMON / f"{module}.py").read_text()
    out, left = rekey(_old_format(facts, src), src)
    assert left == []
    assert out == facts


def _translate(tr: Tr, abi, out) -> tuple[str, str]:
    p = tr.run("translate", VALU, "--abi", str(abi), "--out", str(out))
    return out.read_text(), p.stderr


def test_old_keys_still_read(generated, tmp_path):
    """Facts files written before the qualname keys translate the same."""
    out, _runs = generated
    data = tmp_path / "data"
    shutil.copytree(FIXTURES / "data", data)
    p = data / "facts" / f"{VALU}.json"
    src = (COMMON / f"{VALU}.py").read_text()
    p.write_text(json.dumps(_old_format(json.loads(p.read_text()), src)))
    text, err = _translate(
        Tr(FIXTURES / "library", data), out / "abi", tmp_path / "v.cpp"
    )
    assert normalize(text) == normalize((GOLDEN / f"{VALU}.cpp.golden").read_text())
    assert "facts stale" not in err


def test_line_shift_keeps_types(generated, copy_library, tmp_path):
    """Lines inserted above every function: same translation, no warning
    (with qualname@line keys the types were lost)."""
    out, _runs = generated
    mod = copy_library / "kernels" / "common" / f"{VALU}.py"
    mod.write_text("\n" * 5 + mod.read_text())
    text, err = _translate(Tr(copy_library), out / "abi", tmp_path / "v.cpp")
    assert "facts stale" not in err
    # the /* py: */ comments cite the shifted lines
    text = re.sub(rf"({VALU}\.py:)(\d+)\)", lambda m: f"{m[1]}{int(m[2]) - 5})", text)
    assert normalize(text) == normalize((GOLDEN / f"{VALU}.cpp.golden").read_text())


def test_stale_facts_warn(generated, copy_library, tmp_path):
    """A changed function body warns for that function only."""
    out, _runs = generated
    mod = copy_library / "kernels" / "common" / f"{VALU}.py"
    src = mod.read_text()
    edited = src.replace("n must be even", "n must be an even number")
    assert edited != src
    mod.write_text(edited)
    _text, err = _translate(Tr(copy_library), out / "abi", tmp_path / "v.cpp")
    stale = [ln for ln in err.splitlines() if "facts stale" in ln]
    assert stale == [
        f"warning: facts stale for {VALU}.rne_noguard_bf16; re-run tr.record"
    ]
