# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Facts on a kernel parameter's declaration (``IRBuilder.param``).

An integer parameter states ``ge`` / ``gt`` / ``le`` / ``lt`` / ``eq`` /
``multiple_of``; the host will check them before every launch, and the GPU
trusts them. A bound is an int, an earlier parameter, or a ``sum_of`` /
``product_of`` / ``quotient_of`` / ``difference_of`` over those, stored as an
int or as the canonical JSON of its intexpr tree. These tests pin:

- what the builder stores and every refusal, with its exact text;
- the verifier's rules over stored facts, which also catch IR text the
  builder never wrote (a relation naming a later parameter);
- ``print_ir``, ``serialize`` / ``parse`` and ``canonicalize`` (which renames
  relation leaves by walking the tree);
- ``stated_facts_of``, which resolves Value bounds through their own facts;
- that the facts change no emitted byte, survive ``optimize_kernel``, and are
  what the recording builder records.

The C++ side runs in the ``rocke_param_facts`` ctest. With ``--ir`` that
program prints the C++ ``print_ir``, verifier messages and ``canonicalize``
for one serialized kernel; when ``ROCKE_PARAM_FACTS_CLI`` names it, every
kernel in ``IR_CASES`` goes through both engines and must print the same.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rocke.core.facts import StatedFacts, stated_facts_of
from rocke.core.intexpr import difference_of, product_of, quotient_of, sum_of
from rocke.core.ir import F16, I32, I64, IRBuilder, PtrType
from rocke.core.ir_print import print_ir
from rocke.core.ir_serialize import canonicalize, parse, serialize
from rocke.core.passes import optimize_kernel
from rocke.core.verify import verify
from rocke.portable_ir.src.recording_builder import record_kernel
from rocke.tests.engines import lower_with_both_engines

MAX_EXTENT = 1 << 20


def _gemm_params(b: IRBuilder):
    """The batched-GEMM declarations of the API example: ints, a relation on
    an earlier parameter, a product and a quotient."""
    M = b.param("M", I32, ge=1, le=MAX_EXTENT, multiple_of=64)
    N = b.param("N", I32, ge=1, le=MAX_EXTENT)
    ldc = b.param("ldc", I32, ge=N, le=MAX_EXTENT, multiple_of=8)
    sb = b.param("sb", I32, ge=product_of(M, ldc))
    C = b.param("C", PtrType(I32, "global"), noalias=True, align=16)
    rows = b.param("rows", I32, ge=0, le=quotient_of(sb, M))
    return M, N, ldc, sb, C, rows


def _gemm_kernel():
    b = IRBuilder("param_facts_gemm")
    b.kernel.attrs["max_workgroup_size"] = 64
    M, _, _, _, C, rows = _gemm_params(b)
    tid = b.thread_id_x()
    b.global_store(C, tid, b.add(rows, M))
    b.ret()
    return b.kernel


# --------------------------------------------------------------------------
# what the builder stores
# --------------------------------------------------------------------------


def test_the_builder_stores_ints_and_canonical_relations():
    b = IRBuilder("stored")
    _gemm_params(b)
    attrs = {p.name: p.attrs for p in b.kernel.params}
    assert attrs["M"] == {"ge": 1, "le": MAX_EXTENT, "multiple_of": 64}
    assert attrs["ldc"] == {"ge": '{"param":"N"}', "le": MAX_EXTENT, "multiple_of": 8}
    assert attrs["sb"] == {"ge": '{"mul":[{"param":"M"},{"param":"ldc"}]}'}
    assert attrs["C"] == {"noalias": True, "align": 16}
    assert attrs["rows"] == {"ge": 0, "le": '{"div":[{"param":"sb"},{"param":"M"}]}'}


def test_a_parameter_value_links_its_declaration():
    b = IRBuilder("linked")
    _, _, ldc, *_ = _gemm_params(b)
    assert ldc.param is b.kernel.params[2]
    assert [p.name for p in ldc.param.declared_before] == ["M", "N"]
    assert b.thread_id_x().param is None


def test_sums_and_differences_are_stored_as_their_trees():
    b = IRBuilder("sums")
    total = b.param("total", I32, ge=0, le=4096)
    a = b.param("a", I32, ge=0)
    c = b.param("c", I32, ge=1)
    b.param("b_len", I32, ge=0, le=difference_of(quotient_of(total, c), a))
    s = b.param("s", I32, ge=sum_of(total, a, c))
    assert b.kernel.params[3].attrs["le"] == (
        '{"sub":[{"div":[{"param":"total"},{"param":"c"}]},{"param":"a"}]}'
    )
    assert s.param.attrs["ge"] == (
        '{"add":[{"add":[{"param":"total"},{"param":"a"}]},{"param":"c"}]}'
    )


# --------------------------------------------------------------------------
# what the builder refuses
# --------------------------------------------------------------------------


def _three(b: IRBuilder):
    """M with a lower bound, sb with none, and a computed value k0."""
    M = b.param("M", I32, ge=1)
    sb = b.param("sb", I32)
    return M, sb, b.add(M, M)


# (builds the call on a fresh builder, error type, exact message). The C++
# builder refuses the declarations it can express with the same texts
# (test_param_facts.cpp).
REFUSALS = [
    (
        lambda b, M, sb, k0: b.param("ldc", I32, lt=4096, le=4095),
        ValueError,
        "param ldc: states two upper bounds (lt, le) -- state one",
    ),
    (
        lambda b, M, sb, k0: b.param("lo", I32, gt=0, ge=1),
        ValueError,
        "param lo: states two lower bounds (gt, ge) -- state one",
    ),
    (
        lambda b, M, sb, k0: b.param("D", I32, eq=128, multiple_of=8),
        ValueError,
        "param D: eq excludes every other word -- eq=128 already decides multiple_of",
    ),
    (
        lambda b, M, sb, k0: b.param("D", I32, eq=M, ge=0, le=9),
        ValueError,
        "param D: eq excludes every other word -- eq=M already decides ge, le",
    ),
    (
        lambda b, M, sb, k0: b.param("x", I32, le=quotient_of(M, sb)),
        ValueError,
        (
            "param x: quotient_of(M, sb) needs sb ge 1 -- sb is declared with no lower "
            'bound.\nFix: b.param("sb", I32, ge=1, ...)'
        ),
    ),
    (
        lambda b, M, sb, k0: b.param("x", I32, le=quotient_of(sb, M)),
        ValueError,
        (
            "param x: quotient_of(sb, M) needs sb ge 0 -- sb is declared with no lower "
            'bound.\nFix: b.param("sb", I32, ge=0, ...)'
        ),
    ),
    (
        lambda b, M, sb, k0: b.param("x", I32, le=quotient_of(4096, sum_of(M, M))),
        ValueError,
        (
            "param x: quotient_of(4096, sum_of(M, M)) needs a divisor that is an int or "
            "one kernel parameter, so that its lower bound is stated"
        ),
    ),
    (
        lambda b, M, sb, k0: b.param("x", I32, le=quotient_of(M, 0)),
        ValueError,
        "param x: quotient_of(M, 0) needs a divisor of at least 1, got 0",
    ),
    (
        lambda b, M, sb, k0: b.param("sb2", I32, ge=product_of(M, k0)),
        ValueError,
        (
            "param sb2: product_of(M, add1) uses add1, which is computed in the kernel "
            "-- the host can only evaluate kernel parameters and ints"
        ),
    ),
    (
        lambda b, M, sb, k0: b.param("x", I32, le=k0),
        ValueError,
        (
            "param x: le=add1 uses add1, which is computed in the kernel -- the host can "
            "only evaluate kernel parameters and ints"
        ),
    ),
    (
        lambda b, M, sb, k0: b.param("C", PtrType(F16, "global"), multiple_of=4),
        ValueError,
        (
            "param C: multiple_of states an integer fact, and C is ptr<f16,global> -- "
            "state it on an integer parameter"
        ),
    ),
    (
        lambda b, M, sb, k0: b.param("n", I32, align=16),
        ValueError,
        "param n: align applies to a pointer parameter, and n is i32",
    ),
    (
        lambda b, M, sb, k0: b.param("C", PtrType(F16, "global"), align=12),
        ValueError,
        "param C: align must be a power of two >= 1, got 12",
    ),
    (
        lambda b, M, sb, k0: b.param("n", I32, multiple_of=0),
        ValueError,
        "param n: multiple_of must be >= 1, got 0",
    ),
    (
        lambda b, M, sb, k0: b.param("n", I32, le=1 << 40),
        ValueError,
        "param n: le=1099511627776 is outside the range of i32",
    ),
    (
        lambda b, M, sb, k0: b.param("n", I64, le=1 << 60),
        ValueError,
        (
            "param n: le=1152921504606846976 is outside +/-(2^53 - 1), which a stored "
            "fact cannot hold exactly"
        ),
    ),
    (
        lambda b, M, sb, k0: b.param("n", I32, ge=1.5),
        TypeError,
        (
            "param n: ge is an int, a parameter of this kernel, or sum_of / product_of / "
            "quotient_of / difference_of over those, got 1.5"
        ),
    ),
    (
        lambda b, M, sb, k0: b.param("n", I32, multiple_of=M),
        TypeError,
        "param n: multiple_of is an int, got %M",
    ),
]


@pytest.mark.parametrize("call,error,message", REFUSALS)
def test_the_builder_refuses_what_a_declaration_cannot_state(call, error, message):
    b = IRBuilder("refused")
    M, sb, k0 = _three(b)
    with pytest.raises(error) as raised:
        call(b, M, sb, k0)
    assert str(raised.value) == message
    assert [p.name for p in b.kernel.params] == ["M", "sb"]


def test_a_misspelled_word_is_a_type_error():
    b = IRBuilder("misspelled")
    with pytest.raises(TypeError, match="unexpected keyword argument 'el'"):
        b.param("ldc", I32, el=1 << 20)


def test_a_parameter_of_another_kernel_is_refused():
    other = IRBuilder("other").param("N", I32, ge=1)
    b = IRBuilder("this")
    with pytest.raises(ValueError) as raised:
        b.param("ldc", I32, ge=other)
    assert str(raised.value) == (
        "param ldc: ge=N uses N, which is a parameter of another kernel -- the host "
        "can only evaluate this kernel's parameters and ints"
    )


def test_a_stated_lower_bound_through_a_relation_admits_a_quotient():
    """``M ge N`` with ``N ge 1`` gives M a lower bound of 1, so M divides."""
    b = IRBuilder("transitive")
    N = b.param("N", I32, ge=1)
    M = b.param("M", I32, ge=N)
    sb = b.param("sb", I32, ge=0)
    b.param("rows", I32, le=quotient_of(sb, M))


# --------------------------------------------------------------------------
# stated_facts_of
# --------------------------------------------------------------------------


def test_stated_facts_resolve_relations_to_numbers():
    b = IRBuilder("facts")
    M, _, ldc, sb, C, rows = _gemm_params(b)
    assert stated_facts_of(M) == StatedFacts(
        ("ge", "le", "multiple_of"), 1, MAX_EXTENT, 64
    )
    assert stated_facts_of(ldc) == StatedFacts(
        ("ge", "le", "multiple_of"), 1, MAX_EXTENT, 8
    )
    # sb >= M * ldc >= 1; no upper word.
    assert stated_facts_of(sb) == StatedFacts(("ge",), 1, None)
    # rows <= sb // M, where sb is at most the i32 limit and M at least 1.
    assert stated_facts_of(rows) == StatedFacts(("ge", "le"), 0, (1 << 31) - 1)
    assert stated_facts_of(C) == StatedFacts()
    assert stated_facts_of(7) == StatedFacts(("eq",), 7, 7)
    assert stated_facts_of(b.thread_id_x()) == StatedFacts()


def test_stated_facts_read_a_marker_and_resolve_its_value_end():
    b = IRBuilder("marker_facts")
    T = b.param("T", I32, ge=0, le=4096)
    start = b._assume_range(b.thread_id_x(), ge=0, lt=T, lt_plus=1)
    assert stated_facts_of(start) == StatedFacts(("ge", "lt"), 0, 4096)
    tile = b._assume_range(T, ge=0, lt=512, multiple_of=8, residue=0)
    assert stated_facts_of(tile) == StatedFacts(("ge", "lt", "multiple_of"), 0, 511, 8)


def test_stated_facts_survive_serialize_and_parse():
    kernel = parse(serialize(_gemm_kernel()))
    values = {}
    for op in kernel.body.ops:
        for operand in op.operands:
            if operand.param is not None:
                values[operand.param.name] = operand
    assert stated_facts_of(values["rows"]) == StatedFacts(
        ("ge", "le"), 0, (1 << 31) - 1
    )
    assert stated_facts_of(values["M"]).multiple_of == 64


# --------------------------------------------------------------------------
# print, serialize, canonicalize, verify (both engines through IR_CASES)
# --------------------------------------------------------------------------


def test_print_shows_the_facts_after_the_type():
    text = print_ir(_gemm_kernel())
    assert text.splitlines()[0] == (
        "kernel @param_facts_gemm(%M: i32 {ge = 1, le = 1048576, multiple_of = 64}, "
        "%N: i32 {ge = 1, le = 1048576}, "
        "%ldc: i32 {ge = %N, le = 1048576, multiple_of = 8}, "
        "%sb: i32 {ge = product_of(%M, %ldc)}, %C: ptr<i32,global>, "
        "%rows: i32 {ge = 0, le = quotient_of(%sb, %M)}) {"
    )


def test_serialize_round_trips_the_facts():
    text = serialize(_gemm_kernel())
    assert serialize(parse(text)) == text
    assert verify(parse(text)) == []


def test_canonicalize_renames_relation_leaves():
    canonical = canonicalize(_gemm_kernel())
    assert (
        '%3 : i32 { ge = s:"{\\"mul\\":[{\\"param\\":\\"0\\"},{\\"param\\":\\"2\\"}]}" }'
        in (canonical)
    )


def _small_kernel():
    b = IRBuilder("param_facts_small")
    b.kernel.attrs["max_workgroup_size"] = 64
    M = b.param("M", I32, ge=1)
    x = b.param("x", I32)
    C = b.param("C", PtrType(I32, "global"))
    b.param("y", I32)
    b.global_store(C, b.thread_id_x(), b.add(x, M))
    b.ret()
    return b.kernel


def _with_attrs(changes):
    """The small kernel with some parameters' stored attrs replaced: IR text
    the builder never writes, as an importer or a hand edit could."""
    kernel = _small_kernel()
    for p in kernel.params:
        if p.name in changes:
            p.attrs = dict(changes[p.name])
    return kernel


IR_CASES = {
    "gemm": (_gemm_kernel, []),
    "later": (
        lambda: _with_attrs({"x": {"le": '{"sub":[{"param":"y"},{"param":"M"}]}'}}),
        [
            (
                "x: the relation names y, which is declared later -- state it on the "
                "last-declared term, y (difference_of and quotient_of move a sum or a "
                "product to its side)"
            )
        ],
    ),
    "itself_and_unknown": (
        lambda: _with_attrs({"x": {"ge": '{"add":[{"param":"x"},{"param":"K"}]}'}}),
        [
            "x: the relation names x itself -- a parameter cannot bound itself",
            "x: the relation names K, which is not a parameter of this kernel",
        ],
    ),
    "pointer_term": (
        lambda: _with_attrs({"y": {"le": '{"param":"C"}'}}),
        ["y: the relation names C, which is ptr<i32,global>, not an integer parameter"],
    ),
    "not_a_relation": (
        lambda: _with_attrs({"x": {"ge": '{"pow":[2,3]}', "le": "M+1"}}),
        [
            (
                "param x: ge is not a relation: a relation is an int, a param leaf, or add "
                "/ sub / mul / div of two relations"
            ),
            "param x: le is not a relation: it is not JSON",
        ],
    ),
    "unknown_and_misplaced": (
        lambda: _with_attrs({"x": {"el": 3}, "C": {"multiple_of": 4, "noalias": True}}),
        [
            (
                "param 'x' has unknown attr 'el' -- a param attr is one of noalias, readonly, "
                "writeonly, nocapture, nonnull, align, dereferenceable, addr_space, ge, gt, "
                "le, lt, eq, multiple_of"
            ),
            (
                "param C: multiple_of states an integer fact, and C is ptr<i32,global> -- "
                "state it on an integer parameter"
            ),
        ],
    ),
    "rules": (
        lambda: _with_attrs(
            {
                "M": {"ge": True, "multiple_of": "8"},
                "x": {"gt": 0, "ge": 1, "eq": '{"param":"M"}', "multiple_of": 0},
                "y": {"le": '{"div":[4096,{"sub":[{"param":"M"},1]}]}'},
            }
        ),
        [
            "param M: ge must be an integer or a relation",
            "param M: multiple_of must be an integer",
            "param x: states two lower bounds (gt, ge) -- state one",
            (
                "param x: eq excludes every other word -- eq=M already decides ge, gt, "
                "multiple_of"
            ),
            "param x: multiple_of must be >= 1, got 0",
            (
                "param y: quotient_of(4096, difference_of(M, 1)) needs a divisor that is an "
                "int or one kernel parameter, so that its lower bound is stated"
            ),
        ],
    ),
    "quotient_bounds": (
        lambda: _with_attrs(
            {"x": {"ge": -1}, "y": {"le": '{"div":[{"param":"x"},{"param":"M"}]}'}}
        ),
        [
            (
                "param y: quotient_of(x, M) needs x ge 0 -- x is declared with ge=-1.\n"
                'Fix: b.param("x", I32, ge=0, ...)'
            )
        ],
    ),
}


def _python_report(kernel) -> str:
    """What the C++ program prints for one kernel with ``--ir``."""
    messages = [d.message for d in verify(kernel)]
    return "\n".join(
        [
            print_ir(kernel),
            "-- verify --",
            *messages,
            "-- canonical --",
            canonicalize(kernel),
        ]
    )


@pytest.mark.parametrize("case", sorted(IR_CASES))
def test_the_verifier_reports_every_broken_rule(case):
    build, messages = IR_CASES[case]
    kernel = parse(serialize(build()))
    assert [d.message for d in verify(kernel)] == messages


def test_cpp_prints_verifies_and_canonicalizes_the_same():
    """The C++ ``print_ir``, verifier and ``canonicalize`` agree with Python on
    every kernel in ``IR_CASES``.

    ``ROCKE_PARAM_FACTS_CLI`` names the built ``rocke_param_facts``:
    ``tests/run_all.py`` resolves it from a configured build and the installed
    ``rocke_pytest`` ctest sets it. A value that names no file fails.
    """
    executable = os.environ.get("ROCKE_PARAM_FACTS_CLI")
    if not executable:
        pytest.skip("Set ROCKE_PARAM_FACTS_CLI to the built rocke_param_facts")
    path = Path(executable).resolve()
    assert path.is_file(), f"ROCKE_PARAM_FACTS_CLI names {path}, not a file"
    for case in sorted(IR_CASES):
        kernel = parse(serialize(IR_CASES[case][0]()))
        printed = subprocess.run(
            [str(path), "--ir"],
            input=serialize(kernel),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert printed.stdout == _python_report(kernel), case


# --------------------------------------------------------------------------
# facts emit nothing, survive optimization, and are recorded as stored
# --------------------------------------------------------------------------


def _without_facts():
    b = IRBuilder("param_facts_gemm")
    b.kernel.attrs["max_workgroup_size"] = 64
    M = b.param("M", I32)
    b.param("N", I32)
    b.param("ldc", I32)
    b.param("sb", I32)
    C = b.param("C", PtrType(I32, "global"), noalias=True, align=16)
    rows = b.param("rows", I32)
    tid = b.thread_id_x()
    b.global_store(C, tid, b.add(rows, M))
    b.ret()
    return b.kernel


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22"])
def test_facts_emit_nothing_in_either_engine(flavor):
    assert lower_with_both_engines(
        _gemm_kernel(), "gfx942", llvm_flavor=flavor
    ) == lower_with_both_engines(_without_facts(), "gfx942", llvm_flavor=flavor)


def test_optimize_kernel_keeps_the_facts():
    kernel = _gemm_kernel()
    before = [dict(p.attrs) for p in kernel.params]
    optimize_kernel(kernel)
    assert [p.attrs for p in kernel.params] == before
    assert print_ir(kernel).splitlines()[0] == print_ir(_gemm_kernel()).splitlines()[0]


def test_the_recording_builder_records_the_stored_facts():
    """The recipe carries what the declaration stored: ints, and a relation as
    its canonical JSON string -- never the Value the author passed."""

    def build():
        b = IRBuilder("recorded")
        _gemm_params(b)
        b.ret()
        return b.kernel

    kernel, recipe = record_kernel(build, sys.modules[__name__])
    recorded = {
        i["name"]: i["attrs"] for i in recipe["program"] if i.get("op") == "param"
    }
    assert recorded == {p.name: p.attrs for p in kernel.params}
    assert recorded["sb"] == {"ge": '{"mul":[{"param":"M"},{"param":"ldc"}]}'}
