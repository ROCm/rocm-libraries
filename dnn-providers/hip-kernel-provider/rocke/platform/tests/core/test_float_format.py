# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ engine spells a float exactly as Python's repr() does.

The Python engine writes every float with repr(): in printed IR, in serialized
IR and in HIP literals. The C++ engine writes them all with one formatter,
rocke_append_python_float_repr. This test feeds that formatter (through the
rocke_float_format binary named by ROCKE_FLOAT_FORMAT_TEST) the values whose
shortest spelling is hardest to get right -- every power of two, every power of
ten and the doubles either side of each -- plus a fixed random sample, and
compares each line with repr().

The HIP lowerers in both engines put that spelling into C++ literals through one
adapter each: repr() plus an "f" suffix for a float, repr() alone for a double,
and the <math.h> NAN / INFINITY macros for the values repr() spells in a way
C++ does not accept. A second test lowers float and double constants through
both engines and checks each literal against that rule.
"""

import math
import os
import random
import struct
import subprocess

import pytest
from rocke.core.ir import F32, F64, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip


def _bits(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def _sample() -> list[float]:
    edges = [2.0**e for e in range(-1074, 1024)]
    edges += [float(f"1e{e}") for e in range(-323, 309)]
    edges += [0.0, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308]
    values = []
    for edge in edges:
        values += [edge, math.nextafter(edge, 0.0), math.nextafter(edge, math.inf)]
    rng = random.Random(20261102)
    for _ in range(50000):
        values.append(struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0])
    for _ in range(20000):
        values.append(
            round(rng.uniform(-1.0, 1.0), rng.randint(0, 17))
            * 10.0 ** rng.randint(-20, 20)
        )
    values += [-value for value in values]
    values += [math.inf, -math.inf, math.nan]
    return values


def test_native_float_spelling_matches_python_repr():
    executable = os.environ.get("ROCKE_FLOAT_FORMAT_TEST")
    if not executable:
        pytest.skip("set ROCKE_FLOAT_FORMAT_TEST to the built rocke_float_format test")
    values = _sample()
    native = subprocess.run(
        [executable, "--repr"],
        input="".join(_bits(v) + "\n" for v in values),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert len(native) == len(values)
    mismatches = [
        (repr(value), spelled)
        for value, spelled in zip(values, native)
        if spelled != repr(value)
    ]
    assert (
        not mismatches
    ), f"{len(mismatches)} of {len(values)} differ, e.g. {mismatches[:5]}"


# (value, its float literal, its double literal) in HIP source.
_HIP_LITERALS = [
    (0.0, "0.0f", "0.0"),
    (-0.0, "-0.0f", "-0.0"),
    (1024.0, "1024.0f", "1024.0"),
    (1e-4, "0.0001f", "0.0001"),
    (1e-5, "1e-05f", "1e-05"),
    (1e16, "1e+16f", "1e+16"),
    (2.0**-24, "5.960464477539063e-08f", "5.960464477539063e-08"),
    (-1.4426950408889634, "-1.4426950408889634f", "-1.4426950408889634"),
    (5e-324, "5e-324f", "5e-324"),
    (1.7976931348623157e308, "1.7976931348623157e+308f", "1.7976931348623157e+308"),
    (math.nan, "((float)NAN)", "((double)NAN)"),
    (math.inf, "((float)INFINITY)", "((double)INFINITY)"),
    (-math.inf, "((float)-INFINITY)", "((double)-INFINITY)"),
]


def _constants_kernel() -> str:
    """A kernel that stores every _HIP_LITERALS value as a float and as a
    double, serialized; its HIP source declares one constant per store."""
    builder = IRBuilder("float_literals")
    out_f32 = builder.param("out_f32", PtrType(F32, "global"))
    out_f64 = builder.param("out_f64", PtrType(F64, "global"))
    for index, (value, _, _) in enumerate(_HIP_LITERALS):
        builder.global_store(
            out_f32, builder.const_i32(index), builder.const_f32(value)
        )
        builder.global_store(
            out_f64, builder.const_i32(index), builder.const_f64(value)
        )
    builder.ret()
    return serialize(builder.kernel)


def _declared_literals(hip: str, c_type: str) -> list[str]:
    """The right-hand side of every ``<c_type> <name> = <literal>;`` line."""
    prefix = f"{c_type} "
    return [
        line.split(" = ", 1)[1].rstrip(";")
        for line in (raw.strip() for raw in hip.splitlines())
        if line.startswith(prefix) and " = " in line
    ]


def test_both_hip_lowerers_spell_float_literals_the_same():
    rocke_engine = pytest.importorskip("rocke_engine")
    ir = _constants_kernel()
    python_hip = lower_kernel_to_hip(parse(ir), arch="gfx942")
    native_hip = rocke_engine.lower_serialized_ir_hip(ir, "gfx942")
    for engine, hip in (("Python", python_hip), ("C++", native_hip)):
        assert _declared_literals(hip, "float") == [
            f32 for _, f32, _ in _HIP_LITERALS
        ], engine
        assert _declared_literals(hip, "double") == [
            f64 for _, _, f64 in _HIP_LITERALS
        ], engine
