# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke_py_str_repr, the engine's C++ port of Python's repr() of a str.

The IR printer quotes str attrs with it and spec ports quote names in their
ValueError texts with it, so it must spell a str exactly as Python does. The
C++ side is reached through ``rocke_ir_types --py-str-repr``
(``ROCKE_IR_TYPES_TEST``): it reads hex UTF-8 lines and prints the hex of each
repr.
"""

import os
import subprocess

import pytest

# Every code point the port spells exactly, alone and next to each quote.
_CODE_POINTS = range(0x01, 0x300)


def _cases():
    for cp in _CODE_POINTS:
        ch = chr(cp)
        yield ch
        yield "a" + ch + "'"
        yield '"' + ch + "'"


def test_native_repr_matches_python_repr():
    executable = os.environ.get("ROCKE_IR_TYPES_TEST")
    if not executable:
        pytest.skip("set ROCKE_IR_TYPES_TEST to the built rocke_ir_types test")
    cases = list(_cases())
    stdin = "".join(s.encode().hex() + "\n" for s in cases)
    out = subprocess.run(
        [executable, "--py-str-repr"],
        input=stdin,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert len(out) == len(cases)
    wrong = [
        (s, bytes.fromhex(got).decode(errors="replace"), repr(s))
        for s, got in zip(cases, out)
        if bytes.fromhex(got) != repr(s).encode()
    ]
    assert not wrong, wrong[:10]
