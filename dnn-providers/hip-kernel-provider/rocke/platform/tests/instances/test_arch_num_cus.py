# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ CU-count table (cpp/helpers/arch_num_cus.cpp) is helpers/split_k.py's.

Host-only: reads the C++ table from its source and compares it with
``_ARCH_NUM_CUS`` / ``_DEFAULT_NUM_CUS``, so a row changed on one side fails here.
"""

import re
from pathlib import Path

from rocke.helpers import split_k

CPP = Path(__file__).resolve().parents[2] / "cpp"


def _cpp_table():
    src = (CPP / "helpers" / "arch_num_cus.cpp").read_text(encoding="utf-8")
    body = src.split("k_arch_num_cus[] = {", 1)[1].split("};", 1)[0]
    return {a: int(n) for a, n in re.findall(r'\{"(gfx[0-9a-z]+)",\s*(\d+)\}', body)}


def _cpp_default():
    hdr = (CPP / "include" / "rocke" / "arch_num_cus.h").read_text(encoding="utf-8")
    return int(re.search(r"#define ROCKE_DEFAULT_NUM_CUS (\d+)", hdr).group(1))


def test_table_matches_python():
    assert _cpp_table() == dict(split_k._ARCH_NUM_CUS)


def test_default_matches_python():
    assert _cpp_default() == split_k._DEFAULT_NUM_CUS
