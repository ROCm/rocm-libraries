#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# export_parity.py -- export one of the four parity kernels (scalar|memory|
# forloop|vector) to portable CK-DSL IR JSON (schema ck.dsl.ir/v1), using the
# SAME builders as the existing tests/parity/emit.py so the round-tripped .ll
# can be byte-compared against the Python-lowered reference.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "parity"))

from ck_dsl.core.ir import IRBuilder  # noqa: E402
from ck_dsl.core.ir_export import export_kernel_ir_json  # noqa: E402

import emit  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in emit.BUILDERS:
        sys.stderr.write("usage: export_parity.py <scalar|memory|forloop|vector>\n")
        return 2
    which = sys.argv[1]
    b = IRBuilder("parity_kernel")
    emit.BUILDERS[which](b)
    sys.stdout.write(export_kernel_ir_json(b.kernel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
