# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Child process of ``build_hsaco_from_llvm_ir(..., fresh_process=True)``.

Compiles exactly one module and exits, so nothing compiled before it can
change its code (see :func:`rocke.runtime.comgr.build_hsaco_from_llvm_ir`).

Protocol (internal to :func:`rocke.runtime.comgr._build_in_fresh_process`):

* stdin: one JSON line ``{"isa", "options", "out"}``, then the LLVM IR text
  (utf-8) up to end of input;
* the result goes to the file ``out``: one JSON line ``{"size", "bc",
  "relocatable", "executable"}`` followed by the code object bytes;
* exit status 0 on success, 2 for a :class:`~rocke.runtime.comgr.ComgrError`
  (message on stderr), 1 for any other exception (traceback on stderr).
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    data = sys.stdin.buffer.read()
    head, _, body = data.partition(b"\n")
    header = json.loads(head)
    from rocke.runtime.comgr import ComgrError, build_hsaco_from_llvm_ir

    try:
        hsaco, timings = build_hsaco_from_llvm_ir(
            body.decode("utf-8"), isa=header["isa"], options=header["options"]
        )
    except ComgrError as ex:
        sys.stderr.write(str(ex))
        return 2
    meta = {
        "size": len(hsaco),
        "bc": timings.bc,
        "relocatable": timings.relocatable,
        "executable": timings.executable,
    }
    with open(header["out"], "wb") as fh:
        fh.write(json.dumps(meta).encode("utf-8") + b"\n")
        fh.write(hsaco)
    return 0


if __name__ == "__main__":
    sys.exit(main())
