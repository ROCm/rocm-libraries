# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Child process of ``build_hsaco_from_llvm_ir(..., fresh_process=True)``.

Compiles exactly one module and exits, so nothing compiled before it can
change its code (see :func:`rocke.runtime.comgr.build_hsaco_from_llvm_ir`).
On POSIX it runs under an address-space cap, so a compile that allocates
without bound fails here instead of taking the host's memory.

Protocol (internal to :func:`rocke.runtime.comgr._build_in_fresh_process`):

* stdin: one JSON line ``{"isa", "options", "out", "memory_cap_bytes"}``,
  then the LLVM IR text (utf-8) up to end of input;
* the result goes to the file ``out``: one JSON line ``{"size", "bc",
  "relocatable", "executable"}`` followed by the code object bytes;
* exit status 0 on success, :data:`COMPILE_ERROR_EXIT_STATUS` for a
  :class:`~rocke.runtime.comgr.ComgrError` (message on stderr),
  :data:`OUT_OF_MEMORY_EXIT_STATUS` for a ``MemoryError``, and
  :data:`UNCAUGHT_EXCEPTION_EXIT_STATUS` for any other exception (traceback
  on stderr). When the compiler itself runs out of memory, LLVM writes
  ``LLVM ERROR: out of memory`` to stderr and aborts the process.
"""

from __future__ import annotations

import importlib.util
import json
import sys

COMPILE_ERROR_EXIT_STATUS = 2
# Not 3: on Windows abort() can exit with 3, which would read as out of memory.
OUT_OF_MEMORY_EXIT_STATUS = 4
# Python's own exit status for an uncaught exception.
UNCAUGHT_EXCEPTION_EXIT_STATUS = 1

# Whether this platform can cap a process's address space: POSIX has
# RLIMIT_AS (the ``resource`` module); Windows has neither.
ADDRESS_SPACE_CAP_APPLIES = importlib.util.find_spec("resource") is not None


def limit_address_space(cap_bytes: int) -> None:
    """Lower this process's address-space limit (``RLIMIT_AS``) to
    ``cap_bytes`` where :data:`ADDRESS_SPACE_CAP_APPLIES`; elsewhere do
    nothing. A limit the process inherited that is already lower stays: the
    cap only ever tightens it, and the hard limit is left as it is."""
    if not ADDRESS_SPACE_CAP_APPLIES:
        return
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = cap_bytes
    for inherited in (soft, hard):
        if inherited != resource.RLIM_INFINITY:
            limit = min(limit, inherited)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def main() -> int:
    data = sys.stdin.buffer.read()
    head, _, body = data.partition(b"\n")
    header = json.loads(head)
    limit_address_space(header["memory_cap_bytes"])
    from rocke.runtime.comgr import ComgrError, build_hsaco_from_llvm_ir

    try:
        hsaco, timings = build_hsaco_from_llvm_ir(
            body.decode("utf-8"), isa=header["isa"], options=header["options"]
        )
    except ComgrError as ex:
        sys.stderr.write(str(ex))
        return COMPILE_ERROR_EXIT_STATUS
    except MemoryError:
        sys.stderr.write("MemoryError")
        return OUT_OF_MEMORY_EXIT_STATUS
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
