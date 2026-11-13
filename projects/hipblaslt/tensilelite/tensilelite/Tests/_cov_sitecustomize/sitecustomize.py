# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Start coverage.py in codegen worker subprocesses (coverage-unit lane).

TensileLite codegen emits kernels in joblib/loky worker processes spawned by
``tensilelite/create_library/run.py``. Those workers do not inherit the
parent's coverage tracer, so without this hook the Python emitters they execute
(``SdmaPacketEmitter``, ``MAC_*``, large parts of ``KernelWriterAssembly``, ...)
run unmeasured and show up as uncovered even though the tests exercise them.

``coverage.process_startup()`` is the standard coverage.py subprocess entry
point: it starts measurement only when ``COVERAGE_PROCESS_START`` is set. It is
a no-op everywhere else -- including the main pytest process and its xdist
workers, which pytest-cov already measures -- so this file is inert unless that
variable is exported to the workers.

This directory is placed first on ``PYTHONPATH`` by the ``coverage-unit`` tox
env purely so Python auto-imports this ``sitecustomize`` at interpreter startup.
"""

try:
    import coverage
except Exception:  # coverage not installed (e.g. a normal build): nothing to do.
    pass
else:
    coverage.process_startup()
