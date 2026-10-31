# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensile command-line arguments for the Tensile invocations of the test suite."""


def tensileArgs(getOption, workerLockPath):
    """Arguments the ``tensile_args`` fixture appends to every Tensile call.

    getOption returns the value of a pytest command-line option (e.g.
    ``pytestconfig.getoption``). workerLockPath is the client execution lock
    of this worker, or None.

    Every test run gets ``ClientLogLevel=0`` (Error): a test only needs the
    client's FAILED rows and mismatch lines, which it prints at every log
    level, and Error drops the row of every passed or rejected solution. It
    comes before every user-supplied ``--global-parameters`` (from
    ``--tensile-options`` or ``--global-parameters``): Tensile applies
    repeated ``--global-parameters`` in order, the last value of a key wins,
    and they override the YAML, so a developer can still pass e.g.
    ``ClientLogLevel=3``.
    """
    rv = []
    if workerLockPath:
        rv += ["--client-lock", str(workerLockPath)]

    rv += ["--global-parameters", "ClientLogLevel=0"]

    extraOptions = getOption("--tensile-options")
    if extraOptions is not None:
        rv += extraOptions.split(",")
    if getOption("--global-parameters"):
        rv += ["--global-parameters", getOption("--global-parameters")]
    if not getOption("--no-common-build"):
        if getOption("--prebuilt-client"):
            rv += ["--prebuilt-client", getOption("--prebuilt-client")]

    # Forward --gpu-targets to Tensile. Do NOT forward --build-only or
    # --use-cache here — those are hardcoded in each test function's args.
    if getOption("--gpu-targets"):
        rv += ["--gpu-targets", getOption("--gpu-targets")]

    return rv
