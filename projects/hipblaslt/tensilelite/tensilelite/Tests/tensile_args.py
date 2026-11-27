# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensile command-line arguments for the Tensile invocations of the test suite."""


def tensileArgs(getOption, workerLockPath, world=None):
    """Arguments the ``tensile_args`` fixture appends to every Tensile call.

    getOption returns the value of a pytest command-line option (e.g.
    ``pytestconfig.getoption``). workerLockPath is the client execution lock
    of this worker, or None. world holds the arguments of the test's ``world``
    marker, or None; it becomes ``WorldFilter``, after the user-supplied
    ``--global-parameters``.

    Every test run gets ``ClientLogLevel=0`` (Error), preserving the client's
    FAILED rows and mismatch lines while dropping passed and rejected rows.
    It comes before every user-supplied ``--global-parameters`` (from
    ``--tensile-options`` or ``--global-parameters``): Tensile applies
    repeated ``--global-parameters`` in order, the last value of a key wins,
    and they override the YAML. To restore Debug logging, pass
    ``--tensile-options=--global-parameters,ClientLogLevel=3``. This preserves
    an existing pytest ``--global-parameters`` value, such as
    ``CheckASMCodeSize=True``. If ``--tensile-options`` is already present,
    append the comma-separated pair to that value. YAML ``PrintTensor*``
    settings require a command-line ``ClientLogLevel`` of at least 2 (Verbose).
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
    if world is not None:
        rv += ["--global-parameters", f"WorldFilter={list(world)}"]
    # Forward --gpu-targets to tensilelite. Do NOT forward --build-only or
    # --use-cache here — those are hardcoded in each test function's args.
    if getOption("--gpu-targets"):
        rv += ["--gpu-targets", getOption("--gpu-targets")]

    return rv
