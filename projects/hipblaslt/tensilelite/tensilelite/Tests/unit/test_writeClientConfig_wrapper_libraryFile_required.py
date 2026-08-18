################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
"""Lock in: writeClientConfig (wrapper) must require libraryFile, no default.

Companion to test_writeClientConfigIni_library_file.py — that file guards the
inner function; this one guards the wrapper that calls it. The reviewer noted
that fixing only the inner function leaves a hole: if anyone calls the wrapper
without libraryFile=, None propagates positionally into writeClientConfigIni
and the .ini ends up with library-file=None (a silent miswrite, since the
inner function no longer has the `if libraryFile is None` guard).
"""

import inspect

from Tensile.ClientWriter import writeClientConfig


def test_writeClientConfig_libraryFile_is_required():
    """writeClientConfig must not silently default libraryFile.

    Same rationale as test_libraryFile_is_a_required_parameter for the inner
    writeClientConfigIni function (see test_writeClientConfigIni_library_file.py).
    A default here would let library-file=None leak into ClientParameters.ini.
    """
    sig = inspect.signature(writeClientConfig)
    param = sig.parameters.get("libraryFile")
    assert param is not None, "libraryFile parameter must exist on writeClientConfig"
    assert param.default is inspect.Parameter.empty, (
        "writeClientConfig.libraryFile must be a required parameter (no default). "
        "Defaulting to None would let a None propagate positionally into "
        "writeClientConfigIni and produce 'library-file=None' in the .ini — "
        "same bug class the prior fix eliminated one frame down the stack."
    )
