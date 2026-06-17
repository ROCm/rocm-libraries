################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
"""Unit tests for writeClientConfigIni library-file path emission.

Locks in the invariant: callers must pass libraryFile explicitly.
Regression guard for the SIGSEGV bug where a single-arch build wrote
TensileLibrary.yaml to library/<arch>/ but ClientParameters.ini defaulted
to library/TensileLibrary.yaml (flat) and the client crashed in LLVM.
"""

import inspect

from Tensile.ClientWriter import writeClientConfigIni


def test_libraryFile_is_a_required_parameter():
    """writeClientConfigIni must not silently default libraryFile.

    Why: defaulting to library/TensileLibrary.{yaml,dat} (flat) is wrong
    whenever the build used a per-arch library subdir (single-arch builds).
    Forcing every caller to pass the path makes the bug impossible to recur
    via 'I forgot the kwarg'.
    """
    sig = inspect.signature(writeClientConfigIni)
    param = sig.parameters.get("libraryFile")
    assert param is not None, "libraryFile parameter must exist"
    assert param.default is inspect.Parameter.empty, (
        "libraryFile must be a required parameter (no default). "
        "A default value invites callers to omit it, which produced "
        "the gfx1201 SIGSEGV regression."
    )


def _extract_calls(src, fn_name):
    """Return every text span starting with `<fn_name>(` and ending at the
    matching close paren, balanced. Handles nested parens and multi-line calls.

    Caveat: this is a naive paren-matcher, not a Python parser. It does NOT
    track string literals, comments, raw/f-strings, or interpolation, so a
    `)` inside a string or a `# fn_name(...)` in a comment could mislead it.
    Considered acceptable here because the call sites under check are real
    source code; if a false positive shows up, this docstring is the hint.
    """
    spans = []
    needle = fn_name + "("
    i = 0
    while True:
        start = src.find(needle, i)
        if start == -1:
            return spans
        depth = 0
        j = start + len(fn_name)  # at the '('
        while j < len(src):
            c = src[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    spans.append(src[start:j+1])
                    break
            j += 1
        i = j + 1


def test_benchmarkProblems_cached_path_passes_libraryFile():
    """The cached path at BenchmarkProblems.py:603 must pass libraryFile=.

    Static check: the source of _benchmarkProblemType contains
    'writeClientConfigIni(' with 'libraryFile=' in the same statement.
    Catches the regression where the cached path silently relied on
    writeClientConfigIni's default.
    """
    import Tensile.BenchmarkProblems as bp
    src = inspect.getsource(bp._benchmarkProblemType)
    calls = _extract_calls(src, "writeClientConfigIni")
    assert calls, "expected writeClientConfigIni call(s) in _benchmarkProblemType"
    for call in calls:
        assert "libraryFile" in call, (
            f"writeClientConfigIni call in _benchmarkProblemType is missing "
            f"libraryFile=. This is the exact regression that caused the "
            f"gfx1201 SIGSEGV. Call site:\n{call}"
        )


def test_ClientWriter_internal_caller_passes_libraryFile():
    """CreateBenchmarkClientParametersForSizes (ClientWriter.py:782) must pass libraryFile."""
    import Tensile.ClientWriter as cw
    src = inspect.getsource(cw.CreateBenchmarkClientParametersForSizes)
    calls = _extract_calls(src, "writeClientConfigIni")
    assert calls, "expected writeClientConfigIni call(s) in CreateBenchmarkClientParametersForSizes"
    for call in calls:
        assert "libraryFile" in call, (
            f"writeClientConfigIni call in CreateBenchmarkClientParametersForSizes "
            f"is missing libraryFile=. Same bug class as the cached-path regression.\n{call}"
        )


def test_TensileClientConfig_passes_libraryFile():
    """Tensile/TensileClientConfig.py (standalone bin/TensileClientConfig entry
    point) must also pass libraryFile= now that the default is removed."""
    import Tensile.TensileClientConfig as tcc
    src = inspect.getsource(tcc.TensileClientConfig)
    calls = _extract_calls(src, "writeClientConfigIni")
    assert calls, "expected writeClientConfigIni call(s) in TensileClientConfig"
    for call in calls:
        assert "libraryFile" in call, (
            f"writeClientConfigIni call in TensileClientConfig is missing "
            f"libraryFile=. After Task 1 this call would TypeError.\n{call}"
        )
