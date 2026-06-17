################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
"""Round-trip tests for the LibraryFile field in cache.yaml.

cache.yaml is the source of truth for what the --use-cache phase needs:
the compiled .co files (CodeObjectFiles) AND the library descriptor
(LibraryFile). Recomputing LibraryFile in the cached path produced the
gfx1201 SIGSEGV regression; persisting it eliminates the bug class.
"""

import inspect

import Tensile.BenchmarkProblems as bp


def test_cache_yaml_write_path_persists_LibraryFile():
    """The non-cached write path must include LibraryFile in cacheData.

    Static check: source of _benchmarkProblemType contains a cacheData
    dict literal mentioning 'LibraryFile'.
    """
    src = inspect.getsource(bp._benchmarkProblemType)
    # Look for the cacheData = { ... } block.
    assert '"LibraryFile"' in src or "'LibraryFile'" in src, (
        "cacheData written to cache.yaml must include LibraryFile so the "
        "--use-cache phase can read the actual library path instead of "
        "recomputing it. Without this, the gfx1201 SIGSEGV regression can "
        "recur whenever libraryDir() layout assumptions drift."
    )


def test_readCacheIfValid_returns_both_CodeObjectFiles_and_LibraryFile(tmp_path):
    """_readCacheIfValid must return both fields, not just CodeObjectFiles."""
    from Tensile import LibraryIO
    from Tensile.BenchmarkProblems import _readCacheIfValid, _CACHE_FIELDS

    cachePath = str(tmp_path / "cache.yaml")
    # Build a cache yaml with all required fields. The fixture values must
    # match what _cacheDataMatches checks; we use a stub benchmarkStep.
    class StubBenchmarkStep:
        constantParams = {}
        forkParams = {}
        paramGroups = []
        customKernels = []
        internalSupportParams = {}
        customKernelWildcard = False

    cacheData = {f: getattr(StubBenchmarkStep, attr) for f, attr in _CACHE_FIELDS.items()}
    cacheData["CodeObjectFiles"] = ["library/gfx1201/TensileLibrary_gfx1201.co"]
    cacheData["LibraryFile"] = "library/gfx1201/TensileLibrary.yaml"
    LibraryIO.writeYAML(cachePath, cacheData)

    result = _readCacheIfValid(cachePath, StubBenchmarkStep, "ignored {path}")
    assert result is not None, "valid cache yaml should match"
    assert result["CodeObjectFiles"] == ["library/gfx1201/TensileLibrary_gfx1201.co"]
    assert result["LibraryFile"] == "library/gfx1201/TensileLibrary.yaml"


def test_cache_yaml_without_LibraryFile_is_invalid(tmp_path):
    """Legacy cache.yaml lacking LibraryFile must be treated as invalid → recompile."""
    from Tensile import LibraryIO
    from Tensile.BenchmarkProblems import _readCacheIfValid, _CACHE_FIELDS

    cachePath = str(tmp_path / "cache.yaml")
    class StubBenchmarkStep:
        constantParams = {}
        forkParams = {}
        paramGroups = []
        customKernels = []
        internalSupportParams = {}
        customKernelWildcard = False

    cacheData = {f: getattr(StubBenchmarkStep, attr) for f, attr in _CACHE_FIELDS.items()}
    cacheData["CodeObjectFiles"] = ["library/gfx1201/TensileLibrary_gfx1201.co"]
    # NOTE: deliberately omit LibraryFile to simulate a legacy cache.yaml.
    LibraryIO.writeYAML(cachePath, cacheData)

    result = _readCacheIfValid(cachePath, StubBenchmarkStep, "ignored {path}")
    assert result is None, (
        "cache.yaml without LibraryFile must be treated as invalid so the "
        "--use-cache phase doesn't crash on a missing/wrong library file."
    )


def test_cached_path_uses_loaded_LibraryFile_not_recompute():
    """The cached branch of _benchmarkProblemType must use the LibraryFile
    that flowed back from _loadCacheIfMatches, NOT recompute via libraryDir()."""
    src = inspect.getsource(bp._benchmarkProblemType)
    # In the cached branch (after `# Using cached solution data`), we must NOT
    # be calling libraryDir() to construct libraryFile any more — that's the
    # bug Task 4 retires.
    # Use a stable anchor at the end of the cached branch: writeSolutions
    # (called immediately after the cached writeClientConfigIni block) is the
    # first statement after the else: clause closes.
    cached_branch = src.split("Using cached solution data", 1)[1]
    cached_branch = cached_branch.split("writeSolutions", 1)[0]
    assert "libraryDir(" not in cached_branch, (
        "Cached path must read LibraryFile from cache.yaml, not recompute it "
        "via libraryDir(). Recomputation is the bug class this fix retires."
    )
    # Sanity check: the cached branch should be reading cachedLibraryFile.
    assert "cachedLibraryFile" in cached_branch, (
        "Cached path should be using cachedLibraryFile (loaded from cache.yaml)."
    )
