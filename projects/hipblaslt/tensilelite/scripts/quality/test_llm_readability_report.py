# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Characterization tests for the LLM-readability scanner.

Builds a tiny source tree in a temp dir and asserts the signal counts ``scan``
reports for it. The fixtures deliberately pin the two behaviors that were wrong
in review: nesting is counted per function (not per file), and the
tests-importing-private-paths signal reads the test tree directly (the scan
skips ``Tests/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_readability_report as rr

LIT = "this is a long shared literal used across two files!!"

ALPHA = f'''import typing
from typing import Any

from FeatureB.beta import something
from Common.shared_util import helper

LIT = "{LIT}"


def f(x: Any) -> None:
    y = typing.cast(int, x)
    z = 1  # type: ignore
    return None


def swallow():
    try:
        f(1)
    except Exception:
        pass


def deeply(a, b, c, d, e):
    if a:
        for _ in b:
            while c:
                with d:
                    if e:
                        return LIT
    return None
'''

BETA = f'''LIT = "{LIT}"


def plain():
    return LIT
'''

SHARED_UTIL = '''def helper():
    return 1
'''

TEST_BETA = '''from FeatureA._internal import secret


def test_plain():
    assert secret
'''


def _build_tree(root: Path) -> Path:
    """Write the fixture source tree under ``root`` and return its src root."""
    src = root / "src"
    (src / "FeatureA").mkdir(parents=True)
    (src / "FeatureB").mkdir(parents=True)
    (src / "Common").mkdir(parents=True)
    (src / "Tests").mkdir(parents=True)

    (src / "FeatureA" / "__init__.py").write_text("")
    (src / "FeatureB" / "__init__.py").write_text("")
    (src / "FeatureA" / "alpha.py").write_text(ALPHA)
    (src / "FeatureB" / "beta.py").write_text(BETA)
    (src / "Common" / "shared_util.py").write_text(SHARED_UTIL)
    (src / "Tests" / "test_beta.py").write_text(TEST_BETA)
    return src


def _by_suffix(files, suffix: str) -> rr.FileReport:
    matches = [f for f in files if f.path.replace("\\", "/").endswith(suffix)]
    assert len(matches) == 1, f"expected exactly one {suffix}, got {len(matches)}"
    return matches[0]


def test_scan_skips_tests_and_counts_source(tmp_path):
    """Tests/ is excluded from the scanned file list; the five sources remain."""
    src = _build_tree(tmp_path)
    files, summary = rr.scan(src)
    assert summary["file_count"] == 5
    assert all("Tests" not in Path(f.path).parts for f in files)


def test_nesting_is_per_function(tmp_path):
    """deeply() nests 5 deep and is the only such function; swallow()/f() are not."""
    src = _build_tree(tmp_path)
    files, _ = rr.scan(src)
    alpha = _by_suffix(files, "FeatureA/alpha.py")
    assert alpha.deep_nesting_functions == 1
    assert alpha.max_nesting_depth >= rr.NESTING_THRESHOLD
    assert sum(f.deep_nesting_functions for f in files) == 1


def test_swallowed_error_detected(tmp_path):
    """`except Exception: pass` is the one swallowed handler in the tree."""
    src = _build_tree(tmp_path)
    files, _ = rr.scan(src)
    assert sum(len(f.swallowed_error_lines) for f in files) == 1


def test_typing_escape_hatches(tmp_path):
    """alpha uses Any, typing.cast and one `# type: ignore`."""
    src = _build_tree(tmp_path)
    files, _ = rr.scan(src)
    alpha = _by_suffix(files, "FeatureA/alpha.py")
    assert alpha.any_count == 1
    assert alpha.cast_count == 1
    assert alpha.type_ignore_count == 1


def test_cross_feature_import_excludes_infra(tmp_path):
    """The FeatureB import counts; the Common (infra) import does not."""
    src = _build_tree(tmp_path)
    files, _ = rr.scan(src)
    alpha = _by_suffix(files, "FeatureA/alpha.py")
    assert len(alpha.cross_feature_imports) == 1
    assert all("Common" not in c for c in alpha.cross_feature_imports)


def test_duplicate_long_literal_clustered(tmp_path):
    """The shared 40+ char literal appears in two files -> one cluster."""
    src = _build_tree(tmp_path)
    _, summary = rr.scan(src)
    clusters = summary["duplicate_literal_clusters"]
    assert len(clusters) == 1
    assert clusters[0]["file_count"] == 2


def test_missing_seam_tests(tmp_path):
    """beta has test_beta.py; alpha has none; Common is infra and exempt."""
    src = _build_tree(tmp_path)
    _, summary = rr.scan(src)
    assert summary["missing_seam_tests"] == ["FeatureA/alpha.py"]


def test_internal_test_imports_read_from_test_tree(tmp_path):
    """The private-path import in Tests/ is counted even though scan skips Tests/."""
    src = _build_tree(tmp_path)
    _, summary = rr.scan(src)
    assert summary["internal_test_imports_total"] == 1
