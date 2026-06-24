"""Unit tests for the pre-commit affected-test selector.

The selector logic is pure Python (no rocisa / HIP / pytest execution), so these
run on any host without a ROCm container.
"""

import importlib.util
import json
import pathlib

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts"
    / "precommit_affected_tests.py"
)
_spec = importlib.util.spec_from_file_location("precommit_affected_tests", _SCRIPT)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

TL = hook.TL_REL
P = pathlib.Path


def test_module_dotted_paths():
    assert hook.module_dotted(P("Tensile/Common/Utilities.py")) == "Tensile.Common.Utilities"
    assert hook.module_dotted(P("Tensile/__init__.py")) == "Tensile"
    assert hook.module_dotted(P("Tensile/Common/__init__.py")) == "Tensile.Common"
    assert hook.module_dotted(P("Tensile/data.yaml")) is None


def test_referenced_modules_collects_imports_and_strings(tmp_path):
    f = tmp_path / "test_sample.py"
    f.write_text(
        "import Tensile.Common.Utilities\n"
        "from Tensile.Common import Utilities\n"
        "from . import sibling\n"
        "m = importlib.import_module('Tensile.Foo.Bar')\n"
        "patch('Tensile.Baz.qux')\n",
        encoding="utf-8",
    )
    refs = hook.referenced_modules(f)
    assert "Tensile.Common.Utilities" in refs
    assert "Tensile.Common" in refs
    assert "Tensile.Foo.Bar" in refs
    assert "Tensile.Baz.qux" in refs
    assert "sibling" not in refs


def test_tests_for_module_respects_dotted_boundary():
    index = {
        P("test_a.py"): {"Tensile.Common.Utilities"},
        P("test_b.py"): {"Tensile.Common"},
        P("test_c.py"): {"Tensile.CommonOther"},
        P("test_d.py"): {"Tensile.Common.Utilities.sub"},
    }
    assert hook.tests_for_module("Tensile.Common", index) == {
        P("test_a.py"), P("test_b.py"), P("test_d.py")
    }
    assert hook.tests_for_module("Tensile.Common.Utilities", index) == {
        P("test_a.py"), P("test_d.py")
    }


def test_classify_staged_buckets():
    staged = [
        TL / "pyproject.toml",
        TL / "scripts" / "helper.py",
        TL / "rocisa" / "src.cpp",
        TL / "Tensile" / "Tests" / "unit" / "conftest.py",
        TL / "Tensile" / "Tests" / "unit" / "__snapshots__" / "x.ambr",
        TL / "Tensile" / "Tests" / "unit" / "test_x.py",
        TL / "Tensile" / "Tests" / "common" / "test_y.py",
        TL / "Tensile" / "Common" / "Utilities.py",
        TL / "Tensile" / "Common" / "data.yaml",
    ]
    broad, tests, sources, ignored = hook.classify_staged(staged)
    broad_str = set(broad)
    assert "pyproject.toml" in broad_str
    assert "scripts/helper.py" in broad_str
    assert "rocisa/src.cpp (native ext)" in broad_str
    assert any("conftest.py" in b for b in broad_str)
    assert any("__snapshots__" in b for b in broad_str)
    assert P("Tensile/Tests/unit/test_x.py") in tests
    assert sources == [P("Tensile/Common/Utilities.py")]
    assert P("Tensile/Tests/common/test_y.py") in ignored
    assert P("Tensile/Common/data.yaml") in ignored


def test_select_tests_threshold_and_escalation():
    index = {P(f"test_{i}.py"): {"Tensile.Big"} for i in range(5)}
    index.update({P(f"test_s{i}.py"): {"Tensile.Small"} for i in range(3)})
    index.update({P(f"test_o{i}.py"): {"Tensile.Other"} for i in range(2)})
    # total = 10; threshold = 0.40 * 10 = 4.0
    sources = [P("Tensile/Big.py"), P("Tensile/Small.py"), P("Tensile/Ghost.py")]
    selected, escalations = hook.select_tests(sources, index)
    assert selected == {P(f"test_s{i}.py") for i in range(3)}  # 3 hits -> selected
    joined = " ".join(escalations)
    assert "too broad" in joined          # Big: 5 hits > 4 -> escalate
    assert "no referencing tests" in joined  # Ghost: 0 hits -> escalate


def test_select_tests_unmappable_path():
    selected, escalations = hook.select_tests([P("Tensile/weird")], index={P("t.py"): set()})
    assert selected == set()
    assert any("unmappable path" in e for e in escalations)


def test_failed_test_files_reads_cache(tmp_path):
    cache = tmp_path / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "lastfailed").write_text(
        json.dumps({"a.py::t1": True, "a.py::t2": True, "b.py::t": True}),
        encoding="utf-8",
    )
    assert hook.failed_test_files(tmp_path) == ["a.py", "b.py"]


def test_failed_test_files_missing_or_bad(tmp_path):
    assert hook.failed_test_files(tmp_path) == []
    cache = tmp_path / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "lastfailed").write_text("{not json", encoding="utf-8")
    assert hook.failed_test_files(tmp_path) == []
