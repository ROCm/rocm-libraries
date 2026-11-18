# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""registry_enumerator ``--shapes``: where request-field mappings fall on the cells.

The classification, precedence, strict and error behaviours run over a small
synthetic registry (tests/synthetic_registry.py), so they do not depend on what rocKE
admits. One case reads the shipped manifest against the live registry and asserts only
the report's own shape.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import registry_enumerator as re_  # noqa: E402
from tests import synthetic_registry as synth  # noqa: E402

_MANIFEST = _ROOT / "configs" / "gfx950_attention_dense.engine.yaml"
_ROCKE_LIBRARY = (
    re_._REPO / "dnn-providers" / "hip-kernel-provider" / "rocke" / "library"
)

#: Listed by the synthetic manifest below, and admitted by its candidate.
_COVERED = {
    "hdim": 64,
    "heads": 8,
    "dtype": "bf16",
    "mask": 1,
    "_provenance": {"source": "bench"},
}
_CATEGORIES = ("outside cell axes", "declined by registry", "covered", "GAP")


@pytest.fixture
def one_cell(tmp_path, monkeypatch):
    """The synthetic manifest narrowed to the single cell (hdim 64, heads 8, mask
    CAUSAL), written to disk for the CLI."""
    root = synth.provider(tmp_path, monkeypatch)
    manifest = synth.probe_manifest(
        root,
        cells=[
            {"fields": ["hdim", "heads"], "values": [[64, 8]]},
            {"fields": ["mask"], "values": ["CAUSAL"]},
        ],
    )
    path = synth.write_manifest(tmp_path / "one.engine.yaml", manifest)
    return types.SimpleNamespace(
        path=path, manifest=re_.load_manifest(str(path)), tmp=tmp_path
    )


def _category(text: str, name: str) -> int:
    found = re.search(rf"^  {re.escape(name)}: (\d+) shapes, \d+ cells$", text, re.M)
    assert found, (name, text)
    return int(found.group(1))


def _counts(lines) -> dict:
    text = "\n".join(lines)
    return {name: _category(text, name) for name in _CATEGORIES}


def _run(one_cell, shapes, *flags):
    shapes_file = one_cell.tmp / "shapes.json"
    shapes_file.write_text(json.dumps(shapes))
    out = one_cell.tmp / "missing" / "dir" / "c.yaml"
    argv = ["--manifest", str(one_cell.path), "--out", str(out)]
    argv += ["--shapes", str(shapes_file), *flags]
    return re_.main(argv), out


def test_each_shape_lands_in_its_first_matching_category(one_cell):
    shapes = [
        _COVERED,
        {**_COVERED, "page_size": 16},
        # Outside the cell axes and declined: the earlier category wins.
        {**_COVERED, "batch": 2, "hdim": 256},
        {**_COVERED, "hdim": 256},
        {**_COVERED, "mask": 7},
        {**_COVERED, "hdim": 128},
        # A fixed field the shape sets is the registry's to judge.
        {**_COVERED, "arch": "other_arch"},
    ]
    lines, gaps, covered = re_.shapes_report(re_.Registry(one_cell.manifest), shapes)
    assert (gaps, covered) == (1, 1)
    assert _counts(lines) == {
        "outside cell axes": 2,
        "declined by registry": 3,
        "covered": 1,
        "GAP": 1,
    }, lines
    text = "\n".join(lines)
    assert "1 x page_size (not a request field)" in text
    assert "1 x batch=2" in text


@pytest.mark.parametrize("payload", ['{"hdim": 64}', "[1, 2]", "not json"])
def test_a_malformed_shapes_file_is_exit_2(one_cell, capsys, payload):
    bad = one_cell.tmp / "shapes.json"
    bad.write_text(payload)
    out = one_cell.tmp / "c.yaml"
    argv = ["--manifest", str(one_cell.path), "--out", str(out), "--shapes", str(bad)]
    assert re_.main(argv) == 2
    assert "shapes file" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "extra_shapes, strict_rc",
    [([], 0), ([{**_COVERED, "hdim": 128}], 1)],
    ids=["all-covered", "one-gap"],
)
def test_strict_fails_only_a_shapes_gap(one_cell, capsys, extra_shapes, strict_rc):
    rc, out = _run(one_cell, [_COVERED, *extra_shapes])
    assert rc == 0
    assert out.is_file()
    rc, _ = _run(one_cell, [_COVERED, *extra_shapes], "--strict")
    assert rc == strict_rc
    assert ("GAP shapes" in capsys.readouterr().err) is bool(strict_rc)


def test_strict_fails_a_corpus_in_which_nothing_is_covered(one_cell, capsys):
    """A wrong pin or arch declines every shape; zero GAPs there is not a pass."""
    shapes = [{**_COVERED, "arch": "other_arch"}]
    assert _run(one_cell, shapes)[0] == 0
    assert _run(one_cell, shapes, "--strict")[0] == 1
    assert "no shape is covered" in capsys.readouterr().err


def test_a_null_value_is_that_shapes_decline_not_a_failed_report(one_cell):
    """A corpus that was reportable stays reportable: a null dimension raises
    TypeError inside the registry and is declined like any other bad shape."""
    registry = re_.Registry(one_cell.manifest)
    lines, gaps, covered = re_.shapes_report(
        registry, [_COVERED, {**_COVERED, "hdim": None}]
    )
    counts = _counts(lines)
    assert (counts["covered"], counts["declined by registry"]) == (1, 1)
    assert (gaps, covered) == (0, 1)


def test_an_empty_corpus_reports_nothing_and_only_strict_refuses_it(one_cell, capsys):
    assert _run(one_cell, [])[0] == 0
    assert "shape coverage: 0 shapes" in capsys.readouterr().out
    assert _run(one_cell, [], "--strict")[0] == 1


def test_a_registry_without_admits_is_not_a_decline(one_cell):
    registry = re_.Registry(one_cell.manifest)
    real = registry.candidate

    class Broken:
        def __getattr__(self, name):
            if name == "admits":
                raise AttributeError("no admits")
            return getattr(real, name)

    registry.candidate = Broken()
    with pytest.raises(
        re_.EnumerationError, match=r"cannot judge shape.*AttributeError"
    ):
        re_.shapes_report(registry, [_COVERED])


@pytest.mark.skipif(
    not _ROCKE_LIBRARY.is_dir(),
    reason=f"rocKE library not in this checkout: {_ROCKE_LIBRARY}",
)
def test_the_shipped_manifest_reports_every_shape_in_exactly_one_category(
    tmp_path, capsys
):
    """Against the live registry: which category a shape lands in is rocKE's call;
    the report's shape is the tool's."""
    manifest = re_.load_manifest(str(_MANIFEST))
    decl = manifest["enumerate"]
    cell = {
        f: v
        for group in decl["cells"][:1]
        for f, v in zip(group["fields"], group["values"][0])
    }
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps([{**decl["fixed"], **cell}]))
    out = tmp_path / "c.yaml"
    argv = ["--manifest", str(_MANIFEST), "--out", str(out), "--shapes", str(shapes)]
    assert re_.main(argv) == 0
    assert out.is_file()
    text = capsys.readouterr().out
    assert sum(_category(text, name) for name in _CATEGORIES) == 1
