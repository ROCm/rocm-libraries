# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""registry_enumerator ``--shapes``: where request-field mappings fall on the cells."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import registry_enumerator as re_  # noqa: E402

_MANIFEST = _ROOT / "configs" / "gfx950_attention_dense.engine.yaml"
_ROCKE_LIBRARY = (
    re_._REPO / "dnn-providers" / "hip-kernel-provider" / "rocke" / "library"
)

_COVERED = {
    "batch": 1,
    "nhead_q": 128,
    "nhead_k": 8,
    "seqlen_q": 2048,
    "seqlen_k": 2048,
    "hdim_q": 128,
    "hdim_v": 128,
    "dtype": "bf16",
    "mask_type": 1,
    "sliding_window": 0,
    "use_sinks": False,
    "_provenance": {"source": "bench"},
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    manifest = re_.load_manifest(str(_MANIFEST))
    if not _ROCKE_LIBRARY.is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE_LIBRARY}")
    re_.bind_provider(manifest.get("provider_root"))
    return manifest


def _category(text: str, name: str) -> int:
    found = re.search(rf"^  {re.escape(name)}: (\d+) shapes, \d+ cells$", text, re.M)
    assert found, (name, text)
    return int(found.group(1))


def test_each_shape_lands_in_its_first_matching_category(manifest):
    one = copy.deepcopy(manifest)
    one["enumerate"]["cells"][0]["values"] = [[128, 128, 128, 8]]
    shapes = [
        _COVERED,
        {**_COVERED, "use_sinks": True},
        {**_COVERED, "page_size": 16},
        {**_COVERED, "hdim_q": 256, "hdim_v": 256},
        {**_COVERED, "mask_type": 7},
        {**_COVERED, "nhead_q": 64},
        # A fixed field the shape sets is the registry's to judge.
        {**_COVERED, "arch": "gfx942"},
    ]
    lines, gaps, covered = re_.shapes_report(re_.Registry(one), shapes)
    text = "\n".join(lines)
    assert (gaps, covered) == (1, 1)
    counts = {
        name: _category(text, name)
        for name in ("covered", "outside cell axes", "declined by registry", "GAP")
    }
    assert counts == {
        "covered": 1,
        "outside cell axes": 2,
        "declined by registry": 3,
        "GAP": 1,
    }, text
    assert "1 x use_sinks=True" in text
    assert "1 x page_size (not a request field)" in text
    assert "head_size must be" in text
    assert "request rejected: mask_type must be" in text
    assert "arch 'gfx942' not in" in text
    assert re.search(r"1 x \{hdim_q: 128, .*nhead_q: 64", text), text


@pytest.mark.parametrize("payload", ['{"hdim_q": 64}', "[1, 2]", "not json"])
def test_a_malformed_shapes_file_is_exit_2(manifest, tmp_path, capsys, payload):
    bad = tmp_path / "shapes.json"
    bad.write_text(payload)
    out = tmp_path / "c.yaml"
    argv = ["--manifest", str(_MANIFEST), "--out", str(out), "--shapes", str(bad)]
    assert re_.main(argv) == 2
    assert "shapes file" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "extra_shapes, strict_rc",
    [([], 0), ([{**_COVERED, "nhead_q": 64}], 1)],
    ids=["all-covered", "one-gap"],
)
def test_strict_fails_only_a_shapes_gap(
    manifest, tmp_path, capsys, extra_shapes, strict_rc
):
    one = copy.deepcopy(manifest)
    one["enumerate"]["cells"][0]["values"] = [[128, 128, 128, 8]]
    path = tmp_path / "one.engine.yaml"
    path.write_text(yaml.safe_dump(one))
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps([_COVERED, *extra_shapes]))
    out = tmp_path / "missing" / "dir" / "c.yaml"
    argv = ["--manifest", str(path), "--out", str(out), "--shapes", str(shapes)]
    assert re_.main(argv) == 0
    assert out.is_file()
    assert re_.main(argv + ["--strict"]) == strict_rc
    assert ("GAP shapes" in capsys.readouterr().err) is bool(strict_rc)


def test_strict_fails_a_corpus_in_which_nothing_is_covered(manifest, tmp_path, capsys):
    """A wrong pin or arch declines every shape; zero GAPs there is not a pass."""
    one = copy.deepcopy(manifest)
    one["enumerate"]["cells"][0]["values"] = [[128, 128, 128, 8]]
    path = tmp_path / "one.engine.yaml"
    path.write_text(yaml.safe_dump(one))
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps([{**_COVERED, "arch": "gfx942"}]))
    argv = ["--manifest", str(path), "--out", str(tmp_path / "c.yaml")]
    argv += ["--shapes", str(shapes)]
    assert re_.main(argv) == 0
    assert re_.main(argv + ["--strict"]) == 1
    assert "no shape is covered" in capsys.readouterr().err


def _one_cell_registry(manifest):
    one = copy.deepcopy(manifest)
    one["enumerate"]["cells"][0]["values"] = [[128, 128, 128, 8]]
    return re_.Registry(one)


def test_a_null_value_is_that_shapes_decline_not_a_failed_report(manifest):
    """A corpus that was reportable stays reportable: a null dimension raises
    TypeError inside the registry and is declined like any other bad shape."""
    registry = _one_cell_registry(manifest)
    lines, gaps, covered = re_.shapes_report(
        registry, [_COVERED, {**_COVERED, "seqlen_q": None}]
    )
    text = "\n".join(lines)
    assert (_category(text, "covered"), _category(text, "declined by registry")) == (
        1,
        1,
    )
    assert (gaps, covered) == (0, 1)


def test_an_empty_corpus_reports_nothing_and_only_strict_refuses_it(
    manifest, tmp_path, capsys
):
    shapes = tmp_path / "shapes.json"
    shapes.write_text("[]")
    argv = ["--manifest", str(_MANIFEST), "--out", str(tmp_path / "c.yaml")]
    argv += ["--shapes", str(shapes)]
    assert re_.main(argv) == 0
    assert "shape coverage: 0 shapes" in capsys.readouterr().out
    assert re_.main(argv + ["--strict"]) == 1


def test_a_registry_without_admits_is_not_a_decline(manifest):
    registry = _one_cell_registry(manifest)
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
