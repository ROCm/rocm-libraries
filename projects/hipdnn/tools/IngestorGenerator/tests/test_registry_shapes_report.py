# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""registry_enumerator ``--shapes``: where request-field mappings fall on the cells."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

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
        {**_COVERED, "seqlen_q": 100, "seqlen_k": 77},
    ]
    text = "\n".join(re_.shapes_report(re_.Registry(one), shapes))
    counts = {
        name: _category(text, name)
        for name in ("covered", "outside cell axes", "declined by registry", "GAP")
    }
    assert counts == {
        "covered": 1,
        "outside cell axes": 2,
        "declined by registry": 4,
        "GAP": 1,
    }, text
    assert "1 x use_sinks=True" in text
    assert "1 x page_size (not a request field)" in text
    assert "head_size must be" in text
    assert "request rejected: mask_type must be" in text
    assert "arch 'gfx942' not in" in text
    assert "seqlen_q must be a multiple of block_m" in text
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
