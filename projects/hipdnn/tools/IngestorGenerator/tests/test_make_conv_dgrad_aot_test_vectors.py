# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The committed golden test-vector headers of every dgrad engine are what
tools/make_conv_dgrad_aot_test_vectors.py writes from the committed instance lists and
the rocKE tree beside them, so the C++ geometry and matcher tests cannot drift from the
Python adapter (no compile, no GPU)."""

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch_surface  # noqa: E402
import make_conv_dgrad_aot_test_vectors as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_HEADERS = (
    _REPO / "dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine"
    "/packs"
)
_JOBS = min(16, os.cpu_count() or 1)


@pytest.mark.parametrize(
    "arch, header",
    [
        ("gfx950", "Gfx950ConvDgradTestVectors.hpp"),
        ("gfx942", "Gfx942ConvDgradTestVectors.hpp"),
        ("gfx1250", "Gfx1250ConvDgradTestVectors.hpp"),
    ],
)
def test_the_committed_vectors_are_what_the_tool_writes(
    arch, header, tmp_path, monkeypatch
):
    instances = [
        _CONFIGS / f"{arch}_conv_dgrad_aot_{dtype}.instances.json"
        for dtype in ("fp16", "bf16")
    ]
    missing = sorted(str(p) for p in instances if not p.exists())
    if missing:
        pytest.skip(
            "DVC-tracked instance list(s) not pulled, run `dvc pull` in "
            f"{_CONFIGS}: {missing}"
        )
    # Identities embed the LLVM flavor: lower as the lists were committed.
    flavor = json.loads(instances[0].read_text())[0]["llvm_flavor"]
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    before = list(sys.path)
    try:
        out = tmp_path / header
        argv = ["--rocke-root", str(_ROCKE), "--output", str(out)]
        argv += ["--jobs", str(_JOBS), "--instances"] + [str(p) for p in instances]
        assert tool.main(argv) == 0
    finally:
        sys.path[:] = before
    # Text, both UTF-8: the header's (c) sign is not ASCII, and a checkout's line endings
    # do not matter.
    written = out.read_text(encoding="utf-8")
    assert written == (_HEADERS / header).read_text(encoding="utf-8"), (
        f"{header} is stale: regenerate it with "
        "tools/make_conv_dgrad_aot_test_vectors.py"
    )


def test_without_rocke_root_the_tool_binds_the_in_repo_tree(tmp_path, monkeypatch):
    from test_make_conv_dgrad_aot_config import _bound_root

    argv = ["--instances", str(tmp_path / "x.json"), "--output", str(tmp_path / "o")]
    root = _bound_root(monkeypatch, tool, "_bind_rocke", argv, binder=tool._cfg)
    assert root.resolve() == _ROCKE.resolve()
