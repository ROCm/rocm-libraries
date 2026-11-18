# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The committed golden test-vector header of hipkernel:Gfx942ConvFwdAot is what
tools/make_conv_aot_test_vectors.py writes from the committed instance lists and the
rocKE tree beside them, so the C++ matcher, selection and geometry tests
(TestGfx942ConvFwdGeometry.cpp) cannot drift from the Python adapter (no compile, no
GPU)."""

import json
import os
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch_surface  # noqa: E402
import make_conv_aot_test_vectors as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_PROVIDER = _REPO / "dnn-providers/hip-kernel-provider"
_ROCKE = _PROVIDER / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_HEADERS = _PROVIDER / "src/tests/engines/kernel_ingestor_engine/packs"
_DESCRIPTORS = _PROVIDER / "src/engines/kernel_ingestor_engine/descriptors/rocKE"
_JOBS = min(16, os.cpu_count() or 1)
#: check-added-large-files' default limit (.pre-commit-config.yaml lists no exclusion
#: for the header), which the generated header must stay below.
_ADDED_FILE_LIMIT = 500 * 1024

#: Each engine whose vectors are committed: its header, and its git-tracked instance
#: lists.
_HEADER = {"gfx942": "Gfx942ConvFwdTestVectors.hpp"}


def _instances(arch):
    return [
        _CONFIGS / f"{arch}_conv_fwd_aot_{dtype}.instances.json"
        for dtype in ("fp16", "bf16")
    ]


@pytest.mark.parametrize("arch", sorted(_HEADER))
def test_the_committed_vectors_are_what_the_tool_writes(arch, tmp_path, monkeypatch):
    header = _HEADER[arch]
    instances = _instances(arch)
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
    assert written == (_HEADERS / header).read_text(
        encoding="utf-8"
    ), f"{header} is stale: regenerate it with tools/make_conv_aot_test_vectors.py"


@pytest.mark.parametrize("arch", sorted(_HEADER))
def test_the_committed_vectors_stay_below_the_added_file_limit(arch):
    assert (_HEADERS / _HEADER[arch]).stat().st_size < _ADDED_FILE_LIMIT


@pytest.mark.parametrize("arch", sorted(_HEADER))
def test_the_vectors_state_the_shipped_kmd(arch):
    # The header lists every KMD field in the shipped descriptor's order, types and
    # defaults, so the C++ builds the metadata the engine loads.
    (kmd,) = (_DESCRIPTORS / f"{arch}_conv_fwd_aot").glob("*.kmd.json")
    shipped = [
        {k: field[k] for k in ("name", "type", "default_value") if k in field}
        for field in json.loads(kmd.read_text())["fields"]
    ]
    assert tool.kmd_fields() == shipped


#: A catalogInstances() row: its dtype; a servedMatrix() row: its dtype and served bits.
_INSTANCE_ROW = re.compile(r'^\s*\{"(\w+)", "[^"]*"\},$', re.M)
_SERVED_ROW = re.compile(
    r'^\s*\{\{[-\d, ]+\}, "(\w+)", "([0-9a-f]*)", "[0-9a-f]*", \d+ULL\},$', re.M
)


@pytest.mark.parametrize("arch", sorted(_HEADER))
def test_every_shipped_instance_serves_some_problem_of_the_vectors(arch):
    # Each instance's admission and launch plan is compared with the engine's, not only
    # its refusals: some row has its served bit.
    text = (_HEADERS / _HEADER[arch]).read_text(encoding="utf-8")
    counts = {}
    for dtype in _INSTANCE_ROW.findall(text):
        counts[dtype] = counts.get(dtype, 0) + 1
    rows = _SERVED_ROW.findall(text)
    assert counts and rows
    for dtype, count in counts.items():
        served = [False] * count
        for row_dtype, bits in (r for r in rows if r[0] == dtype):
            assert len(bits) == -(-count // 4)
            for i in range(count):
                served[i] = served[i] or bool(int(bits[i // 4], 16) >> (3 - i % 4) & 1)
        unserved = [i for i, ok in enumerate(served) if not ok]
        assert not unserved, f"{dtype} instances no problem serves: {unserved}"


def test_hex_bits_put_the_first_instance_in_the_top_bit():
    assert tool.hex_bits([True]) == "8"
    assert tool.hex_bits([False, True, False, True, True]) == "58"
    assert tool.hex_bits([]) == ""


def test_without_rocke_root_the_tool_binds_the_in_repo_tree(tmp_path, monkeypatch):
    from test_make_conv_dgrad_aot_config import _bound_root

    argv = ["--instances", str(tmp_path / "x.json"), "--output", str(tmp_path / "o")]
    root = _bound_root(monkeypatch, tool, "_bind_rocke", argv, binder=tool._cfg)
    assert root.resolve() == _ROCKE.resolve()
