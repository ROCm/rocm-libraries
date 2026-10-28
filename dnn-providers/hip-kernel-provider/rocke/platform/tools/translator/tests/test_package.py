# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where a package's files live (tr/package.py): attention backward keeps its
paths, and a platform instance is read from the platform tree."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from trtest import ROCKE, TR

sys.path.insert(0, str(TR))
from tr.package import PACKAGES  # noqa: E402

BWD = PACKAGES["attention_bwd"]
INSTANCE = PACKAGES["transpose"]  # mirrored by hand already


def test_attention_bwd_keeps_its_paths():
    assert (
        BWD.module_name("_attention_bwd_addr") == "kernels.common._attention_bwd_addr"
    )
    assert BWD.header_name("_attention_bwd_addr") == (
        "helper_rocke.instances.common._attention_bwd_addr.h"
    )
    assert BWD.cpp_dir == "platform/cpp/instances/common"
    assert BWD.parity_dir == "library/tests/parity"
    assert BWD.py_globs == ("*attention*bwd*.py",)


def test_platform_instance_paths_exist():
    p = INSTANCE
    assert p.module_name("transpose") == "rocke.instances.common.transpose"
    assert (ROCKE / p.py_root / p.py_dir / "transpose.py").is_file()
    assert (ROCKE / "platform/cpp/include/rocke" / p.header_name("transpose")).is_file()
    assert (ROCKE / p.cpp_dir / "transpose.cpp").is_file()
    assert (ROCKE / p.parity_dir / "transpose_emit.py").is_file()


def test_header_module_round_trip():
    for p, m in ((BWD, "_attention_bwd_valu"), (INSTANCE, "transpose")):
        for name in p.header_names(m):
            assert p.header_module(name) == m
    assert INSTANCE.header_module("ir.h") is None


def test_import_source():
    # from ...core.ir import I32 in rocke/instances/common/<m>.py
    assert INSTANCE.import_source("core.ir", 3) == "rocke.core.ir"
    # from .batched_gemm import ... (a sibling stays bare)
    assert INSTANCE.import_source("batched_gemm", 1) == "batched_gemm"
    # absolute imports are kept as written
    assert BWD.import_source("rocke.helpers.attention_band", 0) == (
        "rocke.helpers.attention_band"
    )


def test_record_reads_a_platform_instance(tmp_path):
    """tr.record finds the module and its parity driver in the platform tree."""
    env = dict(os.environ, TR_PACKAGE="transpose", TR_DATA=str(tmp_path))
    for k in ("TR_LIBRARY", "TR_PARITY", "TR_MAPS", "TR_HOLES"):
        env.pop(k, None)
    r = subprocess.run(
        [sys.executable, "-m", "tr.record", "transpose_emit", "transpose"],
        cwd=TR,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    facts = json.loads((tmp_path / "facts" / "transpose.json").read_text())
    assert facts["built"] == facts["configs"] > 0
    assert "build_transpose2d" in facts["funcs"]


def test_record_names_a_missing_driver(tmp_path):
    env = dict(os.environ, TR_PACKAGE="transpose", TR_DATA=str(tmp_path))
    env.pop("TR_PARITY", None)
    r = subprocess.run(
        [sys.executable, "-m", "tr.record", "no_such_emit", "transpose"],
        cwd=TR,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    assert r.returncode != 0
    assert "no parity driver no_such_emit.py" in r.stderr
