# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The generated pack-shape census test is copied into the provider verbatim, so it
must already be in the provider's clang-format form. Otherwise the copy reformats
the live file and an addition-only change shows unrelated hunks.

The shipped gfx950 config regenerates the provider's live census test byte-for-byte
(no clang-format needed). When a clang-format binary is named, every shipped config's
census test must also be a clang-format fixed point under the provider's style.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from codegen.config_loader import load_config
from codegen.generator import IngestorGenerator

_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PROVIDER = _REPO_ROOT / "dnn-providers" / "hip-kernel-provider"
_LIVE_PACKS = (
    _PROVIDER
    / "src/tests/engines/kernel_ingestor_engine/packs/TestGfx950AttentionDensePacks.cpp"
)
_SHIPPED_CONFIGS = (
    "axes_example.yaml",
    "binary_ops.yaml",
    "gfx950_attention_dense.yaml",
    "scale_add.yaml",
    "variants_example.yaml",
)


def _render_packs_test(config_name: str, out: Path) -> str:
    config = load_config(_ROOT / "configs" / config_name)
    IngestorGenerator(_ROOT / "templates").render(config, out)
    (packs,) = (out / "tests").glob("Test*Packs.cpp")
    return packs.read_text(encoding="utf-8")


@pytest.mark.skipif(not _LIVE_PACKS.is_file(), reason="provider tree not checked out")
def test_gfx950_census_test_regenerates_the_live_file(tmp_path):
    rendered = _render_packs_test("gfx950_attention_dense.yaml", tmp_path)
    assert rendered == _LIVE_PACKS.read_text(encoding="utf-8")


def _clang_format() -> str | None:
    named = os.environ.get("HIPDNN_CLANG_FORMAT")
    if named:
        return named
    return shutil.which("clang-format")


@pytest.mark.skipif(
    _clang_format() is None,
    reason="no clang-format (set HIPDNN_CLANG_FORMAT; pre-commit pins v18.1.4)",
)
@pytest.mark.parametrize("config_name", _SHIPPED_CONFIGS)
def test_census_test_is_a_clang_format_fixed_point(tmp_path, config_name):
    rendered = _render_packs_test(config_name, tmp_path)
    formatted = subprocess.run(
        [
            _clang_format(),
            f"--style=file:{_PROVIDER / '.clang-format'}",
            "--assume-filename=TestPacks.cpp",
        ],
        input=rendered,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    assert rendered == formatted
