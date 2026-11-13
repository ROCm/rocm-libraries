# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The generated pack-shape census test is copied into the provider verbatim, so it
must already be in the provider's clang-format form. Otherwise the copy reformats
the live file and an addition-only change shows unrelated hunks.

Each shipped bundle's config (tests.helpers.shipped_bundles) regenerates the
provider's live census test byte-for-byte (no clang-format needed). When a
clang-format binary is named, every shipped config's census test must also be a
clang-format fixed point under the provider's style, and so must one whose names
straddle the column limit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from codegen.config_loader import load_config
from codegen.generator import IngestorGenerator
from codegen.models import IngestorConfig
from tests.helpers import make_engine, make_minimal_config, shipped_bundles

_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PROVIDER = _REPO_ROOT / "dnn-providers" / "hip-kernel-provider"
_LIVE_PACKS_DIR = _PROVIDER / "src/tests/engines/kernel_ingestor_engine/packs"
_BUNDLES = shipped_bundles()
# Config stems under configs/.
_SHIPPED_CONFIGS = (
    "axes_example",
    "binary_ops",
    "scale_add",
    "variants_example",
    *_BUNDLES,
)
# Names either side of the column limit on each census line the template wraps: the knob
# checks and the score-registration check. A knob past 24 characters, or an engine local
# name past 30, needs a layout the template does not emit.
_KNOBS = ["k" * n for n in range(1, 25)]
_LOCAL_NAME_LENGTHS = range(1, 31)


def _render_packs_file(config: IngestorConfig, out: Path) -> Path:
    IngestorGenerator(_ROOT / "templates").render(config, out)
    (packs,) = (out / "tests").glob("Test*Packs.cpp")
    return packs


def _render_shipped_packs_file(stem: str, out: Path) -> Path:
    return _render_packs_file(load_config(_ROOT / "configs" / f"{stem}.yaml"), out)


@pytest.mark.skipif(
    not _LIVE_PACKS_DIR.is_dir(), reason="provider tree not checked out"
)
@pytest.mark.parametrize("bundle", _BUNDLES)
def test_census_test_regenerates_the_live_file(tmp_path, bundle):
    """Each shipped bundle's census test is committed under the name the generator
    gives it, and is the generator's output."""
    rendered = _render_shipped_packs_file(bundle, tmp_path)
    live = _LIVE_PACKS_DIR / rendered.name
    assert live.is_file(), f"{rendered.name} is not committed in {_LIVE_PACKS_DIR}"
    assert rendered.read_text(encoding="utf-8") == live.read_text(encoding="utf-8")


def _clang_format() -> str | None:
    named = os.environ.get("HIPDNN_CLANG_FORMAT")
    if named:
        return named
    return shutil.which("clang-format")


def _clang_formatted(text: str) -> str:
    return subprocess.run(
        [
            _clang_format(),
            f"--style=file:{_PROVIDER / '.clang-format'}",
            "--assume-filename=TestPacks.cpp",
        ],
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


_NEEDS_CLANG_FORMAT = pytest.mark.skipif(
    _clang_format() is None,
    reason="no clang-format (set HIPDNN_CLANG_FORMAT; pre-commit pins v18.1.4)",
)


@_NEEDS_CLANG_FORMAT
@pytest.mark.parametrize("config", _SHIPPED_CONFIGS)
def test_census_test_is_a_clang_format_fixed_point(tmp_path, config):
    rendered = _render_shipped_packs_file(config, tmp_path).read_text(encoding="utf-8")
    assert rendered == _clang_formatted(rendered)


@_NEEDS_CLANG_FORMAT
@pytest.mark.parametrize("heuristic", ("native", "none"))
def test_census_test_is_a_fixed_point_either_side_of_the_column_limit(
    tmp_path, heuristic
):
    """No shipped config takes the heuristic-free arm, whose registration check wraps
    too."""
    for length in _LOCAL_NAME_LENGTHS:
        engine = make_engine(
            name="hipkernel:E" + "e" * (length - 1), knobs=_KNOBS, heuristic=heuristic
        )
        config = make_minimal_config(engine=engine)
        packs = _render_packs_file(config, tmp_path / str(length))
        rendered = packs.read_text(encoding="utf-8")
        assert rendered == _clang_formatted(rendered), config.engine.name
