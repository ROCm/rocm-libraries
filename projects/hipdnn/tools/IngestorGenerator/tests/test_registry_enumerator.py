# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""registry_enumerator: the manifest + rocKE registry -> generator config.

The manifest-only tests run anywhere. Those that walk the registry are skipped
when rocKE is not importable (a plain IngestorGenerator checkout).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))

import registry_enumerator as re_  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "configs" / "gfx950_attention_dense.engine.yaml"
_PROVIDER = re_._REPO / "dnn-providers" / "hip-kernel-provider"
_SHIPPED = (
    _PROVIDER
    / "src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_attention_dense"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    pytest.importorskip("yaml")
    return re_.load_manifest(str(_MANIFEST))


@pytest.fixture(scope="module")
def bound(manifest):
    try:
        re_.bind_provider(manifest.get("provider_root"))
        re_._attention_slice(manifest)
    except re_.EnumerationError as exc:
        pytest.skip(f"rocKE registry not available: {exc}")


def test_manifest_declares_what_the_enumerator_needs(manifest):
    decl = manifest["enumerate"]
    assert decl["algorithm"] == "attention_dense_grid"
    assert decl["max_kernels"] == 840
    assert set(decl["free_fields"]) == {"block_m", "block_n"}
    # Every metadata field is projected, so no kernel can fail at name time.
    assert set(manifest["specialization"]["metadata_fields"]) <= set(decl["spec_fields"])


def test_manifest_missing_a_section_is_named(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"engine": {}}))
    with pytest.raises(re_.EnumerationError, match="missing 'kmd_fields'"):
        re_.load_manifest(str(path))


def test_missing_provider_root_is_an_error(monkeypatch):
    monkeypatch.delenv("ROCKE_PROVIDER_ROOT", raising=False)
    with pytest.raises(re_.EnumerationError, match="provider_root"):
        re_.bind_provider(None)


def test_enumerates_840(manifest, bound):
    cells = re_.enumerate_specs(manifest)
    assert len(cells) == 840
    assert len({(c, ks.block_m, ks.block_n) for c, ks in cells}) == 840


def test_cell_lists_default_tile_first_then_sorted(manifest, bound):
    cells = re_.enumerate_specs(manifest)
    first_cell = cells[0][0]
    tiles = [(ks.block_m, ks.block_n) for c, ks in cells if c == first_cell]
    assert tiles[1:] == sorted(tiles[1:])
    assert tiles[0] not in tiles[1:]


def test_unserved_cell_is_an_error_naming_it(manifest, bound):
    broken = copy.deepcopy(manifest)
    # The dense candidate serves head sizes 64 and 128 only.
    broken["enumerate"]["head_configs"] = {96: [[8, 8]]}
    with pytest.raises(re_.EnumerationError, match=r"'bf16', 96, 8, 8|head_size must be"):
        re_.enumerate_specs(broken)


def test_over_max_kernels_is_an_error(manifest, bound):
    small = copy.deepcopy(manifest)
    small["enumerate"]["max_kernels"] = 839
    with pytest.raises(re_.EnumerationError, match="max_kernels 839"):
        re_.enumerate_specs(small)


def test_unknown_algorithm_lists_the_available_ones(manifest, bound):
    wrong = copy.deepcopy(manifest)
    wrong["enumerate"]["algorithm"] = "no_such_candidate"
    with pytest.raises(re_.EnumerationError, match="Available"):
        re_._attention_slice(wrong)


def test_unrenderable_name_template_is_an_error(manifest, bound):
    bad = copy.deepcopy(manifest)
    bad["kernel_name_template"] = "{no_such_field}"
    with pytest.raises(re_.EnumerationError, match="cannot be rendered"):
        re_.build_config(bad)


def test_regeneration_is_a_no_op_against_the_shipped_catalog(manifest, bound, tmp_path):
    """enumerate -> generate -> catalog_diff, with only UUIDs allowed to differ."""
    pytest.importorskip("jinja2")
    pytest.importorskip("pydantic")
    if not _SHIPPED.is_dir():
        pytest.skip(f"shipped descriptors not in this checkout: {_SHIPPED}")
    config = tmp_path / "config.yaml"
    out = tmp_path / "out"
    assert re_.main(["--manifest", str(_MANIFEST), "--out", str(config)]) == 0
    gen = subprocess.run(
        [sys.executable, str(_ROOT / "generate.py"), "--config", str(config),
         "--output-dir", str(out), "--force"],
        capture_output=True,
        text=True,
    )
    assert gen.returncode == 0, gen.stderr[-2000:]
    regenerated = next(out.rglob("gfx950_attention_dense.kdp.json"))
    shipped = _SHIPPED / "gfx950_attention_dense.kdp.json"
    for strict in ([], ["--strict-spec"]):
        diff = subprocess.run(
            [sys.executable, str(_TOOLS / "catalog_diff.py"), "--shipped", str(shipped),
             "--regenerated", str(regenerated), "--ignore-ids", *strict],
            capture_output=True,
            text=True,
        )
        assert diff.returncode == 0, diff.stdout + diff.stderr
    assert len(json.loads(regenerated.read_text())["kernelDescriptors"]) == 840


def test_provider_root_is_repository_relative_and_binds_from_any_directory(
    manifest, tmp_path, monkeypatch
):
    """A correct resolver paired with a stale manifest string still fails here."""
    root = manifest["provider_root"]
    assert not os.path.isabs(root), (
        f"the manifest names an absolute provider root ({root!r}); it would only "
        "resolve on the machine that wrote it"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        re_.bind_provider(root)
    except re_.EnumerationError as exc:
        pytest.skip(f"rocKE not in this checkout: {exc}")


def test_manifest_pins_the_non_persistent_eight_argument_arm(manifest):
    """Unpinned, ``dispatch_attention`` returns the unified routing path -- another
    kernel family on the paged ABI -- and the persistent arms are five-argument
    kernels this engine's expectedSignature check would reject."""
    decl = manifest["enumerate"]
    assert decl["algorithm"] == "attention_dense_grid"
    assert decl["spec_id"] == "gfx950_dense_grid"


def test_every_enumerated_spec_is_the_non_persistent_arm(manifest, bound):
    for _, ks in re_.enumerate_specs(manifest):
        assert ks.persistent is False, ks
        assert ks.wide_lds_dma is False, ks


def test_the_persistent_arm_exists_in_the_registry_but_is_not_enumerated(manifest, bound):
    """The control: the persistent algorithm IS in the registry, so the test above
    exercises the manifest's pin rather than a registry that has no such arm."""
    from dispatch.attention import attention_candidates

    algorithms = {c.algorithm for c in attention_candidates()}
    assert "attention_dense_persist" in algorithms
    assert manifest["enumerate"]["algorithm"] != "attention_dense_persist"
