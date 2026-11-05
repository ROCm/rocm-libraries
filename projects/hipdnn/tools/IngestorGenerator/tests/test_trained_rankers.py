# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Trained ``sort_kernel_catalog`` rankers: the config binding the generator emits, the
catalog fingerprint, and the gate that a shipped ranker was trained on the shipped catalog.

The descriptor loader ties a trained model to its engine's UED/KMD/UMD ids only, so a
catalog edit that keeps them (a kernel added, removed or renamed) leaves the model ranking a
catalog it never saw, with no refusal. ``engine.trained_rankers.<arch>`` records the
fingerprint of the catalog the model was trained on, and `TestShippedRankers` fails when the
shipped KDP no longer matches it: retrain (or remove the binding) before shipping the edit.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from codegen.catalog_fingerprint import catalog_fingerprint
from codegen.config_loader import ConfigError, load_config
from codegen.generator import build_ued, mint_ids

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[3]
_DESCRIPTORS = (
    _REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
)

_TFLOPS_ID = "11111111-2222-4333-8444-555555555555"
_TIME_ID = "66666666-7777-4888-9999-aaaaaaaaaaaa"
_SHA = "ab" * 32


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


@pytest.fixture
def scale_add_raw(config_path) -> dict:
    return yaml.safe_load(config_path("scale_add.yaml").read_text())


def _with_rankers(raw: dict, rankers) -> dict:
    out = copy.deepcopy(raw)
    out["engine"]["trained_rankers"] = rankers
    return out


# ---------------------------------------------------------------------------
# config -> UED
# ---------------------------------------------------------------------------


def test_the_ued_binds_trained_rankers_beside_the_native_default(
    scale_add_raw, tmp_path
):
    raw = _with_rankers(
        scale_add_raw,
        {"gfx950": {"uhd_ids": [_TFLOPS_ID, _TIME_ID], "trained_catalog_sha256": _SHA}},
    )
    config = load_config(_write(tmp_path, raw))
    ids = mint_ids(config)
    ued = build_ued(config, ids)
    assert ued["sort_kernel_catalog"] == {
        "default": ids["uhd"],
        "gfx950": [_TFLOPS_ID, _TIME_ID],
    }
    # The default stays first, as uhd_gen promote leaves it.
    assert list(ued["sort_kernel_catalog"])[0] == "default"


def test_no_trained_rankers_leaves_only_the_default(scale_add_config):
    ids = mint_ids(scale_add_config)
    assert build_ued(scale_add_config, ids)["sort_kernel_catalog"] == {
        "default": ids["uhd"]
    }


@pytest.mark.parametrize(
    "rankers",
    [
        {},
        {"default": {"uhd_ids": [_TFLOPS_ID], "trained_catalog_sha256": _SHA}},
        {"gfx950": {"uhd_ids": [], "trained_catalog_sha256": _SHA}},
        {"gfx950": {"uhd_ids": ["not-a-uuid"], "trained_catalog_sha256": _SHA}},
        {
            "gfx950": {
                "uhd_ids": [_TFLOPS_ID, _TFLOPS_ID],
                "trained_catalog_sha256": _SHA,
            }
        },
        {"gfx950": {"uhd_ids": [_TFLOPS_ID], "trained_catalog_sha256": "abc"}},
        {"gfx950": {"uhd_ids": [_TFLOPS_ID]}},
        {"gfx950": {"uhd_ids": [_TFLOPS_ID], "trained_catalog_sha256": _SHA, "x": 1}},
    ],
    ids=[
        "empty",
        "default-key",
        "no-ids",
        "bad-uuid",
        "repeated-id",
        "short-sha",
        "no-sha",
        "unknown-key",
    ],
)
def test_malformed_trained_rankers_are_refused(scale_add_raw, tmp_path, rankers):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, _with_rankers(scale_add_raw, rankers)))


def test_trained_rankers_need_the_native_default(scale_add_raw, tmp_path):
    raw = _with_rankers(
        scale_add_raw,
        {"gfx950": {"uhd_ids": [_TFLOPS_ID], "trained_catalog_sha256": _SHA}},
    )
    raw["engine"]["heuristic"] = "none"
    with pytest.raises(ConfigError, match="heuristic"):
        load_config(_write(tmp_path, raw))


# ---------------------------------------------------------------------------
# catalog fingerprint
# ---------------------------------------------------------------------------


def _kdp(*kernels) -> dict:
    return {"id": "kdp-id", "kernelDescriptors": list(kernels)}


def _kernel(name, *, kid="k-id", tile=64, spec_tile=64) -> dict:
    return {
        "id": kid,
        "name": name,
        "metadata": {"tile_size": tile},
        "kernel_source": {"kind": "rocke", "spec": {"tile_size": spec_tile}},
    }


def test_the_fingerprint_ignores_ids_and_kernel_order():
    a, b = _kernel("a", kid="1"), _kernel("b", kid="2")
    renumbered = _kernel("a", kid="9"), _kernel("b", kid="8")
    assert catalog_fingerprint(_kdp(a, b)) == catalog_fingerprint(
        _kdp(*reversed(renumbered))
    )


@pytest.mark.parametrize(
    "edited",
    [
        _kdp(_kernel("a"), _kernel("b"), _kernel("c")),  # added
        _kdp(_kernel("a")),  # removed
        _kdp(_kernel("a"), _kernel("b2")),  # renamed
        _kdp(_kernel("a"), _kernel("b", tile=32)),  # metadata
        _kdp(_kernel("a"), _kernel("b", spec_tile=32)),  # spec
    ],
    ids=["added", "removed", "renamed", "metadata", "spec"],
)
def test_the_fingerprint_changes_with_the_catalog(edited):
    assert catalog_fingerprint(edited) != catalog_fingerprint(
        _kdp(_kernel("a"), _kernel("b"))
    )


def test_the_fingerprint_covers_multi_launch_specs():
    def launches(reduce_tile):
        return {
            "id": "k",
            "name": "m",
            "metadata": {},
            "sources": {
                "segment": {"spec": {"t": 1}},
                "reduce": {"spec": {"t": reduce_tile}},
            },
        }

    assert catalog_fingerprint(_kdp(launches(1))) != catalog_fingerprint(
        _kdp(launches(2))
    )


# ---------------------------------------------------------------------------
# the shipped rankers: trained on the shipped catalog, bound as configured
# ---------------------------------------------------------------------------

_SELECTIONS = [
    p
    for p in sorted((_ROOT / "configs").glob("*.yaml"))
    if not p.name.endswith(".engine.yaml")
    and (yaml.safe_load(p.read_text()).get("engine") or {}).get("trained_rankers")
]


@pytest.mark.parametrize("selection_path", _SELECTIONS, ids=lambda p: p.name)
class TestShippedRankers:
    @pytest.fixture
    def config(self, selection_path):
        return load_config(selection_path)

    @pytest.fixture
    def bundle(self, config) -> Path:
        return _DESCRIPTORS / config.authored_subpath

    def test_trained_on_the_shipped_catalog(self, config, bundle):
        for pack in config.packs:
            kdp = bundle / "artifacts" / f"{config.kdp_stem(pack)}.kdp.json"
            if not kdp.is_file():
                pytest.skip(f"shipped KDP not in this checkout (dvc pull): {kdp}")
            shipped = catalog_fingerprint(json.loads(kdp.read_text()))
            for arch, ranker in config.engine.trained_rankers.items():
                assert shipped == ranker.trained_catalog_sha256, (
                    f"{config.engine.name}'s {arch} ranker(s) {ranker.uhd_ids} were trained on "
                    f"catalog {ranker.trained_catalog_sha256}, but {kdp.name} is now {shipped}. "
                    f"The loader would keep using the stale model silently: retrain on the new "
                    f"catalog (hipdnn-uhd skill) and record the new fingerprint "
                    f"(tools/catalog_fingerprint.py), or remove the binding."
                )

    def test_the_shipped_ued_binds_the_configured_ids(self, config, bundle):
        ued = json.loads((bundle / f"{config.engine.slug}.ued.json").read_text())
        for arch, ranker in config.engine.trained_rankers.items():
            assert ued["sort_kernel_catalog"].get(arch) == ranker.uhd_ids

    def test_each_bound_id_is_installed(self, config, bundle):
        heuristics = bundle / "heuristics"
        if not heuristics.is_dir():
            pytest.skip(
                f"heuristic models not in this checkout (dvc pull): {heuristics}"
            )
        installed = {
            json.loads(p.read_text())["id"]
            for p in heuristics.rglob("heuristic.uhd.json")
        }
        for arch, ranker in config.engine.trained_rankers.items():
            missing = sorted(set(ranker.uhd_ids) - installed)
            assert not missing, f"{arch}: {missing} bound but not under {heuristics}"
