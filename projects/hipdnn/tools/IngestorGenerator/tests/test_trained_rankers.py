# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Trained ``sort_kernel_catalog`` rankers: ``engine.trained_catalog_sha256`` beside the
``engine.heuristics`` binding, the catalog fingerprint, and the gate that a shipped ranker
was trained on the shipped catalog.

The descriptor loader ties a trained model to its engine's UED/KMD/UMD ids only, so a
catalog edit that keeps them (a kernel added, removed or renamed) leaves the model ranking a
catalog it never saw, with no refusal. ``engine.trained_catalog_sha256.<arch>`` records the
fingerprint of the catalog the models bound under ``engine.heuristics.sort_kernel_catalog.
<arch>`` were trained on, and `TestShippedRankers` fails when the shipped KDP no longer
matches it: retrain (or remove the binding) before shipping the edit.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import ConfigError, load_config
from codegen.generator import build_ued, mint_ids

_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_ROOT / "tools"))

from catalog_fingerprint import catalog_fingerprint  # noqa: E402

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


def _with_rankers(raw: dict, ids, sha256) -> dict:
    """``engine.heuristics.sort_kernel_catalog`` = ``ids`` and
    ``engine.trained_catalog_sha256`` = ``sha256``; ``None`` leaves a key out."""
    out = copy.deepcopy(raw)
    if ids is not None:
        out["engine"]["heuristics"] = {"sort_kernel_catalog": ids}
    if sha256 is not None:
        out["engine"]["trained_catalog_sha256"] = sha256
    return out


# ---------------------------------------------------------------------------
# config -> UED
# ---------------------------------------------------------------------------


def test_the_ued_binds_trained_rankers_beside_the_native_default(
    scale_add_raw, tmp_path
):
    raw = _with_rankers(
        scale_add_raw, {"gfx950": [_TFLOPS_ID, _TIME_ID]}, {"gfx950": _SHA}
    )
    config = load_config(_write(tmp_path, raw))
    assert config.engine.trained_catalog_sha256 == {"gfx950": _SHA}
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
    ("ids", "sha256"),
    [
        (None, {"gfx950": _SHA}),
        ({"gfx950": [_TFLOPS_ID]}, {"gfx942": _SHA}),
        ({"gfx950": [_TFLOPS_ID]}, {"gfx950": _SHA, "gfx942": _SHA}),
        ({"gfx950": [_TFLOPS_ID], "gfx942": [_TIME_ID]}, {"gfx950": _SHA}),
        ({"gfx950": [_TFLOPS_ID]}, {"gfx950": "abc"}),
        ({"gfx950": [_TFLOPS_ID]}, {"gfx950": _SHA.upper()}),
        ({"gfx950": [_TFLOPS_ID]}, {"gfx950": 1}),
        ({"gfx950": [_TFLOPS_ID]}, [_SHA]),
    ],
    ids=[
        "without-a-binding",
        "other-arch",
        "extra-arch",
        "arch-without-sha",
        "short-sha",
        "uppercase-sha",
        "not-a-string",
        "not-a-mapping",
    ],
)
def test_malformed_trained_catalog_sha256_is_refused(
    scale_add_raw, tmp_path, ids, sha256
):
    with pytest.raises(ConfigError, match="trained_catalog_sha256"):
        load_config(_write(tmp_path, _with_rankers(scale_add_raw, ids, sha256)))


def test_a_binding_without_a_fingerprint_gates_nothing(scale_add_raw, tmp_path):
    raw = _with_rankers(scale_add_raw, {"gfx950": [_TFLOPS_ID]}, None)
    assert load_config(_write(tmp_path, raw)).engine.trained_catalog_sha256 == {}


def test_trained_rankers_need_the_native_default(scale_add_raw, tmp_path):
    raw = _with_rankers(scale_add_raw, {"gfx950": [_TFLOPS_ID]}, {"gfx950": _SHA})
    raw["engine"]["heuristic"] = "none"
    with pytest.raises(ConfigError, match="heuristic"):
        load_config(_write(tmp_path, raw))


# ---------------------------------------------------------------------------
# catalog fingerprint
# ---------------------------------------------------------------------------


def _kdp(*kernels) -> dict:
    return {"id": "kdp-id", "kernelDescriptors": list(kernels)}


def _kernel(
    name,
    *,
    kid="k-id",
    tile=64,
    spec_tile=64,
    kind="rocke",
    target=None,
    arch=None,
    priority=None,
) -> dict:
    source = {"kind": kind, "spec": {"tile_size": spec_tile}}
    if target is not None:
        source["target"] = target
    entry = {
        "id": kid,
        "name": name,
        "metadata": {"tile_size": tile},
        "kernel_source": source,
    }
    if arch is not None:
        entry["arch"] = arch
    if priority is not None:
        entry["priority"] = priority
    return entry


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
        _kdp(_kernel("a"), _kernel("b", kind="cktile")),  # builder
        _kdp(_kernel("a"), _kernel("b", target="gfx942")),  # target
        _kdp(_kernel("a"), _kernel("b", arch=["gfx950"])),  # arch
        _kdp(_kernel("a"), _kernel("b", priority=1)),  # priority
    ],
    ids=[
        "added",
        "removed",
        "renamed",
        "metadata",
        "spec",
        "builder",
        "target",
        "arch",
        "priority",
    ],
)
def test_the_fingerprint_changes_with_the_catalog(edited):
    """Every edit ``catalog_diff``'s regeneration gate (g3) catches must also move the
    fingerprint: a builder (kernel_source kind), target, arch or priority edit that the
    gate flags as a different catalog must not leave a trained ranker's fingerprint
    unchanged (it would then rank a catalog it was never trained on, undetected)."""
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


def _binds_a_ranker(path: Path) -> bool:
    """Whether the config binds a trained ``sort_kernel_catalog`` model for any arch.
    Selecting on the binding, not on ``trained_catalog_sha256``, keeps a binding
    that never recorded its catalog from skipping the gate."""
    engine = yaml.safe_load(path.read_text()).get("engine") or {}
    return bool((engine.get("heuristics") or {}).get("sort_kernel_catalog"))


_SELECTIONS = [
    p
    for p in sorted((_ROOT / "configs").glob("*.yaml"))
    if not p.name.endswith(".engine.yaml") and _binds_a_ranker(p)
]


@pytest.mark.parametrize("selection_path", _SELECTIONS, ids=lambda p: p.name)
class TestShippedRankers:
    @pytest.fixture
    def config(self, selection_path):
        return load_config(selection_path)

    @pytest.fixture
    def bundle(self, config) -> Path:
        return _DESCRIPTORS / config.authored_subpath

    def test_every_bound_arch_records_its_trained_catalog(self, config):
        """A ranker bound with no recorded catalog cannot be checked against the
        shipped one, so the binding alone requires the fingerprint."""
        bound = config.engine.heuristics["sort_kernel_catalog"]
        recorded = config.engine.trained_catalog_sha256 or {}
        missing = sorted(set(bound) - set(recorded))
        assert not missing, (
            f"{config.engine.name} binds sort_kernel_catalog for {missing} without "
            f"engine.trained_catalog_sha256: record the catalog the models were trained "
            f"on (tools/catalog_fingerprint.py) in both configs."
        )

    def test_trained_on_the_shipped_catalog(self, config, bundle):
        for pack in config.packs:
            kdp = bundle / "artifacts" / f"{config.kdp_stem(pack)}.kdp.json"
            if not kdp.is_file():
                pytest.skip(f"shipped KDP not in this checkout (dvc pull): {kdp}")
            shipped = catalog_fingerprint(json.loads(kdp.read_text()))
            bound = config.engine.heuristics["sort_kernel_catalog"]
            for arch, trained in config.engine.trained_catalog_sha256.items():
                assert shipped == trained, (
                    f"{config.engine.name}'s {arch} ranker(s) {bound[arch]} were trained on "
                    f"catalog {trained}, but {kdp.name} is now {shipped}. "
                    f"The loader would keep using the stale model silently: retrain on the new "
                    f"catalog (hipdnn-uhd skill) and record the new fingerprint "
                    f"(tools/catalog_fingerprint.py), or remove the binding."
                )

    def test_the_shipped_ued_binds_the_configured_ids(self, config, bundle):
        ued = json.loads((bundle / f"{config.engine.slug}.ued.json").read_text())
        bound = config.engine.heuristics["sort_kernel_catalog"]
        assert set(bound) == set(config.engine.trained_catalog_sha256 or {})
        for arch, ids in bound.items():
            assert ued["sort_kernel_catalog"].get(arch) == ids

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
        for arch, ids in config.engine.heuristics["sort_kernel_catalog"].items():
            missing = sorted(set(ids) - installed)
            assert not missing, f"{arch}: {missing} bound but not under {heuristics}"
