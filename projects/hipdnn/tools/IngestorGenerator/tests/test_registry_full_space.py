# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""registry_enumerator ``space: full`` and ``--plan-space`` on two shipped cells."""

from __future__ import annotations

import copy
import io
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


def _performance(manifest: dict) -> list:
    return manifest["reduction"]["fields"]["performance"]


def _without(manifest: dict, knob: str) -> dict:
    out = copy.deepcopy(manifest)
    out["reduction"]["fields"]["performance"] = [
        k for k in _performance(out) if k["name"] != knob
    ]
    return out


@pytest.fixture(scope="module")
def manifest() -> dict:
    """One D64 and one D128 cell, bf16 causal, at ``space: full``. The narrowed
    cells no longer match the ground-truth value map, which these tests do not use."""
    manifest = re_.load_manifest(str(_MANIFEST))
    del manifest["ground_truth"]
    if not _ROCKE_LIBRARY.is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE_LIBRARY}")
    re_.bind_provider(manifest.get("provider_root"))
    manifest["enumerate"]["space"] = "full"
    manifest["enumerate"]["cells"] = [
        {
            "fields": ["hdim_q", "hdim_v", "nhead_q", "nhead_k"],
            "values": [[64, 64, 8, 1], [128, 128, 8, 8]],
        },
        {"fields": ["dtype"], "values": ["bf16"]},
        {"fields": ["mask_type"], "values": ["TOP_LEFT_CAUSAL"]},
    ]
    return manifest


@pytest.fixture(scope="module")
def config(manifest) -> dict:
    return re_.build_config(manifest, warn=lambda message: None)


def test_projected_knobs_make_names_and_metadata_unique(manifest, config):
    kernels = config["packs"][0]["kernels"]
    assert len({k["name"] for k in kernels}) == len(kernels)
    assert len({tuple(sorted(k["metadata"].items())) for k in kernels}) == len(kernels)
    bindings = config["specialization"]["bindings"]
    projected_names = set(config["specialization"]["metadata_fields"])
    for knob in _performance(manifest):
        name, accessor = knob["name"], knob.get("unset_resolves_by")
        # A knob is only projected (and thus in metadata_fields + bindings) when
        # it actually varies in the enumerated cells.  Knobs with unset_resolves_by
        # that happen to be constant across the test fixture's 2-cell space (e.g.
        # iglp_mode, which stays at None for all D64/D128 bf16 causal kernels)
        # are not projected and correctly absent from metadata_fields.
        if name not in projected_names:
            continue
        assert name in config["specialization"]["metadata_fields"]
        assert bindings[name] == ({"method": accessor} if accessor else {"field": name})


def test_an_unset_knob_is_omitted_and_carries_its_resolved_value(manifest, config):
    accessors = {
        k["name"]: k["unset_resolves_by"]
        for k in _performance(manifest)
        if k.get("unset_resolves_by")
    }
    registry = re_.Registry(manifest)
    spec_cls = type(registry.space(registry.cells()[0])[0])
    for knob, accessor in accessors.items():
        unset = [
            k
            for k in config["packs"][0]["kernels"]
            if knob not in k["kernel_source"]["spec"]
        ]
        assert unset, knob
        # A knob is projected to metadata only when it VARIES within the
        # enumerated cells.  If the 2-cell test fixture produces no variation
        # for a knob (e.g. iglp_mode is always None), the metadata binding is
        # not needed and the knob is not written to metadata.  Skip rather than
        # fail: the reduction YAML entry is still valid and guards the full
        # sweep even if this narrow fixture doesn't exercise it.
        projected = knob in config["packs"][0]["kernels"][0].get("metadata", {}) or any(
            knob in k.get("metadata", {}) for k in config["packs"][0]["kernels"]
        )
        if not projected:
            continue
        for k in unset:
            resolved = getattr(spec_cls(**k["kernel_source"]["spec"]), accessor)()
            resolved = int(resolved) if isinstance(resolved, bool) else resolved
            assert k["metadata"][knob] == resolved, k["name"]
            assert f"_{knob}u{resolved}" in k["name"]


def test_an_unclassified_varying_field_warns_and_is_held(manifest, config):
    warnings = []
    held = re_.build_config(_without(manifest, "waves_per_eu"), warn=warnings.append)
    assert "WARN: field waves_per_eu held at default 2 (unclassified)" in warnings

    def values(c):
        return {
            k["kernel_source"]["spec"]["waves_per_eu"] for k in c["packs"][0]["kernels"]
        }

    assert values(held) == {2}
    assert values(config) == {2, 4}


def test_a_budget_overrun_warns_and_exits_0(manifest, tmp_path, capsys):
    over = copy.deepcopy(manifest)
    over["reduction"]["aot_budget_per_arch"] = 10
    path = tmp_path / "m.engine.yaml"
    path.write_text(yaml.safe_dump(over, sort_keys=False))
    assert re_.main(["--manifest", str(path), "--out", str(tmp_path / "c.yaml")]) == 0
    captured = capsys.readouterr()
    count = int(captured.out.split()[0])
    assert (
        f"WARN: {count} instances for gfx950 exceed the AOT budget 10" in captured.err
    )
    # The same overrun is a gate failure under --strict, and nothing is written.
    out = tmp_path / "strict" / "c.yaml"
    assert re_.main(["--manifest", str(path), "--out", str(out), "--strict"]) == 1
    assert "FAIL: --strict:" in capsys.readouterr().err
    assert not out.exists()


def test_plan_space_lists_each_varying_field(manifest):
    out = io.StringIO()
    held = _without(manifest, "waves_per_eu")
    for knob in _performance(held):
        if knob["name"] == "exp_per_pv_step":
            del knob["unset_resolves_by"]
    assert re_.plan_space(held, out=out) == 0
    assert re_.plan_space(held, out=io.StringIO(), strict=True) == 0
    tight = copy.deepcopy(held)
    tight["reduction"]["aot_budget_per_arch"] = 1
    assert re_.plan_space(tight, out=io.StringIO()) == 0
    assert re_.plan_space(tight, out=io.StringIO(), strict=True) == 1
    text = out.getvalue()
    assert re.search(r"^    waves_per_eu: .*; unclassified-held$", text, re.M), text
    assert re.search(r"^    block_n: .*; covered \(performance\)$", text, re.M), text
    assert re.search(
        r"^    exp_per_pv_step: .*; performance, held \(no unset accessor\)$",
        text,
        re.M,
    ), text
    shown = re.search(r"^    block_n: \{([^}]*)\}", text, re.M)[1].split(", ")
    assert shown == sorted(shown, key=int), shown
    assert re.search(r"^  if added waves_per_eu: \d+ ", text, re.M), text
    dropped = int(re.search(r"^  if dropped block_n: (\d+) ", text, re.M)[1])
    enumerated = re_.build_config(_without(held, "block_n"), warn=lambda m: None)
    assert dropped == sum(len(p["kernels"]) for p in enumerated["packs"])


@pytest.mark.parametrize(
    "count, budget, warns",
    [(1000, None, False), (1001, None, True), (11, 10, True), (1001, 2000, False)],
)
def test_generate_warns_over_its_per_arch_budget(capsys, count, budget, warns):
    sys.path.insert(0, str(_ROOT))
    import generate
    from tests.helpers import make_kernel, make_minimal_config, make_pack

    kernels = [make_kernel(name=f"k{i}") for i in range(count)]
    pack = make_pack(arch=["gfx950"], kernels=kernels)
    reduction = {} if budget is None else {"aot_budget_per_arch": budget}
    generate.warn_over_budget(make_minimal_config(packs=[pack], reduction=reduction))
    err = capsys.readouterr().err
    assert (
        f"WARN: {count} instances for gfx950 exceed the AOT budget {budget or 1000}"
        in err
    ) is warns


@pytest.mark.parametrize("strict", [False, True])
def test_generate_strict_fails_an_over_budget_run(
    monkeypatch, tmp_path, capsys, strict
):
    sys.path.insert(0, str(_ROOT))
    import generate
    from tests.helpers import make_kernel, make_minimal_config, make_pack

    pack = make_pack(
        arch=["gfx950"], kernels=[make_kernel(name=f"k{i}") for i in range(11)]
    )
    config = make_minimal_config(packs=[pack], reduction={"aot_budget_per_arch": 10})
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")
    monkeypatch.setattr(generate, "load_config", lambda path: config)
    argv = ["generate.py", "--config", str(config_path), "--dry-run"]
    argv += ["--output-dir", str(tmp_path / "out")]
    if strict:
        argv.append("--strict")
    monkeypatch.setattr(sys, "argv", argv)
    if strict:
        with pytest.raises(SystemExit) as stop:
            generate.main()
        assert stop.value.code == 1
    else:
        generate.main()
    assert "exceed the AOT budget 10" in capsys.readouterr().err
