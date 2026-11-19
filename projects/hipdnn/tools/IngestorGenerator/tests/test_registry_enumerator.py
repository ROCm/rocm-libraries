# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""registry_enumerator: the manifest + rocKE registry -> generator config. The
slice's agreement with the shipped catalog is gated in test_registry_regeneration.

The contradiction, coercion, vocabulary and cross-family behaviours run over the
synthetic registry in tests/synthetic_registry.py, so they do not move with rocKE's
candidates, tiles or messages. The shipped manifest is read against the live
registry only where the test asserts a fact the tool owns."""

from __future__ import annotations

import copy
import dataclasses
import enum
import io
import json
import os
import re
import subprocess
import sys
import types
import typing
from collections import OrderedDict
from pathlib import Path

import pytest
import yaml

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))

import registry_enumerator as re_  # noqa: E402
from tests import synthetic_registry as synth  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "configs" / "gfx950_attention_dense.engine.yaml"
_ROCKE_LIBRARY = (
    re_._REPO / "dnn-providers" / "hip-kernel-provider" / "rocke" / "library"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return re_.load_manifest(str(_MANIFEST))


@pytest.fixture(scope="module")
def registry(manifest):
    if not _ROCKE_LIBRARY.is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE_LIBRARY}")
    re_.bind_provider(manifest.get("provider_root"))
    return re_.Registry(manifest)


def _one_head_config(manifest: dict) -> dict:
    out = copy.deepcopy(manifest)
    group = out["enumerate"]["cells"][0]
    group["values"] = group["values"][:1]
    return out


def test_manifest_missing_a_section_is_named(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"engine": {}}))
    with pytest.raises(re_.EnumerationError, match="missing 'kmd_fields'"):
        re_.load_manifest(str(path))


def test_provider_root_is_required_and_repository_relative(
    manifest, tmp_path, monkeypatch
):
    monkeypatch.delenv("ROCKE_PROVIDER_ROOT", raising=False)
    with pytest.raises(re_.EnumerationError, match="provider_root"):
        re_.bind_provider(None)
    assert not os.path.isabs(manifest["provider_root"])
    if not _ROCKE_LIBRARY.is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE_LIBRARY}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    re_.bind_provider(manifest["provider_root"])


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """The synthetic probe manifest and the means to write and run it."""
    root = synth.provider(tmp_path, monkeypatch)
    return types.SimpleNamespace(
        manifest=synth.probe_manifest(root), tmp=tmp_path, out=tmp_path / "c.yaml"
    )


def _run_probe(probe, manifest=None, *extra) -> int:
    path = synth.write_manifest(probe.tmp / "m.engine.yaml", manifest or probe.manifest)
    return re_.main(["--manifest", str(path), "--out", str(probe.out), *extra])


def test_the_synthetic_manifest_enumerates(probe):
    """The baseline every refusal below departs from."""
    assert _run_probe(probe) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert len({k["name"] for k in kernels}) == len(kernels) > 0


def test_a_pin_list_adds_each_later_candidates_specs_to_the_cells_it_admits(probe):
    """The first pin gives each cell its default spec; a later one adds its sweep specs
    where it admits the cell, each spec once."""
    single = re_.Registry(probe.manifest)
    probe.manifest["enumerate"]["pin"] = [
        {"algorithm": "probe", "spec_id": "grid"},
        {"algorithm": "persist", "spec_id": "a"},
    ]
    listed = re_.Registry(probe.manifest)
    assert listed.candidate.name == "probe_grid"
    assert [c.name for _, c in listed.extra_pins] == ["probe_persist_a"]
    cell = listed.cells()[0]
    # The probe candidates sweep the same specs: the union holds each once.
    assert listed.space(cell) == single.space(cell)

    extra = listed.extra_pins[0][1]
    base = listed.candidate.select_spec(listed.request(cell))
    added = dataclasses.replace(base, tile_m=256)
    extra.sweep_space = lambda request: [added]
    _, specs = listed.space(cell)
    assert specs == single.space(cell)[1] + [added]

    extra.admits = lambda request: (False, "declined")
    assert listed.space(cell) == single.space(cell)


@dataclasses.dataclass(frozen=True)
class _Tuned:
    """A registry spec wrapped with its tuning id, as rocKE's dispatcher hands back."""

    kernel_spec: object
    tuning_id: str


def test_a_pin_lists_kernels_trace_to_the_pin_that_supplied_them(probe):
    """A spec only a later pin sweeps carries that pin's spec_id and tuning id; a
    spec both pins sweep stays the first pin's."""
    probe.manifest["enumerate"]["pin"] = [
        {"algorithm": "probe", "spec_id": "grid"},
        {"algorithm": "persist", "spec_id": "a"},
    ]
    registry = re_.Registry(probe.manifest)
    first, later = registry.candidate, registry.extra_pins[0][1]
    sweep = synth.ProbeCandidate.sweep_space
    added = dataclasses.replace(
        first.select_spec(registry.request(registry.cells()[0])), tile_n=256
    )
    # Both pins sweep the probe specs; only the later one adds tile_n 256.
    first.sweep_space = lambda r: [
        _Tuned(s, f"grid@{s.tile_n}") for s in sweep(first, r)
    ]
    later.sweep_space = lambda r: [
        _Tuned(s, f"a@{s.tile_n}") for s in [*sweep(later, r), added]
    ]

    kernels = re_.build_config(probe.manifest, registry)["packs"][0]["kernels"]
    provenance = {k["name"]: k["provenance"] for k in kernels}
    assert provenance["probe.bf16_d64_h8_c0_tn32"] == {
        "rocke_spec_id": "grid",
        "rocke_tuning_id": "grid@32",
    }
    assert provenance["probe.bf16_d64_h8_c0_tn256"] == {
        "rocke_spec_id": "a",
        "rocke_tuning_id": "a@256",
    }


def test_base_spec_is_the_named_pins_select_spec(probe):
    """The first pin's by default; a later pin's own, unwrapped; a pin the manifest
    does not list is refused."""
    probe.manifest["enumerate"]["pin"] = [
        {"algorithm": "probe", "spec_id": "grid"},
        {"algorithm": "persist", "spec_id": "a"},
    ]
    registry = re_.Registry(probe.manifest)
    cell = registry.cells()[0]
    later_pin, later = registry.extra_pins[0]
    first = registry.space(cell)[0]
    own = dataclasses.replace(first, tile_m=256)
    later.select_spec = lambda request: _Tuned(own, "a@own")
    later.sweep_space = lambda request: [own]

    assert registry.base_spec(cell) == registry.base_spec(cell, registry.pin) == first
    assert registry.base_spec(cell, later_pin) == own
    with pytest.raises(re_.EnumerationError, match="not one of"):
        registry.base_spec(cell, {"algorithm": "persist", "spec_id": "b"})


@pytest.mark.parametrize(
    "key, value",
    [
        ("spec_overrides", {"tile_m": 128}),
        ("neutral_fields", ["tile_m"]),
        ("selector_fields", ["heads"]),
        ("cell_keyed_by", ["tile_m"]),
    ],
)
def test_base_spec_refuses_a_section_that_reshapes_its_cells(probe, key, value):
    """cell_spaces rewrites or merges such a section's specs, so a raw request combo's
    select_spec is not its cell's default kernel: base_spec names the key instead."""
    probe.manifest["enumerate"][key] = value
    registry = re_.Registry(probe.manifest)
    with pytest.raises(re_.EnumerationError, match=rf"enumerate sets \['{key}'\]"):
        registry.base_spec(registry.cells()[0])


def test_each_pin_of_a_list_must_match_one_candidate(probe, capsys):
    probe.manifest["enumerate"]["pin"] = [
        {"algorithm": "probe", "spec_id": "grid"},
        {"algorithm": "persist"},
    ]
    assert _run_probe(probe) == 2
    assert "matches 2 registry candidates" in capsys.readouterr().err
    probe.manifest["enumerate"]["pin"] = []
    assert _run_probe(probe) == 2
    assert "non-empty list of mappings" in capsys.readouterr().err


_SEGMENT_LAUNCHES = {"segment": {}, "reduce": {"role": "reduce"}}


def test_manifest_launches_reach_every_generated_kernel_source(probe):
    """A split-KV engine (segment + reduce): every generated kernel is a
    multi-launch descriptor, as the generator's loader reads it."""
    from codegen.config_loader import load_config

    probe.manifest["launches"] = copy.deepcopy(_SEGMENT_LAUNCHES)
    assert _run_probe(probe) == 0
    text = probe.out.read_text()
    # One shared object would dump as a YAML anchor and aliases, not per-kernel copies.
    assert "&id" not in text and "*id" not in text
    raw = yaml.safe_load(text)["packs"][0]["kernels"]
    assert all(k["kernel_source"]["launches"] == _SEGMENT_LAUNCHES for k in raw)
    kernels = [k for p in load_config(probe.out).packs for k in p.kernels]
    assert len(kernels) == len(raw) > 0
    for kernel in kernels:
        launches = kernel.kernel_source.launches
        assert list(launches) == ["segment", "reduce"]
        assert launches["reduce"].role == "reduce"
        assert launches["segment"].builder == kernel.kernel_source.builder


def test_each_generated_kernel_owns_its_launches(probe):
    probe.manifest["launches"] = copy.deepcopy(_SEGMENT_LAUNCHES)
    config = re_.build_config(probe.manifest, warn=lambda message: None)
    sources = [k["kernel_source"]["launches"] for k in config["packs"][0]["kernels"]]
    owners = [probe.manifest["launches"], *sources]
    assert len({id(s) for s in owners}) == len(owners)
    assert len({id(s["reduce"]) for s in owners}) == len(owners)


def _run_both_ways(tmp: Path, manifest: dict, capsys, *argv) -> list:
    """Run the CLI on ``manifest`` without, then with, `launches`; return each
    run's exit code, stdout and generated config (parsed, or None)."""
    runs = []
    for launches in (None, _SEGMENT_LAUNCHES):
        m = copy.deepcopy(manifest)
        if launches:
            m["launches"] = copy.deepcopy(launches)
        path = synth.write_manifest(tmp / "m.engine.yaml", m)
        out = tmp / "c.yaml"
        out.unlink(missing_ok=True)
        capsys.readouterr()
        code = re_.main(["--manifest", str(path), *argv])
        text = capsys.readouterr().out.replace(str(out), "<out>")
        runs.append(
            (code, text, yaml.safe_load(out.read_text()) if out.exists() else None)
        )
    return runs


def _without_launches(config: dict) -> dict:
    for kernel in config["packs"][0]["kernels"]:
        assert kernel["kernel_source"].pop("launches") == _SEGMENT_LAUNCHES
    return config


def test_manifest_launches_change_nothing_but_the_kernel_sources(probe, capsys):
    """`--out` and `--shapes`: the same kernels, the same report."""
    (shapes := probe.tmp / "shapes.json").write_text(
        '[{"hdim": 64, "heads": 8, "dtype": "bf16", "mask": 1}]'
    )
    out = str(probe.tmp / "c.yaml")
    for argv in (["--out", out], ["--out", out, "--shapes", str(shapes)]):
        plain, launched = _run_both_ways(probe.tmp, probe.manifest, capsys, *argv)
        assert plain[0] == launched[0] == 0, argv
        assert plain[1] == launched[1], argv
        assert _without_launches(launched[2]) == plain[2], argv
    assert "covered: 1 shapes" in launched[1]


def test_manifest_launches_leave_the_plan_space_unchanged(
    tmp_path, monkeypatch, capsys
):
    manifest = synth.conv_probe_manifest(synth.provider(tmp_path, monkeypatch))
    plain, launched = _run_both_ways(tmp_path, manifest, capsys, "--plan-space")
    assert plain[0] == launched[0] == 0
    assert plain[1] == launched[1] != ""


def test_a_manifest_without_launches_emits_no_launches(probe):
    """Regression guard: the single-launch output is unchanged."""
    assert _run_probe(probe) == 0
    raw = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert all(
        set(k["kernel_source"]) == {"kind", "source", "builder", "spec"} for k in raw
    )


def _declined(m):
    m["enumerate"]["cells"][0]["values"] = [[96, 8]]


def _pin(pin):
    return lambda m: m["enumerate"].update(pin=pin)


def _bad_enum(m):
    m["enumerate"]["fixed"]["mask"] = "NO_SUCH_MASK"


def _enum_cells_by_value(m):
    group = next(g for g in m["enumerate"]["cells"] if g["fields"] == ["mask"])
    group["values"] = [1, 0]


def _max_kernels(m):
    m["enumerate"]["max_kernels"] = 1


def _bad_template(m):
    m["kernel_name_template"] = "{no_such_field}"


def _free(*fields):
    return lambda m: m["enumerate"].update(free_fields=list(fields))


@pytest.mark.parametrize(
    "mutate, offender",
    [
        (_declined, "hdim"),
        (_pin({"algorithm": "no_such_candidate"}), "no_such_candidate"),
        # Two registered candidates share the persistent algorithm.
        (_pin({"algorithm": "persist"}), "persist"),
        (_bad_enum, "NO_SUCH_MASK"),
        (_enum_cells_by_value, "mask"),
        (_max_kernels, "max_kernels"),
        (_bad_template, "no_such_field"),
        (_free("tile_n", "tile_x"), "tile_x"),
    ],
    ids=[
        "declined-cell",
        "pin-matches-none",
        "pin-matches-two",
        "unknown-enum-name",
        "enum-cell-by-value",
        "over-budget",
        "unrenderable-template",
        "unknown-free-field",
    ],
)
def test_a_manifest_the_registry_contradicts_is_exit_2(probe, capsys, mutate, offender):
    mutate(probe.manifest)
    rc = _run_probe(probe)
    err = capsys.readouterr().err
    assert rc == 2
    assert offender in err, err
    assert not probe.out.exists()


def _raises(exc):
    def fail(*args, **kwargs):
        raise exc

    return fail


@pytest.mark.parametrize(
    "attr, replacement, message",
    [
        ("admits", _raises(TypeError("bad")), r"cannot judge cell .*\(TypeError\)"),
        (
            "select_spec",
            _raises(AttributeError("nope")),
            r"rejects cell .*\(AttributeError\)",
        ),
        (
            "select_spec",
            lambda r: types.SimpleNamespace(kernel_spec=None),
            "without a kernel_spec",
        ),
    ],
    ids=["admits-raises", "select_spec-raises", "route-without-kernel"],
)
def test_a_registry_that_breaks_judging_a_cell_is_an_enumeration_error(
    manifest, registry, attr, replacement, message
):
    """Not a traceback, and not a decline: the registry itself is unusable."""
    reg = re_.Registry(_one_head_config(manifest))
    real = reg.candidate

    class Broken:
        def __getattr__(self, name):
            return replacement if name == attr else getattr(real, name)

    reg.candidate = Broken()
    with pytest.raises(re_.EnumerationError, match=message):
        reg.space(reg.cells()[0])


def test_a_candidate_list_that_cannot_be_built_is_an_enumeration_error(
    manifest, registry, monkeypatch
):
    real = re_._load_callable
    monkeypatch.setattr(
        re_,
        "_load_callable",
        lambda ref, what: (
            _raises(RuntimeError("boom")) if what == "candidates" else real(ref, what)
        ),
    )
    with pytest.raises(re_.EnumerationError, match=r"candidate list.*RuntimeError"):
        re_.Registry(_one_head_config(manifest))


@pytest.mark.parametrize("text", [None, "a: [\n"])
def test_an_unreadable_manifest_is_exit_2(tmp_path, capsys, text):
    path = tmp_path / "m.engine.yaml"
    if text is not None:
        path.write_text(text)
    assert re_.main(["--manifest", str(path), "--out", str(tmp_path / "c")]) == 2
    assert "FAIL: manifest" in capsys.readouterr().err


def _no_metadata_fields(m):
    del m["specialization"]["metadata_fields"]


def _enumerate_not_a_mapping(m):
    m["enumerate"] = 5


def _max_kernels_not_an_int(m):
    m["enumerate"]["max_kernels"] = "a"


def _kmd_fields_not_a_list(m):
    m["kmd_fields"] = 5


def _null_bindings_in_the_full_space(m):
    m["enumerate"]["space"] = "full"
    m["specialization"]["bindings"] = None


def _null_vocabulary(m):
    m["specialization"]["vocabulary"] = None


@pytest.mark.parametrize("mode", ["--plan-space", "--out"])
@pytest.mark.parametrize(
    "mutate, field",
    [
        (_no_metadata_fields, "metadata_fields"),
        (_enumerate_not_a_mapping, "enumerate"),
        (_max_kernels_not_an_int, "max_kernels"),
        (_kmd_fields_not_a_list, "kmd_fields"),
        (_null_bindings_in_the_full_space, "bindings"),
        (_null_vocabulary, "vocabulary"),
    ],
    ids=[
        "no-metadata-fields",
        "enumerate-int",
        "max-kernels-str",
        "kmd-fields-int",
        "bindings-null",
        "vocabulary-null",
    ],
)
def test_a_wrongly_typed_manifest_field_is_exit_2_naming_the_field(
    manifest, registry, tmp_path, capsys, mode, mutate, field
):
    broken = copy.deepcopy(manifest)
    mutate(broken)
    path = tmp_path / "m.engine.yaml"
    path.write_text(yaml.safe_dump(broken, sort_keys=False))
    out = tmp_path / "c.yaml"
    argv = ["--manifest", str(path)]
    argv += ["--plan-space"] if mode == "--plan-space" else ["--out", str(out)]
    assert re_.main(argv) == 2
    assert field in capsys.readouterr().err
    assert not out.exists()


def test_a_manifest_key_declared_twice_is_exit_2_naming_the_key(tmp_path, capsys):
    path = tmp_path / "m.engine.yaml"
    path.write_text(_MANIFEST.read_text() + "\npack: other\n")
    assert re_.main(["--manifest", str(path), "--plan-space"]) == 2
    assert "pack" in capsys.readouterr().err


@pytest.fixture
def feature_probe(tmp_path, monkeypatch):
    root = synth.provider(tmp_path, monkeypatch)
    return types.SimpleNamespace(
        manifest=synth.feature_probe_manifest(root),
        plain=synth.probe_manifest(root),
        tmp=tmp_path,
        out=tmp_path / "c.yaml",
    )


def _probe_kernels(probe):
    return yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]


def test_a_feature_field_names_its_kernels_and_narrows_their_slice(feature_probe):
    """Kernels holding the feature off are the plain probe catalog (every free tile,
    no token, the field at its default); kernels turning it on carry the token, the
    field in spec and metadata, and only the default tile its free_fields leave."""
    assert _run_probe(feature_probe) == 0
    kernels = yaml.safe_load(feature_probe.out.read_text())["packs"][0]["kernels"]
    on = [k for k in kernels if k["kernel_source"]["spec"]["bias"]]
    off = [k for k in kernels if not k["kernel_source"]["spec"]["bias"]]
    cells = 4  # two head sizes x two masks
    assert len(off) == 3 * cells and len(on) == cells
    assert all(k["name"].endswith("_bias") and k["metadata"]["bias"] == 1 for k in on)
    assert all("_bias" not in k["name"] and k["metadata"]["bias"] == 0 for k in off)
    assert {k["kernel_source"]["spec"]["tile_n"] for k in on} == {64}
    assert all(k.get("default") for k in on)
    assert len({k["name"] for k in kernels}) == len(kernels)


def test_a_feature_group_leaves_the_plain_kernels_as_they_were(feature_probe):
    """Adding a feature cell group keeps every existing kernel's name, order, spec and
    metadata; the feature-off kernels only gain the field at its default."""
    assert _run_probe(feature_probe, feature_probe.plain) == 0
    plain = _probe_kernels(feature_probe)
    assert _run_probe(feature_probe) == 0
    off = [
        k
        for k in _probe_kernels(feature_probe)
        if not k["kernel_source"]["spec"]["bias"]
    ]
    assert [k["name"] for k in off] == [k["name"] for k in plain]
    for got, want in zip(off, plain):
        spec = dict(got["kernel_source"]["spec"])
        assert spec.pop("bias") is False
        assert spec == want["kernel_source"]["spec"], got["name"]
        meta = dict(got["metadata"])
        assert meta.pop("bias") == 0
        assert meta == want["metadata"], got["name"]


@pytest.mark.parametrize(
    "entry",
    [{"name_token": "_bias"}, {"name_token": "_bias", "free_fields": ["tile_n"]}],
    ids=["no-free-fields", "same-free-fields"],
)
def test_a_feature_field_without_narrowing_sweeps_every_free_tile(feature_probe, entry):
    feature_probe.manifest["enumerate"]["feature_fields"]["bias"] = entry
    assert _run_probe(feature_probe) == 0
    kernels = _probe_kernels(feature_probe)
    on = [k for k in kernels if k["kernel_source"]["spec"]["bias"]]
    off = [k for k in kernels if not k["kernel_source"]["spec"]["bias"]]
    cells = 4  # two head sizes x two masks
    assert len(on) == len(off) == 3 * cells
    assert {k["kernel_source"]["spec"]["tile_n"] for k in on} == {
        k["kernel_source"]["spec"]["tile_n"] for k in off
    }
    assert sum(bool(k.get("default")) for k in on) == cells


@pytest.mark.parametrize(
    "entry",
    [
        {"name_token": "_bias", "select": {"tile_n": 32}},
        {"name_token": "_bias", "free_fields": ["tile_n"], "select": {"tile_n": 32}},
    ],
    ids=["swept-free-fields", "narrowed-free-fields"],
)
def test_a_feature_select_moves_its_cells_off_the_default_tile(feature_probe, entry):
    """``select`` keeps only the specs with its values; the one left per cell is
    the cell's default, though it is not the candidate's default spec."""
    feature_probe.manifest["enumerate"]["feature_fields"]["bias"] = entry
    assert _run_probe(feature_probe) == 0
    kernels = _probe_kernels(feature_probe)
    on = [k for k in kernels if k["kernel_source"]["spec"]["bias"]]
    cells = 4  # two head sizes x two masks
    assert len(on) == cells
    assert {k["kernel_source"]["spec"]["tile_n"] for k in on} == {32}
    assert all(k.get("default") for k in on)
    assert all(k["name"].endswith("_bias") for k in on)
    # select narrows only the cells that turn the feature on.
    off = [k for k in kernels if not k["kernel_source"]["spec"]["bias"]]
    assert len(off) == 3 * cells
    assert {k["kernel_source"]["spec"]["tile_n"] for k in off} == {32, 64, 128}
    assert sum(bool(k.get("default")) for k in off) == cells


def test_a_feature_select_of_the_default_tile_keeps_one_default(feature_probe):
    feature_probe.manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "select": {"tile_n": 64},
    }
    assert _run_probe(feature_probe) == 0
    on = [
        k for k in _probe_kernels(feature_probe) if k["kernel_source"]["spec"]["bias"]
    ]
    assert len(on) == 4  # two head sizes x two masks
    assert {k["kernel_source"]["spec"]["tile_n"] for k in on} == {64}
    assert all(k.get("default") for k in on)


def test_a_partial_select_keeps_the_registry_default_elsewhere(feature_probe):
    """A select pinning one of two swept fields: the cell's default is the kept
    spec that holds the registry default (order unset) in the other."""
    feature_probe.manifest["enumerate"]["free_fields"] = ["tile_n", "order"]
    # order joins the name so its kernels stay distinct.
    feature_probe.manifest["kernel_name_template"] += "_o{order}"
    feature_probe.manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "select": {"tile_n": 32},
    }
    assert _run_probe(feature_probe) == 0
    on = [
        k for k in _probe_kernels(feature_probe) if k["kernel_source"]["spec"]["bias"]
    ]
    defaults = [k for k in on if k.get("default")]
    assert {k["kernel_source"]["spec"]["tile_n"] for k in on} == {32}
    assert len(defaults) == 4  # one per cell
    assert all(k["kernel_source"]["spec"].get("order") is None for k in defaults)


def test_a_select_that_matches_no_spec_is_exit_2(feature_probe, capsys):
    feature_probe.manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "select": {"tile_n": 48},
    }
    assert _run_probe(feature_probe) == 2
    assert "no spec for cell" in capsys.readouterr().err
    assert not feature_probe.out.exists()


def test_a_slice_whose_names_collide_is_exit_2(feature_probe, capsys):
    """A cell group that varies a field no name token, template slot or projected
    knob reaches renders one name twice; ``space: slice`` refuses it too."""
    del feature_probe.manifest["enumerate"]["feature_fields"]
    assert _run_probe(feature_probe) == 2
    err = capsys.readouterr().err
    assert "space: slice renders" in err and "duplicated kernel name" in err, err
    assert not feature_probe.out.exists()


def _feature(entry):
    return lambda m: m["enumerate"]["feature_fields"].update(bias=entry)


def _feature_not_metadata(m):
    m["specialization"]["metadata_fields"].remove("bias")
    del m["specialization"]["bindings"]["bias"]


def _feature_not_in_spec_fields(m):
    m["enumerate"]["spec_fields"] = ["dtype", "tile_m", "tile_n"]


def _feature_not_a_reduction_feature(m):
    m["reduction"] = {"fields": {"feature": ["dtype"], "performance": []}}


def _feature_with_an_empty_reduction_feature(m):
    m["reduction"] = {"fields": {"feature": [], "performance": []}}


def _feature_with_no_reduction_feature(m):
    m["reduction"] = {"fields": {"performance": []}}


def _feature_with_no_reduction_fields(m):
    m["reduction"] = {"aot_budget_per_arch": 1000}


def _free_field_only_a_performance_knob(m):
    """A slice reads a feature's free_fields against enumerate.free_fields, not against
    the reduction block's performance knobs."""
    m["reduction"] = {
        "fields": {"feature": ["bias"], "performance": [{"name": "tile_m"}]}
    }
    m["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_b",
        "free_fields": ["tile_m"],
    }


@pytest.mark.parametrize(
    "mutate, offender",
    [
        (_feature({"free_fields": []}), "name_token"),
        (_feature({"name_token": ""}), "name_token"),
        (_feature({"name_token": "_b", "stride": 2}), "feature_fields.bias"),
        (_feature({"name_token": "_b", "free_fields": "tile_n"}), "free_fields"),
        (_feature({"name_token": "_b", "free_fields": ["tile_x"]}), "subset"),
        (_feature_not_metadata, "metadata_fields"),
        (_feature_not_in_spec_fields, "spec_fields"),
        (_feature_not_a_reduction_feature, "reduction.fields.feature"),
        (_feature_with_an_empty_reduction_feature, "reduction.fields.feature"),
        (_feature_with_no_reduction_feature, "reduction.fields.feature"),
        (_feature_with_no_reduction_fields, "reduction.fields.feature"),
        (_feature({"name_token": "_b", "free_fields": ["tile_m"]}), "subset"),
        (_free_field_only_a_performance_knob, "subset"),
        (
            lambda m: m["enumerate"].update(free_fields=["tile_n", "bias"]),
            "feature_fields.bias",
        ),
        (
            lambda m: m["enumerate"]["feature_fields"].update(
                no_such={"name_token": "_x"}
            ),
            "no_such",
        ),
        (_feature({"name_token": "_b", "select": {"tile_m": 128}}), "select"),
        (_feature({"name_token": "_b", "select": {}}), "select"),
        (_feature({"name_token": "_b", "select": [32]}), "select"),
        (
            _feature({"name_token": "_b", "free_fields": [], "select": {"tile_n": 32}}),
            "select",
        ),
        (_feature({"name_token": "_b", "select": {"tile_n": [32]}}), "select"),
        (
            lambda m: (
                m["enumerate"].update(space="full"),
                m["enumerate"]["feature_fields"].update(
                    bias={"name_token": "_b", "select": {"tile_n": 32}}
                ),
            ),
            "select",
        ),
    ],
    ids=[
        "no-token",
        "empty-token",
        "unknown-key",
        "free-fields-not-a-list",
        "unknown-free-field",
        "not-a-metadata-field",
        "not-a-spec-field",
        "not-a-reduction-feature",
        "empty-reduction-feature",
        "no-reduction-feature",
        "no-reduction-fields",
        "free-fields-not-a-subset",
        "free-field-a-performance-knob-but-not-a-slice-free-field",
        "feature-is-a-free-field",
        "not-on-the-spec",
        "select-not-a-swept-field",
        "select-empty",
        "select-not-a-mapping",
        "select-outside-narrowed-free-fields",
        "select-value-not-a-scalar",
        "select-under-full-space-without-performance-knobs",
    ],
)
def test_a_malformed_feature_field_is_exit_2(feature_probe, capsys, mutate, offender):
    if offender == "no_such":
        # Must also be a metadata field to pass the manifest check and reach the
        # registry, which is what refuses it.
        feature_probe.manifest["specialization"]["metadata_fields"].append("no_such")
    mutate(feature_probe.manifest)
    rc = _run_probe(feature_probe)
    err = capsys.readouterr().err
    assert rc == 2
    assert offender in err, err
    assert not feature_probe.out.exists()


def test_a_feature_in_the_reduction_features_or_with_no_reduction_is_accepted(
    feature_probe,
):
    """Only a manifest with no ``reduction`` block is exempt from naming the feature
    there; with a block, listing it is what passes."""
    assert _run_probe(feature_probe) == 0
    feature_probe.manifest["reduction"] = {
        "fields": {"feature": ["dtype", "bias"], "performance": []}
    }
    assert _run_probe(feature_probe) == 0


def _feature_also(feature_probe, **overrides) -> dict:
    """The feature probe plus a section over ``persist``/``a`` on a cell the primary
    does not list; both sections cross the same cell fields, ``bias`` included."""
    section = {
        "cells": [
            {"fields": ["hdim", "heads"], "values": [[128, 16]]},
            {"fields": ["mask"], "values": ["CAUSAL"]},
            {"fields": ["bias"], "values": [False, True]},
        ],
        "free_fields": ["tile_n"],
        "spec_fields": ["dtype", "tile_m", "tile_n", "causal", "order", "bias"],
        **overrides,
    }
    manifest = copy.deepcopy(feature_probe.manifest)
    manifest["enumerate"]["also"] = [_also(**section)]
    return manifest


def _also_kernels(feature_probe, manifest) -> tuple:
    """``(feature on, feature off)`` kernels of the ``also`` section (spec ``a``)."""
    assert _run_probe(feature_probe, manifest) == 0
    extra = [
        k
        for k in _probe_kernels(feature_probe)
        if k["provenance"]["rocke_spec_id"] == "a"
    ]
    return (
        [k for k in extra if k["kernel_source"]["spec"]["bias"]],
        [k for k in extra if not k["kernel_source"]["spec"]["bias"]],
    )


@pytest.mark.parametrize(
    "entry, tiles",
    [
        ({"name_token": "_bias", "select": {"tile_n": 32}}, {32}),
        ({"name_token": "_bias", "free_fields": []}, {64}),
    ],
    ids=["select", "free-fields"],
)
def test_an_also_section_narrows_its_feature_kernels_like_the_primary(
    feature_probe, entry, tiles
):
    """The feature is declared once, on the primary, and every section's kernels that
    turn it on take its token and its narrowing; the also section's feature-off kernels
    keep every free tile."""
    manifest = _feature_also(feature_probe)
    manifest["enumerate"]["feature_fields"]["bias"] = entry
    on, off = _also_kernels(feature_probe, manifest)
    assert [k["kernel_source"]["spec"]["tile_n"] for k in on] == sorted(tiles)
    assert all(k["name"].endswith("_bias") and k.get("default") for k in on)
    assert {k["kernel_source"]["spec"]["tile_n"] for k in off} == {32, 64, 128}
    assert sum(bool(k.get("default")) for k in off) == 1


def test_an_also_section_cannot_declare_feature_fields_of_its_own(
    feature_probe, capsys
):
    manifest = _feature_also(
        feature_probe, feature_fields={"bias": {"name_token": "_bias"}}
    )
    assert _run_probe(feature_probe, manifest) == 2
    assert "enumerate.also[0].feature_fields" in capsys.readouterr().err


def test_a_feature_missing_from_an_also_sections_spec_fields_is_exit_2(
    feature_probe, capsys
):
    """Metadata would read ``bias`` from the request while the spec the kernel builds
    from omits it: the catalog would advertise a bias kernel that builds without."""
    manifest = _feature_also(
        feature_probe, spec_fields=["dtype", "tile_m", "tile_n", "causal", "order"]
    )
    assert _run_probe(feature_probe, manifest) == 2
    err = capsys.readouterr().err
    assert "enumerate.feature_fields.bias" in err, err
    assert "enumerate.also[0].spec_fields" in err, err
    assert not feature_probe.out.exists()


def test_an_also_section_that_holds_a_selected_field_is_exit_2(feature_probe, capsys):
    """A select value on a field the section does not sweep has nothing to choose
    among: exit 2, naming the field."""
    manifest = _feature_also(feature_probe, free_fields=[])
    manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "select": {"tile_n": 32},
    }
    assert _run_probe(feature_probe, manifest) == 2
    err = capsys.readouterr().err
    assert "tile_n" in err and "free_fields" in err, err


#: Cells of the full feature probe per value of ``bias``: two head sizes x two masks.
_CELLS_PER_BIAS_VALUE = 4
#: The kernels one cell of the full feature probe sweeps unnarrowed: 3 tiles x 2 orders.
_KERNELS_PER_CELL = 6


@pytest.fixture
def full_feature_probe(tmp_path, monkeypatch):
    root = synth.provider(tmp_path, monkeypatch)
    return types.SimpleNamespace(
        manifest=synth.full_feature_probe_manifest(root),
        tmp=tmp_path,
        out=tmp_path / "c.yaml",
    )


def _full_feature_kernels(probe, entry) -> tuple:
    """``(bias-on kernels, bias-off kernels)`` of the full feature probe with
    ``bias``'s feature entry ``entry`` (``name_token`` is filled in)."""
    probe.manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        **entry,
    }
    assert _run_probe(probe) == 0
    kernels = _probe_kernels(probe)
    assert len({k["name"] for k in kernels}) == len(kernels)
    on = [k for k in kernels if k["kernel_source"]["spec"]["bias"]]
    off = [k for k in kernels if not k["kernel_source"]["spec"]["bias"]]
    assert all(k["name"].endswith("_bias") for k in on)
    assert not any("_bias" in k["name"] for k in off)
    return on, off


def _knob_values(kernels, name) -> set:
    return {k["metadata"][name] for k in kernels}


def _defaults(kernels) -> list:
    return [k for k in kernels if k.get("default")]


def test_a_full_space_feature_without_narrowing_sweeps_every_knob(full_feature_probe):
    on, off = _full_feature_kernels(full_feature_probe, {})
    assert len(on) == len(off) == _KERNELS_PER_CELL * _CELLS_PER_BIAS_VALUE
    assert _knob_values(on, "tile_n") == _knob_values(off, "tile_n") == {32, 64, 128}
    assert _knob_values(on, "order") == _knob_values(off, "order") == {"row", "col"}
    assert len(_defaults(on)) == len(_defaults(off)) == _CELLS_PER_BIAS_VALUE


@pytest.mark.parametrize(
    "free, tiles, orders",
    [
        (["tile_n"], {32, 64, 128}, {"row"}),
        (["order"], {64}, {"row", "col"}),
        ([], {64}, {"row"}),
    ],
    ids=["tile-only", "order-only", "none"],
)
def test_a_full_space_feature_sweeps_only_its_free_fields(
    full_feature_probe, capsys, free, tiles, orders
):
    """A cell turning the feature on varies only the performance knobs its free_fields
    name; the rest stay at the cell default (tile 64, order unset, which resolves to
    "row"). Held by narrowing, they are not a WARN. The feature-off cells keep the
    full sweep."""
    on, off = _full_feature_kernels(full_feature_probe, {"free_fields": free})
    assert len(on) == len(tiles) * len(orders) * _CELLS_PER_BIAS_VALUE
    assert _knob_values(on, "tile_n") == tiles and _knob_values(on, "order") == orders
    assert len(off) == _KERNELS_PER_CELL * _CELLS_PER_BIAS_VALUE
    assert _knob_values(off, "tile_n") == {32, 64, 128}
    assert _knob_values(off, "order") == {"row", "col"}
    defaults = _defaults(on)
    assert len(defaults) == _CELLS_PER_BIAS_VALUE
    assert all(k["metadata"]["tile_n"] == 64 for k in defaults)
    assert all("order" not in k["kernel_source"]["spec"] for k in defaults)
    assert "WARN" not in capsys.readouterr().err


def test_a_full_space_select_keeps_its_specs_and_moves_the_cell_default(
    full_feature_probe,
):
    """``select`` keeps only the specs with its values, the knobs it does not name
    still sweep, and the kept spec equal to the registry default elsewhere (order
    unset) is the cell's default though it is not the candidate's default spec."""
    on, off = _full_feature_kernels(full_feature_probe, {"select": {"tile_n": 32}})
    assert _knob_values(on, "tile_n") == {32}
    assert len(on) == 2 * _CELLS_PER_BIAS_VALUE  # 2 resolved orders
    defaults = _defaults(on)
    assert len(defaults) == _CELLS_PER_BIAS_VALUE
    assert all("order" not in k["kernel_source"]["spec"] for k in defaults)
    # select narrows only the cells that turn the feature on.
    assert len(off) == _KERNELS_PER_CELL * _CELLS_PER_BIAS_VALUE
    assert _knob_values(off, "tile_n") == {32, 64, 128}
    assert [k["metadata"]["tile_n"] for k in _defaults(off)] == [
        64
    ] * _CELLS_PER_BIAS_VALUE


def test_a_full_space_select_with_free_fields_keeps_one_kernel_per_cell(
    full_feature_probe,
):
    on, _ = _full_feature_kernels(
        full_feature_probe, {"free_fields": ["tile_n"], "select": {"tile_n": 128}}
    )
    assert (
        len(on) == _CELLS_PER_BIAS_VALUE and len(_defaults(on)) == _CELLS_PER_BIAS_VALUE
    )
    assert _knob_values(on, "tile_n") == {128}
    assert all("order" not in k["kernel_source"]["spec"] for k in on)


def test_a_full_space_select_of_an_explicit_value_keeps_the_default_where_it_is(
    full_feature_probe,
):
    """A select of a value the unset default does not carry (``order=col``) keeps the
    default tile and moves only that knob."""
    on, _ = _full_feature_kernels(full_feature_probe, {"select": {"order": "col"}})
    assert _knob_values(on, "order") == {"col"} and _knob_values(on, "tile_n") == {
        32,
        64,
        128,
    }
    assert [k["metadata"]["tile_n"] for k in _defaults(on)] == [
        64
    ] * _CELLS_PER_BIAS_VALUE


def test_a_full_space_select_that_matches_no_spec_is_exit_2(full_feature_probe, capsys):
    full_feature_probe.manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "select": {"tile_n": 48},
    }
    assert _run_probe(full_feature_probe) == 2
    assert "no spec for cell" in capsys.readouterr().err
    assert not full_feature_probe.out.exists()
    # --plan-space never reaches the build's check, so the narrowing itself refuses.
    with pytest.raises(re_.EnumerationError, match="no spec for cell"):
        re_.plan_space(full_feature_probe.manifest, out=io.StringIO())


def test_a_knob_held_in_every_cell_by_narrowing_is_not_projected(full_feature_probe):
    """Narrowing runs before projection: with only bias cells and ``order`` held in
    all of them, no cell varies ``order``, so it is neither metadata nor a KMD field,
    where unnarrowed it is both."""
    cells = full_feature_probe.manifest["enumerate"]["cells"]
    cells[-1]["values"] = [True]

    def config(entry):
        full_feature_probe.manifest["enumerate"]["feature_fields"]["bias"] = entry
        assert _run_probe(full_feature_probe) == 0
        return yaml.safe_load(full_feature_probe.out.read_text())

    wide = config({"name_token": "_bias"})
    assert "order" in wide["specialization"]["metadata_fields"]
    assert "order" in {f["name"] for f in wide["kmd_fields"]}
    narrow = config({"name_token": "_bias", "free_fields": ["tile_n"]})
    assert "order" not in narrow["specialization"]["metadata_fields"]
    assert "order" not in {f["name"] for f in narrow["kmd_fields"]}
    kernels = narrow["packs"][0]["kernels"]
    assert len(kernels) == 3 * _CELLS_PER_BIAS_VALUE
    assert all("order" not in k["metadata"] for k in kernels)
    assert not any("order" in k["name"] for k in kernels)
    # --plan-space sees the narrowed cells too: no cell varies it.
    shown = io.StringIO()
    assert re_.plan_space(full_feature_probe.manifest, out=shown) == 0
    assert re.search(r"^    tile_n: ", shown.getvalue(), re.M), shown.getvalue()
    assert not re.search(r"^    order: ", shown.getvalue(), re.M), shown.getvalue()


def test_the_budget_is_counted_after_a_full_space_feature_narrows(
    full_feature_probe, capsys
):
    """48 kernels unnarrowed, 36 with the feature's cells sweeping only the tile: over
    and within a budget of 40, for the build and for --plan-space."""
    manifest = full_feature_probe.manifest
    manifest["reduction"]["aot_budget_per_arch"] = 40
    features = manifest["enumerate"]["feature_fields"]
    features["bias"] = {"name_token": "_bias"}
    assert _run_probe(full_feature_probe, None, "--strict") == 1
    assert "48 instances" in capsys.readouterr().err
    assert re_.plan_space(manifest, out=io.StringIO(), strict=True) == 1
    features["bias"]["free_fields"] = ["tile_n"]
    assert _run_probe(full_feature_probe, None, "--strict") == 0
    assert len(_probe_kernels(full_feature_probe)) == 36
    shown = io.StringIO()
    assert re_.plan_space(manifest, out=shown, strict=True) == 0
    assert (
        "instances (current classification): 36 (within budget 40)" in shown.getvalue()
    )


@pytest.mark.parametrize("knob", ["tile_n", "order"])
def test_plan_space_counts_a_dropped_knob_through_the_narrowing(
    full_feature_probe, knob
):
    """``if dropped <knob>`` is the count of a manifest that no longer lists the knob,
    narrowing included."""
    manifest = full_feature_probe.manifest
    manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "free_fields": ["tile_n"],
    }
    shown = io.StringIO()
    assert re_.plan_space(manifest, out=shown) == 0
    assert "instances (current classification): 36 " in shown.getvalue()
    dropped = int(
        re.search(rf"^  if dropped {knob}: (\d+) ", shown.getvalue(), re.M)[1]
    )
    performance = manifest["reduction"]["fields"]["performance"]
    performance[:] = [k for k in performance if k["name"] != knob]
    config = re_.build_config(manifest, warn=lambda message: None)
    assert dropped == len(config["packs"][0]["kernels"])


def _with_causal_feature(manifest: dict, **entry) -> None:
    """Add ``causal`` as a second feature: a cell with the ``CAUSAL`` mask turns it on."""
    manifest["kmd_fields"].append({"name": "causal", "type": "int", "default_value": 0})
    manifest["specialization"]["metadata_fields"].append("causal")
    manifest["specialization"]["bindings"]["causal"] = {"field": "causal"}
    manifest["enumerate"]["feature_fields"]["causal"] = {
        "name_token": "_causal",
        **entry,
    }


def test_a_full_space_cell_with_two_features_sweeps_the_intersection(
    full_feature_probe,
):
    """A cell turning on both features frees only what both free, and keeps the specs
    the selects of both name; a cell with one feature is narrowed by that one alone."""
    manifest = full_feature_probe.manifest
    manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "free_fields": ["tile_n", "order"],
    }
    _with_causal_feature(manifest, free_fields=["order"], select={"order": "col"})
    assert _run_probe(full_feature_probe) == 0
    kernels = _probe_kernels(full_feature_probe)

    def cell(bias, causal) -> list:
        return [
            k
            for k in kernels
            if bool(k["kernel_source"]["spec"]["bias"]) is bias
            and bool(k["kernel_source"]["spec"]["causal"]) is causal
        ]

    both = cell(True, True)
    assert len(both) == 2  # two head sizes, the CAUSAL mask, bias on
    assert _knob_values(both, "tile_n") == {64} and _knob_values(both, "order") == {
        "col"
    }
    assert len(_defaults(both)) == 2
    assert all(k["name"].endswith("_bias_causal") for k in both)
    bias_only = cell(True, False)
    assert len(bias_only) == 2 * _KERNELS_PER_CELL
    assert _knob_values(bias_only, "tile_n") == {32, 64, 128}
    causal_only = cell(False, True)
    assert len(causal_only) == 2 and _knob_values(causal_only, "order") == {"col"}
    assert len(cell(False, False)) == 2 * _KERNELS_PER_CELL
    assert len(kernels) == 28


@pytest.mark.parametrize(
    "bias, causal, offender",
    [
        (
            {"select": {"tile_n": 32}},
            {"select": {"tile_n": 64}},
            "another feature this cell turns on",
        ),
        (
            {"select": {"tile_n": 32}},
            {"free_fields": ["order"]},
            "another feature's free_fields holds",
        ),
    ],
    ids=["selects-disagree", "select-held-by-the-other-features-free-fields"],
)
def test_a_full_space_cell_whose_two_features_conflict_is_exit_2(
    full_feature_probe, capsys, bias, causal, offender
):
    manifest = full_feature_probe.manifest
    manifest["enumerate"]["feature_fields"]["bias"] = {"name_token": "_bias", **bias}
    _with_causal_feature(manifest, **causal)
    assert _run_probe(full_feature_probe) == 2
    assert offender in capsys.readouterr().err
    assert not full_feature_probe.out.exists()


def _fp16_also(manifest: dict, **overrides) -> None:
    """Add an ``also`` section over the primary's own pin and cells, at ``fp16`` so its
    kernel names differ from the primary's; ``overrides`` set the section's keys."""
    primary = manifest["enumerate"]
    manifest["enumerate"]["also"] = [
        {
            "pin": copy.deepcopy(primary["pin"]),
            "fixed": {**primary["fixed"], "dtype": "fp16"},
            "cells": copy.deepcopy(primary["cells"]),
            **overrides,
        }
    ]


def _fp16_bias_on(probe) -> list:
    return [
        k
        for k in _probe_kernels(probe)
        if k["kernel_source"]["spec"]["dtype"] == "fp16"
        and k["kernel_source"]["spec"]["bias"]
    ]


def test_a_full_also_section_is_narrowed_by_the_primarys_feature_entry(
    full_feature_probe,
):
    """``feature_fields`` is declared on the primary and applies to every section: a full
    ``also`` section's cells that turn the feature on vary only the knobs ``free_fields``
    names, as the primary's do, and --plan-space counts the same kernels."""
    manifest = full_feature_probe.manifest
    manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "free_fields": ["tile_n"],
    }
    _fp16_also(manifest)
    assert _run_probe(full_feature_probe) == 0
    extra = _fp16_bias_on(full_feature_probe)
    assert len(extra) == 3 * _CELLS_PER_BIAS_VALUE
    assert _knob_values(extra, "tile_n") == {32, 64, 128}
    assert all("order" not in k["kernel_source"]["spec"] for k in extra)
    assert all(k["name"].endswith("_bias") for k in extra)
    # The cells are the primary's too, so the primary keeps their one default.
    assert not _defaults(extra)
    # The section's bias-off cells keep the full sweep.
    off = [
        k
        for k in _probe_kernels(full_feature_probe)
        if k["kernel_source"]["spec"]["dtype"] == "fp16"
        and not k["kernel_source"]["spec"]["bias"]
    ]
    assert len(off) == _KERNELS_PER_CELL * _CELLS_PER_BIAS_VALUE
    shown = io.StringIO()
    assert re_.plan_space(manifest, out=shown) == 0
    total = len(_probe_kernels(full_feature_probe))
    assert f"instances (current classification): {total} " in shown.getvalue()


def test_a_full_also_cell_a_feature_would_empty_is_exit_2(full_feature_probe, capsys):
    """A select no spec of the also section's cell satisfies fails the build and
    --plan-space, as the primary's does."""
    manifest = full_feature_probe.manifest
    manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "select": {"tile_n": 48},
    }
    manifest["enumerate"]["cells"][-1]["values"] = [False]
    also_cells = copy.deepcopy(manifest["enumerate"]["cells"])
    also_cells[-1]["values"] = [True]
    _fp16_also(manifest, cells=also_cells)
    assert _run_probe(full_feature_probe) == 2
    assert "no spec for cell" in capsys.readouterr().err
    with pytest.raises(re_.EnumerationError, match="no spec for cell"):
        re_.plan_space(manifest, out=io.StringIO())


def test_a_slice_also_section_under_a_full_primary_keeps_what_both_free(
    full_feature_probe,
):
    """A slice ``also`` section sweeps the intersection of its own ``free_fields`` and the
    feature entry's: ``tile_n`` in both keeps every tile, none in the section keeps the
    default tile."""
    manifest = full_feature_probe.manifest
    manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "free_fields": ["tile_n"],
    }
    _fp16_also(manifest, space="slice", free_fields=["tile_n", "order"])
    assert _run_probe(full_feature_probe) == 0
    extra = _fp16_bias_on(full_feature_probe)
    assert _knob_values(extra, "tile_n") == {32, 64, 128}
    assert len(extra) == 3 * _CELLS_PER_BIAS_VALUE
    _fp16_also(manifest, space="slice", free_fields=[])
    assert _run_probe(full_feature_probe) == 0
    extra = _fp16_bias_on(full_feature_probe)
    assert _knob_values(extra, "tile_n") == {64}
    assert len(extra) == _CELLS_PER_BIAS_VALUE


def test_a_full_also_section_under_a_slice_primary_is_narrowed_too(
    full_feature_probe,
):
    """The mirror image: a slice primary whose feature entry names ``tile_n`` (in its
    ``free_fields``) narrows a full ``also`` section's feature cells to that knob."""
    manifest = full_feature_probe.manifest
    reduction = manifest["reduction"]
    primary = manifest["enumerate"]
    primary["space"] = "slice"
    primary["free_fields"] = ["tile_n", "order"]
    primary["feature_fields"]["bias"] = {
        "name_token": "_bias",
        "free_fields": ["tile_n"],
    }
    _fp16_also(manifest, space="full")
    manifest["reduction"] = reduction
    assert _run_probe(full_feature_probe) == 0
    extra = _fp16_bias_on(full_feature_probe)
    assert len(extra) == 3 * _CELLS_PER_BIAS_VALUE
    assert _knob_values(extra, "tile_n") == {32, 64, 128}
    assert all("order" not in k["kernel_source"]["spec"] for k in extra)


_PERFORMANCE = "reduction.fields.performance"


@pytest.mark.parametrize(
    "entry, field, shape",
    [
        ({"free_fields": ["tile_m"]}, "bias.free_fields", _PERFORMANCE),
        ({"free_fields": ["no_such"]}, "bias.free_fields", _PERFORMANCE),
        ({"free_fields": "tile_n"}, "bias.free_fields", "a list of field names"),
        ({"select": {"tile_m": 128}}, "bias.select", _PERFORMANCE),
        (
            {"free_fields": ["order"], "select": {"tile_n": 32}},
            "bias.select",
            "free_fields",
        ),
        ({"free_fields": [], "select": {"tile_n": 32}}, "bias.select", "free_fields"),
        ({"select": {}}, "bias.select", "non-empty mapping"),
        ({"select": ["tile_n"]}, "bias.select", "non-empty mapping"),
        ({"select": {"tile_n": [32]}}, "bias.select", "scalar values"),
    ],
    ids=[
        "free-field-a-spec-field-but-not-a-performance-knob",
        "free-field-unknown",
        "free-fields-not-a-list",
        "select-a-spec-field-but-not-a-performance-knob",
        "select-outside-narrowed-free-fields",
        "select-with-no-free-fields",
        "select-empty",
        "select-not-a-mapping",
        "select-value-not-a-scalar",
    ],
)
def test_a_malformed_full_space_feature_field_is_exit_2(
    full_feature_probe, capsys, entry, field, shape
):
    """Each refusal names the field and what a full space allows there."""
    full_feature_probe.manifest["enumerate"]["feature_fields"]["bias"] = {
        "name_token": "_bias",
        **entry,
    }
    assert _run_probe(full_feature_probe) == 2
    err = capsys.readouterr().err
    assert f"enumerate.feature_fields.{field}" in err, err
    assert shape in err, err
    assert not full_feature_probe.out.exists()


def test_a_builder_value_outside_the_vocabulary_is_refused(probe):
    probe.manifest["specialization"]["vocabulary"]["dtype"] = {"fp16": "FP16"}
    with pytest.raises(re_.EnumerationError, match=r"dtype.*bf16|bf16.*dtype"):
        re_.build_config(probe.manifest, warn=lambda message: None)


@pytest.mark.parametrize("flag", ["--out", "--shapes"])
def test_plan_space_refuses_the_options_it_would_ignore(tmp_path, flag):
    target = tmp_path / "ignored"
    with pytest.raises(SystemExit) as exit_info:
        re_.main(["--manifest", str(_MANIFEST), "--plan-space", flag, str(target)])
    assert exit_info.value.code == 2
    assert not target.exists()


def test_a_tri_state_free_field_enumerates(probe):
    _free("tile_n", "order")(probe.manifest)
    values = {ks.order for _, ks, _ in re_.enumerate_space(probe.manifest).rows}
    assert None in values and len(values) > 1, values


def test_a_fixed_enum_field_resolves_by_name(probe):
    def causal(name):
        probe.manifest["enumerate"]["fixed"]["mask"] = name
        probe.manifest["enumerate"]["cells"] = probe.manifest["enumerate"]["cells"][:1]
        rows = re_.enumerate_space(probe.manifest).rows
        return {ks.causal for _, ks, _ in rows}

    assert causal("NO_MASK") == {False}
    assert causal("CAUSAL") == {True}


def test_a_cell_lists_its_default_first_then_sorted(manifest, registry):
    rows = re_.enumerate_space(_one_head_config(manifest)).rows
    first = [(ks.block_m, ks.block_n, d) for c, ks, d in rows if c == rows[0][0]]
    assert first[0][2] and not any(d for _, _, d in first[1:])
    assert first[1:] == sorted(first[1:])


def test_a_non_attention_family_enumerates_and_generates(tmp_path, monkeypatch):
    """A second family, unrelated to the first: request-only fields reach the name
    and metadata but not the spec, and the generator accepts the result."""
    root = synth.provider(tmp_path, monkeypatch)
    manifest = synth.conv_probe_manifest(root)
    config = tmp_path / "config.yaml"
    path = synth.write_manifest(tmp_path / "conv.engine.yaml", manifest)
    assert re_.main(["--manifest", str(path), "--out", str(config)]) == 0
    kernels = yaml.safe_load(config.read_text())["packs"][0]["kernels"]
    cells = {
        (c, d) for c in (64, 128) for d in manifest["enumerate"]["cells"][1]["values"]
    }
    assert {k["name"] for k in kernels} == {
        f"conv_probe.{d}_c{c}_k{c}_g1_t64x64x64" for c, d in cells
    }
    assert all(not {"C", "K", "G"} & set(k["kernel_source"]["spec"]) for k in kernels)
    assert {(k["metadata"]["C"], k["metadata"]["K"]) for k in kernels} == {
        (64, 64),
        (128, 128),
    }
    gen = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "generate.py"),
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "out"),
            "--force",
        ],
        capture_output=True,
        text=True,
    )
    assert gen.returncode == 0, gen.stderr[-2000:]


@pytest.mark.skipif(
    not _ROCKE_LIBRARY.is_dir(),
    reason=f"rocKE library not in this checkout: {_ROCKE_LIBRARY}",
)
def test_the_shipped_manifest_enumerates_against_the_live_registry(tmp_path):
    """Asserts only what the tool owns; which kernels rocKE yields is g1/g3's."""
    config = tmp_path / "config.yaml"
    assert re_.main(["--manifest", str(_MANIFEST), "--out", str(config)]) == 0
    kernels = yaml.safe_load(config.read_text())["packs"][0]["kernels"]
    assert kernels
    assert len({k["name"] for k in kernels}) == len(kernels)
    declared = re_.load_manifest(str(_MANIFEST))["specialization"]["metadata_fields"]
    assert all(set(k["metadata"]) == set(declared) for k in kernels)


class _Color(enum.Enum):
    RED = "red"


class _Shape(enum.Enum):
    CIRCLE = 2


@dataclasses.dataclass(frozen=True)
class _Spec:
    n: int
    tile: int = 64
    fence: typing.Optional[bool] = None
    tmpl: typing.Optional[bool] = None

    def resolved_fence(self):
        return True if self.fence is None else self.fence

    def resolved_tmpl(self):
        return True if self.tmpl is None else self.tmpl


def test_coerce_finds_a_member_by_name_or_value_in_any_enum_of_a_union():
    registry = object.__new__(re_.Registry)
    registry.hints = {"c": _Color, "u": typing.Union[_Color, _Shape]}
    assert registry.coerce("c", "red", strict=True) is _Color.RED
    assert registry.coerce("u", "CIRCLE", strict=True) is _Shape.CIRCLE
    assert registry.coerce("u", 2, strict=True) is _Shape.CIRCLE
    with pytest.raises(re_.EnumerationError, match=r"_Color or _Shape.*'CIRCLE'"):
        registry.coerce("u", "SQUARE", strict=True)


def test_specs_resolving_alike_collapse_to_the_default_then_the_most_unset():
    accessors = {"fence": "resolved_fence", "tmpl": "resolved_tmpl"}
    kept = [_Spec(1), _Spec(1, fence=True, tmpl=True), _Spec(1, tile=128)]
    assert re_._collapse_unset(kept, accessors, kept[0]) == [kept[0], kept[2]]
    pinned = _Spec(1, fence=True)
    assert re_._collapse_unset([_Spec(1), pinned], accessors, pinned) == [pinned]


@pytest.mark.parametrize("kind", ["performance", "ignored"])
def test_a_classified_name_the_spec_lacks_is_named(kind):
    fields = {"feature": [], "performance": OrderedDict(), "ignored": []}
    fields[kind] = OrderedDict(tile_typo=None) if kind == "performance" else ["x"]
    classification = re_.Classification(budget=10, **fields)
    with pytest.raises(re_.EnumerationError, match=rf"fields\.{kind} names \['"):
        re_._feature_matched([(_Spec(1), [_Spec(1)])], classification)


def test_the_order_key_is_total_on_mixed_values_and_numeric():
    values = [None, True, 3, False, 1, None, 0, "k_major", 2.0]
    orders = {
        tuple(map(repr, sorted(perm, key=re_._order_key)))
        for perm in (values, values[::-1], values[3:] + values[:3])
    }
    assert len(orders) == 1
    assert sorted([128, None, 32, 1024], key=re_._order_key) == [32, 128, 1024, None]


# selector_fields / neutral_fields: a rule-driven candidate (the split probe picks
# num_segments from batch and seqlen) names one kernel from many request combos.


@pytest.fixture
def split(tmp_path, monkeypatch):
    return synth.split_probe_manifest(synth.provider(tmp_path, monkeypatch))


def _dedup(manifest, selector=("batch", "seqlen"), neutral=("num_seqs",)):
    out = copy.deepcopy(manifest)
    out["enumerate"]["selector_fields"] = list(selector)
    out["enumerate"]["neutral_fields"] = list(neutral)
    return out


def _kernels(manifest) -> list:
    config = re_.build_config(manifest, warn=lambda message: None)
    return config["packs"][0]["kernels"]


def test_without_dedup_keys_a_rule_driven_family_repeats_its_names(split):
    """Today's behaviour, kept: one cell per combo, so one name per combo."""
    with pytest.raises(re_.EnumerationError, match="duplicated kernel name"):
        _kernels(split)


def test_selector_only_cells_merge_into_the_union_of_their_picks(split):
    kernels = _kernels(_dedup(split))
    by_cell = OrderedDict()
    for k in kernels:
        by_cell.setdefault(k["metadata"]["hdim"], []).append(k)
    assert list(by_cell) == [64, 128]
    for hdim, cell in by_cell.items():
        spec = [k["kernel_source"]["spec"] for k in cell]
        # Combo picks 1, 8, 1, 4, 1, 1 -> {1, 4, 8}; the first combo's pick is the
        # default, the rest follow sorted.
        assert [s["num_segments"] for s in spec] == [1, 4, 8]
        assert [bool(k.get("default")) for k in cell] == [True, False, False]
        assert [k["name"] for k in cell] == [
            f"split_probe.fp16_d{hdim}_num_segments{n}" for n in (1, 4, 8)
        ]
    # The merged cell carries no selector field: no kernel claims one combo's value.
    assert all("batch" not in k["metadata"] for k in kernels)


def test_the_merged_default_is_the_first_combos_pick(split):
    group = next(g for g in split["enumerate"]["cells"] if g["fields"] == ["seqlen"])
    group["values"] = [1024, 64]
    cell = [k for k in _kernels(_dedup(split)) if k["metadata"]["hdim"] == 64]
    assert [k["kernel_source"]["spec"]["num_segments"] for k in cell] == [8, 1, 4]
    assert [bool(k.get("default")) for k in cell] == [True, False, False]


def test_neutral_fields_take_the_default_specs_value(split):
    specs = [k["kernel_source"]["spec"] for k in _kernels(_dedup(split))]
    assert {s["num_seqs"] for s in specs} == {1}
    batch = next(g for g in split["enumerate"]["cells"] if g["fields"] == ["batch"])
    batch["values"] = [2, 1, 8]
    specs = [k["kernel_source"]["spec"] for k in _kernels(_dedup(split))]
    assert {s["num_seqs"] for s in specs} == {2}


def test_a_combo_pick_the_merged_cell_drops_is_refused(split):
    """Without num_seqs neutral, num_seqs varies, is held at the default, and the
    batch-2 combos' kernels would silently not ship."""
    with pytest.raises(
        re_.EnumerationError, match=r"num_seqs=2\) for request \{hdim: 64, batch: 2"
    ):
        _kernels(_dedup(split, neutral=()))


def test_spec_overrides_pin_a_varying_field_so_its_combos_merge(split):
    """The pick varies num_seqs (refused above); pinning it ships every combo's
    pick with the pinned value instead."""
    manifest = _dedup(split, neutral=())
    manifest["enumerate"]["spec_overrides"] = {"num_seqs": 4}
    kernels = _kernels(manifest)
    specs = [k["kernel_source"]["spec"] for k in kernels]
    assert {s["num_seqs"] for s in specs} == {4}
    cell = [s for s in specs if s["hdim"] == 64]
    assert [s["num_segments"] for s in cell] == [1, 4, 8]


def test_a_spec_override_the_spec_refuses_is_exit_2_naming_the_request(
    split, tmp_path, capsys
):
    """The spec's own validation decides whether an override builds."""
    manifest = _dedup(split, neutral=())
    manifest["enumerate"]["spec_overrides"] = {"num_seqs": 0}
    path = synth.write_manifest(tmp_path / "m.engine.yaml", manifest)
    out = tmp_path / "c.yaml"
    assert re_.main(["--manifest", str(path), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "spec_overrides {'num_seqs': 0} does not build for request" in err, err
    assert "{hdim: 64, batch: 1" in err, err
    assert "num_seqs must be >= 1" in err, err
    assert not out.exists()


def _override_and_neutral(m):
    m["enumerate"]["spec_overrides"] = {"num_seqs": 4}


def test_plan_space_counts_after_merging(split, tmp_path, capsys):
    path = synth.write_manifest(tmp_path / "m.engine.yaml", _dedup(split))
    assert re_.main(["--manifest", str(path), "--plan-space"]) == 0
    out = capsys.readouterr().out
    assert "cells: 2 (12 request combos merged on selector_fields" in out
    assert "instances (current classification): 6 " in out
    assert "num_seqs: {" not in out


def test_plan_space_does_not_claim_a_merge_when_nothing_merges(split, tmp_path, capsys):
    """neutral_fields alone merges no combo: the cell line says so plainly."""
    path = synth.write_manifest(tmp_path / "m.engine.yaml", _dedup(split, selector=()))
    assert re_.main(["--manifest", str(path), "--plan-space"]) == 0
    out = capsys.readouterr().out
    assert "  cells: 12\n" in out
    assert "merged" not in out


def _also_feature(m):
    m["enumerate"]["selector_fields"] = ["hdim", "batch"]


def _also_performance(m):
    m["enumerate"]["neutral_fields"] = ["num_segments"]


@pytest.mark.parametrize(
    "mutate, words",
    [
        (lambda m: m["enumerate"].update(selector_fields="batch"), ["selector_fields"]),
        (lambda m: m["enumerate"].update(neutral_fields=[1]), ["neutral_fields"]),
        (
            lambda m: m["enumerate"].update(selector_fields=["batch", "nope"]),
            ["selector_fields", "nope"],
        ),
        (
            lambda m: m["enumerate"].update(selector_fields=["dtype"]),
            ["selector_fields", "dtype"],
        ),
        (
            lambda m: m["enumerate"].update(neutral_fields=["no_such"]),
            ["neutral_fields", "no_such"],
        ),
        (_also_feature, ["selector_fields", "hdim", "feature"]),
        (_also_performance, ["neutral_fields", "num_segments", "performance"]),
        (
            lambda m: m["enumerate"].update(spec_overrides=["num_seqs"]),
            ["spec_overrides", "mapping"],
        ),
        (
            lambda m: m["enumerate"].update(spec_overrides={"no_such": 1}),
            ["spec_overrides", "no_such"],
        ),
        (
            lambda m: m["enumerate"].update(spec_overrides={"hdim": 128}),
            ["spec_overrides", "hdim", "feature"],
        ),
        (_override_and_neutral, ["spec_overrides", "neutral_fields", "num_seqs"]),
    ],
    ids=[
        "selector-not-a-list",
        "neutral-not-names",
        "selector-unknown",
        "selector-not-a-cell-field",
        "neutral-unknown",
        "selector-is-feature",
        "neutral-is-performance",
        "overrides-not-a-mapping",
        "overrides-unknown",
        "overrides-is-feature",
        "overrides-is-neutral",
    ],
)
@pytest.mark.parametrize("mode", ["--plan-space", "--out"])
def test_a_bad_dedup_key_is_exit_2_naming_it(
    split, tmp_path, capsys, mutate, words, mode
):
    manifest = _dedup(split)
    mutate(manifest)
    path = synth.write_manifest(tmp_path / "m.engine.yaml", manifest)
    out = tmp_path / "c.yaml"
    argv = ["--plan-space"] if mode == "--plan-space" else ["--out", str(out)]
    assert re_.main(["--manifest", str(path), *argv]) == 2
    err = capsys.readouterr().err
    assert all(w in err for w in words), err
    assert not out.exists()


def test_duplicate_names_after_merging_still_fail(split):
    manifest = _dedup(split)
    manifest["kernel_name_template"] = "split_probe.{dtype}"
    with pytest.raises(re_.EnumerationError, match="duplicated kernel name"):
        _kernels(manifest)


@pytest.mark.parametrize("family", ["probe", "conv"])
@pytest.mark.parametrize("mode", ["--plan-space", "--out"])
def test_empty_dedup_keys_are_byte_identical_to_none(
    tmp_path, monkeypatch, capsys, family, mode
):
    """The keys are opt-in: an absent key and an empty one produce the same bytes
    (gfx950's own regeneration gates pin absent == today)."""
    root = synth.provider(tmp_path, monkeypatch)
    base = (
        synth.probe_manifest(root)
        if family == "probe"
        else synth.conv_probe_manifest(root)
    )
    if family == "probe" and mode == "--plan-space":
        base["enumerate"]["space"] = "full"
        base["reduction"] = {
            "fields": {"feature": ["dtype"], "performance": [], "ignored": []}
        }
    outputs = []
    for manifest in (base, _dedup(base, selector=(), neutral=())):
        path = synth.write_manifest(tmp_path / "m.engine.yaml", manifest)
        out = tmp_path / "c.yaml"
        out.unlink(missing_ok=True)
        argv = ["--plan-space"] if mode == "--plan-space" else ["--out", str(out)]
        capsys.readouterr()
        assert re_.main(["--manifest", str(path), *argv]) == 0
        outputs.append(
            (capsys.readouterr().out, out.read_bytes() if out.exists() else None)
        )
    assert outputs[0] == outputs[1]


def _shapes_run(manifest, tmp_path, capsys, shapes) -> str:
    path = synth.write_manifest(tmp_path / "m.engine.yaml", manifest)
    (data := tmp_path / "shapes.json").write_text(json.dumps(shapes))
    out = tmp_path / "c.yaml"
    capsys.readouterr()
    argv = ["--manifest", str(path), "--out", str(out), "--shapes", str(data)]
    assert re_.main(argv) == 0
    return capsys.readouterr().out


def test_the_shapes_report_keys_a_merged_cell_without_its_selector_fields(
    split, tmp_path, capsys
):
    """A graph carries no selector field (or a value the cells do not list): the
    selector fields only steer the rule, so the shape lands in its merged cell."""
    out = _shapes_run(
        _dedup(split),
        tmp_path,
        capsys,
        [{"hdim": 64}, {"hdim": 128, "batch": 16, "seqlen": 4096}],
    )
    assert "  covered: 2 shapes, 2 cells\n" in out, out
    assert "  GAP: 0 shapes, 0 cells\n" in out, out


def test_a_shape_outside_the_merged_cells_is_still_a_gap(split, tmp_path, capsys):
    out = _shapes_run(_dedup(split), tmp_path, capsys, [{"hdim": 256}, {"hdim": 64}])
    assert "  covered: 1 shapes, 1 cells\n" in out, out
    assert "  GAP: 1 shapes, 1 cells\n    1 x {hdim: 256}\n" in out, out


# cell_keyed_by: head counts are runtime values and only the GQA ratio, a spec
# property, is baked, so head pairs sharing a ratio merge into one cell.


@pytest.fixture
def gqa(tmp_path, monkeypatch):
    return synth.gqa_probe_manifest(synth.provider(tmp_path, monkeypatch))


def test_cell_keyed_by_merges_head_pairs_per_ratio(gqa):
    kernels = _kernels(gqa)
    by_ratio = OrderedDict()
    for k in kernels:
        by_ratio.setdefault(k["metadata"]["num_queries_per_kv"], []).append(k)
    # Cells in first-combo order: (4, 1) is ratio 4, then (2, 1), then (2, 2).
    assert list(by_ratio) == [4, 2, 1]
    segments = {
        r: [k["kernel_source"]["spec"]["num_segments"] for k in ks]
        for r, ks in by_ratio.items()
    }
    # Ratio 4 holds both pairs' picks: (4, 1) 8 and 4, (8, 2) 4 and 2.
    assert segments == {4: [8, 2, 4], 2: [8, 4], 1: [4, 2]}
    assert [k["name"] for k in by_ratio[4]] == [
        f"gqa_probe.fp16_d64_g4_num_segments{n}" for n in (8, 2, 4)
    ]
    # The neutral head counts take the cell default's: one kernel per ratio.
    assert {
        (
            k["kernel_source"]["spec"]["num_query_heads"],
            k["kernel_source"]["spec"]["num_kv_heads"],
        )
        for k in by_ratio[4]
    } == {(4, 1)}


def test_a_neutral_field_that_moves_the_keyed_ratio_is_refused(gqa):
    """Only num_query_heads neutral: (8, 2) would become (4, 2), ratio 2, in the
    ratio-4 cell."""
    gqa["enumerate"]["neutral_fields"] = ["num_query_heads"]
    with pytest.raises(
        re_.EnumerationError, match=r"cell_keyed_by.*num_queries_per_kv"
    ):
        _kernels(gqa)


def test_the_shapes_report_keys_a_shape_by_its_picks_ratio(gqa, tmp_path, capsys):
    """(16, 4) is a pair no cell lists but ratio 4 is shipped; (3, 1) is ratio 3."""
    out = _shapes_run(
        gqa,
        tmp_path,
        capsys,
        [{"nhead_q": 16, "nhead_k": 4}, {"nhead_q": 3, "nhead_k": 1}],
    )
    assert "  covered: 1 shapes, 1 cells\n" in out, out
    assert "  GAP: 1 shapes, 1 cells\n    1 x {num_queries_per_kv: 3}\n" in out, out


@pytest.mark.parametrize(
    "keyed, words",
    [
        (["no_such"], ["cell_keyed_by", "no_such"]),
        (["batch"], ["cell_keyed_by", "batch"]),
    ],
    ids=["unknown-attribute", "a-cell-field"],
)
def test_a_bad_cell_keyed_by_is_exit_2_naming_it(gqa, tmp_path, capsys, keyed, words):
    gqa["enumerate"]["cell_keyed_by"] = keyed
    path = synth.write_manifest(tmp_path / "m.engine.yaml", gqa)
    assert re_.main(["--manifest", str(path), "--plan-space"]) == 2
    err = capsys.readouterr().err
    assert all(w in err for w in words), err


def test_each_kernel_carries_the_pins_spec_id(probe):
    """`provenance.rocke_spec_id` is the pin's spec_id, so a shipped kernel traces to
    the registry entry that produced it. The probe candidate hands back unwrapped
    specs, so there is no tuning id to carry."""
    assert _run_probe(probe) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert {k["provenance"]["rocke_spec_id"] for k in kernels} == {"grid"}
    assert not any("rocke_tuning_id" in k["provenance"] for k in kernels)


def test_the_dense_registry_names_a_tuning_id_for_every_kernel(manifest, registry):
    config = re_.build_config(_one_head_config(manifest), registry)
    kernels = [k for pack in config["packs"] for k in pack["kernels"]]
    assert kernels
    pinned = manifest["enumerate"]["pin"]["spec_id"]
    for kernel in kernels:
        assert kernel["provenance"]["rocke_spec_id"] == pinned
        assert kernel["provenance"]["rocke_tuning_id"]


def test_a_mined_dense_shape_row_without_use_attn_bias_is_covered(registry):
    """`mine_shapes.py` rows predate the ``use_attn_bias`` cell axis: such a row is a
    bias-off request, and a row that sets the axis lands in the bias-on cell. The row uses
    the canonical 8 / 8 heads, the only head configuration the dense cells list: the
    kernel reads its head counts at runtime."""
    row = {
        "batch": 1,
        "nhead_q": 8,
        "nhead_k": 8,
        "seqlen_q": 512,
        "seqlen_k": 512,
        "hdim_q": 64,
        "hdim_v": 64,
        "dtype": "bf16",
        "mask_type": 1,
        "sliding_window": 0,
        "use_sinks": False,
    }
    lines, gaps, covered = re_.shapes_report(
        registry, [row, {**row, "use_attn_bias": True}]
    )
    assert (gaps, covered) == (0, 2)
    # Two cells, not one: the omitted axis is bias-off and the set one is bias-on.
    assert "  covered: 2 shapes, 2 cells" in lines


# ---------------------------------------------------------------------------
# enumerate.also: one pack drawn from more than one registry candidate
# ---------------------------------------------------------------------------


def _also(**overrides) -> dict:
    """A second section over the probe family's `persist`/`a` candidate, on a cell
    the primary section does not list."""
    section = {
        "pin": {"algorithm": "persist", "spec_id": "a"},
        "fixed": {"arch": synth.ARCH, "dtype": "bf16"},
        "cells": [
            {"fields": ["hdim", "heads"], "values": [[128, 16]]},
            {"fields": ["mask"], "values": ["CAUSAL"]},
        ],
        "free_fields": [],
        "spec_fields": ["dtype", "tile_m", "tile_n", "causal", "order"],
    }
    section.update(overrides)
    return section


def _with_also(probe, **overrides) -> dict:
    manifest = copy.deepcopy(probe.manifest)
    manifest["enumerate"]["also"] = [_also(**overrides)]
    return manifest


def test_an_also_section_adds_its_kernels_after_the_primary_sections(probe):
    """The pack is one list: primary kernels first, then each section's, each section
    traced to its own pin."""
    assert _run_probe(probe) == 0
    primary = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert _run_probe(probe, _with_also(probe)) == 0
    both = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert [k["name"] for k in both[: len(primary)]] == [k["name"] for k in primary]
    extra = both[len(primary) :]
    assert len(extra) == 1
    assert extra[0]["provenance"]["rocke_spec_id"] == "a"
    assert {k["provenance"]["rocke_spec_id"] for k in primary} == {"grid"}


def test_an_also_section_projects_only_its_own_spec_fields(probe):
    """`spec_fields` is per section: the primary's narrower list must not decide what a
    later section's kernel spec carries, or its kernels would compile at defaults."""
    manifest = _with_also(probe)
    manifest["enumerate"]["spec_fields"] = ["dtype", "tile_m", "tile_n", "causal"]
    assert _run_probe(probe, manifest) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert "order" not in kernels[0]["kernel_source"]["spec"]
    assert "order" in kernels[-1]["kernel_source"]["spec"]


def test_an_also_section_has_its_own_max_kernels(probe):
    assert _run_probe(probe, _with_also(probe, max_kernels=1)) == 0
    assert _run_probe(probe, _with_also(probe, max_kernels=0)) == 2


def test_an_also_section_may_not_repeat_a_kernel_name(probe):
    """Two sections naming one kernel would collide in the descriptor catalog."""
    manifest = _with_also(
        probe,
        cells=[
            {"fields": ["hdim", "heads"], "values": [[64, 8]]},
            {"fields": ["mask"], "values": ["CAUSAL"]},
        ],
        free_fields=["tile_n"],
    )
    assert _run_probe(probe, manifest) == 2


def test_an_also_section_must_cross_the_same_cell_fields(probe):
    manifest = _with_also(probe, cells=[{"fields": ["hdim"], "values": [[128]]}])
    assert _run_probe(probe, manifest) == 2


# ---------------------------------------------------------------------------
# enumerate.also spec_overrides: twins of the primary's kernels with one spec field
# the registry never selects (e.g. an LSE output) set on
# ---------------------------------------------------------------------------


def _twin_section(**overrides) -> dict:
    """The primary's own pin and cells, one kernel per cell (no free fields), with
    ``order`` overridden and a template of its own."""
    section = {
        "pin": {"algorithm": "probe", "spec_id": "grid"},
        "fixed": {"arch": synth.ARCH, "dtype": "bf16"},
        "cells": copy.deepcopy(synth.probe_manifest("unused")["enumerate"]["cells"]),
        "free_fields": [],
        "spec_overrides": {"order": "col"},
        "kernel_name_template": "probe.{dtype}_d{hdim}_h{heads}_c{causal}_tn{tile_n}_col",
    }
    section.update(overrides)
    return section


def _with_twins(probe, **overrides) -> dict:
    manifest = copy.deepcopy(probe.manifest)
    manifest["enumerate"]["also"] = [_twin_section(**overrides)]
    return manifest


def test_spec_overrides_add_twins_and_leave_the_primary_unchanged(probe):
    """One twin per cell, named by the section's template and built with the
    override; every primary kernel keeps its name, spec and default marker."""
    assert _run_probe(probe) == 0
    primary = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert _run_probe(probe, _with_twins(probe)) == 0
    both = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert both[: len(primary)] == primary
    twins = both[len(primary) :]
    assert len(twins) == 4
    assert all(k["name"].endswith("_col") for k in twins)
    assert all(k["kernel_source"]["spec"]["order"] == "col" for k in twins)


def test_a_twin_section_keeps_its_own_default_floor(probe):
    """The twins are their own (cell, overrides) group, so each cell's twin is a
    default that variant reduction keeps, beside the primary's default."""
    assert _run_probe(probe, _with_twins(probe)) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    twins = [k for k in kernels if k["name"].endswith("_col")]
    assert all(k.get("default") for k in twins)
    assert sum(1 for k in kernels if k.get("default")) == 2 * len(twins)


def test_an_override_ships_even_when_spec_fields_omit_it(probe):
    """The override is what sets the twin apart, so its spec always carries it."""
    manifest = _with_twins(probe, spec_fields=["dtype", "tile_m", "tile_n", "causal"])
    assert _run_probe(probe, manifest) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert kernels[-1]["kernel_source"]["spec"]["order"] == "col"


def test_twins_without_their_own_template_collide(probe):
    """The primary's template renders no override slot here, so twin names repeat
    primary names: refused rather than shipped as duplicates."""
    manifest = _with_twins(probe)
    del manifest["enumerate"]["also"][0]["kernel_name_template"]
    assert _run_probe(probe, manifest) == 2


@pytest.mark.parametrize(
    "overrides",
    [{"no_such_field": 1}, {}, ["order"], {"order": ["col"]}],
    ids=["unknown-field", "empty", "not-a-mapping", "non-scalar"],
)
def test_bad_spec_overrides_are_refused(probe, overrides):
    assert _run_probe(probe, _with_twins(probe, spec_overrides=overrides)) == 2


def _feature_order(manifest: dict) -> dict:
    """``manifest`` with ``order`` a feature field, as an LSE flag the twins set is."""
    manifest["reduction"] = {
        "fields": {"feature": ["order"], "performance": [], "ignored": []}
    }
    return manifest


def test_a_twin_override_may_be_a_feature_axis_across_sections(probe):
    """The primary keeps the registry's ``order`` and the twins pin ``col``, so the
    engine's kernels take both values: a feature axis, not a pinned field."""
    assert _run_probe(probe, _feature_order(_with_twins(probe))) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert len([k for k in kernels if k["name"].endswith("_col")]) == 4


def test_an_override_every_section_pins_alike_is_no_feature_axis(probe, capsys):
    """Pinned to ``col`` in every section, ``order`` takes one value engine-wide."""
    manifest = _feature_order(_with_twins(probe))
    manifest["enumerate"]["spec_overrides"] = {"order": "col"}
    assert _run_probe(probe, manifest) == 2
    assert "a pinned field takes one value" in capsys.readouterr().err


def _split_also(split, **overrides) -> dict:
    """``split`` merged on its selector fields, plus a section over hdim 256."""
    manifest = _dedup(split)
    cells = copy.deepcopy(manifest["enumerate"]["cells"])
    cells[0]["values"] = [256]
    manifest["enumerate"]["also"] = [
        {
            "pin": manifest["enumerate"]["pin"],
            "fixed": manifest["enumerate"]["fixed"],
            "cells": cells,
            "neutral_fields": ["num_seqs"],
            **overrides,
        }
    ]
    return manifest


def test_an_also_section_inherits_selector_fields_and_cell_keyed_by(split):
    """Omitted keying keys come from ``enumerate``, so the section merges too."""
    kernels = _kernels(_split_also(split))
    extra = [k for k in kernels if k["metadata"]["hdim"] == 256]
    assert [k["kernel_source"]["spec"]["num_segments"] for k in extra] == [1, 4, 8]
    restated = _split_also(split, selector_fields=["batch", "seqlen"], cell_keyed_by=[])
    assert _kernels(restated) == kernels


@pytest.mark.parametrize(
    "key, value",
    [("selector_fields", ["batch"]), ("cell_keyed_by", ["num_segments"])],
)
def test_an_also_section_with_different_keying_is_refused(split, key, value):
    with pytest.raises(re_.EnumerationError, match=rf"enumerate.also\[0\] sets {key}"):
        _kernels(_split_also(split, **{key: value}))


def _gqa_also(gqa, **overrides) -> dict:
    """``gqa`` plus a section over head pairs (6, 2) and (3, 1), both ratio 3."""
    manifest = copy.deepcopy(gqa)
    primary = manifest["enumerate"]
    primary["also"] = [
        {
            "pin": primary["pin"],
            "fixed": primary["fixed"],
            "cells": [
                {"fields": ["nhead_q", "nhead_k"], "values": [[6, 2], [3, 1]]},
                {"fields": ["batch"], "values": [1, 2]},
            ],
            "neutral_fields": primary["neutral_fields"],
            **overrides,
        }
    ]
    return manifest


def test_an_also_section_inherits_cell_keyed_by_when_the_primary_sets_it(gqa):
    """The ``gqa`` primary keys cells by ratio, so an omitted ``cell_keyed_by`` must
    be copied down: stating it by hand gives the same kernels, and the section's
    head pairs merge into one ratio-3 cell."""
    kernels = _kernels(_gqa_also(gqa))
    ratio3 = [k for k in kernels if k["metadata"]["num_queries_per_kv"] == 3]
    # (6, 2) picks segments 4 and 2, (3, 1) picks 8 and 4; the shared 4 merges.
    segments = [k["kernel_source"]["spec"]["num_segments"] for k in ratio3]
    assert sorted(segments) == [2, 4, 8]
    restated = _gqa_also(
        gqa,
        selector_fields=["nhead_q", "nhead_k", "batch"],
        cell_keyed_by=["num_queries_per_kv"],
    )
    assert _kernels(restated) == kernels


def _full(manifest: dict) -> dict:
    """``manifest`` as a full space: ``tile_n`` a performance knob, ``order`` resolved
    by its accessor, every other varying field held."""
    manifest = copy.deepcopy(manifest)
    manifest["enumerate"]["space"] = "full"
    manifest["enumerate"].pop("free_fields", None)
    manifest["reduction"] = {
        "fields": {
            "feature": ["dtype", "causal", "tile_m"],
            "performance": [
                {"name": "tile_n"},
                {"name": "order", "unset_resolves_by": "resolved_order"},
            ],
            "ignored": [],
        }
    }
    return manifest


def test_full_also_sections_each_project_their_knobs(probe):
    """A full primary section and a full ``also`` section: each keeps every
    performance variant of its own cells, and the catalog's projected knobs are the
    union, carried by every kernel."""
    manifest = _full(_with_also(probe))
    manifest["enumerate"]["also"][0].pop("free_fields")
    assert _run_probe(probe, manifest) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    extra = [k for k in kernels if k["provenance"]["rocke_spec_id"] == "a"]
    # 3 tile_n x 2 resolved orders (an unset order is the "row" binary) for one cell.
    assert len(extra) == 6
    assert {k["metadata"]["tile_n"] for k in extra} == {32, 64, 128}
    assert {k["metadata"]["order"] for k in kernels} == {"row", "col"}
    assert sum(k.get("default", False) for k in extra) == 1


def test_a_full_primary_may_carry_a_slice_also_section(probe):
    """``space`` is inherited from ``enumerate`` unless a section sets its own."""
    manifest = _full(_with_also(probe, free_fields=[], space="slice"))
    assert _run_probe(probe, manifest) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert len([k for k in kernels if k["provenance"]["rocke_spec_id"] == "a"]) == 1


def test_a_cell_two_sections_list_keeps_one_floor(probe):
    """Two sections over the same cells (two geometries of one cell set): every kernel
    ships, but each cell keeps exactly one ``default: true``, the primary section's."""
    manifest = _with_also(
        probe,
        cells=copy.deepcopy(probe.manifest["enumerate"]["cells"]),
        free_fields=["tile_n"],
        # Same cells, different kernels: the dtype is fixed per section, not a cell.
        fixed={"arch": synth.ARCH, "dtype": "fp16"},
    )
    assert _run_probe(probe, manifest) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    defaults = [k for k in kernels if k.get("default")]
    assert len(defaults) == 4  # the primary's two hdim x two mask cells
    assert {k["provenance"]["rocke_spec_id"] for k in defaults} == {"grid"}
    assert any(k["provenance"]["rocke_spec_id"] == "a" for k in kernels)


def test_plan_space_counts_every_section(probe, capsys):
    """--plan-space's instance count is the catalog --out writes, ``also`` included."""
    manifest = _full(_with_also(probe))
    manifest["enumerate"]["also"][0].pop("free_fields")
    path = synth.write_manifest(probe.tmp / "m.engine.yaml", manifest)
    assert re_.main(["--manifest", str(path), "--plan-space"]) == 0
    shown = capsys.readouterr().out
    assert _run_probe(probe, manifest) == 0
    count = len(yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"])
    assert f"instances (current classification): {count} " in shown
    assert "section a (space: full)" in shown


_LAUNCHES = {"main": {}, "reduce": {"role": "reduce"}}


def test_launches_are_copied_onto_every_kernel(probe):
    manifest = copy.deepcopy(probe.manifest)
    manifest["launches"] = _LAUNCHES
    assert _run_probe(probe, manifest) == 0
    kernels = yaml.safe_load(probe.out.read_text())["packs"][0]["kernels"]
    assert kernels and all(k["kernel_source"]["launches"] == _LAUNCHES for k in kernels)


@pytest.mark.parametrize(
    "launches",
    [{"main": {}}, {"Main": {}, "reduce": {"role": "reduce"}}, {"main": {}, "r": 1}],
    ids=["one-launch", "bad-name", "non-mapping"],
)
def test_a_malformed_launches_block_is_refused(probe, launches):
    manifest = copy.deepcopy(probe.manifest)
    manifest["launches"] = launches
    assert _run_probe(probe, manifest) == 2


def test_a_paired_reduce_without_launches_is_refused(probe, capsys):
    """The registry hands back a kernel paired with a reduce; a catalog that does not
    launch it would ship the kernel alone."""
    manifest = copy.deepcopy(probe.manifest)
    manifest["enumerate"]["pin"] = {"algorithm": "split", "spec_id": "s"}
    assert _run_probe(probe, manifest) == 2
    assert "declares no 'launches'" in capsys.readouterr().err
    manifest["launches"] = _LAUNCHES
    assert _run_probe(probe, manifest) == 0


def test_a_shape_only_an_also_section_lists_is_covered(probe, capsys):
    """Coverage is by every section's cells: a shape the primary declines or leaves a
    GAP is covered when a later section lists it."""
    shape = {"hdim": 128, "heads": 16, "dtype": "bf16", "mask": 1}
    shapes = probe.tmp / "shapes.json"
    shapes.write_text(json.dumps([shape]))
    for manifest, gaps in ((probe.manifest, True), (_with_also(probe), False)):
        assert _run_probe(probe, manifest, "--shapes", str(shapes), "--strict") == (
            1 if gaps else 0
        )
    assert "covered: 1 shapes" in capsys.readouterr().out


def test_a_shape_row_omitting_a_cell_axis_takes_the_requests_default(
    feature_probe, capsys
):
    """A row from before the ``bias`` axis (``mine_shapes`` emits none) builds a
    bias-off request, so it is the bias-off cell's shape, not a GAP; an explicit
    bias-on row lands in the bias-on cell."""
    row = {"hdim": 64, "heads": 8, "dtype": "bf16", "mask": 0}
    for shapes, cells in (([row], 1), ([row, {**row, "bias": True}], 2)):
        data = feature_probe.tmp / "shapes.json"
        data.write_text(json.dumps(shapes))
        capsys.readouterr()
        assert _run_probe(feature_probe, None, "--shapes", str(data), "--strict") == 0
        out = capsys.readouterr().out
        assert f"  covered: {len(shapes)} shapes, {cells} cells\n" in out, out
        assert "  GAP: 0 shapes, 0 cells\n" in out, out


def test_a_shape_row_omitting_a_cell_axis_is_not_a_wildcard(feature_probe, capsys):
    """Only the default is assumed: with just the bias-on cells listed, the same
    legacy row is a GAP in the bias-off cell, named as such."""
    group = feature_probe.manifest["enumerate"]["cells"][-1]
    assert group["fields"] == ["bias"]
    group["values"] = [True]
    data = feature_probe.tmp / "shapes.json"
    data.write_text(json.dumps([{"hdim": 64, "heads": 8, "dtype": "bf16", "mask": 0}]))
    assert _run_probe(feature_probe, None, "--shapes", str(data), "--strict") == 1
    out = capsys.readouterr().out
    assert "  GAP: 1 shapes, 1 cells\n" in out, out
    assert "bias: False" in out, out


def test_manifest_heuristics_reach_the_generated_engine_and_its_ued(probe):
    """A promoted model's per-arch binding lives in the manifest's `engine` block: the
    enumerator copies that block into the config it writes (so regeneration gate g4's
    engine equality holds), and the generator emits it after `default`."""
    from codegen.config_loader import load_config
    from codegen.generator import build_ued, mint_ids

    model = "8b3c960a-a47c-4ce3-96f8-343befdbc543"
    manifest = copy.deepcopy(probe.manifest)
    manifest["engine"]["heuristics"] = {"sort_kernel_catalog": {"gfx942": [model]}}
    assert _run_probe(probe, manifest) == 0
    config = load_config(probe.out)
    assert config.engine.heuristics == {"sort_kernel_catalog": {"gfx942": [model]}}
    catalog = build_ued(config, mint_ids(config))["sort_kernel_catalog"]
    assert list(catalog) == ["default", "gfx942"]
    assert catalog["gfx942"] == [model]


# hold_fields / performance overrides: a section pins a performance field to one value (a
# production stack the registry's sweep never combines) and the others hold it.


def _conv_hold(tmp_path, monkeypatch, **enumerate_keys):
    manifest = synth.conv_probe_manifest(synth.provider(tmp_path, monkeypatch))
    manifest["reduction"]["fields"] = {
        "feature": ["direction", "dtype", "arch"],
        "performance": [{"name": "tile_m"}],
        "ignored": ["tile_n", "tile_k"],
    }
    manifest["enumerate"].update(enumerate_keys)
    return manifest


def _tile_ms(manifest) -> list:
    return [k["kernel_source"]["spec"]["tile_m"] for k in _kernels(manifest)]


def test_a_performance_field_varies_unless_the_section_holds_it(tmp_path, monkeypatch):
    manifest = _conv_hold(tmp_path, monkeypatch)
    assert _tile_ms(manifest) == [64, 128, 64, 128, 64, 128, 64, 128]
    manifest["enumerate"]["hold_fields"] = ["tile_m"]
    assert _tile_ms(manifest) == [64, 64, 64, 64]


def test_a_section_may_override_a_performance_field_it_also_holds(
    tmp_path, monkeypatch
):
    """The override moves the section's default spec; holding then keeps every spec of the
    section there, so the performance field is one value per section."""
    manifest = _conv_hold(
        tmp_path, monkeypatch, hold_fields=["tile_m"], spec_overrides={"tile_m": 128}
    )
    assert _tile_ms(manifest) == [128, 128, 128, 128]


def test_an_also_section_inherits_hold_fields_and_pins_its_own_value(
    tmp_path, monkeypatch
):
    """The production-stack pattern: the primary holds the performance field at its
    default and an `also` section, inheriting the hold, overrides it to another value, so
    every cell ships one kernel per value."""
    manifest = _conv_hold(tmp_path, monkeypatch, hold_fields=["tile_m"])
    section = copy.deepcopy(manifest["enumerate"])
    section["spec_overrides"] = {"tile_m": 128}
    del section["hold_fields"]
    manifest["enumerate"]["also"] = [section]
    assert _tile_ms(manifest) == [64] * 4 + [128] * 4


def test_one_rule_admits_a_pinned_performance_field_and_a_feature_axis(
    tmp_path, monkeypatch
):
    """The two things a section may pin in one manifest: `tile_m` (performance) in one
    `also` section, and `tile_n` (feature) in another that no other section pins alike. A
    feature field every section leaves alone, or pins alike, is still refused (see
    test_an_override_every_section_pins_alike_is_no_feature_axis)."""
    manifest = _conv_hold(tmp_path, monkeypatch, hold_fields=["tile_m"])
    manifest["reduction"]["fields"] = {
        "feature": ["direction", "dtype", "arch", "tile_n"],
        "performance": [{"name": "tile_m"}],
        "ignored": ["tile_k"],
    }
    pinned_performance = copy.deepcopy(manifest["enumerate"])
    pinned_performance["spec_overrides"] = {"tile_m": 128}
    pinned_feature = copy.deepcopy(manifest["enumerate"])
    pinned_feature["spec_overrides"] = {"tile_n": 128}
    manifest["enumerate"]["also"] = [pinned_performance, pinned_feature]
    kernels = _kernels(manifest)
    tiles = [
        (k["kernel_source"]["spec"]["tile_m"], k["kernel_source"]["spec"]["tile_n"])
        for k in kernels
    ]
    assert tiles == [(64, 64)] * 4 + [(128, 64)] * 4 + [(64, 128)] * 4
    # The pinned performance field changes how a kernel runs, not what it serves, so the
    # cell's one floor covers it; the feature override serves other graphs, so it has its own.
    assert [bool(k.get("default")) for k in kernels] == (
        [True] * 4 + [False] * 4 + [True] * 4
    )


def test_hold_fields_naming_a_field_the_spec_lacks_is_exit_2(
    tmp_path, monkeypatch, capsys
):
    manifest = _conv_hold(tmp_path, monkeypatch, hold_fields=["no_such"])
    path = synth.write_manifest(tmp_path / "m.engine.yaml", manifest)
    assert re_.main(["--manifest", str(path), "--plan-space"]) == 2
    err = capsys.readouterr().err
    assert "hold_fields" in err and "no_such" in err, err


def test_registry_refuses_hold_fields_the_spec_lacks_when_it_reads_them(
    tmp_path, monkeypatch
):
    """The names are checked as the section is read, not only once a full-space
    enumeration reaches `select_full`, and in every `also` section."""
    manifest = _conv_hold(tmp_path, monkeypatch, hold_fields=["no_such"])
    with pytest.raises(re_.EnumerationError, match=r"hold_fields.*no_such"):
        re_.Registry(manifest)

    manifest = _conv_hold(tmp_path / "also", monkeypatch, hold_fields=["tile_m"])
    section = copy.deepcopy(manifest["enumerate"])
    section["hold_fields"] = ["no_such"]
    manifest["enumerate"]["also"] = [section]
    with pytest.raises(re_.EnumerationError, match=r"hold_fields.*no_such"):
        re_.Registry(manifest)


def test_hold_fields_on_a_slice_section_is_refused(probe, capsys):
    """A slice section projects no performance field, so `hold_fields` there could hold
    nothing: refused by name rather than ignored, as only `select_full` read it."""
    probe.manifest["enumerate"]["hold_fields"] = ["tile_n"]
    with pytest.raises(re_.EnumerationError, match=r"hold_fields.*'slice'"):
        re_.Registry(probe.manifest)

    assert _run_probe(probe) == 2
    err = capsys.readouterr().err
    assert "hold_fields" in err and "slice" in err, err


def test_an_also_slice_section_must_not_inherit_hold_fields(tmp_path, monkeypatch):
    """`hold_fields` is inherited by an `also` section; a slice one that inherits it is
    refused too, and `hold_fields: []` opts that section out."""
    manifest = _conv_hold(tmp_path, monkeypatch, hold_fields=["tile_m"])
    section = copy.deepcopy(manifest["enumerate"])
    section["space"] = "slice"
    del section["hold_fields"]
    manifest["enumerate"]["also"] = [section]
    with pytest.raises(
        re_.EnumerationError, match=r"an enumerate.also section sets hold_fields"
    ):
        re_.Registry(manifest)

    section["hold_fields"] = []
    re_.Registry(manifest)
