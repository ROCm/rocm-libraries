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
