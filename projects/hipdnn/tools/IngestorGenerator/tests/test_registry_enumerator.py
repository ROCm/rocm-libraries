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


_LAUNCHES = {"segment": {}, "reduce": {"role": "reduce"}}


def test_manifest_launches_reach_every_generated_kernel_source(probe):
    """A split-KV engine (segment + reduce): every generated kernel is a
    multi-launch descriptor, as the generator's loader reads it."""
    from codegen.config_loader import load_config

    probe.manifest["launches"] = copy.deepcopy(_LAUNCHES)
    assert _run_probe(probe) == 0
    text = probe.out.read_text()
    # One shared object would dump as a YAML anchor and aliases, not per-kernel copies.
    assert "&id" not in text and "*id" not in text
    raw = yaml.safe_load(text)["packs"][0]["kernels"]
    assert all(k["kernel_source"]["launches"] == _LAUNCHES for k in raw)
    kernels = [k for p in load_config(probe.out).packs for k in p.kernels]
    assert len(kernels) == len(raw) > 0
    for kernel in kernels:
        launches = kernel.kernel_source.launches
        assert list(launches) == ["segment", "reduce"]
        assert launches["reduce"].role == "reduce"
        assert launches["segment"].builder == kernel.kernel_source.builder


def test_each_generated_kernel_owns_its_launches(probe):
    probe.manifest["launches"] = copy.deepcopy(_LAUNCHES)
    config = re_.build_config(probe.manifest, warn=lambda message: None)
    sources = [k["kernel_source"]["launches"] for k in config["packs"][0]["kernels"]]
    owners = [probe.manifest["launches"], *sources]
    assert len({id(s) for s in owners}) == len(owners)
    assert len({id(s["reduce"]) for s in owners}) == len(owners)


def _run_both_ways(tmp: Path, manifest: dict, capsys, *argv) -> list:
    """Run the CLI on ``manifest`` without, then with, `launches`; return each
    run's exit code, stdout and generated config (parsed, or None)."""
    runs = []
    for launches in (None, _LAUNCHES):
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
        assert kernel["kernel_source"].pop("launches") == _LAUNCHES
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


def test_manifest_launches_must_be_a_mapping(probe, capsys):
    probe.manifest["launches"] = ["segment", "reduce"]
    assert _run_probe(probe) == 2
    assert "launches" in capsys.readouterr().err
    assert not probe.out.exists()


@pytest.mark.parametrize(
    "launches",
    [{"segment": {}}, {"segment": {}, "Reduce": {"role": "reduce"}}],
    ids=["single-launch", "bad-name"],
)
def test_bad_manifest_launches_are_refused_when_the_config_loads(probe, launches):
    """The enumerator only carries `launches`; the loader owns the one rule for
    their count and names, and refuses the generated config."""
    from codegen.config_loader import ConfigError, load_config

    probe.manifest["launches"] = launches
    assert _run_probe(probe) == 0
    with pytest.raises(ConfigError, match="launches"):
        load_config(probe.out)


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


def test_plan_space_counts_after_merging(split, tmp_path, capsys):
    path = synth.write_manifest(tmp_path / "m.engine.yaml", _dedup(split))
    assert re_.main(["--manifest", str(path), "--plan-space"]) == 0
    out = capsys.readouterr().out
    assert "cells: 2 (12 request combos merged on selector_fields" in out
    assert "instances (current classification): 6 " in out
    assert "num_seqs: {" not in out


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
    ],
    ids=[
        "selector-not-a-list",
        "neutral-not-names",
        "selector-unknown",
        "selector-not-a-cell-field",
        "neutral-unknown",
        "selector-is-feature",
        "neutral-is-performance",
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
