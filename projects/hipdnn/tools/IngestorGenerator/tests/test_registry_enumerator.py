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
