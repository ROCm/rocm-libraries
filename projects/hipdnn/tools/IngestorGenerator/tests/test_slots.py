# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""RFC 0021 kernel slots: the ``slots`` config section and the slotted descriptors it emits.

The unslotted guarantee is the other half: a config without ``slots`` must render the
bytes it rendered before slots existed.
"""

import copy
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from codegen import generator as generator_module
from codegen.config_loader import ConfigError, load_config
from codegen.generator import IngestorGenerator
from tests.helpers import make_engine, make_minimal_config

FIXTURES = Path(__file__).parent / "fixtures"
SLOTTED = FIXTURES / "slotted_three_slot.yaml"
GOLDEN = FIXTURES / "unslotted_render_golden.json"
UNSLOTTED_REFERENCE_CONFIGS = [
    "scale_add.yaml",
    "binary_ops.yaml",
    "gfx950_attention_dense.yaml",
    "gfx1151_wmma_attention.yaml",
]


def _render_hashes(generator, config, tmp_path, monkeypatch):
    """Render with sequential UUIDs, so two renders of one config are comparable."""
    counter = iter(range(1, 10**6))
    monkeypatch.setattr(
        generator_module.uuid, "uuid4", lambda: uuid.UUID(int=next(counter))
    )
    written = generator.render(config, tmp_path)
    return {
        rel: hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()
        for rel in sorted(written)
    }


def test_unslotted_output_is_byte_identical_to_the_pre_slot_generator(
    generator, config_path, tmp_path, monkeypatch
):
    """The fixture holds the SHA-256 of every file the generator wrote for each
    reference config before slots existed (UUIDs pinned), C++ stubs included."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    configs = {
        name: load_config(config_path(name)) for name in UNSLOTTED_REFERENCE_CONFIGS
    }
    configs["heuristic_free"] = make_minimal_config(
        engine=make_engine(heuristic="none")
    )
    assert set(configs) == set(golden)
    for name, config in configs.items():
        out = tmp_path / name
        out.mkdir()
        assert _render_hashes(generator, config, out, monkeypatch) == golden[name], name


# --- the slotted config and its descriptors ------------------------------------------


def _raw():
    return yaml.safe_load(SLOTTED.read_text(encoding="utf-8"))


def _load(tmp_path, mutate=None):
    raw = _raw()
    if mutate is not None:
        mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


@pytest.fixture
def bundle(generator, tmp_path):
    """The rendered three-slot bundle: (config, descriptor dir, documents by file stem)."""
    config = load_config(SLOTTED)
    generator.render(config, tmp_path)
    ddir = tmp_path / config.descriptor_dir
    documents = {
        path.name[: -len(".json")]: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(ddir.glob("*.json"))
    }
    return config, ddir, documents


def test_a_three_slot_config_emits_a_valid_1_1_ued(bundle):
    _, _, docs = bundle
    ued = docs["split_bwd.ued"]
    assert ued["version"] == "1.1"
    # RFC 0021 §4: the selection members live on the slots, never beside them.
    assert not {
        "metadata",
        "sort_kernel_catalog",
        "knobs",
        "predict_applicable_kernels",
    } & set(ued)
    assert ued["shared_workspace"] == "ws.bytes"
    assert ued["graph_match"] == {"native": "hipkernel.split_bwd.graph_match"}
    assert [slot["name"] for slot in ued["slots"]] == ["prep", "dkdv", "dq"]
    by_name = {slot["name"]: slot for slot in ued["slots"]}
    assert (
        "knobs" not in by_name["prep"] and "sort_kernel_catalog" not in by_name["prep"]
    )
    assert by_name["dkdv"]["knobs"] == ["block_m", "num_warps"]
    assert by_name["dq"]["knobs"] == ["block_m"]


def test_each_slot_has_its_own_kmd_and_a_metric_less_ranker(bundle):
    config, _, docs = bundle
    ued = docs["split_bwd.ued"]
    kmd_ids = {slot["name"]: slot["metadata"] for slot in ued["slots"]}
    assert len(set(kmd_ids.values())) == 3
    for slot in config.slots:
        kmd = docs[f"split_bwd_{slot.name}.kmd"]
        assert kmd["id"] == kmd_ids[slot.name]
        assert [f["name"] for f in kmd["fields"]] == [f.name for f in slot.kmd_fields]
    uhd_ids = {
        slot["name"]: slot["sort_kernel_catalog"]["default"]
        for slot in ued["slots"]
        if "sort_kernel_catalog" in slot
    }
    assert set(uhd_ids) == {"dkdv", "dq"}
    for name, identity in uhd_ids.items():
        uhd = docs[f"split_bwd_{name}.uhd"]
        assert uhd["id"] == identity
        # Metric-less: a slot has no time of its own (RFC 0021 §7).
        assert "score" not in uhd
        assert uhd["native"] == {"symbol": f"hipkernel.split_bwd.{name}_score"}
    assert "split_bwd_prep.uhd" not in docs


def test_every_kdp_is_1_1_and_names_the_slot_whose_kmd_its_kernels_fill(bundle):
    config, _, docs = bundle
    ued = docs["split_bwd.ued"]
    kmd_fields = {
        slot["name"]: {
            f["name"] for f in docs[f"split_bwd_{slot['name']}.kmd"]["fields"]
        }
        for slot in ued["slots"]
    }
    kdps = {name: doc for name, doc in docs.items() if name.endswith(".kdp")}
    assert len(kdps) == len(config.packs) == 4
    assert {doc["slot"] for doc in kdps.values()} == {"prep", "dkdv", "dq"}
    for doc in kdps.values():
        assert doc["version"] == "1.1"
        assert doc["engine"] == ued["id"]
        for kernel in doc["kernelDescriptors"]:
            # The UKD is unchanged by slots, and its metadata is the slot KMD's.
            assert kernel["version"] == "1.0"
            assert set(kernel["metadata"]) <= kmd_fields[doc["slot"]]
        contract = doc["provenance"]["specialization_contract"]["consumers"][0]
        slot_kmd = next(s["metadata"] for s in ued["slots"] if s["name"] == doc["slot"])
        assert contract["kmd_id"] == slot_kmd
        assert set(contract["matcher_only_fields"]) == kmd_fields[doc["slot"]]


def test_only_the_ued_and_kdps_move_to_1_1(bundle):
    _, _, docs = bundle
    for name, doc in docs.items():
        expected = "1.1" if name.endswith((".ued", ".kdp")) else "1.0"
        assert doc["version"] == expected, name


def test_operation_matchers_are_per_slot_not_per_engine(bundle):
    _, _, docs = bundle
    kdps = {
        doc["slot"] if doc["slot"] != "dq" else doc["name"]: doc
        for n, doc in docs.items()
        if n.endswith(".kdp")
    }
    kernel_match = docs["kernel_dtype_matches_graph.umd"]["id"]
    # prep and dkdv have one pack each: nothing to narrow.
    assert kdps["prep"]["matchers"] == [kernel_match]
    assert kdps["dkdv"]["matchers"] == [kernel_match]
    # dq has two: each is narrowed by its own operation matcher.
    for pack_name, discriminator in (("dq_causal", "causal"), ("dq_full", "full")):
        doc = kdps[f"hipkernel:split_bwd_{pack_name}"]
        umd = docs[f"operation_is_{discriminator}.umd"]
        assert doc["matchers"] == [umd["id"], kernel_match]
        assert umd["scope"] == "graph"


def test_the_emitted_file_list_matches_the_preview(generator, tmp_path):
    config = load_config(SLOTTED)
    assert generator.render(config, tmp_path) == generator.preview_files(config)


def test_one_tuple_may_repeat_across_slots_but_not_within_one(generator, tmp_path):
    """RFC 0021 §4: metadata-tuple uniqueness is per slot."""

    def two_slots(raw):
        raw["slots"] = [
            {
                "name": name,
                "kmd_fields": [{"name": "block_m", "type": "int"}],
                "specialization": {
                    "metadata_fields": [],
                    "matcher_only_fields": ["block_m"],
                    "bindings": {},
                    "vocabulary": {},
                },
            }
            for name in ("a", "b")
        ]
        raw["packs"] = [
            {
                "name": f"pack_{name}",
                "slot": name,
                "arch": ["gfx950"],
                "kernels": [
                    {
                        "name": f"{name}.k",
                        "kernel_source": {
                            "kind": "embedded_source",
                            "source_file": f"{name}.cpp",
                            "entry_point": name,
                        },
                        "metadata": {"block_m": 64},
                    }
                ],
            }
            for name in ("a", "b")
        ]

    config = _load(tmp_path, two_slots)
    generator.render(config, tmp_path / "ok")

    def clash(raw):
        two_slots(raw)
        # A second, different binary with the same tuple in slot a.
        raw["packs"][0]["kernels"].append(
            {
                "name": "a.k2",
                "kernel_source": {
                    "kind": "embedded_source",
                    "source_file": "other.cpp",
                    "entry_point": "other",
                },
                "metadata": {"block_m": 64},
            }
        )

    with pytest.raises(ValueError, match="complete to the SAME catalog tuple"):
        generator.render(_load(tmp_path, clash), tmp_path / "clash")


# --- what the tooling downstream makes of the output ----------------------------------


def test_uhd_gen_resolves_the_generated_bundle_per_slot(bundle):
    """The generator's output feeds `uhd_gen`'s provenance: each slot snapshots its own
    KMD and packs, and the engine-level snapshot names every slot."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from uhd_gen.provenance import compare_provenance, snapshot_provenance

    config, ddir, docs = bundle
    ued = docs["split_bwd.ued"]
    for slot in ued["slots"]:
        snapshot = snapshot_provenance(
            ddir, "hipkernel:SplitBwd", "gfx950", slot["name"]
        )
        assert (
            snapshot["kmd"]["id"] == slot["metadata"]
            and snapshot["slot"] == slot["name"]
        )
        compare_provenance(snapshot, snapshot)
    # dq's matchers are dq's packs' only: the causal/full operation UMDs, not dkdv's.
    dkdv = snapshot_provenance(ddir, None, "gfx950", "dkdv")
    dq = snapshot_provenance(ddir, None, "gfx950", "dq")
    assert len(dkdv["umd"]) == 1 and len(dq["umd"]) == 3
    engine = snapshot_provenance(ddir, None, "gfx950")
    assert [entry["slot"] for entry in engine["kmds"]] == ["prep", "dkdv", "dq"]


# --- refusals --------------------------------------------------------------------------


def _expect(tmp_path, mutate, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, mutate)


def test_selection_members_may_not_sit_beside_slots(tmp_path):
    _expect(
        tmp_path,
        lambda raw: raw.update(kmd_fields=[{"name": "x", "type": "int"}]),
        r"key 'kmd_fields' must not appear beside 'slots'",
    )
    _expect(
        tmp_path,
        lambda raw: raw["engine"].update(knobs=["block_m"]),
        r"key 'engine.knobs' must not appear beside 'slots'",
    )
    _expect(
        tmp_path,
        lambda raw: raw.update(specialization={"metadata_fields": []}),
        r"key 'specialization' must not appear beside 'slots'",
    )
    _expect(
        tmp_path,
        lambda raw: raw["engine"].update(heuristic="native"),
        r"engine.heuristic 'native' must not appear beside 'slots'",
    )


def test_an_empty_slot_list_is_refused(tmp_path):
    _expect(
        tmp_path,
        lambda raw: raw.update(slots=[]),
        r"'slots' must declare at least one slot",
    )


@pytest.mark.parametrize(
    "name,message",
    [
        ("Prep", r"slot name 'Prep' in slots must match \^\[a-z\]\[a-z0-9_\]\*\$"),
        ("9p", r"slot name '9p' in slots must match"),
        ("global", r"slot name 'global' in slots is reserved for engine-wide knobs"),
    ],
)
def test_a_slot_name_follows_the_loaders_rule(tmp_path, name, message):
    def rename(raw):
        raw["slots"][0]["name"] = name
        raw["packs"][0]["slot"] = name

    _expect(tmp_path, rename, message)


def test_slot_names_are_unique(tmp_path):
    def duplicate(raw):
        raw["slots"][2]["name"] = "dkdv"

    _expect(tmp_path, duplicate, r"slot name 'dkdv' is declared more than once")


def test_every_pack_names_a_declared_slot(tmp_path):
    _expect(
        tmp_path,
        lambda raw: raw["packs"][0].pop("slot"),
        r"pack 'prep' declares no 'slot', which every pack of a slotted engine must",
    )
    _expect(
        tmp_path,
        lambda raw: raw["packs"][0].update(slot="nope"),
        r"pack 'prep' names slot 'nope', which this config does not declare",
    )


def test_an_unslotted_config_refuses_a_pack_slot_and_a_shared_workspace(
    config_path, tmp_path
):
    raw = yaml.safe_load(config_path("scale_add.yaml").read_text(encoding="utf-8"))
    pinned = copy.deepcopy(raw)
    pinned["packs"][0]["slot"] = "dq"
    path = tmp_path / "pinned.yaml"
    path.write_text(yaml.safe_dump(pinned), encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match=r"pack 'scale_add' declares slot 'dq', but this config declares no 'slots'",
    ):
        load_config(path)
    shared = copy.deepcopy(raw)
    shared["engine"]["shared_workspace"] = "ws.bytes"
    path.write_text(yaml.safe_dump(shared), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"key 'shared_workspace' requires 'slots'"):
        load_config(path)


@pytest.mark.parametrize("token", ["bytes", "ws.", 3])
def test_the_shared_workspace_token_lives_under_the_ws_root(tmp_path, token):
    _expect(
        tmp_path,
        lambda raw: raw["engine"].update(shared_workspace=token),
        r"engine.shared_workspace .* must name a bound token under the 'ws.' root",
    )


def test_a_slot_with_no_pack_is_refused(tmp_path):
    def drop_prep_pack(raw):
        raw["packs"] = [p for p in raw["packs"] if p["slot"] != "prep"]

    _expect(tmp_path, drop_prep_pack, r"slot 'prep' declares no packs")


def test_knobs_are_checked_against_their_own_slots_int_fields(tmp_path):
    # `num_warps` is dkdv's field, not dq's.
    _expect(
        tmp_path,
        lambda raw: raw["slots"][2].update(knobs=["block_m", "num_warps"]),
        r"slots\[dq\].knobs names 'num_warps', which no kmd_fields entry declares",
    )

    def string_knob(raw):
        raw["slots"][1]["knobs"] = ["dtype"]

    _expect(
        tmp_path,
        string_knob,
        r"slots\[dkdv\].knobs names 'dtype', declared in kmd_fields with type 'string'",
    )
    _expect(
        tmp_path,
        lambda raw: raw["slots"][1].update(knobs=["block_m", "block_m"]),
        r"slots\[dkdv\].knobs lists \['block_m'\] more than once",
    )


def test_kernel_metadata_is_checked_against_its_slots_kmd(tmp_path):
    def foreign_field(raw):
        # num_warps belongs to dkdv's KMD, not dq's.
        raw["packs"][2]["kernels"][0]["metadata"]["num_warps"] = 4

    _expect(
        tmp_path,
        foreign_field,
        r"slot 'dq': pack 'dq_causal' kernel 'bwd.dq_causal_m64': metadata names 'num_warps'",
    )


def test_each_slot_declares_its_own_specialization(tmp_path):
    def partial(raw):
        raw["slots"][1]["specialization"]["matcher_only_fields"] = ["block_m", "dtype"]

    _expect(
        tmp_path,
        partial,
        r"slot 'dkdv': 'specialization' must partition the declared kmd_fields",
    )


def test_a_multi_pack_slot_needs_discriminators_and_a_lone_pack_must_not_have_one(
    tmp_path,
):
    _expect(
        tmp_path,
        lambda raw: raw["packs"][3].pop("discriminator"),
        r"declare no 'discriminator', but slot 'dq' has 2 packs",
    )
    _expect(
        tmp_path,
        lambda raw: raw["packs"][1].update(discriminator="x"),
        r"declares a discriminator, but slot 'dkdv' has only one pack",
    )
    _expect(
        tmp_path,
        lambda raw: raw["packs"][3].update(discriminator="causal"),
        r"packs declare duplicate discriminators",
    )


def test_a_slot_ranker_flag_is_a_boolean(tmp_path):
    _expect(
        tmp_path,
        lambda raw: raw["slots"][1].update(ranker="yes"),
        r"slots\[dkdv\].ranker must be true or false",
    )


@pytest.mark.parametrize("slot", [["prep"], {"name": "prep"}])
def test_a_pack_slot_that_is_not_a_name_is_a_config_error(tmp_path, slot):
    # A YAML list or mapping is unhashable: it must not escape as a TypeError.
    _expect(
        tmp_path,
        lambda raw: raw["packs"][0].update(slot=slot),
        r"pack 'prep' declares slot .*, which must be a slot name",
    )


@pytest.mark.parametrize(
    "mutate,scope",
    [
        (lambda raw: raw["slots"][1].update(knob=["block_m"]), r"slot 'dkdv'"),
        (
            lambda raw: raw["slots"][1]["kmd_fields"][0].update(typo="x"),
            r"slot 'dkdv' kmd_fields entry 'block_m'",
        ),
    ],
)
def test_an_unknown_key_on_a_slot_or_its_kmd_field_is_refused(tmp_path, mutate, scope):
    _expect(tmp_path, mutate, scope + r" declares \['(knob|typo)'\]")


@pytest.mark.parametrize("key", ["optional", "default"])
def test_a_deprecated_kmd_field_key_on_a_slot_is_refused(tmp_path, key):
    _expect(
        tmp_path,
        lambda raw: raw["slots"][1]["kmd_fields"][0].update({key: True}),
        rf"kmd_fields\[\]\.{key} is not a real key.*Affected entries: block_m",
    )


# --- sdk_version and behavior_notes reach the slotted UED -------------------------------


def non_default_engine(raw):
    """The slotted fixture with the two engine members it leaves at their defaults."""
    raw["engine"]["sdk_version"] = "2.0.0"
    raw["engine"]["behavior_notes"] = ["runtime_compilation"]


def test_a_slotted_ued_carries_the_engines_sdk_version_and_behavior_notes(
    generator, tmp_path
):
    config = _load(tmp_path, non_default_engine)
    generator.render(config, tmp_path)
    ued = json.loads(
        (tmp_path / config.descriptor_dir / "split_bwd.ued.json").read_text("utf-8")
    )
    assert ued["sdk_version"] == "2.0.0"
    assert ued["behavior_notes"] == ["runtime_compilation"]


# --- the command line over a slotted config ---------------------------------------------

_GENERATE = Path(__file__).resolve().parents[1] / "generate.py"


def _generate_cli(*args):
    return subprocess.run(
        [sys.executable, str(_GENERATE), "--config", str(SLOTTED), *args],
        capture_output=True,
        text=True,
    )


def test_the_cli_says_a_slotted_engine_is_descriptors_only(tmp_path):
    result = _generate_cli("--output-dir", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    assert "Descriptors only: no provider splice for slotted engines" in result.stdout
    assert "Knobs:" not in result.stdout
    assert "splice points" not in result.stdout


def test_placeholder_checking_is_refused_for_a_slotted_engine(tmp_path):
    result = _generate_cli(
        "--output-dir", str(tmp_path / "out"), "--check-placeholders"
    )
    assert result.returncode == 2
    assert "not supported for slotted engines" in result.stderr
    assert "No unfilled placeholders" not in result.stdout
