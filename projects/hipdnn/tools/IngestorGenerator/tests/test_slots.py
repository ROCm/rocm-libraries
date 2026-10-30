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
from codegen.generator import IngestorGenerator, resolve_ids
from tests.helpers import make_engine, make_minimal_config, render_resolved

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
    """Render with sequential UUIDs, so two renders of one config are comparable. The
    result is the SHA-256 of the sorted ``<file>:<SHA-256 of its bytes>`` lines."""
    counter = iter(range(1, 10**6))
    monkeypatch.setattr(
        generator_module.uuid, "uuid4", lambda: uuid.UUID(int=next(counter))
    )
    written = generator.render(config, tmp_path)
    lines = [
        f"{rel}:{hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()}"
        for rel in sorted(written)
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_unslotted_output_is_byte_identical_to_the_pre_slot_generator(
    generator, config_path, tmp_path, monkeypatch
):
    """The fixture holds one digest per reference config of the SHA-256 of every file
    the generator wrote for it before slots existed (UUIDs pinned), C++ stubs included.
    """
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    configs = {
        name: load_config(config_path(name)) for name in UNSLOTTED_REFERENCE_CONFIGS
    }
    configs["heuristic_free"] = make_minimal_config(
        engine=make_engine(heuristic="none")
    )
    configs["verify_multi_launch"] = load_config(
        FIXTURES / "verify_multi_launch" / "config.yaml"
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
        assert doc["engine"] == ued["id"]
        for kernel in doc["kernelDescriptors"]:
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
    assert kdps["prep"]["matchers"] == [kernel_match]
    assert kdps["dkdv"]["matchers"] == [kernel_match]
    for pack_name, discriminator in (("dq_causal", "causal"), ("dq_full", "full")):
        doc = kdps[f"hipkernel:split_bwd_{pack_name}"]
        umd = docs[f"operation_is_{discriminator}.umd"]
        assert doc["matchers"] == [umd["id"], kernel_match]
        assert umd["scope"] == "graph"


def test_each_slot_dispatches_through_its_own_udd(bundle):
    """RFC 0021 §4: packs with different launch sets need different dispatch
    descriptors, so every slot has a UDD and its KDPs name that one."""
    config, _, docs = bundle
    udds = {slot.name: docs[f"split_bwd_{slot.name}.udd"] for slot in config.slots}
    assert len({udd["id"] for udd in udds.values()}) == 3
    assert "split_bwd.udd" not in docs
    for name, udd in udds.items():
        assert udd["dispatch_symbol"] == f"hipkernel.split_bwd.{name}_dispatch"
    for doc in (d for n, d in docs.items() if n.endswith(".kdp")):
        assert doc["dispatch"] == udds[doc["slot"]]["id"]


def _launch_set_config(tmp_path, slot_kernels, slot_fields=None):
    """A packaged slotted rocke engine: one pack per slot; `slot_kernels` maps a
    slot to the launch-name list of each of its kernels (empty: single launch).
    `slot_fields` maps a slot to one extra mandatory int KMD field `{slot: name}`,
    which that slot's kernels state as their index and no other slot declares."""
    slot_fields = slot_fields or {}

    def kernel(slot, index, launches):
        source = {"spec": {"head_size": 64 * (index + 1)}}
        if launches:
            source["launches"] = {name: {} for name in launches}
        return {
            "name": f"{slot}.k{index}",
            "kernel_source": source,
            "metadata": {
                "dtype": "FP16",
                "head_size": 64 * (index + 1),
                **({slot_fields[slot]: index} if slot in slot_fields else {}),
            },
        }

    raw = {
        "dialect": "packaged",
        "kernel_source_kind": "rocke",
        "engine": {"name": "hipkernel:SlotLaunches", "sdk_version": "1.0.0"},
        "slots": [
            {
                "name": slot,
                "kmd_fields": [
                    {"name": "dtype", "type": "string"},
                    {"name": "head_size", "type": "int"},
                    *(
                        [{"name": slot_fields[slot], "type": "int"}]
                        if slot in slot_fields
                        else []
                    ),
                ],
                "specialization": {
                    "metadata_fields": ["head_size"],
                    "matcher_only_fields": [
                        "dtype",
                        *([slot_fields[slot]] if slot in slot_fields else []),
                    ],
                    "bindings": {"head_size": {"field": "head_size"}},
                    "vocabulary": {},
                },
            }
            for slot in slot_kernels
        ],
        "packs": [
            {
                "name": slot,
                "slot": slot,
                "arch": ["gfx1151"],
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": "kernels/gfx1151/wmma_fmha_decode.py",
                    "builder": "build_wmma_fmha_decode_module",
                },
                "kernels": [kernel(slot, i, ls) for i, ls in enumerate(kernels)],
            }
            for slot, kernels in slot_kernels.items()
        ],
    }
    path = tmp_path / "launch_sets.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_slots_may_launch_different_sets_but_a_slot_may_not_mix(tmp_path):
    """RFC 0021 §4: the launch set is a slot's, as each slot has its own handler."""
    with pytest.raises(ConfigError, match=r"One slot \('dkdv'\) has one dispatch"):
        _launch_set_config(tmp_path, {"prep": [[]], "dkdv": [["decode", "merge"], []]})


def test_a_slotted_multi_launch_slot_emits_ukd_1_1_sources_and_its_own_udd(
    generator, tmp_path
):
    """Packaged slotted rocke `launches` become UKD 1.1 `sources`; a single-launch
    slot beside it stays a 1.0 `kernel_source`."""
    config = _launch_set_config(tmp_path, {"prep": [[]], "dkdv": [["decode", "merge"]]})
    generator.render(config, tmp_path / "out")
    ddir = tmp_path / "out" / config.descriptor_dir
    kdps = {
        json.loads(p.read_text())["slot"]: json.loads(p.read_text())
        for p in ddir.glob("*.kdp.json")
    }
    (prep,) = kdps["prep"]["kernelDescriptors"]
    (dkdv,) = kdps["dkdv"]["kernelDescriptors"]
    assert (
        prep["version"] == "1.0" and "kernel_source" in prep and "sources" not in prep
    )
    assert dkdv["version"] == "1.1" and "kernel_source" not in dkdv
    assert list(dkdv["sources"]) == ["decode", "merge"]
    assert kdps["prep"]["dispatch"] != kdps["dkdv"]["dispatch"]


def _hip_hsaco_launch_config(tmp_path):
    """A packaged slotted config: slot `dkdv` authors `hip` launches and slot `dq`
    authors `hsaco` launches."""
    sources = {
        "dkdv": {
            "kind": "hip",
            "source": "kernels/Pair.cpp",
            "entry": "KernelAdd",
            "launches": {"decode": {}, "merge": {"entry": "KernelScale"}},
        },
        "dq": {
            "kind": "hsaco",
            "file": "HsacoFixture.co",
            "symbol": "HsacoFixtureAdd",
            "launches": {"add": {}, "scale": {"symbol": "HsacoFixtureScale"}},
        },
    }
    raw = {
        "dialect": "packaged",
        "kernel_source_kind": "hsaco",
        "engine": {"name": "hipkernel:SlotHipHsaco", "sdk_version": "1.0.0"},
        "slots": [
            {
                "name": slot,
                "kmd_fields": [{"name": "dtype", "type": "string"}],
                "specialization": {
                    "metadata_fields": [],
                    "matcher_only_fields": ["dtype"],
                    "bindings": {},
                    "vocabulary": {},
                },
            }
            for slot in sources
        ],
        "packs": [
            {
                "name": slot,
                "slot": slot,
                "arch": ["gfx950"],
                "kernels": [
                    {
                        "name": f"{slot}.k0",
                        "kernel_source": source,
                        "metadata": {"dtype": "FP16"},
                    }
                ],
            }
            for slot, source in sources.items()
        ],
    }
    path = tmp_path / "hip_hsaco_launches.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_a_slot_authors_hip_and_hsaco_launches_as_ukd_1_1_sources(generator, tmp_path):
    """`launches` work in a slot for `hip` and `hsaco` as they do unslotted: the two
    slots name different launch sets, each agrees within its slot, and each emits a
    1.1 UKD `sources`."""
    config = _hip_hsaco_launch_config(tmp_path)
    generator.render(config, tmp_path / "out")
    ddir = tmp_path / "out" / config.descriptor_dir
    kernels = {}
    for path in ddir.glob("*.kdp.json"):
        kdp = json.loads(path.read_text())
        (kernels[kdp["slot"]],) = kdp["kernelDescriptors"]
    for ukd in kernels.values():
        assert ukd["version"] == "1.1" and "kernel_source" not in ukd
    assert kernels["dkdv"]["sources"] == {
        name: {"kind": "hip", "source": "kernels/Pair.cpp", "entry": entry, "build": {}}
        for name, entry in (("decode", "KernelAdd"), ("merge", "KernelScale"))
    }
    assert kernels["dq"]["sources"] == {
        "add": {
            "kind": "hsaco",
            "file": "HsacoFixture.co",
            "symbol": "HsacoFixtureAdd",
        },
        "scale": {
            "kind": "hsaco",
            "file": "HsacoFixture.co",
            "symbol": "HsacoFixtureScale",
        },
    }


def _gate_slotted(tmp_path, generator, capsys, edit=None, slot_fields=None):
    """Render a packaged slotted bundle (single-launch `prep`, multi-launch `dkdv`
    with two kernels), apply `edit(documents by slot)` to its KDPs and run the
    variant-set gate on it structurally."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import verify_variant_sets as gate_module

    config = _launch_set_config(
        tmp_path,
        {"prep": [[]], "dkdv": [["decode", "merge"], ["decode", "merge"]]},
        slot_fields,
    )
    generator.render(config, tmp_path / "out")
    root = tmp_path / "out" / config.descriptor_dir
    if edit is not None:
        paths = {json.loads(p.read_text())["slot"]: p for p in root.glob("*.kdp.json")}
        documents = {slot: json.loads(p.read_text()) for slot, p in paths.items()}
        edit(documents)
        for slot, path in paths.items():
            path.write_text(json.dumps(documents[slot]))
    code = gate_module.main(["--mode", "structural", "slotted", str(root)])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_the_variant_set_gate_completes_each_pack_against_its_slot(
    generator, tmp_path, capsys
):
    """RFC 0021 §4: tuple uniqueness is per slot. `prep.k0` and `dkdv.k0` carry
    the same metadata and are no collision."""
    code, output = _gate_slotted(tmp_path, generator, capsys)
    assert code == 0, output
    assert "descriptors=    3" in output


def test_the_variant_set_gate_uses_each_slots_own_kmd(generator, tmp_path, capsys):
    """`dkdv`'s KMD has a mandatory field `prep`'s lacks, so no single KMD completes
    every kernel: the gate must complete each pack against its slot's, and still
    name a collision between two `dkdv` kernels."""
    fields = {"dkdv": "block_m"}
    (tmp_path / "ok").mkdir()
    (tmp_path / "clash").mkdir()
    code, output = _gate_slotted(tmp_path / "ok", generator, capsys, slot_fields=fields)
    assert code == 0, output

    def collide(documents):
        first, second = documents["dkdv"]["kernelDescriptors"]
        second["metadata"] = dict(first["metadata"])

    code, output = _gate_slotted(
        tmp_path / "clash", generator, capsys, collide, slot_fields=fields
    )
    assert code == 1
    assert "dkdv.k0 and dkdv.k1 complete to one tuple" in output
    assert "does not complete" not in output


def test_the_packer_resolves_a_generated_slotted_multi_launch_bundle(
    generator, tmp_path
):
    """hkp_pack resolves each slot pack through its slot's KMD and the generated
    multi-launch descriptors expose every `sources` launch."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import verify_variant_sets as gate_module  # puts hkp_pack on sys.path

    from hkp_pack import descriptor_context
    from hkp_pack.sources import kernel_sources

    config = _launch_set_config(
        tmp_path, {"prep": [[]], "dkdv": [["decode", "merge"], ["decode", "merge"]]}
    )
    generator.render(config, tmp_path / "out")
    index = descriptor_context.Index(str(tmp_path / "out" / config.descriptor_dir))
    bundles = {b.kdp_doc["slot"]: b for b in descriptor_context.resolve_bundles(index)}
    assert set(bundles) == {"prep", "dkdv"}
    slot_kmd = {s["name"]: s["metadata"] for s in bundles["dkdv"].engine["slots"]}
    for slot, bundle in bundles.items():
        assert bundle.kmd["id"] == slot_kmd[slot]
    launches = [sorted(kernel_sources(entry.ukd)) for entry in bundles["dkdv"].entries]
    assert launches == [["decode", "merge"]] * 2
    assert [list(kernel_sources(e.ukd)) for e in bundles["prep"].entries] == [[None]]


def test_the_emitted_file_list_matches_the_preview(generator, tmp_path):
    config = load_config(SLOTTED)
    assert generator.render(config, tmp_path) == generator.preview_files(config)


def _two_slots(raw):
    """Two slots with one KMD shape, and one kernel of the same tuple in each."""
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


def test_one_tuple_may_repeat_across_slots_but_not_within_one(generator, tmp_path):
    """RFC 0021 §4: metadata-tuple uniqueness is per slot."""
    two_slots = _two_slots

    config = _load(tmp_path, two_slots)
    generator.render(config, tmp_path / "ok")

    def clash(raw):
        two_slots(raw)
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


# --- stable ids across regeneration (#14109 composed with slots) ------------------------


def _doc_ids(ddir):
    """``{file name: id}`` of every descriptor, and ``{kernel name: id}``."""
    files, kernels = {}, {}
    for path in sorted(ddir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        files[path.name] = doc["id"]
        for entry in doc.get("kernelDescriptors", []):
            kernels[entry["name"]] = entry["id"]
    return files, kernels


def _ddir_bytes(ddir):
    return {path.name: path.read_bytes() for path in sorted(ddir.glob("*.json"))}


def test_regenerating_a_slotted_bundle_keeps_every_id(generator, tmp_path):
    config = load_config(SLOTTED)
    render_resolved(generator, config, tmp_path)
    ddir = tmp_path / config.descriptor_dir
    first = _ddir_bytes(ddir)
    resolution, _ = render_resolved(generator, config, tmp_path)
    assert resolution.report.minted == 0
    assert resolution.report.reused["engine"] == 2 + sum(
        2 + bool(slot.ranker) for slot in config.slots
    )
    assert _ddir_bytes(ddir) == first


def test_a_tuple_shared_by_two_slots_keeps_each_kernels_id(generator, tmp_path):
    """Kernels are matched inside their slot: a key that ignored the slot would
    see two prior kernels of one tuple, call the match ambiguous, and mint both
    again."""
    config = _load(tmp_path, _two_slots)
    render_resolved(generator, config, tmp_path)
    _, before = _doc_ids(tmp_path / config.descriptor_dir)
    assert before["a.k"] != before["b.k"]
    resolution, _ = render_resolved(generator, config, tmp_path)
    assert resolution.report.minted == 0
    assert _doc_ids(tmp_path / config.descriptor_dir)[1] == before


def test_a_slots_ids_follow_its_name_and_only_a_new_slot_is_minted(generator, tmp_path):
    config = load_config(SLOTTED)
    render_resolved(generator, config, tmp_path)
    ddir = tmp_path / config.descriptor_dir
    old_files, old_kernels = _doc_ids(ddir)

    def reorder_and_add(raw):
        raw["slots"].reverse()
        raw["slots"].append(copy.deepcopy(raw["slots"][-1]) | {"name": "extra"})
        raw["packs"].append(
            {
                "name": "extra",
                "slot": "extra",
                "arch": ["gfx950"],
                "kernels": [
                    {
                        "name": "bwd.extra",
                        "kernel_source": {
                            "kind": "embedded_source",
                            "source_file": "Extra.cpp",
                            "entry_point": "Extra",
                        },
                        "metadata": {"dtype": "BFLOAT16"},
                    }
                ],
            }
        )

    bigger = _load(tmp_path, reorder_and_add)
    render_resolved(generator, bigger, tmp_path)
    new_files, new_kernels = _doc_ids(ddir)
    for name in ("prep", "dkdv", "dq"):
        for kind in ("kmd", "uhd"):
            file = f"split_bwd_{name}.{kind}.json"
            if file in old_files:
                assert new_files[file] == old_files[file], file
    assert old_kernels.items() <= new_kernels.items()
    assert old_files["split_bwd.ued.json"] == new_files["split_bwd.ued.json"]
    added = set(new_files) - set(old_files)
    assert "split_bwd_extra.kmd.json" in added
    assert new_files["split_bwd_extra.kmd.json"] not in old_files.values()
    assert new_kernels["bwd.extra"] not in old_kernels.values()


def test_a_retuned_slot_kernel_gets_a_new_id_and_its_siblings_keep_theirs(
    generator, tmp_path
):
    config = load_config(SLOTTED)
    render_resolved(generator, config, tmp_path)
    ddir = tmp_path / config.descriptor_dir
    old_files, old_kernels = _doc_ids(ddir)

    def retune(raw):
        kernel = raw["packs"][1]["kernels"][0]  # dkdv's bwd.dkdv_m64
        assert kernel["name"] == "bwd.dkdv_m64"
        kernel["kernel_source"]["entry_point"] = "DkdvRetuned"

    render_resolved(generator, _load(tmp_path, retune), tmp_path)
    new_files, new_kernels = _doc_ids(ddir)
    assert new_kernels["bwd.dkdv_m64"] != old_kernels["bwd.dkdv_m64"]
    assert {k: v for k, v in new_kernels.items() if k != "bwd.dkdv_m64"} == {
        k: v for k, v in old_kernels.items() if k != "bwd.dkdv_m64"
    }
    assert new_files == old_files


def test_sibling_ids_are_refused_for_a_slotted_engine(generator, tmp_path):
    """Satellite bundles are single-pack; a slotted engine's slots live in one
    bundle, so naming a sibling must say so rather than misread the tree."""
    config = load_config(SLOTTED)
    render_resolved(generator, config, tmp_path)
    with pytest.raises(ValueError, match="not supported for a slotted engine"):
        resolve_ids(
            config,
            tmp_path / "other",
            engine_ids_from=[tmp_path / config.descriptor_dir],
        )


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


def _rename_prep(raw, name):
    raw["slots"][0]["name"] = name
    raw["packs"][0]["slot"] = name


def _drop_prep_pack(raw):
    raw["packs"] = [p for p in raw["packs"] if p["slot"] != "prep"]


# Each row mutates the three-slot config and names the config error that refuses it; the
# message fragment is the loader's wording for the rule.
REFUSALS = [
    (
        "kmd_fields_beside_slots",
        lambda raw: raw.update(kmd_fields=[{"name": "x", "type": "int"}]),
        r"key 'kmd_fields' must not appear beside 'slots'",
    ),
    (
        "engine_knobs_beside_slots",
        lambda raw: raw["engine"].update(knobs=["block_m"]),
        r"key 'engine.knobs' must not appear beside 'slots'",
    ),
    (
        "specialization_beside_slots",
        lambda raw: raw.update(specialization={"metadata_fields": []}),
        r"key 'specialization' must not appear beside 'slots'",
    ),
    (
        "heuristic_beside_slots",
        lambda raw: raw["engine"].update(heuristic="native"),
        r"engine.heuristic 'native' must not appear beside 'slots'",
    ),
    (
        "empty_slot_list",
        lambda raw: raw.update(slots=[]),
        r"'slots' must declare at least one slot",
    ),
    (
        "name_uppercase",
        lambda raw: _rename_prep(raw, "Prep"),
        r"slot name 'Prep' in slots must match \^\[a-z\]\[a-z0-9_\]\*\$",
    ),
    (
        "name_digit_start",
        lambda raw: _rename_prep(raw, "9p"),
        r"slot name '9p' in slots must match",
    ),
    (
        "name_global",
        lambda raw: _rename_prep(raw, "global"),
        r"slot name 'global' in slots is reserved for engine-wide knobs",
    ),
    (
        "duplicate_name",
        lambda raw: raw["slots"][2].update(name="dkdv"),
        r"slot name 'dkdv' is declared more than once",
    ),
    (
        "pack_without_slot",
        lambda raw: raw["packs"][0].pop("slot"),
        r"pack 'prep' declares no 'slot', which every pack of a slotted engine must",
    ),
    (
        "pack_names_undeclared_slot",
        lambda raw: raw["packs"][0].update(slot="nope"),
        r"pack 'prep' names slot 'nope', which this config does not declare",
    ),
    (
        "ws_token_bare",
        lambda raw: raw["engine"].update(shared_workspace="bytes"),
        r"engine.shared_workspace .* must name a bound token under the 'ws.' root",
    ),
    (
        "ws_token_root_only",
        lambda raw: raw["engine"].update(shared_workspace="ws."),
        r"engine.shared_workspace .* must name a bound token under the 'ws.' root",
    ),
    (
        "ws_token_not_a_string",
        lambda raw: raw["engine"].update(shared_workspace=3),
        r"engine.shared_workspace .* must name a bound token under the 'ws.' root",
    ),
    (
        "slot_without_pack",
        _drop_prep_pack,
        r"slot 'prep' declares no packs",
    ),
    (
        # `num_warps` is dkdv's field, not dq's.
        "knob_not_declared_in_slot_kmd",
        lambda raw: raw["slots"][2].update(knobs=["block_m", "num_warps"]),
        r"slots\[dq\].knobs names 'num_warps', which no kmd_fields entry declares",
    ),
    (
        "string_knob",
        lambda raw: raw["slots"][1].update(knobs=["dtype"]),
        r"slots\[dkdv\].knobs names 'dtype', declared in kmd_fields with type 'string'",
    ),
    (
        "duplicate_knob",
        lambda raw: raw["slots"][1].update(knobs=["block_m", "block_m"]),
        r"slots\[dkdv\].knobs lists \['block_m'\] more than once",
    ),
    (
        "kernel_metadata_outside_slot_kmd",
        lambda raw: raw["packs"][2]["kernels"][0]["metadata"].update(num_warps=4),
        r"slot 'dq': pack 'dq_causal' kernel 'bwd.dq_causal_m64': metadata names 'num_warps'",
    ),
    (
        "specialization_not_a_partition",
        lambda raw: raw["slots"][1]["specialization"].update(
            matcher_only_fields=["block_m", "dtype"]
        ),
        r"slot 'dkdv': 'specialization' must partition the declared kmd_fields",
    ),
    (
        "missing_discriminator",
        lambda raw: raw["packs"][3].pop("discriminator"),
        r"declare no 'discriminator', but slot 'dq' has 2 packs",
    ),
    (
        "lone_pack_discriminator",
        lambda raw: raw["packs"][1].update(discriminator="x"),
        r"declares a discriminator, but slot 'dkdv' has only one pack",
    ),
    (
        "duplicate_discriminator",
        lambda raw: raw["packs"][3].update(discriminator="causal"),
        r"packs declare duplicate discriminators",
    ),
    (
        "ranker_not_a_boolean",
        lambda raw: raw["slots"][1].update(ranker="yes"),
        r"slots\[dkdv\].ranker must be true or false",
    ),
    # A YAML list or mapping is unhashable: it must not escape as a TypeError.
    (
        "pack_slot_list",
        lambda raw: raw["packs"][0].update(slot=["prep"]),
        r"pack 'prep' declares slot .*, which must be a slot name",
    ),
    (
        "unknown_slot_key",
        lambda raw: raw["slots"][1].update(knob=["block_m"]),
        r"slot 'dkdv' declares \['(knob|typo)'\]",
    ),
    (
        "unknown_slot_kmd_field_key",
        lambda raw: raw["slots"][1]["kmd_fields"][0].update(typo="x"),
        r"slot 'dkdv' kmd_fields entry 'block_m' declares \['(knob|typo)'\]",
    ),
    (
        "slot_kmd_field_repeated",
        lambda raw: raw["slots"][1]["kmd_fields"].append(
            {"name": "block_m", "type": "int"}
        ),
        r"slots\[dkdv\]\.kmd_fields declares \['block_m'\] more than once",
    ),
    (
        "slot_kmd_field_optional",
        lambda raw: raw["slots"][1]["kmd_fields"][0].update(optional=True),
        r"kmd_fields\[\]\.optional is not a real key.*Affected entries: block_m",
    ),
    (
        "slot_kmd_field_default",
        lambda raw: raw["slots"][1]["kmd_fields"][0].update(default=True),
        r"kmd_fields\[\]\.default is not a real key.*Affected entries: block_m",
    ),
]


def _reduce(raw, *, feature=("dtype",), divides=None):
    knob = {"name": "block_m"}
    if divides is not None:
        knob["divides"] = divides
    raw["reduction"] = {
        "fields": {"feature": list(feature), "performance": [knob], "ignored": []}
    }


REFUSALS += [
    (
        "reduction_divides_names_no_slot_field",
        lambda raw: _reduce(raw, divides="seqlen"),
        r"'divides' names 'seqlen', which is not a kmd_fields name: "
        r"\['block_m', 'dtype', 'num_warps'\]",
    ),
    (
        "reduction_feature_names_no_slot_field",
        lambda raw: _reduce(raw, feature=["dtype", "nonesuch"]),
        r"'reduction.fields.feature' names \['nonesuch'\], which no slot's",
    ),
]


def test_reduction_divides_may_name_a_field_only_one_slot_declares(tmp_path):
    config = _load(tmp_path, lambda raw: _reduce(raw, divides="num_warps"))
    assert config.slots


@pytest.mark.parametrize(
    "mutate,message",
    [(mutate, message) for _, mutate, message in REFUSALS],
    ids=[name for name, _, _ in REFUSALS],
)
def test_a_slotted_config_is_refused(tmp_path, mutate, message):
    _expect(tmp_path, mutate, message)


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
