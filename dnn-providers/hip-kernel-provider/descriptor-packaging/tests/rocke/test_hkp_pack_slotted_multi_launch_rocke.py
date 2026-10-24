"""A slotted engine whose every slot is rocKE, one of them multi-launch, packed with
the real compiler (RFC 0021 composed with RFC 0017 `sources`).

`fixtures/slotted_multi_launch` has the slots `prep` and `dq` as single-launch UKDs and
`dkdv` as a UKD with `sources` (`decode` and `merge`), each slot with its own KMD and
dispatch descriptor, all at gfx1151 over develop's `wmma_fmha_decode` builders. Packing
resolves each pack against its slot's KMD and walks every launch; `desk_check --mode
full` then reads each shipped pack back through the slot-aware context.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from hkp_pack.errors import HkpPackError
from pack_helpers import _run_cli, read_shipped, run_pipeline

ARCH = "gfx1151"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "slotted_multi_launch"
SLOTS = ("prep", "dkdv", "dq")


@pytest.fixture(scope="module")
def packed(tmp_path_factory, hipcc, rocm_kpack_dir, rocke_available):
    tmp_path = tmp_path_factory.mktemp("slotted_multi_launch")
    override = os.environ.get("HKP_PACK_OUTPUT_DIR")
    out = Path(override) / FIXTURE.name if override else tmp_path / "out"
    if out.exists():
        shutil.rmtree(out)
    run_pipeline(
        source_root=FIXTURE,
        arches=[ARCH],
        out_root=out,
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    return out / ARCH


def test_each_slot_packs_its_launches_against_its_own_kmd(packed):
    kernels = {}
    for slot in SLOTS:
        pack = read_shipped(packed / f"{slot}.kdp.json")
        assert pack["slot"] == slot
        (kernels[slot],) = pack["kernelDescriptors"]

    # The multi-launch slot keeps both launches as packed kpack entries; the
    # single-launch slots keep one.
    decode_merge = kernels["dkdv"]["sources"]
    assert set(decode_merge) == {"decode", "merge"}
    assert all(ks["kind"] == "kpack" for ks in decode_merge.values())
    assert decode_merge["decode"]["symbol"] != decode_merge["merge"]["symbol"]
    for slot in ("prep", "dq"):
        assert kernels[slot]["kernel_source"]["kind"] == "kpack"
        assert "sources" not in kernels[slot]
    assert "kernel_source" not in kernels["dkdv"]

    # Each pack completed against its own slot's KMD: the one field it declares.
    for slot, kernel in kernels.items():
        assert kernel["metadata"] == {"family": slot}


def test_each_slot_has_its_own_dispatch_descriptor(packed):
    dispatch = {
        slot: read_shipped(packed / f"{slot}.kdp.json")["dispatch"] for slot in SLOTS
    }
    assert len(set(dispatch.values())) == len(SLOTS)
    for slot, udd in dispatch.items():
        assert (packed / f"{slot}.udd.json").is_file()
        assert udd in (packed / f"{slot}.udd.json").read_text()


@pytest.mark.parametrize("slot", SLOTS)
def test_each_slot_pack_passes_the_full_desk_check(packed, rocm_kpack_dir, slot):
    result = _run_cli(
        str(packed / f"{slot}.kdp.json"),
        "--kpack-python-dir",
        rocm_kpack_dir,
        mode="full",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "compiled specialization agreement: OK for 1 kernel(s)" in result.stdout


def _bound_dkdv_root(tmp_path, merge_head_size):
    """The fixture with `dkdv` declaring `head_size` as a compiled field.

    The consumer binds it to the spec attribute of the same name, so every launch's
    compile must show it. The merge launch carries its own spec (`dtype` spelled out,
    which is the builder's default), so it is a different variant from `decode` and
    gets its own compile and observation.
    """
    root = tmp_path / "src"
    shutil.copytree(FIXTURE, root)

    kmd_path = root / "dkdv.kmd.json"
    kmd = json.loads(kmd_path.read_text(encoding="utf-8"))
    kmd["fields"].append({"name": "head_size", "type": "int"})
    kmd_path.write_text(json.dumps(kmd, indent=2) + "\n", encoding="utf-8")

    kdp_path = root / "dkdv.kdp.json"
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    (ukd,) = kdp["kernelDescriptors"]
    ukd["metadata"]["head_size"] = 128
    decode, merge = ukd["sources"]["decode"], ukd["sources"]["merge"]
    decode["builder"] = "build_wmma_fmha_decode"
    merge["builder"] = "build_wmma_fmha_decode_merge"
    merge.pop("role")
    merge["spec"] = {"head_size": merge_head_size}
    (consumer,) = ukd["provenance"]["specialization_contract"]["consumers"]
    consumer["metadata_fields"] = ["head_size"]
    consumer["bindings"] = {"head_size": {"field": "head_size"}}
    kdp_path.write_text(json.dumps(kdp, indent=2) + "\n", encoding="utf-8")
    return root


def _pack_root(root, tmp_path, hipcc, rocm_kpack_dir):
    run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )


def test_every_launch_of_a_slot_pack_is_observed_against_the_slot_kmd(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available
):
    # Two distinct variants in one slot pack, each compiled and each agreeing with the
    # declared `head_size`: both launches' observations must have been requested.
    root = _bound_dkdv_root(tmp_path, merge_head_size=128)
    _pack_root(root, tmp_path, hipcc, rocm_kpack_dir)

    ukd = read_shipped(tmp_path / "out" / ARCH / "dkdv.kdp.json")["kernelDescriptors"][
        0
    ]
    assert set(ukd["sources"]) == {"decode", "merge"}
    assert set(ukd["provenance"]["effective_spec"]) == {"decode", "merge"}


def test_a_slot_pack_launch_that_disagrees_with_the_slot_metadata_fails_the_pack(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available
):
    # The merge launch compiles at head_size 64 while the slot's metadata says 128;
    # the decode launch alone agrees, so a walk that stopped after the first launch
    # would let it ship.
    root = _bound_dkdv_root(tmp_path, merge_head_size=64)

    with pytest.raises(HkpPackError, match=r"launch 'merge'.*disagrees"):
        _pack_root(root, tmp_path, hipcc, rocm_kpack_dir)
