"""The gfx942 split-KV decode module (segment + reduce), packed with the real compiler.

One rocKE `KernelModule` from `build_unified_attention_3d_module` carries the
segment kernel as its primary and the reduce kernel as its `reduce` companion. A
UKD reaches both through `sources`: `segment` takes the primary, `reduce` names the
companion by role. The spec is the one rocKE's strided decode candidate selects, so
what is packed here is what the runtime would build. Needs rocke and libamd_comgr;
like the other packs in this directory, a machine without them fails
(conftest `rocke_available`) rather than skipping.
"""

import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest

from hkp_pack.errors import HkpPackError
from hkp_pack.kernel_signature import kernel_signature
from hkp_pack.kpack_resolver import load_kpack
from pack_helpers import _run_cli, read_shipped, run_pipeline

ARCH = "gfx942"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "multi_launch_decode_merge"
SOURCE = "kernels/gfx942/attention_tiled_3d.py"
BUILDER = "build_unified_attention_3d_module"
PREDICATE = "supports_unified_attention_3d_module"


def _selected_segment_spec():
    """The segment spec rocKE's strided decode candidate selects for one decode batch.

    D128 fp16, GQA 32/8, 16-token blocks, one query token per sequence. The
    candidate fills in the decode grid, the streaming K/V cache policy and
    `num_segments` from its split rule.
    """
    from dispatch.attention import AttentionRequest, dispatch_attention

    result = dispatch_attention(
        AttentionRequest(
            batch=4,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=1,
            seqlen_k=4096,
            hdim_q=128,
            hdim_v=128,
            arch=ARCH,
            dtype="fp16",
            kv_layout="strided",
            kv_block_size=16,
        )
    )
    assert result.candidate.name == "attention_strided_decode"
    spec = result.spec.kernel_spec
    assert (spec.kv_layout, spec.use_decode_grid, spec.kv_cache_policy) == (
        "strided",
        True,
        "stream",
    )
    return spec


def _root(tmp_path, spec):
    """The multi-launch fixture, retargeted at the gfx942 split-KV module."""
    root = tmp_path / "root"
    shutil.copytree(FIXTURE, root)
    kdp_path = root / "multi_launch.kdp.json"
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    kdp["arch"] = [ARCH]
    ukd = kdp["kernelDescriptors"][0]
    ukd["name"] = "multi_launch.gfx942_unified_3d"
    ukd["arch"] = [ARCH]
    entry = {
        "kind": "rocke",
        "source": SOURCE,
        "builder": BUILDER,
        "spec": dataclasses.asdict(spec),
        "compile_target": ARCH,
    }
    ukd["sources"] = {"segment": dict(entry), "reduce": {**entry, "role": "reduce"}}
    ukd["metadata"] = {"family": "split_kv_decode"}
    kdp_path.write_text(json.dumps(kdp, indent=2) + "\n", encoding="utf-8")
    return root


def _pack(root, tmp_path, hipcc, rocm_kpack_dir):
    out = tmp_path / "out"
    run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=out,
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    return out


def _expected_signature(kernel):
    """The marshalled argument list a builder's params describe: names and kinds."""
    from rocke.core.ir import PtrType

    return [
        (p.name, "global_buffer" if isinstance(p.type, PtrType) else "by_value")
        for p in kernel.params
    ]


def test_the_split_kv_module_packs_both_launches_from_one_object(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available
):
    from kernels.gfx942.attention_tiled_3d import (
        build_unified_attention_3d_module,
        reduce_spec_for_segment,
    )

    spec = _selected_segment_spec()
    out = _pack(_root(tmp_path, spec), tmp_path, hipcc, rocm_kpack_dir)
    kdp = out / ARCH / "multi_launch.kdp.json"
    ukd = read_shipped(kdp)["kernelDescriptors"][0]

    sources = ukd["sources"]
    assert set(sources) == {"segment", "reduce"}
    segment, reduce = sources["segment"], sources["reduce"]
    assert all(ks["kind"] == "kpack" and "role" not in ks for ks in sources.values())
    # One compiled module: one object in the archive, two symbols inside it.
    for key in ("library", "toc_key", "sha256"):
        assert segment[key] == reduce[key], key
    assert segment["symbol"] != reduce["symbol"]
    assert segment["symbol"] == spec.kernel_name()
    assert reduce["symbol"] == reduce_spec_for_segment(spec).kernel_name()

    # Each signature is read off the object for its own symbol, and matches the
    # params of the kernel the builder emitted for that launch.
    module = build_unified_attention_3d_module(spec, arch=ARCH)
    kpack, _comp = load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(out / ARCH / segment["library"])
    blob = bytes(archive.get_kernel(segment["toc_key"], ARCH))
    for ks, kernel in (
        (segment, module.primary),
        (reduce, module.companions["reduce"]),
    ):
        assert ks["signature"] == kernel_signature(blob, ks["symbol"], "t")
        assert [(a["name"], a["kind"]) for a in ks["signature"]] == (
            _expected_signature(kernel)
        )
    assert segment["signature"] != reduce["signature"]

    result = _run_cli(str(kdp), "--kpack-python-dir", rocm_kpack_dir, mode="full")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "compiled specialization agreement: OK for 1 kernel(s)" in result.stdout


def test_a_module_the_predicate_refuses_is_refused_at_pack_time(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available
):
    from kernels.gfx942.attention_tiled_3d import supports_unified_attention_3d_module

    # D256 with 64-token blocks and no tile override: the segment's Q/K/V tiles
    # overflow gfx942's 64 KiB LDS.
    spec = dataclasses.replace(
        _selected_segment_spec(), head_size=256, block_size=64, tile_size_override=None
    )
    ok, reason = supports_unified_attention_3d_module(spec, arch=ARCH)
    assert not ok and "LDS" in reason

    with pytest.raises(HkpPackError) as excinfo:
        _pack(_root(tmp_path, spec), tmp_path, hipcc, rocm_kpack_dir)
    assert re.search(
        rf"spec rejected by {PREDICATE} for builder '{BUILDER}' @ {ARCH}: "
        + re.escape(reason),
        str(excinfo.value),
    ), str(excinfo.value)
