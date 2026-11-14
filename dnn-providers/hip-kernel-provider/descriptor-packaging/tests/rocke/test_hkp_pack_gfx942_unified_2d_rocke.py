"""The gfx942 unified attention 2D kernel, packed with the real compiler.

The specs are the ones the gfx942 tuning candidates the hipkernel:Gfx942AttentionUnified2D
manifest pins select, so what is packed here is what the catalog ships: the D128 default
geometry, and the D64 8-warp x 32-row geometry, which also carries the direct-Q and
conflict-free V store fields its default turns on. Needs rocke and libamd_comgr; like the
other packs in this directory, a machine without them fails (conftest `rocke_available`)
rather than skipping.

`HKP_PACK_OUTPUT_DIR`, when set, receives the packed tree under `gfx942_unified_2d`, for a
later run of `hipdnn_validate_descriptors`.
"""

import dataclasses
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from hkp_pack.errors import HkpPackError
from hkp_pack.kernel_signature import kernel_signature
from hkp_pack.kpack_resolver import load_kpack
from pack_helpers import _run_cli, read_shipped, run_pipeline

ARCH = "gfx942"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rocke"
SOURCE = "kernels/gfx942/attention_tiled_2d.py"
BUILDER = "build_unified_attention_2d_tiled"
PREDICATE = "supports_tiled_2d_spec"

# The kernel arguments the native pack marshals, in order (Gfx942AttentionUnified2DNative.cpp).
PACK_ABI = [
    ("output_ptr", "global_buffer"),
    ("query_ptr", "global_buffer"),
    ("key_cache_ptr", "global_buffer"),
    ("value_cache_ptr", "global_buffer"),
    ("sink_ptr", "global_buffer"),
    ("block_tables_ptr", "global_buffer"),
    ("seq_lens_ptr", "global_buffer"),
    ("alibi_slopes_ptr", "global_buffer"),
    ("qq_bias_ptr", "global_buffer"),
    ("query_start_len_ptr", "global_buffer"),
    ("scale", "by_value"),
    ("k_scale", "by_value"),
    ("v_scale", "by_value"),
    ("out_scale", "by_value"),
    ("softcap", "by_value"),
    ("num_seqs", "by_value"),
    ("block_table_stride", "by_value"),
    ("qq_bias_stride_0", "by_value"),
    ("num_query_heads", "by_value"),
    ("num_kv_heads", "by_value"),
    ("sliding_window", "by_value"),
]


def _selected_spec(spec_id, head_size):
    """The kernel spec the pinned tuning candidate selects for a bottom-right causal bf16
    prefill at GQA ratio 4 (Hq 32 over Hkv 8). Metadata only: no GPU tensors."""
    from dispatch.attention import attention_candidates
    from dispatch.attention.common import AttentionMaskType, AttentionRequest

    (candidate,) = [
        c for c in attention_candidates() if getattr(c, "spec_id", "") == spec_id
    ]
    request = AttentionRequest(
        batch=4,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=head_size,
        hdim_v=head_size,
        arch=ARCH,
        dtype="bf16",
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        kv_block_size=16,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    return candidate.select_spec(request).kernel_spec


def _root(tmp_path, spec):
    """The single-kernel rocke fixture, retargeted at the gfx942 tiled 2D builder."""
    root = tmp_path / "root"
    shutil.copytree(FIXTURE, root)
    kdp_path = root / "attention.kdp.json"
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    kdp["arch"] = [ARCH]
    ukd = kdp["kernelDescriptors"][0]
    ukd["name"] = "attention.gfx942_unified_2d"
    ukd["arch"] = [ARCH]
    ukd["kernel_source"] = {
        "kind": "rocke",
        "source": SOURCE,
        "builder": BUILDER,
        "spec": dataclasses.asdict(spec),
    }
    ukd["metadata"] = {"family": "attention_unified_2d"}
    kdp["kernelDescriptors"] = [ukd]
    kdp_path.write_text(json.dumps(kdp, indent=2) + "\n", encoding="utf-8")
    return root


def _pack(root, tmp_path, hipcc, rocm_kpack_dir, export=None):
    override = os.environ.get("HKP_PACK_OUTPUT_DIR")
    out = Path(override) / export if override and export else tmp_path / "out"
    if out.exists():
        shutil.rmtree(out)
    run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=out,
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    return out


@pytest.mark.parametrize(
    "spec_id, head_size",
    [
        ("gfx942_u2d_transposed_x8_nw2_mw32_t2xb_llvm", 128),
        ("gfx942_u2d_transposed_x8_nw8_mw32_t4xb_llvm", 64),
    ],
)
def test_a_shipped_geometry_packs_with_the_abi_the_native_pack_marshals(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available, spec_id, head_size
):
    from kernels.gfx942.attention_tiled_2d import build_unified_attention_2d_tiled

    spec = _selected_spec(spec_id, head_size)
    out = _pack(
        _root(tmp_path, spec),
        tmp_path,
        hipcc,
        rocm_kpack_dir,
        export=f"gfx942_unified_2d_{spec_id}",
    )
    kdp = out / ARCH / "attention.kdp.json"
    ks = read_shipped(kdp)["kernelDescriptors"][0]["kernel_source"]
    assert ks["kind"] == "kpack"
    assert ks["symbol"] == spec.kernel_name()

    # The signature recorded off the compiled object is the builder's parameter list and
    # the argument list the native pack marshals, name for name.
    kernel = build_unified_attention_2d_tiled(spec, arch=ARCH)
    kpack, _comp = load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(out / ARCH / ks["library"])
    blob = bytes(archive.get_kernel(ks["toc_key"], ARCH))
    assert ks["signature"] == kernel_signature(blob, ks["symbol"], "t")
    assert [(a["name"], a["kind"]) for a in ks["signature"]] == PACK_ABI
    assert [p.name for p in kernel.params] == [name for name, _ in PACK_ABI]

    result = _run_cli(str(kdp), "--kpack-python-dir", rocm_kpack_dir, mode="full")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "compiled specialization agreement: OK for 1 kernel(s)" in result.stdout


def test_a_spec_the_predicate_refuses_is_refused_at_pack_time(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available
):
    from kernels.gfx942.attention_tiled_2d import supports_tiled_2d_spec

    # Narrow 16x16, 4 warps x 16 rows, a 64-token tile at D128: the staged tiles overflow
    # gfx942's 64 KiB LDS, which is why the manifest pins this geometry at D64 only.
    spec = _selected_spec("gfx942_u2d_narrow_nw4_mw16_t4xb_llvm", 64)
    spec = dataclasses.replace(spec, head_size=128)
    ok, reason = supports_tiled_2d_spec(spec, arch=ARCH)
    assert not ok and "LDS" in reason

    with pytest.raises(HkpPackError) as excinfo:
        _pack(_root(tmp_path, spec), tmp_path, hipcc, rocm_kpack_dir)
    assert re.search(
        rf"spec rejected by {PREDICATE} for builder '{BUILDER}' @ {ARCH}: "
        + re.escape(reason),
        str(excinfo.value),
    ), str(excinfo.value)
