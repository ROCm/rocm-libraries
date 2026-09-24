"""gfx950 fp8 decode — static ISA bottleneck classification (runbook 3.1b).

GPU-free: compiles the fp8 decode kernels (D64 64/8, bs16, kv fp8e4m3) via comgr
and objdumps them, reporting the opcode mix (mfma / vmem_load / vmem_store / valu
/ ds_read / ds_write / waitcnt / barrier) so we can classify compute- vs memory-
vs sync-bound (runbook 3.2/3.3/3.4) BEFORE guessing levers.

Two kernel families are inspected:
  * the 3D split-KV segment kernel -- the path this decode cohort actually runs
    on, and the one whose VALU/softmax-bound profile the case study rests on; and
  * the 2D tiled kernel (shipped sync-dequant vs the fp8qk K-in-LDS variant), to
    show what the fp8-K-in-LDS lever trades on the 2D path (context for why it
    does not transfer to the LDS-write-light 3D seg).

Run (rocke env, any host with comgr+objdump; no GPU needed):
    python fp8_decode_isa_inspect.py
"""

from __future__ import annotations

import sys
from dataclasses import replace

from rocke.assets import dsl_docs_dir

sys.path.insert(
    0, str(dsl_docs_dir() / "optimization" / "utilities" / "tools" / "dsl_probes")
)

from probe_isa_inspect import probe_isa_inspect  # noqa: E402
import kernels.common.attention_unified as au  # noqa: E402
from kernels.gfx950.attention_tiled_2d import (  # noqa: E402
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)

_2D_BASE = dict(
    head_size=64,
    block_size=16,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    sliding_window=0,
    has_softcap=False,
    kv_storage_dtype="fp8e4m3",
    num_warps=1,
    tile_size=32,
)


def _seg_3d_kernel(num_seqs: int, kv_len: int):
    """Build the shipped fp8 3D split-KV *segment* kernel for one decode shape,
    at the production num_segments (the gfx950 pre-bump clamp)."""
    au._RESOLVED_ATTENTION_ARCH = "gfx950"
    problem = au.UnifiedAttentionProblem(
        total_q=num_seqs,
        num_seqs=num_seqs,
        num_query_heads=64,
        num_kv_heads=8,
        head_size=64,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=kv_len,
        dtype="bf16",
        use_fp8=True,
        num_cus=256,
    )
    _spec3d, _reduce, build_seg, _build_red, _ = au._tiled_3d_impl("gfx950")
    seg_spec = replace(
        au._tiled_3d_spec_from_problem(problem), num_segments=au._num_segments(problem)
    )
    return build_seg(seg_spec, arch="gfx950")


def main() -> int:
    entries = [
        # The actual decode path: the 3D split-KV segment kernel. Expect
        # valu >> vmem_load and a small ds_write (the VALU/softmax-bound floor).
        ("3d_seg_fp8_b64", _seg_3d_kernel(64, 8192)),
        ("3d_seg_fp8_b1", _seg_3d_kernel(1, 8192)),
        # 2D contrast: sync-dequant vs the fp8qk K-in-LDS lever (a loader
        # ds_write saver -- shown here to explain why it does not help the 3D seg).
        (
            "2d_fp8_sync",
            build_unified_attention_2d_tiled(
                UnifiedAttention2DTiledSpec(**_2D_BASE, use_sinks=False), arch="gfx950"
            ),
        ),
        (
            "2d_fp8qk_KinLDS",
            build_unified_attention_2d_tiled(
                UnifiedAttention2DTiledSpec(
                    **_2D_BASE, use_sinks=False, use_fp8_mfma_qk=True
                ),
                arch="gfx950",
            ),
        ),
    ]
    probe_isa_inspect(entries, mcpu="gfx950")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
