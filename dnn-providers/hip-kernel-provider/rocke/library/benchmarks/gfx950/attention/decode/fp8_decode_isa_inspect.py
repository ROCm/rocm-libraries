"""gfx950 fp8 decode — static ISA bottleneck classification (runbook 3.1b).

GPU-free: compiles the shipped fp8 decode 2D kernel (nw1/t32, bs16, D64 64/8,
kv fp8e4m3) via comgr and objdumps it, reporting the opcode mix (mfma /
vmem_load / vmem_store / valu / waitcnt / barrier) so we can classify
compute- vs memory- vs sync-bound (runbook 3.2/3.3/3.4) BEFORE guessing levers.
Also inspects the fp8qk (K-in-LDS) variant to see which resource it trades.

Run (rocke env, any host with comgr+objdump; no GPU needed):
    python fp8_decode_isa_inspect.py
"""

from __future__ import annotations

import sys

from rocke.assets import dsl_docs_dir

sys.path.insert(0, str(dsl_docs_dir() / "optimization" / "utilities" / "tools" / "dsl_probes"))

from probe_isa_inspect import probe_isa_inspect  # noqa: E402
from kernels.gfx950.attention_tiled_2d import (  # noqa: E402
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)

_BASE = dict(
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


def main() -> int:
    specs = [
        ("flash_shipped(nw1t32)", UnifiedAttention2DTiledSpec(**_BASE, use_sinks=False)),
        ("sink_shipped(nw1t32)", UnifiedAttention2DTiledSpec(**_BASE, use_sinks=True)),
        ("flash_fp8qk", UnifiedAttention2DTiledSpec(
            **_BASE, use_sinks=False, use_fp8_mfma_qk=True)),
    ]
    entries = [
        (label, build_unified_attention_2d_tiled(spec, arch="gfx950"))
        for label, spec in specs
    ]
    probe_isa_inspect(entries, mcpu="gfx950")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
