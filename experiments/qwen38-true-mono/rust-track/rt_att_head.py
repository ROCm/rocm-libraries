"""RT: launch a saved lm_head code object (``rt_head.py`` arm) for a WaveScope (ATT) capture; torch-free.

    python rt_att_head.py --hsaco DIR/head_fused_r4.D.hsaco --epi fused --rows 4 --m 4 --launches 8

The code object is loaded as saved by a timing run (``rt_head.py`` writes ``<label>.<arm>.hsaco``), so the COMGR
and direct objects traced are byte for byte the ones timed (no compile here: the ATT host has another COMGR).
Inputs as ``rt_head.py`` (``head_data``, 3 cold weight copies cycled); the kernel symbol is the build's name.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rust-pilot"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", required=True)
    ap.add_argument("--epi", default="fused", choices=("fused", "logits"))
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--m", type=int, default=0, help="active rows (default rows)")
    ap.add_argument("--launches", type=int, default=8)
    a = ap.parse_args(argv)
    sys.modules["torch"] = None  # rocprofv3 ATT aborts when a second HIP runtime (torch's) registers late
    import r2_gpu as G
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    import kernels.gfx950.mx_decode_gemv as M
    import rt_builds
    from rt_head import head_data

    spec = M.MxDecodeGemvSpec(**rt_builds.HEAD, rows_max=a.rows, epilogue=a.epi, **({"waves_per_eu": 1} if a.epi == "fused" else {}))
    name = rt_builds.build(f"head_{a.epi}_r{a.rows}").name
    hsaco = open(a.hsaco, "rb").read()
    m = a.m or a.rows
    dev = H.Dev(watchdog_s=120)
    try:
        d = head_data(G, dev, 3, a.rows)
        fn = dev.function(G.raw_art(name, hsaco))
        ws = dev.alloc(M.fused_workspace_bytes(spec)) if a.epi == "fused" else None
        y = dev.alloc(64 * a.rows) if a.epi == "fused" else dev.alloc(4 * G.HEAD_N * a.rows)
        for i in range(a.launches):
            dev.launch(fn, spec.ctas, spec.threads, G.head_args(spec, d, y, ws, m, 0, copy=i % 3))
        dev.sync()
        print(f"ATTHEAD {name} launches {a.launches} m {m} ok", flush=True)
    finally:
        dev.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
