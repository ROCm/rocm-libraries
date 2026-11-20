"""EXPERIMENT: build one ``mx_env_sweep`` kernel and launch it a few times.

For profiling (``rocprofv3`` counters or ATT): the process runs only the
chosen kernel, ``--launches`` times back to back on the visible device (set
``HIP_VISIBLE_DEVICES``), after one untimed warmup launch, and prints the
kernel name for ``kernel_include_regex``. The variant, arm and base flags
come from an ``mx_env_sweep`` config; ``--k`` overrides the config shape's K
so a trace stays small.

    python3 -m rocke.examples.gfx1250.gemm_perf_fp8.mx_one \\
        --config <config.json> --variant cs0nbpf4 --arm c4x4 --k 4096
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--k", type=int, default=0, help="K instead of the config shape's (0: keep)")
    ap.add_argument("--launches", type=int, default=4)
    ap.add_argument("--data", choices=("rand", "mxgemm"), default="mxgemm")
    ap.add_argument("--stamps", action="store_true",
                    help="set ROCKE_EXP_MX_RA_STAMP=1 and report the per-tile wave stamps of the last launch")
    o = ap.parse_args()

    cfg = json.loads(o.config.read_text())
    var = next(v for v in cfg["variants"] if v["name"] == o.variant)
    arm = next(a for a in cfg["arms"] if a["name"] == o.arm)
    env = dict(cfg.get("base_env", {}), **var.get("env", {}), **arm.get("env", {}))
    cluster = None
    if arm.get("cluster"):
        env["ROCKE_EXP_CLUSTER"] = arm["cluster"]
        cx, cy = (int(x) for x in arm["cluster"].split(","))
        cluster = (cx, cy, 1)
    for key in [k for k in os.environ if k.startswith("ROCKE_EXP_")]:
        os.environ.pop(key)
    if o.stamps:
        env["ROCKE_EXP_MX_RA_STAMP"] = "1"
    os.environ.update(env)

    from rocke.examples.gfx1250.gemm_perf_fp8.fp8_gemm_sweep import _launch_geometry
    from rocke.examples.gfx1250.gemm_perf_fp8.mx_env_sweep import _Problem, _spec
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.gemm_universal import build_universal_gemm
    from rocke.runtime.hip_module import Runtime

    spec = _spec()
    kdef = build_universal_gemm(spec, arch="gfx1250")
    art = compile_kernel(kdef, arch="gfx1250", backend="python", capture_ir_text=False)
    m, n, k = cfg["shapes"][0]
    shape = (m, n, o.k or k)
    print(f"kernel: {kdef.name}  shape {shape}  cluster {cluster}", flush=True)

    rt = Runtime()
    fn = rt.load_module(art.hsaco).get_function(kdef.name)
    prob = _Problem(rt, shape, o.data)
    gx, gy = (shape[1] + 255) // 256, (shape[0] + 255) // 256
    tiles = -(-shape[2] // 256)
    tiles += tiles % 2
    n_stamp = gx * gy * 4 * tiles * 8 if o.stamps else 0
    if n_stamp:
        rt.free(prob.c)
        prob.c = rt.alloc(shape[0] * shape[1] * 2 + n_stamp * 8)
    prob.put_scales(*prob.scales, env.get("ROCKE_EXP_MX_SWZ") == "1")
    grid, block = _launch_geometry(spec, shape)
    args = prob.args()
    rt.launch(fn, grid, block, args, cluster=cluster)
    rt.wait_stream(0)
    t0 = time.perf_counter()
    for _ in range(o.launches):
        rt.launch(fn, grid, block, args, cluster=cluster)
        rt.wait_stream(0)
    dt = (time.perf_counter() - t0) / o.launches
    print(f"{o.launches} launches, {dt * 1e6:.1f} us each (host-timed, synchronised)", flush=True)
    if n_stamp:
        _report(rt, prob, shape, gx, gy, tiles, n_stamp, cluster)
    prob.free()


def _report(rt, prob, shape, gx, gy, tiles, n_stamp, cluster):
    """Who waits for whom at each tile's end (last launch)."""
    import ctypes

    import numpy as np

    off = shape[0] * shape[1] * 2
    buf = (ctypes.c_uint8 * (n_stamp * 8))()
    rt.memcpy_d2h(buf, prob.c + off, n_stamp * 8)
    st = np.frombuffer(bytes(buf), dtype=np.int64).reshape(gy, gx, 4, tiles, 8).astype(np.float64)
    st = st[:, :, :, 4:tiles - 4]  # drop the first and last tiles
    cyc, rt_ = st[..., 0::2], st[..., 1::2]  # events: arrive, fill done, past WG barrier, past cluster
    fill = cyc[..., 1] - cyc[..., 0]
    wgw = cyc[..., 2] - cyc[..., 1]
    clw = cyc[..., 3] - cyc[..., 2]
    print(f"\nstamps: {gy}x{gx} WGs x 4 waves x {st.shape[3]} tiles (cycles; mean / p90)")
    for name, v in (("own fill wait", fill), ("WG barrier wait", wgw), ("cluster wait + head reads", clw)):
        print(f"  {name:26s} {v.mean():8.1f} {np.percentile(v, 90):8.1f}")
    print("  by wave (0..3; odd waves load A, even B):")
    for w in range(4):
        print(f"    wave {w}: fill {fill[:, :, w].mean():7.1f}  WG wait {wgw[:, :, w].mean():7.1f}  "
              f"arrive-to-fill-done {(cyc[:, :, w, :, 1] - cyc[:, :, :, :, 0].min(axis=2)).mean():7.1f}")
    last = np.argmax(cyc[..., 1], axis=2)  # which wave's fill lands last, per WG and tile
    print("  wave whose fill lands last (share of WG-tiles):",
          " ".join(f"w{w}={np.mean(last == w):.2f}" for w in range(4)))
    print("  by tile parity: WG wait", " ".join(f"p{p}={wgw[:, :, :, p::2].mean():.1f}" for p in (0, 1)),
          " cluster", " ".join(f"p{p}={clw[:, :, :, p::2].mean():.1f}" for p in (0, 1)))
    if cluster:
        cx, cy = cluster[0], cluster[1]
        ready = rt_[..., 2].max(axis=2)  # WG past its barrier (realtime), latest wave
        pos = np.zeros((cy, cx))
        for y in range(0, gy - gy % cy, cy):
            for x in range(0, gx - gx % cx, cx):
                blk = ready[y:y + cy, x:x + cx]  # cy x cx x tiles
                lastpos = np.argmax(blk.reshape(cy * cx, -1), axis=0)
                for i in lastpos:
                    pos[i // cx, i % cx] += 1
        pos /= pos.sum()
        print("  cluster member past its WG barrier last (share; rows = cluster y, cols = x):")
        for y in range(cy):
            print("    " + " ".join(f"{v:.2f}" for v in pos[y]))
        spread = []
        for y in range(0, gy - gy % cy, cy):
            for x in range(0, gx - gx % cx, cx):
                blk = ready[y:y + cy, x:x + cx].reshape(cy * cx, -1)
                spread.append(blk.max(axis=0) - blk.min(axis=0))
        sp = np.concatenate(spread) * 10.0  # realtime ticks are 10 ns
        print(f"  cluster arrival spread (ns): mean {sp.mean():.0f}  p90 {np.percentile(sp, 90):.0f}")


if __name__ == "__main__":
    main()
