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
    prob.free()


if __name__ == "__main__":
    main()
