# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Opt-in benchmark for the 3-D Winograd F(2x2x2,3x3x3) forward prototype.

Runs ``kernels.common.conv_winograd3d`` on halo-form NDHWC bf16 problems (N=1, valid
conv, stride 1), verifies against a float32 torch conv3d with scale-free metrics
(``max|out-ref| / max|ref|`` below 5e-2 and ``mean|out-ref| / mean|ref|`` below
``VERIFY_MEAN_REL``, so scaled data modes are judged like unscaled ones; the exit status
is 1 when a verified run fails), and times it with the
staged protocol (one stalled call per sample, median) -- the same protocol
``benchmark_conv_compare --timing staged`` reports. Two times are printed: the main
kernel alone, and the whole call (weight transform + main kernel).

Example::

    python -m benchmarks.common.benchmark_winograd3d --shape 130,66,66,256,128
"""

from __future__ import annotations

import argparse
import json
import time

# verify verdict: scale-free max and mean relative error of the output against the f32
# reference (both independent of the data scale, so tr@-20 / tv<b>@e data cannot pass a
# zero or garbage output)
VERIFY_MAX_REL = 5e-2
VERIFY_MEAN_REL = 1e-2

UNET3D_SHAPES = {
    1: (258, 130, 130, 64, 64),
    2: (258, 130, 130, 128, 64),
    3: (130, 66, 66, 256, 128),
    4: (130, 66, 66, 128, 128),
    5: (66, 34, 34, 512, 256),
    6: (66, 34, 34, 256, 256),
    7: (34, 18, 18, 512, 512),
    8: (34, 18, 18, 1024, 512),
    9: (130, 66, 66, 64, 128),
    10: (18, 10, 10, 1024, 1024),
}


def _power_probe(fn, seconds=1.5):
    """Back-to-back launches for ~``seconds``: ms/call, mean GFX clock (amdsmi, all XCDs),
    socket W and cycles per call."""
    import os
    import threading

    import amdsmi
    import torch

    amdsmi.amdsmi_init()
    gidx = int((os.environ.get("HIP_VISIBLE_DEVICES") or "0").split(",")[0])
    h = amdsmi.amdsmi_get_processor_handles()[gidx]
    clocks, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                c = [
                    x
                    for x in amdsmi.amdsmi_get_gpu_metrics_info(h)["current_gfxclks"]
                    if isinstance(x, int)
                ]
                if c:
                    clocks.append(sum(c) / len(c))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.002)

    th = threading.Thread(target=sampler)
    e0 = amdsmi.amdsmi_get_energy_count(h)
    th.start()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        for _ in range(20):
            fn()
        n += 20
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    e1 = amdsmi.amdsmi_get_energy_count(h)
    stop.set()
    th.join()
    ej = (
        (e1["energy_accumulator"] - e0["energy_accumulator"])
        * e0["counter_resolution"]
        * 1e-6
    )
    k = len(clocks) // 10
    cs = sorted(clocks[k : len(clocks) - k] or clocks or [0])
    mhz = cs[len(cs) // 2]
    ms = (t1 - t0) / n * 1e3
    return {"b2b_ms": ms, "mhz": mhz, "watt": ej / (t1 - t0), "mcyc": ms * mhz / 1e3}


def _make_data(kind, xs, ws, dev):
    """Operands (bf16). u1: uniform(-1, 1) both (the rocKE verify data); tr: training-like
    -- activations N(0, 1), weights Kaiming-normal std sqrt(2 / (27 C)); a suffix ``@e`` scales
    the activations by 2**e (e.g. tr@-20: gradients ~1e-6, f16 subnormal range)."""
    import torch

    base, _, e = kind.partition("@")
    if base == "u1":
        X = torch.empty(*xs, device=dev).uniform_(-1, 1)
        W = torch.empty(*ws, device=dev).uniform_(-1, 1)
    elif base == "tr":
        X = torch.randn(*xs, device=dev)
        W = torch.randn(*ws, device=dev) * (2.0 / (27 * ws[-1])) ** 0.5
    elif base.startswith("tv"):
        # tv<b>: training-like with a smooth spatial magnitude field 2^(b s(z,y,x)), s in
        # [-1, 1] (gradients whose magnitude varies by +-b binades across the volume)
        import math

        bnd = float(base[2:])
        g = [torch.linspace(0, math.pi * 1.7, n, device=dev) for n in xs[:3]]
        zz, yy, xx = torch.meshgrid(*g, indexing="ij")
        sf = torch.sin(zz + 0.3) * torch.cos(1.3 * yy) * torch.sin(0.7 * xx + 1.0)
        X = torch.randn(*xs, device=dev) * (2.0 ** (bnd * sf))[..., None]
        W = torch.randn(*ws, device=dev) * (2.0 / (27 * ws[-1])) ** 0.5
    else:
        raise ValueError(kind)
    if e:
        X = X * 2.0 ** int(e)
    return X.to(torch.bfloat16), W.to(torch.bfloat16)


def verify_verdict(Yv, ref) -> dict:
    """The verify verdict of an output Yv against the f32 reference: ``rel_err`` =
    max|err| / max|ref|, ``mean_rel`` = mean|err| / mean|ref| (both scale-free) and
    ``verify`` PASS when they are below VERIFY_MAX_REL / VERIFY_MEAN_REL."""
    err = (Yv - ref).abs()
    ra = ref.abs()
    rel = err.max().item() / max(ra.max().item(), 1e-30)
    mrel = err.mean().item() / max(ra.mean().item(), 1e-30)
    ok = rel < VERIFY_MAX_REL and mrel < VERIFY_MEAN_REL
    return {"rel_err": rel, "mean_rel": mrel, "verify": "PASS" if ok else "FAIL"}


def _acc_stats(Yv, ref):
    """Error of Yv (f32 view of the bf16 output) vs the f32 reference: the scale-free
    max|err| / max|ref| and mean|err| / mean|ref| (the verify verdict's two metrics),
    the worst output channel (max_c max|err_c| / max|ref_c|) and the relative error of the
    output's bf16 rounding alone (the floor any bf16-output kernel has)."""
    import torch

    err = (Yv - ref).abs()
    K = ref.shape[-1]
    e2 = err.reshape(-1, K)
    r2 = ref.abs().reshape(-1, K)
    rmax = r2.amax(0).clamp_min(1e-30)
    per_c = e2.amax(0) / rmax
    rnd = (ref.to(torch.bfloat16).float() - ref).abs()
    out = {
        "max_rel_scale_free": (err.max() / ref.abs().max()).item(),
        "mean_rel": (err.mean() / ref.abs().mean()).item(),
        "worst_ch_rel": per_c.max().item(),
        "worst_ch": int(per_c.argmax().item()),
        "median_ch_rel": per_c.median().item(),
        "bf16_round_max_rel": (rnd.max() / ref.abs().max()).item(),
        "bf16_round_mean_rel": (rnd.mean() / ref.abs().mean()).item(),
        "ref_absmax": ref.abs().max().item(),
    }
    del err, e2, r2, rnd
    return out


def run_one(
    shape,
    *,
    arch,
    warmup,
    iters,
    verify,
    spec_kw,
    label="",
    power=False,
    pipe=None,
    dgrad=False,
    acc_out=None,
    data="u1",
    timing=True,
    wtv=1,
    scale_exp=0,
    det_runs=0,
    adapter=False,
    adapter_pd=1,
):
    import torch

    from kernels.common import conv_winograd3d as cw
    from rocke.helpers import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from rocke.runtime.stall_gate import time_launches_staged

    spec_kw = dict(spec_kw)
    variant = spec_kw.pop("variant", "3d")
    torch.manual_seed(0)
    dev = "cuda"
    if dgrad:
        # BWD-data as a forward conv: dX = conv(pad(dY, 2), Wd), Wd[c][dz][dy][dx][k] =
        # W[k][2-dz][2-dy][2-dx][c]. The Winograd problem is (Do+4, Ho+4, Wo+4, C=K, K=C);
        # "X" below is dY (Do, Ho, Wo, K) and "W" the forward filter (K, 3, 3, 3, C).
        if variant != "yx":
            raise ValueError("--dgrad: variant yx only")
        fDi, fHi, fWi, fC, fK = shape
        if adapter:  # dX of the pad-(pd,1,1) conv: conv(pad(dY, (2 - pd, 1, 1)), Wd)
            Di, Hi, Wi, C, K = fDi + 2 - 2 * adapter_pd, fHi, fWi, fK, fC
        else:
            Di, Hi, Wi, C, K = fDi + 2, fHi + 2, fWi + 2, fK, fC
        X, W = _make_data(data, (fDi - 2, fHi - 2, fWi - 2, fK), (fK, 3, 3, 3, fC), dev)
    else:
        Di, Hi, Wi, C, K = shape
        hs_ = (
            2 if adapter else 0
        )  # adapter form: X is the unpadded (pad (pd,1,1)) input
        X, W = _make_data(
            data, (Di - hs_ * adapter_pd, Hi - hs_, Wi - hs_, C), (K, 3, 3, 3, C), dev
        )
    # zero halo the relayout adds around X: halo-form dgrad 2, adapter (pad 1,1,1) form 1;
    # z may differ (adapter pad (0,1,1): FWD 0, BWD-data 2)
    hx = 1 if adapter else (2 if dgrad else 0)
    hz = (2 - adapter_pd if dgrad else adapter_pd) if adapter else hx
    hxs = hx if hz == hx else (hz, hx, hx)
    Wd = W.flip(1, 2, 3).permute(4, 1, 2, 3, 0).contiguous() if dgrad else W
    if variant == "x":
        spec = cw.WinogradXSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, **spec_kw)
        ok, why = cw.is_valid_x_spec(spec)
        build, grid_fn = cw.build_winograd_x_fwd, cw.winograd_x_grid

        def wt():
            return cw.winograd_x_weight_transform(W)

        lay = "c8"
    elif variant == "yx":
        spec = cw.WinogradYxSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, **spec_kw)
        ok, why = cw.is_valid_yx_spec(spec)
        build, grid_fn = cw.build_winograd_yx, cw.winograd_yx_grid
        udt = torch.float16 if spec.op_dtype == "f16" else torch.bfloat16

        def wt():
            return cw.winograd_yx_weight_transform(
                Wd,
                dtype=udt,
                fold=spec.stage in ("reg", "wlds"),
                u16=spec.b16,
                uc=spec.bc,
                my=getattr(spec, "my", 2),
            )

        lay = getattr(spec, "xin", "c8")
    else:
        spec = cw.Winograd3dSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, **spec_kw)
        ok, why = cw.is_valid_spec(spec)
        build, grid_fn = cw.build_winograd3d_fwd, cw.winograd3d_grid
        lay = spec.layout

        def wt():
            return cw.winograd3d_weight_transform(W, layout=lay)

    if not ok:
        print(f"{label} skip: {why}", flush=True)
        return None
    t0 = time.time()
    kern = build(spec)
    art = compile_kernel(kern, arch=arch)
    t_build = time.time() - t0
    frel = variant == "yx" and getattr(spec, "frel", 0)
    sigb = (
        SignatureBuilder()
        .ptr("X", "bf16")
        .ptr(
            "U", "f16" if variant in ("yx", "x") and spec.op_dtype == "f16" else "bf16"
        )
        .ptr(
            "Y", "f32" if variant == "yx" and getattr(spec, "ksplit", 1) > 1 else "bf16"
        )
    )
    if frel:
        sigb = sigb.ptr("Xn", "bf16").ptr("F", "i32")
    boff = variant == "yx" and getattr(spec, "boff", False)
    if boff:
        sigb = sigb.scalar("BOFF", "i32")
    osc = variant == "yx" and getattr(spec, "oscale", False)
    if osc:
        sigb = sigb.scalar("OS", "f32")
    bps = variant == "yx" and getattr(spec, "bps", 0)
    bpo = variant == "yx" and getattr(spec, "bps_out", False)
    if bps in (1, 2):  # bps=3 computes its scale in the kernel (no map)
        sigb = sigb.ptr("E", "i32")
    if bpo:
        sigb = sigb.ptr("EO", "i32")
    SX = (
        2.0**scale_exp
    )  # relayout scale (power of two); the main kernel multiplies by 1/SX
    sig = sigb.build()
    launcher = KernelLauncher(hsaco=art.hsaco, kernel_name=kern.name, signature=sig)
    yc8h = variant == "yx" and getattr(spec, "yc8h", False)
    if yc8h:  # next layer's input layout: c8 with a zero one-voxel halo
        hh = spec.yc8h_halo
        Y = torch.zeros(
            K // 8,
            spec.Do + 2 * hh,
            spec.Ho + 2 * hh,
            spec.Wo + 2 * hh,
            8,
            device=dev,
            dtype=torch.bfloat16,
        )
    else:
        Y = torch.zeros(spec.Do, spec.Ho, spec.Wo, K, device=dev, dtype=torch.bfloat16)
    U = wt()
    if bps == 2 and not dgrad:  # chained input: the producer's yc8h halo is zero
        hb = spec.bps_halo
        Xh = torch.zeros_like(X)
        Xh[hb : Di - hb, hb : Hi - hb, hb : Wi - hb] = X[
            hb : Di - hb, hb : Hi - hb, hb : Wi - hb
        ]
        X = Xh
        del Xh
    if lay == "ndhwc" and variant == "yx":
        # the main kernel reads X / dY in place; a padded form's zero halo by predication
        if cw._xpad(spec) != (hz, hx):
            raise ValueError(f"xin ndhwc: spec xpad (z, yx) must be {(hz, hx)}")
        Xk = X * SX if SX != 1.0 else X
    elif hx:
        Xk = cw.winograd3d_x_to_c8(
            torch.nn.functional.pad(X * SX, (0, 0) + (hx,) * 4 + (hz,) * 2)
        )
    else:
        Xk = {"c8": cw.winograd3d_x_to_c8, "r8": cw.winograd_x_to_r8}.get(
            lay, lambda t: t
        )(X * SX)
    grid = grid_fn(spec)
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    args = {"X": Xk, "U": U, "Y": Y}
    if boff:
        args["BOFF"] = 0
    if osc:
        args["OS"] = 1.0 / SX

    def _absbits(t):  # |x| bf16 bit patterns as int32
        return t.contiguous().view(torch.int16).to(torch.int32) & 0x7FFF

    if bps and osc:
        raise ValueError("bps replaces oscale: use one of them")
    if adapter and (bps == 2 or yc8h or variant != "yx"):
        raise ValueError("--adapter: drop-in yx only (no bps=2 / yc8h chained input)")
    if bps == 1:
        hb = spec.bps_halo
        if hb != hx or cw._bps_hz(spec) != hz:
            raise ValueError(
                f"bps=1: bps_halo must be {hx} and the z halo {hz} (the relayout's)"
            )
        src = (X * SX) if osc else X
        n_ent = cw.x_to_c8_lds_grid(Di - 2 * hz, Hi - 2 * hb, Wi - 2 * hb, C)[0]
        vb = cw._x8_vb(C)
        Em = torch.zeros(n_ent * vb, device=dev, dtype=torch.int32)
        am = _absbits(src).amax(-1).reshape(-1)
        Em[: am.numel()] = am
        E_ref = Em.reshape(n_ent, vb).amax(1)
        E = torch.zeros(n_ent, device=dev, dtype=torch.int32)
        args["E"] = E
        del Em, am
    elif bps == 2:
        hb = spec.bps_halo
        Xin = (
            torch.nn.functional.pad(X * SX, (0, 0, 2, 2, 2, 2, 2, 2))
            if dgrad
            else (X * SX if osc else X)
        )
        P = Xin[hb : Di - hb, hb : Hi - hb, hb : Wi - hb]  # the producer's output
        Dp, Hp, Wp = P.shape[:3]
        ZTp = spec.bps_pz or spec.z_tile
        NBp = spec.bps_pn or C // spec.n_tile
        pmy = spec.bps_pmy or spec.my
        a_ = _absbits(P).reshape(Dp, Hp // pmy, pmy, Wp // 2, 2, C).amax(4).amax(2)
        a_ = a_.reshape(Dp, -1, C)
        ntbp = -(-a_.shape[1] // 32)
        a_ = torch.nn.functional.pad(
            a_, (0, 0, 0, ntbp * 32 - a_.shape[1], 0, -(-Dp // ZTp) * ZTp - Dp)
        )
        E = (
            a_.reshape(-1, ZTp, ntbp, 32, NBp, C // NBp)
            .amax(5)
            .amax(3)
            .amax(1)
            .reshape(-1)
        )
        E = E.repeat_interleave(spec.bps_pw or spec.waves).contiguous()  # one per wave
        assert E.numel() == cw.bps_map_entries(spec), (
            E.numel(),
            cw.bps_map_entries(spec),
        )
        args["E"] = E
        del Xin, P, a_
    if bpo:
        EO = torch.zeros(
            grid[0] * (grid[1] if len(grid) > 1 else 1) * spec.waves,
            device=dev,
            dtype=torch.int32,
        )
        args["EO"] = EO
    if frel:
        Fl = torch.zeros(grid[0], device=dev, dtype=torch.int32)
        args.update({"Xn": X, "F": Fl})
        res_plan = cw._frel_plan(spec)

    KS = getattr(spec, "ksplit", 1) if variant == "yx" else 1
    if KS > 1:
        nel = spec.Do * spec.Ho * spec.Wo * K
        Yp = torch.zeros(KS * nel, device=dev, dtype=torch.float32)
        args["Y"] = Yp
        kr_ = cw.build_ks_reduce(nel, KS)
        lred = KernelLauncher(
            hsaco=compile_kernel(kr_, arch=arch).hsaco,
            kernel_name=kr_.name,
            signature=SignatureBuilder().ptr("P", "f32").ptr("Y", "bf16").build(),
        )
        cred = LaunchConfig(grid=cw.ks_reduce_grid(nel), block=(256, 1, 1))
        _launch0 = launcher

        def launcher(a_, config):  # noqa: F811 -- main kernel + split-K reduction
            _launch0(a_, config=config)
            lred({"P": Yp, "Y": Y}, config=cred)

    def main():
        launcher(args, config=cfg)

    aux = None
    if variant == "x":
        kx = cw.build_x_to_c8_lds(Di, Hi, Wi, C)
        lx = KernelLauncher(
            hsaco=compile_kernel(kx, arch=arch).hsaco,
            kernel_name=kx.name,
            signature=SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16").build(),
        )
        cx = LaunchConfig(grid=cw.x_to_c8_lds_grid(Di, Hi, Wi, C), block=(256, 1, 1))
        kwx = cw.build_x_weight_transform(K, C)
        lwx = KernelLauncher(
            hsaco=compile_kernel(kwx, arch=arch).hsaco,
            kernel_name=kwx.name,
            signature=SignatureBuilder().ptr("W", "bf16").ptr("U", "f16").build(),
        )
        cwx = LaunchConfig(grid=cw.x_weight_transform_grid(K, C), block=(256, 1, 1))
        aux = (lx, cx, lwx, cwx)
    if variant == "yx":
        # rocKE auxiliary kernels for the whole call: X relayout + weight transform
        am1 = bps == 1
        if hx:
            h2 = 2 * hx
            kx = cw.build_x_to_c8_lds(
                Di - 2 * hz, Hi - h2, Wi - h2, C, halo=hxs, scale=osc, amax=am1
            )
            xgrid = cw.x_to_c8_lds_grid(Di - 2 * hz, Hi - h2, Wi - h2, C)
        elif lay == "r8":
            kx = cw.build_x_to_c8(Di, Hi, Wi, C, rows=True)
            xgrid = cw.x_to_c8_grid(Di, Hi, Wi, C)
        else:  # LDS-transposed relayout (coalesced reads and writes)
            kx = cw.build_x_to_c8_lds(Di, Hi, Wi, C, scale=osc, amax=am1)
            xgrid = cw.x_to_c8_lds_grid(Di, Hi, Wi, C)
        sgx = SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16")
        if osc:
            sgx = sgx.scalar("S", "f32")
        if am1:
            sgx = sgx.ptr("E", "i32")
        lx = KernelLauncher(
            hsaco=compile_kernel(kx, arch=arch).hsaco,
            kernel_name=kx.name,
            signature=sgx.build(),
        )
        odt = "f16" if spec.op_dtype == "f16" else "bf16"
        wt3 = (
            wtv in (3, 4)
            and odt == "f16"
            and spec.stage == "lds16"
            and not spec.b16
            and not spec.bc
        )
        my_ = getattr(spec, "my", 2)
        if my_ != 2 and not wt3:
            raise ValueError("my=4: --wtv 3 or 4")
        if wt3:
            kw_ = cw.build_yx_weight_transform_v3(
                K, C, flip=dgrad, swz=wtv == 4, my=my_
            )
        else:
            kw_ = cw.build_yx_weight_transform(
                K,
                C,
                out_dtype=odt,
                fold=spec.stage in ("reg", "wlds"),
                u16=spec.b16,
                uc=spec.bc,
                flip=dgrad,
            )
        lw = KernelLauncher(
            hsaco=compile_kernel(kw_, arch=arch).hsaco,
            kernel_name=kw_.name,
            signature=SignatureBuilder().ptr("W", "bf16").ptr("U", odt).build(),
        )
        cx = LaunchConfig(grid=xgrid, block=(256, 1, 1))
        cwt = LaunchConfig(
            grid=(
                cw.yx_weight_transform_v3_grid if wt3 else cw.yx_weight_transform_grid
            )(K, C),
            block=(256, 1, 1),
        )
        aux = (lx, cx, lw, cwt)
        if frel:
            ka = cw.build_yx_frel_aux(spec)
            la = KernelLauncher(
                hsaco=compile_kernel(ka, arch=arch).hsaco,
                kernel_name=ka.name,
                signature=SignatureBuilder()
                .ptr("X", "bf16")
                .ptr("X8", "bf16")
                .ptr("W", "bf16")
                .ptr("U", odt)
                .ptr("F", "i32")
                .build(),
            )
            ca = LaunchConfig(grid=cw.yx_frel_aux_grid(spec), block=(256, 1, 1))
            aux = ("frel", la, ca)
        elif lay == "c8" and not spec.b16 and not spec.bc:
            # one launch for both passes (relayout and weight transform side by side)
            hd = 2 * hx
            ka = cw.build_yx_aux(
                Di - 2 * hz,
                Hi - hd,
                Wi - hd,
                C,
                K,
                out_dtype=odt,
                fold=spec.stage in ("reg", "wlds"),
                rev=getattr(spec, "xrev", False),
                halo=hxs,
                flip=dgrad,
                wtv=(wtv if wt3 else 1),
                scale=osc,
                amax=am1,
                my=my_,
            )
            sga = (
                SignatureBuilder()
                .ptr("X", "bf16")
                .ptr("X8", "bf16")
                .ptr("W", "bf16")
                .ptr("U", odt)
            )
            if osc:
                sga = sga.scalar("S", "f32")
            if am1:
                sga = sga.ptr("E", "i32")
            la = KernelLauncher(
                hsaco=compile_kernel(ka, arch=arch).hsaco,
                kernel_name=ka.name,
                signature=sga.build(),
            )
            ca = LaunchConfig(
                grid=cw.yx_aux_grid(
                    Di - 2 * hz, Hi - hd, Wi - hd, C, K, wtv=(wtv if wt3 else 1)
                ),
                block=(256, 1, 1),
            )
            aux = ("one", la, ca)

    def run_aux():
        if aux[0] == "frel":
            aux[1]({"X": X, "X8": Xk, "W": W, "U": U, "F": Fl}, config=aux[2])
            return
        ea = {"E": args["E"]} if bps == 1 else {}
        if aux[0] == "one":
            aux[1](
                {"X": X, "X8": Xk, "W": W, "U": U, **({"S": SX} if osc else {}), **ea},
                config=aux[2],
            )
            return
        if lay in ("c8", "r8"):
            aux[0](
                {"X": X, "X8": Xk, **({"S": SX} if osc else {}), **ea}, config=aux[1]
            )
        aux[2]({"W": W, "U": U}, config=aux[3])

    def full():
        if aux is not None:
            run_aux()
            launcher(args, config=cfg)
            return
        if lay == "c8":
            Xk.copy_(cw.winograd3d_x_to_c8(X))
        elif lay == "r8":
            Xk.copy_(cw.winograd_x_to_r8(X))
        U.copy_(wt())
        launcher(args, config=cfg)

    def aux_only():
        run_aux()

    pipe_fn = None
    if boff and pipe:
        # pipelined whole call: weight transform on the main stream; X relayout in z-slab
        # pieces on stream B; main kernel piece i (alternating streams A1/A2, so pieces
        # overlap at their tails) waits only for relayout piece i. Everything forks from and
        # joins back to the current stream (the staged timer's stream).
        nzb = spec.Do // spec.z_tile
        wps = grid[0] // nzb
        cuts = (
            [0] + sorted({max(1, min(nzb - 1, round(f * nzb))) for f in pipe}) + [nzb]
        )
        PV = Hi * Wi
        kr = cw.build_x_to_c8_range(Di, Hi, Wi, C)
        lr = KernelLauncher(
            hsaco=compile_kernel(kr, arch=arch).hsaco,
            kernel_name=kr.name,
            signature=SignatureBuilder()
            .ptr("X", "bf16")
            .ptr("X8", "bf16")
            .scalar("V0", "i32")
            .scalar("V1", "i32")
            .build(),
        )
        kwp = cw.build_yx_weight_transform(K, C, out_dtype="f16")
        lwp = KernelLauncher(
            hsaco=compile_kernel(kwp, arch=arch).hsaco,
            kernel_name=kwp.name,
            signature=SignatureBuilder().ptr("W", "bf16").ptr("U", "f16").build(),
        )
        cwp = LaunchConfig(grid=cw.yx_weight_transform_grid(K, C), block=(256, 1, 1))
        pieces = []
        v_done = 0
        for a, bnd in zip(cuts[:-1], cuts[1:]):
            v1 = min(Di * PV, (bnd * spec.z_tile + 2) * PV)
            pieces.append((v_done, v1, a * wps, (bnd - a) * wps))
            v_done = v1
        sB, sA1, sA2 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        res_pipe = [list(p_) for p_ in pieces]

        def pipe_fn():
            cur = torch.cuda.current_stream()
            fork = torch.cuda.Event()
            fork.record(cur)
            lwp({"W": W, "U": U}, config=cwp)  # weights on the current stream
            wdone = torch.cuda.Event()
            wdone.record(cur)
            evs = []
            sB.wait_event(fork)
            with torch.cuda.stream(sB):
                for v0, v1, _, _ in pieces:
                    lr(
                        {"X": X, "X8": Xk, "V0": v0, "V1": v1},
                        config=LaunchConfig(
                            grid=(-(-(v1 - v0) // 64), 1, 1), block=(256, 1, 1)
                        ),
                    )
                    e = torch.cuda.Event()
                    e.record(sB)
                    evs.append(e)
            done = []
            for i_, (_, _, b0, nbk) in enumerate(pieces):
                st = (sA1, sA2)[i_ % 2]
                st.wait_event(evs[i_])
                st.wait_event(wdone)
                with torch.cuda.stream(st):
                    launcher(
                        dict(args, BOFF=b0),
                        config=LaunchConfig(
                            grid=(nbk, 1, 1), block=(spec.block_size, 1, 1)
                        ),
                    )
                    e = torch.cuda.Event()
                    e.record(st)
                    done.append(e)
            for e in done:
                cur.wait_event(e)

    res = {
        **({"pipe_pieces": res_pipe} if pipe_fn is not None else {}),
        "shape": list(shape),
        "label": label,
        "kernel": kern.name,
        "build_s": round(t_build, 1),
        "flops": spec.flops,
    }
    if frel:
        res["frel_plan"] = list(res_plan)
    if verify and pipe_fn is not None:
        # pipelined whole call from a poisoned c8 buffer
        Xk.view(torch.int16).fill_(-1)
        U.zero_()
        Y.zero_()
        pipe_fn()
        torch.cuda.synchronize()
        res["x8_exact"] = bool(torch.equal(Xk, cw.winograd3d_x_to_c8(X)))
    if verify and frel:
        # whole call from a poisoned c8 buffer (NaN): any read of a voxel before its
        # producer published it shows up as a failure
        Xk.view(torch.int16).fill_(-1)
        U.zero_()
        full()
        torch.cuda.synchronize()
        res["x8_exact"] = bool(torch.equal(Xk, cw.winograd3d_x_to_c8(X)))
        res["flags_set"] = int(Fl.sum().item())
    elif verify and aux is not None:
        # the auxiliary kernels must reproduce the torch reference layouts exactly
        X8r, Ur = Xk.clone(), U.clone()
        if lay in ("c8", "r8"):
            Xk.zero_()
        U.zero_()
        aux_only()
        torch.cuda.synchronize()
        if bps == 1:
            res["map_exact"] = bool(torch.equal(args["E"], E_ref))
        res["aux_exact"] = bool(torch.equal(Xk, X8r)) and bool(
            (U.float() - Ur.float()).abs().max().item()
            <= 1e-3 * Ur.float().abs().max().item()
        )
        if variant == "yx":  # the stand-alone weight transform (chained call) too
            U.zero_()
            lw({"W": W, "U": U}, config=cwt)
            torch.cuda.synchronize()
            res["wt_exact"] = bool(
                (U.float() - Ur.float()).abs().max().item()
                <= 1e-3 * Ur.float().abs().max().item()
            )
    if verify:
        if (
            not frel and pipe_fn is None
        ):  # Y is the poisoned whole call's output (above)
            main()
        torch.cuda.synchronize()
        xr = X.float().permute(3, 0, 1, 2)[None]
        wr = W.float().permute(0, 4, 1, 2, 3)
        pd_ = (adapter_pd, 1, 1) if adapter else 0
        if dgrad:
            ref = torch.nn.functional.conv_transpose3d(xr, wr, padding=pd_)[0].permute(
                1, 2, 3, 0
            )
        else:
            ref = torch.nn.functional.conv3d(xr, wr, padding=pd_)[0].permute(1, 2, 3, 0)
        if yc8h:
            Yc = Y.float()
            hal = Yc.clone()
            hal[:, hh:-hh, hh:-hh, hh:-hh, :] = 0
            res["halo_zero"] = bool(hal.abs().max().item() == 0)
            Yv = (
                Yc[:, hh:-hh, hh:-hh, hh:-hh, :]
                .permute(1, 2, 3, 0, 4)
                .reshape(spec.Do, spec.Ho, spec.Wo, K)
            )
            del Yc, hal
        else:
            Yv = Y.float()
        if bpo:  # the output map this producer wrote vs torch on its bf16 output
            Yb = Yv.to(torch.bfloat16)
            ZT_ = spec.z_tile
            Dq = -(-spec.Do // ZT_) * ZT_
            a_ = (
                _absbits(Yb)
                .reshape(spec.Do, spec.Ho // spec.my, spec.my, spec.Wo // 2, 2, K)
                .amax(4)
                .amax(2)
            )
            a_ = a_.reshape(spec.Do, -1, K)
            ntb_ = -(-a_.shape[1] // 32)
            a_ = torch.nn.functional.pad(
                a_, (0, 0, 0, ntb_ * 32 - a_.shape[1], 0, Dq - spec.Do)
            )
            ref_eo = (
                a_.reshape(-1, ZT_, ntb_, 32, K // spec.n_tile, spec.n_tile)
                .amax(5)
                .amax(3)
            )
            ref_eo = ref_eo.amax(1).reshape(-1)
            d_ = EO.reshape(-1, spec.waves).amax(1).to(torch.int64) - ref_eo.to(
                torch.int64
            )
            # (yc8h_crop: the unstored ring still counts in the kernel's max -> upper bound only)
            res["map_out_ok"] = bool(
                (d_ >= 0).all().item()
                and (getattr(spec, "yc8h_crop", False) or (d_ <= 1).all().item())
            )
            del Yb, a_, ref_eo, d_
        if yc8h and getattr(spec, "yc8h_crop", False):
            # the outer ring is not stored (chained BWD-data layout): compare the rest
            Yv = Yv[1:-1, 1:-1, 1:-1].contiguous()
            ref = ref[1:-1, 1:-1, 1:-1].contiguous()
        res.update(verify_verdict(Yv, ref))
        res.update(_acc_stats(Yv, ref))
        if acc_out:
            torch.save({"Y": Yv.to(torch.bfloat16).cpu()}, acc_out)
        del ref, xr, wr
    from statistics import median

    res["dir"] = "dgrad" if dgrad else "fwd"
    res["form"] = f"adapter{'' if adapter_pd == 1 else '011'}" if adapter else "halo"
    res["scale_exp"] = scale_exp
    res["data"] = data
    if det_runs and aux is not None and aux[0] != "frel" and pipe_fn is None:
        # determinism: the whole drop-in call (aux + main [+ split-K reduce]) det_runs times
        # into a poisoned output, bitwise against the first run; sha256 of the output for
        # cross-GPU / cross-session comparison
        import hashlib

        outs = []
        for _ in range(det_runs):
            Y.view(torch.int16).fill_(-1)
            if KS > 1:
                Yp.fill_(float("nan"))
            full()
            torch.cuda.synchronize()
            outs.append(Y.clone())
        res["det_runs"] = det_runs
        res["det_identical"] = all(
            torch.equal(outs[0].view(torch.int16), o.view(torch.int16))
            for o in outs[1:]
        )
        res["out_sha256"] = hashlib.sha256(
            outs[0].cpu().view(torch.int16).numpy().tobytes()
        ).hexdigest()[:16]
        del outs
    if not timing:
        print(json.dumps(res), flush=True)
        return res
    for _ in range(3):
        main()
    torch.cuda.synchronize()
    tm = median(time_launches_staged(main, warmup=warmup, iters=iters))
    from rocke.runtime.launcher import time_launches

    res["ms_main_block"] = time_launches(main, warmup=warmup, iters=iters)
    tf = median(time_launches_staged(full, warmup=warmup, iters=iters))
    res["ms_main"] = tm
    res["ms_full"] = tf
    if variant == "yx" and aux is not None and aux[0] != "frel":

        def wt_only():
            lw({"W": W, "U": U}, config=cwt)

        res["ms_wt"] = median(time_launches_staged(wt_only, warmup=warmup, iters=iters))
        # chained call: X already in c8 (a yc8h producer wrote it) -> weight transform + main
        _lw, _cw = lw, cwt

        def chain():
            _lw({"W": W, "U": U}, config=_cw)
            launcher(args, config=cfg)

        res["ms_chain"] = median(
            time_launches_staged(chain, warmup=warmup, iters=iters)
        )
        res["eff_tf_chain"] = spec.flops / res["ms_chain"] / 1e9
    if aux is not None:
        res["ms_aux"] = median(
            time_launches_staged(aux_only, warmup=warmup, iters=iters)
        )
    if adapter and not dgrad:
        # the direct alternative's extra pass: zero-pad copy of X into the halo-form input
        # (torch strided copy into the interior of a preallocated, zero-haloed buffer), after
        # which a halo-form (pad 0) direct kernel computes the adapter conv
        Xpad = torch.zeros(Di, Hi, Wi, C, device=dev, dtype=torch.bfloat16)
        Xin_ = Xpad[hz : Di - hz, hx : Hi - hx, hx : Wi - hx]

        def padcopy():
            Xin_.copy_(X)

        res["ms_padcopy"] = median(
            time_launches_staged(padcopy, warmup=warmup, iters=iters)
        )
        # the same pad copy as one rocKE kernel at copy bandwidth (cw.build_pad_copy)
        kpc = cw.build_pad_copy(
            Di - 2 * hz, Hi - 2 * hx, Wi - 2 * hx, C, halo=(hz, hx, hx)
        )
        lpc = KernelLauncher(
            hsaco=compile_kernel(kpc, arch=arch).hsaco,
            kernel_name=kpc.name,
            signature=SignatureBuilder().ptr("X", "bf16").ptr("Xp", "bf16").build(),
        )
        cpc = LaunchConfig(
            grid=cw.pad_copy_grid(Di - 2 * hz, Hi - 2 * hx, Wi - 2 * hx, C),
            block=(256, 1, 1),
        )
        Xpad.zero_()
        lpc({"X": X, "Xp": Xpad}, config=cpc)
        torch.cuda.synchronize()
        res["padcopy_rocke_exact"] = bool(torch.equal(Xin_, X))
        res["ms_padcopy_rocke"] = median(
            time_launches_staged(
                lambda: lpc({"X": X, "Xp": Xpad}, config=cpc),
                warmup=warmup,
                iters=iters,
            )
        )
        del Xpad, Xin_
    if pipe_fn is not None:
        res["ms_pipe"] = median(
            time_launches_staged(pipe_fn, warmup=warmup, iters=iters)
        )
        res["eff_tf_pipe"] = spec.flops / res["ms_pipe"] / 1e9
    res["eff_tf_main"] = spec.flops / tm / 1e9
    if power:
        res.update(_power_probe(main))
    res["eff_tf_full"] = spec.flops / tf / 1e9
    print(json.dumps(res), flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--shapes", default="3", help="3-D UNet shape ids (1..10), comma list"
    )
    p.add_argument("--shape", default=None, help="Di,Hi,Wi,C,K (overrides --shapes)")
    p.add_argument("--arch", default="gfx942")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument(
        "--power", action="store_true", help="also report clock / power / cycles"
    )
    p.add_argument("--results", default=None, help="append JSON lines here")
    p.add_argument(
        "--spec", default="{}", help="JSON dict of extra Winograd3dSpec fields"
    )
    p.add_argument(
        "--dgrad",
        action="store_true",
        help="BWD-data of the shape (yx: conv of dY padded by 2, flipped weights)",
    )
    p.add_argument(
        "--adapter",
        action="store_true",
        help="production pad-(1,1,1) form: X / dY one voxel smaller per side, the "
        "relayout adds the zero halo (drop-in only; FWD and --dgrad)",
    )
    p.add_argument(
        "--adapter-pd",
        type=int,
        default=None,
        choices=(0, 1),
        help="adapter form with pad (pd, 1, 1): 0 = a D-split run's D+2 input",
    )
    p.add_argument(
        "--data",
        default="u1",
        help="operands: u1 | tr [@exp2 scale of X], see _make_data",
    )
    p.add_argument(
        "--acc-out", default=None, help="save the output (bf16) here (torch.save)"
    )
    p.add_argument("--timing", type=int, default=1, help="0: verify / accuracy only")
    p.add_argument(
        "--wtv",
        type=int,
        default=1,
        help="weight transform: 1 (scalar) | 3 (16-B) | 4 (16-B, 8x8 lane swizzle)",
    )
    p.add_argument(
        "--scale-exp",
        type=int,
        default=0,
        help="spec oscale: relayout scales X by 2**e, the epilogue by 2**-e",
    )
    p.add_argument(
        "--det-runs",
        type=int,
        default=0,
        help="determinism: run the drop-in call N times, compare bitwise (+ sha256)",
    )
    p.add_argument(
        "--pipe",
        default="",
        help="yx boff: pipelined whole call, z cut points as "
        "fractions of the slabs, e.g. 0.1,0.4",
    )
    a = p.parse_args(argv)
    spec_kw = json.loads(a.spec)
    if a.shape:
        todo = [("custom", tuple(int(v) for v in a.shape.split(",")))]
    else:
        todo = [(f"s{int(s)}", UNET3D_SHAPES[int(s)]) for s in a.shapes.split(",")]
    out = []
    for label, shp in todo:
        r = run_one(
            shp,
            arch=a.arch,
            warmup=a.warmup,
            iters=a.iters,
            verify=not a.no_verify,
            spec_kw=spec_kw,
            label=label,
            power=a.power,
            pipe=[float(v) for v in a.pipe.split(",")] if a.pipe else None,
            dgrad=a.dgrad,
            acc_out=a.acc_out,
            data=a.data,
            timing=bool(a.timing),
            wtv=a.wtv,
            scale_exp=a.scale_exp,
            det_runs=a.det_runs,
            adapter=a.adapter or a.adapter_pd is not None,
            adapter_pd=1 if a.adapter_pd is None else a.adapter_pd,
        )
        if r is not None:
            r["spec"] = spec_kw
            r["wtv"] = a.wtv
            out.append(r)
    if a.results:
        with open(a.results, "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    return 1 if any(run_failed(r) for r in out) else 0


def run_failed(res: dict) -> bool:
    """A result whose verify verdict, output amax map or determinism check failed."""
    return (
        res.get("verify") == "FAIL"
        or res.get("map_out_ok") is False
        or res.get("det_identical") is False
    )


if __name__ == "__main__":
    raise SystemExit(main())
