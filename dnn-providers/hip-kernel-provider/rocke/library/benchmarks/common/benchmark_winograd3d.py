# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Opt-in benchmark for the 3-D Winograd F(2x2x2,3x3x3) forward prototype.

Runs ``kernels.common.conv_winograd3d`` on halo-form NDHWC bf16 problems (N=1, valid
conv, stride 1), verifies against a float32 torch conv3d with the benchmark's metric
(``max|out-ref| / max(|ref|max, 1)``, bf16 tolerance 5e-2), and times it with the
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

SCAFFOLD = {
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
                c = [x for x in amdsmi.amdsmi_get_gpu_metrics_info(h)["current_gfxclks"]
                     if isinstance(x, int)]
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
    ej = (e1["energy_accumulator"] - e0["energy_accumulator"]) * e0["counter_resolution"] * 1e-6
    k = len(clocks) // 10
    cs = sorted(clocks[k:len(clocks) - k] or clocks or [0])
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
    else:
        raise ValueError(kind)
    if e:
        X = X * 2.0 ** int(e)
    return X.to(torch.bfloat16), W.to(torch.bfloat16)


def _acc_stats(Yv, ref):
    """Error of Yv (f32 view of the bf16 output) vs the f32 reference: the rocKE verify metric
    (max|err| / max(max|ref|, 1)), the scale-free max|err| / max|ref|, mean|err| / mean|ref|,
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


def run_one(shape, *, arch, warmup, iters, verify, spec_kw, label="", power=False, pipe=None,
            dgrad=False, acc_out=None, data="u1", timing=True, wtv=1, scale_exp=0):
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
        Di, Hi, Wi, C, K = fDi + 2, fHi + 2, fWi + 2, fK, fC
        X, W = _make_data(data, (fDi - 2, fHi - 2, fWi - 2, fK), (fK, 3, 3, 3, fC), dev)
    else:
        Di, Hi, Wi, C, K = shape
        X, W = _make_data(data, (Di, Hi, Wi, C), (K, 3, 3, 3, C), dev)
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
                Wd, dtype=udt, fold=spec.stage in ("reg", "wlds"), u16=spec.b16, uc=spec.bc
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
        .ptr("U", "f16" if variant in ("yx", "x") and spec.op_dtype == "f16" else "bf16")
        .ptr("Y", "f32" if variant == "yx" and getattr(spec, "ksplit", 1) > 1 else "bf16")
    )
    if frel:
        sigb = sigb.ptr("Xn", "bf16").ptr("F", "i32")
    boff = variant == "yx" and getattr(spec, "boff", False)
    if boff:
        sigb = sigb.scalar("BOFF", "i32")
    osc = variant == "yx" and getattr(spec, "oscale", False)
    if osc:
        sigb = sigb.scalar("OS", "f32")
    SX = 2.0 ** scale_exp  # relayout scale (power of two); the main kernel multiplies by 1/SX
    sig = sigb.build()
    launcher = KernelLauncher(hsaco=art.hsaco, kernel_name=kern.name, signature=sig)
    yc8h = variant == "yx" and getattr(spec, "yc8h", False)
    if yc8h:  # next layer's input layout: c8 with a zero one-voxel halo
        hh = spec.yc8h_halo
        Y = torch.zeros(K // 8, spec.Do + 2 * hh, spec.Ho + 2 * hh, spec.Wo + 2 * hh, 8,
                        device=dev, dtype=torch.bfloat16)
    else:
        Y = torch.zeros(spec.Do, spec.Ho, spec.Wo, K, device=dev, dtype=torch.bfloat16)
    U = wt()
    if dgrad:
        Xk = cw.winograd3d_x_to_c8(torch.nn.functional.pad(X * SX, (0, 0, 2, 2, 2, 2, 2, 2)))
    else:
        Xk = {"c8": cw.winograd3d_x_to_c8, "r8": cw.winograd_x_to_r8}.get(lay, lambda t: t)(X * SX)
    grid = grid_fn(spec)
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    args = {"X": Xk, "U": U, "Y": Y}
    if boff:
        args["BOFF"] = 0
    if osc:
        args["OS"] = 1.0 / SX
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
            hsaco=compile_kernel(kr_, arch=arch).hsaco, kernel_name=kr_.name,
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
            hsaco=compile_kernel(kx, arch=arch).hsaco, kernel_name=kx.name,
            signature=SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16").build(),
        )
        cx = LaunchConfig(grid=cw.x_to_c8_lds_grid(Di, Hi, Wi, C), block=(256, 1, 1))
        kwx = cw.build_x_weight_transform(K, C)
        lwx = KernelLauncher(
            hsaco=compile_kernel(kwx, arch=arch).hsaco, kernel_name=kwx.name,
            signature=SignatureBuilder().ptr("W", "bf16").ptr("U", "f16").build(),
        )
        cwx = LaunchConfig(grid=cw.x_weight_transform_grid(K, C), block=(256, 1, 1))
        aux = (lx, cx, lwx, cwx)
    if variant == "yx":
        # rocKE auxiliary kernels for the whole call: X relayout + weight transform
        if dgrad:
            kx = cw.build_x_to_c8_lds(Di - 4, Hi - 4, Wi - 4, C, halo=2, scale=osc)
            xgrid = cw.x_to_c8_lds_grid(Di - 4, Hi - 4, Wi - 4, C)
        elif lay == "r8":
            kx = cw.build_x_to_c8(Di, Hi, Wi, C, rows=True)
            xgrid = cw.x_to_c8_grid(Di, Hi, Wi, C)
        else:  # LDS-transposed relayout (coalesced reads and writes)
            kx = cw.build_x_to_c8_lds(Di, Hi, Wi, C, scale=osc)
            xgrid = cw.x_to_c8_lds_grid(Di, Hi, Wi, C)
        lx = KernelLauncher(
            hsaco=compile_kernel(kx, arch=arch).hsaco, kernel_name=kx.name,
            signature=(SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16").scalar("S", "f32")
                       if osc else SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16")).build(),
        )
        odt = "f16" if spec.op_dtype == "f16" else "bf16"
        wt3 = wtv in (3, 4) and odt == "f16" and spec.stage == "lds16" and not spec.b16 \
            and not spec.bc
        if wt3:
            kw_ = cw.build_yx_weight_transform_v3(K, C, flip=dgrad, swz=wtv == 4)
        else:
            kw_ = cw.build_yx_weight_transform(
                K, C, out_dtype=odt, fold=spec.stage in ("reg", "wlds"), u16=spec.b16,
                uc=spec.bc, flip=dgrad,
            )
        lw = KernelLauncher(
            hsaco=compile_kernel(kw_, arch=arch).hsaco, kernel_name=kw_.name,
            signature=SignatureBuilder().ptr("W", "bf16").ptr("U", odt).build(),
        )
        cx = LaunchConfig(grid=xgrid, block=(256, 1, 1))
        cwt = LaunchConfig(grid=(cw.yx_weight_transform_v3_grid if wt3 else
                                 cw.yx_weight_transform_grid)(K, C), block=(256, 1, 1))
        aux = (lx, cx, lw, cwt)
        if frel:
            ka = cw.build_yx_frel_aux(spec)
            la = KernelLauncher(
                hsaco=compile_kernel(ka, arch=arch).hsaco, kernel_name=ka.name,
                signature=SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16")
                .ptr("W", "bf16").ptr("U", odt).ptr("F", "i32").build(),
            )
            ca = LaunchConfig(grid=cw.yx_frel_aux_grid(spec), block=(256, 1, 1))
            aux = ("frel", la, ca)
        elif lay == "c8" and not spec.b16 and not spec.bc:
            # one launch for both passes (relayout and weight transform side by side)
            hd = 4 if dgrad else 0
            ka = cw.build_yx_aux(Di - hd, Hi - hd, Wi - hd, C, K, out_dtype=odt,
                                 fold=spec.stage in ("reg", "wlds"),
                                 rev=getattr(spec, "xrev", False),
                                 halo=2 if dgrad else 0, flip=dgrad, wtv=(wtv if wt3 else 1),
                                 scale=osc)
            sga = (SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16")
                   .ptr("W", "bf16").ptr("U", odt))
            if osc:
                sga = sga.scalar("S", "f32")
            la = KernelLauncher(
                hsaco=compile_kernel(ka, arch=arch).hsaco, kernel_name=ka.name,
                signature=sga.build(),
            )
            ca = LaunchConfig(grid=cw.yx_aux_grid(Di - hd, Hi - hd, Wi - hd, C, K,
                                                  wtv=(wtv if wt3 else 1)),
                              block=(256, 1, 1))
            aux = ("one", la, ca)

    def run_aux():
        if aux[0] == "frel":
            aux[1]({"X": X, "X8": Xk, "W": W, "U": U, "F": Fl}, config=aux[2])
            return
        if aux[0] == "one":
            aux[1]({"X": X, "X8": Xk, "W": W, "U": U, **({"S": SX} if osc else {})},
                   config=aux[2])
            return
        if lay in ("c8", "r8"):
            aux[0]({"X": X, "X8": Xk, **({"S": SX} if osc else {})}, config=aux[1])
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
        cuts = [0] + sorted({max(1, min(nzb - 1, round(f * nzb))) for f in pipe}) + [nzb]
        PV = Hi * Wi
        kr = cw.build_x_to_c8_range(Di, Hi, Wi, C)
        lr = KernelLauncher(
            hsaco=compile_kernel(kr, arch=arch).hsaco, kernel_name=kr.name,
            signature=SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16")
            .scalar("V0", "i32").scalar("V1", "i32").build(),
        )
        kwp = cw.build_yx_weight_transform(K, C, out_dtype="f16")
        lwp = KernelLauncher(
            hsaco=compile_kernel(kwp, arch=arch).hsaco, kernel_name=kwp.name,
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
                    lr({"X": X, "X8": Xk, "V0": v0, "V1": v1},
                       config=LaunchConfig(grid=(-(-(v1 - v0) // 64), 1, 1), block=(256, 1, 1)))
                    e = torch.cuda.Event()
                    e.record(sB)
                    evs.append(e)
            done = []
            for i_, (_, _, b0, nbk) in enumerate(pieces):
                st = (sA1, sA2)[i_ % 2]
                st.wait_event(evs[i_])
                st.wait_event(wdone)
                with torch.cuda.stream(st):
                    launcher(dict(args, BOFF=b0),
                             config=LaunchConfig(grid=(nbk, 1, 1), block=(spec.block_size, 1, 1)))
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
        res["aux_exact"] = bool(torch.equal(Xk, X8r)) and bool(
            (U.float() - Ur.float()).abs().max().item() <= 1e-3 * Ur.float().abs().max().item()
        )
        if variant == "yx":  # the stand-alone weight transform (chained call) too
            U.zero_()
            lw({"W": W, "U": U}, config=cwt)
            torch.cuda.synchronize()
            res["wt_exact"] = bool(
                (U.float() - Ur.float()).abs().max().item() <= 1e-3 * Ur.float().abs().max().item())
    if verify:
        if not frel and pipe_fn is None:  # Y is the poisoned whole call's output (above)
            main()
        torch.cuda.synchronize()
        xr = X.float().permute(3, 0, 1, 2)[None]
        wr = W.float().permute(0, 4, 1, 2, 3)
        if dgrad:
            ref = torch.nn.functional.conv_transpose3d(xr, wr)[0].permute(1, 2, 3, 0)
        else:
            ref = torch.nn.functional.conv3d(xr, wr)[0].permute(1, 2, 3, 0)
        if yc8h:
            Yc = Y.float()
            hal = Yc.clone()
            hal[:, hh:-hh, hh:-hh, hh:-hh, :] = 0
            res["halo_zero"] = bool(hal.abs().max().item() == 0)
            Yv = Yc[:, hh:-hh, hh:-hh, hh:-hh, :].permute(1, 2, 3, 0, 4).reshape(
                spec.Do, spec.Ho, spec.Wo, K)
            del Yc, hal
        else:
            Yv = Y.float()
        err = (Yv - ref).abs()
        scale = max(ref.abs().max().item(), 1.0)
        rel = err.max().item() / scale
        res["rel_err"] = rel
        res["mean_rel"] = err.mean().item() / ref.abs().mean().item()
        res["verify"] = "PASS" if rel < 5e-2 else "FAIL"
        res.update(_acc_stats(Yv, ref))
        if acc_out:
            torch.save({"Y": Yv.to(torch.bfloat16).cpu()}, acc_out)
        del ref, err, xr, wr
    from statistics import median

    res["dir"] = "dgrad" if dgrad else "fwd"
    res["scale_exp"] = scale_exp
    res["data"] = data
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

        res["ms_chain"] = median(time_launches_staged(chain, warmup=warmup, iters=iters))
        res["eff_tf_chain"] = spec.flops / res["ms_chain"] / 1e9
    if aux is not None:
        res["ms_aux"] = median(time_launches_staged(aux_only, warmup=warmup, iters=iters))
    if pipe_fn is not None:
        res["ms_pipe"] = median(time_launches_staged(pipe_fn, warmup=warmup, iters=iters))
        res["eff_tf_pipe"] = spec.flops / res["ms_pipe"] / 1e9
    res["eff_tf_main"] = spec.flops / tm / 1e9
    if power:
        res.update(_power_probe(main))
    res["eff_tf_full"] = spec.flops / tf / 1e9
    print(json.dumps(res), flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shapes", default="3", help="ScaFFold shape ids, comma list")
    p.add_argument("--shape", default=None, help="Di,Hi,Wi,C,K (overrides --shapes)")
    p.add_argument("--arch", default="gfx942")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--power", action="store_true", help="also report clock / power / cycles")
    p.add_argument("--results", default=None, help="append JSON lines here")
    p.add_argument("--spec", default="{}", help="JSON dict of extra Winograd3dSpec fields")
    p.add_argument("--dgrad", action="store_true",
                   help="BWD-data of the shape (yx: conv of dY padded by 2, flipped weights)")
    p.add_argument("--data", default="u1", help="operands: u1 | tr [@exp2 scale of X], see _make_data")
    p.add_argument("--acc-out", default=None, help="save the output (bf16) here (torch.save)")
    p.add_argument("--timing", type=int, default=1, help="0: verify / accuracy only")
    p.add_argument("--wtv", type=int, default=1, help="weight transform: 1 (scalar) | 3 (16-B) | 4 (16-B, 8x8 lane swizzle)")
    p.add_argument("--scale-exp", type=int, default=0,
                   help="spec oscale: relayout scales X by 2**e, the epilogue by 2**-e")
    p.add_argument("--pipe", default="", help="yx boff: pipelined whole call, z cut points as "
                   "fractions of the slabs, e.g. 0.1,0.4")
    a = p.parse_args(argv)
    spec_kw = json.loads(a.spec)
    if a.shape:
        todo = [("custom", tuple(int(v) for v in a.shape.split(",")))]
    else:
        todo = [(f"s{int(s)}", SCAFFOLD[int(s)]) for s in a.shapes.split(",")]
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
        )
        if r is not None:
            r["spec"] = spec_kw
            r["wtv"] = a.wtv
            out.append(r)
    if a.results:
        with open(a.results, "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
