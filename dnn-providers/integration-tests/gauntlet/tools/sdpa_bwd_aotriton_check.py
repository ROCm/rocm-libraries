#!/usr/bin/env python3
"""Grade dumped SDPA-backward cases with AOTriton's tolerance rule, offline.

    python gauntlet/tools/sdpa_bwd_aotriton_check.py --dumps DIR [--dumps DIR ...] [--run NAME] --out results.json [--arch gfx90a]

DIR is a `run_gauntlet_fuzz.py --dump-dir` (or HIPDNN_TEST_DUMP_DIR) tree: every <test name>/manifest.json under it is one case,
keyed `<run>/<case id>` in the output (the case id is the test name after its last '.'). Needs torch (CPU is enough) and numpy.
This is the offline form of a reference-relative validator; inside the harness it would replace the dumps.

The rule is the one projects/hipdnn/tools/sdpa_conformance applies through AOTriton's own tests (_validate in the vendored
_common_test.py). Per gradient (dQ, dK, dV, dBias):
    ref      = the case in fp32 (fp64 for fp32 tests): torch CPU autograd of aten::_scaled_dot_product_attention_math,
               recomputing the softmax from Q, K and V
    lp_ref   = the same math in the test dtype
    limit    = max(1e-5, fudge * max|ref - lp_ref|), fudge per tensor and arch as in the vendored suite
The engine is compared with ref AND with `given`, the fp64 backward of the O and LSE the engine was given (P = exp(S - LSE),
dS = P * (dP - rowsum(dO * O))). A tensor fails only if it misses both: an engine that matches `given` is faithful to its
inputs, and its distance to ref is the inputs' own error. The record also carries `harness_ref_vs_torch`, the distance from the
harness's reference to ref, which shows when a case is too ill-conditioned for any oracle.
"""
import argparse, glob, json, math, os, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument(
    "--dumps", action="append", required=True, help="a dump tree (repeatable)"
)
ap.add_argument(
    "--run",
    help="prefix of the output keys (default: the basename of the first --dumps)",
)
ap.add_argument("--out", required=True)
ap.add_argument(
    "--arch",
    default="gfx90a",
    help="the fudge factors' arch branch (gfx90a, or anything else for the default)",
)
ap.add_argument("--only", help="comma-separated case ids")
a = ap.parse_args()
DT = {"float": torch.float32, "half": torch.float16, "bfloat16": torch.bfloat16}
ATOL_FLOOR = 1e-5  # default_atol[torch.float32] in _common_test.py


def fudge(arch, dtype, hdim):
    """_compute_fudge_factors of the vendored SdpaContext (dQ, dK, dV, dBias); gfx90a branch, non-varlen."""
    q, k, v, b = 148.0, 48.0, 36.0, 17.0
    npot = (hdim & (hdim - 1)) != 0
    if "gfx90a" in arch:
        q = max(q, 80.0)
        k = 500.0 if npot else 340.0
        b = 45.0 if npot else 36.0
    if dtype == torch.float32:
        k, v, b = 180.0, 50.0, 24.0
    return {"dQ": q, "dK": k, "dV": v, "dBias": b}


def load(dumpdir):
    man = json.load(open(os.path.join(dumpdir, "manifest.json")))
    out = {}
    for t in man["tensors"]:
        raw = bytearray(open(os.path.join(dumpdir, t["file"]), "rb").read())
        out[(t["role"], t["uid"])] = (raw, t)
    return out, {t["name"]: t["uid"] for t in man["tensors"] if t.get("name")}, man


def tensor(entry, dtype):
    raw, t = entry
    flat = torch.frombuffer(raw, dtype=dtype).clone()
    return flat.as_strided(t["dims"], t["strides"]).clone()


def gather(entry, dtype, tinfo, offsets, lens):
    """A ragged tensor as dense zero-padded [B, H, S, D]: batch element b is lens[b] rows read from the packed
    buffer at element offset offsets[b] with the tensor's own head/row/column strides.
    """
    raw, t = entry
    flat = torch.frombuffer(raw, dtype=dtype).clone()
    B, H, S, D = t["dims"]
    sh, ss, sd = t["strides"][1], t["strides"][2], t["strides"][3]
    out = torch.zeros(B, H, S, D, dtype=dtype)
    for b in range(B):
        n = min(lens[b], S)
        if n > 0:
            out[b, :, :n, :] = flat.as_strided((H, n, D), (sh, ss, sd), offsets[b])
    return out


def keep_mask(sq, skv, left, right, top_left):
    i = torch.arange(sq)[:, None]
    j = torch.arange(skv)[None, :]
    off = 0 if top_left else skv - sq
    masked = torch.zeros(sq, skv, dtype=torch.bool)
    if right is not None and right >= 0:
        masked |= j >= i + 1 + off + right
    if left is not None and left >= 0:
        masked |= j <= i + off - left
    return ~masked


def sdpa(q, k, v, bias, keep, scale, gqa, dp=0.0, dmask=None):
    """The vendored windowed_sdpa_math with an explicit additive bias: a row that keeps no key attends key 0 and
    has its output multiplied by 0."""
    live = keep.any(dim=-1)
    keep = keep | ((~live)[:, None] & (torch.arange(keep.shape[1])[None, :] == 0))
    band = torch.zeros(keep.shape, dtype=q.dtype).masked_fill_(~keep, float("-inf"))
    mask = band if bias is None else bias + band
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    out = torch.ops.aten._scaled_dot_product_attention_math(
        q,
        k,
        v,
        dropout_p=dp,
        is_causal=False,
        attn_mask=mask,
        scale=scale,
        dropout_mask=dmask,
        enable_gqa=gqa,
    )[0]
    return out * live.to(out.dtype)[:, None]


def grads(q, k, v, b, keep, scale, gqa, dout, dtype, dp=0.0, dmask=None):
    q, k, v = (x.detach().to(dtype).requires_grad_(True) for x in (q, k, v))
    bb = None if b is None else b.detach().to(dtype).requires_grad_(True)
    out = sdpa(q, k, v, bb, keep, scale, gqa, dp, dmask)
    out.backward(dout.to(dtype))
    return {
        "dQ": q.grad.clone(),
        "dK": k.grad.clone(),
        "dV": v.grad.clone(),
        "dBias": None if bb is None else bb.grad.clone(),
    }


def given_grads(q, k, v, do, o, lse, bias, keep, scale, dp=0.0, dmask=None):
    """The backward the engine is told to compute, in fp64, from the O and LSE it was GIVEN (not recomputed):
    P = exp(S - LSE), D = rowsum(dO * O), dS = P * (dP - D). Its noise and input sensitivity are the engine's too, so an
    engine that matches it is faithful to its inputs even where it misses the recomputed truth.
    """
    f = torch.float64
    q, k, v, do, o, lse = (x.to(f) for x in (q, k, v, do, o, lse))
    hq, hk, hv = q.shape[1], k.shape[1], v.shape[1]
    ke, ve = k.repeat_interleave(hq // hk, 1), v.repeat_interleave(hq // hv, 1)
    sc = 1.0 if scale is None else scale
    S = (q @ ke.transpose(-1, -2)) * sc
    if bias is not None:
        S = S + bias.to(f)
    live = keep & torch.isfinite(lse)
    P = torch.where(
        live,
        torch.exp(torch.where(live, S - lse, torch.zeros((), dtype=f))),
        torch.zeros((), dtype=f),
    )
    dP = do @ ve.transpose(-1, -2)
    Pd = P
    if dp and dmask is not None:
        m = dmask.to(f) / (1.0 - dp)
        dP, Pd = dP * m, P * m
    D = (do * o).sum(-1, keepdim=True)
    dS = P * (dP - D)

    def per_kv(x, h):
        return x.reshape(x.shape[0], h, hq // h, x.shape[2], x.shape[3]).sum(2)

    out = {
        "dQ": sc * dS @ ke,
        "dK": per_kv(sc * dS.transpose(-1, -2) @ q, hk),
        "dV": per_kv(Pd.transpose(-1, -2) @ do, hv),
        "dBias": None if bias is None else dS.sum_to_size(bias.shape),
    }
    return out


def lmax(x):
    return x.abs().max().item()


results = {}
run = a.run or os.path.basename(os.path.normpath(a.dumps[0]))
manifests = sorted(
    {
        m
        for d in a.dumps
        for m in glob.glob(os.path.join(d, "**", "manifest.json"), recursive=True)
    }
)
for m in manifests:
    test = os.path.basename(os.path.dirname(m))
    case = test.rsplit(".", 1)[-1] if "." in test else test
    if a.only and case not in a.only.split(","):
        continue
    key = f"{run}/{case}"
    dd = [m]
    if not dd:
        results[key] = {"error": "no dump"}
        continue
    d, dump_uid, man = load(os.path.dirname(dd[0]))
    if "node" in man:
        # the dump carries the node and the tensor types: no other copy of the graph is needed
        nd = man["node"]
        inp = {
            "q_tensor_uid": nd["q"],
            "k_tensor_uid": nd["k"],
            "v_tensor_uid": nd["v"],
            "do_tensor_uid": nd["do"],
            "attn_mask_tensor_uid": nd["attn_mask"],
            "seq_len_q_tensor_uid": nd["seq_len_q"],
            "seq_len_kv_tensor_uid": nd["seq_len_kv"],
            "scale_tensor_uid": nd["scale_tensor"],
            "seed_tensor_uid": nd["seed"],
            "dropout_mask_tensor_uid": nd["dropout_mask"],
            "o_tensor_uid": nd["o"],
            "stats_tensor_uid": nd["stats"],
        }
        outp = {
            "dq_tensor_uid": nd["dq"],
            "dk_tensor_uid": nd["dk"],
            "dv_tensor_uid": nd["dv"],
            "dbias_tensor_uid": nd["dbias"],
        }
        par = {
            "alibi_mask": nd["alibi"],
            "padding_mask": nd["padding"],
            "causal_mask": nd["causal"],
            "causal_mask_bottom_right": nd["causal_bottom_right"],
            "dropout_probability": nd["dropout_p"],
            "attn_scale_value": nd["scale"],
            "left_bound": nd["left_bound"],
            "right_bound": nd["right_bound"],
            "diagonal_alignment": "TOP_LEFT" if nd["top_left"] else "BOTTOM_RIGHT",
        }
        names = {
            "FLOAT": "float",
            "HALF": "half",
            "BFLOAT16": "bfloat16",
            "INT32": "int32",
        }
        tdt = {}
        for (role, uid), (_, t) in d.items():
            if role != "philox_mask":
                tdt[uid] = {
                    "data_type": names.get(t.get("dtype"), "float"),
                    "ragged_offset_tensor_uid": t.get("ragged_off"),
                    "dims": t["dims"],
                    "strides": t["strides"],
                }
    else:
        results[key] = {
            "error": "dump without a node section (written by an older harness)"
        }
        continue
    if par.get("alibi_mask"):
        results[key] = {"error": "unsupported: alibi"}
        continue
    dtype = DT[tdt[inp["q_tensor_uid"]]["data_type"]]
    hp = torch.float64 if dtype == torch.float32 else torch.float32
    lens = None
    if inp.get("seq_len_q_tensor_uid") is not None:
        lens = (
            tensor(d[("input", inp["seq_len_q_tensor_uid"])], torch.int32)
            .flatten()
            .tolist(),
            tensor(d[("input", inp["seq_len_kv_tensor_uid"])], torch.int32)
            .flatten()
            .tolist(),
        )

    def load_t(role, uid, dt=None):
        info = tdt[uid]
        dt = dt or DT[info["data_type"]]
        ro = info.get("ragged_offset_tensor_uid")
        if ro is None:
            return tensor(d[(role, uid)], dt)
        offs = tensor(d[("input", ro)], torch.int32).flatten().tolist()
        is_q = uid in (
            inp["q_tensor_uid"],
            inp["do_tensor_uid"],
            outp["dq_tensor_uid"],
            inp.get("o_tensor_uid"),
            inp.get("stats_tensor_uid"),
        )
        return gather(d[(role, uid)], dt, info, offs, lens[0] if is_q else lens[1])

    q = load_t("input", inp["q_tensor_uid"])
    k = load_t("input", inp["k_tensor_uid"])
    v = load_t("input", inp["v_tensor_uid"])
    do = load_t("input", inp["do_tensor_uid"])
    o_in = load_t("input", inp["o_tensor_uid"])
    lse_in = load_t("input", inp["stats_tensor_uid"], torch.float32)
    bias = None
    if inp.get("attn_mask_tensor_uid") is not None:
        bias = tensor(
            d[("input", inp["attn_mask_tensor_uid"])],
            DT[tdt[inp["attn_mask_tensor_uid"]]["data_type"]],
        )
    sq, skv = q.shape[-2], k.shape[-2]
    left, right = par.get("left_bound"), par.get("right_bound")
    top_left = par.get("diagonal_alignment", "TOP_LEFT") == "TOP_LEFT"
    if par.get("causal_mask"):
        right = 0 if right is None else right
    if par.get("causal_mask_bottom_right"):
        right, top_left = 0, False
    keep = keep_mask(sq, skv, left, right, top_left)
    scale = par.get("attn_scale_value")
    gqa = q.shape[1] != k.shape[1]
    if inp.get("scale_tensor_uid") is not None:
        results[key] = {"error": "unsupported: scale tensor"}
        continue
    dp = par.get("dropout_probability") or 0.0
    dmask = None
    if dp:
        if ("philox_mask", 0) not in d:
            results[key] = {"error": "no philox mask in the dump"}
            continue
        dmask = tensor(d[("philox_mask", 0)], torch.float32) > 0.5
    live = {
        "dQ": torch.ones_like(q, dtype=torch.bool),
        "dK": torch.ones_like(k, dtype=torch.bool),
        "dV": torch.ones_like(v, dtype=torch.bool),
    }
    if lens is None:
        ref = grads(q, k, v, bias, keep, scale, gqa, do, hp, dp, dmask)
        lp = grads(q, k, v, bias, keep, scale, gqa, do, dtype, dp, dmask)
        giv = given_grads(q, k, v, do, o_in, lse_in, bias, keep, scale, dp, dmask)
    else:
        # padding: each batch element on its live lengths (bottom-right alignment measures from them); the
        # rows past them are not part of the result, so they are zero in both references and masked in the engine
        ref = {
            "dQ": torch.zeros(q.shape, dtype=hp),
            "dK": torch.zeros(k.shape, dtype=hp),
            "dV": torch.zeros(v.shape, dtype=hp),
            "dBias": None,
        }
        lp = {
            "dQ": torch.zeros(q.shape, dtype=hp),
            "dK": torch.zeros(k.shape, dtype=hp),
            "dV": torch.zeros(v.shape, dtype=hp),
            "dBias": None,
        }
        giv = {
            "dQ": torch.zeros(q.shape, dtype=torch.float64),
            "dK": torch.zeros(k.shape, dtype=torch.float64),
            "dV": torch.zeros(v.shape, dtype=torch.float64),
            "dBias": None,
        }
        for t in live.values():
            t[:] = False
        for b in range(q.shape[0]):
            lq, lk = lens[0][b], lens[1][b]
            if lq <= 0 or lk <= 0:
                continue
            kp = keep_mask(lq, lk, left, right, top_left)
            sl = lambda x, n: x[b : b + 1, :, :n]
            bb = None
            if bias is not None:
                bb = bias[b : b + 1] if bias.shape[0] > 1 else bias
                bb = bb[:, :, :lq] if bb.shape[2] > 1 else bb
                bb = bb[:, :, :, :lk] if bb.shape[3] > 1 else bb
            args = (sl(q, lq), sl(k, lk), sl(v, lk), bb, kp, scale, gqa, sl(do, lq))
            # the mask is drawn over the declared extents and indexed by (b, h, i, j): its live part is the slice
            dm = None if dmask is None else dmask[b : b + 1, :, :lq, :lk]
            gr, gl = grads(*args, hp, dp, dm), grads(*args, dtype, dp, dm)
            gg = given_grads(
                sl(q, lq),
                sl(k, lk),
                sl(v, lk),
                sl(do, lq),
                sl(o_in, lq),
                sl(lse_in, lq),
                bb,
                kp,
                scale,
                dp,
                dm,
            )
            for nm, n in (("dQ", lq), ("dK", lk), ("dV", lk)):
                ref[nm][b : b + 1, :, :n] = gr[nm]
                lp[nm][b : b + 1, :, :n] = gl[nm].to(hp)
                giv[nm][b : b + 1, :, :n] = gg[nm]
                live[nm][b : b + 1, :, :n] = True
    fud = fudge(a.arch, dtype, q.shape[-1])
    rec = {"dtype": str(dtype).split(".")[-1], "hdim": q.shape[-1], "tensors": {}}
    outs = {
        "dQ": outp["dq_tensor_uid"],
        "dK": outp["dk_tensor_uid"],
        "dV": outp["dv_tensor_uid"],
        "dBias": outp.get("dbias_tensor_uid"),
    }
    for name, uid in outs.items():
        if uid is None or ref[name] is None:
            continue
        eng = load_t("engine", uid).to(hp)
        if name in live:
            eng = torch.where(live[name], eng, torch.zeros((), dtype=hp))
        harness_ref = load_t("ref", uid).to(hp)
        if name in live:
            harness_ref = torch.where(
                live[name], harness_ref, torch.zeros((), dtype=hp)
            )
        ref_error = lmax(ref[name] - lp[name].to(hp))
        thr = max(ATOL_FLOOR, ref_error * fud[name])

        def judge(err):
            bad = math.isnan(err) and not math.isnan(ref_error)
            return (not bad) and err <= thr, (err / thr if thr else math.inf)

        err_truth = lmax(ref[name] - eng)
        ok_truth, ratio_truth = judge(err_truth)
        g = giv.get(name)
        if g is not None:
            g = g.to(torch.float64)
            if name in live:
                g = torch.where(live[name], g, torch.zeros((), dtype=torch.float64))
            err_given = lmax(g - eng.to(torch.float64))
            ok_given, ratio_given = judge(err_given)
        else:
            err_given, ok_given, ratio_given = math.nan, False, math.inf
        # an issue only if the engine misses BOTH the recomputed truth and the backward of its own inputs
        use_given = ok_given or (
            not ok_truth and not math.isnan(ratio_given) and ratio_given < ratio_truth
        )
        rec["tensors"][name] = {
            "test_error": err_given if use_given else err_truth,
            "ref_error": ref_error,
            "fudge": fud[name],
            "threshold": thr,
            "ratio": (
                min(ratio_truth, ratio_given)
                if not math.isnan(ratio_given)
                else ratio_truth
            ),
            "pass": ok_truth or ok_given,
            "pass_truth": ok_truth,
            "pass_given": ok_given,
            "test_error_truth": err_truth,
            "test_error_given": err_given,
            "harness_ref_vs_torch": lmax(ref[name] - harness_ref),
            "ref_max": lmax(ref[name]),
            "harness_ref_vs_given": (
                lmax(harness_ref.to(torch.float64) - g) if g is not None else math.nan
            ),
        }
    rec["pass"] = all(t["pass"] for t in rec["tensors"].values())
    results[key] = rec
    print(
        key,
        "PASS" if rec["pass"] else "FAIL",
        {k2: round(t["ratio"], 2) for k2, t in rec["tensors"].items()},
        flush=True,
    )
json.dump(results, open(a.out, "w"), indent=1)
print("wrote", a.out, len(results))
