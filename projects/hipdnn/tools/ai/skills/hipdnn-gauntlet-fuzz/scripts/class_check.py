#!/usr/bin/env python3
"""Check every WRONG row of a gauntlet run against the ENGINE'S OWN tolerance classes.

  class_check.py --root <work dirs> --run <dir> [--run <dir> ...] --repo <worktree> --toml <engine.toml>
                 [--out class_check.json]

Why: the fuzz grades with flat tolerances, but an engine config (ASM_SDPA_ENGINE.toml, ...) holds per-class
`[[tolerance_overrides]]` rows (`*_told`, `*_tolp`,
`*_tolu`, `*_tols`, `*_tolk1`, `*_tolf32`) that match bundle case ids, never fuzz ids. A WRONG row may be inside the
tolerance the engine itself claims. This script assigns each failing graph the class the bundle generator would
(reference-data-scripts/sdpa_bwd_plan.py: rule order F32, K1, U, S, P, DL, D, then the guard and margin from a
NumPy magnitude estimate), reads that class's atol/rtol from the engine TOML (later rows win, so the TOML's own
overrides count), and compares them with the observed max abs diff.

Output per WRONG row: class (and whether it was escalated, or no class fits and the generator would rescale dO),
the class atol/rtol, the observed per-tensor max abs diff, and a verdict:
  WITHIN   every observed max diff <= the class atol: the engine meets its own contract for this graph.
  EXCEEDS  some max diff > the class atol: it may still pass through rtol*|ref| (the log has no |ref|), so
           replay with `triage_wrong.py --class-tol <out.json>`, which grades the row with the class tolerance.
  NO CLASS no class holds the guard and the margin at this magnitude (the engine contract does not cover this
           graph: the generator would scale dO by 2^k, which a fuzz case cannot).
  UNMODELLED  dropout, padding, ragged or a boolean bias: the generator gives these its provisional class (D or F32)
           without an estimate; the verdict is then weak.
Needs NumPy.
SdpaBackward only: other ops have no class assigner; read their TOML rows by hand and say so in the report.
"""
import argparse, glob, json, math, os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from analyze import tensors as parse_tensors  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--run", action="append", required=True)
ap.add_argument("--repo", required=True, help="worktree holding dnn-providers/integration-tests/reference-data-scripts")
ap.add_argument("--toml", required=True, help="the engine's config TOML")
ap.add_argument("--out", default="class_check.json")
a = ap.parse_args()

sys.path.insert(0, os.path.join(a.repo, "dnn-providers/integration-tests/reference-data-scripts"))
import sdpa_bwd_plan as P  # noqa: E402

DT = {"half": "fp16", "bfloat16": "bf16", "float": "fp32"}


def toml_class_rows(path):
    """{(dtype token, class token): (atol, rtol)} from the [[tolerance_overrides]] rows; later rows win."""
    rows = {}
    for blk in re.split(r"\[\[tolerance_overrides\]\]", open(path, encoding="utf8").read())[1:]:
        f = re.search(r"filters\s*=\s*\[(.*?)\]", blk, re.S)
        at = re.search(r"^\s*atol\s*=\s*([0-9.eE+-]+)", blk, re.M)
        rt = re.search(r"^\s*rtol\s*=\s*([0-9.eE+-]+)", blk, re.M)
        if not (f and at and rt):
            continue
        for filt in re.findall(r'"([^"]+)"', f.group(1)):
            m = re.search(r"_(fp16|bfp16|fp32)_\*_(tol\w+)$", filt)
            if m:
                rows[(m.group(1), m.group(2))] = (float(at.group(1)), float(rt.group(1)))
    return rows


def inputs_spec(meta, uid):
    """The plan's input spec for one tensor uid from the case's fill recipe: [lo, hi] | {"value": v} | None."""
    r = (meta.get("inputs") or {}).get(str(uid))
    if not r:
        return None
    k = r.get("kind")
    if k in ("free", "nonzero"):
        return [float(r.get("lo", -1)), float(r.get("hi", 1))]
    if k == "fixed":
        return {"value": float(r.get("value", 0))}
    if k in ("zeros", "subnormal"):
        return {"value": 0.0}
    if k == "large":
        s = float(r.get("scale", 1))
        return [float(r.get("lo", -1)) * s, float(r.get("hi", 1)) * s]
    return None


def case_of(graph, meta):
    """The plan's case dict, params for the estimator and the modelling flags of one concrete graph."""
    node = graph["nodes"][0]
    inp, par = node["inputs"], node["parameters"]
    tens = {t["uid"]: t for t in graph["tensors"]}
    q, k, v, o = (tens[inp[x]]["dims"] for x in ("q_tensor_uid", "k_tensor_uid", "v_tensor_uid", "o_tensor_uid"))
    dt = DT[graph["io_data_type"]]
    c = dict(B=q[0], Hq=q[1], Hk=k[1], Hv=v[1], Sq=q[2], Skv=k[2], Dqk=q[3], Dv=v[3])
    at = {}
    if par.get("causal_mask"):
        at["causal_mask"] = True
    if par.get("causal_mask_bottom_right"):
        at["causal_mask_bottom_right"] = True
    c.update(W=par.get("left_bound"), R=par.get("right_bound"),
             align="BR" if par.get("diagonal_alignment") == "BOTTOM_RIGHT" else "TL", attrs=at)
    sc = par.get("attn_scale_value")
    c["scale"] = "unset" if sc is None else float(sc)
    c["dropout"] = bool(par.get("dropout_probability")) or inp.get("dropout_mask_tensor_uid") is not None
    c["padding"] = bool(par.get("padding_mask")) or inp.get("seq_len_q_tensor_uid") is not None
    c["ragged"] = any(t.get("ragged_offset_tensor_uid") is not None for t in graph["tensors"])
    spec = {x: inputs_spec(meta, inp[f"{x}_tensor_uid"]) for x in ("q", "k", "v", "do")}
    c["inputs"] = {k_: v_ for k_, v_ in spec.items() if v_ is not None}
    bias_uid = inp.get("attn_mask_tensor_uid")
    bias = tens[bias_uid] if bias_uid is not None else None
    c["bias_bool"] = bias is not None and bias["data_type"] == "boolean"
    return c, dt, bias, meta


def estimate_of(c, dt, bias, meta, inp_uid_bias):
    W, R, al = P.effective_band(c)
    p = {k_: c[k_] for k_ in P.GEOMETRY}
    km = P.kernel_model(c, dt, True)
    p.update(dtype=dt, align=al, W=W, R=R, scale=P.scale_value(c), kernel_model=km, inputs=dict(c["inputs"]))
    if bias is not None:
        p["bias_dims"] = list(bias["dims"])
        p["bias_dtype"] = DT.get(bias["data_type"], dt)
        spec = inputs_spec(meta, inp_uid_bias)
        if spec is not None:
            p["inputs"]["bias"] = spec
    if p["B"] > 2 and p["B"] * p["Hq"] * p["Sq"] * p["Skv"] > 4e6:
        p["B"] = 2
        if p.get("bias_dims") and len(p["bias_dims"]) == 4 and p["bias_dims"][0] > 2:
            p["bias_dims"] = [2] + p["bias_dims"][1:]
    return P.estimate(p)


rows = toml_class_rows(a.toml)
out = {}
for run in a.run:
    s = json.load(open(f"{a.root}/{run}/summary.json"))
    for fd in sorted(glob.glob(f"{a.root}/{run}/failures/*")):
        name = os.path.basename(fd)
        gj = [f for f in glob.glob(fd + "/*.json") if not f.endswith((".meta.json", "row.json", "repro.json", "outcome.json"))]
        mj = glob.glob(fd + "/*.meta.json")
        if not gj or not mj:
            continue
        graph, meta = json.load(open(gj[0])), json.load(open(mj[0]))
        if graph["nodes"][0].get("type") != "SdpaBackwardAttributes" and "dq_tensor_uid" not in graph["nodes"][0].get("outputs", {}):
            continue  # SdpaBackward only
        case = name.rsplit("_", 1)[1]
        c, dt, bias, _ = case_of(graph, meta)
        node = graph["nodes"][0]
        unmodelled = c["dropout"] or c["padding"] or c["ragged"] or c["bias_bool"]
        info = {"unassigned": False}
        scores = P.estimate_scores(c)
        if unmodelled or scores > P.ESTIMATE_LIMIT:
            cls, info = P.provisional_class(dt), {"provisional": "unmodelled" if unmodelled else "too large to estimate"}
            f = 1.0
        else:
            est = estimate_of(c, dt, bias, meta, node["inputs"].get("attn_mask_tensor_uid"))
            cls, info, f = P.assign_class(c, dt, est)
            info = {k_: info[k_] for k_ in ("rule", "min_max_ref", "lambda", "escalated_from", "dO_scale", "unassigned") if k_ in info}
            info["max_ref"] = est["max_ref"]
            info["atol_needed"] = est.get("atol_needed")
        tok = P.TOKEN[cls]
        tol = rows.get((P.DTYPE_TOKEN[dt], tok)) or P.tolerance(cls, dt)
        log = ""
        try:
            log = open(fd + "/log_tail.txt", errors="replace").read()
        except OSError:
            pass
        ts = parse_tensors(log, case)
        worst = max((t["max"] for t in ts if t["max"] is not None), default=None)
        nonfinite = any(t["nonfinite"] for t in ts)
        if info.get("unassigned") or f != 1.0:
            verdict = "NO CLASS"
        elif unmodelled or "provisional" in info:
            verdict = "UNMODELLED"
        elif nonfinite or worst is None or tol is None:
            verdict = "EXCEEDS"
        else:
            verdict = "WITHIN" if worst <= tol[0] else "EXCEEDS"
        out[f"{run}/{name}"] = {
            "run": run, "case": case, "dtype": dt, "class": cls, "token": tok,
            "atol": tol[0] if tol else None, "rtol": tol[1] if tol else None,
            "observed_max": worst, "nonfinite": nonfinite, "verdict": verdict, "info": info,
            "tensors": ts,
        }
        print(f"{run} {case:8s} {dt} class {cls:3s} atol {tol[0] if tol else '?'} rtol {tol[1] if tol else '?'}  "
              f"observed max {worst}  -> {verdict}"
              + (f"  (escalated from {info['escalated_from']})" if info.get("escalated_from") else "")
              + (f"  [{info['provisional']}]" if "provisional" in info else ""), flush=True)
import collections
print("SUMMARY:", dict(collections.Counter(v["verdict"] for v in out.values())))
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
