#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate the batchnorm inference bundles from the test plan.

Reads the declarations in ``batchnorm_inference_plan.py`` and the level rules
below, and writes, for each level (quick, standard, comprehensive, full):

    BatchnormInference/Default              batchnorm inference
    BatchnormInferencePointwise/Default     batchnorm inference -> relu_fwd

Each directory gets ``graph.template.json``, ``sweep.json`` and a ``support.json``
(existing claims are kept for case ids that still exist; claims are harvested with
``--write-support-claims``, never written here). The generator is deterministic:
the same plan gives the same bytes. Every case carries a ``purpose`` (claim |
probe | regression), the plan items it implements (``plan_items``) and its size
class in its metadata. A graph is written once, in the lowest level and the
earliest rule that builds it; the plan items of its duplicates are merged into
it. See ``docs/test-plans/batchnorm-inference.md``.

Usage:
    generate_batchnorm_inference_bundles.py [--bundle-dir <integration-test-bundles>] [--report] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import batchnorm_inference_plan as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import support_sidecar  # noqa: E402

GENERATOR = "generate_batchnorm_inference_bundles.py"
REFERENCE_SOURCE = (
    "hipDNN reference executors, auto verification (GPU reference, then CPU)"
)
DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parents[1] / "integration-test-bundles"

# role -> uid, per topology
UIDS = {
    "plain": dict(y=0, x=1, bias=2, inv_variance=3, mean=4, scale=5),
    "relu": dict(y=0, out=1, bias=2, inv_variance=3, mean=4, scale=5, x=6),
}
INPUT_ROLES = ("x", "scale", "bias", "mean", "inv_variance")
STAT_ROLES = ("bias", "inv_variance", "mean", "scale")


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


def ph(name):
    return "${case." + name + "}"


def _tensor(uid, name, virtual):
    return {
        "data_type": ph("data_type"),
        "dims": ph("dims"),
        "name": name,
        "strides": ph("strides"),
        "uid": uid,
        "virtual": virtual,
    }


def template(topo):
    u = UIDS[topo]
    fused = topo == "relu"
    nodes = [
        {
            "compute_data_type": ph("attributes.bn_compute"),
            "inputs": {
                "bias_tensor_uid": u["bias"],
                "inv_variance_tensor_uid": u["inv_variance"],
                "mean_tensor_uid": u["mean"],
                "scale_tensor_uid": u["scale"],
                "x_tensor_uid": u["x"],
            },
            "name": "BatchnormInference_0",
            "outputs": {"y_tensor_uid": u["y"]},
            "type": "BatchnormInferenceAttributes",
        }
    ]
    names = {"y": "BatchnormInference_0::Y", "x": "x" if fused else "X"}
    if fused:
        nodes.append(
            {
                "compute_data_type": "float",
                "inputs": {
                    "axis_tensor_uid": None,
                    "elu_alpha": None,
                    "in_0_tensor_uid": u["y"],
                    "in_1_tensor_uid": None,
                    "in_2_tensor_uid": None,
                    "operation": "relu_fwd",
                    "relu_lower_clip": ph("attributes.relu_lower_clip"),
                    "relu_lower_clip_slope": ph("attributes.relu_lower_clip_slope"),
                    "relu_upper_clip": ph("attributes.relu_upper_clip"),
                    "softplus_beta": None,
                    "swish_beta": None,
                },
                "name": "Pointwise_1",
                "outputs": {"out_0_tensor_uid": u["out"]},
                "type": "PointwiseAttributes",
            }
        )
        names["out"] = "Pointwise_1::OUT_0"
    tensors = [
        _tensor(uid, names.get(role, role), fused and role == "y")
        for role, uid in sorted(u.items(), key=lambda kv: kv[1])
    ]
    return {
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "io_data_type": ph("io_data_type"),
        "is_override_shape_enabled": False,
        "name": "BatchnormFwd+ActivTest" if fused else "BatchnormInferenceTest",
        "nodes": nodes,
        "tensors": tensors,
    }


def placeholders(tpl):
    """Case-wide placeholder paths (not the per-tensor dims, strides, data_type)."""
    found = set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and v.startswith("${case.") and v.endswith("}"):
            path = v[len("${case.") : -1]
            if path not in ("dims", "strides", "data_type"):
                found.add(path)

    walk(tpl)
    return found


# --------------------------------------------------------------------------
# Case -> sweep entry
# --------------------------------------------------------------------------


def _seed(case_id, uid):
    return (
        int(hashlib.sha1(f"{case_id}/{uid}".encode()).hexdigest()[:8], 16) % (2**31 - 1)
        + 1
    )


def _recipe(spec, seed):
    if spec[0] == "fixed":
        return {"kind": "fixed", "value": spec[1], "seed": seed}
    return {"kind": "free", "lo": spec[1], "hi": spec[2], "seed": seed}


def tensor_specs(c):
    """uid -> (dims, strides, dtype) for every tensor of the case."""
    u = UIDS[c["topo"]]
    xd = P.x_dims(c)
    spec = {
        u["x"]: (xd, P.strides(xd, c["xl"], c["np"]), c["xt"]),
    }
    if c["topo"] == "plain":
        spec[u["y"]] = (xd, P.strides(xd, c["yl"], c["np"]), c["yt"])
    else:
        spec[u["y"]] = (xd, P.strides(xd, c["yl"]), c["inter_t"])
        spec[u["out"]] = (xd, P.strides(xd, c["yl"], c["np"]), c["yt"])
    for role in STAT_ROLES:
        spec[u[role]] = (P.stat_dims(c), P.stat_strides(c), c["pt"])
    return spec


def input_recipes(c, case_id):
    u = UIDS[c["topo"]]
    rec = P.value_recipes(c)
    inputs = {
        u[role]: _recipe(rec[role], _seed(case_id, u[role])) for role in INPUT_ROLES
    }
    return {str(k): v for k, v in sorted(inputs.items())}


def case_values(c):
    attrs = {"bn_compute": c["comp"]}
    if c["topo"] == "relu":
        lo, slope, hi = P.ACTS[c["act"]]
        attrs.update(
            relu_lower_clip=lo, relu_lower_clip_slope=slope, relu_upper_clip=hi
        )
    return {
        "io_data_type": c["xt"],
        "attributes": attrs,
        "tensors": [
            {"uid": uid, "dims": d, "strides": s, "data_type": t}
            for uid, (d, s, t) in sorted(tensor_specs(c).items())
        ],
    }


def _tok(v):
    return "x".join(str(i) for i in v)


def case_id(c):
    """``<shape>_<rank>d[_i<spatial>]_n<N>c<C>_<dtype>[_p..][_y..][_c..]_<layout>[...]``
    (section 8.3)."""
    parts = [c["ids"][0].lower(), f"{c['rank']}d"]
    if c["sp"]:
        parts.append("i" + _tok(c["sp"]))
    parts.append(f"n{c['n']}c{c['c']}")
    dt = P.SHORT[c["xt"]]
    if c["pt"] != "float":
        dt += f"_p{P.SHORT[c['pt']]}"
    if c["yt"] != c["xt"]:
        dt += f"_y{P.SHORT[c['yt']]}"
    if c["comp"] != "float":
        dt += f"_c{P.SHORT[c['comp']]}"
    parts.append(dt)
    parts.append(c["xl"] if c["xl"] == c["yl"] else f"x{c['xl']}y{c['yl']}")
    if c["np"]:
        parts.append(c["np"].replace("_", ""))
    if c["stats_rank"] != "full":
        parts.append("stat2d")
    if c["topo"] == "relu":
        parts.append(c["act"])
        if c["inter_t"] != "float":
            parts.append(f"v{P.SHORT[c['inter_t']]}")
    if P.VALUE_TOKEN[c["vals"]]:
        parts.append(P.VALUE_TOKEN[c["vals"]])
    return "_".join(parts)


def derived_items(c):
    """Feature items a case implements by its fields, beyond the ids its rule names."""
    items = set()
    if c["xl"] != c["yl"]:
        items.add("F8")
    if c["np"]:
        items.add("F9")
    if c["stats_rank"] != "full":
        items.add("F12")
    if c["yt"] != c["xt"]:
        items.add("F14")
    if c["pt"] != "float":
        items.add("F15")
    if c["comp"] != "float":
        items.add("F16")
    if c["vals"] != "default":
        items.add("F17")
    if c["topo"] == "relu":
        items.add("F20")
        if c["inter_t"] != "float":
            items.add("F21")
    return items


@dataclass
class Item:
    """A case, its written id and the plan items of the duplicates merged into it."""

    case: dict
    id: str
    merged: set = field(default_factory=set)

    def plan_items(self):
        return sorted(
            set(self.case["ids"]) | derived_items(self.case) | self.merged,
            key=_item_key,
        )


def _item_key(item):
    return (
        P.PLAN_ITEMS.index(item) if item in P.PLAN_ITEMS else len(P.PLAN_ITEMS),
        item,
    )


def sweep_case(item):
    c, cid = item.case, item.id
    meta = {
        "format_version": 1,
        "generator": GENERATOR,
        "generator_version": "1",
        "reference_source": REFERENCE_SOURCE,
        "operation": "BatchnormInference"
        + ("+relu_fwd" if c["topo"] == "relu" else ""),
        "plan_items": item.plan_items(),
        "purpose": P.purpose(c),
        "size_class": c["size"],
        "seed": _seed(cid, "case"),
        "inputs": input_recipes(c, cid),
    }
    notes = []
    probe = P.probe_reasons(c)
    if probe:
        notes.append("probe: " + "; ".join(probe))
    if c.get("src"):
        notes.append(c["src"])
    if notes:
        meta["notes"] = "; ".join(notes)
    total = P.tensor_bytes(c)
    if total > (1 << 30):
        meta["minimum_vram_mb"] = int(total * 2 / (1 << 20)) + 1
    return {"id": cid, "values": case_values(c), "metadata": meta}


# --------------------------------------------------------------------------
# Level rules (section 8.2)
# --------------------------------------------------------------------------


def pair(i):
    return P.PAIRS[i % len(P.PAIRS)]


def upair(i):
    return P.UPAIRS[i % len(P.UPAIRS)]


def s_cells():
    """(shape id, rank) of every S shape, in catalog order."""
    return [(sid, r) for sid in P.S_SHAPES for r in P.s_ranks(sid)]


def _quick():
    S = P.shape_S
    for r in P.RANKS:
        for lay, dt in P.PAIRS:
            yield S("S1", r, xl=lay, xt=dt, add=("F7", "F13"))
    for r in P.RANKS:
        yield S("S1", r, xl="cf", xt="half", yt="float")
        yield S("S1", r, xl="cl", xt="bfloat16", yt="float")
    yield S("S1", 2, xt="half", pt="half")
    yield S("S1", 2, np="np_row")
    for r in P.RANKS:
        for lay, dt in P.PAIRS:
            yield S("S1", r, topo="relu", xl=lay, xt=dt, add=("F7", "F13"))
    for ri, r in enumerate(P.RANKS):
        for k, act in enumerate(P.CLIPS):
            lay, dt = pair(ri * len(P.CLIPS) + k)
            yield S("S1", r, topo="relu", act=act, xl=lay, xt=dt)
    for act in P.PROBE_ACTS:
        yield S("S1", 2, topo="relu", act=act)


def _standard():
    S = P.shape_S
    for i, (sid, r) in enumerate((sid, r) for sid, r in s_cells() if sid != "S1"):
        lay, dt = pair(i)
        yield S(sid, r, xl=lay, xt=dt)
    for j, (sid, r) in enumerate(
        (sid, r) for sid in ("S4", "S6", "S9") for r in P.s_ranks(sid)
    ):
        lay, dt = upair(j)
        yield S(sid, r, xl=lay, xt=dt, yt="float")
    for k, hid in enumerate(P.H_SHAPES):
        lay, dt = pair(k)
        yield P.shape_H(hid, 2, xl=lay, xt=dt)
    for lay, dt in (pair(1), pair(3)):
        yield P.shape_M("M7", xl=lay, xt=dt)
    for r in P.RANKS:
        lay, dt = pair(r)
        yield S("S1", r, xl=lay, xt=dt, vals="constant")
    yield S("S1", 2, xl="cl", xt="half", vals="zero_scale")
    # rejection probes
    for r in P.RANKS:
        for dt in P.HALF_DTYPES:
            yield S("S1", r, xt=dt, pt=dt)
    for dt in P.HALF_DTYPES:
        yield S("S1", 2, xt=dt, pt=dt, comp=dt)
    for r in P.RANKS:
        for np in P.NONPACKED:
            yield S("S1", r, np=np)
    for r in P.RANKS:
        yield S("S1", r, xl="cf", yl="cl")
    for dt in P.CORE_DTYPES:
        yield P.rank_probe(0, xt=dt)
    yield P.rank_probe(4)
    yield S("S1", 2, stats_rank="2")
    # fused
    for act in P.CLIPS:
        for r in P.RANKS:
            for lay, dt in P.PAIRS:
                yield S("S1", r, topo="relu", act=act, xl=lay, xt=dt)
    for dt in ("float", "half"):
        yield S("S1", 2, topo="relu", act="relu_slope0", xt=dt)
    for r in P.RANKS:
        lay, dt = upair(r)
        yield S("S1", r, topo="relu", xl=lay, xt=dt, yt="float")
    for j, (sid, r) in enumerate(
        (sid, r) for sid in ("S4", "S6", "S9") for r in P.s_ranks(sid)
    ):
        lay, dt = pair(j)
        yield S(sid, r, topo="relu", act=P.CLAIM_ACTS[j % 4], xl=lay, xt=dt)
    for act in P.PROBE_ACTS:
        for r in P.RANKS:
            yield S("S1", r, topo="relu", act=act)
    for dt in P.HALF_DTYPES:
        yield S("S1", 2, topo="relu", xt=dt, inter_t=dt)
    yield S("S1", 2, topo="relu", xt="half", pt="half")


def _comprehensive():
    S = P.shape_S
    for sid, r in s_cells():
        for lay, dt in P.PAIRS:
            yield S(sid, r, xl=lay, xt=dt)
    for sid, r in s_cells():
        for lay, dt in P.UPAIRS:
            yield S(sid, r, xl=lay, xt=dt, yt="float")
    for lid in P.L_SHAPES:
        for r in P.RANKS:
            for lay, dt in P.PAIRS:
                yield P.shape_L(lid, r, xl=lay, xt=dt)
    for k, hid in enumerate(P.H_SHAPES):
        for lay, dt in P.PAIRS:
            yield P.shape_H(hid, 2, xl=lay, xt=dt)
        for r in (1, 3):
            for lay, dt in (pair(k), pair(k + 3)):
                yield P.shape_H(hid, r, xl=lay, xt=dt)
    for k, mid in enumerate(m for m in P.M_SHAPES if m != "M7"):
        for lay, dt in (pair(k), pair(k + 3)):
            yield P.shape_M(mid, xl=lay, xt=dt)
    for lay, dt in P.PAIRS:
        yield P.shape_M("M7", xl=lay, xt=dt)
    for r in P.RANKS:
        for dt in P.CORE_DTYPES:
            yield S("S1", r, xl="cf", xt=dt, vals="tinyvar")
            yield S("S1", r, xl="cl", xt=dt, vals="large")
    for dt in P.CORE_DTYPES:
        yield S("S1", 2, xt=dt, vals="cancel")
    # rejection probes
    for r in P.RANKS:
        for lay in P.LAYOUTS:
            for dt in P.HALF_DTYPES:
                yield S("S1", r, xl=lay, xt=dt, pt=dt)
                yield S("S1", r, xl=lay, xt=dt, pt=dt, comp=dt)
    for ri, r in enumerate(P.RANKS):
        for di, dt in enumerate(P.CORE_DTYPES):
            yield S("S1", r, xt=dt, np=P.NONPACKED[(ri + di) % 2])
            yield S("S1", r, xl="cf", yl="cl", xt=dt)
    # fused
    k = 0
    for sid, r in s_cells():
        for lay, dt in P.PAIRS:
            yield S(sid, r, topo="relu", act=P.CLAIM_ACTS[k % 4], xl=lay, xt=dt)
            k += 1
    for r in P.RANKS:
        for lay, dt in P.PAIRS:
            yield P.shape_L(
                "L1", r, topo="relu", act=P.CLAIM_ACTS[k % 4], xl=lay, xt=dt
            )
            k += 1
    for j, lid in enumerate(("L4", "L5", "L6")):
        lay, dt = pair(j)
        yield P.shape_L(lid, 2, topo="relu", act=P.CLAIM_ACTS[j % 4], xl=lay, xt=dt)
    for j, mid in enumerate(("M1", "M2", "M5")):
        for lay, dt in (pair(j), pair(j + 3)):
            yield P.shape_M(mid, topo="relu", xl=lay, xt=dt)
    for r in P.RANKS:
        for lay, dt in P.PAIRS:
            yield S("S1", r, topo="relu", act="leaky", xl=lay, xt=dt)
    for r in P.RANKS:
        for dt in P.CORE_DTYPES:
            yield S("S1", r, topo="relu", act="relu_lo0p1", xt=dt)


def _full():
    for lid in P.L_SHAPES:
        for r in P.RANKS:
            for lay, dt in P.UPAIRS:
                yield P.shape_L(lid, r, xl=lay, xt=dt, yt="float")
    for mid in P.M_SHAPES:
        for lay, dt in P.PAIRS:
            yield P.shape_M(mid, xl=lay, xt=dt)
    for xid in ("X1", "X2", "X4"):
        for dt in P.CORE_DTYPES:
            yield P.shape_X(xid, xt=dt)
    for dt in P.HALF_DTYPES:
        yield P.shape_X("X3", xt=dt)
    for lid in P.L_SHAPES:
        for r in P.RANKS:
            for lay, dt in P.PAIRS:
                for act in P.CLAIM_ACTS:
                    yield P.shape_L(lid, r, topo="relu", act=act, xl=lay, xt=dt)
    for act in P.CLAIM_ACTS:
        yield P.shape_X("X1", topo="relu", act=act, xt="half")


RULES = {
    "quick": _quick,
    "standard": _standard,
    "comprehensive": _comprehensive,
    "full": _full,
}


def rules(level):
    yield from RULES[level]()
    for gid, (glevel, c) in P.golden_cases().items():
        if glevel == level:
            yield dict(c, ids=tuple(c["ids"]) + (gid,))


def build():
    """All levels, cumulative-deduplicated: a graph appears in the lowest level and
    earliest rule that holds it, and the plan items of its duplicates merge into it.
    Returns {level: [Item]} and the dropped counts."""
    seen = {}
    result = {}
    dropped = Counter()
    for level in P.LEVELS:
        out = []
        for c in rules(level):
            reason = P.legal(c)
            if reason:
                raise SystemExit(f"{level}: illegal case {c['ids']}: {reason}")
            if not P.fits(c, level) and c["size"] != "X":
                raise SystemExit(f"{level}: {case_id(c)} is over the level ceiling")
            key = P.canon(c)
            survivor = seen.get(key)
            if survivor is not None:
                survivor.merged.update(c["ids"])
                dropped[f"{level}: duplicate graph (plan items merged)"] += 1
                continue
            item = Item(c, case_id(c))
            seen[key] = item
            out.append(item)
        counts = Counter((i.case["topo"], i.id) for i in out)
        for i in out:
            if counts[(i.case["topo"], i.id)] > 1:
                i.id = f"{i.id}_{P.short_hash(i.case)}"
        result[level] = out
    return result, dropped


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def dump(obj):
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def support_json(bundle_dir, level, sub, case_ids):
    """The sidecar of the sweep ``sub`` in ``level``. Claims are harvested, never
    written here: the existing sidecars of the same sweep in every level keep the
    claims of cases whose id still exists, so regenerating the bundles, or a case
    moving to another level, does not erase a harvest."""
    claims = set()
    for lv in P.LEVELS:
        old = bundle_dir / lv / sub / "support.json"
        claims |= {c for c in support_sidecar.load(old) if c.case in case_ids}
    path = bundle_dir / level / sub / "support.json"
    if claims:
        return support_sidecar.render(path, claims)
    return json.dumps({"claims": {}, "version": 1}, indent=2, sort_keys=True) + "\n"


def render(result, bundle_dir):
    """{relative path: text} of every file the generator owns."""
    files = {}
    for level, items in result.items():
        by_topo = defaultdict(list)
        for item in items:
            by_topo[item.case["topo"]].append(item)
        for topo in P.TOPOLOGIES:
            op, name = P.BUNDLE_DIR[topo]
            rel = Path(level) / op / name
            tpl = template(topo)
            need = placeholders(tpl)
            entries = []
            for item in by_topo[topo]:
                e = sweep_case(item)
                have = {"io_data_type"} | {
                    f"attributes.{k}" for k in e["values"]["attributes"]
                }
                if have != need:
                    raise SystemExit(
                        f"{rel}/{item.id}: value keys {sorted(have ^ need)} do not match the template"
                    )
                entries.append(e)
            files[rel / "graph.template.json"] = dump(tpl)
            files[rel / "sweep.json"] = dump({"version": 1, "cases": entries})
            files[rel / "support.json"] = support_json(
                bundle_dir, level, Path(op) / name, {i.id for i in by_topo[topo]}
            )
    return files


def traceability(result):
    """Plan items with no case. Every case has plan items by construction."""
    used = set()
    for items in result.values():
        for i in items:
            used.update(i.plan_items())
    return [p for p in P.PLAN_ITEMS if p not in used]


def report(result, dropped):
    print("Cases by level and topology (own level; cumulative in the last column)")
    total = 0
    print("  ".join(f"{h:>14}" for h in ("level", *P.TOPOLOGIES, "own", "cumulative")))
    for level, items in result.items():
        cnt = Counter(i.case["topo"] for i in items)
        total += len(items)
        row = [level] + [cnt[t] for t in P.TOPOLOGIES] + [len(items), total]
        print("  ".join(f"{v:>14}" for v in row))
    print("\nBy purpose and size class")
    for level, items in result.items():
        pur = Counter(P.purpose(i.case) for i in items)
        sz = Counter(i.case["size"] for i in items)
        print(f"  {level:<14} {dict(sorted(pur.items()))}  {dict(sorted(sz.items()))}")
    print("\nLargest x per level (elements)")
    for level, items in result.items():
        own = [
            i for i in items if i.case["size"] != "X" or P.purpose(i.case) == "claim"
        ]
        if own:
            big = max(own, key=lambda i: P.elements(i.case))
            print(f"  {level:<14} {P.elements(big.case):>13,}  {big.id}")
    behaviours = {
        (c["rank"], c["xl"], c["yl"], P.signature(c), c["act"], c["size"])
        for items in result.values()
        for c in (i.case for i in items)
    }
    print(
        f"\nDistinct (rank, layout, signature, activation, size class): {len(behaviours)}"
    )
    golden = {
        g: i.id
        for items in result.values()
        for i in items
        for g in i.plan_items()
        if g.startswith("G")
    }
    print("\nGolden cases")
    for g in sorted(golden, key=lambda g: int(g[1:])):
        print(f"  {g:<4} {golden[g]}")
    print("\nTrade-off register (not built)")
    for tid, text in P.TRADEOFFS.items():
        print(f"  {tid}: {text}")
    if dropped:
        print("\nDropped:")
        for k, v in sorted(dropped.items()):
            print(f"  {k}: {v}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument(
        "--report",
        action="store_true",
        help="print counts, golden cases and the register",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write nothing; list the files that would change",
    )
    args = ap.parse_args(argv)

    result, dropped = build()
    orphans = traceability(result)
    if orphans:
        print(f"plan items with no case: {', '.join(orphans)}", file=sys.stderr)
        return 2
    files = render(result, args.bundle_dir)
    changed = []
    for rel, text in files.items():
        path = args.bundle_dir / rel
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            changed.append(rel)
            if not args.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
    if args.report:
        report(result, dropped)
    verb = "would change" if args.dry_run else "written"
    print(f"\n{len(changed)} file(s) {verb}")
    for rel in changed:
        print(f"  {rel}")
    return 1 if changed and args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
