#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate the convolution backward-weights (wrw) bundles from the test plan.

Reads the declarations in ``conv_wrw_plan.py`` and the level rules below, and
writes, for each level (quick, standard, comprehensive, full), one template
sweep under ``integration-test-bundles/<level>/``:

    ConvolutionWrw/Default                  wrw: dw = conv_wgrad(x, dy)

The directory gets ``graph.template.json``, ``sweep.json`` and a
``support.json`` (claims are harvested with ``--write-support-claims``, never
written here; an existing sidecar keeps the claims of cases that still exist).
The generator is deterministic. Every case carries a ``purpose`` (claim |
probe) and the plan items it implements (``plan_items``). The level rules build
the complete products; the trims declared in ``conv_wrw_plan.TRIMS`` are
removed after case ids are assigned, so a trim never renames a surviving case.
See ``docs/test-plans/convolution-wrw.md``.

Usage:
    generate_conv_wrw_bundles.py --bundle-dir <integration-test-bundles> [--report] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import conv_wrw_plan as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import support_sidecar  # noqa: E402

GENERATOR = "generate_conv_wrw_bundles.py"
BUNDLE_OP, BUNDLE_NAME = "ConvolutionWrw", "Default"


# --------------------------------------------------------------------------
# Template
# --------------------------------------------------------------------------


def ph(name):
    return "${case." + name + "}"


# role -> uid. dw is the wrw output; x and dy are the device inputs, and the
# fill recipes of ``metadata.inputs`` are keyed by these uids.
UIDS = dict(dw=0, dy=1, x=2)
NAMES = dict(dw="ConvolutionWgrad_0::DW", dy="dy", x="x")


def template():
    node = {
        "compute_data_type": ph("attributes.conv_compute"),
        "inputs": {"dy_tensor_uid": UIDS["dy"], "x_tensor_uid": UIDS["x"]},
        "name": "ConvolutionWgrad_0",
        "outputs": {"dw_tensor_uid": UIDS["dw"]},
        "parameters": {
            "conv_mode": ph("attributes.conv_mode"),
            "dilation": ph("attributes.dilation"),
            "post_padding": ph("attributes.post_padding"),
            "pre_padding": ph("attributes.pre_padding"),
            "stride": ph("attributes.stride"),
        },
        "type": "ConvolutionWrwAttributes",
    }
    tensors = [
        {
            "data_type": ph("data_type"),
            "dims": ph("dims"),
            "name": NAMES[role],
            "strides": ph("strides"),
            "uid": uid,
            "virtual": False,
        }
        for role, uid in sorted(UIDS.items(), key=lambda kv: kv[1])
    ]
    return {
        "compute_data_type": "float",
        "intermediate_data_type": ph("intermediate_data_type"),
        "io_data_type": ph("io_data_type"),
        "is_override_shape_enabled": ph("override_shape"),
        "name": "ConvolutionBackwardWeightTest",
        "nodes": [node],
        "tensors": tensors,
    }


def placeholders(node):
    """Case-wide placeholder paths (not the per-tensor dims/strides/data_type)."""
    found = set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            m = re.fullmatch(r"\$\{case\.(.+)\}", v)
            if m and m.group(1) not in ("dims", "strides", "data_type"):
                found.add(m.group(1))

    walk(node)
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
    dwd, dyd, xd = P.dw_dims(c), P.dy_dims(c), P.x_dims(c)
    return {
        UIDS["dw"]: (dwd, P.strides(dwd, c["dwl"], c["dwe"]), c["dwt"]),
        UIDS["dy"]: (dyd, P.strides(dyd, c["dyl"], c["dye"]), c["dyt"]),
        UIDS["x"]: (xd, P.strides(xd, c["xl"], c["xe"]), c["xt"]),
    }


def input_recipes(c, case_id):
    """Fill recipes (``metadata.inputs``) of the two device inputs, one per
    operand and keyed by the operand's own uid: the recipe the plan states for dy
    goes to dy's uid and the one for x to x's. The uids differ from the dgrad
    graph's (there uid 2 is w), so a recipe table must never be copied from
    there."""
    rec = P.value_recipes(c)
    if rec is None:  # fp8, int8, double: only a plain range
        a = 2.0 if c["xt"] == "int8" else 1.0
        rec = dict(dy=("free", -a, a), x=("free", -a, a))
    return {
        str(UIDS[role]): _recipe(rec[role], _seed(case_id, UIDS[role]))
        for role in sorted(("dy", "x"), key=UIDS.get)
    }


def case_values(c):
    """The ``values`` block of a sweep case."""
    attrs = {
        "conv_compute": c["comp"],
        "conv_mode": c["conv_mode"],
        "dilation": list(c["d"]),
        "post_padding": list(c["post"]),
        "pre_padding": list(c["pre"]),
        "stride": list(c["s"]),
    }
    return {
        "io_data_type": c["xt"],
        "intermediate_data_type": c["dwt"],
        "override_shape": c["override"],
        "attributes": attrs,
        "tensors": [
            {"uid": uid, "dims": d, "strides": s, "data_type": t}
            for uid, (d, s, t) in sorted(tensor_specs(c).items())
        ],
    }


def _tok(v):
    return "x".join(str(i) for i in v)


def case_id(c):
    r = c["rank"]
    parts = ["_".join(c["ids"])] if c["ids"] else []
    parts.append(f"{r}d")
    parts.append("i" + _tok(c["ins"]))
    parts.append(f"n{c['n']}c{c['c']}k{c['k']}" + (f"g{c['g']}" if c["g"] > 1 else ""))
    parts.append("f" + _tok(c["f"]))
    dt = P.SHORT[c["xt"]]
    if (c["dyt"], c["dwt"]) != (c["xt"], c["xt"]):
        dt += f"_dy{P.SHORT[c['dyt']]}_dw{P.SHORT[c['dwt']]}"
    parts.append(dt)
    if c["xl"] == c["dyl"] == c["dwl"]:
        parts.append(c["xl"])
    else:
        parts.append(f"x{c['xl']}dy{c['dyl']}dw{c['dwl']}")
    # The bundle verifier requires lowercase_snake_case ids. The plan items stay
    # uppercase in the metadata (plan_items) and in the plan itself.
    return "_".join(p for p in parts if p).lower()


def sweep_case(c, cid):
    probe = P.probe_reasons(c)
    meta = {
        "format_version": 1,
        "generator": GENERATOR,
        "generator_version": "1",
        "reference_source": "hipDNN reference executors, auto verification (GPU reference, then CPU)",
        "operation": BUNDLE_OP,
        "plan_items": list(c["ids"]),
        "purpose": P.purpose(c),
        "size_class": c["size"],
        "seed": _seed(cid, "case"),
        "inputs": input_recipes(c, cid),
    }
    notes = []
    if probe:
        notes.append("probe: " + "; ".join(probe))
    if c.get("src"):
        notes.append(c["src"])
    if c["size"] == "B":
        notes.append(
            "regression: the shape of the flaky split-K cases of "
            "ROCm/rocm-libraries#8029 and #8030"
        )
    if notes:
        meta["notes"] = "; ".join(notes)
    total = sum(P.numel(d) * P.BYTES.get(t, 1) for d, _, t in tensor_specs(c).values())
    if total > (1 << 30):
        meta["minimum_vram_mb"] = int(total * 2 / (1 << 20)) + 1
    return {"id": cid, "values": case_values(c), "metadata": meta}


# --------------------------------------------------------------------------
# Case builders and variants
# --------------------------------------------------------------------------


def with_(c, add_ids=(), **kw):
    d = dict(c)
    d.update(kw)
    d["ids"] = tuple(c["ids"]) + tuple(add_ids)
    return P.finalize(d)


def v_convmode(c):
    return with_(c, ("F13",), conv_mode="CONVOLUTION")


def v_asym(c):
    if c["pre"] != c["post"]:
        return None
    return with_(c, ("F11",), post=tuple(p + 1 for p in c["pre"]))


def v_layouts(c, combo):
    return with_(c, ("F15",), xl=combo[0], dyl=combo[1], dwl=combo[2])


# (x, dy, dw) layouts: the six non-uniform combinations
MIXES = [
    ("cl", "cf", "cf"),
    ("cf", "cl", "cf"),
    ("cf", "cf", "cl"),
    ("cl", "cl", "cf"),
    ("cl", "cf", "cl"),
    ("cf", "cl", "cl"),
]


def v_nonpacked(c, kind):
    r = c["rank"]

    def extra(layout):
        e = [0] * (r + 2)
        if kind == "fast":
            e[P.fastest_dim(layout, r)] = 4
        elif kind == "chan":
            e[1] = 8
        else:  # plane: pad the second-fastest dimension
            e[r + 1 if layout == "cl" else (r if r >= 2 else 1)] = 3
        return tuple(e)

    return with_(
        c,
        (f"F16{kind}",),
        xe=extra(c["xl"]),
        dye=extra(c["dyl"]),
        dwe=extra(c["dwl"]),
    )


def v_type(c, name):
    xt, dyt, dwt, comp = P.EXTRA_TYPES.get(name) or P.MIXED_TYPES[name]
    tag = "F19" if name in P.EXTRA_TYPES else "F20"
    return with_(c, (tag,), xt=xt, dyt=dyt, dwt=dwt, comp=comp)


def v_vals(c, cls):
    if c["xt"] not in P.DTYPE_MAX:
        return None
    d = with_(c, ("F21" + cls,), vals=cls)
    return d if P.value_recipes(d) is not None else None


def v_override(c):
    return with_(c, ("F35",), override=True)


# --------------------------------------------------------------------------
# Level rules
# --------------------------------------------------------------------------


class Builder:
    def __init__(self, level_index):
        self.lv = level_index
        self.level = P.LEVELS[level_index]

    def S(self, sid, r, pair, **kw):
        kwargs = P.shape_S(sid, r, self.level)
        if kwargs is None:
            return None
        return self._make(kwargs, pair, (sid,), "S", **kw)

    def Ls(self, sid, r, pair, **kw):
        kwargs = P.shape_L(sid, r, "comprehensive")
        if kwargs is None:
            return None
        return self._make(kwargs, pair, ("L" + sid[1:],), "L", **kw)

    def LAs(self, sid, r, pair, **kw):
        kwargs = P.shape_LA(sid, r)
        if kwargs is None:
            return None
        return self._make(kwargs, pair, ("LA" + sid[1:],), "LA", **kw)

    def M(self, mid, pair, **kw):
        return self._make(dict(P.M_SHAPES[mid]), pair, (mid,), "M", **kw)

    def H(self, hid, r, pair, **kw):
        kwargs = P.shape_H(hid, r, self.level)
        if kwargs is None:
            return None
        return self._make(kwargs, pair, (hid,), "H", **kw)

    def B(self, bid, pair, **kw):
        return self._make(dict(P.REGRESSION_SHAPES[bid]), pair, (bid,), "B", **kw)

    def X(self, xid, pair, **kw):
        return self._make(dict(P.X_SHAPES[xid]), pair, (xid,), "X", **kw)

    def A(self, aid, pair, **kw):
        return self._make(dict(P.ACCUM_SHAPES[aid]), pair, (aid,), "A", **kw)

    def R(self, rid):
        return P.new(**P.shape_R(rid), ids=(rid,), size="R", src=P.r_source(rid))

    def _make(self, kwargs, pair, ids, size, **kw):
        lay, dt = pair
        return P.new(
            **kwargs, xl=lay, dyl=lay, dwl=lay, xt=dt, ids=ids, size=size, **kw
        )

    def sids(self, r, candidates=P.S_IDS):
        return [s for s in candidates if P.shape_S(s, r, self.level) is not None]

    def lids(self, r, candidates=P.S_IDS):
        return [s for s in candidates if P.shape_L(s, r) is not None]

    def laids(self, r):
        return [s for s in P.LA_IDS if P.shape_LA(s, r) is not None]


def core_cases(b):
    """The wide set that quick used to hold: one class from each group on every
    rank, and the probes. ``b`` is the quick-level builder, so the shapes are the
    quick-level shapes. PR holds it."""
    cases = []
    add = cases.append
    P1 = P.PAIRS[0]  # (cf, fp32)
    for r in P.RANKS:
        for i, s in enumerate(b.sids(r, P.S_QUICK)):
            add(b.S(s, r, P.PAIRS[(i + r - 1) % 6]))
    add(v_convmode(b.S("S3", 2, P1)))
    add(b.S("S10", 2, P1))  # asymmetric padding
    add(v_type(b.S("S3", 2, P1), "fp8_e4m3"))
    add(v_type(b.S("S3", 2, P1), "double"))
    return cases


def rules(lv):
    b = Builder(lv)
    cases = []
    add = cases.append
    P1 = P.PAIRS[0]  # (cf, fp32)
    accum = list(P.ACCUM_SHAPES)
    extra = list(P.EXTRA_TYPES)
    mixed = list(P.MIXED_TYPES)
    classes = [c for c in P.VALUE_CLASSES if c != "uniform"]

    if lv == 0:  # quick: the smallest set that catches gross breakage
        # The 3x3 filter on 2D (fp32 channels-first, fp16 channels-last), a
        # stride-2 filter (bf16), a 3D filter (fp32 channels-last), a 1x1 filter,
        # a depthwise and a grouped class, and a long reduction against a small
        # output.
        add(b.S("S3", 2, P.PAIRS[0]))
        add(b.S("S3", 2, P.PAIRS[3]))
        add(b.S("S17", 2, P.PAIRS[4]))
        add(b.S("S3", 3, P.PAIRS[1]))
        add(b.S("S1", 2, P.PAIRS[5]))
        add(b.S("S53", 2, P.PAIRS[2]))
        add(b.S("S44", 2, P.PAIRS[0]))
        add(b.A("A1", P.PAIRS[1]))

    elif lv == 1:  # PR: what must we test on every PR?
        cases.extend(core_cases(Builder(0)))
        for r in (1, 2, 3, 4):
            for i, s in enumerate(b.sids(r)):
                add(b.S(s, r, P.PAIRS[(i + r + 3) % 6]))

        def S3(pair=P1, r=2):
            return b.S("S3", r, pair)

        for r in P.RANKS:
            add(v_convmode(S3(P1, r)))
        add(v_layouts(S3(P.PAIRS[3]), MIXES[0]))
        add(v_nonpacked(S3(), "fast"))
        for name in extra + mixed:
            add(v_type(S3(), name))
        for cls in classes:
            add(v_vals(S3(P.PAIRS[2 if cls == "large" else 0]), cls))
        add(v_override(S3()))
        for i, h in enumerate(P.H_SHAPES):
            add(b.H(h, 2, P.PAIRS[i % 6]))
        for i, m in enumerate(P.M_PR):
            add(b.M(m, P.PAIRS[(2 * i + 2) % 6]))
        for bid in P.REGRESSION_SHAPES:
            for pair in P.PAIRS:
                add(b.B(bid, pair))
        for i, a in enumerate(accum):
            add(b.A(a, P.PAIRS[(i + 1) % 6]))
        for rid, e in P.R_SHAPES.items():
            if e["level"] == "standard":
                add(b.R(rid))

    elif lv == 2:  # nightly: what should we test each day, wider per engine?
        for r in (1, 2, 3, 4):
            for s in b.sids(r):
                for pair in P.PAIRS:
                    add(b.S(s, r, pair))
        for r in P.RANKS:
            for i, s in enumerate(b.lids(r)):
                for pair in (P.PAIRS[(2 * i) % 6], P.PAIRS[(2 * i + 1) % 6]):
                    add(b.Ls(s, r, pair))
        for r in P.RANKS:
            for i, s in enumerate(b.laids(r)):
                add(b.LAs(s, r, P.PAIRS[(i + r) % 6]))
        for m in P.M_SHAPES:
            for pair in P.PAIRS:
                add(b.M(m, pair))
        for r in P.RANKS:
            for pair in P.PAIRS:
                add(v_convmode(b.S("S3", r, pair)))
            for di, dt in enumerate(P.CORE_DTYPES):
                for j in range(2):
                    add(
                        v_layouts(b.S("S3", r, ("cf", dt)), MIXES[(2 * di + j + r) % 6])
                    )
                for ki, kind in enumerate(("fast", "chan", "plane")):
                    add(v_nonpacked(b.S("S3", r, (P.LAYOUTS[(ki + di) % 2], dt)), kind))
                for cls in classes:
                    add(v_vals(b.S("S3", r, ("cf", dt)), cls))
                add(v_override(b.S("S3", r, ("cf", dt))))
            for pair in P.PAIRS:
                for name in extra + mixed:
                    add(v_type(b.S("S3", r, pair), name))
        for a in accum:
            for pair in P.PAIRS:
                add(b.A(a, pair))
            # Non-cancelling sums: the error of a long reduction shows here.
            for dt in P.CORE_DTYPES:
                add(v_vals(b.A(a, ("cf", dt)), "positive"))
        for h in P.H_SHAPES:
            for pair in P.PAIRS:
                add(b.H(h, 2, pair))
            for r in (1, 3):
                for pair in (P.PAIRS[0], P.PAIRS[3]):
                    add(b.H(h, r, pair))
        for rid, e in P.R_SHAPES.items():
            if e["level"] == "comprehensive":
                add(b.R(rid))

    else:  # full: which wider sweeps must exist in the full suite?
        for r in P.RANKS:
            for s in b.lids(r):
                for pair in P.PAIRS:
                    add(b.Ls(s, r, pair))
        for r in P.RANKS:
            for i, s in enumerate(b.laids(r)):
                for off in (2, 4):
                    add(b.LAs(s, r, P.PAIRS[(i + r + off) % 6]))
        for xid in ("X1", "X2", "X4", "X5"):
            for pair in P.PAIRS:
                add(b.X(xid, pair))
        # X3: 2^31 products into one dw element, in the 16-bit types.
        add(b.X("X3", ("cf", "half")))
        add(b.X("X3", ("cl", "bfloat16")))
        for a in accum:
            for pair in P.PAIRS:
                add(b.A(a, pair))
        for r in (1, 2, 3, 4):
            for i, s in enumerate(b.sids(r, P.S_CROSSED)):

                def base(off=0):
                    return b.S(s, r, P.PAIRS[(i + off) % 6])

                add(v_convmode(base(0)))
                add(v_asym(base(1)))
                add(v_type(base(2), extra[i % len(extra)]))
                add(v_type(base(3), mixed[i % len(mixed)]))
                add(v_layouts(base(4), MIXES[i % 6]))
                add(v_nonpacked(base(5), ("fast", "chan", "plane")[i % 3]))
                add(v_vals(base(0), classes[i % len(classes)]))
                add(v_override(base(1)))
        for r in P.RANKS:
            for pair in P.PAIRS:
                for cls in classes:
                    add(v_vals(b.S("S2", r, pair), cls))
        for rid, e in P.R_SHAPES.items():
            if e["level"] == "full":
                add(b.R(rid))
    return [c for c in cases if c is not None]


def build():
    """All levels, cumulative-deduplicated: a graph appears in the lowest level
    that holds it. Returns {level: [(case, id)]} and the dropped counts."""
    seen = set()
    result = {}
    dropped = Counter()
    for lv, level in enumerate(P.LEVELS):
        out = []
        for c in rules(lv):
            reason = P.legal(c)
            if reason:
                dropped[f"{level}: {reason}"] += 1
                continue
            exempt = c["size"] == "X" or (
                any(i in P.M_PR for i in c["ids"]) and level == "standard"
            )
            if not exempt and not P.fits(c, level) and c["size"] != "L":
                dropped[f"{level}: over ceiling"] += 1
                continue
            key = P.canon(c)
            if key in seen:
                continue
            seen.add(key)
            out.append([c, case_id(c)])
        counts = Counter(cid for _, cid in out)
        for item in out:
            c, cid = item
            if counts[cid] > 1:
                item[1] = f"{cid}_{P.short_hash(c)}"
        result[level] = out
    dropped.update(apply_trims(result))
    return result, dropped


def apply_trims(result):
    """Remove the products of ``P.TRIMS``. Runs after the ids are assigned, so the
    cases that remain keep their ids."""
    removed = Counter()
    for level, items in result.items():
        keep = []
        for item in items:
            t = P.trim(item[0], level)
            if t:
                removed[f"{level}: trimmed {t}"] += 1
            else:
                keep.append(item)
        result[level] = keep
    return removed


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def dump(obj):
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def support_json(path, case_ids):
    """The sidecar of a sweep. Claims are harvested, never written here: an
    existing sidecar keeps the claims of cases that still exist."""
    claims = (
        {c for c in support_sidecar.load(path) if c.case in case_ids}
        if path.exists()
        else set()
    )
    return (
        support_sidecar.render(path, claims)
        if claims
        else json.dumps({"claims": {}, "version": 1}, indent=2, sort_keys=True) + "\n"
    )


def flat_keys(prefix, v, have):
    if isinstance(v, dict):
        for k, x in v.items():
            flat_keys(prefix + k + ".", x, have)
    else:
        have.add(prefix[:-1])


def write_tree(result, bundle_dir, dry_run):
    files = {}
    tpl = template()
    need = placeholders(tpl)
    for level, items in result.items():
        if not items:
            continue
        rel = Path(level) / BUNDLE_OP / BUNDLE_NAME
        entries = []
        for c, cid in items:
            e = sweep_case(c, cid)
            have = set()
            for k, v in e["values"].items():
                if k != "tensors":
                    flat_keys(k + ".", v, have)
            if have != need:
                raise SystemExit(
                    f"{rel}/{cid}: value keys {sorted(have ^ need)} do not match the template placeholders"
                )
            entries.append(e)
        files[rel / "graph.template.json"] = dump(tpl)
        files[rel / "sweep.json"] = dump({"version": 1, "cases": entries})
        files[rel / "support.json"] = support_json(
            bundle_dir / rel / "support.json", {cid for _, cid in items}
        )
    if dry_run:
        return files
    for level in P.LEVELS:
        target = bundle_dir / level / BUNDLE_OP
        if target.exists():
            shutil.rmtree(target)
    for rel, text in files.items():
        path = bundle_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return files


def report(result, dropped):
    print("\nCases by level (own level; cumulative in the last column)")
    total = 0
    for level, items in result.items():
        total += len(items)
        print(f"  {level:<14} {len(items):>6} {total:>8}")
    print("\nBy purpose and size class")
    for level, items in result.items():
        pur = Counter(P.purpose(c) for c, _ in items)
        sz = Counter(c["size"] for c, _ in items)
        print(f"  {level:<14} {dict(pur)}  {dict(sorted(sz.items()))}")
    geo = {
        tuple(
            c[k]
            for k in ("rank", "n", "c", "k", "g", "ins", "f", "s", "d", "pre", "post")
        )
        for items in result.values()
        for c, _ in items
    }
    print(f"\nDistinct conv geometries: {len(geo)}")
    print("\nFLOP, bytes and reduction length A per case, max by level")
    for level, items in result.items():
        if items:
            print(
                f"  {level:<14} max {max(P.flops(c) for c, _ in items) / 1e9:.3f} GFLOP, "
                f"max {max(P.fp32_bytes(c) for c, _ in items) / (1 << 20):.1f} MiB, "
                f"max A {max(P.accum_len(c) for c, _ in items):,}"
            )
    sigs = {P.signature(c) for items in result.values() for c, _ in items}
    print(f"\nDistinct behaviour signatures: {len(sigs)}")
    if dropped:
        print("\nDropped:")
        for k, v in sorted(dropped.items()):
            print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    result, dropped = build()
    write_tree(result, args.bundle_dir, args.dry_run)
    if args.report or args.dry_run:
        report(result, dropped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
