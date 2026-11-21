#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mine the code maps the translator and the lints share.

Writes three JSON files (default dir: data/maps/ next to this tools/ directory):

  sigmap.json    Python IR-builder methods and backward helpers -> C functions:
                 argument slot order, defaults, emits-IR or pure, opts structs,
                 context bundles, many-to-one entries.
  twin_map.json  each Python function, class, method and constant of the
                 backward builder modules -> its C++ twin (symbol, file:line) or
                 "not ported".
  owners.json    each shared C/C++ symbol -> the one file that should define it,
                 and every duplicate definition.

Static analysis only: it reads the rocKE tree and never builds or writes there.

Exit codes: 0 maps written, 2 usage or environment error.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cscan  # noqa: E402
import rsrc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "data", "maps")


class EnvError(Exception):
    """The rocKE tree is not what the tool expects (exit code 2)."""


# Python IRBuilder methods that have no C function of the same name.
IR_SPECIAL = {
    "_op": ("rocke_b_op", "generic op builder"),
    "_emit": ("rocke_b_emit", "append an op to the current region"),
    "_fresh": ("rocke_b_fresh", "fresh SSA name"),
    "__init__": ("rocke_ir_builder_init", "builder construction"),
    "push_region": ("rocke_b_region_enter", "host region stack"),
    "pop_region": ("rocke_b_region_leave", "host region stack"),
    "static_for": (None, "host-time loop: a plain C loop"),
    "unroll": (None, "host-time range: a plain C loop"),
    "static_if": (None, "host-time branch: a plain C if"),
}

# ------------------------------------------------------------------ twin map


def add_folded(tw: dict) -> None:
    """A method whose class has a twin but no symbol of its own is `folded`."""
    syms = tw["symbols"]
    for e in syms.values():
        if e["status"] != "not ported" or e["kind"] not in ("method", "property"):
            continue
        cls = e["name"].split(".")[0]
        ce = syms.get(f"{e['module']}:{cls}")
        if ce and ce["twins"]:
            e["status"] = "folded"
            e["twins"] = [dict(t, evidence="class") for t in ce["twins"][:1]]
            e.pop("reason", None)
            e.pop("same_name_elsewhere", None)


def add_many_to_one(tw: dict) -> None:
    """Python symbols that share one C twin (a kind-tag struct, a merged helper)."""
    rev: Dict[tuple, List[str]] = {}
    for k, e in tw["symbols"].items():
        if e["status"] != "ported":
            continue
        for t in e["twins"]:
            if t["evidence"] in ("table", "convention", "comment"):
                rev.setdefault((t["symbol"], t["file"], t["line"]), []).append(k)
    for keys in rev.values():
        if len(keys) > 1:
            for k in keys:
                tw["symbols"][k].setdefault("many_to_one", sorted(set(keys) - {k}))


def twin_coverage(tw: dict) -> dict:
    syms = tw["symbols"].values()

    def count(sel):
        sel = [e for e in syms if sel(e)]
        return {
            "total": len(sel),
            "ported": sum(e["status"] == "ported" for e in sel),
            "folded": sum(e["status"] == "folded" for e in sel),
            "not_ported": sum(e["status"] == "not ported" for e in sel),
        }

    has_c = {m for m, v in tw["modules"].items() if v["cxx_files"]}
    cov = {
        "all": count(lambda e: True),
        "public": count(lambda e: not e["private"]),
        "modules_with_cxx": count(lambda e: e["module"] in has_c),
        "public_in_modules_with_cxx": count(
            lambda e: e["module"] in has_c and not e["private"]
        ),
        "by_module": {},
    }
    for m in sorted(tw["modules"]):
        cov["by_module"][m] = count(lambda e, m=m: e["module"] == m)
    return cov


def build_twin(tree: rsrc.Tree) -> dict:
    tw = rsrc.build_twin_map(tree)
    add_folded(tw)
    add_many_to_one(tw)
    tw["coverage"] = twin_coverage(tw)
    return tw


# ------------------------------------------------------------------ C signatures


def c_params(sym: cscan.Sym) -> List[dict]:
    out = []
    raw = sym.params.strip()
    if raw in ("", "void"):
        return out
    for piece in cscan._split_top(raw):
        piece = re.sub(r"\s+", " ", piece).strip()
        piece = re.sub(r"=.*$", "", piece).strip()
        m = re.match(r"^(.*?)([A-Za-z_]\w*)(\s*\[[^\]]*\])?$", piece)
        if (
            m
            and m.group(1).strip()
            and not re.search(
                r"(^|\s)(struct|enum|union|const|unsigned)\s*$", m.group(1)
            )
        ):
            typ, name = m.group(1).strip(), m.group(2)
            if m.group(3):
                typ += m.group(3).strip()
        else:
            typ, name = piece, ""
        out.append({"type": typ, "name": name})
    return out


def classify_c_param(p: dict) -> List[str]:
    t, n = p["type"], p["name"]
    tags = []
    if "rocke_ir_builder_t" in t:
        tags.append("builder")
    if re.search(r"(_opts|_options)(_t)?\b", t):
        tags.append("opts")
    if (
        re.search(r"\b\w*_args(_t)?\b", t)
        and "rocke_ir_builder" not in t
        and n.endswith(("args", "a"))
    ):
        tags.append("opts")
    if "&" in t and "std::" not in t:
        tags.append("context")
    if re.search(r"_ctx(_t)?\b", t) or re.search(r"\b(Cta|Main|Kargs)\b", t):
        tags.append("context")
    if n == "out" or n.startswith("out_") or re.search(r"\*\*\s*$", t):
        tags.append("out")
    if (
        re.match(r"^(n|num|count|rank)_?\w*$", n)
        or n.endswith(("_n", "_len", "_count"))
        and "int" in t
    ):
        tags.append("count")
    if re.search(r"rocke_(attn_bwd_iv|int_or_value)_t", t):
        tags.append("int_or_value")
    return tags


def doc_of(sym: cscan.Sym, comments: List[cscan.Comment]) -> str:
    """Comments that sit inside the prototype, and a block comment right above it."""
    texts = []
    for c in comments:
        if c.line >= sym.line and c.end_line <= max(sym.end_line, sym.line):
            texts.append(c.text)
        elif c.block and c.end_line == sym.line - 1:
            texts.insert(0, c.text)
    t = " ".join(re.sub(r"\s*\n\s*\*?\s*", " ", x).strip() for x in texts)
    return re.sub(r"\s+", " ", t).strip()[:300]


# C parameters carrying state Python keeps on the IRBuilder (R-HS1, R-HS2).
# Matching one to a Python argument by position drops that argument
# (decode_thd's s_max was lost to seq_guards this way).
BUILDER_STATE = ("seq_guards", "fence")


def slot_map(
    py_params: List[dict],
    c_ps: List[dict],
    drop_self: bool,
    builder_c: bool,
    ir_self: bool = False,
) -> dict:
    """Pair Python parameters with C parameters: by name, then by position.

    drop_self: the Python function is a method (its `self` is not a slot).
    ir_self: the C builder parameter is the method's `self` (IRBuilder methods).
    """
    P = [p for p in py_params if not (drop_self and p["name"] in ("self", "cls"))]
    C = list(c_ps)
    self_c = None
    if drop_self and C and not ir_self and "builder" not in classify_c_param(C[0]):
        self_c = C[0]  # the struct the method works on
        C = C[1:]
    clash = False
    if builder_c and C and "builder" in classify_c_param(C[0]):
        bname = C[0]["name"]
        C = C[1:]
        if P and P[0]["name"] in ("b", "builder") and not ir_self:
            P = P[1:]
        # The C builder parameter took the name `b`, so the C API renamed any
        # Python parameter `b` and the ones after it. Names can't be trusted.
        clash = bool(bname) and any(p["name"] == bname for p in P)
    used = [False] * len(C)
    pairs = []

    def key(n):
        return re.sub(r"^(in_|out_)", "", n).lstrip("_").lower()

    for p in P:
        hit = None
        if not clash:
            for k, c in enumerate(C):
                if not used[k] and c["name"] and key(c["name"]) == key(p["name"]):
                    hit = k
                    break
        if hit is not None:
            used[hit] = True
        pairs.append([p, hit, "name" if hit is not None else None])
    free = [k for k in range(len(C)) if not used[k]]
    for pr in pairs:
        if pr[1] is None:
            for k in list(free):
                tags = classify_c_param(C[k])
                # builder state the C side threads by hand has no Python source
                if C[k]["name"] in BUILDER_STATE:
                    continue
                if not ({"out", "opts", "context", "count"} & set(tags)):
                    pr[1] = k
                    pr[2] = "position"
                    free.remove(k)
                    used[k] = True
                    break
    idx = [pr[1] for pr in pairs if pr[1] is not None]
    monotonic = all(x < y for x, y in zip(idx, idx[1:]))
    slots = []
    for p, k, how in pairs:
        slots.append(
            {
                "py": p["name"],
                "py_kind": p["kind"],
                "default": p["default"],
                "c": C[k]["name"] if k is not None else None,
                "c_type": C[k]["type"] if k is not None else None,
                "match": how,
            }
        )
    extras = []
    for k, c in enumerate(C):
        if not used[k]:
            extras.append(
                {
                    "c": c["name"],
                    "c_type": c["type"],
                    "role": ",".join(classify_c_param(c)) or "extra",
                }
            )
    out = {"slots": slots, "extra_c": extras, "monotonic": monotonic}
    if self_c:
        out["self_c"] = {"c": self_c["name"], "c_type": self_c["type"]}
    return out


def split_name_sig(entry: dict) -> str:
    return entry.get("match", "")


# ------------------------------------------------------------------ IRBuilder


def _pure_op_names(tree_ast: ast.Module) -> Set[str]:
    for n in tree_ast.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PURE_OP_NAMES" for t in n.targets
        ):
            v = n.value
            if isinstance(v, ast.Call) and v.args:
                v = v.args[0]
            try:
                return set(ast.literal_eval(v))
            except Exception:
                return set()
    return set()


def ir_methods(ir_py: str):
    with open(ir_py, encoding="utf-8") as f:
        t = ast.parse(f.read())
    pure = _pure_op_names(t)
    cls = next(
        (n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "IRBuilder"),
        None,
    )
    if cls is None:
        raise EnvError(f"{ir_py} has no IRBuilder class")
    meths = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    direct_ops: Dict[str, Set[str]] = {}
    direct_emit: Dict[str, bool] = {}
    calls: Dict[str, Set[str]] = {}
    for name, fn in meths.items():
        ops, em, cl = set(), False, set()
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"
            ):
                m = n.func.attr
                if m == "_op":
                    em = True
                    a0 = n.args[0] if n.args else None
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        ops.add(a0.value)
                    else:
                        ops.add("?")
                elif m == "_emit":
                    em = True
                    ops.add("?")
                elif m in meths:
                    cl.add(m)
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "Op"
            ):
                for kw in n.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        ops.add(kw.value.value)
        direct_ops[name], direct_emit[name], calls[name] = ops, em, cl
    emits: Dict[str, bool] = {}
    allops: Dict[str, Set[str]] = {}

    def visit(name: str, stack: Set[str]):
        if name in emits:
            return
        stack = stack | {name}
        e, o = direct_emit[name], set(direct_ops[name])
        for c in calls[name]:
            if c in stack:
                continue
            visit(c, stack)
            if c in emits:
                e = e or emits[c]
                o |= allops[c]
        emits[name], allops[name] = e, o

    for name in meths:
        visit(name, set())
    return meths, emits, allops, pure


def build_ir_sigs(tree: rsrc.Tree, hdr_index: rsrc.CIndex) -> dict:
    meths, emits, allops, pure = ir_methods(tree.ir_py)
    cdecl: Dict[str, List[cscan.Sym]] = {}
    ir_h_names: Set[str] = set()
    for s in hdr_index.syms:
        if s.kind == "function" and (
            s.name.startswith("rocke_b_")
            or s.name in {v[0] for v in IR_SPECIAL.values() if v[0]}
        ):
            cdecl.setdefault(s.name, []).append(s)
            if s.file.endswith("/ir.h") and s.name.startswith("rocke_b_"):
                ir_h_names.add(s.name)
    out = {}
    py_to_c: Dict[str, List[str]] = {}
    for name, fn in meths.items():
        params = rsrc._params(fn)
        ops = sorted(allops[name])
        lit = [o for o in ops if o != "?"]
        entry = {
            "python": {
                "line": fn.lineno,
                "params": params,
                "returns": ast.unparse(fn.returns) if fn.returns else "",
            },
            "private": name.startswith("_"),
            "emits_ir": emits[name],
            "ops": ops,
            "op_pure": (
                (all(o in pure for o in lit) if lit and "?" not in ops else None)
                if emits[name]
                else None
            ),
        }
        cname, note = f"rocke_b_{name}", ""
        if name in IR_SPECIAL:
            cname, note = IR_SPECIAL[name]
        decl = (
            [d for d in cdecl.get(cname, []) if d.file.endswith("/ir.h")]
            if cname
            else None
        )
        if cname and not decl:
            decl = cdecl.get(cname)
        variants = []
        for vn in (f"rocke_b_{name}_ex",):
            if vn in cdecl:
                variants.append(vn)
        if decl:
            s = sorted(decl, key=lambda d: (not d.is_def, d.file))[0]
            cps = c_params(s)
            sm = slot_map(params, cps, drop_self=True, builder_c=True, ir_self=True)
            tags = sorted(
                {t for p in cps for t in classify_c_param(p) if t != "builder"}
            )
            entry["c"] = {
                "name": cname,
                "file": s.file,
                "line": s.line,
                "returns": s.ret,
                "params": cps,
                "monotonic": sm["monotonic"],
                "slots": sm["slots"],
                "extra_c": sm["extra_c"],
                "roles": tags,
                "opts_variants": variants,
                "doc": doc_of(s, hdr_index.comments.get(s.file, [])),
            }
            miss = [x for x in sm["slots"] if not x["c"]]
            extra = [x for x in sm["extra_c"] if x["role"] != "count"]
            if (
                variants
                and miss
                and all(x["default"] is not None for x in miss)
                and not extra
            ):
                ex = sorted(cdecl[variants[0]], key=lambda d: d.file)[0]
                entry["c"]["opts_variant_params"] = c_params(ex)
                entry["match"] = "opts-struct"
            elif not miss and not extra and not sm["monotonic"]:
                entry["match"] = "reordered"
            elif not miss and not extra:
                entry["match"] = "exact"
            elif any("opts" in x["role"] for x in extra):
                entry["match"] = "opts-struct"
            else:
                entry["match"] = "arity-differs"
            py_to_c.setdefault(cname, []).append(name)
        else:
            entry["c"] = None
            entry["match"] = (
                "host-only" if name in IR_SPECIAL and cname is None else "none"
            )
            if note:
                entry["note"] = note
            if cname and note:
                entry["c_name_hint"] = cname
        out[name] = entry
    # C functions with a rocke_b_ name and no Python method of that name
    c_only = sorted(
        n
        for n in ir_h_names
        if n[len("rocke_b_") :] not in meths
        and n not in {v[0] for v in IR_SPECIAL.values()}
        and not any(m for m in meths if f"rocke_b_{m}_ex" == n)
    )
    for n, pys in py_to_c.items():
        if len(pys) > 1:
            for p in pys:
                out[p]["many_to_one"] = sorted(pys)
    return {"methods": out, "c_only": c_only}


# ------------------------------------------------------------------ helpers


def builder_names(ir_sigs: dict) -> Set[str]:
    return {n for n, e in ir_sigs["methods"].items() if e["emits_ir"]}


def helper_emits(tree: rsrc.Tree, emit_methods: Set[str]) -> Dict[str, bool]:
    """Python helper functions of the backward modules that reach a builder call."""
    funcs: Dict[str, ast.FunctionDef] = {}
    for f in tree.bwd_py():
        mod = os.path.splitext(os.path.basename(f))[0]
        t = ast.parse(open(f, encoding="utf-8").read())
        for n in t.body:
            if isinstance(n, ast.FunctionDef):
                funcs[f"{mod}:{n.name}"] = n
            elif isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, ast.FunctionDef):
                        funcs[f"{mod}:{n.name}.{m.name}"] = m
    by_simple: Dict[str, List[str]] = {}
    for k in funcs:
        by_simple.setdefault(k.split(":")[1].split(".")[-1], []).append(k)
    direct: Dict[str, bool] = {}
    calls: Dict[str, Set[str]] = {}
    for k, fn in funcs.items():
        d, cs = False, set()
        for n in ast.walk(fn):
            if not isinstance(n, ast.Call):
                continue
            if isinstance(n.func, ast.Attribute):
                recv = n.func.value
                is_b = (
                    isinstance(recv, ast.Name) and recv.id in ("b", "builder", "bld")
                ) or (
                    isinstance(recv, ast.Attribute)
                    and recv.attr in ("b", "_b", "builder")
                )
                if is_b and n.func.attr in emit_methods:
                    d = True
                cs.add(n.func.attr)
            elif isinstance(n.func, ast.Name):
                cs.add(n.func.id)
        direct[k], calls[k] = d, cs
    res = dict(direct)
    changed = True
    while changed:
        changed = False
        for k in funcs:
            if res[k]:
                continue
            for c in calls[k]:
                if any(res.get(x) for x in by_simple.get(c, [])):
                    res[k] = True
                    changed = True
                    break
    return res


def external_imports(tree: rsrc.Tree) -> Dict[str, Set[str]]:
    """Names the package's modules import from rocke.* other than core.ir (an
    import from a parent package, `from ...helpers.spec import`, by its
    absolute name)."""
    out: Dict[str, Set[str]] = {}
    for f in tree.bwd_py():
        t = ast.parse(open(f, encoding="utf-8").read())
        for n in ast.walk(t):
            if not isinstance(n, ast.ImportFrom):
                continue
            mod = rsrc.PKG.import_source(n.module, n.level)
            if mod.startswith("rocke.") and mod != "rocke.core.ir":
                for a in n.names:
                    out.setdefault(mod, set()).add(a.name)
    return out


def build_helper_sigs(
    tree: rsrc.Tree,
    tw: dict,
    hdr_all: rsrc.CIndex,
    emits_map: Dict[str, bool],
    extern_names: Set[str],
) -> dict:
    """Signatures for the backward helpers that have a C function twin."""
    pysyms = {}
    for f in tree.bwd_py():
        for s in rsrc.py_symbols(f):
            pysyms[s.key] = s
    out = {}
    comm = {}
    dropped: List[str] = []
    for key, e in tw["symbols"].items():
        if e["status"] != "ported" or e["kind"] not in (
            "function",
            "method",
            "class",
            "property",
        ):
            continue  # a folded method has no C function of its own
        ps = pysyms[key]
        fn_twins = []
        for t in e["twins"]:
            for s in hdr_all.by_qual.get(t["symbol"], []):
                if s.kind == "function" and s.file == t["file"] and s.line == t["line"]:
                    fn_twins.append((s, t))
        if ps.kind == "class":
            # a class has no signature; its __init__ / _init does
            continue
        if not fn_twins:
            continue
        # keep the twin the naming convention gives this method; a table pick
        # that names something else is a rename (medium); a group pick that
        # does not match says nothing about this method, so the entry is dropped
        keys = set(rsrc.py_keys(ps))
        named = [(s, t) for s, t in fn_twins if keys & set(rsrc.c_keys(s))]
        renamed = [(s, t) for s, t in fn_twins if t["evidence"] in ("table", "comment")]
        if named:
            fn_twins, conf = named, "high"
        elif renamed:
            fn_twins, conf = renamed, "medium"
        else:
            dropped.append(key)
            continue
        s, tw_ev = sorted(fn_twins, key=lambda d: (not d[0].is_def, d[0].file))[0]
        cps = c_params(s)
        is_method = ps.kind in ("method", "property")
        sm = slot_map(ps.params, cps, drop_self=is_method, builder_c=True)
        roles = sorted({t for p in cps for t in classify_c_param(p)} - {"builder"})
        uses_b = "builder" in {t for p in cps for t in classify_c_param(p)}
        entry = {
            "python": {
                "module": ps.module,
                "line": ps.line,
                "params": ps.params,
                "returns": ps.returns,
            },
            "kind": ps.kind,
            "emits_ir": bool(emits_map.get(key, False)),
            "c": {
                "name": s.qualified,
                "file": s.file,
                "line": s.line,
                "returns": s.ret,
                "params": cps,
                "slots": sm["slots"],
                "extra_c": sm["extra_c"],
                "roles": roles,
                "reordered": not sm["monotonic"],
                "takes_builder": uses_b,
                "doc": doc_of(s, hdr_all.comments.get(s.file, [])),
            },
            "opts_structs": [p["type"] for p in cps if "opts" in classify_c_param(p)],
            "context_bundles": [
                p["type"] for p in cps if "context" in classify_c_param(p)
            ],
            "evidence": tw_ev["evidence"],
            "confidence": conf,
        }
        out[key] = entry
        comm.setdefault(s.qualified + "@" + s.file, []).append(key)
    for k, e in out.items():
        if "many_to_one" in tw["symbols"][k]:
            e["many_to_one"] = tw["symbols"][k]["many_to_one"]
    return out, sorted(dropped)


def build_sigmap(tree: rsrc.Tree, tw: dict) -> dict:
    hdrs = sorted(
        os.path.join(tree.include, f)
        for f in os.listdir(tree.include)
        if f.endswith((".h", ".hpp"))
    )
    hdr_all = rsrc.CIndex(tree, sorted(set(hdrs) | set(tree.bwd_cxx())))
    ir = build_ir_sigs(tree, hdr_all)
    emit_ms = builder_names(ir)
    emits_map = helper_emits(tree, emit_ms)
    helpers, dropped = build_helper_sigs(tree, tw, hdr_all, emits_map, set())
    ext = {}
    for mod, names in sorted(external_imports(tree).items()):
        rel = mod.split(".")
        pyfile = os.path.join(tree.pyroot, *rel) + ".py"
        if not os.path.exists(pyfile):
            pyfile = os.path.join(tree.pyroot, *rel, "__init__.py")
        if not os.path.exists(pyfile):
            ext[mod] = {"python_file": None, "symbols": {}}
            continue
        syms = {s.qual: s for s in rsrc.py_symbols(pyfile)}
        ent = {}
        for n in sorted(names):
            ps = syms.get(n)
            if ps is None:
                ent[n] = {"kind": "unresolved"}
                continue
            keys = rsrc.py_keys(ps)
            cands = []
            for k in keys:
                cands += [
                    c
                    for c in hdr_all.by_key.get(k, [])
                    if c.file.endswith(("." + rel[-1] + ".h", "." + rel[-1] + ".hpp"))
                    or True
                ]
            cands = [c for c in cands if c.kind in rsrc._KIND_OK[ps.kind]]
            same = [c for c in cands if rel[-1] in c.file]
            pick = (same or cands)[:1]
            e = {
                "kind": ps.kind,
                "python_line": ps.line,
                "params": ps.params,
                "returns": ps.returns,
            }
            if pick:
                c = pick[0]
                e["c"] = {
                    "name": c.qualified,
                    "file": c.file,
                    "line": c.line,
                    "kind": c.kind,
                }
                if c.kind == "function":
                    cps = c_params(c)
                    sm = slot_map(ps.params, cps, False, True)
                    e["c"].update(
                        {"params": cps, "slots": sm["slots"], "extra_c": sm["extra_c"]}
                    )
            else:
                e["c"] = None
            ent[n] = e
        ext[mod] = {"python_file": tree.rel(pyfile), "symbols": ent}
    # summary
    m = ir["methods"]
    pub = {k: v for k, v in m.items() if not v["private"]}
    summary = {
        "ir_builder_methods": len(m),
        "ir_builder_public": len(pub),
        "ir_builder_with_c_twin": sum(1 for v in m.values() if v["c"]),
        "ir_builder_public_with_c_twin": sum(1 for v in pub.values() if v["c"]),
        "ir_builder_emitting": sum(1 for v in m.values() if v["emits_ir"]),
        "ir_builder_exact_slots": sum(
            1 for v in m.values() if v.get("match") == "exact"
        ),
        "ir_builder_opts_struct": sum(
            1 for v in m.values() if v.get("match") == "opts-struct"
        ),
        "ir_builder_reordered": sum(
            1 for v in m.values() if v.get("match") == "reordered"
        ),
        "ir_builder_arity_differs": sum(
            1 for v in m.values() if v.get("match") == "arity-differs"
        ),
        "ir_builder_without_c": sorted(k for k, v in m.items() if not v["c"]),
        "c_functions_without_python_method": ir["c_only"],
        "helpers_with_signature": len(helpers),
        "helpers_renamed_medium": sum(
            1 for v in helpers.values() if v["confidence"] == "medium"
        ),
        "helpers_dropped_no_name_match": len(dropped),
        "helpers_emitting": sum(1 for v in helpers.values() if v["emits_ir"]),
        "helpers_with_context_bundle": sum(
            1 for v in helpers.values() if v["context_bundles"]
        ),
        "helpers_with_opts_struct": sum(
            1 for v in helpers.values() if v["opts_structs"]
        ),
        "helpers_many_to_one": sum(1 for v in helpers.values() if "many_to_one" in v),
        "twin_map_many_to_one": sum(
            1 for v in tw["symbols"].values() if "many_to_one" in v
        ),
        "external_helpers_imported": sum(len(v["symbols"]) for v in ext.values()),
        "external_helpers_with_c": sum(
            1 for v in ext.values() for s in v["symbols"].values() if s.get("c")
        ),
    }
    return {
        "summary": summary,
        "ir_builder": ir["methods"],
        "helpers": helpers,
        "helpers_dropped": dropped,
        "external_helpers": ext,
    }


# ------------------------------------------------------------------ owners

# Names another family defines on purpose, per the conventions file.
SHARED_HEADERS = ["helper_rocke.helpers.attention_band.h"]


def include_graph(tree: rsrc.Tree, files: List[str]) -> Dict[str, Set[str]]:
    by_base = {os.path.basename(f): tree.rel(f) for f in files}
    g: Dict[str, Set[str]] = {}
    for f in files:
        rel = tree.rel(f)
        deps = set()
        for line in open(f, encoding="utf-8", errors="replace"):
            m = re.match(r'\s*#\s*include\s*"rocke/([^"]+)"', line)
            if m and m.group(1) in by_base:
                deps.add(by_base[m.group(1)])
        g[rel] = deps
    return g


def reaches(g: Dict[str, Set[str]], a: str, b: str) -> bool:
    """True when file a includes file b, directly or not."""
    seen, todo = set(), [a]
    while todo:
        x = todo.pop()
        if x == b:
            return True
        if x in seen:
            continue
        seen.add(x)
        todo += list(g.get(x, ()))
    return False


def build_owners(tree: rsrc.Tree, tw: dict, parity: bool = False) -> dict:
    files = tree.bwd_cxx(parity)
    shared = [
        os.path.join(tree.include, h)
        for h in SHARED_HEADERS
        if os.path.exists(os.path.join(tree.include, h))
    ]
    idx = rsrc.CIndex(tree, files + shared)
    g = include_graph(tree, files + shared)
    # reverse twin map: C symbol -> python module(s) it is a twin of
    rev: Dict[str, Set[str]] = {}
    for e in tw["symbols"].values():
        if e["status"] != "ported":
            continue
        for t in e["twins"]:
            rev.setdefault(t["symbol"].split("::")[-1], set()).add(e["module"])
    mod_files = {m: v["cxx_files"] for m, v in tw["modules"].items()}

    groups: Dict[tuple, List[cscan.Sym]] = {}
    for s in idx.syms:
        if s.kind in ("namespace", "enum", "struct") and not s.is_def:
            continue
        if s.kind == "namespace" or s.guard or s.in_struct or not s.is_def:
            continue
        if (
            s.kind == "function"
            and s.linkage == "external"
            and "::" in s.scope
            and False
        ):
            continue
        kind = (
            "alias"
            if s.kind == "typedef"
            else ("type" if s.kind in ("struct", "enum", "union", "class") else s.kind)
        )
        sig = ""
        if s.kind == "function":
            # overloads are not duplicates: compare parameter types only
            sig = (
                "("
                + ",".join(
                    (
                        re.sub(r"\s*[A-Za-z_]\w*\s*(\[[^\]]*\])?\s*$", "", p)
                        if re.search(r"[\s*&]\w+\s*$", p)
                        else p
                    )
                    for p in cscan._split_top(s.params)
                    if p.strip() not in ("", "void")
                )
                + ")"
            )
        groups.setdefault((s.qualified + sig, kind), []).append(s)

    dups = []
    owners: Dict[str, str] = {}
    for (qual, kind), defs in sorted(groups.items()):
        files_of = sorted({d.file for d in defs})
        # a struct and its typedef share a file; only count distinct files
        if len(files_of) < 2:
            # a shared symbol that has one definition: the file is the owner
            d = defs[0]
            if d.file.endswith((".h", ".hpp")) and d.linkage == "external":
                owners[qual.split("(")[0]] = d.file
            continue
        ext = [d for d in defs if d.linkage == "external"]
        if kind == "macro":
            cat = "macro"
        elif ext and kind != "alias":
            cat = "external"
        else:
            cat = "internal-copy"
        values = sorted({d.value for d in defs if d.value})
        owner, rule = pick_owner(qual.split("(")[0], defs, rev, mod_files, g, tree)
        dups.append(
            {
                "name": qual.split("(")[0],
                "signature": qual[len(qual.split("(")[0]) :] or None,
                "kind": kind,
                "category": cat,
                "severity": "error" if cat in ("external", "macro") else "warning",
                "owner": owner,
                "owner_rule": rule,
                "same_value": (
                    len(values) <= 1 if kind in ("macro", "variable") else None
                ),
                "definitions": [
                    {"file": d.file, "line": d.line, "linkage": d.linkage}
                    for d in sorted(defs, key=lambda d: (d.file, d.line))
                ],
            }
        )
        owners[qual.split("(")[0]] = owner
    # API symbols that are only declared (a prototype or extern): the header is the owner
    for s in idx.syms:
        if (
            s.is_def
            or s.guard
            or s.in_struct
            or s.kind in ("namespace", "struct", "enum", "union", "class")
        ):
            continue
        if s.file.endswith((".h", ".hpp")) and s.linkage == "external":
            owners.setdefault(s.qualified, s.file)
    cat_count: Dict[str, int] = {}
    for d in dups:
        cat_count[d["category"]] = cat_count.get(d["category"], 0) + 1
    return {
        "summary": {
            "files": len(files),
            "shared_symbols": len(owners),
            "duplicates": len(dups),
            "by_category": cat_count,
        },
        "owners": dict(sorted(owners.items())),
        "duplicates": dups,
    }


def pick_owner(qual: str, defs: List[cscan.Sym], rev, mod_files, g, tree) -> tuple:
    def_files = sorted({d.file for d in defs})
    bare = qual.split("::")[-1]
    # 1. the file of the Python module this symbol is the twin of
    for m in sorted(rev.get(bare, ())):
        hit = [f for f in mod_files.get(m, []) if f in def_files]
        if hit:
            hdr = [f for f in hit if f.endswith((".h", ".hpp"))]
            return (hdr or hit)[0], f"twin of a symbol of {m}"
    # 2. a shared band header named in the conventions
    shared = [f for f in def_files if os.path.basename(f) in SHARED_HEADERS]
    if shared:
        return shared[0], "documented shared helper header"
    # 3. a file every other definition's file includes
    for f in [f for f in def_files if f.endswith((".h", ".hpp"))] + def_files:
        if all(reaches(g, o, f) for o in def_files if o != f):
            return f, "included by every other defining file"
    hdr = [f for f in def_files if f.endswith((".h", ".hpp"))]
    return (hdr or def_files)[
        0
    ], "no common owner: first header (move to a shared header)"


# ------------------------------------------------------------------ main


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def repo_root(path: str) -> str:
    """The git checkout that holds path (or path itself when there is none)."""
    p = os.path.realpath(path)
    q = p
    while q != os.path.dirname(q):
        if os.path.exists(os.path.join(q, ".git")):
            return q
        q = os.path.dirname(q)
    return p


def inside(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rocke", default=rsrc.DEFAULT_ROCKE, help="rocKE root (read only)"
    )
    ap.add_argument(
        "--out", default=DEFAULT_OUT, help="output dir (default: %(default)s)"
    )
    ap.add_argument(
        "--only",
        choices=["sigmap", "twin_map", "owners"],
        action="append",
        help="write only this map",
    )
    ap.add_argument(
        "--json", action="store_true", help="print the coverage summary as JSON"
    )
    ap.add_argument(
        "--parity",
        action="store_true",
        help="include the package's parity drivers (library/tests/parity/*bwd*_emit.c for attention backward)",
    )
    a = ap.parse_args(argv)
    tree = rsrc.Tree(a.rocke)
    err = tree.check()
    if err:
        print("error:", err, file=sys.stderr)
        return 2
    if inside(a.out, tree.root) or inside(a.out, repo_root(tree.root)):
        print(
            f"error: --out {a.out} is inside the rocKE checkout, which is read only",
            file=sys.stderr,
        )
        return 2
    want = set(a.only or ["sigmap", "twin_map", "owners"])
    try:
        return write_maps(tree, a, want)
    except (EnvError, OSError, SyntaxError, UnicodeDecodeError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


def write_maps(tree, a, want) -> int:
    tw = build_twin(tree)
    summary = {}
    if "twin_map" in want:
        write_json(os.path.join(a.out, "twin_map.json"), tw)
        summary["twin_map"] = tw["coverage"]["all"]
        summary["twin_map_public"] = tw["coverage"]["public"]
        summary["twin_map_public_in_modules_with_cxx"] = tw["coverage"][
            "public_in_modules_with_cxx"
        ]
    if "sigmap" in want:
        sg = build_sigmap(tree, tw)
        write_json(os.path.join(a.out, "sigmap.json"), sg)
        summary["sigmap"] = sg["summary"]
    if "owners" in want:
        ow = build_owners(tree, tw, a.parity)
        write_json(os.path.join(a.out, "owners.json"), ow)
        summary["owners"] = ow["summary"]
    if a.json:
        print(json.dumps(summary, indent=1, sort_keys=True))
    else:
        for k, v in summary.items():
            print(f"{k}: {json.dumps(v, sort_keys=True)}")
        print(f"wrote {', '.join(sorted(want))} to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
