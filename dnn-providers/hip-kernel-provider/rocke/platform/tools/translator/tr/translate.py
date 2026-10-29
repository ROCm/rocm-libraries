"""Translate one Python builder module to its C++ twin (.cpp).

usage: python -m tr.translate <module> [--rocke DIR] [--out FILE] [--analyze]

The existing header stays the contract: every public function keeps its C
signature (read from the header through sigmap), so the existing parity driver
links and the byte-identity gate checks the result unchanged. Each Python
function becomes an internal C++ function with natural types plus, when it is
public, an extern "C" shim that adapts the header's ABI. A function the rules
cannot lower is a hole: its refusal is reported, and a checked-in fill under
holes/<module>/<function>.cpp is used when one exists.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

from . import ctypes as T
from .env import (
    ROCKE_HELP,
    Env,
    holes_dir,
    library_dir,
    maps_dir,
    module_path,
    rocke_root,
)
from .family import kind_const
from .host import HOST_VALUE, IR_KIND, host_builders, kind_tags
from .lower import (
    Fn,
    Module,
    Refuse,
    Sig,
    decl_type,
    lit_code,
    py_params,
    split_block_locals,
)
from .package import PKG

CLANG_FORMAT: str | None = None  # --clang-format; see find_clang_format


class Result:
    def __init__(self):
        self.functions = []  # dicts: name, status (STATUSES), reason, line
        self.clones = (
            []
        )  # host clones: dicts name, host, status (translated|hole), reason
        # what tr.abi refused to put in the generated header (--abi only):
        # name -> reason; the header lacks each of them
        self.abi_refused: dict[str, str] = {}
        # typedef names the source would see defined twice, differently (two
        # headers it needs define them): name -> why; the source cannot compile
        self.include_conflicts: dict[str, str] = {}
        self.code = ""


# a function's status: translated by rule, derived (a C-only companion such
# as _free or _default, generated whole), filled by hand, or still a hole.
# The first two are mechanical.
STATUSES = ("translated", "derived", "filled", "hole")


def tally(functions: list[dict]) -> dict[str, int]:
    """How many functions have each status (STATUSES)."""
    out = dict.fromkeys(STATUSES, 0)
    for f in functions:
        out[f["status"]] += 1
    return out


def summary(functions: list[dict], abi_refused: dict | None = None) -> str:
    """The one summary line of a module, shared by --analyze and tr.regen:
    "<mech>/<n> functions mechanical (<t> translated, <d> derived),
    <f> hand fills, <h> holes, <r> ABI refusals". An ABI refusal is a
    struct, function or constant tr.abi left out of the generated header
    (Result.abi_refused): the header is incomplete while any is left."""
    c = tally(functions)
    mech = c["translated"] + c["derived"]
    return (
        f"{mech}/{len(functions)} functions mechanical"
        f" ({c['translated']} translated, {c['derived']} derived),"
        f" {c['filled']} hand fills, {c['hole']} holes,"
        f" {len(abi_refused or {})} ABI refusals"
    )


def refusal_line(name: str, why: str) -> str:
    """One ABI refusal's --analyze line."""
    return f"refused    {name} (not exported: {why})"


def conflict_line(name: str, why: str) -> str:
    """One include conflict's --analyze line (Result.include_conflicts)."""
    return f"conflict   {name} ({why})"


def report_line(f: dict) -> str:
    """One function's --analyze line: its status, name and, for a hole, why;
    for a hand fill, the reason the rule failed that the fill stands in for."""
    if f["status"] == "hole":
        extra = f" ({f['reason']} at line {f['at']})"
    elif f["status"] == "filled":
        extra = f" (hand fill for: {f['reason']} at line {f['at']})"
    else:
        extra = ""
    return f"{f['status']:10} {f['name']}{extra}"


def own_header(env: Env, modname: str) -> str | None:
    for cand in PKG.header_names(modname):
        if cand in env.headers.paths:
            return cand
    return None


def generated_modules(abi_dir: str) -> list[str]:
    """Modules whose generated header is in abi_dir."""
    out = []
    for fn in sorted(os.listdir(abi_dir)):
        m = PKG.header_module(fn)
        if m:
            out.append(m)
    return out


def abi_refusals(abi) -> dict[str, str]:
    """abi.refused by name (a name refused twice, for two reasons, once)."""
    out: dict[str, str] = {}
    for name, why in abi.refused:
        if name not in out:
            out[name] = why
        elif why not in out[name].split("; "):
            out[name] += f"; {why}"
    return out


def install_abi(env: Env, mod: Module):
    """Generated ABI: the module's own sigmap entries come from tr/abi.py, not
    from the mined hand header."""
    from .abi import Abi

    abi = Abi(env, mod).run()
    for k in [k for k in env.sig.helpers if k.startswith(mod.modname + ":")]:
        if k.split(":", 1)[1] not in abi.hand_quals:  # ABI holes keep the hand entry
            del env.sig.helpers[k]
    for cls in abi.structs:
        sname = abi.struct_name(cls)
        if sname in env.headers.structs:
            mod.class_struct[cls] = sname
            mod.struct_class[sname] = cls
    for cls in mod.classes:
        fam = abi.fam(cls)
        if (
            fam is not None
            and fam.module == mod.modname
            and fam.base in abi.structs
            and abi.struct_name(fam.base) in env.headers.structs
        ):
            # tagged values: every member of a family is the family's one tagged struct
            mod.class_struct[cls] = abi.struct_name(fam.base)
            mod.struct_class[abi.struct_name(fam.base)] = fam.base
    for d in abi.decls:
        if "<" in d.py:
            continue  # derived (_free, _default)
        cls, _, meth = d.py.rpartition(".")
        fn = mod.methods[(cls, meth)] if cls else mod.functions[d.py]
        env.sig.helpers[mod.qual(d.py)] = {
            "c": {
                "name": d.name,
                "params": [{"name": n, "type": t} for t, n in d.params],
                "returns": d.ret,
                "slots": d.slots,
                "extra_c": d.extra,
                "takes_builder": any(t == T.B for t, _ in d.params),
                "generated": True,
            },
            "python": {"params": py_params(fn)},
            "kind": "method" if cls else "function",
        }
    return abi


def calls_in(fn: ast.FunctionDef, names: set[str]) -> set[str]:
    out = set()
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in names
        ):
            out.add(n.func.id)
    return out


def topo(mod: Module, roots: list[str]) -> list[str]:
    names = set(mod.functions)
    order, seen = [], set()

    def visit(n, stack):
        if n in seen:
            return
        if n in stack:
            return
        stack.add(n)
        for c in sorted(calls_in(mod.functions[n], names)):
            visit(c, stack)
        seen.add(n)
        order.append(n)

    for r in roots:
        visit(r, set())
    return order


def abi_of(env: Env, mod: Module, name: str):
    """The public C twin: declared in this module's own header, nowhere else."""
    e = env.sig.helpers.get(mod.qual(name))
    if e is None:
        return None, None
    cf = env.c_func(e["c"]["name"])
    if cf is None or cf.file != own_header(env, mod.modname):
        return None, None
    return e, cf


def internal_sig(
    env: Env,
    mod: Module,
    fn: ast.FunctionDef,
    entry,
    cf,
    cls: str | None = None,
    fence: bool = False,
    seq_guards: bool = False,
    trust_c: bool = True,
) -> Sig:
    if not trust_c:
        # a kept hand declaration (ABI hole) with a generated ABI: the internal
        # function takes its types from Python; only the shim maps to the C shape
        entry = None
    slots = {s["py"]: s for s in entry["c"]["slots"]} if entry else {}
    extra = {x["c"]: x["role"] for x in entry["c"].get("extra_c", [])} if entry else {}
    cparams = dict((n, t) for t, n in cf.params) if cf else {}
    cnames = [n for _, n in cf.params] if cf else []
    params = []
    pyps = py_params(fn)
    is_classmethod = any(
        isinstance(d, ast.Name) and d.id == "classmethod" for d in fn.decorator_list
    )
    if cls is not None:
        sname = mod.class_struct.get(cls)
        if sname is None:
            raise Refuse(fn, f"class {cls} has no C struct")
        first = pyps.pop(0)
        if not is_classmethod:
            st = (
                next(
                    (
                        t
                        for t, n in cf.params
                        if T.struct_of(t) and T.struct_of(t)[0] == sname
                    ),
                    None,
                )
                if cf
                else None
            )
            if st is None:
                from .abi import method_mutating

                mut = fn.name in ("__init__", "__post_init__") or method_mutating(
                    env, mod, cls, fn
                )
                st = f"{sname}*" if mut else f"const {sname}*"
            params.append((first["name"], st))
    has_b = any(p["annotation"] == "IRBuilder" or p["name"] == "b" for p in pyps)
    if cf is not None and not has_b and any(t == T.B for t, _ in cf.params):
        params.append(("b", T.B))
    defaults = {}
    a = fn.args
    plain = a.posonlyargs + a.args
    for i, x in enumerate(plain):
        if i >= len(plain) - len(a.defaults):
            defaults[x.arg] = a.defaults[i - (len(plain) - len(a.defaults))]
    for x, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            defaults[x.arg] = d
    fact = mod.fact_of(fn) or {"params": {}, "return": []}

    def expand(txt):
        for al, body in mod.type_aliases.items():
            txt = re.sub(rf"\b{al}\b", body, txt)
        return txt

    for p in pyps:
        ann_txt = expand(p["annotation"]) if p["annotation"] else ""
        ann = (
            T.from_annotation(
                ann_txt,
                {**(mod.tmap.class_struct if mod.tmap else {}), **mod.class_struct},
            )
            if ann_txt and not re.search(r"\bAny\b", ann_txt)
            else None
        )
        if ann is None and p["name"] in fact["params"] and mod.tmap is not None:
            ann = mod.tmap.tags(fact["params"][p["name"]])
        s = slots.get(p["name"])
        ct = cparams.get(s["c"]) if s else None
        seq_c = False
        if ct is not None:
            i = cnames.index(s["c"])
            seq_c = i + 1 < len(cnames) and extra.get(cnames[i + 1]) == "count"
        if p["annotation"] == "IRBuilder":
            t = T.B
        elif s and s.get("dict"):
            if ann is None or not ann.startswith("py::dict<"):
                raise Refuse(fn, f"no C++ type for the dict parameter {p['name']}")
            t = f"const {ann}&"
        elif seq_c and T.pointee_elem(ct):
            t = T.vec(T.pointee_elem(ct))
        elif (
            ct is not None
            and not (ct.endswith("* const*") or ct.endswith("**"))
            and not (ann and T.is_seq(ann))
        ):
            t = ct
        elif ann is not None:
            t = ann
            st_ = T.struct_of(t)
            if (
                st_
                and not st_[1]
                and t in env.headers.structs
                and t not in ("py::kargs", T.IV)
            ):
                t = f"const {t}*"
        elif ct is not None:
            t = ct
        elif p["name"] == "b" and (
            not p["annotation"] or re.fullmatch(r"\s*Any\s*", p["annotation"])
        ):
            # the builder by this codebase's naming (never recorded: no call in the
            # tests); `b: Any` carries no type either (B-R2)
            t = T.B
        elif cls is not None and cls in host_builders(mod):
            t = HOST_VALUE  # host clones: a host builder computes on ints (never recorded: no call in the tests)
        else:
            raise Refuse(fn, f"no type for parameter {p['name']}")
        if s and s.get("has") and not T.optional_inner(t):
            t = T.optional(t)  # a generated has_ flag: the function sees an optional
        elif (
            p["annotation"]
            and re.search(r"\|\s*None\b|\bNone\s*\||Optional\[", p["annotation"])
            and not (
                T.is_ptr(t)
                or t in (T.STR, T.B)
                or T.optional_inner(t)
                or t == "std::nullptr_t"
            )
        ):
            t = T.optional(
                t
            )  # annotated `X | None`: None stays possible even if never recorded
        params.append((p["name"], t))
    if fence:
        params.append(("fence", T.STR))
    if seq_guards:
        params.append(("seq_guards", "int*"))
    ret_ann = ast.unparse(fn.returns) if fn.returns is not None else None
    ret = (
        T.from_annotation(ret_ann, mod.class_struct)
        if ret_ann and not re.search(r"\bAny\b", ret_ann)
        else None
    )
    if (
        ret
        and "None" in ret_ann
        and ret.startswith("const ")
        and ret.endswith("*")
        and T.struct_of(ret)
        and T.struct_of(ret)[0] in env.headers.structs
        and not ret.startswith(
            ("const rocke_mma_op", "const rocke_layout_map", "const rocke_arch_target")
        )
    ):
        # an optional struct the function builds: returned by value (a pointer would dangle)
        ret = T.optional(T.struct_of(ret)[0])
    if ret is None and fact["return"] and mod.tmap is not None:
        rs = [r for r in fact["return"] if r != "None"] or ["None"]
        full = (
            mod.tmap.tags(fact["return"], param=False)
            if "None" in fact["return"] and rs != ["None"]
            else None
        )
        ret = (
            T.VOID
            if rs == ["None"]
            else (
                full
                if full and T.optional_inner(full)
                else mod.tmap.tags(rs, param=False)
            )
        )
    if (
        ret is None
        and ret_ann is not None
        and re.fullmatch(r"(?:dict|Dict)\[str, ?Any\]|dict", ret_ann)
    ):
        ret = T.AUTO  # a record (R-REC): the struct comes from the return statements
    if ret is None and (
        fn.returns is None
        or (re.search(r"\bAny\b", ret_ann or "") and not fact["return"])
    ):
        ret = T.AUTO  # inferred from the return statements (an unrecorded Any too)
    if ret is None:
        raise Refuse(
            fn,
            f"no C++ type for the return {ast.unparse(fn.returns) if fn.returns else '(none)'}",
        )
    if ret == T.STR:
        ret = T.SSTR  # an owned string: a pointer into a temporary would dangle
    if not any(t == T.B for _, t in params) and needs_builder(fn):
        params.insert(0, ("b", T.B))
    name = f"py_{cls}_{fn.name}" if cls else f"py_{fn.name}"
    return Sig(name, params, ret, fn, defaults)


def lifted_sig(mod: Module, site) -> Sig:
    """host clones: the IR signature of a closure stored into a Callable field: its own
    parameters, then the values it captured (designs/host_clones.md)."""
    from .host import site_types
    from .lower import builder_param

    fn = site.node
    fact = mod.fact_of(fn)
    caps = site_types(mod, site)
    if not fact or caps is None:
        raise Refuse(fn, f"closure {fn.name}: no recorded types")
    bp = builder_param(fn)
    params = []
    for p in site.own:
        t = (
            T.B
            if p == bp
            else (
                mod.tmap.tags(kind_tags(fact["params"][p], IR_KIND))
                if p in fact["params"]
                else None
            )
        )
        if t is None:
            raise Refuse(fn, f"closure parameter {p}: no recorded type")
        params.append((p, t))
    for (cap, _k), tg in zip(site.captured, caps):
        t = mod.tmap.tags(kind_tags(tg, IR_KIND))
        if t is None:
            raise Refuse(fn, f"closure {fn.name}: no C++ type for captured {cap}")
        params.append((cap, t))
    rs = [r for r in fact["return"] if r != "None"] or ["None"]
    ret = (
        T.VOID if rs == ["None"] else mod.tmap.tags(kind_tags(rs, IR_KIND), param=False)
    )
    if ret is None:
        raise Refuse(fn, f"closure {fn.name}: no C++ type for the return")
    return Sig(f"py_{site.name}__{fn.name}", params, ret, fn)


def needs_builder(fn: ast.FunctionDef) -> bool:
    """A function that creates the builder or builds IR types needs the C caller's builder."""
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("IRBuilder", "PtrType", "VectorType")
        ):
            return True
    return False


def lower_function(mod: Module, fn: ast.FunctionDef, sig: Sig) -> list[str]:
    hints = {"vars": {}, "expr": {}, "elem": {}}
    saved_consts = dict(mod.private_consts)
    split = set()
    while True:
        try:
            Fn(
                mod,
                fn,
                Sig(
                    sig.name, list(sig.params), sig.ret, fn, sig.defaults, host=sig.host
                ),
                hints,
                dry=True,
            ).run()
            break
        except Refuse as e:
            mod.private_consts = dict(saved_consts)
            m = re.match(r"local (\w+) holds both ", e.reason)
            fn2 = (
                split_block_locals(fn, m.group(1))
                if m and m.group(1) not in split
                else None
            )
            if fn2 is None:
                mod.private_consts = saved_consts
                raise
            split.add(m.group(1))
            fn = fn2  # tagged values: one C++ local per returning block
            hints = {"vars": {}, "expr": {}, "elem": {}}
    mod.private_consts = saved_consts
    if hints.get("ret_widen") and (
        fn.returns is None or re.search(r"\bAny\b", ast.unparse(fn.returns))
    ):
        sig.ret = hints[
            "ret_widen"
        ]  # the dry pass saw a wider return (an unrecorded None)
    body = Fn(mod, fn, sig, hints).run()
    used = re.sub(
        r'"(?:[^"\\]|\\.)*"', '""', "\n".join(body)
    )  # names inside string literals are not uses
    ps = []
    stored = {
        x.id
        for x in ast.walk(fn)
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)
    }
    for n, t in sig.params:
        mu = "" if re.search(rf"\b{re.escape(n)}\b", used) else "[[maybe_unused]] "
        dt = decl_type(t)
        if T.is_seq(t) and n not in stored:
            dt = f"const {dt}&"
        ps.append(f"{mu}{dt} {n}")
    head = f"{decl_type(sig.ret)} {sig.name}({', '.join(ps)})"
    out = (
        [f"/* py: {fn.name} ({os.path.basename(mod.path)}:{fn.lineno}) */", head, "{"]
        + body
        + ["}"]
    )
    return out


def ok_reason_body(call: str) -> list[str]:
    """The body of a C validity gate `bool f(..., char* reason, size_t
    reason_cap)` whose Python returns (ok, reason): the reason copied into the
    caller's buffer (truncated, NUL-terminated), ok returned. Both shim paths
    use it inside py_adapt::with_reason (a raise: its text, false)."""
    return [
        f"const auto res_ = {call};",
        "if(reason != nullptr && reason_cap > 0) {"
        " const std::string why_(std::get<1>(res_));"
        " strncpy(reason, why_.c_str(), reason_cap - 1);"
        " reason[reason_cap - 1] = '\\0'; }",
        "return std::get<0>(res_);",
    ]


def is_ok_reason(t: str) -> bool:
    """A Python (ok, reason) result: std::tuple<bool, str>."""
    items = T.tuple_items(t)
    return items is not None and len(items) == 2 and items[0] == T.BOOL and T.is_str(items[1])


# C int returns a Python int (int64_t inside) narrows to, with a range check
NARROW_INTS = ("int", "int32_t")


def header_doc(env: Env, cf) -> str:
    """The comment block right above a header declaration, markers dropped and
    whitespace collapsed ("" when there is none)."""
    path = env.headers.paths.get(cf.file)
    if path is None or not cf.line:
        return ""
    text = open(path).read().splitlines()
    i = cf.line - 2
    if i < 0 or not text[i].rstrip().endswith("*/"):
        return ""
    block = []
    while i >= 0:
        block.insert(0, text[i])
        if text[i].lstrip().startswith("/*"):
            break
        i -= 1
    doc = re.sub(r"/\*+|\*+/", " ", "\n".join(block))
    doc = re.sub(r"^\s*\*", " ", doc, flags=re.M)
    return re.sub(r"\s+", " ", doc).strip()


def documented_fail(env: Env, cf) -> tuple[str, str, list[str]] | None:
    """What the hand header says an int entry returns when the Python raises:
    (value, the sentence, the pointer parameters it names as NULL), from a
    doc comment that documents a failure value, such as "Returns 0 for a
    NULL spec or vec == 0 (where the Python raises ZeroDivisionError)." None
    when the header documents no such value (then the C entry has no
    convention for a Python failure)."""
    doc = header_doc(env, cf)
    m = re.search(
        r"\b[Rr]eturns?\s+(-?\d+)\s+((?:for|on|when|if)\b[^.()]*(?:\([^)]*\))?[^.]*)",
        doc,
    )
    # a failure case: a NULL argument, a raise, an error, or a zero divisor
    # ("vec == 0"); "Returns 1 if the spec is valid" is not one
    fails = r"\bNULL\b|rais|error|invalid|fail|==\s*0\b|\bzero\b"
    if not m or not re.search(fails, m.group(2), re.I):
        return None
    cond = m.group(2).strip()
    ptrs = {n for t, n in cf.params if t.endswith("*")}
    named = {
        a or b_
        for a, b_ in re.findall(r"\bNULL\s+(\w+)|\b(\w+)\s*(?:==|is)\s*NULL", cond)
    }
    guards = [n for _, n in cf.params if n in ptrs and n in named]
    return m.group(1), f"Returns {m.group(1)} {cond}", guards


def make_shim(
    env: Env,
    mod: Module,
    fn: ast.FunctionDef,
    sig: Sig,
    entry,
    cf,
    fence_arg: str = "nullptr",
    is_init: bool = False,
) -> list[str]:
    """extern "C" entry with the header's signature, calling the internal function."""
    slots = {s["py"]: s for s in entry["c"]["slots"]}
    extra = {x["c"]: x["role"] for x in entry["c"].get("extra_c", [])}
    cnames = [n for _, n in cf.params]
    ctypes = {n: t for t, n in cf.params}
    pro, args = [], []
    used = set()
    for pname, t in sig.params:
        if pname == "b" and t == T.B:
            args.append("b")
            used.add("b")
            continue
        if pname == "fence":
            args.append(fence_arg)
            if fence_arg != "nullptr":
                used.add(fence_arg)
            continue
        if pname == "self" or (
            pname == sig.params[0][0] and T.struct_of(t) and pname not in slots
        ):
            c = next(
                n
                for ct_, n in cf.params
                if T.struct_of(ct_) and T.struct_of(ct_)[0] == T.struct_of(t)[0]
            )
            args.append(c)
            used.add(c)
            continue
        s = slots.get(pname)
        if s is None:
            raise Refuse(fn, f"shim: Python parameter {pname} has no C parameter")
        c = s["c"]
        used.add(c)
        ct = ctypes[c]
        if T.is_seq(t):
            i = cnames.index(c)
            cnt = next((n for n in cnames[i + 1 :] if extra.get(n) == "count"), None)
            if cnt is None:
                raise Refuse(fn, f"shim: sequence {pname} has no count parameter")
            used.add(cnt)
            pro.append(f"const {decl_type(t)} {pname}_v({c}, {c} + {cnt});")
            args.append(f"{pname}_v")
            continue
        if t == T.STR and s.get("default") not in (None, "None"):
            d = ast.literal_eval(s["default"])
            args.append(f"{c} != nullptr ? {c} : {lit_code(d)}")
            continue
        if ct == t or (T.struct_of(ct) and T.struct_of(ct) == T.struct_of(t)):
            args.append(c)
            continue
        raise Refuse(fn, f"shim: cannot pass C {ct} {c} as {t}")
    call = f"{sig.name}({', '.join(args)})"
    ctys = {n: t for t, n in cf.params}
    if sig.ret == "rocke_kernel_def_t*":
        # kernel entry: validate the spec (Python checks it at construction), then build
        pre = []
        for pname, t in sig.params:
            st_ = T.struct_of(t)
            cls_ = mod.struct_class.get(st_[0]) if st_ else None
            if cls_ and f"{cls_}.__post_init__" in mod.sigs:
                pre.append(
                    f"{mod.sigs[cls_ + '.__post_init__'].name}({args[[p for p, _ in sig.params].index(pname)]});"
                )
        inner = " ".join(pro + pre) + f" return {call};"
        ps = [f"{t} {n}" for t, n in cf.params]
        return [
            f"{cf.ret} {cf.name}({', '.join(ps)})",
            "{",
            f"    return py_adapt::kernel_entry(b, [&]() -> rocke_kernel_def_t* {{ {inner} }});",
            "}",
        ]
    if "reason" in ctys and "reason_cap" in ctys:
        fail = "-1" if T.is_int(cf.ret) else "false"
        val = "true" if sig.ret == T.VOID else None
        body_ = f"{call}; return true;" if val else f"return ({cf.ret}){call};"
        if cf.ret == "bool" and is_ok_reason(sig.ret):
            body_ = " ".join(ok_reason_body(call))  # the (ok, reason) validity gate
        elif T.tuple_items(sig.ret) is not None:
            raise Refuse(fn, f"shim: Python return {sig.ret} as C {cf.ret}")
        ps = [f"{t} {n}" for t, n in cf.params]
        return [
            f"{cf.ret} {cf.name}({', '.join(ps)})",
            "{",
            f"    return py_adapt::with_reason<{cf.ret}>(reason, reason_cap, {fail}, [&]() -> {cf.ret} {{ {' '.join(pro)} {body_} }});",
            "}",
        ]
    if (
        "out" in ctys
        and "out_cap" in ctys
        and cf.ret == "rocke_status_t"
        and T.is_str(sig.ret)
    ):
        ps = [f"{t} {n}" for t, n in cf.params]
        return [
            f"{cf.ret} {cf.name}({', '.join(ps)})",
            "{",
            f"    return py_adapt::str_out(out, out_cap, [&]() -> std::string {{ {' '.join(pro)} return {call}; }});",
            "}",
        ]
    for n in cnames:
        if extra.get(n) == "out" and "cap" in n and T.is_int(ctys[n]):
            extra[n] = "extra"
    outs = [n for n in cnames if extra.get(n) == "out"]
    body = list(pro)
    if is_init:
        selfc = args[0]
        body.insert(0, f"memset({selfc}, 0, sizeof(*{selfc}));")
    r = sig.ret
    if r == T.VOID:
        body.append(call + ";")
        if cf.ret == "bool":
            body.append("return rocke_ir_builder_ok(b);")
        elif cf.ret != "void":
            raise Refuse(fn, f"shim: None return as {cf.ret}")
    elif r == cf.ret and not outs:
        body.append(f"return {call};")
    elif (
        T.is_seq(r)
        and outs
        and T.seq_elem(T.seq_elem(r)) is not None
        and T.arr_len(T.seq_elem(r)) is None
    ):
        raise Refuse(fn, f"shim: list of lists {r} into flat out parameters")
    elif T.is_seq(r) and outs:
        el = T.seq_elem(r)
        k = T.arr_len(el) if T.seq_elem(el) else None
        body.append(f"const auto r = {call};")
        if k is not None and len(outs) == k:
            body.append("for(size_t i = 0; i < r.size(); ++i)")
            body.append("{")
            for j, o in enumerate(outs):
                body.append(f"    {o}[i] = r[i][{j}];")
            body.append("}")
        elif len(outs) == 1:
            cap = next(
                (n for n in cnames if extra.get(n) == "extra" and "cap" in n), None
            )
            if cap:
                used.add(cap)
                body.append(
                    f'if((int64_t)r.size() > {cap}) throw ckc::ValueError("result exceeds the output buffer");'
                )
            body.append(f"for(size_t i = 0; i < r.size(); ++i) {outs[0]}[i] = r[i];")
        else:
            raise Refuse(fn, f"shim: {r} into out parameters {outs}")
        used.update(outs)
        if cf.ret == "bool":
            body.append("return rocke_ir_builder_ok(b);")
        elif cf.ret != "void":
            raise Refuse(fn, f"shim: list return as {cf.ret}")
    elif T.is_int(r) and cf.ret in NARROW_INTS and r != cf.ret and not outs:
        # a Python int (int64_t) as a narrower C int: checked, never truncated
        body.append(f"const int64_t r_ = {call};")
        if "b" in used and any(t == T.B and n == "b" for t, n in cf.params):
            # a builder entry: out of range is a raise on the builder (-1)
            body.append(
                f'if((int64_t)({cf.ret})r_ != r_) throw ckc::ValueError("result does not fit in C {cf.ret}");'
            )
            body.append(f"return ({cf.ret})r_;")
        else:
            # the value the header documents for a Python raise, also for a
            # result C cannot hold; no documented value, no convention: a hole
            doc = documented_fail(env, cf)
            if doc is None:
                raise Refuse(
                    fn,
                    f"shim: Python return {r} as C {cf.ret}: the header documents"
                    " no return value for a Python raise",
                )
            val, why, guards = doc
            ps = [f"{t} {n}" if n in used else f"[[maybe_unused]] {t} {n}" for t, n in cf.params]
            out = [f"{cf.ret} {cf.name}({', '.join(ps)})", "{"]
            import textwrap

            note = textwrap.wrap(
                f"the header: {why.replace('*/', '* /')}; the same for a result"
                f" out of {cf.ret} range",
                80,
            )
            out.append("    /* " + "\n     * ".join(note) + " */")
            for g in guards:
                out.append(f"    if({g} == nullptr) return {val};")
            out.append("    try {")
            out += ["        " + x for x in body]
            out.append(f"        if((int64_t)({cf.ret})r_ == r_) return ({cf.ret})r_;")
            out.append("    } catch(const ckc::Error&) {}")
            out.append(f"    return {val};")
            out.append("}")
            return out
    else:
        raise Refuse(fn, f"shim: Python return {r} as C {cf.ret}")
    if (
        any(n == "b" for _, n in cf.params)
        and cf.ret == "bool"
        and "rocke_ir_builder_ok(b)" in "\n".join(body)
    ):
        used.add("b")
    ps = []
    for t, n in cf.params:
        ps.append(f"{t} {n}" if n in used else f"[[maybe_unused]] {t} {n}")
    head = f"{cf.ret} {cf.name}({', '.join(ps)})"
    if "b" in used and any(t == T.B and n == "b" for t, n in cf.params):
        # a Python raise is recorded on the builder, never escapes the C ABI
        inner = " ".join(body)
        if cf.ret == "void":
            return [
                head,
                "{",
                f"    py_adapt::on_builder<int>(b, 0, [&]() -> int {{ {inner} return 0; }});",
                "}",
            ]
        fail = {"bool": "false", "int": "-1"}.get(cf.ret, "nullptr")
        return [
            head,
            "{",
            f"    return py_adapt::on_builder<{cf.ret}>(b, {fail}, [&]() -> {cf.ret} {{ {inner} }});",
            "}",
        ]
    return [head, "{"] + ["    " + x for x in body] + ["}"]


def make_gen_shim(
    env: Env,
    mod: Module,
    fn: ast.FunctionDef,
    sig: Sig,
    entry,
    cf,
    is_init: bool = False,
) -> list[str]:
    """extern "C" entry of a generated ABI declaration (rules: designs/generated_abi.md)."""
    slots = {s["py"]: s for s in entry["c"]["slots"]}
    roles = {x["c"]: x["role"] for x in entry["c"]["extra_c"]}
    cnames = [n for _, n in cf.params]
    ctys = {n: t for t, n in cf.params}
    pro, args = [], []
    for i, (pname, t) in enumerate(sig.params):
        if pname == "b" and t == T.B:
            args.append("b")
            continue
        if pname in ("fence", "seq_guards") and pname not in slots:
            if pname not in ctys:
                raise Refuse(fn, f"gen shim: {pname} has no C parameter")
            args.append(pname)
            continue
        if i == 0 and "self" in ctys and pname not in slots:
            args.append("self")
            continue
        s = slots.get(pname)
        if s is None:
            raise Refuse(fn, f"gen shim: Python parameter {pname} has no C parameter")
        c = s["c"]
        if s.get("dict"):
            m = re.fullmatch(r"(?:const )?(py::dict<(.+)>)&?", t)
            if not m:
                raise Refuse(fn, f"gen shim: dict parameter {pname} lowered as {t}")
            t = m.group(1)
            vals = [
                n
                for n in cnames
                if roles.get(n) == "dict_vals" and n.startswith(pname + "_vals")
            ]
            vt = m.group(2).split(", ", 1)[1]
            v = (
                f"{vals[0]}[i]"
                if len(vals) == 1
                else f"{decl_type(vt)}{{{', '.join(x + '[i]' for x in vals)}}}"
            )
            pro.append(f"{t} {pname}_v;")
            pro.append(
                f"for(int i = 0; i < n_{pname}; ++i) {pname}_v.set({c}[i], {v});"
            )
            args.append(f"{pname}_v")
            continue
        cst = env.headers.structs.get(ctys.get(c, ""))
        if (
            cst is not None
            and "_tuple_" in ctys[c]
            and (T.arr_len(t) is not None or T.tuple_items(t))
        ):
            # a fixed tuple passed as its generated struct
            args.append(f"{decl_type(t)}{{{', '.join(f'{c}.{f}' for f in cst.order)}}}")
            continue
        if T.is_seq(t):
            pro.append(f"const {decl_type(t)} {pname}_v({c}, {c} + n_{pname});")
            args.append(f"{pname}_v")
            continue
        if t == T.STR and s.get("default") not in (None, "None"):
            args.append(
                f"{c} != nullptr ? {c} : {lit_code(ast.literal_eval(s['default']))}"
            )
            continue
        if s.get("has") and T.optional_inner(t):
            args.append(f"({s['has']} ? {decl_type(t)}({c}) : {decl_type(t)}())")
            continue
        args.append(c)
    call = f"{sig.name}({', '.join(args)})"
    body = list(pro)
    if is_init:
        cls_ = next((c for (c, m_), f_ in mod.methods.items() if f_ is fn), None)
        fam = mod.family(cls_)
        if fam is not None and cls_ != fam.base:
            body.insert(
                0, f"self->kind = {kind_const(fam, cls_)};"
            )  # tagged values: the member's tag
        body.insert(0, "memset(self, 0, sizeof(*self));")
    opt_out = next((n for n in cnames if roles.get(n) == "opt_out"), None)
    owned = [n for n in cnames if roles.get(n) == "owned_out"]
    fixed = next((n for n in cnames if roles.get(n, "").startswith("fixed_out")), None)
    if cf.ret == "rocke_kernel_def_t*":
        pre = []
        for pname, t in sig.params:
            st_ = T.struct_of(t)
            cls_ = mod.struct_class.get(st_[0]) if st_ else None
            if cls_ and f"{cls_}.__post_init__" in mod.sigs:
                pre.append(
                    f"{mod.sigs[cls_ + '.__post_init__'].name}({args[[p for p, _ in sig.params].index(pname)]});"
                )
        inner = " ".join(body + pre) + f" return {call};"
        wrap = ("kernel", inner)
    elif "out" in roles and roles["out"] == "str_out":
        wrap = ("str", " ".join(body) + f" return {call};")
    else:
        if owned:
            r = sig.ret
            body.append(f"const auto res_ = {call};")
            el = T.seq_elem(r)
            dm = re.fullmatch(r"py::dict<(.+)>", r)
            if el is None and not dm:
                raise Refuse(fn, f"gen shim: owned outputs from a {r}")
            for j, o in enumerate(owned):
                ot = ctys[o][:-1]  # T** -> T*
                if dm:
                    get = (
                        "x.first"
                        if o == "out_keys"
                        else (
                            "x.second"
                            if o == "out_vals"
                            else f"x.second[{o.rsplit('_', 1)[1]}]"
                        )
                    )
                elif len(owned) > 1:
                    get = f"x[{j}]" if T.arr_len(el) else f"std::get<{j}>(x)"
                else:
                    get = "x"
                tgt = ot[:-1]
                st_ = env.headers.structs.get(tgt)
                if st_ is not None and "_tuple_" in tgt:
                    conv = f"{tgt}{{{', '.join(f'({st_.fields[f]})std::get<{i}>({get})' for i, f in enumerate(st_.order))}}}"
                elif tgt == T.STR:
                    conv = f"py_adapt::own_str({get})"  # each string its own malloc
                else:
                    conv = f"({tgt}){get}"
                body.append(
                    f"py_adapt::own_out(res_, {o}, [](const auto& x) {{ return {conv}; }});"
                )
            body.append("*n_out = (int)res_.size();")
        elif opt_out:
            # Optional[X]: *has_out says whether there is one, *out the value
            body.append(f"const auto res_ = {call};")
            body.append("*has_out = res_.has_value();")
            inner = T.optional_inner(sig.ret) or ""
            el = ctys[opt_out][:-1]
            if T.arr_len(inner) is not None or T.tuple_items(inner):
                st_ = env.headers.structs.get(el)
                n = (
                    T.arr_len(inner)
                    if T.arr_len(inner) is not None
                    else len(T.tuple_items(inner))
                )
                val = (
                    f"{el}{{{', '.join(f'({st_.fields[st_.order[i]]})std::get<{i}>(*res_)' for i in range(n))}}}"
                    if st_
                    else "*res_"
                )
            else:
                val = f"({el})*res_"
            body.append(f"if(res_.has_value()) *{opt_out} = {val};")
        elif fixed:
            k = int(roles[fixed].split(":")[1])
            body.append(f"const auto res_ = {call};")
            body.append(
                f"for(int i = 0; i < {k}; ++i) {fixed}[i] = (int)res_[(size_t)i];"
            )
        elif roles.get("reason") == "reason_str":
            # (ok, reason) as the validity-gate ABI: the reason into the caller's
            # buffer, ok returned (a raise: with_reason writes its text, false)
            body += ok_reason_body(call)
        elif sig.ret == T.VOID:
            body.append(f"{call};")
        elif "_tuple_" in cf.ret and (
            T.arr_len(sig.ret) is not None or T.tuple_items(sig.ret)
        ):
            # std::array / std::tuple -> the generated tuple struct
            n = (
                T.arr_len(sig.ret)
                if T.arr_len(sig.ret) is not None
                else len(T.tuple_items(sig.ret))
            )
            body.append(f"const auto res_ = {call};")
            st_ = env.headers.structs.get(cf.ret)
            body.append(
                f"return {cf.ret}{{{', '.join((f'({st_.fields[st_.order[i]]})' if st_ else '') + f'std::get<{i}>(res_)' for i in range(n))}}};"
            )
        else:
            body.append(f"return {call};")
        if owned or fixed or opt_out or sig.ret == T.VOID:
            if cf.ret == "bool":
                body.append(
                    "return rocke_ir_builder_ok(b);"
                    if ctys.get("b") == T.B
                    else "return true;"
                )
            elif cf.ret != "void":
                raise Refuse(fn, f"gen shim: {sig.ret} as C {cf.ret}")
        wrap = ("plain", " ".join(body))
    ps = ", ".join(f"{t} {n}" for t, n in cf.params) or "void"
    head = f"{cf.ret} {cf.name}({ps})"
    kind, inner = wrap
    fail = {"bool": "false", "int": "-1", "double": "-1.0"}.get(cf.ret) or (
        "nullptr" if cf.ret.endswith("*") else f"{cf.ret}{{}}"
    )
    if kind == "kernel":
        out = f"return py_adapt::kernel_entry(b, [&]() -> rocke_kernel_def_t* {{ {inner} }});"
    elif kind == "str":
        out = f"return py_adapt::str_out(out, out_cap, [&]() -> std::string {{ {inner} }});"
    elif "reason" in ctys:
        out = f"return py_adapt::with_reason<{cf.ret}>(reason, reason_cap, {fail}, [&]() -> {cf.ret} {{ {inner} }});"
    elif ctys.get("b") == T.B and cf.ret != "void":
        out = f"return py_adapt::on_builder<{cf.ret}>(b, {fail}, [&]() -> {cf.ret} {{ {inner} }});"
    elif ctys.get("b") == T.B:
        out = f"py_adapt::on_builder<int>(b, 0, [&]() -> int {{ {inner} return 0; }});"
    else:
        out = inner
    return [head, "{", "    " + out, "}"]


def extern_consts(env: Env, mod: Module, header: str) -> list[str]:
    """Definitions of the header's extern constants from the Python values."""
    path = env.headers.paths[header]
    out = []
    text = open(path).read()
    for m in re.finditer(
        r"extern\s+(const\s+[^;\[]*?)\s*\b(ROCKE_\w+)\s*(\[[^\]]*\])?\s*;", text
    ):
        ty, name, dim = m.group(1).strip(), m.group(2), m.group(3) or ""
        pyname = name.replace(PKG.macro, "")
        val = getattr(mod.pymod, pyname, None)
        if val is None:
            continue
        if isinstance(val, tuple):
            items = ", ".join(lit_code(v) for v in val)
            out.append(f"{ty} {name}{dim} = {{{items}}};")
        else:
            out.append(f"{ty} {name} = {lit_code(val)};")
    return out


def creates_builder(fn: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "IRBuilder"
        for n in ast.walk(fn)
    )


def needs_seq_guards(env: Env, mod: Module, fn: ast.FunctionDef, seen=None) -> bool:
    """R-HS1: the function (or a module function it calls) calls a C helper taking the builder's seq_guards counter,
    or reads the per-builder counter itself (_SEQ_GUARDS)."""
    seen = seen or set()
    if any(isinstance(n, ast.Name) and n.id == "_SEQ_GUARDS" for n in ast.walk(fn)):
        return True
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
            continue
        name = n.func.id
        if name in mod.imports:
            src, attr = mod.imports[name]
            e = env.sig.helpers.get(f"{src.split('.')[-1]}:{attr}")
            cf = env.c_func(e["c"]["name"]) if e else None
            if cf is not None and any(nm == "seq_guards" for _, nm in cf.params):
                return True
        elif name in mod.functions and name not in seen:
            seen.add(name)
            if needs_seq_guards(
                env, mod, mod.functions[name], seen
            ) and not creates_builder(mod.functions[name]):
                return True
    return False


def needs_fence(mod: Module, fn: ast.FunctionDef, seen=None) -> bool:
    """R-HS2: the function (or a module function it calls) emits a barrier that reads b.fence_barriers."""
    seen = seen or set()
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in mod.fence_methods
            and not any(k.arg == "fence" for k in n.keywords)
        ):
            return True
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in mod.functions
            and n.func.id not in seen
        ):
            seen.add(n.func.id)
            if needs_fence(mod, mod.functions[n.func.id], seen):
                return True
    return False


def derived_free(env: Env, mod: Module, header: str) -> dict[str, list[str]]:
    """<struct>_free for structs that own (pointer, count, capacity) list fields."""
    out = {}
    for cls, sname in mod.class_struct.items():
        s = env.headers.structs[sname]
        owned = [f for f in s.fields if "n_" + f in s.fields and "cap_" + f in s.fields]
        if not owned:
            continue
        stem = sname[:-2] if sname.endswith("_t") else sname
        fname = stem + "_free"
        cf = env.headers.funcs.get(fname)
        if cf is None or cf.file != header:
            continue
        w = cf.params[0][1]
        body = [f"{cf.ret} {fname}({cf.params[0][0]} {w})", "{"]
        for f in owned:
            body += [
                f"    free({w}->{f});",
                f"    {w}->{f} = NULL;",
                f"    {w}->n_{f} = 0;",
                f"    {w}->cap_{f} = 0;",
            ]
        body.append("}")
        out[fname] = body
    return out


def derived_default(env: Env, mod: Module, header: str) -> dict[str, list[str]]:
    """<struct>_default(): the dataclass's defaults (None -> has_ flag false; a
    default_factory other than an empty list / tuple / dict only through the
    field struct's own C _default(), else the class gets no derived default)."""
    import dataclasses

    from .family import stem as own_stem

    out = {}
    for cls, sname in mod.class_struct.items():
        fam = mod.family(cls)
        stem = (
            own_stem(cls)
            if fam is not None and cls != fam.base
            else (sname[:-2] if sname.endswith("_t") else sname)
        )
        cf = env.headers.funcs.get(stem + "_default")
        pycls = getattr(mod.pymod, cls, None)
        if (
            cf is None
            or cf.file != header
            or cf.params
            or pycls is None
            or not dataclasses.is_dataclass(pycls)
        ):
            continue
        s_ = env.headers.structs[sname]
        body = [
            f"{cf.ret} {cf.name}(void)",
            "{",
            f"    {sname} s;",
            "    memset(&s, 0, sizeof(s));",
        ]
        if fam is not None and cls != fam.base:
            body.append(
                f"    s.kind = {kind_const(fam, cls)};"
            )  # tagged values: the member's tag
        ok = True
        for f_ in dataclasses.fields(pycls):
            if f_.name not in s_.fields and f_.name.lstrip("_") not in s_.fields:
                continue
            fac = f_.default_factory
            if f_.default is dataclasses.MISSING and fac is not dataclasses.MISSING:
                if fac in (list, tuple, dict):
                    continue  # empty: memset left (NULL, 0, 0)
                # a factory (TraitSpec): its struct's own C default, else no
                # derivation (zeros are not the factory's value)
                ft = s_.fields.get(f_.name, "")
                dcf = (
                    env.headers.funcs.get(ft[:-2] + "_default")
                    if ft.endswith("_t")
                    else None
                )
                if dcf is None or dcf.params or dcf.ret != ft:
                    ok = False
                    break
                body.append(f"    s.{f_.name} = {dcf.name}();")
                continue
            if f_.default is dataclasses.MISSING:
                continue
            if f_.default is None:
                if "has_" + f_.name in s_.fields:
                    body.append(f"    s.has_{f_.name} = false;")
                continue
            d = f_.default
            if isinstance(d, (tuple, list)):
                if not d:
                    continue  # empty list: memset left (NULL, 0, 0)
                items = ", ".join(lit_code(x) for x in d)
                if "n_" + f_.name in s_.fields:  # a list the struct owns
                    el = s_.fields[f_.name][:-1]
                    body.append(
                        f"    s.{f_.name} = py::own_copy<{el}>(std::vector<{el}>{{{items}}});"
                    )
                    body.append(f"    s.n_{f_.name} = {len(d)};")
                    if "cap_" + f_.name in s_.fields:
                        body.append(f"    s.cap_{f_.name} = {len(d)};")
                else:  # a fixed tuple struct
                    body.append(f"    s.{f_.name} = {{{items}}};")
                continue
            body.append(f"    s.{f_.name} = {lit_code(d)};")
        if not ok:
            continue
        body += ["    return s;", "}"]
        out[cf.name] = body
    return out


def load_fill(modname: str, fname: str) -> list[str] | None:
    p = os.path.join(holes_dir(), modname, fname + ".cpp")
    if os.path.exists(p):
        return open(p).read().rstrip("\n").splitlines()
    return None


def translate(modname: str, rocke: str, abi_dir: str | None = None) -> Result:
    env = Env(rocke, overlay=abi_dir)
    path = module_path(rocke, modname)
    mod = Module(env, modname, path)
    res = Result()
    header = own_header(env, modname)
    tm = json.load(open(os.path.join(maps_dir(), "twin_map.json")))["symbols"]
    mod.bind_classes(tm)
    from .kernel import TypeMap, private_structs

    mod.tmap = TypeMap(env, mod)
    if abi_dir:
        # other generated modules first: calls into them use their generated entries
        for other in generated_modules(abi_dir):
            if other != modname:
                try:
                    om = Module(env, other, module_path(rocke, other))
                    om.bind_classes(tm)
                    om.tmap = TypeMap(env, om)
                    install_abi(env, om)
                except (
                    Exception
                ) as exc:  # a module that does not derive keeps its mined entries
                    print(
                        f"generated ABI of {other} not installed: {exc}",
                        file=sys.stderr,
                    )
        own = install_abi(env, mod)
        undeclared = own.undeclared
        res.abi_refused = abi_refusals(own)
    else:
        undeclared = {}
    struct_lines = private_structs(env, mod, mod.facts, mod.tmap)
    mod.private_classes = {
        c for c, sname in mod.class_struct.items() if sname.startswith("py_")
    }
    # a function whose declaration tr.abi refused stays a unit (reported as a
    # hole below), so what it calls is still listed
    public = [
        n for n in mod.functions if env.sig.helpers.get(mod.qual(n)) or n in undeclared
    ]
    order = [(n, None) for n in topo(mod, public)]
    # class methods: dataclass-only classes have none; methods come after the functions they call
    # methods: exported ones, plus private-class methods something translated calls
    exported_m = {
        (c, m)
        for (c, m) in mod.methods
        if env.sig.helpers.get(mod.qual(f"{c}.{m}")) or f"{c}.{m}" in undeclared
    }
    called = set()
    frontier = [mod.functions[n] for n, _ in order] + [
        mod.methods[k] for k in exported_m
    ]
    seen_m = set(exported_m)
    while frontier:
        f_ = frontier.pop()
        for n in ast.walk(f_):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                called.add(n.func.attr)
            if isinstance(n, ast.Attribute):
                called.add(n.attr)  # properties
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in mod.classes
            ):
                called.add("__init__")
        for c, m in mod.methods:
            if (c, m) not in seen_m and (
                m in called
                or (c in mod.private_classes and m in ("__init__", "__post_init__"))
            ):
                seen_m.add((c, m))
                frontier.append(mod.methods[(c, m)])
    method_units = [((f"{c}.{m}"), c) for (c, m) in mod.methods if (c, m) in seen_m]
    # module functions only methods call (helpers of a class) are needed too
    from_methods = set()
    for k in seen_m:
        from_methods |= calls_in(mod.methods[k], set(mod.functions))
    have = {n for n, _ in order}
    extra = [n for n in topo(mod, sorted(from_methods)) if n not in have]
    order = order + [(n, None) for n in extra]
    order = method_units + order if method_units else order
    internals, shims = [], []
    emitted_c = set()
    attempted = set()

    def attempt_clone(name, cls, host):
        """host clones: the host clone of one unit on host builder class `host`
        (designs/host_clones.md): internal only, no C entry."""
        from .host import clone_name, closure_fields, host_type
        from .lower import builder_param, closure_key

        if cls == "<closure>":
            site = next(
                x
                for ss in closure_fields(mod).values()
                for x in ss
                if closure_key(x) == name
            )
            fn = site.node
        else:
            fn = (
                mod.methods[(cls, name.split(".", 1)[1])]
                if cls
                else mod.functions[name]
            )
        rec = {"name": name, "host": host, "line": fn.lineno}
        key = f"{name}@{host}" if host else name
        try:
            ir = (
                lifted_sig(mod, site)
                if cls == "<closure>"
                else internal_sig(env, mod, fn, None, None, cls=cls, trust_c=False)
            )
            if host == IR_KIND:
                # a function on ints whose IR-kind callers pass IR values: its IR
                # version, typed from the recorded IR-value alternatives
                fact = mod.fact_of(fn) or {"params": {}, "return": []}
                params = []
                for pn, t in ir.params:
                    tg = fact["params"].get(pn)
                    t2 = (
                        mod.tmap.tags(kind_tags(tg, IR_KIND))
                        if tg and kind_tags(tg, IR_KIND) != tg
                        else t
                    )
                    if t2 is None:
                        raise Refuse(fn, f"IR version: no C++ type for {pn}")
                    params.append((pn, t2))
                rs = [r for r in fact["return"] if r != "None"]
                ret = (
                    mod.tmap.tags(kind_tags(rs, IR_KIND), param=False)
                    if rs and kind_tags(rs, IR_KIND) != rs
                    else ir.ret
                )
                if ret is None or params == ir.params:
                    raise Refuse(fn, "IR version: the recordings show no IR values")
                sig = Sig(clone_name(ir.name, host), params, ret, fn, ir.defaults)
                mod.sigs[key] = sig
                body = lower_function(mod, fn, sig)
                rec["status"] = "translated"
                return rec, body, None
            if host is None:
                mod.sigs[key] = ir
                body = lower_function(mod, fn, ir)
                rec["status"] = "translated"
                return rec, body, None
            hst = host_builders(mod)[host]
            images = {
                ir_: img for (ir_, h_), img in mod.host_images.items() if h_ == host
            }
            bp = builder_param(fn)
            params = []
            for pn, t in ir.params:
                if pn == bp:
                    params.append((pn, f"const {hst}*"))
                elif t == T.B:
                    raise Refuse(
                        fn, "a host clone of a function that needs the IR builder"
                    )
                else:
                    params.append((pn, host_type(t, images)))
            sig = Sig(
                clone_name(ir.name, host),
                params,
                host_type(ir.ret, images),
                fn,
                ir.defaults,
                host=host,
            )
            mod.sigs[key] = sig
            body = lower_function(mod, fn, sig)
            rec["status"] = "translated"
            return rec, body, None
        except Exception as exc:
            if not isinstance(exc, Refuse) and os.environ.get("TR_DEBUG"):
                import traceback

                traceback.print_exc()
            r = (
                exc
                if isinstance(exc, Refuse)
                else Refuse(fn, f"internal error: {type(exc).__name__}: {exc}")
            )
            mod.sigs.pop(key, None)
            return rec, None, r

    def attempt(name, cls, host=None):
        """One unit: (record, internal body, shim, C names) or a Refuse."""
        if host is not None or cls == "<closure>":
            rec, body, r = attempt_clone(name, cls, host)
            return rec, body, [], set(), r
        fn = mod.methods[(cls, name.split(".", 1)[1])] if cls else mod.functions[name]
        entry, cf = abi_of(env, mod, name)
        rec = {
            "name": name,
            "line": fn.lineno,
            "public": cf is not None,
            "c": cf.name if cf else None,
        }
        if cf is not None:
            attempted.add(cf.name)
        names = set()
        try:
            fence_var = (
                env.headers.funcs.get(cf.name + "_fence") if cf is not None else None
            )
            builds = creates_builder(fn)
            gen = entry is not None and entry["c"].get("generated")
            fence = (
                (bool(fence_var) or cf is None or gen)
                and needs_fence(mod, fn)
                and not builds
            )
            sg = (
                (cf is None or bool(gen))
                and not builds
                and needs_seq_guards(env, mod, fn)
            )
            trust = not abi_dir or bool(gen)
            sig = internal_sig(
                env,
                mod,
                fn,
                entry,
                cf if trust else None,
                cls=cls,
                fence=fence,
                seq_guards=sg,
                trust_c=trust,
            )
            mod.sigs[name] = sig
            body = lower_function(mod, fn, sig)
            is_init = cls is not None and fn.name == "__init__"
            try:
                if cf is not None and entry["c"].get("generated"):
                    shim = make_gen_shim(env, mod, fn, sig, entry, cf, is_init=is_init)
                else:
                    shim = (
                        make_shim(env, mod, fn, sig, entry, cf, is_init=is_init)
                        if cf is not None
                        else []
                    )
            except Refuse as sr:
                # the function translated; only its C entry does not fit the kept
                # hand declaration: an ABI hole (a hole of the report: the source
                # does not compile), the function stays usable inside
                shim = [f'#error "abi hole: {cf.name}: {sr.reason}"']
                rec["shim_hole"] = sr.reason
                rec["shim_at"] = sr.line or fn.lineno
            if fence and fence_var is not None:
                import copy as _copy

                fcf = _copy.copy(fence_var)
                shim += [""] + make_shim(
                    env, mod, fn, sig, entry, fcf, fence_arg=fence_var.params[-1][1]
                )
                names.add(fence_var.name)
            if cf is not None:
                names.add(cf.name)
            rec["status"] = "translated"
            return rec, body, shim, names, None
        except Exception as exc:
            if not isinstance(exc, Refuse) and os.environ.get("TR_DEBUG"):
                import traceback

                traceback.print_exc()
            r = (
                exc
                if isinstance(exc, Refuse)
                else Refuse(fn, f"internal error: {type(exc).__name__}: {exc}")
            )
            mod.sigs.pop(name, None)
            return rec, None, None, names, r

    # Units in order; a unit that failed only because a callee was not
    # translated yet (a method calling a later function) is retried.
    done = {}
    units = [(n, c, None) for n, c in order]
    pending = list(enumerate(units))
    asked = set()
    while True:
        while pending:
            retry = []
            for k, (name, cls, host) in pending:
                got = attempt(name, cls, host)
                r = got[4]
                if r is not None and "not translated" in r.reason:
                    retry.append((k, (name, cls, host), got))
                else:
                    done[k] = got
            if len(retry) == len(pending) and not (mod.host_demand - asked):
                for k, _u, got in retry:
                    done[k] = got
                break
            pending = [(k, u) for k, u, _g in retry]
            if mod.host_demand - asked:
                break
        # host clones: host clones callers asked for become units; the callers are retried
        new = sorted(
            mod.host_demand - asked, key=lambda d: (d[0], d[1] or "", d[2] or "")
        )
        if not new:
            break
        asked |= set(new)
        for n, c, h in new:
            units.append((f"{c}.{n}" if c and c != "<closure>" else n, c, h))
        waiting = {k for k, _u in pending}
        pending = [
            (k, u)
            for k, u in enumerate(units)
            if k in waiting
            or k >= len(units) - len(new)
            or (
                k in done
                and done[k][4] is not None
                and "not translated" in done[k][4].reason
            )
        ]
        for k, _u in pending:
            done.pop(k, None)

    def family_entry(rec, shim):
        """tagged values: the exported entry of a family method dispatches on the kind tag
        (designs/tagged_values.md): members of this module through the
        internal dispatcher, duck-typed peers through their hand C entries."""
        name = rec["name"]
        if (
            "." not in name
            or not shim
            or rec.get("c") is None
            or shim[0].startswith("#error")
        ):
            return shim
        cls, meth = name.split(".", 1)
        fam = mod.family(cls)
        if fam is None or meth in ("__init__", "__post_init__"):
            return shim  # a constructor is called on its own class
        impls = fam.impls(cls, meth)
        inner = {m: i for m, i in impls.items() if not fam.is_peer(m)}
        out = list(shim)
        if len(set(inner.values())) > 1:
            dsig = mod.dispatcher(fam, cls, meth)
            own = mod.sigs[name].name
            out = [re.sub(rf"\b{own}\(", dsig.name + "(", x) for x in out]
        peers = [m for m, i in impls.items() if fam.is_peer(m) and i == m]
        if peers:
            from .family import stem as own_stem

            cf = env.c_func(rec["c"])
            ps = ", ".join(f"{t} {n}" for t, n in cf.params)
            args = ", ".join(n for _, n in cf.params)
            sv = cf.params[0][1]
            ext, pre = [], []
            for m in peers:
                pc = f"{own_stem(m)}_{meth}"
                ext.append(
                    f"/* {m} ({fam.peers[m]}) is a duck-typed peer of {fam.base}; its hand C entry */"
                )
                ext.append(f'extern "C" {cf.ret} {pc}({ps});')
                ret = (
                    f"{{ {pc}({args}); return; }}"
                    if cf.ret == "void"
                    else f"return {pc}({args});"
                )
                pre.append(f"    if({sv}->kind == {kind_const(fam, m)})")
                pre.append(f"        {ret}")
            i = out.index("{")
            out = ext + [""] + out[: i + 1] + pre + out[i + 1 :]
        return out

    for k in sorted(done):
        rec, body, shim, names, r = done[k]
        name = rec["name"]
        if units[k][2] is not None or units[k][1] == "<closure>":
            # a host clone or a lifted closure: an internal function, not a unit of the report
            if r is None:
                internals.append(body)
            else:
                rec.update(status="hole", reason=r.reason, at=r.line)
            res.clones.append(rec)
            continue
        if r is None:
            try:
                shim = family_entry(rec, shim)
            except Refuse as e:
                # the method translated (its internal stays, others call it);
                # its exported entry cannot dispatch yet: a hole there
                internals.append(body)
                shims.append(
                    [
                        f'#error "hole: {name}: {e.reason} (exported dispatch, tagged values)"'
                    ]
                )
                rec.update(
                    status="hole",
                    reason=f"{e.reason} (exported dispatch, tagged values)",
                    at=rec.get("line", 0),
                )
                res.functions.append(rec)
                continue
        forced = load_fill(modname, name.replace(".", "_")) if r is None else None
        if forced is not None and not any(
            "hole fill (dispatch" in x for x in forced[:3]
        ):
            forced = None  # an ordinary fill is for a refused function only
        if r is None and forced is not None:
            # a dispatch fill: the function translates (its internal stays), but
            # its C entry first dispatches on a C-only field the hand callers set
            # (holes/<module>/<Class>.fields); see designs/generated_abi.md
            internals.append(body)
            shims.append(forced)
            emitted_c.update(names)
            rec["status"] = "filled"
            rec["reason"] = "dispatch fill: C-only field of the hand callers"
            rec["at"] = rec.get("line", 0)
            rec["fill_lines"] = len([x for x in forced if x.strip()])
        elif r is None and rec.get("shim_hole"):
            # an ABI hole: the internal stays; a hand fill of the function
            # replaces its C entry, else the entry is a hole of the report
            internals.append(body)
            fill = load_fill(modname, name.replace(".", "_"))
            reason = f"abi hole: {rec['shim_hole']}"
            rec.update(reason=reason, at=rec["shim_at"])
            if fill is not None:
                shims.append(fill)
                emitted_c.update(names)
                rec["status"] = "filled"
                rec["fill_lines"] = len([x for x in fill if x.strip()])
            else:
                shims.append(shim)
                emitted_c.update(names)
                rec["status"] = "hole"
        elif r is None:
            internals.append(body)
            shims.append(shim)
            emitted_c.update(names)
        else:
            fill = load_fill(modname, name.replace(".", "_"))
            rec["reason"] = r.reason
            rec["at"] = r.line
            if fill is not None:
                shims.append(fill)
                emitted_c.update(
                    re.findall(r"^\S.*?\b(rocke_\w+)\(", "\n".join(fill), re.M)
                )
                rec["status"] = "filled"
                rec["fill_lines"] = len([x for x in fill if x.strip()])
            else:
                shims.append([f'#error "hole: {name}: {r.reason} (line {r.line})"'])
                rec["status"] = "hole"
        res.functions.append(rec)
    # exported functions whose declaration tr.abi refused (it uses a struct that
    # is defined nowhere): holes, whatever their body did
    seen_units = {f["name"]: f for f in res.functions}
    for qual, why in undeclared.items():
        rec = seen_units.get(qual)
        if rec is None:
            cls, _, meth = qual.rpartition(".")
            fn = mod.methods.get((cls, meth)) if cls else mod.functions.get(qual)
            rec = {"name": qual, "line": fn.lineno if fn else 0, "public": True}
            res.functions.append(rec)
        elif rec["status"] in ("hole", "filled"):
            continue  # already a hole (or filled) for its own reason
        rec.update(status="hole", reason=f"not exported: {why}", at=rec.get("line", 0))
        shims.append([f'#error "hole: {qual}: not exported: {why}"'])
    # header functions with no Python twin: derived (_free) or holes (count companions)
    if header:
        for fname, body in list(derived_free(env, mod, header).items()) + list(
            derived_default(env, mod, header).items()
        ):
            shims.append(body)
            emitted_c.add(fname)
            res.functions.append(
                {
                    "name": fname,
                    "line": 0,
                    "public": True,
                    "c": fname,
                    "status": "derived",
                }
            )
        for fname, f in sorted(env.headers.funcs.items()):
            if (
                f.file != header
                or fname in emitted_c
                or fname in attempted
                or f.name.startswith("ROCKE")
            ):
                continue
            if (
                "static inline"
                in open(env.headers.paths[header]).read().split(fname)[0][-200:]
            ):
                continue
            fill = load_fill(modname, fname)
            rec = {
                "name": fname,
                "line": 0,
                "public": True,
                "c": fname,
                "reason": "C-only function with no Python twin (ABI companion)",
                "at": 0,
            }
            if fill is not None:
                shims.append(fill)
                rec["status"] = "filled"
                rec["fill_lines"] = len([x for x in fill if x.strip()])
            else:
                shims.append(
                    [f'#error "hole: {fname}: C-only function with no Python twin"']
                )
                rec["status"] = "hole"
            res.functions.append(rec)
    # host images (platform structs built on a host builder) the emitted code uses
    joined_ = "\n".join("\n".join(x) for x in internals + shims)
    for img, dlines in mod.host_image_defs.items():
        if re.search(rf"\b{img}\b", joined_):
            struct_lines += dlines
    # tagged-value dispatchers the emitted code calls (a dry pass may have made others)
    used_text = "\n".join("\n".join(x) for x in internals + shims)
    for dname, dlines in mod.dispatchers.items():
        if re.search(rf"\b{dname}\(", used_text):
            internals.append(dlines)
    # an internal nothing calls (its exported entry was not written) would fail
    # -Wunused-function under -Werror: internals are emitted only when used
    internals = used_internals(internals, struct_lines + [x for s in shims for x in s])
    sha = hashlib.sha256(mod.src.encode()).hexdigest()[:12]
    rel = "/".join(
        (PKG.py_root, os.path.relpath(path, library_dir(rocke)).replace(os.sep, "/"))
    )
    lines = [
        "// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier: MIT",
        f"// first draft from {rel}:{sha} by tr/translate.py; hand-maintained from here.",
        "",
    ]
    if header:
        lines.append(f'#include "rocke/{header}"')
        lines.append("")
    lines += [
        "#include <math.h>",
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
        "#include <algorithm>",
        "#include <array>",
        "#include <string>",
        "#include <vector>",
        "",
        '#include "rocke/error.hpp"',
        '#include "rocke/ir_internal.h"',
        '#include "rocke_py.hpp"',
    ]
    joined = "\n".join("\n".join(x) for x in internals + shims)
    if "py_adapt::" in joined:
        lines.append('#include "rocke_py_adapters.hpp"')
        # the package's own adapters (they need its headers), only when used
        own = PKG.adapters_header()
        if own is not None and any(
            f"py_adapt::{n}(" in joined or f"py_adapt::{n}<" in joined
            for n in PKG.adapter_names()
        ):
            lines.append(f'#include "{own}"')
    # rocKE headers the code needs, each once: the header of every function it
    # calls and of every type of another header it names (a twin struct held by
    # value in a private struct, rocke_gemm_tile_spec_t); a function several
    # headers declare (rocke_kernel_name_join) needs only one of them. A type
    # several headers define (rocke_tensor_descriptor_t: tensor_view.h and
    # transforms.h, two different structs) comes from the header of the
    # functions the code calls with it; two different definitions of one name
    # are never included together (res.include_conflicts instead).
    skip = ("ir.h", "ir_internal.h", "error.hpp", header)
    type_text = "\n".join(struct_lines) + "\n" + joined
    incs = set()
    several = []
    for fname in dict.fromkeys(re.findall(r"\b(rocke_\w+)\(", joined)):
        f = env.headers.funcs.get(fname)
        if f is None:
            continue
        files = env.headers.func_files.get(fname) or [f.file]
        if len(files) > 1:
            several.append(files)
        elif f.file not in skip:
            incs.add(f.file)
    shared = []
    for t in sorted(set(re.findall(r"\b(rocke_\w+_t)\b", type_text))):
        files = env.headers.typedef_files.get(t) or []
        if len(files) > 1:
            shared.append((t, files))
            continue
        f = env.headers.typedef_file.get(t)
        if f is not None and f not in skip:
            if header is None or not _includes(env, header, f):
                incs.add(f)
    for t, files in shared:
        have = _closure(env, [x for x in skip if x] + sorted(incs))
        bodies = env.headers.typedef_bodies.get(t, {})
        if files[0] in have:
            if files[0] not in skip and (
                header is None or not _includes(env, header, files[0])
            ):
                incs.add(files[0])
            continue
        if any(x in have and (x in bodies or not bodies) for x in files):
            continue  # a header the code needs already brings its definition
        ok = [x for x in files if not _adds_conflict(env, have, x)]
        ok.sort(key=lambda x: x not in bodies)  # a definition over a declaration
        incs.add((ok or files)[0])  # none fits: the conflict is reported below
    for files in several:
        if not any(x in incs or x in skip for x in files):
            have = _closure(env, [x for x in skip if x] + sorted(incs))
            ok = [x for x in files if not _adds_conflict(env, have, x)]
            incs.add((ok or files)[0])
    res.include_conflicts = _conflicts(
        env, _closure(env, [x for x in skip if x] + sorted(incs))
    )
    for name, why in res.include_conflicts.items():
        lines.append(f'#error "include conflict: {name}: {why}"')
    for inc in sorted(incs):
        if f'#include "rocke/{inc}"' not in lines:
            lines.append(f'#include "rocke/{inc}"')
    lines.append("")
    if header:
        lines += extern_consts(env, mod, header)
        lines.append("")
    if mod.records:
        # R-REC: one struct per record dict, a field and a has_ flag per key
        for rid, fields in mod.records.items():
            from .lower import record_field

            lines.append(f"struct py_rec_{rid}_t")
            lines.append("{")
            for k, t in fields.items():
                f = record_field(k)
                lines.append(f"    bool has_{f} = false;")
                lines.append(
                    f"    {decl_type(t) if t not in (T.AUTO, T.NONE) else 'int64_t'} {f}{{}};"
                )
            lines.append("};")
            lines.append("")
    if mod.private_consts:
        lines += [d for _, d in mod.private_consts.values()]
        lines.append("")
    lines.append("namespace")
    lines.append("{")
    lines.append("")
    lines += struct_lines
    # prototypes: a retried unit can be defined after a unit that calls it
    heads = []
    for b in internals:
        head = []
        for ln in b:
            if ln.startswith("/*") and not head:
                continue
            if ln.strip() == "{":
                break
            head.append(ln)
        m_ = re.search(r"\b(py_\w+)\(", " ".join(head))
        heads.append(
            (m_.group(1) if m_ else None, " ".join(x.strip() for x in head) + ";")
        )
    # only the ones used before their definition
    later = {n: i for i, (n, _h) in enumerate(heads) if n}
    need = set()
    for i, b in enumerate(internals):
        for n in set(re.findall(r"\b(py_\w+)\(", "\n".join(b))):
            if later.get(n, -1) > i:
                need.add(n)
    protos = [h for n, h in heads if n in need]
    if protos:
        lines += protos + [""]
    for b in internals:
        lines += b
        lines.append("")
    lines.append("} // namespace")
    lines.append("")
    for s in shims:
        if s:
            lines += s
            lines.append("")
    res.code = "\n".join(lines)
    return res


def _internal_name(body: list[str]) -> str | None:
    """The py_* function an internal body defines (its head, before the "{")."""
    head = []
    for ln in body:
        if ln.startswith("/*") and not head:
            continue
        if ln.strip() == "{":
            break
        head.append(ln)
    m = re.search(r"\b(py_\w+)\(", " ".join(head))
    return m.group(1) if m else None


def used_internals(internals: list[list[str]], outside: list[str]) -> list[list[str]]:
    """The internal functions (anonymous namespace) something uses: an exported
    entry, a struct, or another used internal. One with no caller is dropped (a
    static function nothing calls fails -Wunused-function under -Werror), until
    nothing changes. A body that defines no py_* function is kept."""
    keep = list(internals)
    names = [_internal_name(b) for b in keep]
    while True:
        drop = set()
        for n in {n for n in names if n}:
            pat = re.compile(rf"\b{re.escape(n)}\b")
            used = any(pat.search(x) for x in outside) or any(
                pat.search(x) for b, bn in zip(keep, names) if bn != n for x in b
            )
            if not used:
                drop.add(n)
        if not drop:
            return keep
        pairs = [(b, n) for b, n in zip(keep, names) if n not in drop]
        keep, names = [b for b, _ in pairs], [n for _, n in pairs]


def _closure(env: Env, headers: list) -> set:
    """The rocKE headers a source including each of headers sees: those and
    every rocke/ header they include, directly or not."""
    out = set()
    todo = [h for h in headers if h]
    while todo:
        h = todo.pop()
        if h in out or h not in env.headers.paths:
            continue
        out.add(h)
        text = open(env.headers.paths[h]).read()
        todo += re.findall(r'^\s*#\s*include\s+"rocke/([^"]+)"', text, re.M)
    return out


def _norm_c(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _conflicts(env: Env, files: set) -> dict[str, str]:
    """Typedef names that two of the headers files define with different
    bodies (a source that sees both cannot compile): name -> why."""
    out = {}
    for name, bodies in sorted(env.headers.typedef_bodies.items()):
        seen = {f: _norm_c(b) for f, b in bodies.items() if f in files}
        if len(set(seen.values())) > 1:
            out[name] = (
                f"defined differently in {' and '.join(sorted(seen))}, and the"
                " source needs both: include one"
            )
    return out


def _adds_conflict(env: Env, have: set, inc: str) -> bool:
    """Including inc next to the headers have would define a name twice."""
    return bool(
        set(_conflicts(env, have | _closure(env, [inc]))) - set(_conflicts(env, have))
    )


def _includes(env: Env, header: str, other: str, seen=None) -> bool:
    """header includes rocke/<other>, directly or through the headers it includes."""
    seen = set() if seen is None else seen
    if header in seen or header not in env.headers.paths:
        return False
    seen.add(header)
    text = open(env.headers.paths[header]).read()
    for inc in re.findall(r'^\s*#\s*include\s+"rocke/([^"]+)"', text, re.M):
        if inc == other or _includes(env, inc, other, seen):
            return True
    return False


def set_clang_format(exe: str | None) -> None:
    global CLANG_FORMAT
    if exe:
        CLANG_FORMAT = exe


def find_clang_format() -> str:
    """--clang-format, else $CLANG_FORMAT, else the clang-format installed next to
    this Python (the pip wheel in the translator's venv: a pinned version, so the
    output does not move), else clang-format on PATH."""
    exe = CLANG_FORMAT or os.environ.get("CLANG_FORMAT")
    if exe:
        if shutil.which(exe) is None:
            sys.exit(
                f"error: clang-format {exe} (from --clang-format or CLANG_FORMAT) is not an executable"
            )
        return exe
    beside = os.path.join(os.path.dirname(sys.executable), "clang-format")
    if os.access(beside, os.X_OK):
        return beside
    exe = shutil.which("clang-format")
    if exe is None:
        sys.exit(
            "error: no clang-format: pass --clang-format <exe>, set CLANG_FORMAT, or put clang-format on PATH"
        )
    return exe


def clang_format(code: str, rocke: str) -> str:
    style_file = os.path.join(rocke, "platform", "cpp", "instances", "common", "x.cpp")
    p = subprocess.run(
        [find_clang_format(), "-style=file", f"-assume-filename={style_file}"],
        input=code,
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return p.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module")
    ap.add_argument("--rocke", help=ROCKE_HELP)
    ap.add_argument("--out")
    ap.add_argument("--report", help="write the per-function JSON report here")
    ap.add_argument(
        "--analyze", action="store_true", help="only list holes and refusals"
    )
    ap.add_argument("--no-format", action="store_true")
    ap.add_argument(
        "--clang-format",
        metavar="EXE",
        help="clang-format to run (default: $CLANG_FORMAT, the venv's, or PATH)",
    )
    ap.add_argument(
        "--abi",
        metavar="DIR",
        help="generated headers (tr.abi --header DIR): use the generated ABI",
    )
    a = ap.parse_args(argv)
    a.rocke = rocke_root(a.rocke)
    set_clang_format(a.clang_format)
    res = translate(a.module, a.rocke, a.abi)
    for f in res.functions:
        print(report_line(f))
    for c in res.clones:
        extra = f" ({c['reason']} at line {c['at']})" if c["status"] == "hole" else ""
        print(f"clone-{c['status']:4} {c['name']}@{c['host']}{extra}")
    for name, why in res.abi_refused.items():
        print(refusal_line(name, why))
    for name, why in res.include_conflicts.items():
        print(conflict_line(name, why))
    print(summary(res.functions, res.abi_refused))
    if a.report:
        json.dump(res.functions, open(a.report, "w"), indent=1)
    if a.analyze:
        return 0
    code = res.code if a.no_format else clang_format(res.code, a.rocke)
    if a.out:
        open(a.out, "w").write(code)
    else:
        sys.stdout.write(code)
    left = any(f["status"] == "hole" for f in res.functions)
    return 1 if left or res.include_conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
