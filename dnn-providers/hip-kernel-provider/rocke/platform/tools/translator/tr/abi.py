"""Generated C ABI: a module's own exports, derived from the Python alone.

The rules are the table in designs/generated_abi.md. Output is a list of
declarations in the sigmap entry format (so the translator's shim code
consumes them unchanged) plus the header text.

    python -m tr.abi <module> --diff     # each declaration vs the hand header
    python -m tr.abi <module> --header   # print the generated header
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field

from . import ctypes as T
from .env import ROCKE_HELP, Env, data_dir, holes_dir, maps_dir, rocke_root
from .family import family_of, kind_const, kind_type, stem
from .host import (
    closure_fields,
    closure_kind_const,
    closure_kind_type,
    site_field,
    site_types,
)
from .lower import Module, Refuse, py_params
from .package import PKG

PREFIX = PKG.prefix  # the package setting (tr/package.py)

# Python type, parsed from an annotation or a recorded tag:
#   ("b",) ("v",) ("int",) ("bool",) ("float",) ("str",) ("iv",) ("type",)
#   ("mma",) ("kernel",) ("none",) ("opt", X) ("seq", X) ("tuple", [X..])
#   ("dict", K, V) ("fn", [A..], R) ("cls", name) ("obj", name)
SCALARS = {
    "IRBuilder": ("b",),
    "Value": ("v",),
    "int": ("int",),
    "bool": ("bool",),
    "float": ("float",),
    "str": ("str",),
    "None": ("none",),
    "IntOrValue": ("iv",),
    "Type": ("type",),
    "VectorType": ("type",),
    "PtrType": ("type",),
    "MmaOp": ("mma",),
    "KernelDef": ("kernel",),
}


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _split(s: str, seps=",") -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch in seps and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


class Abi:
    def __init__(self, env: Env, mod: Module):
        self.env = env
        self.mod = mod
        self.typedefs: list[str] = []  # generated {fn, ctx} and protocol structs
        self.tuple_structs: dict[str, str] = {}  # fixed tuples, named by item types
        # ABI holes: refused items kept from the existing hand header (C name -> text)
        self.hand: dict[str, str] = {}
        self.hand_quals: set[str] = set()
        self.struct_override: dict[str, str] = {}
        self.support_names: set[str] = set()
        self.structs: dict[str, list[str]] = {}  # python class -> C struct text
        self.decls: list[Decl] = []
        self.consts: list[str] = []
        self.refused: list[tuple[str, str]] = []
        self.flat: set[str] = set()
        self.family_owns: dict[str, bool] = {}
        self._building: set[str] = (
            set()
        )  # family bases whose tagged struct does not derive (old per-class rule)
        self.closure_enums: dict[str, list[str]] = (
            {}
        )  # host clones: kind enums of closure fields, by class

    def fam(self, cls: str):
        """cls's tagged-value family (designs/tagged_values.md), unless its tagged struct
        does not derive: then every class keeps its own struct as before."""
        f = family_of(self.mod, cls)
        if f is None or f.base in self.flat:
            return None
        if f.module != self.mod.modname:
            return f if sibling_family_ok(self.env, self.mod, f) else None
        if (
            f.base not in self.structs
            and f.base not in self.struct_override
            and f.base not in self._building
        ):
            self._building.add(f.base)
            try:
                self.family_struct(f)
            except Refuse as e:
                self.flat.add(f.base)
                self.refused.append(
                    (
                        f.base,
                        f"family struct (tagged values): {e.reason if hasattr(e, 'reason') else e}",
                    )
                )
                return None
            finally:
                self._building.discard(f.base)
        return f

    # ------------------------------------------------------------ names

    def struct_name(self, cls: str) -> str:
        fam = self.fam(cls)
        if fam is not None and cls != fam.base:
            return self.struct_name(
                fam.base
            )  # tagged values: one tagged struct per family
        if cls in self.struct_override:
            return self.struct_override[cls]
        own = cls in self.mod.classes or self.sibling(cls)
        if not own and self.mod.tmap is not None and cls in self.mod.tmap.class_struct:
            return self.mod.tmap.class_struct[cls]
        if not own and f"rocke_{snake(cls)}_t" in self.env.headers.typedef_file:
            return f"rocke_{snake(cls)}_t"  # a platform class with a C struct
        base = cls[len(PKG.class_prefix) :] if cls.startswith(PKG.class_prefix) else cls
        return f"rocke_{PREFIX}_{snake(base)}_t"

    def sibling(self, cls: str) -> bool:
        """Imported from another backward module: its struct is generated by the same rule."""
        src = self.mod.imports.get(cls, ("", ""))[0].split(".")[-1]
        return src.startswith(PKG.modules)

    def known_class(self, t: str) -> bool:
        f = family_of(self.mod, t)
        if (
            f is not None
            and t != f.base
            and self.known_class(f.base)
            and self.fam(t) is not None
        ):
            return True  # tagged values: a member of a known family
        return (
            t in self.mod.classes
            or self.sibling(t)
            or (self.mod.tmap is not None and t in self.mod.tmap.class_struct)
            or f"rocke_{snake(t)}_t" in self.env.headers.typedef_file
        )

    def imported_privates(self) -> list[str]:
        """Private module-level functions a sibling module imports by name
        (`from .<module> import _min64`): Python's real export surface, so
        they are exported like __all__ (the hand header did too)."""
        d = os.path.dirname(self.mod.path)
        pat = re.compile(
            rf"from \.{re.escape(self.mod.modname)} import (?:\(([^)]*)\)|([^\n]*))"
        )
        out = []
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py") or f[:-3] == self.mod.modname:
                continue
            for m in pat.finditer(open(os.path.join(d, f)).read()):
                for n in re.findall(r"\b(_[A-Za-z]\w*)\b", m.group(1) or m.group(2)):
                    if n in self.mod.functions and n not in out:
                        out.append(n)
        return out

    def func_name(self, name: str) -> str:
        name = name.lstrip("_")
        if name.startswith("build_"):
            return f"rocke_{name}_new"
        n = name[len(PREFIX) + 1 :] if name.startswith(PREFIX + "_") else name
        return f"rocke_{PREFIX}_{n}"

    def method_name(self, cls: str, meth: str) -> str:
        fam = self.fam(cls)
        base = (
            stem(cls)
            if fam is not None and cls != fam.base
            else self.struct_name(cls)[: -len("_t")]
        )
        m = {"__init__": "init", "__post_init__": "validate"}.get(meth, meth)
        return f"{base}_{m}"

    # ------------------------------------------------------------ Python types

    def parse_ann(self, node) -> tuple | None:
        if node is None:
            return None
        return self.parse_text(ast.unparse(node))

    def parse_text(self, t: str) -> tuple | None:
        t = t.strip().strip("'\"").replace("typing.", "")
        if t in ("Any", "object"):
            return None  # the recorded type decides
        alts = _split(t, "|")
        if len(alts) > 1:
            ps = [self.parse_text(a) for a in alts]
            if None in ps:
                return None
            if ("none",) in ps and len(ps) == 2:
                return ("opt", next(p for p in ps if p != ("none",)))
            if set(ps) == {("int",), ("v",)}:
                return ("iv",)
            if all(p == ("type",) for p in ps):
                return ("type",)
            return None
        if t in SCALARS:
            return SCALARS[t]
        m = re.fullmatch(r"Optional\[(.+)\]", t)
        if m:
            x = self.parse_text(m.group(1))
            return ("opt", x) if x else None
        m = re.fullmatch(r"(?:Sequence|list|List)\[(.+)\]", t)
        if m:
            x = self.parse_text(m.group(1))
            return ("seq", x) if x else None
        if t in ("Sequence", "list", "tuple"):
            return None  # element type comes from the recorded facts
        m = re.fullmatch(r"tuple\[(.+),\s*\.\.\.\]", t)
        if m:
            x = self.parse_text(m.group(1))
            return ("seq", x) if x else None
        m = re.fullmatch(r"tuple\[(.+)\]", t)
        if m:
            xs = [self.parse_text(p) for p in _split(m.group(1))]
            return ("tuple", tuple(xs)) if None not in xs else None
        m = re.fullmatch(r"dict\[(.+?)[,:](.+)\]", t)
        if m:
            k, v = self.parse_text(m.group(1)), self.parse_text(m.group(2))
            return ("dict", k, v) if k and v else None
        m = re.fullmatch(r"Callable\[\[(.*)\], (.+)\]", t)
        if m:
            args = (
                [self.parse_text(p) for p in _split(m.group(1))] if m.group(1) else []
            )
            r = self.parse_text(m.group(2))
            return ("fn", tuple(args), r) if None not in args and r else None
        if self.known_class(t):
            return ("cls", t)
        if re.fullmatch(r"_?[A-Z]\w*", t):
            return ("obj", t)
        return None

    def recorded(self, tags: list[str]) -> tuple | None:
        """A recorded tag set: one tag, or a known union. Test stand-ins
        (SimpleNamespace, the driver's own classes) drop out when a real type
        was also seen."""
        stand_ins = {"SimpleNamespace"} | {
            k.split(".")[0] for k in self.mod.facts.get("driver", {})
        }
        real = [x for x in tags if x not in stand_ins]
        tags = real or tags
        ps = {self.parse_text(x) for x in tags}
        if None in ps:
            return None
        ps = {self.family_type(p) for p in ps}
        if len(ps) == 1:
            return ps.pop()
        if len(ps) > 1 and all(p == ("v",) or p[0] == "cls" for p in ps):
            # tagged values: one of several unrelated alternatives: a tagged union struct
            return ("union", tuple(sorted(ps, key=lambda p: (p != ("v",), p))))
        if ps == {("int",), ("v",)}:
            return ("iv",)
        if ("none",) in ps and len(ps) == 2:
            return ("opt", next(p for p in ps if p != ("none",)))
        return None

    def family_type(self, p):
        """A class of a family is the family's type (tagged values, designs/tagged_values.md)."""
        if p and p[0] == "cls":
            fam = self.fam(p[1])
            if fam is not None:
                return ("cls", fam.base)
        return p

    def loose_value(self, p, r):
        """A `Value` annotation whose recorded types are only one family's
        classes: the recorder shows the annotation is loose; the family wins."""
        return (
            p == ("v",)
            and r is not None
            and r[0] == "cls"
            and self.fam(r[1]) is not None
        )

    def ptype(self, fn, pname: str, ann: str) -> tuple | None:
        p = self.parse_text(ann) if ann else None
        fact = self.mod.fact_of(fn) or {"params": {}, "return": []}
        rec = fact["params"].get(pname)
        r = self.recorded(rec) if rec else None
        if p is None or self.loose_value(p, r):
            return r
        if r is not None and r[0] == "union" and self.family_type(p) in r[1]:
            return r  # the annotation names one alternative; the record shows them all
        # a bare Sequence / dict annotation: element types from the record
        if r is not None and _refines(p, r):
            return r
        return p

    # ------------------------------------------------------------ C types

    def cty(self, p, where: str) -> str:
        """C type of one value (not a sequence)."""
        k = p[0]
        if k == "v":
            return T.V
        if k == "int":
            return "int"
        if k == "bool":
            return "bool"
        if k == "float":
            return "double"
        if k == "str":
            return T.STR
        if k == "iv":
            return T.IV
        if k == "type":
            return T.TYPE
        if k == "mma":
            return "const rocke_mma_op_t*"
        if k == "b":
            return T.B
        if k == "opt" and p[1][0] in ("v", "str", "type", "mma"):
            return self.cty(p[1], where)
        if k == "opt" and p[1][0] == "cls":
            return f"const {self.struct_name(p[1][1])}*"  # NULL is None
        if k == "cls":
            return self.struct_name(p[1])
        if k == "tuple":
            return self.tuple_struct(p, where)
        if k == "union":
            return self.union_struct(p, where)
        raise Refuse(None, f"{where}: no C type for {p}")

    def union_struct(self, p, where: str) -> str:
        """tagged values: a value of one of several unrelated alternatives (designs/
        tagged_values.md): {kind; one member per alternative}, kind 0 the
        first. Several headers may define the same one, so it is guarded."""
        alts = []
        for a in p[1]:
            if a == ("v",):
                alts.append(("value", T.V))
            else:
                st = self.struct_name(a[1])
                alts.append(
                    (
                        (
                            st[len(f"rocke_{PREFIX}_") : -2]
                            if st.startswith(f"rocke_{PREFIX}_")
                            else snake(a[1])
                        ),
                        st,
                    )
                )
        name = f"rocke_{PREFIX}_" + "_or_".join(n for n, _ in alts)
        if name not in self.tuple_structs:
            guard = name.upper() + "_DEFINED"
            ks = ",\n".join(
                f"    {name.upper()}_KIND_{n.upper()} = {i}"
                for i, (n, _) in enumerate(alts)
            )
            fields = "\n".join(f"    {t} {n};" for n, t in alts)
            self.tuple_structs[name] = (
                f"#ifndef {guard}\n#define {guard}\n/* tagged values: one of several alternatives; kind says which member holds it */\n"
                f"typedef enum {name}_kind\n{{\n{ks}\n}} {name}_kind_t;\n"
                f"typedef struct {name}\n{{\n    {name}_kind_t kind;\n{fields}\n}} {name}_t;\n#endif"
            )
        return f"{name}_t"

    TAGS = {
        "v": "v",
        "int": "int",
        "bool": "bool",
        "float": "f64",
        "str": "str",
        "iv": "iv",
        "type": "type",
    }

    def tuple_struct(self, p, where: str) -> str:
        """A fixed tuple: a generated struct {f0, f1, ...}, named by its item
        types. Several headers may define the same one, so it is guarded."""
        items = p[1]
        if any(x[0] not in self.TAGS and x[0] != "tuple" for x in items):
            raise Refuse(None, f"{where}: tuple item types {items}")
        tags = "_".join(
            self.TAGS.get(x[0])
            or self.tuple_struct(x, where)[len(f"rocke_{PREFIX}_") : -2]
            for x in items
        )
        name = f"rocke_{PREFIX}_tuple_{tags}"
        if name not in self.tuple_structs:
            fields = "\n".join(
                f"    {self.cty(x, where)} f{i};" for i, x in enumerate(items)
            )
            guard = name.upper() + "_DEFINED"
            self.tuple_structs[name] = (
                f"#ifndef {guard}\n#define {guard}\ntypedef struct {name}\n{{\n{fields}\n}} {name}_t;\n#endif"
            )
        return f"{name}_t"

    def param(self, fn, pname, p, cname_fn) -> tuple[list, dict, list]:
        """C parameters, the sigmap slot and extra roles of one Python parameter."""
        k = p[0]
        slot = {"py": pname, "py_kind": "pos", "default": None, "match": "name"}
        where = f"{cname_fn}({pname})"
        if k == "cls":
            st = self.struct_name(p[1])
            slot.update(c=pname, c_type=f"const {st}*")
            return [(f"const {st}*", pname)], slot, []
        if k == "seq":
            el = p[1]
            if el[0] == "fn":
                et = self.callable_struct(cname_fn, pname, el)
            elif el[0] == "obj":
                et = self.protocol_struct(fn, cname_fn, pname, el[1])
            elif el[0] == "seq":
                # a list of lists: row pointers, row lengths, row count
                it = self.cty(el[1], where)
                rt = (
                    f"{it} const* const*" if it.endswith("*") else f"const {it}* const*"
                )
                slot.update(c=pname, c_type=rt, rows=True)
                return (
                    [
                        (rt, pname),
                        ("const int*", f"len_{pname}"),
                        ("int", f"n_{pname}"),
                    ],
                    slot,
                    [
                        {
                            "c": f"len_{pname}",
                            "c_type": "const int*",
                            "role": "row_lens",
                        },
                        {"c": f"n_{pname}", "c_type": "int", "role": "count"},
                    ],
                )
            else:
                et = self.cty(el, where)
            ct = f"{et} const*" if et.endswith("*") else f"const {et}*"
            slot.update(c=pname, c_type=ct)
            return (
                [(ct, pname), ("int", f"n_{pname}")],
                slot,
                [{"c": f"n_{pname}", "c_type": "int", "role": "count"}],
            )
        if k == "dict":
            kt = self.cty(p[1], where)
            kc = f"{kt} const*" if kt.endswith("*") else f"const {kt}*"
            vals = (
                [
                    (self.cty(x, where), f"{pname}_vals_{i}")
                    for i, x in enumerate(p[2][1])
                ]
                if p[2][0] == "tuple"
                else [(self.cty(p[2], where), f"{pname}_vals")]
            )
            vcs = [
                (f"{vt} const*" if vt.endswith("*") else f"const {vt}*", n)
                for vt, n in vals
            ]
            slot.update(c=f"{pname}_keys", c_type=kc, dict=True)
            return (
                [(kc, f"{pname}_keys")] + vcs + [("int", f"n_{pname}")],
                slot,
                [{"c": n, "c_type": vc, "role": "dict_vals"} for vc, n in vcs]
                + [{"c": f"n_{pname}", "c_type": "int", "role": "count"}],
            )
        if k == "fn":
            st = self.callable_struct(cname_fn, pname, p)
            slot.update(c=pname, c_type=st)
            return [(st, pname)], slot, []
        if k == "obj":
            st = self.protocol_struct(fn, cname_fn, pname, p[1])
            slot.update(c=pname, c_type=f"const {st}*")
            return [(f"const {st}*", pname)], slot, []
        if k == "opt" and p[1][0] in ("int", "bool", "float", "iv", "tuple"):
            ct = self.cty(p[1], where)
            slot.update(c=pname, c_type=ct, has=f"has_{pname}")
            return (
                [("bool", f"has_{pname}"), (ct, pname)],
                slot,
                [{"c": f"has_{pname}", "c_type": "bool", "role": "has"}],
            )
        ct = self.cty(p, where)
        slot.update(c=pname, c_type=ct)
        return [(ct, pname)], slot, []

    def callable_struct(self, fname: str, pname: str, p) -> str:
        args, r = p[1], p[2]
        base = f"{fname}_{pname.rstrip('s')}"
        rt = T.VOID if r == ("none",) else self.cty(r, base)
        ps = ", ".join(
            ["void* ctx"] + [f"{self.cty(a, base)} a{i}" for i, a in enumerate(args)]
        )
        self.typedefs.append(
            f"typedef {rt} (*{base}_fn)({ps});\ntypedef struct {base}\n{{\n    {base}_fn fn;\n    void* ctx;\n}} {base}_t;"
        )
        return f"{base}_t"

    def protocol_struct(self, fn, fname: str, pname: str, cls: str) -> str:
        """A duck-typed object: one function pointer per method the function
        calls on it, typed from the driver objects the recorder saw."""
        called = sorted(
            {
                n.func.attr
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            }
        )
        driver = self.mod.facts.get("driver", {})
        meths = {}
        for key, rec in driver.items():
            q = key.split("@")[0]
            if q.startswith(cls + ".") and q.split(".")[-1] in called:
                meths[q.split(".")[-1]] = rec
        if not meths:
            raise Refuse(
                fn,
                f"{fname}({pname}): object of class {cls} used through its methods, none recorded (protocol)",
            )
        base = f"{fname}_{pname.rstrip('s')}"
        fields = []
        for m, rec in sorted(meths.items()):
            args = [x for x in rec.get("args", []) if x != "self"]
            if not args and rec["params"]:
                raise Refuse(fn, f"{cls}.{m}: recorded without argument order")
            ts = []
            for x in args:
                p = self.recorded(rec["params"].get(x, []))
                if p is None:
                    raise Refuse(fn, f"{cls}.{m}({x}): no recorded type")
                ts.append(f"{self.cty(p, base)} {x}")
            r = self.recorded([y for y in rec["return"]]) or ("none",)
            rt = T.VOID if r == ("none",) else self.cty(r, base)
            self.typedefs.append(
                f"typedef {rt} (*{base}_{m}_fn)({', '.join(['void* ctx'] + ts)});"
            )
            fields.append(f"    {base}_{m}_fn {m};")
        self.typedefs.append(
            f"typedef struct {base}\n{{\n"
            + "\n".join(fields)
            + f"\n    void* ctx;\n}} {base}_t;"
        )
        return f"{base}_t"

    # ------------------------------------------------------------ declarations

    def function(self, fn: ast.FunctionDef, cls: str | None = None):
        from .translate import creates_builder, needs_fence, needs_seq_guards

        if cls is None and self.generator_split(fn):
            return
        is_cm = any(
            isinstance(d, ast.Name) and d.id == "classmethod" for d in fn.decorator_list
        )
        cname = self.method_name(cls, fn.name) if cls else self.func_name(fn.name)
        qual = f"{cls}.{fn.name}" if cls else fn.name
        pyps = py_params(fn)
        params, slots, extra = [], [], []
        if cls is not None:
            first = pyps.pop(0)
            if not is_cm:
                st = self.struct_name(cls)
                mut = fn.name == "__init__" or method_mutating(
                    self.env, self.mod, cls, fn
                )
                params.append((f"{st}*" if mut else f"const {st}*", "self"))
        builds = creates_builder(fn) or (
            fn.returns is not None and ast.unparse(fn.returns) == "KernelDef"
        )
        takes_b = any(p["annotation"] == "IRBuilder" or p["name"] == "b" for p in pyps)
        if builds or (not takes_b and self.needs_scratch_builder(fn)):
            params.append((T.B, "b"))
        a = fn.args
        defaults = {}
        plain = a.posonlyargs + a.args
        for i, x in enumerate(plain):
            j = i - (len(plain) - len(a.defaults))
            if j >= 0:
                defaults[x.arg] = ast.unparse(a.defaults[j])
        for x, d in zip(a.kwonlyargs, a.kw_defaults):
            if d is not None:
                defaults[x.arg] = ast.unparse(d)
        has_b = any(t == T.B for t, _ in params)
        for p in pyps:
            pt = self.ptype(fn, p["name"], p["annotation"])
            if (
                pt is None
                and p["name"] == "b"
                and (not p["annotation"] or re.fullmatch(r"\s*Any\s*", p["annotation"]))
            ):
                pt = ("b",)  # the builder by naming convention (`b: Any` too, B-R2)
            if pt is None:
                raise Refuse(fn, f"{cname}: no type for parameter {p['name']}")
            if pt == ("b",):
                has_b = True
            cps, slot, ex = self.param(fn, p["name"], pt, cname)
            slot["py_kind"] = p["kind"]
            slot["default"] = defaults.get(p["name"])
            params += cps
            slots.append(slot)
            extra += ex
        if not builds and needs_seq_guards(self.env, self.mod, fn):
            i = 1 if params and params[0][1] == "b" else 0
            params.insert(i, ("int*", "seq_guards"))
            extra.append({"c": "seq_guards", "c_type": "int*", "role": "seq_guards"})
        if not builds and needs_fence(self.mod, fn):
            params.append((T.STR, "fence"))
            extra.append({"c": "fence", "c_type": T.STR, "role": "fence"})
        ret, more, roles = self.returns(fn, cname, has_b, builds)
        params += more
        extra += roles
        doc = (ast.get_docstring(fn) or "").split("\n\n")[0].replace("\n", " ")
        self.decls.append(Decl(cname, ret, params, slots, extra, qual, doc))

    def needs_scratch_builder(self, fn, seen=None) -> bool:
        """The C side needs a builder the Python does not pass: a helper whose
        adapter or C twin takes one (eval_layout runs a lane map on a scratch
        builder)."""
        from .adapters import function_adapter

        seen = seen if seen is not None else set()
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
                continue
            name = n.func.id
            if name in self.mod.imports:
                src, attr = self.mod.imports[name]
                qual = f"{src.split('.')[-1]}:{attr}"
                ad = function_adapter(qual)
                if ad is not None and "{b}" in ad[0]:
                    return True
                e = self.env.sig.helpers.get(qual)
                if (
                    e
                    and any(p.get("type") == T.B for p in e["c"].get("params", []))
                    and not any(
                        p.get("annotation") == "IRBuilder"
                        for p in e.get("python", {}).get("params", [])
                    )
                ):
                    return True
            elif name in self.mod.functions and name not in seen:
                seen.add(name)
                if self.needs_scratch_builder(self.mod.functions[name], seen):
                    return True
        return False

    def returns(self, fn, cname, has_b, builds) -> tuple[str, list, list]:
        if builds:
            return "rocke_kernel_def_t*", [], []
        fact = self.mod.fact_of(fn) or {"return": []}
        r = self.parse_ann(fn.returns)
        rec = [x for x in fact["return"] if x != "None"]
        rr = self.recorded(rec) if rec else (("none",) if fact["return"] else None)
        if r is None or (rr is not None and _refines(r, rr)):
            r = rr
        if r is None:
            raise Refuse(fn, f"{cname}: no type for the return")
        raises = may_raise(self.mod, fn)
        reason = [("char*", "reason"), ("size_t", "reason_cap")]
        reason_roles = [
            {"c": "reason", "c_type": "char*", "role": "reason"},
            {"c": "reason_cap", "c_type": "size_t", "role": "reason_cap"},
        ]
        k = r[0]
        if k == "none":
            if has_b:
                return "bool", [], []
            return ("bool", reason, reason_roles) if raises else ("void", [], [])
        if k in ("v", "type", "mma") or (
            k == "opt" and r[1][0] in ("v", "type", "mma", "str", "cls")
        ):
            return self.cty(r, cname), [], []
        if k in ("int", "bool", "float"):
            ct = self.cty(r, cname)
            if raises and not has_b:
                return ct, reason, reason_roles
            return ct, [], []
        if k == "str":
            return (
                "rocke_status_t",
                [("char*", "out"), ("size_t", "out_cap")],
                [
                    {"c": "out", "c_type": "char*", "role": "str_out"},
                    {"c": "out_cap", "c_type": "size_t", "role": "str_cap"},
                ],
            )
        if k == "cls":
            return self.struct_name(r[1]), [], []
        fail_ret = "bool"
        tail, troles = ([], []) if has_b or not raises else (reason, reason_roles)
        if k == "tuple" and all(x[0] == "int" for x in r[1]):
            return (
                fail_ret,
                [("int*", "out")] + tail,
                [{"c": "out", "c_type": "int*", "role": f"fixed_out:{len(r[1])}"}]
                + troles,
            )
        if k == "tuple":
            return self.tuple_struct(r, cname), tail, troles
        if k == "opt" and r[1][0] in ("int", "bool", "float", "iv", "tuple", "cls"):
            ct = self.cty(r[1], cname)
            outs = [(f"{ct}*", "out"), ("bool*", "has_out")]
            return (
                fail_ret,
                outs + tail,
                [
                    {"c": "out", "c_type": f"{ct}*", "role": "opt_out"},
                    {"c": "has_out", "c_type": "bool*", "role": "opt_has"},
                ]
                + troles,
            )
        if k == "seq":
            el = r[1]
            if el[0] == "tuple":
                outs = [
                    (f"{self.cty(x, cname)}**", f"out_{i}") for i, x in enumerate(el[1])
                ]
                return (
                    fail_ret,
                    outs + [("int*", "n_out")] + tail,
                    [{"c": n, "c_type": t, "role": "owned_out"} for t, n in outs]
                    + [{"c": "n_out", "c_type": "int*", "role": "owned_count"}]
                    + troles,
                )
            et = self.cty(el, cname)
            return (
                fail_ret,
                [(f"{et}**", "out"), ("int*", "n_out")] + tail,
                [
                    {"c": "out", "c_type": f"{et}**", "role": "owned_out"},
                    {"c": "n_out", "c_type": "int*", "role": "owned_count"},
                ]
                + troles,
            )
        if k == "dict":
            kt = self.cty(r[1], cname)
            vs = (
                [self.cty(x, cname) for x in r[2][1]]
                if r[2][0] == "tuple"
                else [self.cty(r[2], cname)]
            )
            outs = [(f"{kt}**", "out_keys")] + [
                (f"{vt}**", f"out_vals_{i}" if len(vs) > 1 else "out_vals")
                for i, vt in enumerate(vs)
            ]
            return (
                fail_ret,
                outs + [("int*", "n_out")] + tail,
                [{"c": n, "c_type": t, "role": "owned_out"} for t, n in outs]
                + [{"c": "n_out", "c_type": "int*", "role": "owned_count"}]
                + troles,
            )
        raise Refuse(fn, f"{cname}: no C return for {r}")

    def dataclass_struct(self, cls: str, node: ast.ClassDef):
        fam = self.fam(cls)
        if fam is not None and fam.module == self.mod.modname:
            return (
                self.family_owns.get(fam.base, False) if cls == fam.base else False
            )  # the family's one struct
        st = self.struct_name(cls)
        lines, owns, _ = self.field_lines(cls, node, st)
        tag = st[: -len("_t")]
        if not lines:
            lines = ["char reserved; /* no state: C has no empty struct */"]
        self.structs[cls] = (
            self.closure_enums.get(cls, [])
            + [f"typedef struct {tag}", "{"]
            + ["    " + x for x in lines]
            + [f"}} {st};"]
        )
        return owns

    def family_struct(self, fam) -> bool:
        """tagged values: one struct for a class family (designs/tagged_values.md): a kind
        tag, then the base's fields, then each member's own fields in member
        order. A field several members have is one field; an int field joins
        an IntOrValue field of the same name."""
        st = f"rocke_{PREFIX}_{snake(fam.base[len(PKG.class_prefix):] if fam.base.startswith(PKG.class_prefix) else fam.base)}_t"
        kt = kind_type(fam)
        merged: dict[str, str] = {}
        owns = False
        for m in fam.members:
            lines, o, _ = self.field_lines(m, fam.nodes[m], st)
            owns = owns or o
            for ln in lines:
                mm = re.fullmatch(r"(.+?)\s*\b(\w+);(\s*/\*.*\*/)?", ln.strip())
                if mm is None:
                    merged.setdefault(ln, ln)  # a comment line of a hand field file
                    continue
                n, ct = mm.group(2), mm.group(1).strip()
                old = merged.get(n)
                if old is None:
                    merged[n] = ln
                    continue
                oct_ = re.fullmatch(r"(.+?)\s*\b\w+;.*", old.strip()).group(1).strip()
                if oct_ != ct and not (oct_ == T.IV and ct == "int"):
                    raise Refuse(
                        fam.nodes[m],
                        f"{st}: field {n} is {oct_} in one family member and {ct} in {m}",
                    )
        enum = [f"typedef enum {kt[:-2]}", "{"]
        for i, m in enumerate(fam.members):
            sep = "," if i + 1 < len(fam.members) else ""
            where = ", a duck-typed peer" if fam.is_peer(m) else ""
            enum.append(f"    {kind_const(fam, m)} = {i}{sep} /* {m}{where} */")
        enum.append(f"}} {kt};")
        body = [f"{kt} kind; /* the member class (tagged value) */"] + list(
            merged.values()
        )
        self.structs[fam.base] = (
            enum
            + ["", f"typedef struct {st[:-2]}", "{"]
            + ["    " + x for x in body]
            + [f"}} {st};"]
        )
        self.family_owns[fam.base] = owns
        return owns

    def field_lines(
        self, cls: str, node: ast.ClassDef, st: str
    ) -> tuple[list, bool, set]:
        """C field lines of a class, whether it owns lists, and its field names.
        A plain subclass of a class in this module starts with its base's
        fields, so a pointer to it is a valid pointer to the base."""
        base = next(
            (ast.unparse(b) for b in node.bases if ast.unparse(b) in self.mod.classes),
            None,
        )
        lines, owns, names = [], False, set()
        if base is not None and not _is_dataclass(node):
            lines, owns, names = self.field_lines(base, self.mod.classes[base], st)
        attrs = self.mod.facts.get("attrs", {}).get(cls, {})
        fields = [
            (s.target.id, s.annotation)
            for s in node.body
            if isinstance(s, ast.AnnAssign)
            and isinstance(s.target, ast.Name)
            and "ClassVar" not in ast.unparse(s.annotation)
        ]
        pycls = getattr(self.mod.pymod, cls, None)
        if _is_dataclass(node) and pycls is not None:
            import dataclasses as _dc

            # every dataclass field, inherited ones too, with its annotation text
            fields = [
                (
                    f.name,
                    (
                        ast.parse(f.type, mode="eval").body
                        if isinstance(f.type, str)
                        else None
                    ),
                )
                for f in _dc.fields(pycls)
            ]
        if not fields:
            fields = self_fields(node)
        elif _is_dataclass(node):
            # attributes __post_init__ sets with object.__setattr__ (recorded, not annotated)
            set_in_post = {
                c.args[1].value
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "__setattr__"
                and len(c.args) >= 2
                and isinstance(c.args[1], ast.Constant)
            }
            fields += [
                (a, None)
                for a in attrs
                if a in set_in_post and a not in {f for f, _ in fields}
            ]
        for name, ann in fields:
            if name in names:
                continue  # set by the base class too: one field
            sites = closure_fields(self.mod).get((cls, name))
            if sites:
                lines += self.closure_member(cls, name, sites, st)
                continue
            p = self.parse_ann(ann) if ann is not None else None
            if (p is None or p == ("v",)) and name in attrs:
                r = self.recorded(attrs[name])
                if p is None or self.loose_value(p, r):
                    p = r
            if p is None:
                raise Refuse(node, f"{st}: no type for field {name}")
            if p == ("none",):
                continue  # always None: no storage
            n = name.lstrip("_")
            if p[0] == "opt" and p[1][0] in (
                "int",
                "bool",
                "float",
                "iv",
                "tuple",
                "cls",
            ):
                lines += [f"bool has_{n};", f"{self.cty(p[1], st)} {n};"]
            elif p[0] == "seq":
                et = self.cty(p[1], st)
                lines += [f"{et}* {n};", f"int n_{n};", f"int cap_{n};"]
                owns = True
            else:
                lines.append(f"{self.cty(p, st)} {n};")
        lines += self.hand_fields(cls)
        return lines, owns, names | {f for f, _ in fields}

    def closure_member(self, cls: str, name: str, sites, st: str) -> list[str]:
        """host clones: a Callable field filled with closures of this module: a kind per
        definition site plus each site's captured values (designs/host_clones.md)."""
        kt = closure_kind_type(st, name)
        enum = [f"typedef enum {kt[:-2]}", "{"]
        lines = [f"{kt} {name.lstrip('_')}_kind; /* host clones: closure site */"]
        for i, site in enumerate(sites):
            sep = "," if i + 1 < len(sites) else ""
            enum.append(
                f"    {closure_kind_const(st, name, site)} = {i}{sep} /* {site.name}.<locals>.{site.node.name} */"
            )
            tags = site_types(self.mod, site)
            if tags is None:
                raise Refuse(
                    site.node,
                    f"{st}: no recorded type for a value closure {site.node.name} of {site.name} captures",
                )
            for (cap, _k), tg in zip(site.captured, tags):
                p = self.recorded(tg)
                if p is None:
                    raise Refuse(
                        site.node, f"{st}: no C type for captured {cap} ({tg})"
                    )
                ct = self.cty(p, st)
                if ct in (
                    "rocke_layout_map_t",
                    "rocke_arch_target_t",
                    "rocke_mma_catalog_t",
                ):
                    ct = f"const {ct}*"  # shared static platform objects: by pointer, as everywhere
                lines.append(f"{ct} {site_field(name, site, cap)};")
        enum.append(f"}} {kt};")
        self.closure_enums.setdefault(cls, []).extend(enum + [""])
        return lines

    def hand_fields(self, cls: str) -> list[str]:
        """C-only fields a hand fill needs (holes/<module>/<Class>.fields): the
        hand callers' duck-typed peers, e.g. a buffer resource a dispatching
        method fill reads. Appended after the Python fields, so subclasses
        still start with the base's full layout; zero means "not set"."""
        p = os.path.join(holes_dir(), self.mod.modname, f"{cls}.fields")
        if not os.path.exists(p):
            return []
        return [x.rstrip() for x in open(p).read().splitlines() if x.strip()]

    def constant(self, name: str, val):
        c = f"ROCKE_{PREFIX.upper()}_{name}"
        if isinstance(val, bool):
            self.consts.append(f"#define {c} {int(val)}")
        elif isinstance(val, int):
            self.consts.append(f"#define {c} {val}")
        elif isinstance(val, tuple) and val and all(isinstance(x, str) for x in val):
            self.consts.append(
                f"#define {c}_N {len(val)}\nextern const char* const {c}[{c}_N];"
            )
        elif (
            isinstance(val, tuple)
            and val
            and all(isinstance(x, int) and not isinstance(x, bool) for x in val)
        ):
            self.consts.append(
                f"#define {c}_N {len(val)}\nextern const int {c}[{c}_N];"
            )
        elif isinstance(val, str):
            self.consts.append(f"#define {c} {json.dumps(val)}")
        else:
            self.refused.append((name, f"constant of type {type(val).__name__}"))

    def rename(self):
        """data/abi/<module>.names.json: generated name -> the name to keep
        (an existing ABI's spelling). Applied to every C identifier."""
        p = os.path.join(data_dir(), "abi", f"{self.mod.modname}.names.json")
        if not os.path.exists(p):
            return
        names = json.load(open(p))
        pat = re.compile(
            r"\b("
            + "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
            + r")\b"
        )
        sub = lambda t: pat.sub(lambda m: names[m.group(1)], t)
        for d in self.decls:
            d.name = sub(d.name)
            d.ret = sub(d.ret)
            d.params = [(sub(t), n) for t, n in d.params]
            for sl in d.slots:
                if sl.get("c_type"):
                    sl["c_type"] = sub(sl["c_type"])
            for x in d.extra:
                x["c_type"] = sub(x["c_type"])
        self.typedefs = [sub(t) for t in self.typedefs]
        self.structs = {c: [sub(x) for x in ls] for c, ls in self.structs.items()}
        self.consts = [sub(c) for c in self.consts]

    def run(self):
        allv = getattr(self.mod.pymod, "__all__", None) or [
            n for n in vars(self.mod.pymod) if not n.startswith("_")
        ]
        allv = list(allv) + [n for n in self.imported_privates() if n not in allv]
        for name in allv:
            try:
                if name in self.mod.functions:
                    try:
                        self.function(self.mod.functions[name])
                    except Refuse:
                        if not self.keep_hand_func(name):
                            raise
                elif name in self.mod.classes:
                    node = self.mod.classes[name]
                    try:
                        owns = self.dataclass_struct(name, node)
                    except Refuse:
                        if not self.keep_hand_struct(name):
                            raise
                        self.hand_quals.add(name)
                        owns = False
                    for m in node.body:
                        if isinstance(m, ast.FunctionDef) and (
                            not m.name.startswith("_")
                            or m.name in ("__init__", "__post_init__")
                        ):
                            try:
                                self.function(m, cls=name)
                            except Refuse as e:
                                if not self.keep_hand_func(f"{name}.{m.name}"):
                                    self.refused.append(
                                        (
                                            f"{name}.{m.name}",
                                            (
                                                e.reason
                                                if hasattr(e, "reason")
                                                else str(e)
                                            ),
                                        )
                                    )
                    st = self.struct_name(name)
                    if owns:
                        self.decls.append(
                            Decl(
                                f"{st[:-2]}_free",
                                "void",
                                [(f"{st}*", "self")],
                                [],
                                [],
                                f"{name}<free>",
                                "Release the lists the struct owns.",
                            )
                        )
                    if _is_dataclass(node) and any(
                        isinstance(s, ast.AnnAssign) and s.value is not None
                        for s in node.body
                    ):
                        self.decls.append(
                            Decl(
                                self.method_name(name, "default"),
                                st,
                                [],
                                [],
                                [],
                                f"{name}<default>",
                                "Python's field defaults.",
                            )
                        )
                else:
                    self.constant(name, getattr(self.mod.pymod, name))
            except Refuse as e:
                self.refused.append(
                    (name, e.reason if hasattr(e, "reason") else str(e))
                )
        self.struct_closure()
        self.rename()  # final names first: support must not re-add what the rename produced
        self.support()
        self.hand_deps()
        return self

    # ------------------------------------------------------------ ABI holes

    def hand_header(self) -> str | None:
        from .translate import own_header

        return own_header(self.env, self.mod.modname)

    def keep_hand_func(self, qual: str) -> bool:
        """A refused function or method keeps its hand declaration (an ABI hole)."""
        e = self.env.sig.helpers.get(self.mod.qual(qual))
        c = (e or {}).get("c") or {}
        name = c.get("name")
        f = self.env.headers.funcs.get(name) if name else None
        if (
            f is None
            or f.file != self.hand_header()
            or name not in self.env.headers.decl_text
        ):
            return False
        self.hand[name] = self.env.headers.decl_text[name]
        self.hand_quals.add(qual)
        return True

    def generator_split(self, fn: ast.FunctionDef) -> bool:
        """A generator with a C4 split (tr/adapters.py GENERATORS) is exported
        as that split: its count and round companions keep their hand
        declarations, the shape every caller already uses."""
        from .adapters import generator_split

        g = generator_split(self.mod.qual(fn.name))
        if g is None or not any(
            isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn)
        ):
            return False
        names = [re.match(r"\s*(\w+)\(", g[k]).group(1) for k in ("count", "round")]
        h = self.env.headers
        if not all(
            n in h.funcs and h.funcs[n].file == self.hand_header() and n in h.decl_text
            for n in names
        ):
            return False
        for n in names:
            self.hand[n] = h.decl_text[n]
        return True

    def keep_hand_struct(self, cls: str) -> bool:
        sname = self.mod.class_struct.get(cls)
        if (
            not sname
            or self.env.headers.typedef_file.get(sname) != self.hand_header()
            or sname not in self.env.headers.decl_text
        ):
            return False
        self.hand[sname] = self.env.headers.decl_text[sname]
        self.struct_override[cls] = sname
        return True

    def struct_closure(self):
        """Own classes outside __all__ whose struct an exported declaration
        uses (a field of an exported struct, a parameter) get a struct too."""
        changed = True
        while changed:
            changed = False
            text = "\n".join(
                [x for ls in self.structs.values() for x in ls]
                + [d.text() for d in self.decls]
                + self.typedefs
            )
            for cls, node in self.mod.classes.items():
                if cls in self.structs or cls in self.struct_override:
                    continue
                fam = self.fam(cls)
                if fam is not None and cls != fam.base and fam.base in self.structs:
                    continue  # tagged values: the family's one struct is there
                if re.search(rf"\b{self.struct_name(cls)}\b", text):
                    try:
                        self.dataclass_struct(cls, node)
                    except Refuse:
                        if not self.keep_hand_struct(cls):
                            self.refused.append(
                                (cls, "struct an exported declaration uses")
                            )
                            continue
                    changed = True

    def support(self):
        """C-only support of the hand header: typedefs, macros and static
        inline helpers no Python name maps to (rocke_attn_bwd_iv_t and its
        iv_int / iv_val, array caps). Kept as they are."""
        hdr = self.hand_header()
        if hdr is None:
            return
        tm = json.load(open(os.path.join(maps_dir(), "twin_map.json")))["symbols"]
        twins = set()
        for k, e in tm.items():
            for tw in e.get("twins", []):
                twins.add(tw.get("symbol") or tw.get("c") or tw.get("name"))
        h = self.env.headers
        mine = set(self.tuple_structs) | {x + "_t" for x in self.tuple_structs}
        mine |= {d.name for d in self.decls} | set(self.hand)
        gen_all = "\n".join(
            self.typedefs + [x for ls in self.structs.values() for x in ls]
        )
        mine |= set(re.findall(r"}\s*(\w+)\s*;", gen_all)) | set(
            re.findall(r"\(\*(\w+)\)", gen_all)
        )
        mine |= {self.struct_name(c) for c in self.structs}
        gen_text = "\n".join(
            list(self.tuple_structs.values())
            + [x for ls in self.structs.values() for x in ls]
            + self.typedefs
            + self.consts
        )
        for name, text in h.decl_text.items():
            f = (
                h.typedef_file.get(name)
                or h.macro_file.get(name)
                or (h.funcs[name].file if name in h.funcs else None)
            )
            if (
                f != hdr
                or name in twins
                or name in mine
                or name.startswith("ROCKE_HELPER")
                or name.endswith("_H")
            ):
                continue
            if name in h.funcs and name not in h.inline_funcs:
                continue  # an exported C-only function: a hand ABI companion, not support
            if re.search(rf"\b{re.escape(name)}\b", gen_text) and name in h.macro_file:
                continue  # defined here already
            self.hand[name] = text
            self.support_names.add(name)

    def hand_deps(self):
        """Types the kept hand declarations use that this header must define."""
        hdr = self.hand_header()
        generated = "\n".join(
            list(self.tuple_structs.values())
            + [x for ls in self.structs.values() for x in ls]
            + self.typedefs
        )
        changed = True
        while changed:
            changed = False
            for text in list(self.hand.values()):
                for t in set(re.findall(r"\b(rocke_\w+_(?:t|fn))\b", text)):
                    if t in self.hand or re.search(rf"\b{t}\s*;|\(\*{t}\)", generated):
                        continue
                    if (
                        self.env.headers.typedef_file.get(t) == hdr
                        and t in self.env.headers.decl_text
                    ):
                        self.hand[t] = self.env.headers.decl_text[t]
                        changed = True


_SIBLING_OK: dict = {}


def sibling_family_ok(env: Env, mod: Module, f) -> bool:
    """Whether a sibling module's family derives its tagged struct (else that
    module keeps one struct per class, and so must every user)."""
    key = (os.path.dirname(mod.path), f.module, f.base)
    if key not in _SIBLING_OK:
        _SIBLING_OK[key] = True  # a cycle back here assumes it does
        try:
            from .kernel import TypeMap

            om = Module(
                env, f.module, os.path.join(os.path.dirname(mod.path), f.module + ".py")
            )
            om.bind_classes(
                json.load(open(os.path.join(maps_dir(), "twin_map.json")))["symbols"]
            )
            om.tmap = TypeMap(env, om)
            ok = Abi(env, om).fam(f.base) is not None
        except Exception:
            ok = False
        _SIBLING_OK[key] = ok
    return _SIBLING_OK[key]


@dataclass
class Decl:
    name: str
    ret: str
    params: list  # [(ctype, name)]
    slots: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    py: str = ""
    doc: str = ""

    def text(self) -> str:
        ps = ", ".join(f"{t} {n}" for t, n in self.params) or "void"
        return f"{self.ret} {self.name}({ps});"


def self_fields(node: ast.ClassDef) -> list:
    """Attributes a plain class sets in __init__, in order: (name, annotation or None)."""
    init = next(
        (
            m
            for m in node.body
            if isinstance(m, ast.FunctionDef) and m.name == "__init__"
        ),
        None,
    )
    pann = (
        {a.arg: a.annotation for a in (init.args.args + init.args.kwonlyargs)}
        if init is not None
        else {}
    )
    out: dict = {}

    def pairs(tgt, val):
        """(target, value) of one store; a tuple target over a tuple display of
        the same length pairs item by item (self.a, self.b = a, b), any other
        tuple target gives its items with no value."""
        if isinstance(tgt, ast.Tuple):
            vals = (
                val.elts
                if isinstance(val, ast.Tuple) and len(val.elts) == len(tgt.elts)
                else [None] * len(tgt.elts)
            )
            for t, v in zip(tgt.elts, vals):
                yield from pairs(t, v)
        else:
            yield tgt, val

    for n in ast.walk(init) if init is not None else []:
        stores = []
        if isinstance(n, ast.AnnAssign):
            stores = [(n.target, n.annotation)]
        elif isinstance(n, ast.Assign):
            # chained (self.a = self.b = v) and tuple (self.a, self.b = a, b) stores too
            for t in n.targets:
                for tgt, val in pairs(t, n.value):
                    # self.x = param: the parameter's (wider) annotation
                    stores.append(
                        (tgt, pann.get(val.id) if isinstance(val, ast.Name) else None)
                    )
        for tgt, ann in stores:
            if (
                isinstance(tgt, ast.Attribute)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
            ):
                if tgt.attr not in out or out[tgt.attr] is None:
                    out[tgt.attr] = ann
    return list(out.items())


MUTATORS = {
    "append",
    "extend",
    "pop",
    "insert",
    "clear",
    "remove",
    "update",
    "setdefault",
    "sort",
    "reverse",
}


def mutates_self(fn: ast.FunctionDef, me: str) -> bool:
    """self.x = ..., self.x += ..., self.x.append(...) (a list the struct owns),
    or object.__setattr__(self, ...)."""
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name)
            and n.value.id == me
        ):
            return True
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "__setattr__"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "object"
            and n.args
            and isinstance(n.args[0], ast.Name)
            and n.args[0].id == me
        ):
            return True  # object.__setattr__(self, ...) in a frozen dataclass
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in MUTATORS
        ):
            v = n.func.value
            if (
                isinstance(v, ast.Attribute)
                and isinstance(v.value, ast.Name)
                and v.value.id == me
            ):
                return True
    return False


def _field_class(mod: Module, cls: str, field: str) -> str | None:
    """The one class a field self.<field> of cls holds (never None): from the
    recorded attributes of cls's family, else the annotation of the __init__
    parameter stored into it."""
    fam = family_of(mod, cls)
    tags = set()
    for c in fam.members if fam is not None else [cls]:
        tags |= set(mod.facts.get("attrs", {}).get(c, {}).get(field, []))
    if tags:
        return (
            next(iter(tags))
            if len(tags) == 1
            and re.fullmatch(r"[A-Za-z_]\w*", next(iter(tags)))
            and tags != {"None"}
            else None
        )
    init = mod.methods.get((cls, "__init__"))
    if init is None:
        return None
    anns = {
        a.arg: ast.unparse(a.annotation)
        for a in init.args.args + init.args.kwonlyargs
        if a.annotation is not None
    }
    for n in ast.walk(init):
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Attribute)
            and isinstance(n.targets[0].value, ast.Name)
            and n.targets[0].value.id == "self"
            and n.targets[0].attr == field
            and isinstance(n.value, ast.Name)
        ):
            a = anns.get(n.value.id)
            return a if a and re.fullmatch(r"[A-Za-z_]\w*", a) else None
    return None


def _platform_struct(env: Env, mod: Module, c: str) -> str | None:
    """The C struct of a platform class with no twin_map class (PM-R1)."""
    if (
        c in mod.classes
        or c in mod.imports
        and mod.imports[c][0].split(".")[-1].startswith(PKG.modules)
    ):
        return None
    if mod.tmap is not None and c in mod.tmap.class_struct:
        return None
    sname = f"rocke_{snake(c)}_t"
    return sname if sname in env.headers.typedef_file else None


def u3_mutating(
    env: Env, mod: Module, cls: str, fn: ast.FunctionDef, seen: set | None = None
) -> bool:
    """Mutating self: the method calls, on a
    platform struct held by value in a field of self, a PM-R1 method whose C
    twin takes a non-const self (BandEmitter.row_band / cell_keep update the
    emitter's cached constant), or calls a method of its own class that does
    (any family member's version of it: the tagged-value dispatcher passes one self
    type to all). Only this carries through self calls: a method calling one
    that merely stores into self keeps its const self, as before."""
    if not fn.args.args:
        return False
    me = fn.args.args[0].arg
    seen = set() if seen is None else seen
    if (cls, fn.name) in seen:
        return False
    seen.add((cls, fn.name))
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        v, attr = n.func.value, n.func.attr
        if (
            isinstance(v, ast.Attribute)
            and isinstance(v.value, ast.Name)
            and v.value.id == me
        ):
            c = _field_class(mod, cls, v.attr)
            sname = _platform_struct(env, mod, c) if c else None
            if sname is None:
                continue
            from .lower import platform_c_method

            cf = platform_c_method(env, sname, attr)
            if cf is not None and not cf.params[0][0].startswith("const "):
                return True
        elif (
            isinstance(v, ast.Name)
            and v.id == me
            and _u3_method(env, mod, cls, attr, seen)
        ):
            return True
    return False


def _owners(mod: Module, cls: str, name: str) -> set:
    """The classes whose version of method name a call on cls may run: every
    version in cls's family (tagged values), else the one cls inherits."""
    fam = family_of(mod, cls)
    owners = (
        set(fam.impls(fam.base, name).values())
        if fam is not None and fam.module == mod.modname
        else set()
    )
    if not owners:
        c = cls
        while c is not None and (c, name) not in mod.methods:
            node = mod.classes.get(c)
            c = (
                next(
                    (
                        ast.unparse(b)
                        for b in node.bases
                        if ast.unparse(b) in mod.classes
                    ),
                    None,
                )
                if node is not None
                else None
            )
        owners = {c} if c is not None else set()
    return {o for o in owners if (o, name) in mod.methods}


def _u3_method(
    env: Env, mod: Module, cls: str, name: str, seen: set | None = None
) -> bool:
    return any(
        u3_mutating(env, mod, o, mod.methods[(o, name)], seen)
        for o in _owners(mod, cls, name)
    )


def method_mutating(env: Env, mod: Module, cls: str, fn: ast.FunctionDef) -> bool:
    """fn (a method of cls) takes a non-const self: it stores into self
    (mutates_self), or mutating self applies to its name in cls's family."""
    if mutates_self(fn, fn.args.args[0].arg if fn.args.args else "self"):
        return True
    return _u3_method(env, mod, cls, fn.name)


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any("dataclass" in ast.unparse(d) for d in node.decorator_list)


def _refines(a, b) -> bool:
    """b is a with element types filled in (a bare Sequence/dict, or an object)."""
    if a is None:
        return True
    if a == ("v",) and b == ("iv",):
        return True  # callers pass host ints too: the record is wider
    if a[0] != b[0]:
        return a[0] == "obj" and b[0] == "obj"
    if a[0] in ("seq", "opt"):
        return _refines(a[1], b[1])
    return False


SAFE_CALLS = {
    "len",
    "range",
    "min",
    "max",
    "int",
    "bool",
    "tuple",
    "list",
    "enumerate",
    "zip",
    "isinstance",
    "sorted",
    "abs",
    "sum",
    "any",
    "all",
    "dict",
    "str",
    "float",
}


def may_raise(mod: Module, fn: ast.FunctionDef, seen=None) -> bool:
    seen = seen if seen is not None else set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Raise, ast.Assert)):
            return True
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name):
            if f.id in SAFE_CALLS:
                continue
            if f.id in mod.functions:
                if f.id not in seen:
                    seen.add(f.id)
                    if may_raise(mod, mod.functions[f.id], seen):
                        return True
                continue
            return True
        if isinstance(f, ast.Attribute) and f.attr in (
            "append",
            "extend",
            "pop",
            "items",
            "keys",
            "values",
            "get",
            "join",
            "format",
        ):
            continue
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "self"
        ):
            continue
        return True
    return False


def own_decls(env: Env, header: str) -> dict:
    return {n: f for n, f in env.headers.funcs.items() if f.file == header}


def build(modname: str, rocke: str) -> tuple[Abi, Env, Module]:
    from .kernel import TypeMap
    from .env import module_path

    env = Env(rocke)
    mod = Module(env, modname, module_path(rocke, modname))
    tm = json.load(open(os.path.join(maps_dir(), "twin_map.json")))["symbols"]
    mod.bind_classes(tm)
    mod.tmap = TypeMap(env, mod)
    return Abi(env, mod).run(), env, mod


def diff(abi: Abi, env: Env, modname: str) -> list[str]:
    from .translate import own_header

    hdr = own_header(env, modname)
    old = own_decls(env, hdr) if hdr else {}
    out = [f"# {modname}: generated ABI vs {hdr}"]
    seen = set()
    counts = {"same": 0, "changed": 0, "new": 0, "gone": 0}
    for d in abi.decls:
        o = old.get(d.name)
        seen.add(d.name)
        if o is None:
            out.append(f"NEW      {d.text()}")
            counts["new"] += 1
            continue
        same = o.ret == d.ret and [t for t, _ in o.params] == [t for t, _ in d.params]
        counts["same" if same else "changed"] += 1
        if same:
            out.append(f"SAME     {d.name}")
        else:
            oldtxt = f"{o.ret} {o.name}({', '.join(f'{t} {n}' for t, n in o.params)});"
            out.append(f"CHANGED  {d.name}\n    old: {oldtxt}\n    new: {d.text()}")
    for n, o in old.items():
        if n not in seen and n not in abi.hand:
            out.append(f"GONE     {n}")
            counts["gone"] += 1
    for q in sorted(abi.hand_quals):
        out.append(f"HAND     {q} (ABI hole: kept from the hand header)")
    for name, why in abi.refused:
        out.append(f"REFUSED  {name}: {why}")
    out.append(
        f"# same {counts['same']}, changed {counts['changed']}, new {counts['new']}, gone {counts['gone']}, hand {len(abi.hand_quals)}, refused {len(abi.refused)}"
    )
    return out


def header_name(env: Env, modname: str) -> str:
    from .translate import own_header

    return own_header(env, modname) or PKG.header_name(modname)


def header_text(abi: Abi, env: Env, modname: str) -> str:
    """The module's header: constants, structs, callable structs, functions."""
    import hashlib

    name = header_name(env, modname)
    src = open(abi.mod.path, "rb").read()
    sha = hashlib.sha256(src).hexdigest()[:12]
    guard = "ROCKE_" + re.sub(r"\W", "_", name).upper()
    body = []
    for c in abi.consts:
        body += [c, ""]
    hand_types = [
        n
        for n in abi.hand
        if n not in env.headers.funcs or n in env.headers.inline_funcs
    ]
    hand_funcs = [
        n
        for n in abi.hand
        if n in env.headers.funcs and n not in env.headers.inline_funcs
    ]

    def where(n):
        f = (
            env.headers.typedef_file.get(n)
            or env.headers.macro_file.get(n)
            or env.headers.funcs[n].file
        )
        return open(env.headers.paths[f]).read().find(env.headers.decl_text[n])

    hand_types.sort(key=where)
    # every type definition (kept hand types, tuple structs, generated structs)
    # in one dependency order: what a definition names comes before it
    items = []  # (defined names, text, label)
    for n in hand_types:
        items.append(({n}, abi.hand[n], "ABI hole: kept from the hand header"))
    for nm, t in abi.tuple_structs.items():
        items.append(({nm + "_t", nm}, t, None))
    gen_names = []
    for c, lines in abi.structs.items():
        sn = abi.struct_name(c)
        gen_names.append(sn)
        items.append(({sn}, "\n".join(lines), f"py: {c}"))
    owner = {}
    for k, (ns, _t, _l) in enumerate(items):
        for x in ns:
            owner[x] = k
    deps = []
    for k, (ns, t, _l) in enumerate(items):
        used = set(re.findall(r"\b(rocke_\w+|ROCKE_\w+)\b", t)) - ns
        deps.append({owner[u] for u in used if u in owner and owner[u] != k})
    done, order_ = set(), []

    def visit(k, stack=()):
        if k in done or k in stack:
            return
        for d in sorted(deps[k]):
            visit(d, stack + (k,))
        done.add(k)
        order_.append(k)

    for k in range(len(items)):
        visit(k)
    if gen_names:
        # forward declarations: a struct may point to one defined after it
        body += [f"typedef struct {sn[:-2]} {sn};" for sn in gen_names] + [""]
    for k in order_:
        _ns, t, label = items[k]
        body += ([f"/* {label} */"] if label else []) + [t, ""]
    for t in abi.typedefs:
        body += [t, ""]
    for d in abi.decls:
        if d.doc:
            body.append(f"/* py: {d.py}. {d.doc} */")
        else:
            body.append(f"/* py: {d.py} */")
        body += [d.text(), ""]
    if hand_funcs:
        body += ["/* ABI holes: functions kept from the hand header */"]
        for n in hand_funcs:
            body += [abi.hand[n], ""]
    text = "\n".join(body)
    own = set(
        re.findall(r"\b(rocke_\w+_t)\b(?=\s*;\s*$)", text, re.M)
    )  # typedefs defined here
    used = set(re.findall(r"\b(rocke_\w+_t)\b", text)) - own
    incs = {"rocke/ir.h"}
    fwd = []
    for t in sorted(used):
        f = env.headers.typedef_file.get(t)
        if f is None or f == name:
            continue
        by_value = re.search(rf"\b{t}\b(?!\s*\*)", text)
        dt = env.headers.decl_text.get(t, "")
        m = re.match(r"\s*typedef\s+struct\s+(\w+)", dt)
        if (
            not by_value
            and m
            and PKG.header_module(f) is not None
            and PKG.header_module(f) not in PKG.included
        ):
            # only pointed to: a forward declaration (headers of the backward
            # modules include each other; an include could cycle)
            fwd.append(f"typedef struct {m.group(1)} {t};")
            continue
        incs.add(f"rocke/{f}")
    if fwd:
        text = (
            "/* pointed-to types of other backward headers */\n"
            + "\n".join(fwd)
            + "\n\n"
            + text
        )
    std = ["#include <stdbool.h>", "#include <stddef.h>", "#include <stdint.h>"]
    head = (
        [
            "/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.",
            " * SPDX-License-Identifier: MIT",
            " *",
            f" * rocke/{name} -- first draft by tr/abi.py from",
            f" * {PKG.py_root}/{PKG.py_dir}/{modname}.py (sha256 {sha});",
            " * hand-maintained from here.",
            " *",
            " * Rules: designs/generated_abi.md of the translator. Lists a function",
            " * returns are owned by the caller (free with free(); a list of strings:",
            " * free each string, then the list); errors of entries",
            " * that take the builder are recorded on it (false / NULL / -1), the",
            " * others fill reason.",
            " */",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
        ]
        + std
        + [""]
        + [f'#include "{i}"' for i in sorted(incs)]
        + ["", "#ifdef __cplusplus", 'extern "C" {', "#endif", ""]
    )
    tail = ["#ifdef __cplusplus", "}", "#endif", f"#endif /* {guard} */", ""]
    return "\n".join(head) + text + "\n".join(tail)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--rocke", help=ROCKE_HELP)
    ap.add_argument("--diff", action="store_true")
    ap.add_argument(
        "--header", metavar="DIR", help="write the generated header into DIR"
    )
    ap.add_argument(
        "--clang-format",
        metavar="EXE",
        help="clang-format to run (default: $CLANG_FORMAT, the venv's, or PATH)",
    )
    a = ap.parse_args(argv)
    a.rocke = rocke_root(a.rocke)
    abi, env, _mod = build(a.module, a.rocke)
    if a.diff:
        print("\n".join(diff(abi, env, a.module)))
    if a.header:
        os.makedirs(a.header, exist_ok=True)
        path = os.path.join(a.header, header_name(env, a.module))
        from .translate import clang_format, set_clang_format

        set_clang_format(a.clang_format)
        open(path, "w").write(clang_format(header_text(abi, env, a.module), a.rocke))
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
