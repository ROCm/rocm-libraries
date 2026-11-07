# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Kernel-builder rules: recorded types, structs for classes, imported classes with
C twins, the kernel-argument map, builder state, regions and constant tables.

Kept apart from lower.py so the core stays the expression and statement
lowering; Fn mixes these in.
"""

from __future__ import annotations

import ast
import json
import re
import os

from . import ctypes as T
from .env import data_dir, maps_dir

# builder typestate that Python keeps on the IRBuilder and C passes explicitly
STATE = {"fence": T.STR, "seq_guards": "int*"}


# ---------------------------------------------------------------- recorded types


def load_facts(modname: str) -> dict:
    p = os.path.join(data_dir(), "facts", f"{modname}.json")
    return json.load(open(p)) if os.path.exists(p) else {"funcs": {}, "attrs": {}}


def qualnames(tree: ast.Module) -> dict[int, str]:
    """id(FunctionDef) -> CPython co_qualname."""
    out = {}

    def walk(node, prefix):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, ast.FunctionDef):
                q = f"{prefix}{n.name}"
                out[id(n)] = q
                walk(n, q + ".<locals>.")
            elif isinstance(n, ast.ClassDef):
                walk(n, f"{prefix}{n.name}.")
            else:
                walk(n, prefix)

    walk(tree, "")
    return out


def twin_struct(env, mod, cls: str) -> str | None:
    """The C twin struct of class cls that mod imports from another module, found
    in the tree's header of that module (PKG.header_names): the struct its
    Python-to-C table names (`class TileSpec    rocke_gemm_tile_spec_t`), else its
    one struct named rocke_<snake(cls)>_t or rocke_*_<snake(cls)>_t. None when
    the module has no header or the header no such struct (the class then has no
    C type, and what uses it is refused)."""
    from .package import PKG

    if cls in mod.classes or cls not in mod.imports or "." in cls:
        return None
    src, name = mod.imports[cls]
    src = src.split(".")[-1]
    if name != cls or not re.fullmatch(r"[A-Z]\w*", cls) or src.startswith(PKG.modules):
        return None  # a sibling module's class follows the sibling struct rule
    h = env.headers
    snake_ = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls).lower()
    for hdr in PKG.header_names(src):
        path = h.paths.get(hdr)
        if path is None:
            continue
        m = re.search(
            rf"\bclass\s+{re.escape(cls)}\b[^\n]*?\b(rocke_\w+_t)\b", open(path).read()
        )
        if m and m.group(1) in h.structs and h.typedef_file.get(m.group(1)) == hdr:
            return m.group(1)
        cands = sorted(
            s
            for s, f in h.typedef_file.items()
            if f == hdr
            and s in h.structs
            and (s == f"rocke_{snake_}_t" or s.endswith(f"_{snake_}_t"))
        )
        if len(cands) == 1:
            return cands[0]
    return None


class TypeMap:
    """Recorded tags -> C++ types (plan section 5.4, a fixed table)."""

    def __init__(self, env, mod):
        self.env = env
        self.mod = mod
        tm = json.load(open(os.path.join(maps_dir(), "twin_map.json")))["symbols"]
        self.class_struct: dict[str, str] = {}
        for key, ent in tm.items():
            if ent.get("kind") != "class":
                continue
            cname = key.split(":", 1)[1]
            for tw in ent.get("twins", []):
                sym = tw.get("symbol") or tw.get("c") or tw.get("name")
                if sym in env.headers.structs:
                    self.class_struct.setdefault(cname, sym)
                    break
        # a class imported from another module whose C twin struct is in the tree
        # (TileSpec of gemm_universal: rocke_gemm_tile_spec_t)
        for cname in mod.imports:
            if cname not in mod.classes and cname not in self.class_struct:
                sym = twin_struct(env, mod, cname)
                if sym is not None:
                    self.class_struct[cname] = sym
        self.class_module: dict[str, str] = {}
        for key, ent in tm.items():
            if ent.get("kind") == "class":
                m, c = key.split(":", 1)
                self.class_module.setdefault(c, m)
        # Several classes can share one struct (RowSlab and BufferRowSlab are
        # both rocke_attn_bwd_row_slab_t). Prefer the class from a module this
        # module imports; first-seen otherwise.
        # tagged values: a member of a sibling's class family is the family's tagged struct
        from .family import kind_type, package_families

        self.families = package_families(os.path.dirname(mod.path))
        for c, fam in self.families.items():
            bs = self.class_struct.get(fam.base)
            s_ = env.headers.structs.get(bs) if bs else None
            if s_ is not None and s_.fields.get("kind") == kind_type(fam):
                self.class_struct[c] = bs
        imported = {m for m, _ in mod.imports.values()}
        self.struct_class: dict[str, str] = {}
        for c, s in self.class_struct.items():
            cur = self.struct_class.get(s)
            if cur is None or (
                self.class_module.get(c) in imported
                and self.class_module.get(cur) not in imported
            ):
                self.struct_class[s] = c

    def tag(self, t: str) -> str | None:
        table = {
            "Value": T.V,
            "IRBuilder": T.B,
            "int": T.I64,
            "bool": T.BOOL,
            "str": T.STR,
            "float": T.F64,
            "Type": T.TYPE,
            "VectorType": T.TYPE,
            "PtrType": T.TYPE,
            "KernelDef": "rocke_kernel_def_t*",
            "dict[str:Value]": "py::kargs",
            "callable": None,
        }
        if t in table:
            return table[t]
        m = re.fullmatch(r"tuple\[(.+),\.\.\.\]", t)
        if m:
            e = self.tag(m.group(1)) if "|" not in m.group(1) else None
            return T.ptuple(e) if e else None
        if t.startswith("tuple[") and t.endswith("]"):
            parts = _split_tags(t[6:-1])
            es = [self.tag(x) for x in parts]
            if None in es:
                return None
            return (
                T.arr(es[0], len(es))
                if len(set(es)) == 1
                else f"std::tuple<{', '.join(es)}>"
            )
        if t.startswith("dict[") and t.endswith("]") and t != "dict[str:Value]":
            k, _, v = t[5:-1].partition(":")
            kt, vt = self.tag(k), self.tag(v)
            if kt is None or vt is None or "|" in k + v:
                return None
            kt, vt = (T.SSTR if kt == T.STR else kt), (T.SSTR if vt == T.STR else vt)
            return f"py::dict<{kt}, {vt}>"
        if t.startswith("list[") and t.endswith("]"):
            inner = t[5:-1]
            if "|" in inner:
                return None
            e = self.tag(inner)
            return T.vec(e) if e else None
        if t in self.mod.class_struct:
            return self.mod.class_struct[t]
        if t in self.class_struct:
            return self.class_struct[t]
        # shared static platform objects: always referenced through a pointer
        if t in ("MmaOp", "LayoutMap", "ArchTarget", "MmaCatalog"):
            return {
                "MmaOp": "const rocke_mma_op_t*",
                "LayoutMap": "const rocke_layout_map_t*",
                "ArchTarget": "const rocke_arch_target_t*",
                "MmaCatalog": "const rocke_mma_catalog_t*",
            }[t]
        # a platform class with a C struct (BandEmitter -> rocke_band_emitter_t)
        snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", t).lower()
        if (
            re.fullmatch(r"[A-Z]\w*", t)
            and f"rocke_{snake}_t" in self.env.headers.structs
        ):
            return f"rocke_{snake}_t"
        return None

    def tags(self, ts: list[str], param=True) -> str | None:
        s = set(ts)
        # tagged values: classes of one family are its one tagged struct
        from .family import kind_type

        groups: dict = {}
        for x in s:
            fam = self.families.get(x)
            t = self.tag(x) if fam is not None else None
            s_ = self.env.headers.structs.get(t) if t else None
            if s_ is not None and s_.fields.get("kind") == kind_type(fam):
                groups.setdefault(t, []).append(x)
        for t, xs in groups.items():
            for x in sorted(xs)[1:]:
                s.discard(x)
        # test stand-ins drop out when a real type was also seen
        stand_ins = {"SimpleNamespace"} | {
            k.split(".")[0] for k in self.mod.facts.get("driver", {})
        }
        if s - stand_ins:
            s -= stand_ins
        # a host-evaluated builder (HostInt) next to the IR builder: the C side is IR only
        if "IRBuilder" in s and s <= {"IRBuilder", "HostInt", "IntLaneBuilder"}:
            return T.B
        if s == {"int", "Value"}:
            return T.IV
        if s == {"int", "Value", "None"}:
            return T.optional(T.IV)
        if s == {"None"} and param:
            return "std::nullptr_t"  # only ever None in the recordings
        if "None" in s and len(s) == 2:
            other = (s - {"None"}).pop()
            t = self.tag(other)
            if t and (T.is_ptr(t) or t == T.STR):
                return t
            st = T.struct_of(t) if t else None
            if st and not st[1] and t not in ("py::kargs", T.IV):
                # an optional struct: a pointer for a parameter (NULL is None), a
                # value otherwise (a pointer to a returned local would dangle)
                return f"const {t}*" if param else T.optional(t)
            if t and (T.is_int(t) or t in (T.BOOL, T.F64, T.IV)):
                return T.optional(t)
            return None
        tuples = [x for x in s if x.startswith("tuple[") and not x.endswith(",...]")]
        if len(s) > 1 and len(tuples) == len(s):
            # tuples of one arity: merged position by position
            split = [_split_tags(x[6:-1]) for x in tuples]
            if len({len(p) for p in split}) == 1:
                cols = [
                    self.tags([p[i] for p in split], param=False)
                    for i in range(len(split[0]))
                ]
                if None not in cols:
                    return (
                        T.arr(cols[0], len(cols))
                        if len(set(cols)) == 1
                        else f"std::tuple<{', '.join(cols)}>"
                    )
        if len(s) == 2 and "None" not in s:
            ts = [self.tag(x) for x in sorted(s)]
            if all(ts) and len(set(ts)) == 2:
                merged = T.common(ts[0], ts[1]) or (
                    T.ptuple(T.common(T.seq_elem(ts[0]), T.seq_elem(ts[1])))
                    if T.is_seq(ts[0])
                    and T.is_seq(ts[1])
                    and T.common(T.seq_elem(ts[0]), T.seq_elem(ts[1]))
                    else None
                )
                if merged:
                    return (
                        merged  # two shapes of one type (a tuple of 4 and a tuple of n)
                    )
                if T.is_seq(ts[0]) and T.is_seq(ts[1]):
                    return None
                # a tagged union of two kinds: struct values by value, platform objects by pointer
                return (
                    f"std::variant<{', '.join(sorted(ts))}>"  # one spelling per union
                )
        if len(s) == 1:
            t = self.tag(s.pop())
            st = T.struct_of(t) if t else None
            if param and st and not st[1] and t not in ("py::kargs", T.IV):
                return f"const {t}*"
            return t
        return None


def _split_tags(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def gen_struct_name(cls: str) -> str:
    return "py_" + cls.lstrip("_") + "_t"


def private_structs(env, mod, facts, tmap) -> list[str]:
    """C++ structs for module classes with no C twin (fields from recorded attrs)."""
    import dataclasses as _dc

    lines = []
    for cname, cnode in mod.classes.items():
        if cname in mod.class_struct:
            continue
        # a subclass that adds no state shares its base's struct
        bases = [ast.unparse(b) for b in cnode.bases]
        own_state = any(isinstance(x, ast.AnnAssign) for x in cnode.body) or any(
            isinstance(m, ast.FunctionDef) and m.name == "__init__" for m in cnode.body
        )
        base_struct = next(
            (
                mod.class_struct.get(b) or tmap.class_struct.get(b)
                for b in bases
                if (mod.class_struct.get(b) or tmap.class_struct.get(b))
            ),
            None,
        )
        if base_struct and not own_state:
            mod.class_struct[cname] = base_struct
            continue
        attrs = dict(facts.get("attrs", {}).get(cname) or {})
        pycls = getattr(mod.pymod, cname, None)
        ann = {}
        if pycls is not None and _dc.is_dataclass(pycls):
            # a dataclass __init__ is generated code the recorder never sees
            for f in _dc.fields(pycls):
                ann[f.name] = (
                    f.type
                    if isinstance(f.type, str)
                    else getattr(f.type, "__name__", str(f.type))
                )
        sname = gen_struct_name(cname)
        fields = {}
        order = []
        for a in list(ann) + [x for x in attrs if x not in ann]:
            t = tmap.tags(attrs[a], param=False) if a in attrs else None
            if t is None and a in ann:
                t = T.from_annotation(ann[a], {**tmap.class_struct, **mod.class_struct})
                st_ = T.struct_of(t) if t else None
                if st_ and not st_[1] and t in env.headers.structs and t != T.IV:
                    t = f"const {t}*"
            if t is None:
                continue
            fields[a.lstrip("_")] = t
            order.append(a.lstrip("_"))
        if not order and (attrs or ann):
            continue  # state we cannot type: leave the class to a hole
        from .env import CStruct

        env.headers.structs[sname] = CStruct(sname, fields, order, "<generated>")
        mod.class_struct[cname] = sname
        mod.struct_class[sname] = cname
        lines.append(f"/* py: class {cname} */")
        lines.append(f"struct {sname}")
        lines.append("{")
        for f in order:
            lines.append(f"    {fields[f]} {f}{{}};")
        lines.append("};")
        lines.append("")
    return lines


# ---------------------------------------------------------------- constant tables


def const_ctype(v) -> str | None:
    if isinstance(v, bool):
        return T.BOOL
    if isinstance(v, int):
        return T.I64
    if isinstance(v, str):
        return T.STR
    if isinstance(v, (tuple, frozenset, set)):
        items = sorted(v) if isinstance(v, (set, frozenset)) else list(v)
        ts = {const_ctype(x) for x in items}
        if len(ts) == 1:
            e = ts.pop()
            return T.ptuple(e) if e else None
        if not items:
            return T.ptuple(T.STR)
    return None


def const_init(v, t: str, lit_code) -> str:
    if T.seq_elem(t) is not None and isinstance(v, (tuple, set, frozenset)):
        items = sorted(v) if isinstance(v, (set, frozenset)) else list(v)
        el = T.seq_elem(t)
        return "{" + ", ".join(const_init(x, el, lit_code) for x in items) + "}"
    return lit_code(v)
