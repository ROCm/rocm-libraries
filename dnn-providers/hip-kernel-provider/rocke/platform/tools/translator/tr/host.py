# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host clones: one generated version per value kind (designs/host_clones.md).

A host builder is a class with no state whose methods are all named like
IRBuilder methods (HostInt, IntLaneBuilder): Python passes it where it passes
the builder, and its methods compute on ints. A function called with a host
builder gets a host clone: the same body with the builder swapped for the host
builder value and IR values swapped for ints.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

from . import ctypes as T

HOST_VALUE = T.I64


def _is_host_builder_class(env, obj) -> bool:
    if not inspect.isclass(obj) or dataclasses.is_dataclass(obj):
        return False
    own = {
        k: v
        for k, v in vars(obj).items()
        if not (k.startswith("__") and k.endswith("__"))
    }
    if not own or "__init__" in vars(obj) or "__slots__" in vars(obj):
        return False
    if obj.__bases__ != (object,):
        return False
    return all(inspect.isfunction(v) and k in env.sig.ir for k, v in own.items())


def host_builders(mod) -> dict[str, str]:
    """Host builder classes visible in mod (its own and imported ones) -> their C struct."""
    if getattr(mod, "_host_builders", None) is not None:
        return mod._host_builders
    out = {}
    names = set(mod.classes) | set(mod.imports)
    for n in sorted(names):
        obj = getattr(mod.pymod, n, None)
        if obj is None or not _is_host_builder_class(mod.env, obj):
            continue
        st = mod.class_struct.get(n) or (
            mod.tmap.class_struct.get(n) if mod.tmap is not None else None
        )
        if st:
            out[n] = st
    mod._host_builders = out
    return out


def host_class_of(mod, t: str) -> str | None:
    """The host builder class whose struct type t is (by value or pointer)."""
    st = T.struct_of(t) if t else None
    if st is None:
        return None
    for cls, sname in host_builders(mod).items():
        if sname == st[0]:
            return cls
    return None


def host_type(t: str, images: dict | None = None) -> str:
    """t with the value kind swapped: IR values and int-or-values become ints,
    platform structs built on the host builder their host images."""
    if t == T.B:
        return t  # the builder parameter is replaced by the caller
    out = t.replace(T.V, HOST_VALUE).replace(T.IV, HOST_VALUE)
    out = re.sub(r"\bconst int64_t\b", HOST_VALUE, out)
    for ir, img in (images or {}).items():
        out = re.sub(rf"\b{ir}\b", img, out)
    return out


def clone_name(name: str, host_cls: str) -> str:
    return f"{name}__{host_cls}"


# ---------------------------------------------------------------- closures


@dataclasses.dataclass
class Site:
    """A closure of this module stored into a Callable field of a dataclass
    (designs/host_clones.md, "Closures called with both kinds")."""

    name: str  # the enclosing function: names the site (its kind and fields)
    enclosing: "ast.FunctionDef"
    node: "ast.FunctionDef"  # the closure
    own: list  # parameters the caller of the field passes
    captured: list  # [(name, "default" | "cell")]: values the struct holds


def _callable_fields(node) -> list[str]:
    import ast

    out = []
    for s in node.body:
        if (
            isinstance(s, ast.AnnAssign)
            and isinstance(s.target, ast.Name)
            and re.match(r"Callable\b", ast.unparse(s.annotation))
        ):
            out.append(s.target.id)
    return out


def _bindings(fn) -> dict[str, int]:
    """How often each name is bound in fn (parameters count once)."""
    import ast

    n: dict[str, int] = {}
    a = fn.args
    for x in (
        a.posonlyargs
        + a.args
        + a.kwonlyargs
        + ([a.vararg] if a.vararg else [])
        + ([a.kwarg] if a.kwarg else [])
    ):
        n[x.arg] = n.get(x.arg, 0) + 1
    for x in ast.walk(fn):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
            n[x.id] = n.get(x.id, 0) + 1
        elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x is not fn:
            n[x.name] = n.get(x.name, 0) + 1
    return n


def closure_fields(mod) -> dict:
    """(class, field) -> [Site] for each Callable dataclass field that this
    module's functions fill with a local closure, sites in source order."""
    import ast

    if getattr(mod, "_closure_fields", None) is not None:
        return mod._closure_fields
    out: dict = {}
    fields = {c: _callable_fields(n) for c, n in mod.classes.items()}
    for fname, fn in mod.functions.items():
        local = {s.name: s for s in fn.body if isinstance(s, ast.FunctionDef)}
        if not local:
            continue
        for call in ast.walk(fn):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and fields.get(call.func.id)
            ):
                continue
            cls = call.func.id
            for k in call.keywords:
                if (
                    k.arg in fields[cls]
                    and isinstance(k.value, ast.Name)
                    and k.value.id in local
                ):
                    clo = local[k.value.id]
                    a = clo.args
                    plain = a.posonlyargs + a.args
                    nd = len(a.defaults)
                    own = [x.arg for x in plain[: len(plain) - nd]]
                    captured = [(x.arg, "default") for x in plain[len(plain) - nd :]]
                    params = {x.arg for x in plain + a.kwonlyargs}
                    inner_bound = set(_bindings(clo)) - params
                    outer_bound = set(_bindings(fn))
                    cells = []
                    loads = [
                        x
                        for st in clo.body
                        for x in ast.walk(st)
                        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)
                    ]
                    for x in sorted(loads, key=lambda x: (x.lineno, x.col_offset)):
                        if (
                            x.id not in params
                            and x.id not in inner_bound
                            and x.id in outer_bound
                            and x.id not in cells
                        ):
                            cells.append(x.id)  # free variables in source order
                    captured += [(c, "cell") for c in cells]
                    site = Site(fname, fn, clo, own, captured)
                    lst = out.setdefault((cls, k.arg), [])
                    if not any(s.node is clo for s in lst):
                        lst.append(site)
    for v in out.values():
        v.sort(key=lambda s: s.node.lineno)
    mod._closure_fields = out
    return out


def site_field(fld: str, site: Site, cap: str) -> str:
    """The struct member holding captured value `cap` of `site`."""
    return f"{fld}_{site.name}_{cap.lstrip('_')}"


def closure_kind_type(struct: str, fld: str) -> str:
    """rocke_attn_bwd_frag_coords_t, base -> rocke_attn_bwd_frag_coords_base_kind_t"""
    return f"{struct[:-2]}_{fld.lstrip('_')}_kind_t"


def closure_kind_const(struct: str, fld: str, site: Site) -> str:
    return f"{struct[:-2]}_{fld.lstrip('_')}_{site.name.lstrip('_')}".upper()


def site_types(mod, site: Site) -> list | None:
    """Recorded tags of each captured value of site, or None if one is missing."""
    fact = mod.fact_of(site.node)
    if not fact:
        return None
    out = []
    for cap, kind in site.captured:
        tags = (fact.get("params") if kind == "default" else fact.get("cells", {})).get(
            cap
        )
        if not tags:
            return None
        out.append(tags)
    return out


IR_KIND = "IRBuilder"  # the clone name of a function's IR version when its own version is on ints


def kind_tags(tags: list[str], kind: str) -> list[str]:
    """The recorded tags of one value kind. The recorder mixes the calls of
    both kinds of dual code: the IR kind keeps the alternatives with IR
    values, the host kind the ones with ints (other tags stay)."""
    has_v = [t for t in tags if re.search(r"\bValue\b", t)]
    has_i = [
        t for t in tags if re.search(r"\bint\b", t) and not re.search(r"\bValue\b", t)
    ]
    if not has_v or not has_i:
        return list(tags)
    drop = set(has_i) if kind == IR_KIND else set(has_v)
    return [t for t in tags if t not in drop]
