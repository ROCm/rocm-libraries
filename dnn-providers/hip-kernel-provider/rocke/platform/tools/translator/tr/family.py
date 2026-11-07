# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tagged values: class families and duck-typed peers.

A family is one C struct with a kind tag (designs/tagged_values.md):
- a class of a package module and its subclasses in that module, when a
  subclass overrides a method, property or class attribute (other than
  __init__ / __post_init__);
- a class plus the duck-typed peers declared in data/abi/peers.json (classes
  of hand-ported modules that callers pass in its place).
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

from .env import data_dir
from .package import PKG

CTORS = ("__init__", "__post_init__")


@dataclass
class Family:
    base: str
    module: str  # the module defining the base
    members: list[str]  # base first, subclasses in source order, then peers
    parent: dict[
        str, str | None
    ]  # member -> its base inside the family (a peer: the base)
    peers: dict[str, str] = field(default_factory=dict)  # peer -> its module
    nodes: dict[str, ast.ClassDef] = field(default_factory=dict)

    def kind(self, member: str) -> int:
        return self.members.index(member)

    def is_peer(self, member: str) -> bool:
        return member in self.peers

    def derived(self, cls: str) -> list[str]:
        """cls and every member below it (a peer is below the base only)."""
        out = []
        for m in self.members:
            c = m
            while c is not None and c != cls:
                c = self.parent.get(c)
            if c == cls:
                out.append(m)
        return out

    def own_defs(self, cls: str) -> set[str]:
        """Methods, properties and class attributes cls itself defines."""
        node = self.nodes.get(cls)
        out = set()
        if node is None:
            return out
        for s in node.body:
            if isinstance(s, ast.FunctionDef):
                out.add(s.name)
            elif isinstance(s, ast.Assign):
                out |= {t.id for t in s.targets if isinstance(t, ast.Name)}
            elif (
                isinstance(s, ast.AnnAssign)
                and isinstance(s.target, ast.Name)
                and s.value is not None
                and "ClassVar" in ast.unparse(s.annotation)
            ):
                out.add(s.target.id)
        return out

    def impl(self, cls: str, name: str) -> str | None:
        """The member whose definition of `name` cls uses (through its bases)."""
        c = cls
        while c is not None:
            if name in self.own_defs(c):
                return c
            c = self.parent.get(c)
        return None

    def impls(self, cls: str, name: str) -> dict[str, str]:
        """member -> implementing member, for cls and every member below it."""
        return {
            m: i for m in self.derived(cls) if (i := self.impl(m, name)) is not None
        }

    def dispatches(self, cls: str, name: str) -> bool:
        return len(set(self.impls(cls, name).values())) > 1


def _classes(path: str) -> dict[str, ast.ClassDef]:
    tree = ast.parse(open(path).read())
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def _module_families(path: str, modname: str) -> list[Family]:
    cl = _classes(path)
    parent = {}
    for n, node in cl.items():
        parent[n] = next(
            (ast.unparse(b) for b in node.bases if ast.unparse(b) in cl), None
        )
    out = []
    roots = [n for n in cl if parent[n] is None]
    for r in roots:
        members = [n for n in cl if _root(n, parent) == r]
        if len(members) < 2:
            continue
        fam = Family(
            r,
            modname,
            members,
            {m: parent[m] for m in members},
            nodes={m: cl[m] for m in members},
        )
        poly = any(
            (d not in CTORS)
            and fam.parent[m] is not None
            and fam.impl(fam.parent[m], d) is not None
            for m in members
            for d in fam.own_defs(m)
        )
        if poly:
            out.append(fam)
    return out


def _root(n, parent):
    seen = set()
    while parent.get(n) is not None and n not in seen:
        seen.add(n)
        n = parent[n]
    return n


@lru_cache(maxsize=None)
def package_families(kernels_dir: str) -> dict[str, Family]:
    """class name -> its family, over every module of the package in kernels_dir."""
    out: dict[str, Family] = {}
    peers = {}
    p = os.path.join(data_dir(), "abi", "peers.json")
    if os.path.exists(p):
        peers = json.load(open(p))
    for f in sorted(os.listdir(kernels_dir)):
        if not f.endswith(".py") or not f.startswith(PKG.modules):
            continue
        modname = f[:-3]
        path = os.path.join(kernels_dir, f)
        fams = {fam.base: fam for fam in _module_families(path, modname)}
        decl = peers.get(modname, {})
        if decl:
            cl = _classes(path)
            for cls, plist in decl.items():
                fam = fams.get(cls)
                if fam is None:
                    fam = Family(cls, modname, [cls], {cls: None}, nodes={cls: cl[cls]})
                    fams[cls] = fam
                for q in plist:
                    pm, pc = q.split(":")
                    fam.members.append(pc)
                    fam.parent[pc] = cls
                    fam.peers[pc] = pm
                    fam.nodes[pc] = _classes(os.path.join(kernels_dir, pm + ".py"))[pc]
        for fam in fams.values():
            for m in fam.members:
                if not fam.is_peer(m):
                    out[m] = fam
    return out


def family_of(mod, cls: str) -> Family | None:
    """The family of a class this module sees (its own or a sibling's)."""
    if not cls:
        return None
    return package_families(os.path.dirname(mod.path)).get(cls)


def snake(name: str) -> str:
    import re

    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def stem(cls: str) -> str:
    """rocke_<prefix>_<snake class>: a class's own C name stem (functions)."""
    base = cls[len(PKG.class_prefix) :] if cls.startswith(PKG.class_prefix) else cls
    return f"rocke_{PKG.prefix}_{snake(base)}"


def kind_type(fam: Family) -> str:
    return f"{stem(fam.base)}_kind_t"


def kind_const(fam: Family, member: str) -> str:
    return f"{stem(fam.base).upper()}_KIND_{stem(member)[len(f'rocke_{PKG.prefix}_'):].upper()}"


# ---------------------------------------------------------------- unions
# A value of one of several unrelated alternatives (`Value` or a class / a
# family) in an exported signature: a tagged struct of the alternatives,
# rocke_<prefix>_<a>_or_<b>_t {kind; a; b;} (designs/tagged_values.md).


def union_fields(env, sname: str) -> dict[str, str] | None:
    """field -> C type of a generated union struct (kind excluded), else None."""
    s = env.headers.structs.get(sname) if sname else None
    if (
        s is None
        or "_or_" not in sname
        or s.fields.get("kind") != f"{sname[:-2]}_kind_t"
    ):
        return None
    return {f: s.fields[f] for f in s.order if f != "kind"}


def union_kind(sname: str, fld: str) -> str:
    return f"{sname[:-2].upper()}_KIND_{fld.upper()}"
