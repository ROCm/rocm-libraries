# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tables the translator reads: C headers, sigmap, and the imported Python modules.

Everything here is data the translator looks up; nothing here decides how to
lower Python. Header facts come from tools/cscan.py, Python facts from importing
the real modules (S1 of the plan).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import cscan  # noqa: E402

from .package import PKG  # noqa: E402

ROCKE_HELP = "rocKE root (the directory holding platform/ and library/); default: $ROCKE_ROOT, else the tree tr/ sits in"


def data_dir() -> str:
    """Per-module data (facts/, abi/, maps/): $TR_DATA, else data/ here."""
    return os.environ.get("TR_DATA") or os.path.join(REPO, "data")


def holes_dir() -> str:
    """Hand fills (holes/<module>/): $TR_HOLES, else holes/ here."""
    return os.environ.get("TR_HOLES") or os.path.join(REPO, "holes")


def library_dir(rocke: str) -> str:
    """The import root the package's modules are read and imported from (for
    attention backward <rocke>/library, which holds kernels/common/):
    $TR_LIBRARY, else <rocke>/<package py_root>. The self-test points it at
    pinned fixture modules."""
    return os.environ.get("TR_LIBRARY") or os.path.join(rocke, *PKG.py_root.split("/"))


def module_dir(rocke: str) -> str:
    """The directory of the package's Python modules."""
    return os.path.join(library_dir(rocke), *PKG.py_dir.split("/"))


def module_path(rocke: str, modname: str) -> str:
    """The source file of module modname; exits with a message if the package
    has no such module."""
    path = os.path.join(module_dir(rocke), modname + ".py")
    if not os.path.isfile(path):
        sys.exit(
            f"error: no module {modname} in package {PKG.name} ({path});"
            " pick the package with --package or $TR_PACKAGE"
        )
    return path


def parity_dir(rocke: str) -> str:
    """The directory of the package's parity drivers (<module>_emit.py):
    $TR_PARITY (a driver drafted outside the tree), else the package's."""
    return os.environ.get("TR_PARITY") or os.path.join(
        rocke, *PKG.parity_dir.split("/")
    )


def maps_dir() -> str:
    """The code maps tools/mine_maps.py mines from a rocKE tree (sigmap.json,
    twin_map.json): $TR_MAPS, else <data>/maps."""
    return os.environ.get("TR_MAPS") or os.path.join(data_dir(), "maps")


def enclosing_rocke() -> str | None:
    """The rocKE tree this copy of the translator sits in (under platform/tools/), if any."""
    p = REPO
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "platform", "cpp")) and os.path.isdir(
            os.path.join(p, "library", "kernels")
        ):
            return p
        p = os.path.dirname(p)
    return None


def rocke_root(arg: str | None = None) -> str:
    """The rocKE tree to read: --rocke, else $ROCKE_ROOT, else the tree tr/ sits
    in. Exits with a message if there is none."""
    root = arg or os.environ.get("ROCKE_ROOT") or enclosing_rocke()
    if not root:
        sys.exit("error: no rocKE tree: pass --rocke <root> or set ROCKE_ROOT")
    if not os.path.isdir(os.path.join(root, "platform", "cpp")):
        sys.exit(f"error: {root} is not a rocKE root (no platform/cpp)")
    return root


def norm_ctype(t: str) -> str:
    """'rocke_value_t *' -> 'rocke_value_t*', collapse spaces."""
    t = re.sub(r"\s+", " ", t.strip())
    t = re.sub(r"\s*\*\s*", "*", t)
    t = t.replace("*const*", "* const*").replace("*const", "* const")
    return t


def split_params(params: str) -> list[tuple[str, str]]:
    """'rocke_ir_builder_t* b, int n' -> [('rocke_ir_builder_t*', 'b'), ('int', 'n')]."""
    out = []
    params = params.strip()
    if not params or params == "void":
        return out
    depth = 0
    cur = ""
    parts = []
    for ch in params:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    for p in parts:
        p = p.strip()
        m = re.match(r"(.*?)([A-Za-z_]\w*)\s*(\[[^\]]*\])?$", p)
        if not m:
            out.append((norm_ctype(p), ""))
            continue
        ty, name, arr = m.group(1), m.group(2), m.group(3)
        ty = norm_ctype(ty)
        if arr:
            ty = ty + "*" if not ty.endswith("*") else ty + "*"
            ty = "const " + ty if not ty.startswith("const") else ty
        out.append((ty, name))
    return out


@dataclass
class CFunc:
    name: str
    ret: str
    params: list  # [(ctype, name)]
    file: str
    line: int


@dataclass
class CStruct:
    name: str  # typedef name
    fields: dict  # name -> ctype
    order: list = field(default_factory=list)
    file: str = ""


class Headers:
    """Functions, structs, enums, extern variables and macros of include/rocke."""

    def __init__(self, rocke: str, overlay: str | None = None):
        inc = os.path.join(rocke, "platform", "cpp", "include", "rocke")
        # overlay: generated headers that replace the same-named file of the tree
        over = (
            {fn: os.path.join(overlay, fn) for fn in os.listdir(overlay)}
            if overlay
            else {}
        )
        self.funcs: dict[str, CFunc] = {}
        self.structs: dict[str, CStruct] = {}
        self.enum_of: dict[str, str] = {}  # enumerator -> enum typedef
        self.enums: dict[str, list[str]] = {}  # enum typedef -> enumerators
        self.vars: dict[str, str] = {}  # extern name -> ctype text
        self.macros: dict[str, str] = {}
        self.file_of: dict[str, str] = {}
        self.fnptrs: dict[str, tuple] = {}  # typedef name -> (ret, [(ctype, name)])
        self.typedef_file: dict[str, str] = {}  # typedef name -> header defining it
        self.decl_text: dict[str, str] = (
            {}
        )  # function prototype / typedef / macro / inline -> its source text
        self.macro_file: dict[str, str] = {}
        self.inline_funcs: set[str] = set()
        # function -> every header declaring it (kernel_name_join is in
        # helper_rocke.helpers.spec.h and helper_helper_rocke.helpers.spec.h)
        self.func_files: dict[str, list[str]] = {}
        self.paths: dict[str, str] = {}  # header file name -> path read
        for fn in sorted(set(os.listdir(inc)) | set(over)):
            if not fn.endswith((".h", ".hpp")):
                continue
            path = over.get(fn) or os.path.join(inc, fn)
            self.paths[fn] = path
            syms, _ = cscan.scan_file(path)
            raw = open(path).read()
            text = raw.splitlines()
            for m in re.finditer(
                r"typedef\s+([^;()]+?)\s*\(\s*\*\s*(\w+)\s*\)\s*\(([^;]*?)\)\s*;",
                raw,
                re.S,
            ):
                self.fnptrs[m.group(2)] = (
                    norm_ctype(m.group(1)),
                    split_params(re.sub(r"\s+", " ", m.group(3))),
                )
            typedef_ranges = []
            for s in syms:
                if s.in_struct:
                    continue
                if s.kind == "function" and fn not in self.func_files.get(s.name, []):
                    self.func_files.setdefault(s.name, []).append(fn)
                if s.kind == "function" and s.linkage == "external":
                    self.funcs.setdefault(
                        s.name,
                        CFunc(
                            s.name,
                            norm_ctype(s.ret),
                            split_params(s.params),
                            fn,
                            s.line,
                        ),
                    )
                    self.file_of.setdefault(s.name, fn)
                    proto = []
                    for ln in text[s.line - 1 : s.line + 40]:
                        proto.append(ln)
                        if ";" in ln or "{" in ln:
                            break
                    if proto and ";" in proto[-1]:
                        self.decl_text.setdefault(s.name, "\n".join(proto))
                elif s.kind == "function" and s.name not in self.funcs:
                    # static inline helpers in headers are callable too
                    self.funcs[s.name] = CFunc(
                        s.name, norm_ctype(s.ret), split_params(s.params), fn, s.line
                    )
                    self.file_of.setdefault(s.name, fn)
                    if s.end_line:
                        self.decl_text.setdefault(
                            s.name, "\n".join(text[s.line - 1 : s.end_line])
                        )
                        self.inline_funcs.add(s.name)
                elif s.kind == "typedef":
                    typedef_ranges.append(s)
                    self.typedef_file.setdefault(s.name, fn)
                    self.decl_text.setdefault(
                        s.name, "\n".join(text[s.line - 1 : max(s.end_line, s.line)])
                    )
                elif s.kind == "macro" and not s.guard:
                    self.macros[s.name] = s.value
                    self.macro_file.setdefault(s.name, fn)
                    end = s.line
                    while end <= len(text) and text[end - 1].rstrip().endswith("\\"):
                        end += 1
                    self.decl_text.setdefault(s.name, "\n".join(text[s.line - 1 : end]))
                elif s.kind == "variable":
                    self.vars[s.name] = fn
            for s in syms:
                if s.kind == "enumerator":
                    for t in typedef_ranges:
                        if t.line <= s.line <= t.end_line:
                            self.enum_of[s.name] = t.name
                            self.enums.setdefault(t.name, []).append(s.name)
                            break
            for t in typedef_ranges:
                body = "\n".join(text[t.line - 1 : t.end_line])
                if "struct" in body.split("{")[0] and "{" in body:
                    self.structs[t.name] = self._parse_struct(t.name, body, fn)

    @staticmethod
    def _parse_struct(name: str, body: str, fn: str) -> CStruct:
        inner = body[body.index("{") + 1 : body.rindex("}")]
        inner = re.sub(r"/\*.*?\*/", "", inner, flags=re.S)
        inner = re.sub(r"//[^\n]*", "", inner)
        fields: dict[str, str] = {}
        dims: dict[str, str] = {}
        order = []
        depth = 0
        cur = ""
        for ch in inner:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == ";" and depth == 0:
                decl = cur.strip()
                cur = ""
                if not decl or "(" in decl:
                    continue
                m = re.match(
                    r"(.*?)([A-Za-z_]\w*)\s*(\[[^\]]*\])?$", decl.split(",")[0]
                )
                if not m:
                    continue
                ty = norm_ctype(m.group(1))
                names = [m.group(2)] + [
                    x.strip().split("[")[0] for x in decl.split(",")[1:]
                ]
                for n in names:
                    dim = m.group(3).strip("[] ") if m.group(3) else ""
                    fields[n] = ty + ("[]" if m.group(3) else "")
                    if dim:
                        dims[n] = dim  # the C array length (a number or a macro)
                    order.append(n)
            else:
                cur += ch
        cs = CStruct(name, fields, order, fn)
        cs.dims = dims
        return cs


class SigMap:
    def __init__(self, path: str):
        d = json.load(open(path))
        self.ir = d["ir_builder"]
        self.helpers = d["helpers"]
        self.external = d.get("external_helpers", {})

    def external_function(self, module: str, name: str) -> dict | None:
        """A function of another rocke module (rocke.helpers.spec) that has a C
        twin, in the shape of a `helpers` entry ({"python": ..., "c": ...}).

        mine_maps tags the C parameters no Python argument feeds only coarsely
        (out / count / extra); the ones a call can still supply are re-tagged
        here from the C declaration alone:
        - `char* X` then `size_t X_cap` (or `out_cap`), for a Python `-> str`:
          the string result written into a caller buffer (str_out, str_cap);
        - `size_t* X_len` right after it: the result's length, not wanted (null);
        - `rocke_arena_t*` of a helper with no builder: a scratch arena (arena).
        None when the module or name has no C function twin."""
        e = self.external.get(module, {}).get("symbols", {}).get(name)
        if not e or e.get("kind") != "function":
            return None
        c = e.get("c")
        if not c or c.get("kind") != "function" or "slots" not in c:
            return None
        extra = [dict(x) for x in c.get("extra_c", [])]
        returns = e.get("returns") or ""
        takes_b = any(
            p_.get("type", "").replace(" ", "") == "rocke_ir_builder_t*"
            for p_ in c.get("params", [])
        )
        for i, x in enumerate(extra):
            nxt = extra[i + 1] if i + 1 < len(extra) else None
            if (
                x["c_type"].replace(" ", "") == "rocke_arena_t*"
                and x.get("role") == "extra"
                and not takes_b
            ):
                # the arena a builder-free helper formats its strings into: a
                # scratch arena of the caller's scope
                x["role"] = "arena"
                continue
            if (
                returns == "str"
                and x["c_type"].replace(" ", "") == "char*"
                and nxt is not None
                and nxt["c_type"] == "size_t"
                and nxt["c"] in (x["c"] + "_cap", "out_cap")
            ):
                x["role"], nxt["role"] = "str_out", "str_cap"
                aft = extra[i + 2] if i + 2 < len(extra) else None
                if (
                    aft is not None
                    and aft["c_type"].replace(" ", "") == "size_t*"
                    and aft["c"].endswith("_len")
                ):
                    aft["role"] = "null"
        return {
            "python": {"params": e.get("params", []), "returns": returns},
            "c": dict(c, extra_c=extra),
        }


class PyModules:
    """Imports the real modules of the package (S1: import snapshot)."""

    def __init__(self, rocke: str):
        for p in (
            os.path.join(rocke, "platform", "python"),
            library_dir(rocke),
        ):
            if p not in sys.path:
                sys.path.insert(0, p)
        self.cache: dict[str, object] = {}

    def module(self, name: str):
        if name not in self.cache:
            full = PKG.module_name(name)
            self.cache[name] = importlib.import_module(full)
        return self.cache[name]


class Env:
    def __init__(
        self, rocke: str, sigmap: str | None = None, overlay: str | None = None
    ):
        self.rocke = rocke
        self.overlay = overlay
        self.headers = Headers(rocke, overlay)
        self.sig = SigMap(sigmap or os.path.join(maps_dir(), "sigmap.json"))
        self.py = PyModules(rocke)

    def c_func(self, name: str) -> CFunc | None:
        """A C function callable from another file: declared in a header.

        sigmap sometimes points a private Python helper at a file-static C
        function; the exported twin carries the backward prefix instead.
        """
        f = self.headers.funcs.get(name)
        if f is not None:
            return f
        # only the backward prefix: a bare rocke_ name would hit core IR functions
        f = self.headers.funcs.get(PKG.c + name.lstrip("_"))
        return f

    def enum_for_string(self, enum_t: str, s: str) -> str | None:
        """'head_major' in enum rocke_attn_bwd_ws_layout_t -> ROCKE_ATTN_BWD_WS_HEAD_MAJOR."""
        want = "_" + re.sub(r"\W", "_", s).upper()
        hits = [e for e in self.headers.enums.get(enum_t, []) if e.endswith(want)]
        return hits[0] if len(hits) == 1 else None
