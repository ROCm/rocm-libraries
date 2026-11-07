# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""C++ types the translator tracks, spelled as strings, plus the fixed tables
that map Python annotations and IR objects to them (plan section 5.4)."""

from __future__ import annotations

import re

from .package import PKG

V = "rocke_value_t*"
B = "rocke_ir_builder_t*"
I64 = "int64_t"
INT = "int"
BOOL = "bool"
F64 = "double"
STR = "const char*"
SSTR = "std::string"
TYPE = "const rocke_type_t*"
IV = PKG.iv_t
VOID = "void"
NONE = "None"  # the type of the literal None
AUTO = "auto"

INT_TYPES = {I64, INT, "int32_t", "unsigned", "uint32_t", "size_t", "long", "long long"}
# scalar typedefs that end in _t but are not structs
NON_STRUCT = {
    "int64_t",
    "uint64_t",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "size_t",
    "rocke_status_t",
    "rocke_type_t",
    "rocke_value_t",
    "rocke_ir_builder_t",
}

# IR scalar types of rocke.core.ir and their C singletons
IR_SCALARS = {
    "I1": "rocke_i1()",
    "I8": "rocke_i8()",
    "I16": "rocke_i16()",
    "I32": "rocke_i32()",
    "I64": "rocke_i64()",
    "BF16": "rocke_bf16()",
    "F16": "rocke_f16()",
    "F32": "rocke_f32()",
    "F64": "rocke_f64()",
}


def variant_items(t: str) -> list[str] | None:
    m = re.fullmatch(r"std::variant<(.+)>", t or "")
    if not m:
        return None
    out, depth, cur = [], 0, ""
    for ch in m.group(1):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


def optional(t: str) -> str:
    return f"std::optional<{t}>"


def optional_inner(t: str) -> str | None:
    m = re.fullmatch(r"std::optional<(.+)>", t)
    return m.group(1) if m else None


def vec(t: str) -> str:
    return f"std::vector<{t}>"


def ptuple(t: str) -> str:
    return f"py::tuple<{t}>"


def arr(t: str, n: int) -> str:
    return f"std::array<{t}, {n}>"


def is_int(t: str) -> bool:
    return t in INT_TYPES


def is_ptr(t: str) -> bool:
    return t.endswith("*")


def is_str(t: str) -> bool:
    return t in (STR, SSTR)


def seq_elem(t: str) -> str | None:
    """Element type of std::vector<X>, py::tuple<X> or std::array<X, N>."""
    m = re.fullmatch(r"(?:std::vector|py::tuple|py::span|py::olist|py::set)<(.+)>", t)
    if m:
        return m.group(1)
    m = re.fullmatch(r"std::array<(.+), (\d+)>", t)
    if m:
        return m.group(1)
    return None


def arr_len(t: str) -> int | None:
    m = re.fullmatch(r"std::array<(.+), (\d+)>", t)
    return int(m.group(2)) if m else None


def is_seq(t: str) -> bool:
    return seq_elem(t) is not None


def struct_of(t: str) -> tuple[str, bool] | None:
    """'const rocke_x_t*' -> ('rocke_x_t', True); 'rocke_x_t' -> ('rocke_x_t', False)."""
    s = t.replace("const ", "").strip()
    base = s[:-1].strip() if s.endswith("*") and not s.endswith("**") else s
    if base in INT_TYPES or base in NON_STRUCT or not base.endswith("_t"):
        return None
    if s.endswith("*") and not s.endswith("**"):
        return base, True
    if "*" not in s:
        return s, False
    return None


def ann_text(node) -> str:
    import ast

    return ast.unparse(node) if node is not None else ""


def from_annotation(text: str, structs: dict | None = None) -> str | None:
    """Map a Python annotation to a C++ type, or None when the table has no row."""
    t = text.replace("typing.", "").strip().strip("'\"")
    t = re.sub(r"\s+", " ", t)
    if structs and t in structs:
        return structs[t]
    table = {
        "Value": V,
        "IRBuilder": B,
        "int": I64,
        "bool": BOOL,
        "str": STR,
        "float": F64,
        "None": VOID,
        "Value | None": V,
        "Optional[Value]": V,
        "int | Value": IV,
        "Value | int": IV,
        "IntOrValue": IV,
        "MmaOp": "const rocke_mma_op_t*",
        "KernelDef": "rocke_kernel_def_t*",
        "PtrType": TYPE,
        "Type": TYPE,
        "dict[str, Value]": "py::kargs",
        "LayoutMap": "const rocke_layout_map_t*",
    }
    if t in table:
        return table[t]
    m = re.fullmatch(r"Optional\[(.+)\]", t)
    alts = (
        [x.strip() for x in m.group(1).split("|")] + ["None"]
        if m
        else [x.strip() for x in _split_top(t.replace("|", ","))] if "|" in t else None
    )
    if alts and "None" not in alts and len(alts) == 2:
        # two types: a tagged union (isinstance selects the alternative)
        ts = [from_annotation(x, structs) for x in alts]
        if (
            all(ts)
            and len(set(ts)) == 2
            and IV not in ts
            and not any(x in (V, B) for x in ts)
            or (all(ts) and set(ts) == {I64, V})
        ):
            if set(ts) == {I64, V}:
                return IV
            ts = [
                (
                    x[len("const ") : -1]
                    if x.startswith("const ")
                    and x.endswith("*")
                    and struct_of(x)
                    and not x.startswith(
                        (
                            "const rocke_mma_op",
                            "const rocke_layout_map",
                            "const rocke_arch_target",
                        )
                    )
                    else x
                )
                for x in ts
            ]
            return f"std::variant<{', '.join(sorted(ts))}>"  # one spelling per union
    if alts and "None" in alts and len(alts) == 2:
        inner = from_annotation(next(x for x in alts if x != "None"), structs)
        if inner is None or inner == VOID:
            return None
        if is_ptr(inner) or inner == STR:
            return inner  # NULL is None
        st = struct_of(inner)
        if st and not st[1] and inner != IV and not inner.startswith(("std::", "py::")):
            return f"const {inner}*"  # an optional struct: a pointer, NULL is None
        return optional(inner)
    m = re.fullmatch(r"(?:dict|Dict|Mapping)\[(.+)\]", t)
    if m:
        kv = _split_top(m.group(1))
        if len(kv) == 2:
            k, v = from_annotation(kv[0].strip(), structs), from_annotation(
                kv[1].strip(), structs
            )
            if k and v and VOID not in (k, v):
                # strings a dict keeps are owned: a const char* could point at a temporary
                k, v = (SSTR if k == STR else k), (SSTR if v == STR else v)
                return f"py::dict<{k}, {v}>"
        return None
    m = re.fullmatch(r"(?:Sequence|list|List)\[(.+)\]", t)
    if m:
        e = from_annotation(m.group(1), structs)
        return vec(e) if e else None
    m = re.fullmatch(r"tuple\[(.+), \.\.\.\]", t)
    if m:
        e = from_annotation(m.group(1), structs)
        return ptuple(e) if e else None
    m = re.fullmatch(r"tuple\[(.+)\]", t)
    if m:
        parts = [p.strip() for p in _split_top(m.group(1))]
        ts = [from_annotation(p, structs) for p in parts]
        if all(ts) and len(set(ts)) == 1:
            return arr(ts[0], len(ts))
        if all(ts) and VOID not in ts:
            return f"std::tuple<{', '.join(ts)}>"
        return None
    return None


def _split_top(s: str) -> list[str]:
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


def common(a: str, b: str) -> str | None:
    """The type of a ternary or list holding both a and b."""
    if a == b:
        return a
    for x, y in ((a, b), (b, a)):
        if x == "dict?" and (
            y in ("py::attrs", "py::kargs") or y.startswith("py::dict<")
        ):
            return y  # an empty {} takes the other dict's type
        if x == "dict?" and (optional_inner(y) or "").startswith("py::dict<"):
            return optional_inner(y)  # x or {}: a dict either way
    if a == "list?" and seq_elem(b) is not None:
        return ptuple(seq_elem(b)) if b.startswith(("std::array", "py::tuple")) else b
    if b == "list?" and seq_elem(a) is not None:
        return ptuple(seq_elem(a)) if a.startswith(("std::array", "py::tuple")) else a
    ea, eb = seq_elem(a), seq_elem(b)
    if ea is not None and eb is not None and ea != eb and seq_elem(ea) and seq_elem(eb):
        # tuples of rows of different fixed lengths: rows of any length
        inner = common(ea, eb) or (
            ptuple(seq_elem(ea)) if seq_elem(ea) == seq_elem(eb) else None
        )
        if inner:
            return ptuple(inner)
    if ea is not None and eb is not None:
        e = common(ea, eb)
        if e is not None:
            if a.startswith("std::vector") and b.startswith("std::vector"):
                return vec(e)
            return ptuple(e)
    if a == NONE and (is_ptr(b) or b == STR):
        return b
    if b == NONE and (is_ptr(a) or a == STR):
        return a
    for x, y in ((a, b), (b, a)):
        if x == IV and (is_int(y) or y == V):
            return IV  # an int or a Value widens into int | Value
        if is_int(x) and y == V:
            return IV  # an int on one side, a Value on the other
        if optional_inner(x) and (
            optional_inner(x) == y or common(optional_inner(x), y) == optional_inner(x)
        ):
            return x  # optional[X] with X
        if x == NONE and (
            y.startswith(("std::vector<", "py::tuple<", "py::dict<", "std::array<"))
            or (struct_of(y) and not struct_of(y)[1] and y != IV)
        ):
            return optional(y)  # None, later a list / dict / struct
    for x, y in ((a, b), (b, a)):
        if x == NONE and (is_int(y) or y in (BOOL, F64, IV)):
            return optional(I64 if is_int(y) else y)
        if x == NONE and optional_inner(y):
            return y
    if is_int(a) and is_int(b):
        return I64
    if {a, b} <= {STR, SSTR}:
        return SSTR
    ta, tb = tuple_items(a), tuple_items(b)
    if ta is not None and tb is not None and len(ta) == len(tb):
        # mixed tuples of one arity: item by item (a str literal and a
        # formatted str in the same place are both std::string)
        items = [common(x, y) for x, y in zip(ta, tb)]
        if all(i is not None for i in items):
            return f"std::tuple<{', '.join(items)}>"
    return None


def pointee_elem(ct: str) -> str | None:
    """Element type of a C array parameter: 'X* const*' -> 'X*', 'const X*' -> 'X'."""
    ct = ct.strip()
    if ct.endswith("* const*"):
        return ct[: -len(" const*")]
    if ct.endswith("**"):
        return ct[:-1]
    if ct.startswith("const ") and ct.endswith("*"):
        inner = ct[len("const ") : -1].strip()
        if inner.endswith("*"):
            return "const " + inner
        return inner
    return None


def tuple_items(t: str) -> list[str] | None:
    """Item types of std::tuple<A, B, ...>."""
    if not (t.startswith("std::tuple<") and t.endswith(">")):
        return None
    inner = t[len("std::tuple<") : -1]
    out, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


def union_of(a: str, b: str) -> str | None:
    """tagged values (internal): the std::variant holding values of struct types a and b
    (None is std::monostate); tuples of one arity widen item by item. None when
    another kind of type is involved."""
    ia, ib = tuple_items(a), tuple_items(b)
    if ia is not None and ib is not None:
        if len(ia) != len(ib):
            return None
        items = []
        for x, y in zip(ia, ib):
            z = x if x == y else (common(x, y) or union_of(x, y))
            if z is None:
                return None
            items.append(z)
        return f"std::tuple<{', '.join(items)}>"

    def alts(t):
        if t in (NONE, "std::nullptr_t", "std::monostate"):
            return ["std::monostate"]
        vi = variant_items(t)
        if vi:
            return vi
        oi = optional_inner(t)
        if oi:
            return (
                alts(oi) + ["std::monostate"]
                if struct_of(oi) and not struct_of(oi)[1]
                else None
            )
        st = struct_of(t)
        return [t] if st and not st[1] and t.startswith("rocke_") else None

    xa, xb = alts(a), alts(b)
    if xa is None or xb is None:
        return None
    s = sorted(set(xa) | set(xb), key=lambda x: (x == "std::monostate", x))
    if len([x for x in s if x != "std::monostate"]) < 2:
        return None
    return f"std::variant<{', '.join(s)}>"
