# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``ck.dsl.ir/v1`` — round-trippable IR serialization.

This is the machine interchange format specified in
``dsl_docs/architecture/ir_serialization_format.md``. Unlike
:func:`rocke.core.ir_print.print_ir` (human-only, lossy, unparseable) this
captures *everything* needed to reconstruct a :class:`KernelDef` exactly,
including the **explicit SSA value ids**, so the consumer prints what was
assigned rather than re-deriving it (killing the SSA-numbering-drift class).

Public surface:

* :func:`serialize` — ``KernelDef -> str`` (deterministic, canonical).
* :func:`parse`     — ``str -> KernelDef`` (the inverse).
* :func:`canonicalize` / :func:`canonical_equal` — semantic diff with stable
  SSA-id normalization, tolerant of incidental id gaps.

Standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ir import (
    KernelDef,
    Op,
    Param,
    PtrType,
    Region,
    SmemType,
    Type,
    Value,
    VectorType,
)

FORMAT_NAME = "ckdsl.ir"
FORMAT_VERSION = "v1"

_INDENT = "  "


# --------------------------------------------------------------------------
# String escaping (§1.1)
# --------------------------------------------------------------------------


def _escape(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return "".join(out)


def _unescape(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "\\":
                out.append("\\")
            elif nxt == '"':
                out.append('"')
            elif nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            else:
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------
# Type encoding (§2)
# --------------------------------------------------------------------------


def _encode_type(t: Type) -> str:
    # ``Type.name`` is already the canonical spelling used across the stack.
    return t.name


def _parse_type(s: str) -> Type:
    s = s.strip()
    if s.startswith("vec<") and s.endswith(">"):
        inner = s[4:-1]
        # split at the last 'x<digits>' suffix
        idx = inner.rfind("x")
        # find the rightmost 'x' that is followed only by digits
        while idx != -1 and not inner[idx + 1 :].isdigit():
            idx = inner.rfind("x", 0, idx)
        if idx == -1:
            raise ValueError(f"malformed vector type {s!r}")
        elem = _parse_type(inner[:idx])
        count = int(inner[idx + 1 :])
        return VectorType(elem, count)
    if s.startswith("ptr<") and s.endswith(">"):
        inner = s[4:-1]
        # pointee may itself contain commas (vec<...>); split at the last comma
        comma = inner.rfind(",")
        if comma == -1:
            raise ValueError(f"malformed pointer type {s!r}")
        pointee = _parse_type(inner[:comma])
        space = inner[comma + 1 :].strip()
        return PtrType(pointee, space)
    if s.startswith("smem<") and s.endswith(">"):
        inner = s[5:-1]
        # form: "<elem>, [d0xd1x...]"
        lb = inner.rfind("[")
        rb = inner.rfind("]")
        if lb == -1 or rb == -1:
            raise ValueError(f"malformed smem type {s!r}")
        # elem is everything before the comma preceding '['
        head = inner[:lb].rstrip()
        if head.endswith(","):
            head = head[:-1].rstrip()
        elem = _parse_type(head)
        shape_str = inner[lb + 1 : rb].strip()
        shape = tuple(int(x) for x in shape_str.split("x")) if shape_str else ()
        return SmemType(elem, shape)
    return Type(s)


# --------------------------------------------------------------------------
# Attr encoding (§5)
# --------------------------------------------------------------------------


def _encode_attr_value(v: Any) -> str:
    # bool MUST be checked before int (bool is an int subclass).
    if isinstance(v, bool):
        return "b:true" if v else "b:false"
    if isinstance(v, int):
        return f"i:{v}"
    if isinstance(v, float):
        return f"f:{repr(v)}"
    if isinstance(v, str):
        return f's:"{_escape(v)}"'
    if isinstance(v, dict):
        return _encode_attr_map(v)
    if isinstance(v, (list, tuple)):
        return "l:[ " + ", ".join(_encode_attr_value(e) for e in v) + " ]"
    raise ValueError(f"cannot serialize attr value of type {type(v).__name__}: {v!r}")


def _encode_attr_map(attrs: Dict[str, Any]) -> str:
    if not attrs:
        return "{ }"
    parts = [f"{k} = {_encode_attr_value(attrs[k])}" for k in sorted(attrs.keys())]
    return "{ " + ", ".join(parts) + " }"


# --- attr parsing (a small recursive-descent scanner) ---


class _Scanner:
    """Character scanner over a single (logical) line for attr-map parsing."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0
        self.n = len(text)

    def skip_ws(self) -> None:
        while self.i < self.n and self.text[self.i] in " \t":
            self.i += 1

    def peek(self) -> str:
        return self.text[self.i] if self.i < self.n else ""

    def expect(self, ch: str) -> None:
        self.skip_ws()
        if self.peek() != ch:
            raise ValueError(f"expected {ch!r} at offset {self.i} in {self.text!r}")
        self.i += 1

    def at_end(self) -> bool:
        self.skip_ws()
        return self.i >= self.n


def _parse_attr_map(sc: _Scanner) -> Dict[str, Any]:
    sc.expect("{")
    attrs: Dict[str, Any] = {}
    sc.skip_ws()
    if sc.peek() == "}":
        sc.i += 1
        return attrs
    while True:
        sc.skip_ws()
        # key: identifier up to ' ='
        start = sc.i
        while sc.i < sc.n and (sc.text[sc.i].isalnum() or sc.text[sc.i] == "_"):
            sc.i += 1
        key = sc.text[start : sc.i]
        if not key:
            raise ValueError(f"empty attr key at offset {sc.i} in {sc.text!r}")
        sc.skip_ws()
        sc.expect("=")
        value = _parse_attr_value(sc)
        attrs[key] = value
        sc.skip_ws()
        ch = sc.peek()
        if ch == ",":
            sc.i += 1
            continue
        if ch == "}":
            sc.i += 1
            break
        raise ValueError(f"expected ',' or '}}' at offset {sc.i} in {sc.text!r}")
    return attrs


def _parse_attr_value(sc: _Scanner) -> Any:
    sc.skip_ws()
    ch = sc.peek()
    if ch == "{":
        return _parse_attr_map(sc)
    # tagged scalars: <tag>:<payload>
    if sc.text[sc.i : sc.i + 2] == "l:":
        sc.i += 2
        sc.expect("[")
        items: List[Any] = []
        sc.skip_ws()
        if sc.peek() == "]":
            sc.i += 1
            return items
        while True:
            items.append(_parse_attr_value(sc))
            sc.skip_ws()
            c = sc.peek()
            if c == ",":
                sc.i += 1
                continue
            if c == "]":
                sc.i += 1
                break
            raise ValueError(f"expected ',' or ']' at offset {sc.i}")
        return items
    if sc.text[sc.i : sc.i + 2] == "b:":
        sc.i += 2
        if sc.text[sc.i : sc.i + 4] == "true":
            sc.i += 4
            return True
        if sc.text[sc.i : sc.i + 5] == "false":
            sc.i += 5
            return False
        raise ValueError(f"malformed bool at offset {sc.i}")
    if sc.text[sc.i : sc.i + 2] == "i:":
        sc.i += 2
        return int(_read_scalar_token(sc))
    if sc.text[sc.i : sc.i + 2] == "f:":
        sc.i += 2
        return float(_read_scalar_token(sc))
    if sc.text[sc.i : sc.i + 2] == "s:":
        sc.i += 2
        sc.expect('"')
        return _read_quoted(sc)
    raise ValueError(f"unknown attr value tag at offset {sc.i} in {sc.text!r}")


def _read_scalar_token(sc: _Scanner) -> str:
    # read until a delimiter (comma, brace, bracket, ws)
    start = sc.i
    while sc.i < sc.n and sc.text[sc.i] not in ", }]\t":
        sc.i += 1
    return sc.text[start : sc.i]


def _read_quoted(sc: _Scanner) -> str:
    # sc is positioned just after the opening quote
    out = []
    while sc.i < sc.n:
        ch = sc.text[sc.i]
        if ch == "\\" and sc.i + 1 < sc.n:
            out.append(sc.text[sc.i : sc.i + 2])
            sc.i += 2
            continue
        if ch == '"':
            sc.i += 1
            return _unescape("".join(out))
        out.append(ch)
        sc.i += 1
    raise ValueError("unterminated string literal")


# --------------------------------------------------------------------------
# Serialize (§3, §4)
# --------------------------------------------------------------------------


def serialize(kernel: KernelDef) -> str:
    """Render ``kernel`` as ``ck.dsl.ir/v1`` text. Deterministic."""
    lines: List[str] = [f"{FORMAT_NAME} {FORMAT_VERSION}"]
    lines.append(f"kernel @{kernel.name} {{")
    if kernel.attrs:
        lines.append(f"{_INDENT}attrs {_encode_attr_map(kernel.attrs)}")
    # params block (always emitted)
    lines.append(f"{_INDENT}params {{")
    for p in kernel.params:
        line = f"{_INDENT * 2}%{p.name} : {_encode_type(p.type)}"
        if p.attrs:
            line += " " + _encode_attr_map(p.attrs)
        lines.append(line)
    lines.append(f"{_INDENT}}}")
    _serialize_region(kernel.body, 1, lines)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _serialize_region(region: Region, depth: int, lines: List[str]) -> None:
    pad = _INDENT * depth
    lines.append(f"{pad}region @{region.label} {{")
    for op in region.ops:
        _serialize_op(op, depth + 1, lines)
    lines.append(f"{pad}}}")


def _serialize_op(op: Op, depth: int, lines: List[str]) -> None:
    pad = _INDENT * depth
    head = pad
    if op.results:
        head += (
            ", ".join(f"{r.name} : {_encode_type(r.type)}" for r in op.results) + " = "
        )
    head += op.name
    head += " ( " + ", ".join(o.name for o in op.operands) + " )"
    if op.attrs:
        head += " " + _encode_attr_map(op.attrs)
    if op.loc is not None:
        head += f' @loc "{_escape(op.loc)}"'
    lines.append(head)
    for region in op.regions:
        _serialize_region(region, depth + 1, lines)


# --------------------------------------------------------------------------
# Parse (§3, §4)
# --------------------------------------------------------------------------


class _Lines:
    def __init__(self, text: str) -> None:
        raw = text.replace("\r\n", "\n").split("\n")
        self.lines = raw
        self.i = 0

    def _is_skippable(self, ln: str) -> bool:
        stripped = ln.strip()
        return stripped == "" or stripped.startswith("#")

    def peek(self) -> Optional[str]:
        while self.i < len(self.lines) and self._is_skippable(self.lines[self.i]):
            self.i += 1
        if self.i >= len(self.lines):
            return None
        return self.lines[self.i].strip()

    def next(self) -> str:
        ln = self.peek()
        if ln is None:
            raise ValueError("unexpected end of input")
        self.i += 1
        return ln


def parse(text: str) -> KernelDef:
    """Parse ``ck.dsl.ir/v1`` text back into a :class:`KernelDef`."""
    src = _Lines(text)
    header = src.next()
    parts = header.split()
    if len(parts) != 2 or parts[0] != FORMAT_NAME:
        raise ValueError(f"bad header {header!r}; expected '{FORMAT_NAME} <ver>'")
    if parts[1] != FORMAT_VERSION:
        raise ValueError(f"unsupported IR format version {parts[1]!r}")

    kline = src.next()
    if not (kline.startswith("kernel @") and kline.endswith("{")):
        raise ValueError(f"bad kernel line {kline!r}")
    name = kline[len("kernel @") : -1].strip()

    # value table: name -> Value (parsed-in order; defining-first)
    values: Dict[str, Value] = {}

    kernel_attrs: Dict[str, Any] = {}
    params: List[Param] = []

    # optional attrs line
    nxt = src.peek()
    if nxt is not None and nxt.startswith("attrs "):
        src.next()
        sc = _Scanner(nxt[len("attrs ") :])
        kernel_attrs = _parse_attr_map(sc)

    # params block
    pline = src.next()
    if pline != "params {":
        raise ValueError(f"expected 'params {{', got {pline!r}")
    while True:
        ln = src.next()
        if ln == "}":
            break
        p = _parse_param(ln)
        params.append(p)
        v = Value(name=f"%{p.name}", type=p.type)
        values[v.name] = v

    body = _parse_region(src, values)

    # closing kernel brace
    closing = src.next()
    if closing != "}":
        raise ValueError(f"expected closing kernel '}}', got {closing!r}")

    kernel = KernelDef(name=name, params=params, body=body, attrs=kernel_attrs)
    return kernel


def _parse_param(ln: str) -> Param:
    # form: %<name> : <type> [<attr-map>]
    if not ln.startswith("%"):
        raise ValueError(f"bad param line {ln!r}")
    # split off attr-map if present (starts at first ' {')
    attr_idx = ln.find("{")
    attrs: Dict[str, Any] = {}
    head = ln
    if attr_idx != -1:
        head = ln[:attr_idx].rstrip()
        sc = _Scanner(ln[attr_idx:])
        attrs = _parse_attr_map(sc)
    colon = head.index(" : ")
    name = head[1:colon].strip()
    type_str = head[colon + 3 :].strip()
    return Param(name=name, type=_parse_type(type_str), attrs=attrs)


def _parse_region(src: _Lines, values: Dict[str, Value]) -> Region:
    rline = src.next()
    if not (rline.startswith("region @") and rline.endswith("{")):
        raise ValueError(f"expected 'region @<label> {{', got {rline!r}")
    label = rline[len("region @") : -1].strip()
    region = Region(label=label)
    while True:
        ln = src.peek()
        if ln is None:
            raise ValueError("unterminated region")
        if ln == "}":
            src.next()
            break
        op = _parse_op(src, values)
        region.ops.append(op)
    return region


def _parse_op(src: _Lines, values: Dict[str, Value]) -> Op:
    ln = src.next()

    # results: "<r0> : <t0>, <r1> : <t1> = "  (before opname). Detect by ' = '
    # appearing before the ' ( ' operand-list opener.
    results: List[Value] = []
    rest = ln
    eq_idx = _find_results_eq(ln)
    if eq_idx != -1:
        res_str = ln[:eq_idx]
        rest = ln[eq_idx + 3 :]  # skip " = "
        for entry in _split_top(res_str, ","):
            entry = entry.strip()
            colon = entry.index(" : ")
            rname = entry[:colon].strip()
            rtype = _parse_type(entry[colon + 3 :].strip())
            results.append(Value(name=rname, type=rtype))

    # rest: "<opname> ( <operands> ) [<attr-map>] [@loc \"...\"]"
    paren = rest.index(" ( ")
    opname = rest[:paren].strip()
    # find matching close ' )' — operands contain only %ids and commas, so the
    # first " )" after the opener closes the list.
    after = rest[paren + 3 :]
    close = after.index(")")
    operand_str = after[:close].strip()
    tail = after[close + 1 :].strip()

    operands: List[Value] = []
    if operand_str:
        for oid in _split_top(operand_str, ","):
            oid = oid.strip()
            if oid not in values:
                raise ValueError(
                    f"operand {oid!r} used before definition in op {opname!r}"
                )
            operands.append(values[oid])

    attrs: Dict[str, Any] = {}
    loc: Optional[str] = None
    if tail:
        # tail may be "<attr-map>", "@loc \"..\"", or both
        if tail.startswith("{"):
            sc = _Scanner(tail)
            attrs = _parse_attr_map(sc)
            tail = tail[sc.i :].strip()
        if tail.startswith("@loc"):
            q = tail.index('"')
            sc = _Scanner(tail[q + 1 :])
            loc = _read_quoted(sc)

    op = Op(
        name=opname,
        operands=operands,
        results=results,
        attrs=attrs,
        regions=[],
        loc=loc,
    )
    for r in results:
        r.op = op
        values[r.name] = r

    # register block-defined SSA values BEFORE parsing nested regions so body
    # operands resolve (scf.for induction var + iter-arg block values).
    region_scope = _register_block_values(op, values)

    # nested regions follow, while next line is a region opener
    while True:
        nxt = src.peek()
        if nxt is not None and nxt.startswith("region @"):
            op.regions.append(_parse_region(src, values))
        else:
            break

    # block-scope values are not visible to siblings of this op; drop them
    for k in region_scope:
        values.pop(k, None)

    return op


def _register_block_values(op: Op, values: Dict[str, Value]) -> List[str]:
    """Reconstruct the SSA values an op's regions introduce (mirrors the
    builder's own ``scf_for`` / ``scf_for_iter`` registration). Returns the
    list of names to retire once the regions are parsed."""
    added: List[str] = []
    if op.name != "scf.for":
        return added
    iv_name = op.attrs.get("iv")
    iv_type = op.attrs.get("iv_type")
    if isinstance(iv_name, str) and isinstance(iv_type, str):
        v = Value(name=iv_name, type=_parse_type(iv_type))
        v.op = op
        values[iv_name] = v
        added.append(iv_name)
    iter_args = op.attrs.get("iter_args")
    if isinstance(iter_args, list):
        for entry in iter_args:
            if isinstance(entry, dict) and "name" in entry and "type" in entry:
                v = Value(name=entry["name"], type=_parse_type(entry["type"]))
                v.op = op
                values[entry["name"]] = v
                added.append(entry["name"])
    return added


def _find_results_eq(ln: str) -> int:
    """Return the index of the ' = ' that separates results from the opname,
    or -1 if the op has no results. The marker is the ' = ' occurring before
    the ' ( ' operand-list opener."""
    paren = ln.find(" ( ")
    eq = ln.find(" = ")
    if eq != -1 and (paren == -1 or eq < paren):
        return eq
    return -1


def _split_top(s: str, sep: str) -> List[str]:
    """Split ``s`` on ``sep`` ignoring separators nested inside <>, [], {},
    "" — defensive for type strings like ``vec<f32x16>`` in result lists."""
    out: List[str] = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "<[{(":
                depth += 1
            elif ch in ">]})":
                depth -= 1
            elif ch == sep and depth == 0:
                out.append(s[start:i])
                start = i + 1
        i += 1
    out.append(s[start:])
    return out


# --------------------------------------------------------------------------
# Canonicalization for semantic diff (see ir_serialization_format.md §7)
# --------------------------------------------------------------------------


def _walk_values_in_def_order(kernel: KernelDef) -> List[str]:
    """SSA ids in first-definition order: params (ABI order), then a pre-order
    walk of ops (results, then any block-defined iv / iter-args, then regions).
    """
    order: List[str] = []
    seen = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            order.append(name)

    for p in kernel.params:
        add(f"%{p.name}")

    def walk_region(region: Region) -> None:
        for op in region.ops:
            for r in op.results:
                add(r.name)
            if op.name == "scf.for":
                iv = op.attrs.get("iv")
                if isinstance(iv, str):
                    add(iv)
                ia = op.attrs.get("iter_args")
                if isinstance(ia, list):
                    for e in ia:
                        if isinstance(e, dict) and "name" in e:
                            add(e["name"])
            for sub in op.regions:
                walk_region(sub)

    walk_region(kernel.body)
    return order


def canonicalize(kernel: KernelDef) -> str:
    """Return a normalized serialization where SSA ids are renamed to ``%0,
    %1, ...`` in first-definition order and ``loc`` is stripped. Two kernels
    that differ only in incidental id choices / authoring locations produce the
    same canonical string."""
    rename = {old: f"%{i}" for i, old in enumerate(_walk_values_in_def_order(kernel))}

    def rt(v: Value) -> Value:
        return Value(name=rename.get(v.name, v.name), type=v.type)

    def rmap_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        # rename ids that appear in iv / iter_args attrs
        out = dict(attrs)
        if "iv" in out and isinstance(out["iv"], str):
            out["iv"] = rename.get(out["iv"], out["iv"])
        if "iter_args" in out and isinstance(out["iter_args"], list):
            new_list = []
            for e in out["iter_args"]:
                if isinstance(e, dict) and "name" in e:
                    e = dict(e)
                    e["name"] = rename.get(e["name"], e["name"])
                new_list.append(e)
            out["iter_args"] = new_list
        return out

    def rop(op: Op) -> Op:
        new = Op(
            name=op.name,
            operands=[rt(o) for o in op.operands],
            results=[rt(r) for r in op.results],
            attrs=rmap_attrs(op.attrs),
            regions=[rregion(r) for r in op.regions],
            loc=None,  # stripped
        )
        return new

    def rregion(region: Region) -> Region:
        return Region(label=region.label, ops=[rop(o) for o in region.ops])

    norm = KernelDef(
        name=kernel.name,
        params=[
            Param(name=rename[f"%{p.name}"][1:], type=p.type, attrs=p.attrs)
            for p in kernel.params
        ],
        body=rregion(kernel.body),
        attrs=kernel.attrs,
    )
    return serialize(norm)


def canonical_equal(a: KernelDef, b: KernelDef) -> bool:
    """True iff ``a`` and ``b`` are semantically equal (equal up to incidental
    SSA-id gaps and authoring locations)."""
    return canonicalize(a) == canonicalize(b)
