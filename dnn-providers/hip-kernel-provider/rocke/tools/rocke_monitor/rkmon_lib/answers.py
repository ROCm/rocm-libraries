# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The reader of the settings files (the deploy and the build settings files) and of the package's own YAML lists:
JSON, or YAML in a small, common subset, read with the standard library only (rkmon runs on the host's own Python,
where nothing is installed, so no YAML library).

The subset: nested mappings by indentation (spaces only), `key: value`, block lists (`- item`, at the key's
indentation or deeper), lists of mappings (`- key: value`, its first key plain or quoted, the item's further keys two
columns in from its dash), flow lists (`[a, b]`), flow mappings of scalars (`{a: 1, b: x}`), and scalars: plain or
quoted strings, integers, floats, true/false, null or ~, with `#` comments. Anchors, aliases, tags, multi-line strings
and nested flow collections are refused with the line number. A double-quoted string takes JSON's escapes, an escaped
quote among them.

`parse(text, plain=True)` (and `loads`, `load`) keep each plain scalar's text instead, as a `Plain` string, so that the
caller decides its type, as a settings file's schema does: `7.10` stays "7.10" where text is expected. A quoted scalar
is text either way, and an empty value is None. In JSON, each number stays its text the same way. `typed(text)` gives
the value these rules give a plain scalar's text.
"""

import json
import pathlib
import re


class AnswersError(ValueError):
    pass


class Plain(str):
    """A plain (unquoted) scalar's text, as written: parse(text, plain=True) leaves its type to the caller."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"Plain({str.__repr__(self)})"


def load(path, plain: bool = False) -> dict:
    return loads(pathlib.Path(path).read_text(encoding="utf-8"), str(path), plain)


def loads(text: str, source: str = "the text", plain: bool = False) -> dict:
    """The answers in `text` (JSON, or YAML in the subset above), a mapping at the top level. `source` names the text
    in a message; `plain` keeps each plain scalar's text (Plain)."""
    if text.lstrip().startswith("{"):
        kept = dict(parse_int=Plain, parse_float=Plain, parse_constant=Plain)
        try:
            data = json.loads(text, **(kept if plain else {}))
        except ValueError as exc:
            raise AnswersError(f"{source}: not valid JSON: {exc}") from exc
    else:
        data = parse(text, plain)
    if not isinstance(data, dict):
        raise AnswersError(
            f"{source}: the top level is not a mapping of keys to values"
        )
    return data


def _strip_comment(line: str) -> str:
    """The line without its comment: a # at the start or after a space, outside quotes (an escaped \\" inside
    double quotes does not end them)."""
    quote, escaped = None, False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
        elif quote:
            if ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def parse(text: str, plain: bool = False):
    lines = []
    for n, raw in enumerate(text.splitlines(), 1):
        s = _strip_comment(raw).rstrip()
        if not s.strip() or s.strip() in ("---", "..."):
            continue
        lead = s[: len(s) - len(s.lstrip())]
        if "\t" in lead:
            raise AnswersError(f"line {n}: indent with spaces, not tabs")
        lines.append((n, len(lead), s.strip()))
    if not lines:
        return {}
    value, i = _block(lines, 0, lines[0][1], plain)
    if i != len(lines):
        raise AnswersError(f"line {lines[i][0]}: unexpected indentation")
    return value


def _is_item(s: str) -> bool:
    return s == "-" or s.startswith("- ")


def _block(lines, i, indent, plain):
    if _is_item(lines[i][2]):
        return _list(lines, i, indent, plain)
    return _map(lines, i, indent, plain)


# A quoted scalar or key: in double quotes, with JSON's escapes, or in single quotes, with '' for a quote.
QUOTED = r""""(?:[^"\\]|\\.)*"|'(?:[^']|'')*'"""
KEY = re.compile(rf"""^({QUOTED}|[^\s:#][^:#]*?)\s*:(?:\s+(.*))?$""")


def _map(lines, i, indent, plain):
    out = {}
    while i < len(lines):
        n, ind, s = lines[i]
        if ind < indent:
            break
        if ind > indent:
            raise AnswersError(f"line {n}: unexpected indentation")
        if _is_item(s):
            raise AnswersError(f"line {n}: a list item where a key was expected")
        m = KEY.match(s)
        if not m:
            raise AnswersError(f"line {n}: expected `key: value`")
        key = _unquote(m.group(1), n) if m.group(1)[:1] in "\"'" else m.group(1).strip()
        if key in out:
            raise AnswersError(f"line {n}: {key!r} is given twice")
        i += 1
        rest = (m.group(2) or "").strip()
        if rest:
            out[key] = _scalar(rest, n, plain)
        elif i < len(lines) and lines[i][1] > indent:
            out[key], i = _block(lines, i, lines[i][1], plain)
        elif i < len(lines) and lines[i][1] == indent and _is_item(lines[i][2]):
            out[key], i = _list(lines, i, indent, plain)
        else:
            out[key] = None
    return out, i


def _list(lines, i, indent, plain):
    out = []
    while i < len(lines):
        n, ind, s = lines[i]
        if ind != indent or not _is_item(s):
            break
        item = s[1:].strip()
        key = KEY.match(item)
        i += 1
        if not item:
            if i < len(lines) and lines[i][1] > indent:
                value, i = _block(lines, i, lines[i][1], plain)
                out.append(value)
            else:
                out.append(None)
        elif key and (item[:1] not in "\"'[{" or re.fullmatch(QUOTED, key.group(1))):
            # A mapping item: its first key on the dash's line, plain or quoted whole (an item that starts with a
            # quote and has no quoted key, such as "a: b", is a quoted scalar), the rest two columns in from the dash.
            lines[i - 1] = (n, indent + 2, item)
            value, i = _map(lines, i - 1, indent + 2, plain)
            out.append(value)
        else:
            out.append(_scalar(item, n, plain))
    return out, i


def _unquote(s: str, n: int) -> str:
    if s[0] == '"':
        if len(s) < 2 or not s.endswith('"'):
            raise AnswersError(f"line {n}: an unclosed quote")
        try:
            return json.loads(s)
        except ValueError as exc:
            raise AnswersError(f"line {n}: a bad double-quoted string") from exc
    if len(s) < 2 or not s.endswith("'"):
        raise AnswersError(f"line {n}: an unclosed quote")
    return s[1:-1].replace("''", "'")


def _split_flow(s: str, n: int) -> list[str]:
    parts, cur, quote, escaped = [], "", None, False
    for ch in s:
        if quote:
            cur += ch
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            cur += ch
        elif ch == ",":
            parts.append(cur.strip())
            cur = ""
        elif ch in "[]{}":
            raise AnswersError(f"line {n}: nested flow collections are not supported")
        else:
            cur += ch
    if quote:
        raise AnswersError(f"line {n}: an unclosed quote")
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _scalar(s: str, n: int, plain: bool = False):
    if s[0] in "\"'":
        return _unquote(s, n)
    if s[0] in "&*!|>%@`":
        raise AnswersError(
            f"line {n}: anchors, aliases, tags and block strings are not supported"
        )
    if s.startswith("["):
        if not s.endswith("]"):
            raise AnswersError(f"line {n}: an unclosed [")
        return [_scalar(p, n, plain) for p in _split_flow(s[1:-1], n)]
    if s.startswith("{"):
        if not s.endswith("}"):
            raise AnswersError(f"line {n}: an unclosed {{")
        out = {}
        for part in _split_flow(s[1:-1], n):
            key, sep, value = part.partition(":")
            if not sep or not key.strip():
                raise AnswersError(f"line {n}: expected `key: value` in {{...}}")
            key = _unquote(key.strip(), n) if key.strip()[0] in "\"'" else key.strip()
            if key in out:
                raise AnswersError(f"line {n}: {key!r} is given twice")
            out[key] = _scalar(value.strip(), n, plain) if value.strip() else None
        return out
    return Plain(s) if plain else typed(s)


def typed(s: str):
    """A plain scalar's value by this subset's rules: true or false, null or ~ (in any case), an integer, a float with
    a point, else the text itself."""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~"):
        return None
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    if re.fullmatch(r"[-+]?\d+\.\d*(?:[eE][-+]?\d+)?", s):
        return float(s)
    return s
