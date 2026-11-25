# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Layout 1's deploy settings schema (schema/layout1/deploy_settings.schema.json), and the helpers that read any of
the package's JSON Schemas. The schema is the one home of each layout 1 deploy key's description, type, allowed values
and default, and the tests' layout 1 deploy template (deploy_tree.template) is made from it (`template`). `node`,
`leaves`, `sections`, `choices` and `validate` read any of the package's schemas, the deploy and the build settings
files' (file_types.DEPLOY, file_types.BUILD) among them.

The schemas use JSON Schema's own words (type, enum, minimum, maximum, pattern, items, properties,
additionalProperties, required, default, title, description) and one of the package's: x-template, the value a
template shows when it is not the default (a placeholder such as <login>).

docker/rkmon runs this on the host: the standard library only, on the host's Python (deployment.HOST_PYTHON).
"""

from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FOLDER = ROOT / "schema" / "layout1"
FILES = {
    "deploy": "deploy_settings.schema.json",
}
_CACHE: dict = {}


def path(name: str, package_dir=None) -> pathlib.Path:
    """A schema's file: NAME is one of FILES; in the package at PACKAGE_DIR, else in this one."""
    if package_dir is None:
        return FOLDER / FILES[name]
    return pathlib.Path(package_dir) / FOLDER.relative_to(ROOT) / FILES[name]


def load(name: str) -> dict:
    """A schema (NAME, one of FILES), read once."""
    if name not in _CACHE:
        _CACHE[name] = json.loads(path(name).read_text(encoding="utf-8"))
    return _CACHE[name]


def is_section(node: dict) -> bool:
    """A node whose keys the schema lists (a section): every other node is a key of its own."""
    return isinstance(node, dict) and "properties" in node


def node(schema: dict, dotted: str) -> dict | None:
    """The node at a dotted key ("" is the schema itself); None when the schema has none."""
    cur = schema
    for part in [p for p in dotted.split(".") if p]:
        props = cur.get("properties") if isinstance(cur, dict) else None
        if not isinstance(props, dict) or part not in props:
            return None
        cur = props[part]
    return cur


def leaves(schema: dict, prefix: str = "") -> list:
    """Every key a settings file may set, in the schema's order, as (dotted key, its node): the keys of each section,
    never a section itself."""
    out = []
    for k, sub in (schema.get("properties") or {}).items():
        dotted = prefix + k
        if is_section(sub):
            out += leaves(sub, dotted + ".")
        else:
            out.append((dotted, sub))
    return out


def sections(schema: dict, prefix: str = "") -> list:
    """Every section, in the schema's order, as (dotted key, its node), parents before their children."""
    out = []
    for k, sub in (schema.get("properties") or {}).items():
        if is_section(sub):
            out.append((prefix + k, sub))
            out += sections(sub, prefix + k + ".")
    return out


def defaults(schema: dict) -> dict:
    """Every key's default, as a nested mapping (the settings a file that sets nothing has)."""
    out = {}
    for k, sub in (schema.get("properties") or {}).items():
        if is_section(sub):
            out[k] = defaults(sub)
        elif "default" in sub:
            out[k] = json.loads(json.dumps(sub["default"]))
    return out


def template_value(sub: dict):
    """What a template shows for a key: its placeholder (x-template), else its default."""
    return sub.get("x-template", sub.get("default"))


def private_keys(schema: dict) -> list:
    """The keys whose values are private (a value that names a site, a person or an internal system), in the schema's
    order: those marked x-class "private".
    """
    return [dotted for dotted, sub in leaves(schema) if sub.get("x-class") == "private"]


def choices(sub: dict) -> str:
    """The allowed values of a key, in words: `a`, `b` or `c`; a range; or "" when any value of its type goes."""
    if "enum" in sub:
        vals = [
            json.dumps(v) if not isinstance(v, str) else (v or '""')
            for v in sub["enum"]
        ]
        return " or ".join([", ".join(vals[:-1]), vals[-1]] if len(vals) > 1 else vals)
    low, high = sub.get("minimum"), sub.get("maximum")
    if low is not None and high is not None:
        return f"{low} to {high}"
    if low is not None:
        return f"{low} or more"
    return ""


TYPE_WORDS = {
    "string": "text",
    "integer": "a whole number",
    "number": "a number",
    "boolean": "true or false",
    "array": "a list",
    "object": "a mapping",
    "null": "null",
}


def type_words(sub: dict) -> str:
    t = sub.get("type")
    names = t if isinstance(t, list) else [t] if t else []
    return " or ".join(TYPE_WORDS.get(x, x) for x in names)


def _is(value, t: str) -> bool:
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    if t == "null":
        return value is None
    return True


def validate(schema: dict, data, prefix: str = "") -> list:
    """What in `data` the schema refuses, as (dotted key, the line that says what is wrong and what it must be); []
    when nothing. A key's line ends with what the key is (its description), so the fix is in the line.
    """
    sub = schema
    where = prefix.rstrip(".")
    t = sub.get("type")
    names = t if isinstance(t, list) else [t] if t else []
    if names and not any(_is(data, x) for x in names):
        return [
            (
                where,
                f"{where} is {data!r}: it must be {type_words(sub)}" + _what(sub),
            )
        ]
    if "enum" in sub and data not in sub["enum"]:
        return [
            (
                where,
                f"{where} is {data!r}: it must be {choices(sub)}" + _what(sub),
            )
        ]
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        low, high = sub.get("minimum"), sub.get("maximum")
        if (low is not None and data < low) or (high is not None and data > high):
            return [
                (where, f"{where} is {data!r}: it must be {choices(sub)}" + _what(sub))
            ]
    if isinstance(data, (str, int)) and not isinstance(data, bool) and "pattern" in sub:
        if not re.search(sub["pattern"], str(data)):
            return [
                (
                    where,
                    f"{where} is {data!r}, which does not fit" + _what(sub),
                )
            ]
    if "oneOf" in sub:
        return _one_of(sub, data, prefix)
    out = []
    if isinstance(data, list) and isinstance(sub.get("items"), dict):
        for i, item in enumerate(data):
            out += validate(sub["items"], item, f"{where}[{i}].")
    if isinstance(data, dict):
        props = sub.get("properties")
        extra = sub.get("additionalProperties", True)
        key_rule = (sub.get("propertyNames") or {}).get("pattern")
        for k in sub.get("required") or ():
            if k not in data:
                out.append(
                    (
                        prefix + k,
                        f"{prefix + k} is missing" + _what((props or {}).get(k, {})),
                    )
                )
        for k, v in data.items():
            key = prefix + str(k)
            if key_rule and not re.search(key_rule, str(k)):
                out.append(
                    (key, f"{key}: {k!r} is no name here (letters, digits, _ . and -)")
                )
            elif isinstance(props, dict) and k in props:
                out += validate(props[k], v, key + ".")
            elif extra is False:
                out.append(
                    (
                        key,
                        f"{key} is no setting"
                        + (f" of {where}" if where else "")
                        + f" (its keys: {', '.join(props or {})})",
                    )
                )
            elif isinstance(extra, dict):
                out += validate(extra, v, key + ".")
    return out


def _one_of(sub: dict, data, prefix: str) -> list:
    """oneOf: the branch whose provider (the discriminator every branch names) the data names, else the one branch
    that fits."""
    where = prefix.rstrip(".")
    branches = sub["oneOf"]
    tags = [
        (b.get("properties") or {}).get("provider", {}).get("enum") for b in branches
    ]
    if all(tags) and isinstance(data, dict):
        for b, tag in zip(branches, tags):
            if data.get("provider") in tag:
                return validate(b, data, prefix)
        names = ", ".join(t for tag in tags for t in tag)
        return [
            (
                f"{where}.provider" if where else "provider",
                f"{where}.provider is {data.get('provider')!r}: it must be one of {names}"
                + _what(sub),
            )
        ]
    fits = [b for b in branches if not validate(b, data, prefix)]
    if len(fits) == 1:
        return []
    return [
        (
            where,
            f"{where} fits {'none' if not fits else 'more than one'} of its forms"
            + _what(sub),
        )
    ]


def _what(sub: dict) -> str:
    d = str((sub or {}).get("description") or "").strip()
    return f" ({d})" if d else ""


# --- the templates and the docs ---------------------------------------------------------------------------------------

WIDTH = (
    118  # a template's lines at most; a longer comment goes on the lines above its key
)
COLUMN = 36  # where a comment beside its key starts


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_scalar(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{}"
    s = str(v)
    if (
        s == ""
        or not re.fullmatch(r"[A-Za-z0-9_./<>:@~-][A-Za-z0-9 _./<>:@~+=-]*", s)
        or s != s.strip()
    ):
        return json.dumps(s)
    if s.lower() in ("yes", "no", "on", "off", "true", "false", "null", "y", "n"):
        return json.dumps(s)
    return s


def _comment_lines(text: str, pad: str) -> list:
    import textwrap

    return [
        f"{pad}# {x}"
        for x in textwrap.wrap(
            text, WIDTH - len(pad) - 2, break_long_words=False, break_on_hyphens=False
        )
    ]


def _keyed(line: str, note: str, pad: str = "") -> list:
    """A key's line with its description: beside it when the line fits, above it otherwise."""
    beside = f"{line:<{COLUMN}}  # {note}"
    if not note:
        return [line]
    if len(beside) <= WIDTH:
        return [beside]
    return _comment_lines(note, pad) + [line]


def header_keys(schema: dict) -> list:
    """The lines of a settings file's headers (the schema's required keys: kind, schema_version), each with its
    value and its description, after a blank line."""
    props = schema.get("properties") or {}
    out = [""]
    for k in schema.get("required") or ():
        out += _keyed(
            f"{k}: {_scalar(props[k].get('default'))}",
            str(props[k].get("description") or ""),
        )
    return out


def template(
    schema: dict, header: list, values: dict | None = None, skip: tuple = ()
) -> str:
    """A settings template: `header` (comment lines, a list of strings, first of them the schema line), then every key
    in the schema's order with its template value (`values`: dotted key -> a value to show instead) and its
    description as a comment: beside its key when the line fits, above it otherwise; each section after a comment
    with its title and description. `skip`: top-level keys left out (the settings file's other parts).
    """
    values = values or {}
    out = list(header)

    def walk(sub: dict, prefix: str, indent: int) -> None:
        pad = " " * indent
        for k, child in (sub.get("properties") or {}).items():
            dotted = prefix + k
            if not prefix and k in skip:
                continue
            if is_section(child):
                if not prefix:
                    out.append("")
                title = child.get("title") or k
                desc = str(child.get("description") or "").rstrip(".")
                out.extend(_comment_lines(f"{title}: {desc}." if desc else title, pad))
                out.append(f"{pad}{k}:")
                walk(child, dotted + ".", indent + 2)
                continue
            if not prefix:
                out.append("")
            note = str(child.get("description") or "")
            if child.get(
                "x-template-lines"
            ):  # a mapping of your own entries: its examples, as comments
                title = child.get("title") or k
                out.extend(_comment_lines(f"{title}: {note}", pad))
                out.append(f"{pad}{k}:")
                out.extend(f"{pad}  {x}".rstrip() for x in child["x-template-lines"])
                continue
            out.extend(
                _keyed(
                    f"{pad}{k}: {_scalar(values.get(dotted, template_value(child)))}",
                    note,
                    pad,
                )
            )

    walk(schema, "", 0)
    return "\n".join(out).rstrip("\n") + "\n"


def template_keys(text: str) -> list:
    """Each key a settings template sets, as (dotted key, its comment: beside it, else the comment lines right above
    it); a section's own key is left out (its keys are listed)."""
    return [(dotted, comment) for _n, dotted, comment in template_lines(text)]


def template_lines(text: str) -> list:
    """`template_keys` with each key's line number first: (line number, dotted key, its comment)."""
    key_line = re.compile(r"^( *)([A-Za-z0-9_][A-Za-z0-9_.-]*):(.*)$")
    out, stack, above = [], [], []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("#"):
            above.append(s[1:].strip())
            continue
        m = key_line.match(line)
        if not m:
            above = []
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        value, comment = _split_comment(rest)
        if not value:
            stack.append((indent, key))
            above = []
            continue
        dotted = ".".join([k for _i, k in stack] + [key])
        out.append((n, dotted, comment or " ".join(above)))
        above = []
    return out


def _split_comment(rest: str) -> tuple:
    """(the value, the comment) of what follows a key's colon: the comment starts at a # outside quotes."""
    quote = None
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1] in " \t"):
            return rest[:i].strip(), rest[i + 1 :].strip()
    return rest.strip(), ""
