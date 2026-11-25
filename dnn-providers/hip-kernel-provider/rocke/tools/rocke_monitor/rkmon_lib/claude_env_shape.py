# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The older secret claude_env: an older package kept the Claude session's settings there, a JSON object of
the session's variables with the key among them. This is its one reader. It gives the file's shape, its names and
settings, never the key: the translation reads it (tools/translate_production.py), check 16 of `rkmon confirm` compares
it with integrations.claude_code (rkmon_lib/confirm.py, in a one-off container of the image), and root's one-time move
copies the key alone to the claude login (docker/rk_root.py).

- CLAUDE_VARS: the variables the file may set, in their order: Claude Code's own, which a container's Claude session
  took from it, and MODEL_VARS, the four that name a model, which the session takes no more (each call names its model
  by its id). KEY_VARS: the two that hold the key.
- `read(path, looks_secret=None) -> Shape`: the shape of the file at PATH; an empty Shape when there is no such file.
  A variable set to blank space alone is not set. ValueError for a file it refuses: a link, anything but a regular
  file, one larger than LIMIT_BYTES or not UTF-8 text, not a JSON object of strings, or one that names a variable but
  CLAUDE_VARS. LOOKS_SECRET, when given, tells a credential: a function that says what a text looks like when it looks
  like a secret, else "" (rkmon_lib/settings_file.py, secret_shape). It judges each custom header's name and the
  address, and a variable the session does not take is named only when it judges the name none. Without it, such a
  variable is only counted, and the address is taken to hold a credential only when the key is in it.
- `read_key(path) -> (Shape, key)`: `read`'s shape, with no judgement, and the key: ANTHROPIC_API_KEY's, else
  ANTHROPIC_AUTH_TOKEN's, blank space around it aside ("" for none). Only the one-time move calls it.
- `settings(shape) -> (made, cannot)`: integrations.claude_code's settings made from SHAPE: MADE,
  {base_url, api_key_sent_as, api_key_header, api_key_scheme}; CANNOT, a Cannot for each thing the shape has that no
  setting carries ([] when the settings carry it all). A shape with no key gets the defaults: there is no key to send.
- `Cannot(found, unnamed, fix)`: FOUND, what was found, one sentence, which may name a custom header; UNNAMED, the same
  with no header's name in it, only variables' names and counts; FIX, its parts in order, each a text or an Edit of one
  of the file's variables (`Edit(variable, words)`: WORDS say what to set it to, "" to take it out).
- `move(shape) -> str`: why the one-time move copies no key from a file of SHAPE, the shape `read` gives with no
  judgement, as root reads it: the deploy settings cannot carry everything in it (each Cannot's UNNAMED); else
  NO_KEY, when it has no key; "" when the move copies the key. Root's move says it, and check 16 shows it.
- `to_json(shape)`, `from_json(data, looks_secret=None)`: a Shape as one JSON object (check 16's one-off container
  prints it) and back. from_json takes only what `read` makes, else ValueError; with LOOKS_SECRET, a header's name or
  an address that looks like a secret is a ValueError too.

Standard library only, and Python 3.9 too: docker/rkmon reads it on the host, the translation runs it in production's
own image, and root loads it by its path from the image's own code, with rkmon_lib/custom_headers.py beside it.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
from typing import NamedTuple

# This code's own folder: the package's, or the image's own copy of it (rkmon_lib/custom_headers.py is there too).
CODE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# The variables that name models: a default model, and a gateway's own name for each built-in model.
MODEL_VARS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)
CLAUDE_VARS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    # a gateway's bearer token, in place of the API key
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    *MODEL_VARS,
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
)
KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# The largest file it reads: the settings are a few hundred bytes.
LIMIT_BYTES = 64 * 1024
# What the shape says of an address with the key in it, and of a nonessential-traffic switch it does not show.
KEY_IN_ADDRESS = "an address with the key in it"
ANOTHER_VALUE = "another value"
# Why the one-time move copies no key from a file that holds none (`move`).
NO_KEY = "it has no key"
# A header's name as integrations.claude_code.api_key_header takes it; a word before the key, such as Bearer; the values
# of CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC the shape shows as they are; a variable's name a message may show; and
# what the shape may say an address looks like.
_HEADER_NAME = re.compile(r"[A-Za-z0-9-]+")
_SCHEME = re.compile(r"[A-Za-z][A-Za-z-]{0,19}")
_SWITCH = ("0", "1", "true", "false")
_VARIABLE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_LOOKS_LIKE = re.compile(r"[A-Za-z][A-Za-z ]{0,79}")
# A file is read without following a link, and without waiting on a pipe; what open says of a name that is no file it
# opens: a socket, or a device with no driver.
_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_NOT_A_FILE = (errno.ENXIO, errno.ENODEV)


class Header(NamedTuple):
    """One custom header of the file, without its value: its name ("" for a line that does not read as a header: a
    name, a colon and a value), whether its value is the key the file holds, alone or after one word, and that word (""
    for none)."""

    name: str
    value_is_key: bool
    word: str


class Shape(NamedTuple):
    """The file's names and settings, never a credential:

    - `variables`: the CLAUDE_VARS it sets, in that order; `models`, those of them that are MODEL_VARS;
    - `headers`: a Header for each line of ANTHROPIC_CUSTOM_HEADERS, in order;
    - `keys_differ`: ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are two different keys;
    - `base_url`: ANTHROPIC_BASE_URL ("" when it is not set, or has a credential in it); `base_url_credential`: what
      the address looks like when it has one (`read`'s LOOKS_SECRET, or KEY_IN_ADDRESS), else "";
    - `nonessential_traffic`: CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC, "" when it is not set, ANOTHER_VALUE for
      anything but 0, 1, true and false."""

    variables: tuple = ()
    headers: tuple = ()
    keys_differ: bool = False
    base_url: str = ""
    base_url_credential: str = ""
    nonessential_traffic: str = ""

    @property
    def models(self) -> tuple:
        return tuple(v for v in self.variables if v in MODEL_VARS)


class Edit(NamedTuple):
    """A step of a fix: set the file's VARIABLE to what WORDS say; "" takes it out."""

    variable: str
    words: str = ""


class Cannot(NamedTuple):
    """What a shape has that no setting carries: see the module's docstring."""

    found: str
    unnamed: str
    fix: tuple


# --- the read ---------------------------------------------------------------------------------------------------------


def _text(path) -> str | None:
    """The file at PATH as text, read without following a link; None when there is no such file. ValueError for a link,
    anything but a regular file, one larger than LIMIT_BYTES, or one that is not UTF-8 text.
    """
    where = os.fspath(path)
    try:
        if stat.S_ISLNK(os.lstat(where).st_mode):
            raise ValueError(f"{where} is a link, not a file")
        fd = os.open(where, _FLAGS)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in _NOT_A_FILE:
            raise ValueError(f"{where} is not a file") from None
        if exc.errno != errno.ELOOP:
            raise
        raise ValueError(f"{where} is a link, not a file") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{where} is not a file")
        data = b""
        while len(data) <= LIMIT_BYTES:
            more = os.read(fd, LIMIT_BYTES + 1 - len(data))
            if not more:
                break
            data += more
    finally:
        os.close(fd)
    if len(data) > LIMIT_BYTES:
        raise ValueError(f"{where} is larger than {LIMIT_BYTES // 1024} KiB")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{where} is not UTF-8 text") from None


def _not_taken(names: list, looks_secret) -> str:
    """NAMES, variables the session does not take, in words: each by its name when LOOKS_SECRET judges it none and it
    reads as a variable's name; the others counted."""
    shown = [
        n
        for n in names
        if looks_secret is not None and _VARIABLE.fullmatch(n) and not looks_secret(n)
    ]
    rest = len(names) - len(shown)
    if not shown:
        return f"{rest} variable{'s' if rest > 1 else ''} the session does not take"
    more = f" and {rest} more" if rest else ""
    return f"variables the session does not take: {', '.join(shown)}{more}"


def _variables(path, looks_secret) -> dict:
    """The file at PATH: {variable: value} for each of the CLAUDE_VARS it sets to something other than blank space ({}
    when there is no such file); ValueError for a file `read` refuses."""
    text = _text(path)
    if text is None:
        return {}
    where = os.fspath(path)
    try:
        data = json.loads(text)
    except RecursionError:
        raise ValueError(f"{where} is not JSON: it nests too deep") from None
    except ValueError as exc:
        why = getattr(exc, "msg", "") or "it cannot be read"
        raise ValueError(f"{where} is not JSON: {why}") from None
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise ValueError(f"{where} must be a JSON object of strings")
    odd = sorted(k for k in data if k not in CLAUDE_VARS)
    if odd:
        raise ValueError(f"{where} names {_not_taken(odd, looks_secret)}")
    return {k: v for k, v in data.items() if v.strip()}


def _header_lines(text: str) -> list:
    """TEXT's headers, by rkmon_lib/custom_headers.py's header_lines, loaded by its path from this code's own folder."""
    import runpy

    path = os.path.join(CODE, "rkmon_lib", "custom_headers.py")
    return runpy.run_path(path)["header_lines"](text)


def _header(name: str, value: str, keys: set, looks_secret) -> Header:
    """A Header that holds no credential: a name or a word that is not a header's, or that holds a key, or a name
    LOOKS_SECRET judges a secret, gives a line that does not read as a header."""
    word, _, rest = value.partition(" ")
    word, rest = (word, rest.strip()) if rest.strip() in keys else ("", value)
    if (
        not _HEADER_NAME.fullmatch(name)
        or (looks_secret is not None and looks_secret(name))
        or (word and not _SCHEME.fullmatch(word))
        or any(k in name or k in word for k in keys)
    ):
        return Header("", False, "")
    return Header(name, rest in keys, word)


def _shape(env: dict, looks_secret) -> Shape:
    """The shape of the variables ENV, judged by LOOKS_SECRET (None for no judgement)."""
    keys = {env[k] for k in KEY_VARS if k in env}
    base = env.get("ANTHROPIC_BASE_URL", "")
    judged = looks_secret(base) if looks_secret is not None and base else ""
    credential = judged or (KEY_IN_ADDRESS if any(k in base for k in keys) else "")
    text = env.get("ANTHROPIC_CUSTOM_HEADERS", "")
    switch = env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "")
    return Shape(
        variables=tuple(v for v in CLAUDE_VARS if v in env),
        headers=tuple(
            _header(n, v, keys, looks_secret)
            for n, v in (_header_lines(text) if text else [])
        ),
        keys_differ=len(keys) > 1,
        base_url="" if credential else base,
        base_url_credential=credential,
        nonessential_traffic=(
            switch if switch.lower() in _SWITCH or not switch else ANOTHER_VALUE
        ),
    )


def read(path, looks_secret=None) -> Shape:
    """The shape of the file at PATH: see the module's docstring."""
    return _shape(_variables(path, looks_secret), looks_secret)


def read_key(path) -> tuple:
    """(the shape of the file at PATH, with no judgement; its key): see the module's docstring."""
    env = _variables(path, None)
    key = next((env[k].strip() for k in KEY_VARS if k in env), "")
    return _shape(env, None), key


# --- the settings it makes --------------------------------------------------------------------------------------------


def settings(shape: Shape) -> tuple:
    """integrations.claude_code's settings made from SHAPE, and what no setting carries: see the module's docstring."""
    api = "ANTHROPIC_API_KEY" in shape.variables
    bearer = "ANTHROPIC_AUTH_TOKEN" in shape.variables
    made = {
        "base_url": shape.base_url,
        "api_key_sent_as": "header",
        "api_key_header": "",
        "api_key_scheme": "",
    }
    cannot = []

    def row(found: str, fix: tuple, unnamed: str = "") -> None:
        cannot.append(Cannot(found, unnamed or found, tuple(fix)))

    def headers(words: str) -> tuple:
        return (Edit("ANTHROPIC_CUSTOM_HEADERS", words),)

    one_key = (
        Edit("ANTHROPIC_API_KEY"),
        ", or",
        Edit("ANTHROPIC_AUTH_TOKEN"),
        ", whichever the gateway does not take",
    )
    if shape.base_url_credential:
        row(
            f"ANTHROPIC_BASE_URL looks like {shape.base_url_credential}, and a settings file has no credential",
            (
                Edit(
                    "ANTHROPIC_BASE_URL",
                    "the gateway's address, with no credential in it",
                ),
            ),
        )
    if shape.keys_differ:
        row(
            "ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are two different keys, and the claude login is one key",
            one_key,
        )
    bad = sum(1 for h in shape.headers if not h.name)
    if bad:
        row(
            f"ANTHROPIC_CUSTOM_HEADERS has {bad} line{'s' if bad > 1 else ''} that "
            f"{'do' if bad > 1 else 'does'} not read as a header: a name, a colon and a value",
            headers("one header a line: a name, a colon and a value"),
        )
    for h in shape.headers:
        if h.name and not h.value_is_key:
            row(
                f"the custom header {h.name} sends a value that is not the key, and no setting sends a header of "
                "fixed text",
                headers(
                    f"its lines without the header {h.name}, when the gateway takes requests without it"
                ),
                "a custom header sends a value that is not the key, and no setting sends a header of fixed text",
            )
    if len(shape.headers) > 1:
        row(
            f"ANTHROPIC_CUSTOM_HEADERS has {len(shape.headers)} headers, and the settings send the key in one header "
            "at most",
            headers(
                "the one header that sends the key, when the gateway takes requests without the others"
            ),
        )
    if len(shape.headers) == 1 and shape.headers[0].value_is_key:
        if api and bearer:
            row(
                "ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN and a custom header all send the key, and the settings send "
                "it in two ways at most",
                one_key,
            )
        elif api or bearer:
            made["api_key_sent_as"] = "header" if api else "header_and_bearer"
            made["api_key_header"] = shape.headers[0].name
            made["api_key_scheme"] = shape.headers[0].word
    elif not shape.headers and bearer:
        made["api_key_sent_as"] = "header_and_bearer" if api else "bearer"
    return made, cannot


def move(shape: Shape) -> str:
    """Why the one-time move copies no key from a file of SHAPE, "" when it copies it: see the module's docstring."""
    cannot = settings(shape)[1]
    if cannot:
        return "the deploy settings cannot carry everything in it: " + "; ".join(
            c.unnamed for c in cannot
        )
    return "" if any(v in shape.variables for v in KEY_VARS) else NO_KEY


# --- as JSON ----------------------------------------------------------------------------------------------------------


def to_json(shape: Shape) -> dict:
    """SHAPE as one JSON object: see the module's docstring."""
    return {
        "variables": list(shape.variables),
        "headers": [dict(h._asdict()) for h in shape.headers],
        "keys_differ": shape.keys_differ,
        "base_url": shape.base_url,
        "base_url_credential": shape.base_url_credential,
        "nonessential_traffic": shape.nonessential_traffic,
    }


def _header_from(data, looks_secret) -> Header:
    if not isinstance(data, dict) or set(data) != set(Header._fields):
        raise ValueError("a shape's header is a JSON object of its fields")
    name, is_key, word = data["name"], data["value_is_key"], data["word"]
    if not (
        isinstance(name, str) and isinstance(word, str) and isinstance(is_key, bool)
    ):
        raise ValueError(
            "a shape's header has a name, a word and whether it is the key"
        )
    if not name:
        if is_key or word:
            raise ValueError("a line that does not read as a header has no word")
    elif not _HEADER_NAME.fullmatch(name) or (
        looks_secret is not None and looks_secret(name)
    ):
        raise ValueError("a shape's header has a name no header has")
    if word and not _SCHEME.fullmatch(word):
        raise ValueError("a shape's header has a word no header takes before a key")
    return Header(name, is_key, word)


def from_json(data, looks_secret=None) -> Shape:
    """The Shape that DATA, one `to_json` made, gives: see the module's docstring."""
    if not isinstance(data, dict) or set(data) != set(Shape._fields):
        raise ValueError("a shape is a JSON object of its fields")
    variables = data["variables"]
    if not isinstance(variables, list) or variables != [
        v for v in CLAUDE_VARS if v in variables
    ]:
        raise ValueError("a shape's variables are the session's, each once, in order")
    if not isinstance(data["headers"], list):
        raise ValueError("a shape's headers are a list")
    headers = tuple(_header_from(h, looks_secret) for h in data["headers"])
    if not isinstance(data["keys_differ"], bool):
        raise ValueError("a shape says whether its keys differ")
    base, credential = data["base_url"], data["base_url_credential"]
    if not isinstance(base, str) or not isinstance(credential, str):
        raise ValueError("a shape's address is a text")
    if credential and (base or not _LOOKS_LIKE.fullmatch(credential)):
        raise ValueError("a shape's address with a credential is not shown")
    if base and (
        len(base.splitlines()) != 1 or (looks_secret is not None and looks_secret(base))
    ):
        raise ValueError("a shape's address carries no credential")
    switch = data["nonessential_traffic"]
    if not isinstance(switch, str) or not (
        switch in ("", ANOTHER_VALUE) or switch.lower() in _SWITCH
    ):
        raise ValueError("a shape's nonessential-traffic switch is one it shows")
    return Shape(
        tuple(variables),
        headers,
        data["keys_differ"],
        base,
        credential,
        switch,
    )
