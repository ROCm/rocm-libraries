# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Settings files: a whole setup in one file, deploy or build. `header_line` recognizes one by its header, its kind
(file_types.KINDS) or an older layout's first key (file_types.OLD_HEADERS), whatever the file's name: the privacy check
refuses such a file anywhere in the package (rkmon_lib/privacy.py), and the package's .gitignore ignores the name the
README gives them (rkmon-settings*). A deploy settings file is read through rkmon_lib/settings_file.py.
"""

from __future__ import annotations

import json
import re

from . import answers, file_types

# The headers of a settings file of either kind, deploy or build (header_line): file_types.KINDS and
# file_types.OLD_HEADERS.
JSON_HEADER = re.compile(
    r'\s*\{\s*"(' + "|".join(file_types.OLD_HEADERS) + r'|kind)"\s*:'
)
YAML_HEADER = re.compile(
    r"((?:"
    + "|".join(file_types.OLD_HEADERS)
    + r")\s*:|kind\s*:\s*['\"]?(?:"
    + "|".join(file_types.KINDS)
    + r")\b)"
)


def header_line(text: str) -> int:
    """The line of a settings file's header (1-based), or 0 when the text is no settings file of either kind, deploy or
    build. The header is `kind: deploy_settings` or `kind: build_settings`, or an older layout's `rkmon_settings: 1` or
    `rkmon_build_settings: 1`, at the top level: in a JSON object, its first key or any other; in YAML, its first line
    after the comments, or any other top-level key when the text reads as YAML in the subset the settings files are
    read in (rkmon_lib/answers.py). A text that subset cannot read is no settings file, so a document that shows the
    header in an example, or code that names it, is none.
    """
    text = text.lstrip("\ufeff")
    if text.lstrip().startswith("{"):
        m = JSON_HEADER.match(text)
        if m:
            return text[: m.end()].count("\n") + 1
        try:
            data = json.loads(text)
        except ValueError:
            return 0
        return 1 if _header_key(data) else 0
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#") or s == "---":
            continue
        if YAML_HEADER.match(s):
            return i
        break
    if not any(h in text for h in file_types.KINDS + file_types.OLD_HEADERS):
        return 0  # no header word: nothing to read
    try:
        key = _header_key(answers.parse(text))
    except ValueError:
        return 0
    return _top_level_line(text, key) if key else 0


def _header_key(data) -> str:
    """The top-level key that makes `data` a settings file of either kind (an older layout's header, or kind), or ""."""
    if not isinstance(data, dict):
        return ""
    old = [h for h in file_types.OLD_HEADERS if h in data]
    if old:
        return old[0]
    return "kind" if data.get("kind") in file_types.KINDS else ""


def _top_level_line(text: str, key: str) -> int:
    """The line (1-based) of `key` among the top-level keys of YAML text that subset reads (rkmon_lib/answers.py): the
    lines at the first key's indentation."""
    indent = None
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#") or s in ("---", "..."):
            continue
        lead = len(line) - len(line.lstrip(" "))
        indent = lead if indent is None else indent
        if lead == indent and re.match(r"""(['"]?)""" + re.escape(key) + r"\1\s*:", s):
            return i
    return 1
