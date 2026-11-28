# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Whose container a name belongs to, and whose a volume is (check 15). It sits under every docker
call: on OTHER, a command refuses (state 1), and a probe answers "unknown".

- NONE, OURS, OTHER: the three answers.
- `identity(project, here, answers)`: `project` is the compose project, which is `container.name`; `here` is the folder
  this package runs docker compose from; `answers` is what docker said about its containers, as `rows` reads it, or
  those rows already read. A container is the project's when its label com.docker.compose.project is `project`. OURS:
  each of the project's containers has the working-folder label (com.docker.compose.project.working_dir) of `here`,
  the same folder on disk; OTHER: one of them has another folder's, or none; NONE: no container is the project's.
  Other containers are passed over.
- `volume_owner(project, here, volume, users)`: whose a volume is that the deployment of `project`, run from `here`,
  would use: `volume` is its row as `docker volume ls --format json` gives it, None when there is no such volume
  (NONE); `users` are the containers that mount it, running or not, as `rows` reads them. OTHER: its compose project
  label names another project, or a container of another compose project mounts it, or one of `project`'s from
  another folder; a container of no compose project is passed over. OURS otherwise: a volume has no folder's
  label, so one that no other deployment's container mounts reads as this deployment's, one a removed deployment
  left behind included.
- `question(project)`: docker's arguments that ask for those answers: every container of the compose project
  `project`, running or not, one JSON object on each line (`docker ps --all --filter ... --format '{{json .}}'`).
- `exact_name(name)`: the value of docker's name filter (`--filter name=...`) that lists the object named `name`
  alone: docker reads that filter as a pattern that may match any part of a name, so it is anchored at both ends, and
  each character a pattern reads as more than itself is escaped.
- `rows(answers)`: docker's answer read: its JSON, one value on each line that starts with [ or { (as `docker ps
  --all --format json` prints it); else one value, read from the first such line on (docker inspect's indented
  array). Lines that are not JSON (docker's warnings) are passed over, and empty text is no rows. Text that is not
  empty and holds no JSON value, or more JSON after that one value, is a ValueError (UNREADABLE) that quotes none of
  docker's text: whose the containers are is then unknown, never NONE.
- A row's labels are its "Labels", as docker ps gives them ("key=value,key=value"), or as docker inspect gives them
  (a mapping, or under "Config"). The answer is one of the three words, and never says which folder another
  container is from: no message prints another folder's path.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import json
import os
import re

NONE = "NONE"
OURS = "OURS"
OTHER = "OTHER"

PROJECT = "com.docker.compose.project"
WORKING_FOLDER = "com.docker.compose.project.working_dir"
# A comma that starts the next key=value of docker ps's labels.
NEXT_LABEL = re.compile(r",(?=[A-Za-z0-9_.\-/]+=)")
# A line that starts a JSON value, an object or an array, after any indent.
JSON_START = re.compile(r"^[ \t]*[\[{]", re.M)
UNREADABLE = (
    "docker's answer about its containers is neither JSON lines nor one JSON value, "
    "so whose they are is unknown"
)


def _line(text: str, start: int) -> str:
    end = text.find("\n", start)
    return text[start:] if end < 0 else text[start:end]


def _values(text: str) -> list:
    """The JSON values in docker's TEXT: one on each line that starts one, else one from the first such line on."""
    starts = [m.end() - 1 for m in JSON_START.finditer(text)]
    if not starts:
        if text.strip():
            raise ValueError(UNREADABLE)
        return []
    try:
        return [json.loads(_line(text, s)) for s in starts]
    except ValueError:
        pass
    try:
        value, end = json.JSONDecoder().raw_decode(text, starts[0])
    except ValueError:
        raise ValueError(UNREADABLE) from None
    if starts[-1] >= end:
        raise ValueError(UNREADABLE)
    return [value]


def rows(answers) -> list:
    """docker's answers as rows: see the module's docstring; rows already read, as they are."""
    if isinstance(answers, bytes):
        answers = answers.decode("utf-8", "replace")
    if not isinstance(answers, str):
        return [r for r in (answers or ()) if isinstance(r, dict)]
    out = []
    for data in _values(answers):
        out += [
            r
            for r in (data if isinstance(data, list) else [data])
            if isinstance(r, dict)
        ]
    return out


def labels(row: dict) -> dict:
    """A container's labels, from docker ps's "key=value,..." text or docker inspect's mapping."""
    found = row.get("Labels")
    if found is None:
        found = (row.get("Config") or {}).get("Labels")
    if isinstance(found, dict):
        return {str(k): "" if v is None else str(v) for k, v in found.items()}
    out = {}
    for part in NEXT_LABEL.split(str(found or "")):
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip()] = value
    return out


def _same_folder(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(
        os.path.realpath(b)
    )


def question(project: str) -> tuple:
    """docker's arguments that ask for the answers `identity` reads about PROJECT: see the module's docstring."""
    return (
        "ps",
        "--all",
        "--filter",
        f"label={PROJECT}={project}",
        "--format",
        "{{json .}}",
    )


def exact_name(name: str) -> str:
    """The value of docker's name filter that lists NAME alone: see the module's docstring."""
    return "^" + re.sub(r"([\\.+*?()|\[\]{}^$])", r"\\\1", name) + "$"


def volume_owner(project: str, here: str, volume, users) -> str:
    """NONE, OURS or OTHER for a volume: see the module's docstring."""
    if volume is None:
        return NONE
    made_by = labels(volume).get(PROJECT)
    if made_by and made_by != project:
        return OTHER
    for ls in (labels(r) for r in rows(users)):
        theirs = ls.get(PROJECT)
        if theirs and not (
            theirs == project
            and ls.get(WORKING_FOLDER)
            and _same_folder(ls[WORKING_FOLDER], here)
        ):
            return OTHER
    return OURS


def identity(project: str, here: str, answers) -> str:
    """NONE, OURS or OTHER: see the module's docstring."""
    theirs = [labels(r) for r in rows(answers)]
    mine = [ls for ls in theirs if ls.get(PROJECT) == project]
    if not mine:
        return NONE
    if all(
        ls.get(WORKING_FOLDER) and _same_folder(ls[WORKING_FOLDER], here) for ls in mine
    ):
        return OURS
    return OTHER
