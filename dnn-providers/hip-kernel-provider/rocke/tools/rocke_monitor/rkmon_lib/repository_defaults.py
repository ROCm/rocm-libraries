# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The repository defaults: what belongs to one repository rather than to the monitor, such as its CI
job names, its test tiers, its hot spots and its tests' wording, as values of package keys, for the folders of that
repository they were written for. The package ships its home repository's alone.

FOLDER/<org>__<repo>/FILE holds the defaults of the repository <org>/<repo>: {"folders": [each folder of the
repository they were written for], "values": {package keys, nested under the configuration's names, as
config.defaults.json holds them}}. They apply when the configuration's repository is <org>/<repo>, matched without
regard to case as GitHub matches it, and its folder is one of their folders, matched without a slash at either end.
Any other repository or folder takes the package's generic values, config.defaults.json's, and rk doctor and rkmon
check warn, naming what is generic.

- `Found(path, repository, folders, values)`: one repository's defaults: PATH, their file; REPOSITORY, <org>/<repo>
  as their folder names it; FOLDERS, the folders they were written for; VALUES, their values, nested.
- `find(repository, folder) -> Found | None`: the defaults that apply to REPOSITORY and its FOLDER; None when none do.
  A file that `problems` refuses applies to no folder.
- `home() -> Found | None`: the defaults the package ships, its home repository's (the first by name, were there
  more). None when it ships none that `problems` accepts.
- `keys() -> list[str]`: every package key the package's repository defaults set, dotted, in the registry's order:
  the keys a repository without defaults takes generic.
- `keys_of(values) -> list[str]`: the package keys VALUES, nested as a Found's values, set, dotted.
- `own(value, generic) -> bool`: whether a configuration's VALUE of such a key is a value of its own: neither blank
  nor GENERIC, the package's generic value. A value that is not its own takes the repository defaults' value.
- `generic(repository, folder, cfg=None) -> list[str]`: the keys of `keys` that take the package's generic values for
  REPOSITORY and its FOLDER: none when repository defaults apply to them; else each one, but those whose value in CFG,
  the running configuration, is its own (`own`). CFG None: the keys alone.
- `generic_found(repository, folder, keys) -> str`: the words of the WARN that names them.
- `problems() -> list[str]`: what is wrong with the package's repository defaults, one line each: a file that cannot
  be read or is not of the form above, and a value that is no package key (repository defaults set package keys
  only).

Standard library only: rkmon check runs it on the host.
"""

from __future__ import annotations

import json
import pathlib
from typing import NamedTuple

from . import registry, wrapping
from .paths import PACKAGE

FOLDER = PACKAGE / "repository"
FILE = "defaults.json"
SEPARATOR = "__"  # between the organization and the repository in a folder's name


class Found(NamedTuple):
    path: pathlib.Path
    repository: str
    folders: tuple
    values: dict


def _folder(folder) -> str:
    return str(folder or "").strip().strip("/")


def _folders() -> list:
    """Each repository's folder under FOLDER, by name."""
    try:
        return sorted(p for p in FOLDER.iterdir() if p.is_dir() and SEPARATOR in p.name)
    except OSError:
        return []


def _read(folder: pathlib.Path) -> tuple:
    """(Found, None) for the defaults in FOLDER, or (None, what is wrong with them)."""
    path = folder / FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"{folder} has no {FILE}"
    except (OSError, ValueError) as exc:
        return None, f"{path} cannot be read: {exc}"
    if not isinstance(data, dict) or set(data) != {"folders", "values"}:
        return (
            None,
            f'{path} is not of the form {{"folders": [...], "values": {{...}}}}',
        )
    folders = data["folders"]
    if not (
        isinstance(folders, list)
        and folders
        and all(isinstance(f, str) and _folder(f) for f in folders)
    ):
        return (
            None,
            f"{path}: folders must list the repository's folders they were written for",
        )
    if not isinstance(data["values"], dict):
        return None, f"{path}: values must hold package keys, nested"
    repository = folder.name.replace(SEPARATOR, "/", 1)
    return (
        Found(path, repository, tuple(_folder(f) for f in folders), data["values"]),
        None,
    )


def _package_keys() -> set:
    return {k.name for k in registry.keys() if k.owner == registry.PACKAGE}


def keys_of(values) -> list:
    """The package keys VALUES, nested, set: dotted, in VALUES' order."""
    package = _package_keys()
    return [name for name in registry.leaves(values, package) if name in package]


def _stray(found: Found) -> list:
    """The names FOUND's values set that are no package keys."""
    package = _package_keys()
    return [
        name for name in registry.leaves(found.values, package) if name not in package
    ]


def _all() -> list:
    """Every repository's defaults that `problems` does not refuse."""
    out = []
    for folder in _folders():
        found, _problem = _read(folder)
        if found is not None and not _stray(found):
            out.append(found)
    return out


def find(repository, folder) -> Found | None:
    """The repository defaults that apply to REPOSITORY and its FOLDER: see the module's docstring."""
    want, folder = str(repository or "").strip().lower(), _folder(folder)
    if not want or not folder:
        return None
    for found in _all():
        if found.repository.lower() == want and folder in found.folders:
            return found
    return None


def home() -> Found | None:
    """The defaults the package ships, its home repository's: see the module's docstring."""
    found = _all()
    return found[0] if found else None


def keys() -> list:
    """Every package key the package's repository defaults set: see the module's docstring."""
    names = {n for found in _all() for n in keys_of(found.values)}
    return [k.name for k in registry.keys() if k.name in names]


def own(value, generic) -> bool:
    """Whether VALUE is a value of its own: neither blank nor GENERIC. See the module's docstring."""
    if value is registry.MISSING or value in ("", None) or value == [] or value == {}:
        return False
    return value != generic


def generic(repository, folder, cfg=None) -> list:
    """The keys that take the package's generic values for REPOSITORY and its FOLDER: see the module's docstring."""
    if find(repository, folder) is not None:
        return []
    package = registry.package_defaults()
    return [
        name
        for name in keys()
        if cfg is None or not own(registry.get(cfg, name), registry.get(package, name))
    ]


def generic_found(repository, folder, names) -> str:
    """The words of the WARN that names the keys NAMES, which take the package's generic values."""
    return (
        f"none for {repository} and its folder {_folder(folder)}: the package's generic values apply to "
        f"{wrapping.joined(names)}"
    )


def problems() -> list:
    """What is wrong with the package's repository defaults: see the module's docstring."""
    out = []
    for folder in _folders():
        found, problem = _read(folder)
        if problem:
            out.append(problem)
            continue
        out += [
            f"{found.path}: {name} is no package key: repository defaults set package keys only"
            for name in _stray(found)
        ]
    return out
