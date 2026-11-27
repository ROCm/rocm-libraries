# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What a configure read and what its targets compile, from CMake's file API.

Ask before configuring, read after::

    request_codemodel(build_root)
    configure_platform(build_root, ...)
    model = read_codemodel(build_root, "Release")
    model.input_files                       # the files the configure read
    for name in model.linked_targets(["rocke_core"]):
        model.targets[name].sources         # what each target compiles

The replies are CMake's own record of the configure, the same for every
generator. They do not name a file that a build step's command reads unless it
is a declared source (a generator script, for one): CMake does not report those.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CMakeCodemodel", "CMakeTarget", "read_codemodel", "request_codemodel"]

_QUERIES = ("codemodel-v2", "cmakeFiles-v1")


def _api_dir(build_root: Path) -> Path:
    return build_root / ".cmake" / "api" / "v1"


def request_codemodel(build_root: Path) -> None:
    """Ask the next configure of ``build_root`` to write the replies
    :func:`read_codemodel` reads."""
    query = _api_dir(build_root) / "query"
    query.mkdir(parents=True, exist_ok=True)
    for kind in _QUERIES:
        (query / kind).touch()


@dataclass(frozen=True)
class CMakeTarget:
    """One target: the sources it compiles and the directories it includes
    headers from (files the build generates are left out), and the names of
    the targets it depends on."""

    sources: frozenset[Path]
    include_dirs: frozenset[Path]
    dependencies: frozenset[str]


@dataclass(frozen=True)
class CMakeCodemodel:
    """A configured tree, as its file-API replies describe it: the files the
    configure read (but CMake's own modules and the files it wrote), and each
    target by name."""

    input_files: frozenset[Path]
    targets: Mapping[str, CMakeTarget]

    def linked_targets(self, names: Iterable[str]) -> frozenset[str]:
        """``names`` and every target they depend on, transitively. Raises
        ``KeyError`` for a name the tree does not define."""
        pending = list(names)
        reached: set[str] = set()
        while pending:
            name = pending.pop()
            if name in reached:
                continue
            if name not in self.targets:
                raise KeyError(
                    f"reading CMake targets failed - target={name!r}, expected a "
                    "target the configure defined"
                )
            reached.add(name)
            pending.extend(self.targets[name].dependencies)
        return frozenset(reached)


def _reply(reply_dir: Path, name: str) -> dict:
    return json.loads((reply_dir / name).read_text(encoding="utf-8"))


def _target(
    reply_dir: Path, json_file: str, source_root: Path, names_by_id: Mapping[str, str]
) -> CMakeTarget:
    target = _reply(reply_dir, json_file)
    return CMakeTarget(
        sources=frozenset(
            (source_root / source["path"]).resolve()
            for source in target.get("sources", ())
            if not source.get("isGenerated")
        ),
        include_dirs=frozenset(
            (source_root / include["path"]).resolve()
            for group in target.get("compileGroups", ())
            for include in group.get("includes", ())
        ),
        dependencies=frozenset(
            names_by_id[dependency["id"]]
            for dependency in target.get("dependencies", ())
        ),
    )


def read_codemodel(build_root: Path, config: str) -> CMakeCodemodel:
    """The replies the last configure of ``build_root`` wrote, for the build
    configuration ``config``. Raises ``FileNotFoundError`` when there are none
    (:func:`request_codemodel` asks for them) and ``KeyError`` when they
    describe no configuration ``config``."""
    reply_dir = _api_dir(build_root) / "reply"
    indexes = sorted(reply_dir.glob("index-*.json"))
    if not indexes:
        raise FileNotFoundError(
            f"reading the CMake file API failed - no reply in {reply_dir}, "
            "expected request_codemodel before the configure"
        )
    responses = _reply(reply_dir, indexes[-1].name)["reply"]
    codemodel = _reply(reply_dir, responses["codemodel-v2"]["jsonFile"])
    cmake_files = _reply(reply_dir, responses["cmakeFiles-v1"]["jsonFile"])
    source_root = Path(codemodel["paths"]["source"])
    # CMake's own modules and the files the configure writes are left out; a
    # file outside the source tree (isExternal) is kept: the configure read it.
    input_files = frozenset(
        (source_root / item["path"]).resolve()
        for item in cmake_files["inputs"]
        if not (item.get("isCMake") or item.get("isGenerated"))
    )
    configurations = {
        configuration["name"]: configuration
        for configuration in codemodel["configurations"]
    }
    if config not in configurations:
        raise KeyError(
            f"reading the CMake file API failed - config={config!r}, expected one "
            f"of {sorted(configurations)}"
        )
    listed = configurations[config]["targets"]
    names_by_id = {target["id"]: target["name"] for target in listed}
    targets = {
        target["name"]: _target(reply_dir, target["jsonFile"], source_root, names_by_id)
        for target in listed
    }
    return CMakeCodemodel(input_files, targets)
