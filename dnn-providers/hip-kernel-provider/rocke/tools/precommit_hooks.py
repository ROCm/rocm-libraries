# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A hook of the monorepo's pre-commit config, and the paths pre-commit gives it.

One home for reading a ``repo: local`` hook out of ``.pre-commit-config.yaml``
and for pre-commit's rule on which changed paths reach it: a path reaches a hook
when it matches the config's top-level ``files`` and the hook's ``files``, and
neither ``exclude``. A hook's ``types`` filters are not applied: they need the
file on disk, and a caller here may ask about a path a change deleted.

Needs PyYAML, which the rocKE hooks' environment has.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = ["PrecommitHook", "load_local_hook"]

# pre-commit's defaults for a config or hook that leaves the key out.
_DEFAULT_FILES = ""
_DEFAULT_EXCLUDE = "^$"


@dataclass(frozen=True)
class PrecommitHook:
    """One local hook: the command pre-commit runs, the environment it runs in,
    and which paths reach it."""

    hook_id: str
    entry: str
    args: tuple[str, ...]
    language: str
    additional_dependencies: tuple[str, ...]
    pass_filenames: bool
    files: re.Pattern[str]
    exclude: re.Pattern[str]
    config_files: re.Pattern[str]
    config_exclude: re.Pattern[str]

    def takes(self, path: str) -> bool:
        """Whether pre-commit gives this hook ``path``, a repo-relative path
        with ``/`` separators, when a change touches it."""
        return bool(
            self.config_files.search(path)
            and not self.config_exclude.search(path)
            and self.files.search(path)
            and not self.exclude.search(path)
        )


def load_local_hook(config: Path, hook_id: str) -> PrecommitHook:
    """The ``repo: local`` hook ``hook_id`` of the pre-commit config ``config``.

    Raises ``KeyError`` when no local hook has that id, ``ValueError`` when more
    than one has it or ``config`` is not YAML, and ``OSError`` when ``config``
    cannot be read."""
    try:
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(
            f"loading a pre-commit hook failed - config={config} does not parse "
            f"({error}), expected a YAML pre-commit config"
        ) from error
    found = [
        hook
        for repo in document.get("repos", [])
        if repo.get("repo") == "local"
        for hook in repo.get("hooks", [])
        if hook.get("id") == hook_id
    ]
    if not found:
        raise KeyError(
            f"loading a pre-commit hook failed - id={hook_id!r}, expected a local "
            f"hook of {config} with that id"
        )
    if len(found) > 1:
        raise ValueError(
            f"loading a pre-commit hook failed - id={hook_id!r} names "
            f"{len(found)} local hooks of {config}, expected one"
        )
    (hook,) = found
    return PrecommitHook(
        hook_id=hook_id,
        entry=hook["entry"],
        args=tuple(hook.get("args", ())),
        language=hook["language"],
        additional_dependencies=tuple(hook.get("additional_dependencies", ())),
        pass_filenames=hook.get("pass_filenames", True),
        files=re.compile(hook.get("files", _DEFAULT_FILES)),
        exclude=re.compile(hook.get("exclude", _DEFAULT_EXCLUDE)),
        config_files=re.compile(document.get("files", _DEFAULT_FILES)),
        config_exclude=re.compile(document.get("exclude", _DEFAULT_EXCLUDE)),
    )
