# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's home repository's defaults (repository/<org>__<repo>/defaults.json, rkmon_lib/repository_defaults.py),
for the tests whose data is that repository's: FOUND, the defaults the package ships (repository_defaults.home), so that
no test names their repository or its folder; REPOSITORY and FOLDER, the repository and the first folder they apply
to; VALUES, the package keys they set, nested; `config(cfg)`, CFG with them laid over it; `stand_in(repository,
*folders)`, for a test module whose invented repository stands for the home repository; PACKAGE_FOLDER, the package's
own repository defaults' folder, which a process a test starts reads whatever `stand_in` lays over them.
"""

import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import registry, repository_defaults  # noqa: E402

FOUND = repository_defaults.home()
REPOSITORY, FOLDER = FOUND.repository, FOUND.folders[0]
PACKAGE_FOLDER = repository_defaults.FOLDER
VALUES = FOUND.values


def config(cfg: dict | None = None) -> dict:
    """CFG (nested; {} when None), a copy, with each package key VALUES sets in its place."""
    out = copy.deepcopy(cfg or {})
    for name in repository_defaults.keys_of(VALUES):
        registry.put(out, name, copy.deepcopy(registry.get(VALUES, name)))
    return out


def stand_in(repository: str, *folders: str) -> None:
    """Called from a test module's setUpModule: until the module's tests end, the only repository defaults are the
    home repository's, as REPOSITORY's (<org>/<repo>, the module's invented name), for FOLDERS. The module's invented
    data stands for the home repository's, so its checks find repository defaults, and its values are theirs.
    """
    tmp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(tmp.cleanup)
    root = pathlib.Path(tmp.name)
    folder = root / repository.replace("/", repository_defaults.SEPARATOR, 1)
    folder.mkdir()
    data = json.loads(FOUND.path.read_text(encoding="utf-8"))
    data["folders"] = list(folders)
    (folder / repository_defaults.FILE).write_text(json.dumps(data), encoding="utf-8")
    patch = mock.patch.object(repository_defaults, "FOLDER", root)
    patch.start()
    unittest.addModuleCleanup(patch.stop)
