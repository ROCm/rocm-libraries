# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_ALIAS = "_tensilelite_namespace_bridge_test"


def _run_script(script):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_PACKAGE_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


@pytest.mark.parametrize("alias_first", [False, True])
def test_alias_and_canonical_names_share_module_identity(alias_first):
    script = f"""
import importlib
import importlib.resources
import sys

from Tensile._namespace_bridge import install_alias

alias = {_ALIAS!r}
install_alias(alias=alias, canonical="Tensile")

if {alias_first!r}:
    alias_module = importlib.import_module(f"{{alias}}.resources")
    canonical_module = importlib.import_module("Tensile.resources")
else:
    canonical_module = importlib.import_module("Tensile.resources")
    alias_module = importlib.import_module(f"{{alias}}.resources")

assert alias_module is canonical_module
assert sys.modules[f"{{alias}}.resources"] is canonical_module
assert alias_module._STATIC_HEADER_NAMES is canonical_module._STATIC_HEADER_NAMES
assert alias_module.__name__ == "Tensile.resources"
assert alias_module.__package__ == "Tensile"
assert alias_module.__spec__.name == "Tensile.resources"
assert importlib.resources.files(alias).name == "Tensile"
assert importlib.reload(alias_module) is canonical_module
assert sys.modules[f"{{alias}}.resources"] is canonical_module
"""
    _run_script(script)


def test_missing_alias_descendant_reports_the_requested_name():
    script = f"""
import importlib

from Tensile._namespace_bridge import install_alias

alias = {_ALIAS!r}
install_alias(alias=alias, canonical="Tensile")

try:
    importlib.import_module(f"{{alias}}.DoesNotExist")
except ModuleNotFoundError as error:
    assert error.name == f"{{alias}}.DoesNotExist"
else:
    raise AssertionError("missing compatibility module unexpectedly imported")
"""
    _run_script(script)
