################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization of ``LibraryIO``'s optional-dependency import fallbacks.

The module top tries, in order: ``orjson`` -> ``ujson`` -> ``simplejson`` ->
stdlib ``json``; ``CSafeLoader``/``CSafeDumper`` -> ``SafeLoader``/``SafeDumper``;
and ``msgpack`` (warn if absent). In the dev container the fast variants are
installed, so those ``except ImportError`` arms never execute during normal
collection. This test reloads the module with the fast variants *blocked*
(``sys.modules[name] = None`` raises ``ImportError`` on import; ``delattr`` on
the ``yaml`` module hides ``CSafeLoader``/``CSafeDumper``), then reloads it once
more to restore the normal bindings for the rest of the suite.

``yaml`` itself is not blocked — its absence calls ``printExit`` (``sys.exit``)
mid-import, which would leave the shared module half-initialised; that branch is
documented in ``resistance.md`` instead.
"""

import importlib
import sys

import pytest

import Tensile.LibraryIO as L

pytestmark = pytest.mark.unit


def test_optional_dependency_import_fallbacks(monkeypatch):
    import yaml

    # Block the fast JSON variants and msgpack: `import X` raises ImportError.
    for name in ("orjson", "ujson", "simplejson", "msgpack"):
        monkeypatch.setitem(sys.modules, name, None)
    # Hide the C YAML loader/dumper so `from yaml import CSafeLoader` fails.
    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    monkeypatch.delattr(yaml, "CSafeDumper", raising=False)

    try:
        importlib.reload(L)
        # Fell all the way back to stdlib json + SafeLoader/SafeDumper. The
        # msgpack "not detected" branch also runs (captured stdout) but reload
        # keeps the stale global, so it is not asserted here.
        assert L.json is sys.modules["json"]
        assert L.yamlLoader is yaml.SafeLoader
        assert L.yamlDumper is yaml.SafeDumper
    finally:
        # Restore real fast bindings for every other test in the session.
        monkeypatch.undo()
        importlib.reload(L)

    # Sanity: the restored module is functional again.
    assert L._fast_yaml_scalar(1) == "1"
