# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib
import sys
import types
from unittest.mock import Mock

import Tensile.TensileLogic as tensile_logic


def test_main_imports_run_only_when_called(monkeypatch):
    monkeypatch.delitem(sys.modules, "Tensile.TensileLogic.Run", raising=False)
    importlib.reload(tensile_logic)

    assert "Tensile.TensileLogic.Run" not in sys.modules

    run_module = types.ModuleType("Tensile.TensileLogic.Run")
    expected = object()
    run_module.main = Mock(return_value=expected)
    monkeypatch.setitem(sys.modules, run_module.__name__, run_module)

    assert tensile_logic.main() is expected
    run_module.main.assert_called_once_with()
