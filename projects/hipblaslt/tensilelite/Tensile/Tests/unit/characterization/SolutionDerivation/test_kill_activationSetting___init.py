import importlib
import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
pytestmark = pytest.mark.unit

def test_activation_setting_init_sets_empty_string_enum():
    obj = S.activationSetting()
    assert obj.activationEnum == ""
    assert obj.activationEnum is not None
    assert isinstance(obj.activationEnum, str)

def test_activation_setting_empty_enum_is_falsy_and_not_none():
    obj = S.activationSetting()
    assert not obj.activationEnum
    assert obj.activationEnum != None
