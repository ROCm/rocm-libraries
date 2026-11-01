# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""register() against a stand-in vLLM registry that mirrors the pinned vLLM
semantics (AttentionBackendEnum.register raises on a duplicate name), plus the
real registry when the pinned vLLM is importable."""

import enum
import importlib
import sys
import types

import pytest

import hipdnn_vllm


class _FakeEnum(enum.Enum):
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"

    @classmethod
    def register(cls, name, value):
        if name in cls._member_map_:
            raise ValueError(f"Backend {name} already exists")
        member = object.__new__(cls)
        member._name_ = name
        member._value_ = value
        setattr(cls, name, member)
        cls._member_map_[name] = member
        cls._value2member_map_[value] = member
        cls._member_names_.append(name)
        return member

    def is_overridden(self):
        return self in _OVERRIDES

    def get_path(self):
        return _OVERRIDES.get(self, self.value)


_OVERRIDES = {}
_CALLS = []


def _register_backend(backend, class_path=None):
    _CALLS.append((backend, class_path))
    if isinstance(backend, str):
        backend = _FakeEnum.register(backend, class_path or "")
    _OVERRIDES[backend] = class_path
    return lambda x: x


def _fake_modules(rocm):
    """The stand-in vLLM modules; ``rocm["value"]`` drives is_rocm()."""
    platform = types.SimpleNamespace(is_rocm=lambda: rocm["value"])
    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.envs": types.ModuleType("vllm.envs"),
        "vllm.platforms": types.ModuleType("vllm.platforms"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
        "vllm.v1.attention.backends": types.ModuleType("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.registry": types.ModuleType("vllm.v1.attention.backends.registry"),
    }
    mods["vllm"].envs = mods["vllm.envs"]
    mods["vllm.envs"].environment_variables = {}
    mods["vllm.platforms"].current_platform = platform
    reg = mods["vllm.v1.attention.backends.registry"]
    reg.AttentionBackendEnum = _FakeEnum
    reg.register_backend = _register_backend
    return mods


@pytest.fixture
def fake_vllm(monkeypatch):
    rocm = {"value": True}
    for name, mod in _fake_modules(rocm).items():
        monkeypatch.setitem(sys.modules, name, mod)
    # Fresh plugin module state and a clean fake registry per test.
    monkeypatch.setattr(hipdnn_vllm, "_registered", False)
    for name in list(_FakeEnum._member_map_):
        if name != "TRITON_ATTN":
            del _FakeEnum._member_map_[name]
            _FakeEnum._member_names_.remove(name)
    _OVERRIDES.clear()
    _CALLS.clear()
    for var in ("HIPDNN_VLLM_DISABLE", "HIPDNN_VLLM_OPS"):
        monkeypatch.delenv(var, raising=False)
    # The stand-in has no vllm.model_executor: the OOT overrides (encoder
    # attention, LayerNorm, Conv), which subclass real vLLM layers, are
    # covered against the real vLLM in tests/gpu instead.
    monkeypatch.setenv("HIPDNN_VLLM_OPS", "attention")
    return rocm


def test_registers_backend(fake_vllm):
    hipdnn_vllm.register()
    assert "HIPDNN" in _FakeEnum.__members__
    assert _FakeEnum["HIPDNN"].get_path() == hipdnn_vllm.BACKEND_CLASS_PATH
    assert _CALLS == [("HIPDNN", hipdnn_vllm.BACKEND_CLASS_PATH)]


def test_idempotent_in_process(fake_vllm):
    hipdnn_vllm.register()
    hipdnn_vllm.register()
    assert len(_CALLS) == 1


def test_idempotent_after_module_reload(fake_vllm):
    """A second registration (fresh module state, registry already holding
    HIPDNN) must not raise the registry's duplicate-name error."""
    hipdnn_vllm.register()
    hipdnn_vllm._registered = False
    hipdnn_vllm.register()
    assert list(_FakeEnum.__members__).count("HIPDNN") == 1


def test_disabled_registers_nothing(fake_vllm, monkeypatch):
    monkeypatch.setenv("HIPDNN_VLLM_DISABLE", "1")
    hipdnn_vllm.register()
    assert "HIPDNN" not in _FakeEnum.__members__


def test_not_rocm_registers_nothing(fake_vllm):
    fake_vllm["value"] = False
    hipdnn_vllm.register()
    assert "HIPDNN" not in _FakeEnum.__members__


def test_attention_registered_even_when_excluded_from_ops(fake_vllm, monkeypatch):
    monkeypatch.setenv("HIPDNN_VLLM_OPS", "rms_norm")
    hipdnn_vllm.register()
    assert "HIPDNN" in _FakeEnum.__members__


# A fresh interpreter (this process may already hold torch from other tests):
# register() with the stand-in registry, then list the heavy modules loaded.
_REGISTER_NO_TORCH = """
import os, sys
sys.path[:0] = [{here!r}, {pkg_root!r}]
os.environ["HIPDNN_VLLM_OPS"] = "attention"
os.environ.pop("HIPDNN_VLLM_DISABLE", None)
import test_register as t
sys.modules.update(t._fake_modules({{"value": True}}))
import hipdnn_vllm
hipdnn_vllm.register()
assert t._CALLS == [("HIPDNN", hipdnn_vllm.BACKEND_CLASS_PATH)], t._CALLS
heavy = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch.") or m == "hipdnn_vllm._C")
print("HEAVY", heavy)
"""


def test_register_imports_no_torch():
    """register() must not pull torch or _C (API server has no GPU)."""
    import os
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(hipdnn_vllm.__file__)))
    code = _REGISTER_NO_TORCH.format(here=here, pkg_root=pkg_root)
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "HEAVY []", out.stdout


def _compile_factor():
    getter = sys.modules["vllm.envs"].environment_variables[hipdnn_vllm.COMPILE_FACTOR_NAME]
    return getter()


def test_compile_factor_registered(fake_vllm):
    """vLLM's compile caches key on vllm.envs.environment_variables; register()
    adds the plugin's files and dispatch config to it."""
    hipdnn_vllm.register()
    assert '"ops"' in _compile_factor()


def test_compile_factor_follows_ops(fake_vllm, monkeypatch):
    hipdnn_vllm.register()
    default = _compile_factor()
    monkeypatch.setattr(hipdnn_vllm, "_registered", False)
    monkeypatch.setenv("HIPDNN_VLLM_OPS", "rms_norm")
    hipdnn_vllm.register()
    assert _compile_factor() != default


def test_disabled_adds_no_compile_factor(fake_vllm, monkeypatch):
    monkeypatch.setenv("HIPDNN_VLLM_DISABLE", "1")
    hipdnn_vllm.register()
    assert hipdnn_vllm.COMPILE_FACTOR_NAME not in sys.modules["vllm.envs"].environment_variables


@pytest.mark.vllm
def test_real_registry(monkeypatch):
    try:
        importlib.import_module("vllm.v1.attention.backends.registry")
    except Exception:
        pytest.skip("pinned vLLM not importable")
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not current_platform.is_rocm():
        pytest.skip("not a ROCm platform")
    monkeypatch.setattr(hipdnn_vllm, "_registered", False)
    from vllm import envs

    monkeypatch.setattr(envs, "environment_variables", dict(envs.environment_variables))
    hipdnn_vllm.register()
    hipdnn_vllm._registered = False
    hipdnn_vllm.register()
    assert AttentionBackendEnum["HIPDNN"].get_path() == hipdnn_vllm.BACKEND_CLASS_PATH
    cls = AttentionBackendEnum["HIPDNN"].get_class()
    assert cls.get_name() == "HIPDNN"
    assert hipdnn_vllm.COMPILE_FACTOR_NAME in envs.environment_variables
    import torch

    # compile_factors() queries the device; it runs in GPU workers only.
    if torch.cuda.is_available():
        assert hipdnn_vllm.COMPILE_FACTOR_NAME in envs.compile_factors()


_PICKLE_MEMBER = """
import pickle, sys
import hipdnn_vllm
from vllm.v1.attention.backends.registry import AttentionBackendEnum
hipdnn_vllm._register_attention_backend()
sys.stdout.write(pickle.dumps([AttentionBackendEnum["HIPDNN"]]).hex())
"""

# A fresh interpreter that never ran register() or load_general_plugins, like a
# spawned EngineCore unpickling its vLLM config.
_UNPICKLE_MEMBER = """
import pickle, sys
from vllm.v1.attention.backends.registry import AttentionBackendEnum
assert "HIPDNN" not in AttentionBackendEnum.__members__
(member,) = pickle.loads(bytes.fromhex(sys.stdin.read()))
assert member is AttentionBackendEnum["HIPDNN"], member
print(member.name, member.get_path())
"""


@pytest.mark.vllm
def test_member_unpickles_before_plugins_load():
    import subprocess

    try:
        importlib.import_module("vllm.v1.attention.backends.registry")
    except Exception:
        pytest.skip("pinned vLLM not importable")
    run = dict(capture_output=True, text=True, check=True)
    # vLLM logs to stdout; the payload is the last line.
    dumped = subprocess.run([sys.executable, "-c", _PICKLE_MEMBER], **run).stdout.splitlines()[-1]
    out = subprocess.run([sys.executable, "-c", _UNPICKLE_MEMBER], input=dumped, **run)
    assert out.stdout.splitlines()[-1].split() == ["HIPDNN", hipdnn_vllm.BACKEND_CLASS_PATH]
