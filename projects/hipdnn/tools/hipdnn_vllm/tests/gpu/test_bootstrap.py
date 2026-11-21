# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Bundled backend loading and provider selection."""

import json
import os
import subprocess
import sys

import pytest

from hipdnn_vllm import _bootstrap

# Entry points _C resolves through the frontend wrapper or its own lookups.
_SYMBOLS = (
    "hipdnnCreate",
    "hipdnnVersionString_ext",
    "hipdnnBackendExecute",
    "hipdnnSetEnginePluginPaths_ext",
    "hipdnnGetEngineInfo_ext",
)

_PROBE = r"""
import ctypes, json, os, sys
import torch
torch_backend = os.environ.get("PROBE_PRELOAD_GLOBAL")
if torch_backend:
    # Worst case for symbol interposition: torch's copy first, in the global scope.
    ctypes.CDLL(torch_backend, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
from hipdnn_vllm import _bootstrap
st = _bootstrap.load()
from hipdnn_vllm import _C
x = torch.randn(64, 64, device="cuda")
print(json.dumps({
    "backend": st.backend_path,
    "resolved": os.path.realpath(_C.backend_library_path()),
    "origins": {s: _C.backend_symbol_origin(s) for s in sys.argv[1:]},
    "version": _C.loaded_backend_version(),
    "plan": [st.plugin_plan.mode, list(st.plugin_plan.paths)],
    "loaded": [os.path.realpath(p) for p in _C.loaded_plugin_paths(0)],
    "engines": [[e["name"], e["plugin"], e["version"]] for e in _C.engines(0)],
    "mapped": _bootstrap.mapped_backends(),
    "torch_mm": float((x @ x).sum().item()) == float((x @ x).sum().item()),
}))
"""


def _probe(**env):
    """Run _PROBE in a fresh process; an env value of None unsets the variable."""
    full = dict(os.environ)
    for k in ("HIPDNN_VLLM_PLUGIN_MODE", "HIPDNN_VLLM_PROVIDER_DIRS", "HIPDNN_VLLM_OPS", "PROBE_PRELOAD_GLOBAL"):
        full.pop(k, None)
    full.update(env)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, *_SYMBOLS], env=full, capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _torch_backend():
    from hipdnn_vllm import doctor

    return doctor._torch_backend()


def test_c_resolves_only_the_bundled_backend(hipdnn_state, C):
    ours = hipdnn_state.backend_path
    assert os.path.dirname(ours) == os.path.realpath(_bootstrap.bundled_backend_dir())
    assert os.path.realpath(C.backend_library_path()) == ours
    for symbol in _SYMBOLS:
        assert C.backend_symbol_origin(symbol) == ours, symbol
    assert ours in _bootstrap.mapped_backends()
    torch_copy = _torch_backend()
    if torch_copy:
        print("bundled backend:", ours, C.loaded_backend_version())
        print("torch backend:", torch_copy)
        assert torch_copy["path"] != ours
        assert torch_copy["version"] != C.loaded_backend_version()


def test_torch_backend_in_global_scope_does_not_capture_c(hipdnn_state):
    """torch's backend mapped first, RTLD_GLOBAL: both copies are mapped, and
    every symbol _C resolves (by dladdr) still lives in the bundled copy."""
    torch_copy = _torch_backend()
    if not torch_copy:
        pytest.skip("torch ships no libhipdnn_backend")
    out = _probe(PROBE_PRELOAD_GLOBAL=torch_copy["path"])
    print(json.dumps({k: out[k] for k in ("backend", "resolved", "origins", "version", "mapped")}, indent=1))
    assert out["resolved"] == out["backend"]
    assert set(out["origins"].values()) == {out["backend"]}
    assert set(out["mapped"]) == {out["backend"], torch_copy["path"]}
    assert out["version"] == hipdnn_state.backend_version
    assert out["engines"]
    assert out["torch_mm"]


def test_version_guard(C):
    assert C.build_backend_version() == C.loaded_backend_version()
    C.check_backend_version()


def test_set_backend_directory_closes_after_resolution(C):
    assert not C.set_backend_directory("/nonexistent")


def test_engines_and_plugins_listed(C):
    engines = C.engines(0)
    assert engines and all(e["name"] for e in engines)
    assert C.loaded_plugin_paths(0)


def test_configure_after_handle_is_rejected(C):
    with pytest.raises(RuntimeError):
        C.configure_plugins("additive", [])


def test_only_bundled_providers_load(hipdnn_state):
    """auto loads the bundled providers absolutely: MIOpen only with conv, and
    every engine is served by the bundled build."""
    if not hipdnn_state.bundled_dir:
        pytest.skip("no bundled providers")
    bundled = {os.path.realpath(p) for p in _bootstrap._plugin_files(hipdnn_state.bundled_dir)}
    miopen = {p for p in bundled if os.path.basename(p).startswith("libmiopen")}
    assert miopen, bundled

    default = _probe()
    conv = _probe(HIPDNN_VLLM_OPS="attention,rms_norm,conv")
    for run in (default, conv):
        assert run["plan"][0] == "absolute"
        assert set(run["loaded"]) <= bundled, run["loaded"]
    assert set(default["loaded"]) == bundled - miopen
    assert set(conv["loaded"]) == bundled
    print("default engines:", sorted(default["engines"]))
    print("conv engines:", sorted(conv["engines"]))
    assert {e[0] for e in default["engines"]} < {e[0] for e in conv["engines"]}


def test_torch_and_vllm_work_after_bootstrap(hipdnn_state):
    import torch
    import vllm  # noqa: F401

    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    ref = (a.float() @ a.float()).half()
    s = torch.softmax(a.float(), dim=-1)
    torch.testing.assert_close(s.sum(-1), torch.ones(128, device="cuda"))
    torch.testing.assert_close(a @ a, ref, atol=0.5, rtol=1e-2)
