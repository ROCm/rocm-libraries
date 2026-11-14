# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no HIP device", allow_module_level=True)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "conv: needs the MIOpen provider, which the bootstrap loads only when "
        "HIPDNN_VLLM_OPS enables conv (run these with HIPDNN_VLLM_OPS=...,conv -m conv)",
    )


_GPU_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items):
    # This hook sees every collected item of the session, not only this
    # directory's, so mark just the tests that live here.
    for item in items:
        if Path(str(item.path)).resolve().is_relative_to(_GPU_DIR):
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def hipdnn_state():
    from hipdnn_vllm import _bootstrap

    return _bootstrap.load()


@pytest.fixture(scope="session")
def miopen_loaded(hipdnn_state):
    """Whether this process's hipDNN handle loaded the MIOpen provider. The
    plugin set is fixed when the first handle is created, so a process started
    without conv in HIPDNN_VLLM_OPS cannot serve conv graphs."""
    import os

    from hipdnn_vllm import _C

    return any(
        os.path.basename(p).startswith("libmiopen") for p in _C.loaded_plugin_paths(torch.cuda.current_device())
    )


@pytest.fixture(autouse=True)
def _needs_conv(request):
    if request.node.get_closest_marker("conv") is None:
        return
    if not request.getfixturevalue("miopen_loaded"):
        pytest.skip("conv is not in HIPDNN_VLLM_OPS for this process, so the MIOpen provider is not loaded")


@pytest.fixture
def C(hipdnn_state):
    from hipdnn_vllm import _C

    return _C
