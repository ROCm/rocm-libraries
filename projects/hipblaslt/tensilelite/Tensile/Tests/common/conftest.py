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
################################################################################

from pathlib import Path

import pytest

from config_helpers import findAvailableArchs, findConfigs

_COMMON_DIR = Path(__file__).parent


def pytest_ignore_collect(collection_path=None, path=None, config=None):
    """Filter direct children of common/ based on --build-only / --use-cache mode."""
    # collection_path: pytest >= 7; path: pre-7 fallback (py.path.local).
    p = Path(collection_path) if collection_path is not None else Path(str(path))
    if p.parent != _COMMON_DIR:
        return None

    build_only = config.getoption("--build-only", default=False)
    use_cache = config.getoption("--use-cache", default=False)
    name = p.name

    if build_only:
        return name in ("test_config.py", "test_config_run.py")
    if use_cache:
        return name in ("test_config.py", "test_config_build.py")
    return name in ("test_config_build.py", "test_config_run.py")


def pytest_configure(config):
    build_only = config.getoption("--build-only", default=False)
    use_cache = config.getoption("--use-cache", default=False)
    artifact_dir = config.getoption("--artifact-dir", default=None)

    if build_only and use_cache:
        raise pytest.UsageError(
            "Cannot specify both --build-only and --use-cache. Choose one.")

    if (build_only or use_cache) and not artifact_dir:
        raise pytest.UsageError(
            "--artifact-dir is required when using --build-only or --use-cache.")

    if build_only and not config.getoption("--gpu-targets", default=None):
        raise pytest.UsageError(
            "--gpu-targets is required when using --build-only. "
            "Build-only machines typically lack GPUs, so hardware auto-detection "
            "will fail. Provide the target architecture explicitly, e.g. "
            "--gpu-targets gfx942.")


def pytest_generate_tests(metafunc):
    """Parametrize test functions that accept a 'config' fixture.

    Defers findConfigs() to collection time so --gpu-targets is available.
    Gating (which tests actually run) is handled by pytest.skip() inside
    each test function, not here.
    """
    if "config" not in metafunc.fixturenames:
        return
    gpu_targets = metafunc.config.getoption("--gpu-targets", default=None)
    archs = findAvailableArchs(gpu_targets)
    configs = findConfigs(availableArchs=archs)
    metafunc.parametrize("config", configs)
