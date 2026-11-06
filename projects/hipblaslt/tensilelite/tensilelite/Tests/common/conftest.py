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

# Architectures the YAML configs were parametrized for (set in pytest_generate_tests), so the
# deselection below does not detect them a second time.
_ARCHS_KEY = pytest.StashKey[list]()


def pytest_ignore_collect(collection_path: Path, config: pytest.Config):
    """Filter direct children of common/ based on --build-only / --use-cache mode."""
    if collection_path.parent != _COMMON_DIR:
        return None

    build_only = config.getoption("--build-only", default=False)
    use_cache = config.getoption("--use-cache", default=False)
    name = collection_path.name

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
    archs = metafunc.config.stash.get(_ARCHS_KEY, None)
    if archs is None:
        gpu_targets = metafunc.config.getoption("--gpu-targets", default=None)
        archs = findAvailableArchs(gpu_targets)
        metafunc.config.stash[_ARCHS_KEY] = archs
    device_tags = () if metafunc.config.getoption("--build-only", default=False) else None
    configs = findConfigs(availableArchs=archs,
                          deviceTags=device_tags,
                          splitWorlds=not metafunc.config.getoption("--build-only", default=False))
    metafunc.parametrize("config", configs)


def pytest_collection_modifyitems(config, items):
    """Deselect the YAML configs that opt out of every target architecture.

    A config opts out of an architecture with a ``skip-<arch>`` mark in its TestParameters, and
    configMarks then also marks it ``skip``. Such a test would only report a skip, but it would still
    be sent to (and counted by) the pytest-xdist scheduler: on gfx942 that is 563 of the 709 common
    tests, which inflates the chunks the default ``load`` scheduler hands out. Deselecting them
    keeps only the configs that run. This reads the marks configMarks already attached; it opens no
    files.
    """
    archs = config.stash.get(_ARCHS_KEY, None)
    if not archs:
        return
    optOut = {f"skip-{arch}" for arch in archs}
    keep, drop = [], []
    for item in items:
        names = {m.name for m in item.iter_markers()} if item.path.is_relative_to(_COMMON_DIR) else ()
        (drop if optOut.issubset(names) else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
