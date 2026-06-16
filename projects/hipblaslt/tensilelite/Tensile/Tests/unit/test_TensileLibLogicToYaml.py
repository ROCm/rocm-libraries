################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import filecmp
import os
import tempfile
from pathlib import Path

import pytest

from Tensile import TensileLibLogicToYaml

_TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
_LIBLOGIC_FIXTURE_PATH = _TEST_DATA_DIR / "TensileLibLogicToYaml_liblogic.yaml"
_EXPECTED_CONFIG_FIXTURE_PATH = _TEST_DATA_DIR / "TensileLibLogicToYaml_expected_config.yaml"


def _read_liblogic_fixture() -> str:
    """Return the library-logic YAML used by :func:`test_TensileLibLogicToYaml`.

    Returns:
        Raw UTF-8 text of the gfx950 sample library logic.

    Raises:
        FileNotFoundError: If the fixture file is missing from ``test_data/``.
        OSError: If the file cannot be read.
    """
    return _LIBLOGIC_FIXTURE_PATH.read_text(encoding="utf-8")


def _read_expected_config_fixture() -> str:
    """Return the golden Tensile config YAML for :func:`test_TensileLibLogicToYaml`.

    Returns:
        Raw UTF-8 text of the expected generator output.

    Raises:
        FileNotFoundError: If the fixture file is missing from ``test_data/``.
        OSError: If the file cannot be read.
    """
    return _EXPECTED_CONFIG_FIXTURE_PATH.read_text(encoding="utf-8")


def findAvailableArchs():
    from Tensile.Tests.gpu_detection import get_available_archs
    return get_available_archs()


@pytest.mark.skipif(
    "gfx950" not in findAvailableArchs(), reason="Requires gfx950 architecture"
)
def test_TensileLibLogicToYaml():
    solutionIndex = 0
    liblogic_body = _read_liblogic_fixture()
    expected_body = _read_expected_config_fixture()

    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
        f.write(liblogic_body)
        f.flush()
        libLogicFileName = f.name

    with tempfile.TemporaryDirectory() as WORKSPACE:
        configYaml = os.path.join(WORKSPACE, "config.yaml")
        TensileLibLogicToYaml.TensileLibLogicToYaml(
            libLogicFileName, solutionIndex, configYaml, False
        )

        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            f.write(expected_body)
            f.flush()
            configFileName = f.name

        assert filecmp.cmp(configYaml, configFileName, shallow=False)
