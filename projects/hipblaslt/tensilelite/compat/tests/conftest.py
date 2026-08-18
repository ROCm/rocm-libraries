# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-compat",
        action="store_true",
        default=False,
        help="run the installed Tensile compatibility tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-compat"):
        return
    skip = pytest.mark.skip(reason="compatibility tests require --run-compat")
    for item in items:
        item.add_marker(skip)
