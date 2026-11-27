# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Registers the ``quick`` marker the FlyDSL drift-guard CTest entries select on."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "quick: basic device-free unit tests (no DVC content, no staged shard, no compile)",
    )
