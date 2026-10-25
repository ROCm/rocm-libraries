# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run one prepared shape with the artifact's original fixtures and test body."""

import json
import os
from pathlib import Path
import sys

import pytest

pytest_plugins = ["tensilelite.Tests.conftest"]


def pytest_addoption(parser):
    parser.addoption("--rocjitsu-shape-job", type=Path)


def pytest_configure(config):
    common = Path(os.environ["TENSILELITE_ROOT"]) / "tensilelite/Tests/common"
    sys.path.insert(0, str(common))


def pytest_generate_tests(metafunc):
    if "config" not in metafunc.fixturenames:
        return
    job = json.loads(metafunc.config.getoption("--rocjitsu-shape-job").read_text())
    from rocjitsu_shape_plan import digest

    for key, hashkey in (("source", "source_sha256"), ("yaml", "yaml_sha256")):
        if digest(Path(job[key])) != job[hashkey]:
            raise pytest.UsageError(f"shape job's {key} changed after planning")
    arch = metafunc.config.getoption("--gpu-targets")
    os.environ["PyTestBuildArchNames"] = arch
    marks = [
        getattr(pytest.mark, mark["name"])(*mark["args"], **mark["kwargs"])
        for mark in job["markers"]
    ]
    metafunc.parametrize(
        "config", [pytest.param(job["yaml"], marks=marks, id=job["nodeid"])]
    )
