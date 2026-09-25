# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for the IngestorGenerator test suite."""

from pathlib import Path

import pytest

from codegen.config_loader import load_config
from codegen.generator import IngestorGenerator
from tests.helpers import make_engine, make_minimal_config


@pytest.fixture(scope="session")
def configs_dir():
    return Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="session")
def config_path(configs_dir):

    def _config_path(name: str) -> Path:
        return configs_dir / name

    return _config_path


@pytest.fixture(scope="session")
def load_test_config(config_path):

    def _load(name: str):
        return load_config(config_path(name))

    return _load


@pytest.fixture
def scale_add_config(load_test_config):
    return load_test_config("scale_add.yaml")


@pytest.fixture
def binary_ops_config(load_test_config):
    """Multi-pack reference config (exercises the UMD-per-pack policy branch)."""
    return load_test_config("binary_ops.yaml")


@pytest.fixture
def gfx950_attention_dense_config(load_test_config):
    """Packaged-dialect reference config backed by a real rocKE builder. Loading needs
    no rocKE on PYTHONPATH: only the optional ``sources.rocke`` adapter imports it."""
    return load_test_config("gfx950_attention_dense.yaml")


@pytest.fixture
def heuristic_free_config():
    """An engine declaring ``heuristic: none``; nothing under ``configs/`` does, so
    without it the ``{% else %}`` arm of every ``has_heuristic`` branch renders for
    nothing."""
    return make_minimal_config(engine=make_engine(heuristic="none"))


@pytest.fixture(scope="session")
def template_dir():
    return Path(__file__).parent.parent / "templates"


@pytest.fixture
def generator(template_dir):
    return IngestorGenerator(template_dir)


@pytest.fixture
def all_config_names():
    return ["scale_add.yaml", "binary_ops.yaml", "gfx950_attention_dense.yaml"]
