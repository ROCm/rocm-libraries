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
    """Path to the configs/ directory."""
    return Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="session")
def config_path(configs_dir):
    """Factory fixture: returns path to a named config file."""

    def _config_path(name: str) -> Path:
        return configs_dir / name

    return _config_path


@pytest.fixture(scope="session")
def load_test_config(config_path):
    """Factory fixture: loads a named YAML config into an IngestorConfig."""

    def _load(name: str):
        return load_config(config_path(name))

    return _load


@pytest.fixture
def scale_add_config(load_test_config):
    """Single-pack reference config."""
    return load_test_config("scale_add.yaml")


@pytest.fixture
def binary_ops_config(load_test_config):
    """Multi-pack reference config (exercises the UMD-per-pack policy branch)."""
    return load_test_config("binary_ops.yaml")


@pytest.fixture
def gfx950_attention_dense_config(load_test_config):
    """Packaged-dialect reference config, backed by a REAL rocKE builder.

    Loading it needs no rocKE on PYTHONPATH -- the generator never imports the
    builder; only the optional ``sources.rocke`` adapter does, and only when
    asked. That separation is deliberate: descriptor generation must not
    require the kernel toolchain to be installed.
    """
    return load_test_config("gfx950_attention_dense.yaml")


@pytest.fixture
def heuristic_free_config():
    """An engine declaring ``heuristic: none`` -- legal, and shipped by nothing.

    Every config under ``configs/`` declares ``heuristic: native`` and
    ``EngineSpec.heuristic`` defaults to it, so the ``{% else %}`` arm of every
    template that branches on ``has_heuristic`` renders for no fixture at all. An
    engine may ship no ranking model (it then ranks on priority, then descriptor
    id), and the first integration that does would be the first to render those
    arms -- on its own deadline.

    Built from the unit helpers rather than added as a fourth file under
    ``configs/``: the arms belong to the templates, while a config in ``configs/``
    is also a worked example this tool tells its readers to copy.
    """
    return make_minimal_config(engine=make_engine(heuristic="none"))


@pytest.fixture(scope="session")
def template_dir():
    """Path to the templates/ directory."""
    return Path(__file__).parent.parent / "templates"


@pytest.fixture
def generator(template_dir):
    """IngestorGenerator instance configured with the real template dir."""
    return IngestorGenerator(template_dir)


@pytest.fixture
def all_config_names():
    return ["scale_add.yaml", "binary_ops.yaml", "gfx950_attention_dense.yaml"]
