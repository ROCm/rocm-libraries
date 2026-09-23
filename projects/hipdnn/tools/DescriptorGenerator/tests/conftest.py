# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for the descriptor codegen test suite."""

import pytest
from pathlib import Path

from codegen.config_loader import load_config
from codegen.generator import DescriptorGenerator


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
    """Factory fixture: loads a named YAML config into OperationConfig."""

    def _load(name: str):
        return load_config(config_path(name))

    return _load


@pytest.fixture
def convolution_fwd_config(load_test_config):
    """Reference config -- most complete, has all frontend fields + enum_def."""
    return load_test_config("convolution_fwd.yaml")


@pytest.fixture
def pointwise_config(load_test_config):
    """Config with optional tensors, optional scalars, mode field with enum_def."""
    return load_test_config("pointwise.yaml")


@pytest.fixture
def batchnorm_config(load_test_config):
    """Config with tensor arrays, many optional tensors, array return type."""
    return load_test_config("batchnorm.yaml")


@pytest.fixture
def matmul_config(load_test_config):
    """Minimal config -- no data fields, no mode fields."""
    return load_test_config("matmul.yaml")


@pytest.fixture
def sdpa_config(load_test_config):
    """Complex config -- many tensors, multiple mode fields, both shared."""
    return load_test_config("sdpa.yaml")


@pytest.fixture(scope="session")
def template_dir():
    """Path to the templates/ directory."""
    return Path(__file__).parent.parent / "templates"


@pytest.fixture
def generator(template_dir):
    """DescriptorGenerator instance configured with the real template dir."""
    return DescriptorGenerator(template_dir)


@pytest.fixture
def all_config_names():
    """List of all YAML config filenames."""
    from tests.helpers import ALL_CONFIG_NAMES

    return ALL_CONFIG_NAMES
