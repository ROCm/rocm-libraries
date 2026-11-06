# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for device tags and the ``skip-<deviceTag>`` marks.

MI300A and MI300X are both gfx942, so ``skip-gfx942`` cannot skip a test on one
and not the other. A device tag (``mi300a``) names the product. These tests
cover how a tag is found from ``rocminfo`` output and how ``configMarks`` turns
a ``skip-mi300a`` mark into a skip only where that tag is present.

They need no GPU: rocminfo output is canned and tags are passed in directly.
"""

import os

import pytest

from config_helpers import configMarks, findDeviceTags
from tensilelite.Tests.gpu_detection import device_tags_from_rocminfo

_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_COMMON_DIR)

# The configs carrying ``skip-mi300a``, with the ROCM ticket for each.
_MI300A_KNOWN_BUG_CONFIGS = [
    os.path.join(_COMMON_DIR, "streamk", "sk_hgemm_lsu.yaml"),  # ROCM-33647
    os.path.join(_COMMON_DIR, "gemm", "swizzleA.yaml"),  # ROCM-33648
    os.path.join(_COMMON_DIR, "gemm", "swizzleB.yaml"),  # ROCM-33649
    os.path.join(_COMMON_DIR, "gemm", "agntab_coverage_gfx942.yaml"),  # ROCM-33650
]


def _agent(name, compute_units, index=1):
    """One rocminfo agent block, with just the lines the parser reads."""
    return (
        "*******\n"
        f"Agent {index}\n"
        "*******\n"
        f"  Name:                    {name}\n"
        "  Marketing Name:          Some Marketing Name\n"
        f"  Compute Unit:            {compute_units}\n"
    )


_CPU_AGENT = _agent("AMD EPYC 9654 96-Core Processor", 228, index=1)


def _skipped(marks):
    return any(m.name == "skip" for m in marks)


@pytest.fixture
def mi300a_config(tmp_path):
    """A minimal config tagged ``skip-mi300a``."""
    config = tmp_path / "mi300a_known_bug.yaml"
    config.write_text(
        "TestParameters:\n"
        "  marks: [skip-mi300a]\n"
        "GlobalParameters:\n"
        "  Architecture: gfx942\n"
        "BenchmarkProblems:\n"
        "  - - OperationType: GEMM\n"
        "      DataType: s\n"
    )
    return str(config), str(tmp_path)


def test_mi300a_tag_from_rocminfo():
    """A gfx942 agent with 228 CUs is an MI300A."""
    assert device_tags_from_rocminfo(_CPU_AGENT + _agent("gfx942", 228, 2)) == ["mi300a"]


def test_mi300a_tag_with_stepping_name():
    """A feature suffix on the agent name does not hide the architecture."""
    assert device_tags_from_rocminfo(_agent("gfx942:sramecc+:xnack-", 228)) == ["mi300a"]


def test_no_tag_for_mi300x():
    """MI300X is gfx942 with 304 CUs: no tag, so its tests keep running."""
    assert device_tags_from_rocminfo(_CPU_AGENT + _agent("gfx942", 304, 2)) == []


def test_no_tag_for_cpu_agent_with_228_cus():
    """Only a gfx942 agent earns the tag; a CPU agent reports CUs as well."""
    assert device_tags_from_rocminfo(_CPU_AGENT) == []


def test_no_tag_for_other_arch_with_228_cus():
    assert device_tags_from_rocminfo(_agent("gfx950", 228)) == []


def test_tag_is_not_repeated_for_several_mi300a():
    text = _CPU_AGENT + _agent("gfx942", 228, 2) + _agent("gfx942", 228, 3)
    assert device_tags_from_rocminfo(text) == ["mi300a"]


def test_empty_rocminfo_output():
    assert device_tags_from_rocminfo("") == []


def test_env_override_replaces_detection(monkeypatch):
    monkeypatch.setenv("TENSILE_TEST_DEVICE_TAGS", "mi300a")
    assert findDeviceTags() == ["mi300a"]


def test_env_override_can_declare_no_tags(monkeypatch):
    monkeypatch.setenv("TENSILE_TEST_DEVICE_TAGS", "")
    assert findDeviceTags() == []


def test_skip_mi300a_fires_on_mi300a(mi300a_config):
    assert _skipped(configMarks(*mi300a_config, ["gfx942"], ["mi300a"]))


def test_skip_mi300a_inert_without_the_tag(mi300a_config):
    """On MI300X (gfx942, no device tag) the config must still run."""
    assert not _skipped(configMarks(*mi300a_config, ["gfx942"], []))
    assert not _skipped(configMarks(*mi300a_config, ["gfx942"]))


# Not named ``config``: conftest.py parametrizes that fixture name over every YAML.
@pytest.mark.parametrize("known_bug_config", _MI300A_KNOWN_BUG_CONFIGS, ids=os.path.basename)
def test_known_bug_configs_skip_only_on_mi300a(known_bug_config):
    """The known-bug configs skip on MI300A and nowhere else on gfx942."""
    assert _skipped(configMarks(known_bug_config, _TESTS_ROOT, ["gfx942"], ["mi300a"]))
    assert not _skipped(configMarks(known_bug_config, _TESTS_ROOT, ["gfx942"], []))
