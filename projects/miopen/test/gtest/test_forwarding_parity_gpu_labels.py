#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the ex_gpu_* label derivation behind the forwarding parity harness.

Each mirrored parity entry carries one of these labels, and the runner selects by
label, so a label that is missing or misspelled leaves ctest selecting nothing and
reporting success -- coverage that looks present and never ran. That failure is
silent, hence these tests.

The module under test is driven with `cmake -P` against fixture YAML rather than
through a configure, which needs a GPU toolchain and minutes per case.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE = HERE / "ForwardingParityGpuLabels.cmake"

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None, reason="cmake not on PATH"
)


def derive(tmp_path, yaml_text):
    """Return the labels the module derives from `yaml_text`."""
    yaml = tmp_path / "test_categories.yaml"
    yaml.write_text(yaml_text)
    return derive_from_path(tmp_path, yaml)


def derive_from_path(tmp_path, yaml):
    """Return the labels the module derives from `yaml`, which need not exist."""
    # A driver rather than running the module directly, so the module stays free of
    # anything that exists only for this test.
    driver = tmp_path / "driver.cmake"
    driver.write_text(
        f'include("{MODULE.as_posix()}")\n'
        'message("LABELS=${MIOPEN_FORWARDING_PARITY_GPU_LABELS}")\n'
    )
    out = subprocess.run(
        [
            "cmake",
            f"-DMIOPEN_TEST_CATEGORIES_YAML={yaml.as_posix()}",
            "-P",
            str(driver),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(
        l for l in (out.stdout + out.stderr).splitlines() if l.startswith("LABELS=")
    )
    return [label for label in line[len("LABELS=") :].split(";") if label]


def block(key, label):
    return (
        f"  {key}:\n"
        "    test_patterns:\n"
        '      - "*Foo*"\n'
        "    labels:\n"
        '      - "quick"\n'
        f'      - "{label}"\n'
        "\n"
    )


def test_plain_key_yields_its_label(tmp_path):
    assert derive(
        tmp_path, "exclude_gpu:\n" + block("exclude_gpu_gfx950", "ex_gpu_gfx950")
    ) == ["ex_gpu_gfx950"]


@pytest.mark.parametrize("suffix", ["windows", "linux"])
def test_os_suffixed_key_yields_the_unsuffixed_label(tmp_path, suffix):
    """The label the runner selects by has no OS suffix even when the key does."""
    labels = derive(
        tmp_path,
        "exclude_gpu:\n" + block(f"exclude_gpu_gfx110X_{suffix}", "ex_gpu_gfx110X"),
    )
    assert labels == ["ex_gpu_gfx110X"]


def test_both_os_variants_of_one_arch_collapse_to_one_label(tmp_path):
    """Two entries under one name is a hard ctest error, so this must dedupe."""
    labels = derive(
        tmp_path,
        "exclude_gpu:\n"
        + block("exclude_gpu_gfx110X_windows", "ex_gpu_gfx110X")
        + block("exclude_gpu_gfx110X_linux", "ex_gpu_gfx110X"),
    )
    assert labels == ["ex_gpu_gfx110X"]


def test_label_mentioned_in_a_comment_is_not_derived(tmp_path):
    labels = derive(
        tmp_path,
        "exclude_gpu:\n"
        '  # superseded by ex_gpu_gfx110X, see "ex_gpu_gfx9999"\n'
        + block("exclude_gpu_gfx950", "ex_gpu_gfx950"),
    )
    assert labels == ["ex_gpu_gfx950"]


def test_missing_yaml_yields_no_labels_rather_than_failing(tmp_path):
    """No test_categories.yaml is supported, so it must not fail the configure."""
    assert derive_from_path(tmp_path, tmp_path / "absent.yaml") == []


def test_shipped_yaml_declares_a_label_for_every_exclusion_set(tmp_path):
    """Guards the real file: an exclusion set with no label is unreachable coverage."""
    yaml = HERE / "test_categories.yaml"
    keys = [
        line.strip().rstrip(":")
        for line in yaml.read_text().splitlines()
        if line.startswith("  exclude_gpu_") and line.rstrip().endswith(":")
    ]
    assert keys, "no exclusion sets found -- the scrape above is wrong"
    derived = derive(tmp_path, yaml.read_text())
    for key in keys:
        arch = key[len("exclude_gpu_") :]
        for os_suffix in ("_windows", "_linux"):
            arch = arch[: -len(os_suffix)] if arch.endswith(os_suffix) else arch
        assert f"ex_gpu_{arch}" in derived
