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

Written against the standard library's unittest rather than pytest: this runs as a
ctest entry in a wrapper-enabled build, and nothing provisions pytest for a machine
that builds MIOpen.

    python3 -m unittest test_forwarding_parity_gpu_labels
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE / "ForwardingParityGpuLabels.cmake"


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


class GpuLabelDerivationTest(unittest.TestCase):
    def setUp(self):
        # Fail rather than skip: a skip reads as a pass, and what these guard against
        # is itself a silent pass. Every environment meant to run them has cmake.
        if shutil.which("cmake") is None:
            self.fail("cmake is not on PATH; these tests drive it with cmake -P")

        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tmp_path = Path(holder.name)

    def derive(self, yaml_text):
        """Return the labels the module derives from `yaml_text`."""
        yaml = self.tmp_path / "test_categories.yaml"
        yaml.write_text(yaml_text)
        return self.derive_from_path(yaml)

    def derive_from_path(self, yaml):
        """Return the labels the module derives from `yaml`, which need not exist."""
        out = self.run_module(yaml)
        out.check_returncode()
        line = next(
            l for l in (out.stdout + out.stderr).splitlines() if l.startswith("LABELS=")
        )
        return [label for label in line[len("LABELS=") :].split(";") if label]

    def run_module(self, yaml):
        """Run the module on `yaml` and return the finished process, whatever its exit code."""
        # A driver rather than running the module directly, so the module stays free of
        # anything that exists only for this test.
        driver = self.tmp_path / "driver.cmake"
        driver.write_text(
            f'include("{MODULE.as_posix()}")\n'
            'message("LABELS=${MIOPEN_FORWARDING_PARITY_GPU_LABELS}")\n'
        )
        return subprocess.run(
            [
                "cmake",
                f"-DMIOPEN_TEST_CATEGORIES_YAML={yaml.as_posix()}",
                "-P",
                str(driver),
            ],
            capture_output=True,
            text=True,
        )

    def test_plain_key_yields_its_label(self):
        self.assertEqual(
            self.derive(
                "exclude_gpu:\n" + block("exclude_gpu_gfx950", "ex_gpu_gfx950")
            ),
            ["ex_gpu_gfx950"],
        )

    def test_os_suffixed_key_yields_the_unsuffixed_label(self):
        """The label the runner selects by has no OS suffix even when the key does."""
        for suffix in ("windows", "linux"):
            with self.subTest(suffix=suffix):
                labels = self.derive(
                    "exclude_gpu:\n"
                    + block(f"exclude_gpu_gfx110X_{suffix}", "ex_gpu_gfx110X"),
                )
                self.assertEqual(labels, ["ex_gpu_gfx110X"])

    def test_both_os_variants_of_one_arch_collapse_to_one_label(self):
        """Two entries under one name is a hard ctest error, so this must dedupe."""
        labels = self.derive(
            "exclude_gpu:\n"
            + block("exclude_gpu_gfx110X_windows", "ex_gpu_gfx110X")
            + block("exclude_gpu_gfx110X_linux", "ex_gpu_gfx110X"),
        )
        self.assertEqual(labels, ["ex_gpu_gfx110X"])

    def test_label_mentioned_in_a_comment_is_not_derived(self):
        labels = self.derive(
            "exclude_gpu:\n"
            '  # superseded by ex_gpu_gfx110X, see "ex_gpu_gfx9999"\n'
            + block("exclude_gpu_gfx950", "ex_gpu_gfx950"),
        )
        self.assertEqual(labels, ["ex_gpu_gfx950"])

    def test_unquoted_label_is_derived(self):
        """YAML allows a plain scalar, and the file must not go quiet if someone uses one."""
        labels = self.derive(
            "exclude_gpu:\n"
            "  exclude_gpu_gfx950:\n"
            "    labels:\n"
            "      - quick\n"
            "      - ex_gpu_gfx950\n"
        )
        self.assertEqual(labels, ["ex_gpu_gfx950"])

    def test_yaml_with_no_labels_fails_the_configure(self):
        """An existing file that yields nothing would drop every per-arch entry silently."""
        yaml = self.tmp_path / "test_categories.yaml"
        yaml.write_text(
            "exclude_gpu:\n  exclude_gpu_gfx950:\n    labels: [quick, ex_gpu_gfx950]\n"
        )
        out = self.run_module(yaml)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("No ex_gpu_* labels found", out.stderr)
        self.assertIn(yaml.as_posix(), out.stderr)

    def test_missing_yaml_yields_no_labels_rather_than_failing(self):
        """No test_categories.yaml is supported, so it must not fail the configure."""
        self.assertEqual(self.derive_from_path(self.tmp_path / "absent.yaml"), [])

    def test_shipped_yaml_declares_a_label_for_every_exclusion_set(self):
        """Guards the real file: an exclusion set with no label is unreachable coverage."""
        yaml = HERE / "test_categories.yaml"
        keys = [
            line.strip().rstrip(":")
            for line in yaml.read_text().splitlines()
            if line.startswith("  exclude_gpu_") and line.rstrip().endswith(":")
        ]
        self.assertTrue(keys, "no exclusion sets found -- the scrape above is wrong")
        derived = self.derive(yaml.read_text())
        for key in keys:
            arch = key[len("exclude_gpu_") :]
            for os_suffix in ("_windows", "_linux"):
                arch = arch[: -len(os_suffix)] if arch.endswith(os_suffix) else arch
            self.assertIn(f"ex_gpu_{arch}", derived)


if __name__ == "__main__":
    unittest.main()
