# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Three rungs, three questions. A count answers none of them.

A descriptor-count gate passed on an arm that served ZERO graphs. The count was
right -- every descriptor was on disk, correctly named. They never reached a GPU:
a duplicate catalog tuple made the loader reject the whole engine, every graph fell
through to a different one, and the phase ran to completion and exited 0.

The property under test here is therefore not "does the gate pass on good input" but
"does each rung stay separable". The failure mode being defended against is a gate
that quietly stops checking and still prints a reassuring last line, so the tests
that matter are the ones asserting a rung reports NOT RUN loudly and fails, rather
than being skipped into a pass.

The validator is a build artifact, so rung-2 tests skip when it is absent -- and the
absence itself is asserted to be a FAILURE of the gate, not a skip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_GATE = Path(__file__).resolve().parents[1] / "tools" / "coverage_gate.py"
_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "gfx942_attention_dense.profile.yaml"
)
_REPO_ROOT = Path(__file__).resolve().parents[5]
_VALIDATOR = _REPO_ROOT / "build-noasm" / "bin" / "hipdnn_validate_descriptors"
_PACKED = _REPO_ROOT / "build-noasm/lib/hipdnn_plugins/engines/arch_content"

_needs_build = pytest.mark.skipif(
    not (_VALIDATOR.exists() and _PACKED.exists()),
    reason="needs a build configured with HIPDNN_ENABLE_KERNEL_INGESTOR=ON",
)


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_GATE), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _minimal_tree(tmp_path: Path) -> Path:
    """A structurally-valid bundle, so rung 1 can pass without a build."""
    root = tmp_path / "descriptors"
    root.mkdir()
    (root / "test_engine.kmd.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "fields": [{"name": "block_n", "type": "int", "default_value": 64}],
            }
        )
    )
    (root / "test_engine.kdp.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "kernelDescriptors": [
                    {
                        "name": "k0",
                        "kernel_source": {
                            "kind": "kpack",
                            "symbol": "s0",
                            "sha256": "a" * 64,
                        },
                        "metadata": {"block_n": 64},
                    }
                ],
            }
        )
    )
    return root


class TestRungsStaySeparable:
    def test_a_missing_validator_fails_rather_than_skipping_to_a_pass(self, tmp_path):
        """The whole point of the split: rung 1 passing must not imply rung 2.

        A gate that quietly drops a rung reports success for work it did not do,
        which is exactly the shape of the defect this tool exists to prevent.
        """
        result = _run("--tree", str(_minimal_tree(tmp_path)))
        assert result.returncode != 0
        assert "2. LOADS    NOT RUN" in result.stdout
        assert "GATE FAILED" in result.stdout
        assert "loads-not-run" in result.stdout

    def test_serves_is_always_reported_as_owed_never_inferred(self, tmp_path):
        """Rungs 1 and 2 both green still means nothing was served."""
        result = _run("--tree", str(_minimal_tree(tmp_path)))
        assert "3. SERVES   NOT RUN" in result.stdout
        assert "engine_name" in result.stdout, (
            "rung 3 must say to filter by engine_name; an unfiltered aggregate "
            "reports another engine's work as this engine's"
        )

    def test_a_narrowed_static_run_is_not_a_pass(self, tmp_path):
        """No profile means the policy and vocabulary checks did not execute."""
        result = _run("--tree", str(_minimal_tree(tmp_path)))
        assert "NOT CHECKED" in result.stdout
        assert "1. STATIC   FAIL" in result.stdout

    def test_a_missing_tree_is_an_error_not_an_empty_pass(self, tmp_path):
        result = _run("--tree", str(tmp_path / "nope"))
        assert result.returncode == 2


@_needs_build
class TestAgainstTheRealBuild:
    def test_packed_tree_passes_both_runnable_rungs(self):
        result = _run(
            "--tree",
            str(_PACKED),
            "--profile",
            str(_PROFILE),
            "--validator",
            str(_VALIDATOR),
            "--expect-engine",
            "hipkernel:Gfx942AttentionDense",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1. STATIC   PASS" in result.stdout
        assert "2. LOADS    PASS" in result.stdout
        assert "NOT CHECKED" not in result.stdout

    def test_an_engine_that_is_not_loaded_is_named(self):
        """The dropped-engine case. Its only observable is a name missing from the
        loaded list -- the file count is unchanged and the exit code would be 0."""
        result = _run(
            "--tree",
            str(_PACKED),
            "--profile",
            str(_PROFILE),
            "--validator",
            str(_VALIDATOR),
            "--expect-engine",
            "hipkernel:NoSuchEngine",
        )
        assert result.returncode != 0
        assert "MISSING" in result.stdout
        assert "hipkernel:NoSuchEngine" in result.stdout

    def test_the_authored_dialect_fails_rung_two_with_the_loaders_own_reason(
        self, tmp_path
    ):
        """`kind: rocke` is an AUTHORING form that hkp_pack lowers to `kind: kpack`.
        The runtime loader has never heard of `builder`, so pointing rung 2 at the
        authored tree fails -- correctly, and with the same 'dropping it' message
        that a real dropped engine produces.
        """
        root = tmp_path / "authored"
        root.mkdir()
        (root / "e.kmd.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "fields": [{"name": "block_n", "type": "int", "default_value": 64}],
                }
            )
        )
        (root / "e.kdp.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "kernelDescriptors": [
                        {
                            "name": "k0",
                            "kernel_source": {
                                "kind": "rocke",
                                "source": "m.py",
                                "builder": "build_x",
                                "spec": {"block_n": 64},
                            },
                            "metadata": {"block_n": 64},
                        }
                    ],
                }
            )
        )
        result = _run("--tree", str(root), "--validator", str(_VALIDATOR))
        assert result.returncode != 0
        assert "2. LOADS    FAIL" in result.stdout
