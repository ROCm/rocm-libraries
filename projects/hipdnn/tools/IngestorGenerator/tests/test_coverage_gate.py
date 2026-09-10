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


_EXPECT_ENGINE = "hipkernel:Gfx942AttentionDense"


def _find_build_artifacts() -> tuple[Path | None, Path | None]:
    """(validator, packed tree) from a build that actually contains the engine
    these tests assert on.

    Discovered, not hardcoded: this was pinned to `build-noasm/`, one author's
    directory name, so it skipped on every checkout that calls its build
    anything else -- reporting "needs a build" while sitting next to one.

    The engine check is the other half. A build can be present and valid and
    still predate this engine (the packed tree here ships ConvFwd, Pointwise and
    the examples), in which case the assertions below fail on a stale artifact
    rather than on a defect. Skip covers "no build for this engine"; it must
    never cover "the gate is broken".
    """
    for candidate in sorted(_REPO_ROOT.glob("build*")):
        validator = candidate / "bin" / "hipdnn_validate_descriptors"
        packed = candidate / "lib/hipdnn_plugins/engines/arch_content"
        if not (validator.is_file() and packed.is_dir()):
            continue
        try:
            probe = subprocess.run(
                [str(validator), str(packed), "--json"],
                capture_output=True,
                text=True,
                # Per CANDIDATE, and this runs at import time, so N build dirs
                # cost N x this before collection finishes. The real validator
                # answers in ~0.12s; 15s is ~100x headroom for a loaded box and
                # still bounds a hung probe to something a human will wait out.
                timeout=15,
            )
            engines = json.loads(probe.stdout).get("engines", [])
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        if _EXPECT_ENGINE in engines:
            return validator, packed
    return None, None


_VALIDATOR, _PACKED = _find_build_artifacts()

_needs_build = pytest.mark.skipif(
    _VALIDATOR is None or not _PROFILE.is_file(),
    reason=f"needs a build*/ whose packed tree contains {_EXPECT_ENGINE} "
    f"AND {_PROFILE.name} (configure with HIPDNN_ENABLE_KERNEL_INGESTOR=ON on a "
    "branch that ships both)",
)


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_GATE), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


_KMD_ID = "44444444-4444-4444-4444-444444444444"
_UED_ID = "55555555-5555-5555-5555-555555555555"


def _minimal_tree(tmp_path: Path, name: str = "descriptors") -> Path:
    """A structurally-valid, id-wired bundle, so rung 1 can pass without a build.

    Wired by id and not by filename: the static rung reaches a bundle's schema
    through `KDP.engine -> UED.metadata -> KMD`, so a tree whose documents merely
    share a stem has no schema at all as far as it is concerned.
    """
    root = tmp_path / name
    root.mkdir()
    (root / "test_engine.kmd.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": _KMD_ID,
                "fields": [{"name": "block_n", "type": "int", "default_value": 64}],
            }
        )
    )
    (root / "test_engine.ued.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": _UED_ID,
                "name": "test:Engine",
                "metadata": _KMD_ID,
            }
        )
    )
    (root / "test_engine.kdp.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": "66666666-6666-6666-6666-666666666666",
                "engine": _UED_ID,
                "arch": ["gfx942"],
                "kernelDescriptors": [
                    {
                        "version": "1.0",
                        "id": "77777777-7777-7777-7777-777777777777",
                        "name": "k0",
                        "arch": ["gfx942"],
                        "kernel_source": {
                            "kind": "kpack",
                            "library": "kpack/test.kpack",
                            "toc_key": "v0",
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


class TestTheStaticRungNeverOverstatesItself:
    """Rung 1's claim is chosen, never defaulted, and reported as chosen.

    The two modes answer different questions, and the failure being defended
    against is the weaker answer printed under the stronger one's name: a run that
    read no compiled evidence at all, reported "1. STATIC PASS", and let a reader
    conclude the shipped binaries match the metadata selecting them.
    """

    def test_omitting_the_mode_is_a_usage_error(self, tmp_path):
        result = _run("--tree", str(_minimal_tree(tmp_path)))
        assert result.returncode != 0
        assert "--mode" in result.stderr

    def test_structural_reports_rung_one_as_structural_only(self, tmp_path):
        result = _run("--tree", str(_minimal_tree(tmp_path)), "--mode", "structural")
        assert "1. STATIC   PASS (STRUCTURAL ONLY" in result.stdout
        assert "compiled specialization agreement NOT checked" in result.stdout
        assert (
            "1. STATIC   PASS\n" not in result.stdout
        ), "the unqualified line asserts a claim this run never made"

    def test_full_mode_fails_a_tampered_evidence_record(self, tmp_path):
        """The minimal tree carries no producing-build evidence at all, which is
        the limiting case of a record that does not describe the artifact. Full
        mode must fail it rather than report it as an unchecked property."""
        result = _run("--tree", str(_minimal_tree(tmp_path)), "--mode", "full")
        assert result.returncode != 0
        assert "1. STATIC   FAIL" in result.stdout
        assert "static" in result.stdout


class TestRungsStaySeparable:
    def test_a_missing_validator_fails_rather_than_skipping_to_a_pass(self, tmp_path):
        """The whole point of the split: rung 1 passing must not imply rung 2.

        A gate that quietly drops a rung reports success for work it did not do,
        which is exactly the shape of the defect this tool exists to prevent.
        """
        result = _run("--tree", str(_minimal_tree(tmp_path)), "--mode", "structural")
        assert result.returncode != 0
        assert "2. LOADS    NOT RUN" in result.stdout
        assert "GATE FAILED" in result.stdout
        assert "loads-not-run" in result.stdout

    def test_serves_is_always_reported_as_owed_never_inferred(self, tmp_path):
        """Rungs 1 and 2 both green still means nothing was served."""
        result = _run("--tree", str(_minimal_tree(tmp_path)), "--mode", "structural")
        assert "3. SERVES   NOT RUN" in result.stdout
        assert "engine_name" in result.stdout, (
            "rung 3 must say to filter by engine_name; an unfiltered aggregate "
            "reports another engine's work as this engine's"
        )

    def test_a_missing_tree_is_an_error_not_an_empty_pass(self, tmp_path):
        result = _run("--tree", str(tmp_path / "nope"), "--mode", "structural")
        assert result.returncode == 2


@_needs_build
class TestAgainstTheRealBuild:
    def test_packed_tree_passes_both_runnable_rungs(self):
        result = _run(
            "--tree",
            str(_PACKED),
            "--mode",
            "full",
            "--profile",
            str(_PROFILE),
            "--validator",
            str(_VALIDATOR),
            "--expect-engine",
            _EXPECT_ENGINE,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1. STATIC   PASS" in result.stdout
        assert "STRUCTURAL ONLY" not in result.stdout
        assert "2. LOADS    PASS" in result.stdout
        assert "NOT CHECKED" not in result.stdout

    def test_an_engine_that_is_not_loaded_is_named(self):
        """The dropped-engine case. Its only observable is a name missing from the
        loaded list -- the file count is unchanged and the exit code would be 0."""
        result = _run(
            "--tree",
            str(_PACKED),
            "--mode",
            "full",
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
        root = _minimal_tree(tmp_path, "authored")
        kdp_path = root / "test_engine.kdp.json"
        doc = json.loads(kdp_path.read_text())
        # Only the dialect changes. The tree stays id-wired so rung 1 still passes
        # and the failure this test is about belongs unambiguously to rung 2.
        doc["kernelDescriptors"][0]["kernel_source"] = {
            "kind": "rocke",
            "source": "m.py",
            "builder": "build_x",
            "spec": {"block_n": 64},
        }
        kdp_path.write_text(json.dumps(doc))
        result = _run(
            "--tree", str(root), "--mode", "structural", "--validator", str(_VALIDATOR)
        )
        assert result.returncode != 0
        assert "2. LOADS    FAIL" in result.stdout
