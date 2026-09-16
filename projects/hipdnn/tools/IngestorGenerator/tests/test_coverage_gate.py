# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Three rungs, three questions. A count answers none of them.

A descriptor count can be exactly right -- every descriptor on disk, correctly named
-- while nothing reaches a GPU: one duplicate catalog tuple makes the loader reject
the whole engine, every graph falls through to a different one, and the phase runs to
completion and exits 0.

The property under test here is therefore not "does the gate pass on good input" but
"does each rung stay separable", defending against a gate that quietly stops checking
and still prints a reassuring last line. A rung that cannot run must report NOT RUN
loudly and fail, never be skipped into a pass.

The validator is a build artifact and the profile is an author's own input, so the
end-to-end class needs both pointed at from the environment and skips otherwise. The
rung-separation tests need neither, and it is they that assert a missing validator is
a FAILURE of the gate rather than a skip -- the property this module exists for is
therefore checked on every machine, opt-in or not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_GATE = Path(__file__).resolve().parents[1] / "tools" / "coverage_gate.py"
_REPO_ROOT = Path(__file__).resolve().parents[5]

#: The env vars naming the authoring profile handed to `--profile`, in precedence
#: order: the unsuffixed name wins, then gfx942, then gfx950. Nothing here asserts a
#: profile's CONTENT -- only that the gate runs clean on a profile and the build it
#: describes -- so any of the three serves, and accepting all three is what stops a
#: reader who set `test_launch_surface.py`'s per-arch pair from being silently
#: skipped by this module. gfx942 precedes gfx950 because `_EXPECT_ENGINE`'s default
#: names the gfx942 bundle this branch ships; with both set, that is the one more
#: likely to match the build under test. The chosen profile and `_EXPECT_ENGINE` must
#: name the same pack.
_PROFILE_VARS = (
    "HIPDNN_INGESTOR_PROFILE",
    "HIPDNN_INGESTOR_PROFILE_GFX942",
    "HIPDNN_INGESTOR_PROFILE_GFX950",
)

#: The engine name the packed tree under test must expose. Overridable for the same
#: reason the profile is: it identifies a specific pack, not a property of the tool.
#: The default is the engine the provider's shipped production bundle declares
#: (`src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx942_attention_dense/`),
#: because `_find_build_artifacts` below probes a PACKED tree and that bundle is what
#: a build of this branch packs into one. A generator config under `configs/` is an
#: author's input that no build wires, so naming an engine only a config mentions
#: would reject every build dir and turn this class into a skip no machine can
#: satisfy.
_EXPECT_ENGINE = os.environ.get(
    "HIPDNN_INGESTOR_ENGINE", "hipkernel:Gfx942AttentionDense"
)


def _profile_from_env() -> Path | None:
    """The first of `_PROFILE_VARS` naming an existing file, or None.

    Deliberately NO fixture default, unlike `test_launch_surface.py`: the class below
    drives the whole gate against a packed tree an author really built, and the
    committed fixture profiles describe no real pack, so defaulting to one would
    manufacture a failure on the first machine that has a build rather than skip.
    """
    for var in _PROFILE_VARS:
        raw = os.environ.get(var)
        if raw and Path(raw).is_file():
            return Path(raw)
    return None


_PROFILE = _profile_from_env()


def _find_build_artifacts() -> tuple[Path | None, Path | None]:
    """(validator, packed tree) from a build that actually contains the engine
    these tests assert on.

    A build can be present and valid and still predate this engine, in which case
    the assertions below fail on a stale artifact rather than on a defect. Skip
    covers "no build for this engine"; it must never cover "the gate is broken".
    """
    for candidate in sorted(_REPO_ROOT.glob("build*")):
        # Both spellings, because the executable suffix is the platform's and the
        # skip below cannot tell "no build" apart from "a build this probe walked
        # past": a bare name matches nothing on Windows, where every build writes
        # the .exe, so the tests would report no build on a tree that has one.
        validator = next(
            (
                path
                for path in (
                    candidate / "bin" / "hipdnn_validate_descriptors",
                    candidate / "bin" / "hipdnn_validate_descriptors.exe",
                )
                if path.is_file()
            ),
            None,
        )
        packed = candidate / "lib/hipdnn_plugins/engines/arch_content"
        if not (validator and packed.is_dir()):
            continue
        try:
            probe = subprocess.run(
                [str(validator), str(packed), "--json"],
                capture_output=True,
                text=True,
                # Per candidate, at import time, so N build dirs cost N x this
                # before collection finishes. The real validator answers in ~0.12s.
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
    _VALIDATOR is None or _PROFILE is None,
    reason=f"needs BOTH a build*/ whose packed tree contains {_EXPECT_ENGINE} "
    "(configure with HIPDNN_ENABLE_KERNEL_INGESTOR=ON; HIPDNN_INGESTOR_ENGINE "
    f"overrides the name) AND one of {', '.join(_PROFILE_VARS)} (first wins) set to "
    "the existing authoring profile that build was packed from -- a profile is an "
    "author's input this repo does not ship, so this class is opt-in and its "
    "absence is not a broken checkout",
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

    The static rung reaches a bundle's schema through `KDP.engine -> UED.metadata
    -> KMD`, so documents that merely share a filename stem have no schema at all.
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
    """The two modes answer different questions, and the failure defended against is
    the weaker answer printed under the stronger one's name: a run that read no
    compiled evidence at all, reported "1. STATIC PASS", and let a reader conclude
    the shipped binaries match the metadata selecting them.
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
        """The minimal tree carries no producing-build evidence at all, the limiting
        case of a record that does not describe the artifact."""
        result = _run("--tree", str(_minimal_tree(tmp_path)), "--mode", "full")
        assert result.returncode != 0
        assert "1. STATIC   FAIL" in result.stdout
        assert "static" in result.stdout


class TestRungsStaySeparable:
    def test_a_missing_validator_fails_rather_than_skipping_to_a_pass(self, tmp_path):
        """The whole point of the split: rung 1 passing must not imply rung 2."""
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
    """The whole gate driven end to end against a packed tree an author actually
    built and the profile they authored it from.

    Opt-in, on both a build and one of `_PROFILE_VARS`: the classes above pin the
    rung-separation property everywhere, and this class adds only the claim that the
    assembled gate agrees with them on real inputs. Skipping it therefore leaves the
    module's stated property checked; it does not make it conditional.
    """

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
        The runtime loader has never heard of `builder`, so the authored tree fails
        rung 2 with the same 'dropping it' message a real dropped engine produces.
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
