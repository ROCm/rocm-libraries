# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Three rungs, three questions. A count answers none of them.

A descriptor count can be exactly right while nothing reaches a GPU: one duplicate
catalog tuple makes the loader reject the whole engine and the phase still exits 0. The
property under test is that each rung stays separable, and that a rung which cannot run
reports NOT RUN and fails rather than being skipped into a pass.

The end-to-end class needs a build carrying the descriptor validator and skips
otherwise; the rung-separation tests need neither, so the property is checked on every
machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_IG_ROOT = Path(__file__).resolve().parents[1]
_GATE = _IG_ROOT / "tools" / "coverage_gate.py"
_REPO_ROOT = Path(__file__).resolve().parents[5]

sys.path.insert(
    0, str(_REPO_ROOT / "dnn-providers/hip-kernel-provider/descriptor-packaging/python")
)

from hkp_pack import provenance_sidecar  # noqa: E402

#: The engine manifest the shipped production bundle is generated from.
_SHIPPED_MANIFEST = _IG_ROOT / "configs" / "gfx950_attention_dense.engine.yaml"

#: The env vars naming the engine manifest handed to `--manifest`, first wins. The
#: chosen manifest and `_EXPECT_ENGINE` must name the same pack.
_MANIFEST_VARS = (
    "HIPDNN_INGESTOR_MANIFEST",
    "HIPDNN_INGESTOR_MANIFEST_GFX942",
    "HIPDNN_INGESTOR_MANIFEST_GFX950",
)

#: The engine name the packed tree must expose, overridable because it identifies a
#: specific pack. The default names the gfx950 attention-dense engine shipped under the
#: provider's production root
#: (`src/engines/kernel_ingestor_engine/descriptors/<producer>/<bundle>/`), the only
#: place a bundle reaches the PACKED tree `_find_build_artifacts` probes. An author
#: gating their own bundle points `HIPDNN_INGESTOR_ENGINE` at its engine; naming one
#: only a `configs/` file mentions would reject every build dir.
_EXPECT_ENGINE = os.environ.get(
    "HIPDNN_INGESTOR_ENGINE", "hipkernel:Gfx950AttentionDense"
)


def _manifest_from_env() -> Path | None:
    """The first of `_MANIFEST_VARS` naming an existing file; `_SHIPPED_MANIFEST` when
    none is set; None when set but naming no file."""
    if not any(os.environ.get(var) for var in _MANIFEST_VARS):
        return _SHIPPED_MANIFEST
    for var in _MANIFEST_VARS:
        raw = os.environ.get(var)
        if raw and Path(raw).is_file():
            return Path(raw)
    return None


_MANIFEST = _manifest_from_env()


def _loaded_engines(validator: Path, packed: Path) -> list[str] | None:
    """Engine names the validator reports for `packed`, or None when the probe could not
    run: an unusable validator and an empty catalog are different answers."""
    try:
        probe = subprocess.run(
            [str(validator), str(packed), "--json"],
            capture_output=True,
            text=True,
            # The real validator answers in ~0.12s.
            timeout=15,
        )
        return json.loads(probe.stdout).get("engines", [])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _assert_expected_engine(engines: list[str] | None, packed: Path) -> None:
    """Fail, naming what the packed tree does expose, when `_EXPECT_ENGINE` is gone: a
    dropped engine leaves every file count unchanged, so the missing name is the only
    observable, and a build was found, so this is an artifact defect."""
    if engines is None:
        raise AssertionError(
            f"the validator could not be probed against {packed}; a build is present, "
            "so this is a broken artifact, not a missing prerequisite"
        )
    assert _EXPECT_ENGINE in engines, (
        f"{_EXPECT_ENGINE} is absent from the packed tree SELECTED at {packed}; the "
        f"validator loaded {sorted(engines)}. Discovery takes the FIRST build*/ "
        "carrying both a validator and a packed tree, so where several build "
        "directories exist the one named above may be a stale build shadowing the "
        "one under test -- check that before the others; then either set "
        "HIPDNN_INGESTOR_ENGINE to the engine this build actually packs, or repack "
        "the bundle that declares it."
    )


def _find_build_artifacts(
    repo_root: Path = _REPO_ROOT,
) -> tuple[Path | None, Path | None]:
    """(validator, packed tree) from the first `build*/` carrying both, by PRESENCE.

    Selecting on `_EXPECT_ENGINE` would collapse "nothing was built here" into "a build
    is present and the engine was dropped"; engine survival is asserted inside the test
    that depends on it.
    """
    for candidate in sorted(repo_root.glob("build*")):
        # Both spellings, because the executable suffix is the platform's: a bare name
        # matches nothing on Windows, so the skip below would report no build on a tree
        # that has one.
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
        if validator and packed.is_dir():
            return validator, packed
    return None, None


_VALIDATOR, _PACKED = _find_build_artifacts()

_NEEDS_BUILD_REASON = (
    "needs a build*/ carrying the descriptor validator beside a packed tree "
    "(configure with HIPDNN_ENABLE_KERNEL_INGESTOR=ON), and an engine manifest: "
    f"{_SHIPPED_MANIFEST.name} unless one of {', '.join(_MANIFEST_VARS)} (first wins) "
    "names the existing manifest that build was packed from"
)


def _missing_prerequisite(validator: Path | None, manifest: Path | None) -> str | None:
    """The reason the end-to-end class cannot run, or None. Only genuinely absent inputs
    belong here; a present build missing `_EXPECT_ENGINE` is a failure, not a skip."""
    if validator is None or manifest is None:
        return _NEEDS_BUILD_REASON
    return None


_needs_build = pytest.mark.skipif(
    _missing_prerequisite(_VALIDATOR, _MANIFEST) is not None, reason=_NEEDS_BUILD_REASON
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


def _write_bundle(
    root: Path,
    stem: str = "test_engine",
    ids: tuple[str, str, str, str] = (
        _KMD_ID,
        _UED_ID,
        "66666666-6666-6666-6666-666666666666",
        "77777777-7777-7777-7777-777777777777",
    ),
    block_n: int = 64,
    name: str | None = None,
) -> None:
    """One id-wired, packed bundle `<stem>.*.json`; `ids` is (KMD, UED, KDP, UKD)."""
    kmd_id, ued_id, kdp_id, ukd_id = ids
    (root / f"{stem}.kmd.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": kmd_id,
                "fields": [{"name": "block_n", "type": "int", "default_value": 64}],
            }
        )
    )
    (root / f"{stem}.ued.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": ued_id,
                "name": name or f"test:{stem}",
                "metadata": kmd_id,
            }
        )
    )
    kdp = {
        "version": "1.0",
        "id": kdp_id,
        "engine": ued_id,
        "arch": ["gfx942"],
        "kernelDescriptors": [
            {
                "version": "1.0",
                "id": ukd_id,
                "name": f"{stem}_k0",
                "arch": ["gfx942"],
                "kernel_source": {
                    "kind": "kpack",
                    "library": "kpack/test.kpack",
                    "toc_key": "v0",
                    "symbol": f"{stem}_s0",
                    "sha256": "a" * 64,
                },
                "metadata": {"block_n": block_n},
                "provenance": {},
            }
        ],
    }
    # Packed trees carry each UKD's provenance in a sidecar beside the KDP, and the
    # packer's marker in its directory.
    sidecar, data = provenance_sidecar.detach(f"{stem}.kdp.json", kdp)
    (root / sidecar).write_bytes(data)
    (root / provenance_sidecar.PACKED_MARKER).write_bytes(b"")
    (root / f"{stem}.kdp.json").write_text(json.dumps(kdp))


def _minimal_tree(tmp_path: Path, name: str = "descriptors") -> Path:
    """A structurally-valid, id-wired bundle, so rung 1 can pass without a build: the
    static rung reaches the schema through `KDP.engine -> UED.metadata -> KMD`."""
    root = tmp_path / name
    root.mkdir()
    _write_bundle(root)
    return root


class TestTheManifestSelectsTheBundleToGate:
    """Two engines under one root, one carrying a `-1` sentinel: `--manifest` picks the
    bundle rung 1 gates by its `engine.name`."""

    @staticmethod
    def _two_bundle_tree(root: Path, broken: str) -> Path:
        root.mkdir()
        for stem, name in (
            ("gfx950_attention_dense", "hipkernel:Gfx950AttentionDense"),
            ("other_engine", None),
        ):
            _write_bundle(
                root,
                stem,
                ids=tuple(f"{stem}-{part}" for part in ("kmd", "ued", "kdp", "ukd")),
                block_n=-1 if stem == broken else 64,
                name=name,
            )
        return root

    def test_the_shipped_manifest_gates_its_own_bundle(self, tmp_path):
        def static(tag: str, broken: str, *manifest: str) -> str:
            root = self._two_bundle_tree(tmp_path / tag, broken)
            result = _run("--tree", str(root), "--mode", "structural", *manifest)
            return result.stdout

        shipped = ("--manifest", str(_SHIPPED_MANIFEST))
        other_broken = static("other_broken", "other_engine", *shipped)
        own_broken = static("own_broken", "gfx950_attention_dense", *shipped)
        unselected = static("unselected", "other_engine")

        assert "1. STATIC   PASS (STRUCTURAL ONLY" in other_broken, other_broken
        assert "1. STATIC   FAIL" in own_broken, own_broken
        assert "gfx950_attention_dense_k0" in own_broken, own_broken
        assert "1. STATIC   FAIL" in unselected, unselected
        assert "Set engine.name in the manifest" in unselected, unselected


class TestTheStaticRungNeverOverstatesItself:
    """The failure defended against is the weaker answer printed under the stronger
    one's name: a run that read no compiled evidence, reported "1. STATIC PASS", and let
    a reader conclude the shipped binaries match the metadata selecting them."""

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
        """The minimal tree carries no producing-build evidence at all."""
        result = _run("--tree", str(_minimal_tree(tmp_path)), "--mode", "full")
        assert result.returncode != 0
        assert "1. STATIC   FAIL" in result.stdout
        assert "static" in result.stdout

    def test_rung_one_reads_sidecars_from_the_provenance_root(self, tmp_path):
        """An installed tree keeps its sidecars apart; the static rung reads them from
        the mirrored root it is given, and without it refuses the packed tree."""
        root = _minimal_tree(tmp_path)
        provenance = tmp_path / "provenance"
        provenance.mkdir()
        sidecar = root / provenance_sidecar.sidecar_name("test_engine.kdp.json")
        sidecar.rename(provenance / sidecar.name)

        found = _run(
            "--tree",
            str(root),
            "--mode",
            "structural",
            "--provenance-root",
            str(provenance),
        )
        missing = _run("--tree", str(root), "--mode", "structural")

        assert "1. STATIC   PASS (STRUCTURAL ONLY" in found.stdout, found.stdout
        assert "1. STATIC   FAIL" in missing.stdout
        assert "has no provenance sidecar" in missing.stdout

    def test_a_provenance_root_for_an_unmarked_tree_fails_rung_one(self, tmp_path):
        """The root only relocates a packed tree's sidecars. The sidecar waits under
        it here, so a gate taking the root to mean packed would pass."""
        root = _minimal_tree(tmp_path)
        provenance = tmp_path / "provenance"
        provenance.mkdir()
        sidecar = root / provenance_sidecar.sidecar_name("test_engine.kdp.json")
        sidecar.rename(provenance / sidecar.name)
        (root / provenance_sidecar.PACKED_MARKER).unlink()

        result = _run(
            "--tree",
            str(root),
            "--mode",
            "structural",
            "--provenance-root",
            str(provenance),
        )

        assert "1. STATIC   FAIL" in result.stdout
        assert "usage error: --provenance-root" in result.stdout


class TestRungsStaySeparable:
    def test_a_missing_validator_fails_rather_than_skipping_to_a_pass(self, tmp_path):
        """Rung 1 passing must not imply rung 2."""
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

    def test_a_serving_threshold_this_tool_cannot_check_fails_the_gate(self, tmp_path):
        """Rungs 1 and 2 green, rung 3 not run: asking for served graphs must not
        pass on that evidence."""
        validator = tmp_path / "validator"
        validator.write_text(
            f'#!{sys.executable}\nprint(\'{{"engines": ["stub"]}}\')\n'
        )
        validator.chmod(0o755)
        argv = ["--tree", str(_minimal_tree(tmp_path)), "--mode", "structural"]
        argv += ["--validator", str(validator)]
        without = _run(*argv)
        assert without.returncode == 0, without.stdout + without.stderr
        asked = _run(*argv, "--min-served", "1")
        assert asked.returncode == 1
        failed = next(l for l in asked.stdout.splitlines() if "GATE FAILED" in l)
        assert "serving-not-run" in failed

    def test_a_missing_tree_is_an_error_not_an_empty_pass(self, tmp_path):
        result = _run("--tree", str(tmp_path / "nope"), "--mode", "structural")
        assert result.returncode == 2


class TestDiscoverySeparatesNoBuildFromADroppedEngine:
    """Discovery gated on the expected engine makes a dropped engine indistinguishable
    from a machine that never built anything, and the opt-in class below then SKIPS. The
    property is the DISTINCTION, so both trees are built here and compared."""

    @staticmethod
    def _build_tree(root: Path, *, present: bool) -> Path:
        """A `build*/` candidate shaped exactly as discovery reads it."""
        candidate = root / "build"
        candidate.mkdir(parents=True, exist_ok=True)
        if present:
            binary = candidate / "bin" / "hipdnn_validate_descriptors.exe"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"")
            (candidate / "lib/hipdnn_plugins/engines/arch_content").mkdir(parents=True)
        return candidate

    @staticmethod
    def _a_manifest(tmp_path: Path) -> Path:
        """An existing manifest, so each case isolates the build half of the pair."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        return manifest

    def test_an_absent_build_skips_where_a_dropped_engine_fails(self, tmp_path):
        """Both trees in one case: only the pair rules out a discovery rule that answers
        "no build" to both."""
        manifest = self._a_manifest(tmp_path)

        absent = tmp_path / "absent"
        self._build_tree(absent, present=False)
        no_build = _find_build_artifacts(absent)
        assert no_build == (None, None)
        assert _missing_prerequisite(no_build[0], manifest) is not None, (
            "nothing on disk to run the gate against is an absent prerequisite, and "
            "an opt-in class skips on it"
        )

        dropped = tmp_path / "dropped"
        self._build_tree(dropped, present=True)
        validator, packed = _find_build_artifacts(dropped)
        assert (validator, packed) != no_build, (
            "a validator binary beside a packed directory is a build; reporting it "
            "as absent turns the dropped-engine regression test into a skip"
        )
        assert _missing_prerequisite(validator, manifest) is None, (
            "the build is present, so the class runs and the dropped engine has to "
            "surface as a failure"
        )

        with pytest.raises(AssertionError) as caught:
            _assert_expected_engine(["hipkernel:SomethingElse"], packed)
        message = str(caught.value)
        assert _EXPECT_ENGINE in message
        assert "hipkernel:SomethingElse" in message, (
            "a dropped engine's only observable is the list it is missing from, so "
            "the failure has to print that list"
        )
        assert str(packed) in message, (
            "first-match discovery means a stale build*/ can shadow the one under "
            "test, so the failure has to name the tree it actually selected"
        )
        assert "stale build" in message, (
            "a reader whose build was shadowed follows the remedies as written; "
            "neither HIPDNN_INGESTOR_ENGINE nor a repack fixes that cause, so it "
            "has to be offered alongside them"
        )


@_needs_build
class TestAgainstTheRealBuild:
    """The whole gate end to end against a packed tree an author built. Opt-in on a
    build; the classes above pin the rung-separation property everywhere."""

    def test_packed_tree_passes_both_runnable_rungs(self):
        # Checked by name first so a dropped engine reports itself instead of
        # arriving as an opaque nonzero exit from the gate.
        _assert_expected_engine(_loaded_engines(_VALIDATOR, _PACKED), _PACKED)
        result = _run(
            "--tree",
            str(_PACKED),
            "--mode",
            "full",
            "--manifest",
            str(_MANIFEST),
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
        """The dropped-engine case: the file count is unchanged and the exit code would
        be 0, so a name missing from the loaded list is the only observable."""
        result = _run(
            "--tree",
            str(_PACKED),
            "--mode",
            "full",
            "--manifest",
            str(_MANIFEST),
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
        """`kind: rocke` is an AUTHORING form hkp_pack lowers to `kind: kpack`; the
        runtime loader does not know `builder`, so the authored tree fails rung 2."""
        root = _minimal_tree(tmp_path, "authored")
        kdp_path = root / "test_engine.kdp.json"
        doc = json.loads(kdp_path.read_text())
        # Only the dialect changes, so the tree stays id-wired, rung 1 still passes, and
        # the failure belongs unambiguously to rung 2.
        doc["kernelDescriptors"][0]["kernel_source"] = {
            "kind": "rocke",
            "source": "m.py",
            "builder": "build_x",
            "spec": {"block_n": 64},
        }
        kdp_path.write_text(json.dumps(doc))
        # An authored tree carries no marker; one left from the packed fixture would
        # read the tree as packed, bind the sidecar to a different UKD and fail rung 1.
        (root / provenance_sidecar.PACKED_MARKER).unlink()
        result = _run(
            "--tree", str(root), "--mode", "structural", "--validator", str(_VALIDATOR)
        )
        assert result.returncode != 0
        assert "2. LOADS    FAIL" in result.stdout
