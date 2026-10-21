# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build, qualify, and verify pinned live GPU reference bundles for an operation.

Each ``<operation>_reference/cli.py`` declares a ``ReferenceOperation``: its case
contract, architecture adapters, metric, optional Torch answers, and manifest
description. Snapshot export, bundle freezing, authentication, cohort matching,
and the old/current comparison are shared, so every operation certifies its
original tolerance in the same way. The snapshot and qualification commands are
offline maintenance operations. Verification only executes a locked bundle and
current kernels; it never downloads dependencies, generates independent answers,
or promotes a baseline. The cases and diff commands only read the enrolled cohort
and an authenticated manifest, so an author can review a cohort change on a host
without a GPU.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .artifact import display_command, manifest_operation, unpack
from .cases import compare_manifest_cases, describe_case, match_manifest_cases
from .numeric import (
    ErrorBudget,
    array_digest,
    decode,
    file_digest,
    payload_digests,
    write_json,
)
from .paths import default_bundle_path, installed_test_root
from .runner import run_worker
from .session import reuse_workers
from .source import snapshot
from .targets import processor_from_target_id

_WORKFLOW = "library/tests/reference_common/docs/gpu-reference-workflow.md"
# The workflow's triage table says who acts on each failure and links its procedure.
_TRIAGE = f"{_WORKFLOW}#triage"
# Failures name the side that failed: verification replays the frozen baseline
# and runs the current kernel; qualification runs a baseline source snapshot.
_BASELINE = "frozen baseline (replay)"
_CURRENT = "current kernel (source)"
_SNAPSHOT = "baseline snapshot (qualification)"
# Reading a truncated, corrupt, or non-gzip archive raises these; tarfile wraps
# only some of them in ReadError.
_ARCHIVE_ERRORS = (EOFError, gzip.BadGzipFile, tarfile.TarError, zlib.error)


def _triage(problem: str, cause: str, step: str) -> str:
    """Follow a failure with its likely cause and the next step to take."""
    return f"{problem}\n  likely cause: {cause}\n  next step: {step}\n  see {_TRIAGE}"


def _no_parameters(reference: np.ndarray) -> dict[str, Any]:
    return {}


def _budget_policy(manifest: dict) -> str | None:
    policy = manifest.get("budget_policy")
    if policy not in (None, "max-reference-bound-v1"):
        raise ValueError("unsupported reference budget policy")
    return policy


def _shown(path: Path) -> str:
    """Show a path relative to the working directory when it lies inside it."""
    path, here = path.resolve(), Path.cwd().resolve()
    return str(path.relative_to(here) if path.is_relative_to(here) else path)


@dataclass(frozen=True)
class Metric:
    """An operation's comparison against its original tolerance.

    Qualification records ``parameters(reference)`` in each case entry, such as
    a frozen reference scale, so CI never recomputes the independent answer.
    ``distance(left, right, entry)`` bounds the error between two represented
    outputs using an entry's recorded parameters and rejects invalid ones.
    Verification reports the largest current distance under ``name``.
    """

    name: str
    distance: Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], float]
    parameters: Callable[[np.ndarray], dict[str, Any]] = _no_parameters


@dataclass(frozen=True)
class TorchReference:
    """Optional offline Torch answers that qualification also bounds.

    ``devices`` lists the accepted Torch devices, the default first; with one
    device, ``--torch-reference`` takes no value. ``prepare(device, architecture)``
    fails before qualification creates any output if Torch cannot run there, and
    returns ``cross_check(case, inputs, reference, baseline, parameters)``. Its
    report gives every Torch answer's ``baseline_error_bound``.
    """

    devices: tuple[str, ...]
    help: str
    prepare: Callable[[str, str], Callable[..., dict]]


@dataclass(frozen=True)
class ReferenceOperation:
    """The operation-specific half of a pinned GPU reference.

    ``name`` selects the ``<name>_reference`` package, its worker module, its
    bundle directory, and the manifest operation id in its ``operation.json``.
    ``label`` names the operation in messages and ``description`` documents its
    command line. ``contract`` provides ``SCHEMA_VERSION``, ``INPUT_GENERATOR``,
    ``make_inputs``, and ``independent_reference``; ``architectures`` provides
    ``ARCHITECTURES``, ``get_architecture``, and ``baseline_lock``. ``reference``
    describes the independent reference's implementation, metric, and input
    generator in each manifest. Without ``torch``, qualification bounds only the
    independent reference and offers no ``--torch-reference`` option.
    """

    name: str
    label: str
    description: str
    contract: ModuleType
    architectures: ModuleType
    reference: Mapping[str, str]
    metric: Metric
    torch: TorchReference | None = None
    # Published SDPA manifests predate the operation field. SDPA qualification
    # keeps omitting it, so new manifests keep the published form.
    declares_operation: bool = True

    @property
    def worker_module(self) -> str:
        """The module that frozen and current interpreters execute per case."""
        return f"{self.name}_reference.worker"

    @property
    def _package(self) -> Path:
        return Path(self.contract.__file__).resolve().parent

    def _worker(
        self,
        request: dict,
        *,
        side: str,
        case: Any,
        runner: Path,
        platform: Path,
        library: Path | None,
        work: Path,
    ) -> tuple[list[np.ndarray], dict]:
        try:
            return run_worker(
                request,
                runner=runner,
                platform=platform,
                library=library,
                work=work,
                module=self.worker_module,
            )
        except (RuntimeError, TimeoutError) as error:
            # The message keeps the worker's log tail, which holds the cause; the
            # parent's own traceback would only repeat it.
            kind = TimeoutError if isinstance(error, TimeoutError) else RuntimeError
            raise kind(
                self._worker_failure(case, request["architecture"], side, error)
            ) from None

    def _worker_failure(
        self, case: Any, architecture: str, side: str, error: Exception
    ) -> str:
        adapter = (
            f"library/tests/{self.name}_reference/architectures/{architecture}/"
            "__init__.py"
        )
        if side == _BASELINE:
            cause = (
                "this environment cannot run the frozen runtime and code object "
                "(GPU driver, ROCm runtime, or device visibility); the current "
                "kernel did not run"
            )
            step = (
                "read the worker log below and compare this environment with the "
                "one the bundle was qualified in"
            )
        elif side == _CURRENT:
            cause = (
                "the reference adapter no longer matches the current sources, such "
                "as a renamed or re-signed dispatch, build, or launch API, or the "
                "current kernel no longer compiles or launches once per repetition"
            )
            step = (
                "read the worker log below; if a production API changed, update "
                f"{adapter} to call the current one"
            )
        else:
            cause = (
                "the reference adapter does not support the snapshot's production "
                "API, or the baseline kernel does not compile or launch on this GPU"
            )
            step = (
                "read the worker log below; select another committed baseline or "
                f"extend {adapter} for its API"
            )
        problem = (
            f"{self.label} {architecture} case {case.id}: the {side} worker failed"
        )
        return f"{_triage(problem, cause, step)}\n{error}"

    def freeze_runner(self, destination: Path) -> None:
        """Copy the replay runner and its implementation dependencies into a bundle."""
        runner = destination / self._package.name
        runner.mkdir(parents=True)
        for name in ("__init__.py", "contract.py", "worker.py"):
            shutil.copy2(self._package / name, runner / name)
        for name in ("architectures", "implementations"):
            shutil.copytree(
                self._package / name,
                runner / name,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", "baseline_lock.json"
                ),
            )
        # Workers never import the pytest-only helpers, which need pytest.
        shutil.copytree(
            self._package.parent / "reference_common",
            destination / "reference_common",
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "pinned_tests.py", "pytest_support.py"
            ),
        )

    def checked_inputs(
        self, case: Any, digests: dict[str, str]
    ) -> dict[str, np.ndarray]:
        """Require generated input bytes to match the independently qualified corpus."""
        arrays = self.contract.make_inputs(case)
        if {name: array_digest(array) for name, array in arrays.items()} != digests:
            raise ValueError(
                f"generated {self.label} input digest mismatch: {case.id}; "
                "the generator or NumPy implementation differs from qualification"
            )
        return arrays

    def qualify(
        self,
        baseline: Path,
        output: Path,
        repetitions: int,
        architecture: str = "gfx942",
        *,
        torch_reference: str | None = None,
    ) -> None:
        """Measure the old version's bounds and require deterministic old outputs."""
        target = self.architectures.get_architecture(architecture)
        if torch_reference is not None:
            if self.torch is None:
                raise ValueError(f"{self.label} qualification has no Torch reference")
            if torch_reference not in self.torch.devices:
                raise ValueError(
                    f"{self.label} Torch reference device must be one of "
                    + ", ".join(self.torch.devices)
                )
            cross_check = self.torch.prepare(torch_reference, architecture)
        if repetitions < 2:
            raise ValueError("qualification requires at least two old executions")
        metadata = json.loads((baseline / "snapshot.json").read_text())
        source = baseline / "source"
        if payload_digests(source) != metadata["files"]:
            raise ValueError("baseline sources no longer match the committed snapshot")
        output.mkdir(parents=True, exist_ok=False)
        payload = output / "payload"
        payload.mkdir()
        shutil.copytree(
            source / "platform/python/rocke",
            payload / "runtime/rocke",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        self.freeze_runner(payload / "runner")
        entries = {}
        for case in target.CASES:
            case_dir = payload / "cases" / case.id
            case_dir.mkdir(parents=True)
            inputs = self.contract.make_inputs(case)
            input_digests = {
                name: array_digest(array) for name, array in inputs.items()
            }
            with tempfile.TemporaryDirectory(
                prefix=f"rocke-{self.name}-qualify-"
            ) as temporary:
                input_file = Path(temporary) / "inputs.npz"
                np.savez(input_file, **inputs)
                outputs, report = self._worker(
                    {
                        "mode": "source",
                        "architecture": architecture,
                        "case": asdict(case),
                        "inputs": str(input_file),
                        "input_digests": input_digests,
                        "export": str(case_dir / "kernel.hsaco"),
                        "repetitions": repetitions,
                    },
                    side=_SNAPSHOT,
                    case=case,
                    runner=self._package.parent,
                    platform=source / "platform/python",
                    library=source / "library",
                    work=Path(temporary) / "worker",
                )
            digests = {array_digest(array) for array in outputs}
            if len(digests) != 1:
                raise ValueError(
                    _triage(
                        f"{self.label} {architecture} case {case.id}: the "
                        f"{_SNAPSHOT} produced {len(digests)} different outputs in "
                        f"{repetitions} executions; old {self.label} output is not "
                        "deterministic",
                        "the baseline kernel's result depends on execution order "
                        "on this GPU, such as through atomics or a race",
                        "qualify a committed baseline whose kernel is deterministic "
                        "for this case",
                    )
                )
            reference = self.contract.independent_reference(case, inputs)
            parameters = self.metric.parameters(reference)
            old = decode(outputs[0], case.dtype)
            bound = self.metric.distance(old, reference, parameters)
            torch_report = None
            if torch_reference is not None:
                torch_report = cross_check(case, inputs, reference, old, parameters)
                bound = max(
                    bound,
                    *(
                        r["baseline_error_bound"]
                        for r in torch_report["results"].values()
                    ),
                )
            # B covers every enabled independent reference on the same inputs.
            budget = ErrorBudget(case.tolerance, bound)
            entries[case.id] = {
                "case": asdict(case),
                "input_digests": input_digests,
                "output_digest": digests.pop(),
                "reference_digest": array_digest(reference),
                **parameters,
                "budget": asdict(budget),
                "comparison_limit": budget.comparison_limit,
                "kernel": report["kernel"],
                "device_target": report["device_target"],
                "qualification_repetitions": repetitions,
                "compiler": report["compiler"],
            }
            if torch_report is not None:
                entries[case.id]["torch_reference"] = torch_report
            print(
                f"QUALIFIED {case.id}: old_bound={bound:.9g}, "
                f"comparison_limit={budget.comparison_limit:.9g}, "
                f"tolerance={case.tolerance:.9g}",
                flush=True,
            )
        manifest = {
            "schema": self.contract.SCHEMA_VERSION,
            "budget_policy": "max-reference-bound-v1",
            "input_generation": self.contract.INPUT_GENERATOR,
            "baseline_revision": metadata["revision"],
            "baseline_snapshot_sha256": file_digest(baseline / "snapshot.json"),
            "reference": {
                **self.reference,
                "numpy_version": np.__version__,
                "python_version": sys.version.split()[0],
                "contract_sha256": file_digest(self._package / "contract.py"),
            },
            "cases": entries,
            "files": payload_digests(payload),
        }
        if self.declares_operation:
            manifest["operation"] = manifest_operation(self.name)
        write_json(output / "manifest.json", manifest)
        write_json(
            output / "qualification-lock.json",
            {
                "schema": self.contract.SCHEMA_VERSION,
                "baseline_revision": metadata["revision"],
                "manifest_sha256": file_digest(output / "manifest.json"),
            },
        )
        print(
            "Qualification complete; review qualification-lock.json before promotion."
        )

    def load_bundle(
        self,
        bundle: Path,
        lock_path: Path | None = None,
        *,
        architecture: str = "gfx942",
    ) -> dict:
        """Require the independently pinned manifest, payload, and exact case cohort."""
        manifest = self._authenticated_manifest(bundle, lock_path, architecture)
        self.validate_cases(manifest, architecture)
        return manifest

    def _authenticated_manifest(
        self, bundle: Path, lock_path: Path | None, architecture: str
    ) -> dict:
        """Require the pinned manifest and payload; the cohort is checked separately.

        A bundle the lock rejects, such as one extracted from an earlier archive,
        fails with the commands that replace it.
        """
        self.architectures.get_architecture(architecture)
        schema = self.contract.SCHEMA_VERSION
        lock_path = lock_path or self.architectures.baseline_lock(architecture)
        lock = json.loads(lock_path.read_text())
        if lock["schema"] != schema:
            raise ValueError(f"unsupported {self.label} lock schema")
        if bundle.is_file():
            raise ValueError(self.bundle_hint(architecture, bundle, lock_path))
        manifest_path = bundle / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                self._rejected_bundle(
                    f"{self.label} bundle {_shown(bundle)} has no manifest.json",
                    architecture,
                    bundle,
                    lock_path,
                )
            )
        digest = file_digest(manifest_path)
        if digest != lock["manifest_sha256"]:
            raise ValueError(
                self._rejected_bundle(
                    f"{self.label} bundle manifest does not match the pinned lock: "
                    f"{_shown(manifest_path)} has sha256 {digest[:12]}, but "
                    f"{_shown(lock_path)} pins {lock['manifest_sha256'][:12]}",
                    architecture,
                    bundle,
                    lock_path,
                )
            )
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["schema"] != schema
            or manifest["baseline_revision"] != lock["baseline_revision"]
        ):
            raise ValueError(
                f"{self.label} baseline identity mismatch: the manifest records "
                f"schema {manifest['schema']} at {manifest['baseline_revision']}, "
                f"but {_shown(lock_path)} pins schema {lock['schema']} at "
                f"{lock['baseline_revision']}"
            )
        files = payload_digests(bundle / "payload")
        if files != manifest["files"]:
            changed = sorted(
                name
                for name in files.keys() | manifest["files"].keys()
                if files.get(name) != manifest["files"].get(name)
            )
            raise ValueError(
                self._rejected_bundle(
                    f"{self.label} bundle payload has missing, modified, or extra "
                    f"files: {len(changed)} differ, including "
                    + ", ".join(changed[:3]),
                    architecture,
                    bundle,
                    lock_path,
                )
            )
        expected = manifest_operation(self.name)
        # As in artifact.py, the selected lock identifies an undeclared manifest.
        undeclared = None if self.declares_operation else expected
        declared = manifest.get("operation", undeclared)
        if declared != expected:
            raise ValueError(
                f"wrong reference operation: the manifest declares {declared!r}, "
                f"but {self.label} bundles declare {expected!r}"
            )
        if manifest.get("input_generation") != self.contract.INPUT_GENERATOR:
            raise ValueError(f"unsupported {self.label} input generator contract")
        if any(p.is_file() and p.suffix in (".npz", ".npy") for p in bundle.rglob("*")):
            raise ValueError(
                f"generated-input {self.label} bundles must not contain tensor files"
            )
        return manifest

    def _rejected_bundle(
        self, problem: str, architecture: str, bundle: Path, lock: Path
    ) -> str:
        return _triage(
            problem,
            "the extracted bundle is stale (left from an earlier archive), partly "
            "extracted, or modified, or the lock does not belong to it",
            self.bundle_hint(architecture, bundle, lock, stale=True),
        )

    def bundle_hint(
        self,
        architecture: str,
        bundle: Path,
        lock: Path | None = None,
        *,
        stale: bool = False,
        option: str = "--bundle",
    ) -> str:
        """Tell an author how to put a usable extracted bundle at ``bundle``.

        An archive passed in place of the directory gets the command that unpacks
        it. Otherwise a source checkout gets the commands that fetch and extract
        the published archive tracked by the DVC pointer beside the default
        bundle. An installed tree has no pointers or archives; installation
        stages its published bundles, so it gets the installation's requirements
        instead. An architecture without a pointer in a source checkout, or
        without a committed lock in an installed tree, is unpublished and needs
        a qualified candidate. ``stale`` asks for the directory's removal first,
        since neither unpack nor installation replaces an existing bundle.
        ``option`` names the setting that selects a bundle.
        """
        committed = self.architectures.baseline_lock(architecture)
        lock = lock or committed
        if bundle.is_file():
            return self._unpack_hint(bundle, lock, option=option)
        if lock.resolve() != committed.resolve():
            return (
                "pass a candidate bundle directory together with the "
                "qualification-lock.json that qualification wrote with it"
            )
        tests = self._package.parent
        published = default_bundle_path(tests, architecture, operation=self.name)
        if installed_test_root(tests) is not None:
            if not committed.is_file():
                return (
                    f"no published {architecture} {self.label} bundle is pinned at "
                    f"{_shown(committed)}, so installation stages none; an "
                    "unpublished architecture needs a qualified candidate bundle "
                    "and its lock"
                )
            if stale:
                return (
                    f"delete {_shown(bundle)}, then reinstall the tests from one "
                    f"build; installation stages the published {architecture} "
                    f"{self.label} bundle at {_shown(published)} from the archive "
                    "that the installed lock pins"
                )
            return (
                f"installed trees receive the published {architecture} {self.label} "
                f"bundle at {_shown(published)} only when built with "
                "ROCKE_INSTALL_TEST_GPU_REFERENCES=ON and assembled with its "
                "architecture's test artifact"
            )
        archive = published.with_name(f"{published.name}.tar.gz")
        pointer = archive.with_name(f"{archive.name}.dvc")
        if not pointer.is_file():
            return (
                f"no published {architecture} {self.label} archive is tracked at "
                f"{_shown(pointer)}; an unpublished architecture needs a qualified "
                "candidate bundle and its lock"
            )
        lead = (
            f"delete {_shown(bundle)}, then re-run"
            if stale
            else "fetch and extract it with"
        )
        return (
            f"{lead} dvc pull and artifact.py unpack:\n"
            f"    {display_command(['dvc', 'pull', _shown(pointer)])}\n"
            f"    {self._unpack_command(archive, bundle, lock)}"
        )

    def validate_cases(self, manifest: dict, architecture: str) -> None:
        """Preserve the cohort and budgets across qualification and verification."""
        policy = _budget_policy(manifest)
        target = self.architectures.get_architecture(architecture)
        keys = match_manifest_cases(manifest["cases"], target.CASES, label=self.label)
        for case in target.CASES:
            entry = manifest["cases"][keys[case.id]]
            problem = self._entry_problem(case, entry, target, architecture, policy)
            if problem is not None:
                raise ValueError(f"{self.label} {problem}")

    def cohort_differences(self, manifest: dict, architecture: str) -> list[str]:
        """Describe every way the enrolled cohort differs from a manifest's cases.

        This applies ``validate_cases``'s contract matching and per-case checks,
        but collects each difference instead of raising at the first, so an
        author can review a cohort edit on the host before qualifying it. An
        empty list means verification accepts the manifest's cases.
        """
        policy = _budget_policy(manifest)
        target = self.architectures.get_architecture(architecture)
        comparison = compare_manifest_cases(
            manifest["cases"], target.CASES, label=self.label
        )
        differences = [
            f"enrolled but not in the bundle: {describe_case(*item)}"
            for item in comparison.missing.items()
        ]
        differences += [
            f"in the bundle but not enrolled: {describe_case(*item)}"
            for item in comparison.extra.items()
        ]
        differences += [
            f"recorded more than once: {case_id} ({', '.join(keys)})"
            for case_id, keys in comparison.repeated.items()
        ]
        for case in target.CASES:
            if case.id in comparison.keys:
                entry = manifest["cases"][comparison.keys[case.id]]
                problem = self._entry_problem(case, entry, target, architecture, policy)
                if problem is not None:
                    differences.append(problem)
        return differences

    def _entry_problem(
        self,
        case: Any,
        entry: dict,
        target: ModuleType,
        architecture: str,
        policy: str | None,
    ) -> str | None:
        """Explain why a matched manifest entry cannot certify a case, if it cannot."""
        if processor_from_target_id(entry["device_target"]) != architecture:
            return f"bundle targets a different architecture: {case.id}"
        compiler_target = getattr(target, "COMPILER_TARGET", None)
        if (
            compiler_target is not None
            and entry["kernel"].get("compiler_target") != compiler_target
        ):
            return f"bundle has a different compiler target: {case.id}"
        # Reject recorded metric parameters, such as a scale, before any GPU run.
        self.metric.distance(np.zeros(1), np.zeros(1), entry)
        budget = ErrorBudget(**entry["budget"])
        if (
            budget.tolerance != case.tolerance
            or budget.margin != (case.tolerance / 20 if policy is None else 0.0)
            or entry["comparison_limit"] != budget.comparison_limit
        ):
            return (
                f"tolerance or budget changed: {case.id} (enrolled tolerance "
                f"{case.tolerance:g}; recorded tolerance {budget.tolerance:g}, "
                f"margin {budget.margin:g})"
            )
        return None

    def diff_bundle(
        self, bundle: Path, lock_path: Path | None = None, *, architecture: str
    ) -> list[str]:
        """Compare the enrolled cohort with a bundle directory or packed archive.

        An archive is unpacked into a temporary directory by ``artifact.unpack``,
        which rejects unsafe members and authenticates the manifest and payload
        against the lock before the manifest is read. An archive that cannot be
        decompressed or read as tar, such as a truncated fetch or a DVC pointer,
        raises ``ValueError`` with the step that replaces it.
        """
        lock = lock_path or self.architectures.baseline_lock(architecture)
        if bundle.is_dir():
            manifest = self._authenticated_manifest(bundle, lock, architecture)
        else:
            with tempfile.TemporaryDirectory(
                prefix=f"rocke-{self.name}-diff-"
            ) as temporary:
                staged = Path(temporary) / "bundle"
                try:
                    unpack(bundle, staged, lock, operation=self.name)
                except _ARCHIVE_ERRORS as error:
                    raise ValueError(
                        self._unreadable_archive(bundle, architecture, error)
                    ) from None
                manifest = self._authenticated_manifest(staged, lock, architecture)
        return self.cohort_differences(manifest, architecture)

    def _unreadable_archive(
        self, archive: Path, architecture: str, error: Exception
    ) -> str:
        """Explain an archive that cannot be read and how to replace it."""
        directory = default_bundle_path(
            self._package.parent, architecture, operation=self.name
        )
        published = directory.with_name(f"{directory.name}.tar.gz")
        pointer = published.with_name(f"{published.name}.dvc")
        if archive.resolve() == published.resolve() and pointer.is_file():
            fetch = display_command(["dvc", "pull", _shown(pointer)])
            step = f"delete {_shown(archive)}, then fetch it again with: {fetch}"
        elif archive.suffix == ".dvc":
            fetch = display_command(["dvc", "pull", _shown(archive)])
            step = (
                "pass the archive that this DVC pointer tracks, "
                f"{_shown(archive.with_suffix(''))}, after fetching it with: {fetch}"
            )
        else:
            step = (
                "pass an extracted bundle directory or a .tar.gz archive written "
                "by artifact.py pack"
            )
        return _triage(
            f"{self.label} bundle archive {_shown(archive)} cannot be read: {error}",
            "the file is truncated or corrupt, such as after an interrupted fetch, "
            "or is not a bundle archive, such as a DVC pointer",
            step,
        )

    def published_bundle(self, architecture: str) -> Path:
        """Locate the published bundle data that this tree provides by default.

        A source checkout provides the DVC archive, else a bundle extracted
        beside it; an installed tree provides the bundle that CMake staged. The
        path need not exist.
        """
        directory = default_bundle_path(
            self._package.parent, architecture, operation=self.name
        )
        archive = directory.with_name(f"{directory.name}.tar.gz")
        return archive if archive.is_file() else directory

    def manifest_key(
        self, manifest: dict, case: Any, *, architecture: str = "gfx942"
    ) -> str:
        """Return the manifest key, which also names the payload directory, for a case.

        Published bundles may key cases by an earlier id format; the recorded case
        contract, not the key, identifies the entry.
        """
        target = self.architectures.get_architecture(architecture)
        keys = match_manifest_cases(manifest["cases"], target.CASES, label=self.label)
        return keys[case.id]

    def _current_paths(self, current_root: Path | None) -> tuple[Path, Path]:
        if current_root is not None:
            return current_root / "platform/python", current_root / "library"
        # Installed pytest already resolves the installed platform package. Library
        # tests are installed under that library's tests directory in both layouts.
        spec = importlib.util.find_spec("rocke")
        if spec is None or spec.origin is None:
            raise RuntimeError(
                "install the current rocKE platform or pass --current-root"
            )
        return Path(spec.origin).parent.parent, self._package.parent.parent

    def verify_case(
        self,
        case: Any,
        *,
        bundle: Path,
        manifest: dict,
        lock: Path | None = None,
        current_root: Path | None = None,
        repetitions: int = 2,
        architecture: str = "gfx942",
    ) -> dict:
        """Run old/current in separate interpreters and check every current output.

        Each failure names the side that failed, its measured values, the likely
        cause, and the next step. A changed frozen baseline output is a reference
        qualification failure, raised before the current kernel runs; only an
        out-of-budget current output implicates the current kernel. ``lock`` is
        the lock that authenticated ``manifest`` (default: the committed lock);
        the command that reruns a failing case checks the bundle against it.
        """
        if repetitions < 1:
            raise ValueError("verification must execute at least once")
        target = self.architectures.get_architecture(architecture)
        if case not in target.CASES:
            raise ValueError("case is not enrolled for the selected architecture")
        key = self.manifest_key(manifest, case, architecture=architecture)
        entry = manifest["cases"][key]
        subject = f"{self.label} {architecture} case {case.id}"
        payload = bundle / "payload"
        case_dir = payload / "cases" / key
        base = {
            "architecture": architecture,
            "case": asdict(case),
            "input_digests": entry["input_digests"],
            "repetitions": repetitions,
        }
        with tempfile.TemporaryDirectory(
            prefix=f"rocke-{self.name}-verify-"
        ) as temporary:
            temporary = Path(temporary)
            input_file = temporary / "inputs.npz"
            np.savez(input_file, **self.checked_inputs(case, entry["input_digests"]))
            base["inputs"] = str(input_file)
            old, old_report = self._worker(
                dict(
                    base,
                    mode="replay",
                    kernel=entry["kernel"],
                    hsaco=str(case_dir / "kernel.hsaco"),
                ),
                side=_BASELINE,
                case=case,
                runner=payload / "runner",
                platform=payload / "runtime",
                library=None,
                work=temporary / "old",
            )
            for repetition, array in enumerate(old, 1):
                digest = array_digest(array)
                if digest != entry["output_digest"]:
                    raise AssertionError(
                        _triage(
                            f"reference qualification failure for {subject}: the "
                            f"{_BASELINE} output of repetition {repetition} of "
                            f"{len(old)} has digest {digest[:12]}, but "
                            f"qualification recorded {entry['output_digest'][:12]}",
                            "the GPU driver, ROCm runtime, or firmware differs from "
                            "the qualified environment, or the frozen kernel is not "
                            "deterministic here; the current kernel did not run and "
                            "is not the cause",
                            "rerun to confirm that it reproduces and compare this "
                            "environment with the qualified one; do not change the "
                            "current kernel or its tolerance to make it pass",
                        )
                    )
            platform, library = self._current_paths(current_root)
            current, current_report = self._worker(
                dict(base, mode="source"),
                side=_CURRENT,
                case=case,
                runner=self._package.parent,
                platform=platform,
                library=library,
                work=temporary / "current",
            )
        if old_report["device_target"] != entry["device_target"]:
            raise ValueError(
                _triage(
                    f"{subject}: the {_BASELINE} ran on device target "
                    f"{old_report['device_target']}, but the bundle was qualified on "
                    f"{entry['device_target']}; the reference target differs from "
                    "the qualified target",
                    "this GPU's target features, such as xnack or sramecc, are "
                    "configured differently from the qualified GPU",
                    "run on a GPU configured like the qualified one, or qualify a "
                    "bundle for this target",
                )
            )
        if current_report["device_target"] != old_report["device_target"]:
            raise ValueError(
                _triage(
                    f"{subject}: the {_CURRENT} ran on device target "
                    f"{current_report['device_target']}, but the {_BASELINE} ran on "
                    f"{old_report['device_target']}; the workers used different "
                    "target features",
                    "the current and frozen runtimes report this GPU's target "
                    "features differently, or the workers' environments select "
                    "different features",
                    "compare the current runtime's get_device_target_id with the "
                    "frozen runtime's; both sides must run on the qualified target",
                )
            )
        budget = ErrorBudget(**entry["budget"])
        reference = decode(old[0], case.dtype)
        distances = [
            self.metric.distance(decode(array, case.dtype), reference, entry)
            for array in current
        ]
        for repetition, distance in enumerate(distances, 1):
            try:
                budget.check(distance, metric=self.metric.name)
            except AssertionError as error:
                rerun = self._rerun_command(
                    case, architecture, bundle, lock, current_root
                )
                raise AssertionError(
                    _triage(
                        f"{subject}: the {_CURRENT} output of repetition "
                        f"{repetition} of {len(distances)} is outside its qualified "
                        f"budget: {error}",
                        "the current kernel or its dispatch changed the output; the "
                        "frozen baseline reproduced its qualified output on this GPU",
                        f"rerun only this case with `{rerun}` and check the current "
                        "output against the independent reference offline; fix a "
                        "regression, or qualify a replacement baseline for an "
                        "intended numerical change, rather than widening the "
                        "tolerance",
                    )
                ) from None
        return {
            "case": case.id,
            self.metric.name: max(distances),
            "baseline_error_bound": budget.baseline_error_bound,
            "comparison_limit": budget.comparison_limit,
            "original_tolerance": case.tolerance,
            "old_launches": old_report["launches"],
            "current_launches": current_report["launches"],
            "torch_imported": old_report["torch_imported"]
            or current_report["torch_imported"],
        }

    def _parser(self) -> argparse.ArgumentParser:
        architectures = self.architectures.ARCHITECTURES
        parser = argparse.ArgumentParser(
            description=self.description,
            epilog=(
                "The cases and diff commands need no GPU or Torch. See "
                f"{_WORKFLOW} for the authoring and publication procedure."
            ),
        )
        commands = parser.add_subparsers(dest="command", required=True)
        export = commands.add_parser(
            "snapshot",
            help="export an immutable old source revision",
            description=(
                "Export a committed revision's production sources, with their "
                "digests, as the baseline that qualification runs."
            ),
        )
        export.add_argument(
            "--repository",
            required=True,
            type=Path,
            help="git checkout that contains the revision",
        )
        export.add_argument(
            "--revision", required=True, help="committed baseline revision"
        )
        export.add_argument(
            "--output", required=True, type=Path, help="new snapshot directory"
        )
        qualification = commands.add_parser(
            "qualify",
            help=f"qualify old {self.label} on an enrolled architecture",
            description=(
                f"Run a snapshot's {self.label} kernels for every enrolled case "
                "on this GPU, require deterministic outputs, bound their error "
                "against independent references, and write a candidate bundle "
                "and its qualification-lock.json."
            ),
        )
        qualification.add_argument(
            "--arch",
            choices=architectures,
            default="gfx942",
            help="enrolled architecture of this GPU (default: %(default)s)",
        )
        qualification.add_argument(
            "--baseline",
            required=True,
            type=Path,
            help="directory written by the snapshot command",
        )
        qualification.add_argument(
            "--output", required=True, type=Path, help="new candidate bundle directory"
        )
        qualification.add_argument(
            "--repetitions",
            type=int,
            default=3,
            help="old executions per case, at least 2 (default: %(default)s)",
        )
        if self.torch is None:
            qualification.set_defaults(torch_reference=None)
        elif len(self.torch.devices) == 1:
            qualification.add_argument(
                "--torch-reference",
                action="store_const",
                const=self.torch.devices[0],
                help=self.torch.help,
            )
        else:
            qualification.add_argument(
                "--torch-reference",
                nargs="?",
                const=self.torch.devices[0],
                choices=self.torch.devices,
                help=self.torch.help,
            )
        verification = commands.add_parser(
            "verify",
            help="compare frozen and current kernels on this GPU",
            description=(
                "Authenticate a bundle against its lock, then run the frozen and "
                "current kernels of each enrolled case, or of each case selected "
                "with --case, on this GPU and check every current output against "
                "the case's qualified budget. Exits with status 1 when a "
                "comparison fails and 2 when the lock or bundle cannot be used."
            ),
        )
        verification.add_argument(
            "--arch",
            choices=architectures,
            default="gfx942",
            help="enrolled architecture of this GPU (default: %(default)s)",
        )
        verification.add_argument(
            "--bundle",
            required=True,
            type=Path,
            help="extracted bundle directory; unpack a .tar.gz archive with "
            "reference_common/artifact.py first",
        )
        verification.add_argument(
            "--lock",
            type=Path,
            help="lock that authenticates the bundle (default: the committed "
            "baseline_lock.json)",
        )
        verification.add_argument(
            "--current-root",
            type=Path,
            help="rocKE root that provides the current kernels (default: the "
            "importable rocke package)",
        )
        verification.add_argument(
            "--repetitions",
            type=int,
            default=2,
            help="old and current executions per case (default: %(default)s)",
        )
        verification.add_argument(
            "--case",
            action="append",
            dest="cases",
            metavar="ID",
            help="run only this enrolled case; repeat to run several. A subset "
            "run is not a full verification",
        )
        listing = commands.add_parser(
            "cases",
            help="list an architecture's enrolled cases",
            description=(
                "Print each enrolled case id with its tolerance and case contract."
            ),
        )
        listing.add_argument(
            "--arch", choices=architectures, required=True, help="enrolled architecture"
        )
        listing.add_argument(
            "--json", action="store_true", help="print the cohort as JSON"
        )
        comparison = commands.add_parser(
            "diff",
            help="compare the enrolled cases with a qualified bundle",
            description=(
                "Authenticate a bundle's manifest against its lock, then report "
                "enrolled cases it does not record, recorded cases that are not "
                "enrolled, and changed tolerances or budgets, matching cases as "
                "verification does. Exits with status 1 on any difference and 2 "
                "when the lock or bundle cannot be read or authenticated. Reading "
                "the recorded device targets needs the rocKE platform package: "
                "install rocKE or set PYTHONPATH=platform/python from the rocKE "
                "root."
            ),
        )
        comparison.add_argument(
            "--arch", choices=architectures, required=True, help="enrolled architecture"
        )
        comparison.add_argument(
            "--bundle",
            type=Path,
            help="bundle directory or .tar.gz archive (default: the "
            "architecture's DVC archive, else its extracted bundle)",
        )
        comparison.add_argument(
            "--lock",
            type=Path,
            help="lock that authenticates the bundle (default: the committed "
            "baseline_lock.json)",
        )
        return parser

    def main(self, argv: Sequence[str] | None = None) -> None:
        """Run one command; ``argv`` defaults to the process arguments."""
        parser = self._parser()
        args = parser.parse_args(argv)
        if args.command == "snapshot":
            snapshot(args.repository.resolve(), args.revision, args.output.resolve())
        elif args.command == "qualify":
            self.qualify(
                args.baseline.resolve(),
                args.output.resolve(),
                args.repetitions,
                args.arch,
                torch_reference=args.torch_reference,
            )
        elif args.command == "cases":
            self._list_cases(args.arch, as_json=args.json)
        elif args.command == "diff":
            self._diff(parser, args)
        else:
            self._verify(parser, args)

    def _list_cases(self, architecture: str, *, as_json: bool) -> None:
        cases = self.architectures.get_architecture(architecture).CASES
        if as_json:
            listing = {
                "operation": self.name,
                "architecture": architecture,
                "cases": [
                    {"id": case.id, "tolerance": case.tolerance, "case": asdict(case)}
                    for case in cases
                ],
            }
            print(json.dumps(listing, indent=2))
            return
        width = max((len(case.id) for case in cases), default=0)
        for case in cases:
            fields = " ".join(f"{name}={value}" for name, value in asdict(case).items())
            print(f"{case.id:<{width}}  tolerance={case.tolerance:g} {fields}")
        print(f"{self.label} {architecture}: {len(cases)} enrolled cases")

    def _selected_lock(
        self, parser: argparse.ArgumentParser, args: argparse.Namespace
    ) -> Path:
        """Return ``--lock`` or the committed lock, or exit when it does not exist."""
        lock = args.lock or self.architectures.baseline_lock(args.arch)
        if args.lock is None and not lock.is_file():
            parser.error(
                f"{args.arch} has no published {self.label} bundle: {_shown(lock)} "
                "does not exist; pass a candidate's --bundle and --lock"
            )
        if not lock.is_file():
            parser.error(f"lock not found: {_shown(lock)}")
        return lock

    @contextmanager
    def _bundle_errors(
        self, parser: argparse.ArgumentParser, command: str
    ) -> Iterator[None]:
        """Report an unusable lock or bundle as one message and exit status 2.

        The message already names the lock or bundle problem, so a traceback
        would only repeat it, and status 2 keeps these errors apart from a
        reported difference or a failed comparison. Matching recorded device
        targets needs the rocKE platform package, so its absence is reported
        the same way.
        """
        try:
            yield
        except (ValueError, OSError) as error:
            parser.error(str(error))
        except ModuleNotFoundError as error:
            # Any other missing module is a defect; keep its traceback.
            if error.name is None or error.name.partition(".")[0] != "rocke":
                raise
            parser.error(
                f"{command} reads recorded device target ids with the rocKE "
                f"platform package, which cannot be imported ({error}); install "
                "rocKE or set PYTHONPATH=platform/python from the rocKE root"
            )

    def _diff(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        lock = self._selected_lock(parser, args)
        if args.bundle is not None:
            bundle = args.bundle.resolve()
            if not bundle.exists():
                parser.error(f"bundle not found: {_shown(bundle)}")
        else:
            bundle = self.published_bundle(args.arch)
            if not bundle.exists():
                archive = bundle.with_name(f"{bundle.name}.tar.gz")
                parser.error(
                    f"no {args.arch} {self.label} bundle at {_shown(archive)} or "
                    f"{_shown(bundle)}; fetch the archive with DVC or pass --bundle"
                )
        with self._bundle_errors(parser, args.command):
            differences = self.diff_bundle(bundle, lock, architecture=args.arch)
        cases = self.architectures.get_architecture(args.arch).CASES
        if not differences:
            print(
                f"{self.label} {args.arch}: all {len(cases)} enrolled cases match "
                f"{_shown(bundle)}"
            )
            return
        print(
            f"{self.label} {args.arch}: the enrolled cases differ from "
            f"{_shown(bundle)}"
        )
        for difference in differences:
            print(f"  {difference}")
        print(
            "A cohort or budget change needs a complete requalification; see "
            f"{_WORKFLOW}."
        )
        raise SystemExit(1)

    def _verify(
        self, parser: argparse.ArgumentParser, args: argparse.Namespace
    ) -> None:
        lock = self._selected_lock(parser, args)
        bundle = args.bundle.resolve()
        if bundle.is_file():
            parser.error(self._unpack_hint(bundle, lock))
        if not bundle.is_dir():
            parser.error(f"bundle directory not found: {_shown(bundle)}")
        target = self.architectures.get_architecture(args.arch)
        cases = target.CASES
        if args.cases:
            unknown = sorted(set(args.cases) - {case.id for case in cases})
            if unknown:
                parser.error(
                    f"not enrolled for {args.arch}: {', '.join(unknown)}; list the "
                    f"enrolled ids with: {parser.prog} cases --arch {args.arch}"
                )
            cases = tuple(case for case in cases if case.id in args.cases)
        skipped = len(target.CASES) - len(cases)
        with self._bundle_errors(parser, args.command):
            manifest = self.load_bundle(bundle, args.lock, architecture=args.arch)
        if skipped:
            print(
                f"{self.label}: running {len(cases)} of {len(target.CASES)} "
                f"enrolled {args.arch} cases selected with --case; a subset run "
                "is not a full verification",
                flush=True,
            )
        current = args.current_root.resolve() if args.current_root else None
        with reuse_workers(self.worker_module):
            for case in cases:
                report = self.verify_case(
                    case,
                    bundle=bundle,
                    manifest=manifest,
                    lock=args.lock,
                    current_root=current,
                    repetitions=args.repetitions,
                    architecture=args.arch,
                )
                print(json.dumps(report, sort_keys=True), flush=True)
        if skipped:
            print(
                f"{self.label}: {len(cases)}/{len(cases)} selected GPU cases "
                f"passed; this subset run is not a full verification ({skipped} "
                "enrolled cases did not run)",
                flush=True,
            )
        else:
            print(
                f"{self.label}: {len(target.CASES)}/{len(target.CASES)} "
                "required GPU cases passed",
                flush=True,
            )

    def _rerun_command(
        self,
        case: Any,
        architecture: str,
        bundle: Path,
        lock: Path | None,
        current_root: Path | None,
    ) -> str:
        """Show the verify command that repeats one case's comparison.

        The command checks the same bundle against the same lock, so a failing
        candidate is not rerun against the committed lock. The script is
        installed beside the operation package, so its path is valid in source
        and installed trees.
        """
        committed = self.architectures.baseline_lock(architecture)
        candidate = (
            []
            if lock is None or lock.resolve() == committed.resolve()
            else ["--lock", _shown(lock)]
        )
        current = (
            [] if current_root is None else ["--current-root", _shown(current_root)]
        )
        return display_command(
            [
                "python",
                _shown(self._package.parent / f"run_{self.name}_reference.py"),
                "verify",
                "--arch",
                architecture,
                "--bundle",
                _shown(bundle),
                *candidate,
                *current,
                "--case",
                case.id,
            ]
        )

    def _unpack_command(self, archive: Path, bundle: Path, lock: Path) -> str:
        """Show the artifact.py command that extracts and authenticates an archive."""
        return display_command(
            [
                "python",
                _shown(Path(__file__).with_name("artifact.py")),
                "unpack",
                "--operation",
                self.name,
                "--archive",
                _shown(archive),
                "--bundle",
                _shown(bundle),
                "--lock",
                _shown(lock),
            ]
        )

    def _unpack_hint(
        self, archive: Path, lock: Path, *, option: str = "--bundle"
    ) -> str:
        """Explain how to extract an archive passed where a bundle directory belongs."""
        stem = archive.name.removesuffix(".tar.gz")
        directory = archive.with_name(
            stem if stem != archive.name else f"{stem}-bundle"
        )
        return (
            f"{_shown(archive)} is a file, but {option} needs an extracted bundle "
            "directory. Unpack and authenticate the archive with:\n  "
            f"{self._unpack_command(archive, directory, lock)}\n"
            f"then pass {display_command([option, _shown(directory)])}"
        )
