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
or promotes a baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .artifact import manifest_operation
from .cases import match_manifest_cases
from .numeric import (
    ErrorBudget,
    array_digest,
    decode,
    file_digest,
    payload_digests,
    write_json,
)
from .runner import run_worker
from .session import reuse_workers
from .source import snapshot
from .targets import processor_from_target_id


def _no_parameters(reference: np.ndarray) -> dict[str, Any]:
    return {}


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
    generator in each manifest.
    """

    name: str
    label: str
    description: str
    contract: ModuleType
    architectures: ModuleType
    reference: Mapping[str, str]
    metric: Metric
    torch: TorchReference
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
        runner: Path,
        platform: Path,
        library: Path | None,
        work: Path,
    ) -> tuple[list[np.ndarray], dict]:
        return run_worker(
            request,
            runner=runner,
            platform=platform,
            library=library,
            work=work,
            module=self.worker_module,
        )

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
        shutil.copytree(
            self._package.parent / "reference_common",
            destination / "reference_common",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
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
                    runner=self._package.parent,
                    platform=source / "platform/python",
                    library=source / "library",
                    work=Path(temporary) / "worker",
                )
            digests = {array_digest(array) for array in outputs}
            if len(digests) != 1:
                raise ValueError(
                    f"old {self.label} output is not deterministic for {case.id}"
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
        self.architectures.get_architecture(architecture)
        schema = self.contract.SCHEMA_VERSION
        lock = json.loads(
            (lock_path or self.architectures.baseline_lock(architecture)).read_text()
        )
        if lock["schema"] != schema:
            raise ValueError(f"unsupported {self.label} lock schema")
        if file_digest(bundle / "manifest.json") != lock["manifest_sha256"]:
            raise ValueError(
                f"{self.label} bundle manifest does not match the pinned lock"
            )
        manifest = json.loads((bundle / "manifest.json").read_text())
        if (
            manifest["schema"] != schema
            or manifest["baseline_revision"] != lock["baseline_revision"]
        ):
            raise ValueError(f"{self.label} baseline identity mismatch")
        if payload_digests(bundle / "payload") != manifest["files"]:
            raise ValueError(
                f"{self.label} bundle payload has missing, modified, or extra files"
            )
        expected = manifest_operation(self.name)
        # As in artifact.py, the selected lock identifies an undeclared manifest.
        undeclared = None if self.declares_operation else expected
        if manifest.get("operation", undeclared) != expected:
            raise ValueError("wrong reference operation")
        if manifest.get("input_generation") != self.contract.INPUT_GENERATOR:
            raise ValueError(f"unsupported {self.label} input generator contract")
        if any(p.is_file() and p.suffix in (".npz", ".npy") for p in bundle.rglob("*")):
            raise ValueError(
                f"generated-input {self.label} bundles must not contain tensor files"
            )
        self.validate_cases(manifest, architecture)
        return manifest

    def validate_cases(self, manifest: dict, architecture: str) -> None:
        """Preserve the cohort and budgets across qualification and verification."""
        policy = manifest.get("budget_policy")
        if policy not in (None, "max-reference-bound-v1"):
            raise ValueError("unsupported reference budget policy")
        target = self.architectures.get_architecture(architecture)
        keys = match_manifest_cases(manifest["cases"], target.CASES, label=self.label)
        compiler_target = getattr(target, "COMPILER_TARGET", None)
        for case in target.CASES:
            entry = manifest["cases"][keys[case.id]]
            if processor_from_target_id(entry["device_target"]) != architecture:
                raise ValueError(
                    f"{self.label} bundle targets a different architecture: {case.id}"
                )
            if (
                compiler_target is not None
                and entry["kernel"].get("compiler_target") != compiler_target
            ):
                raise ValueError(
                    f"{self.label} bundle has a different compiler target: {case.id}"
                )
            # Reject recorded metric parameters, such as a scale, before any GPU run.
            self.metric.distance(np.zeros(1), np.zeros(1), entry)
            budget = ErrorBudget(**entry["budget"])
            if (
                budget.tolerance != case.tolerance
                or budget.margin != (case.tolerance / 20 if policy is None else 0.0)
                or entry["comparison_limit"] != budget.comparison_limit
            ):
                raise ValueError(f"{self.label} tolerance or budget changed: {case.id}")

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
        current_root: Path | None = None,
        repetitions: int = 2,
        architecture: str = "gfx942",
    ) -> dict:
        """Run old/current in separate interpreters and check every current output."""
        if repetitions < 1:
            raise ValueError("verification must execute at least once")
        target = self.architectures.get_architecture(architecture)
        if case not in target.CASES:
            raise ValueError("case is not enrolled for the selected architecture")
        key = self.manifest_key(manifest, case, architecture=architecture)
        entry = manifest["cases"][key]
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
                runner=payload / "runner",
                platform=payload / "runtime",
                library=None,
                work=temporary / "old",
            )
            for array in old:
                if array_digest(array) != entry["output_digest"]:
                    raise AssertionError(
                        f"reference qualification failure for {case.id}: old output "
                        "does not match the independently qualified result"
                    )
            platform, library = self._current_paths(current_root)
            current, current_report = self._worker(
                dict(base, mode="source"),
                runner=self._package.parent,
                platform=platform,
                library=library,
                work=temporary / "current",
            )
        if old_report["device_target"] != entry["device_target"]:
            raise ValueError("reference target differs from the qualified target")
        if current_report["device_target"] != old_report["device_target"]:
            raise ValueError(
                "current and reference workers used different target features"
            )
        budget = ErrorBudget(**entry["budget"])
        reference = decode(old[0], case.dtype)
        distances = [
            self.metric.distance(decode(array, case.dtype), reference, entry)
            for array in current
        ]
        for distance in distances:
            budget.check(distance)
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

    def main(self) -> None:
        architectures = self.architectures.ARCHITECTURES
        parser = argparse.ArgumentParser(description=self.description)
        commands = parser.add_subparsers(dest="command", required=True)
        export = commands.add_parser(
            "snapshot", help="export an immutable old source revision"
        )
        export.add_argument("--repository", required=True, type=Path)
        export.add_argument("--revision", required=True)
        export.add_argument("--output", required=True, type=Path)
        qualification = commands.add_parser(
            "qualify", help=f"qualify old {self.label} on an enrolled architecture"
        )
        qualification.add_argument("--arch", choices=architectures, default="gfx942")
        qualification.add_argument("--baseline", required=True, type=Path)
        qualification.add_argument("--output", required=True, type=Path)
        qualification.add_argument("--repetitions", type=int, default=3)
        if len(self.torch.devices) == 1:
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
            "verify", help="run every required GPU comparison"
        )
        verification.add_argument("--arch", choices=architectures, default="gfx942")
        verification.add_argument("--bundle", required=True, type=Path)
        verification.add_argument("--lock", type=Path)
        verification.add_argument("--current-root", type=Path)
        verification.add_argument("--repetitions", type=int, default=2)
        args = parser.parse_args()
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
        else:
            bundle = args.bundle.resolve()
            manifest = self.load_bundle(bundle, args.lock, architecture=args.arch)
            target = self.architectures.get_architecture(args.arch)
            current = args.current_root.resolve() if args.current_root else None
            with reuse_workers(self.worker_module):
                for case in target.CASES:
                    report = self.verify_case(
                        case,
                        bundle=bundle,
                        manifest=manifest,
                        current_root=current,
                        repetitions=args.repetitions,
                        architecture=args.arch,
                    )
                    print(json.dumps(report, sort_keys=True), flush=True)
            print(
                f"{self.label}: {len(target.CASES)}/{len(target.CASES)} "
                "required GPU cases passed",
                flush=True,
            )
