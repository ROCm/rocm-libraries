# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build, qualify, and verify a pinned live convolution reference bundle.

The snapshot and qualification commands are offline maintenance operations.
Verification only executes a locked bundle and current kernels; it never
downloads dependencies, generates independent answers, or promotes a baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from reference_common.source import snapshot
from reference_common.runner import run_worker

from reference_common.session import reuse_workers
from .architectures import ARCHITECTURES, baseline_lock, get_architecture

from .contract import (
    SCHEMA_VERSION,
    INPUT_GENERATOR,
    checked_inputs,
    Case,
    independent_reference,
    make_inputs,
    normalized_distance,
    reference_scale,
)
from reference_common.numeric import (
    ErrorBudget,
    array_digest,
    decode,
    file_digest,
    payload_digests,
    write_json,
)

_PACKAGE = Path(__file__).resolve().parent


def _worker(
    request: dict, *, runner: Path, platform: Path, library: Path | None, work: Path
) -> tuple[list[np.ndarray], dict]:
    return run_worker(
        request,
        runner=runner,
        platform=platform,
        library=library,
        work=work,
        module="conv_reference.worker",
    )


def qualify(
    baseline: Path,
    output: Path,
    repetitions: int,
    architecture: str = "gfx942",
    *,
    torch_reference: bool = False,
) -> None:
    """Measure the old version's bounds and require deterministic old outputs."""
    target = get_architecture(architecture)
    if torch_reference:
        import torch  # Fail before starting qualification if unavailable.

        from .torch_reference import cross_check
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
    runner = payload / "runner/conv_reference"
    runner.mkdir(parents=True)
    for name in ("__init__.py", "contract.py", "worker.py"):
        shutil.copy2(_PACKAGE / name, runner / name)
    shutil.copytree(
        _PACKAGE / "architectures",
        runner / "architectures",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "baseline_lock.json"),
    )
    shutil.copytree(
        _PACKAGE.parent / "reference_common",
        payload / "runner/reference_common",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    entries = {}
    for case in target.CASES:
        case_dir = payload / "cases" / case.id
        case_dir.mkdir(parents=True)
        inputs = make_inputs(case)
        input_digests = {name: array_digest(array) for name, array in inputs.items()}
        with tempfile.TemporaryDirectory(prefix="rocke-conv-qualify-") as temporary:
            input_file = Path(temporary) / "inputs.npz"
            np.savez(input_file, **inputs)
            outputs, report = _worker(
                {
                    "mode": "source",
                    "architecture": architecture,
                    "case": asdict(case),
                    "inputs": str(input_file),
                    "input_digests": input_digests,
                    "export": str(case_dir / "kernel.hsaco"),
                    "repetitions": repetitions,
                },
                runner=_PACKAGE.parent,
                platform=source / "platform/python",
                library=source / "library",
                work=Path(temporary) / "worker",
            )
        digests = {array_digest(array) for array in outputs}
        if len(digests) != 1:
            raise ValueError(
                f"old convolution output is not deterministic for {case.id}"
            )
        reference = independent_reference(case, inputs)
        scale = reference_scale(reference)
        bound = normalized_distance(decode(outputs[0], case.dtype), reference, scale)
        torch_report = None
        if torch_reference:
            torch_report = cross_check(
                case, inputs, reference, decode(outputs[0], case.dtype), scale
            )
            bound = max(
                bound,
                *(r["baseline_error_bound"] for r in torch_report["results"].values()),
            )
        budget = ErrorBudget(case.tolerance, bound, case.margin)
        entries[case.id] = {
            "case": asdict(case),
            "input_digests": input_digests,
            "output_digest": digests.pop(),
            "reference_digest": array_digest(reference),
            "reference_scale": scale,
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
            f"margin={case.margin:.9g}, tolerance={case.tolerance:.9g}",
            flush=True,
        )
    manifest = {
        "schema": SCHEMA_VERSION,
        "operation": "conv-fwd",
        "input_generation": INPUT_GENERATOR,
        "baseline_revision": metadata["revision"],
        "baseline_snapshot_sha256": file_digest(baseline / "snapshot.json"),
        "reference": {
            "implementation": "numpy-float64-conv-fwd-rounded-v1",
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "contract_sha256": file_digest(_PACKAGE / "contract.py"),
            "metric": "max-absolute-error/frozen-reference-scale",
            "input_generator": "numpy-PCG64-seed-0-uniform-f32; regenerated and digest-checked quantized bits",
        },
        "cases": entries,
        "files": payload_digests(payload),
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "qualification-lock.json",
        {
            "schema": SCHEMA_VERSION,
            "baseline_revision": metadata["revision"],
            "manifest_sha256": file_digest(output / "manifest.json"),
        },
    )
    print("Qualification complete; review qualification-lock.json before promotion.")


def load_bundle(
    bundle: Path, lock_path: Path | None = None, *, architecture: str = "gfx942"
) -> dict:
    """Require the independently pinned manifest, payload, and exact case cohort."""
    get_architecture(architecture)
    lock = json.loads((lock_path or baseline_lock(architecture)).read_text())
    if lock["schema"] != SCHEMA_VERSION:
        raise ValueError("unsupported convolution lock schema")
    if file_digest(bundle / "manifest.json") != lock["manifest_sha256"]:
        raise ValueError("convolution bundle manifest does not match the pinned lock")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        manifest["schema"] != SCHEMA_VERSION
        or manifest["baseline_revision"] != lock["baseline_revision"]
    ):
        raise ValueError("convolution baseline identity mismatch")
    if payload_digests(bundle / "payload") != manifest["files"]:
        raise ValueError(
            "convolution bundle payload has missing, modified, or extra files"
        )
    if manifest.get("operation") != "conv-fwd":
        raise ValueError("wrong reference operation")
    if manifest.get("input_generation") != INPUT_GENERATOR:
        raise ValueError("unsupported convolution input generator contract")
    if any(p.is_file() and p.suffix in (".npz", ".npy") for p in bundle.rglob("*")):
        raise ValueError(
            "generated-input convolution bundles must not contain tensor files"
        )
    _validate_cases(manifest, architecture)
    return manifest


def _validate_cases(manifest: dict, architecture: str) -> None:
    """Preserve the cohort and budgets across qualification and verification."""
    target = get_architecture(architecture)
    if set(manifest["cases"]) != {case.id for case in target.CASES}:
        raise ValueError(
            "convolution bundle does not cover the complete enrolled cohort"
        )
    for case in target.CASES:
        entry = manifest["cases"][case.id]
        if entry["device_target"].split(":", 1)[0] != architecture:
            raise ValueError(
                f"convolution bundle targets a different architecture: {case.id}"
            )
        if entry["case"] != asdict(case):
            raise ValueError(f"convolution case contract changed: {case.id}")
        normalized_distance(np.zeros(1), np.zeros(1), entry["reference_scale"])
        budget = ErrorBudget(**entry["budget"])
        if (
            budget.tolerance != case.tolerance
            or budget.margin != case.margin
            or entry["comparison_limit"] != budget.comparison_limit
        ):
            raise ValueError(f"convolution tolerance or budget changed: {case.id}")


def _current_paths(current_root: Path | None) -> tuple[Path, Path]:
    if current_root is not None:
        return current_root / "platform/python", current_root / "library"
    # Installed pytest already resolves the installed platform package. Library
    # tests are installed under that library's tests directory in both layouts.
    spec = importlib.util.find_spec("rocke")
    if spec is None or spec.origin is None:
        raise RuntimeError("install the current rocKE platform or pass --current-root")
    return Path(spec.origin).parent.parent, _PACKAGE.parent.parent


def verify_case(
    case: Case,
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
    target = get_architecture(architecture)
    if case not in target.CASES:
        raise ValueError("case is not enrolled for the selected architecture")
    entry = manifest["cases"][case.id]
    payload = bundle / "payload"
    case_dir = payload / "cases" / case.id
    base = {
        "architecture": architecture,
        "case": asdict(case),
        "input_digests": entry["input_digests"],
        "repetitions": repetitions,
    }
    with tempfile.TemporaryDirectory(prefix="rocke-conv-verify-") as temporary:
        temporary = Path(temporary)
        input_file = temporary / "inputs.npz"
        np.savez(input_file, **checked_inputs(case, entry["input_digests"]))
        base["inputs"] = str(input_file)
        old, old_report = _worker(
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
        platform, library = _current_paths(current_root)
        current, current_report = _worker(
            dict(base, mode="source"),
            runner=_PACKAGE.parent,
            platform=platform,
            library=library,
            work=temporary / "current",
        )
    if old_report["device_target"] != entry["device_target"]:
        raise ValueError("reference target differs from the qualified target")
    if current_report["device_target"] != old_report["device_target"]:
        raise ValueError("current and reference workers used different target features")
    budget = ErrorBudget(**entry["budget"])
    reference = decode(old[0], case.dtype)
    distances = [
        normalized_distance(
            decode(array, case.dtype), reference, entry["reference_scale"]
        )
        for array in current
    ]
    for distance in distances:
        budget.check(distance)
    return {
        "case": case.id,
        "normalized_error_upper": max(distances),
        "baseline_error_bound": budget.baseline_error_bound,
        "comparison_limit": budget.comparison_limit,
        "original_tolerance": case.tolerance,
        "old_launches": old_report["launches"],
        "current_launches": current_report["launches"],
        "torch_imported": old_report["torch_imported"]
        or current_report["torch_imported"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "snapshot", help="export an immutable old source revision"
    )
    export.add_argument("--repository", required=True, type=Path)
    export.add_argument("--revision", required=True)
    export.add_argument("--output", required=True, type=Path)
    qualification = commands.add_parser(
        "qualify", help="qualify old convolution on an enrolled architecture"
    )
    qualification.add_argument("--arch", choices=ARCHITECTURES, default="gfx942")
    qualification.add_argument("--baseline", required=True, type=Path)
    qualification.add_argument("--output", required=True, type=Path)
    qualification.add_argument("--repetitions", type=int, default=3)
    qualification.add_argument(
        "--torch-reference",
        action="store_true",
        help="also qualify against CPU Torch float64/float32 conv2d (requires Torch)",
    )
    verification = commands.add_parser(
        "verify", help="run every required GPU comparison"
    )
    verification.add_argument("--arch", choices=ARCHITECTURES, default="gfx942")
    verification.add_argument("--bundle", required=True, type=Path)
    verification.add_argument("--lock", type=Path)
    verification.add_argument("--current-root", type=Path)
    verification.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.repository.resolve(), args.revision, args.output.resolve())
    elif args.command == "qualify":
        qualify(
            args.baseline.resolve(),
            args.output.resolve(),
            args.repetitions,
            args.arch,
            torch_reference=args.torch_reference,
        )
    else:
        bundle = args.bundle.resolve()
        manifest = load_bundle(bundle, args.lock, architecture=args.arch)
        target = get_architecture(args.arch)
        current = args.current_root.resolve() if args.current_root else None
        with reuse_workers("conv_reference.worker"):
            for case in target.CASES:
                report = verify_case(
                    case,
                    bundle=bundle,
                    manifest=manifest,
                    current_root=current,
                    repetitions=args.repetitions,
                    architecture=args.arch,
                )
                print(json.dumps(report, sort_keys=True), flush=True)
        print(
            f"convolution: {len(target.CASES)}/{len(target.CASES)} required GPU cases passed",
            flush=True,
        )
