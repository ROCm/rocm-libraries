# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build, qualify, and verify a pinned live SDPA reference bundle.

The snapshot and qualification commands are offline maintenance operations.
Verification only executes a locked bundle and current kernels; it never
downloads dependencies, generates independent answers, or promotes a baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import numpy as np

from .contract import (
    CASES,
    CASE_BY_ID,
    SCHEMA_VERSION,
    Case,
    ErrorBudget,
    array_digest,
    decode,
    file_digest,
    independent_reference,
    make_inputs,
    max_abs_upper,
    payload_digests,
    write_json,
)

_PACKAGE = Path(__file__).resolve().parent
DEFAULT_LOCK = _PACKAGE / "baseline_lock.json"
_SOURCE_PREFIX = PurePosixPath("dnn-providers/hip-kernel-provider/rocke")


def snapshot(repository: Path, revision: str, output: Path) -> None:
    """Export committed baseline sources, recording every file's identity."""
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", f"{revision}^{{commit}}"],
        text=True,
    ).strip()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                revision,
                str(_SOURCE_PREFIX / "platform/python"),
                str(_SOURCE_PREFIX / "library"),
            ],
            stdout=archive,
            check=True,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                relative = PurePosixPath(member.name).relative_to(_SOURCE_PREFIX)
                if ".." in relative.parts:
                    raise ValueError("invalid source archive path")
                destination = source / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as stream:
                    destination.write_bytes(stream.read())
    write_json(
        output / "snapshot.json",
        {"revision": revision, "files": payload_digests(source)},
    )
    print(f"Exported baseline {revision}", flush=True)


def _worker(
    request: dict, *, runner: Path, platform: Path, library: Path | None, work: Path
) -> tuple[list[np.ndarray], dict]:
    work.mkdir(parents=True, exist_ok=False)
    request = dict(request, platform_root=str(platform.resolve()))
    paths = [platform.resolve()]
    if library is not None:
        paths.append(library.resolve())
        request["library_root"] = str(library.resolve())
    # The runner lives in tests/, which also contains a dispatch package.
    # Production packages must resolve before those test-only names.
    paths.append(runner.resolve())
    write_json(work / "request.json", request)
    env = dict(os.environ)
    # Never let an inherited PYTHONPATH or user site select the other rocKE.
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-s",
            "-m",
            "sdpa_reference.worker",
            str(work / "request.json"),
            str(work),
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode:
        raise RuntimeError(
            f"SDPA {request['mode']} worker failed:\n"
            f"{completed.stdout[-4000:]}{completed.stderr[-12000:]}"
        )
    report = json.loads((work / "report.json").read_text())
    if report["launches"] != request["repetitions"]:
        raise RuntimeError("required SDPA GPU launches did not execute")
    with np.load(work / "outputs.npz", allow_pickle=False) as archive:
        outputs = [archive[f"out_{i}"] for i in range(request["repetitions"])]
    return outputs, report


def qualify(baseline: Path, output: Path, repetitions: int) -> None:
    """Measure the old version's bounds and require deterministic old outputs."""
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
    runner = payload / "runner/sdpa_reference"
    runner.mkdir(parents=True)
    for name in ("__init__.py", "contract.py", "worker.py"):
        shutil.copy2(_PACKAGE / name, runner / name)
    entries = {}
    for case in CASES:
        case_dir = payload / "cases" / case.id
        case_dir.mkdir(parents=True)
        inputs = make_inputs(case)
        np.savez(case_dir / "inputs.npz", **inputs)
        input_digests = {name: array_digest(array) for name, array in inputs.items()}
        with tempfile.TemporaryDirectory(prefix="rocke-sdpa-qualify-") as temporary:
            outputs, report = _worker(
                {
                    "mode": "source",
                    "case": asdict(case),
                    "inputs": str(case_dir / "inputs.npz"),
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
            raise ValueError(f"old SDPA output is not deterministic for {case.id}")
        reference = independent_reference(case, inputs)
        bound = max_abs_upper(decode(outputs[0], case.dtype), reference)
        budget = ErrorBudget(case.tolerance, bound, case.margin)
        entries[case.id] = {
            "case": asdict(case),
            "input_digests": input_digests,
            "output_digest": digests.pop(),
            "reference_digest": array_digest(reference),
            "budget": asdict(budget),
            "comparison_limit": budget.comparison_limit,
            "kernel": report["kernel"],
            "device_target": report["device_target"],
            "qualification_repetitions": repetitions,
            "compiler": report["compiler"],
        }
        print(
            f"QUALIFIED {case.id}: old_bound={bound:.9g}, "
            f"comparison_limit={budget.comparison_limit:.9g}, "
            f"margin={case.margin:.9g}, tolerance={case.tolerance:.9g}",
            flush=True,
        )
    manifest = {
        "schema": SCHEMA_VERSION,
        "baseline_revision": metadata["revision"],
        "baseline_snapshot_sha256": file_digest(baseline / "snapshot.json"),
        "reference": {
            "implementation": "numpy-float64-sdpa-v1",
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "contract_sha256": file_digest(_PACKAGE / "contract.py"),
            "metric": "max-absolute-error",
            "input_generator": "numpy-PCG64-seed-0-f32-normal; stored quantized bits",
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


def load_bundle(bundle: Path, lock_path: Path = DEFAULT_LOCK) -> dict:
    """Require the independently pinned manifest, payload, and exact case cohort."""
    lock = json.loads(lock_path.read_text())
    if lock["schema"] != SCHEMA_VERSION:
        raise ValueError("unsupported SDPA lock schema")
    if file_digest(bundle / "manifest.json") != lock["manifest_sha256"]:
        raise ValueError("SDPA bundle manifest does not match the pinned lock")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        manifest["schema"] != SCHEMA_VERSION
        or manifest["baseline_revision"] != lock["baseline_revision"]
    ):
        raise ValueError("SDPA baseline identity mismatch")
    if payload_digests(bundle / "payload") != manifest["files"]:
        raise ValueError("SDPA bundle payload has missing, modified, or extra files")
    if set(manifest["cases"]) != set(CASE_BY_ID):
        raise ValueError("SDPA bundle does not cover the complete enrolled cohort")
    for case in CASES:
        entry = manifest["cases"][case.id]
        if entry["case"] != asdict(case):
            raise ValueError(f"SDPA case contract changed: {case.id}")
        budget = ErrorBudget(**entry["budget"])
        if (
            budget.tolerance != case.tolerance
            or budget.margin != case.margin
            or entry["comparison_limit"] != budget.comparison_limit
        ):
            raise ValueError(f"SDPA tolerance or budget changed: {case.id}")
    return manifest


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
) -> dict:
    """Run old/current in separate interpreters and check every current output."""
    if repetitions < 1:
        raise ValueError("verification must execute at least once")
    entry = manifest["cases"][case.id]
    payload = bundle / "payload"
    case_dir = payload / "cases" / case.id
    base = {
        "case": asdict(case),
        "inputs": str(case_dir / "inputs.npz"),
        "input_digests": entry["input_digests"],
        "repetitions": repetitions,
    }
    with tempfile.TemporaryDirectory(prefix="rocke-sdpa-verify-") as temporary:
        temporary = Path(temporary)
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
        max_abs_upper(decode(array, case.dtype), reference) for array in current
    ]
    for distance in distances:
        budget.check(distance)
    return {
        "case": case.id,
        "max_abs_upper": max(distances),
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
    qualification = commands.add_parser("qualify", help="qualify old SDPA on gfx942")
    qualification.add_argument("--baseline", required=True, type=Path)
    qualification.add_argument("--output", required=True, type=Path)
    qualification.add_argument("--repetitions", type=int, default=3)
    verification = commands.add_parser(
        "verify", help="run every required GPU comparison"
    )
    verification.add_argument("--bundle", required=True, type=Path)
    verification.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    verification.add_argument("--current-root", type=Path)
    verification.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.repository.resolve(), args.revision, args.output.resolve())
    elif args.command == "qualify":
        qualify(args.baseline.resolve(), args.output.resolve(), args.repetitions)
    else:
        bundle = args.bundle.resolve()
        manifest = load_bundle(bundle, args.lock)
        current = args.current_root.resolve() if args.current_root else None
        for case in CASES:
            report = verify_case(
                case,
                bundle=bundle,
                manifest=manifest,
                current_root=current,
                repetitions=args.repetitions,
            )
            print(json.dumps(report, sort_keys=True), flush=True)
        print(f"SDPA: {len(CASES)}/{len(CASES)} required GPU cases passed", flush=True)
