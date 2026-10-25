# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Required GPU comparisons that every operation's pinned reference suite runs.

``test_<operation>_pinned_reference.py`` defines its ``reference_bundle``
fixture with ``select_bundle`` and binds the tests that ``pinned_reference_tests``
builds to module names, so CTest and pytest select them per operation.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from . import cli
from .cli import ReferenceOperation
from .numeric import decode, encode
from .session import reuse_workers
from .targets import get_device_processor

_TESTS = Path(__file__).resolve().parents[1]


def select_bundle(
    config: pytest.Config,
    operation: ReferenceOperation,
    default_bundle_path: Callable[..., Path],
    load_bundle: Callable[..., dict],
) -> Iterator[tuple[ModuleType, Path, dict]]:
    """Yield the GPU's adapter, bundle, and authenticated manifest, or skip.

    An explicitly selected operation or bundle is required: a missing enrolled
    GPU or bundle fails instead of skipping. Each skip says which of these it
    was, or that options selected another operation, and how to run the tests.
    Workers are reused for the module. The test module passes its own bundle
    lookup helpers so host tests can replace them there.
    """
    selected = config.getoption("--rocke-reference-operation")
    name = operation.name.upper()
    if selected and selected != operation.name:
        pytest.skip(
            f"not selected: --rocke-reference-operation {selected} selects another "
            "operation"
        )
    configured = config.getoption("--rocke-reference-bundle")
    arch = get_device_processor()
    enrolled = ", ".join(operation.architectures.ARCHITECTURES)
    if arch not in operation.architectures.ARCHITECTURES:
        required = selected or configured
        if not arch and required:
            pytest.fail(
                f"the required enrolled {name} GPU is not available: HIP reports no "
                f"GPU; run on an enrolled processor ({enrolled})"
            )
        reason = (
            f"no {name} reference cohort is enrolled for {arch}"
            if arch
            else "no GPU detected: HIP reports no device"
        )
        pytest.skip(f"{reason}; run on an enrolled {name} processor ({enrolled})")
    target = operation.architectures.get_architecture(arch)
    bundle = (
        Path(configured).resolve()
        if configured
        else default_bundle_path(_TESTS, arch, operation=operation.name)
    )
    lock = config.getoption("--rocke-reference-lock")
    lock = Path(lock) if lock else None
    if not bundle.is_dir():
        hint = operation.bundle_hint(
            arch, bundle, lock, option="--rocke-reference-bundle"
        )
        if selected or configured:
            pytest.fail(
                f"required {arch} {name} reference bundle is missing: {bundle}\n{hint}"
            )
        pytest.skip(f"no extracted {arch} {name} reference bundle at {bundle}; {hint}")
    manifest = load_bundle(bundle, lock, architecture=arch)
    with reuse_workers(operation.worker_module):
        yield target, bundle, manifest


def pinned_reference_tests(
    operation: ReferenceOperation, *, perturbation: float
) -> tuple[Callable[..., None], Callable[..., None], Callable[..., None]]:
    """Return the correctness, perturbed-result, and missing-launch tests.

    ``perturbation`` is added to one old or current output element. A changed
    old output fails its qualified digest; a changed current output must exceed
    the first case's comparison limit in the operation's own metric.
    """
    registry = operation.architectures
    cohort = [
        (name, case)
        for name in registry.ARCHITECTURES
        for case in registry.get_architecture(name).CASES
    ]

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "architecture,case",
        cohort,
        ids=[f"{name}-{case.id}" for name, case in cohort],
    )
    def test_correctness_against_qualified_rocke(
        architecture, case, reference_bundle, pytestconfig
    ):
        target, bundle, manifest = reference_bundle
        if architecture != target.NAME:
            pytest.skip(
                f"not selected: this {architecture} case runs only on a "
                f"{architecture} GPU, and this GPU is {target.NAME}"
            )
        # A failure's rerun command checks the bundle against the same lock.
        lock = pytestconfig.getoption("--rocke-reference-lock")
        report = operation.verify_case(
            case,
            bundle=bundle,
            manifest=manifest,
            lock=Path(lock) if lock else None,
            architecture=architecture,
        )
        assert not report["torch_imported"]
        assert report["old_launches"] == report["current_launches"] == 2

    @pytest.mark.gpu
    @pytest.mark.parametrize("mode", ["source", "replay"])
    def test_rejects_perturbed_gpu_results(reference_bundle, monkeypatch, mode):
        target, bundle, manifest = reference_bundle
        case = target.CASES[0]
        real_worker = cli.run_worker

        def perturbed_worker(request, **kwargs):
            outputs, report = real_worker(request, **kwargs)
            if request["mode"] == mode:
                changed = decode(outputs[-1], case.dtype).copy()
                changed.flat[0] += perturbation
                outputs[-1] = encode(changed.astype(np.float32), case.dtype)
            return outputs, report

        monkeypatch.setattr(cli, "run_worker", perturbed_worker)
        expected = "remaining limit" if mode == "source" else "qualification failure"
        with pytest.raises(AssertionError, match=expected):
            operation.verify_case(
                case, bundle=bundle, manifest=manifest, architecture=target.NAME
            )

    @pytest.mark.gpu
    def test_rejects_missing_gpu_launch(reference_bundle, monkeypatch, tmp_path):
        import rocke
        from rocke.runtime import KernelLauncher

        run = importlib.import_module(operation.worker_module).run
        target, bundle, manifest = reference_bundle
        case = target.CASES[0]
        key = operation.manifest_key(manifest, case, architecture=target.NAME)
        entry = manifest["cases"][key]
        case_dir = bundle / "payload/cases" / key
        input_file = tmp_path / "inputs.npz"
        np.savez(input_file, **operation.checked_inputs(case, entry["input_digests"]))
        monkeypatch.setattr(
            KernelLauncher,
            "__call__",
            lambda *args, **kwargs: SimpleNamespace(launches=0),
        )
        with pytest.raises(ValueError, match="non-finite or unwritten"):
            run(
                {
                    "mode": "replay",
                    "architecture": target.NAME,
                    "platform_root": str(Path(rocke.__file__).resolve().parent.parent),
                    "case": asdict(case),
                    "inputs": str(input_file),
                    "input_digests": entry["input_digests"],
                    "kernel": entry["kernel"],
                    "hsaco": str(case_dir / "kernel.hsaco"),
                    "repetitions": 1,
                },
                tmp_path,
            )

    return (
        test_correctness_against_qualified_rocke,
        test_rejects_perturbed_gpu_results,
        test_rejects_missing_gpu_launch,
    )
