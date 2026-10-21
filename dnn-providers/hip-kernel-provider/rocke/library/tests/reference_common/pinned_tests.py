# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Required GPU comparisons that every operation's pinned reference suite runs.

``test_<operation>_pinned_reference.py`` defines its ``reference_bundle``
fixture with ``select_bundle`` and binds the tests that ``pinned_reference_tests``
builds to module names, so CTest and pytest select them per operation. The tests
are parametrized by each enrolled (architecture, variant) slot.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from contextlib import ExitStack
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
) -> Iterator[Callable[[str, str], tuple[ModuleType, Path, dict]]]:
    """Yield the lookup of a slot's adapter, bundle, and authenticated manifest.

    The lookup takes an architecture and variant and skips a slot of another
    processor. An explicitly selected operation or bundle is required: a
    missing enrolled GPU or bundle fails instead of skipping. Each skip says
    which of these it was, or that options selected another operation, and how
    to run the tests. A slot's bundle and reused workers stay open until a test
    opens another slot, so at most one slot's workers are alive at a time. The
    test module passes its own bundle lookup helpers so host tests can replace
    them there.
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
    lock = config.getoption("--rocke-reference-lock")
    lock = Path(lock) if lock else None
    opened: dict[tuple[str, str], tuple[ModuleType, Path, dict]] = {}
    workers = ExitStack()

    def open_slot(architecture: str, variant: str) -> tuple[ModuleType, Path, dict]:
        if architecture != arch:
            pytest.skip(
                f"not selected: this {architecture} {variant} reference runs only "
                f"on a {architecture} GPU, and this GPU is {arch}"
            )
        if (architecture, variant) in opened:
            return opened[architecture, variant]
        workers.close()
        opened.clear()
        target = operation.architectures.get_adapter(arch, variant)
        bundle = (
            Path(configured).resolve()
            if configured
            else default_bundle_path(_TESTS, arch, operation=operation.name)
        )
        if not bundle.is_dir():
            hint = operation.bundle_hint(
                arch, bundle, lock, variant=variant, option="--rocke-reference-bundle"
            )
            if selected or configured:
                pytest.fail(
                    f"required {arch} {name} reference bundle is missing: "
                    f"{bundle}\n{hint}"
                )
            pytest.skip(
                f"no extracted {arch} {name} reference bundle at {bundle}; {hint}"
            )
        manifest = load_bundle(bundle, lock, architecture=arch, variant=variant)
        workers.enter_context(reuse_workers(operation.worker_module))
        opened[architecture, variant] = target, bundle, manifest
        return opened[architecture, variant]

    with workers:
        yield open_slot


def pinned_reference_tests(
    operation: ReferenceOperation, *, perturbation: float
) -> tuple[Callable[..., None], Callable[..., None], Callable[..., None]]:
    """Return the correctness, perturbed-result, and missing-launch tests.

    Each runs for every enrolled (architecture, variant) slot, and the
    correctness test for every case of the slot's cohort. ``perturbation`` is
    added to one old or current output element. A changed old output fails its
    qualified digest; a changed current output must exceed the first case's
    comparison limit in the operation's own metric.
    """
    registry = operation.architectures
    slots = [
        (name, variant)
        for name in registry.ARCHITECTURES
        for variant in registry.variants(name)
    ]
    slot_ids = [f"{name}-{variant}" for name, variant in slots]
    cohort = [
        (name, variant, case)
        for name, variant in slots
        for case in registry.get_adapter(name, variant).CASES
    ]

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "architecture,variant,case",
        cohort,
        ids=[f"{name}-{variant}-{case.id}" for name, variant, case in cohort],
    )
    def test_correctness_against_qualified_rocke(
        architecture, variant, case, reference_bundle, pytestconfig
    ):
        _, bundle, manifest = reference_bundle(architecture, variant)
        # A failure's rerun command checks the bundle against the same lock.
        lock = pytestconfig.getoption("--rocke-reference-lock")
        report = operation.verify_case(
            case,
            bundle=bundle,
            manifest=manifest,
            lock=Path(lock) if lock else None,
            architecture=architecture,
            variant=variant,
        )
        assert not report["torch_imported"]
        assert report["old_launches"] == report["current_launches"] == 2

    @pytest.mark.gpu
    @pytest.mark.parametrize("mode", ["source", "replay"])
    @pytest.mark.parametrize("architecture,variant", slots, ids=slot_ids)
    def test_rejects_perturbed_gpu_results(
        architecture, variant, reference_bundle, monkeypatch, mode
    ):
        target, bundle, manifest = reference_bundle(architecture, variant)
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
                case,
                bundle=bundle,
                manifest=manifest,
                architecture=architecture,
                variant=variant,
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("architecture,variant", slots, ids=slot_ids)
    def test_rejects_missing_gpu_launch(
        architecture, variant, reference_bundle, monkeypatch, tmp_path
    ):
        import rocke
        from rocke.runtime import KernelLauncher

        run = importlib.import_module(operation.worker_module).run
        target, bundle, manifest = reference_bundle(architecture, variant)
        case = target.CASES[0]
        key = operation.manifest_key(
            manifest, case, architecture=architecture, variant=variant
        )
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
                    "architecture": architecture,
                    "variant": variant,
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
