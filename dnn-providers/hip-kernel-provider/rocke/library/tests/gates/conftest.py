# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Preflight for explicit resource-bound library gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NoReturn

import pytest
from rocke.core.llvm_profile import measured_flavor_for_llvm
from rocke.runtime.comgr import loaded_compiler_info
from rocke.tests.lanes import TICKET, skip_until_reason

from tests._attention_bwd_counted_cases import MFMA_ARCHS, selected_isa_arch
from tests._attention_bwd_resources import (
    GATE_COMGR_ROCM,
    GATE_FLAVOR,
    comgr_is_production,
)
from tests._explicit_gate_policy import (
    configured_selection_is_partial,
    selection_is_partial,
)
from tests.gates import mark_preflight_loaded

_COMPILER_GATE = "attention_bwd_counted_compiler_gate.py"
_GATE_FILES = (_COMPILER_GATE, "attention_bwd_counted_structural_gate.py")
_OPTIONAL_ENV = "ROCKE_COUNTED_ISA_OPTIONAL"
_SENTINEL_ENV = "ROCKE_EXPLICIT_GATE_SENTINEL"


def _stop(message: str) -> NoReturn:
    optional = os.environ.get(_OPTIONAL_ENV, "").strip() not in ("", "0")
    suffix = " (optional gate skipped)" if optional else ""
    pytest.exit(message + suffix, returncode=77 if optional else 1)


def _unanswered_query(
    active_flavor: str | None, targets: frozenset[str] | None
) -> str | None:
    """Which toolchain query failed, or ``None``: a broken toolchain, not
    merely another one, fails the gate."""
    if active_flavor is None:
        return "could not query the active comgr compiler flavor"
    if targets is None:
        return "could not query comgr ISA targets"
    return None


def _reference_toolchain_gap(
    requested_flavor: str, active_flavor: str, comgr: str | None
) -> str | None:
    """Why the toolchain is not the one the gate's thresholds were calibrated on
    (``GATE_FLAVOR`` requested and run by the loaded compiler, with the
    ``GATE_COMGR_ROCM`` comgr), or ``None`` when it is. Tests use the compiler
    rocKE's discovery loads, so on any other one the gate skips; the pin stays
    here until AICK-3320 unpins it."""
    if requested_flavor != GATE_FLAVOR:
        return (
            f"resource gate requires LLVM flavor {GATE_FLAVOR}, "
            f"got {requested_flavor!r}"
        )
    if active_flavor != GATE_FLAVOR:
        return (
            f"resource gate requires a loaded compiler of LLVM flavor "
            f"{GATE_FLAVOR}, got {active_flavor!r}"
        )
    if not comgr_is_production(comgr):
        return f"resource gate requires ROCm {GATE_COMGR_ROCM} comgr, got {comgr}"
    return None


# What the preflight does about a toolchain: fail the gate, or skip it.
_STOP = "stop"
_SKIP = "skip"


def _preflight_outcome(
    requested_flavor: str,
    active_flavor: str | None,
    comgr: str | None,
    targets: frozenset[str] | None,
    arches: tuple[str, ...],
) -> tuple[str, str] | None:
    """``(_STOP, why)`` for a broken toolchain (a query that fails, or the
    reference toolchain without a gate arch), ``(_SKIP, why)`` for a toolchain
    that is not the reference one, or ``None`` for the reference toolchain."""
    if failed := _unanswered_query(active_flavor, targets):
        return _STOP, failed
    if gap := _reference_toolchain_gap(requested_flavor, active_flavor, comgr):
        return _SKIP, gap
    missing = [arch for arch in arches if arch not in targets]
    if missing:
        return _STOP, f"comgr {comgr} does not support {', '.join(missing)}"
    return None


def _skip_for_reference_toolchain(session, gap: str) -> None:
    """Skip every gate case, as waiting for AICK-3320, on a compiler that is not
    the gate's reference toolchain. The session then ends with exit 77, which
    the CTest entry's SKIP_RETURN_CODE reports as a skip and a run that requires
    the gate (``run_all.py --require-compiler-gate``) counts as a failure; the
    session attestation carries the reason."""
    reason = skip_until_reason(
        TICKET,
        reason=f"counted ISA gate {gap}; the gate keeps its reference toolchain "
        "until AICK-3320 unpins it",
    )
    for item in session.items:
        item.add_marker(pytest.mark.skip(reason=reason))
    session.config._rocke_gate_skipped_for = reason


def pytest_configure(config) -> None:
    """Mark that the gate's non-bypassable conftest loaded for this invocation."""
    arguments = tuple(map(str, config.invocation_params.args))
    if any(gate in argument for gate in _GATE_FILES for argument in arguments):
        mark_preflight_loaded()
        os.environ["ROCKE_EXPLICIT_GATE_PREFLIGHT"] = "1"


def _gate_invoked(session) -> bool:
    arguments = tuple(map(str, session.config.invocation_params.args))
    return any(
        gate in argument for gate in _GATE_FILES for argument in arguments
    ) or any(Path(item.path).name in _GATE_FILES for item in session.items)


def _is_direct_gate_path(argument: str) -> bool:
    """Whether an argument selects exactly one complete explicit gate file."""
    if "::" in argument:
        return False
    return argument.replace("\\", "/").rsplit("/", 1)[-1] in _GATE_FILES


def _selection_requested(config) -> bool:
    arguments = list(map(str, config.invocation_params.args))
    option = config.option
    configured = any(
        bool(getattr(option, name, False))
        for name in (
            "keyword",
            "markexpr",
            "collectonly",
            "lf",
            "failedfirst",
            "newfirst",
            "ignore",
            "ignore_glob",
            "setupplan",
            "setuponly",
            "stepwise",
            "stepwise_skip",
            "deselect",
        )
    )
    selection_arguments = [
        argument for argument in arguments if not _is_direct_gate_path(argument)
    ]
    selected_paths = [
        str(argument)
        for argument in config.args
        if not _is_direct_gate_path(str(argument))
    ]
    return (
        configured
        or bool(selected_paths)
        or selection_is_partial(selection_arguments)
        or configured_selection_is_partial(
            override_ini=getattr(option, "override_ini", ()) or (),
            plugins=getattr(option, "plugins", ()) or (),
            inifilename=getattr(option, "inifilename", None),
            environment_plugins=os.environ.get("PYTEST_PLUGINS"),
        )
    )


def pytest_deselected(items) -> None:
    """Record any plugin-driven item removal for complete-gate enforcement."""
    if items:
        config = items[0].config
        config._rocke_gate_deselected = getattr(
            config, "_rocke_gate_deselected", 0
        ) + len(items)


def pytest_sessionfinish(session, exitstatus) -> None:
    """End a session whose cases all skipped off the reference toolchain with
    exit 77, and attest that pytest created a real gate session for the parent
    runner."""
    skipped_for = getattr(session.config, "_rocke_gate_skipped_for", None)
    if skipped_for is not None and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = exitstatus = 77
    output = os.environ.get(_SENTINEL_ENV, "").strip()
    if output and _gate_invoked(session):
        Path(output).write_text(
            json.dumps(
                {
                    "collected": session.testscollected,
                    "exitstatus": int(exitstatus),
                    "skipped_for": skipped_for,
                }
            ),
            encoding="utf-8",
        )


def pytest_collection_finish(session) -> None:
    """Enforce complete-gate selection and validate compiler prerequisites once."""
    if not _gate_invoked(session):
        return
    partial = os.environ.get("ROCKE_EXPLICIT_GATE_PARTIAL", "").strip() == "1"
    if (
        os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "").strip() != "1"
        and not partial
    ):
        _stop("complete explicit gates require disabled plugin autoload")
    if os.environ.get("ROCKE_TEST_SHARD", "").strip() and not partial:
        _stop("explicit gate sharding needs explicit partial mode")
    if _selection_requested(session.config) and not partial:
        _stop("explicit gate selection filters need explicit partial mode")
    if getattr(session.config, "_rocke_gate_deselected", 0) and not partial:
        _stop("explicit gate item deselection needs explicit partial mode")
    if not any(Path(item.path).name == _COMPILER_GATE for item in session.items):
        return

    from tests._attention_bwd_resources import (
        _probe_isa_inspect,
        comgr_targets,
        comgr_version,
        current_flavor,
        find_objdump,
        validate_objdump,
    )

    selected_arch = selected_isa_arch()
    if selected_arch and not partial:
        _stop(
            f"counted ISA gate arch filter {selected_arch!r} needs explicit partial mode"
        )
    if selected_arch and selected_arch not in MFMA_ARCHS:
        _stop(f"counted ISA gate does not support arch {selected_arch!r}")
    arches = (selected_arch,) if selected_arch else MFMA_ARCHS

    try:
        flavor = current_flavor()
        info = loaded_compiler_info()
        active_flavor = (
            measured_flavor_for_llvm(info.llvm_version[0])
            if info is not None and info.llvm_version is not None
            else None
        )
        comgr = comgr_version()
        targets = comgr_targets()
    except Exception as exc:  # noqa: BLE001 - one gate-level diagnostic
        _stop(f"counted ISA gate toolchain preflight failed: {exc}")
    outcome = _preflight_outcome(flavor, active_flavor, comgr, targets, arches)
    if outcome is not None:
        action, why = outcome
        if action == _SKIP:
            _skip_for_reference_toolchain(session, why)
            return
        _stop(f"counted ISA gate {why}")
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            "counted-compiler-toolchain: "
            f"flavor={active_flavor} comgr={comgr} arches={','.join(arches)}"
        )
    objdump = find_objdump()
    if objdump is None:
        _stop("counted ISA gate needs llvm-objdump")
    if gap := validate_objdump(objdump):
        _stop(f"counted ISA gate llvm-objdump is unusable: {gap}")
    try:
        _probe_isa_inspect()
    except Exception as exc:  # noqa: BLE001 - one gate-level diagnostic
        _stop(f"counted ISA gate cannot load disassembly support: {exc}")
