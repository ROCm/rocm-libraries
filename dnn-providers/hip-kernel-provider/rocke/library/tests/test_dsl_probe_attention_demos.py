# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only CLI target contracts, stopping at the platform probe boundary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from rocke.assets import dsl_docs_dir
from rocke.core.ir import KernelDef

_PROBE_MODULES = (
    "probe_occupancy",
    "probe_config_sweep",
    "probe_intrinsic_counts",
    "probe_isa_inspect",
    "probe_lowering_compare",
)
_RDNA_ARCHES = ("gfx1151", "gfx1201", "gfx1250")


@pytest.fixture
def demos():
    """Load the loose driver without leaving its probe imports or path behind."""
    driver_path = (
        Path(__file__).resolve().parents[1]
        / "builders"
        / "common"
        / "dsl_probe_attention_demos.py"
    )
    probe_dir = dsl_docs_dir() / "optimization" / "utilities" / "tools" / "dsl_probes"
    if not all((probe_dir / f"{name}.py").is_file() for name in _PROBE_MODULES):
        pytest.skip(f"attention CLI probe source assets unavailable: {probe_dir}")
    saved_path = sys.path[:]
    saved_modules = {
        name: sys.modules[name] for name in _PROBE_MODULES if name in sys.modules
    }
    try:
        # Fresh probe modules prevent another loose tool's imports from choosing
        # which implementation this driver binds; restore existing ones below.
        for name in _PROBE_MODULES:
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(
            "_attention_demo_driver", driver_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = saved_path
        for name in _PROBE_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


@pytest.mark.parametrize("arch", ("gfx942", *_RDNA_ARCHES))
@pytest.mark.parametrize("probe", ("config_sweep", "intrinsic_counts"))
def test_requested_target_reaches_platform_probe(demos, monkeypatch, probe, arch):
    boundary = Mock()
    monkeypatch.setattr(demos, f"probe_{probe}", boundary)

    assert demos.main(["--probe", probe, "--arch", arch]) == 0

    boundary.assert_called_once()
    args, kwargs = boundary.call_args
    # Both platform probes default to gfx950 when the driver forgets arch.
    assert kwargs.get("arch", "gfx950") == arch
    if probe == "config_sweep":
        base_spec = kwargs["base_spec"]
        assert isinstance(base_spec, demos.UnifiedAttention2DTiledSpec)
        # The config demo hands its real builder to the platform sweep rather
        # than building eagerly. Exercise that builder without compiling.
        assert isinstance(kwargs["build_fn"](base_spec), KernelDef)
    else:
        entries = args[0]
        assert entries
        assert all(isinstance(kernel, KernelDef) for _, kernel in entries)


@pytest.mark.parametrize("arch", _RDNA_ARCHES)
@pytest.mark.parametrize(
    "probe, target_argument",
    (("isa_inspect", "mcpu"), ("lowering_compare", "arch")),
)
def test_rdna_target_is_accepted_by_codegen_demos(
    demos, monkeypatch, probe, target_argument, arch
):
    boundary = Mock()
    monkeypatch.setattr(demos, f"probe_{probe}", boundary)

    assert demos.main(["--probe", probe, "--arch", arch]) == 0

    boundary.assert_called_once()
    args, kwargs = boundary.call_args
    assert kwargs[target_argument] == arch
    entries = args[0]
    assert entries
    assert all(isinstance(kernel, KernelDef) for _, kernel in entries)


@pytest.mark.parametrize("arch", _RDNA_ARCHES)
def test_occupancy_rejects_targets_without_resource_caps(
    demos, monkeypatch, capsys, arch
):
    boundary = Mock()
    monkeypatch.setattr(demos, "probe_occupancy", boundary)

    with pytest.raises(SystemExit) as exc_info:
        demos.main(["--probe", "occupancy", "--arch", arch])

    assert exc_info.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "--probe occupancy" in diagnostic
    assert "resource caps only for" in diagnostic
    assert arch in diagnostic
    boundary.assert_not_called()
