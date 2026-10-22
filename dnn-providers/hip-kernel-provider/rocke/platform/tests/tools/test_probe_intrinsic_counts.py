# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Intrinsic histograms must exclude measured target failures and keep going."""

import importlib.util
import warnings

import pytest
from rocke.assets import dsl_docs_dir
from rocke.core.arch import domain as arch_domain
from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import ArchDomainWarning, lower_kernel_to_llvm


@pytest.fixture
def intrinsic_probe():
    path = (
        dsl_docs_dir()
        / "optimization/utilities/tools/dsl_probes/probe_intrinsic_counts.py"
    )
    if not path.is_file():
        pytest.skip("intrinsic probe source tools are absent from this test layout")
    spec = importlib.util.spec_from_file_location("rocke_intrinsic_probe_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measured_target_failure_skips_histogram_and_restores_warning_policy(
    intrinsic_probe, monkeypatch, capsys
):
    arch, flavor = "gfx942", "llvm22"
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    monkeypatch.setenv("ROCKE_ARCH_DOMAIN", "warn")
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)

    table = arch_domain.load(flavor)
    assert table is not None
    negative = table.lookup("ds.read.tr16.b64", arch)
    supported = table.lookup("ds.bpermute", arch)
    assert negative is not None and negative.status == arch_domain.STATUS_ARCH_ABSENT
    assert supported is not None and supported.status == arch_domain.STATUS_OK

    unsupported = IRBuilder("unsupported_transpose_read")
    unsupported.kernel.attrs["max_workgroup_size"] = 64
    smem = unsupported.smem_alloc(F16, [16, 16])
    zero = unsupported.const_i32(0)
    unsupported.ds_read_tr16_b64(smem, zero, zero)
    unsupported.ret()

    valid = IRBuilder("supported_permute")
    valid.kernel.attrs["max_workgroup_size"] = 64
    out = valid.param("out", PtrType(I32, "global"))
    data = valid.param("data", I32)
    valid.store(valid.ds_bpermute(valid.const_i32(0), data), out)
    valid.ret()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ArchDomainWarning)
        rows = intrinsic_probe.probe_intrinsic_counts(
            [("unsupported", unsupported.kernel), ("valid", valid.kernel)], arch=arch
        )
        output = capsys.readouterr().out
        assert [row["label"] for row in rows] == ["valid"]
        # The histogram counts symbol references: one call and one declaration.
        assert rows[0]["intrinsics"]["ds.bpermute"] == 2
        assert rows[0]["intrinsics"]["ds.read.tr16.b64"] == 0
        failure = "=== unsupported === LOWER-FAIL: ArchDomainWarning:"
        assert failure in output
        diagnostic = output.split(failure, 1)[1].splitlines()[0]
        assert arch in diagnostic and "ds.read.tr16.b64" in diagnostic
        assert output.index(failure) < output.index("=== valid ===")
        assert not caught

        # Without changing the caller's filter, ordinary lowering must warn,
        # not raise: the probe's local error policy must have been restored.
        ir = lower_kernel_to_llvm(unsupported.kernel, arch=arch, llvm_flavor=flavor)
        assert "@llvm.amdgcn.ds.read.tr16.b64" in ir
        domain_warnings = [
            warning.message
            for warning in caught
            if isinstance(warning.message, ArchDomainWarning)
        ]
        assert [(w.key, w.arch, w.flavor) for w in domain_warnings] == [
            ("ds.read.tr16.b64", arch, flavor)
        ]
