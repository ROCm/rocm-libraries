# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""dsl_probes: target selection, per-entry failures and instruction counting.

Host-only. The compile probes are driven with a kernel the requested target
can't lower, so each entry fails during lowering and no compiler runs.
"""

import importlib.util
import sys
import warnings
from dataclasses import dataclass

import pytest
from rocke.assets import dsl_docs_dir
from rocke.core.arch import domain as arch_domain
from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import ArchDomainWarning, lower_kernel_to_llvm

_ARCH, _FLAVOR = "gfx942", "llvm22"


def _load(name):
    path = dsl_docs_dir() / "optimization/utilities/tools/dsl_probes" / f"{name}.py"
    if not path.is_file():
        pytest.skip("dsl_probes source tools are absent from this test layout")
    spec = importlib.util.spec_from_file_location(f"rocke_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve string annotations through sys.modules while the
    # module body runs; unregister afterwards so nothing leaks between tests.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture
def domain_lane(monkeypatch):
    """Pin the lane so the measured target failure is deterministic."""
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    monkeypatch.setenv("ROCKE_ARCH_DOMAIN", "warn")
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", _FLAVOR)
    cell = arch_domain.load(_FLAVOR).lookup("ds.read.tr16.b64", _ARCH)
    assert cell is not None and cell.status == arch_domain.STATUS_ARCH_ABSENT


def _transpose_read_kernel(name):
    """A kernel gfx942 provably can't lower (ds.read.tr16.b64 is gfx950+)."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    smem = b.smem_alloc(F16, [16, 16])
    zero = b.const_i32(0)
    b.ds_read_tr16_b64(smem, zero, zero)
    b.ret()
    return b.kernel


# ---------------------------------------------------------------------------
# Compile probes: the requested target reaches the compiler, and a kernel it
# can't lower fails only its own entry.
# ---------------------------------------------------------------------------


def _assert_target_failures(output, marker, labels):
    for label in labels:
        line = next(ln for ln in output.splitlines() if label in ln and marker in ln)
        assert "ArchDomainWarning" in line, line
        assert _ARCH in line and "ds.read.tr16.b64" in line, line


def test_isa_inspect_fails_each_entry_for_the_requested_target(domain_lane, capsys):
    probe = _load("probe_isa_inspect")
    entries = [(n, _transpose_read_kernel(n)) for n in ("first", "second")]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ArchDomainWarning)
        assert probe.probe_isa_inspect(entries, mcpu=_ARCH) == []
    _assert_target_failures(capsys.readouterr().out, "BUILD-FAIL", ("first", "second"))


def test_occupancy_fails_each_entry_for_the_requested_target(domain_lane, capsys):
    probe = _load("probe_occupancy")
    entries = [(n, _transpose_read_kernel(n), 1) for n in ("first", "second")]
    rows = probe.probe_occupancy(entries, arch=probe.ARCH_BY_NAME[_ARCH])
    assert rows == []
    _assert_target_failures(capsys.readouterr().out, "BUILD-FAIL", ("first", "second"))


def test_lowering_compare_fails_each_entry_for_the_requested_target(
    domain_lane, capsys, tmp_path
):
    probe = _load("probe_lowering_compare")
    entries = [(n, _transpose_read_kernel(n)) for n in ("first", "second")]
    assert probe.probe_lowering_compare(entries, arch=_ARCH, out_dir=tmp_path) == []
    _assert_target_failures(capsys.readouterr().out, "LLVM-FAIL", ("first", "second"))


@dataclass(frozen=True)
class _Spec:
    variant: int = 0


def test_config_sweep_fails_each_variant_for_the_requested_target(domain_lane, capsys):
    probe = _load("probe_config_sweep")
    rows = probe.probe_config_sweep(
        build_fn=lambda spec: _transpose_read_kernel(f"v{spec.variant}"),
        base_spec=_Spec(),
        overrides=[{}, {"variant": 1}],
        only_build=True,
        arch=_ARCH,
    )
    assert [r["label"] for r in rows] == ["baseline", "variant=1"]
    for row in rows:
        assert row["error"].startswith("BUILD: ArchDomainWarning:")
        assert _ARCH in row["error"] and "ds.read.tr16.b64" in row["error"]
        assert "hsaco_bytes" not in row
    assert capsys.readouterr().out.count("BUILD-FAIL") == 2


def test_intrinsic_counts_skips_unsupported_entry_and_restores_warning_policy(
    domain_lane, capsys
):
    probe = _load("probe_intrinsic_counts")
    unsupported = _transpose_read_kernel("unsupported")

    valid = IRBuilder("supported_permute")
    valid.kernel.attrs["max_workgroup_size"] = 64
    out = valid.param("out", PtrType(I32, "global"))
    data = valid.param("data", I32)
    valid.store(valid.ds_bpermute(valid.const_i32(0), data), out)
    valid.ret()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ArchDomainWarning)
        rows = probe.probe_intrinsic_counts(
            [("unsupported", unsupported), ("valid", valid.kernel)], arch=_ARCH
        )
        output = capsys.readouterr().out
        assert [row["label"] for row in rows] == ["valid"]
        # The histogram counts symbol references: one call and one declaration.
        assert rows[0]["intrinsics"]["ds.bpermute"] == 2
        assert rows[0]["intrinsics"]["ds.read.tr16.b64"] == 0
        failure = "=== unsupported === LOWER-FAIL: ArchDomainWarning:"
        assert failure in output
        diagnostic = output.split(failure, 1)[1].splitlines()[0]
        assert _ARCH in diagnostic and "ds.read.tr16.b64" in diagnostic
        assert output.index(failure) < output.index("=== valid ===")
        assert not caught

        # Without changing the caller's filter, ordinary lowering must warn,
        # not raise: the probe's local error policy must have been restored.
        ir = lower_kernel_to_llvm(unsupported, arch=_ARCH, llvm_flavor=_FLAVOR)
        assert "@llvm.amdgcn.ds.read.tr16.b64" in ir
        domain_warnings = [
            w.message for w in caught if isinstance(w.message, ArchDomainWarning)
        ]
        assert [(w.key, w.arch, w.flavor) for w in domain_warnings] == [
            ("ds.read.tr16.b64", _ARCH, _FLAVOR)
        ]


# ---------------------------------------------------------------------------
# Counting: the names rocKE actually emits on CDNA, RDNA and gfx12.
# ---------------------------------------------------------------------------


def test_intrinsic_patterns_match_pre_gfx950_mfma_and_wmma_names():
    probe = _load("probe_intrinsic_counts")
    # Intrinsic names as rocKE's lowering emits them. Pre-gfx950 MFMA names
    # have no dot before the type; gfx950's 16x16x32 must not count as 16x16x16.
    ir = "\n".join(
        f"  %r{i} = call <4 x float> @llvm.amdgcn.{name}()"
        for i, name in enumerate(
            (
                "mfma.f32.16x16x16f16",
                "mfma.f32.16x16x16bf16.1k",
                "mfma.f32.32x32x8f16",
                "mfma.f32.4x4x4f16",
                "mfma.f32.16x16x32.f16",
                "wmma.f32.16x16x16.f16.v8f32.v8f16",
                "wmma.f32.16x16x32.bf16.v8f32.v16bf16",
                "swmmac.f32.16x16x32.f16.v8f32.v8f16.v16f16.i16",
            )
        )
    )
    got = probe.count_intrinsics(ir)["intrinsics"]
    assert got["mfma.f32.16x16x16.f16"] == 1
    assert got["mfma.f32.16x16x16.bf16"] == 1
    assert got["mfma.f32.32x32x8.f16"] == 1
    assert got["mfma.f32.4x4x4.f16"] == 1
    assert got["mfma.f32.16x16x32.f16"] == 1
    assert got["wmma (any)"] == 2
    assert got["swmmac (any)"] == 1


def test_isa_inspect_categories_count_rdna_and_gfx12_instructions():
    probe = _load("probe_isa_inspect")
    # Verbatim llvm-objdump lines from gfx1201 / gfx1151 WMMA FMHA builds.
    isa = """
\tv_wmma_f32_16x16x16_f16 v[32:39], v[40:43], v[91:94], 0    // 000000001D90: CC404020 1A02B728
\tglobal_load_b128 v[40:43], v[0:1], off                     // 000000001C50: EE05C07C 00000028 00000000
\tglobal_store_b16 v[46:47], v50, off                        // 000000002CE4: EE06407C 19000000 0000002E
\tscratch_load_b128 v[154:157], off, off                     // 000000001F5C: DC5D0000 9A7C0000
\tscratch_store_b128 off, v[2:5], off                        // 000000001DE0: DC750000 007C0200
\tds_load_b128 v[84:87], v82                                 // 00000000297C: DBFC0000 54000052
\tds_store_b128 v80, v[91:94] offset:512                     // 000000002864: DB7C0200 00005B50
\ts_wait_loadcnt 0x0                                         // 000000001D8C: BFC00000
\ts_wait_dscnt 0x0                                           // 000000001E6C: BFC60000
\ts_wait_alu 0xfffe                                          // 000000001BBC: BF88FFFE
\ts_barrier_signal -1                                        // 000000002974: BE804EC1
\ts_barrier_wait 0xffff                                      // 000000002978: BF94FFFF
"""
    cats, waits = probe.categorize_isa(isa)
    assert cats["wmma"] == 1
    assert cats["vmem_load"] == 1 and cats["vmem_store"] == 1
    assert cats["scratch_load"] == 1 and cats["scratch_store"] == 1
    assert cats["ds_read"] == 1 and cats["ds_write"] == 1
    assert cats["waitcnt"] == 2 and cats["s_wait_alu"] == 1
    assert cats["barrier"] == 1
    # Memory ops counted as a headline category are not also "other".
    assert "other" not in cats
    # gfx12 split waits keep their counter name.
    assert waits == {"s_wait_loadcnt 0x0": 1, "s_wait_dscnt 0x0": 1}
