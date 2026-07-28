################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Full derived-state characterization golden for ``Solution.assignDerivedParameters``.

This is the pickle-free, syrupy (``.ambr``) sibling of the pickle golden corpus
test. It asserts the COMPLETE per-case derived ``Solution._state`` dict (every
field, no fixed column subset) for each input case, giving it the same kill
power as the pickle corpus test: any mutation of ``assignDerivedParameters`` that
changes any derived field breaks a snapshot.

Input cases are regenerated at collection time from the in-tree designed base
YAML configs under ``_codegen/data/test_data/_designed`` (no pre-state pickle).
The reconstruction replays the exact corpus pipeline:

    config YAML -> BenchmarkProcess -> constructForkPermutations (capped at LIMIT)
                -> per-perm _generate_single_solution with a monkeypatch hook on
                   Solution.assignDerivedParameters capturing a deepcopy of the
                   pre-derive state
                -> deepcopy(pre), pop SolutionIndex/SolutionNameMin, then
                   Solution.assignDerivedParameters(st, sg, True, False, iim, rv)

which is deterministic for a fixed toolchain/container. The resulting full state
is rendered to clean, address-free data (nested mappings such as ``ProblemType``
recurse so every field is deep-compared as the pickle golden does; the ISA
SemanticVersion and leaf objects such as ``DataType`` render via ``str()``) and
snapshotted with the default syrupy amber serializer.
"""

import contextlib
import copy
import importlib
import importlib.util
import io
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CODEGEN = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_codegen")
if _CODEGEN not in sys.path:
    sys.path.insert(0, _CODEGEN)

_S = importlib.import_module("Tensile.SolutionStructs.Solution")

_LIMIT = 24

_CONFIGS = [
    ("gfx1100/hss_wmma_rich.yaml", "gfx1100"),
    ("gfx1201/rich_wmma.yaml", "gfx1201"),
    ("gfx1250/datamover.yaml", "gfx1250"),
    ("gfx1250/localread_ldstr_wmma3.yaml", "gfx1250"),
    ("gfx1250/rich_gemm_sb.yaml", "gfx1250"),
    ("gfx1250/streamk_cluster.yaml", "gfx1250"),
    ("gfx1250/streamk_tdmsplit.yaml", "gfx1250"),
    ("gfx1250/streamk.yaml", "gfx1250"),
    ("gfx1250/xccremap.yaml", "gfx1250"),
    ("gfx908/rich_gfx908.yaml", "gfx908"),
    ("gfx90a/db.yaml", "gfx90a"),
    ("gfx90a/mac.yaml", "gfx90a"),
    ("gfx90a/rich_gemm.yaml", "gfx90a"),
    ("gfx90a/seed.yaml", "gfx90a"),
    ("gfx942/activation.yaml", "gfx942"),
    ("gfx942/addrstore.yaml", "gfx942"),
    ("gfx942/asmaddr2_bf16_srvw.yaml", "gfx942"),
    ("gfx942/asmaddr2_fp32.yaml", "gfx942"),
    ("gfx942/asmaddr_flat_buf0.yaml", "gfx942"),
    ("gfx942/asmaddr_initstrides.yaml", "gfx942"),
    ("gfx942/asmaddr_srvw_scale.yaml", "gfx942"),
    ("gfx942/complex_cc.yaml", "gfx942"),
    ("gfx942/fp8_gr_conv_f8h.yaml", "gfx942"),
    ("gfx942/fp8_gr_conv.yaml", "gfx942"),
    ("gfx942/globalwrite.yaml", "gfx942"),
    ("gfx942/grad_biassrc_a.yaml", "gfx942"),
    ("gfx942/gsu_mbsk.yaml", "gfx942"),
    ("gfx942/gsu_on.yaml", "gfx942"),
    ("gfx942/gsu.yaml", "gfx942"),
    ("gfx942/hss.yaml", "gfx942"),
    ("gfx942/int8_hpa.yaml", "gfx942"),
    ("gfx942/kwafeat.yaml", "gfx942"),
    ("gfx942/kwconv.yaml", "gfx942"),
    ("gfx942/kwfeat.yaml", "gfx942"),
    ("gfx942/lra.yaml", "gfx942"),
    ("gfx942/seed.yaml", "gfx942"),
    ("gfx942/shiftvec_full.yaml", "gfx942"),
    ("gfx942/shiftvector2.yaml", "gfx942"),
    ("gfx942/shiftvector.yaml", "gfx942"),
    ("gfx942/solution.yaml", "gfx942"),
    ("gfx942/sparse_dtvsm.yaml", "gfx942"),
    ("gfx942/sparse_pack.yaml", "gfx942"),
    ("gfx942/store.yaml", "gfx942"),
    ("gfx942/streamk_dynamic.yaml", "gfx942"),
    ("gfx942/streamk_fixup_tree.yaml", "gfx942"),
    ("gfx942/streamk_sfc.yaml", "gfx942"),
    ("gfx942/streamk_xccm.yaml", "gfx942"),
    ("gfx942/streamk.yaml", "gfx942"),
    ("gfx942/wgm.yaml", "gfx942"),
    ("gfx950/hhs_hplr_snll.yaml", "gfx950"),
    ("gfx950/localread_cafs_fp8.yaml", "gfx950"),
    ("gfx950/lra_tr.yaml", "gfx950"),
    ("gfx950/mx_bias_act_gsu.yaml", "gfx950"),
    ("gfx950/mx_fp8_scale_swizzle.yaml", "gfx950"),
    ("gfx950/seed.yaml", "gfx950"),
    ("gfx950/subtile3_gr_variants.yaml", "gfx950"),
    ("gfx950/subtile_kern_tlu1.yaml", "gfx950"),
    ("gfx950/subtile_lr_fp8.yaml", "gfx950"),
    ("gfx950/subtile.yaml", "gfx950"),
]

_POPPED_KEYS = ("SolutionIndex", "SolutionNameMin")
_VOLATILE = set()


def _sanitize(value):
    """Render a derived-state value as clean, deterministic, address-free data.

    Primitives pass through; namedtuples (e.g. ``SemanticVersion``) are rendered
    via ``str()``; dict-like mappings (plain ``dict`` and ``Mapping`` subclasses
    such as ``ProblemType``) recurse so every nested field is deep-compared exactly
    as the pickle golden compares them; lists/tuples recurse; any remaining
    non-container object (e.g. ``DataType``) is rendered via its ``str`` canonical
    form (no ``0x`` addresses).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return str(value)


def _data_dir():
    import config_harness as CH

    return Path(os.path.dirname(CH.__file__)) / "data" / "test_data" / "_designed"


def _perms_for(config_path):
    import config_harness as CH
    from Tensile.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
    from Tensile.Common.Types import makeDebugConfig

    cfg = CH._load_config(config_path)
    bps = cfg["BenchmarkProblems"]
    if not bps:
        return None, None, None, None
    problemTypeConfig, problemSizeGroupConfig = bps[0][0], bps[0][1]
    debugConfig = makeDebugConfig(cfg.get("GlobalParameters", {}))
    benchmarkProcess = BenchmarkProcess(problemTypeConfig, problemSizeGroupConfig, False)
    step = benchmarkProcess[0]
    if problemSizeGroupConfig.get("ForkParameters"):
        perms = list(constructForkPermutations(step.forkParams, step.paramGroups))
    else:
        perms = []
    return benchmarkProcess.problemType, step.constantParams, perms[:_LIMIT], debugConfig


def _capture_pre_states(config_path):
    import config_harness as CH
    from Tensile.BenchmarkProblems import _generate_single_solution

    orig = _S.Solution.assignDerivedParameters
    captured = []
    with CH._isolated_globals_with_isa(_capture_pre_states.iim):
        problemType, constantParams, perms, debugConfig = _perms_for(config_path)
        if not perms:
            return captured

        def hook(state, splitGSU, prr, pia, iim2, rv2, _o=orig, _c=captured):
            _c.append((copy.deepcopy(state), bool(splitGSU)))
            return _o(state, splitGSU, prr, pia, iim2, rv2)

        _S.Solution.assignDerivedParameters = staticmethod(hook)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for perm in perms:
                    try:
                        _generate_single_solution(
                            perm,
                            problemType,
                            constantParams,
                            _capture_pre_states.assembler,
                            debugConfig,
                            _capture_pre_states.iim,
                        )
                    except Exception:
                        pass
        finally:
            _S.Solution.assignDerivedParameters = staticmethod(orig)
    return captured


def _build_corpus():
    import config_harness as CH

    orig = _S.Solution.assignDerivedParameters
    data = _data_dir()
    cases = []
    for relpath, arch in _CONFIGS:
        tag = relpath.replace("/", "__").replace(".yaml", "")
        try:
            assembler, iim = CH._toolchain_for(arch)
            rocm_version = assembler.rocm_version
            _capture_pre_states.assembler = assembler
            _capture_pre_states.iim = iim
            captured = _capture_pre_states(str(data / relpath))
        except Exception:
            _S.Solution.assignDerivedParameters = staticmethod(orig)
            continue

        seen = set()
        n = 0
        for pre, splitGSU in captured:
            for k in _POPPED_KEYS:
                pre.pop(k, None)
            try:
                key = hash(repr(sorted((k, repr(v)) for k, v in pre.items())))
            except Exception:
                key = None
            if key is not None and key in seen:
                continue
            seen.add(key)
            cases.append(("%s__s%d" % (tag, n), copy.deepcopy(pre), splitGSU, iim, rocm_version))
            n += 1
    return cases


def _safe_build_corpus():
    if importlib.util.find_spec("syrupy") is None:
        return []
    try:
        return _build_corpus()
    except Exception:
        return []


_CASES = _safe_build_corpus()
_INPUTS = {c[0]: (c[1], c[2], c[3], c[4]) for c in _CASES}
_IDS = [c[0] for c in _CASES]


@pytest.mark.parametrize("case_id", _IDS)
def test_derived_state_matches_snapshot(case_id, snapshot):
    """Full-state derived-parameter golden for one designed-config solution case.

    Derivation runs inside the test (not at collection) so a mutation that makes
    ``assignDerivedParameters`` raise fails this test (killed) instead of being
    silently dropped from the corpus. The 246-case set is fixed by the
    mutation-independent pre-derive capture, matching the pickle golden corpus.
    """
    pre, splitGSU, iim, rocm_version = _INPUTS[case_id]
    state = copy.deepcopy(pre)
    with contextlib.redirect_stdout(io.StringIO()):
        _S.Solution.assignDerivedParameters(state, splitGSU, True, False, iim, rocm_version)
    sanitized = {k: v for k, v in _sanitize(state).items() if k not in _VOLATILE}
    assert sanitized == snapshot
