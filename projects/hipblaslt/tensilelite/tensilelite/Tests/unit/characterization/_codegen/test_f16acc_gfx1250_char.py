# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# f16 accumulation kernels emitted in process, one size group at a time (the
# harness emits the first group of an entry, so every group is written out alone):
# - f16acc_streamk_gfx1250.yaml: every solution of every group builds; every
#   StreamK kernel has an f16 fixup that adds the partials in f32 and rounds the
#   sum to f16 once; a StaticGrid group also has the tree fixup; a linear fixup
#   adds the workgroup's own partial before the earlier pieces' partials under
#   StaticGrid, after them under DynamicWorkQueue, and a Hybrid kernel has both.
# - the gfx1250-strict yamls with the strict arch caps, so no gfx1250-strict
#   assembler is needed: the f16-accumulate fp8 WMMA, no pk8 convert and no
#   v_pk_maximum3_f16, the v_perm_b32 decode on each fp4 operand, and for an fp8
#   output the per-element store that merges the second pair of each VGPR with
#   v_perm_b32 0x5040100.
################################################################################

import contextlib
import copy
import os
import re

import pytest
import yaml

import config_harness
from tensilelite.Common.Capabilities import applyArchCapOverrides
from tensilelite.Common.GlobalParameters import globalParameters

pytestmark = pytest.mark.unit

_GFX12 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common", "gemm", "gfx12")
_STRICT = "gfx1250-strict"


def _groups(config):
    with open(os.path.join(_GFX12, config)) as f:
        entries = yaml.safe_load(f)["BenchmarkProblems"]
    return [(config, p, g) for p, entry in enumerate(entries) for g in range(1, len(entry))]


def _fork(group, name):
    return [v for item in group.get("ForkParameters", []) for v in item.get(name, [])]


def _emit(config, p, g, tmp_path, arch):
    with open(os.path.join(_GFX12, config)) as f:
        doc = yaml.safe_load(f)
    entry = doc["BenchmarkProblems"][p]
    path = tmp_path / ("%s_%d_%d.yaml" % (config[:-5], p, g))
    path.write_text(yaml.safe_dump({**{k: v for k, v in doc.items() if k != "BenchmarkProblems"},
                                    "BenchmarkProblems": [[entry[0], entry[g]]]}))
    results = config_harness.emit_kernels_from_config(str(path), limit=None, arch=arch, canonical=False)
    assert results and not [base for base, src, err in results if err or not src], (config, p, g)
    return entry[g], [src for _, src, _ in results]


def _lines(src):
    return [l for l in (l.split("//")[0].strip() for l in src.splitlines()) if l]


def _own_partial_order(src):
    # A linear fixup works block by block; each block adds the workgroup's own f16 partial (v_fma_mix_f32 from
    # v[vgprValuC..]) before or after its loop over the earlier pieces' partials.
    lines = _lines(src)
    starts = [i for i, l in enumerate(lines) if re.fullmatch(r"label_SK_F16_AddPartial(?!End)\w*:", l)]
    ends = [i for i, l in enumerate(lines) if re.fullmatch(r"label_SK_F16_AddPartialEnd\w*:", l)]
    own = [i for i, l in enumerate(lines) if re.match(r"v_fma_mix_f32 \S+, v\[vgprValuC", l)]
    assert own and starts and len(starts) == len(ends)
    assert not [i for i in own for s, e in zip(starts, ends) if s < i < e], "own partial inside a loop"
    first, last = any(i < starts[0] for i in own), any(i > ends[-1] for i in own)
    return "first" if first and not last else "last" if last and not first else "mixed"


_STREAMK = _groups("f16acc_streamk_gfx1250.yaml")


@pytest.mark.parametrize("config, p, g", _STREAMK, ids=["entry%d_group%d" % (p, g) for _, p, g in _STREAMK])
def test_f16acc_streamk_groups_emit_their_f16_fixups(tmp_path, config, p, g):
    group, sources = _emit(config, p, g, tmp_path, "gfx1250")
    fixups = [s for s in sources if "label_SK_F16_AddPartial:" in s]
    if set(_fork(group, "TileProcessingStrategy")) == {"StreamK"}:
        assert len(fixups) == len(sources)
    else:
        assert 0 < len(fixups) < len(sources)  # persistent DataParallel kernels have no fixup
    assert all("round the sum to f16" in s for s in fixups)
    assignments = set(_fork(group, "WorkAssignment"))
    tree = [s for s in fixups if "label_SK_F16_StoreSums:" in s]
    assert len(tree) < len(fixups) and (tree or "StaticGrid" not in assignments)
    # A Hybrid kernel holds a static and a dynamic fixup, so both orders.
    want = {order for assignment, order in (("StaticGrid", "first"), ("DynamicWorkQueue", "last"), ("Hybrid", "mixed"))
            if assignment in assignments}
    assert {_own_partial_order(s) for s in fixups if s not in tree} == want


@pytest.fixture
def _strict_caps(monkeypatch):
    # The harness builds the base ISA's caps; a gfx1250-strict build applies the stepping's caps and StinkyTofu tables.
    toolchain, isolated, cache = config_harness._toolchain_for, config_harness._isolated_globals_with_isa, {}

    def toolchainFor(arch):
        if arch not in cache:
            assembler, iim = toolchain(arch)
            iim = copy.deepcopy(iim)
            applyArchCapOverrides(iim, [arch])
            cache[arch] = (assembler, iim)
        return cache[arch]

    @contextlib.contextmanager
    def isolatedForArch(isaInfoMap):
        with isolated(isaInfoMap):
            globalParameters["StinkyTofuArchName"] = _STRICT
            yield

    monkeypatch.setattr(config_harness, "_toolchain_for", toolchainFor)
    monkeypatch.setattr(config_harness, "_isolated_globals_with_isa", isolatedForArch)


_CVT = re.compile(r"^v_cvt_pk_fp8_f32 (\S+), \S+, \S+(?: op_sel:\[0,0,([01])\])?$")
_PERM = re.compile(r"^v_perm_b32 (\S+), (\S+), (\S+), 0x5040100$")


def _fp8_merges(sources):
    # (v_perm_b32 merges of a just-converted pair into a destination VGPR, converts that write a high word)
    merges = high = 0
    for src in sources:
        recent = []
        for line in _lines(src):
            cvt, perm = _CVT.match(line), _PERM.match(line)
            high += bool(cvt) and cvt.group(2) == "1"
            merges += bool(perm) and perm.group(1) == perm.group(3) and perm.group(2) in recent
            recent = (recent + [cvt.group(1) if cvt else None])[-8:]
    return merges, high


_STRICT_CONFIGS = ["f8f8hh", "f8f4hh", "f4f8hh", "f4f4hh", "f8f8f8h", "f8f4f8h", "f8f8bh", "f8f4bh", "f4f8f8h",
                   "f4f4f8h", "f4f8bh", "f4f4bh", "f16acc_overflow", "f16acc_largek", "f16acc_mnf4"]
# Configs with an fp8 output at a store width of 4 or more.
_FP8_STORE_4 = {"f8f8f8h", "f8f4f8h", "f4f8f8h", "f4f4f8h", "f16acc_mnf4"}


@pytest.mark.parametrize("config", _STRICT_CONFIGS)
def test_gfx1250_strict_kernels_decode_fp4_and_store_fp8_without_converts(_strict_caps, tmp_path, config):
    name = config + "_gfx1250-strict.yaml"
    sources = [s for _, p, g in _groups(name) for s in _emit(name, p, g, tmp_path, _STRICT)[1]]
    text = "\n".join(sources)
    # f16acc_mnf4 has MN-contiguous fp4 A and B, read with ds_load_tr4_b64 before the same decode.
    fp4 = {"A", "B"} if config == "f16acc_mnf4" else set() if config.startswith("f16acc") else \
        {tc for tc, t in zip("AB", (config[0:2], config[2:4])) if t == "f4"}
    assert re.search(r"v_wmma_f16_16x16x(64|128)_fp8_fp8", text)
    assert "v_cvt_scale_pk8" not in text
    assert "v_cvt_scalef32_pk8_fp8_f16" not in text and "v_pk_maximum3_f16" not in text
    assert ("0x4c484440" in text) == bool(fp4)
    assert {tc for tc in "AB" if re.search(r"^\s*v_perm_b32 v\[vgprValu%s_" % tc, text, re.M)} == fp4
    merges, high = _fp8_merges(sources)
    assert high == 0 and (merges > 0 or config not in _FP8_STORE_4), (merges, high)
