# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Retained real store-D harness coverage for store trimming.

Feature nodes exercise deferred-body removal. Roundtrip nodes preserve exact
stores and canaries; replacing the real buffer-store emitter with no stores
breaks them. This harness masks incompatible sizes with subtile guards, so it
is deliberately NOT used as a launch-refusal oracle. Full GEMM and launch-path
coverage live in an external acceptance test outside this tree.
"""
import math
import re

import numpy as np
import pytest

import test_storeD_roundtrip as T
from gpu_test_helpers import TileConfig, init_rocisa


class _Kernel(dict):
    subtile = True

    def __setitem__(self, key, value):
        if key == "UseSubtileImpl":
            value = self.subtile
        super().__setitem__(key, value)


@pytest.fixture
def trimmed(monkeypatch):
    original = T._build_store_kernel

    def configure(af0, af1, subtile):
        def build(cfg, mi_wave_group=None, use_bf16=False):
            kernel = _Kernel(original(cfg, mi_wave_group=mi_wave_group, use_bf16=use_bf16))
            kernel.subtile = subtile
            kernel["AssertFree0ElementMultiple"] = af0
            kernel["AssertFree1ElementMultiple"] = af1
            kernel["BespokeNarrowing"] = True  # a bespoke request; stock keeps its edges
            kernel["UseSubtileImpl"] = subtile
            return kernel
        monkeypatch.setattr(T, "_build_store_kernel", build)
    return configure


C128 = TileConfig(mt_a=128, mt_b=128, depth_u=64)
C256 = TileConfig(mt_a=256, mt_b=128, depth_u=64)


def _asserts(mode, cfg):
    return (cfg.mt_a if mode in ("af0", "both") else 1,
            cfg.mt_b if mode in ("af1", "both") else 1)


@pytest.mark.parametrize("mode,deferred_vws", [
    ("none", {"1", "4"}), ("af0", {"4"}), ("af1", {"1"}), ("both", set()),
])
def test_subtile_deferred_edge_bodies(trimmed, monkeypatch, tmp_path, mode, deferred_vws):
    captured = {}

    def capture_assembly(asm, tmp_path, label, output_size, **kwargs):
        captured["asm"] = asm
        return bytes(output_size)

    # Only assembly/execution is replaced; notLocalSplitUGlobalWrite and the
    # real deferred modules still run. The returned bytes are not the oracle.
    monkeypatch.setattr(T, "assemble_and_run", capture_assembly)
    trimmed(*_asserts(mode, C128), subtile=True)
    init_rocisa(target="gfx950")
    T._run_storeD(C128, tmp_path, C128.mt_a, C128.mt_b, init_mode="random")
    src = captured["asm"]
    vws = set(re.findall(r"^label_GW_B0_FD0_VW(\d+)_Then_Deferred:", src, re.M))
    assert vws == deferred_vws, "unreachable deferred edge bodies survive"
    if mode == "both":
        assert not re.search(r"\bs_cbranch_scc1\s+label_GW_B0_FD0_VW\d+_(?:Then|Else)\b", src)


CASES = []
for cfg in (C128, C256):
    for sj in (1, 17, 64, 127, 128):
        CASES.append(("af0", cfg, cfg.mt_a, sj))
    for si in (1, 17, 64, cfg.mt_a - 1, cfg.mt_a):
        CASES.append(("af1", cfg, si, cfg.mt_b))
    CASES.append(("both", cfg, cfg.mt_a, cfg.mt_b))
    CASES.append(("none", cfg, 17, 23))


@pytest.mark.gpu
@pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
@pytest.mark.parametrize("mode,cfg,si,sj", CASES,
                         ids=[f"{m}_{c.label}_m{si}n{sj}" for m, c, si, sj in CASES])
def test_trimmed_store_roundtrip(trimmed, tmp_path, mode, cfg, si, sj, use_bf16):
    trimmed(*_asserts(mode, cfg), subtile=True)
    init_rocisa(target="gfx950")
    out, _, ref, r0, r1 = T._run_storeD(cfg, tmp_path, si, sj, init_mode="random", use_bf16=use_bf16)
    if use_bf16:
        T._verify_bf16_random_matrix_positions(out, ref, r0, r1, si, sj)
    else:
        T._verify_random_matrix_positions(out, ref, r0, r1)
    T._verify_oob_sentinel(out, r0, r1, si, sj, label="bf16" if use_bf16 else "f32")
    got = np.asarray(out, dtype=np.float64).reshape(r0, r1, order="F")[:si, :sj]
    expected = ref.astype(np.float64).reshape(r0, r1, order="F")[:si, :sj]
    norm_error = float(np.linalg.norm(got - expected) / np.linalg.norm(expected))
    tolerance = 0.1 if use_bf16 else 0.00001
    assert math.isfinite(norm_error) and norm_error < tolerance, f"norm_error={norm_error}, tolerance={tolerance}"
