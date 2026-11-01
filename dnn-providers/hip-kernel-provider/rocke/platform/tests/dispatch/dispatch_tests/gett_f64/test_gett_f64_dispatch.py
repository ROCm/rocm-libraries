# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection, sweep and build tests for the f64 GETT dispatcher family."""

from __future__ import annotations

import pytest
from rocke.dispatch.families.gett_f64 import (
    GettF64Request,
    dispatch_gett_f64,
    dispatch_gett_f64_all,
    gett_f64_sweep_space,
)
from rocke.instances.common.gett_f64 import (
    TUNED_CONFIGS,
    arch_spec,
    gemm_problem,
    gett_f64_grid,
    gett_f64_shape_class,
    is_valid_spec,
    pipe_spec,
    tuned_spec,
)
from rocke.instances.common.gett_f64_plan import packed_strides, plan_contraction


# CU counts of the parts the shape classes were tuned on.
CUS = {"gfx90a": 104, "gfx942": 304, "gfx950": 256}


def _req(n, arch="gfx950", k=512, cx=False, **kw):
    p = gemm_problem(
        n, n, k, a_strides=(k, 1), b_strides=(k, 1), d_strides=(n, 1), complex_=cx
    )
    kw.setdefault("num_cus", CUS.get(arch, 256))
    return GettF64Request(problem=p, arch=arch, **kw)


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
def test_the_tuned_candidate_comes_first(arch):
    for n, cx in ((8192, False), (1024, False), (4096, True), (1024, True)):
        req = _req(n, arch, cx=cx)
        r = dispatch_gett_f64(req)
        piped = pipe_spec(req.problem, arch, num_cus=req.cus())
        if piped is not None:  # the pipelined candidate goes first where it applies
            assert r.candidate.name == "gett_f64_pipe"
            assert r.spec == piped
            continue
        assert r.candidate.name == "gett_f64_tuned"
        assert r.spec == tuned_spec(req.problem, arch, num_cus=req.cus())
        assert r.block == (r.spec.threads, 1, 1)
        assert r.grid == gett_f64_grid(r.spec)
        assert r.request.dtype == ("c128" if cx else "f64")


@pytest.mark.parametrize(
    "arch,n,k,cx,cls",
    [
        ("gfx950", 1024, 512, False, "small"),  # 64 tiles of 128 < 256 CUs
        ("gfx950", 4096, 512, False, "medium"),
        ("gfx950", 8192, 512, False, "large"),  # >= 8 tiles per CU
        ("gfx950", 4096, 4096, False, "large"),  # long K
        ("gfx942", 1024, 512, False, "small"),
        ("gfx942", 4096, 512, False, "medium"),
        ("gfx942", 4096, 8192, False, "medium"),
        ("gfx90a", 1536, 512, False, "small"),  # 144 tiles < 2 x 104 CUs
        ("gfx90a", 4096, 512, False, "kk"),  # both operands K-contiguous
        ("gfx950", 1024, 512, True, "small"),
        ("gfx950", 4096, 1024, True, "medium"),
        ("gfx950", 4096, 4096, True, "large"),
        ("gfx942", 1024, 512, True, "small"),
        ("gfx942", 4096, 512, True, "medium"),
        ("gfx90a", 1024, 512, True, "medium"),
        ("gfx90a", 4096, 2048, True, "large"),
    ],
)
def test_shape_classes(arch, n, k, cx, cls):
    p = _req(n, arch, k=k, cx=cx).problem
    assert gett_f64_shape_class(p, arch, CUS[arch]) == cls


def test_gfx90a_classes_by_the_operands_read_along_k():
    def cls(a, b):
        p = gemm_problem(
            4096, 4096, 4096, a_strides=a, b_strides=b, d_strides=(1, 4096)
        )
        return gett_f64_shape_class(p, "gfx90a", 104)

    assert cls((4096, 1), (4096, 1)) == "kk"
    assert cls((4096, 1), (1, 4096)) == "kmixed"
    assert cls((1, 4096), (1, 4096)) == "medium"
    # K rows 64 KiB apart (8192-element leading dimension): the pow2 class
    p = gemm_problem(
        8192, 8192, 8192, a_strides=(1, 8192), b_strides=(1, 8192), d_strides=(1, 8192)
    )
    assert gett_f64_shape_class(p, "gfx90a", 104) == "pow2"
    # many tiles and a short K: the wide class
    p = gemm_problem(
        15000,
        15000,
        768,
        a_strides=(1, 15000),
        b_strides=(1, 15000),
        d_strides=(1, 15000),
    )
    assert gett_f64_shape_class(p, "gfx90a", 104) == "wide"
    pc = gemm_problem(
        15000,
        15000,
        768,
        a_strides=(1, 15000),
        b_strides=(1, 15000),
        d_strides=(1, 15000),
        complex_=True,
    )
    assert gett_f64_shape_class(pc, "gfx90a", 104) == "wide"


def test_tuned_configs_are_valid_and_knobs_override():
    for (arch, kind, cls), cfg in TUNED_CONFIGS.items():
        p = gemm_problem(
            4096,
            4096,
            2048,
            a_strides=(1, 4096),
            b_strides=(1, 4096),
            d_strides=(1, 4096),
            complex_=kind == "complex",
        )
        spec = arch_spec(p, arch, **cfg)
        assert is_valid_spec(spec, arch=arch)[0], (arch, kind, cls)
    p = _req(8192).problem
    assert tuned_spec(p, "gfx950", num_cus=256, tile_k=8).tile_k == 8


def test_complex_signature_carries_complex_scalars():
    r = dispatch_gett_f64(_req(4096, cx=True))
    assert [a["name"] for a in r.signature][-4:] == [
        "alpha_re",
        "alpha_im",
        "beta_re",
        "beta_im",
    ]


def test_unsupported_targets_are_refused():
    for arch in ("gfx1151", "gfx1201", "gfx1250"):
        with pytest.raises(ValueError):
            dispatch_gett_f64(_req(4096, arch))


def test_sweep_space_is_valid_and_starts_at_the_default():
    req = _req(8192)
    specs = gett_f64_sweep_space(req)
    assert len(specs) >= 4
    assert dispatch_gett_f64(req).spec in specs
    assert len(set(specs)) == len(specs)
    # the other tuned classes of the arch are in the tuned candidate's space
    kinds = {(s.tile_m, s.tile_n) for s in specs}
    assert {(32, 32), (64, 128), (128, 128)} <= kinds
    assert any(sp.unroll2 for sp in specs)


def test_every_result_builds():
    ext = {"m0": 12, "k0": 6, "m1": 10, "n0": 14, "k1": 8}
    ma, mb, md = ["m0", "k0", "m1", "k1"], ["n0", "k0", "k1"], ["m0", "n0", "m1"]
    plan = plan_contraction(
        ma,
        packed_strides(ma, ext),
        mb,
        packed_strides(mb, ext),
        md,
        packed_strides(md, ext),
        ext,
    )
    req = GettF64Request(problem=plan.launches[0].problem, arch="gfx942", num_cus=304)
    results = dispatch_gett_f64_all(req)
    assert results
    for r in results:
        kernel = r.build()
        assert kernel.name == r.spec.kernel_name()


def test_cu_count_comes_from_the_caller_or_the_device(monkeypatch):
    import rocke.runtime.hip_module as hm
    from rocke.instances.common.gett_f64 import resolve_num_cus

    assert resolve_num_cus("gfx942", 228) == 228
    with pytest.raises(ValueError):
        resolve_num_cus("gfx942", 0)
    monkeypatch.setattr(hm, "get_device_arch", lambda device=0: None)
    with pytest.raises(ValueError, match="num_cus is required"):
        resolve_num_cus("gfx942")
    with pytest.raises(ValueError, match="num_cus is required"):
        GettF64Request(problem=_req(4096).problem, arch="gfx942").cus()
    monkeypatch.setattr(hm, "get_device_arch", lambda device=0: "gfx942")
    monkeypatch.setattr(hm, "get_device_num_cus", lambda device=0: 228)
    assert resolve_num_cus("gfx942") == 228
    assert resolve_num_cus("gfx950", 256) == 256
    with pytest.raises(ValueError):
        resolve_num_cus("gfx950")  # the device is another arch


def test_cu_count_moves_the_shape_class():
    p = _req(2048, "gfx942").problem  # 256 tiles of 128
    assert gett_f64_shape_class(p, "gfx942", 304) == "small"
    assert gett_f64_shape_class(p, "gfx942", 228) == "medium"


def test_runtime_size_requests_share_one_kernel_per_layout():
    a = dispatch_gett_f64(_req(4096, "gfx950", runtime_sizes=True))
    b = dispatch_gett_f64(_req(4000, "gfx950", k=1024, runtime_sizes=True))
    assert a.spec.runtime_sizes and b.spec.runtime_sizes
    assert a.spec == b.spec and a.kernel_id.spec_hash == b.kernel_id.spec_hash
    assert a.grid == gett_f64_grid(a.spec, a.request.problem)
    assert b.grid == gett_f64_grid(b.spec, b.request.problem)
    assert a.grid != b.grid
    names = [x["name"] for x in a.signature]
    assert names[6:10] == ["M", "N", "K", "L"]
    assert a.build().name == b.build().name


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
@pytest.mark.parametrize("cx", [False, True])
def test_long_k_few_ctas_split_k(arch, cx):
    from rocke.instances.common.gett_f64 import (
        SPLIT_CTAS_PER_CU,
        SPLIT_K_MAX,
        SPLIT_MIN_K,
        gett_f64_workspace_bytes,
    )

    def spec(n, k):
        return dispatch_gett_f64(_req(n, arch, k=k, cx=cx)).spec

    s = spec(256, 16384)  # few CTAs, long K
    ctas = -(-256 // s.tile_m) * -(-256 // s.tile_n)
    assert 1 < s.split_k <= SPLIT_K_MAX
    assert s.split_k & (s.split_k - 1) == 0  # a power of two
    assert 16384 // s.split_k >= SPLIT_MIN_K
    assert ctas * s.split_k >= min(SPLIT_CTAS_PER_CU * CUS[arch], ctas * SPLIT_K_MAX)
    assert not s.unroll2 and not s.k_stagger
    assert gett_f64_workspace_bytes(s) > 0
    r = dispatch_gett_f64(_req(256, arch, k=16384, cx=cx))
    assert r.grid[2] == s.split_k
    assert spec(256, 512).split_k == 1  # short K: one CTA per tile
    assert spec(8192, 16384).split_k == 1  # many CTAs already
    p = _req(256, arch, k=16384, cx=cx).problem
    assert tuned_spec(p, arch, num_cus=CUS[arch], split_k=1).split_k == 1


def test_k_blocks_inside_a_long_split():
    from rocke.instances.common.gett_f64 import K_BLOCK_ELEMS

    p = _req(256, "gfx950", k=262144).problem  # K / split stays long
    s = tuned_spec(p, "gfx950", num_cus=256)
    assert s.split_k > 1 and s.k_block == max(1, K_BLOCK_ELEMS // s.tile_k)
