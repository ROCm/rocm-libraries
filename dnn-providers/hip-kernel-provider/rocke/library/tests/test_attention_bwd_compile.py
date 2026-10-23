# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the backward batch compile and on-disk compile cache.

CPU only. The cache and batch logic run with a stubbed compiler (and one
process-pool batch with a worker that dies); one test cross-compiles a real
backward main kernel through comgr (skipped when the local comgr cannot
compile for the target).
"""

from __future__ import annotations

import hashlib
import time

import pytest

from benchmarks.common import attention_bwd_compile as bc
from kernels.common.attention_bwd import AttnBwdSpec, validate_attn_bwd_spec

TC = bc.Toolchain("llvm22", "/x/libamd_comgr.so.3", "7.2", "abcdef012345")


def _spec(**kw):
    base = dict(head_size=64, dtype="bf16")
    base.update(kw)
    return validate_attn_bwd_spec(AttnBwdSpec(**base), "gfx942")


# --------------------------------------------------------------------------- source hash
def test_source_files_cover_the_lowering_and_the_kernels():
    rel = {p.relative_to(bc._ROOT).as_posix() for p in bc.source_files()}
    for must in (
        "platform/python/rocke/core/lower_llvm.py",
        "platform/python/rocke/helpers/compile.py",
        "platform/python/rocke/runtime/comgr.py",
        "library/kernels/common/attention_bwd.py",
        "library/kernels/common/_attention_bwd_body.py",
        "library/kernels/common/_attention_bwd_lds.py",
        "library/kernels/common/attention_bwd_aux.py",
    ):
        assert must in rel, must
    assert not any("__pycache__" in r for r in rel)


def test_source_hash_is_line_ending_neutral_and_content_sensitive(tmp_path):
    def tree(root, text):
        for sub in ("platform/python/rocke/core", "library/kernels/common"):
            d = root / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / "m.py").write_bytes(text)

    tree(tmp_path / "a", b"x = 1\ny = 2\n")
    tree(tmp_path / "b", b"x = 1\r\ny = 2\r\n")
    tree(tmp_path / "c", b"x = 1\ny = 3\n")
    ha, hb, hc = (bc.backward_source_hash(tmp_path / n) for n in "abc")
    assert ha == hb != hc
    (tmp_path / "a/library/kernels/common/extra.py").write_bytes(b"")
    assert bc.backward_source_hash(tmp_path / "a") != ha  # a new file counts
    assert len(bc.backward_source_hash()) == 16


# --------------------------------------------------------------------------- keys
def test_entry_key_covers_spec_arch_flavor_comgr_and_sources():
    s = _spec()
    k = bc.entry_key("main", s, arch="gfx942", toolchain=TC, source_hash="h")
    other = [
        bc.entry_key("main", _spec(edge_tiles=True), arch="gfx942", toolchain=TC,
                     source_hash="h"),
        bc.entry_key("main", s, arch="gfx950", toolchain=TC, source_hash="h"),
        bc.entry_key("main", s, arch="gfx942",
                     toolchain=bc.Toolchain("llvm20", TC.comgr_path, TC.comgr_rocm,
                                            TC.comgr_sha), source_hash="h"),
        bc.entry_key("main", s, arch="gfx942",
                     toolchain=bc.Toolchain("llvm22", TC.comgr_path, "7.1",
                                            TC.comgr_sha), source_hash="h"),
        bc.entry_key("main", s, arch="gfx942",
                     toolchain=bc.Toolchain("llvm22", TC.comgr_path, TC.comgr_rocm,
                                            "000000000000"), source_hash="h"),
        bc.entry_key("main", s, arch="gfx942", toolchain=TC, source_hash="g"),
    ]  # fmt: skip
    assert len({k, *other}) == 1 + len(other)
    with pytest.raises(ValueError):
        # not a stage ("dq" is: split dQ; "reduce" is: the dK / dV slice sink)
        bc.spec_identity("page_zero", s)


# --------------------------------------------------------------------------- cache
def test_cache_roundtrip_and_rejects_foreign_or_corrupt_entries(tmp_path):
    c = bc.CompileCache(tmp_path, toolchain=TC, source_hash="h")
    s = _spec()
    assert c.read("main", s, "gfx942") is None and c.misses == 1
    e = c.write("main", s, "gfx942", kernel_name="k", hsaco=b"\x7fELFabc")
    got = c.read("main", s, "gfx942")
    assert got is not None and got.hsaco == b"\x7fELFabc" and got.kernel_name == "k"
    assert got.hsaco_sha == hashlib.sha256(b"\x7fELFabc").hexdigest()[:16]
    # another comgr or flavor never sees the entry (different key) ...
    other = bc.CompileCache(
        tmp_path,
        toolchain=bc.Toolchain("llvm22", TC.comgr_path, "7.1", TC.comgr_sha),
        source_hash="h",
    )
    assert other.read("main", s, "gfx942") is None
    # ... and an entry whose metadata or bytes do not match is rejected
    hp, mp = c._paths(e.key)
    hp.write_bytes(b"\x7fELFabd")
    assert c.read("main", s, "gfx942") is None and c.rejected == 1
    hp.write_bytes(b"\x7fELFabc")
    mp.write_text(mp.read_text().replace('"llvm22"', '"llvm20"'))
    assert c.read("main", s, "gfx942") is None and c.rejected == 2
    assert not list(tmp_path.rglob("*.tmp"))


# --------------------------------------------------------------------------- batch
@pytest.fixture
def fake_compile(monkeypatch):
    calls = []

    def compile_one(stage, spec, arch):
        calls.append((stage, spec))
        if spec.edge_tiles:
            raise NotImplementedError("not built: edge")
        if spec.ws_layout == "token_major":
            raise RuntimeError("boom")
        return f"k_{spec.kernel_name()}", b"\x7fELF" + repr(spec).encode(), 0.01

    monkeypatch.setattr(bc, "compile_one", compile_one)
    return calls


@pytest.mark.parametrize("mode", ["serial", "thread"])
def test_batch_tokens_dedupe_and_cache_hits(tmp_path, fake_compile, mode):
    c = bc.CompileCache(tmp_path, toolchain=TC, source_hash="h")
    a, b = _spec(), _spec(edge_tiles=True)
    t = _spec(ws_layout="token_major")
    items = [("main", a), ("main", b), ("main", a), ("main", t)]
    res = bc.compile_batch(items, arch="gfx942", cache=c, workers=4, mode=mode)
    assert [r.ok for r in res] == [True, False, True, False]
    assert res[1].error == "not_built" and res[3].error == "compile:RuntimeError"
    assert res[2].cached and res[2].entry.key == res[0].entry.key
    assert len(fake_compile) == 3  # the duplicate compiled once
    again = bc.compile_batch(items[:1], arch="gfx942", cache=c, workers=4, mode=mode)
    assert again[0].cached and len(fake_compile) == 3


def test_batch_deadline_and_toolchain_mismatch(tmp_path, fake_compile, monkeypatch):
    c = bc.CompileCache(tmp_path, toolchain=TC, source_hash="h")
    res = bc.compile_batch(
        [("main", _spec())], arch="gfx942", cache=c, mode="serial",
        deadline=time.time() - 1,
    )  # fmt: skip
    assert res[0].error == "deadline" and not fake_compile
    # a pool worker reports its own toolchain; a result from another comgr
    # than the cache's is rejected, not written
    other = bc.Toolchain("llvm22", TC.comgr_path, TC.comgr_rocm, "ffffffffffff")
    monkeypatch.setattr(bc, "_WORKER_TOOLCHAIN", other)
    out = bc._compile_task("main", _spec(scheduler_strategy="max-ilp"), "gfx942")
    assert out["comgr"] == other.comgr

    class _Pool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args):
            import concurrent.futures as cf

            f = cf.Future()
            f.set_result(fn(*args))
            return f

    monkeypatch.setattr(bc.cf, "ProcessPoolExecutor", _Pool)
    res = bc.compile_batch(
        [("main", _spec(scheduler_strategy="max-ilp"))], arch="gfx942", cache=c,
        workers=2, mode="process",
    )  # fmt: skip
    assert res[0].error == "toolchain_mismatch"
    assert c.read("main", _spec(scheduler_strategy="max-ilp"), "gfx942") is None
    with pytest.raises(ValueError):
        bc.compile_batch([], arch="gfx942", cache=c, mode="fork")


# Dense-family knob values registered as not built (dense_bwd_not_built and
# the general-shell body levers, _shell_not_built): the compile path refuses
# them before building or touching a toolchain.
_DENSE_NOT_BUILT = (
    {"body_variant": "trload"},
    {"dkv_mode": "atomic", "body_knobs": (("dkv_sink", "slices"),)},
)


def _dense_not_built(**kw):
    from kernels.gfx950.attention_dense_bwd import Gfx950AttentionDenseBwdSpec

    return Gfx950AttentionDenseBwdSpec(
        seqlen_q=256, seqlen_kv=256, num_query_heads=8, num_kv_heads=2,
        head_size=128, **kw,
    )  # fmt: skip


class _DiesInTheWorker:
    """An item whose unpickling ends the worker process (a stand-in for a
    toolchain crash such as a comgr segfault)."""

    def __reduce__(self):
        import os

        return (os._exit, (3,))

    def __repr__(self):
        return "_DiesInTheWorker()"


def test_a_dying_worker_is_charged_to_its_item_only(tmp_path):
    """Process mode: a worker that dies breaks the pool; the item that kills
    it comes back ``compile_crash`` (re-run alone), every other item of the
    batch still gets its own result."""
    c = bc.CompileCache(tmp_path, toolchain=TC, source_hash="h")
    # not built (refused before any toolchain is touched): the dense family's
    # registered-not-built values on gfx950 (the per-arch transpose-load body,
    # the slice sink of the general shell)
    nb = [_dense_not_built(**kw) for kw in _DENSE_NOT_BUILT]
    # one worker: the items queued behind the crash die with the pool and
    # are re-run alone (each must come back with its own result)
    items = [("main", _DiesInTheWorker()), ("dense", nb[0]), ("dense", nb[1])]
    for workers in (1, 2):
        res = bc.compile_batch(
            items, arch="gfx950", cache=c, workers=workers, mode="process"
        )
        assert [r.error for r in res] == [bc.COMPILE_CRASH, "not_built", "not_built"]
        assert not any(r.ok for r in res)


def test_a_dying_fresh_compile_process_is_a_compile_crash(tmp_path, monkeypatch):
    """A kernel that compiles in a fresh process of its own (iglp_opt(1)) and
    whose process dies comes back ``compile_crash`` in every mode, like a
    dying pool worker; a compile error stays ``compile:ComgrError``."""
    from rocke.runtime.comgr import ComgrError, ComgrProcessDied

    def compile_one(stage, spec, arch):
        raise (ComgrError if spec.ws_layout == "token_major" else ComgrProcessDied)("x")

    monkeypatch.setattr(bc, "compile_one", compile_one)
    items = [("main", _spec()), ("main", _spec(ws_layout="token_major"))]
    for mode in ("serial", "thread"):
        c = bc.CompileCache(tmp_path / mode, toolchain=TC, source_hash="h")
        res = bc.compile_batch(items, arch="gfx950", cache=c, workers=2, mode=mode)
        assert [r.error for r in res] == [bc.COMPILE_CRASH, "compile:ComgrError"]


def test_not_built_is_detected_before_building(monkeypatch):
    # the dense family's registered-not-built values (gfx950)
    for kw in _DENSE_NOT_BUILT:
        d = _dense_not_built(**kw)
        assert bc.not_built_reasons("dense", d)
        with pytest.raises(NotImplementedError):
            bc.compile_one("dense", d, "gfx950")
    # the main kernel's registered-not-built value (the arrival-counter dK / dV
    # sink) is refused before building
    lc = validate_attn_bwd_spec(
        AttnBwdSpec(head_size=64, dkv_mode="atomic", dkv_sink="last_cta"), "gfx942"
    )
    assert bc.not_built_reasons("main", lc) == ["dkv_sink='last_cta'"]
    with pytest.raises(NotImplementedError, match="last_cta"):
        bc.compile_one("main", lc, "gfx942")
    # every other main-kernel value is built (tr_read with the K-permuted
    # 16x16x32 G1 / G3 B operand included): a finding of the emission check
    # is refused before building
    s = validate_attn_bwd_spec(
        AttnBwdSpec(
            head_size=64, dtype="bf16", transpose_source="tr_read",
            atom_g13="16x16x32",
        ),
        "gfx950",
    )  # fmt: skip
    assert bc.not_built_reasons("main", s) == []
    import kernels.common.attention_bwd as ab

    monkeypatch.setattr(ab, "_not_built", lambda spec: ["an unbuilt emission"])
    assert bc.not_built_reasons("main", s) == ["an unbuilt emission"]
    with pytest.raises(NotImplementedError, match="an unbuilt emission"):
        bc.compile_one("main", s, "gfx950")
    assert bc.not_built_reasons("prep", s) == []


# --------------------------------------------------------------------------- real compile
def _can_compile():
    try:
        tc = bc.toolchain_identity()
    except Exception:  # noqa: BLE001
        return False
    return tc.comgr_path is not None


@pytest.mark.skipif(not _can_compile(), reason="no comgr library found")
def test_real_main_kernel_compiles_and_is_cached(tmp_path):
    tc = bc.toolchain_identity()
    c = bc.CompileCache(tmp_path, toolchain=tc, source_hash=bc.backward_source_hash())
    s = _spec()
    res = bc.compile_batch([("main", s)], arch="gfx942", cache=c, mode="serial")
    assert res[0].ok, res[0].detail
    assert res[0].entry.hsaco[:4] == b"\x7fELF"
    assert res[0].entry.kernel_name == s.kernel_name("main")
    again = bc.compile_batch([("main", s)], arch="gfx942", cache=c, mode="serial")
    assert again[0].cached and again[0].entry.hsaco == res[0].entry.hsaco


@pytest.mark.skipif(not _can_compile(), reason="no comgr library found")
def test_thread_pool_compiles_the_serial_code_objects(tmp_path):
    """Threaded in-process compiles give byte-identical code objects."""
    tc = bc.toolchain_identity()
    src = bc.backward_source_hash()
    specs = [
        ("main", _spec()),
        ("main", _spec(edge_tiles=True)),
        ("main", _spec(ws_layout="token_major")),
        ("main", _spec(scheduler_strategy="iterative-minreg")),
        ("main", _spec(waves=4, block_n=64, block_m=32, warp_grid_g4=(2, 2))),
        ("main", _spec(head_size=128)),
    ]
    a = bc.compile_batch(
        specs, arch="gfx942", mode="serial",
        cache=bc.CompileCache(tmp_path / "s", toolchain=tc, source_hash=src),
    )  # fmt: skip
    b = bc.compile_batch(
        specs, arch="gfx942", mode="thread", workers=6,
        cache=bc.CompileCache(tmp_path / "t", toolchain=tc, source_hash=src),
    )  # fmt: skip
    assert all(r.ok for r in a + b), [r.detail for r in a + b if not r.ok]
    assert [r.entry.hsaco for r in a] == [r.entry.hsaco for r in b]
    assert len({r.entry.hsaco_sha for r in a}) == len(specs)


@pytest.mark.skipif(not _can_compile(), reason="no comgr library found")
def test_iglp1_code_objects_do_not_depend_on_what_the_process_compiled_before():
    """LLVM's ``iglp_opt(1)`` strategy keeps counts across the compiles of a
    process: with the production comgr, the probe kernel below compiled in a
    process after the other one gets another code object (SGPR spills) than
    in a process that compiles it alone. The backward compile puts every
    ``iglp_opt(1)`` kernel in a fresh process, so its code object is the
    alone one whatever this process compiled before."""
    import dataclasses

    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.helpers.compile import _comgr_options_for_kernel
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    from kernels.common.attention_bwd import build_attn_bwd_main

    arch, isa = "gfx950", "amdgcn-amd-amdhsa--gfx950"

    def spec(dkv_mode, **kw):
        base = AttnBwdSpec(
            head_size=64, dtype="fp16", dkv_mode=dkv_mode, mask_class="band"
        )
        base = validate_attn_bwd_spec(base, arch)
        kw.update(sched="iglp1", edge_tiles=True)
        return validate_attn_bwd_spec(dataclasses.replace(base, **kw), arch)

    def lowered(s):
        k = build_attn_bwd_main(s, arch=arch)
        return lower_kernel_to_llvm(k, arch=arch), _comgr_options_for_kernel(k)

    probe = spec("atomic", stage_order="g4g1g3")
    ll, opts = lowered(probe)
    alone, _ = build_hsaco_from_llvm_ir(ll, isa=isa, options=opts, fresh_process=True)
    # in-process iglp_opt(1) compiles move the strategy's counts here: another
    # kernel, then the probe itself (the history of the observed failure)
    for s in (spec("direct", head_pack=True, stage_order="g3g4g1"), probe):
        ll, opts = lowered(s)
        build_hsaco_from_llvm_ir(ll, isa=isa, options=opts)
    _name, blob, _secs = bc.compile_one("main", probe, arch)
    assert blob == alone
