# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The opt-in on-disk HSACO cache of the backward run glue, the session
pre-build and the device shard runner.

CPU only. ``attn_bwd_kernels`` runs with stubbed builders, compiler and HIP
runtime (the cache logic, not the kernels, is under test):

* off by default: without ``ROCKE_BWD_HSACO_CACHE`` nothing touches the disk
  and every kernel compiles, as before;
* on: the second call (a new process, simulated) loads every kernel from disk
  and compiles nothing; a change of toolchain, emission switches, source hash
  or spec misses; a corrupt entry is rejected and rebuilt; a swapped builder
  is cached apart under its declared variant and bypasses the cache without
  one; concurrent writer processes never expose a bad entry;
* the pre-build dry run walks past a test's first ``run_case`` and names the
  kernels of every plan, then compiles each distinct spec once (a second
  pre-build compiles nothing); the timing attempts' load-only path never
  compiles;
* the shard filter partitions the items, keeps a parameter set in one shard,
  and the runner runs shards side by side on one shared cache.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from benchmarks.common import attention_bwd_compile as bc
from kernels.common import attention_bwd_cache as ac
from kernels.common import attention_bwd_run as run_mod
from kernels.common.attention_bwd_plan import attn_bwd_plan

from . import _attention_bwd_prebuild as pb
from . import run_device_shards as rds
from ._attention_bwd_harness import run_case
from .sdpa.bwd_cases import get_bwd_case
from .sdpa.bwd_kernel_cases import config_policy, request_from_case

TC = ac.Toolchain("llvm22", "/x/libamd_comgr.so.3", "7.2", "abcdef012345")
ARCH = "gfx942"


def _plan(case_id="bwd_dtype_fp16_d64_mha", **knobs):
    req = request_from_case(get_bwd_case(case_id))
    return attn_bwd_plan(req, ARCH, policy=config_policy("default", ARCH, knobs))


# --------------------------------------------------------------------------- stubs
class _Module:
    def __init__(self, blob):
        self.blob = blob

    def get_function(self, name):
        return (name, self.blob)


class _Runtime:
    loads: list = []

    def load_module(self, blob):
        _Runtime.loads.append(bytes(blob))
        return _Module(bytes(blob))


@pytest.fixture
def device(monkeypatch, tmp_path):
    """Stub builders / compiler / runtime; the fixed toolchain ``TC``."""
    import rocke.helpers.compile as compile_mod
    import rocke.runtime.hip_module as hip_module

    compiles = []
    _Runtime.loads = []

    def builder(stage):
        def build(spec, *, arch):
            return (stage, spec, arch)

        return build

    stock = {s: builder(s) for s in ac.STAGES}
    monkeypatch.setattr(run_mod, "_BUILDERS", dict(stock))
    monkeypatch.setattr(run_mod, "_STOCK_BUILDERS", dict(stock))

    def compile_kernel(kernel, *, arch, capture_ir_text, backend):
        stage, spec, _arch = kernel
        compiles.append((stage, repr(spec)))
        name = spec.kernel_name(stage)
        return types.SimpleNamespace(
            kernel_name=name, hsaco=b"\x7fELF" + f"{stage}|{spec!r}".encode()
        )

    monkeypatch.setattr(compile_mod, "compile_kernel", compile_kernel)
    monkeypatch.setattr(hip_module, "Runtime", _Runtime)
    monkeypatch.setattr(ac, "toolchain_identity", lambda: TC)
    monkeypatch.setattr(ac, "process_source_hash", lambda: "h")
    monkeypatch.setattr(ac, "_ENV_CACHES", {})
    monkeypatch.delenv(ac.HSACO_CACHE_ENV, raising=False)
    return types.SimpleNamespace(compiles=compiles, root=tmp_path / "hsaco")


def _new_process(monkeypatch):
    """A later process: empty in-memory caches (the disk stays)."""
    monkeypatch.setattr(ac, "_ENV_CACHES", {})
    return {}


# --------------------------------------------------------------------------- default off
def test_without_the_variable_nothing_touches_the_disk(device, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("disk cache used without ROCKE_BWD_HSACO_CACHE")

    monkeypatch.setattr(ac.CompileCache, "read", boom)
    monkeypatch.setattr(ac.CompileCache, "write", boom)
    plan = _plan()
    assert ac.env_cache() is None
    got = run_mod.attn_bwd_kernels(plan, cache={})
    assert set(got) == {s.kernel for s in plan.launches}
    assert len(device.compiles) == len({s.kernel for s in plan.launches})
    # the in-memory cache keeps working as before: a second call compiles nothing
    mem: dict = {}
    run_mod.attn_bwd_kernels(plan, cache=mem)
    n = len(device.compiles)
    run_mod.attn_bwd_kernels(plan, cache=mem)
    assert len(device.compiles) == n
    assert not device.root.exists()
    # a blank value is off too
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, "  ")
    assert ac.env_cache() is None


# --------------------------------------------------------------------------- hits / misses
def test_second_process_loads_every_kernel_and_compiles_nothing(device, monkeypatch):
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))
    plan = _plan()
    names = {s.kernel for s in plan.launches}
    first = run_mod.attn_bwd_kernels(plan, cache={})
    assert len(device.compiles) == len(names)
    assert len(list(device.root.rglob("*.hsaco"))) == len(names)
    blobs = {n: first[n][0].hsaco for n in names}

    mem = _new_process(monkeypatch)
    second = run_mod.attn_bwd_kernels(plan, cache=mem)
    assert len(device.compiles) == len(names)  # nothing compiled again
    for n in names:
        art = second[n][0]
        assert isinstance(art, ac.CachedArtifact)
        assert art.hsaco == blobs[n] and art.kernel_name == n
        assert second[n][2] == (n, blobs[n])  # the loaded code object
    stats = ac.env_cache()
    assert stats.hits == len(names) and stats.misses == 0 and stats.rejected == 0
    assert ac.env_cache_stats() == {
        "hits": len(names),
        "misses": 0,
        "rejected": 0,
        "writes": 0,
        "bypassed": 0,
    }


@pytest.mark.parametrize("change", ["flavor", "comgr", "emit", "source", "spec"])
def test_a_key_change_misses(device, monkeypatch, change):
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))
    plan = _plan()
    run_mod.attn_bwd_kernels(plan, cache={})
    n = len(device.compiles)
    other = {
        "flavor": dataclasses.replace(TC, flavor="llvm23"),
        "comgr": dataclasses.replace(TC, comgr_sha="0" * 12),
        "emit": dataclasses.replace(TC, emit="ROCKE_DEBUG_LOC=1"),
    }.get(change)
    if other is not None:
        monkeypatch.setattr(ac, "toolchain_identity", lambda: other)
    elif change == "source":
        monkeypatch.setattr(ac, "process_source_hash", lambda: "g")
    else:
        plan = _plan(edge_tiles=True)
    mem = _new_process(monkeypatch)
    run_mod.attn_bwd_kernels(plan, cache=mem)
    stats = ac.env_cache()
    if change == "spec":  # only the main kernel's spec changed
        assert len(device.compiles) == n + 1 and stats.hits == 2
    else:
        assert len(device.compiles) == 2 * n and stats.hits == 0
    assert stats.misses == len(device.compiles) - n


def test_plain_keys_are_unchanged_and_emission_switches_are_keyed(monkeypatch):
    import hashlib

    s = _plan().specs["main"]
    blob = "\n".join((ac.spec_identity("main", s), ARCH, "llvm22", TC.comgr, "h"))
    want = hashlib.sha256(blob.encode()).hexdigest()[:32]
    assert ac.entry_key("main", s, arch=ARCH, toolchain=TC, source_hash="h") == want
    dbg = dataclasses.replace(TC, emit="ROCKE_DEBUG_LOC=1")
    assert ac.entry_key("main", s, arch=ARCH, toolchain=dbg, source_hash="h") != want
    monkeypatch.delenv("ROCKE_DEBUG_LOC", raising=False)
    assert ac.emission_env() == ""
    monkeypatch.setenv("ROCKE_DEBUG_LOC", "0")
    assert ac.emission_env() == ""
    monkeypatch.setenv("ROCKE_DEBUG_LOC", "1")
    assert ac.emission_env() == "ROCKE_DEBUG_LOC=1"
    assert ac.toolchain_identity().emit == "ROCKE_DEBUG_LOC=1"


@pytest.mark.parametrize("damage", ["hsaco", "json", "meta", "name"])
def test_a_corrupt_entry_is_rejected_and_rebuilt(device, monkeypatch, damage):
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))
    plan = _plan()
    first = run_mod.attn_bwd_kernels(plan, cache={})
    n = len(device.compiles)
    main = plan.launches[1]
    cache = ac.env_cache()
    hp, mp = cache._paths(cache.key("main", plan.specs["main"], ARCH))
    if damage == "hsaco":
        hp.write_bytes(hp.read_bytes()[:-3])  # truncated code object
    elif damage == "json":
        mp.write_text("{not json", encoding="utf-8")
    elif damage == "meta":
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["source_hash"] = "other"
        mp.write_text(json.dumps(meta), encoding="utf-8")
    else:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        del meta["kernel_name"]
        mp.write_text(json.dumps(meta), encoding="utf-8")
    mem = _new_process(monkeypatch)
    got = run_mod.attn_bwd_kernels(plan, cache=mem)
    stats = ac.env_cache()
    assert stats.rejected == 1 and stats.hits == n - 1
    assert len(device.compiles) == n + 1  # only the damaged kernel rebuilt
    assert got[main.kernel][0].hsaco == first[main.kernel][0].hsaco
    again = ac.CompileCache(device.root, toolchain=TC, source_hash="h")
    entry = again.read("main", plan.specs["main"], ARCH)  # rewritten whole
    assert entry is not None and entry.hsaco == first[main.kernel][0].hsaco


def test_a_cached_entry_of_another_kernel_is_refused(device, monkeypatch):
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))
    plan = _plan()
    c = ac.CompileCache(device.root, toolchain=TC, source_hash="h")
    c.write("main", plan.specs["main"], ARCH, kernel_name="other", hsaco=b"\x7fELFx")
    with pytest.raises(AssertionError, match="other"):
        run_mod.attn_bwd_kernels(plan, cache={})


def test_a_swapped_builder_bypasses_the_cache(device, monkeypatch):
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))
    plan = _plan()
    run_mod.attn_bwd_kernels(plan, cache={})
    n = len(device.compiles)
    stock = run_mod._BUILDERS["main"]

    def switched(spec, *, arch):
        return stock(spec, arch=arch)

    monkeypatch.setitem(run_mod._BUILDERS, "main", switched)
    mem = _new_process(monkeypatch)
    run_mod.attn_bwd_kernels(plan, cache=mem)
    stats = ac.env_cache()
    assert len(device.compiles) == n + 1  # main rebuilt, not read from disk
    assert stats.hits == n - 1 and stats.writes == 0  # and not written back


def test_a_declared_builder_variant_is_cached_apart(device, monkeypatch):
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))
    plan = _plan()
    run_mod.attn_bwd_kernels(plan, cache={})
    n = len(device.compiles)
    stock = run_mod._BUILDERS["main"]

    def switched(spec, *, arch):
        return stock(spec, arch=arch)

    switched.cache_variant = "_emit_main:sentinel=False"
    with monkeypatch.context() as m:
        m.setitem(run_mod._BUILDERS, "main", switched)
        run_mod.attn_bwd_kernels(plan, cache=_new_process(monkeypatch))
        assert len(device.compiles) == n + 1  # its own entry, not the stock one
        run_mod.attn_bwd_kernels(plan, cache=_new_process(monkeypatch))
        assert len(device.compiles) == n + 1  # then loaded
        assert ac.env_cache().hits == n
    # the shipped kernel's entry is intact beside it
    run_mod.attn_bwd_kernels(plan, cache=_new_process(monkeypatch))
    stats = ac.env_cache()
    assert len(device.compiles) == n + 1 and stats.hits == n and not stats.rejected
    c = ac.CompileCache(device.root, toolchain=TC, source_hash="h")
    spec = plan.specs["main"]
    assert c.key("main", spec, ARCH) != c.key(
        "main", spec, ARCH, switched.cache_variant
    )


def _patch_emitter(monkeypatch, where):
    """A negative control's patch: a test-defined function in place of a
    block-GEMM method (``method``) or a prefetch-ring function (``function``),
    or a mock (``mock``)."""
    from unittest import mock

    from kernels.common import _attention_bwd_gemm as gemm
    from kernels.common import _attention_bwd_ring as ring

    if where == "method":
        real = gemm.BlockGemm.load_k_slice

        def broken(self, b, a, bop, s, k_slice, **kw):
            return real(self, b, a, bop, 0, k_slice, **kw)

        monkeypatch.setattr(gemm.BlockGemm, "load_k_slice", broken)
        return "kernels.common._attention_bwd_gemm.BlockGemm.load_k_slice"
    if where == "function":
        real_counts = ring.dma_wait_counts

        def no_top_wait(depth, n_qd, n_a):
            return real_counts(depth, n_qd, n_a)

        monkeypatch.setattr(ring, "dma_wait_counts", no_top_wait)
    else:
        monkeypatch.setattr(ring, "dma_wait_counts", mock.MagicMock())
    return "kernels.common._attention_bwd_ring.dma_wait_counts"


def test_the_shipped_emission_code_is_not_patched(device):
    """The stock emission modules (and this file's stubs of the compiler,
    runtime and toolchain, which are not emission) report no patch."""
    import kernels.common._attention_bwd_gemm  # noqa: F401
    import kernels.common._attention_bwd_ring  # noqa: F401
    import kernels.common.attention_bwd_dq  # noqa: F401
    import kernels.common.attention_bwd_unified  # noqa: F401

    assert ac.patched_emission() == []


@pytest.mark.parametrize("where", ["method", "function", "mock"])
def test_a_patched_emitter_neither_reads_nor_writes_the_disk(
    device, monkeypatch, where
):
    """A warm cache is not read while an emitter is patched (the mutant is
    built, not the shipped kernel loaded), nothing is written (no mutant under
    the shipped key), and the shipped entries serve again once it is gone."""
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))
    plan = _plan()
    names = {s.kernel for s in plan.launches}
    run_mod.attn_bwd_kernels(plan, cache={})
    entries = sorted(p.read_bytes() for p in device.root.rglob("*.hsaco"))
    n = len(device.compiles)
    with monkeypatch.context() as m:
        name = _patch_emitter(m, where)
        assert ac.patched_emission() == [name]
        run_mod.attn_bwd_kernels(plan, cache=_new_process(monkeypatch))
        stats = ac.env_cache()
        assert len(device.compiles) == n + len(names)  # every kernel built
        assert (stats.hits, stats.misses, stats.writes) == (0, 0, 0)
        assert stats.bypassed == len(names)
        assert ac.env_cache_stats()["bypassed"] == len(names)
    assert sorted(p.read_bytes() for p in device.root.rglob("*.hsaco")) == entries
    assert ac.patched_emission() == []
    run_mod.attn_bwd_kernels(plan, cache=_new_process(monkeypatch))
    assert len(device.compiles) == n + len(names) and ac.env_cache().hits == len(names)


def test_a_patched_emitter_never_fills_a_cold_cache(device, monkeypatch):
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))
    plan = _plan()
    _patch_emitter(monkeypatch, "method")
    run_mod.attn_bwd_kernels(plan, cache={})
    assert not list(device.root.rglob("*.hsaco"))
    assert ac.env_cache().writes == 0


def test_foreign_code_is_told_from_installed_and_hashed_code():
    import json as stdlib_json

    from kernels.common import _attention_bwd_gemm as gemm

    assert not ac._foreign(gemm.BlockGemm.load_k_slice)  # hashed source
    assert not ac._foreign(stdlib_json.dumps)  # the Python installation
    assert ac._foreign(_patch_emitter)  # this test file (rocke tree, unhashed)
    assert ac._foreign(staticmethod(_patch_emitter))
    assert ac._foreign(property(_patch_emitter))
    assert ac._foreign(__import__("functools").partial(gemm.BlockGemm.load_k_slice))
    generated = eval("lambda: 0")  # noqa: S307 - code without a source file
    assert not ac._foreign(generated)  # generated code is not foreign


def test_main_built_with_declares_its_variant():
    from ._attention_bwd_harness import main_built_with

    with main_built_with(sentinel=False):
        assert run_mod._BUILDERS["main"].cache_variant == "_emit_main:sentinel=False"
    assert run_mod._BUILDERS["main"] is run_mod._STOCK_BUILDERS["main"]


def test_a_read_only_cache_costs_a_rebuild_not_an_error(device, monkeypatch):
    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(device.root))

    def denied(*a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr(ac.CompileCache, "write", denied)
    plan = _plan()
    got = run_mod.attn_bwd_kernels(plan, cache={})
    assert set(got) == {s.kernel for s in plan.launches}


_WRITER = textwrap.dedent(
    """
    import sys
    from kernels.common import attention_bwd_cache as ac

    root, wid, rounds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tc = ac.Toolchain("llvm22", "/x/libamd_comgr.so.3", "7.2", "abcdef012345")
    c = ac.CompileCache(root, toolchain=tc, source_hash="h")
    blob = {k: (b"\\x7fELF" + bytes([k]) * (1 << 18)) for k in range(4)}
    for r in range(rounds):
        k = (r + wid) % 4
        c.write("prep", ("spec", k), "gfx942", kernel_name=f"k{k}", hsaco=blob[k])
        for j in range(4):
            e = c.read("prep", ("spec", j), "gfx942")
            if e is not None and (e.hsaco != blob[j] or e.kernel_name != f"k{j}"):
                print("BAD", j)
                sys.exit(1)
    if c.rejected:  # a partial file was visible
        print("REJECTED", c.rejected)
        sys.exit(2)
    """
)


def test_concurrent_writer_processes_never_expose_a_bad_entry(tmp_path):
    """Four processes write and read the same four keys at once: every read
    is either a miss or the exact entry (never a partial file, which the
    digest check would reject); nothing temporary is left."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(tmp_path), str(w), "60"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for w in range(4)
    ]
    outs = [(p.wait(timeout=300), p.stdout.read().decode()) for p in procs]
    assert all(rc == 0 for rc, _ in outs), outs
    c = ac.CompileCache(tmp_path, toolchain=TC, source_hash="h")
    for k in range(4):
        e = c.read("prep", ("spec", k), "gfx942")
        assert e is not None and e.kernel_name == f"k{k}"
    assert not list(tmp_path.rglob("*.tmp"))


def test_the_identity_never_imports_torch():
    code = (
        "import sys; from kernels.common.attention_bwd_cache import "
        "toolchain_identity as t; t(); print('torch' in sys.modules)"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


# --------------------------------------------------------------------------- load only
def test_timing_attempts_only_load(tmp_path, monkeypatch):
    calls = []

    def compile_one(stage, spec, arch):
        calls.append(stage)
        return spec.kernel_name(stage), b"\x7fELF" + stage.encode(), 0.0

    monkeypatch.setattr(bc, "compile_one", compile_one)
    c = bc.CompileCache(tmp_path, toolchain=TC, source_hash="h")
    plan = _plan()
    with pytest.raises(RuntimeError, match=bc.CACHE_MISS):
        bc.plan_entries(plan, c, compile_missing=False)
    assert not calls and not list(tmp_path.rglob("*.hsaco"))
    built = bc.plan_entries(plan, c)  # the reference process compiles
    n = len(calls)
    loaded = bc.plan_entries(plan, c, compile_missing=False)  # an attempt
    assert len(calls) == n and loaded.keys() == built.keys()
    assert all(loaded[k].hsaco == built[k].hsaco for k in built)


def test_the_bench_takes_the_shared_cache_variable(monkeypatch, tmp_path):
    from benchmarks.common import attention_bwd_bench as bench

    monkeypatch.setenv(ac.HSACO_CACHE_ENV, str(tmp_path))
    assert bench.make_parser().parse_args([]).compile_cache == str(tmp_path)
    monkeypatch.delenv(ac.HSACO_CACHE_ENV)
    assert bench.make_parser().parse_args([]).compile_cache is None


def test_pool_workers_take_the_callers_comgr_digests(monkeypatch):
    seen = {}

    class _Pool:
        def __init__(self, *a, **k):
            seen.update(k)

    monkeypatch.setattr(bc.cf, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(ac, "_FILE_SHA", {("lib", 1, 2): "abc"})
    bc._process_pool(2)
    assert seen["initializer"] is bc._worker_init
    assert seen["initargs"] == ({("lib", 1, 2): "abc"},)


def test_seeded_comgr_digest_is_keyed_by_the_file_version(monkeypatch, tmp_path):
    lib = tmp_path / "libamd_comgr.so"
    lib.write_bytes(b"one")
    st = lib.stat()
    monkeypatch.setattr(ac, "_FILE_SHA", {})
    ac.seed_comgr_digests({(str(lib), st.st_size, st.st_mtime_ns): "seeded"})
    assert ac._comgr_sha(str(lib)) == "seeded"  # no read of the file
    lib.write_bytes(b"other content")  # another size: digested again
    assert ac._comgr_sha(str(lib)) == ac._sha256_file(lib, 12) != "seeded"
    ac.seed_comgr_digests({"bad": "x", ("a", 1, 2): 3})  # malformed: ignored
    assert "bad" not in ac._FILE_SHA and ("a", 1, 2) not in ac._FILE_SHA


# --------------------------------------------------------------------------- pre-build
def _probe_two_runs(case_id, config):
    """A test body for the dry run: two plans of one case (the second run
    happens only if the first returns)."""
    case = get_bwd_case(case_id)
    a = run_case(case, arch=ARCH, config=config)
    b = run_case(case, arch=ARCH, config=config, overrides={"edge_tiles": True})
    assert a.plan.specs["main"] != b.plan.specs["main"]
    assert not a.integrity and a.dq.shape == a.ref.dq.shape


def _probe_compiles(case_id, config):
    from rocke.helpers.compile import compile_kernel

    try:
        compile_kernel(None, arch=ARCH)  # refused in the dry run
    except Exception as ex:  # noqa: BLE001
        if isinstance(ex, pb._Planned):
            raise
    # reaching here: the dry run let a compile through (counted "completed")


def _item(fn, params, *, skip=False, cls=None):
    mark = types.SimpleNamespace(args=(True,), kwargs={})
    return types.SimpleNamespace(
        module=sys.modules[__name__],
        function=fn,
        cls=cls,
        callspec=types.SimpleNamespace(
            params=params, id="-".join(map(str, params.values()))
        ),
        nodeid=f"{__name__}::{fn.__name__}",
        get_closest_marker=lambda name: None,
        iter_markers=lambda name: [mark] if skip and name == "skipif" else [],
    )


_CASE = "bwd_dtype_fp16_d64_mha"


def test_dry_run_tasks_take_plain_parametrized_harness_tests():
    params = {"case_id": _CASE, "config": "default"}
    items = [
        _item(_probe_two_runs, params),
        _item(_probe_two_runs, params, skip=True),
        _item(_probe_two_runs, params, cls=object),
        _item(_probe_two_runs, {"case_id": _CASE}),  # needs a fixture
        types.SimpleNamespace(module=json, function=None, cls=None),
    ]
    tasks, left = pb.dry_run_tasks(items)
    assert left == 4 and len(tasks) == 1
    assert tasks[0].function == "_probe_two_runs" and tasks[0].params == params


def test_the_dry_run_names_every_plan_and_never_compiles():
    params = {"case_id": _CASE, "config": "default"}
    tasks, _ = pb.dry_run_tasks(
        [_item(_probe_two_runs, params), _item(_probe_compiles, params)]
    )
    out = pb._dry_run_chunk(tasks)
    assert out["counts"] == {"completed": 1, "stopped": 1}
    main = [s for st, s, a in out["specs"] if st == "main"]
    assert len(main) == 2 and main[0] != main[1]  # past the first run_case
    want = {(s.stage, repr(_plan().specs[s.stage])) for s in _plan().launches}
    assert want <= {(st, repr(s)) for st, s, a in out["specs"]}
    assert {a for *_x, a in out["specs"]} == {ARCH}
    # the patches are gone after the dry run
    assert run_case.__name__ == "run_case" and run_mod.attn_bwd_kernels.__module__ == (
        run_mod.__name__
    )


def test_the_dry_run_pool_imports_the_test_module_by_name():
    params = {"case_id": _CASE, "config": "default"}
    tasks, _ = pb.dry_run_tasks([_item(_probe_two_runs, params)])
    specs, counts = pb.collect_specs(tasks, workers=1)
    assert counts == {"completed": 1}
    assert len({repr(s) for st, s, a in specs if st == "main"}) == 2


def test_prebuild_compiles_each_spec_once(tmp_path, monkeypatch):
    plan_a, plan_b = _plan(), _plan(edge_tiles=True)
    found = [
        (s.stage, p.specs[s.stage], p.arch)
        for p in (plan_a, plan_b, plan_a)
        for s in p.launches
    ]
    dry_runs = []

    def collect_specs(tasks, workers):
        dry_runs.append(len(tasks))
        return found, collections.Counter()

    monkeypatch.setattr(pb, "collect_specs", collect_specs)
    calls = []

    def compile_one(stage, spec, arch):
        calls.append((stage, repr(spec)))
        return spec.kernel_name(stage), b"\x7fELF" + repr(spec).encode(), 0.0

    monkeypatch.setattr(bc, "compile_one", compile_one)
    monkeypatch.setattr(ac, "toolchain_identity", lambda: TC)
    distinct = {(st, repr(s)) for st, s, a in found}
    counts = pb.prebuild([], cache_root=tmp_path, workers=2, mode="serial")
    assert sorted(calls) == sorted(distinct)  # each spec once
    assert counts["kernels"] == counts["compiled"] == len(distinct)
    assert dry_runs == [0]
    # the same selection again: the manifest skips the dry run and compiles
    again = pb.prebuild([], cache_root=tmp_path, workers=2, mode="serial")
    assert dry_runs == [0] and again["manifest"] == 1
    assert len(calls) == len(distinct) and again["cached"] == len(distinct)
    assert not (tmp_path / pb.LOCK_NAME).exists()
    line = pb.format_counts(again, TC)
    assert (
        line.startswith("bwd-prebuild: items=0 ") and f"cached={len(distinct)}" in line
    )
    # an entry gone from the cache: dry run again, only that kernel compiles
    gone = next(tmp_path.rglob("*.hsaco"))
    gone.unlink()
    third = pb.prebuild([], cache_root=tmp_path, workers=2, mode="serial")
    assert dry_runs == [0, 0] and "manifest" not in third
    assert len(calls) == len(distinct) + 1 and third["compiled"] == 1
    # another selection environment: another manifest (dry run again, all cached)
    monkeypatch.setenv("ROCKE_BWD_TIER", "full")
    fourth = pb.prebuild([], cache_root=tmp_path, workers=2, mode="serial")
    assert dry_runs == [0, 0, 0] and fourth["cached"] == len(distinct)
    assert len(calls) == len(distinct) + 1


def test_the_prebuild_lock_serialises_and_survives_a_dead_holder(tmp_path):
    with pb.prebuild_lock(tmp_path, wait_s=5) as held:
        assert held and (tmp_path / pb.LOCK_NAME).exists()
        assert not pb._acquire(tmp_path / pb.LOCK_NAME, 0.0, 3600)  # busy: gives up
    assert not (tmp_path / pb.LOCK_NAME).exists()
    stale = tmp_path / pb.LOCK_NAME
    stale.write_text("1\n")
    old = stale.stat().st_mtime - 3600
    os.utime(stale, (old, old))
    with pb.prebuild_lock(tmp_path, wait_s=60) as held:
        assert held  # a lock older than the wait is a dead holder's


def test_prebuild_needs_the_cache_variable(monkeypatch):
    monkeypatch.setenv(pb.PREBUILD_ENV, "1")
    monkeypatch.delenv(ac.HSACO_CACHE_ENV, raising=False)
    with pytest.raises(pytest.UsageError):
        pb.prebuild_session(types.SimpleNamespace(items=[]))
    monkeypatch.setenv(pb.PREBUILD_ENV, "0")
    assert pb.prebuild_session(types.SimpleNamespace(items=[])) is None


# --------------------------------------------------------------------------- shards
def test_shard_parse_and_partition(monkeypatch):
    assert rds.parse_shard("3/8") == (frozenset({3}), 8)
    assert rds.parse_shard("0,2/4") == (frozenset({0, 2}), 4)
    for bad in ("3", "8/8", "-1/4", "/4", "1/0"):
        with pytest.raises(ValueError):
            rds.parse_shard(bad)

    def item(fn, pid):
        cs = types.SimpleNamespace(id=pid) if pid else None
        nodeid = f"tests/test_x.py::{fn}" + (f"[{pid}]" if pid else "")
        return types.SimpleNamespace(nodeid=nodeid, callspec=cs)

    items = [item(f, f"c{i}-debug") for i in range(40) for f in ("t_a", "t_b")]
    items += [item(f"t_plain{i}", None) for i in range(10)]
    dropped = []
    config = types.SimpleNamespace(
        hook=types.SimpleNamespace(
            pytest_deselected=lambda items: dropped.extend(items)
        )
    )
    n, seen = 3, []
    for i in range(n):
        monkeypatch.setenv(rds.SHARD_ENV, f"{i}/{n}")
        sel = list(items)
        rds.select_items(config, sel)
        ids = {x.nodeid for x in sel}
        seen.append(ids)
        for x in sel:  # both tests of one parameter set land in one shard
            if "::t_a[" in x.nodeid:
                assert x.nodeid.replace("::t_a[", "::t_b[") in ids
    assert sum(len(s) for s in seen) == len(items)
    assert set().union(*seen) == {x.nodeid for x in items}
    assert all(seen) and len(dropped) == (n - 1) * len(items)
    monkeypatch.setenv(rds.SHARD_ENV, "0,1,2/3")  # a job's shards together
    sel = list(items)
    rds.select_items(config, sel)
    assert len(sel) == len(items)
    monkeypatch.delenv(rds.SHARD_ENV)
    sel = list(items)
    rds.select_items(config, sel)
    assert len(sel) == len(items)


def test_devices_and_shard_env():
    assert rds.resolve_devices("3", {}) == ["0", "1", "2"]
    assert rds.resolve_devices("2,5", {}) == ["2", "5"]
    assert rds.resolve_devices("auto", {"HIP_VISIBLE_DEVICES": "4,6"}) == ["4", "6"]
    assert rds.resolve_devices("auto", {"ROCR_VISIBLE_DEVICES": "GPU-a,GPU-b"}) == [
        "0",
        "1",
    ]
    assert rds.resolve_devices("auto", {}) == ["0"]
    base = {rds.SHARD_ENV: "x", rds.BWD_SHARD_ENV: "y", rds.PREBUILD_ENV: "1"}
    env = rds.build_env(
        base, cache="c", shard_env=rds.SHARD_ENV, shard="1/4", device="2",
        prebuild=False,
    )  # fmt: skip
    assert env[rds.SHARD_ENV] == "1/4" and rds.BWD_SHARD_ENV not in env
    assert env["HIP_VISIBLE_DEVICES"] == "2" and env[rds.CACHE_ENV] == "c"
    assert rds.PREBUILD_ENV not in env
    args = rds.make_parser().parse_args(["--shards", "4", "--first-shard", "4"])
    assert rds.shard_plan(args, ["0"]) == [(4, "0"), (5, "0"), (6, "0"), (7, "0")]


def test_children_share_bytecode_when_the_job_forbids_writing_it():
    """Under PYTHONDONTWRITEBYTECODE the children get one bytecode directory
    outside the tree (not a recompile of every source per process); an
    explicit prefix, an unset variable or --no-pycache keep the environment."""
    no_pyc = {"PYTHONDONTWRITEBYTECODE": "1"}
    d = rds.shared_pycache(no_pyc)
    try:
        assert d is not None and Path(d).is_dir()
        env = rds.build_env(
            no_pyc, cache="c", shard_env=rds.SHARD_ENV, shard="", device=None,
            prebuild=True, pycache=d,
        )  # fmt: skip
        assert env["PYTHONPYCACHEPREFIX"] == d
        assert "PYTHONDONTWRITEBYTECODE" not in env
    finally:
        Path(d).rmdir()
    assert rds.shared_pycache(no_pyc, disabled=True) is None
    assert rds.shared_pycache({}) is None
    assert rds.shared_pycache({**no_pyc, "PYTHONPYCACHEPREFIX": "p"}) is None
    env = rds.build_env(
        no_pyc, cache="c", shard_env=rds.SHARD_ENV, shard="", device=None,
        prebuild=False,
    )  # fmt: skip
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and "PYTHONPYCACHEPREFIX" not in env


def test_the_runner_runs_shards_side_by_side(tmp_path, capsys):
    """Two shards of the layering test on one device: every item runs once."""
    target = str(Path(__file__).with_name("test_library_layering.py"))
    rc = rds.main(
        ["--devices", "1", "--shards", "2", "--no-prebuild",
         "--cache", str(tmp_path / "c"), "--log-dir", str(tmp_path / "logs"),
         "--", target, "-p", "no:cacheprovider", "-q"]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0, out
    lines = [ln for ln in out.splitlines() if ln.startswith("shard ")]
    assert len(lines) == 2
    import re

    passed = sum(int(m) for m in re.findall(r"(\d+) passed", out))
    assert passed == 4  # the four layers, split across the two shards
    assert (tmp_path / "logs" / "shard-0-of-2.log").exists()


def test_every_launch_stage_is_a_cache_stage():
    """The run glue launches main, prep, convert, the split-dQ kernel and the
    dense main kernel: each must have a cache key (an unknown stage raises in
    ``spec_identity``, which would fail every cached launch of it)."""
    assert set(run_mod._STOCK_BUILDERS) <= set(ac.STAGES)
