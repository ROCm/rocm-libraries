"""Selection, worker-knob, and failure-reporting tests for the parallel prewarm.

The corpus below has a double duty. It backs the in-process golden-sequence
assertion pytest runs against a `tmp_path`, and it backs the out-of-process
staged-tree capture a plain script runs against a pristine checkout of the base
commit. That second consumer is why `_write_corpus` is a standalone
standard-library-only function rather than a fixture body: the capture script
copies it verbatim into a tree that has never seen this file.
"""

import ast
import concurrent.futures
import inspect
import itertools
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest

from hkp_pack import pipeline
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from hkp_pack.hip_compile import hip_source_relpath, hip_variant_key

# The one arch the corpus is authored for. Every consumer references this
# constant instead of restating the literal: a capture script that ran a
# different arch would take the copy-through branch for every KDP, compile
# nothing, and report two trees identical over nothing at all.
TARGET_ARCH = "gfx942"

# The arch the corpus uses to express exclusion. Never packed for.
OTHER_ARCH = "gfx90a"

# The define every corpus source reads, so distinct build blocks produce
# distinct variant keys and genuinely distinct code objects.
_BLOCK_DEFINE = "HKP_PARALLEL_BLOCK"

_ROCKE_STUB_PKG = "hkp_parallel_stub"
_ROCKE_STUB_SOURCE = f"{_ROCKE_STUB_PKG}/kernels/attention.py"
_ROCKE_STUB_BUILDER = "build_attention"
_ROCKE_STUB_SPEC = {"tile": 64}

_K1_SOURCE = "k1.cpp"
_K2_SOURCE = "k2.cpp"

_HIP_SOURCE_TEMPLATE = """\
#include <hip/hip_runtime.h>

extern "C" __global__ void {first}(const float* a, float* b)
{{
    unsigned i = blockIdx.x * {define} + threadIdx.x;
    b[i] = a[i] + 1.0f;
}}

extern "C" __global__ void {second}(const float* a, float* b)
{{
    unsigned i = blockIdx.x * {define} + threadIdx.x;
    b[i] = a[i] * 2.0f;
}}
"""

_ROCKE_STUB_MODULE = """
    import dataclasses

    @dataclasses.dataclass
    class AttentionSpec:
        tile: int

    def build_attention(spec: AttentionSpec, *, arch="gfx942"):
        return ("kernel", spec, arch)
"""


def _hip_ks(source, entry, block):
    return {
        "kind": "hip",
        "source": source,
        "entry": entry,
        "build": {"defines": {_BLOCK_DEFINE: block}},
    }


def _rocke_ks():
    return {
        "kind": "rocke",
        "source": _ROCKE_STUB_SOURCE,
        "builder": _ROCKE_STUB_BUILDER,
        "spec": dict(_ROCKE_STUB_SPEC),
    }


def _ukd(uid, kernel_source, arch=None):
    doc = {
        "version": "0.1",
        "id": uid,
        "name": uid,
        "kernel_source": kernel_source,
        "metadata": {},
        "priority": 0,
    }
    if arch is not None:
        doc["arch"] = arch
    return doc


def _kdp(kid, arch, entries):
    # matchers/engine/dispatch are authored empty: the loader requires the keys
    # and resolves only non-null references, and this corpus is about variant
    # selection, so carrying generics would add files without adding a case.
    return {
        "version": "0.1",
        "id": kid,
        "name": kid,
        "arch": arch,
        "matchers": [],
        "engine": None,
        "dispatch": None,
        "kernelDescriptors": entries,
    }


def _write_json(dest, name, doc):
    (dest / name).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _write_corpus(dest, *, hip_only=False):
    """Write the selection corpus into `dest`, returning `dest`.

    Standard library only, and no interpreter state is touched, so the whole
    function can be copied into a checkout that does not contain this test file
    and run outside pytest.

    `hip_only=True` omits the two rocke cases (an inline rocke UKD and a KDP
    referencing a standalone rocke one). Outside pytest there is no stub for the
    rocke compiler, so a rocke UKD would reach comgr for real. Every other case
    stays: the variant-key dedup pair and the shared standalone UKD are authored
    hip precisely so the subset keeps them.

    The cases, in the order the loader sees them (`sorted(rglob("*.json"))`):

    1.  c01 -- a KDP whose arch excludes the target, carrying a standalone-UKD
        ref whose own (wildcard) arch matches. The KDP-level filter must
        short-circuit the standalone branch, so the ref is expected ABSENT.
    2.  c02 -- a matching KDP with an inline hip UKD.
    3.  c03 -- a matching KDP with an inline rocke UKD.
    4.  c04 -- a matching KDP with one admitted inline UKD and one whose own
        arch excludes the target.
    5.  c05 -- a matching KDP referencing a standalone hip UKD by id.
    6.  c06 -- a matching KDP referencing a standalone rocke UKD by id.
    7.  c07 -- a matching KDP referencing a standalone UKD whose own arch
        excludes the target, plus an admitted inline UKD so the KDP survives.
    8.  c08 -- two entries that hash to the same variant key.
    9.  c09a / c09b -- one standalone UKD referenced from two KDPs: listed by
        both, compiled once.
    10. an orphan standalone UKD no KDP references. Legal, warns, packs on, and
        is expected ABSENT from the selection.
    11. c11 -- a matching KDP whose entries all filter out, so it is dropped.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    (dest / _K1_SOURCE).write_text(
        _HIP_SOURCE_TEMPLATE.format(first="K1", second="K1B", define=_BLOCK_DEFINE),
        encoding="utf-8",
    )
    (dest / _K2_SOURCE).write_text(
        _HIP_SOURCE_TEMPLATE.format(first="K2", second="K2B", define=_BLOCK_DEFINE),
        encoding="utf-8",
    )

    if not hip_only:
        pkg = dest / _ROCKE_STUB_PKG
        (pkg / "kernels").mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "kernels" / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "kernels" / "attention.py").write_text(
            textwrap.dedent(_ROCKE_STUB_MODULE), encoding="utf-8"
        )

    # Case 1 -- the KDP-level filter must suppress the standalone ref too.
    _write_json(
        dest,
        "c01_excluded.kdp.json",
        _kdp("kdp-c01-excluded", [OTHER_ARCH], ["ukd-standalone-wild"]),
    )
    _write_json(
        dest,
        "u_standalone_wild.ukd.json",
        _ukd("ukd-standalone-wild", _hip_ks(_K1_SOURCE, "K1", 1024)),
    )

    # Case 2 -- inline hip.
    _write_json(
        dest,
        "c02_inline_hip.kdp.json",
        _kdp(
            "kdp-c02",
            [TARGET_ARCH],
            [_ukd("ukd-inline-hip", _hip_ks(_K1_SOURCE, "K1", 64))],
        ),
    )

    # Case 3 -- inline rocke.
    if not hip_only:
        _write_json(
            dest,
            "c03_inline_rocke.kdp.json",
            _kdp("kdp-c03", [TARGET_ARCH], [_ukd("ukd-inline-rocke", _rocke_ks())]),
        )

    # Case 4 -- a per-entry arch that excludes the target.
    _write_json(
        dest,
        "c04_inline_arch.kdp.json",
        _kdp(
            "kdp-c04",
            [TARGET_ARCH, OTHER_ARCH],
            [
                _ukd("ukd-inline-kept", _hip_ks(_K2_SOURCE, "K2", 64)),
                _ukd(
                    "ukd-inline-dropped",
                    _hip_ks(_K2_SOURCE, "K2", 128),
                    arch=[OTHER_ARCH],
                ),
            ],
        ),
    )

    # Case 5 -- a standalone hip UKD referenced by id.
    _write_json(
        dest,
        "c05_ref_standalone_hip.kdp.json",
        _kdp("kdp-c05", [TARGET_ARCH], ["ukd-standalone-hip"]),
    )
    _write_json(
        dest,
        "u_standalone_hip.ukd.json",
        _ukd("ukd-standalone-hip", _hip_ks(_K1_SOURCE, "K1", 256)),
    )

    # Case 6 -- a standalone rocke UKD referenced by id.
    if not hip_only:
        _write_json(
            dest,
            "c06_ref_standalone_rocke.kdp.json",
            _kdp("kdp-c06", [TARGET_ARCH], ["ukd-standalone-rocke"]),
        )
        _write_json(
            dest,
            "u_standalone_rocke.ukd.json",
            _ukd("ukd-standalone-rocke", _rocke_ks()),
        )

    # Case 7 -- a standalone UKD whose own arch excludes the target, listed
    # ahead of an admitted inline UKD so the surrounding order is observable.
    _write_json(
        dest,
        "c07_ref_standalone_arch.kdp.json",
        _kdp(
            "kdp-c07",
            [TARGET_ARCH, OTHER_ARCH],
            [
                "ukd-standalone-other-arch",
                _ukd("ukd-c07-inline", _hip_ks(_K1_SOURCE, "K1", 512)),
            ],
        ),
    )
    _write_json(
        dest,
        "u_standalone_other_arch.ukd.json",
        _ukd(
            "ukd-standalone-other-arch",
            _hip_ks(_K2_SOURCE, "K2", 256),
            arch=[OTHER_ARCH],
        ),
    )

    # Case 8 -- two UKDs sharing (source, build) and so one variant key.
    _write_json(
        dest,
        "c08_dedup.kdp.json",
        _kdp(
            "kdp-c08",
            [TARGET_ARCH],
            [
                _ukd("ukd-dedup-a", _hip_ks(_K1_SOURCE, "K1", 2048)),
                _ukd("ukd-dedup-b", _hip_ks(_K1_SOURCE, "K1B", 2048)),
            ],
        ),
    )

    # Case 9 -- one standalone UKD referenced from two KDPs.
    _write_json(
        dest,
        "c09a_shared_ref.kdp.json",
        _kdp("kdp-c09a", [TARGET_ARCH], ["ukd-standalone-shared"]),
    )
    _write_json(
        dest,
        "c09b_shared_ref.kdp.json",
        _kdp("kdp-c09b", [TARGET_ARCH], ["ukd-standalone-shared"]),
    )
    _write_json(
        dest,
        "u_standalone_shared.ukd.json",
        _ukd("ukd-standalone-shared", _hip_ks(_K2_SOURCE, "K2", 512)),
    )

    # Case 10 -- an orphan standalone UKD.
    _write_json(
        dest,
        "u_standalone_orphan.ukd.json",
        _ukd("ukd-standalone-orphan", _hip_ks(_K2_SOURCE, "K2", 1024)),
    )

    # Case 11 -- a matching KDP whose only entry filters out.
    _write_json(
        dest,
        "c11_all_filtered.kdp.json",
        _kdp(
            "kdp-c11",
            [TARGET_ARCH, OTHER_ARCH],
            [
                _ukd(
                    "ukd-c11-dropped",
                    _hip_ks(_K2_SOURCE, "K2", 4096),
                    arch=[OTHER_ARCH],
                )
            ],
        ),
    )

    return dest


# Derived by hand from the corpus above and the three arch filters, never by
# running the implementation and pasting its output. Cross-KDP order is stable
# because `load_flat_input` walks `sorted(root.rglob("*.json"))`, so KDP order
# is lexicographic on path -- if that walk is ever changed to an unsorted rglob
# this sequence goes flaky with no recorded dependency to point at.
GOLDEN_SEQUENCE = [
    "ukd-inline-hip",
    "ukd-inline-rocke",
    "ukd-inline-kept",
    "ukd-standalone-hip",
    "ukd-standalone-rocke",
    "ukd-c07-inline",
    "ukd-dedup-a",
    "ukd-dedup-b",
    "ukd-standalone-shared",
    "ukd-standalone-shared",
]

# The same derivation over the hip-only subset (cases 3 and 6 omitted).
HIP_ONLY_GOLDEN_SEQUENCE = [
    "ukd-inline-hip",
    "ukd-inline-kept",
    "ukd-standalone-hip",
    "ukd-c07-inline",
    "ukd-dedup-a",
    "ukd-dedup-b",
    "ukd-standalone-shared",
    "ukd-standalone-shared",
]

# What the hip-only subset stages into an intermediate arch tree: one .co per
# distinct variant key, and one JSON per KDP that either survives or is copied
# through. c11 is dropped, so it contributes neither.
HIP_ONLY_EXPECTED_CO_COUNT = 6
HIP_ONLY_EXPECTED_KDP_JSON_COUNT = 8

# Entries the generator must NOT yield. The orphan is reachable only from a
# `ukd_by_id()`-driven enumeration, and the wildcard standalone only if the
# KDP-level filter fails to short-circuit the standalone branch.
EXPECTED_ABSENT = ("ukd-standalone-orphan", "ukd-standalone-wild")


def _silent(*_args, **_kwargs):
    pass


@pytest.fixture
def corpus(tmp_path):
    return _write_corpus(tmp_path / "corpus")


def _entry_identity(entry_id, ukd_doc, sdesc):
    if entry_id is None:
        assert sdesc is None, "an inline entry must yield no standalone descriptor"
        return ukd_doc["id"]
    assert sdesc is not None, "a standalone entry must yield its descriptor"
    assert ukd_doc is sdesc.doc
    return entry_id


def _observed_sequence(corpus_dir):
    flat = load_flat_input(corpus_dir, log=_silent)
    ukd_by_id = flat.ukd_by_id()
    observed = []
    for kdp in flat.kdps():
        for tup in pipeline._selected_entries(kdp.doc, TARGET_ARCH, ukd_by_id):
            observed.append(_entry_identity(*tup))
    return observed


@pytest.mark.quick
def test_selected_entries_matches_golden_sequence(corpus):
    """The shared generator selects what the serial walk's loop selected.

    Compared as a sequence, not a set. Order is load-bearing: the walk appends
    to `new_kds` in yield order, that order flows into the emitted KDP JSON, and
    `pack_arch` builds its variant map by iterating the recorded UKDs in walk
    order, which fixes archive layout. A reordering defect is invisible to a set
    comparison and visible to this one.

    The absences are as much the assertion as the presences: the orphan
    standalone UKD and the standalone ref inside an arch-excluded KDP are both
    legal input the walk never compiles, and the generator must not yield them.
    """
    observed = _observed_sequence(corpus)
    assert observed == GOLDEN_SEQUENCE
    for absent in EXPECTED_ABSENT:
        assert absent not in observed


@pytest.mark.quick
def test_selected_entries_matches_golden_sequence_hip_only(tmp_path):
    """The same sequence over the hip-only corpus the pool tests are built on.

    The pool tests all run on `hip_only=True` so they need no rocKE toolchain,
    which means the corpus they select from is not the one the sequence above
    pins. Dropping the two rocke cases must remove exactly those entries and
    disturb the order of nothing else.
    """
    corpus = _write_corpus(tmp_path / "hip-only-golden", hip_only=True)
    assert _observed_sequence(corpus) == HIP_ONLY_GOLDEN_SEQUENCE


@pytest.mark.quick
def test_prewarm_jobs_are_deduped_on_variant_key(corpus):
    """The pool never compiles one variant twice.

    Corpus case 8 authors two UKDs onto one variant key and case 9 references
    one standalone UKD from two KDPs, so a job list that failed to dedup would
    be longer than the set of keys it carries.
    """
    flat = load_flat_input(corpus, log=_silent)
    jobs = pipeline._prewarm_jobs(flat, corpus, TARGET_ARCH)
    assert jobs, "the corpus selects variants, so the job list cannot be empty"
    assert len({j.vk for j in jobs}) == len(jobs)


def _arch_matches_call_sites():
    """`arch_matches` call counts in pipeline.py, keyed by enclosing function.

    Parsed rather than counted as strings: an explanatory comment naming
    `arch_matches` is not a call.
    """
    tree = ast.parse(inspect.getsource(pipeline))
    counts = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        found = 0
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == "arch_matches":
                found += 1
        if found:
            counts[node.name] = found
    return counts


@pytest.mark.quick
def test_arch_matches_call_sites_are_pinned():
    """All three selection filters live in the generator and nowhere else.

    `compile_intermediate` keeps exactly one call, and it is not a filter: it
    decides KDP disposition -- copy the authored KDP through verbatim -- before
    the deepcopy the generator would consume. A call in any other function is a
    fourth selection site, which is the divergence a single shared generator
    exists to make impossible.
    """
    assert _arch_matches_call_sites() == {
        "_selected_entries": 3,
        "compile_intermediate": 1,
    }


@pytest.mark.quick
def test_pack_jobs_env_parsing(monkeypatch):
    # Pinned to a fixed budget rather than the runner's core count: restating
    # `min(32, _cpu_budget())` asserts nothing, and a runner with 32 or fewer
    # visible CPUs never exercises the cap.
    monkeypatch.delenv("HKP_PACK_JOBS", raising=False)
    monkeypatch.setattr(pipeline, "_cpu_budget", lambda: 200)
    assert pipeline._pack_jobs() == 32
    monkeypatch.setattr(pipeline, "_cpu_budget", lambda: 6)
    assert pipeline._pack_jobs() == 6

    monkeypatch.setenv("HKP_PACK_JOBS", "1")
    assert pipeline._pack_jobs() == 1

    # An explicit value outruns the budget on purpose: oversubscribing is the
    # caller's call, and silently reducing it would make the knob a suggestion.
    monkeypatch.setenv("HKP_PACK_JOBS", "64")
    assert pipeline._pack_jobs() == 64

    monkeypatch.setenv("HKP_PACK_JOBS", "lots")
    with pytest.raises(HkpPackError, match="HKP_PACK_JOBS"):
        pipeline._pack_jobs()

    # A negative clears the parse, so the rejection is pinned here: clamped
    # onto the serial path it would produce a correct pack and no signal at all.
    monkeypatch.setenv("HKP_PACK_JOBS", "-4")
    with pytest.raises(HkpPackError, match="1 or greater"):
        pipeline._pack_jobs()

    # Zero is rejected rather than read as serial, because `0` means "auto"
    # elsewhere in this repository and a caller who writes it here means that.
    monkeypatch.setenv("HKP_PACK_JOBS", "0")
    with pytest.raises(HkpPackError, match="1 or greater"):
        pipeline._pack_jobs()


@pytest.mark.quick
def test_cpu_budget_takes_the_narrowest_limit(monkeypatch):
    """The budget is the smallest limit in force, not the host core count.

    The failure defended against is silent and container-only: the pack still
    succeeds, just several times slower than a correctly sized pool, which no
    assertion about output can catch.
    """
    monkeypatch.setattr(os, "cpu_count", lambda: 384)
    monkeypatch.setattr(pipeline, "_cgroup_v2_cpu_quota", lambda: None)
    # Absent before 3.13, so `raising=False` installs it rather than patching
    # it -- otherwise this branch is skipped on exactly those interpreters.
    monkeypatch.setattr(os, "process_cpu_count", lambda: 8, raising=False)
    assert pipeline._cpu_budget() == 8

    # A cgroup quota is invisible to every CPU-count API, so it has to be able
    # to win on its own.
    monkeypatch.setattr(pipeline, "_cgroup_v2_cpu_quota", lambda: 4)
    assert pipeline._cpu_budget() == 4

    # Without `process_cpu_count` the affinity mask carries the platforms that
    # have one, and the host count is the last resort for the ones that do not.
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(pipeline, "_cgroup_v2_cpu_quota", lambda: None)
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(12)))
        assert pipeline._cpu_budget() == 12
    else:
        assert pipeline._cpu_budget() == 384


def _fake_cgroup(monkeypatch, tmp_path, own, limits):
    """Stand up a cgroup tree: `own` is this process's cgroup, `limits` maps a
    cgroup-relative directory to the `cpu.max` text written there."""
    root = tmp_path / "cgroup"
    (root / own).mkdir(parents=True, exist_ok=True)
    for rel, text in limits.items():
        node = root / rel if rel else root
        node.mkdir(parents=True, exist_ok=True)
        (node / "cpu.max").write_text(text, encoding="utf-8")

    proc = tmp_path / "proc-self-cgroup"
    proc.write_text(f"0::/{own}\n", encoding="utf-8")

    mapping = {"/sys/fs/cgroup": root, "/proc/self/cgroup": proc}
    monkeypatch.setattr(pipeline, "Path", lambda p: mapping.get(str(p), Path(p)))


@pytest.mark.quick
def test_cgroup_quota_reads_whole_cpus(tmp_path, monkeypatch):
    """`cpu.max` parses to whole CPUs, and `max` means no limit.

    Parsed here rather than trusted because the real files are read from fixed
    paths that do not exist on Windows and are unlimited on most Linux hosts,
    so the parsing is never exercised by simply running the suite.
    """

    trees = itertools.count()

    def _quota(own, limits):
        # A fresh tree per case, so a stale `cpu.max` cannot answer for the next.
        _fake_cgroup(monkeypatch, tmp_path / f"case{next(trees)}", own, limits)
        return pipeline._cgroup_v2_cpu_quota()

    assert _quota("", {"": "800000 100000"}) == 8
    assert _quota("", {"": "max 100000"}) is None
    # A fractional allocation floors to zero CPUs, which would disable the pool
    # entirely; one worker is the smallest honest answer.
    assert _quota("", {"": "50000 100000"}) == 1
    assert _quota("", {"": "garbage"}) is None


@pytest.mark.quick
def test_cgroup_quota_walks_up_from_the_process_cgroup(tmp_path, monkeypatch):
    """A limit on an ancestor cgroup counts, not just one at the root.

    Where the limit sits depends on the cgroup namespace. Docker and Kubernetes
    give one, so the limit is at the root; Slurm and a systemd login session do
    not, and the root then has no `cpu.max` whatsoever. Measured on a real
    cgroup-v2 login node: the process sat in
    `/user.slice/user-N.slice/session-N.scope` and only those three levels
    carried the file. Reading the root alone reports every such host as
    unlimited, which is the case this pins.
    """
    scope = "user.slice/user-1.slice/session-9.scope"

    trees = itertools.count()

    def _quota(own, limits):
        # A fresh tree per case, so a stale `cpu.max` cannot answer for the next.
        _fake_cgroup(monkeypatch, tmp_path / f"case{next(trees)}", own, limits)
        return pipeline._cgroup_v2_cpu_quota()

    # The nested-scope shape, with nothing at the root at all.
    assert _quota(scope, {scope: "400000 100000"}) == 4

    # The tightest limit in the chain governs, wherever it sits.
    assert _quota(scope, {scope: "1600000 100000", "user.slice": "400000 100000"}) == 4
    assert _quota(scope, {scope: "400000 100000", "user.slice": "1600000 100000"}) == 4

    # An unlimited ancestor does not mask a limited descendant.
    assert _quota(scope, {scope: "400000 100000", "user.slice": "max 100000"}) == 4

    # Unlimited the whole way up is genuinely unlimited.
    assert _quota(scope, {scope: "max 100000", "": "max 100000"}) is None


@pytest.mark.quick
def test_cgroup_quota_is_none_without_a_v2_cgroup(tmp_path, monkeypatch):
    """A cgroup-v1-only host has no `0::` line, and reports no limit."""
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text("3:cpu,cpuacct:/some/slice\n1:name=systemd:/\n", encoding="utf-8")
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cpu.max").write_text("800000 100000", encoding="utf-8")

    mapping = {"/sys/fs/cgroup": root, "/proc/self/cgroup": proc}
    monkeypatch.setattr(pipeline, "Path", lambda p: mapping.get(str(p), Path(p)))

    assert pipeline._cgroup_v2_cpu_quota() is None


_HSACO_SOURCE = "hsaco_kernel.cpp"


@pytest.fixture
def hsaco_corpus(tmp_path):
    """A KDP carrying an hsaco UKD ahead of a compilable hip one.

    Kept out of the selection corpus deliberately: it makes
    `compile_intermediate` raise, which would stop the golden-sequence corpus
    from being walkable. The hsaco entry is authored first so the walk reaches
    its error before it would need a real hipcc for the hip entry.
    """
    dest = tmp_path / "hsaco-corpus"
    dest.mkdir()
    (dest / _HSACO_SOURCE).write_text(
        _HIP_SOURCE_TEMPLATE.format(first="H1", second="H1B", define=_BLOCK_DEFINE),
        encoding="utf-8",
    )
    (dest / "prebuilt.co").write_bytes(b"\x7fELF")
    hsaco_ukd = _ukd(
        "ukd-hsaco",
        {"kind": "hsaco", "file": "prebuilt.co", "symbol": "H1"},
    )
    hip_ukd = _ukd("ukd-hsaco-sibling", _hip_ks(_HSACO_SOURCE, "H1", 64))
    _write_json(
        dest,
        "hsaco.kdp.json",
        _kdp("kdp-hsaco", [TARGET_ARCH], [hsaco_ukd, hip_ukd]),
    )
    return dest


@pytest.mark.quick
def test_prewarm_skips_hsaco_kind(hsaco_corpus, tmp_path):
    """An hsaco UKD produces no job, and the walk stays the sole error reporter.

    This pins *current* behaviour: `_compile_ukd_variant` reads
    `kernel_source.source` before the kind dispatch and neither `hsaco` nor
    `kpack` carries one, so the walk raises `KeyError` and its `unsupported
    kind` branch is unreachable for a validly-authored UKD of either kind. That
    the `KeyError` escapes as itself rather than as an `HkpPackError` is a
    pre-existing defect this test pins rather than fixes; fixing it belongs with
    the walk's error contract, not with the prewarm.

    The raise is asserted first on purpose: with the job-list assertion ahead of
    it, a stub job list ends the test before the walk is ever exercised.
    """
    flat = load_flat_input(hsaco_corpus, log=_silent)

    with pytest.raises(KeyError, match="source"):
        pipeline.compile_intermediate(
            flat,
            hsaco_corpus,
            TARGET_ARCH,
            "hipcc",
            tmp_path / "inter",
            log=_silent,
        )

    hsaco_ukd = flat.kdps()[0].doc["kernelDescriptors"][0]
    assert pipeline._variant_key_for(hsaco_ukd, Path(".")) is None

    sibling_vk = hip_variant_key(
        hip_source_relpath(Path("."), _HSACO_SOURCE),
        {"defines": {_BLOCK_DEFINE: 64}},
    )
    jobs = pipeline._prewarm_jobs(flat, hsaco_corpus, TARGET_ARCH)
    assert [j.vk for j in jobs] == [sibling_vk]


def _synthetic_job(corpus, block):
    """One prewarm job over a real corpus source, keyed by its build block.

    Distinct blocks give distinct variant keys and so distinct output names,
    which is what lets a test fail exactly one job out of many. The key is
    computed the way the walk computes it rather than invented, so the output
    the producer writes is the one the caches are checked against.
    """
    ks = _hip_ks(_K1_SOURCE, "K1", block)
    return pipeline._VariantJob(
        vk=hip_variant_key(hip_source_relpath(Path("."), _K1_SOURCE), ks["build"]),
        kind="hip",
        ukd=_ukd(f"ukd-synthetic-{block}", ks),
        rel_dir=".",
        source_root=str(corpus),
        out_dir="",
        hipcc="",
        arch=TARGET_ARCH,
    )


@pytest.mark.quick
def test_prewarm_pool_stops_at_first_failure(tmp_path, monkeypatch):
    """The pack stops on the first failure in walk order and cancels the rest.

    Exactly one job fails, and every other one sleeps, so a pool that ran the
    queue to the end is distinguishable from one that abandoned it. The attempt
    count is bounded rather than pinned: the executor dispatches a few jobs
    beyond the running two before the parent observes the failure, so the exact
    number depends on scheduling even though `< len(jobs)` does not.

    The failing job is deliberately not the first. With `jobs[0]` failing, no
    job ever succeeds, so the empty-cache assertions below hold by construction
    and would survive the caches being filled on the failure path.

    The job list is synthesised rather than taken from the corpus, which yields
    six -- below the executor's own dispatch depth, so every job would reach a
    worker before the first result is consumed and cancellation would have
    nothing left to cancel. A queue long enough for the property to exist is
    part of the setup.
    """
    corpus = _write_corpus(tmp_path / "fail-fast", hip_only=True)
    monkeypatch.setenv("HKP_PACK_JOBS", "2")

    jobs = [_synthetic_job(corpus, 64 + i) for i in range(24)]
    monkeypatch.setattr(pipeline, "_prewarm_jobs", lambda *_a, **_k: list(jobs))
    flat = load_flat_input(corpus, log=_silent)

    tally = tmp_path / "tally"
    tally.mkdir()
    failing = jobs[3]
    hipcc = _stub_hipcc(tmp_path, fail_out=f"{failing.vk}.co", delay=1.0, tally=tally)

    variant_co = {}
    variant_symbol = {}
    with pytest.raises(HkpPackError) as excinfo:
        pipeline._prewarm_variants(
            flat,
            corpus,
            TARGET_ARCH,
            hipcc,
            tmp_path / "inter",
            variant_co,
            variant_symbol,
            log=_silent,
        )

    # Named by UKD id, which a reader can look up in the descriptors. The vk is
    # not asserted absent: the producer's stderr is quoted in the detail, and
    # the stub names the output file it refused, which is vk-derived.
    message = str(excinfo.value)
    assert f"variant '{failing.ukd['id']}'" in message
    assert TARGET_ARCH in message

    attempts = len(list(tally.iterdir()))
    assert attempts < len(jobs), (
        f"every one of the {len(jobs)} jobs was attempted -- the queue was not "
        "cancelled, so the pack is not failing fast"
    )

    # A half-filled cache is worse than an empty one: the walk skips a compile
    # for any key it finds, so a surviving entry suppresses the compile of an
    # artefact this run never produced. Three jobs succeed before the failure,
    # so there is something for a leak to leave behind.
    assert variant_co == {}
    assert variant_symbol == {}


@pytest.mark.quick
def test_variant_key_for_uses_module_globals(monkeypatch):
    """Both key functions resolve through the `pipeline` module globals.

    Two existing tests monkeypatch `pipeline.hip_variant_key` and
    `pipeline.rocke_variant_key` to a constant so every job collapses onto one
    key and the pack stays on the serial path. A function-local import, an alias
    bound at import time, or a key computed inside a worker process would all
    bypass those patches and silently disagree with the walk.
    """
    monkeypatch.setattr(pipeline, "hip_variant_key", lambda *a, **k: "SENTINEL-HIP")
    monkeypatch.setattr(pipeline, "rocke_variant_key", lambda *a, **k: "SENTINEL-ROCKE")

    hip_ukd = _ukd("ukd-key-hip", _hip_ks(_K1_SOURCE, "K1", 64))
    rocke_ukd = _ukd("ukd-key-rocke", _rocke_ks())

    assert pipeline._variant_key_for(hip_ukd, Path(".")) == "SENTINEL-HIP"
    assert pipeline._variant_key_for(rocke_ukd, Path(".")) == "SENTINEL-ROCKE"


def _child_sys_path(_ignored):
    """Run in a pool worker; returns the child's `sys.path`."""
    return list(sys.path)


def _conftest_inserted_paths():
    packaging_root = Path(__file__).resolve().parent.parent
    candidates = [packaging_root / "python"]
    rocke_root = packaging_root.parent / "rocke"
    candidates += [rocke_root / "platform" / "python", rocke_root / "library"]
    return [str(p) for p in candidates if str(p) in sys.path]


@pytest.mark.quick
def test_worker_inherits_parent_sys_path():
    """A pool worker starts with the parent's `sys.path`, conftest inserts and all.

    CPython propagates `sys.path` to children under both `spawn` and
    `forkserver`, so a worker can import `hkp_pack` and the rocKE platform
    without a `PYTHONPATH` export.

    A probe of the interpreter rather than of this package -- no change to
    `pipeline.py` can fail it. Should a future interpreter stop propagating
    `sys.path`, every rocKE variant fails to import in its worker, and this
    says why. The related constraint it does not check, that the pool must be
    built after parent-side path setup, is documented at the construction site.
    """
    expected = _conftest_inserted_paths()
    assert expected, "conftest inserts at least the hkp_pack package root"

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        child_path = list(pool.map(_child_sys_path, [None], chunksize=1))[0]

    assert set(expected) <= set(child_path)


_MISSING_MODULE = "hkp_parallel_absent/kernels/nowhere.py"


@pytest.fixture
def failing_corpus(tmp_path):
    """A KDP with two rocke UKDs naming a module that does not exist.

    Two entries rather than one because the prewarm returns without a pool for
    a single job, and the failure has to come from a real worker process. They
    carry different specs so they key apart and stay two jobs. The module is
    absent, so the child raises before it reaches the rocKE compiler and the
    case needs no toolchain at all.
    """
    dest = tmp_path / "failing-corpus"
    dest.mkdir()
    entries = [
        _ukd(
            f"ukd-absent-{tile}",
            {
                "kind": "rocke",
                "source": _MISSING_MODULE,
                "builder": _ROCKE_STUB_BUILDER,
                "spec": {"tile": tile},
            },
        )
        for tile in (64, 128)
    ]
    _write_json(dest, "absent.kdp.json", _kdp("kdp-absent", [TARGET_ARCH], entries))
    return dest


@pytest.mark.quick
def test_prewarm_failure_names_variant(failing_corpus, tmp_path, monkeypatch):
    """A pool failure names one variant, not N tracebacks.

    Which variant is named is not host-dependent and is asserted exactly: it is
    the first failure in submission order, which is walk order, so the parallel
    path names the variant the serial path would have named. How many others
    would have failed is deliberately not asserted, and the message carries no
    count.

    No `PYTHONPATH` export: children inherit the parent's `sys.path` under both
    start methods, which `test_worker_inherits_parent_sys_path` is the detector
    for.
    """
    monkeypatch.setenv("HKP_PACK_JOBS", "2")
    flat = load_flat_input(failing_corpus, log=_silent)

    jobs = pipeline._prewarm_jobs(flat, failing_corpus, TARGET_ARCH)
    assert len(jobs) >= 2, "a single job returns before starting a pool"

    with pytest.raises(HkpPackError) as excinfo:
        pipeline.compile_intermediate(
            flat,
            failing_corpus,
            TARGET_ARCH,
            "hipcc",
            tmp_path / "inter",
            log=_silent,
        )

    message = str(excinfo.value)
    assert re.search(rf"variant '\S+' failed to compile for {TARGET_ARCH}", message)
    assert f"'{jobs[0].ukd['id']}'" in message
    assert "module not importable" in message


@pytest.mark.quick
def test_compile_one_variant_returns_errors_and_computes_no_keys(tmp_path, monkeypatch):
    """The worker returns its failure and never computes a key.

    Both halves are asserted because both are invisible from the parent: a
    worker that raised instead of returning would lose an unpicklable
    diagnosis, and one that recomputed `vk` would bypass the patches in force
    when the parent computed it.
    """
    calls = []

    def _record_key(*_args, **_kwargs):
        calls.append("key")
        return "RECOMPUTED"

    def _boom(*_args, **_kwargs):
        raise HkpPackError("compile failed for k1.cpp @ gfx942 (exit 1): boom")

    monkeypatch.setattr(pipeline, "hip_variant_key", _record_key)
    monkeypatch.setattr(pipeline, "rocke_variant_key", _record_key)
    monkeypatch.setattr(pipeline, "compile_hip_variant", _boom)

    job = pipeline._VariantJob(
        vk="VK-FROM-PARENT",
        kind="hip",
        ukd=_ukd("ukd-worker", _hip_ks(_K1_SOURCE, "K1", 64)),
        rel_dir=".",
        source_root=str(tmp_path),
        out_dir=str(tmp_path / "inter"),
        hipcc="hipcc",
        arch=TARGET_ARCH,
    )

    compile_one = getattr(pipeline, "_compile_one_variant", None)
    assert compile_one is not None, "pipeline._compile_one_variant does not exist"

    vk, co_path, symbol, err = compile_one(job)
    assert vk == "VK-FROM-PARENT"
    assert co_path is None and symbol is None
    assert err.startswith("HkpPackError: ")
    assert "boom" in err
    assert calls == []


_STUB_HIPCC_BODY = r"""
import hashlib
import os
import sys
import time

FAIL_OUT = {fail_out!r}
DELAY = {delay!r}
TALLY = {tally!r}

args = sys.argv[1:]
oi = args.index("-o")
out = args[oi + 1]

# Recorded before the failure branch, so the tally counts attempts rather than
# successes -- a fail-fast assertion needs to see the job that failed.
if TALLY:
    open(os.path.join(TALLY, os.path.basename(out)), "wb").close()

if FAIL_OUT and os.path.basename(out) == FAIL_OUT:
    sys.stderr.write("stub hipcc: refusing " + FAIL_OUT + chr(10))
    sys.exit(2)

if DELAY:
    time.sleep(DELAY)

seed = repr([a for i, a in enumerate(args) if i not in (oi, oi + 1)])
with open(out, "wb") as fh:
    fh.write(bytes([127]) + b"ELF" + hashlib.sha256(seed.encode()).digest())
"""


def _stub_hipcc(tmp_path, *, fail_out=None, delay=0.0, tally=None):
    """Path to a hipcc stand-in that writes a .co and exits 0.

    Lets the pool run to success on a box with no toolchain, which is what makes
    the success path testable at all. The bytes are derived from every argument
    except the output path, so two variants of one source differ in content and
    a variant cannot pass by being confused with its sibling.

    `fail_out` is matched against the output basename rather than the source, so
    exactly one variant fails even where several share a source file -- the
    fail-fast test needs the other jobs to survive long enough to be cancelled.
    `delay` slows every other job so cancellation is observable rather than a
    race, and `tally` collects one marker per attempt.

    A launcher script rather than the interpreter directly, because the producer
    invokes `hipcc` as argv[0] of a subprocess.
    """
    stub = tmp_path / "stub_hipcc.py"
    stub.write_text(
        _STUB_HIPCC_BODY.format(
            fail_out=fail_out, delay=delay, tally=str(tally) if tally else None
        ),
        encoding="utf-8",
    )
    if sys.platform == "win32":
        launcher = tmp_path / "stub_hipcc.bat"
        launcher.write_text(
            f'@echo off\r\n"{sys.executable}" "{stub}" %*\r\n', encoding="utf-8"
        )
    else:
        launcher = tmp_path / "stub_hipcc.sh"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{stub}" "$@"\n', encoding="utf-8"
        )
        launcher.chmod(0o755)
    return launcher


@pytest.mark.quick
def test_prewarm_pool_populates_both_caches(tmp_path, monkeypatch):
    """A pool that runs to success fills both caches, each with its own value.

    The one test that exercises the pool's success path. Every other test in
    this file stops short of it: the failure test raises before the unpack loop,
    the hsaco test has a single job so no pool starts, and the rest call the
    selection helpers directly. The inherited suite never starts a pool either
    -- its packs are below the two-job threshold or patch the compile out -- so
    without this, swapping the two cache assignments changes no test result.

    `_prewarm_jobs` supplies the expected symbols. It is pinned independently by
    the golden-sequence test, and reading the authored `entry` back out of the
    jobs keeps this test from restating a vk-to-symbol table that the corpus
    would silently outgrow.
    """
    corpus = _write_corpus(tmp_path / "hip-only", hip_only=True)
    monkeypatch.setenv("HKP_PACK_JOBS", "2")

    variant_co = {}
    variant_symbol = {}
    pipeline._prewarm_variants(
        load_flat_input(corpus, log=_silent),
        corpus,
        TARGET_ARCH,
        _stub_hipcc(tmp_path),
        tmp_path / "inter",
        variant_co,
        variant_symbol,
        log=_silent,
    )

    jobs = pipeline._prewarm_jobs(
        load_flat_input(corpus, log=_silent), corpus, TARGET_ARCH
    )
    expected_symbol = {job.vk: job.ukd["kernel_source"]["entry"] for job in jobs}
    assert len(expected_symbol) == HIP_ONLY_EXPECTED_CO_COUNT

    # Asserted separately from the symbols so a swap of the two assignments
    # fails on both dicts rather than on whichever is checked first.
    assert set(variant_co) == set(expected_symbol)
    for vk, co in variant_co.items():
        assert isinstance(co, Path), f"{vk} cached a {type(co).__name__}, not a Path"
        assert co.is_file(), f"{vk} cached a path that does not exist: {co}"
        assert co.name == f"{vk}.co"

    assert variant_symbol == expected_symbol

    # Distinct sources and distinct build blocks must not collapse onto one
    # artifact: equal bytes here would mean the key space, not the pool, is wrong.
    assert len({co.read_bytes() for co in variant_co.values()}) == len(variant_co)


def _staged_tree(corpus, out_dir):
    """Every staged file under `out_dir`, keyed by path relative to it."""
    pipeline.compile_intermediate(
        load_flat_input(corpus, log=_silent),
        corpus,
        TARGET_ARCH,
        _stub_hipcc(out_dir.parent),
        out_dir,
        log=_silent,
    )
    return {
        p.relative_to(out_dir).as_posix(): p.read_bytes()
        for p in sorted(out_dir.rglob("*"))
        if p.is_file()
    }


@pytest.mark.quick
def test_serial_and_parallel_stage_identical_trees(tmp_path, monkeypatch):
    """Serial and parallel packs stage byte-identical trees.

    Asserted over the whole staged tree rather than the two caches, because the
    caches are the mechanism and the tree is the product. A prewarm writing its
    artefacts somewhere the walk does not read would leave both caches looking
    correct, and every other test here passing, while the tree diverged.
    """
    corpus = _write_corpus(tmp_path / "equiv", hip_only=True)

    monkeypatch.setenv("HKP_PACK_JOBS", "1")
    serial = _staged_tree(corpus, tmp_path / "inter-serial")
    monkeypatch.setenv("HKP_PACK_JOBS", "4")
    parallel = _staged_tree(corpus, tmp_path / "inter-parallel")

    # Pinned against the corpus so a staging change that silently drops files
    # cannot make two empty trees compare equal.
    assert len(serial) == HIP_ONLY_EXPECTED_CO_COUNT + HIP_ONLY_EXPECTED_KDP_JSON_COUNT
    assert serial == parallel


@pytest.mark.quick
def test_pack_jobs_one_starts_no_pool(tmp_path, monkeypatch):
    """`HKP_PACK_JOBS=1` returns before any pool is constructed.

    The documented escape hatch, and the only path with a serial traceback.
    Relaxing the guard to `workers < 1` would start a one-worker pool that
    still packs correctly, so no assertion on the output can catch it -- the
    pool's absence is the property, so it is asserted directly.
    """
    corpus = _write_corpus(tmp_path / "serial", hip_only=True)
    monkeypatch.setenv("HKP_PACK_JOBS", "1")

    def _no_pool(*_args, **_kwargs):
        raise AssertionError("HKP_PACK_JOBS=1 built a process pool")

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", _no_pool)

    variant_co = {}
    pipeline._prewarm_variants(
        load_flat_input(corpus, log=_silent),
        corpus,
        TARGET_ARCH,
        _stub_hipcc(tmp_path),
        tmp_path / "inter",
        variant_co,
        {},
        log=_silent,
    )

    # The walk, not the prewarm, compiles everything on this path.
    assert variant_co == {}
