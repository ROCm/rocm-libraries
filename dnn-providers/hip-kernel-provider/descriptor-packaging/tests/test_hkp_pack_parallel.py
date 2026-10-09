"""Selection, worker-knob, and failure-reporting tests for the parallel prewarm.

The corpus below has a double duty. It backs the in-process golden-sequence
assertion pytest runs against a `tmp_path`, and it backs the out-of-process
staged-tree capture a plain script runs against a pristine checkout of the base
commit. That second consumer is why `_write_corpus` is a standalone
standard-library-only function rather than a fixture body: the capture script
copies it verbatim into a tree that has never seen this file.
"""

import concurrent.futures
import itertools
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest

from hkp_pack import agreement, pipeline
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from hkp_pack.hip_compile import hip_source_relpath, hip_variant_key
from pack_helpers import (
    _child_sys_path,
    _conftest_only_paths,
    _kdp,
    _silent,
    _ukd,
    _write_json,
)

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

# An embedded_source names its file relative to its descriptor and is never
# handed to a producer, so it sits in a child folder no compile ever reads.
_EMBEDDED_SOURCE_FILE = "kernels/embedded.cpp"

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


def _embedded_ks(entry_point):
    return {
        "kind": "embedded_source",
        "source_file": _EMBEDDED_SOURCE_FILE,
        "entry_point": entry_point,
    }


def _rocke_ks():
    return {
        "kind": "rocke",
        "source": _ROCKE_STUB_SOURCE,
        "builder": _ROCKE_STUB_BUILDER,
        "spec": dict(_ROCKE_STUB_SPEC),
    }


def _write_corpus(dest, *, hip_only=False, with_embedded=False):
    """Write the selection corpus into `dest`, returning `dest`.

    Standard library only, as are the `pack_helpers` authoring helpers it calls,
    and no interpreter state is touched, so it can be copied with them into a
    checkout that does not contain these files and run outside pytest.

    `hip_only=True` omits the two rocke cases (an inline rocke UKD and a KDP
    referencing a standalone rocke one). Outside pytest there is no stub for the
    rocke compiler, so a rocke UKD would reach comgr for real. Every other case
    stays: the variant-key dedup pair and the shared standalone UKD are authored
    hip precisely so the subset keeps them.

    `with_embedded=True` appends the two embedded_source cases. It is purely
    additive: the default corpus, and the `hip_only` subset of it, are what
    every other consumer here pins.

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
    12. c12 -- a matching KDP with an inline embedded_source UKD (`with_embedded`).
    13. c13 -- a matching KDP referencing a standalone embedded_source UKD
        (`with_embedded`). Both pass-through cases run no producer, so neither
        yields a variant key and neither stages a .co.
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

    if with_embedded:
        embedded = dest / _EMBEDDED_SOURCE_FILE
        embedded.parent.mkdir(parents=True, exist_ok=True)
        embedded.write_text(
            _HIP_SOURCE_TEMPLATE.format(first="E1", second="E1B", define="64"),
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

    if with_embedded:
        # Case 12 -- an inline pass-through UKD.
        _write_json(
            dest,
            "c12_inline_embedded.kdp.json",
            _kdp(
                "kdp-c12",
                [TARGET_ARCH],
                [_ukd("ukd-inline-embedded", _embedded_ks("E1"))],
            ),
        )

        # Case 13 -- a standalone pass-through UKD referenced by id.
        _write_json(
            dest,
            "c13_ref_standalone_embedded.kdp.json",
            _kdp("kdp-c13", [TARGET_ARCH], ["ukd-standalone-embedded"]),
        )
        _write_json(
            dest,
            "u_standalone_embedded.ukd.json",
            _ukd("ukd-standalone-embedded", _embedded_ks("E1B")),
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

# The same, over the hip-only subset with the embedded cases added. The two
# extra KDPs stage their JSON and nothing else; a standalone UKD is emitted by
# `pack_arch`, not staged here, so c13's own file is not counted.
MIXED_EXPECTED_CO_COUNT = 6
MIXED_EXPECTED_KDP_JSON_COUNT = 10

# Entries the generator must NOT yield. The orphan is reachable only from a
# `ukd_by_id()`-driven enumeration, and the wildcard standalone only if the
# KDP-level filter fails to short-circuit the standalone branch.
EXPECTED_ABSENT = ("ukd-standalone-orphan", "ukd-standalone-wild")


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

    Kept out of the selection corpus deliberately: its hsaco entry has no
    compile, so it yields no prewarm job, and only the hip sibling does.
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
        arch=[TARGET_ARCH],
    )
    hip_ukd = _ukd("ukd-hsaco-sibling", _hip_ks(_HSACO_SOURCE, "H1", 64))
    _write_json(
        dest,
        "hsaco.kdp.json",
        _kdp("kdp-hsaco", [TARGET_ARCH], [hsaco_ukd, hip_ukd]),
    )
    return dest


@pytest.mark.quick
def test_prewarm_skips_hsaco_kind(hsaco_corpus):
    """An hsaco UKD produces no job: it has no compile.

    `_variant_key_for` declines the kind, so the prewarm drops it and only the
    compilable hip sibling is scheduled; the walk keys the hsaco UKD itself.
    """
    flat = load_flat_input(hsaco_corpus, log=_silent)

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
    variant_observations = {}
    with pytest.raises(HkpPackError) as excinfo:
        pipeline._prewarm_variants(
            flat,
            corpus,
            TARGET_ARCH,
            hipcc,
            tmp_path / "inter",
            variant_co,
            variant_symbol,
            variant_observations,
            {},
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


@pytest.mark.quick
def test_worker_inherits_parent_sys_path():
    """A pool worker starts with the parent's `sys.path`, conftest inserts and all.

    CPython propagates `sys.path` to children under both `spawn` and
    `forkserver`, so a worker can import `hkp_pack`, and any producer module a
    conftest put on the path, without a `PYTHONPATH` export.

    A probe of the interpreter rather than of this package -- no change to
    `pipeline.py` can fail it. Should a future interpreter stop propagating
    `sys.path`, every variant whose producer is imported in the worker fails
    there, and this says why. The related constraint it does not check, that the
    pool must be built after parent-side path setup, is documented at the
    construction site.

    The path watched is the rocm_kpack directory conftest inserts from
    HIPKERNELPROVIDER_ROCM_KPACK_DIR, which no `PYTHONPATH` carries. The hkp_pack
    package root conftest also inserts is exported by the ctest environment, so a
    child that re-read `PYTHONPATH` would have it too.
    """
    kpack_dir = os.environ.get("HIPKERNELPROVIDER_ROCM_KPACK_DIR")
    if not kpack_dir:
        pytest.skip(
            "HIPKERNELPROVIDER_ROCM_KPACK_DIR is unset, so conftest inserts no "
            "path that PYTHONPATH does not already carry"
        )
    expected = _conftest_only_paths([kpack_dir])
    assert expected, "the premise: conftest inserted it and PYTHONPATH lacks it"

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        child_path = list(pool.map(_child_sys_path, [None], chunksize=1))[0]

    assert set(expected) <= set(child_path)


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

    vk, co_path, symbol, err, observations, origins = compile_one(job)
    assert vk == "VK-FROM-PARENT"
    assert co_path is None and symbol is None and observations is None
    assert err.startswith("HkpPackError: ")
    assert "boom" in err
    assert calls == []
    # A failed variant contributes no producer identity. A partial one would let
    # the parent observer record a SHA for a compile that produced no artefact.
    assert origins == {}


_STUB_HIPCC_BODY = r"""
import hashlib
import os
import sys
import time

FAIL_OUT = {fail_out!r}
SLOW_FAIL_OUT = {slow_fail_out!r}
DELAY = {delay!r}
TALLY = {tally!r}

args = sys.argv[1:]
oi = args.index("-o")
out = args[oi + 1]
name = os.path.basename(out)

# Recorded before the failure branch, so the tally counts attempts rather than
# successes -- a fail-fast assertion needs to see the job that failed.
if TALLY:
    open(os.path.join(TALLY, name), "wb").close()

if name == SLOW_FAIL_OUT:
    time.sleep(DELAY)
if name in (FAIL_OUT, SLOW_FAIL_OUT):
    sys.stderr.write("stub hipcc: refusing " + name + chr(10))
    sys.exit(2)

if DELAY:
    time.sleep(DELAY)

seed = repr([a for i, a in enumerate(args) if i not in (oi, oi + 1)])
with open(out, "wb") as fh:
    fh.write(bytes([127]) + b"ELF" + hashlib.sha256(seed.encode()).digest())
"""


def _stub_hipcc(tmp_path, *, fail_out=None, slow_fail_out=None, delay=0.0, tally=None):
    """Path to a hipcc stand-in that writes a .co and exits 0.

    Lets the pool run to success on a box with no toolchain, which is what makes
    the success path testable at all. The bytes are derived from every argument
    except the output path, so two variants of one source differ in content and
    a variant cannot pass by being confused with its sibling.

    `fail_out` is matched against the output basename rather than the source, so
    exactly one variant fails even where several share a source file -- the
    fail-fast test needs the other jobs to survive long enough to be cancelled.
    `delay` slows every other job so cancellation is observable rather than a
    race, and `tally` collects one marker per attempt. `slow_fail_out` names a
    second output that fails only after `delay`, so a failure submitted earlier
    can be made to finish later.

    A launcher script rather than the interpreter directly, because the producer
    invokes `hipcc` as argv[0] of a subprocess.
    """
    stub = tmp_path / "stub_hipcc.py"
    stub.write_text(
        _STUB_HIPCC_BODY.format(
            fail_out=fail_out,
            slow_fail_out=slow_fail_out,
            delay=delay,
            tally=str(tally) if tally else None,
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
def test_the_pool_names_the_first_failure_in_walk_order(tmp_path, monkeypatch):
    """With several variants failing, the one named is the first in walk order,
    not the first to finish: that is the variant the serial path would have
    stopped on, and a name chosen by completion order changes run to run.

    Made deterministic rather than left to the scheduler: the first job fails
    after a delay and the second at once, so a pool that took the first failure
    to complete would name the second. Driven through `compile_intermediate`,
    the entry the pack itself calls, with real worker processes.
    """
    corpus = _write_corpus(tmp_path / "two-failures", hip_only=True)
    monkeypatch.setenv("HKP_PACK_JOBS", "2")
    flat = load_flat_input(corpus, log=_silent)

    first, second = pipeline._prewarm_jobs(flat, corpus, TARGET_ARCH)[:2]
    hipcc = _stub_hipcc(
        tmp_path,
        fail_out=f"{second.vk}.co",
        slow_fail_out=f"{first.vk}.co",
        delay=1.0,
    )

    with pytest.raises(HkpPackError) as excinfo:
        pipeline.compile_intermediate(
            flat,
            corpus,
            TARGET_ARCH,
            hipcc,
            tmp_path / "inter",
            log=_silent,
        )

    message = str(excinfo.value)
    assert re.search(rf"variant '\S+' failed to compile for {TARGET_ARCH}", message)
    assert f"variant '{first.ukd['id']}'" in message
    assert second.ukd["id"] not in message


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
    variant_observations = {}
    pipeline._prewarm_variants(
        load_flat_input(corpus, log=_silent),
        corpus,
        TARGET_ARCH,
        _stub_hipcc(tmp_path),
        tmp_path / "inter",
        variant_co,
        variant_symbol,
        variant_observations,
        {},
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
def test_mixed_kinds_stage_identical_trees_under_a_real_pool(tmp_path, monkeypatch):
    """A corpus mixing compiled and pass-through kinds packs the same either way.

    The equivalence above runs on a corpus every entry of which the prewarm
    keys, so a KDP the pool contributes nothing to is never staged in a parallel
    pack. `_variant_key_for` declines every embedded_source entry, and those
    KDPs must reach the tree regardless of which path compiled the rest.

    The pool's construction is asserted rather than assumed. `_prewarm_variants`
    returns without one below two jobs or two workers, so a selection change
    that took the corpus under that threshold would leave the second pack serial
    too -- the trees would still match, over a property no longer exercised.
    """
    corpus = _write_corpus(tmp_path / "mixed", hip_only=True, with_embedded=True)

    monkeypatch.setenv("HKP_PACK_JOBS", "1")
    serial = _staged_tree(corpus, tmp_path / "inter-serial")

    real_pool = pipeline.ProcessPoolExecutor
    built = []

    def _spy_pool(*args, **kwargs):
        built.append(kwargs["max_workers"])
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", _spy_pool)
    monkeypatch.setenv("HKP_PACK_JOBS", "4")
    parallel = _staged_tree(corpus, tmp_path / "inter-parallel")

    # One pool for the one arch, sized to the requested worker count rather than
    # clamped to the job count, which this corpus keeps above it.
    assert built == [4]

    # Pinned against the corpus so a staging change that silently drops files
    # cannot make two empty trees compare equal.
    assert len(serial) == MIXED_EXPECTED_CO_COUNT + MIXED_EXPECTED_KDP_JSON_COUNT
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
        {},
        {},
        log=_silent,
    )

    # The walk, not the prewarm, compiles everything on this path.
    assert variant_co == {}


@pytest.mark.quick
def test_absorbed_and_observed_identities_share_one_key():
    """An exported identity lands on the key a direct observation would build. The
    two reach the observer as different types (a worker's string, an in-process
    `Path`), and a mismatch files one file twice, letting a genuine SHA
    disagreement pass as two unrelated producers.
    """
    observed = agreement.OriginObserver()
    observed.identity(pipeline._pack_jobs)
    exported = observed.exported()
    assert len(exported) == 1

    merged = agreement.OriginObserver()
    merged.identity(pipeline._pack_jobs)
    merged.absorb(exported)
    assert len(merged.files) == 1


@pytest.mark.quick
def test_absorbing_a_disagreeing_sha_is_refused():
    """The merge goes through the conflict check, not over it: `dict.update` would
    take the later SHA and drop the earlier one, the condition this observer exists
    to report.
    """
    merged = agreement.OriginObserver()
    merged.identity(pipeline._pack_jobs)

    with pytest.raises(HkpPackError, match="producer changed during compilation"):
        merged.absorb({key: "0" * 64 for key in merged.exported()})
