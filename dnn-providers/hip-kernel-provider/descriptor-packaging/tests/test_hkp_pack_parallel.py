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
import json
import os
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
    """A yielded tuple reduced to the id of the UKD it selects."""
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

    Parsed rather than counted as strings: this plan requires an explanatory
    comment naming `arch_matches`, and a comment is not a call.
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
    monkeypatch.delenv("HKP_PACK_JOBS", raising=False)
    assert pipeline._pack_jobs() == min(32, os.cpu_count() or 1)

    monkeypatch.setenv("HKP_PACK_JOBS", "1")
    assert pipeline._pack_jobs() == 1

    monkeypatch.setenv("HKP_PACK_JOBS", "lots")
    with pytest.raises(HkpPackError, match="HKP_PACK_JOBS"):
        pipeline._pack_jobs()


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

    This pins *current* behaviour: `pipeline.py:183` reads `kernel_source.source`
    before the kind dispatch and neither `hsaco` nor `kpack` carries one, so the
    walk raises `KeyError` and the `unsupported kind` branch at
    `pipeline.py:227-228` is unreachable for a validly-authored UKD of either
    kind -- a separate ticket tracks that the `KeyError` is not an `HkpPackError`.

    The raise is asserted first on purpose. With the job-list assertion ahead of
    it, a stub job list ends the test before the walk is ever exercised, which is
    how the contradiction between this test and the code stayed hidden.

    `_variant_key_for` returns None for a kind the prewarm cannot compile rather
    than raising. Had it raised, the prewarm would become the reporter of the
    failure the walk produces -- a different error context, a different
    traceback, and the walk would no longer own its own error reporting.
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


@pytest.mark.quick
def test_first_failure_is_submission_order():
    """The reported failure is the first among failures in submission order.

    Not first-completed and not lowest-key: submission order is walk order, so
    this names the same variant the serial path would have named. `pool.map`
    yields in submission order for exactly this reason; switching to
    `as_completed` would make the reported variant depend on scheduling.
    """
    results = [
        ("vk-0", "/inter/vk-0.co", "S0", None),
        ("vk-1", None, None, "HkpPackError: module not importable"),
        ("vk-2", "/inter/vk-2.co", "S2", None),
        ("vk-3", None, None, "HkpPackError: compile failed"),
    ]
    outcome = pipeline._first_failure(results)
    assert outcome is not None, "two of the four results carry an error"
    first, failure_count = outcome
    assert first[0] == "vk-1"
    assert first[3] == "HkpPackError: module not importable"
    assert failure_count == 2


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
    without a `PYTHONPATH` export. This test is the detector for that, not a
    driver of any change: it is expected green from the first run.

    What it would catch is the constraint that rides on the finding. Workers
    snapshot `sys.path` at process start, so the pool must be created after all
    parent-side path setup and must never be cached in a module global -- a
    reused pool would freeze a stale path list from whenever it was first built.
    """
    expected = _conftest_inserted_paths()
    assert expected, "conftest inserts at least the hkp_pack package root"

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        child_path = list(pool.map(_child_sys_path, [None], chunksize=1))[0]

    assert set(expected) <= set(child_path)


@pytest.mark.quick
def test_compile_one_variant_returns_errors_and_computes_no_keys(
    tmp_path, monkeypatch
):
    """The worker returns its failure and never computes a key.

    Returning rather than raising, because rocke and comgr exceptions are not
    guaranteed picklable and an exception that cannot cross the process boundary
    loses the diagnosis. Computing no key, because the parent already computed
    it under whatever patches are in force; a key recomputed in the child would
    bypass them and disagree with the walk.
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
    )

    compile_one = getattr(pipeline, "_compile_one_variant", None)
    assert compile_one is not None, "pipeline._compile_one_variant does not exist"

    vk, co_path, symbol, err = compile_one(job)
    assert vk == "VK-FROM-PARENT"
    assert co_path is None and symbol is None
    assert err.startswith("HkpPackError: ")
    assert "boom" in err
    assert calls == []
