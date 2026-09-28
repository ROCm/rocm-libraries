"""The parallel prewarm's rocKE side: a rocKE variant failing in a pool worker,
and the producer-origin agreement a pool of rocKE variants must keep. Both run
the rocKE producer path in `hkp_pack.rocke_compile`, stubbed at comgr, over the
selection corpus helpers of `tests/test_hkp_pack_parallel.py`.
"""

import importlib
import pickle
import re
import sys
import textwrap
from dataclasses import replace

import pytest

from hkp_pack import pipeline, rocke_compile
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from test_hkp_pack_parallel import (
    TARGET_ARCH,
    _ROCKE_STUB_BUILDER,
    _kdp,
    _silent,
    _ukd,
    _write_json,
)

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


# One stable producing invocation stands behind a whole pack, not behind each
# variant separately. A pooled variant observes its producer in a worker process,
# so the property survives only if the worker's observations come back.

_EDITABLE_PKG = "hkp_parallel_editable"
_EDITABLE_SOURCE = f"{_EDITABLE_PKG}/kernels/editable.py"
_EDITABLE_BUILDER = "build_editable"

_EDITABLE_MODULE = """
    import dataclasses

    @dataclasses.dataclass
    class EditableSpec:
        tile: int

    def build_editable(spec: EditableSpec, *, arch="gfx942"):
        return ("kernel", spec, arch)
"""


class _FakeRockeArtifact:
    def __init__(self, name, data):
        self.kernel_name = name
        self.hsaco = data


class _FakeComgrError(Exception):
    pass


@pytest.fixture
def editable_producer(tmp_path, monkeypatch):
    """A two-variant rocke corpus whose producer module can be edited mid-pack.

    Both variants name the same defining file, so an edit between their compiles is
    a disagreement rather than two unrelated observations; their specs differ so
    they stay two jobs. The comgr entry is stubbed in this process, so the workers
    run in-process too via `_SerialPool`. Yields (corpus, producer_path).
    """
    corpus = tmp_path / "editable-corpus"
    pkg = corpus / _EDITABLE_PKG / "kernels"
    pkg.mkdir(parents=True)
    (corpus / _EDITABLE_PKG / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    producer = pkg / "editable.py"
    producer.write_text(textwrap.dedent(_EDITABLE_MODULE), encoding="utf-8")

    entries = [
        _ukd(
            f"ukd-editable-{tile}",
            {
                "kind": "rocke",
                "source": _EDITABLE_SOURCE,
                "builder": _EDITABLE_BUILDER,
                "spec": {"tile": tile},
            },
        )
        for tile in (64, 128)
    ]
    _write_json(
        corpus, "editable.kdp.json", _kdp("kdp-editable", [TARGET_ARCH], entries)
    )

    monkeypatch.syspath_prepend(str(corpus))
    importlib.invalidate_caches()

    def _fake_compile(kernel, *, arch, capture_ir_text=False, backend=None):
        return _FakeRockeArtifact("editable_symbol", b"\x7fELF-stub")

    monkeypatch.setattr(
        rocke_compile, "_load_compiler", lambda: (_fake_compile, _FakeComgrError)
    )

    try:
        yield corpus, producer
    finally:
        # The module name is fixed while its file lives under a per-test
        # directory, so a cached entry would hand the next test a producer
        # whose source file no longer exists.
        for name in [n for n in sys.modules if n.split(".")[0] == _EDITABLE_PKG]:
            del sys.modules[name]


class _SerialPool:
    """An in-process pool stand-in running jobs one at a time in order, because the
    case below turns on one variant compiling before an edit to the producer and
    the other after it; `between` runs after each result. Jobs and results are
    pickled across the call as the real boundary does, so an unpicklable origin map
    cannot pass here and fail a real pack.
    """

    def __init__(self, between=None):
        self.between = between

    def map(self, fn, jobs, chunksize=1):
        def _run():
            for index, job in enumerate(jobs):
                if index and self.between is not None:
                    self.between()
                result = fn(pickle.loads(pickle.dumps(job)))
                yield pickle.loads(pickle.dumps(result))

        return _run()

    def shutdown(self, **_kwargs):
        pass


def _serial_pool(monkeypatch, between=None):
    """Put `_SerialPool` where `_prewarm_variants` builds its pool."""
    monkeypatch.setattr(
        pipeline, "ProcessPoolExecutor", lambda **_kwargs: _SerialPool(between)
    )


def _editable_jobs(corpus, out_dir):
    flat = load_flat_input(corpus, log=_silent)
    jobs = pipeline._prewarm_jobs(flat, corpus, TARGET_ARCH)
    return flat, [replace(job, out_dir=str(out_dir), hipcc="hipcc") for job in jobs]


@pytest.mark.quick
def test_worker_returns_picklable_producer_origins(editable_producer, tmp_path):
    """A worker hands back the producer identities it observed, in picklable form.

    The one end of the merge the parent cannot reconstruct: an empty map, or one
    keyed by something that does not pickle, leaves nothing to compare.
    """
    corpus, producer = editable_producer
    _flat, jobs = _editable_jobs(corpus, tmp_path / "inter")
    assert len(jobs) == 2, "the corpus authors two distinct rocke variants"

    _vk, _co, _symbol, err, _observations, origins = pipeline._compile_one_variant(
        jobs[0]
    )
    assert err is None, err

    assert pickle.loads(pickle.dumps(origins)) == origins
    assert all(isinstance(key, str) for key in origins), (
        "a worker's origin map crosses a process boundary; string keys are the "
        "one spelling both sides build for a file"
    )
    # The builder and its spec class both live in the producer module, so its
    # resolved path is the key the parent's own records of that file land on.
    assert str(producer.resolve()) in origins


@pytest.mark.quick
def test_pool_rejects_variants_built_by_different_producer_revisions(
    editable_producer, tmp_path, monkeypatch
):
    """Two pooled variants whose producer SHAs disagree fail the pack. The property
    is cross-variant: each compile is internally consistent, so only an observer
    spanning the arch sees two revisions of one file. Run through the pool branch,
    since the serial path shares one observer and could not fail this way.
    """
    corpus, producer = editable_producer
    monkeypatch.setenv("HKP_PACK_JOBS", "2")

    def _edit_producer():
        # Appended between two compiles, after the first variant's own stability
        # check has re-read the file and agreed with itself: an edit inside a
        # compile would be caught there and prove nothing cross-variant.
        with producer.open("a", encoding="utf-8") as fh:
            fh.write("\n# a revision the first variant was not built from\n")

    _serial_pool(monkeypatch, between=_edit_producer)

    with pytest.raises(HkpPackError) as excinfo:
        pipeline.compile_intermediate(
            load_flat_input(corpus, log=_silent),
            corpus,
            TARGET_ARCH,
            "hipcc",
            tmp_path / "inter",
            log=_silent,
        )

    message = str(excinfo.value)
    assert "producer changed during compilation" in message
    assert str(producer.resolve()) in message


@pytest.mark.quick
def test_pool_accepts_variants_built_by_one_producer_revision(
    editable_producer, tmp_path, monkeypatch
):
    """An unedited producer packs both variants, merge and all: a merge that raised
    for any two variants sharing a producer would satisfy the rejection test while
    making every real pack fail.
    """
    corpus, producer = editable_producer
    monkeypatch.setenv("HKP_PACK_JOBS", "2")
    _serial_pool(monkeypatch)

    inter = pipeline.compile_intermediate(
        load_flat_input(corpus, log=_silent),
        corpus,
        TARGET_ARCH,
        "hipcc",
        tmp_path / "inter",
        log=_silent,
    )

    assert len(inter.variant_co) == 2
    assert all(co.is_file() for co in inter.variant_co.values())
    assert producer.read_text(encoding="utf-8") == textwrap.dedent(_EDITABLE_MODULE)
