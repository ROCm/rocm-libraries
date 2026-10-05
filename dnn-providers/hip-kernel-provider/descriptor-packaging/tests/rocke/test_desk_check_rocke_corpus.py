"""The four desk-check invariants of the packaging README's "Desk-check a variant
set" against real rocKE output: the `desk_check` fixture packed through the rocKE
producer, and the rocKE bundles the repository ships.

Invariant 1 reads the authored spec, which packing moves from ``kernel_source`` to
``provenance.spec``, so a check reading ``kernel_source.spec`` on packed output
always sees ``{}`` and reports "none" regardless of real drift
(``test_runbook_scripts_invariant_1_is_dead_on_packed_output`` pins that against a
real ``run_pipeline`` pack with injected drift). Invariants 2-4 read only
``metadata`` and post-pack ``kernel_source`` fields, which packing populates.

Each invariant gets a positive case (a real packed fixture) and a negative one (a
fixture engineered to violate it), so no check only ever sees valid data. The
producer-agnostic halves of invariants 2 and 3 are also held on a real hip pack
in ``tests/test_desk_check_invariants.py``.
"""

import json
import shutil

import pytest

from hkp_pack.desk_check import (
    duplicate_matcher_tuples,
    metadata_spec_drift,
    symbol_distinctness,
    toc_key_uniqueness,
)
from hkp_pack.pipeline import run_pipeline
from pack_helpers import _EXAMPLES, _ROOT_IDS, _read, _require_bundles, _run_cli

ARCH = "gfx950"
# The KMD fields the desk-check compares -- `DEFAULT_MATCHER_FIELDS`, narrowed
# to what this fixture's KMD actually declares.
_MATCHER_FIELDS = ("batch", "head_size")
_ROCKE_EXAMPLE = [root / "rocKE" for root in _EXAMPLES]


def _kernels(shipped_kdp):
    return shipped_kdp["kernelDescriptors"]


# Fixtures: pack the real desk_check fixture bundle (valid) plus small
# purpose-built variants that violate one invariant each.
@pytest.fixture(scope="module")
def desk_check_fixture(fixtures_dir):
    return fixtures_dir / "desk_check"


@pytest.fixture(scope="module")
def packed_desk_check(tmp_path_factory, desk_check_fixture, hipcc, rocm_kpack_dir):
    """Real pack of the desk_check fixture (two genuinely distinct
    attention_dense variants: head_size 64 and 128, both batch=1)."""
    tmp_path = tmp_path_factory.mktemp("desk_check_pack")
    run_pipeline(
        source_root=desk_check_fixture,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    return _read(tmp_path / "out" / ARCH / "attention.kdp.json")


def _pack_mutated(tmp_path, desk_check_fixture, hipcc, rocm_kpack_dir, mutate):
    """Copy the desk_check fixture, apply `mutate` to its KDP doc, pack it
    for real, and return the shipped KDP doc."""
    src = tmp_path / "src"
    shutil.copytree(desk_check_fixture, src)
    kdp_path = src / "attention.kdp.json"
    doc = _read(kdp_path)
    mutate(doc)
    kdp_path.write_text(json.dumps(doc), encoding="utf-8")
    run_pipeline(
        source_root=src,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    return _read(tmp_path / "out" / ARCH / "attention.kdp.json")


# ---------------------------------------------------------------------------
# Invariant 1: metadata/spec drift.
# ---------------------------------------------------------------------------
class TestInvariant1MetadataSpecDrift:
    def test_runbook_scripts_invariant_1_is_dead_on_packed_output(
        self, packed_desk_check
    ):
        """The RUNBOOK's literal script (kernel_source.get('spec', {})) reports
        'none' even when a real drift is injected: a dead check on the exact data
        it is documented to run against."""
        kernels = _kernels(packed_desk_check)
        # Inject a genuine drift: corrupt one kernel's metadata so it
        # disagrees with its own real provenance.spec.
        corrupted = json.loads(json.dumps(kernels[1]))  # deep copy
        assert corrupted["metadata"]["head_size"] == 128
        corrupted["metadata"]["head_size"] = 999  # real, injected drift

        # The RUNBOOK's literal invariant-1 comprehension, verbatim in shape.
        bad = [
            (k["name"], f)
            for k in [corrupted]
            for f in _MATCHER_FIELDS
            if f in k["kernel_source"].get("spec", {})
            and str(k["kernel_source"]["spec"][f]).lower()
            != str(k["metadata"][f]).lower()
        ]
        assert bad == [], (
            "the RUNBOOK's literal script found the injected drift -- if this "
            "assertion now fails, kernel_source carries a 'spec' key on packed "
            "output again and the dead-check finding needs re-verification"
        )

    def test_corrected_check_finds_no_drift_on_clean_packed_output(
        self, packed_desk_check
    ):
        assert metadata_spec_drift(_kernels(packed_desk_check), _MATCHER_FIELDS) == []

    def test_corrected_check_catches_real_injected_drift(self, packed_desk_check):
        kernels = json.loads(json.dumps(_kernels(packed_desk_check)))
        kernels[1]["metadata"]["head_size"] = 999
        bad = metadata_spec_drift(kernels, _MATCHER_FIELDS)
        assert bad == [(kernels[1]["name"], "head_size")]

    def test_corrected_check_also_works_on_the_authored_tree(self, desk_check_fixture):
        """The pre-pack case still works: an authored tree's
        kernel_source.spec."""
        authored = _read(desk_check_fixture / "attention.kdp.json")
        assert metadata_spec_drift(_kernels(authored), _MATCHER_FIELDS) == []


# ---------------------------------------------------------------------------
# Invariant 2: no two kernels share a matcher tuple on the same arch.
# ---------------------------------------------------------------------------
class TestInvariant2DuplicateMatcherTuples:
    def test_distinct_variants_report_no_duplicates(self, packed_desk_check):
        assert (
            duplicate_matcher_tuples(_kernels(packed_desk_check), _MATCHER_FIELDS) == {}
        )

    def test_real_pack_of_two_identical_matcher_tuples_is_detected(
        self, tmp_path, desk_check_fixture, hipcc, rocm_kpack_dir
    ):
        """Negative case, packed for real: two kernels whose spec differs only in a
        field outside the matcher tuple (seqlen_q) collapse to one
        (batch, head_size) tuple, leaving one variant unreachable."""

        def mutate(doc):
            dup = json.loads(json.dumps(doc["kernelDescriptors"][0]))
            dup["id"] = "ukd-attention-dense-d64-dup"
            dup["name"] = "Attention dense d64 duplicate seqlen"
            dup["kernel_source"]["spec"]["seqlen_q"] = 512
            dup["kernel_source"]["spec"]["seqlen_kv"] = 512
            # metadata (the matcher tuple) is UNCHANGED -- same (batch, head_size).
            doc["kernelDescriptors"].append(dup)

        shipped = _pack_mutated(
            tmp_path, desk_check_fixture, hipcc, rocm_kpack_dir, mutate
        )
        dupes = duplicate_matcher_tuples(_kernels(shipped), _MATCHER_FIELDS)
        assert dupes == {(1, 64): 2}, dupes


# ---------------------------------------------------------------------------
# Invariant 3: every variant individually addressable (toc_key uniqueness).
# ---------------------------------------------------------------------------
class TestInvariant3TocKeyUniqueness:
    def test_distinct_variants_have_distinct_toc_keys(self, packed_desk_check):
        distinct, total = toc_key_uniqueness(_kernels(packed_desk_check))
        assert distinct == total == 2

    def test_real_pack_of_a_genuine_duplicate_spec_collides_on_one_toc_key(
        self, tmp_path, desk_check_fixture, hipcc, rocm_kpack_dir
    ):
        """Negative case, packed for real: two UKDs with byte-identical
        (source, builder, spec) collapse onto ONE toc_key."""

        def mutate(doc):
            twin = json.loads(json.dumps(doc["kernelDescriptors"][0]))
            twin["id"] = "ukd-attention-dense-d64-twin"
            twin["name"] = "Attention dense d64 twin (accidental duplicate)"
            doc["kernelDescriptors"] = [doc["kernelDescriptors"][0], twin]

        shipped = _pack_mutated(
            tmp_path, desk_check_fixture, hipcc, rocm_kpack_dir, mutate
        )
        distinct, total = toc_key_uniqueness(_kernels(shipped))
        assert total == 2
        assert distinct == 1, (
            "expected the twin variant to collide onto the same toc_key as "
            "the original -- if this now shows 2, the collision no longer "
            "reproduces and the invariant-3 negative case needs revisiting"
        )


# ---------------------------------------------------------------------------
# Invariant 4: symbol names are NOT unique, and that is fine.
# ---------------------------------------------------------------------------
class TestInvariant4SymbolNonUniquenessTolerated:
    def test_distinct_shapes_get_distinct_symbols(self, packed_desk_check):
        # head_size 64 vs 128 changes the kernel_name() the builder derives,
        # so THIS fixture shows distinct symbols per kernel.
        distinct, total = symbol_distinctness(_kernels(packed_desk_check))
        assert distinct == total == 2

    def test_real_pack_where_symbol_is_shared_but_toc_key_disambiguates(
        self, tmp_path, desk_check_fixture, hipcc, rocm_kpack_dir
    ):
        """Packed for real: attention_dense's kernel_name() omits `batch`, so two
        variants differing only in batch share one symbol while remaining two
        distinct, individually-addressable toc_keys. Invariant 4 exists to say
        that is fine."""

        def mutate(doc):
            other_batch = json.loads(json.dumps(doc["kernelDescriptors"][1]))
            other_batch["id"] = "ukd-attention-dense-d128-b4"
            other_batch["name"] = "Attention dense d128 batch4"
            other_batch["kernel_source"]["spec"]["batch"] = 4
            other_batch["metadata"]["batch"] = 4
            doc["kernelDescriptors"] = [doc["kernelDescriptors"][1], other_batch]

        shipped = _pack_mutated(
            tmp_path, desk_check_fixture, hipcc, rocm_kpack_dir, mutate
        )
        kernels = _kernels(shipped)
        distinct_sym, total = symbol_distinctness(kernels)
        assert total == 2
        assert distinct_sym == 1, (
            "expected batch to be omitted from the symbol so both kernels "
            "share it -- if this now shows 2, attention_dense's kernel_name() "
            "no longer omits batch and the fixture premise needs revisiting"
        )
        # But toc_key still disambiguates them -- the tolerance is safe.
        distinct_toc, _ = toc_key_uniqueness(kernels)
        assert distinct_toc == 2


# The CLI on real rocKE output. Only a rocKE tree carries the authored spec
# invariant 1 compares, so only a rocKE tree can take the CLI to a clean exit.
class TestCliEndToEnd:
    def test_clean_real_pack_exits_zero(self, packed_desk_check, tmp_path):
        kdp_path = tmp_path / "clean.kdp.json"
        kdp_path.write_text(
            json.dumps({"kernelDescriptors": _kernels(packed_desk_check)})
        )
        proc = _run_cli(str(kdp_path))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "metadata/authored-spec drift: none" in proc.stdout
        assert "toc_key: distinct=2 of 2 OK" in proc.stdout

    def test_full_mode_refuses_the_unpacked_dialect(self, desk_check_fixture):
        """A rocKE tree read BEFORE packing has no bytes, so no producing-build
        record can bind and full mode must refuse rather than report an unverified
        pass. `verify_variant_sets` refuses the same artifact, so a pass here would
        make the two readers disagree."""
        proc = _run_cli(str(desk_check_fixture / "attention.kdp.json"), mode="full")
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "packed dialect" in proc.stdout
        assert "'rocke'" in proc.stdout
        assert "NOT VERIFIED HERE" not in proc.stdout

    def test_real_injected_drift_exits_nonzero(self, packed_desk_check, tmp_path):
        """A real packed tree with a genuine metadata/spec mismatch must fail the
        CLI, not just the underlying function: the script wires `report.ok` into
        its exit code."""
        kernels = json.loads(json.dumps(_kernels(packed_desk_check)))
        kernels[1]["metadata"]["head_size"] = 999
        kdp_path = tmp_path / "drifted.kdp.json"
        kdp_path.write_text(json.dumps({"kernelDescriptors": kernels}))

        proc = _run_cli(str(kdp_path))

        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "head_size" in proc.stdout


# Real-bundle regressions against every git-tracked rocKE bundle the repository
# carries -- no pack, no hipcc, no GPU.
@pytest.mark.quick
@pytest.mark.parametrize("rocke_root", _ROCKE_EXAMPLE, ids=_ROOT_IDS)
def test_real_rocke_example_dtype_vocabularies_are_not_drift(rocke_root):
    """rocKE specs and hipDNN KMDs spell dtype in two DELIBERATE vocabularies:
    `spec.dtype` is the builder's Python spelling ("bf16"), `metadata.dtype` the
    hipDNN DataType enum name ("BF16" here, "BFLOAT16"/"HALF" in the gfx950 dense
    bundle -- data_types.fbs:6-26), and a raw string compare false-positives on
    every rocKE kernel that ships."""
    for kdp in _require_bundles(rocke_root):
        kernels = _kernels(_read(kdp))
        spec = kernels[0]["kernel_source"]["spec"]
        meta = kernels[0]["metadata"]
        # The premise: two different spellings of one type. A failure here
        # means the bundle changed and the regression needs re-grounding.
        assert (spec["dtype"], meta["dtype"]) == ("bf16", "BF16"), kdp
        assert metadata_spec_drift(kernels, ("dtype",)) == [], kdp


@pytest.mark.quick
@pytest.mark.parametrize("rocke_root", _ROCKE_EXAMPLE, ids=_ROOT_IDS)
def test_real_rocke_example_passes_out_of_the_box(rocke_root):
    """The CLI, run exactly as an agent runs it at RUNBOOK §4's host boundary,
    against the real rocKE bundles this repository ships."""
    for kdp in _require_bundles(rocke_root):
        proc = _run_cli(str(kdp))
        assert proc.returncode == 0, str(kdp) + proc.stdout + proc.stderr
        assert "metadata/authored-spec drift: none" in proc.stdout, kdp
        assert "duplicate matcher tuples: none" in proc.stdout, kdp
