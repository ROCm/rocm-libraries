"""The four desk-check invariants of the packaging README's "Desk-check a variant
set", exercising the SHIPPED `hkp_pack.desk_check` module rather than a copy.

Invariant 1 reads the authored spec, which packing moves from ``kernel_source`` to
``provenance.spec``, so a check reading ``kernel_source.spec`` on packed output
always sees ``{}`` and reports "none" regardless of real drift
(``test_runbook_scripts_invariant_1_is_dead_on_packed_output`` pins that against a
real ``run_pipeline`` pack with injected drift). Invariants 2-4 read only
``metadata`` and post-pack ``kernel_source`` fields, which packing populates.

Each invariant gets a positive case (a real packed fixture) and a negative one (a
fixture engineered to violate it), so no check only ever sees valid data.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hkp_pack.desk_check import (
    DEFAULT_MATCHER_FIELDS,
    MODES,
    DeskCheckNoSpecFound,
    DeskCheckReport,
    compiled_agreement,
    duplicate_matcher_tuples,
    load_kernels,
    load_variant_set,
    metadata_identity_fields,
    metadata_spec_drift,
    symbol_distinctness,
    toc_key_uniqueness,
)
from hkp_pack.errors import HkpPackError
from hkp_pack.pipeline import run_pipeline

ARCH = "gfx950"
# The KMD fields the desk-check compares -- `DEFAULT_MATCHER_FIELDS`, narrowed
# to what this fixture's KMD actually declares.
_MATCHER_FIELDS = ("batch", "head_size")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


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

    def test_corrected_check_raises_when_no_spec_found_anywhere(self):
        """A tree that is neither authored (kernel_source.spec) nor packed
        (provenance.spec) must not silently report 'no drift'. Distinguishing
        'clean' from 'nothing to check' is the point of this check."""
        kernel = {
            "name": "mystery",
            "kernel_source": {"kind": "kpack"},
            "metadata": {"head_size": 128},
        }
        with pytest.raises(DeskCheckNoSpecFound):
            metadata_spec_drift([kernel], ["head_size"])

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

    @staticmethod
    def _twins(left_arch, right_arch):
        """Two kernels identical but for the arches they declare."""
        return [
            {"name": n, "metadata": {"dtype": "FLOAT"}, **({"arch": a} if a else {})}
            for n, a in (("left", left_arch), ("right", right_arch))
        ]

    def test_one_tuple_on_disjoint_arches_is_not_a_duplicate(self):
        """Each is the only candidate on its own device, so neither is
        unreachable. The runtime refuses a duplicate only on an arch both kernels
        reach."""
        kernels = self._twins(["gfx942"], ["gfx950"])
        assert duplicate_matcher_tuples(kernels, ("dtype",)) == {}

    def test_one_tuple_on_a_shared_arch_is_still_a_duplicate(self):
        """Control for the case above: arch scoping narrows the check rather than
        switching it off. A single overlapping arch is enough."""
        kernels = self._twins(["gfx942", "gfx950"], ["gfx950"])
        assert duplicate_matcher_tuples(kernels, ("dtype",)) == {("FLOAT",): 2}

    def test_an_arch_less_kernel_collides_with_every_arch(self):
        """An absent arch is the wildcard `arch_matches` reads it as, so it
        reaches the other kernel's device and the two are a real collision."""
        kernels = self._twins(None, ["gfx942"])
        assert duplicate_matcher_tuples(kernels, ("dtype",)) == {("FLOAT",): 2}

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


# The CLI itself, end to end: `tools/hkp_desk_check.py` is what an agent runs at
# RUNBOOK §4's host boundary. The invariant-function tests above import the library
# directly and would stay green even if the CLI's argument parsing, exit-code
# mapping, or output path were broken.
_TOOL = Path(__file__).resolve().parent.parent / "tools" / "hkp_desk_check.py"


def _run_cli(*args, mode="structural"):
    """The CLI as an agent runs it. The mode is always explicit, because the tool
    requires it."""
    return subprocess.run(
        [sys.executable, str(_TOOL), "--mode", mode, *args],
        capture_output=True,
        text=True,
    )


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

    def test_structural_mode_never_reports_compiled_agreement(
        self, packed_desk_check, tmp_path
    ):
        """A structural pass is a statement about the documents, so a clean
        structural run must say what it did NOT check.
        """
        kdp_path = tmp_path / "clean.kdp.json"
        kdp_path.write_text(
            json.dumps({"kernelDescriptors": _kernels(packed_desk_check)})
        )
        proc = _run_cli(str(kdp_path))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "compiled specialization agreement: NOT CHECKED" in proc.stdout
        assert "mode=structural" in proc.stdout

    def test_the_mode_is_required(self, packed_desk_check, tmp_path):
        """No default: a run whose mode is unstated cannot be read back out of a
        log, and the weaker result would read as the stronger."""
        kdp_path = tmp_path / "clean.kdp.json"
        kdp_path.write_text(
            json.dumps({"kernelDescriptors": _kernels(packed_desk_check)})
        )
        proc = subprocess.run(
            [sys.executable, str(_TOOL), str(kdp_path)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "--mode" in proc.stderr

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

    def test_authored_tree_reports_toc_key_not_applicable_and_exits_zero(
        self, desk_check_fixture
    ):
        """A pre-pack authored tree has no toc_key/symbol yet: that reads as
        NOT-APPLICABLE, never as a false 'None == None' collision, and does not
        fail the run on its own."""
        proc = _run_cli(str(desk_check_fixture / "attention.kdp.json"))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "NOT-APPLICABLE" in proc.stdout


# Real-bundle regressions: everything above runs against a purpose-built fixture,
# these against every git-tracked bundle the repository carries -- no pack, no
# hipcc, no GPU. Two roots are wired and both read: `examples/descriptors` is the
# documented sample tree, and the engine root is what a consumer loads. The engine
# root ships no bundle today, so its parametrizations skip; authoring one closes
# that with no change here, since the roots are globbed.
_PACKAGING = Path(__file__).resolve().parent.parent
_EXAMPLES = [
    _PACKAGING / "examples" / "descriptors",
    _PACKAGING.parent / "src" / "engines" / "kernel_ingestor_engine" / "descriptors",
]
_ROCKE_EXAMPLE = [root / "rocKE" for root in _EXAMPLES]
_HIP_EXAMPLE = [root / "hip" for root in _EXAMPLES]
#: Case ids that name the tree, so a failure or a skip says WHICH root it was.
_ROOT_IDS = [root.parent.name for root in _EXAMPLES]


def _require_bundles(producer_root):
    """Every `.kdp.json` under one producer subtree of one root, or a NAMED skip
    when it holds none: an emptied root and an absent producer are both legitimate,
    but the skip must name which, or a root that stopped being read looks like one
    that passed."""
    kdps = sorted(producer_root.glob("*/*.kdp.json"))
    if not kdps:
        pytest.skip(
            f"{producer_root.parent} carries no '{producer_root.name}' bundle "
            f"-- nothing to check for this producer in this root"
        )
    return kdps


@pytest.mark.quick
class TestRealBundleDtypeVocabulary:
    """rocKE specs and hipDNN KMDs spell dtype in two DELIBERATE vocabularies:
    `spec.dtype` is the builder's Python spelling ("bf16"), `metadata.dtype` the
    hipDNN DataType enum name ("BF16" here, "BFLOAT16"/"HALF" in the gfx950 dense
    bundle -- data_types.fbs:6-26), and a raw string compare false-positives on
    every rocKE kernel that ships."""

    @pytest.mark.parametrize("rocke_root", _ROCKE_EXAMPLE, ids=_ROOT_IDS)
    def test_real_rocke_example_dtype_vocabularies_are_not_drift(self, rocke_root):
        for kdp in _require_bundles(rocke_root):
            kernels = _kernels(_read(kdp))
            spec = kernels[0]["kernel_source"]["spec"]
            meta = kernels[0]["metadata"]
            # The premise: two different spellings of one type. A failure here
            # means the bundle changed and the regression needs re-grounding.
            assert (spec["dtype"], meta["dtype"]) == ("bf16", "BF16"), kdp
            assert metadata_spec_drift(kernels, ("dtype",)) == [], kdp

    @pytest.mark.parametrize(
        "spec_dtype,meta_dtype",
        [
            ("bf16", "BFLOAT16"),  # gfx950 attention_dense spelling
            ("bf16", "BF16"),  # gfx942 tiled spelling
            ("fp16", "HALF"),  # gfx950 attention_dense spelling
            ("fp32", "FLOAT"),
            ("weird_t", "weird_t"),  # unknown vocabulary, but agreeing
        ],
    )
    def test_equivalent_spellings_do_not_report_drift(self, spec_dtype, meta_dtype):
        kernels = [
            {
                "name": "k",
                "kernel_source": {"spec": {"dtype": spec_dtype}},
                "metadata": {"dtype": meta_dtype},
            }
        ]
        assert metadata_spec_drift(kernels, ("dtype",)) == []

    @pytest.mark.parametrize(
        "spec_dtype,meta_dtype",
        [
            ("bf16", "HALF"),  # the real, fatal case: wrong precision baked
            ("fp16", "BFLOAT16"),
            ("fp16", "FLOAT"),
            ("weird_t", "other_t"),  # unknown vocabulary must stay COMPARED
        ],
    )
    def test_genuine_dtype_drift_still_fails(self, spec_dtype, meta_dtype):
        """Normalising the vocabulary must not disarm the check: silencing this
        row with `--field` would make the field most worth checking the one field
        never checked."""
        kernels = [
            {
                "name": "k",
                "kernel_source": {"spec": {"dtype": spec_dtype}},
                "metadata": {"dtype": meta_dtype},
            }
        ]
        assert metadata_spec_drift(kernels, ("dtype",)) == [("k", "dtype")]


@pytest.mark.quick
class TestDriftAndTupleFieldsAreIndependent:
    """Invariant 1's drift fields and invariant 2's matcher-tuple fields are
    separate lists. Dropping `dtype` to silence the false drift above must not
    remove it from the tuple identity, which would manufacture false duplicate
    collisions."""

    def _two_variants_differing_only_in_dtype(self):
        return [
            {
                "name": "bf16",
                "kernel_source": {"spec": {"dtype": "bf16", "head_size": 64}},
                "metadata": {"dtype": "BFLOAT16", "head_size": 64},
            },
            {
                "name": "fp16",
                "kernel_source": {"spec": {"dtype": "fp16", "head_size": 64}},
                "metadata": {"dtype": "HALF", "head_size": 64},
            },
        ]

    def _two_variants_with_a_translated_field(self):
        """Two distinct variants whose `layout` the engine deliberately translates
        (spec spelling vs KMD spelling), which no alias table can know about: the
        general case `--drift-field` exists for."""
        return [
            {
                "name": "nhwc",
                "kernel_source": {"spec": {"layout": "nhwc_packed", "head_size": 64}},
                "metadata": {"layout": "NHWC", "head_size": 64},
            },
            {
                "name": "nchw",
                "kernel_source": {"spec": {"layout": "nchw_packed", "head_size": 64}},
                "metadata": {"layout": "NCHW", "head_size": 64},
            },
        ]

    def test_narrowing_drift_fields_silences_drift_but_keeps_the_tuple(self):
        kernels = self._two_variants_with_a_translated_field()
        coupled = DeskCheckReport(
            kernels, fields=("layout", "head_size"), mode="structural"
        )
        # The premise: with one shared list, `layout` false-positives.
        assert coupled.drift == [("nhwc", "layout"), ("nchw", "layout")]

        narrowed = DeskCheckReport(
            kernels,
            fields=("layout", "head_size"),
            drift_fields=("head_size",),
            mode="structural",
        )
        assert narrowed.drift == [], "drift comparison should have dropped layout"
        assert narrowed.duplicate_tuples == {}, (
            "layout was dropped from the DRIFT comparison only -- dropping it "
            "from the matcher tuple too collapses two distinct variants into "
            "a false collision, which is the defect this parameter exists for"
        )
        assert narrowed.ok

    def test_narrowing_drift_fields_does_not_narrow_the_matcher_tuple(self):
        kernels = self._two_variants_differing_only_in_dtype()
        report = DeskCheckReport(
            kernels,
            fields=("dtype", "head_size"),
            drift_fields=("head_size",),
            mode="structural",
        )
        assert report.duplicate_tuples == {}, (
            "dtype was dropped from the DRIFT comparison only -- it must "
            "still distinguish these two variants in the matcher tuple"
        )
        assert report.ok

    def test_a_real_duplicate_is_still_caught_with_narrowed_drift_fields(self):
        kernels = self._two_variants_differing_only_in_dtype()
        kernels[1]["metadata"]["dtype"] = "BFLOAT16"  # genuinely unreachable now
        kernels[1]["kernel_source"]["spec"]["dtype"] = "bf16"
        report = DeskCheckReport(
            kernels,
            fields=("dtype", "head_size"),
            drift_fields=("head_size",),
            mode="structural",
        )
        assert report.duplicate_tuples == {("BFLOAT16", 64): 2}
        assert not report.ok

    def test_the_drift_default_is_wider_than_the_matcher_field_list(self):
        """The independence runs in both directions: narrowing the matcher tuple
        never grants invariant 1 leave to stop comparing a field.

        Breaking mutation: `drift_comparable_fields`'s `return tuple(fields)`
        -> `return tuple(fields[:1])`."""
        kernels = self._two_variants_differing_only_in_dtype()
        kernels[0]["metadata"]["head_size"] = 999  # real drift
        report = DeskCheckReport(kernels, fields=("dtype",), mode="structural")
        assert report.fields == ("dtype",)
        assert report.drift_fields == ("dtype", "head_size")
        assert report.drift == [("bf16", "head_size")]


@pytest.mark.quick
class TestHeterogeneousMetadataTupleIdentity:
    """A field only some kernels declare still takes part in the tuple
    identity, wherever those kernels sit in the list: the identity must not
    depend on list order."""

    def _mixed(self):
        return [
            {
                "name": "with_block_n",
                "kernel_source": {"spec": {"head_size": 64}},
                "metadata": {"head_size": 64, "block_n": 64},
            },
            {
                "name": "without_block_n",
                "kernel_source": {"spec": {"head_size": 64}},
                "metadata": {"head_size": 64},
            },
        ]

    def test_absent_field_is_distinguishing_not_a_collision(self):
        assert duplicate_matcher_tuples(self._mixed(), ("head_size", "block_n")) == {}

    def test_result_is_independent_of_kernel_order(self):
        fields = ("head_size", "block_n")
        forward = duplicate_matcher_tuples(self._mixed(), fields)
        reverse = duplicate_matcher_tuples(list(reversed(self._mixed())), fields)
        assert forward == reverse == {}

    def test_two_kernels_both_missing_the_field_still_collide(self):
        """A field no kernel declares drops out of the identity entirely, so the
        two kernels are indistinguishable to the matcher and must collide."""
        kernels = self._mixed()
        del kernels[0]["metadata"]["block_n"]
        assert duplicate_matcher_tuples(kernels, ("head_size", "block_n")) == {(64,): 2}

    def test_absent_marker_distinguishes_only_when_some_kernel_declares_it(self):
        """Complement of the case above: once any kernel declares the field,
        "declares no block_n" and "declares block_n=64" are different variants,
        which is what `_ABSENT` encodes."""
        kernels = self._mixed() + [
            {
                "name": "third_without_block_n",
                "kernel_source": {"spec": {"head_size": 64}},
                "metadata": {"head_size": 64},
            }
        ]
        # Two kernels share (64, absent); the block_n=64 one stands alone.
        assert duplicate_matcher_tuples(kernels, ("head_size", "block_n")) == {
            (64, "<absent>"): 2
        }


@pytest.mark.quick
class TestCliOnRealShippedBundles:
    """The CLI, run exactly as an agent runs it at RUNBOOK §4's host boundary,
    against the real bundles this repository ships."""

    @pytest.mark.parametrize("rocke_root", _ROCKE_EXAMPLE, ids=_ROOT_IDS)
    def test_real_rocke_example_passes_out_of_the_box(self, rocke_root):
        for kdp in _require_bundles(rocke_root):
            proc = _run_cli(str(kdp))
            assert proc.returncode == 0, str(kdp) + proc.stdout + proc.stderr
            assert "metadata/authored-spec drift: none" in proc.stdout, kdp
            assert "duplicate matcher tuples: none" in proc.stdout, kdp

    @pytest.mark.parametrize("hip_root", _HIP_EXAMPLE, ids=_ROOT_IDS)
    def test_hip_producer_bundle_reports_could_not_check_not_a_false_clean(
        self, hip_root
    ):
        """A non-rocKE producer has no authored spec anywhere. That is "nothing to
        check", and must exit non-zero rather than render identically to
        "checked, found nothing wrong"."""
        for kdp in _require_bundles(hip_root):
            proc = _run_cli(str(kdp))
            assert proc.returncode == 1, str(kdp) + proc.stdout + proc.stderr
            assert "COULD-NOT-CHECK" in proc.stdout, kdp

    def test_drift_field_flag_is_independent_of_field_flag(self, tmp_path):
        kernels = [
            {
                "name": "bf16",
                "kernel_source": {"spec": {"dtype": "bf16", "head_size": 64}},
                "metadata": {"dtype": "BFLOAT16", "head_size": 64},
            },
            {
                "name": "fp16",
                "kernel_source": {"spec": {"dtype": "fp16", "head_size": 64}},
                "metadata": {"dtype": "HALF", "head_size": 64},
            },
        ]
        kdp = tmp_path / "two.kdp.json"
        kdp.write_text(json.dumps({"kernelDescriptors": kernels}))
        proc = _run_cli(
            str(kdp),
            "--field",
            "dtype",
            "--field",
            "head_size",
            "--drift-field",
            "head_size",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "duplicate matcher tuples: none" in proc.stdout


@pytest.mark.quick
class TestMatcherFieldsComeFromTheBundlesOwnContract:
    """The matcher-tuple identity is the bundle's OWN declaration of what the
    producing compiler specialized on. A generic list standing in for it collapses
    distinct kernels: a 2733-kernel rocKE attention bundle declares fourteen
    fields, five outside the generic list, and reported 661 false collisions."""

    def _two_kernels_differing_only_in_a_declared_field(self):
        return [
            {
                "id": f"waves-{waves}",
                "name": f"waves-{waves}",
                "kernel_source": {"spec": {"head_size": 64, "waves_per_eu": waves}},
                "metadata": {"head_size": 64, "waves_per_eu": waves},
            }
            for waves in (1, 2)
        ]

    def _bundle(self, tmp_path, metadata_fields):
        doc = {
            "kernelDescriptors": self._two_kernels_differing_only_in_a_declared_field()
        }
        if metadata_fields is not None:
            doc["provenance"] = {
                "specialization_contract": {
                    "schema_version": 1,
                    "consumers": [{"metadata_fields": list(metadata_fields)}],
                }
            }
        kdp = tmp_path / "declared.kdp.json"
        kdp.write_text(json.dumps(doc))
        return kdp

    def test_declared_fields_distinguish_what_the_generic_list_collapses(
        self, tmp_path
    ):
        kdp = self._bundle(tmp_path, ("head_size", "waves_per_eu"))
        kernels, fields = load_variant_set(kdp)
        assert fields == ("head_size", "waves_per_eu")
        assert duplicate_matcher_tuples(kernels, fields) == {}
        # The premise: waves_per_eu is not in the generic list, so the same
        # two kernels are indistinguishable under it.
        assert duplicate_matcher_tuples(kernels, DEFAULT_MATCHER_FIELDS) == {(64,): 2}
        proc = _run_cli(str(kdp))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "duplicate matcher tuples: none" in proc.stdout

    def test_a_bundle_declaring_no_contract_is_keyed_on_its_own_metadata(
        self, tmp_path
    ):
        """A bundle with nothing to say about its specialization is still keyed on
        the fields it carries. A fixed list reports a collision neither the runtime
        nor the bundle has; keyed on nothing, every kernel collides.
        """
        kdp = self._bundle(tmp_path, None)
        assert load_variant_set(kdp)[1] is None
        assert metadata_identity_fields(load_kernels(kdp)) == (
            "head_size",
            "waves_per_eu",
        )
        proc = _run_cli(str(kdp))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "duplicate matcher tuples: none" in proc.stdout

    def test_a_kernel_carrying_no_metadata_derives_an_empty_identity(self):
        """Nothing stated is nothing to key on, and the empty identity is the
        honest answer. No fallback follows it, since `duplicate_matcher_tuples`
        drops every field no kernel carries.
        """
        assert metadata_identity_fields([{"name": "k"}]) == ()
        assert duplicate_matcher_tuples(
            [{"name": "a"}, {"name": "b"}], ()
        ) == duplicate_matcher_tuples(
            [{"name": "a"}, {"name": "b"}], ("dtype", "batch")
        )

    def test_an_explicit_field_still_outranks_the_declaration(self, tmp_path):
        """A caller who names the fields is answering a different question than
        the bundle is, and must not be overruled by it."""
        kdp = self._bundle(tmp_path, ("head_size", "waves_per_eu"))
        proc = _run_cli(str(kdp), "--field", "head_size")
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "duplicate matcher tuples: {(64,): 2}" in proc.stdout


@pytest.mark.quick
class TestDriftFieldsAreNotBoundedByTheDeclaredContract:
    """The declared contract sets the matcher-tuple identity (invariant 2) and must
    NOT set the drift comparison (invariant 1): for the tuple the bundle is the
    authority, for drift it is the thing under audit. Each case declares ONE field
    and drifts on a second."""

    #: Deliberately narrow: `block_m` is real, specialized and undeclared.
    _NARROW_CONTRACT = ("head_size",)

    def _bundle(self, tmp_path, *, metadata_block_m, name="narrow"):
        """One kernel declaring only `head_size`, whose `block_m` metadata the
        caller sets to agree with or contradict the spec's 256."""
        doc = {
            "kernelDescriptors": [
                {
                    "id": "ukd-narrow",
                    "name": "narrow",
                    "kernel_source": {"spec": {"head_size": 64, "block_m": 256}},
                    "metadata": {"head_size": 64, "block_m": metadata_block_m},
                }
            ],
            "provenance": {
                "specialization_contract": {
                    "schema_version": 1,
                    "consumers": [{"metadata_fields": list(self._NARROW_CONTRACT)}],
                }
            },
        }
        root = tmp_path / name
        root.mkdir()
        kdp = root / f"{name}.kdp.json"
        kdp.write_text(json.dumps(doc))
        return kdp

    def test_drift_outside_the_declared_contract_is_reported(self, tmp_path):
        """The decisive case: `block_m` 256 was compiled in and the metadata says
        128, so a drift list drawn from the narrow declaration exits 0.

        Breaking mutation: `DeskCheckReport.__init__`'s
        `drift_comparable_fields(kernels)` -> `self.fields`."""
        kdp = self._bundle(tmp_path, metadata_block_m=128)
        # The premise: the declaration really is narrow, so the coupled default
        # would have had nothing to say about block_m.
        assert load_variant_set(kdp)[1] == self._NARROW_CONTRACT
        proc = _run_cli(str(kdp))
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "block_m" in proc.stdout

    def test_the_same_bundle_without_drift_stays_clean(self, tmp_path):
        """The control: the identical bundle whose `block_m` agrees exits 0, so the
        failure above is a detected disagreement rather than a check that fails
        everything.

        Breaking mutation: `_values_agree`'s final compare -> `return False`."""
        kdp = self._bundle(tmp_path, metadata_block_m=256)
        proc = _run_cli(str(kdp))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "metadata/authored-spec drift: none" in proc.stdout

    def test_the_generic_fallback_would_also_miss_this_field(self, tmp_path):
        """`DEFAULT_MATCHER_FIELDS` is artifact-independent, the property the
        declared list lacks, but it is a fixed attention-shaped guess without
        `block_m` while a real dense bundle specializes on block_m, waves_per_eu,
        persistent, num_persistent and use_exp2_fast.

        Breaking mutation: `DeskCheckReport.__init__`'s
        `drift_comparable_fields(kernels)` -> `DEFAULT_MATCHER_FIELDS`."""
        kdp = self._bundle(tmp_path, metadata_block_m=128)
        kernels, declared = load_variant_set(kdp)
        assert "block_m" not in DEFAULT_MATCHER_FIELDS
        assert metadata_spec_drift(kernels, DEFAULT_MATCHER_FIELDS) == []
        assert metadata_spec_drift(kernels, declared) == []
        report = DeskCheckReport(kernels, fields=declared, mode="structural")
        assert report.drift == [("narrow", "block_m")]

    def test_drift_field_still_narrows_deliberately(self, tmp_path):
        """`--drift-field` stays the explicit escape for a field whose sides speak
        vocabularies no alias table bridges, confining the comparison to
        `head_size` even though `block_m` drifts.

        Breaking mutation: `hkp_desk_check.main`'s `drift_fields = ... else None`
        -> `drift_fields = None`."""
        kdp = self._bundle(tmp_path, metadata_block_m=128)
        proc = _run_cli(str(kdp), "--drift-field", "head_size")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "metadata/authored-spec drift: none" in proc.stdout

    def test_widening_the_drift_list_does_not_widen_the_matcher_tuple(self, tmp_path):
        """Invariant 2 is untouched: two kernels differing only in the undeclared
        `block_m` are indistinguishable to a matcher keyed on `head_size`, so the
        collision is still reported.

        Breaking mutation: `DeskCheckReport.__init__`'s
        `duplicate_matcher_tuples(kernels, self.fields)` ->
        `duplicate_matcher_tuples(kernels, self.drift_fields)`."""
        kdp = self._bundle(tmp_path, metadata_block_m=256)
        doc = _read(kdp)
        twin = json.loads(json.dumps(doc["kernelDescriptors"][0]))
        twin["id"] = "ukd-narrow-twin"
        twin["name"] = "narrow twin"
        twin["kernel_source"]["spec"]["block_m"] = 128
        twin["metadata"]["block_m"] = 128  # agrees with its own spec: no drift
        doc["kernelDescriptors"].append(twin)
        kdp.write_text(json.dumps(doc))

        kernels, declared = load_variant_set(kdp)
        report = DeskCheckReport(kernels, fields=declared, mode="structural")
        assert report.fields == self._NARROW_CONTRACT
        assert "block_m" in report.drift_fields
        assert report.drift == []
        assert report.duplicate_tuples == {(64,): 2}

        proc = _run_cli(str(kdp))
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "duplicate matcher tuples: {(64,): 2}" in proc.stdout


@pytest.mark.quick
class TestStructuralDescriptorContext:
    def test_structural_entries_need_no_engine_and_keep_disjoint_arches(self, tmp_path):
        kernel = {
            "id": "standalone",
            "name": "structural",
            "arch": ["gfx950"],
            "kernel_source": {"spec": {"head_size": 64}},
            "metadata": {"head_size": 64},
        }
        kdp = tmp_path / "structural.kdp.json"
        kdp.write_text(
            json.dumps({"arch": ["gfx942"], "kernelDescriptors": [kernel["id"]]})
        )
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "kernel.ukd.json").write_text(json.dumps(kernel))
        assert load_kernels(kdp) == [kernel]
        result = _run_cli(str(kdp), "--field", "head_size")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_reference_resolution_does_not_search_above_kdp_parent(self, tmp_path):
        root = tmp_path / "shard"
        root.mkdir()
        kernel = {
            "id": "standalone",
            "name": "structural",
            "kernel_source": {"spec": {"head_size": 64}},
            "metadata": {"head_size": 64},
        }
        kdp = root / "selected.kdp.json"
        kdp.write_text(json.dumps({"kernelDescriptors": [kernel["id"]]}))
        ukd = root / "kernel.ukd.json"
        ukd.write_text(json.dumps(kernel))
        control = _run_cli(str(kdp), "--field", "head_size")
        assert control.returncode == 0, control.stdout + control.stderr
        ukd.rename(tmp_path / ukd.name)
        result = _run_cli(str(kdp), "--field", "head_size")
        assert result.returncode == 1, result.stdout + result.stderr
        assert kernel["id"] in result.stderr and str(root) in result.stderr


# A shard the FULL mode can walk. Full mode resolves every KDP under the root to
# its engine and the KMD that governs it before either path lookup runs, so the
# minimal structural fixtures above -- a lone KDP with no `engine` -- fail on that
# hop and never reach the code these last two classes cover.
def _bundle_root(root, ukd):
    """One resolvable shard holding `ukd` inline: a KDP walking by id to a UED
    and to the KMD that governs it. Returns the KDP's path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "shard.kdp.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": "kdp-shard",
                "name": "shard",
                "arch": [ARCH],
                "engine": "ued-shard",
                "kernelDescriptors": [ukd],
            }
        )
    )
    (root / "shard.ued.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": "ued-shard",
                "name": "shard",
                "metadata": "kmd-shard",
            }
        )
    )
    (root / "shard.kmd.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": "kmd-shard",
                "name": "shard",
                "fields": [
                    {"name": "block_size", "type": "int", "default_value": 16},
                    {"name": "dtype", "type": "string"},
                ],
            }
        )
    )
    return root / "shard.kdp.json"


def _inline_ukd():
    """One inline rocKE kernel whose contract exhausts `_bundle_root`'s KMD."""
    return {
        "version": "1.0",
        "id": "ukd-shard",
        "name": "shard kernel",
        "arch": [ARCH],
        "kernel_source": {
            "kind": "rocke",
            "source": "k.py",
            "builder": "b",
            "spec": {"block_size": 16, "dtype": "bf16"},
        },
        "metadata": {"block_size": 16, "dtype": "BF16"},
        "priority": 0,
        "provenance": {
            "specialization_contract": {
                "schema_version": 1,
                "consumers": [
                    {
                        "engine_id": "ued-shard",
                        "kmd_id": "kmd-shard",
                        "metadata_fields": ["block_size", "dtype"],
                        "matcher_only_fields": [],
                        "bindings": {
                            "block_size": {"field": "block_size"},
                            "dtype": {"field": "dtype"},
                        },
                        "vocabulary": {"dtype": {"bf16": "BF16"}},
                    }
                ],
            }
        },
    }


@pytest.mark.quick
class TestAKdpPathMatchingNothingIsReportedNotRaised:
    """A `kdp` argument naming no indexed descriptor is a mistyped or stale path.

    Both entry points look the file up in an index built from its parent directory,
    and a bare `StopIteration` escapes the CLI's `HkpPackError` handlers. Structural
    drives `_resolve` and full reaches `compiled_agreement`'s bundle selection
    first, so each lookup is covered.
    """

    def _typo_beside_a_real_shard(self, tmp_path):
        """A populated index and a path that matches nothing in it, which is the
        mistyped-filename case rather than an empty directory."""
        _bundle_root(tmp_path / "shard", _inline_ukd())
        return tmp_path / "shard" / "typo.kdp.json"

    def test_structural_lookup_raises_a_named_finding(self, tmp_path):
        typo = self._typo_beside_a_real_shard(tmp_path)
        with pytest.raises(HkpPackError, match="typo.kdp.json"):
            load_variant_set(typo)

    def test_full_mode_lookup_raises_a_named_finding(self, tmp_path):
        typo = self._typo_beside_a_real_shard(tmp_path)
        with pytest.raises(HkpPackError, match="typo.kdp.json"):
            compiled_agreement(typo)

    @pytest.mark.parametrize("mode", MODES)
    def test_the_cli_prints_the_finding_rather_than_a_traceback(self, tmp_path, mode):
        typo = self._typo_beside_a_real_shard(tmp_path)
        proc = _run_cli(str(typo), mode=mode)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "Traceback" not in proc.stderr, proc.stderr
        assert "StopIteration" not in proc.stderr, proc.stderr
        assert str(typo) in proc.stdout + proc.stderr

    def test_the_real_shard_in_the_same_directory_still_resolves(self, tmp_path):
        """The control: the refusal is of the path that matches nothing, not of
        every lookup the same index serves."""
        kdp = _bundle_root(tmp_path / "shard", _inline_ukd())
        kernels, _declared = load_variant_set(kdp)
        assert [k["name"] for k in kernels] == ["shard kernel"]


@pytest.mark.quick
class TestAnIdLessInlineKernelIsReportedNotAKeyError:
    """Full mode keys every consumer record on the UKD id, and nothing on the READ
    path requires one: `_require(ukd, ["id", ...])` runs in the packing pipeline,
    while `descriptor_context.Index` only parses JSON. Only an inline entry can
    reach `consumer_records` id-less, since `by_id` indexes nothing id-less.
    """

    def test_an_id_less_inline_kernel_names_itself_in_the_failure(self, tmp_path):
        ukd = _inline_ukd()
        del ukd["id"]
        kdp = _bundle_root(tmp_path / "shard", ukd)
        proc = _run_cli(str(kdp), mode="full")
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "KeyError" not in proc.stderr, proc.stderr
        assert "declares no 'id'" in proc.stdout
        assert "shard kernel" in proc.stdout

    def test_the_same_bundle_carrying_an_id_gets_past_the_keying(self, tmp_path):
        """The control: an id is what the record keying is missing, not the
        bundle. Carrying one, the run reaches the pre-pack refusal full mode owes
        an unpacked rocKE tree."""
        kdp = _bundle_root(tmp_path / "shard", _inline_ukd())
        proc = _run_cli(str(kdp), mode="full")
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "declares no 'id'" not in proc.stdout
        assert "packed dialect" in proc.stdout
