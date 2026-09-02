"""RUNBOOK.md Step 5d's four desk-check invariants, exercising the SHIPPED
`hkp_pack.desk_check` module (not a private copy of its logic -- a copy is
exactly how invariant 1 went dead in the first place: the RUNBOOK's prose
snippet and reality drifted apart with nothing to notice).

RUNBOOK.md ("Desk-check the shipped set -- no GPU required") states four
invariants over a shipped variant set. Before this module existed they lived
only as a shell-embedded Python snippet in the markdown -- untestable prose --
and that snippet was WRONG on the exact data it is documented to run against
("KDP=<the shipped .kdp.json under the packed tree>"): after packing,
``kernel_source`` is rewritten to kpack form (``{kind, library, toc_key,
symbol, sha256}``); the authored ``spec`` dict moves to ``provenance.spec``.
So the RUNBOOK's original ``kernel_source.get("spec", {})`` was always ``{}``
on real packed output, and invariant 1 (metadata/spec drift) silently
reported "none" regardless of real drift. Verified here with a real
``run_pipeline`` pack (real hipcc + comgr + rocm_kpack) and an injected
genuine drift the old literal script missed
(``test_runbook_scripts_invariant_1_is_dead_on_packed_output`` below).

Invariants 2-4 (duplicate matcher tuple, toc_key uniqueness, symbol
non-uniqueness tolerance) are verified CORRECT on real packed output -- they
read only ``metadata`` and post-pack ``kernel_source`` fields
(``toc_key``/``symbol``), which the pack step does populate.

Each invariant gets a POSITIVE case (a real packed fixture that satisfies
it) and a NEGATIVE case (a fixture engineered to violate it, proving the
check would actually catch the real defect it exists for -- a check that
only ever sees valid data is decoration).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hkp_pack.desk_check import (
    DeskCheckNoSpecFound,
    DeskCheckReport,
    duplicate_matcher_tuples,
    metadata_spec_drift,
    symbol_distinctness,
    toc_key_uniqueness,
)
from hkp_pack.pipeline import run_pipeline

ARCH = "gfx950"
# The KMD fields the desk-check compares -- mirrors RUNBOOK.md's F list,
# narrowed to what this fixture's KMD actually declares.
_MATCHER_FIELDS = ("batch", "head_size")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _kernels(shipped_kdp):
    return shipped_kdp["kernelDescriptors"]


# ---------------------------------------------------------------------------
# Fixtures: pack the real desk_check fixture bundle (valid) plus small
# purpose-built variants that violate one invariant each.
# ---------------------------------------------------------------------------
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
        """The RUNBOOK's literal script (kernel_source.get('spec', {})) must
        report 'none' even when a real drift is injected -- proving it is a
        dead check on the exact data it is documented to run against."""
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
        (provenance.spec) -- e.g. a hip-producer UKD, or a badly hand-edited
        one -- must not silently report 'no drift'. Distinguishing 'clean'
        from 'nothing to check' is the whole point of the fix."""
        kernel = {
            "name": "mystery",
            "kernel_source": {"kind": "kpack"},
            "metadata": {"head_size": 128},
        }
        with pytest.raises(DeskCheckNoSpecFound):
            metadata_spec_drift([kernel], ["head_size"])

    def test_corrected_check_also_works_on_the_authored_tree(self, desk_check_fixture):
        """The fix must not regress the pre-pack case the RUNBOOK's script
        DID handle correctly: an authored tree's kernel_source.spec."""
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
        """Negative case, packed for real: two kernels whose spec differs
        only in a field NOT in the matcher tuple (seqlen_q) collapse to one
        (batch, head_size) tuple -- one variant would be unreachable."""

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
        (source, builder, spec) collapse onto ONE toc_key -- exactly the
        'two variants share one blob' case invariant 3 exists to catch."""

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
        # so THIS fixture happens to show distinct symbols per kernel --
        # itself a real, verified fact worth pinning.
        distinct, total = symbol_distinctness(_kernels(packed_desk_check))
        assert distinct == total == 2

    def test_real_pack_where_symbol_is_shared_but_toc_key_disambiguates(
        self, tmp_path, desk_check_fixture, hipcc, rocm_kpack_dir
    ):
        """Negative-for-uniqueness / positive-for-tolerance case, packed for
        real: attention_dense's kernel_name() omits `batch`
        (attention_dense.py; rocke-mining.md's stated omission), so two
        variants differing ONLY in batch legitimately share one symbol while
        remaining two distinct, individually-addressable toc_keys. A desk
        check that key on symbol alone would wrongly flag this as a
        collision; invariant 4 exists to say that is fine."""

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


# ---------------------------------------------------------------------------
# The CLI itself, end to end: `tools/hkp_desk_check.py` is the shipped thing
# RUNBOOK step 5d will actually tell an agent to run -- these are not
# redundant with the invariant-function tests above, which import the
# library directly and would stay green even if the CLI's argument parsing,
# exit-code mapping, or output path were broken.
# ---------------------------------------------------------------------------
_TOOL = Path(__file__).resolve().parent.parent / "tools" / "hkp_desk_check.py"


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, str(_TOOL), *args], capture_output=True, text=True
    )


class TestCliEndToEnd:
    def test_clean_real_pack_exits_zero(self, packed_desk_check, tmp_path):
        kdp_path = tmp_path / "clean.kdp.json"
        kdp_path.write_text(
            json.dumps({"kernelDescriptors": _kernels(packed_desk_check)})
        )
        proc = _run_cli(str(kdp_path))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "metadata/spec drift: none" in proc.stdout
        assert "toc_key: distinct=2 of 2 OK" in proc.stdout

    def test_real_injected_drift_exits_nonzero(self, packed_desk_check, tmp_path):
        """The exact defect this whole tool exists for: a real packed tree
        with a genuine metadata/spec mismatch must fail the CLI, not just
        the underlying function -- proving the shipped script wires the
        library's `report.ok` into its own exit code correctly."""
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
        """A pre-pack authored tree has no toc_key/symbol yet -- that must
        read as NOT-APPLICABLE, never as a false 'None == None' collision,
        and must not fail the run on its own."""
        proc = _run_cli(str(desk_check_fixture / "attention.kdp.json"))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "NOT-APPLICABLE" in proc.stdout


# ---------------------------------------------------------------------------
# Real-bundle regressions. Every test above this line runs against a fixture
# built for the test; all three defects below survived those 179 tests and a
# careful reading, and appeared the moment the CLI was pointed at a REAL
# shipped bundle. So these run against the real, git-tracked rocKE example
# under `examples/descriptors/` -- no pack, no hipcc, no GPU, so they run
# everywhere the suite does.
# ---------------------------------------------------------------------------
_EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "descriptors"
_ROCKE_EXAMPLE = _EXAMPLES / "rocKE" / "gfx942_tiled_attention"
_HIP_EXAMPLE = _EXAMPLES / "hip" / "pointwise_add"


@pytest.mark.quick
class TestRealBundleDtypeVocabulary:
    """rocKE specs and hipDNN KMDs spell dtype in two DELIBERATE
    vocabularies: `spec.dtype` is what the builder's Python takes ("bf16"),
    `metadata.dtype` is the hipDNN DataType enum name the matcher compares
    against the graph ("BF16" here, "BFLOAT16"/"HALF" in the gfx950 dense
    bundle -- data_types.fbs:6-26). A raw string compare called that drift
    and false-positived on EVERY rocKE kernel that ships; the 32-kernel
    gfx950 bundle reported 32 failures out of the box. `grep -c
    "BFLOAT16\\|bf16"` on this file returned 0 before these tests, which is
    exactly why it shipped.
    """

    def test_real_rocke_example_dtype_vocabularies_are_not_drift(self):
        kernels = _kernels(_read(_ROCKE_EXAMPLE / "tiled_attention.kdp.json"))
        spec = kernels[0]["kernel_source"]["spec"]
        meta = kernels[0]["metadata"]
        # The premise: two different spellings of one type. If this ever
        # fails, the bundle changed and the regression needs re-grounding.
        assert (spec["dtype"], meta["dtype"]) == ("bf16", "BF16")
        assert metadata_spec_drift(kernels, ("dtype",)) == []

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
        """Normalising the vocabulary must not disarm the check. Silencing
        this row with `--field` -- the tool's original advice -- would have
        made the field most worth checking the one field never checked."""
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
    """One field list used to feed both invariant 1 and invariant 2. An
    agent following the tool's own advice -- drop `dtype` to silence the
    false drift above -- also dropped it from the matcher-tuple identity,
    and the duplicate check then reported 16 false collisions on the
    32-kernel gfx950 bundle: one false positive silenced, another
    manufactured, in the check whose entire job is catching unreachable
    variants."""

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
        """Two distinct variants whose `layout` the engine deliberately
        translates (spec spelling vs KMD spelling), which no alias table
        can know about -- the general case `--drift-field` exists for, and
        the only shape that proves the two lists are really independent."""
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
        coupled = DeskCheckReport(kernels, fields=("layout", "head_size"))
        # The premise: with one shared list, `layout` false-positives.
        assert coupled.drift == [("nhwc", "layout"), ("nchw", "layout")]

        narrowed = DeskCheckReport(
            kernels, fields=("layout", "head_size"), drift_fields=("head_size",)
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
            kernels, fields=("dtype", "head_size"), drift_fields=("head_size",)
        )
        assert report.duplicate_tuples == {("BFLOAT16", 64): 2}
        assert not report.ok

    def test_drift_fields_defaults_to_fields(self):
        kernels = self._two_variants_differing_only_in_dtype()
        kernels[0]["metadata"]["head_size"] = 999  # real drift
        report = DeskCheckReport(kernels, fields=("dtype", "head_size"))
        assert report.drift_fields == report.fields
        assert report.drift == [("bf16", "head_size")]


@pytest.mark.quick
class TestHeterogeneousMetadataTupleIdentity:
    """`duplicate_matcher_tuples` derived its field set from `kernels[0]`
    alone. A set where only a LATER kernel declared a field raised
    KeyError; in the other list order it silently dropped the field from
    the identity and reported a collision between two genuinely distinct
    variants. The tuple identity must not depend on list order."""

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
        """A field NO kernel declares drops out of the identity entirely --
        it distinguishes nothing, so the two kernels are genuinely
        indistinguishable to the matcher and must collide."""
        kernels = self._mixed()
        del kernels[0]["metadata"]["block_n"]
        assert duplicate_matcher_tuples(kernels, ("head_size", "block_n")) == {(64,): 2}

    def test_absent_marker_distinguishes_only_when_some_kernel_declares_it(self):
        """The complement of the case above: once ANY kernel declares the
        field, "declares no block_n" and "declares block_n=64" are different
        variants and must not collide -- which is what `_ABSENT` encodes."""
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
    """The CLI, run exactly as RUNBOOK step 5d tells an agent to run it,
    against the real bundles this repository ships. The out-of-box run on a
    real bundle is the case that was never exercised."""

    def test_real_rocke_example_passes_out_of_the_box(self):
        proc = _run_cli(str(_ROCKE_EXAMPLE / "tiled_attention.kdp.json"))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "metadata/spec drift: none" in proc.stdout
        assert "duplicate matcher tuples: none" in proc.stdout

    def test_hip_producer_bundle_reports_could_not_check_not_a_false_clean(self):
        """A non-rocKE producer has no authored spec anywhere. That is
        "nothing to check", and must exit non-zero rather than render
        identically to "checked, found nothing wrong"."""
        proc = _run_cli(str(_HIP_EXAMPLE / "pointwise_add.kdp.json"))
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "COULD-NOT-CHECK" in proc.stdout

    def test_drift_field_flag_is_independent_of_field_flag(self, tmp_path):
        """End-to-end proof of the escape hatch that used to corrupt a
        second invariant: narrowing --drift-field leaves --field's tuple
        identity intact."""
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
