"""The four desk-check invariants of the packaging README's "Desk-check a variant
set", exercising the SHIPPED `hkp_pack.desk_check` module rather than a copy.

Everything here is producer-agnostic: synthesised variant sets, the hip bundles
the repository ships, and a real hip pack for invariants 2 and 3, which read only
``metadata`` and the ``toc_key`` packing stamps. Invariants 1 and 4 on real
output need what only rocKE produces -- an authored spec to drift from, and a
builder-derived symbol that can omit a field -- and are held in
``tests/rocke/test_desk_check_rocke_corpus.py``.
"""

import json
import shutil
import subprocess
import sys

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
    toc_key_uniqueness,
)
from hkp_pack.errors import HkpPackError
from hkp_pack.pipeline import run_pipeline
from pack_helpers import _EXAMPLES, _ROOT_IDS, _TOOL, _read, _require_bundles, _run_cli

ARCH = "gfx950"


# ---------------------------------------------------------------------------
# Invariant 1: metadata/spec drift.
# ---------------------------------------------------------------------------
class TestInvariant1MetadataSpecDrift:
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


# ---------------------------------------------------------------------------
# Invariant 2: no two kernels share a matcher tuple on the same arch.
# ---------------------------------------------------------------------------
class TestInvariant2DuplicateMatcherTuples:
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


# ---------------------------------------------------------------------------
# Invariants 2 and 3 on a real hip pack.
# ---------------------------------------------------------------------------
# Both read fields packing writes -- metadata carried through, toc_key stamped
# from the variant key -- so a pack by either producer exercises them. A hip
# kernel has no authored spec, so the CLI's invariant-1 verdict on this output
# is COULD-NOT-CHECK and its exit status is 1 whatever the other invariants say;
# each invariant's own report line is asserted instead.
_HIP_ARCH = "gfx942"
_HIP_BLOCK_DEFINE = "HIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE"


def _second_variant(build_block, metadata_block):
    """A second kernel for the solo shard: its first kernel with the compiled
    block size and the declared one set independently."""

    def mutate(ukd):
        ukd["kernel_source"]["build"]["defines"][_HIP_BLOCK_DEFINE] = build_block
        ukd["metadata"]["block_size"] = metadata_block

    return mutate


@pytest.fixture
def packed_hip_variant_set(tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir):
    """Pack the empty_arch solo shard with a second inline kernel derived from its
    first by `mutate`, returning the shipped KDP's path."""

    def _pack(mutate):
        src = tmp_path / "src"
        shutil.copytree(empty_arch_fixture, src)
        kdp_path = src / "solo.kdp.json"
        doc = _read(kdp_path)
        second = json.loads(json.dumps(doc["kernelDescriptors"][0]))
        second["id"] += "-second"
        second["name"] += " (second)"
        mutate(second)
        doc["kernelDescriptors"].append(second)
        kdp_path.write_text(json.dumps(doc), encoding="utf-8")
        run_pipeline(
            source_root=src,
            arches=[_HIP_ARCH],
            out_root=tmp_path / "out",
            hipcc=hipcc,
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=tmp_path / "inter",
        )
        return tmp_path / "out" / _HIP_ARCH / "solo.kdp.json"

    return _pack


class TestInvariantsOnARealHipPack:
    @pytest.mark.parametrize(
        "second,duplicates,distinct_toc",
        [
            # Compiled and declared apart: nothing to report.
            (_second_variant(128, 128), {}, 2),
            # Compiled apart, declared alike: two blobs the matcher cannot tell
            # apart, so one is unreachable.
            (_second_variant(128, 64), {(64, "FLOAT"): 2}, 2),
            # An accidental copy: one compile, one toc_key for two kernels.
            (_second_variant(64, 64), {(64, "FLOAT"): 2}, 1),
        ],
        ids=["distinct", "shared-matcher-tuple", "twin"],
    )
    def test_the_packed_variant_set_reports_what_packing_made_of_it(
        self, packed_hip_variant_set, second, duplicates, distinct_toc
    ):
        kdp = packed_hip_variant_set(second)
        kernels, declared = load_variant_set(kdp)
        # The premise: the bundle's own contract, not a generic list, keys the
        # tuple, and packing stamped a toc_key on both kernels.
        assert declared == ("block_size", "dtype")
        assert all(k["kernel_source"].get("toc_key") for k in kernels)

        assert duplicate_matcher_tuples(kernels, declared) == duplicates
        assert toc_key_uniqueness(kernels) == (distinct_toc, 2)

        proc = _run_cli(str(kdp))
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "metadata/authored-spec drift: COULD-NOT-CHECK" in proc.stdout
        assert f"duplicate matcher tuples: {duplicates or 'none'}" in proc.stdout
        verdict = "OK" if distinct_toc == 2 else "COLLISION"
        assert f"toc_key: distinct={distinct_toc} of 2 {verdict}" in proc.stdout


# The CLI itself, end to end: `tools/hkp_desk_check.py` is what an agent runs at
# RUNBOOK §4's host boundary. The invariant-function tests import the library
# directly and would stay green even if the CLI's argument parsing, exit-code
# mapping, or output path were broken.


# The CLI's own contract, on synthesised variant sets: nothing here depends on
# which producer made the descriptors.
def _variant_set(tmp_path, *, packed, drift=False):
    """Two kernels in one KDP, in post-pack shape or pre-pack shape.

    Pre-pack, the authored spec sits in `kernel_source.spec`. Post-pack, it sits
    in `provenance.spec` and `kernel_source` carries the toc_key and symbol
    packing stamps instead, which is the shape a pack writes. `drift` makes the
    second kernel's metadata disagree with its spec.
    """
    kernels = []
    for head_size in (64, 128):
        spec = {"head_size": head_size}
        declared = head_size * 2 if drift and head_size == 128 else head_size
        kernel = {"name": f"d{head_size}", "metadata": {"head_size": declared}}
        if packed:
            kernel["kernel_source"] = {
                "toc_key": f"toc-{head_size}",
                "symbol": f"k{head_size}",
            }
            kernel["provenance"] = {"spec": spec}
        else:
            kernel["kernel_source"] = {"spec": spec}
        kernels.append(kernel)
    kdp = tmp_path / "variants.kdp.json"
    kdp.write_text(json.dumps({"kernelDescriptors": kernels}))
    return kdp


@pytest.mark.quick
class TestCliEndToEnd:
    def test_structural_mode_never_reports_compiled_agreement(self, tmp_path):
        """A structural pass is a statement about the documents, so a clean
        structural run must say what it did NOT check.
        """
        proc = _run_cli(str(_variant_set(tmp_path, packed=True)))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "compiled specialization agreement: NOT CHECKED" in proc.stdout
        assert "mode=structural" in proc.stdout

    def test_drift_in_a_packed_variant_set_fails_the_run(self, tmp_path):
        """Packing moves the authored spec to `provenance.spec`, so that is where
        invariant 1 reads it on packed output. Drift found there fails the run."""
        proc = _run_cli(str(_variant_set(tmp_path, packed=True, drift=True)))
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "metadata/authored-spec drift: [('d128', 'head_size')]" in proc.stdout

    def test_the_mode_is_required(self, tmp_path):
        """No default: a run whose mode is unstated cannot be read back out of a
        log, and the weaker result would read as the stronger."""
        proc = subprocess.run(
            [sys.executable, str(_TOOL), str(_variant_set(tmp_path, packed=True))],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "--mode" in proc.stderr

    def test_a_pre_pack_variant_set_reports_toc_key_not_applicable(self, tmp_path):
        """A pre-pack variant set has no toc_key/symbol yet: that reads as
        NOT-APPLICABLE, never as a false 'None == None' collision, and does not
        fail the run on its own. Two kernels, so the false collision would show."""
        proc = _run_cli(str(_variant_set(tmp_path, packed=False)))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "toc_key: NOT-APPLICABLE" in proc.stdout
        assert "symbols: NOT-APPLICABLE" in proc.stdout
        assert "COLLISION" not in proc.stdout


# Real-bundle regressions: everything above runs against a purpose-built fixture,
# these against every git-tracked bundle the repository carries -- no pack, no
# hipcc, no GPU. Two roots are wired and both read: `examples/descriptors` is the
# documented sample tree, and the engine root is what a consumer loads. A producer
# subtree a root does not carry skips with a named reason. New bundles need no change
# here, since the roots are globbed.
_HIP_EXAMPLE = [root / "hip" for root in _EXAMPLES]


@pytest.mark.quick
class TestRealBundleDtypeVocabulary:
    """rocKE specs and hipDNN KMDs spell dtype in two DELIBERATE vocabularies:
    `spec.dtype` is the builder's Python spelling ("bf16"), `metadata.dtype` the
    hipDNN DataType enum name ("BF16" here, "BFLOAT16"/"HALF" in the gfx950 dense
    bundle -- data_types.fbs:6-26), and a raw string compare false-positives on
    every rocKE kernel that ships."""

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
