"""Multi-launch UKDs (`sources`), the checks that need no rocke: authored and packed
descriptor validation, `agreement.verify` over a shipped UKD whose launches share
or split code objects, and every reader of a UKD's kernel given a `sources` UKD.
The cases that compile a builder are in `rocke/test_hkp_pack_sources_rocke.py`.
"""

import copy
import dataclasses
import hashlib
import json
import itertools
import re
from pathlib import Path

import pytest

from hkp_pack import agreement, desk_check, pipeline, provenance_sidecar, sources
from hkp_pack.descriptor_context import Entry
from hkp_pack.descriptors import (
    Descriptor,
    FlatInput,
    _validate_ukd_fields,
    load_flat_input,
)
from hkp_pack.errors import HkpPackError
from hkp_pack.kernel_signature import exported_kernels, kernel_signature
from pack_helpers import GENERIC_TARGETS, _kdp, _restamp, _silent, _write_json
from synthesised_objects import _arg, _kernel, _object, requires_msgpack
from test_specialization_agreement import (
    ARCH,
    ENGINE,
    FULL_KMD,
    DemoSpec,
    consumer,
    demo_builder,
    observe_for,
)

PRIMARY = "decode_split"
MERGE = "decode_merge"
REDUCE = "decode_reduce"
SEGMENT = "attn_segment"

ARGS = {
    PRIMARY: [_arg("global_buffer", 8, 0, "X"), _arg("by_value", 4, 8, "N")],
    MERGE: [
        _arg("global_buffer", 8, 0, "A"),
        _arg("global_buffer", 8, 8, "B"),
        _arg("by_value", 4, 16, "N"),
    ],
    REDUCE: [_arg("global_buffer", 8, 0, "R")],
    SEGMENT: [_arg("global_buffer", 8, 0, "S"), _arg("by_value", 4, 8, "K")],
}
METADATA = {"head_size": 64, "dtype": "BF16", "causal": 1, "block_n": 64}
METADATA["use_exp2_fast"] = 1

#: One module holding the primary and a `merge` companion.
MODULE = {"primary": PRIMARY, "companions": {"merge": MERGE}}
#: Two launches taking one module's two kernels.
ONE_MODULE = {"decode": ("vk-a", None), "merge": ("vk-a", "merge")}
#: Two builders, one kernel each (the segment + reduce shape).
TWO_BUILDERS = {"segment": ("vk-seg", None), "reduce": ("vk-red", None)}
MODULES = {
    "vk-a": MODULE,
    "vk-seg": {"primary": SEGMENT, "companions": {}},
    "vk-red": {"primary": REDUCE, "companions": {}},
}


def _code_object(module):
    requires_msgpack()
    names = [module["primary"], *module["companions"].values()]
    return _object([_kernel(name, ARGS[name]) for name in names])


def shipped(launches=ONE_MODULE, version="1.0"):
    """A shipped rocke `sources` UKD over synthesised objects.

    `launches` is `{launch: (variant key, role)}`; each variant key names a module
    in `MODULES`. Returns `(doc, records, payload)` with `payload` keyed by launch.
    """
    entry = consumer()
    records = agreement.canonical_records(
        [
            agreement.consumer_record(
                {"id": "ukd-demo", "metadata": METADATA},
                ENGINE,
                FULL_KMD,
                {"id": "kdp-demo", "engine": "ued-demo", "arch": [ARCH]},
                ARCH,
                entry,
            )
        ]
    )
    objects, observed, entries = {}, {}, {}
    for name, (vk, role) in launches.items():
        module = MODULES[vk]
        if vk not in objects:
            objects[vk] = _code_object(module)
            observations = observe_for(DemoSpec(head_size=64), entry)
            observations["symbol"] = module["primary"]
            observations["compile_target"] = ARCH
            observations["code_object_sha256"] = hashlib.sha256(objects[vk]).hexdigest()
            if module["companions"]:
                observations["companions"] = dict(module["companions"])
            observed[vk] = observations
        symbol = module["primary"] if role is None else module["companions"][role]
        entries[name] = pipeline.LaunchEntry(
            variant_key=vk,
            symbol=symbol,
            source=f"kernels/demo/{vk}.py",
            builder=f"build_{vk.replace('-', '_')}",
            spec={"head_size": 64},
            compile_target=ARCH,
            observations=observed[vk],
            role=role,
            origin_kind="rocke",
        )
    record = pipeline.InlineUKD(
        id="ukd-demo",
        name="demo variant",
        metadata=METADATA,
        priority=0,
        source=None,
        entry=None,
        build=None,
        symbol=None,
        variant_key=None,
        extra={"version": version},
        origin_kind="rocke",
        consumers=records,
        launches=entries,
    )
    doc = pipeline._rewrite_ukd_kpack(
        record,
        ARCH,
        None,
        None,
        signature=None,
        packed_launches={
            name: {
                "toc_key": launch.variant_key,
                "sha256": hashlib.sha256(objects[launch.variant_key]).hexdigest(),
                "signature": kernel_signature(
                    objects[launch.variant_key], launch.symbol, "t"
                ),
            }
            for name, launch in entries.items()
        },
    )
    payload = {name: objects[launch.variant_key] for name, launch in entries.items()}
    return doc, records, payload


def _evidence(doc, name):
    return doc["provenance"]["effective_spec"][name]


class TestShippedSources:
    @pytest.mark.parametrize("launches", [ONE_MODULE, TWO_BUILDERS])
    def test_each_entry_is_a_full_kpack_kernel_source_without_a_role(self, launches):
        doc, records, payload = shipped(launches)
        assert "kernel_source" not in doc
        assert set(doc["sources"]) == set(launches)
        for name, ks in doc["sources"].items():
            assert set(ks) == {
                "kind",
                "library",
                "toc_key",
                "symbol",
                "sha256",
                "signature",
            }
            assert ks["kind"] == "kpack"
            assert ks["signature"] == kernel_signature(payload[name], ks["symbol"], "t")
            assert ks["sha256"] == hashlib.sha256(payload[name]).hexdigest()
        agreement.verify(doc, records, payload)

    def test_entries_on_one_module_share_the_object_and_differ_in_symbol(self):
        doc, _records, _payload = shipped(ONE_MODULE)
        decode, merge = doc["sources"]["decode"], doc["sources"]["merge"]
        assert (decode["library"], decode["toc_key"], decode["sha256"]) == (
            merge["library"],
            merge["toc_key"],
            merge["sha256"],
        )
        assert (decode["symbol"], merge["symbol"]) == (PRIMARY, MERGE)
        assert decode["signature"] != merge["signature"]
        # Entries sharing a compile carry identical evidence.
        assert _evidence(doc, "decode") == _evidence(doc, "merge")

    def test_entries_on_two_builders_have_their_own_objects_and_evidence(self):
        doc, _records, _payload = shipped(TWO_BUILDERS)
        seg, red = doc["sources"]["segment"], doc["sources"]["reduce"]
        assert seg["toc_key"] != red["toc_key"] and seg["sha256"] != red["sha256"]
        assert _evidence(doc, "segment") != _evidence(doc, "reduce")
        assert set(doc["provenance"]["sources"]) == {"segment", "reduce"}

    @pytest.mark.parametrize("authored", ["0.1", "1.0", "1.1", "1.2", "2.0"])
    def test_a_sources_ukd_ships_at_no_less_than_1_1(self, authored):
        doc, records, payload = shipped(version=authored)
        major, minor = (int(p) for p in authored.split("."))
        assert doc["version"] == (authored if (major, minor) >= (1, 1) else "1.1")
        # The stamp is in the bytes the evidence digests.
        agreement.verify(doc, records, payload)

    def test_the_role_is_authored_only_and_never_on_a_packed_entry(self):
        doc, _records, _payload = shipped()
        assert all("role" not in ks for ks in doc["sources"].values())
        # It is recorded where the descriptor digest binds it, for the verifier.
        assert doc["provenance"]["sources"]["merge"]["role"] == "merge"
        assert "role" not in doc["provenance"]["sources"]["decode"]

    def test_the_shipped_document_round_trips_through_json(self):
        doc, records, payload = shipped()
        agreement.verify(
            json.loads(json.dumps(doc)), json.loads(json.dumps(records)), payload
        )


class TestVerifyBindsEachEntryToItsOwnEvidence:
    def test_a_swapped_symbol_across_entries_fails(self):
        doc, records, payload = shipped()
        a, b = doc["sources"]["decode"], doc["sources"]["merge"]
        a["symbol"], b["symbol"] = b["symbol"], a["symbol"]
        a["signature"], b["signature"] = b["signature"], a["signature"]
        _restamp(doc)
        with pytest.raises(HkpPackError, match="launch 'decode': .*symbol binding"):
            agreement.verify(doc, records, payload)

    def test_a_symbol_the_module_exports_under_the_wrong_launch_fails(self):
        doc, records, payload = shipped()
        # `merge` is given the primary's symbol and signature, which the object
        # exports; only the role recorded for `merge` can tell it apart.
        doc["sources"]["merge"]["symbol"] = PRIMARY
        doc["sources"]["merge"]["signature"] = doc["sources"]["decode"]["signature"]
        _restamp(doc)
        with pytest.raises(HkpPackError, match="launch 'merge': .*symbol binding"):
            agreement.verify(doc, records, payload)

    def test_a_signature_that_disagrees_with_the_object_fails(self):
        doc, records, payload = shipped()
        doc["sources"]["merge"]["signature"] = []
        _restamp(doc)
        with pytest.raises(HkpPackError, match="launch 'merge': signature does not"):
            agreement.verify(doc, records, payload)

    def test_a_signature_of_a_sibling_kernel_fails(self):
        doc, records, payload = shipped()
        doc["sources"]["merge"]["signature"] = doc["sources"]["decode"]["signature"]
        _restamp(doc)
        with pytest.raises(HkpPackError, match="launch 'merge': signature does not"):
            agreement.verify(doc, records, payload)

    def test_a_symbol_the_object_lacks_names_the_kernels_it_holds(self):
        doc, records, payload = shipped()
        doc["sources"]["merge"]["symbol"] = "decode_gone"
        _evidence(doc, "merge")["observations"]["companions"]["merge"] = "decode_gone"
        _restamp(doc)
        with pytest.raises(HkpPackError, match="decode_gone") as info:
            agreement.verify(doc, records, payload)
        assert PRIMARY in str(info.value) and MERGE in str(info.value)

    def test_an_entry_bound_to_another_launchs_code_object_fails(self):
        doc, records, payload = shipped(TWO_BUILDERS)
        payload["segment"], payload["reduce"] = payload["reduce"], payload["segment"]
        with pytest.raises(HkpPackError, match="launch 'segment': .*payload binding"):
            agreement.verify(doc, records, payload)

    def test_an_entry_whose_evidence_names_another_object_fails(self):
        doc, records, payload = shipped(TWO_BUILDERS)
        stolen = _evidence(doc, "reduce")["observations"]["code_object_sha256"]
        _evidence(doc, "segment")["observations"]["code_object_sha256"] = stolen
        with pytest.raises(HkpPackError, match="launch 'segment': .*code-object"):
            agreement.verify(doc, records, payload)

    def test_evidence_swapped_between_entries_fails(self):
        doc, records, payload = shipped(TWO_BUILDERS)
        evidence = doc["provenance"]["effective_spec"]
        evidence["segment"], evidence["reduce"] = (
            evidence["reduce"],
            evidence["segment"],
        )
        with pytest.raises(HkpPackError, match="launch 'reduce'|launch 'segment'"):
            agreement.verify(doc, records, payload)

    def test_authored_input_swapped_between_entries_fails(self):
        doc, records, payload = shipped(TWO_BUILDERS)
        launches = doc["provenance"]["sources"]
        launches["segment"]["builder"], launches["reduce"]["builder"] = (
            launches["reduce"]["builder"],
            launches["segment"]["builder"],
        )
        _restamp(doc)
        with pytest.raises(HkpPackError, match="authored-input binding"):
            agreement.verify(doc, records, payload)

    @pytest.mark.parametrize("where", ["sources", "evidence", "authored", "payload"])
    def test_a_dropped_entry_fails(self, where):
        doc, records, payload = shipped(TWO_BUILDERS)
        if where == "sources":
            del doc["sources"]["reduce"]
        elif where == "evidence":
            del doc["provenance"]["effective_spec"]["reduce"]
        elif where == "authored":
            del doc["provenance"]["sources"]["reduce"]
        else:
            del payload["reduce"]
        _restamp(doc)
        with pytest.raises(HkpPackError, match="do not match the UKD's launches"):
            agreement.verify(doc, records, payload)

    def test_a_kernel_left_unclaimed_by_a_dropped_entry_fails_even_when_restamped(self):
        """Entry, authored input and evidence all dropped together: only the claim
        check over the module's compiled kernels still sees `merge` unclaimed."""
        doc, records, payload = shipped({**ONE_MODULE, "other": ("vk-seg", None)})
        del doc["sources"]["merge"]
        del doc["provenance"]["sources"]["merge"]
        del doc["provenance"]["effective_spec"]["merge"]
        del payload["merge"]
        _restamp(doc)
        with pytest.raises(HkpPackError, match=r"claim the kernels \['\(primary\)'\]"):
            agreement.verify(doc, records, payload)

    def test_each_entry_is_compared_against_the_consumers_declaration(self):
        doc, records, payload = shipped(TWO_BUILDERS)
        observations = _evidence(doc, "reduce")["observations"]
        for entry in observations["requests"].values():
            entry["values"]["head_size"] = 128
        with pytest.raises(
            HkpPackError, match="launch 'reduce': compiled specialization disagrees"
        ):
            agreement.verify(doc, records, payload)

    def test_a_ukd_with_both_or_neither_kernel_key_is_refused(self):
        doc, records, payload = shipped()
        both = copy.deepcopy(doc)
        both["kernel_source"] = copy.deepcopy(next(iter(doc["sources"].values())))
        with pytest.raises(HkpPackError, match="exactly one of"):
            agreement.verify(both, records, payload)
        neither = copy.deepcopy(doc)
        del neither["sources"]
        with pytest.raises(HkpPackError, match="exactly one of"):
            agreement.verify(neither, records, payload)

    def test_a_single_launch_ukd_verifies_as_before(self):
        """A single `kernel_source` does not take the launch path or its prefix."""
        from test_specialization_agreement import shipped_ukd

        doc, records = shipped_ukd()
        doc["provenance"]["effective_spec"]["observations"]["symbol"] = "other"
        doc["provenance"]["effective_spec"]["descriptor_digest"] = (
            agreement.descriptor_binding(doc)
        )
        from test_specialization_agreement import PAYLOAD

        with pytest.raises(HkpPackError) as info:
            agreement.verify(doc, records, PAYLOAD)
        assert str(info.value) == "effective_spec symbol binding mismatch"


class TestEvidenceIsHeldToTheObject:
    """The compile observations are not covered by the descriptor digest, so the
    claims they support are also held to what the code object exports."""

    @staticmethod
    def _detached(launches=ONE_MODULE):
        """`shipped` through JSON, so each launch's evidence is its own copy."""
        doc, records, payload = shipped(launches)
        return json.loads(json.dumps(doc)), json.loads(json.dumps(records)), payload

    def test_the_object_exports_what_its_launches_claim(self):
        _doc, _records, payload = shipped()
        assert exported_kernels(payload["decode"], "t") == sorted([PRIMARY, MERGE])
        assert exported_kernels(shipped(TWO_BUILDERS)[2]["segment"], "t") == [SEGMENT]

    def test_a_dropped_launch_with_its_companion_evidence_popped_fails(self):
        doc, records, payload = self._detached()
        del doc["sources"]["merge"]
        del doc["provenance"]["sources"]["merge"]
        del doc["provenance"]["effective_spec"]["merge"]
        del payload["merge"]
        del doc["provenance"]["effective_spec"]["decode"]["observations"]["companions"]
        _restamp(doc)
        with pytest.raises(HkpPackError) as info:
            agreement.verify(doc, records, payload)
        assert f"claim the symbols ['{PRIMARY}']" in str(info.value)
        assert f"exports ['{MERGE}', '{PRIMARY}']" in str(info.value)

    def test_the_same_drop_without_popping_is_caught_by_the_evidence(self):
        doc, records, payload = self._detached()
        del doc["sources"]["merge"]
        del doc["provenance"]["sources"]["merge"]
        del doc["provenance"]["effective_spec"]["merge"]
        del payload["merge"]
        _restamp(doc)
        with pytest.raises(HkpPackError, match=r"claim the kernels \['\(primary\)'\]"):
            agreement.verify(doc, records, payload)

    def test_siblings_on_one_object_with_disagreeing_evidence_fail(self):
        doc, records, payload = self._detached()
        agreement.verify(doc, records, payload)
        del doc["provenance"]["effective_spec"]["decode"]["observations"]["companions"]
        _restamp(doc)
        with pytest.raises(HkpPackError, match="share one code object but carry"):
            agreement.verify(doc, records, payload)

    def test_launches_on_separate_objects_may_carry_different_evidence(self):
        doc, records, payload = self._detached(TWO_BUILDERS)
        assert (
            doc["provenance"]["effective_spec"]["segment"]["observations"]
            != doc["provenance"]["effective_spec"]["reduce"]["observations"]
        )
        agreement.verify(doc, records, payload)

    def test_a_single_launch_whose_evidence_names_companions_is_refused(self):
        """A plain kernel_source speaks for the primary only; a module's companion
        recorded in its evidence is a shipped kernel nothing claims."""
        from test_specialization_agreement import PAYLOAD, shipped_ukd

        doc, records = shipped_ukd()
        agreement.verify(doc, records, PAYLOAD)
        evidence = doc["provenance"]["effective_spec"]
        evidence["observations"]["companions"] = {"merge": "other_kernel"}
        with pytest.raises(HkpPackError) as info:
            agreement.verify(doc, records, PAYLOAD)
        assert "claim the kernels ['(primary)']" in str(info.value)
        assert "compile produced ['(primary)', 'merge']" in str(info.value)


@dataclasses.dataclass(frozen=True)
class LeanSpec:
    """A stage's spec that shows `head_size`, `dtype` and `causal` only: it has no
    `block_n` and no `resolved_use_exp2_fast`, which the consumer's declaration
    also binds."""

    head_size: int
    dtype: str = "bf16"
    causal: bool = True


@dataclasses.dataclass(frozen=True)
class MalformedSpec:
    """Has every bound attribute, but `resolved_use_exp2_fast` is not an accessor."""

    head_size: int = 64
    dtype: str = "bf16"
    causal: bool = True
    block_n: int = 64
    resolved_use_exp2_fast: bool = True


class _RaisingPropertySpec:
    """`block_n` exists, as a property whose body raises AttributeError."""

    head_size = 64
    dtype = "bf16"
    causal = True

    @property
    def block_n(self):
        return self.missing_helper

    def resolved_use_exp2_fast(self):
        return True


class _RaisingMethodSpec:
    """`resolved_use_exp2_fast` exists, as a method whose body raises
    AttributeError."""

    head_size = 64
    dtype = "bf16"
    causal = True
    block_n = 64

    def resolved_use_exp2_fast(self):
        return self.missing_helper


def _requests():
    request = agreement.observation_request(consumer(), FULL_KMD)
    return {agreement.digest(request): request}


def _observed(spec, tolerate_missing=True):
    return agreement.observe(
        spec,
        demo_builder,
        _requests(),
        agreement.OriginObserver(),
        tolerate_missing=tolerate_missing,
    )


def _entry_of(observations):
    (entry,) = observations["requests"].values()
    return entry


class TestFieldsALaunchDoesNotReport:
    """One declaration speaks for every launch of a `sources` UKD. A launch whose
    spec lacks a bound attribute does not report that field; each field must be
    reported by at least one launch and match wherever reported."""

    def test_a_missing_attribute_raises_unless_tolerated(self):
        with pytest.raises(HkpPackError, match="cannot observe block_n via"):
            _observed(LeanSpec(head_size=64), tolerate_missing=False)
        entry = _entry_of(_observed(LeanSpec(head_size=64)))
        assert entry["values"] == {"head_size": 64, "dtype": "BF16", "causal": 1}
        assert entry["unreported"] == ["block_n", "use_exp2_fast"]
        assert set(entry["accessors"]) == {"head_size", "dtype", "causal"}

    def test_evidence_in_which_everything_was_reported_keeps_its_shape(self):
        entry = _entry_of(_observed(DemoSpec(head_size=64)))
        assert set(entry) == {"values", "accessors"}

    def test_tolerance_covers_an_absent_attribute_and_nothing_else(self):
        """An attribute that exists but is not an accessor is a defect, not a field
        this stage lacks."""
        with pytest.raises(HkpPackError, match="not a bound effective accessor"):
            _observed(MalformedSpec())

    @pytest.mark.parametrize(
        "spec, attribute",
        [
            (_RaisingPropertySpec(), "block_n"),
            (_RaisingMethodSpec(), "use_exp2_fast"),
        ],
        ids=["property", "method"],
    )
    def test_an_attribute_that_raises_while_observed_is_not_unreported(
        self, spec, attribute
    ):
        """An AttributeError from an attribute the spec does have is a defect in
        it; only an attribute the spec lacks is a field it does not report."""
        with pytest.raises(HkpPackError, match=f"cannot observe {attribute} via"):
            _observed(spec)

    def test_a_field_one_launch_reports_and_matches_passes(self):
        launches = {
            "zero": _observed(DemoSpec(head_size=64)),
            "one": _observed(LeanSpec(head_size=64)),
        }
        agreement.compare_launches(consumer(), FULL_KMD, METADATA, launches)

    def test_a_launch_reporting_a_value_the_metadata_does_not_hold_fails(self):
        launches = {
            "zero": _observed(DemoSpec(head_size=64, block_n=16)),
            "one": _observed(LeanSpec(head_size=64)),
        }
        with pytest.raises(
            HkpPackError,
            match="launch 'zero': compiled specialization disagrees .*block_n "
            "observed=16, metadata=64",
        ):
            agreement.compare_launches(consumer(), FULL_KMD, METADATA, launches)

    def test_a_field_no_launch_reports_fails_naming_it_and_the_launches(self):
        launches = {
            "zero": _observed(LeanSpec(head_size=64)),
            "one": _observed(LeanSpec(head_size=64)),
        }
        with pytest.raises(HkpPackError) as info:
            agreement.compare_launches(consumer(), FULL_KMD, METADATA, launches)
        assert "field(s) ['block_n', 'use_exp2_fast'] reported by no launch" in str(
            info.value
        )
        assert "checked launches ['one', 'zero']" in str(info.value)

    def test_no_launches_at_all_fails(self):
        with pytest.raises(HkpPackError, match="reported by no launch"):
            agreement.compare_launches(consumer(), FULL_KMD, METADATA, {})

    def test_a_field_reported_on_every_launch_must_match_on_each(self):
        launches = {
            "zero": _observed(DemoSpec(head_size=64)),
            "one": _observed(LeanSpec(head_size=128)),
        }
        with pytest.raises(
            HkpPackError, match="launch 'one': .*head_size observed=128, metadata=64"
        ):
            agreement.compare_launches(consumer(), FULL_KMD, METADATA, launches)

    def test_a_binding_misspelled_for_every_launch_fails(self):
        typo = consumer(
            metadata_fields=["head_sizee"],
            matcher_only_fields=[
                "dtype",
                "causal",
                "block_n",
                "use_exp2_fast",
                "family",
                "head_size",
            ],
            bindings={"head_sizee": {"field": "head_sizee"}},
            vocabulary={},
        )
        kmd = {
            **FULL_KMD,
            "fields": [*FULL_KMD["fields"], {"name": "head_sizee", "type": "int"}],
        }
        request = agreement.observation_request(typo, kmd)
        requests = {agreement.digest(request): request}
        launches = {
            name: agreement.observe(
                spec,
                demo_builder,
                requests,
                agreement.OriginObserver(),
                tolerate_missing=True,
            )
            for name, spec in {
                "zero": DemoSpec(head_size=64),
                "one": LeanSpec(head_size=64),
            }.items()
        }
        with pytest.raises(HkpPackError, match=r"\['head_sizee'\] reported by no"):
            agreement.compare_launches(
                typo, kmd, {**METADATA, "head_sizee": 64}, launches
            )

    @pytest.mark.parametrize(
        "mutate, diagnostic",
        [
            (lambda e: e["values"].pop("head_size"), "evidence accounts for"),
            (lambda e: e.update(unreported=["head_size"]), "evidence accounts for"),
            (
                lambda e: e.update(unreported=[*e.get("unreported", []), "block_n"]),
                "evidence accounts for",
            ),
        ],
        ids=["report-dropped", "reported-and-unreported", "unreported-twice"],
    )
    def test_a_launch_must_account_for_every_declared_field_exactly_once(
        self, mutate, diagnostic
    ):
        lean = _observed(LeanSpec(head_size=64))
        mutate(_entry_of(lean))
        launches = {"zero": _observed(DemoSpec(head_size=64)), "one": lean}
        with pytest.raises(HkpPackError, match=f"launch 'one': {diagnostic}"):
            agreement.compare_launches(consumer(), FULL_KMD, METADATA, launches)

    def test_a_launch_without_the_consumers_observation_fails_naming_it(self):
        other = _observed(LeanSpec(head_size=64))
        other["requests"] = {}
        launches = {"zero": _observed(DemoSpec(head_size=64)), "one": other}
        with pytest.raises(HkpPackError, match="launch 'one': compile result lacks"):
            agreement.compare_launches(consumer(), FULL_KMD, METADATA, launches)

    def test_a_single_launch_must_still_report_every_declared_field(self):
        """`compare`, the single-launch rule, is unchanged: an unreported field is
        a failure naming it."""
        with pytest.raises(HkpPackError, match="cannot observe block_n via"):
            agreement.compare(consumer(), FULL_KMD, METADATA, _observed(LeanSpec(64)))
        agreement.compare(consumer(), FULL_KMD, METADATA, _observed(DemoSpec(64)))

    def test_verify_re_derives_the_verdict_from_the_stored_evidence(self):
        """Through `shipped`: stage zero reports `block_n`, stage one does not."""
        doc, records, payload = shipped(TWO_BUILDERS)
        agreement.verify(doc, records, payload)
        evidence = doc["provenance"]["effective_spec"]
        lean = _entry_of(evidence["reduce"]["observations"])
        lean["unreported"] = sorted(lean["values"])
        lean["values"] = {}
        # Nothing is reported by `reduce`, and `segment` still reports everything.
        agreement.verify(doc, records, payload)
        _entry_of(evidence["segment"]["observations"])["values"]["head_size"] = 128
        with pytest.raises(HkpPackError, match="launch 'segment': compiled"):
            agreement.verify(doc, records, payload)


class TestSameKernelUnderTwoLaunches:
    def test_two_compile_groups_with_identical_bytes_verify(self, monkeypatch):
        """Two compiles (distinct variant keys, toc_keys and builders) can yield
        byte-identical code objects; they are two compiles, not one."""
        monkeypatch.setitem(MODULES, "vk-x", {"primary": PRIMARY, "companions": {}})
        monkeypatch.setitem(MODULES, "vk-y", {"primary": PRIMARY, "companions": {}})
        doc, records, payload = shipped({"a": ("vk-x", None), "b": ("vk-y", None)})
        assert payload["a"] == payload["b"]
        sa, sb = doc["sources"]["a"], doc["sources"]["b"]
        assert sa["toc_key"] != sb["toc_key"]
        sources_provenance = doc["provenance"]["sources"]
        assert sources_provenance["a"]["builder"] != sources_provenance["b"]["builder"]
        agreement.verify(doc, records, payload)

    def test_equal_bytes_with_different_observations_are_still_two_compiles(
        self, monkeypatch
    ):
        """Compiles group by (library, toc_key), not by their bytes: two builders
        may emit identical objects and each keep its own evidence."""
        monkeypatch.setitem(MODULES, "vk-x", {"primary": PRIMARY, "companions": {}})
        monkeypatch.setitem(MODULES, "vk-y", {"primary": PRIMARY, "companions": {}})
        doc, records, payload = shipped({"a": ("vk-x", None), "b": ("vk-y", None)})
        assert payload["a"] == payload["b"]
        evidence = doc["provenance"]["effective_spec"]
        evidence["b"]["observations"]["producer"]["builder"]["sha256"] = "f" * 64
        assert evidence["a"]["observations"] != evidence["b"]["observations"]
        agreement.verify(doc, records, payload)

    def test_one_compile_claimed_by_two_launches_verifies(self):
        doc, records, payload = shipped(
            {
                "first": ("vk-a", None),
                "second": ("vk-a", None),
                "merge": ("vk-a", "merge"),
            }
        )
        assert doc["sources"]["first"]["symbol"] == doc["sources"]["second"]["symbol"]
        agreement.verify(doc, records, payload)

    def test_a_companion_claimed_twice_with_the_primary_unclaimed_fails(self):
        doc, records, payload = shipped(
            {"one": ("vk-a", "merge"), "two": ("vk-a", "merge")}
        )
        with pytest.raises(HkpPackError, match=r"claim the kernels \['merge'\]"):
            agreement.verify(doc, records, payload)


class TestLaunchOrderCarriesNoMeaning:
    """The order of `sources` keys selects nothing: the verdict and the shipped
    bytes are the same for every order."""

    LAUNCHES = {
        "first": ("vk-a", None),
        "second": ("vk-a", None),
        "merge": ("vk-a", "merge"),
        "seg": ("vk-seg", None),
    }

    @staticmethod
    def _orders():
        return list(itertools.permutations(TestLaunchOrderCarriesNoMeaning.LAUNCHES))

    def _shipped(self, order):
        return shipped({name: self.LAUNCHES[name] for name in order})

    def test_every_order_ships_the_same_document_and_verifies(self):
        reference = None
        for order in self._orders():
            doc, records, payload = self._shipped(order)
            agreement.verify(doc, records, payload)
            canonical = json.dumps(doc, sort_keys=True)
            reference = reference or canonical
            assert canonical == reference

    def test_every_order_refuses_a_launch_whose_evidence_departs_from_its_siblings(
        self,
    ):
        for order in self._orders():
            doc, records, payload = self._shipped(order)
            doc = json.loads(json.dumps(doc))
            evidence = doc["provenance"]["effective_spec"]
            evidence["merge"]["observations"]["producer"]["builder"]["sha256"] = (
                "f" * 64
            )
            with pytest.raises(HkpPackError, match="share one code object but carry"):
                agreement.verify(doc, records, payload)

    def test_the_two_launch_refusal_names_the_pair_in_sorted_order(self):
        for order in itertools.permutations(["decode", "merge"]):
            doc, records, payload = shipped({n: ONE_MODULE[n] for n in order})
            doc = json.loads(json.dumps(doc))
            evidence = doc["provenance"]["effective_spec"]
            evidence["merge"]["observations"]["producer"]["builder"]["sha256"] = (
                "f" * 64
            )
            with pytest.raises(HkpPackError) as info:
                agreement.verify(doc, records, payload)
            assert "launches ['decode', 'merge'] share one code object" in str(
                info.value
            )

    def test_the_packer_compiles_and_claims_the_same_in_every_order(self, monkeypatch):
        compiled = []

        def fake_compile(
            source, builder, spec, arch, _dir, _req, _origins, target, **_
        ):
            compiled.append((source, builder, target))
            observations = {
                "arch": arch,
                "compile_target": arch,
                "symbol": PRIMARY,
            }
            if builder == "b1":
                observations["companions"] = {"merge": MERGE}
            return f"{builder}.co", PRIMARY, observations

        monkeypatch.setattr(pipeline, "compile_rocke_variant", fake_compile)

        def entry(builder, role=None):
            ks = {"kind": "rocke", "source": "k/m.py", "builder": builder, "spec": {}}
            return {**ks, "role": role} if role else ks

        authored = {
            "first": entry("b1"),
            "second": entry("b1"),
            "merge": entry("b1", "merge"),
            "seg": entry("b2"),
        }
        results = []
        for order in itertools.permutations(authored):
            compiled.clear()
            ukd = {"id": "ukd-demo", "sources": {n: dict(authored[n]) for n in order}}
            _vk, _symbol, fields = pipeline._compile_sources_ukd(
                ukd,
                "UKD 'ukd-demo'",
                ".",
                ".",
                ARCH,
                "hipcc",
                "dir",
                {},
                {},
                {},
                {},
                {},
                None,
            )
            launches = {
                name: (launch.variant_key, launch.symbol, launch.role)
                for name, launch in fields["launches"].items()
            }
            results.append((sorted(compiled), sorted(launches.items())))
        assert all(result == results[0] for result in results)
        assert [builder for _s, builder, _t in results[0][0]] == ["b1", "b2"]


class TestClaimCheck:
    OBSERVED = {"symbol": PRIMARY, "companions": {"merge": MERGE}}

    def test_the_primary_and_every_companion_claimed_once_matches(self):
        sources.check_claims("W", "b", ["a", "m"], [None, "merge"], self.OBSERVED)

    @pytest.mark.parametrize(
        "roles",
        [
            [None, None, "merge"],
            [None, None, "merge", "merge"],
        ],
        ids=["primary-twice", "both-twice"],
    )
    def test_one_kernel_under_two_launch_names_is_one_claim(self, roles):
        names = [f"n{i}" for i in range(len(roles))]
        sources.check_claims("W", "b", names, roles, self.OBSERVED)

    @pytest.mark.parametrize(
        "roles, claimed",
        [
            ([None], r"\['\(primary\)'\]"),
            (["merge"], r"\['merge'\]"),
            ([None, "reduce"], r"\['\(primary\)', 'reduce'\]"),
            (["merge", "merge"], r"\['merge'\]"),
            ([], r"\[\]"),
        ],
        ids=[
            "companion-unclaimed",
            "primary-unclaimed",
            "renamed-role",
            "companion-twice",
            "nothing-claimed",
        ],
    )
    def test_a_claim_set_other_than_the_compiled_kernels_fails_naming_both(
        self, roles, claimed
    ):
        names = [f"n{i}" for i in range(len(roles))]
        with pytest.raises(HkpPackError) as info:
            sources.check_claims("W", "b", names, roles, self.OBSERVED)
        message = str(info.value)
        assert re.search(rf"claim the kernels {claimed}", message)
        assert re.search(r"compile produced \['\(primary\)', 'merge'\]", message)

    def test_a_distinct_claim_set_is_sorted_and_deduplicated(self):
        assert sources.claimed_roles([None, "merge", None]) == ["(primary)", "merge"]

    def test_a_single_kernel_compile_has_only_a_primary(self):
        sources.check_claims("W", "b", ["a"], [None], {"symbol": PRIMARY})
        with pytest.raises(HkpPackError, match=r"produced \['\(primary\)'\]"):
            sources.check_claims("W", "b", ["a"], ["merge"], {"symbol": PRIMARY})


def _hip_entry(**overrides):
    ks = {"kind": "hip", "source": "k.cpp", "entry": "e", "build": {}}
    ks.update(overrides)
    return ks


def _hsaco_entry(symbol="s", **overrides):
    ks = {"kind": "hsaco", "file": "k.co", "symbol": symbol}
    ks.update(overrides)
    return ks


def _without(entry, field):
    return {k: v for k, v in entry.items() if k != field}


def _rocke_entry(**overrides):
    ks = {"kind": "rocke", "source": "k/m.py", "builder": "b", "spec": {}}
    ks.update(overrides)
    return ks


def _sources_ukd(entries=None, **top):
    doc = {
        "id": "k",
        "name": "k",
        "sources": (
            entries
            if entries is not None
            else {"k0": _rocke_entry(), "k1": _rocke_entry(role="merge")}
        ),
        "metadata": {},
        "priority": 0,
    }
    doc.update(top)
    return doc


def _packed_entry(symbol, **overrides):
    ks = {
        "kind": "kpack",
        "library": "kpack/x.kpack",
        "toc_key": "vk",
        "symbol": symbol,
        "sha256": "0" * 64,
        "signature": [],
    }
    ks.update(overrides)
    return ks


def _packed_ukd(version="1.1", entries=None):
    return _sources_ukd(
        (
            entries
            if entries is not None
            else {"k0": _packed_entry(PRIMARY), "k1": _packed_entry(MERGE)}
        ),
        version=version,
    )


class TestAuthoredValidation:
    def test_two_rocke_entries_with_and_without_a_role_are_accepted(self):
        _validate_ukd_fields(_sources_ukd(), "UKD 'k'")

    def test_both_kernel_keys_are_refused(self):
        doc = _sources_ukd(kernel_source=_rocke_entry())
        with pytest.raises(HkpPackError, match="both 'kernel_source' and 'sources'"):
            _validate_ukd_fields(doc, "UKD 'k'")

    def test_neither_kernel_key_is_refused(self):
        doc = _sources_ukd()
        del doc["sources"]
        with pytest.raises(
            HkpPackError, match="missing required field 'kernel_source'"
        ):
            _validate_ukd_fields(doc, "UKD 'k'")

    @pytest.mark.parametrize(
        "entries",
        [{}, {"only": _rocke_entry()}, [], None, "k0"],
        ids=["empty", "single", "list", "null", "string"],
    )
    def test_fewer_than_two_launches_or_a_non_object_is_refused(self, entries):
        doc = _sources_ukd(entries={} if entries is None else entries)
        if entries is None:
            doc["sources"] = None
        with pytest.raises(HkpPackError, match="at least two launches"):
            _validate_ukd_fields(doc, "UKD 'k'")

    @pytest.mark.parametrize("name", ["K0", "0k", "", "a-b", "a b", "k0\n", "ä"])
    def test_a_launch_name_must_be_a_lowercase_identifier(self, name):
        doc = _sources_ukd({name: _rocke_entry(), "k1": _rocke_entry()})
        with pytest.raises(HkpPackError, match="invalid launch name"):
            _validate_ukd_fields(doc, "UKD 'k'")

    @pytest.mark.parametrize("name", ["k0", "a_b", "decode2", "x"])
    def test_a_lowercase_identifier_launch_name_is_accepted(self, name):
        _validate_ukd_fields(
            _sources_ukd({name: _rocke_entry(), "other": _rocke_entry()}), "UKD 'k'"
        )

    def test_an_embedded_source_entry_is_refused_at_validation(self):
        entry = {"kind": "embedded_source", "source_file": "k.cpp", "entry_point": "e"}
        doc = _sources_ukd({"k0": _rocke_entry(), "k1": entry}, arch=["gfx942"])
        with pytest.raises(HkpPackError, match="'k1' has kind 'embedded_source'"):
            _validate_ukd_fields(doc, "UKD 'k'")

    @pytest.mark.parametrize(
        "kind, says",
        [
            ("embedded_source", "not packed into a code object"),
            ("kpack", "not pack input"),
        ],
    )
    def test_each_refused_kind_says_why(self, kind, says):
        message = str(sources.refuse_authored_kind("W", "k1", kind))
        assert f"'k1' has kind '{kind}'" in message and says in message

    def test_hip_and_hsaco_entries_may_be_authored(self):
        _validate_ukd_fields(
            _sources_ukd({"k0": _hip_entry(), "k1": _hip_entry()}), "U"
        )
        _validate_ukd_fields(
            _sources_ukd(
                {"k0": _hsaco_entry(), "k1": _hsaco_entry("s2")}, arch=["gfx942"]
            ),
            "U",
        )

    @pytest.mark.parametrize("make", [_hip_entry, _hsaco_entry], ids=["hip", "hsaco"])
    def test_role_is_refused_on_a_hip_or_hsaco_entry(self, make):
        doc = _sources_ukd({"k0": make(), "k1": make(role="merge")}, arch=["gfx942"])
        with pytest.raises(HkpPackError, match="'k1' carries 'role'"):
            _validate_ukd_fields(doc, "UKD 'k'")

    @pytest.mark.parametrize(
        "entry, match",
        [
            (_hip_entry(build={"flags": [1]}), "invalid build"),
            (_hip_entry(build={"flags": ["-fuse-cuid=hash"]}), "-fuse-cuid"),
            (_without(_hip_entry(), "build"), "invalid build \\(absent\\)"),
            (_without(_hip_entry(), "entry"), "missing required field 'entry'"),
            (_without(_hsaco_entry(), "file"), "missing required field 'file'"),
            (_hsaco_entry(symbol=""), "'symbol' must be a non-empty string"),
            (_hsaco_entry(symbol="s\u00ff"), "must be ASCII"),
        ],
        ids=[
            "flags-not-strings",
            "reserved-flag",
            "build-absent",
            "entry-absent",
            "file-absent",
            "symbol-empty",
            "symbol-non-ascii",
        ],
    )
    def test_a_hip_or_hsaco_entry_is_checked_like_a_single_launch(self, entry, match):
        other = _hip_entry() if entry["kind"] == "hip" else _hsaco_entry("s2")
        doc = _sources_ukd({"k0": other, "k1": entry}, arch=["gfx942"])
        with pytest.raises(HkpPackError, match=match) as info:
            _validate_ukd_fields(doc, "UKD 'k'")
        assert "sources 'k1'" in str(info.value)

    @pytest.mark.parametrize("arch", [None, []], ids=["absent", "empty"])
    def test_hsaco_entries_need_a_non_empty_ukd_arch(self, arch):
        doc = _sources_ukd({"k0": _hsaco_entry(), "k1": _hsaco_entry("s2")})
        if arch is not None:
            doc["arch"] = arch
        with pytest.raises(HkpPackError, match="hsaco requires a non-empty 'arch'"):
            _validate_ukd_fields(doc, "UKD 'k'")

    @pytest.mark.parametrize(
        "entries",
        [
            {"k0": _hip_entry(), "k1": _hsaco_entry()},
            {"k0": _rocke_entry(), "k1": _hip_entry()},
            {"k0": _rocke_entry(), "k1": _hsaco_entry()},
        ],
        ids=["hip-hsaco", "rocke-hip", "rocke-hsaco"],
    )
    def test_entries_of_two_kinds_cannot_mix(self, entries):
        doc = _sources_ukd(entries, arch=["gfx942"])
        with pytest.raises(HkpPackError, match="mix kernel kinds"):
            _validate_ukd_fields(doc, "UKD 'k'")

    def test_an_unknown_kind_and_a_missing_kind_are_refused(self):
        with pytest.raises(
            HkpPackError,
            match=(
                "'k1' has kind 'flydsl', which is not a sources entry kind "
                r"\(expected 'rocke', 'hip' or 'hsaco', or 'kpack' once packed\)"
            ),
        ):
            _validate_ukd_fields(
                _sources_ukd({"k0": _rocke_entry(), "k1": {"kind": "flydsl"}}),
                "UKD 'k'",
            )
        with pytest.raises(HkpPackError, match="'k1' missing 'kind'"):
            _validate_ukd_fields(
                _sources_ukd({"k0": _rocke_entry(), "k1": {"source": "x"}}), "UKD 'k'"
            )

    @pytest.mark.parametrize("missing", ["source", "builder", "spec"])
    def test_a_rocke_entry_needs_source_builder_and_spec(self, missing):
        entry = _rocke_entry()
        del entry[missing]
        with pytest.raises(HkpPackError, match=f"missing required field '{missing}'"):
            _validate_ukd_fields(_sources_ukd({"k0": entry, "k1": _rocke_entry()}), "U")

    def test_an_entrys_spec_and_target_are_validated_like_a_single_launch(self):
        with pytest.raises(HkpPackError, match="invalid spec"):
            _validate_ukd_fields(
                _sources_ukd({"k0": _rocke_entry(spec=3), "k1": _rocke_entry()}), "U"
            )
        with pytest.raises(HkpPackError, match="invalid target"):
            _validate_ukd_fields(
                _sources_ukd({"k0": _rocke_entry(target="nope"), "k1": _rocke_entry()}),
                "U",
            )

    @pytest.mark.parametrize("role", ["Merge", "1x", "", "a-b", 3, None, ["merge"]])
    def test_a_role_must_be_a_lowercase_identifier_string(self, role):
        doc = _sources_ukd({"k0": _rocke_entry(), "k1": _rocke_entry(role=role)})
        with pytest.raises(HkpPackError, match="invalid role"):
            _validate_ukd_fields(doc, "UKD 'k'")

    def test_a_single_kernel_source_still_validates_and_refuses_a_role_free_miss(self):
        ukd = {
            "id": "k",
            "name": "k",
            "kernel_source": _rocke_entry(),
            "metadata": {},
            "priority": 0,
        }
        _validate_ukd_fields(ukd, "UKD 'k'")


class TestPackedValidation:
    @pytest.mark.parametrize("version", ["1.1", "1.2", "2.0"])
    def test_a_packed_sources_ukd_at_1_1_or_later_is_accepted(self, version):
        _validate_ukd_fields(_packed_ukd(version), "UKD 'k'")

    @pytest.mark.parametrize("version", ["1.0", "0.9", None])
    def test_a_packed_sources_ukd_below_1_1_is_refused(self, version):
        doc = _packed_ukd(version)
        if version is None:
            del doc["version"]
        with pytest.raises(HkpPackError, match="needs 1.1 or later"):
            _validate_ukd_fields(doc, "UKD 'k'")

    def test_two_launch_names_for_one_kernel_are_accepted(self):
        doc = _packed_ukd(
            entries={"k0": _packed_entry(PRIMARY), "k1": _packed_entry(PRIMARY)}
        )
        _validate_ukd_fields(doc, "UKD 'k'")

    def test_entries_sharing_a_module_but_not_a_symbol_are_accepted(self):
        _validate_ukd_fields(_packed_ukd(), "UKD 'k'")

    def test_a_packed_entry_may_not_carry_a_role(self):
        doc = _packed_ukd(
            entries={
                "k0": _packed_entry(PRIMARY),
                "k1": _packed_entry(MERGE, role="merge"),
            }
        )
        with pytest.raises(HkpPackError, match="'role', which is authored-only"):
            _validate_ukd_fields(doc, "UKD 'k'")

    def test_a_packed_entry_needs_the_full_kernel_source(self):
        entry = _packed_entry(MERGE)
        del entry["signature"]
        with pytest.raises(HkpPackError, match="missing required field 'signature'"):
            _validate_ukd_fields(
                _packed_ukd(entries={"k0": _packed_entry(PRIMARY), "k1": entry}), "U"
            )

    def test_authored_and_packed_entries_cannot_mix(self):
        doc = _packed_ukd(entries={"k0": _packed_entry(PRIMARY), "k1": _rocke_entry()})
        with pytest.raises(HkpPackError, match="mix kernel kinds"):
            _validate_ukd_fields(doc, "UKD 'k'")

    def test_a_standalone_packed_ukd_below_1_1_is_refused_on_load(self, tmp_path):
        """The version reaches the check on the standalone path too."""
        ukd = _packed_ukd("1.0")
        ukd["arch"] = [ARCH]
        _write_json(tmp_path, "k.ukd.json", ukd)
        with pytest.raises(HkpPackError, match="needs 1.1 or later"):
            load_flat_input(tmp_path, GENERIC_TARGETS, log=_silent)


class TestEveryReaderOfAUkdsKernel:
    """Each reader given a `sources` UKD either handles it or refuses; none reads
    the absent `kernel_source` key and skips."""

    def test_kernel_sources_names_every_launch_and_refuses_both_or_neither(self):
        assert set(sources.kernel_sources(_sources_ukd())) == {"k0", "k1"}
        assert sources.kernel_sources({"kernel_source": {"kind": "x"}}) == {
            None: {"kind": "x"}
        }
        for doc in (
            _sources_ukd(kernel_source={"kind": "rocke"}),
            {"id": "k"},
            _sources_ukd(entries={}),
        ):
            with pytest.raises(HkpPackError):
                sources.kernel_sources(doc)

    def test_kind_of_is_the_shared_kind_and_refuses_a_mixed_ukd(self):
        assert sources.kind_of(_sources_ukd()) == "rocke"
        assert sources.kind_of(_packed_ukd()) == "kpack"
        with pytest.raises(HkpPackError, match="mix kernel_source kinds"):
            sources.kind_of(
                _sources_ukd({"k0": _rocke_entry(), "k1": _packed_entry(PRIMARY)})
            )

    def test_the_variant_key_of_a_sources_ukd_is_refused_not_keyed_by_none(self):
        with pytest.raises(HkpPackError, match="one variant per launch"):
            pipeline._variant_key_for(_sources_ukd(), ".")

    def test_launch_variants_yield_one_single_launch_document_per_entry(self):
        ukd = _sources_ukd(
            {
                "k0": _rocke_entry(),
                "k1": _rocke_entry(role="merge"),
                "k2": _rocke_entry(builder="other"),
            }
        )
        launches = list(pipeline._launch_variants(ukd, "."))
        keys = [vk for vk, _name, _doc in launches]
        assert [name for _vk, name, _doc in launches] == ["k0", "k1", "k2"]
        # `role` is no part of what is compiled: k0 and k1 key one compile.
        assert keys[0] == keys[1] != keys[2]
        for _vk, _name, doc in launches:
            assert "sources" not in doc and doc["id"] == "k"
            assert "role" not in doc["kernel_source"]
        assert ukd["sources"]["k1"]["role"] == "merge"

    def test_a_kpack_entry_reaching_the_compile_is_refused(self):
        ukd = _sources_ukd({"k0": _rocke_entry(), "k1": _packed_entry(PRIMARY)})
        with pytest.raises(HkpPackError) as info:
            pipeline._compile_ukd_variant(
                ukd, "W", ".", ".", ARCH, "hipcc", ".", {}, {}, {}, {}, {}, None
            )
        assert "'k1' has kind 'kpack'" in str(info.value)

    def test_the_authored_extras_exclude_sources(self):
        assert "sources" not in pipeline._ukd_extra(_sources_ukd(version="1.0"))
        assert pipeline._ukd_extra(_sources_ukd(version="1.0")) == {"version": "1.0"}

    def test_a_sources_ukd_is_not_a_passthrough(self):
        assert not pipeline._is_passthrough(_sources_ukd())
        assert pipeline._is_passthrough(
            {
                "kernel_source": {
                    "kind": "embedded_source",
                    "source_file": "k.cpp",
                    "entry_point": "e",
                }
            }
        )

    def test_a_rocke_sources_ukd_under_a_generic_arch_is_refused(self, tmp_path):
        """The generic-target rule reads the kind; a `sources` UKD has no
        `kernel_source` to read it from, which must not exempt it."""
        generic = GENERIC_TARGETS.names()[0]
        ukd = _sources_ukd(arch=[generic], version="1.0")
        _write_json(tmp_path, "k.kdp.json", _kdp("kdp-k", [generic], [ukd]))
        with pytest.raises(HkpPackError, match="rocKE does not support generic"):
            load_flat_input(tmp_path, GENERIC_TARGETS, log=_silent)

    def test_a_disabled_kind_prunes_a_sources_ukd(self, tmp_path):
        standalone = _sources_ukd(version="1.0")
        standalone["id"] = "solo"
        _write_json(tmp_path, "solo.ukd.json", standalone)
        _write_json(
            tmp_path,
            "k.kdp.json",
            _kdp("kdp-k", [], [_sources_ukd(version="1.0")]),
        )
        pruned = load_flat_input(
            tmp_path, GENERIC_TARGETS, log=_silent, disabled_kinds=("rocke",)
        )
        assert pruned.kdps() == [] and pruned.ukds() == []
        kept = load_flat_input(
            tmp_path, GENERIC_TARGETS, log=_silent, disabled_kinds=("hip",)
        )
        assert len(kept.kdps()) == 1 and len(kept.ukds()) == 1


def _kernel_doc(name, entries, **extra):
    doc = {"name": name, "metadata": {"head_size": 64}, "sources": entries}
    doc.update(extra)
    return doc


class TestDeskCheckReaders:
    def test_the_launches_of_a_sources_kernel_are_each_read(self):
        kernel = _kernel_doc("k", {"a": {"toc_key": "t"}, "b": {"toc_key": "t"}})
        assert set(desk_check._launch_sources(kernel)) == {"a", "b"}
        assert desk_check._launch_sources({"name": "x"}) == {None: {}}
        with pytest.raises(HkpPackError, match="exactly one of"):
            desk_check._launch_sources({"sources": {"a": {}}, "kernel_source": {}})

    def test_toc_key_and_symbol_applicability_sees_a_sources_kernel(self):
        """A packed `sources` kernel carries its toc_key in each entry; reading only
        `kernel_source` would report a pre-pack tree and skip the check."""
        kernels = [_kernel_doc("k", {"a": {"toc_key": "t"}, "b": {"toc_key": "t"}})]
        assert desk_check._field_applicable(kernels, "toc_key")
        assert not desk_check._field_applicable(kernels, "symbol")
        assert not desk_check._field_applicable(
            [_kernel_doc("k", {"a": {}, "b": {}})], "toc_key"
        )

    def test_launches_sharing_a_toc_key_are_one_variant_and_two_kernels_collide(self):
        siblings = _kernel_doc("k", {"a": {"toc_key": "t"}, "b": {"toc_key": "t"}})
        other = _kernel_doc("o", {"a": {"toc_key": "u"}, "b": {"toc_key": "v"}})
        # Not a collision: one kernel's siblings share a module.
        assert desk_check.toc_key_uniqueness([siblings, other]) == (2, 2)
        # A second kernel on exactly the same objects is.
        twin = copy.deepcopy(siblings)
        assert desk_check.toc_key_uniqueness([siblings, twin]) == (1, 2)
        # A single kernel_source kernel counts as it always did.
        plain = [
            {"kernel_source": {"toc_key": "x"}},
            {"kernel_source": {"toc_key": "y"}},
        ]
        assert desk_check.toc_key_uniqueness(plain) == (2, 2)

    def test_a_toc_key_shared_through_one_launch_collides_across_kernels(self):
        two_launch = _kernel_doc("k", {"a": {"toc_key": "t"}, "b": {"toc_key": "u"}})
        reuse = {"name": "o", "kernel_source": {"toc_key": "u"}}
        disjoint = {"name": "o", "kernel_source": {"toc_key": "v"}}
        assert desk_check.toc_key_uniqueness([two_launch, reuse]) == (1, 2)
        assert desk_check.toc_key_uniqueness([two_launch, disjoint]) == (2, 2)
        shared = _kernel_doc("k", {"a": {"toc_key": "t"}, "b": {"toc_key": "t"}})
        assert desk_check.toc_key_uniqueness([shared]) == (1, 1)

    def test_symbols_of_a_sources_kernel_are_counted_per_kernel(self):
        one = _kernel_doc("k", {"a": {"symbol": "s1"}, "b": {"symbol": "s2"}})
        two = _kernel_doc("o", {"a": {"symbol": "s1"}, "b": {"symbol": "s3"}})
        assert desk_check.symbol_distinctness([one, two]) == (2, 2)
        assert desk_check.symbol_distinctness([one, copy.deepcopy(one)]) == (1, 2)

    def test_the_spec_of_each_launch_is_read_from_its_own_location(self):
        authored = _kernel_doc(
            "k",
            {"a": {"spec": {"head_size": 64}}, "b": {"spec": {"head_size": 32}}},
        )
        assert desk_check._authored_specs(authored) == [
            ("a", {"head_size": 64}),
            ("b", {"head_size": 32}),
        ]
        packed = _kernel_doc(
            "k",
            {"a": {"toc_key": "t"}, "b": {"toc_key": "t"}},
            provenance={
                "sources": {
                    "a": {"spec": {"head_size": 64}},
                    "b": {"spec": {"head_size": 32}},
                }
            },
        )
        assert [s for _n, s in desk_check._authored_specs(packed)] == [
            {"head_size": 64},
            {"head_size": 32},
        ]

    @pytest.mark.parametrize(
        "provenance",
        [{}, {"sources": {}}, {"sources": {"a": {"spec": {"head_size": 64}}}}],
        ids=["no-provenance", "empty", "one-launch-missing"],
    )
    def test_a_launch_without_a_spec_is_could_not_check_not_agreement(self, provenance):
        kernel = _kernel_doc(
            "k",
            {"a": {"toc_key": "t"}, "b": {"toc_key": "t"}},
            provenance=provenance,
        )
        with pytest.raises(desk_check.DeskCheckNoSpecFound, match="launch"):
            desk_check._authored_specs(kernel)
        with pytest.raises(desk_check.DeskCheckNoSpecFound):
            desk_check.metadata_spec_drift([kernel], ("head_size",))

    def test_drift_is_found_in_any_launch_and_names_it(self):
        kernel = _kernel_doc(
            "k",
            {"a": {"spec": {"head_size": 64}}, "b": {"spec": {"head_size": 32}}},
        )
        assert desk_check.metadata_spec_drift([kernel], ("head_size",)) == [
            ("k[b]", "head_size")
        ]
        agreeing = _kernel_doc(
            "k", {"a": {"spec": {"head_size": 64}}, "b": {"spec": {"head_size": 64}}}
        )
        assert desk_check.metadata_spec_drift([agreeing], ("head_size",)) == []

    def test_comparable_fields_are_taken_from_every_launch(self):
        kernel = _kernel_doc(
            "k",
            {"a": {"spec": {"head_size": 64}}, "b": {"spec": {"causal": 1}}},
            metadata={"head_size": 64, "causal": 1},
        )
        assert desk_check.drift_comparable_fields([kernel]) == ("head_size", "causal")

    def test_full_mode_holds_every_launch_to_the_packed_dialect(self):
        desk_check._require_packed(_packed_ukd())
        with pytest.raises(HkpPackError, match="sources 'k1' kernel_source.kind is"):
            desk_check._require_packed(
                _sources_ukd({"k0": _packed_entry(PRIMARY), "k1": _rocke_entry()})
            )
        with pytest.raises(HkpPackError, match="sources 'k0' kernel_source.kind is"):
            desk_check._require_packed(_sources_ukd())
        with pytest.raises(HkpPackError) as info:
            desk_check._require_packed({"kernel_source": {"kind": "hip"}})
        assert "and kernel_source.kind is 'hip'." in str(info.value)


class TestDeskCheckReadsEachLaunchsObject:
    class _Archive:
        def __init__(self, kernels):
            self._kernels = kernels

        def get_kernel(self, toc_key, arch):
            assert arch == ARCH
            return self._kernels.get(toc_key)

    def _serve(self, monkeypatch, tmp_path, doc, kernels):
        """Write the library file `doc` names and make `load_kpack` read `kernels`
        (toc_key to bytes) out of it."""
        library = tmp_path / doc["sources"]["segment"]["library"]
        library.parent.mkdir(parents=True)
        library.write_bytes(b"archive")
        archive = self._Archive(kernels)
        kpack = type(
            "FakeKpack",
            (),
            {"PackedKernelArchive": type("A", (), {"read": lambda _path: archive})},
        )
        monkeypatch.setattr(desk_check, "load_kpack", lambda _dir: (kpack, None))

    def test_each_launch_is_read_from_its_own_toc_key_and_verifies(
        self, monkeypatch, tmp_path
    ):
        doc, records, payload = shipped(TWO_BUILDERS)
        assert payload["segment"] != payload["reduce"]
        by_toc_key = {doc["sources"][n]["toc_key"]: payload[n] for n in payload}
        self._serve(monkeypatch, tmp_path, doc, by_toc_key)
        read = desk_check._payload(Entry(doc, str(tmp_path), [ARCH]), ARCH)
        assert read == payload
        agreement.verify(doc, records, read)

    def test_a_toc_key_absent_from_the_archive_names_its_launch(
        self, monkeypatch, tmp_path
    ):
        doc, _records, payload = shipped(TWO_BUILDERS)
        self._serve(
            monkeypatch,
            tmp_path,
            doc,
            {doc["sources"]["segment"]["toc_key"]: payload["segment"]},
        )
        with pytest.raises(HkpPackError, match="launch 'reduce': toc_key 'vk-red'"):
            desk_check._payload(Entry(doc, str(tmp_path), [ARCH]), ARCH)


class TestProvenanceSidecarGuard:
    def test_a_packed_sources_ukd_in_an_unmarked_directory_is_refused(self, tmp_path):
        """`_require_authored` refuses a directory that holds kpack UKDs but no
        packed marker. A `sources` UKD has no `kernel_source` kind to find."""
        doc = {
            "id": "kdp-k",
            "kernelDescriptors": [_packed_ukd()],
        }
        kdp_path = tmp_path / "k.kdp.json"
        with pytest.raises(HkpPackError, match="looks packed"):
            provenance_sidecar.attach(kdp_path, doc)

    def test_an_authored_sources_ukd_is_left_alone(self, tmp_path):
        doc = {"id": "kdp-k", "kernelDescriptors": [_sources_ukd(version="1.0")]}
        assert provenance_sidecar.attach(tmp_path / "k.kdp.json", doc) is doc

    @pytest.mark.parametrize(
        "sources_value, expected",
        [
            ({"a": {"kind": "kpack"}, "b": {"kind": "kpack"}}, "kpack"),
            ({"a": {"kind": "rocke"}, "b": {"kind": "rocke"}}, "rocke"),
            ({"a": {"kind": "kpack"}, "b": {"kind": "rocke"}}, None),
            ({"a": {"kind": "kpack"}, "b": 3}, None),
            ({}, None),
            (3, None),
        ],
    )
    def test_the_kind_of_a_sources_ukd_is_the_one_its_entries_share(
        self, sources_value, expected
    ):
        assert provenance_sidecar._kind({"sources": sources_value}) == expected


def _inline(ukd_id, names):
    """An inline compiled UKD of a shard, naming `names` as its launches; no names
    is a single-launch kernel."""
    return pipeline.InlineUKD(
        id=ukd_id,
        name=ukd_id,
        metadata={},
        priority=0,
        source=None,
        entry=None,
        build=None,
        symbol=None,
        variant_key=None,
        launches={name: None for name in names},
    )


def _kdp_of(kdp_id, *entries, dispatch=None):
    """A shard KDP over `entries`; its UDD is its own unless `dispatch` names one."""
    return pipeline.ArchKDP(
        id=kdp_id,
        filename=f"{kdp_id}.kdp.json",
        header={"dispatch": dispatch or f"udd-{kdp_id}"},
        entries=list(entries),
    )


def _shard(*kdps, standalone=()):
    return pipeline.IntermediateArch(
        arch=ARCH,
        directory=None,
        kdps=list(kdps),
        standalone_ukds={u.id: u for u in standalone},
    )


def _udds(inter, symbols=None):
    """The input holding one UDD per dispatch the shard's KDPs name. Each UDD names
    the handler `handler.<udd id>` unless `symbols` maps its id to another."""
    symbols = symbols or {}
    ids = sorted({kdp.header["dispatch"] for kdp in inter.kdps} - {None})
    return FlatInput(
        descriptors=[
            Descriptor(
                path=Path(f"{udd_id}.udd.json"),
                doc={
                    "id": udd_id,
                    "name": udd_id,
                    "dispatch_symbol": symbols.get(udd_id, f"handler.{udd_id}"),
                },
            )
            for udd_id in ids
        ]
    )


def _check(inter, symbols=None):
    pipeline._check_launch_sets(_udds(inter, symbols), inter)


class TestOneLaunchSetPerDispatchHandler:
    """One native dispatch handler declares one launch set and serves every kernel
    of every KDP whose UDD names it; the packer refuses kernels of one handler that
    disagree, whether they share a KDP, a UDD, or only the dispatch symbol."""

    def test_kernels_naming_different_launch_sets_are_refused(self, tmp_path):
        inter = _shard(
            _kdp_of(
                "kdp-mixed",
                _inline("ukd-one", ["decode", "merge"]),
                _inline("ukd-two", ["segment", "reduce"]),
            )
        )
        with pytest.raises(HkpPackError) as info:
            pipeline.pack_arch(_udds(inter), inter, tmp_path, None, None)
        message = str(info.value)
        assert "dispatch handler 'handler.udd-kdp-mixed'" in message
        assert "KDP 'kdp-mixed'" in message
        assert "['ukd-one'] name launches ['decode', 'merge']" in message
        assert "['ukd-two'] name launches ['reduce', 'segment']" in message

    def test_a_single_launch_kernel_beside_a_multi_launch_one_is_refused(self):
        inter = _shard(
            _kdp_of(
                "kdp-mixed", _inline("ukd-plain", []), _inline("ukd-multi", ["a", "b"])
            )
        )
        with pytest.raises(HkpPackError) as info:
            _check(inter)
        message = str(info.value)
        assert "KDP 'kdp-mixed'" in message
        assert "['ukd-plain'] name launches []" in message
        assert "['ukd-multi'] name launches ['a', 'b']" in message

    def test_a_standalone_reference_is_held_to_its_launch_set(self):
        inter = _shard(
            _kdp_of("kdp-mixed", _inline("ukd-multi", ["a", "b"]), "ukd-alone"),
            standalone=[_inline("ukd-alone", [])],
        )
        with pytest.raises(HkpPackError, match=r"\['ukd-alone'\] name launches \[\]"):
            _check(inter)

    def test_the_same_launch_set_in_a_different_key_order_is_accepted(self):
        _check(
            _shard(
                _kdp_of(
                    "kdp-demo",
                    _inline("ukd-one", ["decode", "merge"]),
                    _inline("ukd-two", ["merge", "decode"]),
                )
            )
        )

    def test_two_kdps_sharing_one_udd_are_held_to_one_launch_set(self):
        # Each KDP agrees with itself; the loader still resolves one handler for
        # both, so the second KDP's kernel would be dropped at load.
        inter = _shard(
            _kdp_of("kdp-a", _inline("ukd-a", ["decode", "merge"]), dispatch="udd-x"),
            _kdp_of("kdp-b", _inline("ukd-b", ["decode", "reduce"]), dispatch="udd-x"),
        )
        with pytest.raises(HkpPackError) as info:
            _check(inter)
        message = str(info.value)
        assert "dispatch handler 'handler.udd-x'" in message
        assert "KDP 'kdp-a' (kdp-a.kdp.json) UKD(s) ['ukd-a']" in message
        assert "KDP 'kdp-b' (kdp-b.kdp.json) UKD(s) ['ukd-b']" in message
        assert "name launches ['decode', 'merge']" in message
        assert "name launches ['decode', 'reduce']" in message

    def test_two_udds_naming_one_dispatch_symbol_are_held_to_one_launch_set(self):
        inter = _shard(
            _kdp_of("kdp-multi", _inline("ukd-multi", ["decode", "merge"])),
            _kdp_of("kdp-plain", _inline("ukd-plain", [])),
        )
        symbols = {"udd-kdp-multi": "sdpa.fwd", "udd-kdp-plain": "sdpa.fwd"}
        with pytest.raises(HkpPackError, match=r"dispatch handler 'sdpa\.fwd'"):
            _check(inter, symbols)

    def test_each_dispatch_handler_may_serve_its_own_launch_set(self):
        shared = _inline("ukd-shared", ["a", "b"])
        _check(
            _shard(
                _kdp_of("kdp-ab", _inline("ukd-x", ["a", "b"]), "ukd-shared"),
                _kdp_of("kdp-cd", _inline("ukd-y", ["c", "d"])),
                _kdp_of("kdp-plain", _inline("ukd-z", [])),
                # Shares kdp-ab's UDD, and its launch set with it.
                _kdp_of("kdp-ab-again", "ukd-shared", dispatch="udd-kdp-ab"),
                standalone=[shared],
            )
        )

    def test_a_kdp_naming_no_udd_is_held_to_one_launch_set_on_its_own(self):
        unbound = _kdp_of("kdp-a", _inline("ukd-a", ["a", "b"]), _inline("ukd-b", []))
        unbound.header["dispatch"] = None
        with pytest.raises(
            HkpPackError,
            match=r"KDP 'kdp-a', whose dispatch names no UDD in this root,",
        ):
            _check(_shard(unbound))

    def test_kdps_naming_no_udd_share_no_handler(self):
        # No UDD resolves, so there is no handler for them to share; each answers for
        # itself, as a KDP whose dispatch the loader cannot resolve is refused whole.
        first = _kdp_of("kdp-a", _inline("ukd-a", ["a", "b"]))
        second = _kdp_of("kdp-b", _inline("ukd-b", []))
        first.header["dispatch"] = second.header["dispatch"] = None
        _check(_shard(first, second))
