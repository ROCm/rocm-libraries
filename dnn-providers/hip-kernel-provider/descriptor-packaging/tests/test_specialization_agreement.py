"""Compiled-specialization agreement: what a declaration observes and what binds it.

Every case is DISCRIMINATING -- each passes before the property it names is broken
and fails after, and no two fail for the same reason. They need no GPU, comgr or
rocKE: the observer takes readouts off whatever object it is handed.

The spec classes mirror the coupling shapes of the real gfx942 kernel
(`rocke/library/kernels/gfx942/attention_dense.py`): a tri-state field whose
accessor consults a policy when the raw value is None, and a second accessor that
returns False whenever its partner is off. Mirrored rather than imported, because
importing the producer to test the verifier is the dependency this design refuses.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hkp_pack import agreement, pipeline
from hkp_pack.errors import HkpPackError
from hkp_pack.desk_check import compiled_agreement, load_kernels
from hkp_pack.kpack_resolver import load_kpack

ARCH = "gfx942"
PAYLOAD = b"\x7fELF-not-really-a-code-object"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
SYMBOL = "attention_dense_bf16_d128"
# One argument in the shape kernel_signature emits -- `name` is present only when the
# code object carries one, so a bare kind/size/offset triple is a valid produced entry.
DEMO_SIGNATURE = [{"kind": "global_buffer", "size": 8, "offset": 0}]


@dataclass(frozen=True)
class DemoSpec:
    """A spec with the three readout shapes the contract distinguishes: plain
    fields (``head_size``, ``causal``), a constructor default (``block_n``) that
    still arrives as a definite value when the key is omitted, and tri-state fields
    (``use_exp2_fast``, ``use_v_swizzle``) whose ``None`` means the kernel's own
    policy decides -- never a wildcard, never false.
    """

    head_size: int
    dtype: str = "bf16"
    causal: bool = True
    block_n: int = 64
    tags: list = field(default_factory=list)
    use_cfvst: bool | None = None
    use_v_swizzle: bool | None = None
    use_exp2_fast: bool | None = None

    def resolved_use_cfvst(self) -> bool:
        if self.use_cfvst is None:
            return self.head_size == 128 and self.dtype == "fp16"
        return bool(self.use_cfvst)

    def resolved_use_v_swizzle(self) -> bool:
        # The coupling: with the conflict-free-V path off there is no V_lds to
        # swizzle, so the swizzle is off whatever the raw field says.
        if not self.resolved_use_cfvst():
            return False
        if self.use_v_swizzle is None:
            return True
        return bool(self.use_v_swizzle)

    def resolved_use_exp2_fast(self) -> bool:
        if self.use_exp2_fast is None:
            return not (self.dtype == "bf16" and self.head_size == 128)
        return bool(self.use_exp2_fast)

    def unstable_counter(self) -> int:
        self.tags.append(1)
        return len(self.tags)


def demo_builder(spec, *, arch):
    """Stands in for the real builder. Nothing observes its return value."""
    return (spec, arch)


def kmd(*fields) -> dict:
    return {"version": "1.0", "id": "kmd-demo", "name": "demo", "fields": list(fields)}


FULL_KMD = kmd(
    {"name": "head_size", "type": "int", "default_value": 128},
    {"name": "dtype", "type": "string"},
    {"name": "causal", "type": "int", "default_value": 1},
    {"name": "block_n", "type": "int", "default_value": 64},
    {"name": "use_exp2_fast", "type": "int", "default_value": 1},
    {"name": "family", "type": "string", "default_value": "attention_dense"},
)

ENGINE = {
    "version": "1.0",
    "id": "ued-demo",
    "name": "demo:Engine",
    "metadata": "kmd-demo",
}


def consumer(**overrides) -> dict:
    base = {
        "engine_id": "ued-demo",
        "kmd_id": "kmd-demo",
        "metadata_fields": ["head_size", "dtype", "causal", "block_n", "use_exp2_fast"],
        "matcher_only_fields": ["family"],
        "bindings": {
            "head_size": {"field": "head_size"},
            "dtype": {"field": "dtype"},
            "causal": {"field": "causal"},
            "block_n": {"field": "block_n"},
            "use_exp2_fast": {"method": "resolved_use_exp2_fast"},
        },
        "vocabulary": {"dtype": {"bf16": "BF16", "fp16": "FP16"}},
    }
    base.update(overrides)
    return base


def observe_for(spec, *consumers, schema=FULL_KMD):
    """The observations one compile produces for every consumer that asked."""
    requests = {}
    for entry in consumers:
        request = agreement.observation_request(entry, schema)
        requests[agreement.digest(request)] = request
    observations = agreement.observe(
        spec, demo_builder, requests, agreement.OriginObserver()
    )
    observations["arch"] = ARCH
    observations["symbol"] = SYMBOL
    observations["code_object_sha256"] = PAYLOAD_SHA
    return observations


class TestObservedValues:
    """What the observer reads off a real hydrated object."""

    def test_null_tri_state_resolves_to_true_or_false_by_the_kernels_own_policy(self):
        """`None` is a question the object answers, not a wildcard and not false.
        A checker mapping it to `False` would agree with the bf16/D128 descriptor
        and mislabel every other shape; treating it as "matches anything" would
        agree with both and mean nothing.
        """
        on = observe_for(DemoSpec(head_size=64), consumer())
        off = observe_for(DemoSpec(head_size=128, dtype="bf16"), consumer())
        key = agreement.digest(agreement.observation_request(consumer(), FULL_KMD))
        assert on["requests"][key]["values"]["use_exp2_fast"] == 1
        assert off["requests"][key]["values"]["use_exp2_fast"] == 0

    def test_a_constructor_default_is_observed_as_the_value_it_hydrates_to(self):
        """An omitted authored key is a definite value on the object, not an
        absence: `block_n` is never authored here, so only the dataclass default
        can supply the 64 the comparison agrees with.
        """
        observations = observe_for(DemoSpec(head_size=64), consumer())
        key = agreement.digest(agreement.observation_request(consumer(), FULL_KMD))
        assert observations["requests"][key]["values"]["block_n"] == 64

    def test_effective_accessor_overrides_an_explicitly_true_raw_field(self):
        """Raw swizzle True, effective False, because the cfvst path is disabled --
        the case that makes "read the raw field when non-null" wrong. The
        declaration binds the accessor, so the observation reports the builder's
        decision rather than the author's request.
        """
        spec = DemoSpec(head_size=64, use_cfvst=False, use_v_swizzle=True)
        assert spec.use_v_swizzle is True
        swizzle_kmd = kmd({"name": "use_v_swizzle", "type": "bool"})
        entry = consumer(
            metadata_fields=["use_v_swizzle"],
            matcher_only_fields=[],
            bindings={"use_v_swizzle": {"method": "resolved_use_v_swizzle"}},
            vocabulary={},
        )
        observations = observe_for(spec, entry, schema=swizzle_kmd)
        key = agreement.digest(agreement.observation_request(entry, swizzle_kmd))
        assert observations["requests"][key]["values"]["use_v_swizzle"] is False

    def test_a_boolean_observed_against_an_int_destination_is_encoded_as_an_integer(
        self,
    ):
        """The 0/1 projection is a property of the declared destination type."""
        observations = observe_for(DemoSpec(head_size=64, causal=True), consumer())
        key = agreement.digest(agreement.observation_request(consumer(), FULL_KMD))
        value = observations["requests"][key]["values"]["causal"]
        assert value == 1 and type(value) is int

    def test_a_boolean_observed_against_a_bool_destination_stays_boolean(self):
        bool_kmd = kmd({"name": "causal", "type": "bool"})
        entry = consumer(
            metadata_fields=["causal"],
            matcher_only_fields=[],
            bindings={"causal": {"field": "causal"}},
            vocabulary={},
        )
        observations = observe_for(DemoSpec(head_size=64), entry, schema=bool_kmd)
        key = agreement.digest(agreement.observation_request(entry, bool_kmd))
        assert observations["requests"][key]["values"]["causal"] is True


class TestUnsupportedReadouts:
    """An unsupported readout is a failure, never a wildcard success."""

    def test_a_binding_naming_an_absent_accessor_fails(self):
        entry = consumer(
            bindings={
                **consumer()["bindings"],
                "use_exp2_fast": {"method": "resolved_nothing"},
            }
        )
        with pytest.raises(HkpPackError, match="resolved_nothing"):
            observe_for(DemoSpec(head_size=64), entry)

    def test_a_method_binding_naming_a_plain_field_fails(self):
        """The wrong KIND of readout is as wrong as a missing one: a field read
        through a method binding would observe the raw value the builder resolved
        past."""
        entry = consumer(
            bindings={
                **consumer()["bindings"],
                "use_exp2_fast": {"field": "use_exp2_fast"},
            }
        )
        with pytest.raises(HkpPackError):
            # The raw tri-state is None, which is no legal int for this destination.
            observe_for(DemoSpec(head_size=64), entry)

    def test_a_non_repeatable_accessor_fails(self):
        """Agreement is a claim about a decision, and a decision that differs
        between two reads of the same object is not one."""
        counter_kmd = kmd({"name": "count", "type": "int"})
        entry = consumer(
            metadata_fields=["count"],
            matcher_only_fields=[],
            bindings={"count": {"method": "unstable_counter"}},
            vocabulary={},
        )
        with pytest.raises(HkpPackError, match="non-repeatable"):
            observe_for(DemoSpec(head_size=64), entry, schema=counter_kmd)


class TestDeclarationValidation:
    """What a declaration is allowed to say."""

    def test_a_partition_that_misses_a_kmd_field_fails(self):
        with pytest.raises(HkpPackError, match="partition"):
            agreement.validate_consumer(consumer(matcher_only_fields=[]), FULL_KMD)

    def test_an_overlapping_partition_fails(self):
        with pytest.raises(HkpPackError, match="partition"):
            agreement.validate_consumer(
                consumer(matcher_only_fields=["family", "causal"]), FULL_KMD
            )

    def test_binding_keys_must_equal_metadata_fields(self):
        bindings = dict(consumer()["bindings"])
        bindings.pop("causal")
        with pytest.raises(HkpPackError, match="binding keys"):
            agreement.validate_consumer(consumer(bindings=bindings), FULL_KMD)

    def test_a_binding_naming_both_a_field_and_a_method_fails(self):
        bindings = {
            **consumer()["bindings"],
            "causal": {"field": "causal", "method": "resolved_use_cfvst"},
        }
        with pytest.raises(HkpPackError, match="exactly one"):
            agreement.validate_consumer(consumer(bindings=bindings), FULL_KMD)

    def test_a_declaration_cannot_name_an_import_root(self):
        """There is nowhere in a declaration to redirect the compiler's imports:
        the consumer's key set is closed, so a `provider_root` is rejected outright
        rather than ignored, keeping a declaration from becoming a second way to
        choose which producer runs.
        """
        entry = consumer()
        entry["provider_root"] = "/some/unrelated/editable/checkout"
        with pytest.raises(HkpPackError, match="missing/unknown keys"):
            agreement.validate_consumer(entry, FULL_KMD)

    def test_two_entries_for_one_engine_and_kmd_pair_fail(self):
        ukd = {
            "id": "ukd-demo",
            "provenance": {
                "specialization_contract": {
                    "schema_version": 1,
                    "consumers": [consumer(), consumer()],
                }
            },
        }
        with pytest.raises(HkpPackError, match="duplicate/conflicting"):
            agreement.contracts(ukd, {"kmd-demo": FULL_KMD})

    def test_one_standalone_ukd_may_declare_several_distinct_consumers(self):
        """A UKD several engines reference carries one entry per engine."""
        second_kmd = dict(FULL_KMD, id="kmd-other")
        ukd = {
            "id": "ukd-demo",
            "provenance": {
                "specialization_contract": {
                    "schema_version": 1,
                    "consumers": [
                        consumer(),
                        consumer(engine_id="ued-other", kmd_id="kmd-other"),
                    ],
                }
            },
        }
        entries = agreement.contracts(
            ukd, {"kmd-demo": FULL_KMD, "kmd-other": second_kmd}
        )
        assert len(entries) == 2

    def test_selecting_a_declaration_for_an_engine_that_declares_none_fails(self):
        ukd = {
            "id": "ukd-demo",
            "provenance": {
                "specialization_contract": {
                    "schema_version": 1,
                    "consumers": [consumer(engine_id="ued-other")],
                }
            },
        }
        with pytest.raises(HkpPackError, match="expected exactly one"):
            agreement.select_declaration(ukd, ENGINE, FULL_KMD, {"kmd-demo": FULL_KMD})


def contract_of(*consumers) -> dict:
    return {"schema_version": 1, "consumers": list(consumers)}


class TestSharedCarriage:
    """Where the declaration is written, and what each kernel resolves to.

    One engine's inline kernels share one declaration, so the KDP may carry it
    once; a reader still gets the consumers in force for the kernel before it.
    """

    @staticmethod
    def kdp(contract=None, *kernels) -> dict:
        doc = {
            "id": "kdp-demo",
            "engine": "ued-demo",
            "kernelDescriptors": list(kernels),
        }
        if contract is not None:
            doc["provenance"] = {"specialization_contract": contract}
        return doc

    def test_a_kernel_with_none_of_its_own_inherits_the_kdps(self):
        kernel = {"id": "ukd-a"}
        doc = self.kdp(contract_of(consumer()), kernel)
        entries = agreement.contracts(kernel, {"kmd-demo": FULL_KMD}, doc)
        assert [e["engine_id"] for e in entries] == ["ued-demo"]

    def test_a_kernels_own_declaration_overrides_the_kdps_wholesale(self):
        """Never a merge: a merged block would let a kernel inherit a consumer it
        never declared, which is the one thing the declaration exists to rule out."""
        second_kmd = dict(FULL_KMD, id="kmd-other")
        kernel = {
            "id": "ukd-a",
            "provenance": {
                "specialization_contract": contract_of(
                    consumer(engine_id="ued-other", kmd_id="kmd-other")
                )
            },
        }
        doc = self.kdp(contract_of(consumer()), kernel)
        entries = agreement.contracts(
            kernel, {"kmd-demo": FULL_KMD, "kmd-other": second_kmd}, doc
        )
        assert [(e["engine_id"], e["kmd_id"]) for e in entries] == [
            ("ued-other", "kmd-other")
        ]

    def test_a_kernel_with_neither_is_the_same_hard_error(self):
        kernel = {"id": "ukd-a"}
        with pytest.raises(HkpPackError, match="missing/invalid"):
            agreement.contracts(kernel, {"kmd-demo": FULL_KMD}, self.kdp(None, kernel))

    def test_a_standalone_ukd_has_no_enclosing_kdp_and_must_carry_its_own(self):
        """It is its own file and several KDPs may reference it, so no KDP speaks
        for it -- which is why every reader passes None for a standalone entry."""
        with pytest.raises(HkpPackError, match="missing/invalid"):
            agreement.contracts({"id": "ukd-standalone"}, {"kmd-demo": FULL_KMD})

    def test_an_inherited_declaration_is_validated_like_an_authored_one(self):
        """Inheritance moves where the claim is written, never what it must satisfy."""
        kernel = {"id": "ukd-a"}
        doc = self.kdp(contract_of(consumer(kmd_id="kmd-absent")), kernel)
        with pytest.raises(HkpPackError, match="dangling KMD"):
            agreement.contracts(kernel, {"kmd-demo": FULL_KMD}, doc)


class TestComparison:
    """Observations against completed metadata."""

    @staticmethod
    def metadata(**overrides):
        base = {
            "head_size": 64,
            "dtype": "BF16",
            "causal": 1,
            "block_n": 64,
            "use_exp2_fast": 1,
        }
        base.update(overrides)
        return base

    def test_agreeing_metadata_compares_clean(self):
        observations = observe_for(DemoSpec(head_size=64), consumer())
        agreement.compare(consumer(), FULL_KMD, self.metadata(), observations)

    def test_metadata_written_in_the_builders_vocabulary_fails(self):
        """`bf16` loads cleanly, reconciles on every count, and matches nothing."""
        observations = observe_for(DemoSpec(head_size=64), consumer())
        with pytest.raises(HkpPackError, match="disagrees"):
            agreement.compare(
                consumer(), FULL_KMD, self.metadata(dtype="bf16"), observations
            )

    def test_an_absent_metadata_key_is_completed_from_the_kmd_default(self):
        """The loader substitutes the KMD default, so the comparison must too:
        otherwise a descriptor omitting a key would be checked against nothing
        while the runtime checked it against 64."""
        observations = observe_for(DemoSpec(head_size=64), consumer())
        without_block_n = self.metadata()
        del without_block_n["block_n"]
        agreement.compare(consumer(), FULL_KMD, without_block_n, observations)

    def test_an_absent_key_whose_kmd_default_disagrees_still_fails(self):
        observations = observe_for(DemoSpec(head_size=64, block_n=32), consumer())
        without_block_n = self.metadata()
        del without_block_n["block_n"]
        with pytest.raises(HkpPackError, match="disagrees"):
            agreement.compare(consumer(), FULL_KMD, without_block_n, observations)

    def test_a_second_contradictory_consumer_of_one_compile_result_fails(self):
        """A shared variant is checked once per consumer, and the first one's
        agreement certifies nothing about the second: both ask the same object the
        same questions, so one compile answers both, and the second consumer's
        metadata contradicts the answer.
        """
        spec = DemoSpec(head_size=64)
        first = consumer()
        second = consumer(engine_id="ued-second")
        observations = observe_for(spec, first, second)
        agreement.compare(first, FULL_KMD, self.metadata(), observations)
        with pytest.raises(HkpPackError, match="disagrees"):
            agreement.compare(
                second, FULL_KMD, self.metadata(head_size=128), observations
            )

    def test_a_consumer_whose_request_was_never_collected_fails(self):
        """Collecting every request BEFORE compiling is what makes a shared result
        usable; a request discovered afterwards has no observation and must not be
        answered from another consumer's."""
        observations = observe_for(DemoSpec(head_size=64), consumer())
        late = consumer(
            metadata_fields=["head_size"],
            matcher_only_fields=[
                "dtype",
                "causal",
                "block_n",
                "use_exp2_fast",
                "family",
            ],
            bindings={"head_size": {"field": "head_size"}},
            vocabulary={},
        )
        with pytest.raises(HkpPackError, match="lacks this consumer's observation"):
            agreement.compare(late, FULL_KMD, self.metadata(), observations)


def shipped_ukd(**overrides):
    """A shipped kpack UKD with producer evidence, as `_rewrite_ukd_kpack` writes it."""
    spec = DemoSpec(head_size=64)
    entry = consumer()
    observations = observe_for(spec, entry)
    record = pipeline.InlineUKD(
        id="ukd-demo",
        name="demo variant",
        metadata=TestComparison.metadata(),
        priority=0,
        source="kernels/demo/attention_dense.py",
        entry=None,
        build=None,
        symbol=SYMBOL,
        variant_key="vk-demo",
        extra=overrides.pop("extra", {}),
        origin_kind="rocke",
        builder="demo_builder",
        spec={"head_size": 64},
        provenance=overrides.pop("provenance", {}),
        observations=observations,
        consumers=agreement.canonical_records(
            [
                agreement.consumer_record(
                    {"id": "ukd-demo", "metadata": TestComparison.metadata()},
                    ENGINE,
                    FULL_KMD,
                    {"id": "kdp-demo", "engine": "ued-demo", "arch": [ARCH]},
                    ARCH,
                    entry,
                )
            ]
        ),
    )
    doc = pipeline._rewrite_ukd_kpack(
        record, ARCH, "vk-demo", PAYLOAD_SHA, signature=DEMO_SIGNATURE
    )
    # The producer stamps the ABI signature into the same kernel_source this
    # helper's callers read the provenance out of, so assert it here: it is where
    # the signature thread and the agreement thread write one document.
    assert doc["kernel_source"]["signature"] == DEMO_SIGNATURE
    return doc, record.consumers


def observations_of(doc):
    """The compiler's own readouts, which the descriptor digest excludes."""
    return doc["provenance"]["effective_spec"]["observations"]


def restamp_authored_spec(doc):
    """Change the authored spec and re-stamp the descriptor digest over it, since
    `provenance.spec` sits inside the digested document and a bare edit is refused
    by the descriptor binding first. Re-stamping leaves the authored-input check as
    the one that speaks.
    """
    doc["provenance"]["spec"] = {"head_size": 128}
    evidence = doc["provenance"]["effective_spec"]
    evidence["descriptor_digest"] = agreement.descriptor_binding(doc)


class TestProducerEvidenceReservation:
    """Only the compiler writes the record, and nothing overwrites it."""

    def test_a_shipped_ukd_carries_the_record_and_keeps_the_authored_spec(self):
        doc, _records = shipped_ukd()
        assert doc["provenance"]["effective_spec"]["schema_version"] == 1
        assert doc["provenance"]["spec"] == {"head_size": 64}

    def test_an_authored_effective_spec_is_rejected(self):
        """A forged record is refused at the write boundary rather than merged."""
        with pytest.raises(HkpPackError, match="only the producing compiler creates"):
            shipped_ukd(provenance={"effective_spec": {"schema_version": 1}})

    def test_authored_extra_may_not_name_a_produced_field(self):
        """The passthrough cannot land on top of a field this function writes --
        including `provenance`, which is what carries the evidence."""
        with pytest.raises(HkpPackError, match="produced UKD field"):
            shipped_ukd(extra={"metadata": {"head_size": 999}})

    def test_the_evidence_binds_the_document_that_actually_ships(self):
        """`descriptor_digest` is taken after the passthrough and the shard arch,
        so it describes the bytes written to disk rather than an intermediate."""
        doc, records = shipped_ukd()
        agreement.verify(doc, records, PAYLOAD)


class TestPackedVerification:
    """Full checks on a shipped artifact, with no producer importable."""

    def test_a_valid_packed_artifact_verifies(self):
        doc, records = shipped_ukd()
        agreement.verify(doc, records, PAYLOAD)

    @pytest.mark.parametrize(
        "mutate,diagnostic",
        [
            pytest.param(
                lambda d: d["metadata"].__setitem__("head_size", 128),
                "descriptor binding mismatch",
                id="metadata",
            ),
            pytest.param(
                lambda d: d["kernel_source"].__setitem__("symbol", "other"),
                "descriptor binding mismatch",
                id="authored-symbol",
            ),
            pytest.param(
                lambda d: d["kernel_source"].__setitem__("toc_key", "other"),
                "descriptor binding mismatch",
                id="authored-toc-key",
            ),
            pytest.param(
                lambda d: d.__setitem__("arch", ["gfx950"]),
                "descriptor binding mismatch",
                id="authored-arch",
            ),
            pytest.param(
                restamp_authored_spec,
                "authored-input binding mismatch",
                id="authored-spec",
            ),
            pytest.param(
                lambda d: observations_of(d).__setitem__(
                    "code_object_sha256", hashlib.sha256(b"other object").hexdigest()
                ),
                "code-object binding mismatch",
                id="observed-code-object",
            ),
            pytest.param(
                lambda d: observations_of(d).__setitem__("symbol", "other"),
                "symbol binding mismatch",
                id="observed-symbol",
            ),
            pytest.param(
                lambda d: observations_of(d).__setitem__("arch", "gfx950"),
                "architecture binding mismatch",
                id="observed-arch",
            ),
        ],
    )
    def test_a_changed_descriptor_fails_its_binding(self, mutate, diagnostic):
        """Each mutation is one field of the binding, and each fails on its own.

        The expected diagnostic travels with the mutation because `verify` raises
        one exception class from nine sites. The first four cases share one message
        by layering: the descriptor digest covers the whole authored document, so an
        edit to the symbol, toc key or arch is refused before the evidence's own
        copy is compared, which the `observed-` cases reach by moving the evidence.
        """
        doc, records = shipped_ukd()
        mutate(doc)
        with pytest.raises(HkpPackError, match=diagnostic):
            agreement.verify(doc, records, PAYLOAD)

    def test_changed_payload_bytes_fail(self):
        doc, records = shipped_ukd()
        with pytest.raises(HkpPackError, match="payload binding"):
            agreement.verify(doc, records, PAYLOAD + b"tampered")

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda r: r["engine"].__setitem__("name", "demo:Other"), id="engine"
            ),
            pytest.param(
                # The KMD is what the descriptor's metadata is read against, so a
                # moved default has to fail here rather than quietly re-complete
                # the metadata and agree with an observation taken under the old
                # one.
                lambda r: r["kmd"]["fields"][3].__setitem__("default_value", 32),
                id="kmd",
            ),
            pytest.param(
                lambda r: r["kdp"].__setitem__("name", "changed consumer"), id="kdp"
            ),
            pytest.param(
                lambda r: r["metadata"].__setitem__("head_size", 128), id="metadata"
            ),
            pytest.param(
                lambda r: r["declaration"].__setitem__("vocabulary", {}),
                id="declaration",
            ),
            pytest.param(
                lambda r: r.__setitem__("effective_arch", "gfx950"),
                id="effective_arch",
            ),
        ],
    )
    def test_a_changed_consumer_document_fails_its_binding(self, mutate):
        """Every document a record names, moved on its own, on the rebuild side.
        The stored list names and digests the documents rather than carrying them,
        so six mutations and six failures say the projection kept every binding an
        embedded document would have carried.
        """
        doc, records = shipped_ukd()
        altered = copy.deepcopy(records)
        mutate(altered[0])
        with pytest.raises(HkpPackError, match="consumer binding"):
            agreement.verify(doc, altered, PAYLOAD)

    def test_a_declaration_re_evidenced_over_its_own_edit_still_fails(self):
        """A consistent edit on both sides is refused by the observation instead:
        re-deriving the evidence takes the consumer binding out of the way, leaving
        the compile itself, where the edited declaration asks for readouts the
        observation was never taken for.
        """
        doc, records = shipped_ukd()
        records[0]["declaration"]["vocabulary"] = {}
        doc["provenance"]["effective_spec"]["consumers"] = [
            agreement.stored_record(entry) for entry in records
        ]
        with pytest.raises(HkpPackError, match="lacks this consumer's observation"):
            agreement.verify(doc, records, PAYLOAD)

    def test_an_absent_record_is_a_failure_and_not_an_unchecked_property(self):
        doc, records = shipped_ukd()
        del doc["provenance"]["effective_spec"]
        with pytest.raises(HkpPackError, match="missing compiler-owned"):
            agreement.verify(doc, records, PAYLOAD)

    def test_an_unsupported_record_schema_version_is_a_failure(self):
        doc, records = shipped_ukd()
        doc["provenance"]["effective_spec"]["schema_version"] = 2
        with pytest.raises(HkpPackError, match="missing compiler-owned"):
            agreement.verify(doc, records, PAYLOAD)

    def test_a_forged_record_copied_onto_another_descriptor_fails(self):
        """Lifting a valid record onto a descriptor it was not written for is the
        cheapest forgery available, and the descriptor digest refuses it."""
        doc, records = shipped_ukd()
        other = copy.deepcopy(doc)
        other["id"] = "ukd-other"
        other["provenance"]["effective_spec"] = copy.deepcopy(
            doc["provenance"]["effective_spec"]
        )
        with pytest.raises(HkpPackError, match="descriptor binding"):
            agreement.verify(other, records, PAYLOAD)

    def test_stripped_producing_object_identity_fails(self):
        doc, records = shipped_ukd()
        doc["provenance"]["effective_spec"]["observations"]["producer"].pop("builder")
        with pytest.raises(HkpPackError, match="producing-object identity"):
            agreement.verify(doc, records, PAYLOAD)

    def test_verification_reads_only_json_and_bytes(self):
        """The whole record round-trips through JSON, which is what lets a checker
        with no producer installed reach the same verdict the compiler did."""
        doc, records = shipped_ukd()
        reloaded = json.loads(json.dumps(doc))
        agreement.verify(reloaded, json.loads(json.dumps(records)), PAYLOAD)


_MACHINE_PRODUCER = '''
"""A producer whose only variable is the directory it was read from."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineSpec:
    head_size: int = 64
    dtype: str = "bf16"
    causal: bool = True
    block_n: int = 64

    def resolved_use_exp2_fast(self) -> bool:
        return not (self.dtype == "bf16" and self.head_size == 128)


def build(spec, *, arch):
    return (spec, arch)
'''


@contextlib.contextmanager
def loaded_producer(directory):
    """One copy of the same producer bytes, imported from `directory`: two roots of
    identical source is what a second build machine presents to the observer -- one
    module name, one content hash, a different absolute path. Loaded by location
    under a fixed module name and registered in `sys.modules`, where
    `inspect.getsourcefile` reads from.
    """
    directory.mkdir(parents=True)
    path = directory / "machine_demo.py"
    path.write_text(_MACHINE_PRODUCER)
    spec = importlib.util.spec_from_file_location("machine_demo", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


def published_evidence(directory):
    """The record `publish` writes for one compile of the producer in `directory`."""
    with loaded_producer(directory) as module:
        return _publish_for(module)


def _publish_for(module):
    entry = consumer()
    request = agreement.observation_request(entry, FULL_KMD)
    observations = agreement.observe(
        module.MachineSpec(),
        module.build,
        {agreement.digest(request): request},
        agreement.OriginObserver(),
    )
    observations["arch"] = ARCH
    observations["symbol"] = SYMBOL
    observations["code_object_sha256"] = PAYLOAD_SHA
    ukd = {
        "id": "ukd-demo",
        "metadata": TestComparison.metadata(),
        "provenance": {
            "source": "kernels/demo/attention_dense.py",
            "builder": "build",
            "spec": {"head_size": 64},
        },
    }
    agreement.publish(
        ukd,
        observations,
        agreement.canonical_records(
            [
                agreement.consumer_record(
                    ukd,
                    ENGINE,
                    FULL_KMD,
                    {"id": "kdp-demo", "engine": "ued-demo", "arch": [ARCH]},
                    ARCH,
                    entry,
                )
            ]
        ),
    )
    return ukd["provenance"]["effective_spec"]


def strings_in(value):
    """Every string anywhere inside a JSON-shaped value, keys included."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)
    elif isinstance(value, str):
        yield value


class TestMachineIndependence:
    """What the shipped record may be a function of: its inputs, and nothing else."""

    def test_two_producer_directories_emit_byte_identical_evidence(self, tmp_path):
        """The property the other reproducibility tests cannot see: they compare a
        serial run against a parallel one, where every producer resolves to the
        same file. Two roots of differing depth is what a second machine looks like
        from inside the observer.
        """
        left = published_evidence(tmp_path / "a")
        right = published_evidence(tmp_path / "b/deeper/still")
        assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)

    def test_no_published_value_names_a_location_on_the_building_machine(
        self, tmp_path
    ):
        """No separator and no install root anywhere, keys included. Stated over
        every string rather than the fields known to have held one, so a path
        reintroduced under a new name fails here too.
        """
        evidence = published_evidence(tmp_path / "a")
        for value in strings_in(evidence):
            assert "site-packages" not in value
            assert "/" not in value and "\\" not in value

    def test_a_producer_identity_carrying_a_path_is_rejected(self):
        """A four-key identity is refused rather than accepted and ignored: the
        record is the only statement a checker has about what produced the payload,
        so a shape this module does not write is one it cannot reason about.
        """
        doc, records = shipped_ukd()
        observations_of(doc)["producer"]["builder"]["file"] = str(
            Path(__file__).resolve()
        )
        with pytest.raises(HkpPackError, match="producing-object identity"):
            agreement.verify(doc, records, PAYLOAD)


class _ReaderArtifact:
    """A real archive with data-only compiler observations, never executable code.

    Records use agreement's producer-side functions, not the descriptor context
    under test. Both CLI readers must accept the frozen control before mutation.
    """

    def __init__(
        self,
        root,
        python_dir,
        *,
        inherited=False,
        shared=False,
        distinct_kmd=False,
        nested=False,
    ):
        self.root = root
        self.root.mkdir()
        self.python_dir = python_dir
        self.inline = not shared and not nested
        self.ukd, _ = shipped_ukd()
        self.schemas = [copy.deepcopy(FULL_KMD)]
        self.engines = [copy.deepcopy(ENGINE)]
        self.kdps = [{"id": "kdp-selected", "engine": ENGINE["id"], "arch": [ARCH]}]
        declarations = [consumer()]
        if shared:
            if distinct_kmd:
                schema = copy.deepcopy(FULL_KMD)
                schema["id"] = "kmd-second"
                schema["fields"][0]["default_value"] = 256
                self.schemas.append(schema)
                engine = dict(ENGINE, id="ued-second", metadata=schema["id"])
                self.engines.append(engine)
                declarations.append(
                    consumer(engine_id=engine["id"], kmd_id=schema["id"])
                )
            self.kdps.append(
                {"id": "kdp-sibling", "engine": self.engines[-1]["id"], "arch": [ARCH]}
            )
        contract = contract_of(*declarations)
        if inherited and self.inline:
            self.kdps[0]["provenance"] = {"specialization_contract": contract}
        else:
            self.ukd["provenance"]["specialization_contract"] = contract
        self.ukd["kernel_source"]["library"] = "kpack/reader.kpack"
        self.ukd_path = root / (
            "nested/kernels/shared.ukd.json" if nested else "shared.ukd.json"
        )
        self.origin = root if self.inline else self.ukd_path.parent
        self.archive_path = self.origin / self.ukd["kernel_source"]["library"]
        schemas = {schema["id"]: schema for schema in self.schemas}
        engines = {engine["id"]: engine for engine in self.engines}
        records = []
        for kdp in self.kdps:
            engine = engines[kdp["engine"]]
            schema = schemas[engine["metadata"]]
            declaration = agreement.select_declaration(
                self.ukd, engine, schema, schemas, kdp if self.inline else None
            )
            records.append(
                agreement.consumer_record(
                    self.ukd, engine, schema, kdp, ARCH, declaration
                )
            )
        observations = self.ukd["provenance"]["effective_spec"]["observations"]
        agreement.publish(self.ukd, observations, agreement.canonical_records(records))
        # No shared Python references between mutable descriptors and old evidence.
        self.ukd = json.loads(json.dumps(self.ukd))
        self.profile = root / "profile.json"
        self.profile.write_text(
            json.dumps(
                {"bundle": "selected", "vocabulary": {"dtype": ["BF16", "FP16"]}}
            )
        )
        self.save()
        self.write_archive(PAYLOAD)

    @property
    def kdp_path(self):
        return self.root / "selected.kdp.json"

    def save(self):
        for index, schema in enumerate(self.schemas):
            (self.root / f"schema-{index}.kmd.json").write_text(json.dumps(schema))
        for index, engine in enumerate(self.engines):
            (self.root / f"engine-{index}.ued.json").write_text(json.dumps(engine))
        for name, kdp in zip(("selected", "sibling"), self.kdps):
            doc = dict(
                kdp, kernelDescriptors=[self.ukd if self.inline else self.ukd["id"]]
            )
            (self.root / f"{name}.kdp.json").write_text(json.dumps(doc))
        if not self.inline:
            self.ukd_path.parent.mkdir(parents=True, exist_ok=True)
            self.ukd_path.write_text(json.dumps(self.ukd))

    def write_archive(self, payload):
        kpack, compression = load_kpack(self.python_dir)
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive = kpack.PackedKernelArchive(
            group_name="reader",
            gfx_arch_family=ARCH,
            gfx_arches=[ARCH],
            compressor=compression.ZstdCompressor(compression_level=3),
        )
        archive.add_kernel(
            archive.prepare_kernel(
                relative_path=self.ukd["kernel_source"]["toc_key"],
                gfx_arch=ARCH,
                hsaco_data=payload,
                metadata={},
            )
        )
        archive.finalize_archive()
        archive.write(self.archive_path)

    def retire_evidence(self, origin_kind):
        """Drop the producer's record AND the claim that would demand it back --
        the shape a tree presents once its evidence is lost, not the shape a
        deliberate strip presents, which would move `origin_kind` too. That is why
        `origin_kind` is a parameter (`None` removes it) and the only other thing
        that moves, so a verdict is attributable to the origin.
        """
        provenance = self.ukd["provenance"]
        # The frozen control is rocKE-produced, which is what makes "hip" and
        # "absent" below real mutations.
        assert provenance["origin_kind"] == "rocke"
        provenance.pop("effective_spec")
        if origin_kind is None:
            provenance.pop("origin_kind")
        else:
            provenance["origin_kind"] = origin_kind
        declaration = provenance["specialization_contract"]["consumers"][0]
        declaration["matcher_only_fields"] += declaration["metadata_fields"]
        declaration["metadata_fields"] = []
        declaration["bindings"] = {}
        declaration["vocabulary"] = {}
        self.save()

    def assert_qualified_waiver(self):
        """Both readers pass, and both say out loud that they bound nothing."""
        failures, unclaimed, verified = compiled_agreement(
            self.kdp_path, self.python_dir
        )
        assert failures == [] and len(unclaimed) == 1 and verified == 0
        for result in self.run_clis():
            assert result.returncode == 0, result.stdout + result.stderr
            assert "NOT VERIFIED HERE" in result.stdout

    def run_clis(self):
        package = Path(__file__).resolve().parents[1]
        repo = Path(__file__).resolve().parents[4]
        desk = [
            sys.executable,
            str(package / "tools/hkp_desk_check.py"),
            str(self.kdp_path),
            "--mode",
            "full",
            "--field",
            "head_size",
        ]
        variant = [
            sys.executable,
            str(
                repo
                / "projects/hipdnn/tools/IngestorGenerator/tools/verify_variant_sets.py"
            ),
            "set",
            str(self.root),
            "--mode",
            "full",
            "--profile",
            str(self.profile),
        ]
        extra = ["--kpack-python-dir", str(self.python_dir)] if self.python_dir else []
        return [
            subprocess.run(args + extra, capture_output=True, text=True)
            for args in (desk, variant)
        ]

    def assert_agreement(self):
        assert compiled_agreement(self.kdp_path, self.python_dir) == ([], [], 1)
        for result in self.run_clis():
            assert result.returncode == 0, result.stdout + result.stderr
            assert "NOT VERIFIED HERE" not in result.stdout

    def assert_failure(self, reason):
        failures, unclaimed, verified = compiled_agreement(
            self.kdp_path, self.python_dir
        )
        assert failures and not unclaimed and verified == 0
        assert any(reason in failure for failure in failures), failures
        for result in self.run_clis():
            assert result.returncode == 1, result.stdout + result.stderr
            assert reason in result.stdout + result.stderr
            assert "Traceback" not in result.stderr


@pytest.fixture
def reader_artifact(tmp_path, rocm_kpack_dir):
    def build(**kwargs):
        return _ReaderArtifact(tmp_path / "shard", rocm_kpack_dir, **kwargs)

    return build


class TestRealArchiveReaders:
    """Agreement through actual archive IO and both shipped CLI readers."""

    def test_literal_bracket_root_preserves_desk_selection_and_agreement(
        self, tmp_path, rocm_kpack_dir
    ):
        artifact = _ReaderArtifact(tmp_path / "shard[1]", rocm_kpack_dir)
        assert load_kernels(artifact.kdp_path) == [artifact.ukd]
        artifact.assert_agreement()

    @pytest.mark.parametrize("inherited", [False, True])
    def test_empty_fields_cannot_waive_existing_evidence(
        self, reader_artifact, inherited
    ):
        artifact = reader_artifact(inherited=inherited)
        artifact.assert_agreement()
        owner = artifact.kdps[0] if inherited else artifact.ukd
        declaration = owner["provenance"]["specialization_contract"]["consumers"][0]
        declaration["matcher_only_fields"] += declaration["metadata_fields"]
        declaration["metadata_fields"] = []
        declaration["bindings"] = {}
        declaration["vocabulary"] = {}
        artifact.save()
        artifact.assert_failure("binding mismatch")

    def test_payload_mutation_fails_after_real_archive_control(self, reader_artifact):
        artifact = reader_artifact()
        artifact.assert_agreement()
        artifact.write_archive(PAYLOAD + b"-tampered")
        artifact.assert_failure("payload binding mismatch")

    def test_missing_evidence_is_not_an_unclaimed_kernel(self, reader_artifact):
        artifact = reader_artifact()
        artifact.assert_agreement()
        artifact.ukd["provenance"].pop("effective_spec")
        artifact.save()
        artifact.assert_failure("missing compiler-owned effective_spec")

    @pytest.mark.parametrize("distinct_kmd", [False, True])
    @pytest.mark.parametrize("mutation", ["missing-record", "stale-consumer"])
    def test_shared_standalone_requires_every_consumer(
        self, reader_artifact, distinct_kmd, mutation
    ):
        artifact = reader_artifact(shared=True, distinct_kmd=distinct_kmd)
        artifact.assert_agreement()
        sibling_path = artifact.root / "sibling.kdp.json"
        assert compiled_agreement(sibling_path, artifact.python_dir) == ([], [], 1)
        if mutation == "missing-record":
            artifact.ukd["provenance"]["effective_spec"]["consumers"].pop()
        else:
            artifact.kdps[1]["name"] = "changed consumer"
        artifact.save()
        artifact.assert_failure("consumer binding mismatch")

    @pytest.mark.parametrize("mutation", ["moved", "tampered"])
    def test_nested_standalone_reads_its_own_named_archive(
        self, reader_artifact, mutation
    ):
        artifact = reader_artifact(shared=True, distinct_kmd=True, nested=True)
        artifact.assert_agreement()
        if mutation == "tampered":
            artifact.write_archive(PAYLOAD + b"-nested-tamper")
            artifact.assert_failure("payload binding mismatch")
        else:
            # Same library spelling under KDP root is not the UKD's named file.
            wrong_origin = artifact.root / artifact.ukd["kernel_source"]["library"]
            wrong_origin.parent.mkdir(parents=True, exist_ok=True)
            artifact.archive_path.rename(wrong_origin)
            failures, unclaimed, verified = compiled_agreement(
                artifact.kdp_path, artifact.python_dir
            )
            assert failures and not unclaimed and verified == 0
            for result in artifact.run_clis():
                assert result.returncode == 1, result.stdout + result.stderr
                assert str(artifact.archive_path) in result.stdout + result.stderr

    def test_recordless_packed_no_claim_from_a_hip_origin_remains_qualified(
        self, reader_artifact
    ):
        """A hip-origin kernel with no claim and no record is the legitimate shape:
        the same mutation `test_a_rocke_origin_cannot_waive_its_own_evidence` makes,
        differing only in the origin left behind, so the pair proves the check
        discriminates on origin.
        """
        artifact = reader_artifact()
        artifact.assert_agreement()
        artifact.retire_evidence("hip")
        artifact.assert_qualified_waiver()

    def test_a_rocke_origin_cannot_waive_its_own_evidence(self, reader_artifact):
        """The waiver keyed on the claim alone is a self-service exemption: the
        packer publishes `effective_spec` onto every rocKE UKD it ships, so this
        descriptor's `origin_kind` contradicts the absent record, and leaving it
        waivable takes a shipped rocKE shard to a clean exit with the bytes unread.
        """
        artifact = reader_artifact()
        artifact.assert_agreement()
        artifact.retire_evidence("rocke")
        artifact.assert_failure(
            "A rocKE-produced kernel is required to carry its compiler evidence"
        )
        for result in artifact.run_clis():
            # The report must name which kernel.
            assert artifact.ukd["name"] in result.stdout + result.stderr
            assert "NOT VERIFIED HERE" not in result.stdout

    def test_an_absent_origin_kind_is_not_read_as_rocke(self, reader_artifact):
        """Silence is not a claim of rocKE origin, so it keeps the waiver:
        descriptors packed before `origin_kind` existed and hand-authored trees
        carry none, and inferring rocKE would fail every one of them.
        """
        artifact = reader_artifact()
        artifact.assert_agreement()
        artifact.retire_evidence(None)
        assert "origin_kind" not in artifact.ukd["provenance"]
        artifact.assert_qualified_waiver()

    def test_kdp_effective_spec_is_never_inherited(self, reader_artifact):
        artifact = reader_artifact(inherited=True)
        artifact.assert_agreement()
        artifact.kdps[0]["provenance"]["effective_spec"] = artifact.ukd[
            "provenance"
        ].pop("effective_spec")
        artifact.save()
        artifact.assert_failure("missing compiler-owned effective_spec")

    def test_standalone_requires_own_declaration_despite_kdp_contracts(
        self, reader_artifact
    ):
        artifact = reader_artifact(shared=True, distinct_kmd=True)
        artifact.assert_agreement()
        contract = artifact.ukd["provenance"].pop("specialization_contract")
        for kdp in artifact.kdps:
            kdp["provenance"] = {"specialization_contract": contract}
        artifact.save()
        failures, unclaimed, verified = compiled_agreement(
            artifact.kdp_path, artifact.python_dir
        )
        assert failures and not unclaimed and verified == 0
        assert any("missing/invalid specialization_contract" in f for f in failures)
        for result in artifact.run_clis():
            assert result.returncode == 1, result.stdout + result.stderr
            assert "specialization" in result.stdout + result.stderr
            assert "Traceback" not in result.stderr

    def test_own_override_cannot_borrow_kdp_consumer(self, reader_artifact):
        artifact = reader_artifact()
        artifact.assert_agreement()
        artifact.kdps[0]["provenance"] = {
            "specialization_contract": copy.deepcopy(
                artifact.ukd["provenance"]["specialization_contract"]
            )
        }
        artifact.ukd["provenance"]["specialization_contract"] = contract_of(
            consumer(engine_id="another-engine")
        )
        artifact.save()
        with pytest.raises(HkpPackError, match="0 specialization consumers"):
            compiled_agreement(artifact.kdp_path, artifact.python_dir)
        for result in artifact.run_clis():
            assert result.returncode == 1, result.stdout + result.stderr
            assert "0 specialization consumers" in result.stdout + result.stderr
            assert "Traceback" not in result.stderr
