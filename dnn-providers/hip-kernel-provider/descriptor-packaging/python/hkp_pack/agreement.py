"""Data-only specialization declarations and producing-build evidence.

No producer imports occur here. Only the compiler observes builder objects; readers
verify those observations against the current descriptors and archive payload.
"""

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path

from .errors import HkpPackError


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def canonical(value, kind, *, observed=False):
    if observed and kind == "int" and type(value) is bool:
        value = int(value)
    valid = {
        "bool": type(value) is bool,
        "int": type(value) is int and -(2**63) <= value < 2**63,
        "float": type(value) in (int, float) and math.isfinite(value),
        "string": type(value) is str,
        "int_list": isinstance(value, list)
        and all(type(v) is int and -(2**63) <= v < 2**63 for v in value),
    }
    if not valid.get(kind, False):
        raise HkpPackError(f"invalid {kind} metadata value {value!r}")
    return float(value) if kind == "float" else value


def complete_metadata(metadata, kmd):
    fields = {f["name"]: f for f in kmd["fields"]}
    if len(fields) != len(kmd["fields"]) or set(metadata) - fields.keys():
        raise HkpPackError("duplicate KMD fields or undeclared metadata")
    result = {}
    for name, field in fields.items():
        if name not in metadata and "default_value" not in field:
            raise HkpPackError(f"missing mandatory metadata {name!r}")
        result[name] = canonical(
            metadata.get(name, field.get("default_value")), field["type"]
        )
    return result


def overlap(left, right):
    return not left or not right or bool(set(left) & set(right))


def validate_consumer(consumer, kmd):
    required = {
        "engine_id",
        "kmd_id",
        "metadata_fields",
        "matcher_only_fields",
        "bindings",
        "vocabulary",
    }
    if not isinstance(consumer, dict) or set(consumer) != required:
        raise HkpPackError("specialization consumer has missing/unknown keys")
    if consumer["kmd_id"] != kmd["id"]:
        raise HkpPackError("specialization consumer references the wrong KMD")
    names = {f["name"] for f in kmd["fields"]}
    for key in ("metadata_fields", "matcher_only_fields"):
        value = consumer[key]
        if (
            not isinstance(value, list)
            or any(not isinstance(n, str) for n in value)
            or len(set(value)) != len(value)
        ):
            raise HkpPackError(f"invalid specialization {key}")
    checked, matcher = set(consumer["metadata_fields"]), set(
        consumer["matcher_only_fields"]
    )
    if checked & matcher or checked | matcher != names:
        raise HkpPackError("specialization fields must exhaustively partition the KMD")
    bindings = consumer["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != checked:
        raise HkpPackError("specialization binding keys must equal metadata_fields")
    for binding in bindings.values():
        if not isinstance(binding, dict) or set(binding) not in ({"field"}, {"method"}):
            raise HkpPackError("each binding must name exactly one field or method")
        name = next(iter(binding.values()))
        if not isinstance(name, str) or not name.isidentifier():
            raise HkpPackError("binding must name one explicit attribute")
    vocabulary = consumer["vocabulary"]
    if not isinstance(vocabulary, dict) or set(vocabulary) - checked:
        raise HkpPackError("vocabulary must map metadata_fields")
    if any(not isinstance(v, dict) for v in vocabulary.values()):
        raise HkpPackError("vocabulary entries must be explicit translations")
    return consumer


def contracts(ukd, schemas):
    contract = ukd.get("provenance", {}).get("specialization_contract")
    if (
        not isinstance(contract, dict)
        or set(contract) != {"schema_version", "consumers"}
        or contract["schema_version"] != 1
    ):
        raise HkpPackError(
            f"UKD {ukd.get('id')}: missing/invalid specialization_contract"
        )
    consumers = contract["consumers"]
    if not isinstance(consumers, list) or not consumers:
        raise HkpPackError("specialization_contract needs consumers")
    seen = set()
    for consumer in consumers:
        if not isinstance(consumer, dict) or consumer.get("kmd_id") not in schemas:
            raise HkpPackError("specialization consumer has dangling KMD")
        validate_consumer(consumer, schemas[consumer["kmd_id"]])
        key = (consumer["engine_id"], consumer["kmd_id"])
        if key in seen:
            raise HkpPackError("duplicate/conflicting specialization consumer")
        seen.add(key)
    return consumers


def observation_request(consumer, kmd):
    return {
        "bindings": consumer["bindings"],
        "vocabulary": consumer["vocabulary"],
        "types": {
            f["name"]: f["type"]
            for f in kmd["fields"]
            if f["name"] in consumer["metadata_fields"]
        },
    }


def select_declaration(ukd, engine, kmd, schemas):
    """The single consumer entry this UKD declares for one (engine, KMD) pair.

    A standalone UKD several engines reference carries one entry per pair, so the
    pair -- not the UKD -- selects. Zero matches is an unfulfilled specialization
    obligation and more than one is a conflict; both fail rather than picking, since
    either would certify this compile against a declaration it was not written for.
    """
    declarations = contracts(ukd, schemas)
    matching = [
        c
        for c in declarations
        if c["engine_id"] == engine["id"] and c["kmd_id"] == kmd["id"]
    ]
    if len(matching) != 1:
        raise HkpPackError(
            f"UKD {ukd.get('id')}: {len(matching)} specialization consumers for "
            f"engine {engine['id']} / KMD {kmd['id']}, expected exactly one"
        )
    return matching[0]


def consumer_record(ukd, engine, kmd, kdp_header, arch, declaration):
    """One consumer's binding of this compile to the descriptors that consume it.

    The producer writes these into the evidence and a reader rebuilds them from the
    descriptors in front of it; equality of the two lists is what makes a changed
    KMD, metadata, KDP header or effective architecture fail its binding. Both sides
    therefore build the record HERE and nowhere else -- a second construction with
    the same intent and a different key order would compare unequal for no reason,
    or equal despite a real difference.

    `kdp_header` is the KDP document without its `kernelDescriptors`, carrying the
    single shard `arch`, so the record binds the pack the UKD ships under rather than
    the authored multi-arch list.
    """
    return {
        "ukd_id": ukd["id"],
        "engine": engine,
        "kmd": kmd,
        "metadata": ukd["metadata"],
        "kdp": kdp_header,
        "effective_arch": arch,
        "declaration": declaration,
    }


def canonical_records(records):
    """One consumer list in an order neither side chooses.

    Two KDPs referencing one standalone UKD under the same engine, KMD and arch
    describe the same consumer, so the duplicate collapses. Ordering by digest keeps
    the list independent of descriptor traversal order, which differs between the
    producer's walk and a reader's directory scan.
    """
    unique = {digest(record): record for record in records}
    return [unique[key] for key in sorted(unique)]


class OriginObserver:
    """One stable producing invocation's defining-file identities."""

    def __init__(self):
        self.files = {}

    def identity(self, obj):
        obj = getattr(obj, "__func__", obj)
        try:
            path = Path(inspect.getsourcefile(obj)).resolve(strict=True)
            content = path.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            if path in self.files and self.files[path] != sha:
                raise HkpPackError(f"producer changed during compilation: {path}")
            self.files[path] = sha
            return {
                "module": obj.__module__,
                "qualname": obj.__qualname__,
                "file": str(path),
                "sha256": sha,
            }
        except (TypeError, OSError, AttributeError) as exc:
            raise HkpPackError(f"unresolvable producer origin: {obj!r}: {exc}") from exc

    def stable(self):
        for path, sha in self.files.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise HkpPackError(f"producer changed during compilation: {path}")


def observe(spec_obj, builder_fn, requests, origins):
    producer = {
        "builder": origins.identity(builder_fn),
        "spec": origins.identity(type(spec_obj)),
    }
    observations = {}
    for key, request in requests.items():
        if key != digest(request):
            raise HkpPackError("invalid observation request digest")
        values, accessors = {}, {}
        for name, binding in request["bindings"].items():
            mode, attribute = next(iter(binding.items()))
            try:
                selected = getattr(spec_obj, attribute)
                if mode == "method":
                    if (
                        not inspect.ismethod(selected)
                        or selected.__self__ is not spec_obj
                    ):
                        raise HkpPackError(
                            f"{attribute} is not a bound effective accessor"
                        )
                    inspect.signature(selected).bind()
                    accessors[name] = origins.identity(selected)
                    value, repeated = selected(), selected()
                else:
                    if isinstance(
                        inspect.getattr_static(type(spec_obj), attribute, None),
                        property,
                    ):
                        raise HkpPackError(
                            f"{attribute} is a property, not a direct field"
                        )
                    value, repeated = selected, getattr(spec_obj, attribute)
                    accessors[name] = {"field": attribute, "owner": producer["spec"]}
                mapping = request["vocabulary"].get(name, {})
                if isinstance(value, str):
                    value = mapping.get(value, value)
                if isinstance(repeated, str):
                    repeated = mapping.get(repeated, repeated)
                value = canonical(value, request["types"][name], observed=True)
                repeated = canonical(repeated, request["types"][name], observed=True)
                if digest(value) != digest(repeated):
                    raise HkpPackError(f"non-repeatable effective accessor {attribute}")
                values[name] = value
            except HkpPackError:
                raise
            except Exception as exc:
                raise HkpPackError(
                    f"cannot observe {name} via {binding}: {exc}"
                ) from exc
        observations[key] = {"values": values, "accessors": accessors}
    return {"producer": producer, "requests": observations}


def compare(consumer, kmd, metadata, observations):
    key = digest(observation_request(consumer, kmd))
    if key not in observations["requests"]:
        raise HkpPackError("compile result lacks this consumer's observation")
    actual = observations["requests"][key]["values"]
    completed = complete_metadata(metadata, kmd)
    expected = {name: completed[name] for name in consumer["metadata_fields"]}
    if digest(actual) != digest(expected):
        raise HkpPackError(
            f"compiled specialization disagrees with metadata: observed={actual}, metadata={expected}"
        )
    return key


def descriptor_binding(ukd):
    doc = copy.deepcopy(ukd)
    doc.get("provenance", {}).pop("effective_spec", None)
    return digest(doc)


def publish(ukd, observations, records):
    """Write the producing compiler's evidence onto a shipped UKD.

    Only this function creates `provenance.effective_spec`, and it runs inside the
    invocation that actually compiled the payload. The authored `provenance.spec`
    sits beside it untouched: one is what was asked for, the other is what was
    observed, and collapsing them would lose the disagreement the whole check exists
    to find.

    `descriptor_binding` excludes the evidence from its own digest, so the record
    binds the descriptor it ships in without having to describe itself.
    """
    provenance = ukd["provenance"]
    authored = {key: provenance[key] for key in ("source", "builder", "spec")}
    provenance["effective_spec"] = {
        "schema_version": 1,
        "authored_digest": digest(authored),
        "observations": observations,
        "consumers": records,
        "descriptor_digest": descriptor_binding(ukd),
    }


def verify(ukd, records, payload):
    """Check one shipped UKD's evidence against the descriptors and bytes in hand.

    `records` is built from the CURRENT descriptors through `consumer_record` and
    `canonical_records`, so equality with the stored list is what fails a changed
    KMD, metadata, KDP header, declaration or effective architecture. Nothing here
    imports a producer: the evidence is self-contained, which is what lets a packed
    artifact be checked on a machine that has never had rocKE installed.

    Every deviation raises. A missing or unsupported record is a failure and never
    an unchecked property -- an artifact that cannot say what it was built from has
    not established agreement, which is the case this exists to reject.
    """
    record = ukd.get("provenance", {}).get("effective_spec")
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise HkpPackError("missing compiler-owned effective_spec evidence")
    if record.get("descriptor_digest") != descriptor_binding(ukd):
        raise HkpPackError("effective_spec descriptor binding mismatch")
    authored = {
        key: ukd["provenance"].get(key) for key in ("source", "builder", "spec")
    }
    if record.get("authored_digest") != digest(authored):
        raise HkpPackError("effective_spec authored-input binding mismatch")
    if record.get("consumers") != records:
        raise HkpPackError("effective_spec consumer binding mismatch")
    source = ukd["kernel_source"]
    payload_sha = hashlib.sha256(payload).hexdigest()
    if payload_sha != source.get("sha256"):
        raise HkpPackError("effective_spec payload binding mismatch")
    observations = record.get("observations", {})
    # The observations were taken on the object handed to the builder; these three
    # bind them to the artifact that came back out of it. Without them the record
    # would describe a compile that happened, with no evidence it is the compile
    # whose bytes this descriptor names.
    if observations.get("code_object_sha256") != payload_sha:
        raise HkpPackError("effective_spec code-object binding mismatch")
    if observations.get("symbol") != source.get("symbol"):
        raise HkpPackError("effective_spec symbol binding mismatch")
    if [observations.get("arch")] != ukd.get("arch"):
        raise HkpPackError("effective_spec architecture binding mismatch")
    producer = observations.get("producer", {})
    for role in ("builder", "spec"):
        identity = producer.get(role, {})
        if set(identity) != {"module", "qualname", "file", "sha256"} or not all(
            isinstance(v, str) and v for v in identity.values()
        ):
            raise HkpPackError("missing producing-object identity")
    for entry in records:
        compare(entry["declaration"], entry["kmd"], ukd["metadata"], observations)
