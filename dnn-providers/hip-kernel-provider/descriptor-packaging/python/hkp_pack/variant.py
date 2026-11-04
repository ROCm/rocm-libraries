import hashlib
import json


def _hash_payload(stem, payload):
    """Byte-stable variant key: '<stem>_<sha256(payload)[:12]>'.

    The shared hashing core each producer's type-specific key function calls.
    payload is serialized with sort_keys=True so a nested dict (e.g. a rocke
    spec) hashes deterministically at every level regardless of key order.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"
