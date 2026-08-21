import hashlib
import json
from pathlib import Path


def variant_key(source, build):
    """Stable input hash over (source, build), independent of toolchain.

    Drives both the toc_key and the intermediate .co filename. Keyed on
    (source, build) only: two UKDs sharing source+build (differing entry) hash
    identically and therefore share one blob; a different build hashes apart.

    Producer-agnostic: keyed on the authored (source, build) inputs, not on any
    hip-specific detail, so non-hip producers reuse the same addressing rule.
    """
    payload = json.dumps(
        {"source": source, "build": build},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    stem = Path(source).stem
    return f"{stem}_{digest}"
