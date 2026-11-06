"""Print the catalog fingerprint of a KDP, the value an engine config records as
``engine.trained_catalog_sha256.<arch>`` when a ranker is trained on it.

    python3 tools/catalog_fingerprint.py <bundle>/artifacts/<pack>.kdp.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import catalog_diff


def catalog_fingerprint(kdp: dict) -> str:
    """sha256 hex of a KDP's kernels, independent of descriptor ids and kernel order.

    Keyed on the same identity ``catalog_diff``'s regeneration gate (g3) uses --
    ``catalog_diff.entry_key``: the full ``kernel_source``/``sources`` (every key,
    not just ``spec``) plus ``arch``, ``metadata``, ``priority`` and ``version``.
    A builder, target, arch or priority edit the regeneration gate catches
    therefore changes this fingerprint too, so a trained ranker is never left
    silently ranking a catalog it never saw.
    """
    entries = sorted(
        json.dumps(catalog_diff.entry_key(k), sort_keys=True, separators=(",", ":"))
        for k in kdp.get("kernelDescriptors", [])
    )
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kdp", type=Path, help="the KDP the ranker was trained on")
    args = parser.parse_args(argv)
    print(catalog_fingerprint(json.loads(args.kdp.read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
