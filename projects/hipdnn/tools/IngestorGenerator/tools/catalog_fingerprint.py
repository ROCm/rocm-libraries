"""Print the catalog fingerprint of a KDP, the value an engine config records as
``engine.trained_catalog_sha256.<arch>`` when a ranker is trained on it. A slot ranker
(``slots[].trained_catalog_sha256.<arch>``) is trained on every pack serving its slot on
that arch: name each of those KDPs.

    python3 tools/catalog_fingerprint.py <bundle>/artifacts/<pack>.kdp.json [<kdp>...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import catalog_diff


def catalog_fingerprint(kdp: dict) -> str:
    """The fingerprint of one KDP's catalog (see `catalog_fingerprints`)."""
    return catalog_fingerprints([kdp])


def catalog_fingerprints(kdps) -> str:
    """sha256 hex of the kernels of @p kdps together, independent of descriptor ids,
    kernel order and how the kernels are split across KDPs.

    Keyed on the same identity ``catalog_diff``'s regeneration gate (g3) uses --
    ``catalog_diff.entry_key``: the full ``kernel_source``/``sources`` (every key,
    not just ``spec``) plus ``arch``, ``metadata``, ``priority`` and ``version``.
    A builder, target, arch or priority edit the regeneration gate catches
    therefore changes this fingerprint too, so a trained ranker is never left
    silently ranking a catalog it never saw.
    """
    entries = sorted(
        json.dumps(catalog_diff.entry_key(k), sort_keys=True, separators=(",", ":"))
        for kdp in kdps
        for k in kdp.get("kernelDescriptors", [])
    )
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "kdps", type=Path, nargs="+", help="the KDP(s) the ranker was trained on"
    )
    args = parser.parse_args(argv)
    print(catalog_fingerprints(json.loads(p.read_text()) for p in args.kdps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
