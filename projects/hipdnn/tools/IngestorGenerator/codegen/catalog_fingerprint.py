"""The identity of a kernel catalog that a trained ranker depends on.

A trained ``sort_kernel_catalog`` model is tied by the descriptor loader only to its engine's
UED, KMD and UMD ids (``trained_against``). A catalog edit that keeps those ids (a kernel
added, removed or renamed, a knob or spec changed) would leave the model ranking a catalog it
never saw, with no refusal. The engine config records this fingerprint of the catalog the
model was trained on (``engine.trained_catalog_sha256.<arch>``) and the
regeneration gates compare it with the shipped KDP's, so such an edit fails until the model is
retrained.

The fingerprint covers what selects and builds each kernel -- its name, metadata and kernel
spec(s) -- and nothing that a regeneration mints or reorders: descriptor ids and kernel order
are left out.
"""

from __future__ import annotations

import hashlib
import json


def _kernel_identity(kernel: dict) -> dict:
    source = kernel.get("kernel_source") or {}
    sources = kernel.get("sources") or {}
    return {
        "name": kernel["name"],
        "metadata": kernel.get("metadata") or {},
        "spec": source.get("spec"),
        "launch_specs": {name: launch.get("spec") for name, launch in sources.items()},
    }


def catalog_fingerprint(kdp: dict) -> str:
    """sha256 hex of a KDP's kernels, independent of descriptor ids and kernel order."""
    entries = sorted(
        json.dumps(_kernel_identity(k), sort_keys=True, separators=(",", ":"))
        for k in kdp.get("kernelDescriptors", [])
    )
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
