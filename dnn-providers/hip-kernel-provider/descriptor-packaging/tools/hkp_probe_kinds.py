"""The producer kinds a packaging probe supports: one entry per kernel_source.kind.

Shared by hkp_probe_derive_root.py (compile-group selection, expect.json) and
hkp_probe_assert.py (per-UKD checks). A kind with no entry is rejected by derive at
configure time and fails `no-rules-for-kind` in the assertion, so a new producer cannot
pass a probe without being described here.

Each entry also names the packed OUTPUT TYPE of the kind:
    kpack        the packer compiles/packs the UKD into the kpack archive and ships a
                 `kind: kpack` UKD (hip, rocke, hsaco)
    passthrough  the packer ships the UKD as authored; nothing enters the archive
                 (embedded_source)
A new kind whose output is one of these, with provenance checks that already exist by
name, is one entry in KINDS plus one end-to-end test case. A kind needing a new
provenance check adds it to hkp_probe_assert._PROVENANCE_CHECKS; a new output type adds
its check set to hkp_probe_assert._OUTPUT_CHECKS; a kind the packer does not support
needs packer work first. A UKD of an unregistered kind in a KDP that ships for a probed
arch makes configure fail; it must live in a KDP that does not ship for a probed arch,
or the probe needs a decision.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    # kernel_source fields that define a compile group within a KDP: UKDs of one kind
    # agreeing on all of them share a compile path. Every field must be present in
    # kernel_source. Empty means the kind compiles nothing and the whole KDP is one
    # group. Authors who want specific UKDs of a group packed list them by name
    # (hkp_add_packaging_probe UKDS).
    group_by: tuple[str, ...]
    # Names of provenance checks (hkp_probe_assert._PROVENANCE_CHECKS) a shipped UKD of
    # this kind must satisfy, beyond provenance.origin_kind == the kind itself.
    provenance: tuple[str, ...]
    # Packed output type: a key of hkp_probe_assert._OUTPUT_CHECKS.
    output: str = "kpack"
    # kernel_source field holding an object whose values select code within one
    # compile group (rocke: `spec`, which the builder bakes into the kernel). With a
    # sweep field, a group keeps UKDs until each value of each field that varies within
    # the group is packed at least once (greedy, ties by sorted name); combinations of
    # values are not guaranteed. None keeps one UKD per group, the first by sorted name.
    sweep_field: str | None = None


KINDS = {
    "rocke": Kind(
        group_by=("source", "builder"),
        provenance=("wheel", "comgr"),
        sweep_field="spec",
    ),
    "hip": Kind(group_by=("source", "build"), provenance=()),
    "hsaco": Kind(group_by=(), provenance=()),
    "embedded_source": Kind(group_by=(), provenance=(), output="passthrough"),
}
