"""Companion kernels: further kernels exported by a rocke UKD's own code object.

A rocke builder may return a `rocke.helpers.KernelModule` -- a primary kernel
plus role-keyed companions (a split-KV decode and the kernel merging its splits,
say) -- compiled into ONE code object. The UKD declares the roles it expects in
`kernel_source.companion_kernels`, and the packer surfaces each role in the
shipped metadata as two producer-written string fields:

- `<role>_symbol`: the companion's symbol in the same code object, the one the
  primary's `kernel_source.symbol` names;
- `<role>_signature`: its argument list as compact JSON, in exactly the shape of
  `kernel_source.signature` (read out of the object by `kernel_signature`).

A native derives the companion's descriptor from the primary's by swapping the
symbol and signature, which keeps `requireSignatureMatch` meaningful for it.

Role-keyed fields rather than one list field: KMD metadata has no list-of-strings
type, and a native asks for its companion by purpose, not by position. The
fields are matcher-only (no compile input binds them) and authored metadata may
not supply them, because only the compiled object knows their values.
"""

import json
import re

from .errors import HkpPackError
from .kernel_signature import kernel_signature

# A role names metadata fields, so it is held to a lowercase identifier; the
# same rule `rocke.helpers.KernelModule` enforces on the producing side.
_ROLE = re.compile(r"[a-z][a-z0-9_]*\Z")


def metadata_fields(role):
    """`(symbol field, signature field)` one companion role writes into metadata."""
    return f"{role}_symbol", f"{role}_signature"


def declared_roles(ukd):
    """The companion roles a rocke UKD declares; empty for every other kind."""
    ks = ukd["kernel_source"]
    if ks.get("kind") != "rocke":
        return []
    return list(ks.get("companion_kernels", []))


def produced_fields(ukd):
    """Every metadata field the producer writes for this UKD's companions."""
    return [name for role in declared_roles(ukd) for name in metadata_fields(role)]


def validate_roles(roles, where):
    """A `companion_kernels` value: a non-empty list of distinct role identifiers."""
    if (
        not isinstance(roles, list)
        or not roles
        or any(not isinstance(r, str) or not _ROLE.match(r) for r in roles)
        or len(set(roles)) != len(roles)
    ):
        raise HkpPackError(
            f"{where} kernel_source has invalid companion_kernels {roles!r} "
            "(expected a non-empty list of distinct lowercase identifiers)"
        )


def encode_signature(signature):
    """One argument list as the compact JSON a `<role>_signature` field holds."""
    return json.dumps(signature, separators=(",", ":"))


def companion_metadata(code_object, companions, where):
    """`{field: value}` for `companions` (`{role: symbol}`) read off `code_object`.

    Each signature is read out of the object itself, exactly as the primary's
    `kernel_source.signature` is, so a companion missing from the object fails
    here rather than at launch.
    """
    produced = {}
    for role, symbol in companions.items():
        symbol_field, signature_field = metadata_fields(role)
        produced[symbol_field] = symbol
        produced[signature_field] = encode_signature(
            kernel_signature(code_object, symbol, f"{where} companion '{role}'")
        )
    return produced


def check_authored(ukd, kmd, where):
    """The authored side of a UKD's companions, before anything compiles.

    Authored metadata may not state a produced field, and the consuming KMD must
    declare each one as a string -- the loader drops a pack whose UKD supplies an
    undeclared field. Returns the produced field names.
    """
    produced = produced_fields(ukd)
    supplied = sorted(set(produced) & set(ukd["metadata"]))
    if supplied:
        raise HkpPackError(
            f"{where}: authored metadata supplies {supplied}, which the packer "
            "writes from the compiled companion kernels"
        )
    declared = {f["name"]: f.get("type") for f in kmd["fields"]}
    missing = [name for name in produced if declared.get(name) != "string"]
    if missing:
        raise HkpPackError(
            f"{where}: kernel_source.companion_kernels needs KMD '{kmd.get('id')}' "
            f"to declare string metadata field(s) {missing}"
        )
    return produced


def check_observed(ukd, observations, where):
    """The compiled companions must be exactly the roles the UKD declares.

    Returns `{role: symbol}` in the UKD's declared order.
    """
    declared = declared_roles(ukd)
    observed = observations.get("companions", {})
    if sorted(declared) != sorted(observed):
        raise HkpPackError(
            f"{where}: kernel_source.companion_kernels declares {declared}, but "
            f"builder '{ukd['kernel_source'].get('builder')}' emitted companions "
            f"{sorted(observed)}"
        )
    return {role: observed[role] for role in declared}
