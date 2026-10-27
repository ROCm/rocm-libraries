"""Multi-launch UKDs: a `sources` map in place of `kernel_source` (RFC 0017 §15.1).

A UKD carries exactly one of `kernel_source` (one launch) or `sources` (two or
more launches keyed by launch name). Every reader of a UKD's kernel goes through
`kernel_sources`, so a document that carries both, or neither, is refused in one
place rather than skipped by whichever reader looked at the wrong key.

An authored entry is a rocke `kernel_source` plus an optional `role` naming one of
the builder's companion kernels; without one it takes the builder's primary
kernel. A packed entry is a full kpack `kernel_source` with no `role`: its symbol
and signature are read out of the compiled object, never authored. A UKD with
`sources` reads at document version 1.1, which the packer stamps.

Launch order carries no meaning here: the pack's native dispatch handler launches
the entries, orders them, and owns the workspace between them.
"""

import re

from .errors import HkpPackError

# A launch name or a role names a kernel by purpose, so it is held to a lowercase
# identifier; the same rule `rocke.helpers.KernelModule` enforces on roles.
LAUNCH_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")

# The first UKD document version that may carry `sources`.
SOURCES_VERSION = "1.1"
_SOURCES_VERSION = (1, 1)

# Stands for the builder's primary kernel in claim lists. Not an identifier, so no
# role can spell it.
PRIMARY = "(primary)"

#: Entry kinds the packer refuses to take as authored `sources` entries.
UNSUPPORTED_AUTHORED_KINDS = ("hip", "hsaco", "embedded_source")


def refuse_authored_kind(where, name, kind):
    """The refusal for a `sources` entry that is not authored as rocke.

    Authored entries are rocke only. hip and hsaco are not supported yet,
    embedded_source is never packed into a code object, and kpack entries are
    already packed. Any other kind is not a kind at all.
    """
    head = f"{where} sources '{name}' has kind '{kind}'"
    if kind in ("hip", "hsaco"):
        tail = "which is not yet supported as a sources entry; only 'rocke' entries are"
    elif kind == "embedded_source":
        tail = "which is not packed into a code object, so it cannot be a sources entry"
    elif kind == "kpack":
        tail = "and already-packed entries are not pack input"
    else:
        tail = "which is not a sources entry kind (expected 'rocke')"
    return HkpPackError(f"{head}, {tail}")


def reads_sources(version):
    """Whether a UKD document `version` ('<major>.<minor>') may carry `sources`."""
    major, minor = version.split(".")
    return (int(major), int(minor)) >= _SOURCES_VERSION


def kernel_sources(ukd):
    """A UKD's kernel_source objects keyed by launch name.

    `{None: kernel_source}` for a single-launch UKD, `{name: kernel_source}` for a
    UKD with `sources`. Raises on a UKD carrying both or neither, and on a
    `sources` that is not a non-empty object, so no caller can read past a
    malformed pair.
    """
    has_ks = "kernel_source" in ukd
    has_sources = "sources" in ukd
    if has_ks == has_sources:
        raise HkpPackError(
            f"UKD '{ukd.get('id', '?')}' must carry exactly one of 'kernel_source' "
            f"or 'sources', found {'both' if has_ks else 'neither'}"
        )
    if has_ks:
        return {None: ukd["kernel_source"]}
    sources = ukd["sources"]
    if not isinstance(sources, dict) or not sources:
        raise HkpPackError(
            f"UKD '{ukd.get('id', '?')}' sources must be a non-empty object keyed "
            "by launch name"
        )
    return dict(sources)


def kind_of(ukd):
    """The one kernel_source kind of a UKD; every entry of `sources` shares it."""
    kinds = {ks.get("kind") for ks in kernel_sources(ukd).values()}
    if len(kinds) != 1:
        raise HkpPackError(
            f"UKD '{ukd.get('id', '?')}' sources mix kernel_source kinds "
            f"{sorted(map(str, kinds))}"
        )
    return kinds.pop()


def claimed_roles(roles):
    """The distinct kernels a variant's entries claim, as a sorted list.

    `roles` holds each entry's `role`, None for a primary claim. Entries naming
    one kernel under different launch names make one claim.
    """
    return sorted({PRIMARY if role is None else role for role in roles})


def compiled_roles(observations):
    """The kernels one compile produced: the primary plus every companion role."""
    return sorted([PRIMARY, *observations.get("companions", {})])


def check_claims(where, builder, names, roles, observations):
    """The kernels the entries on one variant claim must be what the compile made.

    A compiled companion no entry claims, or a claimed role the module lacks,
    fails here, naming both lists. `names` are the launches, `roles` their roles.
    """
    claimed = claimed_roles(roles)
    compiled = compiled_roles(observations)
    if claimed != compiled:
        raise HkpPackError(
            f"{where}: launches {sorted(names)} claim the kernels {claimed} of "
            f"builder '{builder}', but the compile produced {compiled}"
        )


def entry_symbol(role, observations):
    """The symbol one claimed kernel has in the compiled object."""
    if role is None:
        return observations["symbol"]
    return observations["companions"][role]
