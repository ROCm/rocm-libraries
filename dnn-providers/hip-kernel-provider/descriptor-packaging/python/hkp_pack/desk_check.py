"""RUNBOOK.md step 5d's four desk-check invariants, as real, importable code.

Extracted from a shell-embedded Python snippet the RUNBOOK carried in prose.
Prose could not be tested, so nothing tested it: invariant 1 read
``kernel_source.spec``, which packing rewrites away (the authored spec moves
to ``provenance.spec``; ``kernel_source`` becomes ``{kind: kpack, library,
toc_key, symbol, sha256}``), so on the exact packed tree the step told an
agent to point it at, the check silently printed "none" regardless of real
drift. This module is the fix, plus the three invariants that were already
correct, all in one place a test can import instead of copy.

Runs over a single loaded KDP document's ``kernelDescriptors`` list -- works
on an authored (pre-pack) tree via ``kernel_source.spec`` or a shipped
(post-pack) one via ``provenance.spec`` interchangeably, and treats "neither
location has a spec" as a distinct, reported outcome rather than a silent
"no drift".
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

# The KMD fields a desk-check typically compares. Callers should narrow this
# to fields their own KMD actually declares (see `--field`); it is a default,
# not something this module can discover on its own -- there is no schema
# object here to introspect, only a list of kernel dicts.
DEFAULT_MATCHER_FIELDS = (
    "dtype",
    "batch",
    "head_size",
    "num_query_heads",
    "num_kv_heads",
    "seqlen_q",
    "seqlen_kv",
    "causal",
    "sliding_window",
    "block_n",
)


class DeskCheckNoSpecFound(RuntimeError):
    """Raised when a kernel's authored spec cannot be found anywhere this
    check knows to look (neither ``kernel_source.spec`` nor
    ``provenance.spec``) -- distinct from finding a spec that agrees with
    metadata, which is a genuine "no drift" result. Conflating the two was
    exactly how the original invariant went dead: "found nothing to check"
    and "checked, found nothing wrong" rendered identically."""


def load_kernels(kdp_path: Path) -> list[dict]:
    """Load a `.kdp.json`'s ``kernelDescriptors`` list."""
    doc = json.loads(Path(kdp_path).read_text(encoding="utf-8"))
    return doc["kernelDescriptors"]


def _authored_spec(kernel: dict) -> dict:
    ks_spec = kernel.get("kernel_source", {}).get("spec")
    if ks_spec is not None:
        return ks_spec
    prov_spec = kernel.get("provenance", {}).get("spec")
    if prov_spec is not None:
        return prov_spec
    raise DeskCheckNoSpecFound(
        f"kernel '{kernel.get('name')}' has no spec in kernel_source OR "
        "provenance -- wrong tree, or a non-rocke producer?"
    )


def metadata_spec_drift(
    kernels: list[dict], fields=DEFAULT_MATCHER_FIELDS
) -> list[tuple[str, str]]:
    """Invariant 1: metadata must agree with the spec it claims to describe.

    The matcher reads ``metadata``; the compiler read ``spec``. A drift
    between them is invisible (nothing errors) and fatal (the kernel that
    runs is not the kernel the matcher thinks it picked). Checks whichever of
    ``kernel_source.spec`` (authored tree) or ``provenance.spec`` (packed
    tree) is present per kernel; raises `DeskCheckNoSpecFound` if a kernel has
    neither, rather than silently treating it as clean.

    dtype is commonly a SPELLING, not a raw value (one integration paired
    spec "bf16"/"fp16" with metadata "BFLOAT16"/"HALF") -- if your engine
    translates the dtype vocabulary, drop "dtype" from `fields` or the row
    false-positives.
    """
    bad = []
    for k in kernels:
        spec = _authored_spec(k)
        meta = k["metadata"]
        for f in fields:
            if f not in spec or f not in meta:
                continue
            spec_v, meta_v = spec[f], meta[f]
            if isinstance(spec_v, bool):
                if int(spec_v) != meta_v:
                    bad.append((k["name"], f))
            elif str(spec_v).lower() != str(meta_v).lower():
                bad.append((k["name"], f))
    return bad


def duplicate_matcher_tuples(
    kernels: list[dict], fields=DEFAULT_MATCHER_FIELDS
) -> dict[tuple, int]:
    """Invariant 2: no two kernels may share a matcher tuple on the same
    arch -- one of them is unreachable. Returns {tuple: count} for every
    tuple shared by more than one kernel (empty means none)."""
    present = [f for f in fields if kernels and f in kernels[0].get("metadata", {})]
    tups = collections.Counter(
        tuple(k["metadata"][f] for f in present) for k in kernels
    )
    return {t: c for t, c in tups.items() if c > 1}


def toc_key_uniqueness(kernels: list[dict]) -> tuple[int, int]:
    """Invariant 3: every variant individually addressable in the archive.
    Returns (distinct toc_key count, kernel count); equal means OK.

    Only meaningful once ``toc_key`` exists, i.e. post-pack -- see
    `_field_applicable` for the pre-pack "not yet assigned" case, which the
    report (not this function) is responsible for distinguishing from a
    genuine collision."""
    toc = [k.get("kernel_source", {}).get("toc_key") for k in kernels]
    return len(set(toc)), len(kernels)


def symbol_distinctness(kernels: list[dict]) -> tuple[int, int]:
    """Invariant 4 (informational, NOT a failure condition): symbol names are
    not guaranteed unique -- rocKE's ``kernel_name()`` may omit a field it
    still bakes in. Uniqueness comes from (toc_key, symbol), never the symbol
    alone. Returns (distinct symbol count, kernel count); fewer is legal."""
    sym = [k.get("kernel_source", {}).get("symbol") for k in kernels]
    return len(set(sym)), len(kernels)


def _field_applicable(kernels: list[dict], field: str) -> bool:
    """False when NOT ONE kernel's ``kernel_source`` carries `field` at all --
    the normal, expected shape of an AUTHORED (pre-pack) tree, where
    ``toc_key``/``symbol`` are assigned by packing and simply do not exist
    yet. True (applicable) the moment even one kernel carries it, so a
    heterogeneous tree (some packed, some not) still gets checked rather
    than silently waved through as "not applicable"."""
    return any(field in k.get("kernel_source", {}) for k in kernels)


class DeskCheckReport:
    """All four invariants over one kernel list, plus a pass/fail verdict.

    Works on both an authored (pre-pack) tree and a shipped (post-pack) one:
    invariants 3 and 4 key on ``toc_key``/``symbol``, which packing assigns,
    so on an authored tree they report NOT-APPLICABLE rather than a false
    "all None -- collision". Invariant 4 is informational and never fails
    the report even when applicable -- a shared symbol with distinct
    toc_keys is a documented, tolerated shape, not a defect.
    """

    def __init__(self, kernels: list[dict], fields=DEFAULT_MATCHER_FIELDS):
        self.kernel_count = len(kernels)
        self.fields = tuple(fields)
        self.spec_drift_error: str | None = None
        self.drift: list[tuple[str, str]] = []
        try:
            self.drift = metadata_spec_drift(kernels, fields)
        except DeskCheckNoSpecFound as exc:
            self.spec_drift_error = str(exc)
        self.duplicate_tuples = duplicate_matcher_tuples(kernels, fields)

        self.toc_applicable = _field_applicable(kernels, "toc_key")
        self.toc_distinct, self.toc_total = (
            toc_key_uniqueness(kernels) if self.toc_applicable else (0, 0)
        )
        self.symbol_applicable = _field_applicable(kernels, "symbol")
        self.symbol_distinct, self.symbol_total = (
            symbol_distinctness(kernels) if self.symbol_applicable else (0, 0)
        )

    @property
    def ok(self) -> bool:
        """False on any invariant this check can actually enforce failing.
        A COULD-NOT-CHECK spec-drift result also fails the report -- it is
        not a clean bill of health, it is a check that could not run, and
        reporting it as green is the exact defect this module exists to
        remove. toc_key NOT-APPLICABLE (pre-pack tree) does NOT fail the
        report -- that is an expected state, not an unchecked one."""
        toc_ok = (not self.toc_applicable) or (self.toc_distinct == self.toc_total)
        return (
            self.spec_drift_error is None
            and not self.drift
            and not self.duplicate_tuples
            and toc_ok
        )

    def render(self) -> str:
        lines = [f"kernels={self.kernel_count}"]
        if self.spec_drift_error is not None:
            lines.append(
                f"metadata/spec drift: COULD-NOT-CHECK -- {self.spec_drift_error}"
            )
        else:
            lines.append(f"metadata/spec drift: {self.drift or 'none'}")
        lines.append(
            "duplicate matcher tuples: " + str(self.duplicate_tuples or "none")
        )
        if not self.toc_applicable:
            lines.append(
                "toc_key: NOT-APPLICABLE -- no kernel_source carries toc_key "
                "(pre-pack tree)"
            )
        else:
            lines.append(
                f"toc_key: distinct={self.toc_distinct} of {self.toc_total} "
                + ("OK" if self.toc_distinct == self.toc_total else "COLLISION")
            )
        if not self.symbol_applicable:
            lines.append(
                "symbols: NOT-APPLICABLE -- no kernel_source carries symbol "
                "(pre-pack tree)"
            )
        else:
            lines.append(
                f"symbols: distinct={self.symbol_distinct} of {self.symbol_total} "
                "(fewer is legal -- toc_key disambiguates)"
            )
        return "\n".join(lines)
