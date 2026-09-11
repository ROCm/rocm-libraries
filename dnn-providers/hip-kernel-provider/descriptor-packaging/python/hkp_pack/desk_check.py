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

Three things this module got wrong in its first form, all invisible to its
own 179 tests and all found by pointing the CLI at a real 32-kernel bundle:

1. ``dtype`` compared two DELIBERATE vocabularies as if they were one. rocKE
   specs spell it ``"bf16"``; hipDNN metadata carries the enum name
   ``"BFLOAT16"`` (or, in the tiled bundle, ``"BF16"``). Both describe the
   same type, so a raw string compare false-positived on every rocKE kernel
   that ships. See ``_DTYPE_ALIASES`` -- the fix normalises the vocabularies
   rather than dropping the field, because dtype is the field most worth
   checking: ``spec "bf16"`` against ``metadata "HALF"`` is a real, fatal
   drift and still fails.
2. One field list fed BOTH invariant 1 and invariant 2. Narrowing it to
   silence a drift false-positive silently removed the same field from the
   matcher-tuple identity, manufacturing false collisions in the check whose
   entire job is catching unreachable variants. The two now take independent
   lists (``fields`` vs ``drift_fields``).
3. ``duplicate_matcher_tuples`` derived its field set from ``kernels[0]``
   alone, so a heterogeneous variant set either raised ``KeyError`` or --
   depending only on list order -- silently dropped a field from the tuple
   identity and reported false collisions. It now takes the union across all
   kernels and represents an absent field explicitly.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

# Two vocabularies describe one type: a rocKE spec spells the dtype the way
# the builder's Python takes it ("bf16"), while the KMD metadata carries the
# hipDNN DataType enum name the matcher compares against the graph
# ("BFLOAT16" -- projects/hipdnn/flatbuffers_sdk/schemas/data_types.fbs:6-26).
# Neither is wrong, and the difference is not drift. Normalising both sides
# through this table keeps the check live on the field most likely to drift
# for real: spec "bf16" against metadata "HALF" is a genuine, fatal mismatch
# and still reports. An unrecognised spelling on either side falls back to a
# plain case-insensitive compare, so an engine with its own vocabulary is
# still checked rather than waved through.
_DTYPE_ALIASES = {
    "BF16": "BFLOAT16",
    "BFLOAT16": "BFLOAT16",
    "FP16": "HALF",
    "HALF": "HALF",
    "FLOAT16": "HALF",
    "FP32": "FLOAT",
    "FLOAT": "FLOAT",
    "FLOAT32": "FLOAT",
    "FP64": "DOUBLE",
    "DOUBLE": "DOUBLE",
    "FP8E4M3": "FP8_E4M3",
    "FP8E5M2": "FP8_E5M2",
    "FP8E4M3FNUZ": "FP8_E4M3_FNUZ",
    "FP8E5M2FNUZ": "FP8_E5M2_FNUZ",
    "FP4E2M1": "FP4_E2M1",
    "FP6E2M3": "FP6_E2M3",
    "FP6E3M2": "FP6_E3M2",
}

# Sentinel for "this kernel does not declare that field at all", so a tuple
# identity can say so explicitly instead of silently shortening.
_ABSENT = "<absent>"


def _canonical_dtype(value) -> str:
    """A dtype spelling reduced to the one token both vocabularies mean, or
    the plain lowercased string when the spelling is not one this module
    knows -- an unknown vocabulary stays compared, never skipped."""
    token = "".join(ch for ch in str(value) if ch.isalnum()).upper()
    return _DTYPE_ALIASES.get(token, str(value).lower())


def _values_agree(field: str, spec_v, meta_v) -> bool:
    """One spec value against one metadata value, per-field.

    Booleans compare as ints because a KMD carries ``causal: 1`` for a spec's
    ``causal: True``; dtype compares through the vocabulary table; everything
    else is the case-insensitive string compare the original snippet did,
    which is what a numeric field wants."""
    if isinstance(spec_v, bool):
        return int(spec_v) == meta_v
    if field == "dtype":
        return _canonical_dtype(spec_v) == _canonical_dtype(meta_v)
    return str(spec_v).lower() == str(meta_v).lower()


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

    ``dtype`` is a SPELLING on both sides and the two sides speak different
    vocabularies on purpose -- a rocKE spec's ``"bf16"`` and a KMD's
    ``"BFLOAT16"`` are the same type. ``_values_agree`` normalises them, so
    this stays a live check on the field rather than a wall of false
    positives (spec ``"bf16"`` against metadata ``"HALF"`` still fails).
    This function's `fields` is INDEPENDENT of the matcher-tuple identity
    used by `duplicate_matcher_tuples`: narrowing one must never silently
    narrow the other.
    """
    bad = []
    for k in kernels:
        spec = _authored_spec(k)
        meta = k["metadata"]
        for f in fields:
            if f not in spec or f not in meta:
                continue
            if not _values_agree(f, spec[f], meta[f]):
                bad.append((k["name"], f))
    return bad


def duplicate_matcher_tuples(
    kernels: list[dict], fields=DEFAULT_MATCHER_FIELDS
) -> dict[tuple, int]:
    """Invariant 2: no two kernels may share a matcher tuple on the same
    arch -- one of them is unreachable. Returns {tuple: count} for every
    tuple shared by more than one kernel (empty means none).

    The compared field set is the UNION of `fields` present in ANY kernel's
    metadata, not the fields of ``kernels[0]``. Keying off the first kernel
    made the tuple identity depend on list order: a set where only a later
    kernel declared a field either raised ``KeyError`` or silently dropped
    that field from the identity and reported collisions that do not exist.
    A kernel that does not declare a field in the union gets `_ABSENT` for
    it, which is itself distinguishing -- "declares no block_n" and
    "declares block_n=64" are genuinely different variants.
    """
    present = [f for f in fields if any(f in k.get("metadata", {}) for k in kernels)]
    tups = collections.Counter(
        tuple(k.get("metadata", {}).get(f, _ABSENT) for f in present) for k in kernels
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

    `fields` is the MATCHER-TUPLE identity (invariant 2). `drift_fields` is
    the set invariant 1 compares against the spec, and defaults to `fields`
    only because they usually coincide. They are separate parameters because
    one list feeding both is a trap: narrowing the comparison to silence a
    drift report used to delete the same field from the tuple identity and
    manufacture false collisions in the check whose entire job is catching
    unreachable variants. Narrow one, and the other is untouched.
    """

    def __init__(
        self,
        kernels: list[dict],
        fields=DEFAULT_MATCHER_FIELDS,
        drift_fields=None,
    ):
        self.kernel_count = len(kernels)
        self.fields = tuple(fields)
        self.drift_fields = self.fields if drift_fields is None else tuple(drift_fields)
        self.spec_drift_error: str | None = None
        self.drift: list[tuple[str, str]] = []
        try:
            self.drift = metadata_spec_drift(kernels, self.drift_fields)
        except DeskCheckNoSpecFound as exc:
            self.spec_drift_error = str(exc)
        self.duplicate_tuples = duplicate_matcher_tuples(kernels, self.fields)

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
