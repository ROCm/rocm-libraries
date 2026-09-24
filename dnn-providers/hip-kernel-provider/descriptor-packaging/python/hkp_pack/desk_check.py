"""The four desk-check invariants of the packaging README's "Desk-check a variant
set", as real, importable code.

Runs over one loaded KDP document's ``kernelDescriptors`` list, reading an authored
tree's ``kernel_source.spec`` or a shipped tree's ``provenance.spec``
interchangeably, and reports "neither location has a spec" as its own outcome.
"""

from __future__ import annotations

import collections
from pathlib import Path

from . import agreement, descriptor_context
from .errors import HkpPackError
from .kpack_resolver import load_kpack

# Two vocabularies describe one type: a rocKE spec spells the dtype the way the
# builder's Python takes it ("bf16"), while KMD metadata carries the hipDNN
# DataType enum name ("BFLOAT16" --
# projects/hipdnn/flatbuffers_sdk/schemas/data_types.fbs:6-26). Normalising both
# sides keeps dtype a live check; an unrecognised spelling falls back to a
# case-insensitive compare rather than being waved through.
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
    """A dtype spelling reduced to the token both vocabularies mean, or the
    lowercased string when the spelling is unknown -- an unknown vocabulary stays
    compared, never skipped."""
    token = "".join(ch for ch in str(value) if ch.isalnum()).upper()
    return _DTYPE_ALIASES.get(token, str(value).lower())


def _values_agree(field: str, spec_v, meta_v) -> bool:
    """One spec value against one metadata value, per-field.

    Booleans compare as ints because a KMD carries ``causal: 1`` for a spec's
    ``causal: True``; dtype compares through the vocabulary table; everything else
    is a case-insensitive string compare."""
    if isinstance(spec_v, bool):
        return int(spec_v) == meta_v
    if field == "dtype":
        return _canonical_dtype(spec_v) == _canonical_dtype(meta_v)
    return str(spec_v).lower() == str(meta_v).lower()


# One engine's attention-shaped field list. Nothing resolves to it: a bundle's own
# declaration answers first and `metadata_identity_fields` otherwise, because a
# generic guess collapses distinct kernels onto one matcher tuple. Kept as the
# reference the tests measure that failure against, and as a list `--field` supplies.
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
    """Raised when a kernel's authored spec is in neither ``kernel_source.spec``
    nor ``provenance.spec`` -- distinct from a spec that agrees with metadata.
    Conflating "found nothing to check" with "checked, found nothing wrong" is a
    dead check."""


def _selected(kdp_path: Path, matches: list):
    """The one indexed KDP the caller's path names.

    A path matching nothing raises `HkpPackError` naming the path, rather than
    escaping a gate as a `StopIteration` traceback.
    """
    if not matches:
        raise HkpPackError(
            f"{kdp_path}: the descriptor index under {kdp_path.parent} holds no "
            f"KDP at this path -- the argument must name an existing "
            f"`.kdp.json` file."
        )
    return matches[0]


def _resolve(kdp_path: Path) -> tuple[dict, list[descriptor_context.Entry]]:
    """One `.kdp.json`'s own document and its resolved entries."""
    kdp_path = Path(kdp_path).resolve()
    index = descriptor_context.Index(str(kdp_path.parent))
    kdp = _selected(
        kdp_path, [d for d in index.of_type("kdp") if Path(d.path) == kdp_path]
    )
    return kdp.doc, descriptor_context.resolve_entries(index, kdp)


def load_kernels(kdp_path: Path) -> list[dict]:
    """A `.kdp.json`'s kernel descriptors, standalone-UKD references resolved."""
    return [entry.ukd for entry in _resolve(kdp_path)[1]]


def declared_matcher_fields(kdp_doc: dict, entries) -> tuple[str, ...] | None:
    """The matcher-tuple identity this bundle declares for itself, in declaration
    order, or None when no entry declares a contract.

    The specialization contract states what the producing compiler specialized on,
    so it, not a generic guess, distinguishes variants. Unioned across entries,
    since one shard may carry several consumers. Both halves count:
    `validate_consumer` makes them exhaust the KMD and `KernelIngestorStateManager`
    keys its catalog on the whole tuple.

    `agreement.resolved_contract` offers the enclosing KDP only to inline entries,
    since a standalone UKD inherits nothing.
    """
    fields: list[str] = []
    for entry in entries:
        contract = agreement.resolved_contract(
            entry.ukd, kdp_doc if entry.inline else None
        )
        if not isinstance(contract, dict):
            continue
        for consumer in contract.get("consumers") or []:
            if not isinstance(consumer, dict):
                continue
            for key in ("metadata_fields", "matcher_only_fields"):
                for field in consumer.get(key) or []:
                    if field not in fields:
                        fields.append(field)
    return tuple(fields) or None


def metadata_identity_fields(kernels: list[dict]) -> tuple[str, ...]:
    """Every field any kernel states in its metadata, first-appearance order.

    The identity for a bundle declaring no contract. Derived rather than fixed: a
    fixed list describes one engine's shape.
    """
    fields: list[str] = []
    for kernel in kernels:
        for field in kernel.get("metadata") or {}:
            if field not in fields:
                fields.append(field)
    return tuple(fields)


def load_variant_set(kdp_path: Path) -> tuple[list[dict], tuple[str, ...] | None]:
    """A `.kdp.json`'s kernel descriptors plus the matcher fields it declares.

    Both come from one walk of the descriptor tree: a shipped shard's KDP runs to
    megabytes.
    """
    kdp_doc, entries = _resolve(kdp_path)
    return (
        [entry.ukd for entry in entries],
        declared_matcher_fields(kdp_doc, entries),
    )


def _payload(
    entry: descriptor_context.Entry, arch: str, kpack_python_dir=None
) -> bytes:
    """The archive bytes this descriptor names, read from the archive itself.

    Comparing the descriptor's own ``sha256`` against a digest of that field
    establishes nothing. Needs the archive reader only, never the producer.
    """
    kernel = entry.ukd
    source = kernel.get("kernel_source", {})
    library = (Path(entry.origin_dir) / source.get("library", "")).resolve()
    if not library.is_file():
        raise HkpPackError(
            f"kernel '{kernel.get('name')}' names library '{source.get('library')}', "
            f"which is not a file at {library}"
        )
    kpack, _compression = load_kpack(kpack_python_dir)
    try:
        archive = kpack.PackedKernelArchive.read(library)
        blob = archive.get_kernel(source.get("toc_key"), arch)
    except Exception as exc:
        raise HkpPackError(
            f"kernel '{kernel.get('name')}': cannot read {library}: {exc}"
        ) from exc
    if blob is None:
        raise HkpPackError(
            f"kernel '{kernel.get('name')}': toc_key '{source.get('toc_key')}' is "
            f"absent from {library} for {arch}"
        )
    return bytes(blob)


def compiled_agreement(
    kdp_path: Path, kpack_python_dir=None
) -> tuple[list[str], list[str], int]:
    """Compiled-specialization agreement over one shipped KDP.

    Checks the declaration and the producing-build record against the descriptors
    and archive bytes in hand; nothing imports the producer, so a valid artifact
    verifies where rocKE was never installed. An artifact that cannot present a
    record fails, as does a non-kpack kernel, which has no bytes to bind.

    Returns `(failures, unclaimed, verified)`. A packed declaration with no
    `metadata_fields` is the legitimate shape for a non-compiled source and counts
    as unclaimed; only rocKE-origin kernels carry that evidence today.

    The waiver is keyed on origin: `provenance.origin_kind == "rocke"` was
    published with its evidence, so the same shape fails there -- otherwise a
    descriptor could retire its evidence by dropping `effective_spec` and moving
    its `metadata_fields` into `matcher_only_fields`. An absent `origin_kind` is
    not rocKE. That reaches evidence lost by accident, not removed on purpose:
    `origin_kind` is bound only by a digest inside the record being dropped.
    """
    kdp_path = Path(kdp_path).resolve()
    index = descriptor_context.Index(str(kdp_path.parent))
    schemas = index.schemas()
    bundles = descriptor_context.resolve_bundles(index)
    bundle = _selected(kdp_path, [b for b in bundles if Path(b.kdp_path) == kdp_path])
    doc, engine, kmd = bundle.kdp_doc, bundle.engine, bundle.kmd
    arches = doc.get("arch") or []
    if len(arches) != 1:
        return (
            [
                f"{kdp_path.name}: a shipped shard carries exactly one arch, not "
                f"{arches!r}"
            ],
            [],
            0,
        )
    arch = arches[0]
    all_records = descriptor_context.consumer_records(bundles, schemas, arch)
    failures: list[str] = []
    unclaimed: list[str] = []
    verified = 0
    for entry in bundle.entries:
        kernel = entry.ukd
        name = kernel.get("name")
        try:
            kind = kernel.get("kernel_source", {}).get("kind")
            if kind != "kpack":
                raise HkpPackError(
                    f"--mode full needs the packed dialect, and kernel_source.kind "
                    f"is {kind!r}. The producing compiler's evidence exists only "
                    f"once the bytes do; check the packed tree."
                )
            agreement.select_declaration(
                kernel, engine, kmd, schemas, doc if entry.inline else None
            )
            records = all_records[kernel["id"]]
            provenance = kernel.get("provenance") or {}
            claimed = any(r["declaration"]["metadata_fields"] for r in records)
            if not claimed and "effective_spec" not in provenance:
                # The packer publishes `effective_spec` onto every rocKE UKD it
                # ships, so only a non-rocKE origin may waive.
                if provenance.get("origin_kind") == "rocke":
                    raise HkpPackError(
                        "provenance.origin_kind is 'rocke', so the packer published "
                        "this kernel's compiler-owned provenance.effective_spec when "
                        "it shipped it. The descriptor in hand declares no "
                        "specialized metadata_fields AND carries no effective_spec, "
                        "so there is no record left to bind and the archive bytes "
                        "were never read. A rocKE-produced kernel is required to "
                        "carry its compiler evidence; relabelling its specialized "
                        "fields as matcher-only does not make it an unspecialized "
                        "source."
                    )
                unclaimed.append(
                    f"{name}: declares no specialized metadata_fields, so there is "
                    f"no producing-build record to bind and nothing here was "
                    f"verified against a binary"
                )
                continue
            payload = _payload(entry, arch, kpack_python_dir)
            agreement.verify(kernel, records, payload)
            verified += 1
        except HkpPackError as exc:
            failures.append(f"{name}: {exc}")
    return failures, unclaimed, verified


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


def drift_comparable_fields(kernels: list[dict]) -> tuple[str, ...]:
    """Every field invariant 1 can compare: one carrying BOTH a spec value and a
    metadata value on at least one kernel, unioned in first-appearance order.

    Derived from the descriptors rather than the declared contract, because that
    declaration is one of the things invariant 1 polices: a narrow declaration
    would confine the audit to the fields the artifact chose to mention. Widest is
    nearly free, since `metadata_spec_drift` skips a field missing from either
    side; the residual cost is a deliberately translated field reporting as drift,
    which `--drift-field` narrows.

    A kernel with no spec contributes nothing rather than raising, leaving the
    COULD-NOT-CHECK verdict to `metadata_spec_drift`.
    """
    fields: list[str] = []
    for kernel in kernels:
        try:
            spec = _authored_spec(kernel)
        except DeskCheckNoSpecFound:
            continue
        metadata = kernel.get("metadata") or {}
        for field in spec:
            if field in metadata and field not in fields:
                fields.append(field)
    return tuple(fields)


def metadata_spec_drift(kernels: list[dict], fields=None) -> list[tuple[str, str]]:
    """Invariant 1: metadata must agree with the spec it claims to describe.

    The matcher reads ``metadata``; the compiler read ``spec``. A drift between
    them is invisible and fatal. Checks whichever of ``kernel_source.spec``
    (authored) or ``provenance.spec`` (packed) is present, raising
    `DeskCheckNoSpecFound` when a kernel has neither. ``dtype`` spellings differ on
    purpose and `_values_agree` normalises them.

    `fields` is independent of `duplicate_matcher_tuples`' identity: narrowing one
    must never narrow the other. `None` means `drift_comparable_fields`, and
    specifically not the declared contract, which is an input to this check.
    """
    if fields is None:
        fields = drift_comparable_fields(kernels)
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


def _reachable_together(group: list[dict]) -> int:
    """The largest number of kernels in `group` one device reaches.

    A tuple shared across disjoint arches is no collision. An absent or empty
    `arch` is a wildcard and counts against every arch in the group.
    """
    sets = [frozenset(k.get("arch") or ()) for k in group]
    named = frozenset().union(*sets) if sets else frozenset()
    if not named:
        return len(group)
    return max(sum(1 for s in sets if not s or arch in s) for arch in named)


def duplicate_matcher_tuples(
    kernels: list[dict], fields=DEFAULT_MATCHER_FIELDS
) -> dict[tuple, int]:
    """Invariant 2: no two kernels may share a matcher tuple on the same arch --
    one is unreachable. Returns {tuple: count} for every tuple two kernels reach
    one device with, the scope the runtime refuses in.

    The compared set is the union of `fields` present in any kernel's metadata,
    never ``kernels[0]``'s, which would make the identity list-order dependent. A
    kernel not declaring a field gets `_ABSENT`, itself distinguishing.
    """
    present = [f for f in fields if any(f in k.get("metadata", {}) for k in kernels)]
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for kernel in kernels:
        key = tuple(kernel.get("metadata", {}).get(f, _ABSENT) for f in present)
        groups[key].append(kernel)
    counts = {t: _reachable_together(g) for t, g in groups.items()}
    return {t: c for t, c in counts.items() if c > 1}


def toc_key_uniqueness(kernels: list[dict]) -> tuple[int, int]:
    """Invariant 3: every variant individually addressable in the archive.
    Returns (distinct toc_key count, kernel count); equal means OK.

    Only meaningful once ``toc_key`` exists, i.e. post-pack. `_field_applicable`
    covers the pre-pack case, which the report distinguishes from a collision."""
    toc = [k.get("kernel_source", {}).get("toc_key") for k in kernels]
    return len(set(toc)), len(kernels)


def symbol_distinctness(kernels: list[dict]) -> tuple[int, int]:
    """Invariant 4 (informational, NOT a failure condition): symbol names are not
    guaranteed unique -- rocKE's ``kernel_name()`` may omit a field it still bakes
    in. Uniqueness comes from (toc_key, symbol). Returns (distinct symbol count,
    kernel count); fewer is legal."""
    sym = [k.get("kernel_source", {}).get("symbol") for k in kernels]
    return len(set(sym)), len(kernels)


def _field_applicable(kernels: list[dict], field: str) -> bool:
    """False when no kernel's ``kernel_source`` carries `field` -- the expected
    shape of an authored tree, where ``toc_key``/``symbol`` do not exist yet. True
    as soon as one kernel carries it, so a heterogeneous tree is still checked."""
    return any(field in k.get("kernel_source", {}) for k in kernels)


#: `structural` reads the descriptors against themselves; `full` additionally binds
#: each descriptor to the producing compiler's record and the archive bytes it
#: names. Separate modes rather than a strength dial: their conclusions differ in
#: kind, one about the documents and one about the binary.
MODES = ("full", "structural")


class DeskCheckReport:
    """All four invariants over one kernel list, plus a pass/fail verdict.

    Invariants 3 and 4 key on ``toc_key``/``symbol``, which packing assigns, so an
    authored tree reports them NOT-APPLICABLE rather than a false collision.

    `fields` is the matcher-tuple identity (invariant 2); `drift_fields` is what
    invariant 1 compares. One list feeding both would let narrowing a drift report
    manufacture false collisions, so their defaults differ: the bundle's declared
    contract for `fields`, `drift_comparable_fields` for `drift_fields`.

    `mode` decides what the verdict may mean (see `MODES`); ``full`` additionally
    requires `compiled_agreement`'s result.
    """

    def __init__(
        self,
        kernels: list[dict],
        fields=None,
        drift_fields=None,
        *,
        mode: str,
        agreement_failures=None,
        agreement_unclaimed=None,
        agreement_verified=0,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "full" and agreement_failures is None:
            raise ValueError(
                "full mode requires the compiled-agreement result; None would make "
                "an unrun check indistinguishable from a clean one"
            )
        self.mode = mode
        self.agreement_failures = list(agreement_failures or [])
        self.agreement_unclaimed = list(agreement_unclaimed or [])
        self.agreement_verified = agreement_verified
        self.kernel_count = len(kernels)
        # Derived, not fixed: a caller omitting `fields` gets the identity these
        # descriptors carry, not one engine's shape.
        self.fields = (
            metadata_identity_fields(kernels) if fields is None else tuple(fields)
        )
        self.drift_fields = (
            drift_comparable_fields(kernels)
            if drift_fields is None
            else tuple(drift_fields)
        )
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

        A COULD-NOT-CHECK spec-drift result also fails: it is a check that could
        not run. toc_key NOT-APPLICABLE is an expected state and does not. In full
        mode any compiled-agreement failure fails the report.
        """
        toc_ok = (not self.toc_applicable) or (self.toc_distinct == self.toc_total)
        return (
            self.spec_drift_error is None
            and not self.drift
            and not self.duplicate_tuples
            and toc_ok
            and not self.agreement_failures
        )

    def render(self) -> str:
        lines = [f"mode={self.mode}", f"kernels={self.kernel_count}"]
        if self.mode == "full":
            if self.agreement_failures:
                body = "\n  ! ".join(["FAILED"] + self.agreement_failures)
            elif self.agreement_verified:
                body = (
                    f"OK for {self.agreement_verified} kernel(s) -- declaration and "
                    "producing-build record bind the current descriptors, schema, "
                    "arch and archive bytes"
                )
            else:
                body = (
                    "NOT VERIFIED HERE -- no kernel in this KDP declares a "
                    "specialized metadata field, so no producing-build record was "
                    "read and nothing here was bound to a binary. Only rocKE-origin "
                    "kernels currently carry compiled-specialization evidence; a hip "
                    "kernel AOT-compiled with specializing preprocessor defines is a "
                    "real compiled specialization that this check does not yet "
                    "verify, so absence of a claim is a limit of this tool, not a "
                    "property of the kernel."
                )
            lines.append("compiled specialization agreement: " + body)
            if self.agreement_unclaimed:
                lines.append(
                    "\n  ? ".join(
                        ["compiled specialization NOT VERIFIED HERE:"]
                        + self.agreement_unclaimed
                    )
                )
        else:
            lines.append(
                "compiled specialization agreement: NOT CHECKED -- structural mode "
                "reads the descriptors only; it establishes nothing about the "
                "compiled binary. Re-run with --mode full to bind them."
            )
        if self.spec_drift_error is not None:
            lines.append(
                f"metadata/authored-spec drift: COULD-NOT-CHECK -- {self.spec_drift_error}"
            )
        else:
            lines.append(f"metadata/authored-spec drift: {self.drift or 'none'}")
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
