"""The four desk-check invariants of the packaging README's "Desk-check a
variant set", as real, importable code.

Runs over a single loaded KDP document's ``kernelDescriptors`` list -- works
on an authored (pre-pack) tree via ``kernel_source.spec`` or a shipped
(post-pack) one via ``provenance.spec`` interchangeably, and treats "neither
location has a spec" as a distinct, reported outcome rather than a silent
"no drift".

``dtype`` names one type in two DELIBERATE vocabularies: rocKE specs spell it
``"bf16"``, hipDNN metadata carries the enum name ``"BFLOAT16"`` (or, in the
tiled bundle, ``"BF16"``). ``_DTYPE_ALIASES`` normalises them rather than
dropping the field, because dtype is the field most worth checking: ``spec
"bf16"`` against ``metadata "HALF"`` is a real, fatal drift and must still fail.
"""

from __future__ import annotations

import collections
from pathlib import Path

from . import agreement, descriptor_context
from .errors import HkpPackError
from .kpack_resolver import load_kpack

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


# The last-resort field list, for a bundle that declares no specialization
# contract at all. A bundle that DOES declare one states exactly which fields
# its kernels were specialized on, and `load_variant_set` reads that statement
# instead: a generic guess standing in for the bundle's own declaration is how
# distinct kernels collapse onto one matcher tuple and a clean bundle reports
# hundreds of false collisions.
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
    metadata, which is a genuine "no drift" result. Conflating the two would
    render "found nothing to check" identically to "checked, found nothing
    wrong", which is a dead check."""


def _resolve(kdp_path: Path) -> tuple[dict, list[descriptor_context.Entry]]:
    """One `.kdp.json`'s own document and its resolved entries."""
    kdp_path = Path(kdp_path).resolve()
    index = descriptor_context.Index(str(kdp_path.parent))
    kdp = next(d for d in index.of_type("kdp") if Path(d.path) == kdp_path)
    return kdp.doc, descriptor_context.resolve_entries(index, kdp)


def load_kernels(kdp_path: Path) -> list[dict]:
    """A `.kdp.json`'s kernel descriptors, standalone-UKD references resolved."""
    return [entry.ukd for entry in _resolve(kdp_path)[1]]


def declared_matcher_fields(kdp_doc: dict, entries) -> tuple[str, ...] | None:
    """The matcher-tuple identity this bundle declares for itself, in
    declaration order, or None when no entry declares a contract.

    The specialization contract is the bundle's own statement of what the
    producing compiler specialized on, so it -- not a generic list this module
    guesses -- is the field set that distinguishes one variant from another.
    Union across entries, because one shard may carry several consumers and a
    field any of them keys on is distinguishing for the shard.

    Resolution goes through `agreement.resolved_contract`, which already looks
    in both places a contract may live: the kernel's own `provenance`, then the
    enclosing KDP's. The enclosing document is offered only to inline entries,
    the same restriction the rest of the packaging applies -- a standalone UKD
    is its own file and inherits nothing.
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
            for field in consumer.get("metadata_fields") or []:
                if field not in fields:
                    fields.append(field)
    return tuple(fields) or None


def load_variant_set(kdp_path: Path) -> tuple[list[dict], tuple[str, ...] | None]:
    """A `.kdp.json`'s kernel descriptors plus the matcher fields it declares.

    Both come from one walk of the descriptor tree: a shipped shard's KDP runs
    to megabytes, and reading it twice to answer two questions about the same
    document buys nothing.
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

    A check that compares the descriptor's own ``sha256`` against a digest of that
    same field establishes nothing. Reading the named blob is what makes the payload
    binding real, and it needs the packaging archive reader only -- never the
    producer that emitted the kernel.
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
    verifies on a machine that has never had rocKE installed. An artifact that
    cannot present a record is a failure rather than an unchecked property --
    absence is exactly the state a forged or stale tree is in. A non-kpack kernel
    fails too, having no bytes to bind before packing; that is the same refusal
    `verify_variant_sets` makes, so both readers agree about one artifact.

    Returns `(failures, unclaimed, verified)`. A packed declaration with no
    `metadata_fields` is the legitimate shape for a non-compiled source and binds no
    record, so it counts as unclaimed rather than as a pass nothing was read for.
    Only rocKE-origin kernels carry that evidence today.

    The waiver is keyed on origin. A kernel whose `provenance.origin_kind` is
    `"rocke"` was published with its evidence by construction, so the same shape is
    a failure there: otherwise a descriptor could retire its own evidence by
    dropping `effective_spec` and moving its `metadata_fields` into
    `matcher_only_fields`, leaving the archive bytes unread. An ABSENT
    `origin_kind` is not rocKE -- descriptors packed before the field existed and
    hand-authored fixtures have none.

    That reaches evidence lost by accident, not evidence removed on purpose.
    Nothing outside the record binds `origin_kind`: `descriptor_binding()` digests
    it, and that digest is inside the record being dropped, so an edit that removes
    the record can set `origin_kind` to `"hip"` in the same pass and present as a
    source that never owed evidence. Detecting that needs provenance bound
    somewhere the shipped tree cannot rewrite.
    """
    kdp_path = Path(kdp_path).resolve()
    index = descriptor_context.Index(str(kdp_path.parent))
    schemas = index.schemas()
    bundles = descriptor_context.resolve_bundles(index)
    bundle = next(b for b in bundles if Path(b.kdp_path) == kdp_path)
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
    """Every field invariant 1 is able to compare: one carrying BOTH a spec
    value and a metadata value on at least one kernel, unioned across the set in
    first-appearance order so one bundle always renders one field list.

    Derived from the descriptors in hand rather than from the bundle's declared
    specialization contract, because that declaration is one of the things
    invariant 1 polices. A narrow declaration would otherwise confine the drift
    comparison to the fields the artifact chose to mention, and a genuine
    spec-vs-metadata disagreement on any other field -- a `block_m` in the
    metadata that is not the `block_m` the compiler baked in -- would never be
    compared at all, leaving a clean exit as the only possible verdict. An
    artifact does not get to set the width of the audit that polices it.

    Widest is also nearly free: `metadata_spec_drift` already skips any field
    missing from either side, so this set asks for no comparison that function
    would have refused to make. The residual cost is a field whose two sides
    speak deliberately different vocabularies that `_values_agree` cannot
    normalise -- an engine-translated `layout`, say -- which reports as drift.
    That is what `--drift-field` deliberately narrows, and a false report that
    argues with a human beats a field silently never compared.

    A kernel with no spec anywhere contributes nothing here rather than raising:
    `metadata_spec_drift` owns that refusal, and raising from the field
    derivation would move a COULD-NOT-CHECK verdict out of the check that
    reports it.
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
    narrow the other. `None` means `drift_comparable_fields` -- the widest
    set this data admits, and specifically NOT the bundle's declared contract,
    which is an input to this check rather than a bound on it.
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


def duplicate_matcher_tuples(
    kernels: list[dict], fields=DEFAULT_MATCHER_FIELDS
) -> dict[tuple, int]:
    """Invariant 2: no two kernels may share a matcher tuple on the same
    arch -- one of them is unreachable. Returns {tuple: count} for every
    tuple shared by more than one kernel (empty means none).

    The compared field set is the UNION of `fields` present in ANY kernel's
    metadata, not the fields of ``kernels[0]``. Keying off the first kernel
    would make the tuple identity depend on list order: a set where only a
    later kernel declared a field would either raise ``KeyError`` or silently
    drop that field from the identity and report collisions that do not exist.
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


#: The two things a desk check can be asked. `structural` reads the descriptors
#: against themselves and against each other; `full` additionally binds each
#: descriptor to the producing compiler's record and to the archive bytes it names.
#: They are separate MODES rather than a strength dial because their conclusions are
#: different in kind: a structural pass is a statement about the documents, and only
#: a full pass is a statement about the binary.
MODES = ("full", "structural")


class DeskCheckReport:
    """All four invariants over one kernel list, plus a pass/fail verdict.

    Invariants 3 and 4 key on ``toc_key``/``symbol``, which packing assigns, so an
    authored (pre-pack) tree reports them NOT-APPLICABLE rather than a false
    "all None -- collision".

    `fields` is the MATCHER-TUPLE identity (invariant 2); `drift_fields` is what
    invariant 1 compares against the spec. One list feeding both would be a trap:
    narrowing the comparison to silence a drift report would also narrow the tuple
    identity and manufacture false collisions in the check whose entire job is
    catching unreachable variants.

    So the two defaults are drawn from different places on purpose. `fields`
    falls back to the bundle's own declared contract, which is the only
    statement of what distinguishes one variant from another. `drift_fields`
    falls back to `drift_comparable_fields` -- every field carrying both a spec
    and a metadata value -- because a bundle that declared its way to a narrow
    contract would otherwise also narrow the check that audits that bundle,
    and real drift on a field it does not declare would exit 0.

    `mode` decides what the verdict is allowed to MEAN (see `MODES`). ``full``
    additionally requires `compiled_agreement`'s result, of which only an empty
    `agreement_failures` is clean.
    """

    def __init__(
        self,
        kernels: list[dict],
        fields=DEFAULT_MATCHER_FIELDS,
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
        self.fields = tuple(fields)
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

        A COULD-NOT-CHECK spec-drift result also fails the report -- it is
        not a clean bill of health, it is a check that could not run, and
        reporting it as green is the exact defect this module exists to
        remove. toc_key NOT-APPLICABLE (pre-pack tree) does NOT fail the
        report -- that is an expected state, not an unchecked one.

        In full mode any compiled-agreement failure fails the report, including the
        failure that says a record is absent: an artifact that cannot say what it was
        built from has not shown agreement with anything.
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
