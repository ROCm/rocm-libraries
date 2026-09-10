"""RUNBOOK.md step 5d's desk-check invariants, as a real, runnable CLI.

The invariants used to live only as a shell-embedded Python snippet inside
markdown; that snippet's invariant 1 was dead on every real packed tree (it
read `kernel_source.spec`, which packing rewrites away) and nothing noticed
because nothing could run it. This is the fix, made runnable:

    python3 tools/hkp_desk_check.py --mode structural <path/to/*.kdp.json>
    python3 tools/hkp_desk_check.py --mode full <path/to/shipped.kdp.json>

TWO MODES, TWO DIFFERENT CONCLUSIONS. `--mode structural` reads the descriptors
against themselves and against each other: drift between metadata and the AUTHORED
spec, duplicate matcher tuples, toc_key uniqueness, symbol tolerance. It says so in
its own output, and it never reports compiled agreement, because nothing it looked
at is evidence about a binary.

`--mode full` additionally binds every kernel to the producing compiler's
`provenance.effective_spec` record and to the archive bytes the descriptor names:
the self-contained declaration, the current metadata, KMD schema, KDP header,
effective arch, captured symbol and payload SHA256 must all agree with what the
compiler observed. A missing, unsupported or mismatched record is a FAILURE, not an
unchecked property. Nothing here imports rocKE, so a valid packed artifact verifies
on a machine that has never had the producer installed.

`--mode full` needs the packed dialect. Before packing there are no bytes, so
there is no producing-build record for a declaration to bind and the mode has
nothing to check; a non-kpack kernel is a failure here, the same refusal
`verify_variant_sets` makes, so the two readers agree about one artifact.

A packed kernel whose declaration lists no specialized metadata field is reported
as NOT VERIFIED HERE and never folded into the agreement line. Only rocKE-origin
kernels currently carry compiled-specialization evidence: a hip kernel AOT-built
with specializing preprocessor defines is a real compiled specialization that this
check does not yet verify, so the absence of a claim is a limit of this tool
rather than a property of the kernel. Reading it as compiled agreement would be a
success this tool never earned, which is the substitution the two modes exist to
prevent.

A KDP's `kernelDescriptors` may hold standalone-UKD id references as bare strings
after packing; they are resolved against the shard, the same hop
`verify_variant_sets` makes, so both readers see the same descriptor set.

The mode is REQUIRED. A default would let a structural run be mistaken for a full
one in a log, which is the substitution this split exists to prevent.

Exits 0 when every enforced invariant is clean, 1 when any is violated OR
could not be checked (a "COULD-NOT-CHECK" spec-drift result is a failure, not
a silent pass -- see `hkp_pack.desk_check.DeskCheckReport.ok`). Symbol
non-uniqueness is informational only and never causes a non-zero exit on its own.

Structural mode works on an authored (pre-pack) KDP -- `kernel_source.spec` --
and a shipped (post-pack) one -- `provenance.spec` -- since the drift check
falls back between the two automatically. Full mode needs a shipped shard: the
record and the archive it binds exist only after packing.
"""

import argparse
import sys
from pathlib import Path

# Same shadowing hazard hkp_pack.py's own tool guards against: tools/ must
# never resolve `hkp_pack` to itself.
_PKG_ROOT = str(Path(__file__).resolve().parent.parent / "python")
while _PKG_ROOT in sys.path:
    sys.path.remove(_PKG_ROOT)
sys.path.insert(0, _PKG_ROOT)

from hkp_pack.desk_check import (  # noqa: E402
    DEFAULT_MATCHER_FIELDS,
    MODES,
    DeskCheckReport,
    compiled_agreement,
    load_kernels,
)
from hkp_pack.errors import HkpPackError  # noqa: E402


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="hkp_desk_check",
        description="Desk-check a KDP's variant set: compiled specialization "
        "agreement (full mode only), metadata/authored-spec drift, duplicate "
        "matcher tuples, toc_key uniqueness, and symbol non-uniqueness tolerance.",
    )
    p.add_argument(
        "kdp",
        help="Path to a `.kdp.json`. Structural mode accepts an authored "
        "(pre-pack) or shipped (post-pack) file; full mode requires a shipped "
        "shard, whose producing-build record and archive it binds.",
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        required=True,
        help="'full' binds every kernel to the producing compiler's "
        "provenance.effective_spec record and to the archive bytes the "
        "descriptor names; a missing, unsupported or mismatched record fails. "
        "'structural' checks the descriptors against each other only and "
        "reports compiled agreement as NOT CHECKED. Required: a default would "
        "let a structural run read as a full one.",
    )
    p.add_argument(
        "--kpack-python-dir",
        default=None,
        help="The rocm-kpack 'python' directory, used in full mode to read the "
        "named archive blob. Only the archive reader is needed -- the original "
        "producer is never imported.",
    )
    p.add_argument(
        "--field",
        action="append",
        dest="fields",
        default=[],
        help="A KMD field the matcher keys on; repeatable. This is the "
        "MATCHER-TUPLE identity (invariant 2). Defaults to a generic "
        "attention-shaped list -- narrow it to your own KMD's fields for a "
        "meaningful check. Narrowing this does NOT narrow --drift-field.",
    )
    p.add_argument(
        "--drift-field",
        action="append",
        dest="drift_fields",
        default=[],
        help="A field to compare between metadata and the authored spec "
        "(invariant 1); repeatable. Defaults to whatever --field resolves "
        "to. Separate from --field on purpose: dropping a field here to "
        "silence a drift report must never remove it from the matcher-tuple "
        "identity, which would manufacture false duplicate collisions.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    fields = tuple(args.fields) if args.fields else DEFAULT_MATCHER_FIELDS
    drift_fields = tuple(args.drift_fields) if args.drift_fields else None
    kdp = Path(args.kdp)
    failures = None
    unclaimed: list = []
    verified = 0
    if args.mode == "full":
        # A tree that cannot be read at all is one failure message, not a crash
        # and not a skip: the caller asked whether this bundle agrees with its
        # binaries, and "the record could not be reached" answers that with no.
        try:
            failures, unclaimed, verified = compiled_agreement(
                kdp, args.kpack_python_dir
            )
        except HkpPackError as exc:
            failures = [str(exc)]
    try:
        kernels = load_kernels(kdp)
    except HkpPackError as exc:
        # An unresolvable standalone-UKD reference means the descriptor set is not
        # readable at all. Reported, not raised: a traceback out of a gate reads as
        # a broken tool rather than as a broken artifact.
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    report = DeskCheckReport(
        kernels,
        fields,
        drift_fields,
        mode=args.mode,
        agreement_failures=failures,
        agreement_unclaimed=unclaimed,
        agreement_verified=verified,
    )
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
