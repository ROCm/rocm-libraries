"""The desk-check invariants of RUNBOOK §4's host boundary, as a real, runnable CLI.

    python3 tools/hkp_desk_check.py --mode structural <path/to/*.kdp.json>
    python3 tools/hkp_desk_check.py --mode full <path/to/shipped.kdp.json>

`--mode structural` reads the descriptors against themselves and each other: drift
between metadata and the authored spec, duplicate matcher tuples, toc_key
uniqueness, symbol tolerance. It never reports compiled agreement. It accepts an
authored (`kernel_source.spec`) or shipped (`provenance.spec`) KDP, since the drift
check falls back between the two.

`--mode full` additionally binds every kernel to the producing compiler's
`provenance.effective_spec` record and to the archive bytes the descriptor names:
declaration, metadata, KMD schema, KDP header, effective arch, captured symbol and
payload SHA256 must all agree with what the compiler observed. A missing,
unsupported or mismatched record is a FAILURE. Nothing imports rocKE, so a valid
artifact verifies where the producer was never installed. It needs the packed
dialect: a non-kpack kernel fails, the same refusal `verify_variant_sets` makes.

A packed non-rocKE kernel whose declaration lists no specialized metadata field is
reported NOT VERIFIED HERE, never folded into the agreement line: only rocKE-origin
kernels carry compiled-specialization evidence today. A kernel whose
`provenance.origin_kind` is `rocke` cannot reach that report -- the packer published
its `effective_spec` -- so declaring no specialized field AND carrying no record is
a FAILURE. An ABSENT `origin_kind` is not rocKE.

Standalone-UKD id references in a KDP's `kernelDescriptors` resolve against the
shard, the same hop `verify_variant_sets` makes.

The mode is REQUIRED: a default would let a structural run read as a full one.

Exits 0 when every enforced invariant is clean, 1 when any is violated or could not
be checked (see `hkp_pack.desk_check.DeskCheckReport.ok`). Symbol non-uniqueness is
informational and never causes a non-zero exit on its own.
"""

import argparse
import sys
from pathlib import Path

# Same shadowing hazard hkp_pack.py's own tool guards against: tools/ must never
# resolve `hkp_pack` to itself.
_PKG_ROOT = str(Path(__file__).resolve().parent.parent / "python")
while _PKG_ROOT in sys.path:
    sys.path.remove(_PKG_ROOT)
sys.path.insert(0, _PKG_ROOT)

from hkp_pack.desk_check import (  # noqa: E402
    MODES,
    DeskCheckReport,
    compiled_agreement,
    load_variant_set,
    metadata_identity_fields,
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
        "MATCHER-TUPLE identity (invariant 2). Defaults to the fields the "
        "bundle's own specialization_contract declares it specialized on, and "
        "otherwise to the fields derived from the bundle's own kernel "
        "metadata -- there is no generic attention-shaped fallback. Neither "
        "this flag nor that declaration narrows --drift-field.",
    )
    p.add_argument(
        "--drift-field",
        action="append",
        dest="drift_fields",
        default=[],
        help="A field to compare between metadata and the authored spec "
        "(invariant 1); repeatable. Defaults to EVERY field carrying both a "
        "spec and a metadata value, never to --field or to the bundle's "
        "declared contract: this check audits the bundle, so letting the "
        "bundle's own declaration set its width would let a narrow "
        "declaration hide real drift on an undeclared field. Separate from "
        "--field on purpose: dropping a field here to silence a drift report "
        "must never remove it from the matcher-tuple identity, which would "
        "manufacture false duplicate collisions.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    drift_fields = tuple(args.drift_fields) if args.drift_fields else None
    kdp = Path(args.kdp)
    failures = None
    unclaimed: list = []
    verified = 0
    if args.mode == "full":
        # A tree that cannot be read at all is one failure message, not a crash and
        # not a skip: "the record could not be reached" answers the caller's
        # question with no.
        try:
            failures, unclaimed, verified = compiled_agreement(
                kdp, args.kpack_python_dir
            )
        except HkpPackError as exc:
            failures = [str(exc)]
    try:
        kernels, declared_fields = load_variant_set(kdp)
    except HkpPackError as exc:
        # An unresolvable standalone-UKD reference means the descriptor set is not
        # readable at all. Reported, not raised: a traceback out of a gate reads as
        # a broken tool rather than a broken artifact.
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    # No generic fallback after the derived one: it is reached only by a bundle
    # whose kernels carry no metadata, and `duplicate_matcher_tuples` drops every
    # field absent from all of them.
    fields = tuple(args.fields) or declared_fields or metadata_identity_fields(kernels)
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
