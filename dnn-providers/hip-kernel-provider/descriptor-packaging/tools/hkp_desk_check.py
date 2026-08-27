"""RUNBOOK.md step 5d's four desk-check invariants, as a real, runnable CLI.

The invariants used to live only as a shell-embedded Python snippet inside
markdown; that snippet's invariant 1 was dead on every real packed tree (it
read `kernel_source.spec`, which packing rewrites away) and nothing noticed
because nothing could run it. This is the fix, made runnable:

    python3 tools/hkp_desk_check.py <path/to/*.kdp.json>

Exits 0 when every enforced invariant is clean, 1 when any is violated OR
could not be checked (a "COULD-NOT-CHECK" spec-drift result is a failure, not
a silent pass -- see `hkp_pack.desk_check.DeskCheckReport.ok`). Invariant 4
(symbol non-uniqueness) is informational only and never causes a non-zero
exit on its own.

Works on both an authored (pre-pack) KDP -- `kernel_source.spec` -- and a
shipped (post-pack) one -- `provenance.spec` -- since the check falls back
between the two automatically.
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
    DeskCheckReport,
    load_kernels,
)


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="hkp_desk_check",
        description="Desk-check a KDP's variant set for the four RUNBOOK "
        "step 5d invariants: metadata/spec drift, duplicate matcher tuples, "
        "toc_key uniqueness, and symbol non-uniqueness tolerance.",
    )
    p.add_argument(
        "kdp",
        help="Path to a `.kdp.json` -- authored (pre-pack) or shipped "
        "(post-pack), either works.",
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
    kernels = load_kernels(Path(args.kdp))
    report = DeskCheckReport(kernels, fields, drift_fields)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
