"""rpp-test-viz <capture dir> [--filter GLOB] [--list | -o OUT]"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

from .bundle import find_cases
from .html_export import write_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rpp-test-viz",
        description="Write a self-contained HTML viewer for the reference/actual/diff "
        "bundles written by rpp_tests_capture.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="RPP_TEST_CAPTURE_DIR of a run, any directory below it, or a manifest",
    )
    parser.add_argument(
        "--filter", default="*", help="glob over case names (GTest style)"
    )
    parser.add_argument("--list", action="store_true", help="print the cases and exit")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="HTML file to write (default: <path>/viewer.html)",
    )
    args = parser.parse_args(argv)

    cases = [
        c for c in find_cases(args.path) if fnmatch.fnmatchcase(c.name, args.filter)
    ]
    if not cases:
        print(
            f"no capture bundles under {args.path} match {args.filter!r}",
            file=sys.stderr,
        )
        return 1

    if args.list:
        for case in cases:
            for k, entry in enumerate(case.comparisons):
                verdict = (
                    "PASS" if case.passed else f"{entry['failed']}/{entry['compared']}"
                )
                suffix = f" [{k}]" if len(case.comparisons) > 1 else ""
                print(f"{verdict:>13}  {case.name}{suffix}")
        return 0

    root = args.path.parent if args.path.is_file() else args.path
    out = args.output or root / "viewer.html"
    size = write_html(cases, out)
    print(f"wrote {out} ({size / 1e6:.1f} MB, {len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
