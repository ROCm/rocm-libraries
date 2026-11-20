"""rpp-test-viz <capture dir> [--filter GLOB] [--list | --html [OUT]]"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

from .bundle import find_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rpp-test-viz",
        description="Browse reference/actual/diff bundles written by rpp_tests_capture.",
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
        "--html",
        nargs="?",
        type=Path,
        const=True,
        metavar="OUT",
        help="write a self-contained HTML viewer instead of opening napari "
        "(default OUT: <path>/viewer.html)",
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

    if args.html:
        from .html_export import write_html

        out = args.path / "viewer.html" if args.html is True else args.html
        size = write_html(cases, out)
        print(f"wrote {out} ({size / 1e6:.1f} MB, {len(cases)} cases)")
        return 0

    try:
        import napari  # deferred: --list and --html work without a display or Qt
    except ImportError:
        print(
            "napari is not installed: use --html, or run with `uv run --extra napari`",
            file=sys.stderr,
        )
        return 1

    from .viewer import launch

    browser = launch(cases)  # noqa: F841 -- must outlive napari.run()
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
