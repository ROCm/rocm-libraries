import argparse
import sys
from pathlib import Path

# Put the package dir ahead of everything, including this script's own dir:
# tools/ also holds a module literally named hkp_pack (this file), which would
# otherwise shadow the hkp_pack package when tools/ is sys.path[0].
_PKG_ROOT = str(Path(__file__).resolve().parent.parent / "python")
while _PKG_ROOT in sys.path:
    sys.path.remove(_PKG_ROOT)
sys.path.insert(0, _PKG_ROOT)

from hkp_pack.errors import HkpPackError  # noqa: E402
from hkp_pack.pipeline import run_pipeline  # noqa: E402


def _split_arches(values):
    arches = []
    for value in values:
        for tok in value.replace(",", ";").split(";"):
            tok = tok.strip()
            if tok:
                # GPU_TARGETS entries may carry feature suffixes (gfx942:xnack-).
                arches.append(tok.split(":", 1)[0])
    # De-dup, preserve order.
    seen = set()
    ordered = []
    for a in arches:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="hkp_pack",
        description="Compile authored hip UKDs, prune per arch, and pack a "
        "per-arch kpack release tree for the hip-kernel-provider.",
    )
    p.add_argument(
        "--source-root",
        required=True,
        help="Flat authored source folder (KDP + generic JSON + HIP sources).",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Root under which <out-root>/<gfx>/ release folders are written.",
    )
    p.add_argument(
        "--arches",
        action="append",
        default=[],
        help="';'- or ','-separated gfx arch list; repeatable. The tool loops "
        "internally over the full list in one run. Empty installs nothing.",
    )
    p.add_argument(
        "--hipcc",
        required=True,
        help="Path to the hipcc driver used for --genco compilation.",
    )
    p.add_argument(
        "--inter-root",
        default=None,
        help="Build-only intermediate root (never shipped). Defaults beside "
        "out-root.",
    )
    p.add_argument(
        "--kpack-python-dir",
        default=None,
        help="Path to the rocm-kpack 'python' directory (overrides any "
        "installed rocm_kpack).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    arches = _split_arches(args.arches)
    run_pipeline(
        source_root=Path(args.source_root),
        arches=arches,
        out_root=Path(args.out_root),
        hipcc=args.hipcc,
        kpack_python_dir=args.kpack_python_dir,
        inter_root=Path(args.inter_root) if args.inter_root else None,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HkpPackError as exc:
        print(f"hkp_pack: error: {exc}", file=sys.stderr)
        sys.exit(1)
