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
from hkp_pack.pipeline import GROUP_NAME, run_pipeline  # noqa: E402


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
        description="Compile authored hip and rocKE UKDs, prune per arch, and "
        "pack a per-arch kpack release tree for the hip-kernel-provider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "environment:\n"
            "  HKP_PACK_JOBS  Worker processes used to compile distinct "
            "variants within one\n"
            "                 arch. Defaults to min(32, cpus available to this "
            "process),\n"
            "                 which respects a cgroup or affinity limit rather "
            "than the\n"
            "                 host core count. The cap is per-worker startup, "
            "not memory:\n"
            "                 a worker costs about 250 ms to load rocke+comgr, "
            "so too many\n"
            "                 workers make a pack slower. The best count grows "
            "with the\n"
            "                 variant count, and 32 is a compromise -- measured "
            "on gfx942,\n"
            "                 a 1672-variant pack is fastest near 48 workers "
            "while a\n"
            "                 200-variant pack is 2.5x faster on 12 than on 32, "
            "so set this\n"
            "                 explicitly when packing a small arch. Arches are "
            "always\n"
            "                 packed one at a time.\n"
            "                 Set HKP_PACK_JOBS=1 to force serial execution -- "
            "a compile\n"
            "                 failure then raises from the walk itself, with a "
            "single clean\n"
            "                 traceback and nothing else in flight. Anything "
            "that is not an\n"
            "                 integer of 1 or more -- including 0 -- is an "
            "error, not a\n"
            "                 fallback to the default. On any worker count "
            "the first\n"
            "                 variant that fails to compile stops the pack and "
            "the queued\n"
            "                 variants are cancelled."
        ),
    )
    p.add_argument(
        "--source-root",
        required=True,
        help="The authored source root (KDP + generic JSON + HIP sources). "
        "Walked recursively; child folders scope the content (e.g. hip/, "
        "rocKE/, per-integration folders) and each descriptor's authored "
        "subpath is preserved into the staged and installed trees. Producer "
        "selection is per-UKD on kernel_source.kind, not per-folder.",
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
    p.add_argument(
        "--group",
        default=GROUP_NAME,
        help="Archive group name for this root. The shipped archive is "
        "<arch>/kpack/<group>_<arch>.kpack, so two roots staged into one "
        "descriptor tree MUST NOT share a group -- otherwise the second "
        "overwrites the first and its descriptors name an archive that no "
        "longer holds their kernels. Defaults to the shipped group.",
    )
    p.add_argument(
        "--rocke-wheel-stamp",
        default=None,
        help="Path to the rocke wheel content-digest stamp. Its digest is "
        "recorded in each rocKE UKD's provenance, so a shipped kernel names "
        "the wheel that produced it.",
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
        rocm_kpack_dir=args.kpack_python_dir,
        inter_root=Path(args.inter_root) if args.inter_root else None,
        rocke_wheel_stamp=args.rocke_wheel_stamp,
        group=args.group,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HkpPackError as exc:
        print(f"hkp_pack: error: {exc}", file=sys.stderr)
        sys.exit(1)
