# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`python -m corpus_build --out <dir> --count 1000`.

Run from `projects/hipdnn/tools`, the same working directory `uhd_gen` is run from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import assemble, build as pipeline, model_shapes
from .shapes import DTYPES, Filter


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m corpus_build",
        description=__doc__ + "\n" + (
            "Assemble one engine-agnostic SDPA graph corpus from the packed kernel\n"
            "geometries, the recorded model shapes and the declared parameter space,\n"
            "deduplicated on the full shape tuple and tagged with a regime."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path,
                        help="output directory: graphs/*.json, manifest.json, manifest.csv")
    parser.add_argument("--count", type=int, default=pipeline.DEFAULT_COUNT,
                        help=f"graphs to emit (default {pipeline.DEFAULT_COUNT})")
    parser.add_argument("--seed", type=int, default=0,
                        help="sampling seed; the same seed rebuilds the same corpus")
    parser.add_argument("--kdp-root", type=Path, action="append", dest="kdp_roots",
                        help="directory searched for *.kdp.json (repeatable; default "
                             "the in-tree rocKE example descriptors)")
    parser.add_argument("--kdp", type=Path, action="append", default=[],
                        help="an additional kernel pack file (repeatable) -- point this "
                             "at the gfx950 dense/tiled packs when they are not in tree")
    parser.add_argument("--model-catalog", type=Path, default=None,
                        help="MODEL_CATALOG.md (default the in-tree one)")
    parser.add_argument("--model-shapes", type=Path, action="append", default=[],
                        dest="shape_dirs",
                        help="directory of published shape files (repeatable): hipDNN "
                             "graph JSON, record JSON, CSV, or key=value lines. This is "
                             "where the cluster's ~/model-shapes goes")
    parser.add_argument("--model-shapes-arch", default=None, dest="arch",
                        help="keep only shape-file rows whose `arch` column matches")
    parser.add_argument("--model-batches", type=int, nargs="+",
                        default=list(model_shapes.DEFAULT_BATCHES),
                        help="batches each recorded model geometry is expanded over "
                             f"(default {' '.join(map(str, model_shapes.DEFAULT_BATCHES))})")
    parser.add_argument("--declaration", type=Path, default=None,
                        help="sdpa_fwd.opmeta.json (default the in-tree one)")
    parser.add_argument("--min-candidates", type=int,
                        default=pipeline.DEFAULT_MIN_CANDIDATES,
                        help="skip packed geometries with fewer competing kernels "
                             f"(default {pipeline.DEFAULT_MIN_CANDIDATES})")
    parser.add_argument("--max-bytes", type=int, default=pipeline.DEFAULT_MAX_BYTES,
                        help="skip shapes whose Q/K/V/O exceed this (default 2 GiB)")
    parser.add_argument("--dtype", action="append", default=[], choices=list(DTYPES),
                        dest="dtypes",
                        help="keep only shapes of this dtype (repeatable; default all). "
                             "AITER's gfx942 forward kernels are bf16 only, so a corpus "
                             "meant to make it compete asks for `--dtype bf16`")
    parser.add_argument("--head-dim", type=int, action="append", default=[],
                        dest="head_dims",
                        help="keep only shapes of this head dimension (repeatable; "
                             "default all). AITER's gfx942 forward table carries "
                             "hdim_v=128 exclusively")
    parser.add_argument("--causal", action="append", default=[], type=int, choices=[0, 1],
                        dest="causal",
                        help="keep only shapes with this mask setting (repeatable; default "
                             "both). AITER's gfx950 forward table has no causal kernel, so a "
                             "corpus built to train it asks for `--causal 0`")
    for source, share in assemble.DEFAULT_SHARES.items():
        parser.add_argument(f"--{source}-share", type=float, default=share,
                            dest=f"{source}_share",
                            help=f"{source} source's share of --count (default {share}); "
                                 "a source that cannot fill its share hands the rest back")
    return parser.parse_args(argv)


def report(manifest: dict) -> None:
    print(f"corpus                  {manifest['emitted']} graphs "
          f"(requested {manifest['requested']}, seed {manifest['seed']})")
    for source, taken in sorted(manifest["mix"].items()):
        dropped = manifest["duplicates_dropped"].get(source, 0)
        share = 100.0 * taken / max(1, manifest["emitted"])
        print(f"  {source:<10} {taken:5d}  {share:5.1f}%   "
              f"{dropped} duplicate(s) dropped")
    print("regimes")
    for regime, total in manifest["regimes"].items():
        print(f"  {regime:<26} {total:5d}")
    applied = manifest["reports"]["filter"]
    if applied["dtypes"] or applied["head_dims"] or applied.get("causal"):
        excluded = manifest["reports"]["filtered_out"]
        dropped = manifest["reports"]["sweep"].get("filtered", 0)
        print(f"filter                  dtypes={applied['dtypes'] or 'any'} "
              f"head_dims={applied['head_dims'] or 'any'} "
              f"causal={applied.get('causal') or 'any'}")
        for source, total in sorted(excluded.items()):
            print(f"  {source:<10} {total:5d} excluded")
        print(f"  {'sweep':<10} {dropped:5d} draws rejected")
    for skipped in manifest["reports"]["model"]:
        for entry in skipped.get("skipped", []):
            print(f"  model skipped: {entry['model']:<10} {entry['reason']}")


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = pipeline.build(
        args.out, count=args.count, seed=args.seed, kdp_roots=args.kdp_roots,
        kdps=args.kdp, catalog=args.model_catalog, shape_dirs=args.shape_dirs,
        arch=args.arch, batches=tuple(args.model_batches),
        declaration=args.declaration, min_candidates=args.min_candidates,
        max_bytes=args.max_bytes,
        keep=Filter(dtypes=tuple(args.dtypes), head_dims=tuple(args.head_dims),
                    causal=tuple(bool(value) for value in args.causal)),
        shares={source: getattr(args, f"{source}_share")
                for source in assemble.DEFAULT_SHARES})
    report(manifest)
    print(f"written                 {args.out}")
    if manifest["emitted"] < manifest["requested"]:
        # Short is a real outcome, not a failure: the packs carry what they carry and
        # the declared space is finite. Say so rather than pretending the corpus is
        # the size that was asked for.
        print(f"NOTE: {manifest['requested'] - manifest['emitted']} short of "
              f"--count; every source pool was exhausted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
