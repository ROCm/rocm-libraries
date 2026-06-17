#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Enumerate tile-boundary "danger" GEMM sizes for gfx950 StreamK kernels.

Background (ROCM-26298)
-----------------------
A bf16 transposed GEMM on gfx950 produced an out-of-bounds device store. Root
cause was a StreamK kernel running in the *degenerate data-parallel* regime
(``tiles % skGrid == 0``) while storing a *partial* macro-tile: the store-edge
mask let a store run past the end of the D buffer. This tool systematically
enumerates GEMM sizes that land partial macro-tile edges in the same "danger
band", so the same architectural edge-case family can be hunted for and guarded
against rather than fixed one size at a time.

What it does
------------
1. Parses the committed gfx950 StreamK (``StreamK >= 2``) bf16 ``BBS`` logic
   YAMLs and extracts each SK solution's macro-tile geometry. Geometry is read
   from ``SolutionNameMin`` (e.g. ``..._MT128x192x64_..._SVW4_..._SK3_...``),
   which encodes MacroTile0/1, DepthU, StoreVectorWidth, GlobalReadVectorWidth
   and MatrixInstruction -- this avoids loading the (multi-hundred-thousand
   line) logic files through a full YAML parser.
2. Dedupes geometries by ``(MacroTile0, MacroTile1)`` macro-tile shape, ranks
   them by frequency, and keeps the most common ``--max-shapes`` (the
   ROCM-26298 shape ``128x192`` is always included if present).
3. For each shape, emits candidate sizes that place a *partial* macro-tile on
   the M edge and on the N edge, with the remainder inside a danger band derived
   from StoreVectorWidth / MIWaveTile grouping, while choosing tile counts that
   are likely to drive the degenerate ``tiles == skGrid`` regime (near the CU
   count). Tile-aligned controls (remainder 0) are emitted alongside.
4. Writes a hipBLASLt gtest YAML stress suite plus a human-readable manifest.

Two-leg test structure (see plan)
---------------------------------
Each boundary size is exercised two ways:
  * Leg A - fault hunt (positional, value-independent): heuristic selection
    (``algo_method: 0``) at realistic large K (default 2048) to actually drive
    StreamK selection and the degenerate regime. Detection is the gtest NaN
    guard-pad + hard GPU fault; ``norm_check`` is a sanity signal.
  * Leg B - correctness (``integer_exact``): same M/N boundary sweep with
    bitwise-exact ``unit_check``. ``integer_exact`` constrains bf16 to K <= 256
    and alpha=2 / beta in {0,-2}; M and N are unconstrained, so the boundary
    axis is fully covered. To force coverage of the overrunning SK3 kernel at
    small K (where the heuristic tends to avoid StreamK), the SK3 solution can
    be pinned via ``algo_method: 2`` + ``solution_index``. Solution indices are
    build-specific, so they are resolved per build by the discovery runner and
    injected via ``--pin-map``; without a pin map Leg B falls back to portable
    heuristic selection.

Regenerate
----------
    python3 generate_tile_boundary_stress.py \
        --logic-dir projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950 \
        --out projects/hipblaslt/clients/tests/data/matmul_stress_gtest.yaml \
        --manifest projects/hipblaslt/clients/tests/tile_boundary_stress_manifest.md

This is a TEST-ONLY tool: it generates test data and does not touch any
library/runtime code.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

# gfx950 SPX device (75a0) compute-unit count, from the logic header
# ``{Architecture: gfx950, CUCount: 256}``. Used only to choose tile-count
# targets near the degenerate ``tiles == skGrid`` region; the empirical run is
# authoritative (skGrid is not a closed-form CU formula).
DEFAULT_CU_COUNT = 256

# Transpose layouts to cover, keyed by the Tensile A/B operand naming embedded
# in the logic file names. ``Alik`` = A transposed (T); ``Ailk`` = A normal (N);
# ``Bljk`` = B normal (N). ROCM-26298 was T/N.
LAYOUTS = {
    "Alik_Bljk": {"transA": "T", "transB": "N", "tag": "TN"},
    "Ailk_Bljk": {"transA": "N", "transB": "N", "tag": "NN"},
}

# Geometry fields encoded in SolutionNameMin.
_RE_MT = re.compile(r"_MT(\d+)x(\d+)x(\d+)_")
_RE_SVW = re.compile(r"_SVW(\d+)_")
_RE_GRVWA = re.compile(r"_GRVWA(\d+)_")
_RE_GRVWB = re.compile(r"_GRVWB(\d+)_")
_RE_MI = re.compile(r"_MI(\d+)x(\d+)x(\d+)_")
_RE_SK = re.compile(r"_SK(\d+)_")
_RE_NAME = re.compile(r"SolutionNameMin:\s*(\S+)")


class Geometry:
    """Macro-tile geometry of an SK solution, deduped by (mt0, mt1)."""

    __slots__ = ("mt0", "mt1", "depthus", "svw", "grvwa", "grvwb", "mi", "count",
                 "example_name")

    def __init__(self, mt0, mt1):
        self.mt0 = mt0
        self.mt1 = mt1
        self.depthus = set()
        self.svw = 0           # keep the largest SVW seen (widest store vector)
        self.grvwa = 0
        self.grvwb = 0
        self.mi = None
        self.count = 0
        self.example_name = ""

    def update(self, depthu, svw, grvwa, grvwb, mi, name):
        self.depthus.add(depthu)
        self.svw = max(self.svw, svw)
        self.grvwa = max(self.grvwa, grvwa)
        self.grvwb = max(self.grvwb, grvwb)
        if self.mi is None:
            self.mi = mi
        self.count += 1
        # Prefer the largest-DepthU example name (closest to the ROCM-26298
        # MT128x192x128 faulting solution) as the pin target reference.
        if not self.example_name or depthu == max(self.depthus):
            self.example_name = name

    @property
    def key(self):
        return f"MT{self.mt0}x{self.mt1}"


def parse_logic_file(path):
    """Yield (mt0, mt1, depthu, svw, grvwa, grvwb, mi, name) for SK>=2 sols.

    Reads line by line and parses only ``SolutionNameMin`` lines, so the huge
    logic files never go through a YAML parser.
    """
    with open(path, "r") as fh:
        for line in fh:
            if "SolutionNameMin:" not in line:
                continue
            m = _RE_NAME.search(line)
            if not m:
                continue
            name = m.group(1)
            sk = _RE_SK.search(name)
            if not sk or int(sk.group(1)) < 2:
                continue
            mt = _RE_MT.search(name)
            if not mt:
                continue
            mt0, mt1, depthu = int(mt.group(1)), int(mt.group(2)), int(mt.group(3))
            svw = int(_RE_SVW.search(name).group(1)) if _RE_SVW.search(name) else 1
            grvwa = int(_RE_GRVWA.search(name).group(1)) if _RE_GRVWA.search(name) else 1
            grvwb = int(_RE_GRVWB.search(name).group(1)) if _RE_GRVWB.search(name) else 1
            mim = _RE_MI.search(name)
            mi = [int(mim.group(1)), int(mim.group(2)), int(mim.group(3))] if mim else None
            yield (mt0, mt1, depthu, svw, grvwa, grvwb, mi, name)


def collect_geometries(logic_dir, layout):
    """Collect deduped (mt0, mt1) geometries for one transpose layout."""
    geos = {}
    pat = re.compile(r"gfx950.*_Cijk_%s_BBS_.*\.yaml$" % re.escape(layout))
    for root, _dirs, files in os.walk(logic_dir):
        for fn in files:
            if not pat.search(fn):
                continue
            for (mt0, mt1, depthu, svw, grvwa, grvwb, mi, name) in \
                    parse_logic_file(os.path.join(root, fn)):
                g = geos.get((mt0, mt1))
                if g is None:
                    g = Geometry(mt0, mt1)
                    geos[(mt0, mt1)] = g
                g.update(depthu, svw, grvwa, grvwb, mi, name)
    return geos


def select_shapes(geos, max_shapes, always_include):
    """Rank shapes by frequency; keep top max_shapes plus forced includes."""
    ranked = sorted(geos.values(), key=lambda g: (-g.count, g.mt0, g.mt1))
    chosen = ranked[:max_shapes]
    chosen_keys = {(g.mt0, g.mt1) for g in chosen}
    for (mt0, mt1) in always_include:
        if (mt0, mt1) in geos and (mt0, mt1) not in chosen_keys:
            chosen.append(geos[(mt0, mt1)])
            chosen_keys.add((mt0, mt1))
    return chosen


def danger_remainders(mt, svw, max_rems):
    """Partial-tile remainders most likely to expose a store-edge overrun.

    Combines the classic StoreVectorWidth overrun zone (just under the tile
    width) with a wider band attributable to MIWaveTile output grouping
    (ROCM-26298's overrun remainder was tile - 16). Remainders are returned
    largest-first (closest to the tile edge) and capped to ``max_rems``.
    """
    offsets = {1, 8, 16, svw, 2 * svw,
               max(1, round(mt * 0.09)), max(1, round(mt * 0.18))}
    rems = sorted({mt - o for o in offsets if 0 < mt - o < mt}, reverse=True)
    return rems[:max_rems]


def gen_boundary_points(geo, target_tiles, edge_tile_count, max_rems):
    """Generate (M, N, edge, remainder, reason) boundary points for a shape.

    For an N-edge probe the N tile is partial and M is tile-aligned; tile counts
    are chosen so the total ``ceil(M/MT0)*ceil(N/MT1)`` lands near ``target``
    (the degenerate ``tiles == skGrid`` region). M-edge probes are symmetric.
    Aligned controls (remainder 0) accompany each group.
    """
    pts = []
    mt0, mt1 = geo.mt0, geo.mt1

    for target in target_tiles:
        # ---- N edge: partial N tile, aligned M ----
        n_tiles = edge_tile_count
        m_tiles = max(1, round(target / n_tiles))
        M = m_tiles * mt0
        for rem in danger_remainders(mt1, geo.svw, max_rems):
            N = (n_tiles - 1) * mt1 + rem
            pts.append((M, N, "Nedge", rem,
                        f"N partial tile rem={rem}/{mt1} (svw={geo.svw}); "
                        f"tiles~={m_tiles}x{n_tiles}={m_tiles * n_tiles} near {target}"))
        pts.append((M, n_tiles * mt1, "Nctrl", 0,
                    f"N tile-aligned control; tiles={m_tiles}x{n_tiles}"))

        # ---- M edge: partial M tile, aligned N ----
        m_tiles = edge_tile_count
        n_tiles = max(1, round(target / m_tiles))
        N = n_tiles * mt1
        for rem in danger_remainders(mt0, geo.svw, max_rems):
            M = (m_tiles - 1) * mt0 + rem
            pts.append((M, N, "Medge", rem,
                        f"M partial tile rem={rem}/{mt0} (svw={geo.svw}); "
                        f"tiles~={m_tiles}x{n_tiles}={m_tiles * n_tiles} near {target}"))
        pts.append((m_tiles * mt0, N, "Mctrl", 0,
                    f"M tile-aligned control; tiles={m_tiles}x{n_tiles}"))

    # De-duplicate identical (M, N, edge) points across targets.
    seen = set()
    uniq = []
    for p in pts:
        k = (p[0], p[1], p[2])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def yaml_size_list(points, k):
    lines = []
    for (M, N, edge, rem, reason) in points:
        lines.append(f"    - {{ M: {M:>7}, N: {N:>7}, K: {k:>5} }}  # {edge} {reason}")
    return "\n".join(lines)


def emit_yaml(shapes_by_layout, args, pin_map):
    cmd = ("python3 generate_tile_boundary_stress.py "
           f"--logic-dir {args.logic_dir} --out {args.out} "
           f"--manifest {args.manifest} --max-shapes {args.max_shapes} "
           f"--max-rems {args.max_rems} --target-tiles {','.join(map(str, args.target_tiles))} "
           f"--leg-a-k {args.leg_a_k} --leg-b-k {args.leg_b_k}")
    out = []
    out.append("---")
    out.append("# GENERATED FILE - DO NOT EDIT BY HAND.")
    out.append("# Tile-boundary StreamK stress suite (ROCM-26298 hardening).")
    out.append(f"# Generated: {datetime.now(timezone.utc).isoformat()}")
    out.append("# Regenerate with:")
    out.append(f"#   {cmd}")
    out.append("#")
    out.append("# Category 'stress' is on-demand: excluded from quick/pre_checkin/nightly.")
    out.append("# Leg A: heuristic (algo_method 0), large K, NaN guard-pad + norm_check.")
    out.append("# Leg B: integer_exact (K<=256, alpha=2/beta in {0,-2}), bitwise unit_check;")
    out.append("#        SK3 solution pinned via --pin-map when available, else heuristic.")
    out.append("include: hipblaslt_common.yaml")
    out.append("include: known_bugs.yaml")
    out.append("include: matmul_common.yaml")
    out.append("")
    out.append("Tests:")

    n_cases = 0
    for layout, shapes in shapes_by_layout.items():
        meta = LAYOUTS[layout]
        for geo in shapes:
            points = gen_boundary_points(geo, args.target_tiles,
                                         args.edge_tile_count, args.max_rems)
            if not points:
                continue
            base = f"stress_{meta['tag']}_{geo.key}"

            # ---- Leg A: heuristic fault hunt at large K ----
            out.append(f"- name: {base}_legA")
            out.append("  category: stress")
            out.append("  function:")
            out.append("    matmul: *hpa_bf16_precision")
            out.append(f"  transA: {meta['transA']}")
            out.append(f"  transB: {meta['transB']}")
            out.append("  gpu_arch: '950'")
            out.append("  algo_method: 0")
            out.append("  norm_check: 1")
            out.append("  unit_check: 0")
            out.append("  alpha: 1")
            out.append("  beta: 0")
            if args.pad is not None:
                out.append(f"  pad: {args.pad}")
            out.append("  matrix_size:")
            out.append(yaml_size_list(points, args.leg_a_k))
            out.append("")
            n_cases += len(points)

            # ---- Leg B: integer_exact correctness, K<=256 ----
            out.append(f"- name: {base}_legB")
            out.append("  category: stress")
            out.append("  function:")
            out.append("    matmul: *hpa_bf16_precision")
            out.append(f"  transA: {meta['transA']}")
            out.append(f"  transB: {meta['transB']}")
            out.append("  gpu_arch: '950'")
            pin = pin_map.get(f"{meta['tag']}_{geo.key}")
            if pin:
                pins = pin if isinstance(pin, list) else [pin]
                pins = pins[:args.max_pins]
                out.append("  algo_method: 2")
                out.append(f"  solution_index: {pins}  # resolved SK3 indices for this build")
            else:
                out.append("  algo_method: 0  # heuristic (no pin map); see --pin-map")
            out.append("  initialization: integer_exact")
            out.append("  norm_check: 0")
            out.append("  unit_check: 1")
            # integer_exact bf16: alpha=2, beta in {0,-2} (inline; the
            # *integer_exact_alpha_beta anchor lives in matmul_gtest.yaml and
            # does not resolve across this separate include).
            out.append("  alpha_beta:")
            out.append("    - { alpha: 2, beta: 0 }")
            out.append("    - { alpha: 2, beta: -2 }")
            if args.pad is not None:
                out.append(f"  pad: {args.pad}")
            out.append("  matrix_size:")
            out.append(yaml_size_list(points, args.leg_b_k))
            out.append("")
            n_cases += len(points)

    with open(args.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return n_cases


def emit_pin_hunt(shapes_by_layout, args, pin_map):
    """Write a build-specific pinned-SK large-K fault-hunt suite.

    The heuristic rarely selects StreamK kernels, so to actually exercise each
    SK kernel's store edge in the degenerate large-K regime we pin the build's
    resolved SK indices (from --pin-map) and sweep the same boundary sizes at
    K=leg_a_k. NOT committed: solution indices are build-specific. Detection is
    the NaN guard-pad + hard GPU fault, so norm_check is off for speed.
    """
    out = ["---", "# GENERATED, build-specific pinned-SK fault hunt - DO NOT COMMIT.",
           "include: hipblaslt_common.yaml", "include: known_bugs.yaml",
           "include: matmul_common.yaml", "", "Tests:"]
    n = 0
    for layout, shapes in shapes_by_layout.items():
        meta = LAYOUTS[layout]
        for geo in shapes:
            pins = pin_map.get(f"{meta['tag']}_{geo.key}")
            if not pins:
                continue
            pins = (pins if isinstance(pins, list) else [pins])[:args.max_pins]
            pts = gen_boundary_points(geo, args.target_tiles,
                                      args.edge_tile_count, args.max_rems)
            out.append(f"- name: hunt_{meta['tag']}_{geo.key}_legP")
            out.append("  category: stress")
            out.append("  function:")
            out.append("    matmul: *hpa_bf16_precision")
            out.append(f"  transA: {meta['transA']}")
            out.append(f"  transB: {meta['transB']}")
            out.append("  gpu_arch: '950'")
            out.append("  algo_method: 2")
            out.append(f"  solution_index: {pins}")
            out.append("  norm_check: 0")
            out.append("  unit_check: 0")
            out.append("  alpha: 1")
            out.append("  beta: 0")
            out.append("  matrix_size:")
            out.append(yaml_size_list(pts, args.leg_a_k))
            out.append("")
            n += len(pts) * len(pins)
    with open(args.pin_hunt_out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return n


def emit_manifest(shapes_by_layout, args, n_cases):
    lines = []
    lines.append("# Tile-boundary StreamK stress suite - manifest")
    lines.append("")
    lines.append("Generated by `generate_tile_boundary_stress.py` "
                 "(ROCM-26298 hardening). TEST-ONLY.")
    lines.append("")
    lines.append(f"- Logic dir: `{args.logic_dir}`")
    lines.append(f"- Output data: `{args.out}`")
    lines.append(f"- CU count (skGrid target basis): {args.cu_count}")
    lines.append(f"- Target tile counts: {args.target_tiles}")
    lines.append(f"- Danger remainders per edge: up to {args.max_rems}")
    lines.append(f"- Leg A K: {args.leg_a_k}; Leg B K: {args.leg_b_k}")
    lines.append(f"- Total emitted (M,N) probe points across legs: {n_cases}")
    lines.append("")
    lines.append("## Pin targets for Leg B (resolve solution_index per build)")
    lines.append("")
    lines.append("For each shape below, resolve the build's SK3 (StreamK) "
                 "solution index whose name matches the example, then pass a "
                 "JSON `--pin-map` mapping `TAG_MTxxXyy` -> index to regenerate "
                 "Leg B with `algo_method: 2`.")
    lines.append("")
    for layout, shapes in shapes_by_layout.items():
        meta = LAYOUTS[layout]
        lines.append(f"### {meta['tag']} (transA={meta['transA']}, transB={meta['transB']})")
        lines.append("")
        lines.append("| shape | freq | DepthU set | SVW | GRVWA/B | MI | example SK solution |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for geo in shapes:
            lines.append(
                f"| `{meta['tag']}_{geo.key}` | {geo.count} | "
                f"{sorted(geo.depthus)} | {geo.svw} | {geo.grvwa}/{geo.grvwb} | "
                f"{geo.mi} | `{geo.example_name}` |")
        lines.append("")
    with open(args.manifest, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logic-dir", required=True,
                    help="gfx950 Logic/asm_full dir containing SK bf16 BBS YAMLs")
    ap.add_argument("--out", required=True, help="output gtest YAML data file")
    ap.add_argument("--manifest", required=True, help="output manifest (markdown)")
    ap.add_argument("--max-shapes", type=int, default=12,
                    help="max distinct (MT0,MT1) shapes per layout (default 12)")
    ap.add_argument("--max-rems", type=int, default=4,
                    help="max danger remainders per edge (default 4)")
    ap.add_argument("--target-tiles", default="242",
                    help="comma-separated target total tile counts (default 242)")
    ap.add_argument("--edge-tile-count", type=int, default=11,
                    help="tile count along the partial edge dimension (default 11)")
    ap.add_argument("--cu-count", type=int, default=DEFAULT_CU_COUNT)
    ap.add_argument("--leg-a-k", type=int, default=2048,
                    help="K for Leg A heuristic fault hunt (default 2048)")
    ap.add_argument("--leg-b-k", type=int, default=256,
                    help="K for Leg B integer_exact (bf16 limit 256, default 256)")
    ap.add_argument("--pad", type=int, default=None,
                    help="optional D/C guard pad (elements) for OOB detection")
    ap.add_argument("--pin-map", default=None,
                    help="JSON mapping TAG_MTxxXyy -> SK3 solution index (or list) "
                         "from resolve_sk3_pinmap.py; pins Leg B (and the hunt)")
    ap.add_argument("--max-pins", type=int, default=6,
                    help="max SK indices pinned per shape (default 6)")
    ap.add_argument("--pin-hunt-out", default=None,
                    help="also write a build-specific pinned-SK large-K fault-hunt "
                         "YAML here (requires --pin-map; do not commit)")
    return ap.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    args.target_tiles = [int(x) for x in str(args.target_tiles).split(",") if x != ""]
    if args.leg_b_k > 256:
        print("warning: integer_exact for bf16 requires K<=256; clamping leg-b-k.",
              file=sys.stderr)
        args.leg_b_k = 256

    pin_map = {}
    if args.pin_map:
        with open(args.pin_map) as fh:
            pin_map = json.load(fh)

    # ROCM-26298 shape is always covered if present.
    always_include = [(128, 192)]

    shapes_by_layout = {}
    for layout in LAYOUTS:
        geos = collect_geometries(args.logic_dir, layout)
        if not geos:
            print(f"warning: no SK bf16 {layout} geometries found under "
                  f"{args.logic_dir}", file=sys.stderr)
            continue
        shapes_by_layout[layout] = select_shapes(geos, args.max_shapes, always_include)

    if not shapes_by_layout:
        print("error: no geometries collected; check --logic-dir", file=sys.stderr)
        return 1

    n_cases = emit_yaml(shapes_by_layout, args, pin_map)
    emit_manifest(shapes_by_layout, args, n_cases)

    if args.pin_hunt_out:
        if not pin_map:
            print("error: --pin-hunt-out requires --pin-map", file=sys.stderr)
            return 1
        n_hunt = emit_pin_hunt(shapes_by_layout, args, pin_map)
        print(f"Wrote pinned-SK hunt {args.pin_hunt_out} (~{n_hunt} instances)")

    total_shapes = sum(len(s) for s in shapes_by_layout.values())
    print(f"Wrote {args.out} ({total_shapes} shapes, {n_cases} probe points "
          f"x 2 legs) and manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
