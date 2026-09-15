# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate an HTML table showing the MFMA accumulator register layout."""

import argparse
import colorsys
import sys
from pathlib import Path


def parseArgs():
    p = argparse.ArgumentParser(
        description="Generate MFMA accumulator layout HTML table."
    )
    p.add_argument(
        "--mi", nargs=9, type=int, required=True, metavar="N",
        help="MatrixInstruction list (9 integers: mfma_m mfma_n mfma_k mfma_b ? wt_m wt_n wg_m wg_n)"
    )
    p.add_argument("--out", default="mfma_layout.html", help="Output HTML file path")
    p.add_argument("--wavesize", type=int, default=64, help="Wavefront size (default: 64)")
    p.add_argument("--verify", action="store_true", help="Check bijectivity and exit")
    return p.parse_args()


def computeWaveGroup(mi):
    """Derive MIWaveGroup [wg_m, wg_n] from the 9-element MatrixInstruction."""
    mfma_m, _, _, mfma_b, mi4, _, _, mi7, mi8 = mi
    waves = mi7 * mi8
    wg0 = mi4 * mfma_m * mi7
    blkBm = min(wg0 // mfma_m, mfma_b)
    miwg0 = min((wg0 // mfma_m) // blkBm, waves)
    return miwg0, waves // miwg0


def derivedParams(mi, wavesize):
    """Return a dict of all layout parameters derived from mi and wavesize."""
    mfma_m, mfma_n, mfma_k, mfma_b = mi[0], mi[1], mi[2], mi[3]
    wt_m, wt_n = mi[5], mi[6]
    wg_m, wg_n = computeWaveGroup(mi)
    rowsPerLane = (mfma_m * mfma_n) // wavesize
    return {
        "mfma_m": mfma_m, "mfma_n": mfma_n, "mfma_k": mfma_k, "mfma_b": mfma_b,
        "wt_m": wt_m, "wt_n": wt_n, "wg_m": wg_m, "wg_n": wg_n,
        "rowsPerLane": rowsPerLane, "wavesize": wavesize,
        "totalRows": wg_m * wt_m * mfma_m,
        "totalCols": wg_n * wt_n * mfma_n,
        "numWaves": wg_m * wg_n,
    }


def buildGrid(mi, p):
    """Fill a 2-D grid (totalRows x totalCols) with cell labels and wave indices.

    Returns (grid, waveGrid) where each entry is a string label or int wave index.
    """
    mfma_m  = p["mfma_m"]
    mfma_n  = p["mfma_n"]
    wt_m    = p["wt_m"]
    wt_n    = p["wt_n"]
    wg_m    = p["wg_m"]
    wg_n    = p["wg_n"]
    rpl     = p["rowsPerLane"]
    ws      = p["wavesize"]
    tR      = p["totalRows"]
    tC      = p["totalCols"]

    grid     = [[None] * tC for _ in range(tR)]
    waveGrid = [[None] * tC for _ in range(tR)]

    for tid in range(wg_m * wg_n * ws):
        laneId   = tid % ws
        waveIdx  = tid // ws
        waveId0  = waveIdx % wg_m
        waveId1  = waveIdx // wg_m
        colInWave = laneId % mfma_n
        laneGroup = laneId // mfma_n

        for mmaNId in range(wt_n):
            for mmaMId in range(wt_m):
                vtileIdx = mmaNId * wt_m + mmaMId
                for elemMId in range(rpl):
                    agprIdx = vtileIdx * rpl + elemMId
                    row = waveId0 * (wt_m * mfma_m) + mmaMId * mfma_m + laneGroup * rpl + elemMId
                    col = waveId1 * (wt_n * mfma_n) + mmaNId * mfma_n + colInWave
                    grid[row][col]     = f"a[{agprIdx}][{elemMId},0,{mmaMId},{mmaNId},{laneId},{waveIdx}]"
                    waveGrid[row][col] = waveIdx

    return grid, waveGrid


def verifyBijectivity(grid, p):
    """Assert every cell is filled exactly once; return True on success."""
    errors = []
    for r in range(p["totalRows"]):
        for c in range(p["totalCols"]):
            if grid[r][c] is None:
                errors.append(f"cell ({r},{c}) is empty")
    if errors:
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        print(f"bijectivity check failed: {len(errors)} empty cell(s)", file=sys.stderr)
        return False
    print(f"bijectivity check passed: all {p['totalRows']}x{p['totalCols']} cells filled")
    return True


def waveColor(waveIdx, numWaves):
    """Return a light background hex color for the given wave index."""
    hue = waveIdx / max(numWaves, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.30, 1.0)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def buildCaption(mi, p):
    tR, tC = p["totalRows"], p["totalCols"]
    return (
        f"MatrixInstruction={mi} | "
        f"mfma={p['mfma_m']}x{p['mfma_n']} k={p['mfma_k']} b={p['mfma_b']} | "
        f"MIWaveTile=[{p['wt_m']},{p['wt_n']}] MIWaveGroup=[{p['wg_m']},{p['wg_n']}] | "
        f"wavesize={p['wavesize']} rows_per_lane={p['rowsPerLane']} | "
        f"tile {tR}x{tC}"
    )


def renderHtml(mi, grid, waveGrid, p):
    """Render the full HTML document as a string."""
    numWaves = p["numWaves"]
    colors   = {w: waveColor(w, numWaves) for w in range(numWaves)}

    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "table { border-collapse: collapse; font-family: monospace; font-size: 9px; }",
        "th, td { border: 1px solid #888; padding: 2px 4px; white-space: nowrap; }",
        "th { background: #ddd; text-align: center; }",
        "</style>",
        "</head><body>",
        "<table>",
        f"<caption>{buildCaption(mi, p)}</caption>",
    ]

    # Header row.
    header = "<tr><th>#</th>" + "".join(f"<th>{c}</th>" for c in range(p["totalCols"])) + "</tr>"
    lines.append(header)

    for r in range(p["totalRows"]):
        row_parts = [f"<tr><th>{r}</th>"]
        for c in range(p["totalCols"]):
            bg  = colors[waveGrid[r][c]]
            lbl = grid[r][c]
            row_parts.append(f"<td style='background:{bg}'>{lbl}</td>")
        row_parts.append("</tr>")
        lines.append("".join(row_parts))

    lines += ["</table>", "</body></html>"]
    return "\n".join(lines)


def main():
    args = parseArgs()
    mi   = args.mi
    p    = derivedParams(mi, args.wavesize)
    grid, waveGrid = buildGrid(mi, p)

    if args.verify:
        ok = verifyBijectivity(grid, p)
        sys.exit(0 if ok else 1)

    html = renderHtml(mi, grid, waveGrid, p)
    outPath = Path(args.out)
    outPath.write_text(html, encoding="utf-8")
    print(outPath.resolve())


if __name__ == "__main__":
    main()
