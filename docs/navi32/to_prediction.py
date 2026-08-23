#!/usr/bin/env python3
"""
Convert a GridBased Tensile logic file into a Prediction (Origami-selected) one.

A GridBased library selects by nearest-neighbour lookup in a dense shape->solution table
(element [7]). A Prediction library has **no table at all** -- [7] is None -- and ranks the
solutions in [5] with Origami's analytical latency model at call time.

Confirmed against the shipped gfx942 StreamK Prediction files on develop:
    [6] [7] [8] [9] = None      [10] = metric string      [11] = 'Prediction'

That is why this matters for navi32. The shipped navi32 GridBased table has only **471 rows**
against navi31's 9 680, so most real shapes are served by a distant neighbour. A Prediction
library replaces that sparse table with a model that is evaluated for the actual shape --
and Origami now knows gfx1101 (see P1), so it can be evaluated for a 60-CU part.

Dropping [7] also shrinks the file by roughly an order of magnitude.

    python3 to_prediction.py IN.yaml OUT.yaml [--metric DeviceEfficiency]
"""

import argparse
import pathlib

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--metric", default=None,
                    help="element [10]; default keeps whatever the source had")
    a = ap.parse_args()

    doc = yaml.safe_load(pathlib.Path(a.src).open())
    rows = len(doc[7]) if doc[7] else 0

    doc[6] = None
    doc[7] = None          # the shape table is what Prediction replaces
    doc[8] = None
    doc[9] = None
    if a.metric:
        doc[10] = a.metric
    doc[11] = "Prediction"

    with pathlib.Path(a.dst).open("w") as fh:
        yaml.safe_dump(doc, fh, default_flow_style=None, width=10**6, sort_keys=False)

    src_kb = pathlib.Path(a.src).stat().st_size / 1024
    dst_kb = pathlib.Path(a.dst).stat().st_size / 1024
    print(f"  {pathlib.Path(a.src).name}")
    print(f"    GridBased: {len(doc[5])} solutions + {rows} table rows  ({src_kb:.0f} KB)")
    print(f"    Prediction: {len(doc[5])} solutions, table dropped     ({dst_kb:.0f} KB)")
    print(f"    isa={doc[2]} arch={doc[1]} metric={doc[10]!r} type={doc[11]!r}")
    print(f"    -> {a.dst}")


if __name__ == "__main__":
    main()
