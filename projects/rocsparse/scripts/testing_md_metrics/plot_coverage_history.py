#!/usr/bin/env python3
# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

"""Plot the rocSPARSE develop coverage history from coverage_history.csv.

Usage: plot_coverage_history.py [coverage_history.csv] [out.svg]

Blue is the develop coverage job (the `rocSPARSE` Codecov flag). Green squares are
the `rocSPARSE-unit` flag, which has no reading before 14 Oct 2026. The two lines do
not measure the same files and cannot be averaged or subtracted.
"""
import csv
import datetime as dt
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# (date, label, color, linestyle). Keep in sync with the Coverage history section.
EVENTS = [
    ("2026-08-12", "12 Aug unit tests", "#16803d", ":"),
    ("2026-09-23", "23 Sep testing.md", "#c25a0c", "--"),
    ("2026-11-12", "12 Nov #15395", "#6e6e6e", ":"),
    ("2026-11-18", "18 Nov #15569 (union)", "#6e6e6e", ":"),
]
TARGET = 80.0


def read(path):
    comb, unit = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            day = dt.date.fromisoformat(row["date"])
            if row["develop_combined"]:
                comb.append((day, float(row["develop_combined"])))
            if row["unit_only"]:
                unit.append((day, float(row["unit_only"])))
    return comb, unit


def main(csv_path="coverage_history.csv", out="coverage-history.svg"):
    comb, unit = read(csv_path)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(*zip(*comb), color="#2563eb", marker="o", ms=2.5, lw=2,
            label="Develop job (rocSPARSE flag), daily")
    ax.plot(*zip(*unit), color="#16803d", marker="s", ms=3.5, lw=2,
            label="Unit-only (rocSPARSE-unit flag)")
    ax.axhline(TARGET, color="#6e6e6e", ls="--", lw=1.2, label="80% target (codecov.yml)")
    for day, label, color, style in EVENTS:
        x = dt.date.fromisoformat(day)
        ax.axvline(x, color=color, ls=style, lw=1.4)
        ax.text(x, 60.6, " " + label, rotation=90, va="bottom", ha="right",
                fontsize=7.5, color=color)
    ax.set_ylim(60, 90)
    ax.set_xlim(comb[0][0], comb[-1][0] + dt.timedelta(days=4))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Line coverage")
    ax.set_xlabel(f"2026 (last point {comb[-1][0].isoformat()})")
    ax.grid(axis="y", color="#dddddd")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, metadata={"Date": None})
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
