#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Recompute the polynomial coefficients of ``rocke.helpers.elementary``.

Each table is the minimax polynomial (relative error, Remez exchange in ``mpmath``) of one
reduced function on its interval, with the coefficients rounded to the nearest double. The
module stores the printed hex literals; ``tests/helpers/test_elementary.py`` re-checks the
error bound of the stored doubles against ``mpmath`` (it does not rerun the exchange).

Usage:
  python rocke/platform/tools/gen_elementary_coefficients.py          # print the tables
  python rocke/platform/tools/gen_elementary_coefficients.py --check  # exit 1 if they differ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import mpmath as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

mp.mp.dps = 60


def _sin_s(z):
    # (sin(sqrt z) / sqrt z - 1) / z
    if z == 0:
        return mp.mpf(-1) / 6
    r = mp.sqrt(z)
    return (mp.sin(r) / r - 1) / z


def _cos_c(z):
    # (cos(sqrt z) - 1 + z / 2) / z^2
    if z == 0:
        return mp.mpf(1) / 24
    return (mp.cos(mp.sqrt(z)) - 1 + z / 2) / (z * z)


def _atan_a(z):
    # (atan(sqrt z) / sqrt z - 1) / z
    if z == 0:
        return mp.mpf(-1) / 3
    r = mp.sqrt(z)
    return (mp.atan(r) / r - 1) / z


def _asin_r(z):
    # (asin(sqrt z) / sqrt z - 1) / z
    if z == 0:
        return mp.mpf(1) / 6
    r = mp.sqrt(z)
    return (mp.asin(r) / r - 1) / z


def _sinh_h(z):
    # (sinh(sqrt z) / sqrt z - 1) / z
    if z == 0:
        return mp.mpf(1) / 6
    r = mp.sqrt(z)
    return (mp.sinh(r) / r - 1) / z


def _exp_p(r):
    # (e^r - 1 - r - r^2 / 2) / r^3
    if r == 0:
        return mp.mpf(1) / 6
    return (mp.expm1(r) - r - r * r / 2) / (r**3)


def _log_g(z):
    # (2 atanh(sqrt z) / sqrt z - 2) / z  (log1p(f) = f - f^2/2 + s (f^2/2 + z g(z)),
    # s = f / (2 + f), z = s^2)
    if z == 0:
        return mp.mpf(2) / 3
    s = mp.sqrt(z)
    return (2 * mp.atanh(s) / s - 2) / z


#: (name, function, interval, degree). The intervals carry a small margin past the reduced
#: ranges the module produces (see the module docstring of rocke.helpers.elementary).
TABLES: List[Tuple[str, Callable, Tuple[float, float], int]] = [
    ("SIN_S", _sin_s, (0.0, 0.6169), 6),
    ("COS_C", _cos_c, (0.0, 0.6169), 6),
    ("ATAN_A", _atan_a, (0.0, 0.19141), 11),
    ("ASIN_R", _asin_r, (0.0, 0.25), 13),
    ("SINH_H", _sinh_h, (0.0, 1.0), 7),
    ("EXP_P", _exp_p, (-0.34658, 0.34658), 10),
    ("LOG_G", _log_g, (0.0, 0.02944), 7),
]


def _poly(c: Sequence, z):
    acc = mp.mpf(0)
    for ck in reversed(c):
        acc = acc * z + ck
    return acc


def _rel_err(c, f, z):
    fz = f(z)
    return (_poly(c, z) - fz) / fz


def _extrema(c, f, a, b, n_grid=3000):
    """Local extrema of the relative error on [a, b] (end points included)."""
    xs = [
        (a + b) / 2 - (b - a) / 2 * mp.cos(mp.pi * i / (n_grid - 1))
        for i in range(n_grid)
    ]
    es = [_rel_err(c, f, x) for x in xs]
    idx = [0]
    for i in range(1, n_grid - 1):
        if (es[i] - es[i - 1]) * (es[i + 1] - es[i]) <= 0:
            idx.append(i)
    idx.append(n_grid - 1)
    out = []
    for i in idx:
        x = xs[i]
        if 0 < i < n_grid - 1:
            lo, hi = xs[i - 1], xs[i + 1]
            sign = 1 if es[i] > 0 else -1
            # golden-section refinement of |e|
            g = (mp.sqrt(5) - 1) / 2
            for _ in range(60):
                x1 = hi - g * (hi - lo)
                x2 = lo + g * (hi - lo)
                if sign * _rel_err(c, f, x1) > sign * _rel_err(c, f, x2):
                    hi = x2
                else:
                    lo = x1
            x = (lo + hi) / 2
        out.append((x, _rel_err(c, f, x)))
    return out


def remez(f, a, b, deg, iters=40):
    a, b = mp.mpf(a), mp.mpf(b)
    n = deg + 2
    nodes = [(a + b) / 2 - (b - a) / 2 * mp.cos(mp.pi * i / (n - 1)) for i in range(n)]
    c, emax = None, None
    for _ in range(iters):
        A = mp.matrix(n, n)
        y = mp.matrix(n, 1)
        for i, z in enumerate(nodes):
            for k in range(deg + 1):
                A[i, k] = z**k
            A[i, deg + 1] = (-1) ** i * abs(f(z))
            y[i] = f(z)
        try:
            sol = mp.lu_solve(A, y)
        except ZeroDivisionError:
            if c is None:
                raise
            break  # two nodes merged: the exchange has converged
        c = [sol[k] for k in range(deg + 1)]
        E = abs(sol[deg + 1])
        ext = _extrema(c, f, a, b)
        # keep alternating extrema: merge runs of one sign by their largest member
        alt = []
        for x, e in ext:
            if alt and (alt[-1][1] > 0) == (e > 0):
                if abs(e) > abs(alt[-1][1]):
                    alt[-1] = (x, e)
            else:
                alt.append((x, e))
        while len(alt) > n:
            # drop the smaller end
            if abs(alt[0][1]) < abs(alt[-1][1]):
                alt.pop(0)
            else:
                alt.pop()
        emax = max(abs(e) for _, e in ext)
        if len(alt) == n:
            nodes = [x for x, _ in alt]
        if emax - E <= E * mp.mpf("1e-6"):
            break
    return c, emax


def max_rel_err_of_doubles(f, interval, coefs: Sequence[float], n=4000) -> float:
    """Max relative error of the polynomial with the double coefficients ``coefs`` (exact
    arithmetic) on a Chebyshev grid of ``n`` points plus the end points."""
    a, b = mp.mpf(interval[0]), mp.mpf(interval[1])
    c = [mp.mpf(x) for x in coefs]
    worst = mp.mpf(0)
    for i in range(n):
        z = (a + b) / 2 - (b - a) / 2 * mp.cos(mp.pi * i / (n - 1))
        worst = max(worst, abs(_rel_err(c, f, z)))
    return float(worst)


def tables():
    out = {}
    for name, f, (a, b), deg in TABLES:
        c, emax = remez(f, a, b, deg)
        dbl = [float(x) for x in c]
        out[name] = (dbl, float(emax), max_rel_err_of_doubles(f, (a, b), dbl))
    return out


def render(tabs) -> str:
    lines = []
    for name, (dbl, emax, edbl) in tabs.items():
        lines.append(
            f"# minimax rel. error {emax:.3e} (2^{float(mp.log(emax, 2)):.1f}); "
            f"double coefficients {edbl:.3e}"
        )
        lines.append(f"{name} = (")
        for x in dbl:
            lines.append(f'    _h("{x.hex()}"),')
        lines.append(")")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    tabs = tables()
    if a.check:
        from rocke.helpers import elementary as el

        bad = [
            name
            for name, (dbl, _, _) in tabs.items()
            if tuple(dbl) != tuple(getattr(el, name))
        ]
        if bad:
            print("stale coefficient tables: " + ", ".join(bad))
            return 1
        print("coefficient tables are current")
        return 0
    sys.stdout.write(render(tabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
