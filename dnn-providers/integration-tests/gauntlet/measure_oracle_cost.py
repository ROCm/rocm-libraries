#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Measures what the oracle costs per space (RFC 0023, section 14).

A space bounds the size of its samples by a cost-ceiling constraint, ``<cost> <= <ceiling>``,
whose ``reason`` calls it a cost ceiling (``b*hq*sq*skv*dqk <= 2**31``). The ceiling is there
because the reference the engine is graded against takes time in proportion to that cost. This
script measures the proportion: per space it builds a ladder of explicit rows that sweep the
cost expression from small to the space's current ceiling (every other axis kept at a legal
value), runs each rung through ``run_gauntlet_fuzz.py --row`` on the same binary and engine,
reads the ``reference_ms`` the binary reported for each rung from its ``outcomes.jsonl``, and fits

    reference_ms = a + b * cost

by least squares over the rungs that ran a reference (``PASS`` or ``WRONG``). It extrapolates the
cost at which the reference takes 60 s, the budget a case may spend on its oracle.

Synopsis::

    measure_oracle_cost.py --binary BIN [--test-article A] [--test-engine E] [--test-config C]
        [--binary-arg ARG ...] [--filter REGEX] [--spaces-dir DIR] [--work-dir DIR] [--json FILE]
        [--rungs N] [--ratio R] [--base-seed S] [--pin AXIS=VALUE ...] [--data-seed D]
        [--case-timeout T]

``--binary``, ``--test-article``, ``--test-engine``, ``--test-config``, ``--binary-arg``,
``--spaces-dir`` and ``--case-timeout`` go to the driver unchanged. ``--filter`` is a regex on
``<Op>/<Space>`` (default every space). A space whose cost-ceiling constraint names more than one
cost (``ResampleFwd/Default`` has two) is measured once per constraint.

The ladder has ``--rungs`` rungs (default 8) whose target costs fall by ``--ratio`` (default 4)
from the ceiling, so the last rung is the largest row the space allows. A rung is built from a
legal base row (the sample ``--base-seed`` draws, default 1, or the first seed after it that is
legal with the ``--pin`` values) by first lowering every axis the cost depends on, then raising
them in turn until the cost nears the rung's target. ``--pin AXIS=VALUE`` (a JSON value, else a
string; repeatable) holds an axis at a value in every row, for the engine's sake: an engine that
declines the base row's dtype declines every rung, and the space is reported ``unmeasured``.

Output: a table on stdout and ``--json`` (default ``<work-dir>/oracle_cost.json``)::

    {"budget_ms": 60000, "measurements": [{
        "space": "SdpaFwd/Default", "cost_driver": "b*hq*sq*skv*dqk", "ceiling": 2147483648,
        "status": "measured", "lane": "cpu", "fit_points": 7,
        "a_ms": 4.2, "b_ms_per_cost": 3.1e-8, "r2": 0.998,
        "cost_at_60s": 1.9e12, "predicted_ms_at_ceiling": 66000.0,
        "points": [{"case": "r0", "cost": 1048576, "outcome": "PASS",
                    "reference_ms": 36.8, "reference_lane": "cpu"}, ...]}, ...]}

A measurement is ``unmeasured`` (with ``reason``) when no rung ran a reference (every rung was
declined, say: the reason is the engine's), when fewer than two distinct costs did, or when the
space has no cost-ceiling constraint. When the rungs that ran a reference used both lanes, the
fit uses the lane with more rungs (the CPU one on a tie), and ``fit_points`` says how many.

The exit status is 0 when the run measured or reported every space, 2 on a usage or driver error.
"""

import argparse
import ast
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sampler  # noqa: E402
import space as space_lib  # noqa: E402

DRIVER = HERE / "run_gauntlet_fuzz.py"
BUDGET_MS = 60000.0
DEFAULT_RUNGS = 8
DEFAULT_RATIO = 4
DEFAULT_BASE_SEED = 1
DEFAULT_DATA_SEED = 1
DEFAULT_CASE_TIMEOUT = "10m"
# The outcomes whose reference ran to the end: the engine's output was compared with it.
FITTED_OUTCOMES = ("PASS", "WRONG")
# How many base seeds are tried before a set of pins is given up on.
MAX_BASE_ATTEMPTS = 1000


class OracleCostError(Exception):
    """A usage or driver error; the run cannot go on."""


class LadderError(Exception):
    """No ladder can be built for one space: it is reported ``unmeasured``, the run goes on."""


# --------------------------------------------------------------------------
# The cost driver of a space
# --------------------------------------------------------------------------


class CostDriver:
    """One cost-ceiling constraint of a space: ``expr`` (the cost) ``<= ceiling``."""

    def __init__(self, space, source, expr, ceiling, axes):
        self.space = space
        self.source = source
        self.expr = expr
        self.ceiling = ceiling
        # The axes the cost depends on, directly or through derived values, in axis order.
        self.axes = axes


def _names(source):
    return {
        n.id
        for n in ast.walk(ast.parse(source, mode="eval"))
        if isinstance(n, ast.Name)
    }


def cost_drivers(space):
    """The cost-ceiling constraints of ``space``: ``<expr> <= <constant>`` whose reason says
    ``cost``. The expression is read from the space, never assumed."""
    drivers = []
    known = set(space.axes) | set(space.derive)
    for constraint in space.constraints:
        if "cost" not in (constraint.reason or "").lower():
            continue
        tree = ast.parse(constraint.rule, mode="eval").body
        if not (
            isinstance(tree, ast.Compare)
            and len(tree.ops) == 1
            and isinstance(tree.ops[0], ast.LtE)
        ):
            continue
        lhs = ast.get_source_segment(constraint.rule, tree.left)
        rhs = ast.get_source_segment(constraint.rule, tree.comparators[0])
        try:
            ceiling = space_lib._Expr(rhs, space.space_id, "cost ceiling", ()).evaluate(
                {}
            )
            expr = space_lib._Expr(lhs, space.space_id, "cost driver", known)
        except space_lib.SpaceError:
            continue  # a bound that names axes is not a ceiling
        reach = _names(lhs)
        grew = True
        while grew:
            grew = False
            for name, derived in space.derive.items():
                if name in reach:
                    more = _names(derived.source) - reach
                    if more:
                        reach |= more
                        grew = True
        axes = [name for name in space.axes if name in reach]
        if axes and isinstance(ceiling, (int, float)) and ceiling > 0:
            drivers.append(CostDriver(space, lhs.strip(), expr, ceiling, axes))
    return drivers


def evaluate(driver, axes):
    """``(legal, cost)`` of the axis values: every constraint of the space holds, and the value
    of the cost expression. ``(False, None)`` for values the space cannot evaluate."""
    try:
        env = driver.space.derive_env(axes)
        legal = driver.space.first_violation(env) is None
        return legal, driver.expr.evaluate(env)
    except space_lib.SpaceError:
        return False, None


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def _options(axis):
    """The values of an axis a ladder may take, ascending: a ``values`` axis (numbers sorted,
    anything else as declared), a ``range`` axis as its ends and declared boundaries."""
    if axis.is_range:
        return sorted({axis.lo, axis.hi, *axis.boundaries})
    values = []
    for value in axis.values:  # a list may repeat a value to weight the draw
        if value not in values:
            values.append(value)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return sorted(values)
    return values


def _next_up(axis, value):
    """The next larger value of an axis, or None at its top. A ``range`` axis doubles."""
    if axis.is_range:
        if value >= axis.hi:
            return None
        return min(axis.hi, max(value + 1, value * 2))
    options = _options(axis)
    index = options.index(value)
    return options[index + 1] if index + 1 < len(options) else None


def find_base(space, base_seed, data_seed, pins):
    """The axis values of the first legal row from ``base_seed`` on that carries the pins."""
    last = None
    for seed in range(base_seed, base_seed + MAX_BASE_ATTEMPTS):
        try:
            axes = dict(sampler.sample(space, seed).axis_values)
        except sampler.InfeasibleSpaceError as e:
            raise LadderError(str(e)) from None
        axes.update({name: value for name, value in pins.items() if name in space.axes})
        try:
            sampler.sample_from_axes(space, axes, data_seed)
        except sampler.InvalidRowError as e:
            last = e
            continue
        return axes
    raise LadderError(
        f"space '{space.space_id}': no legal base row in {MAX_BASE_ATTEMPTS} seeds from"
        f" {base_seed} with the pins {pins}: {last}"
    )


def rung_axes(driver, base, movable, target):
    """Axis values whose cost is as near ``target`` from below as the movable axes allow."""
    space = driver.space
    axes = dict(base)
    # Lowest first: each axis the cost depends on goes to its smallest value that is legal.
    for name in movable:
        for value in _options(space.axes[name]):
            if evaluate(driver, {**axes, name: value})[0]:
                axes[name] = value
                break
    # Then one step up at a time, round-robin, so no axis takes the whole budget.
    progressed = True
    while progressed:
        progressed = False
        for name in movable:
            value = _next_up(space.axes[name], axes[name])
            if value is None:
                continue
            legal, cost = evaluate(driver, {**axes, name: value})
            if legal and cost <= target:
                axes[name] = value
                progressed = True
    # A range axis doubles in steps; settle the last of the gap by bisection.
    for name in movable:
        axis = space.axes[name]
        if not axis.is_range:
            continue
        low, high = axes[name], axis.hi
        while low < high:
            mid = (low + high + 1) // 2
            legal, cost = evaluate(driver, {**axes, name: mid})
            if legal and cost <= target:
                low = mid
            else:
                high = mid - 1
        axes[name] = low
    return axes


def build_ladder(driver, rungs, ratio, base_seed, data_seed, pins):
    """``[(cost, axes)]`` ascending, one entry per distinct cost: rows of ``driver.space`` that
    sweep the cost expression from small to the ceiling."""
    base = find_base(driver.space, base_seed, data_seed, pins)
    movable = [name for name in driver.axes if name not in pins]
    by_cost = {}
    for i in range(rungs):
        target = driver.ceiling / ratio ** (rungs - 1 - i)
        axes = rung_axes(driver, base, movable, target)
        legal, cost = evaluate(driver, axes)
        if legal and cost > 0:
            by_cost.setdefault(cost, axes)
    return sorted(by_cost.items(), key=lambda item: item[0])


# --------------------------------------------------------------------------
# Running a rung
# --------------------------------------------------------------------------


def run_rung(args, space, axes, case_id, work_dir):
    """The outcome record of one rung: the driver's ``--row`` run in ``work_dir``."""
    work_dir.mkdir(parents=True, exist_ok=True)
    row = work_dir / "row.json"
    row.write_text(
        json.dumps(
            {
                "space": space.space_id,
                "axes": axes,
                "data_seed": args.data_seed,
                "case_id": case_id,
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(DRIVER),
        "--binary",
        args.binary,
        "--work-dir",
        str(work_dir / "work"),
        "--spaces-dir",
        args.spaces_dir,
        "--case-timeout",
        args.case_timeout,
        "--row",
        str(row),
    ]
    for flag, value in (
        ("--test-article", args.test_article),
        ("--test-engine", args.test_engine),
        ("--test-config", args.test_config),
    ):
        if value is not None:
            command += [flag, value]
    for extra in args.binary_arg:
        command += ["--binary-arg", extra]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    (work_dir / "driver.log").write_text(
        f"$ {' '.join(command)}\n{done.stdout}\n{done.stderr}", encoding="utf-8"
    )
    outcomes = work_dir / "work" / "outcomes.jsonl"
    # 0 and 1 are a run (1: the case was WRONG or CRASH); 2 is a usage or space error.
    if done.returncode not in (0, 1) or not outcomes.exists():
        raise OracleCostError(
            f"{space.space_id} {case_id}: the driver exited {done.returncode} without an"
            f" outcome: {(done.stderr or done.stdout).strip()[-400:]}"
        )
    lines = [
        ln for ln in outcomes.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    if len(lines) != 1:
        raise OracleCostError(
            f"{space.space_id} {case_id}: expected one outcome record, found {len(lines)}"
        )
    return json.loads(lines[0])


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


def least_squares(xs, ys):
    """``(a, b, r2)`` of ``y = a + b * x``. Needs two distinct ``x``."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    residual = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    mean = sy / n
    total = sum((y - mean) ** 2 for y in ys)
    r2 = 1.0 if total == 0 else 1.0 - residual / total
    return a, b, r2


def _round(value):
    return float(f"{value:.6g}")


def summarize(driver, points):
    """The measurement of one driver from the recorded points of its rungs."""
    result = {
        "space": driver.space.space_id,
        "cost_driver": driver.source,
        "ceiling": driver.ceiling,
        "points": points,
    }
    timed = [
        p
        for p in points
        if p["outcome"] in FITTED_OUTCOMES and p.get("reference_ms") is not None
    ]
    if not timed:
        counts = collections.Counter(p["outcome"] for p in points)
        reasons = collections.Counter(p.get("reason", "") for p in points)
        top = reasons.most_common(1)[0][0] if reasons else ""
        shown = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        result.update(
            status="unmeasured",
            reason=f"no rung ran a reference ({shown or 'no rungs built'}); {top}".rstrip(
                "; "
            ),
        )
        return result
    lanes = collections.Counter(p["reference_lane"] for p in timed)
    lane = max(lanes, key=lambda name: (lanes[name], name == "cpu"))
    fitted = [p for p in timed if p["reference_lane"] == lane]
    if len({p["cost"] for p in fitted}) < 2:
        result.update(
            status="unmeasured",
            lane=lane,
            reason=f"only {len(fitted)} rung(s) of one cost ran a {lane} reference; a fit"
            " needs two distinct costs",
        )
        return result
    a, b, r2 = least_squares(
        [p["cost"] for p in fitted], [p["reference_ms"] for p in fitted]
    )
    result.update(
        status="measured",
        lane=lane,
        fit_points=len(fitted),
        a_ms=_round(a),
        b_ms_per_cost=b,
        r2=_round(r2),
        cost_at_60s=_round((BUDGET_MS - a) / b) if b > 0 else None,
        predicted_ms_at_ceiling=_round(a + b * driver.ceiling),
    )
    return result


# --------------------------------------------------------------------------
# One space
# --------------------------------------------------------------------------


def measure_driver(args, driver, index, pins):
    """The measurement of one cost driver: build its ladder, run it, fit it."""
    space = driver.space
    ladder = build_ladder(
        driver, args.rungs, args.ratio, args.base_seed, args.data_seed, pins
    )
    tag = re.sub(r"[^A-Za-z0-9_]", "_", space.space_id) + f"_{index}"
    points = []
    for i, (cost, axes) in enumerate(ladder):
        case_id = f"r{i}"
        record = run_rung(
            args, space, axes, case_id, Path(args.work_dir) / tag / case_id
        )
        point = {
            "case": case_id,
            "cost": cost,
            "outcome": record.get("outcome"),
            "reference_ms": record.get("reference_ms"),
            "reference_lane": record.get("reference_lane"),
        }
        if point["reference_ms"] is None:
            point["reason"] = record.get("reason") or ""
        points.append(point)
        print(
            f"  {space.space_id} {case_id}: cost {cost}, {point['outcome']}"
            + (
                ""
                if point["reference_ms"] is None
                else f", reference {point['reference_ms']} ms"
            ),
            flush=True,
        )
    return summarize(driver, points)


def measure_space(args, space, pins):
    drivers = cost_drivers(space)
    if not drivers:
        return [
            {
                "space": space.space_id,
                "status": "unmeasured",
                "reason": "the space has no cost-ceiling constraint",
                "points": [],
            }
        ]
    results = []
    for index, driver in enumerate(drivers):
        try:
            results.append(measure_driver(args, driver, index, pins))
        except LadderError as e:
            results.append(
                {
                    "space": space.space_id,
                    "cost_driver": driver.source,
                    "ceiling": driver.ceiling,
                    "status": "unmeasured",
                    "reason": str(e),
                    "points": [],
                }
            )
    return results


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def parse_pin(text):
    name, sep, raw = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"--pin wants AXIS=VALUE, got {text!r}")
    try:
        return name, json.loads(raw)
    except ValueError:
        return name, raw


def build_parser():
    p = argparse.ArgumentParser(
        prog="measure_oracle_cost.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--binary",
        required=True,
        help="hipdnn_integration_tests, as the driver takes it",
    )
    p.add_argument("--test-article", help="passed to the binary as --test-article")
    p.add_argument("--test-engine", help="passed to the binary as --test-engine")
    p.add_argument("--test-config", help="passed to the binary as --test-config")
    p.add_argument(
        "--binary-arg",
        action="append",
        default=[],
        help="extra binary argument (repeatable)",
    )
    p.add_argument("--filter", help="regex on <Op>/<Space> (default every space)")
    p.add_argument("--spaces-dir", default=str(space_lib.DEFAULT_SPACES_DIR))
    p.add_argument(
        "--work-dir", default="oracle-cost-work", help="one subdirectory per rung"
    )
    p.add_argument("--json", help="the report (default <work-dir>/oracle_cost.json)")
    p.add_argument(
        "--rungs", type=int, default=DEFAULT_RUNGS, help="rungs per cost driver"
    )
    p.add_argument(
        "--ratio", type=float, default=DEFAULT_RATIO, help="cost ratio between rungs"
    )
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    p.add_argument("--case-timeout", default=DEFAULT_CASE_TIMEOUT)
    p.add_argument(
        "--pin",
        action="append",
        default=[],
        type=parse_pin,
        metavar="AXIS=VALUE",
        help="hold an axis at a value in every row (JSON value, else string; repeatable)",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rungs < 1:
        print("measure_oracle_cost.py: error: --rungs must be >= 1", file=sys.stderr)
        return 2
    if args.ratio <= 1:
        print("measure_oracle_cost.py: error: --ratio must be > 1", file=sys.stderr)
        return 2
    try:
        spaces = space_lib.load_spaces(args.spaces_dir)
        if args.filter:
            pattern = re.compile(args.filter)
            spaces = [s for s in spaces if pattern.search(s.space_id)]
        if not spaces:
            raise OracleCostError(
                f"no space in {args.spaces_dir} matches {args.filter!r}"
            )
        pins = dict(args.pin)
        stray = sorted(name for name in pins if not any(name in s.axes for s in spaces))
        if stray:
            raise OracleCostError(f"--pin: no selected space has the axes {stray}")
        measurements = []
        for space in spaces:
            measurements += measure_space(args, space, pins)
    except (OracleCostError, space_lib.SpaceError, re.error) as e:
        print(f"measure_oracle_cost.py: error: {e}", file=sys.stderr)
        return 2
    report = {"budget_ms": BUDGET_MS, "measurements": measurements}
    out = Path(args.json) if args.json else Path(args.work_dir) / "oracle_cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"report: {out}")
    return 0


def print_report(report):
    print("")
    for m in report["measurements"]:
        head = f"{m['space']}  cost = {m.get('cost_driver', '-')}  ceiling {m.get('ceiling', '-')}"
        if m["status"] != "measured":
            print(f"{head}\n  unmeasured: {m['reason']}")
            continue
        cost_at_60s = m["cost_at_60s"]
        print(
            f"{head}\n  {m['lane']} reference: ms = {m['a_ms']} + {m['b_ms_per_cost']:.4g} * cost"
            f" (r2 {m['r2']}, {m['fit_points']} rungs); 60 s at cost "
            f"{'never (no positive slope)' if cost_at_60s is None else cost_at_60s};"
            f" {m['predicted_ms_at_ceiling']} ms at the ceiling"
        )


if __name__ == "__main__":
    sys.exit(main())
