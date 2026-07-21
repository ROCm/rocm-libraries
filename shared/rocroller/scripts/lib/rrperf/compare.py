# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Result comparison routines."""

import argparse
import datetime
import io
import os
import re
import statistics
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rrperf
import rrperf.args as args
import scipy.stats
from rrperf.problems import GEMMResult, RRPerfResult
from rrperf.specs import MachineSpecs


def priority_problems():
    """Load priority problem args from rrsuites.py."""
    return rrperf.utils.load_suite("priority_problems")


@dataclass
class ComparisonResult:
    mean: list[float]
    median: list[float]
    moods_pval: float

    results: list[Any] = field(repr=False)

    problem: str


@dataclass
class PlotData:
    timestamp: list[float] = field(default_factory=list)
    commit: list[str] = field(default_factory=list)
    median: list[float] = field(default_factory=list)
    min: list[float] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    kernel: list[float] = field(default_factory=list)
    machine: list[int] = field(default_factory=list)
    box_data: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["timestamp", "commit", "runs"])
    )


@dataclass
class ComparisonCounts:
    compared: int = 0
    significant: int = 0
    insignificant: int = 0
    reference_only: int = 0
    candidate_only: int = 0

    @property
    def not_compared_total(self) -> int:
        return self.reference_only + self.candidate_only


class PerformanceRun:
    timestamp: float
    commit: str
    directory: str
    machine_spec: MachineSpecs
    results: OrderedDict

    def __init__(self, timestamp, commit, directory, machine_spec, results):
        self.timestamp = timestamp
        self.commit = commit
        self.directory = directory
        self.machine_spec = machine_spec
        self.results = results

    def __lt__(self, other):
        if self.timestamp == other.timestamp:
            return self.directory < other.directory
        return self.timestamp < other.timestamp

    def name(self):
        return os.path.basename(self.directory)

    @staticmethod
    def get_comparable_tokens(ref, runs):
        # compute intersection
        common = set(ref.results.keys())
        for run in runs:
            common = common.intersection(set(run.results.keys()))

        common = sorted(common)

        return common

    @staticmethod
    def get_all_tokens(runs):
        return sorted({token for run in runs for token in run.results})

    @staticmethod
    def get_all_specs(runs):
        return sorted({run.machine_spec for run in runs})

    @staticmethod
    def get_timestamp(wrkdir):
        try:
            return datetime.datetime.fromtimestamp(
                float((wrkdir / "timestamp.txt").read_text().strip())
            )
        except Exception:
            try:
                return datetime.datetime.strptime(wrkdir.stem[0:10], "%Y-%m-%d")
            except Exception:
                return datetime.datetime.fromtimestamp(0)

    @staticmethod
    def get_commit(wrkdir):
        try:
            return (wrkdir / "git-commit.txt").read_text().strip()[0:8]
        except Exception:
            try:
                return wrkdir.stem[11:19]
            except Exception:
                return "XXXXXXXX"

    @staticmethod
    def load_perf_runs(directories):
        perf_runs = list()
        for directory in directories:
            wrkdir = Path(directory)
            results = OrderedDict()
            for path in wrkdir.glob("*.yaml"):
                try:
                    result = rrperf.problems.load_results(path)
                except Exception as e:
                    print('Error loading results in "{}": {}'.format(path, e))
                    raise
                for element in result:
                    if element.run_invariant_token in results:
                        # TODO: Handle result files that have multiple results in
                        # a single yaml file.
                        results[element.run_invariant_token] = element
                    else:
                        results[element.run_invariant_token] = element
            spec = rrperf.specs.load_machine_specs(wrkdir / "machine-specs.txt")
            timestamp = PerformanceRun.get_timestamp(wrkdir)
            commit = PerformanceRun.get_commit(wrkdir)
            perf_runs.append(
                PerformanceRun(
                    timestamp=timestamp,
                    commit=commit,
                    directory=directory,
                    machine_spec=spec,
                    results=results,
                )
            )

        return perf_runs


def summary_statistics(perf_runs):
    """Compare results in `results_by_directory` and compute summary statistics.

    The first run is the reference run.
    """

    # first directory is reference, remaining are runs
    ref = perf_runs[0]
    runs = perf_runs[1:]
    common = PerformanceRun.get_comparable_tokens(ref, runs)
    # compute comparison statistics
    stats = defaultdict(dict)

    for token in common:
        A = ref.results[token]
        ka = np.asarray(A.kernelExecute).ravel()

        ka_median = statistics.median(ka) if ka.size > 0 else 0
        ka_mean = statistics.mean(ka) if ka.size > 0 else 0

        for run in runs:
            B = run.results[token]
            kb = np.asarray(B.kernelExecute).ravel()

            kb_median = statistics.median(kb) if kb.size > 0 else 0
            kb_mean = statistics.mean(kb) if kb.size > 0 else 0

            try:
                _, p, _, _ = scipy.stats.median_test(ka, kb)
            except Exception:
                p = 1.0
            stats[run][token] = A.run_invariant_token, ComparisonResult(
                mean=[ka_mean, kb_mean],
                median=[ka_median, kb_median],
                moods_pval=p,
                results=[A, B],
                problem=A.problem_token(priority_problems()),
            )
    return stats


def comparison_counts(perf_runs, summary, threshold=0.05):
    counts = ComparisonCounts()

    if len(perf_runs) < 2:
        return counts

    ref = perf_runs[0]
    runs = perf_runs[1:]
    compared_tokens = set(PerformanceRun.get_comparable_tokens(ref, runs))
    counts.compared = len(compared_tokens)

    token_pvals = defaultdict(list)
    for run_summary in summary.values():
        for token, (_, comparison) in run_summary.items():
            token_pvals[token].append(comparison.moods_pval)

    for token in compared_tokens:
        pvals = token_pvals.get(token, [])
        if any(p < threshold for p in pvals):
            counts.significant += 1
        else:
            counts.insignificant += 1

    ref_tokens = set(ref.results.keys())
    candidate_tokens = set()
    for run in runs:
        candidate_tokens.update(run.results.keys())

    counts.reference_only = len(ref_tokens - compared_tokens)
    counts.candidate_only = len(candidate_tokens - compared_tokens)

    return counts


def summary_as_df(summary, ResultType):
    rows = []
    for run in summary:
        for result in summary[run]:
            token, comparison = summary[run][result]
            A, B = comparison.results
            if not isinstance(A, ResultType):
                continue
            row = A.compact()
            row.update(
                {
                    "meanA(ns)": comparison.mean[0],
                    "meanB(ns)": comparison.mean[1],
                    "medianA(ns)": comparison.median[0],
                    "medianB(ns)": comparison.median[1],
                    "pval": comparison.moods_pval,
                    "reldiff": (
                        100
                        * (comparison.median[1] - comparison.median[0])
                        / comparison.median[0]
                        if comparison.median[0] != 0
                        else 0.0
                    ),
                    "genA(ns)": A.kernelGenerate,
                    "genB(ns)": B.kernelGenerate,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def significant_changes(summary, threshold=0.05):
    result_diffs = dict()
    for run in summary:
        for result in summary[run]:
            token, comparison = summary[run][result]
            A, B = comparison.results
            percent = (
                ((comparison.median[1] - comparison.median[0]) * 100.0)
                / comparison.median[0]
                if comparison.median[0] != 0
                else 0.0
            )
            if comparison.moods_pval < threshold:
                sign = "+" if percent < 0 else "-"
                result_diffs[
                    A.problem_token(priority_problems()) + A.run_invariant_token
                ] = (
                    f"{sign} {(abs(percent)):6.2f}% "
                    f"| p={comparison.moods_pval:.4e} "
                    f"\n\t| {A.problem_token(priority_problems())}"
                    f"| {A.solution_token} "
                    f"| {token}\n"
                )
    keys = sorted(result_diffs.keys())
    result_diff = ""
    for key in keys:
        result_diff += result_diffs[key]
    return result_diff


def resource_usage_changes(
    summary: dict[str, dict[str, tuple[str, ComparisonResult]]],
) -> str:
    """Report resource usage changes between runs."""
    resource_diffs = dict()
    for run in summary:
        for result in summary[run]:
            token, comparison = summary[run][result]
            A: RRPerfResult
            B: RRPerfResult
            A, B = comparison.results

            sgprA = A.sgprCount
            sgprB = B.sgprCount
            vgprA = A.vgprCount
            vgprB = B.vgprCount
            agprA = A.agprCount
            agprB = B.agprCount
            ldsBytesA = A.ldsBytes
            ldsBytesB = B.ldsBytes

            if (
                sgprA != sgprB
                or vgprA != vgprB
                or agprA != agprB
                or ldsBytesA != ldsBytesB
            ):
                # Have diff viewer show red if *any* resource increased, green otherwise
                any_increase = (
                    sgprB > sgprA
                    or vgprB > vgprA
                    or agprB > agprA
                    or ldsBytesB > ldsBytesA
                )
                line_sign = "-" if any_increase else "+"

                changes = []
                register_pairs = [
                    ("SGPR", sgprA, sgprB),
                    ("VGPR", vgprA, vgprB),
                    ("AGPR", agprA, agprB),
                ]

                for register_type, value_a, value_b in register_pairs:
                    if value_a != value_b:
                        diff = value_b - value_a
                        sign = "+" if diff > 0 else ""
                        changes.append(
                            f"{register_type}: {value_a} -> {value_b} ({sign}{diff})"
                        )

                if ldsBytesA != ldsBytesB:
                    diff = ldsBytesB - ldsBytesA
                    sign = "+" if diff > 0 else ""
                    changes.append(
                        f"LDS: {ldsBytesA} -> {ldsBytesB} bytes ({sign}{diff})"
                    )

                resource_diffs[
                    A.problem_token(priority_problems()) + A.run_invariant_token
                ] = (
                    f"{line_sign} {' | '.join(changes)}"
                    f"\n\t| {A.problem_token(priority_problems())}"
                    f"| {A.solution_token} "
                    f"| {token}\n"
                )

    keys = sorted(resource_diffs.keys())
    result_diff = ""
    for key in keys:
        result_diff += resource_diffs[key]
    return result_diff


def markdown_summary(md, perf_runs, detail=False):
    """Create Markdown report of summary statistics."""

    summary = summary_statistics(perf_runs)
    counts = comparison_counts(perf_runs, summary)

    header = [
        "Problem",
        "Median Diff %",
        "Moods p-val",
        "Gen A (ns)",
        "Gen B (ns)",
    ]

    result_diff = significant_changes(summary)

    print("## Comparison Summary\n", file=md)
    print(f"- Compared: {counts.compared}", file=md)
    print(f"- Significant diffs: {counts.significant}", file=md)
    print(f"- Insignificant diffs: {counts.insignificant}", file=md)
    print(f"- Not compared (reference-only): {counts.reference_only}", file=md)
    print(f"- Not compared (candidate-only): {counts.candidate_only}", file=md)
    print(f"- Not compared (total): {counts.not_compared_total}\n", file=md)

    result_table = ""
    if detail:
        for run in summary:
            for result in summary[run]:
                token, comparison = summary[run][result]
                A, B = comparison.results
                percent = (
                    ((comparison.median[1] - comparison.median[0]) * 100.0)
                    / comparison.median[0]
                    if comparison.median[0] != 0
                    else 0.0
                )
                row_str = [
                    f"{token}",
                    f"{(percent):.2f}%",
                    f"{comparison.moods_pval:0.4e}",
                    f"{A.kernelGenerate:,.0f}",
                    f"{B.kernelGenerate:,.0f}",
                ]
                result_table += " | ".join(row_str) + "\n"

    if len(result_diff) > 0:
        print("```diff", file=md)
        print(
            "@@            Significant (p-val <0.05) Performance Diffs            @@",
            file=md,
        )
        print("=" * 100, file=md)
        print(result_diff, file=md)
        print("```\n\n", file=md)
    else:
        print(
            ":heavy_check_mark: **_No Statistically Significant Performance Diff_** :heavy_check_mark:\n\n",
            file=md,
        )

    if detail:
        print("<details><summary>Full table of results</summary>\n", file=md)
        print(" | ".join(header), file=md)
        print(" | ".join(["---"] * len(header)), file=md)
        print(result_table, file=md)
        print("\n</details>", file=md)


def markdown_resource_summary(md, perf_runs):
    """Create Markdown report of resource usage changes."""

    summary = summary_statistics(perf_runs)
    resource_diff = resource_usage_changes(summary)

    if len(resource_diff) > 0:
        print("```diff", file=md)
        print(
            "@@                    Resource Usage Changes                         @@",
            file=md,
        )
        print("=" * 100, file=md)
        print(resource_diff, file=md)
        print("```\n\n", file=md)
    else:
        print(
            ":heavy_check_mark: **_No Resource Usage Changes_** :heavy_check_mark:\n\n",
            file=md,
        )


def html_overview_table(html_file, summary, problems):
    """Create HTML table with summary statistics."""

    print("<table><tr><td>", file=html_file)

    header = [
        "Problem",
        "Mean A (ns)",
        "Mean B (ns)",
        "Median A (ns)",
        "Median B (ns)",
        "Median Diff %",
        "Moods p-val",
        "Run A (ref)",
        "Run B",
        "Gen A (ns)",
        "Gen B (ns)",
        "SGPR A",
        "SGPR B",
        "VGPR A",
        "VGPR B",
        "AGPR A",
        "AGPR B",
        "LDS A",
        "LDS B",
    ]

    print("</td><td> ".join(header), file=html_file)
    print("</td></tr>", file=html_file)

    for run in summary:
        for i, result in enumerate(summary[run]):
            token, comparison = summary[run][result]
            A: RRPerfResult
            B: RRPerfResult
            A, B = comparison.results
            relative_diff = (
                (comparison.median[1] - comparison.median[0]) / comparison.median[0]
                if comparison.median[0] != 0
                else 0.0
            )
            link_target = (
                problems.index(comparison.problem)
                if comparison.problem in problems
                else i
            )

            print(
                f"""
                <tr>
                    <td><a href="#plot{link_target}"> {token} </a></td>
                    <td> {comparison.mean[0]:,} </td>
                    <td> {comparison.mean[1]:,} </td>
                    <td> {comparison.median[0]:,} </td>
                    <td> {comparison.median[1]:,} </td>
                    <td> {relative_diff:.2%} </td>
                    <td> {comparison.moods_pval:0.4e} </td>
                    <td> {A.path.parent.stem} </td>
                    <td> {B.path.parent.stem} </td>
                    <td> {A.kernelGenerate:,.0f} </td>
                    <td> {B.kernelGenerate:,.0f} </td>
                    <td> {A.sgprCount} </td>
                    <td> {B.sgprCount} </td>
                    <td> {A.vgprCount} </td>
                    <td> {B.vgprCount} </td>
                    <td> {A.agprCount} </td>
                    <td> {B.agprCount} </td>
                    <td> {A.ldsBytes:,} </td>
                    <td> {B.ldsBytes:,} </td>
                </tr>""",
                file=html_file,
            )

    print("</table>", file=html_file)


def email_html_summary(html_file, perf_runs):
    """Create HTML email report of summary statistics."""

    summary = summary_statistics(perf_runs)

    print("<h2>Results</h2>", file=html_file)

    html_overview_table(html_file, summary, [])

    perf_runs.sort()

    machines = dict()
    for run in perf_runs:
        if run.machine_spec not in machines:
            machines[run.machine_spec] = list()
        machines[run.machine_spec].append(run.name())

    print("<h2>Machines</h2>", file=html_file)
    for machine in machines:
        print(
            "<h3>Machine for {}:</h3>".format(", ".join(machines[machine])),
            file=html_file,
        )
        print(
            "<blockquote>{}</blockquote>".format(
                machine.pretty_string().replace("\n", "<br>")
            ),
            file=html_file,
        )


def get_common_args(tokens):
    splitTokens = [re.split(r",|\(|\)| ", x) for x in tokens]
    return {
        arg for arg in splitTokens[0] if all([arg in token for token in splitTokens])
    }


def html_summary(  # noqa: C901
    html_file,
    perf_runs,
    normalize=False,
    y_zero=False,
    plot_box=True,
    plot_median=False,
    plot_min=False,
    x_value="timestamp",
    group_results=False,
):
    """Create HTML report of summary statistics."""
    import plotly.express as px
    from plotly import graph_objs as go

    if plot_box and group_results:
        raise Exception(
            "Result grouping and box plots cannot be used at the same time."
        )

    perf_runs.sort()
    comparison_runs = perf_runs[-2:]
    summary = summary_statistics(comparison_runs)
    counts = comparison_counts(comparison_runs, summary)

    plots = []

    # Get problem and test tokens from the most recent run and sort them for consistent results.
    problems = sorted(
        {
            val.problem_token(priority_problems())
            for val in perf_runs[-1].results.values()
        }
    )
    tests = sorted(perf_runs[-1].results.keys())

    # Get all unique machine specs and sort them for consistent results.
    configs = PerformanceRun.get_all_specs(perf_runs)

    # Don't group when using box plots.
    for problem in problems if group_results else tests:
        runs = defaultdict(lambda: PlotData())
        for token in tests if group_results else [problem]:
            normalizer = None if normalize else 1
            for run in perf_runs:
                if token not in run.results:
                    continue
                if (
                    group_results
                    and run.results[token].problem_token(priority_problems()) != problem
                ):
                    continue
                name = (
                    run.name()
                    + "<br>Machine ID: "
                    + str(configs.index(run.machine_spec))
                    + "<br>"
                    + run.results[token].solution_token
                    + "<br>"
                )

                A = run.results[token]
                ka = np.asarray(A.kernelExecute).ravel()
                median = statistics.median(ka) if ka.size > 0 else 0

                if normalizer is None:
                    normalizer = median
                if normalizer > 0:
                    ka = ka / normalizer
                    median = median / normalizer
                min = np.min(ka) if ka.size > 0 else 0
                runs[token].timestamp.append(run.timestamp)
                runs[token].commit.append(run.commit)
                runs[token].median.append(median)
                runs[token].min.append(min)
                runs[token].name.append(name)
                runs[token].kernel.append(ka)
                runs[token].machine.append(configs.index(run.machine_spec))

        plot = go.Figure()
        common_args = get_common_args(runs.keys())
        if plot_box:
            dfs = pd.concat([runs[token].box_data for token in runs])
            box = px.box(dfs, x=x_value, y="runs").select_traces()
            for trace in box:
                plot.add_trace(trace)
        for token in runs:
            legend = sorted(set(re.split(r",|\(|\)| ", token)) - common_args)
            legend = "<br>".join(legend) + "<br>-----------------------"
            if plot_median:
                scatter = go.Scatter(
                    x=getattr(runs[token], x_value),
                    y=runs[token].median,
                    name="Median for<br>" + legend,
                    text=runs[token].name,
                    marker_color=runs[token].machine,
                    mode="lines+markers",
                )
                plot.add_trace(scatter)
            if plot_min:
                scatter = go.Scatter(
                    x=getattr(runs[token], x_value),
                    y=runs[token].min,
                    name="Min for<br>" + legend,
                    text=runs[token].name,
                    marker_color=runs[token].machine,
                    mode="lines+markers",
                )
                plot.add_trace(scatter)

        if y_zero:
            plot.update_yaxes(rangemode="tozero")

        plot.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date" if x_value == "timestamp" else "category",
            ),
            yaxis=dict(
                fixedrange=False,
            ),
        )
        plot.update_layout(
            height=1000,
            title_text=str(problem),
        )
        if not normalize:
            plot.update_yaxes(title={"text": "Runtime (ns)"})
        else:
            plot.update_yaxes(title={"text": "Normalized Runtime"})
        plots.append(plot)

    # Make a table of machines for lookup.
    machine_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Machine {}".format(i) for i in range(len(configs))],
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="left",
                ),
                cells=dict(
                    values=[
                        config.pretty_string().replace("\n", "<br>")
                        for config in configs
                    ],
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="left",
                ),
            )
        ]
    )

    print(
        """
<html>
  <head>
    <title>{}</title>
  </head>
  <body>
""".format("Performance"),
        file=html_file,
    )

    print("<h1>rocRoller performance</h1>", file=html_file)
    print("<h2>Comparison Summary</h2>", file=html_file)
    print("<ul>", file=html_file)
    print(f"<li>Compared: {counts.compared}</li>", file=html_file)
    print(f"<li>Significant diffs: {counts.significant}</li>", file=html_file)
    print(f"<li>Insignificant diffs: {counts.insignificant}</li>", file=html_file)
    print(
        f"<li>Not compared (reference-only): {counts.reference_only}</li>",
        file=html_file,
    )
    print(
        f"<li>Not compared (candidate-only): {counts.candidate_only}</li>",
        file=html_file,
    )
    print(
        f"<li>Not compared (total): {counts.not_compared_total}</li>",
        file=html_file,
    )
    print("</ul>", file=html_file)

    if len(perf_runs) == 2:
        print("<h2>Overview</h2>", file=html_file)
        html_overview_table(html_file, summary, problems if group_results else [])

    print("<h2>Results</h2>", file=html_file)
    if group_results:
        print("<ul>", file=html_file)
        print(
            "\n".join(
                [
                    f'<li><a href="#plot{link_target}"> {token} </a></li>'
                    for link_target, token in enumerate(problems)
                ]
            ),
            file=html_file,
        )
        print("</ul>", file=html_file)

    print('<table width="100%">', file=html_file)
    for i in range(len(plots)):
        print("<tr><td>", file=html_file)
        print(
            plots[i].to_html(
                full_html=False, include_plotlyjs=i == 0, div_id=f"plot{i}"
            ),
            file=html_file,
        )
        print("</td></tr>", file=html_file)
    print(
        """
    </table>
    """,
        file=html_file,
    )

    print("<h2>Machines</h2>", file=html_file)
    print("<tr><td>", file=html_file)
    print(
        machine_table.to_html(full_html=False, include_plotlyjs=False), file=html_file
    )
    print("</td></tr>", file=html_file)

    print(
        """
    </body>
    </html>
    """,
        file=html_file,
    )


def console_summary(f, perf_runs):
    summary = summary_statistics(perf_runs)
    result_diff = significant_changes(summary)
    resource_diff = resource_usage_changes(summary)

    if len(result_diff) > 0:
        print("Significant Diffs (p-val < 0.05)", file=f)
        print(result_diff, file=f)
    else:
        print("No statistically significant performance diffs", file=f)

    if len(resource_diff) > 0:
        print("\nResource Usage Changes", file=f)
        print(resource_diff, file=f)
    else:
        print("\nNo resource usage changes", file=f)


def get_args(parser: argparse.ArgumentParser):
    common_args = [
        args.directories,
        args.x_value,
        args.normalize,
        args.y_zero,
        args.plot_median,
        args.plot_min,
        args.exclude_boxplot,
        args.group_results,
    ]
    for arg in common_args:
        arg(parser)

    parser.add_argument(
        "--format",
        choices=["md", "html", "email_html", "console", "gemmdf", "resource_md"],
        default="md",
        help="Output format.",
    )


def run(args):
    """Compare previous performance runs."""
    compare(**args.__dict__)


def compare(
    directories=None,
    format="md",
    output=None,
    normalize=False,
    y_zero=False,
    plot_median=False,
    plot_min=False,
    exclude_boxplot=False,
    x_value="timestamp",
    group_results=False,
    **kwargs,
):
    """Compare multiple run directories.

    Implements the CLI 'compare' subcommand.
    """

    perf_runs = PerformanceRun.load_perf_runs(directories)

    print_final = False

    if output is None:
        print_final = True
        output = io.StringIO()

    if format == "html":
        html_summary(
            output,
            perf_runs,
            normalize=normalize,
            y_zero=y_zero,
            plot_median=plot_median,
            plot_min=plot_min,
            plot_box=not exclude_boxplot,
            x_value=x_value,
            group_results=group_results,
        )
    elif format == "email_html":
        email_html_summary(output, perf_runs)
    elif format == "md":
        markdown_summary(output, perf_runs)
    elif format == "resource_md":
        markdown_resource_summary(output, perf_runs)
    elif format == "console":
        console_summary(output, perf_runs)
    elif format == "gemmdf":
        summary = summary_statistics(perf_runs)
        df = summary_as_df(summary, GEMMResult)
        column_mapping = {"m": "macM", "n": "macN", "k": "macK"}
        df = df.rename(columns=column_mapping)
        cols = [
            "PREC",
            "AB",
            "M",
            "N",
            "K",
            "macM",
            "macN",
            "macK",
            "SCH",
            "LDS",
            "WG",
            "reldiff",
            "pval",
            "medianA(ns)",
            "medianB(ns)",
            "genA(ns)",
            "genB(ns)",
            "sgprA",
            "sgprB",
            "vgprA",
            "vgprB",
            "agprA",
            "agprB",
            "ldsBytesA",
            "ldsBytesB",
        ]
        scols = [
            "PREC",
            "AB",
            "M",
            "N",
            "K",
            "medianB(ns)",
            "macM",
            "macN",
            "macK",
            "SCH",
            "LDS",
            "WG",
            "reldiff",
            "pval",
            "medianA(ns)",
            "genA(ns)",
            "genB(ns)",
            "sgprA",
            "sgprB",
            "vgprA",
            "vgprB",
            "agprA",
            "agprB",
            "ldsBytesA",
            "ldsBytesB",
        ]

        print(df[cols].sort_values(scols, axis="index"), file=output)

    else:
        raise RuntimeError("Invalid format: " + format)

    if print_final:
        print(output.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_args(parser)

    parsed_args = parser.parse_args()
    run(parsed_args)
