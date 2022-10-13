"""Result comparison routines."""

import pathlib
import io
import os
import datetime

import numpy as np
import scipy.stats
import statistics

from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Any, List

import rrperf

from rrperf.specs import MachineSpecs


@dataclass
class ComparisonResult:
    mean: List[float]
    median: List[float]
    moods_pval: float

    results: List[Any] = field(repr=False)


class PerformanceRun:
    timestamp: float
    directory: str
    machine_spec: MachineSpecs
    results: OrderedDict

    def __init__(self, timestamp, directory, machine_spec, results):
        self.timestamp = timestamp
        self.directory = directory
        self.machine_spec = machine_spec
        self.results = results

    def __lt__(self, other):
        if self.timestamp == other.timestamp:
            return self.directory < other.directory
        return self.timestamp < other.timestamp

    def name(self):
        return os.path.basename(self.directory)

    def get_comparable_tokens(ref, runs):
        # compute intersection
        common = set(ref.results.keys())
        for run in runs:
            common = common.intersection(set(run.results.keys()))

        common = list(common)
        common.sort()

        return common

    def get_all_tokens(runs):
        tests = list({token for run in runs for token in run.results})
        tests.sort()
        return tests

    def get_all_specs(runs):
        configs = list({run.machine_spec for run in runs})
        configs.sort()
        return configs

    def get_timestamp(wrkdir):
        try:
            return datetime.datetime.fromtimestamp(
                float((wrkdir / "timestamp.txt").read_text().strip())
            )
        except:
            try:
                return datetime.datetime.strptime(wrkdir.stem[0:10], "%Y-%m-%d")
            except:
                return datetime.datetime.fromtimestamp(0)

    def load_perf_runs(directories):
        perf_runs = list()
        for directory in directories:
            wrkdir = pathlib.Path(directory)
            results = OrderedDict()
            for path in wrkdir.glob("*.yaml"):
                try:
                    result = rrperf.problems.load_results(path)
                except Exception as e:
                    print('Error loading results in "{}": {}'.format(path, e))
                for element in result:
                    if element.token in results:
                        #TODO: Handle result files that have multiple results in a single yaml file.
                        results[element.token] = element
                    else:
                        results[element.token] = element
            spec = rrperf.specs.load_machine_specs(wrkdir / "machine-specs.txt")
            timestamp = PerformanceRun.get_timestamp(wrkdir)
            perf_runs.append(PerformanceRun(timestamp, directory, spec, results))

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
        ka = np.asarray(A.kernelExecute)
        ka_median = statistics.median(ka)
        ka_mean = statistics.mean(ka)

        for run in runs:
            B = run.results[token]
            kb = np.asarray(B.kernelExecute)

            kb_median = statistics.median(kb)
            kb_mean = statistics.mean(kb)

            _, p, _, _ = scipy.stats.median_test(ka, kb)

            stats[run][token] = A.token, ComparisonResult(
                mean=[ka_mean, kb_mean],
                median=[ka_median, kb_median],
                moods_pval=p,
                results=[A, B],
            )

    return stats


header = [
        "Problem",
        "Run A (ref)",
        "Run B",
        "Mean A",
        "Mean B",
        "Median A",
        "Median B",
        "Moods p-val",
    ]

def markdown_summary(md, perf_runs):
    """Create Markdown report of summary statistics."""

    summary = summary_statistics(perf_runs)

    print(" | ".join(header), file=md)
    print(" | ".join(["---"] * len(header)), file=md)

    for run in summary:
        for result in summary[run]:
            token, comparison = summary[run][result]
            A, B = comparison.results
            print(
                f"{token} | {A.path.parent.stem} | {B.path.parent.stem} | {comparison.mean[0]} | {comparison.mean[1]} | {comparison.median[0]} | {comparison.median[1]} | {comparison.moods_pval:0.4e}",
                file=md,
            )

    perf_runs.sort()

    print("\n\n## Machines\n", file=md)
    for run in perf_runs:
        print("### Machine for {}:\n".format(run.name()), file=md)
        print(run.machine_spec.pretty_string(), file=md)
        print("\n")


def html_overview_table(html_file, summary):
    """Create HTML table with summary statistics."""

    print("<table><tr><td>", file=html_file)

    print("</td><td> ".join(header), file=html_file)
    print("</td></tr>", file=html_file)

    for run in summary:
        for i, result in enumerate(summary[run]):
            token, comparison = summary[run][result]
            A, B = comparison.results
            print(
                f'<tr><td><a href="#plot{i}"> {token} </a></td><td> {A.path.parent.stem} </td><td> {B.path.parent.stem} </td><td> {comparison.mean[0]} </td><td> {comparison.mean[1]} </td><td> {comparison.median[0]} </td><td> {comparison.median[1]} </td><td> {comparison.moods_pval:0.4e}</td><tr>',
                file=html_file,
            )

    print("</table>", file=html_file)


def email_html_summary(html_file, perf_runs):
    """Create HTML email report of summary statistics."""

    summary = summary_statistics(perf_runs)

    print("<h2>Results</h2>", file=html_file)

    html_overview_table(html_file, summary)

    perf_runs.sort()
    print("<h2>Machines</h2>", file=html_file)
    for run in perf_runs:
        print("<h3>Machine for {}:</h3>".format(run.name()), file=html_file)
        print(
            "<blockquote>{}</blockquote>".format(
                run.machine_spec.pretty_string().replace("\n", "<br>")
            ),
            file=html_file,
        )


def html_summary(html_file, perf_runs):
    """Create HTML report of summary statistics."""

    from plotly import graph_objs as go
    from plotly.subplots import make_subplots
    from scipy import stats

    perf_runs.sort()
    summary = summary_statistics(perf_runs[-2:])

    plots = []

    # Get all unique test tokens and sort them for consistent results.
    tests = PerformanceRun.get_all_tokens(perf_runs)

    # Get all unique machine specs and sort them for consistent results.
    configs = PerformanceRun.get_all_specs(perf_runs)

    for token in tests:
        plot = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.06,
            specs=[[{"type": "box"}], [{"type": "table"}]],
        )
        medians = []
        xs = []
        runs = []
        names = []
        for run in perf_runs:
            if token in run.results:
                name = (
                    run.name()
                    + " <br> Machine ID: "
                    + str(configs.index(run.machine_spec))
                )
                A = run.results[token]
                ka = np.asarray(A.kernelExecute) / A.numInner
                runs.append(ka)
                median = statistics.median(ka)
                no_outliers = ka
                count = 1
                while no_outliers.size != count:
                    count = no_outliers.size
                    no_outliers = no_outliers[(np.abs(stats.zscore(no_outliers)) < 2)]
                plot.add_trace(go.Box(x0=name, y=no_outliers, name=name, boxpoints=False), row=1, col=1)
                xs.append(name)
                medians.append(median)
                names.append(name)

        plot.add_trace(go.Scatter(x=xs, y=medians, name="Median"))

        table = go.Table(
            header=dict(
                values=names,
                line_color="darkslategray",
                fill_color="lightskyblue",
                align="left",
            ),
            cells=dict(
                values=runs,
                line_color="darkslategray",
                fill_color="lightcyan",
                align="left",
            ),
        )

        plot.add_trace(table, row=2, col=1)
        plot.update_layout(
            height=1000,
            showlegend=False,
            title_text=str(token),
        )
        plot.update_yaxes(title={"text": "Time (ns)"}, row=1, col=1)
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
""".format(
            "Performance"
        ),
        file=html_file,
    )

    print("<h1>rocRoller performance</h1>", file=html_file)
    print("<h2>Overview</h2>", file=html_file)
    html_overview_table(html_file, summary)

    print("<h2>Results</h2>", file=html_file)
    print('<table width="100%">', file=html_file)

    print("<tr><td>", file=html_file)
    print(machine_table.to_html(full_html=False, include_plotlyjs=True), file=html_file)
    print("</td></tr>", file=html_file)

    for i in range(len(plots)):
        print("<tr><td>", file=html_file)
        print(
            plots[i].to_html(
                full_html=False, include_plotlyjs=False, div_id=f"plot{i}"
            ),
            file=html_file,
        )
        print("</td></tr>", file=html_file)
    print(
        """
    </table>
    </body>
    </html>
    """,
        file=html_file,
    )


def compare(directories=None, format="md", **kwargs):
    """Compare multiple run directories.

    Implements the CLI 'compare' subcommand.
    """

    perf_runs = PerformanceRun.load_perf_runs(directories)

    output = io.StringIO()
    if format == "html":
        html_summary(
            output,
            perf_runs,
        )

    elif format == "email_html":
        email_html_summary(output, perf_runs)
    else:
        markdown_summary(output, perf_runs)
    print(output.getvalue())
