"""Result comparison routines."""

import datetime
import io
import os
import pathlib
import statistics
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import pandas as pd
import rrperf
import scipy.stats
from rrperf.specs import MachineSpecs


@dataclass
class ComparisonResult:
    mean: List[float]
    median: List[float]
    moods_pval: float

    results: List[Any] = field(repr=False)


@dataclass
class PlotData:
    timestamp: List[float] = field(default_factory=list)
    median: List[float] = field(default_factory=list)
    name: List[str] = field(default_factory=list)
    kernel: List[float] = field(default_factory=list)
    machine: List[int] = field(default_factory=list)
    box_data: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["timestamp", "runs"])
    )


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
        except Exception:
            try:
                return datetime.datetime.strptime(wrkdir.stem[0:10], "%Y-%m-%d")
            except Exception:
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
                        # TODO: Handle result files that have multiple results in
                        # a single yaml file.
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
    "Median Diff %",
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
            row_str = [
                f"{token}",
                f"{A.path.parent.stem}",
                f"{B.path.parent.stem}",
                f"{comparison.mean[0]}",
                f"{comparison.mean[1]}",
                f"{comparison.median[0]}",
                f"{comparison.median[1]}",
                f"{(((comparison.median[1] - comparison.median[0]) * 100.0)/comparison.median[0]):.2f}%",
                f"{comparison.moods_pval:0.4e}",
            ]
            print(
                " | ".join(row_str),
                file=md,
            )

    perf_runs.sort()

    machines = dict()
    for run in perf_runs:
        if run.machine_spec not in machines:
            machines[run.machine_spec] = list()
        machines[run.machine_spec].append(run.name())

    print("\n\n## Machines\n", file=md)
    for machine in machines:
        print("### Machine for {}:\n".format(", ".join(machines[machine])), file=md)
        print(machine.pretty_string(), file=md)
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
                f"""<tr>
                <td><a href="#plot{i}"> {token} </a></td>
                <td> {A.path.parent.stem} </td>
                <td> {B.path.parent.stem}</td>
                <td> {comparison.mean[0]} </td>
                <td> {comparison.mean[1]} </td>
                <td> {comparison.median[0]} </td>
                <td>{comparison.median[1]} </td>
                <td>{(((comparison.median[1] - comparison.median[0]) * 100.0)/comparison.median[0]):.2f}% </td>
                <td> {comparison.moods_pval:0.4e}</td>
                <tr>""",
                file=html_file,
            )

    print("</table>", file=html_file)


def email_html_summary(html_file, perf_runs):
    """Create HTML email report of summary statistics."""

    summary = summary_statistics(perf_runs)

    print("<h2>Results</h2>", file=html_file)

    html_overview_table(html_file, summary)

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


def html_summary(html_file, perf_runs):
    """Create HTML report of summary statistics."""

    import plotly.express as px
    from plotly import graph_objs as go

    perf_runs.sort()
    summary = summary_statistics(perf_runs[-2:])

    plots = []

    # Get all unique test tokens and sort them for consistent results.
    tests = PerformanceRun.get_all_tokens(perf_runs)

    # Get all unique machine specs and sort them for consistent results.
    configs = PerformanceRun.get_all_specs(perf_runs)

    for token in tests:
        machine_filtered_runs = defaultdict(lambda: PlotData())

        for run in perf_runs:
            if token not in run.results:
                continue

            name = (
                run.name() + " <br> Machine ID: " + str(configs.index(run.machine_spec))
            )

            A = run.results[token]
            ka = np.asarray(A.kernelExecute) / A.numInner
            median = statistics.median(ka)

            for machine in ["all", run.machine_spec]:
                machine_filtered_runs[machine].timestamp.append(run.timestamp)
                machine_filtered_runs[machine].median.append(median)
                machine_filtered_runs[machine].name.append(name)
                machine_filtered_runs[machine].kernel.append(ka)
                machine_filtered_runs[machine].machine.append(
                    configs.index(run.machine_spec)
                )
                machine_filtered_runs[machine].box_data = pd.concat(
                    [
                        machine_filtered_runs[machine].box_data,
                        pd.DataFrame({"timestamp": run.timestamp, "runs": ka}),
                    ]
                )

        drop_down_options = [
            {
                "method": "update",
                "label": "All Machines",
                "args": [{"visible": [True, True] + ([False] * len(configs) * 2)}],
            }
        ]

        plot = go.Figure()
        box = px.box(
            machine_filtered_runs["all"].box_data, x="timestamp", y="runs"
        ).select_traces()
        scatter = go.Scatter(
            x=machine_filtered_runs["all"].timestamp,
            y=machine_filtered_runs["all"].median,
            name="Median",
            text=machine_filtered_runs["all"].name,
            marker_color=machine_filtered_runs["all"].machine,
            mode="lines+markers",
        )
        plot.add_trace(next(box))
        plot.add_trace(scatter)

        for i, config in enumerate(configs):
            box = px.box(
                machine_filtered_runs[config].box_data, x="timestamp", y="runs"
            ).select_traces()

            scatter = go.Scatter(
                x=machine_filtered_runs[config].timestamp,
                y=machine_filtered_runs[config].median,
                visible=False,
                name="Median",
                text=machine_filtered_runs[config].name,
                mode="lines+markers",
            )

            plot.add_trace(next(box))
            plot.add_trace(scatter)
            filter = [False] * ((len(configs) + 1) * 2)
            filter[(i + 1) * 2] = True
            filter[(i + 1) * 2 + 1] = True
            drop_down_options.append(
                {
                    "method": "update",
                    "label": "Machine ID: {}".format(str(configs.index(config))),
                    "args": [{"visible": filter}],
                }
            )

        plot.update_yaxes(
            range=[
                min(machine_filtered_runs["all"].median) * 0.97,
                max(machine_filtered_runs["all"].median) * 1.03,
            ]
        )

        plot.update_layout(
            updatemenus=[
                {
                    "buttons": drop_down_options,
                    "direction": "down",
                    "showactive": True,
                }
            ],
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
            ),
            yaxis=dict(
                fixedrange=False,
            ),
        )
        plot.update_layout(
            height=1000,
            title_text=str(token),
        )
        plot.update_yaxes(title={"text": "Time (ns)"})
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


def compare(directories=None, format="md", output=None, **kwargs):
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
        )
    elif format == "email_html":
        email_html_summary(output, perf_runs)
    else:
        markdown_summary(output, perf_runs)

    if print_final:
        print(output.getvalue())
