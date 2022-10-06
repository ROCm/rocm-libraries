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


@dataclass
class ComparisonResult:
    mean: List[float]
    median: List[float]
    moods_pval: float

    results: List[Any] = field(repr=False)


def summary_statistics(results_by_directory):
    """Compare results in `results_by_directory` and compute summary statistics.

    The first run is the reference run.
    """

    # build lookup
    results = defaultdict(dict)
    for run in results_by_directory.keys():
        for result in results_by_directory[run]:
            results[run][result.token] = result

    # first directory is reference, remaining are runs
    ref, *runs = results_by_directory.keys()

    # compute intersection
    common = {x for x in results[ref].keys()}
    for run in runs:
        common = common.intersection({x for x in results[run].keys()})

    common = list(common)
    common.sort()

    # compute comparison statistics
    stats = defaultdict(dict)
    for result in common:
        A = results[ref][result]
        ka = np.asarray(A.kernelExecute)
        ka_median = statistics.median(ka)
        ka_mean = statistics.mean(ka)

        for run in runs:
            B = results[run][result]
            kb = np.asarray(B.kernelExecute)

            kb_median = statistics.median(kb)
            kb_mean = statistics.mean(kb)

            _, p, _, _ = scipy.stats.median_test(ka, kb)

            stats[run][result] = A.token, ComparisonResult(
                mean=[ka_mean, kb_mean],
                median=[ka_median, kb_median],
                moods_pval=p,
                results=[A, B],
            )

    return stats


def markdown_summary(md, summary, specs_by_directory):
    """Create Markdown report of summary statistics."""

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

    runs = list(specs_by_directory.keys())
    runs.sort()

    print("\n\n## Machines\n", file=md)
    for run in runs:
        print("### Machine for {}:\n".format(os.path.basename(run)), file=md)
        print(specs_by_directory[run].pretty_string(), file=md)
        print("\n")


def email_html_summary(html_file, summary, specs_by_directory):
    """Create Markdown report of summary statistics."""

    print("<h2>Results</h2>", file=html_file)

    print("<table><tr><td>", file=html_file)

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
    print(" </td><td> ".join(header), file=html_file)
    print("</td></tr>", file=html_file)

    for run in summary:
        for result in summary[run]:
            token, comparison = summary[run][result]
            A, B = comparison.results
            print(
                f"<tr><td> {token} </td><td> {A.path.parent.stem} </td><td> {B.path.parent.stem} </td><td> {comparison.mean[0]} </td><td> {comparison.mean[1]} </td><td> {comparison.median[0]} </td><td> {comparison.median[1]} </td><td> {comparison.moods_pval:0.4e}</td><tr>",
                file=html_file,
            )
    
    print("</table>", file=html_file)

    runs = list(specs_by_directory.keys())
    runs.sort()

    print("<h2>Machines</h2>", file=html_file)
    for run in runs:
        print("<h3>Machine for {}:</h3>".format(os.path.basename(run)), file=html_file)
        print("<blockquote>{}</blockquote>".format(specs_by_directory[run].pretty_string().replace("\n", "<br>")), file=html_file)


def html_summary(
    html_file, results_by_directory, specs_by_directory, timestamp_by_directory
):
    """Create HTML report of summary statistics."""

    from plotly import graph_objs as go
    from plotly.subplots import make_subplots

    plots = []

    # build lookup
    results = defaultdict(dict)
    for run in results_by_directory.keys():
        for result in results_by_directory[run]:
            results[run][result.token] = result

    # Order directories by timestamp so they are plotted in order.
    timestamps = list(timestamp_by_directory.values())
    timestamps.sort()
    directories = [dir for timestamp in timestamps for dir in timestamp_by_directory if timestamp_by_directory[dir] == timestamp]

    # Get all unique test tokens and sort them for consistent results.
    tests = list({x for y in results.keys() for x in results[y].keys()})
    tests.sort()

    # Get all unique machine specs and sort them for consistent results.
    configs = list(set(specs_by_directory.values()))
    configs.sort()

    for token in tests:
        plot = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.06,
            specs=[[{"type": "box"}], [{"type": "table"}]],
        )
        means = []
        xs = []
        runs = []
        names = []
        for run in directories:
            if token in results[run]:
                name = (
                    os.path.basename(str(run))
                    + " <br> Machine ID: "
                    + str(configs.index(specs_by_directory[run]))
                )
                A = results[run][token]
                ka = np.asarray(A.kernelExecute)
                runs.append(ka)
                plot.add_trace(go.Box(x0=name, y=ka, name=name), row=1, col=1)
                xs.append(name)
                means.append(statistics.mean(ka))
                names.append(name)

        plot.add_trace(go.Scatter(x=xs, y=means, name="Mean"))

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

    print('<table width="100%">', file=html_file)

    print("<tr><td>", file=html_file)
    print(machine_table.to_html(full_html=False, include_plotlyjs=True), file=html_file)
    print("</td></tr>", file=html_file)

    for i in range(len(plots)):
        print("<tr><td>", file=html_file)

        print(plots[i].to_html(full_html=False, include_plotlyjs=False), file=html_file)

        print("</td></tr>", file=html_file)
    print(
        """
    </table>
    </body>
    </html>
    """,
        file=html_file,
    )


def get_timestamp(wrkdir):
    try:
        return datetime.datetime.fromtimestamp(float((wrkdir / "timestamp.txt").read_text().strip()))
    except:
        try:
            return datetime.datetime.strptime(wrkdir.stem[0:10], "%Y-%m-%d")
        except:
            return datetime.datetime.fromtimestamp(0)


def compare(directories=None, format="md", **kwargs):
    """Compare multiple run directories.

    Implements the CLI 'compare' subcommand.
    """

    # mapping from directory to list of results
    results_by_directory = OrderedDict()
    # mapping from directory to machine info
    specs_by_directory = OrderedDict()
    # mapping from directory to timestamp
    timestamp_by_directory = OrderedDict()

    for directory in directories:
        wrkdir = pathlib.Path(directory)
        results = []
        for path in wrkdir.glob("*.yaml"):
            try:
                results.extend(rrperf.problems.load_results(path))
            except Exception as e:
                print('Error loading results in "{}": {}'.format(path, e))
        results_by_directory[directory] = results

        specs_by_directory[directory] = rrperf.specs.load_machine_specs(
            wrkdir / "machine-specs.txt"
        )

        timestamp_by_directory[directory] = get_timestamp(wrkdir)

    summary = summary_statistics(results_by_directory)

    output = io.StringIO()
    if format == "html":
        html_summary(
            output, results_by_directory, specs_by_directory, timestamp_by_directory
        )
    elif format == "email_html":
        email_html_summary(output, summary, specs_by_directory)
    else:
        markdown_summary(output, summary, specs_by_directory)
    print(output.getvalue())
