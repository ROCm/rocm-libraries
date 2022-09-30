"""Result comparison routines."""

import pathlib
import io
import traceback

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


def markdown_summary(md, summary):
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


def compare(directories=None, **kwargs):
    """Compare multiple run directories.

    Implements the CLI 'compare' subcommand.
    """

    results_by_directory = OrderedDict()  # mapping from directory to list of results

    for directory in directories:
        wrkdir = pathlib.Path(directory)
        results = []
        for path in wrkdir.glob("*.yaml"):
            try:
                results.extend(rrperf.problems.load_results(path))
            except Exception as e:
                print("Error loading results in \"{}\": {}".format(path, e))
        results_by_directory[directory] = results

    summary = summary_statistics(results_by_directory)

    md = io.StringIO()
    markdown_summary(md, summary)
    print(md.getvalue())
