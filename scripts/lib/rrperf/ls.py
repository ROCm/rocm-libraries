"""List benchmark suites."""

import argparse

import rrperf.args as args
import rrperf.rrsuites


def get_args(parser: argparse.ArgumentParser):
    args.suite(parser)


def run(args):
    """List benchmarks."""

    suite = args.suite
    if suite is None:
        suite = "all"

    generator = getattr(rrperf.rrsuites, suite)
    for x in generator():
        print(x)
