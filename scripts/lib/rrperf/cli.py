"""rocRoller performance tracking suite command line interface."""

import argparse
import rrperf


def main():
    parser = argparse.ArgumentParser(
        description="rocRoller performance tracking suite."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument("--suite", help="Benchmark suite to run.")
    run_cmd.add_argument(
        "--submit", help="Submit results to SOMEWHERE.", action="store_true", default=False
    )
    run_cmd.add_argument("--token", help="Benchmark token to run.")
    run_cmd.add_argument("--filter", help="Filter benchmarks...")

    args = parser.parse_args()
    command = {"run": rrperf.run.run}[args.command]
    command(**args.__dict__)
