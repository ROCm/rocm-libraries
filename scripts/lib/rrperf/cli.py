"""rocRoller performance tracking suite command line interface."""

import argparse

import rrperf


def main():
    parser = argparse.ArgumentParser(
        description="rocRoller performance tracking suite."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument(
        "--submit",
        help="Submit results to SOMEWHERE.",
        action="store_true",
        default=False,
    )
    run_cmd.add_argument("--token", help="Benchmark token to run.")
    run_cmd.add_argument("--filter", help="Filter benchmarks...")
    run_cmd.add_argument(
        "--rocm_smi",
        default="rocm-smi",
        help="Location of rocm-smi.",
    )
    run_cmd.add_argument(
        "--pin_clocks",
        action="store_true",
        help="Pin clocks before launching benchmark clients.",
    )

    compare_cmd = subparsers.add_parser("compare")
    compare_cmd.add_argument(
        "directories", nargs="*", help="Output directories to compare."
    )
    compare_cmd.add_argument(
        "--format",
        choices=["md", "html", "email_html"],
        default="md",
        help="Output format.",
    )

    autoperf_cmd = subparsers.add_parser(
        "autoperf",
        help="Run performance tests against multiple commits",
        description="""Run multiple performance tests against multiple commits and/or
        the current workspace.
        HEAD is tested if commits are not provided. Tags (like HEAD) can be specified.
        The output is in {clonedir}/doc_{datetime}.""",
    )
    autoperf_cmd.add_argument(
        "--clonedir",
        type=str,
        help="Base directory for repo clone destinations.",
        default=".",
    )
    autoperf_cmd.add_argument(
        "--ancestral",
        action="store_true",
        help="Test every commit between first and last commits.  Off by default.",
        required=False,
    )
    autoperf_cmd.add_argument(
        "--current",
        action="store_true",
        help="Test the repository in its current state.  Off by default.",
        required=False,
    )
    autoperf_cmd.add_argument(
        "commits", type=str, nargs="*", help="Commits/tags/branches to test."
    )

    for cmd in [run_cmd, autoperf_cmd]:
        cmd.add_argument(
            "--rundir",
            help="Location to run tests and store performance results.",
            default=None,
        )
        cmd.add_argument("--suite", help="Benchmark suite to run.")

    for cmd in [compare_cmd, autoperf_cmd]:
        cmd.add_argument(
            "--normalize",
            action="store_true",
            help="Normalize data before plotting in html.",
        )
        cmd.add_argument(
            "--y_zero",
            action="store_true",
            help="Start the y-axis at 0 when plotting in html.",
        )
        cmd.add_argument(
            "--plot_median",
            action="store_true",
            help="Include a plot of the median when plotting in html.",
        )
        cmd.add_argument(
            "--plot_min",
            action="store_true",
            help="Include a plot of the min when plotting in html.",
        )
        cmd.add_argument(
            "--exclude_boxplot",
            action="store_true",
            help="Exclude the box plots when plotting in html.",
        )
        cmd.add_argument(
            "--x_value",
            help="Choose which value to use for the x-axis.",
            default="timestamp",
            choices=["timestamp", "commit"],
        )

    profile_cmd = subparsers.add_parser(
        "profile",
        help="Run Omniperf against a RocRoller kernel or a Tensile guidepost.",
        description="""
        Run Omniperf against a RocRoller kernel or a Tensile guidepost.
        Specify a YAML config file to invoke Tensile to build a kernel to be profiled.
        Alternatively, specify an rrperf suite to profile a RocRoller kernel.
        These kernels are profiled with Omniperf.
        """,
    )
    profile_cmd.add_argument("--config", help="Location of Tensile YAML config file.")
    profile_cmd.add_argument("--suite", help="Benchmark suite to run.")
    profile_cmd.add_argument(
        "--output_dir",
        help="Directory where the Omniperf results are written.",
        default=".",
    )
    profile_cmd.add_argument(
        "--tensile_repo",
        help="Directory where Tensile repository is located.",
        default="/home/tensile",
    )

    args = parser.parse_args()

    command = {
        "run": rrperf.run.run,
        "compare": rrperf.compare.compare,
        "autoperf": rrperf.autoperf.run,
        "profile": rrperf.profile.run,
    }[args.command]
    command(**args.__dict__)
