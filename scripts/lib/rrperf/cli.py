"""
rocRoller performance tracking suite command line interface.

New, modular way to add subcommands to rrperf:

Make a top-level module inside rrperf. It must implement get_args() and run().

- get_args() should take an argparse.ArgumentParser and add all command-line arguments
  to it.
- run() should take the parsed arguments object and run the relevant command.

The docstring for the module will be used as the description for the subcommand (under
`rrperf <command> --help`).  The docstring for the `run` function will be used as the
 help text for the subcommand (under `rrperf --help`).

"""

import argparse
import inspect
import rrperf


def main():
    parser = argparse.ArgumentParser(
        description="rocRoller performance tracking suite."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run", help="Run a benchmark suite.")
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

    compare_cmd = subparsers.add_parser(
        "compare", help="Compare previous performance runs."
    )
    compare_cmd.add_argument(
        "directories", nargs="*", help="Output directories to compare."
    )
    compare_cmd.add_argument(
        "--format",
        choices=["md", "html", "email_html", "console", "gemmdf"],
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

    autoperf_cmd.add_argument(
        "--no-fail",
        action="append",
        default=[],
        help="Commits/tags/branches where a failure does not cause the"
        " overall command to fail.",
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
            help="Exclude the box plots when plotting in html. (Must be used with --group_results)",
        )
        cmd.add_argument(
            "--x_value",
            help="Choose which value to use for the x-axis.",
            default="timestamp",
            choices=["timestamp", "commit"],
        )
        cmd.add_argument(
            "--group_results",
            action="store_true",
            help="Group data with the same problem args on the same graph.\n"
            "(Not compatible with boxplots.)",
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

    for name, mod in rrperf.__dict__.items():
        if hasattr(mod, "get_args") and hasattr(mod, "run"):
            mod_parser = subparsers.add_parser(
                name,
                help=inspect.getdoc(mod.run),
                description=inspect.getdoc(mod),
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            mod.get_args(mod_parser)

    args = parser.parse_args()

    if (
        "group_results" in args.__dict__
        and args.group_results
        and not args.exclude_boxplot
    ):
        parser.error("--group_results cannot be used without --exclude_boxplot")

    manual_commands = {
        "run": rrperf.run.run,
        "compare": rrperf.compare.compare,
        "autoperf": rrperf.autoperf.run,
        "profile": rrperf.profile.run,
    }

    if args.command in manual_commands:
        manual_commands[args.command](**args.__dict__)
    else:
        getattr(rrperf, args.command).run(args)
