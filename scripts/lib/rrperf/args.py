import argparse


def directories(parser: argparse.ArgumentParser):
    parser.add_argument("directories", nargs="*", help="Output directories to compare.")


def rundir(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--rundir",
        help="Location to run tests and store performance results.",
        default=None,
    )


def suite(parser: argparse.ArgumentParser):
    parser.add_argument("--suite", help="Benchmark suite to run.")


def x_value(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--x_value",
        help="Choose which value to use for the x-axis.",
        default="timestamp",
        choices=["timestamp", "commit"],
    )


def normalize(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data before plotting in html.",
    )


def y_zero(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--y_zero",
        action="store_true",
        help="Start the y-axis at 0 when plotting in html.",
    )


def plot_median(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--plot_median",
        action="store_true",
        help="Include a plot of the median when plotting in html.",
    )


def plot_min(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--plot_min",
        action="store_true",
        help="Include a plot of the min when plotting in html.",
    )


def exclude_boxplot(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--exclude_boxplot",
        action="store_true",
        help="Exclude the box plots when plotting in html. (Must be used with --group_results)",
    )


def group_results(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--group_results",
        action="store_true",
        help="Group data with the same problem args on the same graph.\n"
        "(Not compatible with boxplots.)",
    )
