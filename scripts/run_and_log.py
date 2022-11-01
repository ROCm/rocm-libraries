#!/usr/bin/env python3

import argparse
import re
import pathlib
import subprocess
from datetime import datetime


def runAndLog(args):
    fname = pathlib.Path(args.output_file)
    output = subprocess.check_output(args.command, cwd=args.working_directory).decode()
    print(output)
    if args.regex:
        regex = "|".join(["(?:{})".format(r) for r in args.regex])
        output = ", ".join(re.findall(regex, output))
    if not args.exclude_commit:
        commit = subprocess.check_output("git rev-parse --short HEAD".split(), cwd=args.git_directory).decode()
        output = ", ".join(["Git Commit: {}".format(commit.strip()), output])
    output = ", ".join([str(datetime.now()), output])
    print(output)
    mode = 'w' if args.overwrite else 'a'
    with fname.open(mode) as log:
        log.write(output)
        log.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a program and log its output to a file along with current git commit.')
    parser.add_argument('--output_file', type=str, default='run.log', help='Target log file.')
    parser.add_argument('--overwrite', action="store_true", help='Overwrite the output file instead of appending.')
    parser.add_argument('--exclude_commit', action="store_true", help='Don\t include commit ID in log.')
    parser.add_argument('--git_directory', type=str, default='./', help='Directory to query for git commit.')
    parser.add_argument('--working_directory', type=str, default='./', help='Directory to run script in.')
    parser.add_argument('-r', '--regex', default=[], type=str, action='append', help='Regex for what to capture and log.')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command to run and log.')
    args = parser.parse_args()

    runAndLog(args)
