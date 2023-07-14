#!/usr/bin/env python3

"""
No print by default. `-v` or `--verbose` flag to print. `-vv` flag to print line numbers.
"""

import argparse
import sys
import subprocess
import shutil
import os
import re


def script(args: argparse.Namespace, print: callable, shell: callable) -> int:
    # START <=====
    def copy_files(source_dir, target_dir, regex_pattern):
        regex = re.compile(regex_pattern)
        paths = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if regex.match(file):
                    source_path = os.path.join(root, file)
                    target_path = os.path.join(target_dir, file)
                    shutil.copy2(source_path, target_path)
                    paths.append(target_path)
                    print(f"Copied file: {source_path} -> {target_path}", 1)
        return paths

    copy_files("../../build/", "./kernels/", r"^GEMM.*\.s$")
    print(
        "\n\nKernels availible: \n"
        + shell(
            "ls kernels/* | grep -E '\.s$|\.yaml$' | tee ./kernels/kernels.txt"
        ).stdout,
        1,
    )
    # END ======


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity level"
    )

    parser_args = parser.parse_args()

    default_print = print

    def print(*args, verbosity: int = 1, **kwargs):
        if parser_args.verbose >= verbosity:
            if parser_args.verbose >= 2:
                # Prepend line number
                default_print(
                    f"[{sys.argv[0]}:{sys._getframe().f_back.f_lineno}]",
                    *args,
                    **kwargs,
                )
            else:
                default_print(*args, **kwargs)

    def shell(
        command: str,
        *args,
        shell: bool = True,  # Run the command in a shell
        check: bool = True,  # Ensure that the command executed successfully
        text: bool = True,  # Return stdout and stderr as strings instead of bytes
        capture_output: bool = True,  # Capture stdout and stderr
        **kwargs,
    ) -> subprocess.CompletedProcess:
        if parser_args.verbose >= 2:
            default_print(
                f"[{sys.argv[0]}:{sys._getframe().f_back.f_lineno}] Running shell cmd:",
                command,
            )
        return subprocess.run(
            command,
            *args,
            shell=shell,
            check=check,
            text=text,
            capture_output=capture_output,
            **kwargs,
        )

    print(f"Running script: {sys.argv[0]} with args: {vars(parser_args)}", verbosity=2)
    sys.exit(script(parser_args, print, shell))
