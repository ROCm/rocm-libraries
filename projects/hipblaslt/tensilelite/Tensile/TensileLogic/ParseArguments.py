################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from argparse import ArgumentParser
from pathlib import Path

from Tensile.Toolchain.Validators import ToolchainDefaults

BUNDLED_KNOWN_BUGS = object()


def parseArguments():
    """
    Returns:
        A dictionary containing the keys representing options and their values.
    """

    argParser = ArgumentParser(
        description="TensileLogic runs critical checks to ensure the "
        "integrity of the supplied logic files.",
    )
    argParser.add_argument("LogicPath", help="path to library logic (yaml) files")

    argParser.add_argument(
        "-v",
        "--verbose",
        dest="Verbose",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="set print level with `-v 2`",
    )
    argParser.add_argument(
        "--jobs",
        "-j",
        dest="Jobs",
        action="store",
        default=48,
        help="number of worker processes to use during validation checks",
    )
    argParser.add_argument(
        "--cxx-compiler",
        dest="CxxCompiler",
        action="store",
        default=ToolchainDefaults.CXX_COMPILER,
        help=f"default: {ToolchainDefaults.CXX_COMPILER}",
    )
    argParser.add_argument(
        "--architecture",
        dest="Architecture",
        action="store",
        default="all",
        help="semicolon-separated list of gfx architectures to validate "
        "(e.g. 'gfx1151;gfx942'); 'all' validates every logic file",
    )

    group = argParser.add_mutually_exclusive_group()
    group.add_argument(
        "--check-all", dest="CheckAll", action="store_true", help="run all logic file checks"
    )
    group.add_argument(
        "--check-only-custom-kernels",
        dest="CheckOnlyCustomKernels",
        action="store_true",
        help="run logic file checks only on custom kernels",
    )

    knownBugsGroup = argParser.add_mutually_exclusive_group()
    knownBugsGroup.add_argument(
        "--known-bugs",
        dest="KnownBugs",
        type=Path,
        default=None,
        metavar="FILE",
        help="YAML file containing (path, solution_name) pairs to skip validation for "
        "(documented exceptions; paths relative to LogicPath). solution_name is the "
        "solution's SolutionNameMin, which is stable across library re-tuning",
    )
    knownBugsGroup.add_argument(
        "--use-bundled-known-bugs",
        dest="KnownBugs",
        action="store_const",
        const=BUNDLED_KNOWN_BUGS,
        help="use the known-bugs YAML bundled with TensileLogic",
    )
    argParser.add_argument(
        "--strict-known-bugs",
        dest="StrictKnownBugs",
        action="store_true",
        help="exit non-zero when a known-bugs entry no longer fails validation "
        "(a landed fix); use in CI to force removal of stale entries",
    )
    args = argParser.parse_args()

    return args
