################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import re
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional

from Tensile.Common import coVersionMap, printExit
from Tensile.Common.Architectures import architectureMap
from Tensile.Toolchain.Validators import ToolchainDefaults


def parseArguments(input: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse command line arguments for TensileCreateLibrary.

    Args:
        input: List of strings representing command line arguments used when
               calling parseArguments prgrammatically e.g. in testing.

    Returns:
        A dictionary containing the keys representing options and their values.
    """

    argParser = ArgumentParser(
        description="TensileCreateLibrary generates libraries and code object files "
        "for a set of supplied logic files.",
    )

    argParser.add_argument("LogicPath", help="Path to LibraryLogic.yaml files.")
    argParser.add_argument("OutputPath", help="Where to write library files?")
    argParser.add_argument(
        "RuntimeLanguage", help="Which runtime language?", choices=["OCL", "HIP", "HSA"]
    )
    argParser.add_argument(
        "--cxx-compiler",
        dest="CxxCompiler",
        action="store",
        default=ToolchainDefaults.CXX_COMPILER,
        help=f"Default: {ToolchainDefaults.CXX_COMPILER}",
    )
    argParser.add_argument(
        "--c-compiler", dest="CCompiler", action="store", default=ToolchainDefaults.C_COMPILER
    )
    argParser.add_argument("--cmake-cxx-compiler", dest="CmakeCxxCompiler", action="store")
    argParser.add_argument(
        "--offload-bundler",
        dest="OffloadBundler",
        action="store",
        default=ToolchainDefaults.OFFLOAD_BUNDLER,
    )
    argParser.add_argument(
        "--assembler", dest="Assembler", action="store", default=ToolchainDefaults.ASSEMBLER
    )
    argParser.add_argument(
        "--code-object-version",
        dest="CodeObjectVersion",
        choices=["4", "5", "V4", "V5", "default"],
        default="4",
        action="store",
    )
    argParser.add_argument(
        "--architecture",
        dest="Architecture",
        type=str,
        action="store",
        default="all",
        help="Supported archs: " + " ".join(architectureMap.keys()),
    )
    argParser.add_argument(
        "--no-compress",
        dest="NoCompress",
        action="store_true",
        help="Don't compress assembly code objects.",
    )
    argParser.add_argument(
        "--experimental",
        dest="Experimental",
        action="store_true",
        help="Include logic files in directories named 'Experimental'.",
    )
    argParser.add_argument(
        "--no-enumerate", action="store_true", help="Do not run rocm_agent_enumerator."
    )
    argParser.add_argument("--version", help="Version string to embed into library file.")
    argParser.add_argument(
        "--logic-format",
        dest="LogicFormat",
        choices=["yaml", "json"],
        action="store",
        default="yaml",
        help="select which logic format to use",
    )
    argParser.add_argument(
        "--library-format",
        dest="LibraryFormat",
        choices=["yaml", "msgpack"],
        action="store",
        default="msgpack",
        help="select which library format to use",
    )
    argParser.add_argument(
        "--jobs",
        "-j",
        dest="CpuThreads",
        type=int,
        default=-1,
        help="Number of parallel jobs to launch.",
    )
    argParser.add_argument(
        "--verbose",
        "-v",
        dest="PrintLevel",
        type=int,
        default=1,
        help="Set printout verbosity level.",
    )
    argParser.add_argument(
        "--no-lazy-library-loading",
        dest="LazyLibraryLoading",
        action="store_false",
        default=True,
        help="Disable building for lazy library loading.",
    )
    argParser.add_argument(
        "--enable-marker",
        dest="EnableMarker",
        action="store_true",
        default=False,
        help="Enable marker in Tensile.",
    )
    argParser.add_argument(
        "--no-generate-solution-table",
        dest="GenSolTable",
        action="store_false",
        default=True,
        help="Skip generating solution-yaml matching table",
    )
    argParser.add_argument(
        "--asm-debug",
        dest="AsmDebug",
        action="store_true",
        default=False,
        help="Keep debug information for built code objects",
    )
    argParser.add_argument("--build-id", dest="BuildIdKind", action="store", default="sha1")
    argParser.add_argument(
        "--address-sanitizer",
        dest="AsanBuild",
        action="store_true",
        default=False,
        help="Enable ASAN build.",
    )
    argParser.add_argument(
        "--keep-build-tmp",
        dest="KeepBuildTmp",
        action="store_true",
        default=False,
        help="Do not remove the temporary build directory (may required hundreds of GBs of space)",
    ),
    argParser.add_argument(
        "--logic-filter",
        dest="LogicFilter",
        action="store",
        default="*",
        type=str,
        help="Cutomsized logic filter, default is *, i.e. all logics."
        " Example: gfx942/Equality/* for building equality of gfx942 only",
    )
    argParser.add_argument(
        "--disable-asm-comments",
        dest="DisableAsmComments",
        action="store_true",
        default=False,
        help="Disable assembly comments in generated assembly code"
    )

    args = argParser.parse_args()

    # NOTE: The feature that allows predicate based tensilelite library builds
    # via `--architecture=gfx950[id=XXXX]` is not fully supported until the
    # tensilelite runtime lookup and selection for these predicates is complete.
    # As a staged approach, the TensileCreateLibrary code to enable this feature
    # is implemented, but will remain disabled until the work is complete.
    predicatePattern = re.compile(r"\[(.*?)\]")
    if predicatePattern.search(args.Architecture):
        printExit(
            "Predicate based tensilelite library builds are supported; however, since "
            "the runtime lookup and selection of these libraries is incomplete, the "
            "requested build is currently disabled. Please re-run using only the "
            "architecture name without a predicate (`--architecture=gfx950`)."
        )

    arguments = {}
    arguments["RuntimeLanguage"] = args.RuntimeLanguage
    arguments["CodeObjectVersion"] = coVersionMap[args.CodeObjectVersion]
    arguments["Architecture"] = args.Architecture
    arguments["LazyLibraryLoading"] = args.LazyLibraryLoading
    arguments["EnableMarker"] = args.EnableMarker
    arguments["DisableAsmComments"] = args.DisableAsmComments
    if args.CmakeCxxCompiler:
        os.environ["CMAKE_CXX_COMPILER"] = args.CmakeCxxCompiler
    arguments["LogicFormat"] = args.LogicFormat
    arguments["LibraryFormat"] = args.LibraryFormat
    arguments["CpuThreads"] = args.CpuThreads
    arguments["PrintLevel"] = args.PrintLevel
    arguments["AsmDebug"] = args.AsmDebug
    arguments["BuildIdKind"] = args.BuildIdKind
    arguments["KeepBuildTmp"] = args.KeepBuildTmp
    arguments["AsanBuild"] = args.AsanBuild
    arguments["UseCompression"] = not args.NoCompress
    arguments["CxxCompiler"] = args.CxxCompiler
    arguments["CCompiler"] = args.CCompiler
    arguments["OffloadBundler"] = args.OffloadBundler
    arguments["Assembler"] = args.Assembler
    arguments["LogicPath"] = args.LogicPath
    arguments["LogicFilter"] = args.LogicFilter
    arguments["OutputPath"] = args.OutputPath
    arguments["Experimental"] = args.Experimental
    arguments["GenSolTable"] = args.GenSolTable

    return arguments
