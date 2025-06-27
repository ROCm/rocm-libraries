################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
import warnings
from argparse import Action, ArgumentParser
from typing import Any, Dict, List, Optional

from ..Common import DeveloperWarning, architectureMap
from ..Utilities.Toolchain import ToolchainDefaults, validateToolchain


class DeprecatedOption(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        warnings.warn(
            f"[DEPRECATED] The option {option_string} will be removed in future versions."
        )


def splitExtraParameters(par):
    """Allows the --global-parameters option to specify any parameters from the command line."""

    (key, value) = par.split("=")
    value = eval(value)
    return (key, value)


def parseArguments(input: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse command line arguments for TensileCreateLibrary.

    Args:
        input: List of strings representing command line arguments used when
               calling parseArguments prgrammatically e.g. in testing.

    Returns:
        A dictionary containing the keys representing options and their values.
    """

    parser = ArgumentParser(
        description="TensileCreateLibrary generates libraries and code object files "
        "for a set of supplied logic files.",
    )

    # Positional arguments
    parser.add_argument("LogicPath", help="Path to Library Logic (YAML) files.")
    parser.add_argument("OutputPath", help="Build path for library files.")
    parser.add_argument(
        "RuntimeLanguage",
        help="Runtime language for generated library.",
        choices=["HIP", "HSA"],
    )

    # Optional arguments
    parser.add_argument(
        "--cxx-compiler",
        dest="CxxCompiler",
        default=ToolchainDefaults.CXX_COMPILER,
        type=str,
        help="C++ compiler used when generating binaries."
        "On Linux, amdclang++ (default) or hipcc. On Windows clang++ (default) or hipcc. "
        "On Windows, include the file extension, or extensions will be searched according to the PATHEXT environment variable. "
        "Pass a fully-qualified path to override environment inspection when searching for the compiler.",
    )
    parser.add_argument(
        "--c-compiler",
        dest="CCompiler",
        default=ToolchainDefaults.C_COMPILER,
        type=str,
    )
    parser.add_argument(
        "--assembler",
        dest="Assembler",
        default=ToolchainDefaults.ASSEMBLER,
        type=str,
    )
    parser.add_argument(
        "--offload-bundler",
        dest="OffloadBundler",
        default=ToolchainDefaults.OFFLOAD_BUNDLER,
        type=str,
    )
    parser.add_argument(
        "--architecture",
        dest="Architecture",
        default="all",
        type=str,
        help="Architectures to generate a library for. When specifying multiple options, "
        "use quoted, semicolon delimited architectures, e.g., --architecture='gfx908;gfx1012'. "
        "Supported archiectures include: " + ", ".join(architectureMap.keys()),
    )
    parser.add_argument(
        "--code-object-version",
        dest="CodeObjectVersion",
        choices=["default", "V4", "V5"],
        type=str,
        help="HSA code-object version.",
    )
    parser.add_argument(
        "--cmake-cxx-compiler",
        dest="CmakeCxxCompiler",
        action=DeprecatedOption,
        help="(Deprecated) Set the environment variable CMAKE_CXX_COMPILER.",
    )

    # Boolean flags
    parser.add_argument(
        "--merge-files",
        dest="MergeFiles",
        action="store_true",
        help="Store all solutions in single file.",
    )
    parser.add_argument(
        "--no-merge-files",
        dest="MergeFiles",
        action="store_false",
        help="Store every solution and kernel in separate file.",
    )
    parser.add_argument(
        "--num-merged-files",
        dest="NumMergedFiles",
        default=1,
        type=int,
        help="Number of files the kernels should be written into.",
    )
    parser.add_argument(
        "--short-file-names",
        dest="ShortNames",
        action="store_true",
        help="Converts solution and kernel names to serial IDs.",
    )
    parser.add_argument(
        "--no-short-file-names",
        dest="ShortNames",
        action="store_false",
        help="Disables short files names.",
    )
    parser.add_argument(
        "--no-enumerate",
        dest="NoEnumerate",
        action="store_true",
        help="Do not run rocm_agent_enumerator.",
    )
    parser.add_argument(
        "--embed-library",
        dest="EmbedLibrary",
        type=str,
        help="Embed (new) library files into static variables. Specify the name of the library",
    )
    parser.add_argument(
        "--embed-library-key",
        dest="EmbedLibraryKey",
        default=None,
        help="Access key for embedding library files.",
    )
    parser.add_argument(
        "--version",
        dest="Version",
        type=str,
        help="Version string to embed into library file.",
    )
    parser.add_argument(
        "--generate-manifest-and-exit",
        dest="GenerateManifestAndExit",
        action="store_true",
        default=False,
        help="Output manifest file with list of expected library objects and exit.",
    )
    parser.add_argument(
        "--generate-sources-and-exit",
        dest="GenerateSourcesAndExit",
        action="store_true",
        default=False,
        help="Output source files only and exit.",
    )
    parser.add_argument(
        "--verify-manifest",
        dest="VerifyManifest",
        action="store_true",
        default=False,
        help="Verify manifest file against generated library files and exit.",
    )
    parser.add_argument(
        "--keep-build-tmp",
        dest="KeepBuildTmp",
        action="store_true",
        default=False,
        help="Do not remove the temporary build directory (may required hundreds of GBs of space)",
    )
    parser.add_argument(
        "--library-format",
        dest="LibraryFormat",
        choices=["yaml", "msgpack"],
        action="store",
        default="msgpack",
        help="Select which library format to use.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        dest="CpuThreads",
        default=-1,
        type=int,
        help="Number of parallel jobs to launch. "
        "If jobs < 1 or jobs > nproc the number of parallel jobs will be set to the "
        "number of cores, up to a maximum of 64.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="PrintLevel",
        choices=[0, 1, 2, 3],
        default=1,
        type=int,
        help="Set printing verbosity level.",
    )
    parser.add_argument(
        "--separate-architectures",
        dest="SeparateArchitectures",
        action="store_true",
        default=False,
        help="Separates generated library files by architecture.",
    )
    parser.add_argument(
        "--lazy-library-loading",
        dest="LazyLibraryLoading",
        action="store_true",
        default=False,
        help="Loads libraries when needed instead of eagerly.",
    )
    parser.add_argument(
        "--client-config",
        dest="ClientConfig",
        action="store_true",
        help="Creates best-solution.ini in the output directory for the library and "
        "code object files.",
    )
    parser.add_argument(
        "--ignore-asm-cap-cache",
        dest="IgnoreAsmCapCache",
        action="store_true",
        default=False,
        help="Ignore ASM capabilities cache and derive the capabilities at runtime.",
    )
    parser.add_argument(
        "--write-master-solution-index",
        dest="WriteMasterSolutionIndex",
        action="store_true",
        default=False,
        help="Output master solution index in csv format including number of kernels per architecture.",
    )
    parser.add_argument(
        "--global-parameters",
        nargs="+",
        dest="GlobalParameters",
        type=splitExtraParameters,
        default=[],
        action=DeprecatedOption,
        help="(Deprecated) Additional global parameters.",
    )
    args = parser.parse_args(input)

    arguments = {
        "LogicPath": args.LogicPath,
        "OutputPath": args.OutputPath,
        "RuntimeLanguage": args.RuntimeLanguage,
        "CxxCompiler": args.CxxCompiler,
        "CCompiler": args.CCompiler,
        "Assembler": args.Assembler,
        "OffloadBundler": args.OffloadBundler,
        "CmakeCxxCompiler": args.CmakeCxxCompiler,
        "CodeObjectVersion": args.CodeObjectVersion,
        "Architecture": args.Architecture,
        "MergeFiles": args.MergeFiles,
        "NumMergedFiles": args.NumMergedFiles,
        "ShortNames": args.ShortNames,
        "CodeFromFiles": False,
        "EmbedLibrary": args.EmbedLibrary,
        "EmbedLibraryKey": args.EmbedLibraryKey,
        "Version": args.Version,
        "LibraryFormat": args.LibraryFormat,
        "GenerateManifestAndExit": args.GenerateManifestAndExit,
        "VerifyManifest": args.VerifyManifest,
        "GenerateSourcesAndExit": args.GenerateSourcesAndExit,
        "CpuThreads": args.CpuThreads,
        "PrintLevel": args.PrintLevel,
        "SeparateArchitectures": args.SeparateArchitectures,
        "LazyLibraryLoading": args.LazyLibraryLoading,
        "ClientConfig": args.ClientConfig,
        "IgnoreAsmCapCache": args.IgnoreAsmCapCache,
        "WriteMasterSolutionIndex": args.WriteMasterSolutionIndex,
        "KeepBuildTmp": args.KeepBuildTmp,
    }

    if args.CmakeCxxCompiler:
        os.environ["CMAKE_CXX_COMPILER"] = args.CmakeCxxCompiler
    if args.GenerateSourcesAndExit:
        # Generated sources are preserved and go into output directory
        arguments["WorkingPath"] = arguments["OutputPath"]
    if args.PrintLevel <= 0:
        warnings.filterwarnings("ignore", category=DeveloperWarning)

    for k, v in args.GlobalParameters:
        arguments[k] = v
    (
        arguments["CxxCompiler"],
        arguments["CCompiler"],
        arguments["Assembler"],
        arguments["OffloadBundler"],
        arguments["HipConfig"],
    ) = validateToolchain(
        arguments["CxxCompiler"],
        arguments["CCompiler"],
        arguments["Assembler"],
        arguments["OffloadBundler"],
        ToolchainDefaults.HIP_CONFIG,
    )
    if args.NoEnumerate:
        arguments["ROCmAgentEnumeratorPath"] = False
    else:
        arguments["ROCmAgentEnumeratorPath"] = validateToolchain(
            ToolchainDefaults.DEVICE_ENUMERATOR
        )

    return arguments
