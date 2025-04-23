#!/usr/bin/python3

# ########################################################################
# Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################
#
# Author: rocsparse-maintainer@amd.com
#

"""Copyright 2021 Advanced Micro Devices, Inc. All rights Reserved.
Manage build and installation"""

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib
from fnmatch import fnmatchcase

args = {}
param = {}
OS_info = {}

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""Checks build arguments""")

    #
    # Common options
    #

    # Arch
    parser.add_argument('-a', '--architecture', dest='gpu_architecture', required=False, default="all",
                        help='Set GPU architectures, e.g. all, auto, "gfx803;gfx906:xnack-", gfx1030, gfx1101 (optional, default: all)')

    # Debug
    parser.add_argument('-g', '--debug', required=False, default = False,  action='store_true',
                        help='Generate Debug build (optional, default: False, Release is default mode)')

    # Build directory
    parser.add_argument(      '--build_dir', type=str, required=False, default = "build",
                        help='Build directory path (optional, default: build)')

    # ?
    parser.add_argument(      '--skip_ld_conf_entry', required=False, default = False)

    # Static library
    parser.add_argument(      '--static', required=False, default = False, dest='static_lib', action='store_true',
                        help='Generate static library build (optional, default: False)')

    # Build client
    parser.add_argument('-c', '--clients', required=False, default = False, dest='build_clients', action='store_true',
                        help='Generate all client builds (optional, default: False)')

    # Install after build.
    parser.add_argument('-i', '--install', required=False, default = False, dest='install', action='store_true',
                        help='Install after build (optional, default: False)')

    # Add definition
    parser.add_argument(      '--cmake-darg', required=False, dest='cmake_dargs', action='append', default=[],
                        help='List of additional cmake defines for builds (optional, e.g. CMAKE)')

    # Verbose mode
    parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
                        help='Verbose build (optional, default: False)')


    # rocalution
    parser.add_argument(     '--clients-only', dest='clients_only', required=False, default = False, action='store_true',
                        help='Build only clients with a pre-built library')
    parser.add_argument(     '--rocprim_dir', dest='rocprim_dir', type=str, required=False, default = "",
                             help='Specify path to an existing rocPRIM install directory (optional, default: /opt/rocm/rocprim)')
    return parser.parse_args()

def os_detect():
    global OS_info
    if os.name == "nt":
        OS_info["ID"] = platform.system()
    else:
        inf_file = "/etc/os-release"
        if os.path.exists(inf_file):
            with open(inf_file) as f:
                for line in f:
                    if "=" in line:
                        k,v = line.strip().split("=")
                        OS_info[k] = v.replace('"','')
    OS_info["NUM_PROC"] = os.cpu_count()
    print(OS_info)

def create_dir(dir_path):
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        run_cmd( "rm" , f"-rf {dir_path}")

def cmake_path(os_path):
    if os.name == "nt":
        return os_path.replace("\\", "/")
    else:
        return os_path

def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    cmake_executable = "cmake"
    cmake_options = []
    src_path = cmake_path(cwd_path)
    cmake_platform_opts = []
    if os.name == "nt":
        generator = f"-G Ninja"
        cmake_options.append( generator )

        # CMAKE_PREFIX_PATH set to rocm_path and HIP_PATH set BY SDK Installer
        raw_rocm_path = cmake_path(os.getenv('HIP_PATH', "C:/hip"))
        rocm_path = f'"{raw_rocm_path}"' # guard against spaces in path
        # CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
        cmake_platform_opts.append( f"-DCPACK_PACKAGING_INSTALL_PREFIX=" )
        cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"C:/hipSDK\"" )
        toolchain = os.path.join( src_path, "toolchain-windows.cmake" )
    else:
        rocm_path = os.getenv( 'ROCM_PATH', "/opt/rocm")
        cmake_platform_opts.append( f"-DROCM_DIR:PATH={rocm_path} -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}" )
        cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"rocalution-install\"" )
        toolchain = "toolchain-linux.cmake"

    print( f"Build source path: {src_path}")

    tools = f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
    cmake_options.append( tools )

    cmake_options.extend( cmake_platform_opts )

    cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path}"
    cmake_options.append( cmake_base_options )

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF"
    cmake_options.append( cmake_pack_options )

    if os.getenv('CMAKE_CXX_COMPILER_LAUNCHER'):
        cmake_options.append( f"-DCMAKE_CXX_COMPILER_LAUNCHER={os.getenv('CMAKE_CXX_COMPILER_LAUNCHER')}" )

#   cmake_options.append("-DBUILD_TESTING=OFF")

    print( cmake_options )

    # build type
    cmake_config = ""
    build_dir = os.path.abspath(args.build_dir)
    if not args.debug:
        build_path = os.path.join(build_dir, "release")
        cmake_config="Release"
    else:
        build_path = os.path.join(build_dir, "debug")
        cmake_config="Debug"

    cmake_options.append( f"-DCMAKE_BUILD_TYPE={cmake_config}" )

    # clean
    delete_dir( build_path )

    create_dir( os.path.join(build_path, "clients") )
    os.chdir( build_path )

    if args.static_lib:
        cmake_options.append( f"-DBUILD_SHARED_LIBS=OFF" )

    if args.skip_ld_conf_entry:
        cmake_options.append( f"-DROCM_DISABLE_LDCONFIG=ON" )

    if args.build_clients:
        cmake_build_dir = cmake_path(build_dir)
        cmake_options.append( f"-DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_DIR={cmake_build_dir}" )


    if args.gpu_architecture == "auto":
        gpu_detect()
        if len(OS_info["GPU"]):
            args.gpu_architecture = OS_info["GPU"]
        else:
            fatal("Could not detect GPU as requested. Not continuing.")
    cmake_options.append(f'-DGPU_TARGETS=\"{args.gpu_architecture}\"')

 #   if args.clients_only:
 #       if args.library_dir_installed:
 #           library_dir = args.library_dir_installed
 #       else:
 #           library_dir = f"{rocm_path}/rocblas"
 #       cmake_lib_dir = cmake_path(library_dir)
 #       cmake_options.append( f"-DSKIP_LIBRARY=ON -DROCBLAS_LIBRARY_DIR={cmake_lib_dir}" )



# Reject
#    if args.cpu_ref_lib == 'blis':
#        cmake_options.append( f"-DLINK_BLIS=ON" )

#
# Reject for now
#
#    cmake_options.append( f"-DAMDGPU_TARGETS={args.gpu_architecture}" )

    if args.cmake_dargs:
        for i in args.cmake_dargs:
          cmake_options.append( f"-D{i}" )

    cmake_options.append( f"{src_path}")
    cmd_opts = " ".join(cmake_options)

    return cmake_executable, cmd_opts


def make_cmd():
    global args
    global OS_info

    make_options = []

    nproc = OS_info["NUM_PROC"]
    if os.name == "nt":
        make_executable = f"cmake.exe --build . " # ninja
        if args.verbose:
          make_options.append( "--verbose" )
        make_options.append( "--target all" )
        if args.install:
          make_options.append( "--target package --target install" )
    else:
        make_executable = f"make -j{nproc}"
        if args.verbose:
          make_options.append( "VERBOSE=1" )
        if True: # args.install:
         make_options.append( "install" )
    cmd_opts = " ".join(make_options)

    return make_executable, cmd_opts

def run_cmd(exe, opts):
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode

def main():
    global args
    os_detect()
    args = parse_args()

    # configure
    exe, opts = config_cmd()
    run_cmd(exe, opts)

    # make
    exe, opts = make_cmd()
    run_cmd(exe, opts)

if __name__ == '__main__':
    main()

