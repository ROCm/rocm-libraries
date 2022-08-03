#!/usr/bin/python3
""" Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
Manage build and installation"""

import re
import sys
import os
import subprocess
import argparse
import ctypes
import pathlib
from fnmatch import fnmatchcase

args = {}
param = {}
OS_info = {}

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""
    Checks build arguments
    """)
    parser.add_argument('-g', '--debug', required=False, default=False,  action='store_true',
                        help='Generate Debug build (default: False)')
    parser.add_argument(      '--build_dir', type=str, required=False, default="build",
                        help='Build directory path (default: build)')
    parser.add_argument(      '--deps_dir', type=str, required=False, default=None,
                        help='Dependencies directory path (default: build/deps)')
    parser.add_argument(      '--skip_ld_conf_entry', required=False, default=False)
    parser.add_argument(      '--static', required=False, default=False, dest='static_lib', action='store_true',
                        help='Generate static library build (default: False)')
    parser.add_argument('-c', '--clients', required=False, default=False, dest='build_clients', action='store_true',
                        help='Generate all client builds (default: False)')
    parser.add_argument('-t', '--tests', required=False, default=False, dest='build_tests', action='store_true',
                        help='Generate unit tests only (default: False)')
    parser.add_argument('-i', '--install', required=False, default=False, dest='install', action='store_true',
                        help='Install after build (default: False)')
    parser.add_argument(      '--cmake-darg', required=False, dest='cmake_dargs', action='append', default=[],
                        help='List of additional cmake defines for builds (e.g. CMAKE_CXX_COMPILER_LAUNCHER=ccache)')
    parser.add_argument('-a', '--architecture', dest='gpu_architecture', required=False, default="gfx906;gfx1030", #:sramecc+:xnack-" ) #gfx1030" ) #gfx906" ) # gfx1030" )
                        help='Set GPU architectures, e.g. all, gfx000, gfx803, gfx906:xnack-;gfx1030 (optional, default: all)')
    parser.add_argument('-v', '--verbose', required=False, default=False, action='store_true',
                        help='Verbose build (default: False)')
    return parser.parse_args()

def os_detect():
    inf_file = "/etc/os-release"
    if os.path.exists(inf_file):
        with open(inf_file) as f:
            for line in f:
                if "=" in line:
                    k,v = line.strip().split("=")
                    OS_info[k] = v.replace('"','')
    else:
        OS_info["ID"] = 'windows'
        OS_info["VERSION_ID"] = 10
    OS_info["NUM_PROC"] = os.cpu_count()
    print(OS_info)

def create_dir(dir_path):
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        fullpath = os.path.join( os.getcwd(), dir_path )
    pathlib.Path(fullpath).mkdir(parents=True, exist_ok=True)
    return

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if (OS_info["ID"] == 'windows'):
        run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        linux_path = pathlib.Path(dir_path).absolute()
        #print( linux_path )
        run_cmd( "rm" , f"-rf {linux_path}")

def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    src_path = cwd_path.replace("\\", "/")

    print( f"***************** {src_path}")
    cmake_executable = ""
    cmake_options = []
    cmake_platform_opts = []
    if (OS_info["ID"] == 'windows'):
        # we don't have ROCM on windows but have hip, ROCM can be downloaded if required
        rocm_path = os.getenv( 'ROCM_PATH', "C:/hipsdk/rocm-cmake-master") #C:/hip") # rocm/Utils/cmake-rocm4.2.0"
        cmake_executable = "cmake.exe"
        toolchain = os.path.join( src_path, "toolchain-windows.cmake" )
        #set CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
        cmake_platform_opts.append( f"-DWIN32=ON -DCPACK_PACKAGING_INSTALL_PREFIX=") #" -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}"
        cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"C:/hipSDK\"" )
        generator = f"-G Ninja"
        # "-G \"Visual Studio 16 2019\" -A x64"  #  -G NMake ")  #
        cmake_options.append( generator )
    else:
        rocm_path = os.getenv( 'ROCM_PATH', "/opt/rocm")
        if (OS_info["ID"] in ['centos', 'rhel']):
          cmake_executable = "cmake3"
        else:
          cmake_executable = "cmake"
        toolchain = "toolchain-linux.cmake"
        cmake_platform_opts = f"-DROCM_DIR:PATH={rocm_path} -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}"

    tools = f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
    cmake_options.append( tools )

    cmake_options.extend( cmake_platform_opts)


  # build type
    cmake_config = ""
    build_dir = args.build_dir
    if not args.debug:
        build_path = os.path.join(build_dir, "release")
        cmake_config="Release"
    else:
        build_path = os.path.join(build_dir, "debug")
        cmake_config="Debug"

    cmake_options.append( f"-DCMAKE_BUILD_TYPE={cmake_config}" ) #--build {build_path}" )

    if args.deps_dir is None:
        deps_dir = os.path.abspath(os.path.join(build_dir, 'deps')).replace('\\','/')
    else:
        deps_dir = args.deps_dir
    cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path}" # -DCMAKE_INSTALL_PREFIX=rocmath-install" #-DCMAKE_INSTALL_LIBDIR=
    cmake_options.append( cmake_base_options )

    print( cmake_options )

    # clean
    delete_dir( build_path )

    create_dir( os.path.join(build_path, "clients") )
    os.chdir( build_path )

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF -DCPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF"
    cmake_options.append( cmake_pack_options )

    if args.static_lib:
        cmake_options.append( f"-DBUILD_SHARED_LIBS=OFF" )

    if args.skip_ld_conf_entry:
        cmake_options.append( f"-DROCM_DISABLE_LDCONFIG=ON" )

    if args.build_tests:
        cmake_options.append( f"-DBUILD_TEST=ON -DBUILD_DIR={build_dir}" )

    if args.build_clients:
        cmake_options.append( f"-DBUILD_TEST=ON -DBUILD_EXAMPLE=ON -DBUILD_DIR={build_dir}" )

    cmake_options.append( f"-DAMDGPU_TARGETS={args.gpu_architecture}" )

    if args.cmake_dargs:
        for i in args.cmake_dargs:
          cmake_options.append( f"-D{i}" )

    cmake_options.append( f"{src_path}")

#   case "${ID}" in
#     centos|rhel)
#     cmake_options="${cmake_options} -DCMAKE_FIND_ROOT_PATH=/usr/lib64/llvm7.0/lib/cmake/"
#     ;;
#     windows)
#     cmake_options="${cmake_options} -DWIN32=ON -DROCM_PATH=${rocm_path} -DROCM_DIR:PATH=${rocm_path} -DCMAKE_PREFIX_PATH:PATH=${rocm_path}"
#     cmake_options="${cmake_options} --debug-trycompile -DCMAKE_MAKE_PROGRAM=nmake.exe -DCMAKE_TOOLCHAIN_FILE=toolchain-windows.cmake"
#     # -G '"NMake Makefiles JOM"'"
#     ;;
#   esac
    cmd_opts = " ".join(cmake_options)

    return cmake_executable, cmd_opts


def make_cmd():
    global args
    global OS_info

    make_options = []

    if (OS_info["ID"] == 'windows'):
        make_executable = "cmake.exe --build ." # ninja"
        if args.verbose:
          make_options.append( "--verbose" )
        make_options.append( "--target all" )
        if args.install:
          make_options.append( "--target package --target install" )
    else:
        nproc = OS_info["NUM_PROC"]
        make_executable = f"make -j{nproc}"
        if args.verbose:
          make_options.append( "VERBOSE=1" )
        if args.install:
          make_options.append( "install" )
    cmd_opts = " ".join(make_options)

    return make_executable, cmd_opts

def run_cmd(exe, opts):
    program = f"{exe} {opts}"
    if sys.platform.startswith('win'):
        sh = True
    else:
        sh = True
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=sh)
    #proc = subprocess.Popen(cmd, cwd=os.getcwd())
    #cwd=os.path.join(workingdir,"..",".."), stdout=fout, stderr=fout,
     #                       env=os.environ.copy())
    #proc.wait()
    return proc.returncode

def main():
    global args
    os_detect()
    args = parse_args()
    # configure
    exe, opts = config_cmd()
    run_cmd(exe, opts)

    # make/build/install
    exe, opts = make_cmd()
    run_cmd(exe, opts)


if __name__ == '__main__':
    main()
