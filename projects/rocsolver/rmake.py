#!/usr/bin/python3
"""Copyright 2020-2021 Advanced Micro Devices, Inc.
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

    # common
    parser.add_argument('-g', '--debug', required=False, default = False,  action='store_true',
                        help='Generate Debug build (optional, default: False)')
    parser.add_argument(      '--build_dir', type=str, required=False, default = "build",
                        help='Build directory path (optional, default: build)')
    parser.add_argument(      '--skip_ld_conf_entry', required=False, default=False)
    parser.add_argument(      '--static', required=False, default = False, dest='static_lib', action='store_true',
                        help='Generate static library build (optional, default: False)')
    parser.add_argument('-n', required=False, default=True, dest='optimal', action='store_false',
                        help='Include specialized kernels for small matrix sizes for better performance. (optional, default: True')
    parser.add_argument('-c', '--clients', required=False, default = False, dest='build_clients', action='store_true',
                        help='Generate all client builds (optional, default: False)')
    parser.add_argument('-i', '--install', required=False, default = False, dest='install', action='store_true',
                        help='Install after build (optional, default: False)')
    parser.add_argument(      '--cmake-darg', required=False, dest='cmake_dargs', action='append', default=[],
                        help='List of additional cmake defines for builds (optional, e.g. CMAKE)')
    parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
                        help='Verbose build (optional, default: False)')
    # rocsolver
    parser.add_argument(     '--clients-only', dest='clients_only', required=False, default = False, action='store_true',
                        help='Build only clients with a pre-built library')
    parser.add_argument(     '--rocblas_dir', dest='rocblas_dir', type=str, required=False, default = "",
                        help='Specify path to an existing rocBLAS install directory (optional, default: /opt/rocm/rocblas)')
    parser.add_argument(     '--rocsolver_dir', dest='rocsolver_dir', type=str, required=False, default = "",
                        help='Specify path to an existing rocSOLVER install directory (optional, default: /opt/rocm/rocsolver)')
    parser.add_argument('-a', '--architecture', dest='gpu_architecture', type=str, required=False, default=None,
                        help='Set GPU architectures to build for (optional)')
    parser.add_argument(      '--ci_labels', nargs='?', type=str, required=False, default="",
                        help='Semicolon (";") separated list of GitHub Labels with build options (not implemented, silently ignored)')
    parser.add_argument(      '--ci_gfx', nargs='?', type=str, required=False, default="",
                        help='Semicolon (";") separated list of gfx targets expected on test runs (not implemented, silently ignored)')

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
    cmake_executable = ""
    cmake_options = []
    src_path = cmake_path(cwd_path)
    cmake_platform_opts = []
    cmake_prefix_path = []
    if os.name == "nt":
        # not really rocm path as none exist, HIP_DIR set in toolchain is more important
        rocm_path = os.getenv( 'ROCM_CMAKE_PATH', "C:/github/rocm-cmake-master/share/rocm")
        cmake_executable = "cmake"
        #set CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
        cmake_platform_opts.append( "-DCPACK_PACKAGING_INSTALL_PREFIX=" )
        cmake_platform_opts.append( "-DCMAKE_INSTALL_PREFIX=C:/hipSDK" )
        lapack_dir = os.getenv("LAPACK_DIR")
        if lapack_dir is not None:
            cmake_prefix_path.append(lapack_dir)
        cblas_dir = os.getenv("cblas_DIR")
        if cblas_dir is not None:
            cmake_prefix_path.append(cblas_dir)

        if lapack_dir is not None:
            cmake_options.append(f"-DROCSOLVER_LAPACK_PATH={lapack_dir}")
        elif cblas_dir is not None:
            cmake_options.append(f"-DROCSOLVER_LAPACK_PATH={cblas_dir}")

        cmake_platform_opts.append('-DCMAKE_CXX_COMPILER=clang++.exe')
        cmake_platform_opts.append('-DCMAKE_C_COMPILER=clang.exe')
        if os.getenv("NO_VCPKG") is None:
            vcpkg_path =  pathlib.Path(os.getenv("VCPKG_PATH", "C:/github/vcpkg"))
            vcpkg_toolchain = vcpkg_path / 'scripts/buildsystems/vcpkg.cmake'
            cmake_options.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain.as_posix()} -DVCPKG_TARGET_TRIPLET=x64-windows")
        cmake_options.append("-DCMAKE_STATIC_LIBRARY_SUFFIX=.a")
        cmake_options.append("-DCMAKE_STATIC_LIBRARY_PREFIX=static_")
        cmake_options.append("-DCMAKE_SHARED_LIBRARY_SUFFIX=.dll")
        cmake_options.append("-DCMAKE_SHARED_LIBRARY_PREFIX=")
        cmake_options.append("-G Ninja")
    else:
        rocm_path = os.getenv( 'ROCM_PATH', "/opt/rocm")
        cmake_executable = "cmake"
        cmake_platform_opts.append( f"-DROCM_DIR:PATH={rocm_path} -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}" )
        cmake_platform_opts.append("-DCMAKE_INSTALL_PREFIX=rocsolver-install")

    if not args.optimal:
        cmake_options.append('-DOPTIMAL=OFF')

    print( f"Build source path: {src_path}")

    cmake_options.extend( cmake_platform_opts )

    cmake_prefix_path.append(rocm_path)
    cmake_base_options = "-DROCM_PATH={0} -DCMAKE_PREFIX_PATH={1}".format(rocm_path, ';'.join(cmake_prefix_path))
    cmake_options.append( cmake_base_options )

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF"
    cmake_options.append( cmake_pack_options )

    if os.getenv('CMAKE_CXX_COMPILER_LAUNCHER'):
        cmake_options.append( f"-DCMAKE_CXX_COMPILER_LAUNCHER={os.getenv('CMAKE_CXX_COMPILER_LAUNCHER')}" )

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
        cmake_options.append("-DBUILD_SHARED_LIBS=OFF" )

    if args.skip_ld_conf_entry:
        cmake_options.append("-DROCM_DISABLE_LDCONFIG=ON" )

    if args.build_clients:
        cmake_build_dir = cmake_path(build_dir)
        cmake_options.append("-DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON" )
        cmake_options.append("-DBUILD_TESTING=ON -DBUILD_CLIENTS_EXTRA_TESTS=ON" )

    if args.clients_only:
        cmake_options.append("-DBUILD_LIBRARY=OFF")

    if args.rocsolver_dir:
        cmake_options.append("-Drocsolver_DIR=" + cmake_path(args.rocsolver_dir))
        os.environ['rocsolver_DIR'] = cmake_path(args.rocsolver_dir)

    if args.rocblas_dir:
        cmake_options.append("-Drocblas_DIR=" + cmake_path(args.rocblas_dir))
        os.environ['rocblas_DIR'] = cmake_path(args.rocblas_dir)

    if args.gpu_architecture is not None:
        cmake_options.append( f"-DAMDGPU_TARGETS={args.gpu_architecture}" )

    if args.cmake_dargs:
        for i in args.cmake_dargs:
          cmake_options.append( f"-D{i}" )

    cmake_options.append( f"{src_path}")
    cmd_opts = " ".join(cmake_options)

    try:
        os.environ['HIP_DIR'] = cmake_path(os.environ['HIP_DIR'])
    except:
        pass

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

