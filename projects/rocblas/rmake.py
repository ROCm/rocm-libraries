#!/usr/bin/python3
"""Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib

try:
  import psutil
  psutil_imported = True
except ImportError:
  psutil_imported = False

args = {}
OS_info = {}


# yapf: disable
def parse_args():
    """Parse command-line arguments"""
    global OS_info

    parser = argparse.ArgumentParser(description="""Checks build arguments""")
    general_opts = parser.add_argument_group('General Build Options')
    experimental_opts = parser.add_argument_group('Experimental Build Options')

    general_opts.add_argument('-a', '--architecture', dest='gpu_architecture', type=str, required=False, default="all",
                        help='Set GPU architectures, e.g. all, auto, "gfx900;gfx906:xnack-", gfx1030 (optional, default: all, recommended: auto, builds for architecture detected on the build machine)')

    experimental_opts.add_argument(       '--address-sanitizer', dest='address_sanitizer', required=False, default=False, action='store_true',
                        help='Build with address sanitizer enabled. (optional, default: False')

    experimental_opts.add_argument('-b', '--branch', dest='tensile_tag', type=str, required=False, default="",
                        help='deprecated: Specify the legacy Tensile GitHub repository branch or tag to use. (eg. develop, or <commit hash>). note: tensile_tag.txt containing commit hash has been deprecated.')

    general_opts.add_argument(      '--build_dir', type=str, required=False, default="build",
                        help='Specify path to configure & build process output directory.(optional, default: ./build)')

    general_opts.add_argument(      '--ci_labels', type=str, required=False, default="",
                        help='Semi-colon seperated list of labels that may modify build (optional, e.g. "gfx12;noTensile")')

    general_opts.add_argument(      '--ci_gfx', type=str, required=False, default="",
                        help='Semi-colon seperated list of gfx targets expected on test runs (optional, e.g. "gfx1030;gfx1201")')

    general_opts.add_argument(      '--cleanup', required=False, default=False, action='store_true',
                        help='Remove intermediary build files after build to reduce disk usage. (Linux only handled by install.sh)')

    general_opts.add_argument('-c', '--clients', dest='build_clients', required=False, default=False, action='store_true',
                        help='Build the library clients benchmark and gtest (optional, default: False, Generated binaries will be located at <build_dir>/clients/staging)')

    general_opts.add_argument(     '--clients_no_fortran', required=False, default=False, action='store_true',
                        help='When building clients, build them without Fortran API testing or Fortran examples. (optional, default:False')

    general_opts.add_argument(     '--clients-only', dest='clients_only', required=False, default=False, action='store_true',
                        help='Skip building the library and only build the clients with a pre-built library.')

    general_opts.add_argument(      '--cmake-arg', dest='cmake_args', type=str, required=False, default="",
                        help='Forward the given argument to CMake when configuring the build.')

    general_opts.add_argument(      '--cmake-darg', dest='cmake_dargs', required=False, action='append', default=[],
                        help='List of additional cmake defines for builds (optional, e.g. CMAKE_)')

    general_opts.add_argument(      '--cmake_install', required=False, default=False, action='store_true',
                        help='Linux only: Handled by install.sh')

    experimental_opts.add_argument(      '--codecoverage', required=False, default=False, action='store_true',
                        help='Code coverage build. Requires Debug (-g|--debug) or RelWithDebInfo mode (-k|--relwithdebinfo), (optional, default: False)')

    general_opts.add_argument( '-d', '--dependencies', required=False, default=False, action='store_true',
                        help='Build and install external dependencies. (Handled by install.sh and on Windows rdeps.py')

    experimental_opts.add_argument('-f', '--fork', dest='tensile_fork', type=str, required=False, default="",
                        help='deprecated: Specify the username to the fork of the legacy Tensile GitHub repository (e.g., ROCm or MyUserName)')
    
    general_opts.add_argument('-g', '--debug', required=False, default=False,  action='store_true',
                        help='Build in Debug mode (optional, default: False)')

    general_opts.add_argument('-i', '--install', required=False, default=False, dest='install', action='store_true',
                        help='Generate and install library package after build. Windows only. Linux use install.sh (optional, default: False)')

    experimental_opts.add_argument(     '--install_invoked', required=False, default=False, action='store_true',
                        help='rmake invoked from install.sh so do not do dependency or package installation (default: False)')

    general_opts.add_argument('-j', '--jobs', type=int, required=False, default=0,
                        help='Specify number of parallel jobs to launch, affects memory usage (default: heuristic around logical core count)')

    general_opts.add_argument('--ninja', required=False, default=False, action='store_true',
                        help='Build using ninja generator (default: false on Linux, true on windows)')

    experimental_opts.add_argument('-k', '--relwithdebinfo', required=False, default=False, action='store_true',
                        help='Build in Release with Debug Info (optional, default: False)')

    experimental_opts.add_argument('-l', '--logic', dest='tensile_logic', type=str, required=False, default="asm_full",
                        help='Specify the Tensile logic target, e.g., asm_full, asm_lite, etc. (optional, default: asm_full)')

    experimental_opts.add_argument(    '--lazy-library-loading', dest='tensile_lazy_library_loading', required=False, default=True, action='store_true',
                        help='Enable on-demand loading of Tensile Library files, speeds up the rocblas initialization. (Default is enabled)')

    experimental_opts.add_argument(    '--no-lazy-library-loading', dest='tensile_lazy_library_loading', required=False, default=True, action='store_false',
                        help='Disable on-demand loading of Tensile Library files. (Default is enabled)')

    experimental_opts.add_argument(     '--library-path', dest='library_dir_installed', type=str, required=False, default="",
                        help='Specify path to a pre-built rocBLAS library, when building clients only using --clients-only flag. (optional, default: /opt/rocm/rocblas)')

    experimental_opts.add_argument('-n', '--no_tensile', dest='build_tensile', required=False, default=True, action='store_false',
                        help='Build a subset of rocBLAS library which does not require Tensile.')

    experimental_opts.add_argument(      '--no_hipblaslt', dest='build_hipblaslt', required=False, default=True, action='store_false',
                        help='Build a subset of rocBLAS library which does not require HipBLASLt.')

    experimental_opts.add_argument(     '--merge-architectures', dest='merge_architectures', required=False, default=False, action='store_true',
                        help='Merge TensileLibrary files for different architectures into single file (optional, was behavior in ROCm 5.1 and earlier)')

    experimental_opts.add_argument(     '--no-merge-architectures', dest='merge_architectures', required=False, default=False, action='store_false',
                        help='Keep TensileLibrary files separated by architecture (optional)')

    experimental_opts.add_argument(     '--msgpack', dest='tensile_msgpack_backend', required=False, default=True, action='store_true',
                        help='Build Tensile backend to use MessagePack (optional, default: True)')

    experimental_opts.add_argument(     '--no-msgpack', dest='tensile_msgpack_backend', required=False, default=True, action='store_false',
                        help='Build Tensile backend not to use MessagePack and so use YAML (optional)')

    general_opts.add_argument( '--no-offload-compress', dest='no_offload_compress', required=False, default=False, action='store_true',
                        help='Do not apply offload compression.')

    general_opts.add_argument( '-r', '--relocatable', required=False, default=False, action='store_true',
                        help='Linux only: Add RUNPATH (based on ROCM_RPATH) and remove ldconf entry.')

    experimental_opts.add_argument(      '--run_header_testing', required=False, default=False, action='store_true',
                        help='Linux only: Run post build header testing. (options, default: False')

    general_opts.add_argument(      '--skipldconf', dest='skip_ld_conf_entry', required=False, default=False, action='store_true',
                        help='Linux only: Skip ld.so.conf entry.')

    general_opts.add_argument('-s', '--static', required=False, default=False, dest='static_lib', action='store_true',
                        help='Build rocblas as a static library. (optional, default: False)')

    experimental_opts.add_argument(      '--src_path', type=str, required=False, default="",
                        help='Source path. (optional, default: Current directory)')

    experimental_opts.add_argument('-t', '--test_local_path', dest='tensile_test_local_path', type=str, required=False, default="",
                        help='Use a local path for Tensile instead of relative shared/tensile folder (optional).  This is required to build a different commit of Tensile than rocBLAS. note: tensile_tag.txt containing commit hash has been deprecated.')

    experimental_opts.add_argument(      '--hipblaslt_path', dest='hipblaslt_path', type=str, required=False, default="",
                        help='Use a local path for HipBLASLt (optional)')

    general_opts.add_argument(      '--upgrade_tensile_venv_pip', required=False, default=False, action='store_true',
                        help='Upgrade python pip and packaging versions during Tensile installation (optional, default: False)')

    experimental_opts.add_argument('-u', '--use-custom-version', dest='tensile_version', type=str, required=False, default="",
                        help='Ignore Tensile version and just use the Tensile tag (optional)')

    general_opts.add_argument('-v', '--verbose', required=False, default=False, action='store_true',
                        help='Verbose build (optional, default: False)')

    experimental_opts.add_argument('-X', '--exclude-checks', dest='exclude_checks', required=False, default=False, action='store_true',
                        help='Exclude compiler and configuration checks (optional, default: False)')

    return parser.parse_args()
# yapf: enable

def get_ram_GB():
    """
    Total amount of GB RAM available or zero if unknown
    """
    gb = 0
    env_limit = os.getenv('ROCM_CI_RAM_GB_LIMIT', "")
    if len(env_limit):
        gb = int(env_limit)
    if gb == 0:
        if psutil_imported:
            gb = round(psutil.virtual_memory().total / pow(1024, 3))
            print( "psutil: virtual_memory ", str(gb), " GB" )
        else:
            print( "psutil: not installed so can't estimate RAM limit" )
    return gb

def strip_ECC(token):
    return token.replace(':sramecc+', '').replace(':sramecc-', '').strip()

def gpu_detect():
    global OS_info
    OS_info["GPU"] = ""
    if os.name == "nt":
        cmd = "hipinfo.exe"
    else:
        cmd = "rocminfo"
    process = subprocess.run([cmd], stdout=subprocess.PIPE)
    for line_in in process.stdout.decode().splitlines():
        if os.name == "nt":
            if 'gcnArchName' in line_in:
                OS_info["GPU"] = strip_ECC( line_in.split(":")[1] )
                break
        else:
            if 'amdgcn-amd-amdhsa' in line_in:
                OS_info["GPU"] = strip_ECC( line_in.split("--")[1] )
                break

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
                        k, v = line.strip().split("=")
                        OS_info[k] = v.replace('"', '')
    OS_info["NUM_PROC"] = os.cpu_count()
    OS_info["RAM_GB"] = get_ram_GB()

def get_arch_parallelism() -> int:
    global args
    if (args.gpu_architecture == "all"):
        num_parallel = 4
    else:
        num_parallel = min( 4, len(args.gpu_architecture.split(';')) )
    return num_parallel

def get_compiler_jobs(env_var: str) -> int:
        cjobs = 1
        pjstr = env_var.split("parallel-jobs=")
        if len(pjstr) > 1:
            arg = pjstr[1].split(" ")
            if len(arg[0]):
                cjobs = int(arg[0])
        return cjobs

def get_env_compiler_parallelism() -> int:
    # disable for now due to windows compiler issue and linux ignored warning
    return 1

    # new and legacy env
    hip_clang_job_env = os.getenv('HIP_CLANG_NUM_PARALLEL_JOBS', "0")

    if len(hip_clang_job_env) and int(hip_clang_job_env) > 0:
        return int(hip_clang_job_env)
    else:
        clang_flags = os.getenv('CCC_OVERRIDE_OPTIONS', "")
        hipcc_flags = os.getenv('HIPCC_COMPILE_FLAGS_APPEND', "")
        cjobs = 1
        if len(clang_flags) > 1:
            cjobs = get_compiler_jobs( clang_flags )
        elif len(hipcc_flags) > 1:
            cjobs = get_compiler_jobs( hipcc_flags )
        if (cjobs == 1 and len(clang_flags) < 1) and len(hipcc_flags) < 1:
            cjobs = get_arch_parallelism()
            # use HIP_ define to capture to makefiles, as env override would be compile time
            # if cjobs > 1:
            #     custom_env["CCC_OVERRIDE_OPTIONS"] = f"#+-parallel-jobs={cjobs}"
        cjobs = min(8, cjobs)
        return cjobs

def jobs_heuristic() -> int:
    # auto jobs heuristics
    nprocs = min(OS_info["NUM_PROC"], 128) # disk limiter
    ram = OS_info["RAM_GB"]
    jobs = nprocs
    if (ram >= 16): # don't apply if below minimum RAM
        jobs = min(round(ram/2), jobs) # RAM limiter
    pjobs = get_env_compiler_parallelism()
    if (pjobs > 1 and pjobs < jobs):
        jobs = round(jobs / pjobs)
    if os.name == "nt":
        jobs = min(61, jobs) # multiprocessing limit (used by tensile)
    return int(jobs)

def create_dir(dir_path):
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join(os.getcwd(), dir_path)
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return


def delete_dir(dir_path):
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        run_cmd("RMDIR", f"/S /Q {dir_path}")
    else:
        run_cmd("rm", f"-rf {dir_path}")


def cmake_path(os_path):
    if os.name == "nt":
        return os_path.replace("\\", "/")
    else:
        return os.path.realpath(os_path)


def fatal(msg, code=1):
    print(msg)
    exit(code)


def deps_cmd():
    if os.name == "nt":
        exe = f"python3 rdeps.py"
        stripped_args = ""
    else:
        exe = f"./install.sh --rmake_invoked -d "
        all_args = ' '.join(sys.argv[1:])
        stripped_args = all_args.split('--ci_labels')[0] # last args in CI
        stripped_args = stripped_args.split('-a ')[0]
    return exe, stripped_args


def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    cmake_executable = "cmake"
    cmake_options = []
    if len(args.src_path):
        src_path = args.src_path
    else:
        src_path = cmake_path(cwd_path)
    cmake_platform_opts = []
    if os.name == "nt":
        generator = f"-G Ninja"
        cmake_options.append(generator)

        # CMAKE_PREFIX_PATH set to rocm_path and HIP_PATH set BY SDK Installer
        raw_rocm_path = cmake_path(os.getenv('HIP_PATH', "C:/hip"))
        rocm_path = f'"{raw_rocm_path}"' # guard against spaces in path
        # CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
        cmake_platform_opts.append(f"-DCPACK_PACKAGING_INSTALL_PREFIX=")
        cmake_platform_opts.append(f'-DCMAKE_INSTALL_PREFIX="C:/hipSDK"')
        toolchain = os.path.join(src_path, "toolchain-windows.cmake")
    else:
        rocm_raw_path = os.getenv('ROCM_PATH', "/opt/rocm")
        rocm_path = rocm_raw_path
        if (args.ninja):
            cmake_platform_opts.append(f"-G Ninja")
        cmake_platform_opts.append(f"-DROCM_DIR:PATH={rocm_path} -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}")
        cmake_platform_opts.append(f'-DCMAKE_INSTALL_PREFIX="rocblas-install"')
        toolchain = "toolchain-linux.cmake"

    print(f"Build source path: {src_path}")

    cjobs = get_env_compiler_parallelism()
    if cjobs > 1:
        compile_args = f"-DHIP_CLANG_NUM_PARALLEL_JOBS={cjobs}"
        cmake_options.append(compile_args)

    tools = f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
    cmake_options.append(tools)

    cmake_options.extend(cmake_platform_opts)

    if args.cmake_args:
        cmake_options.append(args.cmake_args)

    cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path}"
    cmake_options.append(cmake_base_options)

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF"
    cmake_options.append(cmake_pack_options)

    if os.getenv('CMAKE_CXX_COMPILER_LAUNCHER'):
        cmake_options.append(f'-DCMAKE_CXX_COMPILER_LAUNCHER={os.getenv("CMAKE_CXX_COMPILER_LAUNCHER")}')

    # build type
    cmake_config = ""
    build_dir = os.path.realpath(args.build_dir)
    if args.debug:
        build_path = os.path.join(build_dir, "debug")
        cmake_config = "Debug"
    elif args.relwithdebinfo:
        build_path = os.path.join(build_dir, "release-debug")
        cmake_config = "RelWithDebInfo"
    else:
        build_path = os.path.join(build_dir, "release")
        cmake_config = "Release"

    cmake_options.append(f"-DCMAKE_BUILD_TYPE={cmake_config}")

    if args.codecoverage:
        if args.debug or args.relwithdebinfo:
            cmake_options.append(f"-DBUILD_CODE_COVERAGE=ON")
        else:
            fatal("*** Code coverage is not supported for Release build! Aborting. ***")

    if args.address_sanitizer:
        cmake_options.append(f"-DBUILD_ADDRESS_SANITIZER=ON")

    if args.no_offload_compress:
        cmake_options.append(f"-DBUILD_OFFLOAD_COMPRESS=OFF")

    # clean
    delete_dir(build_path)

    create_dir(os.path.join(build_path, "clients"))
    os.chdir(build_path)

    if args.static_lib:
        cmake_options.append(f"-DBUILD_SHARED_LIBS=OFF")

    if args.relocatable:
        rocm_rpath = os.getenv('ROCM_RPATH', "/opt/rocm/lib:/opt/rocm/lib64")
        cmake_options.append(f'-DCMAKE_SHARED_LINKER_FLAGS=" -Wl,--enable-new-dtags -Wl,--rpath,{rocm_rpath}"')

    if args.skip_ld_conf_entry or args.relocatable:
        cmake_options.append(f"-DROCM_DISABLE_LDCONFIG=ON")

    if args.clients_only:
        args.build_clients = True # Implied
        if args.library_dir_installed:
            library_dir = args.library_dir_installed
        else:
            library_dir = f"{rocm_path}"
        cmake_lib_dir = cmake_path(library_dir)
        cmake_options.append(f"-DSKIP_LIBRARY=ON -DROCBLAS_LIBRARY_DIR={cmake_lib_dir}")

    if args.build_clients:
        cmake_build_dir = cmake_path(build_dir)
        cmake_options.append(
            f"-DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_DIR={cmake_build_dir}"
        )
        if args.clients_no_fortran:
            cmake_options.append(f"-DBUILD_FORTRAN_CLIENTS=OFF")

    if args.gpu_architecture == "auto":
        gpu_detect()
        if len(OS_info["GPU"]):
            args.gpu_architecture = OS_info["GPU"]
        else:
            fatal("Could not detect GPU as requested. Not continuing.")
    # not just for tensile
    cmake_options.append(f'-DGPU_TARGETS=\"{args.gpu_architecture}\"')

    if not args.build_tensile:
        cmake_options.append(f"-DBUILD_WITH_TENSILE=OFF")
    else:
        cmake_options.append(f"-DTensile_CODE_OBJECT_VERSION=default")
        if args.tensile_logic:
            cmake_options.append(f"-DTensile_LOGIC={args.tensile_logic}")
        if args.tensile_fork:
            cmake_options.append(f"-Dtensile_fork={args.tensile_fork}")
        if args.tensile_tag:
            cmake_options.append(f"-Dtensile_tag={args.tensile_tag}")
        if args.tensile_test_local_path:
            cmake_options.append(f"-DTensile_TEST_LOCAL_PATH={args.tensile_test_local_path}")
        if args.tensile_version:
            cmake_options.append(f"-DTENSILE_VERSION={args.tensile_version}")
        if args.upgrade_tensile_venv_pip:
            cmake_options.append(f"-DTENSILE_VENV_UPGRADE_PIP=ON")
        if not args.merge_architectures:
            cmake_options.append(f"-DTensile_SEPARATE_ARCHITECTURES=ON")
        else:
            cmake_options.append(f"-DTensile_SEPARATE_ARCHITECTURES=OFF")
        if args.tensile_lazy_library_loading:
            cmake_options.append(f"-DTensile_LAZY_LIBRARY_LOADING=ON")
        else:
            cmake_options.append(f"-DTensile_LAZY_LIBRARY_LOADING=OFF")
        if args.tensile_msgpack_backend:
            cmake_options.append(f"-DTensile_LIBRARY_FORMAT=msgpack")
        else:
            cmake_options.append(f"-DTensile_LIBRARY_FORMAT=yaml")
        if args.jobs != OS_info["NUM_PROC"]:
            # tensile doesn't use HIP_CLANG_NUM_PARALLEL_JOBS so multiply by cjobs
            cmake_options.append(f"-DTensile_CPU_THREADS={str(args.jobs*cjobs)}")
        if not args.build_hipblaslt:
            cmake_options.append(f"-DBUILD_WITH_HIPBLASLT=OFF")
        else:
            cmake_options.append(f"-DBUILD_WITH_HIPBLASLT=ON")
            if args.hipblaslt_path:
                cmake_options.append(f"-Dhipblaslt_path={args.hipblaslt_path}")

    if args.run_header_testing:
        cmake_options.append(f"-DRUN_HEADER_TESTING=ON")

    if args.exclude_checks:
        cmake_options.append(f"-DCONFIG_NO_COMPILER_CHECKS=ON")

    if args.cmake_dargs:
        for i in args.cmake_dargs:
            cmake_options.append(f"-D{i}")

    cmake_options.append(f"{src_path}")
    cmd_opts = " ".join(cmake_options)

    return cmake_executable, cmd_opts


def make_cmd():
    global args
    global OS_info

    make_options = []

    if os.name == "nt" or args.ninja:
        # the CMAKE_BUILD_PARALLEL_LEVEL currently doesn't work for windows build, so using -j
        # make_executable = f"cmake.exe -DCMAKE_BUILD_PARALLEL_LEVEL=4 --build . " # ninja
        exe_suffix = ".exe" if os.name == "nt" else ""
        make_executable = f"ninja{exe_suffix} -j {args.jobs}"
        if args.verbose:
            make_options.append("--verbose")
        make_options.append("all")
        if args.install:
            make_options.append("package")
    else:
        make_executable = f"make -j{args.jobs}"
        if args.verbose:
            make_options.append("VERBOSE=1")
    if not args.clients_only:
        make_options.append("install")
    cmd_opts = " ".join(make_options)

    return make_executable, cmd_opts

def arg_into_list(arg) -> list:
    arg = re.sub(r"['\"]|['\']",'', arg)
    return arg.split(';')

def label_modifiers(labels):
    global args
    processed = ["noTensile", "dependencies"]
    overlap = [v for v in processed if v in labels]
    if len(overlap):
        if "noTensile" in overlap:
            args.build_tensile = False
        # TODO: Deferred
        # if "dependencies" in overlap and os.name != "nt":
        #     args.dependencies = True

def gfx_modifiers(ci_gfx_list):
    global args
    if len(ci_gfx_list):
        gfx = arg_into_list( args.gpu_architecture )
        overlap = [v for v in ci_gfx_list if (v in gfx or "all" in gfx)]
        if len(overlap):
            args.gpu_architecture = ';'.join(overlap)

def run_cmd(exe, opts):
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode

def main():
    global args
    os_detect()
    args = parse_args()

    if args.ci_labels != "":
        label_modifiers( arg_into_list(args.ci_labels) )
    if args.ci_gfx != "":
        gfx_modifiers( arg_into_list(args.ci_gfx) )

    if args.jobs == 0:
        args.jobs = jobs_heuristic()
    if os.name == "nt" and args.jobs > 61:
        print( f"WARNING: jobs > 61 may fail on windows python multiprocessing (jobs = {args.jobs}).")

    if args.install_invoked:
        # ignore any install handled options
        args.dependencies = False
        args.install = False
        # args.cleanup = False

    print(OS_info)

    root_dir = os.curdir

    # depdendency install
    if (args.dependencies):
        exe, opts = deps_cmd()
        if run_cmd(exe, opts):
            fatal("Dependency install failed. Not continuing.")

    # configure
    exe, opts = config_cmd()
    if run_cmd(exe, opts):
        fatal("Configuration failed. Not continuing.")

    # make
    exe, opts = make_cmd()
    if run_cmd(exe, opts):
        fatal("Build failed. Not continuing.")

    # Linux install and cleanup not supported from rmake yet


if __name__ == '__main__':
    main()
