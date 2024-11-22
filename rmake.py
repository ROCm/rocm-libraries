#!/usr/bin/python3
""" Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
Manage build and installation"""

import re
import sys
import os
import subprocess
import argparse
import ctypes
import pathlib
from fnmatch import fnmatchcase
import platform as pf
import shutil

class AutoBuilder:
    def __init__(self):
        self.param = {}
        self.parser = argparse.ArgumentParser(description="""
        Checks build arguments
        """)
        
        sysInfo = pf.uname()

        # getting os information
        self.OS_info = {
            "Machine" : sysInfo.machine,
            "Node Name" : sysInfo.node,
            "Num Processor" : os.cpu_count(),
            "Processor" : sysInfo.processor,
            "Release" : sysInfo.release,
            "System" : sysInfo.system,
            "Version" : sysInfo.version,
        }
        m = ' System Information '

        print()
        print(f'{m:-^100}')
        
        for k in self.OS_info:
            print(f'\t {k}: {self.OS_info[k]}')
        print(f'\n\n')
        

        self.lib_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.toolchain = f'toolchain-linux.cmake' if self.OS_info['System'] == 'Linux' else f'toolchain-windows.cmake'

    def __parse_args__(self):
        """Parse command-line arguments"""
        default_gpus = 'gfx906:xnack-,gfx1030,gfx1100,gfx1101,gfx1102,gfx1151,gfx1200,gfx1201'

        self.parser.add_argument('-g', '--debug', required=False, default=False,  action='store_true',
                            help='Generate Debug build (default: False)')
        self.parser.add_argument(      '--build_dir', type=str, required=False, default="build",
                            help='Build directory path (default: build)')
        self.parser.add_argument(      '--deps_dir', type=str, required=False, default=None,
                            help='Dependencies directory path (default: build/deps)')
        self.parser.add_argument(      '--skip_ld_conf_entry', required=False, default=False)
        self.parser.add_argument(      '--static', required=False, default=False, dest='static_lib', action='store_true',
                            help='Generate static library build (default: False)')
        self.parser.add_argument('-c', '--clients', required=False, default=False, dest='build_clients', action='store_true',
                            help='Generate all client builds (default: False)')
        self.parser.add_argument('-t', '--tests', required=False, default=False, dest='build_tests', action='store_true',
                            help='Generate unit tests only (default: False)')
        self.parser.add_argument('-i', '--install', required=False, default=False, dest='install', action='store_true',
                            help='Install after build (default: False)')
        self.parser.add_argument(      '--cmake-darg', required=False, dest='cmake_dargs', action='append', default=[],
                            help='List of additional cmake defines for builds (e.g. CMAKE_CXX_COMPILER_LAUNCHER=ccache)')
        self.parser.add_argument('-a', '--architecture', dest='gpu_architecture', required=False, default=default_gpus, #:sramecc+:xnack-" ) #gfx1030" ) #gfx906" ) # gfx1030" )
                            help='Set GPU architectures, e.g. all, gfx000, gfx803, gfx906:xnack-;gfx1030;gfx1100 (optional, default: all)')
        self.parser.add_argument('-v', '--verbose', required=False, default=False, action='store_true',
                            help='Verbose build (default: False)')  
        
        self.args = self.parser.parse_args()

    def __mk_dir__(self, dir_path: str):
        if os.path.isabs(dir_path):
            full_path = dir_path
        else:
            full_path = os.path.join(self.lib_dir, dir_path)

        try:
            os.mkdir(full_path)
            print(f'{full_path} created')
        except FileExistsError:
            print(f'{full_path} already exists ...')

    def __rm_dir__(self, dir_path: str):
        if os.path.isabs(dir_path):
            full_path = dir_path
        else:
            full_path = os.path.join(self.lib_dir, dir_path)
            print(f'{full_path} deleted')
        try:
            shutil.rmtree(full_path)
        except FileNotFoundError:
            print(f'{full_path} does not exists ...')

    def __get_cmake_cmd__(self):

        m = ' Current Working Directory '
        print(f'{m:-^100}\n\t{self.lib_dir}')

        cmake_exe = ''
        cmake_options = [f'-DCMAKE_TOOLCHAIN_FILE={self.toolchain}']
 
        if self.args.debug:
            build_path = os.path.join(self.args.build_dir, 'debug')
            cmake_config = 'Debug'
        else:
            build_path = os.path.join(self.args.build_dir, 'release')
            cmake_config = 'Release'
        
        cmake_options.append(f"-DCMAKE_BUILD_TYPE={cmake_config}")

        if self.args.deps_dir is None:
            deps_dir = os.path.abspath(os.path.join(self.args.build_dir, 'deps'))
        else:
            deps_dir = self.args.deps_dir

        if self.OS_info['System'] == 'Linux':
            cmake_exe = shutil.which('cmake3')

            if cmake_exe is None:
                cmake_exe = shutil.which('cmake')
            
            if cmake_exe is None:
                raise(SystemError('Did not find cmake or cmake3 in system'))

            rocm_path = os.getenv('ROCM_PATH', '/opt/rocm')
            
            cmake_options.append(f'-DROCM_DIR:PATH={rocm_path}')
            cmake_options.append(f'-DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}')
            cmake_options.append(f'-DROCM_PATH={rocm_path}')
            cmake_options.append(f'-DCMAKE_PREFIX_PATH:PATH={rocm_path},{rocm_path}')



    # if (OS_info["ID"] == 'windows'):
    #     cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path[:-1]};{rocm_cmake_path[1:]}" # -DCMAKE_INSTALL_PREFIX=rocmath-install" #-DCMAKE_INSTALL_LIBDIR=
    # else:
    

        print(cmake_options)
    
    def run(self):
        self.__parse_args__()
        self.__get_cmake_cmd__()
        



# if (OS_info["ID"] == 'windows'):
#         # we don't have ROCM on windows but have hip, ROCM can be downloaded if required
#         # CMAKE_PREFIX_PATH set to rocm_path and HIP_PATH set BY SDK Installer
#         raw_rocm_path = cmake_path(os.getenv('HIP_PATH', "C:/hip"))
#         rocm_path = f'"{raw_rocm_path}"' # guard against spaces in path
#         cmake_executable = "cmake.exe"
#         toolchain = os.path.join( src_path, "toolchain-windows.cmake" )
#         #set CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
#         cmake_platform_opts.append( f"-DWIN32=ON -DCPACK_PACKAGING_INSTALL_PREFIX=") #" -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}"
#         cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"C:/hipSDK\"" )

#         # MSVC requires acknowledgement of using extended aligned storage.
#         # Before VS 2017 15.8, has non-conforming alignment. VS 2017 15.8 fixes this, but inherently changes layouts of
#         # aligned storage with extended alignment, and thus binary compatibility with such types.
#         cmake_platform_opts.append( "-DCMAKE_CXX_FLAGS=\"-D_ENABLE_EXTENDED_ALIGNED_STORAGE\"")

#         rocm_cmake_path = '"' + cmake_path(os.getenv("ROCM_CMAKE_PATH", "C:/hipSDK")) + '"'
#         generator = f"-G Ninja"
#         # "-G \"Visual Studio 16 2019\" -A x64"  #  -G NMake ")  #
#         cmake_options.append( generator )

#     if (OS_info["ID"] == 'windows'):
#         cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path[:-1]};{rocm_cmake_path[1:]}" # -DCMAKE_INSTALL_PREFIX=rocmath-install" #-DCMAKE_INSTALL_LIBDIR=
#     else:
#         cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path[:-1]},{rocm_cmake_path[1:-1]}" # -DCMAKE_INSTALL_PREFIX=rocmath-install" #-DCMAKE_INSTALL_LIBDIR=
    
#     cmake_options.append( cmake_base_options )

#     print( cmake_options )

#     # clean
#     delete_dir( build_path )

#     create_dir( os.path.join(build_path, "clients") )
#     os.chdir( build_path )

#     # packaging options
#     cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF -DCPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF"
#     cmake_options.append( cmake_pack_options )

#     if args.static_lib:
#         cmake_options.append( f"-DBUILD_SHARED_LIBS=OFF" )

#     if args.skip_ld_conf_entry:
#         cmake_options.append( f"-DROCM_DISABLE_LDCONFIG=ON" )

#     if args.build_tests:
#         cmake_options.append( f"-DBUILD_TEST=ON -DBUILD_DIR={build_dir}" )

#     if args.build_clients:
#         cmake_options.append( f"-DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DBUILD_EXAMPLE=ON -DBUILD_DIR={build_dir}" )

#     cmake_options.append( f"-DAMDGPU_TARGETS={args.gpu_architecture}" )

#     if args.cmake_dargs:
#         for i in args.cmake_dargs:
#           cmake_options.append( f"-D{i}" )

#     cmake_options.append( f"{src_path}")

# #   case "${ID}" in
# #     centos|rhel)
# #     cmake_options="${cmake_options} -DCMAKE_FIND_ROOT_PATH=/usr/lib64/llvm7.0/lib/cmake/"
# #     ;;
# #     windows)
# #     cmake_options="${cmake_options} -DWIN32=ON -DROCM_PATH=${rocm_path} -DROCM_DIR:PATH=${rocm_path} -DCMAKE_PREFIX_PATH:PATH=${rocm_path}"
# #     cmake_options="${cmake_options} --debug-trycompile -DCMAKE_MAKE_PROGRAM=nmake.exe -DCMAKE_TOOLCHAIN_FILE=toolchain-windows.cmake"
# #     # -G '"NMake Makefiles JOM"'"
# #     ;;
# #   esac
#     cmd_opts = " ".join(cmake_options)

#     return cmake_executable, cmd_opts

# def run_cmd(exe, opts):
#     program = f"{exe} {opts}"
#     if sys.platform.startswith('win'):
#         sh = True
#     else:
#         sh = True
#     print(program)
#     proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=sh)
#     #proc = subprocess.Popen(cmd, cwd=os.getcwd())
#     #cwd=os.path.join(workingdir,"..",".."), stdout=fout, stderr=fout,
#      #                       env=os.environ.copy())
#     #proc.wait()
#     return proc.returncode

# def cmake_path(os_path):
#     if OS_info["ID"] == "windows":
#         return os_path.replace("\\", "/")
#     else:
#         return os.path.realpath(os_path)



# def make_cmd():
#     global args
#     global OS_info

#     make_options = []

#     if (OS_info["ID"] == 'windows'):
#         make_executable = "cmake.exe --build ." # ninja"
#         if args.verbose:
#           make_options.append( "--verbose" )
#         make_options.append( "--target all" )
#         if args.install:
#           make_options.append( "--target package --target install" )
#     else:
#         nproc = OS_info["NUM_PROC"]
#         make_executable = f"make -j {nproc}"
#         if args.verbose:
#           make_options.append( "VERBOSE=1" )
#         if args.install:
#           make_options.append( "install" )
#     cmd_opts = " ".join(make_options)

#     return make_executable, cmd_opts


# def main():
#     global args
#     os_detect()
#     args = parse_args()
#     # configure
#     exe, opts = config_cmd()
#     run_cmd(exe, opts)

#     # make/build/install
#     exe, opts = make_cmd()
#     run_cmd(exe, opts)


if __name__ == '__main__':
    builder = AutoBuilder()
    
    builder.run()

    # builder.__mk_dir__(f'build')
    # builder.__mk_dir__(f'/home/zenguyen/forks/rocPRIM/build')
    # builder.__rm_dir__(f'build')
    # builder.__rm_dir__(f'/home/zenguyen/forks/rocPRIM/build')


