#!/usr/bin/python3
""" Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
Manage build and installation"""

import os
import subprocess
import argparse
import platform as pf
import shutil

class AutoBuilder:
    def __init__(self):
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
        print()
        

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
        self.parser.add_argument('-b', '--benchmarks', required=False, default=False, dest='build_bench', action='store_true',
                                 help='Generate benchmarks only (default: False)')
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
            os.makedirs(full_path)
        except FileExistsError:
            ... # file already exists

    def __rm_dir__(self, dir_path: str):
        if os.path.isabs(dir_path):
            full_path = dir_path
        else:
            full_path = os.path.join(self.lib_dir, dir_path)
        try:
            if self.OS_info['System'] == 'Linux':
                subprocess.run(f'rm -rf "{full_path}"', shell=True)
            else:
                subprocess.run(f'RMDIR /S /Q {full_path}', shell=True)

        except FileNotFoundError:
            ... # no file to remove
    
    def __get_cmake_cmd__(self):

        m = ' Current Working Directory '
        print(f'{m:-^100}\n\t{self.lib_dir}')
        print()

        cmake_exe = ''
        cmake_options = [
            f'--toolchain={self.toolchain}',
            f'-DCPACK_SET_DESTDIR=OFF', 
            f'-DCPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF',
            f'-DGPU_TARGETS={self.args.gpu_architecture}'
        ]
 
        if self.args.debug:
            build_path = os.path.join(self.args.build_dir, 'debug')
            cmake_config = 'Debug'
        else:
            build_path = os.path.join(self.args.build_dir, 'release')
            cmake_config = 'Release'
        
        self.build_path = build_path
        
        cmake_options.append(f"-DCMAKE_BUILD_TYPE={cmake_config}")

        if self.OS_info['System'] == 'Linux':
            cmake_exe = shutil.which('cmake3')

            if cmake_exe is None:
                cmake_exe = shutil.which('cmake')
            
            if cmake_exe is None:
                raise(SystemError('Did not find cmake or cmake3 in system'))

            rocm_path = os.getenv('ROCM_PATH', '/opt/rocm')

            cmake_options.append(f"-DROCM_DIR:PATH={rocm_path}")
            cmake_options.append(f"-DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}")
            cmake_options.append(f"-DROCM_PATH={rocm_path}")
            cmake_options.append(f"-DCMAKE_PREFIX_PATH:PATH={rocm_path}")
            
        else:
            cmake_exe = shutil.which('cmake.exe')
            
            if cmake_exe is None:
                cmake_exe = shutil.which('cmake3.exe')

            if cmake_exe is None:
                raise(SystemError('Did not find cmake or cmake3 in system'))
            
            rocm_path = os.getenv('ROCM_PATH', 'C:/hip')
            rocm_cmake_path = os.getenv('ROCM_CMAKE_PATH', r'C:/hipSDK')

            rocm_path.replace('\\', '/')
            rocm_cmake_path.replace('\\', '/')

            cmake_options.append(f'-G Ninja')
            cmake_options.append( f"-DWIN32=ON -DCPACK_PACKAGING_INSTALL_PREFIX=") #" -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}"
            cmake_options.append( f'-DCMAKE_INSTALL_PREFIX="C:/hipSDK"' )
            cmake_options.append( f'-DCMAKE_CXX_FLAGS="-D_ENABLE_EXTENDED_ALIGNED_STORAGE"')
            cmake_options.append(f'-DROCM_PATH={rocm_path}') 
            cmake_options.append(f'-DCMAKE_PREFIX_PATH:PATH={rocm_path};{rocm_cmake_path}')


        if self.args.static_lib:
           cmake_options.append( f'-DBUILD_SHARED_LIBS=OFF')

        if self.args.skip_ld_conf_entry:
            cmake_options.append( f'-DROCM_DISABLE_LDCONFIG=ON' )

        if self.args.build_clients:
            self.args.build_tests = True
            self.args.build_bench = True
            cmake_options.append(f'-DBUILD_EXAMPLE=ON')

        if self.args.build_tests:
          cmake_options.append(f'-DBUILD_TEST=ON')
        
        if self.args.build_bench:
          cmake_options.append(f'-DBUILD_BENCHMARK=ON')

        if self.args.cmake_dargs:
            cmake_options += [f'-D{i}' for i in self.args.cmake_dargs]
        
        # putting '' around paths to avoid white space in pathing
        if self.OS_info['System'] == 'Linux':
            command_str = f"{cmake_exe}"  
        else:
            command_str = f'"{cmake_exe}"'

        m = 'CMAKE Options'
        print(f'{m:-^100}')
        for op in cmake_options:
            print(f'\t{op}')
            command_str += f' {op}'
        print()

        command_str += f' "{self.lib_dir}"'
        m = 'Final Command'
        print(f'{m:-^100}')
        print(command_str)
        print()
        return command_str
    
    def __call__(self):
        self.__parse_args__()
        cmake_command = self.__get_cmake_cmd__()

        self.__rm_dir__(self.build_path)
        self.__rm_dir__('build')
        self.__mk_dir__(self.build_path)

        curr_dir = os.path.abspath(os.curdir)
        os.chdir(self.build_path)


        subprocess.run(cmake_command, shell=True)
        
        if self.OS_info['System'] == 'Linux':
            v = ''
            if self.args.verbose:
                v = ' VERBOSE=1'
            subprocess.run(f' make -j {self.OS_info["Num Processor"]}{v}', shell=True)

            if self.args.install:
                subprocess.run(f'make install', shell=True)
        else:
            v = ''
            if self.args.verbose:
                v = ' --verbose'
            subprocess.run(f'ninja -j {self.OS_info["Num Processor"]}{v}', shell=True)
            if self.args.install:
                subprocess.run(f'ninja install', shell=True)    

        os.chdir(curr_dir)

if __name__ == '__main__':
    builder = AutoBuilder()
    builder()
