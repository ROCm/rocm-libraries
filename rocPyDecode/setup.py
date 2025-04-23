# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import subprocess
import os
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    @classmethod
    def has_ext_modules(self):
        return True

def get_rev_based_on_os():
    os.system("cat /etc/os-release | grep ID= > os_release")
    if os.path.exists('os_release'):
        with open('os_release', 'r') as file:
            os_release_info = file.read()
        if "ubuntu" in os_release_info:
            os.system('apt show rocm-libs -a | grep Version > rev_file')
            return True
        elif "rhel" in os_release_info:
            os.system('yum info rocm-libs | grep Version > rev_file')
            return True
        elif "sles" in os_release_info:
            os.system('zypper info rocm-libs | grep Version > rev_file')
            return True
        elif "centos" in os_release_info:
            os.system('yum info rocm-libs | grep Version > rev_file')
            return True
    return False

def get_rocm_rev():
    try:
        if get_rev_based_on_os() == False:
            return "0" # "Unknown os"
        # "apt/yum/zepper" abs in file: rev_file
        if os.path.exists('rev_file'):
            with open("rev_file", 'r') as file:
                rev_data = file.read().split(".")
                return rev_data[3][:5]
    except Exception as e:
        print("An error occurred:", e)
    return "0" # error, can't get the rev

class custom_bdist_wheel(_bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        # Customize the platform tag, CI team requirements
        plat = 'manylinux_2_28_x86_64'
        return python, abi, plat

# Call CMake to configure and build the project
build_dir = os.path.join(os.getcwd(), 'build')
os.makedirs(build_dir, exist_ok=True)
cmake_args=["cmake", ".", "-B"+build_dir, "-H"+os.getcwd()]

subprocess.check_call(cmake_args,cwd=os.getcwd())

# Invoke cmake --build to build the project
subprocess.check_call(['cmake', '--build', build_dir, '--config', 'Release', '--parallel'],cwd=os.getcwd())

# Install the built binaries
subprocess.check_call(['cmake', '--install', build_dir],cwd=os.getcwd())

setup(
    name='amd_rocpydecode_instinct',
    description='AMD ROCm Video Decoder Library',
    url='https://github.com/ROCm/rocPyDecode',
    version='0.4.0' + '.' + get_rocm_rev(),
    author='AMD',
    license='MIT License',
    include_package_data=True,
    packages=[ 'pyRocVideoDecode/samples','lib',''],
    package_dir={'pyRocVideoDecode/samples':'samples', '':'./build/rocpydecode/' },
    package_data={'': ['*.so']},  # Include .so files in the package
    cmdclass={'bdist_wheel': custom_bdist_wheel,},
    )

# Test built binaries -- TBD: Optional
# subprocess.check_call(['ctest', '--test-dir', build_dir, '-VV'])
