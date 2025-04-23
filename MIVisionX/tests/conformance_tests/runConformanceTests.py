# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from datetime import datetime
from subprocess import Popen, PIPE
import os
import platform
import argparse

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018 - 2024, AMD MIVisionX - Conformance System Report"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"


def shell(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output = p.communicate()[0][0:-1]
    return output


def write_formatted(output, f):
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")


def write_lines_as_table(header, lines, f):
    for h in header:
        f.write("|%s" % h)
    f.write("|\n")

    for h in header:
        f.write("|:---")
    f.write("|\n")

    for l in lines:
        fields = l.split()
        for field in fields:
            f.write("|%s" % field)
        f.write("|\n")


# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--directory', 	type=str, default='~/mivisionx-conformance',
                    help='conformance build directory - optional (default:~/)')
parser.add_argument('--backend_type',       type=str, default='ALL',
                    help='Backend type - optional (default:ALL [options:ALL/HOST/HIP/OCL])')
args = parser.parse_args()
setupDir = args.directory
backendType = args.backend_type

if backendType not in ('ALL', 'HOST', 'HIP', 'OCL'):
    print("ERROR: OpenVX Backends supported - [ALL or HOST or HIP or OCL]")
    exit()

# Directory for Conformance
if setupDir == '~/mivisionx-conformance':
    setupDir_cts = setupDir
else:
    setupDir_cts = setupDir+'/mivisionx-conformance'

# cts directory path
cts_dir = os.path.expanduser(setupDir_cts)
cts_dir = os.path.abspath(cts_dir)
scriptPath = os.path.dirname(os.path.realpath(__file__))

# clean old conformance
if os.path.exists(cts_dir):
    os.system('sudo -v')
    os.system('sudo rm -rf '+cts_dir)
    print("\nMIVisionX Conformance Tests: Removing Previous Logs -- "+cts_dir+"\n")


# Build MIVisionX OpenCL, HIP, HOST
os.system('(mkdir -p '+cts_dir+')')
os.system('(cd '+cts_dir +
          '; mkdir -p build-opencl; mkdir -p build-hip; mkdir -p build-host;)')
if backendType in ('ALL', 'HOST'):
    os.system('(cd '+cts_dir+'/build-host; cmake -DGPU_SUPPORT=OFF ' +
              scriptPath+'/../../; make -j8)')
if backendType in ('ALL', 'OCL'):
    os.system('(cd '+cts_dir+'/build-opencl; cmake -DBACKEND=OCL ' +
              scriptPath+'/../../; make -j8)')
if backendType in ('ALL', 'HIP'):
    os.system('(cd '+cts_dir+'/build-hip; cmake -DBACKEND=HIP ' +
              scriptPath+'/../../; make -j8)')


openvxIncludePath = scriptPath+'/../../amd_openvx/openvx/include'
openvxHostLib = cts_dir+'/build-host/lib'
openvxOpenclLib = cts_dir+'/build-opencl/lib'
openvxHipLib = cts_dir+'/build-hip/lib'

ctsHost = cts_dir+'/conformance_tests/build-cts-host'
ctsOpenCL = cts_dir+'/conformance_tests/build-cts-opencl'
ctsHIP = cts_dir+'/conformance_tests/build-cts-hip'

testDataExport = 'export VX_TEST_DATA_PATH=' + \
    cts_dir+'/conformance_tests/OpenVX-cts/test_data/'

# OpenVX Lib Deps
openvx_host_lib = openvx_ocl_lib = openvx_hip_lib = ''

# get system data
platform_name = platform.platform()
platform_name_fq = shell('hostname --all-fqdns')
platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

file_dtstr = datetime.now().strftime("%Y%m%d")
reportFilename = 'system_info_report_%s_%s.md' % (platform_name, file_dtstr)
report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
sys_info = shell('inxi -c0 -S')

cpu_info = shell('inxi -c0 -C')
cpu_info = cpu_info.rstrip()  # strip out clock speeds

gpu_info = shell('inxi -c0 -G')
gpu_info = gpu_info.rstrip()  # strip out X info

memory_info = shell('inxi -c 0 -m')
board_info = shell('inxi -c0 -M')

vbios = shell('(cd /opt/rocm/bin/; ./rocm-smi -v)')

#rocmInfo = shell('(cd /opt/rocm/bin/; ./rocm-smi -a)')

rocm_packages = shell('dpkg-query -W | grep rocm')
rocm_packages = rocm_packages.splitlines()

lib_type = '.so'
dep_libs = 'rt'
HIP_deps = '/opt/rocm/lib/libamdhip64.so'
LDD_Links = 'ldd'
if platform.system() == 'Darwin':
    lib_type = '.dylib'
    dep_libs = ''
    HIP_deps = ''
    LDD_Links = 'otool -L'

# Write Report
with open(reportFilename, 'w') as f:
    f.write("MIVisionX - System Report\n")
    f.write("================================\n")
    f.write("\n")

    f.write("Generated: %s\n" % report_dtstr)
    f.write("\n")

    f.write("Platform: %s (%s)\n" % (platform_name_fq, platform_ip))
    f.write("--------\n")
    f.write("\n")

    write_formatted(sys_info, f)
    write_formatted(cpu_info, f)
    write_formatted(gpu_info, f)
    write_formatted(board_info, f)
    write_formatted(memory_info, f)

    if platform.system() != 'Darwin':
        f.write("\n\n\n")
        f.write("ROCm Package and Version Report\n")
        f.write("-------------\n")
        f.write("\n")
        write_lines_as_table(['Package', 'Version'], rocm_packages, f)
        f.write("\n\n\n")

        f.write("Vbios Report\n")
        f.write("-------------\n")
        f.write("\n")
        write_formatted(vbios, f)
        f.write("\n")
        #f.write("ROCm Device Info Report\n")
        # f.write("-------------\n")
        # f.write("\n")
        #write_formatted(rocmInfo, f)
    f.write("\n\n")
    f.write("OpenVX Shared Library Dependencies\n")
    f.write("-------------\n")
    f.write("\n")
    if backendType in ('ALL', 'HOST'):
        openvx_host_lib = shell(LDD_Links+' '+openvxHostLib+'/libopenvx'+lib_type)
        f.write("* OpenVX HOST Library\n")
        write_formatted(openvx_host_lib, f)
    if backendType in ('ALL', 'OCL'):
        openvx_ocl_lib = shell(LDD_Links+' '+openvxOpenclLib+'/libopenvx'+lib_type)
        f.write("* OpenVX OpenCL Library\n")
        write_formatted(openvx_ocl_lib, f)
    if backendType in ('ALL', 'HIP'):
        openvx_hip_lib = shell(LDD_Links+' '+openvxHipLib+'/libopenvx'+lib_type)
        f.write("* OpenVX HIP Library\n")
        write_formatted(openvx_hip_lib, f)
    f.write("\n")

    f.write("\n\n---\n**Copyright AMD ROCm MIVisionX 2018 - 2024 -- runConformanceTests.py V-"+__version__+"**\n")

# system report file
reportFileDir = os.path.abspath(reportFilename)
os.system('(cp '+reportFileDir+' '+cts_dir+')')
print("\nSTATUS: Output Report File - "+reportFileDir)

# Build CTS - OpenVX 1.3
os.system('(cd '+cts_dir+'; mkdir conformance_tests && cd conformance_tests && git clone -b openvx_1.3 https://github.com/KhronosGroup/OpenVX-cts.git)')
os.system('(cd '+cts_dir+'/conformance_tests; mkdir -p build-cts-opencl && mkdir -p build-cts-hip && mkdir -p build-cts-host)')
# Build CTS Host
if backendType in ('ALL', 'HOST'):
    os.system('(cd '+ctsHost+' && cmake -DOPENVX_INCLUDES='+openvxIncludePath+' -DOPENVX_LIBRARIES='+openvxHostLib +
              '/libopenvx'+lib_type+'\;'+openvxHostLib+'/libvxu'+lib_type+'\;pthread\;dl\;m\;'+dep_libs+' -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts)')
    os.system('(cd '+ctsHost+' && cmake --build . )')
# Build CTS OpenCL
if backendType in ('ALL', 'OCL'):
    os.system('(cd '+ctsOpenCL+' && cmake -DOPENVX_INCLUDES='+openvxIncludePath+' -DOPENVX_LIBRARIES='+openvxOpenclLib +
              '/libopenvx'+lib_type+'\;'+openvxOpenclLib+'/libvxu'+lib_type+'\;pthread\;dl\;m\;'+dep_libs+' -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts)')
    os.system('(cd '+ctsOpenCL+' && cmake --build . )')
# Build CTS HIP
if backendType in ('ALL', 'HIP'):
    os.system('(cd '+ctsHIP+' && cmake -DOPENVX_INCLUDES='+openvxIncludePath+' -DOPENVX_LIBRARIES='+openvxHipLib+'/libopenvx'+lib_type+'\;' +
              openvxHipLib+'/libvxu'+lib_type+'\;'+HIP_deps+'\;pthread\;dl\;m\;'+dep_libs+' -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts)')
    os.system('(cd '+ctsHIP+' && cmake --build . )')

# Run Host
if backendType in ('ALL', 'HOST'):
    os.system('(cd '+ctsHost+' && '+testDataExport +
              ' && LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-HOST-CTS.md )')
    # save conformance logs
    host_data = cts_log_data = ""
    with open(reportFileDir) as fp:
        host_data = fp.read()
    with open(ctsHost+'/OpenVX-HOST-CTS.md') as fp:
        cts_log_data = fp.read()
    host_data += "\n\n"
    host_data += cts_log_data
    with open('HOST_Conformance_Logs.md', 'w') as fp:
        fp.write(host_data)

# Run OpenCL
if backendType in ('ALL', 'OCL'):
    os.system('(cd '+ctsOpenCL+' && '+testDataExport +
              ' && AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-OCL.md )')
    os.system('(cd '+ctsOpenCL+' && '+testDataExport +
              ' && AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-OCL.md )')
    # save conformance logs
    ocl_data = cts_log_data = ""
    with open(reportFileDir) as fp:
        ocl_data = fp.read()
    with open(ctsOpenCL+'/OpenVX-GPU-CTS-OCL.md') as fp:
        cts_log_data = fp.read()
    ocl_data += "\n\n"
    ocl_data += cts_log_data
    with open('OCL_Conformance_Logs.md', 'w') as fp:
        fp.write(ocl_data)

# Run HIP
if backendType in ('ALL', 'HIP'):
    os.system('(cd '+ctsHIP+' && '+testDataExport +
              ' && AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-HIP.md )')
    os.system('(cd '+ctsHIP+' && '+testDataExport +
              ' && AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-HIP.md )')
    # save conformance logs
    hip_data = cts_log_data = ""
    with open(reportFileDir) as fp:
        hip_data = fp.read()
    with open(ctsHIP+'/OpenVX-GPU-CTS-HIP.md') as fp:
        cts_log_data = fp.read()
    hip_data += "\n\n"
    hip_data += cts_log_data
    with open('HIP_Conformance_Logs.md', 'w') as fp:
        fp.write(hip_data)

print("\nrunConformanceTests.py - V:"+__version__+"\n")
