# Copyright (c) 2020 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
import argparse
import os
import platform

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018 - 2024, AMD MIVisionX - Library Tests Report"
__license__ = "MIT"
__version__ = "1.2.0"
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


# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--install_directory',    type=str, default='/opt/rocm',
                    help='MIVisionX Install Directory - optional')
parser.add_argument('--backend_type',       type=str, default='HIP',
                    help='Backend type - optional (default:CPU [options:CPU/HIP/OCL])')
args = parser.parse_args()

installDir = args.install_directory
backendType = args.backend_type

# check arguments
if backendType not in ('CPU', 'HIP', 'OCL'):
    print("ERROR: Backends supported - CPU or HIP or OCL]")
    exit()

# check install
runVX_exe = installDir+'/bin/runvx'
if (os.path.isfile(runVX_exe)):
    print("STATUS: MIVisionX Install Path Found - "+installDir)
else:
    print("\nERROR: MIVisionX Install Path Not Found\n")
    exit()

# get absolute path
MIVisionXAbsPath = os.path.abspath(installDir)

# get data
platform_name = platform.platform()

if os.path.exists('/usr/bin/yum'):
    if "centos" not in platform_name or "redhat" not in platform_name:
        platfromInfo = platform_name+'-CentOS-RedHat'
elif os.path.exists('/usr/bin/apt-get'):
    if "Ubuntu" not in platform_name:
        platform_name = platform_name+'-Ubuntu'
elif os.path.exists('/usr/bin/zypper'):
    if "SLES" not in platform_name:
        platform_name = platform_name+'-SLES'
else:
    print("\nMIVisionX Library Test on "+platform_name+" is unsupported")
    print("MIVisionX Library Test Supported on: Ubuntu 20/22; CentOS 7/8; RedHat 8/9; & SLES 15 SP4")
    exit(1)

# TBD - Install inxi package

print("\nMIVisionX Library Test V:"+__version__ +
      " on "+platform_name+" is supported")

platform_name_fq = shell('hostname --all-fqdns')
platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

file_dtstr = datetime.now().strftime("%Y%m%d")
reportFilename = 'library_report_%s_%s_%s.md' % (
    backendType, platform_name, file_dtstr)
report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
sys_info = shell('inxi -c0 -S')

cpu_info = shell('inxi -c0 -C')

gpu_info = shell('inxi -c0 -G')

memory_info = shell('inxi -c 0 -m')
board_info = shell('inxi -c0 -M')

# level 1 - Libraries
openvx_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libopenvx.so')
vxu_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libvxu.so')
# level 2 - Libraries
loom_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libvx_loomsl.so')
# level 3 - libraries
opencv_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libvx_opencv.so')
media_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libvx_amd_media.so')
custom_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libvx_amd_custom.so')
# level 4 - libraries
nn_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libvx_nn.so')
migraphx_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libvx_amd_migraphx.so')
# level 5 - libraries
rpp_lib = shell('ldd '+MIVisionXAbsPath+'/lib/libvx_rpp.so')

# Executables
runvx_exe = shell('ldd '+MIVisionXAbsPath+'/bin/runvx')
runcl_exe = shell('ldd '+MIVisionXAbsPath+'/bin/runcl')
loom_exe = shell('ldd '+MIVisionXAbsPath+'/bin/loom_shell')
mv_compile_exe = shell('ldd '+MIVisionXAbsPath+'/bin/mv_compile')

warning = 0

# Write Report
with open(reportFilename, 'w') as f:
    f.write("MIVisionX - Libraries Report\n")
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

    f.write("\nLibraries Report - Backend:%s\n" % (backendType))
    f.write("--------\n")
    f.write("\n")
    f.write("\nInstall Path: %s\n" % (MIVisionXAbsPath))
    f.write("\n")
    # OpenVX Libraries
    f.write("* OpenVX Library\n")
    if not openvx_lib:
        f.write("WARNING: OpenVX Library Not Built\n")
        print("WARNING: OpenVX Library Not Built\n")
        warning = 1
    else:
        write_formatted(openvx_lib, f)
    f.write("\n")
    # OpenVX VXU Libraries
    f.write("* OpenVX VXU Library\n")
    if not vxu_lib:
        f.write("WARNING: OpenVX VXU Library Not Built\n")
        print("WARNING: OpenVX VXU Library Not Built\n")
        warning = 1
    else:
        write_formatted(vxu_lib, f)
    f.write("\n")
    # OpenCV Ext Libraries
    f.write("* VX OpenCV Ext Library\n")
    if not opencv_lib:
        f.write("WARNING: VX OpenCV Ext Library Not Built\n")
        print("WARNING: VX OpenCV Ext Library Not Built\n")
        warning = 1
    else:
        write_formatted(opencv_lib, f)
    f.write("\n")
    # VX RPP Libraries
    f.write("* VX RPP Library\n")
    if not rpp_lib:
        f.write("WARNING: VX RPP Library Not Built\n")
        print("WARNING: VX RPP Library Not Built\n")
        warning = 1
    else:
        write_formatted(rpp_lib, f)
    f.write("\n")
    if backendType == 'OCL':
        # Loom Libraries
        f.write("* Loom Library\n")
        if not loom_lib:
            f.write("WARNING: Loom Library Not Built\n")
            print("WARNING: Loom Library Not Built\n")
            warning = 1
        else:
            write_formatted(loom_lib, f)
        f.write("\n")
    if backendType == 'OCL' or backendType == 'HIP':
        # AMD Media Libraries
        f.write("* AMD Media Library\n")
        if not media_lib:
            f.write("WARNING: AMD Media Library Not Built\n")
            print("WARNING: AMD Media Library Not Built\n")
            warning = 1
        else:
            write_formatted(media_lib, f)
        f.write("\n")
        # AMD Custom Libraries
        f.write("* AMD Custom Library\n")
        if not custom_lib:
            f.write("WARNING: AMD Custom Library Not Built\n")
            print("WARNING: AMD Custom Library Not Built\n")
            warning = 1
        else:
            write_formatted(custom_lib, f)
        f.write("\n")
        if backendType == 'HIP':
            # VX NN Libraries
            f.write("* VX Neural Net Library\n")
            if not nn_lib:
                f.write("WARNING: VX Neural Net Library Not Built\n")
                print("WARNING: VX Neural Net Library Not Built\n")
                warning = 1
            else:
                write_formatted(nn_lib, f)
            f.write("\n")
            # VX MIGraphX Libraries
            f.write("* VX MIGraphX Library\n")
            if not migraphx_lib:
                f.write("WARNING: VX MIGraphX Library Not Built\n")
                print("WARNING: VX MIGraphX Library Not Built\n")
                warning = 1
            else:
                write_formatted(migraphx_lib, f)
            f.write("\n")

    f.write("\nExecutables Report\n")
    f.write("--------\n")
    f.write("\n")
    # RunVX
    f.write("* RunVX\n")
    if not runvx_exe:
        f.write("WARNING: RunVX Not Built\n")
        print("WARNING: RunVX Not Built\n")
        warning = 1
    else:
        write_formatted(runvx_exe, f)
    f.write("\n")
    if backendType == 'OCL':
        # RunCL
        f.write("* RunCL\n")
        if not runcl_exe:
            f.write("WARNING: RunCL Not Built\n")
            print("WARNING: RunCL Not Built\n")
            warning = 1
        else:
            write_formatted(runcl_exe, f)
        f.write("\n")
        # Loom Shell
        f.write("* Loom Shell\n")
        if not loom_exe:
            f.write("WARNING: Loom Shell Not Built\n")
            print("WARNING: Loom Shell Not Built\n")
            warning = 1
        else:
            write_formatted(loom_exe, f)
        f.write("\n")
    if backendType == 'OCL' or backendType == 'HIP':
        # MV Compile
        f.write("* MV Compile\n")
        if not mv_compile_exe:
            f.write("WARNING: MV Compile Not Built\n")
            print("WARNING: MV Compile Not Built\n")
            warning = 1
        else:
            write_formatted(mv_compile_exe, f)
        f.write("\n")
        f.write("\n")

    if warning == 1:
        f.write(
            "WARNING: Not all modules of MIVisionX is built, check for missing dependencies")
        print(
            "WARNING: Not all modules of MIVisionX is built, check for missing dependencies")
    else:
        f.write("SUCCESS: All modules of MIVisionX built")
        print("SUCCESS: All modules of MIVisionX built")
    f.write("\n")

    f.write("\n\n---\n**Copyright AMD ROCm MIVisionX 2018 - 2024 -- runLibraryTests.py V-"+__version__+"**\n")
    f.write("\n")

# report file
reportFileDir = os.path.abspath(reportFilename)
print("STATUS: Output Report File - "+reportFileDir)
if warning == 1:
    print("WARNING: Not all modules of MIVisionX is built, check for missing dependencies")
print("MIVisionX Tests - runLibraryTests.py - V:"+__version__+"\n")
