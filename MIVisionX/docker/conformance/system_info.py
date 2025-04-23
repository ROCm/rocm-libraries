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

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018 - 2024, AMD MIVisionX - System Report"
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

rocmInfo = shell('(cd /opt/rocm/bin/; ./rocm-smi -a)')

rocm_packages = shell('dpkg-query -W | grep rocm')
rocm_packages = rocm_packages.splitlines()

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
    f.write("ROCm Device Info Report\n")
    f.write("-------------\n")
    f.write("\n")
    write_formatted(rocmInfo, f)
    f.write("\n")

    f.write("\n\n---\n**Copyright AMD ROCm MIVisionX 2018 - 2024 -- system_info.py V-"+__version__+"**\n")

# report file
reportFileDir = os.path.abspath(reportFilename)
print("\nSTATUS: Output Report File - "+reportFileDir)
print("\nsystem_info.py - V:"+__version__+"\n")
