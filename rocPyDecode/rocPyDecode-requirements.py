# Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sys
import argparse
import platform
import traceback
if sys.version_info[0] < 3:
    import commands
else:
    import subprocess

__copyright__ = "Copyright (c) 2024, AMD ROCm rocPyDecode"
__version__ = "0.4.0"
__status__ = "Shipping"

# error check calls
def ERROR_CHECK(waitval):
    if(waitval != 0): # return code and signal flags
        print('ERROR_CHECK failed with status:'+str(waitval))
        traceback.print_stack()
        status = ((waitval >> 8) | waitval) & 255 # combine exit code and wait flags into single non-zero byte
        exit(status)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rocm_path', type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')

args = parser.parse_args()
ROCM_PATH = args.rocm_path

# override default path if env path set 
if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
print("\nROCm PATH set to -- "+ROCM_PATH+"\n")

# check ROCm installation
if os.path.exists(ROCM_PATH):
    print("\nROCm Installation Found -- " + ROCM_PATH + "\n")
    os.system('echo ROCm Info -- && ' + ROCM_PATH + '/bin/rocminfo')
else:
    print(
        "WARNING: If ROCm installed, set ROCm Path with \"--rocm_path\" option for full installation [Default:/opt/rocm]\n")
    print("ERROR: rocPyDecode Setup requires ROCm install\n")
    exit(-1)

# get platform info
platformInfo = platform.platform()

# sudo requirement check
sudoLocation = ''
userName = ''
if sys.version_info[0] < 3:
    status, sudoLocation = commands.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = commands.getstatusoutput("whoami")
else:
    status, sudoLocation = subprocess.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = subprocess.getstatusoutput("whoami")

# check os version
os_info_data = 'NOT Supported'
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as os_file:
        os_info_data = os_file.read().replace('\n', ' ')
        os_info_data = os_info_data.replace('"', '')

# setup for Linux
linuxSystemInstall = ''
linuxCMake = 'cmake'
linuxSystemInstall_check = ''
linuxFlag = ''
sudoValidate = 'sudo -v'
osUpdate = ''
if "centos" in os_info_data or "redhat" in os_info_data or "Oracle" in os_info_data:
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    osUpdate = 'makecache'
    if "VERSION_ID=7" in os_info_data:
        linuxCMake = 'cmake3'
        sudoValidate = 'sudo -k'
        platformInfo = platformInfo+'-redhat-7'
    elif "VERSION_ID=8" in os_info_data:
        platformInfo = platformInfo+'-redhat-8'
    elif "VERSION_ID=9" in os_info_data:
        platformInfo = platformInfo+'-redhat-9'
    else:
        platformInfo = platformInfo+'-redhat-centos-undefined-version'
elif "Ubuntu" in os_info_data:
    linuxSystemInstall = 'apt-get -y'
    linuxSystemInstall_check = '--allow-unauthenticated'
    linuxFlag = '-S'
    osUpdate = 'update'
    if "VERSION_ID=20" in os_info_data:
        platformInfo = platformInfo+'-Ubuntu-20'
    elif "VERSION_ID=22" in os_info_data:
        platformInfo = platformInfo+'-Ubuntu-22'
    elif "VERSION_ID=24" in os_info_data:
        platformInfo = platformInfo+'-Ubuntu-24'
    else:
        platformInfo = platformInfo+'-Ubuntu-undefined-version'
elif "SLES" in os_info_data:
    linuxSystemInstall = 'zypper -n'
    linuxSystemInstall_check = '--no-gpg-checks'
    platformInfo = platformInfo+'-SLES'
    osUpdate = 'refresh'
elif "Mariner" in os_info_data:
    linuxSystemInstall = 'tdnf -y'
    linuxSystemInstall_check = '--nogpgcheck'
    platformInfo = platformInfo+'-Mariner'
    osUpdate = 'makecache'
else:
    print("\rocPyDecode Setup on "+platformInfo+" is unsupported\n")
    print("\rocPyDecode Setup Supported on: Ubuntu 20/22, RedHat 8/9, & SLES 15\n")
    exit(-1)

# rocPyDecode Setup
print("\nrocPyDecode Setup on: "+platformInfo+"\n")
print("\nrocPyDecode Dependencies Installation with rocPyDecode-setup.py V-"+__version__+"\n")

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' update'))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# source install - common package dependencies
commonPackages = [
    'cmake',
    'pkg-config'
]

# Debian packages
coreDebianPackages = [
    'rocdecode-dev',
    'python3-dev',
    'python3-pip',
    'python3-pybind11',
    'libdlpack-dev'
]

# core RPM packages
# TODO: dlpack package missing in RPM
coreRPMPackages = [
    'rocdecode-devel',
    'python3-devel',
    'python3-pybind11',
    'python3-pip'
]

# update
ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +' '+linuxSystemInstall_check+' '+osUpdate))

# common packages
ERROR_CHECK(os.system(sudoValidate))
for i in range(len(commonPackages)):
    ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
            ' '+linuxSystemInstall_check+' install '+ commonPackages[i]))

# rocPyDecode Requirements
ERROR_CHECK(os.system(sudoValidate))
if "Ubuntu" in platformInfo:
    # core debian packages
    for i in range(len(coreDebianPackages)):
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                    ' '+linuxSystemInstall_check+' install '+ coreDebianPackages[i]))
elif "redhat" in platformInfo:
    # core RPM packages
    for i in range(len(coreRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                    ' '+linuxSystemInstall_check+' install '+ coreRPMPackages[i]))

print("rocPyDecode Dependencies Installed with rocPyDecode-setup.py V-"+__version__)
