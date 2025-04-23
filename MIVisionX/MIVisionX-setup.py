# Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

__copyright__ = "Copyright 2018 - 2024, AMD ROCm MIVisionX"
__license__ = "MIT"
__version__ = "3.8.1"
__email__ = "mivisionx.support@amd.com"
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
parser.add_argument('--directory', 	type=str, default='~/mivisionx-deps',
                    help='Setup home directory - optional (default:~/)')
parser.add_argument('--opencv',    	type=str, default='4.6.0',
                    help='OpenCV Version - optional (default for non Ubuntu OS:4.6.0)')
parser.add_argument('--ffmpeg',    	type=str, default='OFF',
                    help='FFMPEG Installation - optional (default:OFF) [options:ON/OFF]')
parser.add_argument('--neural_net',	type=str, default='ON',
                    help='MIVisionX Neural Net Dependency Install - optional (default:ON) [options:ON/OFF]')
parser.add_argument('--inference',	type=str, default='ON',
                    help='MIVisionX Neural Net Inference Dependency Install - optional (default:ON) [options:ON/OFF]')
parser.add_argument('--amd_rpp',	 	type=str, default='ON',
                    help='MIVisionX amd_rpp Dependency Install - optional (default:ON) [options:ON/OFF]')
parser.add_argument('--developer', 	type=str, default='OFF',
                    help='Setup Developer Options - optional (default:OFF) [options:ON/OFF]')
parser.add_argument('--reinstall', 	type=str, default='OFF',
                    help='Remove previous setup and reinstall - optional (default:OFF) [options:ON/OFF]')
parser.add_argument('--backend', 	type=str, default='HIP',
                    help='MIVisionX Dependency Backend - optional (default:HIP) [options:HIP/CPU/OCL]')
parser.add_argument('--rocm_path', 	type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')
args = parser.parse_args()

setupDir = args.directory
opencvVersion = args.opencv
ffmpegInstall = args.ffmpeg.upper()
neuralNetInstall = args.neural_net.upper()
inferenceInstall = args.inference.upper()
amdRPPInstall = args.amd_rpp.upper()
developerInstall = args.developer.upper()
reinstall = args.reinstall.upper()
backend = args.backend.upper()
ROCM_PATH = args.rocm_path

# override default path if env path set 
if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
print("\nROCm PATH set to -- "+ROCM_PATH+"\n")

# check developer inputs
if ffmpegInstall not in ('OFF', 'ON'):
    print(
        "ERROR: FFMPEG Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit()
if neuralNetInstall not in ('OFF', 'ON'):
    print(
        "ERROR: Neural Net Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit()
if inferenceInstall not in ('OFF', 'ON'):
    print(
        "ERROR: Inference Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit()
if amdRPPInstall not in ('OFF', 'ON'):
    print(
        "ERROR: Neural Net Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit()
if developerInstall not in ('OFF', 'ON'):
    print(
        "ERROR: Developer Option Not Supported - [Supported Options: OFF or ON]\n")
    exit()
if reinstall not in ('OFF', 'ON'):
    print(
        "ERROR: Re-Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit()
if backend not in ('OCL', 'HIP', 'CPU'):
    print(
        "ERROR: Backend Option Not Supported - [Supported Options: CPU or OCL or HIP]\n")
    parser.print_help()
    exit()

# check ROCm installation
if os.path.exists(ROCM_PATH) and backend != 'CPU':
    print("\nROCm Installation Found -- "+ROCM_PATH+"\n")
    os.system('echo ROCm Info -- && '+ROCM_PATH+'/bin/rocminfo')
else:
    if backend != 'CPU':
        print("\nWARNING: ROCm Not Found at -- "+ROCM_PATH+"\n")
        print(
            "WARNING: If ROCm installed, set ROCm Path with \"--rocm_path\" option for full installation [Default:/opt/rocm]\n")
        print("WARNING: Limited dependencies will be installed\n")
        backend = 'CPU'
    else:
        print("\nSTATUS: CPU Backend Install\n")
    neuralNetInstall = 'OFF'
    inferenceInstall = 'OFF'

# get platfrom info
platfromInfo = platform.platform()

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

# Setup Directory for Deps
if setupDir == '~/mivisionx-deps':
    setupDir_deps = setupDir
else:
    setupDir_deps = setupDir+'/mivisionx-deps'

# setup directory path
deps_dir = os.path.expanduser(setupDir_deps)
deps_dir = os.path.abspath(deps_dir)

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
if "centos" in os_info_data or "redhat" in os_info_data:
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    osUpdate = 'makecache'
    if "VERSION_ID=7" in os_info_data:
        linuxCMake = 'cmake3'
        sudoValidate = 'sudo -k'
        platfromInfo = platfromInfo+'-redhat-7'
    elif "VERSION_ID=8" in os_info_data:
        platfromInfo = platfromInfo+'-redhat-8'
    elif "VERSION_ID=9" in os_info_data:
        platfromInfo = platfromInfo+'-redhat-9'
    else:
        platfromInfo = platfromInfo+'-redhat-centos-undefined-version'
elif "Ubuntu" in os_info_data:
    linuxSystemInstall = 'apt-get -y'
    linuxSystemInstall_check = '--allow-unauthenticated'
    osUpdate = 'update'
    linuxFlag = '-S'
    if "VERSION_ID=20" in os_info_data:
        platfromInfo = platfromInfo+'-Ubuntu-20'
    elif "VERSION_ID=22" in os_info_data:
        platfromInfo = platfromInfo+'-Ubuntu-22'
    elif "VERSION_ID=24" in os_info_data:
        platfromInfo = platfromInfo+'-Ubuntu-24'
    else:
        platfromInfo = platfromInfo+'-Ubuntu-undefined-version'
elif "SLES" in os_info_data:
    linuxSystemInstall = 'zypper -n'
    linuxSystemInstall_check = '--no-gpg-checks'
    osUpdate = 'refresh'
    platfromInfo = platfromInfo+'-SLES'
elif "Mariner" in os_info_data:
    linuxSystemInstall = 'tdnf -y'
    linuxSystemInstall_check = '--nogpgcheck'
    platfromInfo = platfromInfo+'-Mariner'
    osUpdate = 'makecache'
else:
    print("\nMIVisionX Setup on "+platfromInfo+" is unsupported\n")
    print("\nMIVisionX Setup Supported on: Ubuntu 20/22, RedHat 8/9, & SLES 15\n")
    exit()

# MIVisionX Setup
print("\nMIVisionX Setup on: "+platfromInfo+"\n")

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' '+osUpdate))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# Delete previous install
if os.path.exists(deps_dir) and reinstall == 'ON':
    ERROR_CHECK(os.system(sudoValidate))
    ERROR_CHECK(os.system('sudo rm -rf '+deps_dir))
    print("\nMIVisionX Setup: Removing Previous Install -- "+deps_dir+"\n")

# source install - package dependencies
libpkgConfig = "pkg-config"
if "centos" in os_info_data and "VERSION_ID=7" in os_info_data:
    libpkgConfig = "pkgconfig"
commonPackages = [
    'cmake',
    'wget',
    'unzip',
    str(libpkgConfig),
    'inxi'
]

neuralNetDebianPackages = [
    'half',
    'rocblas-dev',
    'miopen-hip-dev',
    'migraphx-dev'
]

inferenceDebianPackages = [
    'python3-dev',
    'python3-pip'
]

neuralNetRPMPackages = [
    'half',
    'rocblas-devel',
    'miopen-hip-devel',
    'migraphx-devel'
]

inferenceRPMPackages = [
    'python3-devel',
    'python3-pip'
]

pipNumpyVersion = "numpy==1.23.0"
pipProtoVersion= "protobuf==3.12.4"
pipONNXVersion = "onnx==1.12.0"

if "VERSION_ID=24" in os_info_data:
    pipNumpyVersion = "numpy==2.0.0"
    pipONNXVersion = "onnx==1.16.0"
    pipProtoVersion= "protobuf==3.20.2"
pip3InferencePackagesUbuntu = [
    'future==1.0.0',
    'pytz==2022.1',
    'google==3.0.0',
    str(pipNumpyVersion),
    str(pipProtoVersion),
    str(pipONNXVersion),
    'nnef==1.0.7'
]

pipONNXversion = "onnx==1.11.0"
pipNNEFversion = "nnef==1.0.7"
if "VERSION_ID=7" in os_info_data or "VERSION_ID=8" in os_info_data:
    pipNumpyVersion = "numpy==1.19.5"
    pipNNEFversion = "protobuf==3.12.4" # TBD: NO NNEF Package for RHEL 7/8
if "NAME=SLES" in os_info_data:
    pipNumpyVersion = "numpy==1.19.5"
    pipProtoVersion= "protobuf==3.19.5"
    pipNNEFversion = "protobuf==3.19.5" # TBD: NO NNEF Package for SLES

pip3InferencePackagesRPM = [
    'future==0.18.2',
    'pytz==2022.1',
    'google==3.0.0',
    str(pipNumpyVersion),
    str(pipProtoVersion),
    str(pipONNXversion),
    str(pipNNEFversion)
]

ffmpegDebianPackages = [
    'libavcodec-dev',
    'libavformat-dev',
    'libavutil-dev',
    'libswscale-dev'
]

rppDebianPackages = [
    'rpp',
    'rpp-dev'
]

rppRPMPackages = [
    'rpp',
    'rpp-devel'
]

rocdecodeDebianPackages = [
    'rocdecode',
    'rocdecode-dev'
]

rocdecodeRPMPackages = [
    'rocdecode',
    'rocdecode-devel'
]

openclDebianPackages = [
    'ocl-icd-opencl-dev'
]

openclRPMPackages = [
    'ocl-icd-devel'
]

opencvDebianPackages = [
    'libopencv-dev'
]

opencvRPMPackages = [
    'gtk2-devel',
    'libjpeg-devel',
    'libpng-devel',
    'libtiff-devel',
    'libavc1394'
]

# update
ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +' '+linuxSystemInstall_check+' '+osUpdate))

# Re-Install
if os.path.exists(deps_dir):
    print("\nMIVisionX Setup: Re-Installing Libraries from -- "+deps_dir+"\n")

    # common packages
    for i in range(len(commonPackages)):
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ commonPackages[i]))
    if "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install cmake3'))

    # neural net packages
    if neuralNetInstall == 'ON' and backend == 'HIP':
        ERROR_CHECK(os.system(sudoValidate))
        if "Ubuntu" in platfromInfo:
            for i in range(len(neuralNetDebianPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ neuralNetDebianPackages[i]))
        else:
            for i in range(len(neuralNetRPMPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ neuralNetRPMPackages[i]))
    # RPP
    if "Ubuntu" in platfromInfo:
        for i in range(len(rppDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rppDebianPackages[i]))
    else:
        for i in range(len(rppRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rppRPMPackages[i]))
    
    # rocDecode - Disable -- TBD: Turn on with package update
    # if "Ubuntu" in platfromInfo:
        # for i in range(len(rocdecodeDebianPackages)):
            # ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        # ' '+linuxSystemInstall_check+' install -y '+ rocdecodeDebianPackages[i]))
    # elif "redhat-7" not in platfromInfo:
        # for i in range(len(rocdecodeRPMPackages)):
            # ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        # ' '+linuxSystemInstall_check+' install -y '+ rocdecodeRPMPackages[i]))

    print("\nMIVisionX Dependencies Re-Installed with MIVisionX-setup.py V-"+__version__+" on "+platfromInfo+"\n")
    exit()

# Clean Install
else:
    print("\nMIVisionX Dependencies Installation with MIVisionX-setup.py V-"+__version__+"\n")
    ERROR_CHECK(os.system('mkdir '+deps_dir))
    # Create Build folder
    ERROR_CHECK(os.system('(cd '+deps_dir+'; mkdir build )'))
    # install pre-reqs
    ERROR_CHECK(os.system(sudoValidate))
    # common packages
    for i in range(len(commonPackages)):
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ commonPackages[i]))
    if "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install cmake3'))

    # neural net packages
    if neuralNetInstall == 'ON' and backend == 'HIP':
        ERROR_CHECK(os.system(sudoValidate))
        if "Ubuntu" in platfromInfo:
            for i in range(len(neuralNetDebianPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ neuralNetDebianPackages[i]))
        else:
            for i in range(len(neuralNetRPMPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ neuralNetRPMPackages[i]))

        # Install Model Compiler Deps
        if inferenceInstall == 'ON':
            modelCompilerDeps = os.path.expanduser('~/.mivisionx-model-compiler-deps')

            # Delete previous install
            if os.path.exists(modelCompilerDeps) and reinstall == 'ON':
                ERROR_CHECK(os.system(sudoValidate))
                ERROR_CHECK(os.system('sudo rm -rf '+modelCompilerDeps))
                print("\nMIVisionX Setup: Removing Previous Inference Install -- "+modelCompilerDeps+"\n")

            if not os.path.exists(modelCompilerDeps):
                print("STATUS: Model Compiler Deps Install - " +modelCompilerDeps+"\n")
                os.makedirs(modelCompilerDeps)
                ERROR_CHECK(os.system(sudoValidate))
                if "Ubuntu" in platfromInfo:
                    for i in range(len(inferenceDebianPackages)):
                        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                                ' '+linuxSystemInstall_check+' install -y '+ inferenceDebianPackages[i]))
                    # Install base Deps
                    for i in range(len(pip3InferencePackagesUbuntu)):
                        ERROR_CHECK(os.system('pip3 install '+ pip3InferencePackagesUbuntu[i]))
                else:
                    for i in range(len(inferenceRPMPackages)):
                        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                                ' '+linuxSystemInstall_check+' install -y '+ inferenceRPMPackages[i]))
                    # Install base Deps
                    for i in range(len(pip3InferencePackagesRPM)):
                            ERROR_CHECK(os.system('pip3 install '+ pip3InferencePackagesRPM[i]))
                    # NNEF Manual Install
                    if "SLES" in platfromInfo or "Mariner" in platfromInfo or "redhat-8" in platfromInfo:
                        ERROR_CHECK(os.system('mkdir -p '+modelCompilerDeps+'/nnef-deps'))
                        ERROR_CHECK(os.system(
                            '(cd '+modelCompilerDeps+'/nnef-deps; git clone -b nnef-v1.0.0 https://github.com/KhronosGroup/NNEF-Tools.git)'))
                        ERROR_CHECK(os.system(
                            '(cd '+modelCompilerDeps+'/nnef-deps/NNEF-Tools/parser/cpp; mkdir -p build && cd build; '+linuxCMake+' ..; make -j$(nproc); sudo make install)'))
                        ERROR_CHECK(os.system(
                            '(cd '+modelCompilerDeps+'/nnef-deps/NNEF-Tools/parser/python; sudo python3 setup.py install)'))
            else:
                print("STATUS: Model Compiler Deps Pre-Installed - " +modelCompilerDeps+"\n")
    else:
        print("\nSTATUS: MIVisionX Setup: Neural Network only supported with HIP backend\n")

    if amdRPPInstall == 'ON' and backend == 'HIP':
    # RPP
        if "Ubuntu" in platfromInfo:
            for i in range(len(rppDebianPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rppDebianPackages[i]))
        else:
            for i in range(len(rppRPMPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rppRPMPackages[i]))
    else:
        print("\nSTATUS: MIVisionX Setup: AMD VX RPP only supported with HIP backend\n")
        
    # rocDecode - Disable TBD: Turn on with package update
    # if "Ubuntu" in platfromInfo:
        # for i in range(len(rocdecodeDebianPackages)):
            # ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        # ' '+linuxSystemInstall_check+' install -y '+ rocdecodeDebianPackages[i]))
    # elif "redhat-7" not in platfromInfo:
        # for i in range(len(rocdecodeRPMPackages)):
                # ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        # ' '+linuxSystemInstall_check+' install -y '+ rocdecodeRPMPackages[i]))

    # Install ffmpeg
    if ffmpegInstall == 'ON':
        if "Ubuntu" in platfromInfo:
            for i in range(len(ffmpegDebianPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                                ' '+linuxSystemInstall_check+' install -y '+ ffmpegDebianPackages[i]))

        elif "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install epel-release'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' localinstall --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg ffmpeg-devel'))
        elif "centos-8" in platfromInfo or "redhat-8" in platfromInfo:
            # el8 x86_64 packages
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/SDL2-2.0.10-2.el8.x86_64.rpm'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg ffmpeg-devel'))
        elif "centos-9" in platfromInfo or "redhat-9" in platfromInfo:
            # el9 x86_64 packages
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install https://dl.fedoraproject.org/pub/epel/epel-next-release-latest-9.noarch.rpm'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %rhel).noarch.rpm'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install ffmpeg ffmpeg-free-devel'))
        elif "SLES" in platfromInfo:
            # FFMPEG-4 packages
            ERROR_CHECK(os.system(
                    'sudo zypper ar -cfp 90 \'https://ftp.gwdg.de/pub/linux/misc/packman/suse/openSUSE_Leap_$releasever/Essentials\' packman-essentials'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                    ' install ffmpeg-4'))


    # Install OpenCV -- TBD cleanup
    ERROR_CHECK(os.system('(cd '+deps_dir+'/build; mkdir OpenCV )'))
    # Install
    ERROR_CHECK(os.system(sudoValidate))
    if "Ubuntu" in platfromInfo:
        for i in range(len(opencvDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ opencvDebianPackages[i]))
    else:
        if "centos" in platfromInfo or "redhat" in platfromInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' groupinstall \'Development Tools\''))
        elif "SLES" in platfromInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install -t pattern devel_basis'))
        for i in range(len(opencvRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ opencvRPMPackages[i]))
        # OpenCV 4.6.0
        # Get Source and install
        ERROR_CHECK(os.system(
            '(cd '+deps_dir+'; wget https://github.com/opencv/opencv/archive/'+opencvVersion+'.zip )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'; unzip '+opencvVersion+'.zip )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; '+linuxCMake +
                        ' -D WITH_EIGEN=OFF \
                        -D WITH_GTK=ON \
                        -D WITH_JPEG=ON \
                        -D BUILD_JPEG=ON \
                        -D WITH_OPENCL=OFF \
                        -D WITH_OPENCLAMDFFT=OFF \
                        -D WITH_OPENCLAMDBLAS=OFF \
                        -D WITH_VA_INTEL=OFF \
                        -D WITH_OPENCL_SVM=OFF  \
                        -D CMAKE_INSTALL_PREFIX=/usr/local \
                        -D BUILD_LIST=core,features2d,highgui,imgcodecs,imgproc,photo,video,videoio  \
                        -D CMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
                        ../../opencv-'+opencvVersion+' )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; make -j$(nproc))'))
        ERROR_CHECK(os.system(sudoValidate))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; sudo make install)'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; sudo ldconfig)'))

    if developerInstall == 'ON':
        ERROR_CHECK(os.system(sudoValidate))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install autoconf texinfo wget'))
        if "Ubuntu" in platfromInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                    ' install build-essential libgmp-dev'))
        else:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                    ' install gmp-devel'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                    ' groupinstall \'Development Tools\' '))
        ERROR_CHECK(os.system(
            '(cd '+deps_dir+'; wget https://ftp.gnu.org/gnu/gdb/gdb-12.1.tar.gz )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'; tar -xvzf gdb-12.1.tar.gz )'))
        ERROR_CHECK(os.system(
            '(cd '+deps_dir+'/gdb-12.1; ./configure --with-python3; make CXXFLAGS="-static-libstdc++" -j$(nproc); sudo make install)'))

    # Install OpenCL ICD Loader
    if backend == 'OCL':
        if "Ubuntu" in platfromInfo:
            for i in range(len(openclDebianPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ openclDebianPackages[i]))
        else:
            for i in range(len(openclRPMPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ openclRPMPackages[i]))
                        
    print("\nMIVisionX Dependencies Installed with MIVisionX-setup.py V-"+__version__+" on "+platfromInfo+"\n")
