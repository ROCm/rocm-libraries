# Copyright (c) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

__copyright__ = "Copyright 2022 - 2025, AMD ROCm Augmentation Library"
__license__ = "MIT"
__version__ = "3.0.0"
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
parser.add_argument('--directory', 	type=str, default='~/rocal-deps',
                    help='Setup home directory - optional (default:~/)')
parser.add_argument('--rocm_path', 	type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')
parser.add_argument('--backend', 	type=str, default='HIP',
                    help='rocAL Dependency Backend - optional (default:HIP) [options:CPU/OCL/HIP]')
parser.add_argument('--ffmpeg',    	type=str, default='OFF',
                    help='FFMPEG Installation - optional (default:OFF) [options:ON/OFF]')
parser.add_argument('--reinstall', 	type=str, default='OFF',
                    help='Remove previous setup and reinstall - optional (default:OFF) [options:ON/OFF]')
args = parser.parse_args()

setupDir = args.directory
ROCM_PATH = args.rocm_path
backend = args.backend.upper()
ffmpegInstall = args.ffmpeg.upper()
reinstall = args.reinstall.upper()

# override default path if env path set 
if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
print("\nROCm PATH set to -- "+ROCM_PATH+"\n")

# check developer inputs
if backend not in ('OCL', 'HIP', 'CPU'):
    print(
        "ERROR: Backend Option Not Supported - [Supported Options: CPU or OCL or HIP]\n")
    parser.print_help()
    exit()
if ffmpegInstall not in ('OFF', 'ON'):
    print(
        "ERROR: FFMPEG Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit()
if reinstall not in ('OFF', 'ON'):
    print(
        "ERROR: Re-Install Option Not Supported - [Supported Options: OFF or ON]\n")
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

# Setup Directory for Deps
if setupDir == '~/rocal-deps':
    setupDir_deps = setupDir
else:
    setupDir_deps = setupDir+'/rocal-deps'

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
if "centos" in os_info_data or "redhat" in os_info_data or "Oracle" in os_info_data:
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    osUpdate = 'makecache'
    if "VERSION_ID=8" in os_info_data:
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
    if "VERSION_ID=22" in os_info_data:
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
    print("\nrocAL Setup on "+platformInfo+" is unsupported\n")
    print("\nrocAL Setup Supported on: Ubuntu 20/22, RedHat 8/9, & SLES 15\n")
    exit()

# rocAL Setup
print("\nrocAL Setup on: "+platformInfo+"\n")

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' '+osUpdate))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# Delete previous install
if os.path.exists(deps_dir) and reinstall == 'ON':
    ERROR_CHECK(os.system(sudoValidate))
    ERROR_CHECK(os.system('sudo rm -rf '+deps_dir))
    print("\nrocAL Setup: Removing Previous Install -- "+deps_dir+"\n")

# Core package dependencies
libpkgConfig = "pkg-config"
if "centos" in os_info_data and "VERSION_ID=7" in os_info_data:
    libpkgConfig = "pkgconfig"
commonPackages = [
    'cmake',
    'git',
    'wget',
    'unzip',
    str(libpkgConfig)
]

rocmDebianPackages = [
    'half',
    'mivisionx-dev'
]

rocmRPMPackages = [
    'half',
    'mivisionx-devel'
]

rocdecodeDebianPackages = [
    'rocdecode-dev'
]

rocdecodeRPMPackages = [
    'rocdecode-devel'
]

rocJpegDebianPackages = [
    'rocjpeg-dev'
]

rocJpegRPMPackages = [
    'rocjpeg-devel'
]

opencvRPMPackages = [
    'gtk2-devel',
    'libjpeg-devel',
    'libpng-devel',
    'libtiff-devel',
    'libavc1394',
    'unzip'
]

coreDebianPackages = [
    'nasm',
    'yasm',
    'liblmdb-dev',
    #'rapidjson-dev',
    'libsndfile1-dev', # for audio features
    'python3-dev',
    'python3-pip',
    'python3-protobuf',
    'libprotobuf-dev',
    'libprotoc-dev',
    'protobuf-compiler',
    'libturbojpeg0-dev'
]

libsndFile = "libsndfile-devel"
libPythonProto = "python3-protobuf"
libProtoCompiler = "protobuf-compiler"
libTurboJPEG = "turbojpeg-devel"
if "SLES" in os_info_data:
    libProtoCompiler = "libprotobuf-c-devel"
    libsndFile = "cmake" # TBD - libsndfile-devel  fails to install in SLES
    libTurboJPEG = "cmake" # TBD libturbojpeg0 dev/devel package unavailable in SLES
coreRPMPackages = [
    'nasm',
    'yasm',
    'lmdb-devel',
    'jsoncpp-devel',
    #'rapidjson-devel',
    str(libsndFile), # for audio features
    'python3-devel',
    'python3-pip',
    str(libPythonProto),
    'protobuf-devel',
    str(libProtoCompiler),
    str(libTurboJPEG)
]

pip3Packages = [
    'pytest==7.0.0',
    'wheel==0.37.0'
]

debianOptionalPackages = [
    'ffmpeg',
    'libavcodec-dev',
    'libavformat-dev',
    'libavutil-dev',
    'libswscale-dev',
    'libopencv-dev',
]

# Install
ERROR_CHECK(os.system(sudoValidate))
if os.path.exists(deps_dir):
    print("\nrocAL Setup: install found -- "+deps_dir)
    print("\nrocAL Setup: use option --reinstall ON to reinstall all dependencies")
    print("\nrocAL Dependencies Installed with rocAL-setup.py on "+platformInfo+"\n")
    exit(0)
# Clean Install
else:
    print("\nrocAL Dependencies Installation with rocAL-setup.py V-"+__version__+"\n")
    ERROR_CHECK(os.system('mkdir '+deps_dir))
    # Create Build folder
    ERROR_CHECK(os.system('(cd '+deps_dir+'; mkdir build )'))
    # update
    ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +' '+linuxSystemInstall_check+' '+osUpdate))
    # common packages
    for i in range(len(commonPackages)):
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ commonPackages[i]))
    if "redhat-7" in platformInfo:
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install cmake3'))

    # ROCm Packages
    if "Ubuntu" in platformInfo:
        for i in range(len(rocmDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rocmDebianPackages[i]))
    else:
        for i in range(len(rocmRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rocmRPMPackages[i]))
            
    # rocDecode
    if "Ubuntu" in platformInfo:
        for i in range(len(rocdecodeDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rocdecodeDebianPackages[i]))
    elif "redhat" in platformInfo and "SLES" in platformInfo:
        for i in range(len(rocdecodeRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rocdecodeRPMPackages[i]))

    # rocJpeg
    if "Ubuntu" in platformInfo:
        for i in range(len(rocJpegDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rocJpegDebianPackages[i]))
    elif "redhat" in platformInfo and "SLES" in platformInfo:
        for i in range(len(rocJpegRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rocJpegRPMPackages[i]))

    ERROR_CHECK(os.system(sudoValidate))
    # rocAL Core Packages
    if "Ubuntu" in platformInfo:
        for i in range(len(coreDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ coreDebianPackages[i]))
    else:
        for i in range(len(coreRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ coreRPMPackages[i]))
    
    # turbo-JPEG - https://github.com/libjpeg-turbo/libjpeg-turbo.git -- 3.0.2
    if "SLES" in platformInfo:
        turboJpegVersion = '3.0.2'
        ERROR_CHECK(os.system(
                    '(cd '+deps_dir+'; git clone -b '+turboJpegVersion+' https://github.com/libjpeg-turbo/libjpeg-turbo.git )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/libjpeg-turbo; mkdir build; cd build; '+linuxCMake +
                    ' -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib -DWITH_JPEG8=TRUE ..; make -j$(nproc); sudo make install )'))

    #pip3 packages
    for i in range(len(pip3Packages)):
        ERROR_CHECK(os.system('pip3 install '+ pip3Packages[i]))

    # PyBind11 - https://github.com/pybind/pybind11 -- v2.11.1
    pybind11Version = 'v2.11.1'
    ERROR_CHECK(os.system('(cd '+deps_dir+'; git clone -b '+pybind11Version+' https://github.com/pybind/pybind11; cd pybind11; mkdir build; cd build; ' +
            linuxCMake+' -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON ../; make -j$(nproc); sudo make install)'))
    
    # dlpack - https://github.com/dmlc/dlpack
    if "Ubuntu-22" in platformInfo or "Ubuntu-24" in platformInfo:
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install libdlpack-dev'))
    elif "SLES" in platformInfo:
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install dlpack-devel'))
    elif "redhat" in platformInfo:
        # no package avialable -- using prebuilt RPM
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://rpmfind.net/linux/opensuse/distribution/leap/16.0/repo/oss/x86_64/dlpack-devel-0.8-lp160.1.1.x86_64.rpm'))


    # RapidJSON - Source TBD: Package install of RapidJSON has compile issues - https://github.com/Tencent/rapidjson.git -- master
    os.system('(cd '+deps_dir+'; git clone https://github.com/Tencent/rapidjson.git; cd rapidjson; mkdir build; cd build; ' +	
            linuxCMake+' ../; make -j$(nproc); sudo make install)')

    # libtar - https://repo.or.cz/libtar.git ; version - v1.2.20
    libtar_version = 'v1.2.20'
    ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install autoconf libtool'))
    ERROR_CHECK(os.system(
        '(cd '+deps_dir+'; git clone -b '+ libtar_version+' https://repo.or.cz/libtar.git )'))
    ERROR_CHECK(os.system('(cd '+deps_dir+'/libtar; '+
            ' autoreconf --force --install; CFLAGS="-fPIC" ./configure; make -j$(nproc); sudo make install )'))

    # Optional Deps
    if "Ubuntu" in platformInfo:
        for i in range(len(debianOptionalPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                                ' '+linuxSystemInstall_check+' install -y '+ debianOptionalPackages[i]))
    else:
        # Install ffmpeg
        if ffmpegInstall == 'ON':
            if "redhat-7" in platformInfo:
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install epel-release'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' localinstall --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg ffmpeg-devel'))
            elif "redhat-8" in platformInfo:
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/SDL2-2.0.10-2.el8.x86_64.rpm'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg ffmpeg-devel'))
            elif "redhat-9" in platformInfo:
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://dl.fedoraproject.org/pub/epel/epel-next-release-latest-9.noarch.rpm'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %rhel).noarch.rpm'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg ffmpeg-devel'))
            elif "SLES" in platformInfo:
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg-4 ffmpeg-4-libavcodec-devel ffmpeg-4-libavformat-devel'))
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg-4-libavutil-devel ffmpeg-4-libswscale-devel'))

        # Install OpenCV -- TBD cleanup
        opencvVersion = '4.6.0'
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build; mkdir OpenCV )'))
        # Install pre-reqs
        ERROR_CHECK(os.system(sudoValidate))
        if "redhat" in platformInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' groupinstall \'Development Tools\''))
        for i in range(len(opencvRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ opencvRPMPackages[i]))
        # OpenCV 4.6.0 
        # Get Installation Source
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

print("\nrocAL Dependencies Installed with rocAL-setup.py V-"+__version__+" on "+platformInfo+"\n")
