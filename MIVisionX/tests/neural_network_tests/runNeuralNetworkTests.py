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

import argparse
import os
import shutil
import sys
import traceback
import platform
if sys.version_info[0] < 3:
    import commands
else:
    import subprocess
from datetime import datetime
from subprocess import Popen, PIPE

__copyright__ = "Copyright 2018 - 2024, AMD MIVisionX - Neural Net Test Full Report"
__license__ = "MIT"
__version__ = "2.1.0"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"
    
# error check calls
def ERROR_CHECK(call):
    status = call
    if(status != 0):
        print('ERROR_CHECK failed with status:'+str(status))
        traceback.print_stack()
        exit(status)


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


def strip_libtree_addresses(lib_tree):
    return lib_tree


def script_info():
    print("\nMIVisionX runNeuralNetworkTests V-"+__version__+"\n")
    print(
        "--profiler_mode       - NN Profile Mode: optional (default:0 [range:0 - 9])")
    print("    --profiler_mode 0 -- Run All Tests")
    print("    --profiler_mode 1 -- Run caffe2nnir2openvx No Fuse flow")
    print("    --profiler_mode 2 -- Run caffe2nnir2openvx Fuse flow")
    print("    --profiler_mode 3 -- Run caffe2nnir2openvx FP16 flow")
    print("    --profiler_mode 4 -- Run onnx2nnir2openvx No Fuse flow")
    print("    --profiler_mode 5 -- Run onnx2nnir2openvx Fuse flow")
    print("    --profiler_mode 6 -- Run onnx2nnir2openvx FP16 flow")
    print("    --profiler_mode 7 -- Run nnef2nnir2openvx No Fuse flow")
    print("    --profiler_mode 8 -- Run nnef2nnir2openvx Fuse flow")
    print("    --profiler_mode 9 -- Run nnef2nnir2openvx FP16 flow")
    print(
        "--profiler_level      - NN Profile Batch Size in powers of 2: optional (default:7 [range:1 - N])")
    print(
        "--miopen_find         - MIOPEN_FIND_ENFORCE mode: optional (default:1 [range:1 - 5])")


# models to run - add new models `modelname` , c, h, w
caffeModelConfig = [
    ('caffe-mnist', 1, 28, 28)
]

onnxModelConfig = [
    ('onnx-squeezenet', 3, 224, 224)
]

nnefModelConfig = [
    ('nnef-mnist', 1, 28, 28)
]

# REPORT
reportConfig = [
    ('CAFFE no fused OPs', 'caffe2nnir2openvx_noFuse_profile.md'),
    ('CAFFE fused OPs', 'caffe2nnir2openvx_Fuse_profile.md'),
    ('CAFFE fp16', 'caffe2nnir2openvx_FP16_profile.md'),
    ('ONNX no fused OPs', 'onnx2nnir2openvx_noFuse_profile.md'),
    ('ONNX fused Ops', 'onnx2nnir2openvx_Fuse_profile.md'),
    ('ONNX fp16', 'onnx2nnir2openvx_FP16_profile.md'),
    ('NNEF no fused OPs', 'nnef2nnir2openvx_noFuse_profile.md'),
    ('NNEF fused OPs', 'nnef2nnir2openvx_Fuse_profile.md'),
    ('NNEF fp16', 'nnef2nnir2openvx_FP16_profile.md')
]

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--profiler_mode',      type=int, default=0,
                    help='NN Profile Mode - optional (default:0 [range:0 - 9])')
parser.add_argument('--profiler_level',     type=int, default=7,
                    help='NN Profile Batch Size in powers of 2 - optional (default:7 [range:1 - N])')
parser.add_argument('--miopen_find',        type=int, default=1,
                    help='MIOPEN_FIND_ENFORCE mode - optional (default:1 [range:1 - 5])')
parser.add_argument('--test_info',          type=str, default='no',
                    help='Show test info - optional (default:no [options:no/yes])')
parser.add_argument('--backend_type',       type=str, default='HIP',
                    help='Backend type - optional (default:HIP [options:HOST/HIP/OCL])')
parser.add_argument('--install_directory',    type=str, default='/opt/rocm',
                    help='MIVisionX Install Directory - optional')
parser.add_argument('--reinstall', 	type=str, default='ON',
                    help='Remove previous setup and reinstall - optional (default:OFF) [options:ON/OFF]')
args = parser.parse_args()

profileMode = args.profiler_mode
profileLevel = args.profiler_level
miopenFind = args.miopen_find
testInfo = args.test_info
backendType = args.backend_type
installDir = args.install_directory
reinstall = args.reinstall.upper()

platfromInfo = platform.platform()

returnStatus = 0

# check arguments
if not 0 <= profileMode <= 9:
    print(
        "\nERROR: NN Profile Mode not in range - [0 - 9]\n")
    exit()
if not 1 <= profileLevel <= 10:
    print(
        "\nERROR: NN Profile Batch Size in powers of 2 not in range - [1 - 10]\n")
    exit()
if not 1 <= miopenFind <= 5:
    print(
        "\nERROR: MIOPEN_FIND_ENFORCE not in range - [1 - 5]\n")
    exit()
if testInfo not in ('no', 'yes'):
    print("ERROR: Show test info options supported - [no or yes]")
    script_info()
    exit()

if testInfo == 'yes':
    script_info()
    exit()

# check backend
if backendType not in ('HOST', 'HIP', 'OCL'):
    print("ERROR: Backends supported - HOST or HIP or OCL]")
    exit()

if backendType == 'HOST':
    print("ERROR: HOST Backend currently NOT Supported [Supported: OCL/HIP]")
    exit()

if reinstall not in ('OFF', 'ON'):
    print(
        "ERROR: Re-Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit()

# check install
runVX_exe = installDir+'/bin/runvx'
if (os.path.isfile(runVX_exe)):
    print("STATUS: MIVisionX Install Path Found - "+installDir)
else:
    print("\nERROR: MIVisionX Install Path Not Found\n")
    exit()

print("\nMIVisionX runNeuralNetworkTests V-"+__version__+"\n")

# check for Scripts
scriptPath = os.path.dirname(os.path.realpath(__file__))
modelCompilerDir = os.path.expanduser(
    installDir+'/libexec/mivisionx/model_compiler/python')
pythonScript = modelCompilerDir+'/caffe_to_nnir.py'
modelCompilerScript = os.path.abspath(pythonScript)
if (os.path.isfile(modelCompilerScript)):
    print("\nMIVisionX Neural Net Tests on "+platfromInfo+"\n")
    print("STATUS: Model Compiler Scripts Used from - "+modelCompilerDir+"\n")
else:
    print("ERROR: Model Compiler Scripts Not Found at - "+modelCompilerDir)
    print("ERROR: MIVisionX Not Installed, install MIVisionX and rerun")
    exit()

# Install Model Compiler Deps
modelCompilerDeps = os.path.expanduser('~/.mivisionx-model-compiler-deps')
linuxCMake = 'cmake'

# check os version
os_info_data = 'NOT Supported'
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as os_file:
        os_info_data = os_file.read().replace('\n', ' ')
        os_info_data = os_info_data.replace('"', '')
        
# Linux Packages
inferenceDebianPackages = [
    'inxi',
    'python3-dev',
    'python3-pip'
]

inferenceRPMPackages = [
    'inxi',
    'python3-devel',
    'python3-pip'
]

# Debian based
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

# RPM based
pipONNXversion = "onnx==1.11.0"
pipNNEFversion = "nnef==1.0.7"
if "VERSION_ID=7" in os_info_data or "VERSION_ID=8" in os_info_data:
    pipNumpyVersion = "numpy==1.19.5"
    pipNNEFversion = "protobuf==3.12.4" # TBD: NO NNEF Package for SLES
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

# Delete previous install
if os.path.exists(modelCompilerDeps) and reinstall == 'ON':
    os.system('sudo -v')
    os.system('sudo rm -rf '+modelCompilerDeps)
    print("\nMIVisionX runNeuralNetworkTests: Removing Previous Install -- " +modelCompilerDeps+"\n")

if not os.path.exists(modelCompilerDeps):
    print("STATUS: Model Compiler Deps Install - "+modelCompilerDeps+"\n")
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
    
    # setup for Linux
    linuxSystemInstall = ''
    linuxCMake = 'cmake'
    linuxSystemInstall_check = ''
    linuxFlag = ''
    sudoValidate = 'sudo -v'
    if "centos" in os_info_data or "redhat" in os_info_data:
        linuxSystemInstall = 'yum -y'
        linuxSystemInstall_check = '--nogpgcheck'
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
        platfromInfo = platfromInfo+'-SLES'
    elif "Mariner" in os_info_data:
        linuxSystemInstall = 'tdnf -y'
        linuxSystemInstall_check = '--nogpgcheck'
        platfromInfo = platfromInfo+'-Mariner'
    else:
        print("\nMIVisionX runNeuralNetworkTests.py on "+platfromInfo+" is unsupported\n")
        print("\nMIVisionX Setup Supported on: Ubuntu 20/22; CentOS 7/8; RedHat 8/9; & SLES 15 SP5\n")
        exit()

    if userName == 'root':
        ERROR_CHECK(os.system(linuxSystemInstall+' update'))
        ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

    os.makedirs(modelCompilerDeps)
    os.system('sudo -v')
    if "Ubuntu" in platfromInfo:
        for i in range(len(inferenceDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ inferenceDebianPackages[i]))
        for i in range(len(pip3InferencePackagesUbuntu)):
                            ERROR_CHECK(os.system('pip3 install '+ pip3InferencePackagesUbuntu[i]))
    else:
        for i in range(len(inferenceRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ inferenceRPMPackages[i]))
        for i in range(len(pip3InferencePackagesRPM)):
                            ERROR_CHECK(os.system('pip3 install '+ pip3InferencePackagesRPM[i]))
        if "SLES" in platfromInfo or "Mariner" in platfromInfo or "redhat-8" in platfromInfo:
            ERROR_CHECK(os.system('mkdir -p '+modelCompilerDeps+'/nnef-deps'))
            ERROR_CHECK(os.system(
                '(cd '+modelCompilerDeps+'/nnef-deps; git clone -b nnef-v1.0.0 https://github.com/KhronosGroup/NNEF-Tools.git)'))
            ERROR_CHECK(os.system(
                '(cd '+modelCompilerDeps+'/nnef-deps/NNEF-Tools/parser/cpp; mkdir -p build && cd build; '+linuxCMake+' ..; make -j$(nproc); sudo make install)'))
            ERROR_CHECK(os.system(
                '(cd '+modelCompilerDeps+'/nnef-deps/NNEF-Tools/parser/python; sudo python3 setup.py install)'))
else:
    print("STATUS: Model Compiler Deps Pre-Installed - "+modelCompilerDeps+"\n")
    if "centos-7" in platfromInfo:
        linuxCMake = 'cmake3'

currentWorkingDirectory = os.getcwd()

# Create working directory
outputDirectory = currentWorkingDirectory+'/vx_nn_test'
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
else:
    shutil.rmtree(outputDirectory)
    os.makedirs(outputDirectory)

# run caffe2nnir2openvx no fuse flow
if profileMode == 0 or profileMode == 1:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/caffeNoFuse'
    os.makedirs(outputDirectory)
    for i in range(len(caffeModelConfig)):
        modelName, channel, height, width = caffeModelConfig[i]
        print("\n caffe2nnir2openvx with NO FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/caffe_to_nnir.py '+scriptPath+'/models/' +
                    modelName+'/model.caffemodel . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --fuse-ops 0 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/caffe_no_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/caffe_no_fuse_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_no_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_noFuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_no_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_noFuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/caffe2nnir2openvx_noFuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_no_fuse_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_noFuse_profile.md'''
    os.system(runAwk_md)

# run caffe2nnir2openvx with fuse flow
if profileMode == 0 or profileMode == 2:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/caffeFuse'
    os.makedirs(outputDirectory)
    for i in range(len(caffeModelConfig)):
        modelName, channel, height, width = caffeModelConfig[i]
        print("\n caffe2nnir2openvx with FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/caffe_to_nnir.py '+scriptPath+'/models/' +
                    modelName+'/model.caffemodel . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --fuse-ops 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/caffe_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/caffe_fuse_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_Fuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_Fuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/caffe2nnir2openvx_Fuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_fuse_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_Fuse_profile.md'''
    os.system(runAwk_md)

# run caffe2nnir2openvx with fp16 flow
if profileMode == 0 or profileMode == 3:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/caffeFP16'
    os.makedirs(outputDirectory)
    for i in range(len(caffeModelConfig)):
        modelName, channel, height, width = caffeModelConfig[i]
        print("\n caffe2nnir2openvx FP16 -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/caffe_to_nnir.py '+scriptPath+'/models/' +
                    modelName+'/model.caffemodel . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --convert-fp16 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/caffe_fp16_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/caffe_fp16_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_fp16_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_FP16_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_fp16_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_FP16_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/caffe2nnir2openvx_FP16_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe_fp16_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/caffe2nnir2openvx_FP16_profile.md'''
    os.system(runAwk_md)

# run onnx2nnir2openvx no fuse flow
if profileMode == 0 or profileMode == 4:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/onnxNoFuse'
    os.makedirs(outputDirectory)
    for i in range(len(onnxModelConfig)):
        modelName, channel, height, width = onnxModelConfig[i]
        print("\n onnx2nnir2openvx with NO FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/onnx_to_nnir.py '+scriptPath+'/models/' +
                    modelName+'/model.onnx . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --fuse-ops 0 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/onnx_no_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/onnx_no_fuse_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_no_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_noFuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_no_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_noFuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/onnx2nnir2openvx_noFuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_no_fuse_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_noFuse_profile.md'''
    os.system(runAwk_md)

# run onnx2nnir2openvx with fuse flow
if profileMode == 0 or profileMode == 5:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/onnxFuse'
    os.makedirs(outputDirectory)
    for i in range(len(onnxModelConfig)):
        modelName, channel, height, width = onnxModelConfig[i]
        print("\n onnx2nnir2openvx with FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/onnx_to_nnir.py '+scriptPath+'/models/' +
                    modelName+'/model.onnx . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --fuse-ops 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/onnx_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/onnx_fuse_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_Fuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_Fuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/onnx2nnir2openvx_Fuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_fuse_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_Fuse_profile.md'''
    os.system(runAwk_md)

# run onnx2nnir2openvx with fp16 flow
if profileMode == 0 or profileMode == 6:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/onnxFP16'
    os.makedirs(outputDirectory)
    for i in range(len(onnxModelConfig)):
        modelName, channel, height, width = onnxModelConfig[i]
        print("\n onnx2nnir2openvx FP16 -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/onnx_to_nnir.py '+scriptPath+'/models/' +
                    modelName+'/model.onnx . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --convert-fp16 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/onnx_fp16_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/onnx_fp16_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_fp16_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_FP16_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_fp16_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_FP16_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/onnx2nnir2openvx_FP16_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx_fp16_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/onnx2nnir2openvx_FP16_profile.md'''
    os.system(runAwk_md)

# run nnef2nnir2openvx no fuse flow
if profileMode == 0 or profileMode == 7:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/nnefNoFuse'
    os.makedirs(outputDirectory)
    for i in range(len(nnefModelConfig)):
        modelName, channel, height, width = nnefModelConfig[i]
        print("\n nnef2nnir2openvx with No FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/nnef_to_nnir.py '+scriptPath+'/models/' +
                    modelName+' . )')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --fuse-ops 0 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/nnef_no_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/nnef_no_fuse_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_no_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_noFuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_no_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_noFuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/nnef2nnir2openvx_noFuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_no_fuse_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_noFuse_profile.md'''
    os.system(runAwk_md)

# run nnef2nnir2openvx fuse flow
if profileMode == 0 or profileMode == 8:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/nnefFuse'
    os.makedirs(outputDirectory)
    for i in range(len(nnefModelConfig)):
        modelName, channel, height, width = nnefModelConfig[i]
        print("\n nnef2nnir2openvx with FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/nnef_to_nnir.py '+scriptPath+'/models/' +
                    modelName+' . )')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --fuse-ops 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/nnef_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/nnef_fuse_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_Fuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_fuse_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_Fuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/nnef2nnir2openvx_Fuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_fuse_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_Fuse_profile.md'''
    os.system(runAwk_md)

# run nnef2nnir2openvx FP16 flow
if profileMode == 0 or profileMode == 9:
    outputDirectory = currentWorkingDirectory+'/vx_nn_test/nnefFP16'
    os.makedirs(outputDirectory)
    for i in range(len(nnefModelConfig)):
        modelName, channel, height, width = nnefModelConfig[i]
        print("\n nnef2nnir2openvx with FP16 Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                    '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/nnef_to_nnir.py '+scriptPath+'/models/' +
                    modelName+' . )')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_update.py --convert-fp16 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                    modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                    currentWorkingDirectory+'/vx_nn_test/nnef_fp16_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                    ' ./anntest weights.bin | tee -a '+currentWorkingDirectory+'/vx_nn_test/nnef_fp16_output.log)')
            annTestResults = shell(
                '(cd '+modelBuildDir+x+'; ./anntest weights.bin)')
            annTestResults = annTestResults.decode()
            if annTestResults.find("successful") == -1:
                returnStatus = -1

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_fp16_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_FP16_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_fp16_output.log > ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_FP16_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        currentWorkingDirectory+'/vx_nn_test/nnef2nnir2openvx_FP16_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef_fp16_output.log | tee -a ''' + \
        currentWorkingDirectory+'''/vx_nn_test/nnef2nnir2openvx_FP16_profile.md'''
    os.system(runAwk_md)

# get system data
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
    print("\nMIVisionX Neural Network Test on "+platform_name+" is unsupported")
    print("MIVisionX Neural Network Test Supported on: Ubuntu 20/22; CentOS 7/8; RedHat 8/9; & SLES 15 SP3")
    print("\nMIVisionX Neural Network Test on "+platform_name+" is unreliable")

platform_name_fq = shell('hostname --all-fqdns')
platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

file_dtstr = datetime.now().strftime("%Y%m%d")
reportFilename = 'inference_report_%s_%s_%s.md' % (
    backendType, platform_name, file_dtstr)
report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
sys_info = shell('inxi -c0 -S')

cpu_info = shell('inxi -c0 -C')
cpu_info = cpu_info.rstrip()  # strip out clock speeds

gpu_info = shell('inxi -c0 -G')
gpu_info = gpu_info.rstrip()  # strip out X info

memory_info = shell('inxi -c 0 -m')
board_info = shell('inxi -c0 -M')

lib_tree = shell('ldd -v '+installDir+'/lib/libvx_nn.so')
lib_tree = strip_libtree_addresses(lib_tree)

vbios = shell('(cd /opt/rocm/bin/; ./rocm-smi -v)')

rocmInfo = shell('(cd /opt/rocm/bin/; ./rocm-smi -a)')

rocm_packages = shell('dpkg-query -W | grep rocm')
rocm_packages = rocm_packages.splitlines()

# Write Report
with open(reportFilename, 'w') as f:
    f.write("MIVisionX - ML Inference Report\n")
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

    f.write("\n\nBenchmark Report\n")
    f.write("--------\n")
    f.write("\n")

    if profileMode == 0:
        for i in range(len(reportConfig)):
            modelType, reportFile = reportConfig[i]
            f.write("\n### MODEL FORMAT: %s\n" % modelType)
            with open(currentWorkingDirectory+'/vx_nn_test/'+reportFile) as benchmarkFile:
                for line in benchmarkFile:
                    f.write("%s" % line)
    else:
        modelType, reportFile = reportConfig[profileMode - 1]
        f.write("\n### MODEL FORMAT: %s\n" % modelType)
        with open(currentWorkingDirectory+'/vx_nn_test/'+reportFile) as benchmarkFile:
            for line in benchmarkFile:
                f.write("%s" % line)

    f.write("\n")

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

    f.write("Dynamic Libraries Report\n")
    f.write("-----------------\n")
    f.write("\n")
    write_formatted(lib_tree, f)
    f.write("\n")

    f.write("\n\n---\n**Copyright AMD ROCm MIVisionX 2018 - 2024 -- runNeuralNetworkTests.py V-"+__version__+"**\n")

# report file
reportFileDir = os.path.abspath(reportFilename)
print("\nSTATUS: Output Report File - "+reportFileDir)

print("\nrunNeuralNetworkTests.py completed - V:"+__version__+"\n")

exit(returnStatus)
