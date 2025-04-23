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
import platform
import os

def is_conda_installed():
    try:
        # get Conda version to check if Conda is installed
        subprocess.run(['conda', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        # Conda is not installed
        return False
    except FileNotFoundError:
        # Conda is not installed
        return False

def get_python_version_platform():
    return platform.python_version()

def get_conda_python_version():
    try:
        # Execute the "conda list" command and capture the output
        result = subprocess.run(['conda', 'list', 'python'], capture_output=True, text=True, check=True)       
        # Split the output into lines and find the line containing the Python package
        lines = result.stdout.split('\n')
        for line in lines:
            if 'python' in line and not line.startswith('#'):
                parts = line.split()
                if parts[0] == 'python':  # Ensure that it's the Python package
                    return parts[1]  # The version should be in the second column
    except subprocess.CalledProcessError as e:
        print("Failed to run conda command:", e)
    except Exception as e:
        print("An error occurred:", e)
    return "Python version not found"

if (is_conda_installed() ):
    conda_python_version = get_conda_python_version()
    sys_python_version = get_python_version_platform()

    # for user
    print("\nPreparing for installation, please wait.\n")

    # temp: make both python versions the same
    subprocess.check_call(['conda','install', '-y', 'python='+sys_python_version], stdout=open(os.devnull, 'wb'))

    # install requirements
    os.system('python rocPyDecode-requirements.py')

    # UN-INSTALL older rocPyDecode
    subprocess.check_call(['rm', '-rf', 'build'], stdout=open(os.devnull, 'wb'))
    subprocess.check_call(['rm', '-rf', 'rocPyDecode.egg-info'], stdout=open(os.devnull, 'wb'))
    subprocess.check_call(['python', '-m', 'pip', 'uninstall', 'rocPyDecode', '-y'], stdout=open(os.devnull, 'wb'))

    # Install rocPyDecode
    subprocess.check_call(['python', '-m', 'pip', 'install', '.'])

    # return the conda py version back
    subprocess.check_call(['conda','install', '-y', 'python='+conda_python_version], stdout=open(os.devnull, 'wb'))
else:
    # install requirements
    os.system('python rocPyDecode-requirements.py')
    # install for all other dockers
    os.system('python3 setup.py install')

# how to test
print("\nTo test rocPyDecode run the following command:\n","python3 samples/videodecode.py -i /opt/rocm/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4","\n")