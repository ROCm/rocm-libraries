#!/bin/bash

################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# This script needs to be edited to bump version for new release.
# Version will be bumped in Tensile/__init__.py and in .yaml files

OLD_VERSION="4.32.1"
NEW_VERSION="4.33.0"

OLD_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.7.2"
NEW_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.8.0"

sed -i "s/${OLD_VERSION}/${NEW_VERSION}/g" Tensile/Common/Common.py
sed -i "s/${OLD_VERSION}/${NEW_VERSION}/g" HostLibraryTests/CMakeLists.txt

echo "The version number also needs to be fixed in Tensile/cmake/TensileConfigVersion.cmake ."

#only update when there is a major version change
#for FILE in Tensile/Configs/*yaml
#do
#  sed -i "s/${OLD_MINIMUM_REQUIRED_VERSION}/${NEW_MINIMUM_REQUIRED_VERSION}/" $FILE
#done
