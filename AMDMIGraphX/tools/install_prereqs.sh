#!/bin/bash

#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#####################################################################################
#
# Build MIGraphX prerequisites for docker container

set -e

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

source /etc/os-release

# hipcc added as workaround for broken hip-dev package
if [[ ("${ID}" == "sles") ]]; then
  zypper -n --gpg-auto-import-keys install -y \
    cmake \
    miopen-hip-devel \
    openmp-extras-devel \
    python3-devel \
    python3-pip \
    rocblas-devel \
    rocm-cmake \
    perl-File-BaseDir \
    libgfortran5 \
    hipblas-devel \
    hipblaslt-devel
else
  # Need pip3 and Python headers to build dependencies
  apt update && apt install -y \
    cmake \
    libnuma-dev \
    miopen-hip-dev \
    openmp-extras \
    python3-dev \
    python3-pip \
    python3-venv \
    rocblas-dev \
    libgfortran5 \
    hipblas-dev \
    hipblaslt-dev \
    hipcc \
    rocm-cmake \
    rocm-llvm-dev \
    libtbb-dev
fi


# Needed for cmake to build various pip packages
pip3 install setuptools wheel

# install rbuild to build dependencies
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz


PREFIX=/usr/local
REQ_FILE_DIR="$(dirname -- "$0")"
if [ "$#" -ge 2 ]; then
  PREFIX=$1
  cd $2
elif [ "$#" -eq 1 ]; then
  PREFIX=$1
fi

echo "Dependencies are installed at $PREFIX"

# Install deps with rbuild
rbuild prepare -d $PREFIX -s develop

if [[ ("${ID}" != "sles") ]]; then
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
pip3 install -r $REQ_FILE_DIR/requirements-py.txt
fi
