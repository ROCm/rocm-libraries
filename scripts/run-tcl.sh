################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#!/bin/bash

# Variables
tensile_path=""
logic_path=""
jobs=""
arch="gfx900"
compiler="amdclang++"

# Constants
language="HIP"
build_dir="build-tcl-$(date +'%Y-%m-%dT%H-%M-%S')"
log_file="tcl-profile-$(date +'%Y-%m-%dT%H-%M-%S').log"

usage() {
    echo "Run TensileCreateLibrary with timestamped log and build directory"
    echo ""
    echo "Usage: $0 --tensile-path=<tensile-path> --logic-path=<logic-path> --jobs=<jobs> [--arch=<arch>] [--compiler=<compiler>]"
    echo ""
    echo "Parameters:"
    echo "  --tensile-path: Path to root directory of Tensile"
    echo "  --logic-path: Path to directory containing logic files"
    echo "  --jobs: Number of concurrent processes to use"
    echo "  --arch: Target Gfx architecture(s) [default: $arch]"
    echo "  --compiler: HIP-enabled compiler (must be in PATH) [default: $compiler]"
    echo ""
    echo "Example:"
    echo "  $0 --tensile-path=/mnt/host/Tensile --logic-path=/mnt/host/Logic --jobs=16"
}

main() {
  cd $tensile_path
  echo "+ Writing logs to: `pwd`/$log_file"
  echo "+ Building output to: `pwd`/$build_dir"
  export TENSILE_PROFILE=ON
  export PYTHONPATH="$tensile_path"
  $tensile_path/Tensile/bin/TensileCreateLibrary $logic_path $build_dir $language \
     --merge-files \
     --separate-architecture \
     --lazy-library-loading \
     --no-short-file-names \
     --code-object-version=default \
     --cxx-compiler=$compiler \
     --jobs=$jobs \
     --library-format=msgpack \
     --architecture=$arch | tee "$tensile_path/$log_file" 2>&1
}

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --tensile-path=*) tensile_path="${arg#*=}" ;;
        --logic-path=*) logic_path="${arg#*=}" ;;
        --jobs=*) jobs="${arg#*=}" ;;
        --arch=*) arch="${arg#*=}" ;;
        --compiler=*) compiler="${arg#*=}" ;;
        --help) usage; exit 0 ;;
        *) echo "Invalid option: $arg"; usage; exit 1 ;;
    esac
done

# Check if all parameters are provided
if [ -z "$tensile_path" ] || [ -z "$logic_path" ] || [ -x "$jobs" ] || [ -z "$arch" ] || [ -z "$compiler" ]; then
    usage
    exit 1
fi

main
