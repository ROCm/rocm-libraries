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
source `find . -name "utils.sh"`

# Variables
tags=""
branch="develop"
arch="gfx900"
compiler="amdclang++"
jobs="16,32"

base_image_envvar="REGISTRY_IMAGE"

usage() {
    echo "Run grid-based profiling analysis for TensileCreateLibrary under variable inputs"
    echo ""
    echo "Usage: $0 --tags=<tags> [--jobs=<jobs>] [--branch=<branch>] [--arch=<arch>] [--compiler=<compiler>]"
    echo ""
    echo "Parameters:"
    echo "  --tags: Docker tags to to launch run from [required]"
    echo "  --jobs: Number of concurrent processes to use [default: $jobs]"
    echo "  --arch: Target Gfx architecture(s) [default: $arch]"
    echo "  --branch: The target branch [default: $branch]"
    echo "  --compiler: HIP-enabled compiler (must be in PATH) [default: $compiler]"
    echo ""
    echo "Environment variables:"
    echo "  REGISTRY_IMAGE: Base Docker image to use [required]"
    echo ""
    echo "Example:"
    echo "  export REGISTRY_IMAGE='rocm/rocm-terminal'"
    echo "  $0 --tags='12345-ubuntu22.04' --jobs='16,32' --branch='develop' --arch='gfx90a'"
    echo ""
    echo "Dependencies:"
    echo "  docker: Docker is implicitly called and may install images"
}

check_installed_docker_images() {
    local base_img=$1
    local tag=$2
    if ! docker images $base_img | grep "$tag" | grep -q .; then
        echoerr "+ Error: Docker image $base_img:$tag is not installed. Please pull the image and retry."
        exit 1
    fi
}

find_tensile() {
  local query="*/Tensile/setup.py"
  local full_image=$1
  # If several Tensile projects are found, use the first one
  docker run \
    --rm \
    --volume="$HOME:/mnt/host" \
    "$full_image" bash -c "find /mnt/host -path $query -exec dirname {} \; 2>&1 | head -n1"
}

find_logic() {
  local query="*/Tensile/Logic/asm_full"
  local full_image=$1
  docker run \
    --rm \
    --volume="$HOME:/mnt/host" \
    "$full_image" bash -c "find /mnt/host -path $query"
}

run_suite() {
    for tag in "${tags[@]}"; do
        check_installed_docker_images $base_image $tag
    done

    echo "+ Running profiling suite..."
    for tag in "${tags[@]}"; do
        local full_image="$base_image:$tag"
        echo "+ In container: $full_image"

        local tensile_path=$(find_tensile $full_image)
        local logic_path=$(find_logic $full_image)
        echo "    using Tensile: $tensile_path"
        echo "    using logic files: $logic_path"
        for n in "${jobs[@]}"; do
            docker run --rm \
                       --security-opt seccomp=unconfined \
                       --device=/dev/kfd \
                       --device=/dev/dri \
                       --group-add=video \
                       --volume="$HOME:/mnt/host" "$full_image" \
                       /bin/bash -c "$tensile_path/scripts/run-tcl.sh \
                           --tensile-path=$tensile_path \
                           --logic-path=$logic_path \
                           --jobs=$n \
                           --arch=$arch \
                           --compiler=$compiler"
        done
    done
}

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --tags=*) tags="${arg#*=}" ;;
        --jobs=*) jobs="${arg#*=}" ;;
        --arch=*) arch="${arg#*=}" ;;
        --branch=*) branch="${arg#*=}" ;;
        --compiler=*) compiler="${arg#*=}" ;;
        --help) usage; exit 0 ;;
        *) echo "Invalid option: $arg"; usage; exit 1 ;;
    esac
done

assert_envvar_exists $base_image_envvar
base_image=${!base_image_envvar}

declare -a tags=($(convert_comma_separated_to_array "$tags"))
declare -a jobs=($(convert_comma_separated_to_array "$jobs"))

# Check if all parameters are provided
if [ -z "$tags" ] || \
   [ -z "$jobs" ]     || \
   [ -z "$arch" ]     || \
   [ -z "$branch" ]   || \
   [ -z "$compiler" ]; then
    usage
    exit 1
fi

echo "+ Profiling..."
echo "    tags:     ${tags[@]}"
echo "    jobs:     ${jobs[@]}"
echo "    arch:     $arch"
echo "    branch:   $branch"
echo "    compiler: $compiler"

run_suite
