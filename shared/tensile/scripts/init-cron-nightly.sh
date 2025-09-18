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
tensile_path="`pwd`"

# Constants
script_to_run="profile-tcl.sh"

usage() {
    echo "Set up cron table for nightly Tensile profiling reports"
    echo ""
    echo "Usage: $0 --tags=<tags> [--tensile-path=<path>]"
    echo ""
    echo "Parameters:"
    echo "  --tags: The target docker image tags [required]"
    echo "  --tensile-path: Path to root directory of Tensile [default: $tensile_path]"
    echo ""
    echo "Example:"
    echo "  $0 --build-id=14354 --tensile-path='path/to/tensile'"
}

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --tags=*) tags="${arg#*=}" ;;
        --tensile-path=*) tensile_path="${arg#*=}" ;;
        --help) usage; exit 0 ;;
        *) echo "Invalid option: $arg"; usage; exit 1 ;;
    esac
done

# Check if all parameters are provided
if [ -z "$tags" ] || [ -z "$tensile_path" ]; then
    usage
    exit 1
fi

if [ ! -e "$tensile_path" ]; then
    echoerr "+ Path to Tensile does not exist"
    echoerr "+ Cannot find: $tensile_path"
    exit 1
fi

if crontab -l | grep -q "$script_to_run"; then
    echoerr "+ Cron job with same command already exists."
    echoerr "+ Clean your crontab manually with 'crontab -e' and rerun."
    echoerr "+ Conflict line:\n+   `crontab -l | grep $script_to_run`"
    exit 1
else
    cron_log="$tensile_path/tcl-profile-$(date +'%Y-%m-%dT%H-%M-%S').log.cron"
    (crontab -l 2>/dev/null; echo "00 22 * * 0-4 $tensile_path/scripts/$script_to_run --tags=$tags | tee $cron_log") | crontab -
    echoinfo "Added cron job:\n  `crontab -l | tail -n 1`"
fi
