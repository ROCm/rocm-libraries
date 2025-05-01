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

