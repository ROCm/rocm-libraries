#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Check if coverage tests failed and report status
# Usage: check_coverage_exit_code.sh <coverage_dir>

coverage_dir="$1"
exit_code_file="${coverage_dir}/test_exit_code.txt"

if [ -f "${exit_code_file}" ]; then
    echo ""
    echo "WARNING: Some tests failed (see above), but coverage report was still generated"
    echo "HTML report: ${coverage_dir}/index.html"
    exit $(cat "${exit_code_file}")
else
    echo "All tests passed"
    exit 0
fi
