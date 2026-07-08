#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set -o pipefail
set -u

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <clang-format> <source-dir> <check|format> <files-per-invocation> <jobs>" >&2
    exit 2
fi

clang_format=$1
source_dir=$2
format_mode=$3
files_per_invocation=$4
jobs=$5

if [[ ${format_mode} == "check" ]]; then
    clang_format_args=(--dry-run --Werror)
elif [[ ${format_mode} == "format" ]]; then
    clang_format_args=(--verbose -i)
else
    echo "FORMAT_MODE must be 'check' or 'format'" >&2
    exit 2
fi

if ! [[ ${files_per_invocation} =~ ^[1-9][0-9]*$ ]]; then
    echo "files-per-invocation must be greater than 0" >&2
    exit 2
fi

if ! [[ ${jobs} =~ ^[0-9]+$ ]]; then
    echo "jobs must be greater than or equal to 0" >&2
    exit 2
fi

if [[ ${jobs} -eq 0 ]]; then
    jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi
if [[ ${jobs} -lt 1 ]]; then
    jobs=1
fi

find "${source_dir}" \
    \( \
        -path "${source_dir}/build" -o \
        -path "${source_dir}/flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/data_objects" \
    \) -prune -o \
    -type f \( \
        -name '*.cpp' -o \
        -name '*.hpp' -o \
        -name '*.c' -o \
        -name '*.h' \
    \) -print0 \
    | sort -z \
    | xargs -0 -r -n "${files_per_invocation}" -P "${jobs}" "${clang_format}" "${clang_format_args[@]}"
