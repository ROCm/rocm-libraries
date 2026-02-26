#!/usr/bin/env bash
set -euo pipefail

# Configure and compile all discovered test_*conv* targets.
#
# Usage:
#   ./script/build_all_test_targets.sh [build_dir] [jobs]
#
# Defaults:
#   build_dir: <repo_root>/build
#   jobs:      0 (let backend default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-$(cd "${SRC_DIR}/../.." && pwd)/build}"
JOBS="${2:-0}"

echo "[INFO] Source dir: ${SRC_DIR}"
echo "[INFO] Build dir : ${BUILD_DIR}"
echo "[INFO] Jobs      : ${JOBS}"

mkdir -p "${BUILD_DIR}"

echo "[INFO] Configuring CMake..."
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}"

echo "[INFO] Discovering test_*conv* targets..."
mapfile -t TARGETS < <(
    cmake --build "${BUILD_DIR}" --target help 2>/dev/null \
        | sed -n 's/^[.][.][.] //p' \
        | awk '{print $1}' \
        | sed 's/:$//' \
        | grep '^test_.*conv' \
        | sort -u
)

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    echo "[WARN] Could not discover targets via 'cmake --build --target help'."
    echo "[WARN] Falling back to source-file-based target list (conv only)."
    mapfile -t TARGETS < <(
        {
            for f in "${SRC_DIR}/test"/*.cpp "${SRC_DIR}/test/gtest"/*.cpp; do
                [[ -f "${f}" ]] || continue
                b="$(basename "${f}" .cpp)"
                if [[ "test_${b}" == *conv* ]]; then
                    echo "test_${b}"
                fi
            done
        } | sort -u
    )
fi

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    echo "[ERROR] No test_*conv* targets found."
    exit 2
fi

echo "[INFO] Found ${#TARGETS[@]} test_*conv* targets."

ok=0
fail=0
FAILED_TARGETS=()

for t in "${TARGETS[@]}"; do
    echo "[INFO] Building ${t} ..."
    if [[ "${JOBS}" -gt 0 ]]; then
        if cmake --build "${BUILD_DIR}" --target "${t}" --parallel "${JOBS}"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
            FAILED_TARGETS+=("${t}")
        fi
    else
        if cmake --build "${BUILD_DIR}" --target "${t}"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
            FAILED_TARGETS+=("${t}")
        fi
    fi
done

echo
echo "[SUMMARY] Built successfully: ${ok}"
echo "[SUMMARY] Failed          : ${fail}"

if [[ "${fail}" -gt 0 ]]; then
    echo "[SUMMARY] Failed targets:"
    for t in "${FAILED_TARGETS[@]}"; do
        echo "  - ${t}"
    done
    exit 1
fi

echo "[INFO] All discovered test_*conv* targets compiled successfully."
