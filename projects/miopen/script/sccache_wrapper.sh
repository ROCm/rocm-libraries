#!/bin/bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT


set -e
COMPILERS_HASH_DIR=${COMPILERS_HASH_DIR:-"/tmp/.sccache"}
SCCACHE_EXTRAFILES=${SCCACHE_EXTRAFILES:-"${COMPILERS_HASH_DIR}/rocm_compilers_hash_file"}
SCCACHE_BIN=${SCCACHE_BIN:-"${SCCACHE_INSTALL_LOCATION}/sccache"}
ENFORCE_REDIS="false"
while [ "$1" != "" ];
do
    case $1 in
        --enforce_redis )
            shift; ENFORCE_REDIS="true" ;;
        --no-hipcc )
            shift ;;
        *)
            break ;;
    esac
done
setup_rocm_compilers_hash_file() {
    mkdir -p "$COMPILERS_HASH_DIR"
    # MD5 checksums of clang and clang-offload-bundler cannot be used since they will keep changing
    # if the ROCM_PATH changes, ie; for every mainline build.
    # This is because ROCM_PATH gets encoded into the clang/clang-offload-bundler binaries as part
    # of RPATH.
    # Note: Since compiler version strings include the commit hash, they vary with each build of TheRock.
    # Exclude the commit hash to prevent unnecessary cache invalidation.
    # Omit hipcc and bitcode hashing for now; hipcc is unused, and bitcode hashing can be restored if needed
    CLANG_VERSION="$("${ROCM_PATH}/llvm/bin/clang" --version | head -n 1)"
    CLANG_OFFLOAD_BUNDLER_VERSION="$("${ROCM_PATH}/llvm/bin/clang-offload-bundler" --version | head -n 1)"
    printf '%s: %s\n' 'clang version' "${CLANG_VERSION}" | sed 's/ (.*)//' | tee -a "$SCCACHE_EXTRAFILES"
    printf '%s: %s\n' 'clang-offload-bundler version' "${CLANG_OFFLOAD_BUNDLER_VERSION}" | sed 's/ (.*)//' | tee -a "$SCCACHE_EXTRAFILES"
    echo "sccache-wrapper: compilers hash file set up at ${SCCACHE_EXTRAFILES}"
    cat "$SCCACHE_EXTRAFILES"
}
if [ "${ENFORCE_REDIS}" == "true" ]; then
    if [ -z "${SCCACHE_REDIS}" ]; then
        echo "SCCACHE_REDIS not set. Not wrapping compilers with sccache."
        exit 10
    else
        response=$(redis-cli -u ${SCCACHE_REDIS} ping) || true
        if [ "${response}" != "PONG" ]; then
            echo "Redis server unreachable. Not wrapping compilers with sccache."
            exit 20
        fi
    fi
fi
setup_rocm_compilers_hash_file
$SCCACHE_BIN --version
$SCCACHE_BIN --start-server
