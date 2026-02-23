# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Backwards compatibility for HIPBLASLT_ENABLE_LLVM → HIPBLASLT_ENABLE_YAML
if(DEFINED HIPBLASLT_ENABLE_LLVM)
    message(DEPRECATION
        "HIPBLASLT_ENABLE_LLVM is deprecated. Use HIPBLASLT_ENABLE_YAML instead.\n"
        "  Old: -DHIPBLASLT_ENABLE_LLVM=${HIPBLASLT_ENABLE_LLVM}\n"
        "  New: -DHIPBLASLT_ENABLE_YAML=${HIPBLASLT_ENABLE_LLVM}")
    if(NOT DEFINED HIPBLASLT_ENABLE_YAML)
        set(HIPBLASLT_ENABLE_YAML ${HIPBLASLT_ENABLE_LLVM} CACHE BOOL "Use YAML for parsing configuration files." FORCE)
    endif()
    unset(HIPBLASLT_ENABLE_LLVM CACHE)
endif()
