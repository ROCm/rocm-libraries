# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Mutual exclusion check: cannot have both HIPBLASLT_ENABLE_YAML and HIPBLASLT_ENABLE_MSGPACK
if((DEFINED HIPBLASLT_ENABLE_YAML OR HIPBLASLT_ENABLE_LLVM) AND DEFINED HIPBLASLT_ENABLE_MSGPACK)
    message(FATAL_ERROR
        "Cannot set both HIPBLASLT_ENABLE_YAML and HIPBLASLT_ENABLE_MSGPACK.\n"
        "These options are mutually exclusive. Please use only HIPBLASLT_ENABLE_YAML.\n"
        "  HIPBLASLT_ENABLE_MSGPACK is deprecated - use HIPBLASLT_ENABLE_YAML instead.")
endif()

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

# Backwards compatibility for HIPBLASLT_ENABLE_MSGPACK → HIPBLASLT_ENABLE_YAML
if(DEFINED HIPBLASLT_ENABLE_MSGPACK)
    if(HIPBLASLT_ENABLE_MSGPACK)
        set(_yaml_val OFF)
    else()
        set(_yaml_val ON)
    endif()
    message(DEPRECATION
      "HIPBLASLT_ENABLE_MSGPACK is deprecated. Use HIPBLASLT_ENABLE_YAML instead.\n"
      "  Old: -DHIPBLASLT_ENABLE_MSGPACK=${HIPBLASLT_ENABLE_MSGPACK}\n"
      "  New: -DHIPBLASLT_ENABLE_YAML=${_yaml_val}")
    if(NOT DEFINED HIPBLASLT_ENABLE_YAML)
        set(HIPBLASLT_ENABLE_YAML ${_yaml_val} CACHE BOOL "Use YAML for parsing configuration files." FORCE)
    endif()
    unset(HIPBLASLT_ENABLE_MSGPACK CACHE)
endif()
