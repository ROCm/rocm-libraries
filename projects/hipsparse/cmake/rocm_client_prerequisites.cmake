# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# ROCm Client Prerequisites
# ==============================================================================
# This file handles OS detection and sets up package prerequisites for clients.
# It determines the correct package names for OpenMP libraries
# based on the detected operating system and version.
#
# Output Variables:
# -----------------
# CLIENTS_OS          - Detected OS name (lowercase)
# CLIENTS_OS_VERSION  - Detected OS version
# OPENMP_RPM          - OpenMP library package name for RPM-based systems
# OPENMP_DEB          - OpenMP library package name for DEB-based systems
# ==============================================================================

# Detect operating system
if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
endif()

message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")

# Set OpenMP library package names if OpenMP is enabled
if(HIPSPARSE_ENABLE_OPENMP)
    set(OPENMP_RPM "libgomp")
    set(OPENMP_DEB "libomp-dev")
    if(CLIENTS_OS STREQUAL "sles")
        set(OPENMP_RPM "libgomp1")
    endif()
else()
    set(OPENMP_RPM "")
    set(OPENMP_DEB "")
endif()
