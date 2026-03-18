# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# ROCm Client Prerequisites
# ==============================================================================
# This file handles OS detection and sets up package prerequisites for clients.
# It determines the correct package names for Fortran and OpenMP libraries
# based on the detected operating system and version.
#
# Output Variables:
# -----------------
# CLIENTS_OS          - Detected OS name (lowercase)
# CLIENTS_OS_VERSION  - Detected OS version
# GFORTRAN_RPM        - Fortran library package name for RPM-based systems
# GFORTRAN_DEB        - Fortran library package name for DEB-based systems
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

# Set default Fortran library package names
set(GFORTRAN_RPM "libgfortran4")
set(GFORTRAN_DEB "libgfortran4")

# Adjust Fortran package names based on OS and version
if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel" OR CLIENTS_OS STREQUAL "almalinux")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
        set(GFORTRAN_RPM "libgfortran")
    endif()
elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
endif()

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
