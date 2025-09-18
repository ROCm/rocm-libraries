# ROCmClientPackages.cmake
# Provides functions for setting up client package dependencies based on OS

include_guard(GLOBAL)

#[=======================================================================[.rst:
ROCmClientPackages
------------------

Provides functions for setting up ROCm client package dependencies
based on the detected operating system.

Functions:
  rocm_setup_openmp_packages() - Sets OPENMP_RPM and OPENMP_DEB variables
  rocm_setup_gfortran_packages() - Sets GFORTRAN_RPM and GFORTRAN_DEB variables
  rocm_setup_client_packages() - Sets up all client package variables
  rocm_setup_client_components() - Sets up packages and ROCm components

Variables set:
  OPENMP_RPM - OpenMP package name for RPM-based systems
  OPENMP_DEB - OpenMP package name for DEB-based systems
  GFORTRAN_RPM - GFortran package name for RPM-based systems
  GFORTRAN_DEB - GFortran package name for DEB-based systems
#]=======================================================================]

# Internal macro to detect the OS of the build host, if not already done
macro(_rocm_detect_os)
    if(NOT HOST_OS)
        rocm_set_os_id(HOST_OS)
        string(TOLOWER "${HOST_OS}" HOST_OS)
        rocm_read_os_release(HOST_OS_VERSION VERSION_ID)
    endif()
endmacro()

# Set up OpenMP package variables for current OS
macro(rocm_setup_posix_openmp_packages)
    _rocm_detect_os()

    set(OPENMP_RPM "libgomp")
    set(OPENMP_DEB "libomp-dev")

    # OS-specific overrides
    if(HOST_OS STREQUAL "sles")
        set(OPENMP_RPM "libgomp1")
    endif()
endmacro()

# Set up GFortran package variables for current OS
macro(rocm_setup_posix_gfortran_packages)
    _rocm_detect_os()

    set(GFORTRAN_RPM "libgfortran4")
    set(GFORTRAN_DEB "libgfortran4")

    if(HOST_OS STREQUAL "centos" OR HOST_OS STREQUAL "rhel" OR HOST_OS STREQUAL "almalinux")
        if(HOST_OS_VERSION VERSION_GREATER_EQUAL "8")
            set(GFORTRAN_RPM "libgfortran")
        endif()
    elseif(HOST_OS STREQUAL "ubuntu" AND HOST_OS_VERSION VERSION_GREATER_EQUAL "20.04")
        set(GFORTRAN_DEB "libgfortran5")
    elseif(HOST_OS STREQUAL "mariner" OR HOST_OS STREQUAL "azurelinux")
        set(GFORTRAN_RPM "gfortran")
    endif()
endmacro()

# Macro to set up all client package variables
macro(rocm_setup_client_packages)
    _rocm_detect_os()
    message(STATUS "Detected OS: ${HOST_OS} ${HOST_OS_VERSION}")

    rocm_setup_posix_openmp_packages()
    rocm_setup_posix_gfortran_packages()

    message(STATUS "OpenMP packages: RPM=${OPENMP_RPM}, DEB=${OPENMP_DEB}")
    message(STATUS "GFortran packages: RPM=${GFORTRAN_RPM}, DEB=${GFORTRAN_DEB}")
endmacro()

# Convenience function that also sets up standard components
function(rocm_setup_client_components component_name client_component_name)
    rocm_setup_client_packages()
    rocm_package_setup_component(${component_name})
    rocm_package_setup_client_component(${client_component_name})
endfunction()
