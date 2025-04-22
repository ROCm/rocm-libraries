CMAKE_MINIMUM_REQUIRED(VERSION 3.18...3.31)

#Setting Default value for libdir
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")

if(DEFINED HIPFORT_COMPILER)
  message(DEPRECATION "HIPFORT_COMPILER is deprecated. Use CMAKE_Fortran_COMPILER instead.")
  set(CMAKE_Fortran_COMPILER "${HIPFORT_COMPILER}" CACHE STRING "Fortran compiler")
endif()

if(DEFINED HIPFORT_AR)
  message(DEPRECATION "HIPFORT_AR is deprecated. Use CMAKE_AR instead.")
  set(CMAKE_AR "${HIPFORT_AR}" CACHE STRING "Archiving tool for static libraries")
endif()

if(DEFINED HIPFORT_RANLIB)
  message(DEPRECATION "HIPFORT_RANLIB is deprecated. Use CMAKE_RANLIB instead.")
  set(CMAKE_RANLIB "${HIPFORT_RANLIB}" CACHE STRING "Randomizing tool for static libraries")
endif()

if(DEFINED HIPFORT_COMPILER_FLAGS)
  message(DEPRECATION "HIPFORT_COMPILER_FLAGS is deprecated. Use CMAKE_Fortran_FLAGS instead.")
  set(CMAKE_Fortran_FLAGS "${HIPFORT_COMPILER_FLAGS}" CACHE STRING "Flags for the Fortran compiler and linker")
endif()

if(DEFINED HIPFORT_BUILD_TYPE)
  message(DEPRECATION "HIPFORT_BUILD_TYPE is deprecated. Use CMAKE_BUILD_TYPE instead.")
  set(CMAKE_BUILD_TYPE "${HIPFORT_BUILD_TYPE}" CACHE STRING "Build type to generate")
endif()

if(DEFINED HIPFORT_INSTALL_DIR)
  message(DEPRECATION "HIPFORT_INSTALL_DIR is deprecated. Use CMAKE_INSTALL_PREFIX instead.")
  set(CMAKE_INSTALL_PREFIX "${HIPFORT_INSTALL_DIR}" CACHE STRING "Install directory")
endif()

# hip-config.cmake requires CXX enabled
PROJECT(hipfort VERSION 0.7.0 LANGUAGES Fortran C CXX)

#include definitions for GNU install dir cmake flags.
include(GNUInstallDirs)

message("-- HIPFORT -------------  cmake START -------------------")
message("-- Fortran Compiler:       ${CMAKE_Fortran_COMPILER}")
message("-- Build Type:             ${CMAKE_BUILD_TYPE}")
message("-- Installation Directory: ${CMAKE_INSTALL_PREFIX}")
message("-- hipfort Version:        ${hipfort_VERSION}")
message("-- HIPFORT ----------------------------------------------")

set(CMAKE_Fortran_FORMAT FREE)
set(CMAKE_Fortran_PREPROCESS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

INCLUDE(FortranCInterface)
FortranCInterface_VERIFY()
IF(NOT FortranCInterface_VERIFIED_C)
  MESSAGE(FATAL_ERROR "Fortran compiler must support C Interface")
ENDIF(NOT FortranCInterface_VERIFIED_C)
	
IF(NOT CMAKE_Fortran_COMPILER_SUPPORTS_F90)
  MESSAGE(FATAL_ERROR "Fortran compiler does not support F90")
ENDIF(NOT CMAKE_Fortran_COMPILER_SUPPORTS_F90)

# Test for Fortran 08 support by using an f08-specific construct.
IF(NOT DEFINED CMAKE_Fortran_COMPILER_SUPPORTS_F08)
  MESSAGE(CHECK_START "Checking whether ${CMAKE_Fortran_COMPILER} supports Fortran 08")
  INCLUDE(CheckFortranSourceCompiles)
  CHECK_FORTRAN_SOURCE_COMPILES("
      module mod
        interface foo
          module procedure :: foo_a,foo_b
        end interface
      contains
        subroutine foo_a(a)
          use iso_c_binding
          integer,target,dimension(:) :: a
          type(c_ptr) :: a_ptr
          a_ptr = c_loc(a) ! gfortran < 4.9 fails here
        end
        subroutine foo_b(b)
          integer,pointer,dimension(:,:) :: b
        end
      end
      PROGRAM TESTFortran08
        use mod
        implicit none
        integer :: a(5)
        integer,allocatable :: b(:)
        allocate(b,mold=a)
        deallocate(b)
      END PROGRAM TESTFortran08
    " CMAKE_Fortran_COMPILER_SUPPORTS_F08)
  IF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    MESSAGE(CHECK_PASS "yes")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Determining if the Fortran compiler supports Fortran 08 passed with "
      "the following output:\n${OUTPUT}\n\n")
    set(CMAKE_Fortran_COMPILER_SUPPORTS_F08 1)
  ELSE(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    MESSAGE(CHECK_FAIL "no")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
      "Determining if the Fortran compiler supports Fortran 08 failed with "
      "the following output:\n${OUTPUT}\n\n")
    set(CMAKE_Fortran_COMPILER_SUPPORTS_F08 0)
  ENDIF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
  unset(CMAKE_Fortran_COMPILER_SUPPORTS_F08 CACHE)
ENDIF(NOT DEFINED CMAKE_Fortran_COMPILER_SUPPORTS_F08)

# Set compile flags for DEBUG, # RELEASE, or TESTING.  
INCLUDE(${CMAKE_MODULE_PATH}/SetFortranFlags.cmake) 
message("-- Done setting FortranFlags")
message("-- CMAKE_Fortran_FLAGS_DEBUG:   ${CMAKE_Fortran_FLAGS_DEBUG}")
message("-- CMAKE_Fortran_FLAGS_RELEASE: ${CMAKE_Fortran_FLAGS_RELEASE}")
message("-- CMAKE_Fortran_FLAGS_TESTING: ${CMAKE_Fortran_FLAGS_TESTING}")

# There is an error in CMAKE with this flag for pgf90.  Unset it
GET_FILENAME_COMPONENT(FCNAME ${CMAKE_Fortran_COMPILER} NAME)
IF(FCNAME STREQUAL "pgf90")
    UNSET(CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS)
ENDIF(FCNAME STREQUAL "pgf90")

option(BUILD_TESTING "Build tests" OFF)
if(BUILD_TESTING)
  enable_testing()
endif()

# Subdirectories and packaging
# NOTE: rocm-cmake must be be included before
# adding subdirectories.
INCLUDE(${CMAKE_MODULE_PATH}/rocm-cmake.cmake)
rocm_setup_version(VERSION ${VERSION})
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/lib)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/test)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/bin)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-runtime-amd (>= 4.5.0)")
set(CPACK_RPM_PACKAGE_REQUIRES "hip-runtime-amd >= 4.5.0")
# Package name changed from hipfort to hipfort-devel/dev
# Backward compatibility support for old package name
set(CPACK_DEBIAN_PACKAGE_PROVIDES "hipfort")
set(CPACK_DEBIAN_PACKAGE_REPLACES "hipfort")
set(CPACK_DEBIAN_PACKAGE_CONFLICTS "hipfort")
set(CPACK_RPM_PACKAGE_PROVIDES "hipfort")
set(CPACK_RPM_PACKAGE_OBSOLETES "hipfort")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")
# Prevent rpmbuild from stripping binaries, which caused issues on CentOS
set(CPACK_RPM_SPEC_INSTALL_POST "/bin/true")

rocm_create_package(
  NAME hipfort
  DESCRIPTION "Fortran Interface For GPU Kernel Libraries"
  MAINTAINER "Hipfort maintainer <hipfort-maintainer@amd.com>"
  HEADER_ONLY # Enabled For generating Only -devel pkg.
)

message("-- HIPFORT -------------  cmake DONE --------------------")
