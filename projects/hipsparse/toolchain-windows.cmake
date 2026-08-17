
if (DEFINED ENV{HIP_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
elseif (DEFINED ENV{HIP_DIR})
  file(TO_CMAKE_PATH "$ENV{HIP_DIR}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
else()
  set(HIP_DIR "C:/hip")
  set(rocm_bin "C:/hip/bin")
endif()

set(CXX_COMPILER_PATH "${rocm_bin}/clang++.exe")
set(C_COMPILER_PATH "${rocm_bin}/clang.exe")
set(CXX_COMPILER_PATH_ALT "${rocm_bin}/../lib/llvm/bin/clang++.exe")
set(C_COMPILER_PATH_ALT "${rocm_bin}/../lib/llvm/bin/clang.exe")

# Check for the first path (preferred)
if(EXISTS "${CXX_COMPILER_PATH}" AND EXISTS "${C_COMPILER_PATH}")
    set(CMAKE_CXX_COMPILER "${CXX_COMPILER_PATH}")
    set(CMAKE_C_COMPILER "${C_COMPILER_PATH}")
elseif(EXISTS "${CXX_COMPILER_PATH_ALT}" AND EXISTS "${C_COMPILER_PATH_ALT}")
    set(CMAKE_CXX_COMPILER "${CXX_COMPILER_PATH_ALT}")
    set(CMAKE_C_COMPILER "${C_COMPILER_PATH_ALT}")
else()
    message(WARNING "Compiler was not found. CMAKE_CXX_COMPILER will not be explicitly set.")
endif()

# working
#set(CMAKE_Fortran_COMPILER "C:/Strawberry/c/bin/gfortran.exe")

#set(CMAKE_Fortran_PREPROCESS_SOURCE "<CMAKE_Fortran_COMPILER> -E <INCLUDES> <FLAGS> -cpp <SOURCE> -o <PREPROCESSED_SOURCE>")

# TODO remove, just to speed up slow cmake
#set(CMAKE_C_COMPILER_WORKS 1)
#set(CMAKE_CXX_COMPILER_WORKS 1)
#set(CMAKE_Fortran_COMPILER_WORKS 1)


# our usage flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWIN32 -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING")

# flags for clang direct use
# -Wno-ignored-attributes to avoid warning: __declspec attribute 'dllexport' is not supported [-Wignored-attributes] which is used by msvc compiler
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14-fms-extensions -fms-compatibility -Wno-ignored-attributes")
# # -I${HIP_PATH}/include -I${HIP_PATH}/include/hip  add  -x hip ??
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__ -D__HIP_ROCclr__ -DHIP_CLANG_HCC_COMPAT_MODE=1")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
endif()


if (DEFINED ENV{VCPKG_PATH})
  file(TO_CMAKE_PATH "$ENV{VCPKG_PATH}" VCPKG_PATH)
else()
  set(VCPKG_PATH "C:/github/vcpkg")
endif()
include("${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake")

set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_STATIC_LIBRARY_PREFIX "static_")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
set(CMAKE_SHARED_LIBRARY_PREFIX "")
