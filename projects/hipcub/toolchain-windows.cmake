#set(CMAKE_MAKE_PROGRAM "nmake.exe")
#set(CMAKE_GENERATOR "Ninja")
# Ninja doesn't support platform
#set(CMAKE_GENERATOR_PLATFORM x64)

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

set(CMAKE_CXX_COMPILER "${rocm_bin}/clang++.exe")
set(CMAKE_C_COMPILER "${rocm_bin}/clang.exe")

if (NOT python)
  set(python "python3") # take default for windows
endif()

# our usage flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWIN32 -D_CRT_SECURE_NO_WARNINGS")

# flags for clang direct use
# -Wno-ignored-attributes to avoid warning: __declspec attribute 'dllexport' is not supported [-Wignored-attributes] which is used by msvc compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fms-extensions -fms-compatibility -Wno-ignored-attributes")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__ -D__HIP_ROCclr__")

if (DEFINED ENV{VCPKG_PATH})
  file(TO_CMAKE_PATH "$ENV{VCPKG_PATH}" VCPKG_PATH)
else()
  set(VCPKG_PATH "C:/github/vcpkg")
endif()
include("${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake")
