#set(CMAKE_MAKE_PROGRAM "nmake.exe")
#set(CMAKE_GENERATOR "Ninja")
# Ninja doesn't support platform
#set(CMAKE_GENERATOR_PLATFORM x64)

if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/hip/bin")
else()
  set(rocm_bin "/opt/rocm/hip/bin")
endif()


# set(CMAKE_CXX_COMPILER "hipcc")
# set(CMAKE_C_COMPILER "hipcc")
set(CMAKE_CXX_COMPILER "${rocm_bin}/hipcc")
set(CMAKE_C_COMPILER "${rocm_bin}/hipcc")

#set(CMAKE_CXX_LINKER "hipcc" )

# TODO remove, just to speed up slow cmake
# set(CMAKE_C_COMPILER_WORKS 1)
# set(CMAKE_CXX_COMPILER_WORKS 1)
#

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -IC:/hip/include -IC:/hip/lib/clang/12.0.0 -DWIN32 -D_CRT_SECURE_NO_WARNINGS")

# flags for clang direct use
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fms-extensions -fms-compatibility")
# -Wno-ignored-attributes to avoid warning: __declspec attribute 'dllexport' is not supported [-Wignored-attributes] which is used by msvc compiler
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fms-extensions -fms-compatibility -Wno-ignored-attributes")

# flags for clang direct use with hip
# -x hip causes linker error
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -x hip -IC:/hip/include/hip -D__HIP_PLATFORM_HCC__ -D__HIP_ROCclr__ -DHIP_CLANG_HCC_COMPAT_MODE=1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -IC:/hip/include/hip -D__HIP_PLATFORM_HCC__ -D__HIP_ROCclr__ -DHIP_CLANG_HCC_COMPAT_MODE=1")


# set(GTEST_DIR "C:/rocm/Utils/GTestMSVC")
# set(GTEST_INCLUDE_DIR "${GTEST_DIR}/include")
# set(GTEST_LIBRARY "${GTEST_DIR}/lib/Release/gtest.lib")
# set(GTEST_MAIN_LIBRARY "${GTEST_DIR}/lib/Release/gtest_main.lib")
# set(GTEST_LIBRARIES "${GTEST_DIR}/lib/Release/gtest.lib;${GTEST_DIR}/lib/Release/gtest_main.lib")
