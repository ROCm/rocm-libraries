#set(CMAKE_MAKE_PROGRAM "nmake.exe")
#set(CMAKE_GENERATOR "Ninja")
# Ninja doesn't support platform
#set(CMAKE_GENERATOR_PLATFORM x64)

if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(rocm_bin "/opt/rocm/bin")
endif()


# set(CMAKE_CXX_COMPILER "hipcc")
# set(CMAKE_C_COMPILER "hipcc")
set(CMAKE_CXX_COMPILER "${rocm_bin}/hipcc")
set(CMAKE_C_COMPILER "${rocm_bin}/hipcc")
