#set(CMAKE_MAKE_PROGRAM "nmake.exe")
#set(CMAKE_GENERATOR "Ninja")
# Ninja doesn't support platform
#set(CMAKE_GENERATOR_PLATFORM x64)
if (NOT python)
  set(python "python3") # default for linux
endif()


if (NOT DEFINED ENV{ROCM_PATH})
  set(ENV{ROCM_PATH} "/opt/rocm" CACHE PATH "Path to the ROCm installation.")
endif()

set(rocm_bin "$ENV{ROCM_PATH}/bin")
if (NOT DEFINED CMAKE_PREFIX_PATH) 
  list( APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH}/llvm $ENV{ROCM_PATH})
endif()

if (NOT DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER "${rocm_bin}/amdclang++" CACHE PATH "Path to the C++ compiler")
  set(CMAKE_CXX_FLAGS "-mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false")
else()
  set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE PATH "Path to the C++ compiler")
endif()
