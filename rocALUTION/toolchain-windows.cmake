

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
# set(CMAKE_CXX_COMPILER "${rocm_bin}/hipcc.bat")
# set(CMAKE_C_COMPILER "${rocm_bin}/hipcc.bat")
# C:\hip\bin\clang++.exe -Drocalution_EXPORTS -I../../library/src/include -I../../library/include -Iinclude -DWIN32 -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING -Wno-ignored-attributes -DHIP_CLANG_HCC_COMPAT_MODE=1 -fms-extensions -fms-compatibility -D__HIP_ROCclr__=1 -D__HIP_PLATFORM_AMD__=1  -DWIN32 -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING -Wno-ignored-attributes -DHIP_CLANG_HCC_COMPAT_MODE=1 -fms-extensions -fms-compatibility -D__HIP_ROCclr__=1 -D__HIP_PLATFORM_AMD__=1  -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -fvisibility-inlines-hidden -IC:/hip/include -IC:/Users/amd/rocPRIM/rocprim/include -Wno-unused-command-line-argument -Wall -std=c++14 -MD -MT library/CMakeFiles/rocalution.dir/src/conversion/rocalution_prune_dense2csr.cpp.obj -MF library\CMakeFiles\rocsparse.dir\src\conversion\rocsparse_prune_dense2csr.cpp.obj.d -o library/CMakeFiles/rocsparse.dir/src/conversion/rocsparse_prune_dense2csr.cpp.obj -c ../../library/src/conversion/rocsparse_prune_dense2csr.cpp
#  "C:\\hip\\bin\\clang.exe" -cc1 -triple x86_64-pc-windows-msvc19.29.30038 -aux-triple amdgcn-amd-amdhsa -emit-obj -mincremental-linker-compatible --mrelax-relocations -disable-free -disable-llvm-verifier -discard-value-names -main-file-name rocsparse_spgemm.cpp -mrelocation-model pic -pic-level 2 -mframe-pointer=none -fmath-errno -fno-rounding-math -mconstructor-aliases -munwind-tables -target-cpu x86-64 -tune-cpu generic -v "-fcoverage-compilation-dir=C:\\Users\\amd\\rocSPARSE-internal\\build\\release" -resource-dir "C:\\hip\\lib\\clang\\13.0.0" -dependency-file "library\\CMakeFiles\\rocsparse.dir\\src\\extra\\rocsparse_spgemm.cpp.obj.d" -MT library/CMakeFiles/rocsparse.dir/src/extra/rocsparse_spgemm.cpp.obj -sys-header-deps -internal-isystem "C:\\hip\\lib\\clang\\13.0.0\\include\\cuda_wrappers" -internal-isystem "C:\\hip\\include" -include __clang_hip_runtime_wrapper.h -isystem C:/hip/lib/clang/13.0.0/include/.. -isystem "C:\\hip/include" -D __HIP_ROCclr__=1 -D __HIP_PLATFORM_AMD__=1 -D rocsparse_EXPORTS -I ../../library/src/include -I ../../library/include -I include -D WIN32 -D WIN32_LEAN_AND_MEAN -D NOMINMAX -D _CRT_SECURE_NO_WARNINGS -D _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING -D HIP_CLANG_HCC_COMPAT_MODE=1 -D __HIP_ROCclr__=1 -D __HIP_PLATFORM_AMD__=1 -D WIN32 -D WIN32_LEAN_AND_MEAN -D NOMINMAX -D _CRT_SECURE_NO_WARNINGS -D _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING -D HIP_CLANG_HCC_COMPAT_MODE=1 -D __HIP_ROCclr__=1 -D __HIP_PLATFORM_AMD__=1 -D NDEBUG -D _DLL -D _MT -I C:/hip/include -I C:/Users/amd/rocPRIM/rocprim/include -internal-isystem "C:\\hip\\lib\\clang\\13.0.0\\include" -internal-isystem "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30037\\ATLMFC\\include" -internal-isystem "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30037\\include" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\NETFXSDK\\4.8\\include\\um" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\shared" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\um" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\winrt" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\cppwinrt" -internal-isystem "C:\\hip\\lib\\clang\\13.0.0\\include" -internal-isystem "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30037\\ATLMFC\\include" -internal-isystem "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30037\\include" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\NETFXSDK\\4.8\\include\\um" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\shared" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\um" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\winrt" -internal-isystem "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\cppwinrt" -O3 -Wno-ignored-attributes -Wno-ignored-attributes -Wno-unused-command-line-argument -Wall -std=c++14 -fdeprecated-macro "-fdebug-compilation-dir=C:\\Users\\amd\\rocSPARSE-internal\\build\\release" -ferror-limit 19 -fmessage-length=237 -fvisibility-inlines-hidden -fhip-new-launch-api -fno-use-cxa-atexit -fms-extensions -fms-compatibility -fms-compatibility-version=19.29.30038 -fdelayed-template-parsing -fcxx-exceptions -fexceptions -fcolor-diagnostics -vectorize-loops -vectorize-slp --dependent-lib=msvcrt -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -fcuda-include-gpubinary "C:\\Users\\amd\\AppData\\Local\\Temp\\rocsparse_spgemm-d475e8.hipfb" -cuid=ab70d53fbec9a493 -fcuda-allow-variadic-functions -faddrsig -o library/CMakeFiles/rocsparse.dir/src/extra/rocsparse_spgemm.cpp.obj -x hip ../../library/src/extra/rocsparse_spgemm.cpp

set(python "python")

# working
#set(CMAKE_Fortran_COMPILER "C:/Strawberry/c/bin/gfortran.exe")

#set(CMAKE_Fortran_PREPROCESS_SOURCE "<CMAKE_Fortran_COMPILER> -E <INCLUDES> <FLAGS> -cpp <SOURCE> -o <PREPROCESSED_SOURCE>")

# TODO remove, just to speed up slow cmake
#set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
#set(CMAKE_Fortran_COMPILER_WORKS 1)


# our usage flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWIN32 -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING")

# flags for clang direct use

# -Wno-ignored-attributes to avoid warning: __declspec attribute 'dllexport' is not supported [-Wignored-attributes] which is used by msvc compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIP_CLANG_HCC_COMPAT_MODE=1")

# args also in hipcc.bat
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions -fms-compatibility -D__HIP_ROCclr__=1 -D__HIP_PLATFORM_AMD__=1 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_OPENMP_LLVM_RUNTIME")

if (DEFINED ENV{LAPACK_DIR})
  file(TO_CMAKE_PATH "$ENV{LAPACK_DIR}" LAPACK_DIR)
else()
  set(LAPACK_DIR "C:/lapack/build")
endif()

#if (DEFINED ENV{BLIS_DIR})
#  file(TO_CMAKE_PATH "$ENV{BLIS_DIR}" BLIS_DIR)
#else()
#  set(BLIS_DIR "C:/blis/blis-dll-pthread/blis")
#endif()

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
