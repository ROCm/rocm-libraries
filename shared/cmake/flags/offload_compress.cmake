include_guard(GLOBAL)

option(BUILD_OFFLOAD_COMPRESS "Build with offload compression" ON)

include(CheckCXXCompilerFlag)

# Function that checks if a HIP compiler function exists.
function(hip_check_compiler_flag _flag _var)
  if(USE_HIPCXX)
    cmake_check_compiler_flag(HIP "${_flag}" ${_var})
  else()
    check_cxx_compiler_flag("-xhip ${_flag}" ${_var})
  endif()
endfunction(hip_check_compiler_flag)

# Enable offload compress.
if(BUILD_OFFLOAD_COMPRESS)
  # Detect capability and add flag.
  hip_check_compiler_flag("--offload-compress" HIP_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if(HIP_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    if(USE_HIPCXX)
      add_compile_options($<$<COMPILE_LANGUAGE:HIP>:--offload-compress>)
    else()
      add_compile_options("--offload-compress")
    endif()
  else()
    message(WARNING "BUILD_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option.")
  endif()
endif()
