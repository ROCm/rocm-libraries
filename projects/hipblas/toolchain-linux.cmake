
# LEGACY: Set -DPython3_EXECUTABLE=python3 for pipelines that expect that
if (NOT Python3_EXECUTABLE)
  set(Python3_EXECUTABLE "python3")
endif()

if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(rocm_bin "/opt/rocm/bin")
endif()

# relying on env and path for backward compatibility with external recipes
if (NOT DEFINED ENV{CXX} AND NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${rocm_bin}/amdclang++")
endif()

# Prefer ROCm's amdflang; fall back to gfortran only when ROCm ships none.
if (NOT DEFINED ENV{FC} AND NOT CMAKE_Fortran_COMPILER)
  set(_rocm_fc_hints)
  foreach(_rocm_fc_root "${ROCM_PATH}" "$ENV{ROCM_PATH}" "/opt/rocm")
    if (_rocm_fc_root)
      list(APPEND _rocm_fc_hints "${_rocm_fc_root}/bin" "${_rocm_fc_root}/lib/llvm/bin")
    endif()
  endforeach()
  unset(_rocm_amdflang CACHE)
  find_program(_rocm_amdflang NAMES amdflang HINTS ${_rocm_fc_hints} NO_DEFAULT_PATH)
  if (_rocm_amdflang)
    set(CMAKE_Fortran_COMPILER "${_rocm_amdflang}")
  else()
    set(CMAKE_Fortran_COMPILER "gfortran")
  endif()
  unset(_rocm_amdflang CACHE)
  unset(_rocm_fc_hints)
  unset(_rocm_fc_root)
endif()

if (CONFIG_NO_COMPILER_CHECKS)
  set(CMAKE_CXX_COMPILER_WORKS 1)
  set(CMAKE_Fortran_COMPILER_WORKS 1)
endif()
