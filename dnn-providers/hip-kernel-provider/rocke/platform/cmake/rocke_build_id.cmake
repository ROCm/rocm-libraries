# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocke_build_id.cmake -- compute a deterministic, git-independent content hash
# of the portable-IR native artifact bundle. The hash covers the inputs owned by
# this repository that determine the shared engine and replay CLI. It deliberately
# does not identify the compiler, toolchain, or final binary bytes.
#
# rocke_compute_build_id(<out_var> <root_dir>)
#   Hashes the engine C++ sources and headers, replay CLI source, and CMake build
#   recipes that produce those artifacts. Each file contributes its repo-relative
#   path followed by a SHA-256 of its bytes; the sorted list is hashed again to
#   yield the build-id. A rename or byte edit therefore changes the ID.

# Collect repository-owned inputs to the shared-engine/replay-CLI bundle.
function(rocke_build_id_inputs out_var root_dir)
  # Private headers are engine inputs too. pybind sources are not: CMake excludes
  # cpp/bindings from rocke_core, and neither bundle artifact compiles them.
  file(GLOB_RECURSE _rocke_all
       "${root_dir}/cpp/*.cpp"
       "${root_dir}/cpp/*.h"
       "${root_dir}/cpp/*.hpp")
  list(FILTER _rocke_all EXCLUDE REGEX "/cpp/bindings/")

  # Include the CLI frontend and the build recipes that select, stamp, compile,
  # and link the two native artifacts. Keep this list mirrored in online.py.
  foreach(_rocke_input
          "tests/portable_ir/replay_cli.cpp"
          "CMakeLists.txt"
          "tests/CMakeLists.txt"
          "cmake/rocke_build_id.cmake"
          "cmake/generate_rocke_build_id.cmake")
    if(EXISTS "${root_dir}/${_rocke_input}")
      list(APPEND _rocke_all "${root_dir}/${_rocke_input}")
    endif()
  endforeach()
  list(SORT _rocke_all)
  set(${out_var} "${_rocke_all}" PARENT_SCOPE)
endfunction()

# Compute the stable short digest over the complete native bundle source surface.
function(rocke_compute_build_id out_var root_dir)
  rocke_build_id_inputs(_rocke_all "${root_dir}")

  set(_rocke_lines "")
  foreach(_f IN LISTS _rocke_all)
    file(RELATIVE_PATH _rel "${root_dir}" "${_f}")
    file(SHA256 "${_f}" _fh)
    list(APPEND _rocke_lines "${_rel}:${_fh}")
  endforeach()

  # Sort for a stable, filesystem-order-independent digest.
  list(SORT _rocke_lines)
  string(REPLACE ";" "\n" _rocke_blob "${_rocke_lines}")
  string(SHA256 _rocke_digest "${_rocke_blob}")

  # 16 hex chars is plenty to distinguish builds while staying readable in logs.
  string(SUBSTRING "${_rocke_digest}" 0 16 _rocke_short)
  set(${out_var} "${_rocke_short}" PARENT_SCOPE)
endfunction()
