# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocke_build_id.cmake -- compute a deterministic, git-independent content hash of
# the engine source tree. The hash changes if and only if a tracked source byte
# changes; it is stable across rebuilds of the same sources, across machines,
# and regardless of any version-control state.
#
# rocke_compute_build_id(<out_var> <root_dir>)
#   Hashes the sorted concatenation of every src/**/*.cpp and include/**/*.h*
#   under <root_dir>. Each file contributes its repo-relative path followed by a
#   SHA-256 of its bytes; the per-file lines are sorted (locale-independent) and
#   the whole list is hashed once more to yield the build-id. Including the path
#   means a rename alone changes the id; including per-file content means an edit
#   to any byte changes it.

# Compute a deterministic build id for the rocKE C++ source tree.
function(rocke_compute_build_id out_var root_dir)
  # Collect the tracked source surface. GLOB (not GLOB_RECURSE with
  # CONFIGURE_DEPENDS) is fine here: this runs at configure time, and the
  # top-level CMakeLists already re-globs sources with CONFIGURE_DEPENDS, so a
  # newly added file triggers a reconfigure and thus a recompute.
  file(GLOB_RECURSE _rocke_src
       "${root_dir}/Cpp/*.cpp")
  file(GLOB_RECURSE _rocke_hdr
       "${root_dir}/Cpp/include/*.h"
       "${root_dir}/Cpp/include/*.hpp")
  set(_rocke_all ${_rocke_src} ${_rocke_hdr})

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
