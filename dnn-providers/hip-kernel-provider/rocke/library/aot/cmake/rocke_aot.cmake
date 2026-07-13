# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared CMake support for provider-owned rocKE ahead-of-time artifacts.
#
# rocKE (`rocke`) and the library packages (`kernels`) are consumed from the
# build-local rocke-pyenv (rocke/CMakeLists.txt), so every Python invocation runs
# under ${ROCKE_PYENV_PYTHON} and depends on ${ROCKE_PYENV_STAMP}. The only extra
# import root is this tree's own `rocke_client_aot` tooling package (not part of
# the editable library install) plus the rocm_kpack source dir for the packer.

if(NOT DEFINED ROCKE_PYENV_PYTHON OR NOT DEFINED ROCKE_PYENV_STAMP)
    message(FATAL_ERROR
        "rocke_aot.cmake requires the rocke-pyenv variables (ROCKE_PYENV_PYTHON / "
        "ROCKE_PYENV_STAMP) from rocke/CMakeLists.txt; enable ROCKE_BUILD_PYENV."
    )
endif()

get_filename_component(_ROCKE_CLIENT_AOT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_ROCKE_CLIENT_AOT_BUILD_TOOL "${_ROCKE_CLIENT_AOT_ROOT}/tools/rocke_aot_build.py")
set(_ROCKE_CLIENT_AOT_PACK_TOOL "${_ROCKE_CLIENT_AOT_ROOT}/tools/rocke_kpack_pack.py")
set(_ROCKE_CLIENT_AOT_PYTHON_ROOT "${_ROCKE_CLIENT_AOT_ROOT}/python")
set(_ROCKE_CLIENT_AOT_SCHEMA_ROOT "${_ROCKE_CLIENT_AOT_ROOT}/schemas")
set(_ROCKE_CLIENT_AOT_BUNDLE_SCHEMA "${_ROCKE_CLIENT_AOT_SCHEMA_ROOT}/bundle.schema.json")

# engine_build_id / llvm_flavor recorded in the per-arch bundle manifest. Plan 3
# consumes these for compatibility gating; expose them as overridable cache vars.
set(ROCKE_AOT_ENGINE_BUILD_ID "rocke-client" CACHE STRING
    "engine_build_id recorded in rocKE AOT bundle manifests")
if(NOT DEFINED ROCKE_AOT_LLVM_FLAVOR OR ROCKE_AOT_LLVM_FLAVOR STREQUAL "")
    if(DEFINED ENV{ROCKE_LLVM_FLAVOR} AND NOT "$ENV{ROCKE_LLVM_FLAVOR}" STREQUAL "")
        set(ROCKE_AOT_LLVM_FLAVOR "$ENV{ROCKE_LLVM_FLAVOR}")
    else()
        set(ROCKE_AOT_LLVM_FLAVOR "llvm20")
    endif()
endif()

# --- rocm_kpack source availability -----------------------------------------
# The packer imports rocm_kpack from source (never pip -- it is not an installable
# package). Three resolution paths, in order:
#   1. -DROCKE_KPACK_PYTHON_DIR=<rocm-systems>/shared/kpack/python: an explicit
#      source dir. TheRock/CI pass this as a configure arg pointing at their
#      pinned rocm-systems checkout.
#   2. Otherwise: a sparse partial clone of rocm-systems (only shared/kpack/python,
#      ~4 MB) into the build tree.
#   3. If that git path fails (older git, no partial-clone/sparse support): fall
#      back to FetchContent's shallow clone (whole tree, ~1 GB, but more robust).
# The resolved dir is filesystem-checked below and gates loud -- never silently
# skipped.
set(ROCKE_KPACK_GIT_REPOSITORY "https://github.com/ROCm/rocm-systems.git"
    CACHE STRING "rocm-systems repository fetched for the rocm_kpack source dep")
set(ROCKE_KPACK_GIT_TAG "develop"
    CACHE STRING "rocm-systems git ref fetched for the rocm_kpack source dep")

if(DEFINED ROCKE_KPACK_PYTHON_DIR AND NOT "${ROCKE_KPACK_PYTHON_DIR}" STREQUAL "")
    get_filename_component(_ROCKE_KPACK_PYTHON_DIR "${ROCKE_KPACK_PYTHON_DIR}" ABSOLUTE)
    message(STATUS "rocKE AOT kpack packer: rocm_kpack from -DROCKE_KPACK_PYTHON_DIR ${_ROCKE_KPACK_PYTHON_DIR}")
else()
    # No explicit dir: consume rocm_kpack from a source checkout of rocm-systems.
    # Try a sparse partial clone first -- it pulls only shared/kpack/python (blobs
    # deferred via --filter=blob:none, tree limited via cone sparse-checkout), so a
    # few MB rather than the whole monorepo. The checkout lives in the build tree;
    # the EXISTS guard makes reconfigures a no-op.
    set(_ROCKE_KPACK_SRC "${CMAKE_CURRENT_BINARY_DIR}/rocm_systems_kpack")
    set(_ROCKE_KPACK_PYTHON_DIR "${_ROCKE_KPACK_SRC}/shared/kpack/python")
    if(NOT EXISTS "${_ROCKE_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py")
        find_package(Git REQUIRED)
        message(STATUS
            "rocKE AOT kpack packer: sparse-cloning rocm-systems "
            "(${ROCKE_KPACK_GIT_REPOSITORY}@${ROCKE_KPACK_GIT_TAG})")
        file(REMOVE_RECURSE "${_ROCKE_KPACK_SRC}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --filter=blob:none --sparse
                    --branch "${ROCKE_KPACK_GIT_TAG}"
                    "${ROCKE_KPACK_GIT_REPOSITORY}" "${_ROCKE_KPACK_SRC}"
            RESULT_VARIABLE _ROCKE_KPACK_CLONE_RC
        )
        if(_ROCKE_KPACK_CLONE_RC EQUAL 0)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${_ROCKE_KPACK_SRC}"
                        sparse-checkout set shared/kpack/python
                RESULT_VARIABLE _ROCKE_KPACK_SPARSE_RC
            )
        endif()
    endif()
    if(EXISTS "${_ROCKE_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py")
        message(STATUS "rocKE AOT kpack packer: rocm_kpack from sparse checkout ${_ROCKE_KPACK_PYTHON_DIR}")
    else()
        # Sparse path unavailable: drop any partial tree and fall back to
        # FetchContent's shallow clone. SOURCE_SUBDIR names a directory with no
        # CMakeLists.txt so FetchContent populates without configuring rocm-systems
        # as a CMake subproject; GIT_SUBMODULES "" skips the heavy submodule trees.
        message(STATUS "rocKE AOT kpack packer: sparse clone unavailable; falling back to FetchContent shallow clone")
        file(REMOVE_RECURSE "${_ROCKE_KPACK_SRC}")
        include(FetchContent)
        FetchContent_Declare(
            rocke_rocm_systems
            GIT_REPOSITORY "${ROCKE_KPACK_GIT_REPOSITORY}"
            GIT_TAG        "${ROCKE_KPACK_GIT_TAG}"
            GIT_SHALLOW    TRUE
            GIT_SUBMODULES ""
            GIT_PROGRESS   TRUE
            SOURCE_SUBDIR  shared/kpack/python
        )
        FetchContent_MakeAvailable(rocke_rocm_systems)
        set(_ROCKE_KPACK_PYTHON_DIR "${rocke_rocm_systems_SOURCE_DIR}/shared/kpack/python")
        message(STATUS
            "rocKE AOT kpack packer: rocm_kpack from fetched rocm-systems "
            "${_ROCKE_KPACK_PYTHON_DIR}")
    endif()
endif()

if(NOT EXISTS "${_ROCKE_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py")
    message(FATAL_ERROR
        "rocm_kpack source not found: ${_ROCKE_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py "
        "is missing. Set -DROCKE_KPACK_PYTHON_DIR=<rocm-systems>/shared/kpack/python "
        "or check ROCKE_KPACK_GIT_TAG (${ROCKE_KPACK_GIT_TAG})."
    )
endif()

# --- libamd_comgr availability (compile_kernel backend="python") -------------
# rocke.helpers.compile_kernel ctypes-loads libamd_comgr at build time. We locate
# it through comgr's own CMake package (amd_comgr): TheRock's amd-comgr subproject
# provides it, and a standalone build finds it under /opt/rocm. The imported target
# points at comgr in its assembled tree, where comgr's own RUNPATH
# ($ORIGIN, $ORIGIN/llvm/lib, $ORIGIN/rocm_sysdeps/lib) resolves its entire
# dependency closure (libLLVM, libclang-cpp, the vendored rocm_sysdeps libs). So
# once comgr is loaded from that tree nothing about its deps needs forwarding.
# Resolution order:
#   1. explicit ROCKE_COMGR_LIB (cache/env) override,
#   2. find_package(amd_comgr) imported target location,
#   3. otherwise unset, and compile_kernel's own resolver applies at build time.
set(_ROCKE_COMGR_LIB "")
if(DEFINED ROCKE_COMGR_LIB AND NOT "${ROCKE_COMGR_LIB}" STREQUAL "")
    set(_ROCKE_COMGR_LIB "${ROCKE_COMGR_LIB}")
elseif(DEFINED ENV{ROCKE_COMGR_LIB} AND NOT "$ENV{ROCKE_COMGR_LIB}" STREQUAL "")
    set(_ROCKE_COMGR_LIB "$ENV{ROCKE_COMGR_LIB}")
else()
    find_package(amd_comgr CONFIG QUIET)
    if(TARGET amd_comgr)
        get_target_property(_ROCKE_COMGR_CFGS amd_comgr IMPORTED_CONFIGURATIONS)
        foreach(_cfg RELEASE ${_ROCKE_COMGR_CFGS})
            get_target_property(_ROCKE_COMGR_LOC amd_comgr IMPORTED_LOCATION_${_cfg})
            if(_ROCKE_COMGR_LOC)
                set(_ROCKE_COMGR_LIB "${_ROCKE_COMGR_LOC}")
                break()
            endif()
        endforeach()
    endif()
endif()
if("${_ROCKE_COMGR_LIB}" STREQUAL "")
    message(STATUS
        "rocKE AOT build: libamd_comgr not resolved at configure; "
        "compile_kernel's runtime resolver will apply")
elseif(NOT EXISTS "${_ROCKE_COMGR_LIB}")
    message(FATAL_ERROR "ROCKE_COMGR_LIB does not exist: ${_ROCKE_COMGR_LIB}")
else()
    get_filename_component(_ROCKE_COMGR_LIB "${_ROCKE_COMGR_LIB}" ABSOLUTE)
    message(STATUS "rocKE AOT build: libamd_comgr from ${_ROCKE_COMGR_LIB}")
endif()

# Reconfigure when common AOT Python helpers or shared JSON Schemas change.
# Kernel families add their handler + family schemas via the freshness globs in
# rocke_client_add_aot_instances().
file(GLOB_RECURSE _ROCKE_CLIENT_AOT_PACKAGE_MODULES CONFIGURE_DEPENDS
    "${_ROCKE_CLIENT_AOT_PYTHON_ROOT}/rocke_client_aot/*.py"
)
file(GLOB _ROCKE_CLIENT_AOT_COMMON_SCHEMA_DEPENDS CONFIGURE_DEPENDS
    "${_ROCKE_CLIENT_AOT_SCHEMA_ROOT}/*.schema.json"
)

# The HSACO bytes are determined by the kernels package (builders/signatures) and
# the rocke platform (compile_kernel lowering). Both are editable-installed in
# the pyenv, so edits take effect at runtime but the pyenv stamp never re-fires
# for them. Track their sources so `cmake --build` regenerates HSACO (and thus
# the .kpack) when the code that produces it changes, rather than shipping stale
# artifacts. Coarse globs: correctness over minimal rebuilds.
get_filename_component(_ROCKE_LIBRARY_DIR "${_ROCKE_CLIENT_AOT_ROOT}/.." ABSOLUTE)
get_filename_component(_ROCKE_ROOT_DIR "${_ROCKE_LIBRARY_DIR}/.." ABSOLUTE)
file(GLOB_RECURSE _ROCKE_CLIENT_AOT_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${_ROCKE_LIBRARY_DIR}/kernels/*.py"
)
file(GLOB_RECURSE _ROCKE_CLIENT_AOT_PLATFORM_SOURCES CONFIGURE_DEPENDS
    "${_ROCKE_ROOT_DIR}/platform/python/rocke/*.py"
)

# Return the PYTHONPATH used by rocKE client AOT tooling in OUT_VAR.
#
# The pyenv already resolves `rocke` and `kernels`; only this tree's tooling
# package and the rocm_kpack source dir are prepended. An incoming developer
# PYTHONPATH is preserved last.
function(rocke_client_aot_pythonpath OUT_VAR)
    set(_ROCKE_CLIENT_AOT_PYTHONPATH "${_ROCKE_CLIENT_AOT_PYTHON_ROOT}")
    if(NOT "${_ROCKE_KPACK_PYTHON_DIR}" STREQUAL "")
        list(APPEND _ROCKE_CLIENT_AOT_PYTHONPATH "${_ROCKE_KPACK_PYTHON_DIR}")
    endif()
    if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
        cmake_path(CONVERT "$ENV{PYTHONPATH}" TO_CMAKE_PATH_LIST
                   _ROCKE_CLIENT_AOT_INCOMING_PYTHONPATH)
        list(APPEND _ROCKE_CLIENT_AOT_PYTHONPATH
             ${_ROCKE_CLIENT_AOT_INCOMING_PYTHONPATH})
    endif()
    cmake_path(CONVERT "${_ROCKE_CLIENT_AOT_PYTHONPATH}" TO_NATIVE_PATH_LIST
               _ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE)
    set(${OUT_VAR} "${_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE}" PARENT_SCOPE)
endfunction()

# Return the CMake -E env / CTest ENVIRONMENT entries for AOT Python commands.
#
# PYTHONDONTWRITEBYTECODE keeps configure/build/test runs from writing __pycache__
# into source trees, which matters because generated artifacts live in the build
# tree and the source tree should stay reviewable.
function(rocke_client_aot_pythonpath_environment OUT_VAR)
    rocke_client_aot_pythonpath(_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE)
    string(REPLACE ";" "\\;" _ROCKE_CLIENT_AOT_PYTHONPATH_ESCAPED
           "${_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE}")
    set(_ROCKE_CLIENT_AOT_ENV
        "PYTHONPATH=${_ROCKE_CLIENT_AOT_PYTHONPATH_ESCAPED}"
        "PYTHONDONTWRITEBYTECODE=1"
    )
    # Forward the resolved libamd_comgr path; compile_kernel honors ROCKE_COMGR_LIB.
    # comgr is loaded from its assembled tree, where comgr's own RUNPATH resolves
    # its full dependency closure, so no dependency directories are forwarded.
    if(NOT "${_ROCKE_COMGR_LIB}" STREQUAL "")
        list(APPEND _ROCKE_CLIENT_AOT_ENV "ROCKE_COMGR_LIB=${_ROCKE_COMGR_LIB}")
    endif()
    set(${OUT_VAR} "${_ROCKE_CLIENT_AOT_ENV}" PARENT_SCOPE)
endfunction()

# Derive the per-instance HSACO + sidecar output paths from an aot_list.json
# array (one object per instance, keyed by "name"). Returns two lists in
# GEN_VAR (hsaco + sidecar) and SIDE_VAR (sidecar only) in the caller scope.
function(_rocke_client_aot_derive_outputs GEN_VAR SIDE_VAR ARCH_OUTPUT_DIR AOT_LIST)
    file(READ "${AOT_LIST}" _JSON)
    string(JSON _COUNT ERROR_VARIABLE _JSON_ERROR LENGTH "${_JSON}")
    if(_JSON_ERROR)
        message(FATAL_ERROR "Failed to parse ${AOT_LIST}: ${_JSON_ERROR}")
    endif()
    if(_COUNT EQUAL 0)
        message(FATAL_ERROR "No rocKE client AOT instances found in ${AOT_LIST}")
    endif()
    set(_GEN)
    set(_SIDE)
    set(_INDEX 0)
    while(_INDEX LESS _COUNT)
        string(JSON _NAME GET "${_JSON}" ${_INDEX} "name")
        list(APPEND _GEN "${ARCH_OUTPUT_DIR}/${_NAME}.hsaco"
             "${ARCH_OUTPUT_DIR}/${_NAME}.sidecar.json")
        list(APPEND _SIDE "${ARCH_OUTPUT_DIR}/${_NAME}.sidecar.json")
        math(EXPR _INDEX "${_INDEX} + 1")
    endwhile()
    set(${GEN_VAR} "${_GEN}" PARENT_SCOPE)
    set(${SIDE_VAR} "${_SIDE}" PARENT_SCOPE)
endfunction()

# Pack one architecture's HSACO into its own rocke_client_<arch>.kpack + bundle
# manifest and register the install rules. Per-arch files carry the arch in their
# name so a multi-arch build -- or TheRock's per-family artifact union -- merges
# them by simple co-location, with no kpack fusing. Split from
# rocke_client_add_aot_instances to stay within the cmake-lint statement budget.
# Positional: NAME ARCH; the per-instance sidecar outputs follow as ARGN. The
# per-arch build dir and stamp are reconstructed from NAME/ARCH.
function(_rocke_client_aot_pack_and_install NAME ARCH)
    rocke_client_aot_pythonpath_environment(_AOT_ENV)
    set(_SIDECAR_OUTPUTS ${ARGN})
    set(_ARCH_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifacts/${ARCH}/${NAME}")
    set(_BUILD_STAMP "${_ARCH_OUTPUT_DIR}/build.stamp")
    set(_KPACK_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/kpack/${ARCH}")
    set(_KPACK_FILE "${_KPACK_OUTPUT_DIR}/rocke_client_${ARCH}.kpack")
    set(_KPACK_MANIFEST "${_KPACK_OUTPUT_DIR}/rocke_client_${ARCH}.json")
    add_custom_command(
        OUTPUT "${_KPACK_FILE}" "${_KPACK_MANIFEST}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_KPACK_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E env ${_AOT_ENV}
                "${ROCKE_PYENV_PYTHON}" "${_ROCKE_CLIENT_AOT_PACK_TOOL}"
                --artifact-dir "${_ARCH_OUTPUT_DIR}"
                --arch "${ARCH}"
                --out-dir "${_KPACK_OUTPUT_DIR}"
                --engine-build-id "${ROCKE_AOT_ENGINE_BUILD_ID}"
                --llvm-flavor "${ROCKE_AOT_LLVM_FLAVOR}"
                --bundle-schema "${_ROCKE_CLIENT_AOT_BUNDLE_SCHEMA}"
        DEPENDS "${ROCKE_PYENV_STAMP}"
                "${_BUILD_STAMP}"
                ${_SIDECAR_OUTPUTS}
                "${_ROCKE_CLIENT_AOT_PACK_TOOL}"
                "${_ROCKE_CLIENT_AOT_BUNDLE_SCHEMA}"
                ${_ROCKE_CLIENT_AOT_PACKAGE_MODULES}
        VERBATIM
        COMMENT "Pack rocKE client ${ARCH} AOT kpack + bundle manifest"
    )
    add_custom_target("${NAME}_kpack"
        DEPENDS "${_KPACK_FILE}" "${_KPACK_MANIFEST}"
        COMMENT "Pack ${NAME} kpack + manifest"
    )
    add_dependencies("${NAME}_kpack" "${NAME}")
    add_dependencies(rocke_client_aot_kpack "${NAME}_kpack")

    # Per-arch, single source of truth: the .kpack + its manifest only. The
    # manifest carries every selection/launch field; aot_list.json is a
    # build-time input and is intentionally not installed.
    set(_INSTALL_ROOT
        "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/arch_content/rocke/${ARCH}")
    install(FILES "${_KPACK_FILE}" "${_KPACK_MANIFEST}" DESTINATION "${_INSTALL_ROOT}")
endfunction()

# Register one AOT kernel architecture instance list, its kpack packing, and
# install rules.
#
# Required arguments:
#   NAME      Target name and per-arch artifact directory component.
#   ARCH      rocKE architecture component, e.g. gfx942 or gfx1151.
#   ARCH_DIR  Directory holding this arch's aot_list.json
#             (kernels/<arch>/<family>/).
#
# Optional arguments:
#   HANDLER        Family AOT handler module (kernels/common/<family>_aot.py).
#   SCHEMA_DIR     Family schema overlay dir (defaults to the shared AOT schemas).
#   PYTHON_DEPENDS Extra Python files whose edits should rebuild the artifacts.
function(rocke_client_add_aot_instances)
    cmake_parse_arguments(ARG "" "NAME;ARCH;ARCH_DIR;HANDLER;SCHEMA_DIR" "PYTHON_DEPENDS" ${ARGN})
    if(NOT ARG_NAME OR NOT ARG_ARCH OR NOT ARG_ARCH_DIR OR NOT ARG_HANDLER)
        message(FATAL_ERROR
            "rocke_client_add_aot_instances requires NAME, ARCH, ARCH_DIR, and HANDLER"
        )
    endif()

    if(NOT TARGET rocke_client_aot_artifacts OR NOT TARGET rocke_client_aot_kpack)
        message(FATAL_ERROR
            "Create rocke_client_aot_artifacts and rocke_client_aot_kpack before "
            "calling rocke_client_add_aot_instances"
        )
    endif()

    get_filename_component(_ARCH_DIR "${ARG_ARCH_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT IS_DIRECTORY "${_ARCH_DIR}")
        message(FATAL_ERROR "rocKE client AOT arch directory does not exist: ${_ARCH_DIR}")
    endif()
    get_filename_component(_HANDLER "${ARG_HANDLER}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_HANDLER}")
        message(FATAL_ERROR "rocKE client AOT handler does not exist: ${_HANDLER}")
    endif()

    set(_AOT_LIST "${_ARCH_DIR}/aot_list.json")
    if(NOT EXISTS "${_AOT_LIST}")
        message(FATAL_ERROR "rocKE client AOT arch directory is missing aot_list.json: ${_ARCH_DIR}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_AOT_LIST}")

    # Family schema overlay (optional) + freshness globs.
    set(_SCHEMA_ARGS)
    set(_SCHEMA_DEPENDS)
    if(ARG_SCHEMA_DIR)
        get_filename_component(_SCHEMA_DIR "${ARG_SCHEMA_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        set(_SCHEMA_ARGS --schema-dir "${_SCHEMA_DIR}")
        file(GLOB _SCHEMA_DEPENDS CONFIGURE_DEPENDS "${_SCHEMA_DIR}/*.schema.json")
    endif()

    rocke_client_aot_pythonpath_environment(_AOT_ENV)

    set(_ARCH_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifacts/${ARG_ARCH}/${ARG_NAME}")
    set(_BUILD_STAMP "${_ARCH_OUTPUT_DIR}/build.stamp")

    # Derive the generated HSACO and sidecar outputs from the instance names in
    # aot_list.json so Ninja tracks each artifact precisely.
    _rocke_client_aot_derive_outputs(
        _GENERATED_OUTPUTS _SIDECAR_OUTPUTS "${_ARCH_OUTPUT_DIR}" "${_AOT_LIST}")

    # Recreate the artifact directory on every rebuild so removed/renamed
    # instances cannot leave stale HSACO or sidecar files behind.
    add_custom_command(
        OUTPUT "${_BUILD_STAMP}" ${_GENERATED_OUTPUTS}
        COMMAND "${CMAKE_COMMAND}" -E remove_directory "${_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_AOT_LIST}" "${_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E env ${_AOT_ENV}
                "${ROCKE_PYENV_PYTHON}" "${_ROCKE_CLIENT_AOT_BUILD_TOOL}"
                --artifact-dir "${_ARCH_OUTPUT_DIR}"
                --handler "${_HANDLER}"
                ${_SCHEMA_ARGS}
                --arch "${ARG_ARCH}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_BUILD_STAMP}"
        DEPENDS "${ROCKE_PYENV_STAMP}"
                "${_HANDLER}"
                "${_AOT_LIST}"
                "${_ROCKE_CLIENT_AOT_BUILD_TOOL}"
                ${_ROCKE_CLIENT_AOT_PACKAGE_MODULES}
                ${_ROCKE_CLIENT_AOT_KERNEL_SOURCES}
                ${_ROCKE_CLIENT_AOT_PLATFORM_SOURCES}
                ${_ROCKE_CLIENT_AOT_COMMON_SCHEMA_DEPENDS}
                ${_SCHEMA_DEPENDS}
                ${ARG_PYTHON_DEPENDS}
        VERBATIM
        COMMENT "Build rocKE client ${ARG_ARCH} ${ARG_NAME} AOT artifacts"
    )
    add_custom_target("${ARG_NAME}"
        DEPENDS "${_BUILD_STAMP}" ${_GENERATED_OUTPUTS}
        COMMENT "Build ${ARG_NAME} AOT artifacts"
    )
    add_dependencies(rocke_client_aot_artifacts "${ARG_NAME}")

    # kpack pack + bundle manifest + install rules (split out to keep this
    # registration function within the cmake-lint statement budget).
    _rocke_client_aot_pack_and_install(
        "${ARG_NAME}" "${ARG_ARCH}" ${_SIDECAR_OUTPUTS})
endfunction()
