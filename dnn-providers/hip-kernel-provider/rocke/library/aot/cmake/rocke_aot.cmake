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
# The packer imports rocm_kpack from source (never pip). Resolve the source dir
# from ROCKE_KPACK_PYTHON_DIR (TheRock passes it; overridable standalone) or the
# ambient PYTHONPATH, and gate loud at configure time -- never silently skip.
if(DEFINED ROCKE_KPACK_PYTHON_DIR AND NOT "${ROCKE_KPACK_PYTHON_DIR}" STREQUAL "")
    set(_ROCKE_KPACK_PYTHON_DIR "${ROCKE_KPACK_PYTHON_DIR}")
elseif(DEFINED ENV{ROCKE_KPACK_PYTHON_DIR} AND NOT "$ENV{ROCKE_KPACK_PYTHON_DIR}" STREQUAL "")
    set(_ROCKE_KPACK_PYTHON_DIR "$ENV{ROCKE_KPACK_PYTHON_DIR}")
else()
    set(_ROCKE_KPACK_PYTHON_DIR "")
endif()

if(NOT "${_ROCKE_KPACK_PYTHON_DIR}" STREQUAL "")
    get_filename_component(_ROCKE_KPACK_PYTHON_DIR "${_ROCKE_KPACK_PYTHON_DIR}" ABSOLUTE)
    if(NOT EXISTS "${_ROCKE_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py")
        message(FATAL_ERROR
            "ROCKE_KPACK_PYTHON_DIR does not contain rocm_kpack/kpack.py: "
            "${_ROCKE_KPACK_PYTHON_DIR}"
        )
    endif()
    message(STATUS "rocKE AOT kpack packer: rocm_kpack from ${_ROCKE_KPACK_PYTHON_DIR}")
else()
    # No explicit dir: rocm_kpack must be importable from the ambient environment
    # (local dev). find_spec locates a top-level package without executing it.
    #
    # This checks the system Python3, whereas the packer runs under the not-yet-
    # built ROCKE_PYENV_PYTHON. That is an adequate proxy only because the pyenv
    # is a --system-site-packages venv of this same base interpreter and
    # rocm_kpack is never pip-installed into it, so ambient importability is
    # shared. If that assumption changes, prefer the explicit
    # ROCKE_KPACK_PYTHON_DIR path above, which filesystem-checks rocm_kpack.
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('rocm_kpack') else 1)"
        RESULT_VARIABLE _ROCKE_KPACK_IMPORT_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _ROCKE_KPACK_IMPORT_RESULT EQUAL 0)
        message(FATAL_ERROR
            "rocm_kpack is not importable and ROCKE_KPACK_PYTHON_DIR is unset. "
            "Set -DROCKE_KPACK_PYTHON_DIR=<rocm-systems>/shared/kpack/python (or put "
            "rocm_kpack on PYTHONPATH). The kpack packer requires it."
        )
    endif()
    message(STATUS "rocKE AOT kpack packer: rocm_kpack found on the ambient environment")
endif()

# --- libamd_comgr availability (compile_kernel backend="python") -------------
# rocke.helpers.compile_kernel ctypes-loads libamd_comgr at build time.
# Resolution order:
#   1. explicit ROCKE_COMGR_LIB (cache/env) override,
#   2. find_library(amd_comgr): resolves the system comgr in a standalone build
#      (/opt/rocm) and the staged comgr in a superbuild where amd-comgr is a
#      build dependency of this subproject (TheRock ml-libs declares it),
#   3. otherwise unset, and compile_kernel's own resolver (torch-bundled / rpath)
#      applies at build time.
# The resolved path is forwarded into the AOT build env (see the env helper).
set(_ROCKE_COMGR_LIB "")
if(DEFINED ROCKE_COMGR_LIB AND NOT "${ROCKE_COMGR_LIB}" STREQUAL "")
    set(_ROCKE_COMGR_LIB "${ROCKE_COMGR_LIB}")
elseif(DEFINED ENV{ROCKE_COMGR_LIB} AND NOT "$ENV{ROCKE_COMGR_LIB}" STREQUAL "")
    set(_ROCKE_COMGR_LIB "$ENV{ROCKE_COMGR_LIB}")
else()
    find_library(ROCKE_AMD_COMGR_LIBRARY NAMES amd_comgr)
    if(ROCKE_AMD_COMGR_LIBRARY)
        set(_ROCKE_COMGR_LIB "${ROCKE_AMD_COMGR_LIBRARY}")
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

# libamd_comgr's dependency closure (libclang-cpp, libLLVM, the vendored
# rocm_sysdeps libs, ...) lives in the amd-llvm toolchain, staged separately from
# comgr, so comgr's own $ORIGIN rpath cannot find it at build time and the AOT
# tool's dlopen(libamd_comgr) needs those directories on its search path.
#
# Rather than hardcode the toolchain layout, ask CMake's runtime-dependency
# resolver -- file(GET_RUNTIME_DEPENDENCIES), the machinery behind
# install(RUNTIME_DEPENDENCY_SET) -- to read comgr's actual closure and forward
# the exact directories it resolves. comgr declares amd-llvm as a RUNTIME_DEP, so
# the toolchain (located via the compiler) is where the resolver looks; the set we
# forward is whatever comgr genuinely pulls, and anything left unresolved is
# reported at configure instead of failing as a runtime OSError.
set(_ROCKE_COMGR_DEP_LIBDIRS "")
if(NOT "${_ROCKE_COMGR_LIB}" STREQUAL "")
    # Search roots for the resolver: comgr's own dir plus the amd-llvm toolchain
    # that owns its deps, derived from the compiler. Probe both the POSIX (lib/)
    # and Windows (bin/) layouts and keep whichever exist.
    set(_ROCKE_COMGR_SEARCH_DIRS "")
    get_filename_component(_ROCKE_COMGR_LIB_DIR "${_ROCKE_COMGR_LIB}" DIRECTORY)
    list(APPEND _ROCKE_COMGR_SEARCH_DIRS "${_ROCKE_COMGR_LIB_DIR}")
    if(CMAKE_CXX_COMPILER)
        execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
            OUTPUT_VARIABLE _ROCKE_CLANG_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _ROCKE_CLANG_RD_RESULT
        )
        if(_ROCKE_CLANG_RD_RESULT EQUAL 0 AND _ROCKE_CLANG_RESOURCE_DIR)
            # .../lib/llvm/lib/clang/<ver> -> .../lib/llvm/lib -> .../lib
            get_filename_component(_ROCKE_LLVM_LIB_DIR "${_ROCKE_CLANG_RESOURCE_DIR}" DIRECTORY)
            get_filename_component(_ROCKE_LLVM_LIB_DIR "${_ROCKE_LLVM_LIB_DIR}" DIRECTORY)
            get_filename_component(_ROCKE_LLVM_ROOT "${_ROCKE_LLVM_LIB_DIR}" DIRECTORY)
            get_filename_component(_ROCKE_TC_LIB_DIR "${_ROCKE_LLVM_ROOT}" DIRECTORY)
            get_filename_component(_ROCKE_TC_ROOT "${_ROCKE_TC_LIB_DIR}" DIRECTORY)
            foreach(_cand
                    "${_ROCKE_LLVM_LIB_DIR}"
                    "${_ROCKE_LLVM_ROOT}/bin"
                    "${_ROCKE_TC_LIB_DIR}/rocm_sysdeps/lib"
                    "${_ROCKE_TC_LIB_DIR}/rocm_sysdeps/bin"
                    "${_ROCKE_TC_LIB_DIR}"
                    "${_ROCKE_TC_ROOT}/bin")
                if(IS_DIRECTORY "${_cand}")
                    list(APPEND _ROCKE_COMGR_SEARCH_DIRS "${_cand}")
                endif()
            endforeach()
        endif()
    endif()
    list(REMOVE_DUPLICATES _ROCKE_COMGR_SEARCH_DIRS)

    # file(GET_RUNTIME_DEPENDENCIES) needs a resolver tool; outside install() its
    # platform vars may be unset. Configure them from CMAKE_OBJDUMP when present,
    # else skip cleanly and fall back to the search roots (never fatal).
    set(_ROCKE_GRD_OK FALSE)
    if(CMAKE_OBJDUMP AND EXISTS "${CMAKE_OBJDUMP}")
        if(NOT CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM)
            if(WIN32)
                set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM "windows+pe")
            elseif(APPLE)
                set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM "macos+macho")
            else()
                set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM "linux+elf")
            endif()
            set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL "objdump")
            set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND "${CMAKE_OBJDUMP}")
        endif()
        set(_ROCKE_GRD_OK TRUE)
    endif()

    if(_ROCKE_GRD_OK)
        # POST_EXCLUDE drops OS/system libraries (already on the loader's default
        # path); we only forward toolchain-provided dependency directories.
        # cmake-lint: disable=E1126
        file(GET_RUNTIME_DEPENDENCIES
            LIBRARIES "${_ROCKE_COMGR_LIB}"
            RESOLVED_DEPENDENCIES_VAR _ROCKE_COMGR_RESOLVED
            UNRESOLVED_DEPENDENCIES_VAR _ROCKE_COMGR_UNRESOLVED
            DIRECTORIES ${_ROCKE_COMGR_SEARCH_DIRS}
            POST_EXCLUDE_REGEXES
                "^/usr/" "^/lib/" "^/lib64/" "^/bin/"
                "^/System/" "[Ss]ystem32"
        )
        foreach(_dep IN LISTS _ROCKE_COMGR_RESOLVED)
            get_filename_component(_ROCKE_DEP_DIR "${_dep}" DIRECTORY)
            list(APPEND _ROCKE_COMGR_DEP_LIBDIRS "${_ROCKE_DEP_DIR}")
        endforeach()
        list(REMOVE_DUPLICATES _ROCKE_COMGR_DEP_LIBDIRS)
        if(_ROCKE_COMGR_UNRESOLVED)
            message(STATUS
                "rocKE AOT build: libamd_comgr deps left to the default loader "
                "path: ${_ROCKE_COMGR_UNRESOLVED}")
        endif()
    endif()

    if(_ROCKE_COMGR_DEP_LIBDIRS)
        message(STATUS
            "rocKE AOT build: libamd_comgr dependency dirs (resolved) "
            "${_ROCKE_COMGR_DEP_LIBDIRS}")
    else()
        # Resolver skipped or produced nothing usable: forward the toolchain
        # search roots so the build still resolves comgr's deps.
        set(_ROCKE_COMGR_DEP_LIBDIRS ${_ROCKE_COMGR_SEARCH_DIRS})
        message(STATUS
            "rocKE AOT build: runtime-dependency resolver unavailable; forwarding "
            "toolchain search dirs ${_ROCKE_COMGR_DEP_LIBDIRS}")
    endif()
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
    # Forward the explicit libamd_comgr path (compile_kernel honors ROCKE_COMGR_LIB)
    # and comgr's whole dependency-dir set so the AOT tool's dlopen resolves the
    # full chain on both platforms. POSIX: fold the dirs into LD_LIBRARY_PATH for
    # the dynamic loader. Windows: forward them in ROCKE_COMGR_DEP_DIRS, which the
    # comgr loader feeds to os.add_dll_directory (PATH is not a reliable dependent
    # DLL search path under Python's secure DLL loading). Only when comgr resolved.
    if(NOT "${_ROCKE_COMGR_LIB}" STREQUAL "")
        get_filename_component(_ROCKE_COMGR_LIB_DIR "${_ROCKE_COMGR_LIB}" DIRECTORY)
        list(APPEND _ROCKE_CLIENT_AOT_ENV "ROCKE_COMGR_LIB=${_ROCKE_COMGR_LIB}")
        set(_ROCKE_DEP_DIRS "${_ROCKE_COMGR_LIB_DIR}" ${_ROCKE_COMGR_DEP_LIBDIRS})
        list(REMOVE_DUPLICATES _ROCKE_DEP_DIRS)
        if(WIN32)
            # Escape the ';' list separator so cmake -E env passes a single
            # ';'-joined string (os.pathsep on Windows).
            string(REPLACE ";" "\\;" _ROCKE_DEP_DIRS_ENV "${_ROCKE_DEP_DIRS}")
        else()
            list(JOIN _ROCKE_DEP_DIRS ":" _ROCKE_DEP_DIRS_ENV)
        endif()
        list(APPEND _ROCKE_CLIENT_AOT_ENV "ROCKE_COMGR_DEP_DIRS=${_ROCKE_DEP_DIRS_ENV}")
        if(NOT WIN32)
            set(_ROCKE_LDLIB_DIRS ${_ROCKE_DEP_DIRS})
            if(DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
                list(APPEND _ROCKE_LDLIB_DIRS "$ENV{LD_LIBRARY_PATH}")
            endif()
            list(JOIN _ROCKE_LDLIB_DIRS ":" _ROCKE_LDLIB_PATH)
            list(APPEND _ROCKE_CLIENT_AOT_ENV "LD_LIBRARY_PATH=${_ROCKE_LDLIB_PATH}")
        endif()
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
        "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/rocke/${ARCH}")
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
