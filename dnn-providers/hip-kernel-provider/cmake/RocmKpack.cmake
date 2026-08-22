# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Single point of resolution for rocm-kpack. Two halves of this provider consume
# it: descriptor-packaging drives the Python packer that writes .kpack archives
# at build time, and the kernel ingestor links the C runtime that reads them back
# at dispatch time.
#
# ONE REPOSITORY, ONE COMMIT, BOTH HALVES -- DO NOT SPLIT THIS PIN. The writer and
# the reader are format-locked only because they come from the same ref, and that
# is the entire mitigation for format skew: nothing negotiates a container version
# at runtime, so a packer and a reader taken from different commits do not report a
# version mismatch. They surface as a corrupt archive, arbitrarily far from the
# cause. Moving HIPKERNELPROVIDER_KPACK_GIT_REF moves both halves together, which
# is the property that keeps them honest; never introduce a second ref for one half.
#
# Nothing may resolve either half through find_package(), for the same reason: an
# installed rocm-kpack is unsafe here, not merely unsupported. Its package config
# carries no commit or SHA and hardcodes PACKAGE_VERSION "0.1.0", so a reader taken
# from it cannot be checked against the packer's ref by this build or by any other.
# Linking one splits the pin and leaves nothing behind that could detect the split.
#
# Scope: the guarantee holds between those two consumers and is not a repo-wide
# property. asm_sdpa_engine independently produces .kpack archives from an unpinned
# clone of a different repository. Nothing is broken today -- those install outside
# the arch_content descriptor tree and no UKD names them, so the ingestor cannot
# reach them -- but routing such a producer through a UKD would falsify the above.
#
# Both resolvers take the same two tiers: an explicit directory override, then the
# shared pinned fetch. The two overrides are counterparts but not interchangeable
# paths -- one names a subdirectory of the other:
#   HIPKERNELPROVIDER_KPACK_PYTHON_DIR   packer only, a rocm-kpack python/ dir
#   HIPKERNELPROVIDER_KPACK_RUNTIME_DIR  reader only, a rocm-kpack source tree root
# Setting only one of them re-opens the skew this module exists to close, so when
# overriding, set both against one tree: RUNTIME_DIR=<root>, PYTHON_DIR=<root>/python.

include_guard(GLOBAL)

# Enforce the pairing the comment above asks for. Leaving it to prose means the
# failure mode is an archive written by one rocm-kpack and read by another --
# silent until a format difference bites at dispatch, and very hard to attribute
# back to a half-applied override. The two names are checked together, once,
# before either tier resolves.
#
# The check is deliberately on "defined and non-empty" rather than on the
# directories agreeing: PYTHON_DIR names <root>/python while RUNTIME_DIR names
# <root>, and demanding a particular relationship between them would forbid the
# legitimate case of a python/ tree relocated beside its source root.
set(_kpack_py_set FALSE)
set(_kpack_rt_set FALSE)
if(DEFINED HIPKERNELPROVIDER_KPACK_PYTHON_DIR AND HIPKERNELPROVIDER_KPACK_PYTHON_DIR)
    set(_kpack_py_set TRUE)
endif()
if(DEFINED HIPKERNELPROVIDER_KPACK_RUNTIME_DIR AND HIPKERNELPROVIDER_KPACK_RUNTIME_DIR)
    set(_kpack_rt_set TRUE)
endif()
if(NOT _kpack_py_set STREQUAL _kpack_rt_set)
    message(FATAL_ERROR
        "rocm-kpack: HIPKERNELPROVIDER_KPACK_PYTHON_DIR (packer) and "
        "HIPKERNELPROVIDER_KPACK_RUNTIME_DIR (reader) must be set together or "
        "not at all. Setting one leaves the other on the pinned fetch, so the "
        "packer and the runtime resolve DIFFERENT rocm-kpack trees -- the exact "
        "format skew the shared pin exists to prevent.\n"
        "  PYTHON_DIR  = '${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}'\n"
        "  RUNTIME_DIR = '${HIPKERNELPROVIDER_KPACK_RUNTIME_DIR}'\n"
        "Set both against one tree: RUNTIME_DIR=<root>, PYTHON_DIR=<root>/python.")
endif()

# The default ref is pinned to a known-good SHA for reproducible builds. Override
# the ref to test a newer kpack: set HIPKERNELPROVIDER_KPACK_GIT_REF to any SHA,
# tag, or branch (e.g. "main"), and HIPKERNELPROVIDER_KPACK_GIT_REPO to fetch from
# a fork. Changing the ref on an existing build tree takes `cmake --fresh`: a plain
# reconfigure leaves ROCM_KPACK_SOURCE_DIR in the cache and rocm_kpack_populate()
# returns early on it, never re-reading the ref. `--fresh` clears that entry, so the
# populate runs again and FetchContent re-resolves the declared ref -- including a
# branch, whose head it picks up on that populate. Pin to a specific SHA rather than
# a branch when the fetched commit needs to be reproducible.
set(HIPKERNELPROVIDER_KPACK_GIT_REPO "https://github.com/ROCm/rocm-kpack.git"
    CACHE STRING "rocm-kpack git repository to fetch (override for a fork).")
set(HIPKERNELPROVIDER_KPACK_GIT_REF "e3483286e751060b3a70b792792cc122632c66e8"
    CACHE STRING "rocm-kpack git ref (SHA, tag, or branch) to fetch. Defaults \
to a pinned SHA for reproducibility; set to a branch or newer SHA to test the \
latest tool.")

# ---------------------------------------------------------------------------
# rocm_kpack_populate()
#   Fetch the pinned rocm-kpack source tree and publish it as
#   ROCM_KPACK_SOURCE_DIR, an internal cache entry so that every directory scope
#   -- the packaging module, the provider, and the test suite that needs an
#   upstream archive fixture -- reads one answer. Idempotent: a later call
#   returns without re-fetching. Only the source tree is produced here;
#   configuring the subproject is rocm_kpack_add_runtime()'s decision.
# ---------------------------------------------------------------------------
function(rocm_kpack_populate)
    if(ROCM_KPACK_SOURCE_DIR AND EXISTS "${ROCM_KPACK_SOURCE_DIR}")
        return()
    endif()

    message(STATUS "rocm-kpack: fetching \
${HIPKERNELPROVIDER_KPACK_GIT_REPO}@${HIPKERNELPROVIDER_KPACK_GIT_REF}")
    include(FetchContent)
    FetchContent_Declare(
        rocm_kpack
        GIT_REPOSITORY "${HIPKERNELPROVIDER_KPACK_GIT_REPO}"
        GIT_TAG "${HIPKERNELPROVIDER_KPACK_GIT_REF}"
    )
    # CMP0169 (CMake >= 3.30) deprecates the single-argument FetchContent_Populate;
    # keep it valid, since a bare source tree is exactly what this function means.
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_GetProperties(rocm_kpack)
    if(NOT rocm_kpack_POPULATED)
        FetchContent_Populate(rocm_kpack)
    endif()

    set(ROCM_KPACK_SOURCE_DIR "${rocm_kpack_SOURCE_DIR}" CACHE INTERNAL
        "rocm-kpack source tree in use by this build.")
endfunction()

# ---------------------------------------------------------------------------
# rocm_kpack_python_dir(<out_var>)
#   Set <out_var> to the rocm-kpack tree's python/ directory, fetching the tree
#   if it is not present yet. The caller decides whether an unusable result is
#   fatal; this function only resolves.
# ---------------------------------------------------------------------------
function(rocm_kpack_python_dir out_var)
    rocm_kpack_populate()
    set(${out_var} "${ROCM_KPACK_SOURCE_DIR}/python" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# rocm_kpack_add_third_party()
#   Make msgpack-cxx and zstd resolvable for rocm-kpack's top-level
#   find_package(... REQUIRED) calls. Neither is discoverable on a stock Windows
#   toolchain -- a bare configure of rocm-kpack dies on the msgpack-cxx lookup --
#   so both are fetched and declared OVERRIDE_FIND_PACKAGE, which redirects those
#   find_package() calls onto the fetched trees instead of failing.
#
#   OVERRIDE_FIND_PACKAGE is not directory-scoped: every later
#   find_package(msgpack-cxx) or find_package(zstd) anywhere in the configure
#   resolves to the versions pinned here. Nothing else is affected today only
#   because add_subdirectory ordering happens to place hip-kernel-provider last and
#   the other msgpack consumer, tensile, is FATAL_ERROR-blocked in the superbuild.
#   Neither is enforced, so a reordering would silently substitute versions in
#   another project.
#
#   Called only by rocm_kpack_add_runtime(); the versions here are the ones that
#   build against the pinned rocm-kpack.
# ---------------------------------------------------------------------------
function(rocm_kpack_add_third_party)
    include(FetchContent)

    # Both projects declare a pre-3.13 cmake_minimum_required, which leaves CMP0077
    # OLD for them: their option() calls would discard a normal variable of the same
    # name. Cache entries are the only settings they honour. The names are all
    # vendor-prefixed, so nothing else in this build reads them.
    set(ZSTD_BUILD_PROGRAMS OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_TESTS OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_SHARED OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_STATIC ON CACHE INTERNAL "")
    set(ZSTD_LEGACY_SUPPORT OFF CACHE INTERNAL "")
    FetchContent_Declare(
        zstd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG v1.5.6
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR build/cmake
        OVERRIDE_FIND_PACKAGE
    )

    set(MSGPACK_USE_BOOST OFF CACHE INTERNAL "")
    set(MSGPACK_BUILD_TESTS OFF CACHE INTERNAL "")
    set(MSGPACK_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(MSGPACK_BUILD_DOCS OFF CACHE INTERNAL "")
    set(MSGPACK_CXX17 ON CACHE INTERNAL "")
    FetchContent_Declare(
        msgpack-cxx
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
        GIT_TAG cpp-7.0.0
        GIT_SHALLOW TRUE
        OVERRIDE_FIND_PACKAGE
    )

    FetchContent_MakeAvailable(zstd msgpack-cxx)

    # Neither dependency belongs in this provider's install tree; both ship their
    # own headers and package config files and would otherwise land beside it. The
    # EXCLUDE_FROM_ALL directory property is what drops a directory out of the
    # parent's install traversal, and it is applied here rather than passed to
    # add_subdirectory() because FetchContent_MakeAvailable() makes that call
    # itself and cannot forward the keyword before CMake 3.28. Their targets stay
    # buildable: rocm_kpack links both, so building it builds them. Confirmed the
    # same way as rocm-kpack's own rules -- install_manifest.txt lists neither
    # msgpack's headers nor either project's package config.
    set_property(DIRECTORY "${zstd_SOURCE_DIR}/build/cmake" PROPERTY EXCLUDE_FROM_ALL TRUE)
    set_property(DIRECTORY "${msgpack-cxx_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)

    # zstd v1.5.6 names its static library libzstd_static and exports no namespaced
    # alias, while rocm-kpack links zstd::libzstd unconditionally; without this the
    # configure fails at generate time. The guard is load-bearing: a shared zstd
    # build does define the namespaced target, and an unconditional alias collides.
    if(NOT TARGET zstd::libzstd AND TARGET libzstd_static)
        add_library(zstd::libzstd ALIAS libzstd_static)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# rocm_kpack_add_runtime()
#   Make the rocm_kpack C library target available to link, resolving in two
#   tiers: a HIPKERNELPROVIDER_KPACK_RUNTIME_DIR source tree, then the pinned
#   fetch. Idempotent. Both tiers build the reader from source, so consumers get
#   one answer either way: link rocm_kpack, include <rocm_kpack/kpack.h>, and the
#   library is static. ROCM_KPACK_SOURCE_DIR names the tree that was used and is
#   always populated, for consumers needing an upstream file such as a test asset.
# ---------------------------------------------------------------------------
function(rocm_kpack_add_runtime)
    if(TARGET rocm_kpack)
        return()
    endif()

    # Both tiers add rocm-kpack's TOP LEVEL, never its runtime/ folder. The top
    # level is where msgpack-cxx and zstd are looked up, and runtime/ links both
    # unconditionally, so descending straight into runtime/ trades the tests'
    # GTest lookup for a missing-target failure.
    if(DEFINED HIPKERNELPROVIDER_KPACK_RUNTIME_DIR AND EXISTS "${HIPKERNELPROVIDER_KPACK_RUNTIME_DIR}")
        set(_kpack_source "${HIPKERNELPROVIDER_KPACK_RUNTIME_DIR}")
        if(NOT EXISTS "${_kpack_source}/runtime/CMakeLists.txt")
            message(FATAL_ERROR
                "rocm-kpack: HIPKERNELPROVIDER_KPACK_RUNTIME_DIR must name a rocm-kpack "
                "source tree root -- the directory that holds runtime/ -- not the "
                "runtime/ folder itself. Got: ${_kpack_source}")
        endif()
        set(ROCM_KPACK_SOURCE_DIR "${_kpack_source}" CACHE INTERNAL
            "rocm-kpack source tree in use by this build.")
        message(STATUS "rocm-kpack: using the source tree at \
HIPKERNELPROVIDER_KPACK_RUNTIME_DIR=${_kpack_source}")
    else()
        rocm_kpack_populate()
        set(_kpack_source "${ROCM_KPACK_SOURCE_DIR}")
    endif()

    rocm_kpack_add_third_party()

    # rocm-kpack's own knobs, as normal variables rather than cache entries:
    # BUILD_TESTING and BUILD_SHARED_LIBS are project-wide names, and forcing them
    # into the shared cache would reach every other target in this build. A normal
    # variable reaches only the directory added below, and rocm-kpack's
    # cmake_minimum_required(3.20) puts CMP0077 in NEW, so its option() calls
    # honour it.
    #
    # BUILD_TESTING defaults ON upstream and pulls find_package(GTest REQUIRED) at
    # the top level; leaving it on fails the configure. BUILD_SHARED_LIBS is OFF so
    # that there is no separate kpack runtime library to place beside the provider
    # -- the reader is linked into it. That also gives the target a PUBLIC
    # ROCM_KPACK_STATIC definition, which is what makes the header's declarations
    # come out right on Windows; consumers get it by linking and need define nothing.
    set(BUILD_TESTING OFF)
    set(BUILD_RUNTIME ON)
    set(BUILD_SHARED_LIBS OFF)

    # EXCLUDE_FROM_ALL is what keeps rocm-kpack out of this provider's install
    # tree; CMAKE_SKIP_INSTALL_RULES is not needed. Upstream's
    # runtime/CMakeLists.txt installs its headers, an export set under the rocm::
    # namespace, and a package config; none of that is ours to ship. Verified
    # against install_manifest.txt rather than assumed: the directory's own
    # cmake_install.cmake is still generated, but the parent never includes it, so
    # the manifest carries no include/rocm_kpack and no lib/cmake/rocm-kpack entry.
    #
    # SYSTEM marks <rocm_kpack/kpack.h> a system include for everything linking the
    # target. Upstream's own sources emit deprecation warnings under the MSVC ABI,
    # and this project's warning options carry -Werror; they are applied PRIVATE per
    # target and so never reach here, but consumers including the header would
    # otherwise be exposed to it.
    add_subdirectory("${_kpack_source}" "${CMAKE_CURRENT_BINARY_DIR}/rocm-kpack" SYSTEM
                     EXCLUDE_FROM_ALL)

    if(NOT TARGET rocm_kpack)
        message(FATAL_ERROR
            "rocm-kpack: ${_kpack_source} configured without defining the rocm_kpack "
            "target. Check that the tree is a rocm-kpack checkout and that its "
            "runtime build was not disabled.")
    endif()
endfunction()
