# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocke project umbrella:
#   - platform/ : the authoring SDK + engine (former rocKE/).
# The library/ tree is build-time-only Python (kernels/builders/dispatch); it is
# editable-installed and wheeled by the rocke-pyenv / rocke-wheels blocks below,
# not added as a CMake subdirectory.
#
# NOTE: add_subdirectory(platform) is deferred to after the ROCKE_BUILD_PYENV /
# ROCKE_WHEEL_DIR / ROCKE_WHEEL_VERSION declarations below so that
# platform/CMakeLists.txt sees those variables when it registers the wheel
# install() rules.

# ---------------------------------------------------------------------------
# rocke-pyenv: build-time editable Python environment.
#
# The platform `rocke` package and the library packages (kernels/builders/
# dispatch) live under two separate source roots. Rather than scatter sys.path
# surgery across every script/test, we editable-install BOTH into a build-local
# venv. Editable installs point at live source, so code edits need no reinstall;
# the install only re-runs when a pyproject.toml changes (gated by the stamp).
# Build-time Python codegen should use ${ROCKE_PYENV_PYTHON} and depend on the
# stamp. See rocke/BUILDING.md.
# ---------------------------------------------------------------------------
option(ROCKE_BUILD_PYENV "Create the build-local editable rocke Python env" ON)
if(ROCKE_BUILD_PYENV)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(ROCKE_PYENV_DIR "${CMAKE_BINARY_DIR}/rocke-pyenv")
    set(ROCKE_PYENV_STAMP "${CMAKE_BINARY_DIR}/rocke-pyenv.stamp")
    if(WIN32)
        # The venv mirrors the invoking interpreter's filename (python.exe or
        # python3.exe depending on the installation).  Derive it from
        # Python3_EXECUTABLE so the path always matches what venv actually creates.
        cmake_path(GET Python3_EXECUTABLE FILENAME _rocke_py_name)
        set(_rocke_pyenv_python "${ROCKE_PYENV_DIR}/Scripts/${_rocke_py_name}")
    else()
        set(_rocke_pyenv_python "${ROCKE_PYENV_DIR}/bin/python")
    endif()
    set(ROCKE_PYENV_PYTHON "${_rocke_pyenv_python}"
        CACHE FILEPATH "Interpreter for the rocke editable Python env")

    add_custom_command(
        OUTPUT "${ROCKE_PYENV_STAMP}"
        # --system-site-packages so the venv inherits the system ROCm torch
        # (it is never reinstalled here).
        COMMAND "${Python3_EXECUTABLE}" -m venv --system-site-packages --copies "${ROCKE_PYENV_DIR}"
        COMMAND "${ROCKE_PYENV_PYTHON}" -m pip install -q --upgrade pip "setuptools>=61" wheel
        # editable_mode=compat writes a .pth pointing at the source root, so even
        # newly added modules are importable without reinstalling.
        COMMAND "${ROCKE_PYENV_PYTHON}" -m pip install -q
                --config-settings editable_mode=compat
                -e "${CMAKE_CURRENT_SOURCE_DIR}/platform"
        COMMAND "${ROCKE_PYENV_PYTHON}" -m pip install -q --no-deps
                --config-settings editable_mode=compat
                -e "${CMAKE_CURRENT_SOURCE_DIR}/library"
        COMMAND "${CMAKE_COMMAND}" -E touch "${ROCKE_PYENV_STAMP}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/platform/pyproject.toml"
                "${CMAKE_CURRENT_SOURCE_DIR}/library/pyproject.toml"
        COMMENT "rocke-pyenv: editable-installing platform + library"
        VERBATIM)

    add_custom_target(rocke-pyenv ALL
        DEPENDS "${ROCKE_PYENV_STAMP}"
        COMMENT "rocke-pyenv: build-time editable Python environment")

    # -----------------------------------------------------------------------
    # rocke wheels: build pure-Python wheels for the platform and library
    # packages so they can be shipped in the test artifact and pip-installed
    # in CI instead of relying on raw directory copies + PYTHONPATH hacks.
    # -----------------------------------------------------------------------
    set(ROCKE_WHEEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheels"
        CACHE PATH "Output directory for rocke Python wheels")
    set(ROCKE_WHEEL_VERSION "0.1.0" CACHE STRING "rocke Python wheel version")

    # Each wheel depends on ITS OWN Python sources, not just its pyproject.toml.
    # Without these globs a kernel edit never rebuilds the wheel, so the whole
    # downstream staleness chain (wheel -> content digest -> venv reinstall ->
    # repack) is inert in exactly the case it exists for: someone editing a
    # kernel. There are two wheels and they need separate globs -- the kernel
    # bodies live under library/, so wiring only the platform wheel would leave
    # the real case broken. The platform glob takes the package's JSON data too:
    # the arch-domain columns ship in that wheel and drive the lowerer, so
    # regenerating one has to rebuild it like any source edit.
    file(GLOB_RECURSE _rocke_platform_sources CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/platform/python/*.py"
         "${CMAKE_CURRENT_SOURCE_DIR}/platform/python/*.json")
    file(GLOB_RECURSE _rocke_library_sources CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/library/*.py")

    # SOURCE_DATE_EPOCH makes the wheels byte-reproducible. Without it, two
    # builds of identical sources produce wheels with different sha256s (zip
    # entry timestamps come from file mtimes), which was measured here: two
    # back-to-back `pip wheel` runs over an unmodified tree differed. That
    # would make the content-digest staleness chain fire on every mtime bump --
    # a git checkout, a touch, a fresh clone -- and force a full recompile of
    # every kernel for every arch, which is exactly the churn the digest exists
    # to prevent. Pinned to a fixed epoch rather than the commit date so the
    # value does not itself become a rebuild trigger.
    set(_rocke_wheel_env "${CMAKE_COMMAND}" -E env "SOURCE_DATE_EPOCH=1700000000")

    set(_rocke_platform_wheel "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    add_custom_command(
        OUTPUT "${_rocke_platform_wheel}"
        COMMAND ${_rocke_wheel_env}
                "${ROCKE_PYENV_PYTHON}" -m pip wheel --no-deps --no-build-isolation
                --wheel-dir "${ROCKE_WHEEL_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/platform"
        DEPENDS "${ROCKE_PYENV_STAMP}"
                "${CMAKE_CURRENT_SOURCE_DIR}/platform/pyproject.toml"
                ${_rocke_platform_sources}
        COMMENT "Building rocke platform wheel"
        VERBATIM)

    # The library wheel is built from a copy of library/ in a temporary
    # directory, not in place: setuptools writes build/lib/**/*.py next to the
    # sources, which would match the glob above and force a CMake re-run on
    # the next build. (Filtering the list does not help: CONFIGURE_DEPENDS
    # re-checks the unfiltered glob.) See cmake/build_library_wheel.py.
    set(_rocke_library_wheel_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_library_wheel.py")
    set(_rocke_library_wheel "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    add_custom_command(
        OUTPUT "${_rocke_library_wheel}"
        COMMAND ${_rocke_wheel_env}
                "${ROCKE_PYENV_PYTHON}" "${_rocke_library_wheel_script}"
                "${CMAKE_CURRENT_SOURCE_DIR}/library" "${ROCKE_WHEEL_DIR}"
        DEPENDS "${ROCKE_PYENV_STAMP}"
                "${_rocke_library_wheel_script}"
                "${CMAKE_CURRENT_SOURCE_DIR}/library/pyproject.toml"
                ${_rocke_library_sources}
        COMMENT "Building rocke-library wheel"
        VERBATIM)

    add_custom_target(rocke-wheels ALL
        DEPENDS "${_rocke_platform_wheel}" "${_rocke_library_wheel}"
        COMMENT "rocke-wheels: building platform + library Python wheels")
endif()

# ---------------------------------------------------------------------------
# Optional embedding assets for installed library tests.
#
# platform/ remains standalone and knows only its resolved install destination
# and optional SDPA pytest arguments. This umbrella owns integration with files
# outside rocKE because it already joins the platform and library test trees.
# Missing external assets preserve the historical behavior: their checks skip.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
    set(ROCKE_TEST_INSTALL_DIR
        "${CMAKE_INSTALL_BINDIR}/${ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR}")
else()
    set(ROCKE_TEST_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
endif()

if(NOT DEFINED ROCKE_HIPDNN_SDPA_TEMPLATE)
    set(ROCKE_HIPDNN_SDPA_TEMPLATE "" CACHE FILEPATH
        "hipDNN SDPA graph template staged with installed rocKE tests")
endif()
if(NOT DEFINED ROCKE_HIPDNN_SDPA_BUNDLES)
    set(ROCKE_HIPDNN_SDPA_BUNDLES "" CACHE PATH
        "hipDNN golden-bundle directory staged with installed rocKE tests")
endif()
if(NOT DEFINED ROCKE_PROVIDER_PACK_HEADERS)
    set(ROCKE_PROVIDER_PACK_HEADERS "" CACHE PATH
        "Provider pack-header directory staged with installed rocKE tests")
endif()

if(DEFINED ROCM_LIBRARIES_ROOT)
    set(_rocke_embedding_root "${ROCM_LIBRARIES_ROOT}")
else()
    # Preserve automatic staging for the established in-repository embedding.
    set(_rocke_embedding_root "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
endif()

set(_rocke_sdpa_template "${ROCKE_HIPDNN_SDPA_TEMPLATE}")
if(NOT _rocke_sdpa_template)
    set(_rocke_sdpa_template
        "${_rocke_embedding_root}/projects/hipdnn/tools/IngestorGenerator/configs/templates/attention_sdpa_fwd.graph.json")
endif()
set(_rocke_sdpa_bundles "${ROCKE_HIPDNN_SDPA_BUNDLES}")
if(NOT _rocke_sdpa_bundles)
    set(_rocke_sdpa_bundles
        "${_rocke_embedding_root}/dnn-providers/integration-tests/integration-test-bundles")
endif()
set(_rocke_pack_headers "${ROCKE_PROVIDER_PACK_HEADERS}")
if(NOT _rocke_pack_headers)
    set(_rocke_pack_headers
        "${_rocke_embedding_root}/dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs")
endif()

set(_rocke_install_external_assets ON)
if((DEFINED ROCKE_INSTALL_TESTS AND NOT ROCKE_INSTALL_TESTS)
   OR (DEFINED ROCKE_INSTALL_PYTHON_TESTS AND NOT ROCKE_INSTALL_PYTHON_TESTS))
    set(_rocke_install_external_assets OFF)
endif()

set(_rocke_sdpa_graphs "")
if(_rocke_install_external_assets
   AND EXISTS "${_rocke_sdpa_template}"
   AND NOT IS_DIRECTORY "${_rocke_sdpa_template}"
   AND IS_DIRECTORY "${_rocke_sdpa_bundles}")
    file(GLOB_RECURSE _rocke_sdpa_graphs CONFIGURE_DEPENDS
         RELATIVE "${_rocke_sdpa_bundles}" "${_rocke_sdpa_bundles}/*.json")
    # This is the existing hipdnn_assets.sdpa_fwd_golden_graphs() selection.
    list(FILTER _rocke_sdpa_graphs INCLUDE REGEX "^[^/]+/SdpaFwd[^/]*/.+\\.json$")
    list(FILTER _rocke_sdpa_graphs EXCLUDE REGEX "(meta|support|sweep)\\.json$")
    list(FILTER _rocke_sdpa_graphs EXCLUDE REGEX "(^|/)Default/")
endif()
set(_rocke_sdpa_assets_available OFF)
if(_rocke_sdpa_graphs)
    set(_rocke_sdpa_assets_available ON)
    set(ROCKE_LIBRARY_SDPA_PYTEST_ARGS "--rocke-require-hipdnn-assets")
endif()

set(_rocke_dense_geometry_header
    "${_rocke_pack_headers}/Gfx950AttentionDenseGeometry.hpp")
set(_rocke_unified_geometry_header
    "${_rocke_pack_headers}/Gfx950AttentionUnified3DGeometry.hpp")
set(_rocke_pack_headers_available OFF)
if(_rocke_install_external_assets
   AND EXISTS "${_rocke_dense_geometry_header}"
   AND EXISTS "${_rocke_unified_geometry_header}")
    set(_rocke_pack_headers_available ON)
endif()

add_subdirectory(platform)

if(_rocke_sdpa_assets_available)
    set(_rocke_sdpa_hipdnn_dest
        "${ROCKE_TEST_INSTALL_DIR}/tests/library/tests/sdpa/hipdnn")
    install(FILES "${_rocke_sdpa_template}"
            DESTINATION "${_rocke_sdpa_hipdnn_dest}"
            RENAME "attention_sdpa_fwd.graph.json")
    foreach(_rocke_sdpa_graph IN LISTS _rocke_sdpa_graphs)
        cmake_path(GET _rocke_sdpa_graph PARENT_PATH _rocke_sdpa_graph_dir)
        install(FILES "${_rocke_sdpa_bundles}/${_rocke_sdpa_graph}"
                DESTINATION "${_rocke_sdpa_hipdnn_dest}/integration-test-bundles/${_rocke_sdpa_graph_dir}")
    endforeach()
endif()

if(_rocke_pack_headers_available)
    install(FILES "${_rocke_dense_geometry_header}"
                  "${_rocke_unified_geometry_header}"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/library/tests/hkp_packs")
endif()
