# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocke project umbrella:
#   - platform/ : the authoring SDK + engine (former rocKE/).
# The library/ tree is build-time-only Python (kernels/builders/dispatch); it is
# editable-installed and wheeled by the rocke-pyenv / rocke-wheels blocks below,
# not added as a CMake subdirectory.
add_subdirectory(platform)

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

    set(_rocke_platform_wheel "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    add_custom_command(
        OUTPUT "${_rocke_platform_wheel}"
        COMMAND "${ROCKE_PYENV_PYTHON}" -m pip wheel --no-deps --no-build-isolation
                --wheel-dir "${ROCKE_WHEEL_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/platform"
        DEPENDS "${ROCKE_PYENV_STAMP}"
                "${CMAKE_CURRENT_SOURCE_DIR}/platform/pyproject.toml"
        COMMENT "Building rocke platform wheel"
        VERBATIM)

    set(_rocke_library_wheel "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    add_custom_command(
        OUTPUT "${_rocke_library_wheel}"
        COMMAND "${ROCKE_PYENV_PYTHON}" -m pip wheel --no-deps --no-build-isolation
                --wheel-dir "${ROCKE_WHEEL_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/library"
        DEPENDS "${ROCKE_PYENV_STAMP}"
                "${CMAKE_CURRENT_SOURCE_DIR}/library/pyproject.toml"
        COMMENT "Building rocke-library wheel"
        VERBATIM)

    add_custom_target(rocke-wheels ALL
        DEPENDS "${_rocke_platform_wheel}" "${_rocke_library_wheel}"
        COMMENT "rocke-wheels: building platform + library Python wheels")
endif()

# ---------------------------------------------------------------------------
# AOT catalog families: kernel-authoring + build for the throwaway AOT catalog
# engine (src/engines/aot_catalog_engine/). Only meaningful when that consumer is
# built; its producers run under the rocke build interpreter (${ROCKE_PYENV_PYTHON})
# defined above, so this must follow the rocke-pyenv block.
# ---------------------------------------------------------------------------
if(ENABLE_AOT_CATALOG_ENGINE)
    add_subdirectory(aot_catalog)
endif()
