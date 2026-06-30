# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocke project umbrella:
#   - library/  : the SDPA product (C-api provider plugin + build-time Python).
#                 Added first to mirror the historical rocke-client-before-rocKE
#                 ordering; both complete before src/ (added by the parent) reads
#                 the staged CTest entries.
#   - platform/ : the authoring SDK + engine (former rocKE/).
add_subdirectory(library)
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
    set(ROCKE_PYENV_PYTHON "${ROCKE_PYENV_DIR}/bin/python"
        CACHE FILEPATH "Interpreter for the rocke editable Python env")

    add_custom_command(
        OUTPUT "${ROCKE_PYENV_STAMP}"
        # --system-site-packages so the venv inherits the system ROCm torch
        # (it is never reinstalled here).
        COMMAND "${Python3_EXECUTABLE}" -m venv --system-site-packages "${ROCKE_PYENV_DIR}"
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
endif()
