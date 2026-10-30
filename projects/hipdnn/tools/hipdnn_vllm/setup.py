# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build hipdnn-vllm and its ``_C`` torch extension.

``_C`` is a plain torch C++ extension (host code only, no device kernels)
compiled against:

* torch from the pinned vLLM venv (headers + ABI), and
* the hipDNN frontend, backend and data-SDK headers of the hipDNN install the
  wheel bundles (``HIPDNN_VLLM_HIPDNN_PREFIX``, staged by
  scripts/build_wheel.sh), in the frontend's runtime-load mode
  (``HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND``). ``_C`` links no
  ``libhipdnn_backend``: the bootstrap points the frontend at the bundled copy
  by explicit path, and torch's copy (rocm-sdk-libraries) is never used by
  ``_C``. HIP headers come from the venv's ``rocm-sdk-devel``.

Build inside the pinned venv with ``--no-build-isolation`` (see
scripts/build_wheel.sh). Environment:

* ``HIPDNN_VLLM_HIPDNN_PREFIX``: staged hipDNN install prefix (required).
* ``HIPDNN_VLLM_FAMILY``: TheRock family the bundled backend and providers
  were built for (adds ``.<family>`` to the version's local label).
* ``ROCM_SDK_PATH``: override the rocm-sdk-devel root.
"""

from __future__ import annotations

import ctypes
import glob
import os
import subprocess
import sys
import tomllib

from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.1.0"


def _pins() -> dict:
    with open(os.path.join(HERE, "BUILD_PINS.toml"), "rb") as f:
        return tomllib.load(f)


def _sdk_root() -> str:
    root = os.environ.get("ROCM_SDK_PATH")
    if not root:
        root = subprocess.check_output(["rocm-sdk", "path", "--root"], text=True).strip()
    if not os.path.isdir(os.path.join(root, "include")):
        sys.exit(f"rocm-sdk-devel root {root!r} has no include/; run `rocm-sdk init` first")
    return root


def _dir_containing(root: str, relpath: str) -> str:
    hits = sorted(glob.glob(os.path.join(root, "include", "**", relpath), recursive=True))
    hits += sorted(glob.glob(os.path.join(root, "include", relpath)))
    if not hits:
        sys.exit(f"{relpath} not found under {root}/include")
    hit = hits[0]
    return hit[: -len(relpath)].rstrip("/")


def _hipdnn_prefix() -> str:
    prefix = os.environ.get("HIPDNN_VLLM_HIPDNN_PREFIX")
    if not prefix or not os.path.isdir(os.path.join(prefix, "include")):
        sys.exit(
            "set HIPDNN_VLLM_HIPDNN_PREFIX to a staged hipDNN install (include/ and "
            "lib/libhipdnn_backend.so); scripts/build_wheel.sh builds one"
        )
    return prefix


def _backend_lib(prefix: str) -> str:
    import torch  # noqa: F401  (loads the HIP runtime libraries the backend needs)

    lib = os.path.join(prefix, "lib", "libhipdnn_backend.so")
    if not os.path.isfile(lib):
        sys.exit(f"{lib} not found")
    return lib


def _backend_version(lib: str) -> str:
    handle = ctypes.CDLL(lib, mode=os.RTLD_LAZY | os.RTLD_LOCAL)
    fn = handle.hipdnnVersionString_ext
    fn.restype = ctypes.c_char_p
    return fn().decode()


def _local_version(pins: dict, family: str | None) -> str:
    label = "rocm" + pins["pins"]["rocm"]
    if family:
        label += "." + "".join(ch for ch in family.lower() if ch.isalnum())
    return f"{VERSION}+{label}"


def main() -> None:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    pins = _pins()
    family = os.environ.get("HIPDNN_VLLM_FAMILY") or None
    root = _sdk_root()
    prefix = _hipdnn_prefix()
    backend_version = _backend_version(_backend_lib(prefix))

    # The staged frontend, backend and data_sdk include roots first, so no
    # rocm-sdk-devel hipDNN header (an older API) can be picked up; the SDK root
    # last, for the HIP and nlohmann headers.
    include_dirs = list(
        dict.fromkeys(
            [
                os.path.join(prefix, "include"),
                _dir_containing(prefix, "hipdnn_frontend.hpp"),
                _dir_containing(prefix, "hipdnn_backend.h"),
                _dir_containing(prefix, os.path.join("hipdnn_data_sdk", "utilities", "EngineNames.hpp")),
                os.path.join(root, "include"),
            ]
        )
    )
    ext = CppExtension(
        name="hipdnn_vllm._C",
        sources=[
            "csrc/ops.cpp",
            "csrc/runtime.cpp",
            "csrc/plan_cache.cpp",
            "csrc/stats.cpp",
            "csrc/graph_api.cpp",
            "csrc/graphs/norm.cpp",
            "csrc/graphs/norm_rows.cpp",
            "csrc/graphs/linear.cpp",
            "csrc/graphs/sdpa.cpp",
            "csrc/graphs/encoder_attention.cpp",
            "csrc/graphs/layer_norm.cpp",
            "csrc/graphs/conv.cpp",
            "csrc/graphs/custom_op.cpp",
        ],
        include_dirs=include_dirs,
        define_macros=[
            ("USE_ROCM", "1"),
            ("__HIP_PLATFORM_AMD__", "1"),
            ("HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND", "1"),
            # Defined for every source so all TUs see one Graph definition.
            ("HIPDNN_ENABLE_SDPA", "1"),
            ("HIPDNN_VLLM_BUILD_BACKEND_VERSION", f'"{backend_version}"'),
        ],
        extra_compile_args=["-std=c++20", "-O2", "-Wall"],
        libraries=["dl"],
    )

    p = pins["pins"]
    setup(
        version=_local_version(pins, family),
        install_requires=[
            f"vllm=={p['vllm']}",
            f"torch=={p['torch']}",
            f"rocm=={p['rocm']}",
        ],
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension},
        package_data={
            "hipdnn_vllm": [
                "_build_info.json",
                "backend/*/*",
                "providers/*/*",
                "providers/*/**/*",
            ]
        },
    )


main()
