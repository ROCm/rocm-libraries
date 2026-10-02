# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

from rocm_docs import ROCmDocs

# NOTE: hipCCL's umbrella version lives in hipccl3/CMakeLists.txt (the
# actively-developed, default layout) as three separate HIPCCL_VERSION_MAJOR/
# MINOR/PATCH set() calls, rather than a single VERSION_STRING like the
# component projects use - see that file's "hipCCL umbrella version" comment.
with open("../hipccl3/CMakeLists.txt", encoding="utf-8") as f:
    cmake_contents = f.read()


def _version_part(name):
    match = re.search(rf"set\(HIPCCL_VERSION_{name}\s+([0-9]+)\)", cmake_contents)
    if not match:
        raise ValueError(f"HIPCCL_VERSION_{name} not found!")
    return match[1]


version_number = ".".join(
    _version_part(part) for part in ("MAJOR", "MINOR", "PATCH")
)
left_nav_title = f"hipCCL {version_number} Documentation"

# for PDF output on Read the Docs
project = "hipCCL Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

# NOTE: no run_doxygen()/enable_api_reference() here, unlike
# rocprim/hipcub/rocthrust's own conf.py - hipCCL itself defines no new C++
# header API to run doxygen over (only the hipccl::hipccl CMake interface
# target and CPack package). Per-algorithm API reference stays in each
# component's own docs/ site, linked from index.rst instead.
docs_core = ROCmDocs(left_nav_title)
docs_core.setup()

external_projects_current_project = "hipccl"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Theme-related settings
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm",
    "repository_url": "https://github.com/ROCm/rocm-libraries",
    "path_to_docs": "projects/hipccl/docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}

# Suppresses "WARNING: toctree directive not expected with external-toc"
# Ideally suppression wouldn't be needed; see sphinx-external-toc#36
suppress_warnings = ["etoc.toctree"]
