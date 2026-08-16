# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

with open("../CMakeLists.txt", encoding="utf-8") as f:
    match = re.search(r'set\(VERSION_STRING\s+"?([0-9.]+)', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"hipThreads {version_number} Documentation"

# for PDF output on Read the Docs
project = "hipThreads"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

# Only the curated .rst pages are part of the site. Keep Sphinx from globbing the
# local virtualenv, build output, generated Doxygen, and legacy markdown notes.
exclude_patterns = [
    ".venv",
    "_build",
    "_doxygen",
    "doxygen",
    "*.md",
]

extensions = ["rocm_docs", "rocm_docs.doxygen"]

# Theme-related settings
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm",
    "repository_url": "https://github.com/ROCm/rocm-libraries",
    "path_to_docs": "projects/hipthreads/docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}

external_projects_current_project = "hipthreads"

# hipThreads is class-heavy (wthread, mutex, condition_variable, ...). By default
# Breathe shows only the class summary, not its members. Expand documented public
# members so the API reference lists join/detach/lock/etc. (A member without a
# Doxygen comment will not appear; add "undoc-members" here to show those too.)
breathe_default_members = ("members",)

doxygen_root = "doxygen"
doxygen_project = {
    "name": project,
    "path": "doxygen/xml",
}

cpp_id_attributes = [
    "__device__",
    "__host__",
    "__global__",
    "__forceinline__",
    "__shared__",
    "_LIBHIPTHREADS_EXPORTED_FROM_ABI",
    "_LIBHIPTHREADS_TEMPLATE_VIS",
    "_LIBHIPTHREADS_TYPE_VIS",
    "_LIBHIPTHREADS_HIDE_FROM_ABI",
    "_LIBHIPTHREADS_NODISCARD_EXT",
    "_LIBHIPTHREADS_CONSTEXPR",
]
