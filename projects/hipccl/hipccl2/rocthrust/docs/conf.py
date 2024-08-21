# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

with open("../CMakeLists.txt", encoding="utf-8") as f:
    match = re.search(r"rocm_setup_version\(VERSION\s+\"?([0-9.]+)[^0-9.]+", f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"rocThrust {version_number} Documentation"

# for PDF output on Read the Docs
project = "rocThrust Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

extensions = ["rocm_docs", "rocm_docs.doxygen"]
html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}

external_projects_current_project = "rocthrust"

doxygen_root = "doxygen"
doxygen_project = {
    "name": project,
    "path": "doxygen/xml",
}

cpp_id_attributes = ["__device__", "__host__"]
