# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import shutil
import sys
import re


# We need to add the location of the rocrand Python module to the PATH
# in order to build the documentation of that module
docs_dir_path = pathlib.Path(__file__).parent
python_dir_path = docs_dir_path.parent / 'python' / 'rocrand'
sys.path.append(str(python_dir_path))

with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'rocm_setup_version\( VERSION\s+\"?([0-9.]+)[^0-9.]+', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"rocRAND {version_number} Documentation"
shutil.copy2('../library/src/fortran/README.md', './fortran-api-reference.md')

# for PDF output on Read the Docs
project = "rocRAND Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "rocrand"

# Suppresses "WARNING: toctree directive not expected with external-toc"
# Ideally suppression wouldn't be needed; see sphinx-external-toc#36
suppress_warnings = ["etoc.toctree"]

cpp_id_attributes = ["__forceinline__", "__device__", "__host__", "ROCRANDAPI"]

doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}

html_theme = "rocm_docs_theme"
html_theme_options = {
    "announcement": f"This is ROCm 7.13.0 technology preview release documentation. For the latest production stream release, refer to <a id='rocm-banner' href='https://rocm.docs.amd.com/en/latest/'>ROCm documentation</a>.",
    "flavor": "generic",
    "header_title": f"ROCm™ 7.13.0 Preview",
    "header_link": f"https://rocm.docs.amd.com/en/7.13.0-preview/index.html",
    "version_list_link": "",
    "nav_secondary_items": {
        "GitHub": "https://github.com/ROCm/ROCm",
        "Community": "https://github.com/ROCm/ROCm/discussions",
        "Blogs": "https://rocm.blogs.amd.com/",
        "System and Infra Docs": "https://instinct.docs.amd.com/",
        "Support": "https://github.com/ROCm/ROCm/issues/new/choose",
    },
    "link_main_doc": False,
}

extensions = [
    "rocm_docs", 
    "rocm_docs.doxygen",
]
