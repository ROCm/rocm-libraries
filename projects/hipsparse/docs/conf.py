# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re


with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'\bproject\s*\(\s*hipsparse\s+VERSION\s+([0-9.]+)', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"hipSPARSE {version_number} Documentation"

# for PDF output on Read the Docs
project = "hipSPARSE Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "hipsparse"


# Extend the extensions list with additional extensions
if 'extensions' not in globals():
    extensions = []
extensions.extend([
    'rocm_docs',
    'breathe',
    'sphinx_tabs.tabs',  # Add the tabs extension
    'sphinx_design',     # Add the design extension for grid directive
])

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

# Configure Breathe (Doxygen integration)
breathe_projects = {"hipsparse": "doxygen/xml"}
breathe_default_project = "hipsparse"

# Configure sphinx-tabs to prevent collapsing when clicking the same tab
sphinx_tabs_disable_tab_closing = True

# Add custom static files
if 'html_static_path' not in globals():
    html_static_path = []
if '_static' not in html_static_path:
    html_static_path.append('_static')

if 'html_js_files' not in globals():
    html_js_files = []
html_js_files.append('custom_tabs.js')
