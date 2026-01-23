# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

from rocm_docs import ROCmDocs


with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'\bproject\s*\(\s*hipsparse\s+VERSION\s+([0-9.]+)', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"hipSPARSE {version_number} Documentation"

# for PDF output on Read the Docs
project = "hipSPARSE Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.setup()

external_projects_current_project = "hipsparse"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Extend the extensions list with additional extensions
if 'extensions' not in globals():
    extensions = []
extensions.extend([
    'breathe',
    'sphinx_tabs.tabs',  # Add the tabs extension
    'sphinx_design',     # Add the design extension for grid directive
])

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
