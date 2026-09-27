# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from rocm_docs import ROCmDocs

name = "StinkyTofu"
version_number = "0.1.0"

# for PDF output on Read the Docs
project = f"{name}"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(f"{name} {version_number} documentation")
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.setup()

external_projects_current_project = "StinkyTofu"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

"""
html_theme is usually unchanged (rocm_docs_theme).
flavor defines the site header display, select the flavor for the corresponding portals
flavor options: rocm, rocm-docs-home, rocm-blogs, rocm-ds, instinct, ai-developer-hub, local, generic
"""
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "generic",
    "header_title": f"{name} {version_number}",
    "header_link": "https://rocm.docs.amd.com/projects/stinkytofu/en/latest/",
    "nav_secondary_items": {
        "GitHub": "https://github.com/ROCm/rocm-libraries/tree/develop/shared/stinkytofu",
        "Community": "https://github.com/ROCm/rocm-libraries/discussions",
        "Support": "https://github.com/ROCm/rocm-libraries/issues/new/choose",
    },
    "link_main_doc": False,
}
