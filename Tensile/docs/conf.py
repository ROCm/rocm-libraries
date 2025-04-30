# Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options.
# For a full list, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import re
import sys
from pathlib import Path

from rocm_docs import ROCmDocs

# Add the Tensile module to PYTHON_PATH
sys.path.append(str(Path(__file__).resolve().parent.parent))


def get_semantic_version_from_file(file_path: str, search_prefix: str):
    regex = rf'{search_prefix}"(\d+\.\d+\.\d+)"'
    with open(file_path, "r") as f:
        match = re.search(regex, f.read())
        if not match:
            raise ValueError(
                f"Version is either not found or malformed. File: {file_path}, search prefix: {search_prefix}"
            )
    return match.group(1)


semantic_version = get_semantic_version_from_file("../bump-version.sh", "NEW_VERSION=")
left_nav_title = f"Tensile {semantic_version} Documentation"

# for PDF output on Read the Docs
project = "Tensile Documentation"
author = "Advanced Micro Devices, Inc."
year = datetime.date.today().strftime("%Y")
copyright = f"Copyright (c) {year} Advanced Micro Devices, Inc. All rights reserved."
version = f"{semantic_version}"
release = f"{semantic_version}"

numfig_reset = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "special-members": "__init__, __getitem__",
    "inherited-members": True,
    "show-inheritance": True,
    "imported-members": False,
    "member-order": "bysource",  # bysource: seems unfortunately not to work for Cython modules
}

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "Tensile"

docs_core = ROCmDocs(left_nav_title)
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

extensions += [
    "sphinx.ext.autodoc",  # Automatically create API documentation from Python docstrings
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
