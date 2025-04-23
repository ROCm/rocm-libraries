#####################################################################################
#  The MIT License (MIT)
#
#  Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#####################################################################################

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-docs-home"}

templates_path = ["."]  # Use the current folder for templates

setting_all_article_info = True
all_article_info_os = ["linux"]

with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'.*\brocm_setup_version\(VERSION\s+([0-9.]+)[^0-9.]+',
                      f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]

# for PDF output on Read the Docs
project = "MIGraphX"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

extensions = ["rocm_docs", "rocm_docs.doxygen", "sphinx_collapse"]
external_toc_path = "./sphinx/_toc.yml"
doxygen_root = "doxygen"
doxysphinx_enabled = False
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}

html_title = f"{project} {version_number} documentation"

external_projects_current_project = "amdmigraphx"
