################################################################################
#
# MIT License
#
# Copyright 2023-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath("."))
import sphinx_rtd_theme  # noqa: F401,E402


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["amsmath", "dollarmath"]

# doxygen_xml_output = ""
# breathe_projects = { "rocroller": doxygen_xml_output }
# breathe_default_project = "rocroller"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The document name of the "root" document, that is, the document that
# contains the root toctree directive. Default is 'index'.
root_doc = "index"

# General information about the project.
project = "rocRoller"
copyright = "2023, Advanced Micro Devices, Inc. All rights reserved"
author = "Advanced Micro Devices, Inc"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
# with open('../../CMakeLists.txt') as file:
#     for line in file:
#         if 'rocm_setup_version' in line:
#             version = re.findall('[0-9.]+', line)[0]
#             break
# # The full version, including alpha/beta/rc tags.
# release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
}

# The "title" for HTML documentation generated with Sphinx's own templates.
# This is appended to the <title> tag of individual pages, and used in the
# navigation bar as the "topmost" element. It defaults
# to '<project> v<revision> documentation'.
html_title = "rocRoller: AMD's assembly kernel generator"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "rocrollerdoc"


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, "rocRoller", "rocRoller Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        root_doc,
        "rocRoller",
        "rocRoller Documentation",
        author,
        "rocRoller",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# The name of the default domain. Can also be None to disable a default
# domain. The default is 'py'.
primary_domain = "cpp"

# -- Options for the C++ domain ------------------------------------------

# A list of strings that the parser additionally should accept as
# attributes. This can for example be used when attributes have been
# #define d for portability.
# cpp_id_attributes = ['ROCROLLER']
