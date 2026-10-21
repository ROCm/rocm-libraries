"""
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re

from rocm_docs import ROCmDocs

with open("../CMakeLists.txt", encoding="utf-8") as f:
    match = re.search(r".*\bset\(VERSION\s+\"?([0-9.]+)[^0-9.]+", f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"RPP {version_number} documentation"

# Expose the version to the Doxyfile via $(RPP_VERSION) so the API reference
# stays in sync with CMakeLists.txt instead of hardcoding PROJECT_NUMBER.
os.environ["RPP_VERSION"] = version_number

# for PDF output on Read the Docs
project = "RPP documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.enable_api_reference()
docs_core.setup()
docs_core.myst_heading_anchors = 6

external_projects_current_project = "rpp"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Theme-related settings
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm",
    "repository_url": "https://github.com/ROCm/rocm-libraries",
    "path_to_docs": "projects/rpp/docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}
# Generate llms.txt (https://llmstxt.org/)
rocm_docs_generate_llms = True

# Breathe's visit_docimage reads only the file name from a Doxygen \image and
# discards its caption, returning a bare inline image node. Consecutive \image
# directives then run together on one line with nothing identifying them. Wrap
# each image in a figure carrying its caption, which matches how Doxygen's own
# HTML output presents them.
from breathe import parser as _breathe_parser  # noqa: E402
from breathe.renderer.sphinxrenderer import SphinxRenderer, url_re  # noqa: E402
from docutils import nodes as _nodes  # noqa: E402


def _docimage_text(item):
    if isinstance(item, str):
        return item
    value = getattr(item, "value", None)
    if isinstance(value, str):
        return value
    return ""


def _docimage_caption(node):
    # Breathe 5 stores the Doxygen caption attribute on the node. Text inside
    # the image element is a child of the node, which is a list.
    caption = getattr(node, "caption", None)
    if isinstance(caption, str) and caption.strip():
        return caption.strip()

    if isinstance(node, list):
        text = "".join(_docimage_text(item) for item in node).strip()
        if text:
            return text

    # Breathe 4 kept the same text on valueOf_ or content_.
    text = getattr(node, "valueOf_", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts = [
        str(item.value)
        for item in getattr(node, "content_", []) or []
        if getattr(item, "value", None)
    ]
    return "".join(parts).strip()


def _visit_docimage(self, node):
    path_to_image = node.name or ""
    if path_to_image and not url_re.match(path_to_image):
        path_to_image = self.project_info.sphinx_abs_path_to_file(path_to_image)

    caption = _docimage_caption(node)
    image = _nodes.image("", uri=path_to_image, alt=caption)
    if not caption:
        return [image]

    figure = _nodes.figure("", image)
    figure += _nodes.caption(caption, "", _nodes.Text(caption))
    return [figure]


# Replacing visit_docimage leaves the dispatch table pointing at the original.
# Breathe 5 dispatches through node_handlers, keyed by parser node type.
# Breathe 4 dispatches through methods, keyed by the Doxygen element name.
_image_type = getattr(_breathe_parser, "Node_docImageType", None)
if _image_type is not None and hasattr(SphinxRenderer, "node_handlers"):
    SphinxRenderer.node_handlers[_image_type] = _visit_docimage
if hasattr(SphinxRenderer, "methods"):
    SphinxRenderer.methods["docimage"] = _visit_docimage
SphinxRenderer.visit_docimage = _visit_docimage
