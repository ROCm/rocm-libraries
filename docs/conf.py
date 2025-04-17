# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from rocm_docs import ROCmDocs

# Preprocess the fortran sources with gfortran, because doxygen doesn't seem
# to correctly parse them, even with preprocessing turned on (and with
# upper-case file names as convention)
# preprocessed_out_dir has to be kept in sync with the Doxyfile's INPUT
preprocessed_out_dir = Path("doxygen", "input")
try:
    os.mkdir(preprocessed_out_dir)
except FileExistsError:
    pass

gfortran_exe = shutil.which("gfortran")
if gfortran_exe is None:
    raise RuntimeError("Couldn't find the fortran compiler!")

for filename in glob.glob("../lib/hipfort/*.f"):
    path = Path(filename)
    # -P is to disable embedding line information
    subprocess.check_call(
        [
            gfortran_exe,
            "-E",
            "-cpp",
            "-P",
            "-DUSE_FPOINTER_INTERFACES=1",
            "-UUSE_CUDA_NAMES",
            str(path),
            "-o",
            str(preprocessed_out_dir / path.name),
        ]
    )

with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'.*\bhipfort VERSION\s+\"?([0-9.]+)[^0-9.]+', f.read())
    if not match:
        raise ValueError("HIPFORT_VERSION not found!")
    version_number = match[1]
left_nav_title = f"hipfort {version_number} Documentation"

# for PDF output on Read the Docs
project = "hipfort Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.enable_api_reference()
docs_core.setup()

external_projects_current_project = "hipfort"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# rocm-docs-core might or might not have changed these yet (depending on version),
# and we don't want to wipe their settings if they did
if not "html_theme_options" in globals():
    html_theme_options: Dict[str, Any] = {}
if not "exclude_patterns" in globals():
    exclude_patterns: List[str] = []

html_theme_options["show_navbar_depth"] = 2
exclude_patterns.append("doxygen/input")
