import re

from rocm_docs import ROCmDocs

with open('../CMakeLists.txt', encoding='utf-8') as f:
    content = f.read()
    major = re.search(r'set\s*\(\s*ROCFFT_VERSION_MAJOR\s+"(\d+)"\s*\)', content)
    minor = re.search(r'set\s*\(\s*ROCFFT_VERSION_MINOR\s+"(\d+)"\s*\)', content)
    patch = re.search(r'set\s*\(\s*ROCFFT_VERSION_PATCH\s+"(\d+)"\s*\)', content)
    
    if not (major and minor and patch):
        raise ValueError("VERSION not found!")
    
    version_number = f"{major[1]}.{minor[1]}.{patch[1]}"

left_nav_title = f"rocFFT {version_number} Documentation"


# for PDF output on Read the Docs
project = "rocFFT Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.setup()

external_projects_current_project = "rocfft"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
