# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image's build context as docker reads .dockerignore, for the tests: `patterns(root)`, the patterns of ROOT's
.dockerignore in their order, and `ignored(patterns, path)`, whether those patterns leave PATH (relative to the context)
out of it, are image.dockerignore and image.ignored; `left_out_here(*rels)`, why this run has no copy of a file the
image leaves out, which a test that reads the file skips with.
"""

import os

from rkmon_lib import build_plan, image
from rkmon_lib.image import dockerignore as patterns, ignored  # noqa: F401


def left_out_here(*rels: str) -> str:
    """Why this run has no copy of some of RELS (files, relative to the package), or "" when it has them all: it runs in
    the image, which keeps its build record (build_plan.RECORD_PATH) and leaves out of its build context each entry of
    image.PACKAGE_ONLY, and one of RELS is under such an entry. A test that reads the file skips with this reason. The
    key is the image, not the file being missing: in a checkout without the file, the test that reads it fails.
    """
    tops = sorted({rel.split("/", 1)[0] for rel in rels} & set(image.PACKAGE_ONLY))
    if tops and os.path.isfile(build_plan.RECORD_PATH):
        return f"the image leaves {' and '.join(tops)} out (image.PACKAGE_ONLY)"
    return ""
