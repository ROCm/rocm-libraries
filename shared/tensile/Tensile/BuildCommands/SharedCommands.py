################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
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

import os
import subprocess
from pathlib import Path
from typing import Union

from ..Common import tPrint


def compressCodeObject(
    coPathSrc: Union[Path, str], coPathDest: Union[Path, str], gfx: str, bundler: str
):
    """Compresses a code object file using the provided bundler.

    Args:
        coPathSrc: The source path of the code object file to be compressed.
        coPathDest: The destination path for the compressed code object file.
        gfx: The target GPU architecture.
        bundler: The path to the Clang Offload Bundler executable.

    Raises:
        RuntimeError: If compressing the code object file fails.
    """
    args = [
        bundler,
        "--compress",
        "--type=o",
        "--bundle-align=4096",
        f"--targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--{gfx}",
        f"--input={os.devnull}",
        f"--input={str(coPathSrc)}",
        f"--output={str(coPathDest)}",
    ]

    tPrint(2, f"Bundling/compressing code objects: {' '.join(args)}")
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        tPrint(2, f"Output: {out}")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error compressing code object via bundling: {err.output}\nFailed command: {' '.join(args)}"
        )
