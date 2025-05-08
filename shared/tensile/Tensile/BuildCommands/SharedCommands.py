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
