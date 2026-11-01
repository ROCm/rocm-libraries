# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Chunked helper-kernel compile.

The helper kernels of a build (Kernels.cpp) are independent, so they can be split into a few
translation units ("chunks") that compile in parallel and are linked into one code object. Each
chunk is compiled with the compile job of the driver's own command for that chunk (`-###`: the
`clang -cc1` job, with -include-pch and every other flag exactly as the driver passes them), and
the chunk objects are linked and bundled with the driver's own `lld` and `clang-offload-bundler`
jobs. That keeps every kernel's code and metadata identical to the one-piece compile; the
supported route to separate compilation, -fgpu-rdc, changed every kernel's code
(timing_runs/python_kernel_build_src_co_investigation.md, option C).

A kernel's chunk is a hash of its name (chunkOf): content-stable, so adding or removing one
kernel changes one chunk only.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from os import name as os_name
from pathlib import Path
from typing import List, Optional, Sequence, Union

from .Component import Compiler, _invoke, driverJobs


def chunkOf(kernelName: str, chunkCount: int) -> int:
    """The chunk (0 .. chunkCount-1) of the helper kernel kernelName. A hash of the name only,
    so a kernel stays in its chunk whatever other kernels the build has, and the same in every
    process (Python's hash() of a str is salted per process)."""
    digest = hashlib.sha256(kernelName.encode()).digest()
    return int.from_bytes(digest[:8], "big") % chunkCount


def supportsChunkedCompile(compiler: Compiler, targets: Sequence[str]) -> bool:
    """Whether compileHelperChunks can compile for targets with compiler: one target (one compile
    job per chunk), no compiler launcher (the jobs run without it, so ccache would be bypassed),
    no --save-temps (more jobs per chunk, and the temporaries are wanted for Kernels.cpp), and
    not on Windows (the `-###` output is parsed with POSIX shell quoting)."""
    return (
        len(targets) == 1
        and not compiler.launcher
        and "--save-temps" not in compiler.default_args
        and os_name != "nt"
    )


def _jobKind(args: List[str]) -> str:
    tool = Path(args[0]).name
    if args[1:2] == ["-cc1"]:
        return "compile"
    if tool in ("lld", "ld.lld"):
        return "link"
    if tool.startswith("clang-offload-bundler"):
        return "bundle"
    return tool


def _valueAfter(args: List[str], option: str) -> str:
    if args.count(option) != 1 or args.index(option) == len(args) - 1:
        raise RuntimeError(f"expected one {option} <value> in {' '.join(args)}")
    return args[args.index(option) + 1]


def _replaced(args: List[str], old: str, new: List[str]) -> List[str]:
    """args with the single argument equal to old replaced by the arguments new."""
    if args.count(old) != 1:
        raise RuntimeError(f"expected {old} once in {' '.join(args)}")
    index = args.index(old)
    return [*args[:index], *new, *args[index + 1 :]]


def _driverCompileJobs(
    compiler: Compiler,
    includeDir: str,
    target: str,
    srcPath: Union[Path, str],
    objPath: str,
    pchPath: Optional[str],
) -> List[List[str]]:
    """The jobs of the driver's one-piece compile command for srcPath: compile, link, bundle."""
    args = compiler.compileArgs(includeDir, [target], str(srcPath), objPath, pch_path=pchPath)
    jobs = driverJobs(args)
    kinds = [_jobKind(job) for job in jobs]
    if kinds != ["compile", "link", "bundle"]:
        raise RuntimeError(f"unexpected driver jobs {kinds} for {' '.join(map(str, args))}")
    return jobs


def compileHelperChunks(
    compiler: Compiler,
    includeDir: str,
    target: str,
    chunkPaths: Sequence[Union[Path, str]],
    objPath: str,
    pchPath: Optional[str] = None,
):
    """Compiles the chunk sources chunkPaths for target into objPath, the same bundled object the
    one-piece compile of their kernels writes (Compiler.__call__): every chunk's compile job at
    once (one process each), then one link of all chunk objects and the bundle job. The chunk
    objects are written next to objPath.

    Raises:
        RuntimeError: If a job fails or the driver's jobs are not the expected compile, link and
            bundle (the caller then compiles in one piece).
        OSError, CalledProcessError: If the driver cannot be run.
        ValueError: If chunkPaths is empty.
    """
    if not chunkPaths:
        raise ValueError("no chunks to compile")
    objDir = Path(objPath).parent
    compileJobs = []
    chunkObjs = []
    driverObjs = []
    for chunkPath in chunkPaths:
        compileJob, link, bundle = _driverCompileJobs(
            compiler, includeDir, target, chunkPath, objPath, pchPath
        )
        driverObj = _valueAfter(compileJob, "-o")
        chunkObj = str(objDir / (Path(chunkPath).stem + ".o"))
        compileJobs.append(_replaced(compileJob, driverObj, [chunkObj]))
        chunkObjs.append(chunkObj)
        driverObjs.append(driverObj)
        if len(compileJobs) == 1:
            firstLink, firstBundle = link, bundle

    # The link and bundle jobs are the same for every chunk, up to the driver's temporary file
    # names: the first chunk's, linking all chunk objects instead of its one, to a named output.
    linkedObj = str(Path(objPath).with_suffix(".linked.o"))
    driverLinked = _valueAfter(firstLink, "-o")
    linkJob = _replaced(_replaced(firstLink, driverObjs[0], chunkObjs), driverLinked, [linkedObj])
    bundleJob = _replaced(firstBundle, f"-input={driverLinked}", [f"-input={linkedObj}"])
    if f"-output={objPath}" not in bundleJob:
        raise RuntimeError(f"the bundle job does not write {objPath}: {' '.join(bundleJob)}")

    desc = "Compiling a chunk of the HIP source kernels (.cpp -> .o)"
    with ThreadPoolExecutor(max_workers=len(compileJobs)) as executor:
        # list(): wait for every compile, and raise the first failure.
        list(executor.map(lambda job: _invoke(job, desc), compileJobs))
    _invoke(linkJob, "Linking the HIP source kernel chunks (.o -> .o)")
    _invoke(bundleJob, "Bundling the linked HIP source kernels (.o -> .o)")
