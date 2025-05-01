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
# SPDX-License-Identifier: MIT
################################################################################
from os import name as os_name
from os import environ
from pathlib import Path
from re import search, IGNORECASE
from shlex import split
from subprocess import check_output, STDOUT, CalledProcessError, PIPE, run
from typing import List

from Tensile.Common import SemanticVersion, print1
from .Validators import ToolchainDefaults, validateToolchain

def _invoke(args: List[str], desc: str="", working_dir=None):
  """Invokes a command with the provided arguments in a subprocess.
  Args:
      args: A list of arguments to pass to the subprocess.
      desc: A description of the subprocess invocation.
  Raises:
      RuntimeError: If the subprocess invocation fails.
  Return:
      subprocess output
  """
  #print1(f"{desc}: {' '.join(args)}")
  try:
      out = check_output(args, stderr=STDOUT, cwd=working_dir)
  except:
      raise RuntimeError(
          f"Failed command: {' '.join(args)}"
      )
  return out


def _getVersion(executable: str, versionFlag: str, regex: str) -> str:
    """Compute the version string of a toolchain component.

    Args:
        executable: The toolchain component to check the version of.
        versionFlag: The flag to pass to the executable to get the version.
        regex: pattern used to extract version string.
    Raises:
        RuntimeError: If querying executable for version fails.
    Return:
        Executable version
    """
    executable = validateToolchain(executable)
    args = f'"{executable}" "{versionFlag}"'
    try:
        output = run(args, stdout=PIPE, shell=True).stdout.decode().strip()
        match = search(regex, output, IGNORECASE)
        if match:
            result = match.group(1)
            return SemanticVersion(*[int(c.split("-")[0]) for c in result.split(".")[:3]])
        return None
        raise Exception(f"No version from {output} matches regex {regex}")
    except Exception as e:
        raise RuntimeError(f"Failed to get version when calling {args}: {e}")


def get_rocm_version() -> str:
    """Compute the ROCm version string using hipconfig.

    Raises:
        RuntimeError: If hipconfig fails to execute.
    Return:
        ROCm version string
    """
    try:
        return _getVersion(ToolchainDefaults.HIP_CONFIG, "--version", r'(.+)')
    except:
        return None


class Component:
    """A class used to represent a ROCm toolchain component such as clang++"""

    _rocm_version = get_rocm_version()

    def __init__(self, component_path: Path, version_flag: str="--version", version_regex: str=r"version\s+([\d.]+)"):
        self._version = _getVersion(str(component_path), version_flag, version_regex)
        self._component_path = component_path

    def __str__(self):
        result = f"ROCm {Component._rocm_version.major}.{Component._rocm_version.minor}.{Component._rocm_version.patch} "
        result += f"Component path: {self._component_path} version: {self._version.major}.{self._version.minor}.{self._version.patch}"
        return result

    @property
    def path(self):
        return self._component_path

    @property
    def version(self):
        return self._version

    @property
    def rocm_version(self):
        return Component._rocm_version


class Assembler(Component):
    """
    ROCm assembler class used to build objects from assembly source files.

    ...

    Attributes
    ----------
    version : str
        the version of the component
    rocm_version : str
        the ROCm version
    path : str
        path to assembler

    Methods
    -------
    __call__(self, targetGfx: str, wavefrontSize: int, debug: bool, srcPath: str, destPath: str)
        Invokes the assembler on the provided arguments
    """

    def __init__(self, component_path: Path, co_version: str, debug: bool=False):
        """Constructs instance of assmebler.

        Args:
            assembler_path: The path to the assember.
            co_version: The code object version to use.
        """

        super(Assembler, self).__init__(component_path)
        self._code_object_version = co_version

        self._default_args = [
            *split(environ.get('Tensile_ASM_COMPILER_LAUNCHER', '')),
            str(component_path),
            "-x", "assembler",
            "--target=amdgcn-amd-amdhsa",
            "-g" if debug else "",
            f"-mcode-object-version={co_version}",
            "-c",
        ]

    def __call__(self, targetGfx: str, wavefrontSize: int, srcPath: str, destPath: str):
        """Assemble an assembly source file into an object file.
        Args:
            targetGfx: The target GPU gfx architecture.
            wavefrontSize: The wavefront size to use.
            debug: add debug flags if True.
            srcPath: The path to the assembly source file.
            destPath: The destination path for the generated object file.
        """
        args = self._default_args
        args = [
            *args,
            f"-mcpu={targetGfx}",
            "-mwavefrontsize64" if wavefrontSize == 64 else "-mno-wavefrontsize64",
            srcPath,
            "-o",
            destPath
        ]
        return _invoke(args, "Assembling assembly source code into object file (.s -> .o)")

    @property
    def code_object_version(self):
        return self._code_object_version

class Compiler(Component):
    """
    ROCm compiler class used to build objects from C++ source files.

    ...

    Attributes
    ----------
    version : str
        the version of the component
    rocm_version : str
        the ROCm version
    path : str
        path to compiler

    Methods
    -------
    __call__(self, include_path: str, target_list: List[str], srcPath: str, destPath: str):
        Invokes the compiler on the provided arguments
    """

    def __init__(self, compiler_path: Path, build_id_kind: str, cpu_threads: int, asan_build: bool=False, save_temps: bool=False):
        """Constructs and instance of a Compiler."""
        super(Compiler, self).__init__(compiler_path)

        self.default_args = [
            *split(environ.get("Tensile_CXX_COMPILER_LAUNCHER", "")),
             compiler_path,
            "-shared",
            "-fPIC",
            "-fgpu-rdc",
            "-Xoffload-linker",
            f"--build-id={build_id_kind}",
            "-D__HIP_HCC_COMPAT_MODE__=1",
            "-x", "hip", "-O3",
            "-std=c++17",
            f"-parallel-jobs={cpu_threads}"
        ]

        if asan_build:
            self.default_args.extend(["-fsanitize=address", "-shared-libasan", "-fuse-ld=lld"])
        if save_temps:
            self.default_args.append("--save-temps")
        if os_name == "nt":                                                    # should we use fPIIC on all arches?
            self.default_args.extend(["-fms-extensions", "-fms-compatibility", "-fPIC", "-Wno-deprecated-declarations"])

    def __call__(self, src_paths: List[str], dest_path: str, include_path: str, gfxs: List[str]):
        """Compiles a source file into an object file.

        Args:
            include_path: Path appened to "-I" to directory with required include files.
            target_list: List of offload architectures of the form gfxXYZ e.g. gfx942.
            sercPath: The path to the source file.
            destPath: Path to the object file output during compilation.
        Raises:
            RuntimeError: If the compilation command fails.
        """

        args = list(self.default_args)
        args += [f"--offload-arch={gfx}" for gfx in gfxs]
        args += ["-I", include_path]
        args += src_paths
        args +=["-o", dest_path]
        return _invoke(args, f"Compiling HIP source kernels into objects (.cpp -> .o)")


class Bundler(Component):
    """
    ROCm bundler class used to unbundle objects into code object files.

    ...

    Attributes
    ----------
    version : str
        the version of the component
    rocm_version : str
        the ROCm version
    Methods
    -------
    __call__(self, targetGfx: str, wavefrontSize: int, debug: bool, srcPath: str, destPath: str)
        Invokes the assembler on the provided arguments
    def targets(self, objFile: str):
      returns a list of target triple strings of the form amdgcn-amd--gfx942
    def compress(self, srcPath: str, destPath: str, target: str):
        Compresses a code object file using the provided bundler.
    """

    def __init__(self, bundler_path: Path):
        """Constructs and instance of a Bunder."""
        super(Bundler, self).__init__(bundler_path)

    def targets(self, objFile: str):
        """returns a list of target triple strings of the form amdgcn-amd--gfx942"""
        args = [self._component_path, "--type=o", f"--input={objFile}", "-list"]
        return _invoke(args, f"Listing target triples in object file").decode().split("\n")

    def compress(self, srcPath: str, destPath: str, target: str):
        """Compresses a code object file using the provided bundler.

        Args:
            srcPath: The source path of the code object file to be compressed.
            destPath: The destination path for the compressed code object file.
            gfx: The target GPU architecture.

        Raises:
            RuntimeError: If compressing the code object file fails.
        """
        devnull = "/dev/null" if os_name != "nt" else "NUL"
        args = [
            self._component_path,
            "--compress",
            "--type=o",
            "--bundle-align=4096",
            f"--targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa-unknown-{target}",
            f"--input={devnull}",
            f"--input={srcPath}",
            f"--output={destPath}",
        ]

        return _invoke(args, "Bundling/compressing code object file (.co -> .co)")

    def __call__(self, target: str, srcPath: str, destPath: str):
        """Unbundles source code object files using the Clang Offload Bundler.
        Args:
            target: The target triple, see https://llvm.org/docs/AMDGPUUsage.html#target-triples.
            srcPath: The path to the input object file.
            destPath: The path to the unbundled code object.
        Raises:
            RuntimeError: If unbundling the source code object file fails.
        """
        args = [
            self._component_path,
            "--type=o",
            f"--targets={target}",
            f"--input={srcPath}",
            f"--output={destPath}",
            "--unbundle",
        ]
        return _invoke(args, f"Unbundling source code object file")


class Linker(Component):
    """
    ROCm Linker class used to link objects into code object files.

    ...

    Attributes
    ----------
    version : str
        the version of the component
    rocm_version : str
        the ROCm version
    Methods
    -------
    __call__(self, srcPaths: List[str], destPath: str):
        Invokes the linker on the provided arguments
    """

    def __init__(self, linker_path: Path, build_id_kind: str):
        """Constructs and instance of a Linker."""
        super(Linker, self).__init__(linker_path)
        self.default_args = [
                self._component_path,
                "--target=amdgcn-amd-amdhsa",
                "-Xlinker", f"--build-id={build_id_kind}",
        ]


    def __call__(self, srcPaths: List[str], destPath: str):
        """Links object files into a code object file.

        Args:
            srcPaths: A list of paths to object files.
            destPath: A destination path for the generated code object file.
        Raises:
            RuntimeError: If linker invocation fails.
        """
        if os_name == "nt":
            # Use args file on Windows b/c the command may exceed the limit of 8191 characters
            with open(Path.cwd() / "clang_args.txt", "wt") as file:
                file.write(" ".join(srcPaths).replace('\\', '\\\\'))
            args = [*(self.default_args), "-o", destPath, "@clang_args.txt"]
        else:
            args = [*(self.default_args), *srcPaths, "-o", destPath]
        return _invoke(args, "Linking assembly object files into code object (*.o -> .co)")


class RocObjLs(Component):
    """
    ROCm roc-obj-ls class used to list code objects.

    ...

    Attributes
    ----------
    version : str
        the version of the component
    rocm_version : str
        the ROCm version
    Methods
    -------
    __call__(self, srcPaths: List[str], destPath: str):
        Invokes roc-obj-ls on the provided arguments
    """

    def __init__(self, ls_path: Path):
        """Constructs and instance of roc-obj-ls."""
        # Use `-h` because roc-obj-ls doesn't have a version flag
        super(RocObjLs, self).__init__(ls_path, "-h")

    def __call__(self, sharedObjFile):
        """Lists the code objects in shared object.

        Args:
            sharedObjFile: Name of object file to list.

        Returns:
            List of objects embedded in shared object file.
        """
        args = [self._component_path, sharedObjFile]
        return [e.strip().split()[1:] for e in _invoke(args, f"Listing code objects in shared object", Path(sharedObjFile).parent).decode().split("\n") if e.strip().split()[1:]]


class RocObjExtract(Component):
    """
    ROCm roc-obj-extract class used to get code objects bundle from shared library.

    ...

    Attributes
    ----------
    version : str
        the version of the component
    rocm_version : str
        the ROCm version
    Methods
    -------
    __call__(self, srcPaths: List[str], destPath: str):
        Invokes roc-obj-extract on the provided arguments
    """

    def __init__(self, extract_path: Path):
        """Constructs and instance of roc-obj-extract."""
        # Use `-h` because roc-obj-extract doesn't have a version flag
        # and will print  to stdout "Error: No arguments."
        super(RocObjExtract, self).__init__(extract_path, "-h")

    def __call__(self, filename: str):
        """Extracts code objects from a shared object.

        Args:
            objFile: Name pf object file to extract.

        Returns:
            Code object file.
        """
        args = [self._component_path, filename]
        return _invoke(args, f"Extracting code object.", Path(filename.replace("file:","")).parent)
