################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from io import TextIOWrapper
from pathlib import Path
from typing import List, Optional, Tuple

from ..Common import printWarning


def _openKernelFiles(
    numMergedFiles: int,
    mergeFiles: bool,
    lazyLoading: bool,
    outputPath: Path,
    kernelFiles: List[str],
) -> Tuple[Optional[TextIOWrapper], Optional[TextIOWrapper]]:
    """Opens kernel source and header files based on merging and loading configurations.

    Decides to open existing files for appending or create new ones based on **numMergedFiles**,
    **mergeFiles**, **lazyLoading**, and the presence of **kernelFiles** with the following branching:
    1. If **numMergedFiles** > 1 and **kernelFiles** is not empty, it opens files based on the first kernel file name.
    2. If **mergeFiles** or **lazyLoading** is enabled, it opens fixed-named files for appending: Kernels.cpp and Kernels.h.
    3. If **mergeFiles** or **lazyLoading** is False, it returns (None, None) to indicate no that
       files are not to be merged, and it is the caller's responsibility to handle file creation.
       This is the default behaviour when TensileCreateLibrary is called during rocBLAS installation.

    Args:
        numMergedFiles: The number of files to merge.
        mergeFiles: Flag indicating if files should be merged.
        lazyLoading: Flag indicating if lazy loading is enabled.
        outputPath: Path to the directory for creating or appending files.
        kernelFiles: Optional list of kernel file names, used for naming if not empty.

    Returns:
        A tuple of `TextIOWrapper` objects for the kernel source and header files, or (None, None)
        if no files need to be opened based on the conditions.

    Notes:
        - Relies on `openFilesBasedOnFirstKernel` and `openFilesWithFixedNames` for file handling.
        - The caller is responsible for closing the returned file objects.
    """
    kernelSourceFile, kernelHeaderFile = None, None
    if numMergedFiles > 1 and len(kernelFiles) > 0:
        kernelSourceFile, kernelHeaderFile = _openFilesBasedOnFirstKernel(kernelFiles)
    elif mergeFiles or lazyLoading:
        kernelSourceFile, kernelHeaderFile = _openFilesWithFixedNames(outputPath)

    return kernelSourceFile, kernelHeaderFile


def _closeKernelFiles(
    kernelSourceFile: Optional[TextIOWrapper], kernelHeaderFile: Optional[TextIOWrapper]
):
    """Closes the kernel source and header file objects if they are open.

    This function checks if the provided file objects for the kernel source and header files are
    not None (they are open) and closes them. It's a cleanup function to ensure that file resources
    are properly released after their use.

    Args:
        kernelSourceFile: The file object for the kernel's source code, or None if it's not open.
        kernelHeaderFile: The file object for the kernel's header code, or None if it's not open.
    """
    if kernelSourceFile:
        kernelSourceFile.close()
    if kernelHeaderFile:
        kernelHeaderFile.close()


def _openFilesWithFixedNames(outputPath: Path) -> Tuple[TextIOWrapper, TextIOWrapper]:
    """Opens files for kernel source code (Kernels.cpp) and header code (Kernels.h).

    Args:
        outputPath: The directory path where the files should be opened.

    Returns:
        A tuple containing two `TextIOWrapper` objects for the source and header files respectively.

    Notes:
        - The source file is named "Kernels.cpp" and the header file is named "Kernels.h".
        - Both files are opened in append mode with UTF-8 encoding.
    """
    srcFilename = Path(outputPath).resolve() / "Kernels.cpp"
    hdrFilename = Path(outputPath).resolve() / "Kernels.h"
    return open(srcFilename, "a", encoding="utf-8"), open(hdrFilename, "a", encoding="utf-8")


def _openFilesBasedOnFirstKernel(kernelFiles: List[str]) -> Tuple[TextIOWrapper, TextIOWrapper]:
    """Opens two files for appending based on the name of the first kernel file in the list.

    Args:
        kernelFiles: A list of kernel file names, where the first element is used to determine the base file name.

    Returns:
        A tuple containing two `TextIOWrapper` objects for appending to the source and header files respectively.

    Notes:
        - The base file name is derived by removing the ".cpp" extension from the first kernel file name.
        - The source and header files are named using the base name with ".cpp" and ".h" extensions respectively.
        - Files are opened in append mode with UTF-8 encoding.
    """
    filename = kernelFiles[0].replace(".cpp", "")
    return open(filename + ".cpp", "a", encoding="utf-8"), open(
        filename + ".h", "a", encoding="utf-8"
    )


class KernelFileContextManager:
    """A context manager for opening kernel files and ensuring they are closed after use.

    This context manager uses the provided parameters to open kernel source and header files
    and automatically closes them when exiting the context. It simplifies the management of
    file resources in operations involving kernel file manipulation.

    Usage:
        with KernelFileContextManager(params, outputPath, kernelFiles) as (kernelSourceFile, kernelHeaderFile):
            # Use kernelSourceFile and kernelHeaderFile here
    """

    def __init__(
        self,
        lazyLoading: bool,
        mergeFiles: bool,
        numMergedFiles: int,
        outputPath: Path,
        kernelFiles: List[str],
    ):
        """Initializes the context manager with the necessary parameters for opening kernel files.

        Args:
            params (dict): A dictionary containing parameters for opening kernel files.
            outputPath (Path): The output path where kernel files are located or will be created.
            kernelFiles (Optional[List[str]]): A list of kernel file names.
        """
        self.numMergedFiles = numMergedFiles
        self.mergeFiles = mergeFiles
        self.lazyLoading = lazyLoading
        self.outputPath = outputPath
        self.kernelFiles = kernelFiles
        self.kernelSourceFile = None
        self.kernelHeaderFile = None

        self._preconditions()

    def __enter__(self):
        """Opens the kernel files based on the provided parameters and returns them.

        Returns:
            A tuple containing the opened kernel source and header files.
        """
        self.kernelSourceFile, self.kernelHeaderFile = _openKernelFiles(
            self.numMergedFiles,
            self.mergeFiles,
            self.lazyLoading,
            self.outputPath,
            self.kernelFiles,
        )
        return self.kernelSourceFile, self.kernelHeaderFile

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures that the kernel files are closed when exiting the context.

        Args:
            exc_type: The exception type if an exception was raised within the context.
            exc_val: The exception value if an exception was raised.
            exc_tb: The traceback if an exception was raised.
        """
        _closeKernelFiles(self.kernelSourceFile, self.kernelHeaderFile)

    def _preconditions(self):
        """Checks if the context manager has been initialized with valid parameters.

        Raises:
            ValueError: If any, or some combination of, the required parameters are invalid.
        """
        if not self.outputPath:
            raise ValueError("Output path cannot be empty.")

        if not self.kernelFiles:
            printWarning("Kernel file context manager opened without any kernel files.")
            if self.numMergedFiles > 1:
                raise ValueError(
                    f"Number of merged files is {self.numMergedFiles}, but no kernel files were "
                    " provided to place the generated code into. Provide at least one kernel file."
                )

        if self.lazyLoading:
            if not self.mergeFiles:
                raise ValueError("If lazy loading is enabled, merge files must be as well")

        if not self.mergeFiles:
            if self.numMergedFiles > 1:
                # This behaviour matches that in Common.assignGlobalParameters.
                # It is enforced again to reconcile the discrepancy in the case that assignGlobalParameters is not called.
                printWarning(
                    f"Merging files is disabled, but {self.numMergedFiles} merged files were requested. "
                    "The number of merged files will be ignored and separate files will be created."
                )
                self.numMergedFiles = 1
