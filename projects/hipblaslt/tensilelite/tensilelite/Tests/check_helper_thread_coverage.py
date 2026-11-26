# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check that the coverage lane traces the production helper compiler thread."""

import inspect
from pathlib import Path
import tempfile
from unittest.mock import patch

import coverage

from tensilelite.Toolchain import Source
from tensilelite.create_library.run import helperCompileExecutor


def main():
    config = Path(__file__).resolve().parents[2] / "pyproject.toml"
    sourceLines, firstLine = inspect.getsourcelines(Source.buildSourceCodeObjectFiles)
    cacheHitReturn = firstLine + next(
        index for index, line in enumerate(sourceLines) if line.strip() == "return coPaths"
    )

    with tempfile.TemporaryDirectory(prefix="helper-thread-coverage-") as work:
        root = Path(work)
        measured = coverage.Coverage(
            config_file=str(config),
            source=["tensilelite.Toolchain.Source"],
            data_file=str(root / ".coverage"),
        )
        with patch.object(Source, "HelperKernelCache") as cache:
            cache.return_value.restore.return_value = (True, ["cached.hsaco"])
            measured.start()
            try:
                with helperCompileExecutor() as executor:
                    result = executor.submit(
                        Source.buildSourceCodeObjectFiles,
                        None, None, root / "dest", root / "obj", root / "inc",
                        root / "Kernels.cpp", ["gfx942"],
                    ).result()
            finally:
                measured.stop()

        assert result == ["cached.hsaco"], result
        lines = measured.get_data().lines(str(Path(Source.__file__).resolve())) or []
        assert cacheHitReturn in lines, (
            "Coverage did not trace the helper compiler thread's cache-hit return; "
            "include thread in tool.coverage.run.concurrency"
        )
    print("Helper compiler thread coverage passed")


if __name__ == "__main__":
    main()
