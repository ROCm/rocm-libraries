# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scikit-build metadata and SDK initialization for installed-library builds."""

import os
from pathlib import Path
import re
import sys

from packaging.version import Version


def sdk_version() -> str | None:
    value = os.environ.get("ROCM_SDK_VERSION")
    return str(Version(value)) if value else None


def dynamic_metadata(
    field: str, settings: dict
) -> str | list[str] | dict[str, list[str]]:
    version = sdk_version()
    if field == "version":
        source = Path(__file__).parent / "src/origami/__init__.py"
        base = re.search(r'^__version__ = "([^"]+)"', source.read_text(), re.MULTILINE)[
            1
        ]
        return base if version is None else f"{base}+rocm{version.replace('+', '.')}"
    if field == "dependencies":
        return [] if version is None else [f"rocm[libraries]=={version}"]
    if field == "optional-dependencies":
        extras = {
            "test": [
                "pytest>=7.0.0",
                "pytest-cov>=4.0.0",
                "pyyaml>=6.0",
                "packaging>=24.0",
            ],
            "torch": ["torch>=2.0.0"],
        }
        if version is not None:
            targets = os.environ.get("ROCM_SDK_TARGETS", "").split(",")
            if not all(re.fullmatch(r"gfx[0-9a-z]+", target) for target in targets):
                raise ValueError(
                    "ROCM_SDK_TARGETS must list SDK targets, e.g. gfx942,gfx950"
                )
            for target in sorted(set(targets)):
                extras[f"device-{target}"] = [f"rocm[device-{target}]=={version}"]
        return extras
    raise ValueError(f"Unsupported metadata field: {field}")


def write_initializer(output: Path) -> None:
    version = sdk_version()
    output.write_text(
        "# Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
        "# SPDX-License-Identifier: MIT\n\n"
        f"SDK_VERSION = {version!r}\n\n"
        "def initialize():\n"
        "    if SDK_VERSION is None:\n"
        "        return\n"
        "    import rocm_sdk\n"
        "    rocm_sdk.initialize_process(\n"
        "        preload_shortnames=['amdhip64', 'origami'],\n"
        "        env_override=False,\n"
        "        check_version=SDK_VERSION,\n"
        "        fail_on_version_mismatch=True,\n"
        "    )\n"
    )


if __name__ == "__main__":
    write_initializer(Path(sys.argv[1]))
