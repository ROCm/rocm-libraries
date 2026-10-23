# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate pip requirements from the project's dependency groups and uv.lock."""

import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--check", action="store_true", help="Report stale exports without changing files."
)
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]

# Exclude the root package but retain its dependencies in every profile.
command = [
    "uv",
    "export",
    "--format",
    "requirements.txt",
    "--locked",
    "--no-default-groups",
    "--no-emit-project",
    "--no-hashes",
    "--no-header",
]
profiles = {
    "requirements.txt": (
        "Minimal Python requirements to build and use the core project.",
        ["--group", "build"],
    ),
    "requirements-test.txt": (
        "Minimal Python requirements to test a prebuilt installation; "
        "source-only tests require requirements-dev.txt.",
        ["--group", "test"],
    ),
    # Preserve pip's noneditable install; pip does not apply tool.uv build settings.
    "requirements-dev.txt": (
        "Full Python requirements for core builds, tests, and contributor tasks.",
        ["--group", "dev", "--no-editable"],
    ),
}
header = (
    "# Generated from pyproject.toml and uv.lock; do not edit directly.\n"
    "# Regenerate with: python scripts/generate_requirements.py\n\n"
)

# Finish every export before writing, so a failed export cannot partially update files.
exports: dict[Path, str] = {}
for name, (description, flags) in profiles.items():
    result = subprocess.run(
        command + flags, cwd=project, stdout=subprocess.PIPE, text=True
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    exports[project / name] = header + "# " + description + "\n\n" + result.stdout

stale: list[str] = []
for path, content in exports.items():
    if args.check:
        if not path.exists() or path.read_text() != content:
            stale.append(path.name)
    else:
        path.write_text(content)
        print(f"Generated {path.name}")

if stale:
    print("Out-of-date requirements: " + ", ".join(stale))
    print(
        "Run uv lock, then python scripts/generate_requirements.py from the TensileLite directory."
    )
    raise SystemExit(1)
