#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
One-time migration script to convert old full MIT license headers
to the new short SPDX format.
"""

import re
import subprocess
from pathlib import Path

repo_dir = Path(__file__).resolve().parent.parent

# New license text for different file types
NEW_LICENSE_CPP = """// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

"""

NEW_LICENSE_HASH = """# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""

# Patterns to match old license blocks
OLD_LICENSE_CPP_PATTERN = re.compile(
    r"/\*{5,}\s*\*\s*\n"
    r"\s*\*\s*MIT License\s*\n"
    r".*?"
    r"\*{5,}/\s*\n*",
    re.DOTALL,
)

OLD_LICENSE_CPP_SLASHSLASH_PATTERN = re.compile(
    r"^(//\s*MIT License\s*\n"
    r"(//.*\n)*?"
    r"//.*SOFTWARE\.\s*\n*)",
    re.MULTILINE,
)

OLD_LICENSE_HASH_PATTERN = re.compile(
    r"^(#{5,}\s*\n)?"
    r"#\s*\n"
    r"#\s*MIT License\s*\n"
    r"(#.*\n)*?"
    r"#.*SOFTWARE\.\s*\n"
    r"#\s*\n"
    r"(#{5,}\s*\n)?",
    re.MULTILINE,
)

# Files/directories to exclude
EXCLUDE_PATHS = []

EXCLUDE_NAMES = [
    "CODEOWNERS",
    "requirements.txt",
    ".clang-format",
    "LICENSE.md",
    ".gitignore",
    ".gitattributes",
]


def git_ls_tree(directory: Path):
    """List files tracked by git in directory."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Fallback: list all files
        for f in directory.rglob("*"):
            if f.is_file():
                yield str(f.relative_to(directory))
    else:
        for line in result.stdout.strip().split("\n"):
            if line:
                yield line


def should_exclude(path: Path) -> bool:
    """Check if a file should be excluded from processing."""
    path_str = str(path)
    for exclude in EXCLUDE_PATHS:
        if exclude in path_str:
            return True
    if path.name in EXCLUDE_NAMES:
        return True
    return False


def has_spdx_license(text: str) -> bool:
    """Check if file already has the new SPDX license format."""
    return "SPDX-License-Identifier" in text


def has_old_license(text: str) -> bool:
    """Check if file has the old MIT license format."""
    return "MIT License" in text and "AMD ROCm" in text


def get_file_type(path: Path) -> str:
    """Determine file type for license format selection."""
    if path.suffix in [".hpp", ".cpp", ".h", ".c", ".in", ".cl"]:
        return "cpp"
    elif path.suffix in [".py", ".sh", ".cmake", ".txt", ".yml", ".yaml"]:
        return "hash"
    elif path.suffix == "" and path.name not in EXCLUDE_NAMES:
        # Check shebang for scripts without extension
        try:
            with open(path, "r") as f:
                first_line = f.readline()
                if first_line.startswith("#!"):
                    return "hash"
        except:
            pass
    # Handle Dockerfiles and similar
    if "dockerfile" in path.name.lower():
        return "hash"
    return None


def replace_license(text: str, file_type: str) -> str:
    """Replace old license with new SPDX format."""
    if file_type == "cpp":
        # Try the block comment pattern first
        new_text = OLD_LICENSE_CPP_PATTERN.sub("", text)
        if new_text != text:
            return NEW_LICENSE_CPP + new_text.lstrip("\n")
        # Try the // comment pattern
        new_text = OLD_LICENSE_CPP_SLASHSLASH_PATTERN.sub("", text)
        if new_text != text:
            return NEW_LICENSE_CPP + new_text.lstrip("\n")
    elif file_type == "hash":
        # Handle shebang
        lines = text.split("\n")
        shebang = ""
        coding = ""
        rest_start = 0

        if lines and lines[0].startswith("#!"):
            shebang = lines[0] + "\n"
            rest_start = 1

        if len(lines) > rest_start and "coding:" in lines[rest_start]:
            coding = lines[rest_start] + "\n"
            rest_start += 1

        rest = "\n".join(lines[rest_start:])
        new_rest = OLD_LICENSE_HASH_PATTERN.sub("", rest)
        if new_rest != rest:
            return shebang + coding + NEW_LICENSE_HASH + "\n" + new_rest.lstrip("\n")

    return text


def process_file(path: Path, dry_run: bool = False) -> bool:
    """Process a single file, returning True if it was modified."""
    if should_exclude(path):
        return False

    file_type = get_file_type(path)
    if file_type is None:
        return False

    try:
        text = path.read_text()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return False

    if has_spdx_license(text):
        return False

    if not has_old_license(text):
        return False

    new_text = replace_license(text, file_type)
    if new_text == text:
        print(f"Warning: Could not replace license in {path}")
        return False

    if dry_run:
        print(f"Would update: {path}")
    else:
        print(f"Updating: {path}")
        path.write_text(new_text)

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate old MIT license headers to new SPDX format"
    )
    parser.add_argument(
        "dir",
        type=Path,
        default=repo_dir,
        nargs="?",
        help="Directory to process (default: repo root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    args = parser.parse_args()

    updated = 0
    errors = []

    for x in git_ls_tree(args.dir):
        path = args.dir / Path(x)
        if path.is_file():
            try:
                if process_file(path, args.dry_run):
                    updated += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")
                errors.append(path)

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {updated} files")
    if errors:
        print(f"Errors in {len(errors)} files")


if __name__ == "__main__":
    main()
