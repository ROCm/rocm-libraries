#!/usr/bin/env python3
# Stage a frozen-build copy of ck_dsl + ck_dsl_provider for the MicroPython embed.
#
# The source is now directly MicroPython-compatible -- the dataclass fields carry
# explicit `= field()`, displays avoid PEP-448 star-unpacking, file reads use
# `open(str(...))`, env reads use `os.getenv`, and frozen __init__s use
# `object.__setattr__` (enabled in the embed via MICROPY_PY_DELATTR_SETATTR). So
# this step no longer rewrites any code. It only:
#   1. copies the two packages into BUNDLE_DIR (the source trees stay untouched), and
#   2. trims the heavy package __init__.py eager-import roots to empty stubs, so the
#      frozen closure stays bounded -- importing a package must not eagerly pull the
#      whole library, parts of which need deps the embed build does not carry.
import os
import shutil

# Paths supplied by the CMake freeze pipeline (build_embed.py) via the environment.
SRC = os.environ["CK_DSL_SRC"]
PROVIDER_SRC = os.environ["CK_DSL_PROVIDER_SRC"]
BUNDLE_DIR = os.environ["BUNDLE_DIR"]
DST = os.path.join(BUNDLE_DIR, "ck_dsl")
PROVIDER_DST = os.path.join(BUNDLE_DIR, "ck_dsl_provider")

# Heavy package __init__s replaced with empty stubs (eager-import roots).
TRIM_INITS = [
    "__init__.py",
    "helpers/__init__.py",
    "instances/__init__.py",
    "runtime/__init__.py",
    "analysis/__init__.py",
    "benchmark/__init__.py",
]


def main():
    if os.path.exists(BUNDLE_DIR):
        shutil.rmtree(BUNDLE_DIR)
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    # examples/ are standalone demo scripts (ctypes, etc.) never in the frozen
    # closure, so they are excluded from the staged copy.
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "examples"))
    shutil.copytree(
        PROVIDER_SRC, PROVIDER_DST, ignore=shutil.ignore_patterns("__pycache__")
    )

    for rel in TRIM_INITS:
        p = os.path.join(DST, rel)
        if os.path.exists(p):
            with open(p, "w") as f:
                f.write(
                    "# trimmed for the MicroPython frozen build (eager-import root removed)\n"
                )
    print("staged bundle at %s (%d __init__ stubs)" % (BUNDLE_DIR, len(TRIM_INITS)))


if __name__ == "__main__":
    main()
