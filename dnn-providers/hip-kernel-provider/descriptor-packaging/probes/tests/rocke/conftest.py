"""Probe tests that pack rocKE content, which live in this directory and nowhere else.

A build with HIPKERNELPROVIDER_ENABLE_ROCKE=OFF makes no rocKE toolchain, and its
`hkp-probe-tools` entry `--ignore`s this directory, so neither this conftest nor any
test beside it is collected there. Everything here may therefore assume the build
asked for rocKE, and packs with the rocKE toolchain (probe_support.pack_root(rocke=True)).
"""

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
