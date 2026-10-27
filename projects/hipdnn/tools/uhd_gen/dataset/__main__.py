#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Entry point for `python -m uhd_gen.dataset`.

Separate from `python -m uhd_gen`, whose `__main__` imports the training stack at module
scope; publishing needs only pandas and a Parquet engine.
"""
from __future__ import annotations

import sys

from .publish import main

if __name__ == "__main__":
    # `add` converts a source into a stored dataset, `export` reads rows back out, and
    # `remove` / `retag` correct what is stored (store.py); anything else publishes.
    if sys.argv[1:2] == ["add"]:
        from .store import main as add_main

        sys.exit(add_main(sys.argv[2:]))
    if sys.argv[1:2] == ["export"]:
        from .store import export_main

        sys.exit(export_main(sys.argv[2:]))
    if sys.argv[1:2] == ["remove"]:
        from .store import remove_main

        sys.exit(remove_main(sys.argv[2:]))
    if sys.argv[1:2] == ["retag"]:
        from .store import retag_main

        sys.exit(retag_main(sys.argv[2:]))
    sys.exit(main())
