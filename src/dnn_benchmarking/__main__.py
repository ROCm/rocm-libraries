# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Entry point for running dnn_benchmarking as a module.

Usage:
    python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json
"""

import sys

from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())
