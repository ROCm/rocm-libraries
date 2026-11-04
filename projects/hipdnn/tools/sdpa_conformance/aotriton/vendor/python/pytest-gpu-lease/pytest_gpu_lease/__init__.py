# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lease exactly one GPU to each pytest-xdist worker.

See ``pytest_gpu_lease.plugin`` for the fixtures. Discovery happens through the
``pytest11`` entry point, so fixtures are intentionally not re-exported here.
"""

__version__ = '0.1.0'
