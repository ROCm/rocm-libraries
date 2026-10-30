# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stand-in for ``hipdnn_frontend``: maps a ``libhipdnn_backend.so`` from ``$FAKE_ROCM/lib``."""

import ctypes
import os

ctypes.CDLL(os.path.join(os.environ["FAKE_ROCM"], "lib", "libhipdnn_backend.so"))
