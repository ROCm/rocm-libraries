# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Former name of :mod:`.mx_gemm_bench`, kept for existing callers.

Imported, this is the :mod:`.mx_gemm_bench` module object itself, so
attributes set on it reach :func:`main`; run with ``-m``, it runs that main.
"""

from __future__ import annotations

import sys

from . import mx_gemm_bench as _mx_gemm_bench

if __name__ == "__main__":
    sys.exit(_mx_gemm_bench.main())

sys.modules[__name__] = _mx_gemm_bench
