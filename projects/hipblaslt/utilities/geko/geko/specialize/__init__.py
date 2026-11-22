# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""geko --specialize: one bespoke kernel per traced call identity.

Provides:
- ``calls``: mask-128 trace ingestion into call identities with summed counts.
"""

from . import calls
