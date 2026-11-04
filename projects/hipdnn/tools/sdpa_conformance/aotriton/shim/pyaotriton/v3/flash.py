# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""AOTriton's backend vocabulary (`FWD_IMPL` / `BWD_IMPL` names).

The plugin refuses those variables, so the names never select anything here; the vendored
`attn_torch_function` imports the tables at module level.
"""


class OpAttnFwdBackend:
    by_index = {0: "triton", 1: "aiter", 2: "flyc"}


class OpAttnBwdBackend:
    by_index = {0: "triton_split", 1: "triton_fuse", 2: "aiter", 3: "flyc"}
