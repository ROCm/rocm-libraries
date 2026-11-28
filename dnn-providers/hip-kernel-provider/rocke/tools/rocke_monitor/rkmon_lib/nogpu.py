# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How the checks keep their runs off the GPU, in one place.

HIP_VISIBLE_DEVICES=-1 makes this host's HIP runtime abort (rc 134) as soon as a run imports torch or touches HIP, so the
checks never set it: they unset it and hide the GPU with an empty CUDA_VISIBLE_DEVICES, which the HIP runtime honours by
showing no device. torch's CPU paths need none.
"""

SHELL = "unset HIP_VISIBLE_DEVICES; export CUDA_VISIBLE_DEVICES="  # a bash line for the checks' scripts


def env(base: dict) -> dict:
    """`base` (an environment) with no HIP_VISIBLE_DEVICES and an empty CUDA_VISIBLE_DEVICES."""
    out = {k: v for k, v in base.items() if k != "HIP_VISIBLE_DEVICES"}
    out["CUDA_VISIBLE_DEVICES"] = ""
    return out
