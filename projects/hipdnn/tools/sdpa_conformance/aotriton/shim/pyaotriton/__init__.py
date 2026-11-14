# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stub of the pyaotriton names the vendored suite imports outside `aotriton_flash.py`.

The shim (`shim/aotriton_flash.py`) replaces the module that drives AOTriton; hipDNN runs the
operations. What remains are the status codes and the backend vocabulary the vendored tests read.
"""


class hipError_t:
    hipSuccess = 0
    hipErrorPeerAccessUnsupported = 217
    hipErrorIllegalAddress = 700


def hipGetLastError():
    return hipError_t.hipSuccess


def get_name_suffix():
    return "hipdnn"
