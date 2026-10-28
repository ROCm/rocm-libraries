# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Errors raised by the kernel generators."""


class GeneratorKernelRejected(ValueError):
    """The generator's emitter rejected a parameter set that passed the generator's own validation."""
