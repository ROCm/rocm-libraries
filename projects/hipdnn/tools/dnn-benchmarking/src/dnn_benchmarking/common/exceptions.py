# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Custom exceptions for dnn-benchmarking."""


class GraphLoadError(Exception):
    """Raised when a graph fails to load or validate."""

    pass


class ExecutionError(Exception):
    """Raised when graph execution fails."""

    pass


class UnsupportedGraphError(ExecutionError):
    """Raised when a graph/engine combination is unsupported rather than broken.

    Distinct from a generic failure: callers (e.g. the suite runner) catch this
    to *skip* a graph the provider/engine cannot handle, while letting genuine
    errors surface. The PyTorch reference normalizes every unsupported condition
    to this type.
    """

    pass


class ValidationError(Exception):
    """Raised when validation fails."""

    pass
