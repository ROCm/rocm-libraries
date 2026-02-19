# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Validation module for dnn-benchmarking."""

from .comparison import ArrayComparator, ComparisonResult
from .reference_provider import (
    ReferenceOutput,
    ReferenceProvider,
    ReferenceProviderRegistry,
)
from .validator import Validator

# Import providers to register them with the registry
from . import providers  # noqa: F401

__all__ = [
    "ArrayComparator",
    "ComparisonResult",
    "ReferenceOutput",
    "ReferenceProvider",
    "ReferenceProviderRegistry",
    "Validator",
]
