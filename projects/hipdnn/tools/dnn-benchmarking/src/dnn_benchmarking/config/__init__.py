# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Configuration module for dnn-benchmarking."""

from .benchmark_config import (
    BenchmarkConfig,
    EngineSelection,
    MetricsConfig,
    REFERENCE_PROVIDER_CHOICES,
    ReferenceProviderName,
    SuiteConfig,
    ValidationConfig,
)

__all__ = [
    "BenchmarkConfig",
    "EngineSelection",
    "MetricsConfig",
    "REFERENCE_PROVIDER_CHOICES",
    "ReferenceProviderName",
    "SuiteConfig",
    "ValidationConfig",
]
