# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public, CPU-only LDS conflict result model and serialization."""

from .model import (
    SCHEMA_VERSION,
    AccessClassification,
    AccessResult,
    ConflictGroup,
    ConflictSummary,
    Diagnostic,
    DiagnosticSeverity,
    GroupKind,
    LdsAccess,
    LdsConflictResult,
    ModelValidationError,
    NormalizedRequest,
    ProfileIdentity,
)
from .serialization import dumps, loads

__all__ = [
    "SCHEMA_VERSION",
    "AccessClassification",
    "AccessResult",
    "ConflictGroup",
    "ConflictSummary",
    "Diagnostic",
    "DiagnosticSeverity",
    "GroupKind",
    "LdsAccess",
    "LdsConflictResult",
    "ModelValidationError",
    "NormalizedRequest",
    "ProfileIdentity",
    "dumps",
    "loads",
]
