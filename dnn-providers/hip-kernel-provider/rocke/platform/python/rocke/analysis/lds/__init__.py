# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public, CPU-only LDS conflict prediction and serialization."""

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
from .opcodes import OpcodeSpec, UnsupportedLdsOpcodeError, supported_opcodes
from .predict import LdsPredictionError, predict_lds_conflicts
from .registry import UnsupportedLdsTargetError, registered_targets
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
    "LdsPredictionError",
    "ModelValidationError",
    "NormalizedRequest",
    "OpcodeSpec",
    "ProfileIdentity",
    "UnsupportedLdsOpcodeError",
    "UnsupportedLdsTargetError",
    "dumps",
    "loads",
    "predict_lds_conflicts",
    "registered_targets",
    "supported_opcodes",
]
