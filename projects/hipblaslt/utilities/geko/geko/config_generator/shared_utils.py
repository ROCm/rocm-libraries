# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ParamMeta:
    """Static metadata for a parameter, sourced from Tensile.
    Loaded once. Provides default values and valid ranges for comments."""
    name: str
    default_value: Any
    valid_range: str


@dataclass
class ForkParameter:
    """A single fork parameter — the universal output unit.
    Used for independent params, group entries, and MI bundles alike."""
    name: str
    values: List[Any] = field(default_factory=list)
    comment: str = ""
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


GroupDimension = List[Dict[str, ForkParameter]]


@dataclass
class SizeContext:
    """Per-size state. Created fresh for each generate_for_size() call.
    Carries dimensions + accumulates results so downstream params
    can reference upstream values (inter-param dependencies)."""
    M: int
    N: int
    B: int
    K: int
    params: Dict[str, ForkParameter] = field(default_factory=dict)
    groups: List[GroupDimension] = field(default_factory=list)


@dataclass
class ConfigEntry:
    """All data for one output config (one size or one merged cluster)."""
    sizes: List[List[int]]
    fork_params: Dict[str, ForkParameter]
    nkernels: int
    mis_per_size: Dict[Tuple, int]
