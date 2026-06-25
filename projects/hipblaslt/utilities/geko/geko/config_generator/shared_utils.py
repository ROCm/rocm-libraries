################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

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
