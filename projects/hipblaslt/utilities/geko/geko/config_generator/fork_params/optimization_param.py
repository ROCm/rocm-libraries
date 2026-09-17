# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Optional, Tuple

from geko.config_generator.fork_params.param_meta import load_tensile_metadata
from geko.config_generator.shared_utils import (
    ForkParameter,
    GroupDimension,
    SizeContext,
)


class BaseParamBuilder:
    """Common base for classes that create ForkParameter instances.

    Loads Tensile metadata once and provides _make_param for building
    ForkParameter with auto-generated comments (default value + valid range).
    Inherited by both BaseOptimizationParams and BasePostProcessor.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._gt = config["GemmProblem"].gemm_type
        self._meta = load_tensile_metadata()

    def _make_param(
        self,
        name: str,
        values: List[Any],
        comment: Optional[str] = None,
        active: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ForkParameter:
        """Build a ForkParameter.  Auto-generates comment from Tensile
        metadata when *comment* is not provided."""
        if comment is None and name in self._meta:
            m = self._meta[name]
            comment = f" Default Value: {m.default_value} # Range: {m.valid_range}"
        return ForkParameter(
            name=name, values=values, comment=comment or "", active=active, metadata=metadata
        )


def param(fn):
    """Mark a method as a parameter generator."""
    fn._is_param = True
    return fn


def group(fn):
    """Mark a method as a group generator."""
    fn._is_group = True
    return fn


class BaseOptimizationParams(BaseParamBuilder):
    """Base class for generating non-MI fork parameters.

    Instantiated once per GEMM type.

    Decorate per-param methods with @param, per-group methods with @group.
    generate_for_size discovers them automatically via the method resolution
    order (MRO).
    Methods returning None are stripped.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._param_methods = []
        self._group_methods = []
        for name, method in vars(type(self)).items():
            if getattr(method, "_is_param", False):
                self._param_methods.append(name)
            if getattr(method, "_is_group", False):
                self._group_methods.append(name)

    # -----------------------------------------------------------------
    # Orchestrator
    # -----------------------------------------------------------------

    def generate_for_size(
        self, size: Tuple[int, int, int, int]
    ) -> Tuple[Dict[str, ForkParameter], List[GroupDimension]]:
        """Compute all params and groups for a single size.

        *size* is ``(M, N, B, K)``.

        Discovers @param and @group methods defined on the concrete class.
        Results returning None are stripped.  The SizeContext accumulates
        results so downstream methods can reference upstream values.
        """
        M, N, B, K = size
        ctx = SizeContext(M=M, N=N, B=B, K=K)

        # Independent fork parameters — each @param method returns a
        # ForkParameter or None.  Results are accumulated in ctx so
        # downstream params can reference upstream values.
        params: Dict[str, ForkParameter] = {}
        for method_name in self._param_methods:
            result = getattr(self, method_name)(ctx)
            if result is not None:
                params[result.name] = result
                ctx.params[result.name] = result

        # Group dimensions — each @group method returns a GroupDimension
        # (list of correlated param dicts) or None.  Groups are
        # cross-producted with independent params in the final config.
        groups: List[GroupDimension] = []
        for method_name in self._group_methods:
            result = getattr(self, method_name)(ctx)
            grp = result if result is not None else []
            groups.append(grp)
            ctx.groups.append(grp)

        return params, groups
