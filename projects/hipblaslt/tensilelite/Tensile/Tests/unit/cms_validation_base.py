################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
from collections.abc import Callable
from typing import Any, Optional
import pytest
from test_CustomSchedule import create_base_kernel, update_kernel, ScheduleInfo
from Tensile.Components.CMSValidator import (
    create_unified_timeline, ValidationContext, validate_timeline,
)
from cms_test_utils import (
    make_mock_id_map, make_mock_mfma_code,
    generate_real_idmap, subset_id_map, _frozen_config_key,
)


class CMSValidationTestBase:
    """
    Base class for CMS validation tests that provides common setup and helper methods.

    Timeline-based tests set ``validator_passes`` to a list of constraint
    functions (e.g. ``[add_local_read_constraints]``).  The base
    ``validation_function`` creates the ValidationContext, calls each pass,
    then runs ``validate_timeline``.

    Structural-only tests (no Timeline needed) override ``validation_function``
    directly and set ``needs_timeline = False``.
    """
    # Structural-only tests (e.g. verify_ascending_order) should set this to False
    # if their schedule format is incompatible with Timeline construction.
    needs_timeline = True

    # Subclasses set this to the list of add_*_constraints functions to run.
    validator_passes: list[Callable[['Timeline', 'ValidationContext'], None]] = []

    # Subclasses opt in to real (kernel-writer-produced) idMaps by setting this
    # to a Solution config dict. When None, validate() uses make_mock_id_map.
    # Parametrized classes can override _resolve_real_id_map_config() to return
    # different configs per parametrization.
    real_id_map_config: Optional[dict] = None

    @pytest.fixture(autouse=True)
    def _inject_isa(self, isa_infrastructure):
        """Stash assembler + isaInfoMap so _get_real_idmap can use them.

        isa_infrastructure returns (None, isaInfoMap, asm) per conftest.py:54.
        Always present at autouse scope; only consumed lazily by validate()
        when real_id_map_config is set, so it costs nothing for mock-path tests
        beyond the one-time fixture init.
        """
        self._isaInfoMap = isa_infrastructure[1]
        self._asm = isa_infrastructure[2]

    def _resolve_real_id_map_config(self):
        """Override in parametrized subclasses to return a per-param config."""
        return self.real_id_map_config

    def _get_real_idmap(self):
        """Lazy generate + class-level cache keyed by config key.

        Cached on the concrete subclass so parametrized variants reuse a
        single kernel-writer run (~140 ms each) when their config is identical.
        """
        config = self._resolve_real_id_map_config()
        if config is None:
            raise RuntimeError(
                "_get_real_idmap called but real_id_map_config is None"
            )
        key = _frozen_config_key(config)
        if not hasattr(self.__class__, '_idmap_cache'):
            self.__class__._idmap_cache = {}
        cache = self.__class__._idmap_cache
        if key not in cache:
            cache[key] = generate_real_idmap(config, self._asm, self._isaInfoMap)
        return cache[key]

    def validation_function(self, sched, ctx, codePathIdx, timeline=None):
        """
        Run each pass in ``self.validator_passes`` on the timeline, then validate.

        Structural-only tests override this method directly.

        Args:
            sched: ScheduleInfo object to validate
            ctx: ValidationContext object
            codePathIdx: Code path index to validate
            timeline: Timeline object for timeline-based validation (None for structural checks)

        Returns:
            Tuple of (status: bool, message: str)
        """
        if not self.validator_passes:
            raise NotImplementedError("Subclasses must set validator_passes or override validation_function")
        for pass_fn in self.validator_passes:
            pass_fn(timeline, ctx)
        message = validate_timeline(timeline)
        if message:
            return False, message
        return True, ""
    
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None):
        """Initialize kernel and compute number of VMFMAs.

        Args:
            method: The test method about to run (passed by pytest automatically).
                    Defaulted to None so subclasses can call super().setup_method(method)
                    even when invoked without it.
            kernel_updates: Optional dict of kernel config overrides.
        """
        self.kernel = create_base_kernel()
        if kernel_updates:
            update_kernel(self.kernel, kernel_updates)
        
        self.num_vmfma = self.kernel["MIWaveTileA"] * self.kernel["MIWaveTileB"]
        self.num_vmfma *= self.kernel["DepthU"] // self.kernel["MatrixInstruction"][2]
        if self.kernel.get("UseF32XEmulation", False):
            self.num_vmfma *= 3

    def validate(
        self,
        optSchedule,
        syncCode,
        numCodePaths: int,
        nglshift: int,
        nllshift: int,
        codePathIdx: int,
        expected_message: Optional[str],
        nllZeroDscnt: bool = False,
        mfmaReorder: list[int] = None,
        snopCode: list[Any] = None,
    ):
        """
        Creates a ScheduleInfo and validates it using the validation function from the subclass.
        
        Args:
            optSchedule: The schedule dictionary mapping instruction types to indices
            syncCode: List of sync instructions (SWaitCnt, SBarrier, etc.)
            numCodePaths: Number of code paths in the schedule
            nglshift: NGL shift value
            nllshift: NLL shift value
            codePathIdx: Code path index to validate
            expected_message: Expected error message (None if validation should pass, str if validation should fail)
            nllZeroDscnt: Whether to use zero dscnt for NLL loop (default: False)
            mfmaReorder: List of MFMA reorder indices
            snopCode: List of SNOP instructions
        """
        if mfmaReorder is None:
            mfmaReorder = []
        if snopCode is None:
            snopCode = []
        
        sched = ScheduleInfo(numCodePaths, self.num_vmfma, optSchedule, syncCode, nglshift, nllshift, nllZeroDscnt, mfmaReorder, snopCode)

        if self._resolve_real_id_map_config() is not None:
            real_id_map, real_mfma_code, _ = self._get_real_idmap()
            # Mock fallback supplies entries for optSchedule keys absent from
            # the registered twin (e.g. PackA3/PackB3 for ForceUnrollSubIter
            # tests). Real entries — including the swap-pack registers we
            # actually need — take precedence.
            mock_fallback = make_mock_id_map(sched, self.kernel)
            id_map = subset_id_map(real_id_map, optSchedule,
                                   syncCode=syncCode, snopCode=snopCode,
                                   fallback_id_map=mock_fallback)
            mfma_code = real_mfma_code[:int(self.num_vmfma)]
        else:
            id_map = make_mock_id_map(sched, self.kernel)
            mfma_code = make_mock_mfma_code(self.num_vmfma)

        ctx = ValidationContext(
            kernel=self.kernel,
            id_map=id_map,
            mfma_code=mfma_code,
            mfma_reorder=mfmaReorder or [],
        )

        timeline = None
        if self.needs_timeline:
            timeline = create_unified_timeline(sched, self.kernel, codePathIdx, id_map=id_map, mfma_code=mfma_code)
        status, message = self.validation_function(sched, ctx, codePathIdx, timeline=timeline)
        
        if expected_message is None:
            assert status, f"Schedule should have passed validation but did not. {message}"
        else:
            assert not status, f"Schedule should have failed but passed."
            assert message == expected_message, f"Expected: {expected_message}, Got: {message}"

