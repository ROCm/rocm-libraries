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
        nllZeroDscnt: bool = False,
        mfmaReorder: list[int] = None,
        snopCode: list[Any] = None,
        expected_failure: Optional[type] = None,
        expected_fields: Optional[dict] = None,
        expected_message: Optional[str] = None,
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
            nllZeroDscnt: Whether to use zero dscnt for NLL loop (default: False)
            mfmaReorder: List of MFMA reorder indices
            snopCode: List of SNOP instructions
            expected_failure: Expected Failure subclass (e.g. OrderInvertedFailure).
                When set, the test asserts isinstance(timeline._last_failure,
                expected_failure) AND that expected_fields (if given) match
                attribute-by-attribute. When unset, the test asserts validation
                passed.
            expected_fields: dict of {attr_name: expected_value} to assert on
                the typed Failure. Use this with expected_failure to bind the
                test to semantic state rather than message text.
            expected_message: Exact-string assertion path for rules that still
                return raw strings (not typed Failures) — e.g.
                `verify_scc_overlap`. Mutually exclusive with expected_failure.
                Prefer expected_failure for any rule that emits typed Failures.

        For tests where the rule raises a Python AssertionError before
        validate_timeline is reached (e.g. preprocessing precondition
        violations), call this without expected_failure inside a
        pytest.raises(AssertionError, match=...) block.
        """
        assert not (expected_failure is not None and expected_message is not None), (
            "pass either expected_failure OR expected_message, not both"
        )
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

        if expected_failure is not None:
            assert not status, "Schedule should have failed but passed."
            failure = getattr(timeline, "_last_failure", None) if timeline else None
            assert failure is not None, (
                f"Expected typed Failure but timeline._last_failure is None. "
                f"The rule may still emit a raw string. Got message: {message!r}"
            )
            assert isinstance(failure, expected_failure), (
                f"Expected {expected_failure.__name__}, got {type(failure).__name__}: {message}"
            )
            for attr_name, expected_value in (expected_fields or {}).items():
                actual_value = getattr(failure, attr_name)
                assert actual_value == expected_value, (
                    f"{type(failure).__name__}.{attr_name}: expected {expected_value!r}, got {actual_value!r}"
                )
            return failure

        if expected_message is not None:
            assert not status, "Schedule should have failed but passed."
            assert message == expected_message, (
                f"Expected: {expected_message!r}, Got: {message!r}"
            )
            return None

        assert status, f"Schedule should have passed validation but did not. {message}"

    @staticmethod
    def assert_order_inverted(failure, *, producer_name, producer_idx,
                              consumer_name, consumer_idx):
        """Assert OrderInvertedFailure carries the expected producer/consumer
        identity AND vmfma_index positions. Pinning positions matters because
        a name-only check passes if any of N similarly-named instructions
        violate ordering — but we want to verify the *specific* schedule slot
        the test set up is the one being flagged."""
        assert failure.producer.name == producer_name, (
            f"producer.name: expected {producer_name!r}, got {failure.producer.name!r}"
        )
        assert failure.producer.issued_at.vmfma_index == producer_idx, (
            f"producer.issued_at.vmfma_index: expected {producer_idx}, "
            f"got {failure.producer.issued_at.vmfma_index}"
        )
        assert failure.consumer.name == consumer_name, (
            f"consumer.name: expected {consumer_name!r}, got {failure.consumer.name!r}"
        )
        assert failure.consumer.issued_at.vmfma_index == consumer_idx, (
            f"consumer.issued_at.vmfma_index: expected {consumer_idx}, "
            f"got {failure.consumer.issued_at.vmfma_index}"
        )

    @staticmethod
    def assert_timing_too_close(failure, *, producer_name, producer_idx,
                                consumer_name, consumer_idx,
                                expected_quad_cycles, actual_quad_cycles):
        """Assert TimingTooCloseFailure carries the expected producer/consumer
        positions AND quad-cycle counts."""
        assert failure.producer.name == producer_name
        assert failure.producer.issued_at.vmfma_index == producer_idx
        assert failure.consumer.name == consumer_name
        assert failure.consumer.issued_at.vmfma_index == consumer_idx
        assert failure.expected_quad_cycles == expected_quad_cycles, (
            f"expected_quad_cycles: expected {expected_quad_cycles}, "
            f"got {failure.expected_quad_cycles}"
        )
        assert failure.actual_quad_cycles == actual_quad_cycles, (
            f"actual_quad_cycles: expected {actual_quad_cycles}, "
            f"got {failure.actual_quad_cycles}"
        )

    @staticmethod
    def assert_wrong_interleaving(failure, *, pack_name, pack_idx,
                                  expected_next_name, expected_next_idx,
                                  actual_next_name, actual_next_idx):
        """Assert WrongInterleavingFailure carries the expected pack +
        expected/actual successor identities and positions."""
        assert failure.pack.name == pack_name
        assert failure.pack.issued_at.vmfma_index == pack_idx
        assert failure.expected_next.name == expected_next_name
        assert failure.expected_next.issued_at.vmfma_index == expected_next_idx
        assert failure.actual_next.name == actual_next_name
        assert failure.actual_next.issued_at.vmfma_index == actual_next_idx

