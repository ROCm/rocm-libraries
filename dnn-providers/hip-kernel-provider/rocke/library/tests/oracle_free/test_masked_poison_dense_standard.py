# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU masked-token poisoning relation for the shipped dense attention
kernels: the standard-tier cases.

A row of masked attention must not change when K and V change at tokens that are
masked for it. Each case launches the subject on a baseline, on inputs whose
masked tokens carry a large key and value, and on boundary controls, all on raw
FP16/BF16 bits through the architecture's public dense runner. Rows that cannot
see a poisoned token are compared bit for bit; rows that can must be taken over
by it. No expected output is computed.

The only skip is a processor that is not the case's architecture, through the
shared ``rocke.tests.gpu.current_device_arch_in`` gate, so
``ROCKE_REQUIRE_<ARCH>=1`` turns that skip into a failure. On the architecture,
a kernel outside the path the bit equality is derived for fails the test before
any launch; tuning changes inside it keep running.
"""

from __future__ import annotations

import pytest
from rocke.tests.gpu import current_device_arch_in

from ._dense_subject import MASKED_POISON_STANDARD_SUBJECT_CASES
from ._masked_poison import MaskedPoisonCase
from ._masked_poison_lane import check_masked_poison_case

pytestmark = pytest.mark.gpu


def _subject_case_id(item: tuple[str, MaskedPoisonCase]) -> str:
    arch, case = item
    return f"{arch}-{case.case_id}"


@pytest.mark.parametrize(
    "subject_case", MASKED_POISON_STANDARD_SUBJECT_CASES, ids=_subject_case_id
)
def test_masked_token_poisoning(
    subject_case: tuple[str, MaskedPoisonCase],
) -> None:
    """Gate one masked dense case: unaffected rows bit for bit, poisoned rows
    taken over, boundary controls visible."""
    arch, case = subject_case
    current_device_arch_in(
        (arch,), subject=f"the {arch} dense masked-token poisoning gate"
    )
    check_masked_poison_case(arch, case)
