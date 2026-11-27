# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``parse_kernel_arguments`` reads each kernel's argument list, and
``parse_kernel_fields`` its other scalar values, from the code-object metadata
``llvm-readelf --notes`` prints.

The fixture is the shape the tool prints for an AMDHSA code object: a list of
kernel entries, each with a nested ``.args`` list whose entries the compiler may
tag (``!str N``), followed by kernel-level keys and a scalar list. The COMGR
check that the packer's layout matches these offsets is in
``tests/runtime/test_packing.py``.
"""

from __future__ import annotations

import pytest
from rocke.analysis import KernelArgument, parse_kernel_arguments, parse_kernel_fields

_NOTES = """\
Displaying notes found in: .note
  Owner                Data size 	Description
  AMDGPU               0x000002b0	NT_AMDGPU_METADATA (AMDGPU Metadata)
    AMDGPU Metadata:
        ---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           X
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           !str N
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
    .group_segment_fixed_size: 256
    .name:           first
    .reqd_workgroup_size:
      - 64
      - 1
      - 1
    .symbol:         first.kd
  - .args:
      - .name:           alpha
        .offset:         0
        .size:           4
        .value_kind:     by_value
    .name:           second
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...
"""


def test_each_kernel_gets_its_own_arguments_in_order():
    assert parse_kernel_arguments(_NOTES) == {
        "first": (
            KernelArgument("X", "global_buffer", 0, 8),
            KernelArgument("N", "by_value", 8, 4),
            KernelArgument(None, "hidden_block_count_x", 16, 4),
        ),
        "second": (KernelArgument("alpha", "by_value", 0, 4),),
    }


def test_each_kernel_gets_its_own_scalar_fields():
    # .reqd_workgroup_size is a nested list, so it is not a field; an empty
    # quoted string is.
    notes = _NOTES.replace(
        "    .name:           second\n",
        "    .name:           second\n    .language: ''\n",
    )
    assert parse_kernel_fields(notes) == {
        "first": {
            "agpr_count": "0",
            "group_segment_fixed_size": "256",
            "symbol": "first.kd",
        },
        "second": {"language": ""},
    }


def test_a_kernel_without_a_name_has_no_fields():
    notes = _NOTES.replace("    .name:           second\n", "")
    with pytest.raises(
        ValueError, match=r"^a kernel in the code-object metadata has no \.name$"
    ):
        parse_kernel_fields(notes)


def test_text_without_kernel_metadata_has_no_kernels():
    assert parse_kernel_arguments("no notes here\n") == {}


def test_a_kernel_without_a_name_is_refused():
    notes = _NOTES.replace("    .name:           second\n", "")
    with pytest.raises(
        ValueError, match=r"^a kernel in the code-object metadata has no \.name$"
    ):
        parse_kernel_arguments(notes)


def test_an_argument_without_an_offset_is_refused():
    notes = _NOTES.replace(
        "        .offset:         0\n        .size:           4\n",
        "        .size:           4\n",
    )
    with pytest.raises(
        ValueError, match=r"^an argument of kernel 'second' has no \.offset$"
    ):
        parse_kernel_arguments(notes)


def test_an_argument_missing_several_keys_names_them_all():
    notes = _NOTES.replace(
        "      - .name:           alpha\n        .offset:         0\n"
        "        .size:           4\n        .value_kind:     by_value\n",
        "      - .name:           alpha\n",
    )
    with pytest.raises(
        ValueError,
        match=r"^an argument of kernel 'second' has no \.offset, \.size, \.value_kind$",
    ):
        parse_kernel_arguments(notes)


def test_a_kernel_list_that_does_not_start_with_an_entry_is_refused():
    notes = _NOTES.replace("  - .agpr_count:     0\n", "    .agpr_count:     0\n")
    with pytest.raises(
        ValueError, match=r"^amdhsa.kernels does not start with a list entry: "
    ):
        parse_kernel_arguments(notes)
