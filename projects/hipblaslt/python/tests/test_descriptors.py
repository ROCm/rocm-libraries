# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Descriptors and the table-driven attribute API. No GPU needed.

hipBLASLt checks only that an attribute buffer is big enough, copies sizeof(attribute) bytes, and
accepts most values unchecked. So the bindings' type table is the only guard against a wrong-size
or out-of-range value, and these tests prove it for every attribute.
"""

import pytest

import hipblaslt

c = hipblaslt._core
TYPES = c._attribute_types()


def _new(enum_name):
    if enum_name == "MatmulDescAttr":
        return c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    if enum_name == "MatrixLayoutAttr":
        return c.MatrixLayout(c.DataType.R_16F, 4, 4, 4)
    return c.Preference()


# A value per C type that only round-trips when the type is right: bits above bit 31 for the
# 64-bit types, the top bit for the unsigned ones.
_WIDE = {
    "int32": -(2**31),
    "uint32": 2**32 - 1,
    "int64": 2**40 + 3,  # several int64 attributes reject negative values
    "uint64": 2**40 + 5,
    "float32": 1.5,
    "pointer": 0x7F00_1234_5670,
}

# Attributes hipBLASLt accepts only some values for. Each still uses the widest value it takes.
_VALUE = {
    ("MatmulDescAttr", "BIAS_BATCH_STRIDE"): 2**31 - 1,  # rejects negative values
    ("MatmulDescAttr", "ALPHA_BATCH_STRIDE"): 1,  # 0 or 1
    ("MatmulDescAttr", "BETA_BATCH_STRIDE"): 1,
    ("MatmulDescAttr", "UNIFORM_SUMMATION_ORDER_EXT"): 1,  # 0 or 1
    ("MatmulDescAttr", "SM_COUNT_TARGET"): 2**31 - 1,  # rejects negative values
    ("MatrixLayoutAttr", "BATCH_COUNT"): 2**31 - 1,
    (
        "MatrixLayoutAttr",
        "BATCH_MODE",
    ): c.BatchMode.POINTER_ARRAY,  # GROUPED needs a grouped layout
    ("PreferenceAttr", "SM_COUNT_TARGET"): 2**31 - 1,
    # hipBLASLt clamps this to UINT32_MAX; 5 still fails if only 4 of its 8 bytes are written.
    ("PreferenceAttr", "MAX_WORKSPACE_BYTES"): 5,
}

# hipBLASLt accepts these but has no getter: reading one back returns HIPBLAS_STATUS_INVALID_VALUE.
_WRITE_ONLY = {
    ("MatmulDescAttr", n)
    for n in (
        "C_SCALE_POINTER",
        "D_SCALE_POINTER",
        "EPILOGUE_AUX_SCALE_POINTER",
        "EPILOGUE_AUX_POINTER",
        "EPILOGUE_AUX_LD",
        "EPILOGUE_AUX_BATCH_STRIDE",
        "EPILOGUE_ACT_ARG0_EXT",
        "EPILOGUE_ACT_ARG1_EXT",
    )
}
# A non-NULL value is documented as not supported (no kernel scales the AUX output).
_VALUE[("MatmulDescAttr", "EPILOGUE_AUX_SCALE_POINTER")] = None

# These only apply to grouped layouts, which the bindings can't create yet. On an ordinary layout
# hipBLASLt rejects them, which test_grouped_only_attributes_are_rejected pins.
_GROUPED_ONLY = {
    ("MatrixLayoutAttr", n)
    for n in (
        "GROUPED_MATRIX_LAYOUT_ROWS_ARRAY",
        "GROUPED_MATRIX_LAYOUT_COLS_ARRAY",
        "GROUPED_MATRIX_LAYOUT_LD_ARRAY",
        "GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH",
        "GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH",
    )
}


def _cases():
    for enum_name, table in TYPES.items():
        enum = getattr(c, enum_name)
        for value, (kind, enum_type) in sorted(table.items()):
            attr = enum(value)
            key = (enum_name, attr.name)
            if key in _VALUE:
                v = _VALUE[key]
            elif enum_type:
                v = list(getattr(c, enum_type))[-1]
            else:
                v = _WIDE[kind]
            yield pytest.param(enum_name, attr, kind, v, id=f"{enum_name}.{attr.name}")


CASES = list(_cases())


@pytest.mark.parametrize("enum_name,attr,kind,value", CASES)
def test_round_trip(enum_name, attr, kind, value):
    key = (enum_name, attr.name)
    obj = _new(enum_name)
    if key in _GROUPED_ONLY:
        pytest.skip(
            "grouped layouts aren't bound yet; see test_grouped_only_attributes_are_rejected"
        )
    obj.set_attribute(attr, value)
    if key in _WRITE_ONLY:
        with pytest.raises(hipblaslt.HipblasLtError, match="INVALID_VALUE"):
            obj.get_attribute(attr)
        return
    got = obj.get_attribute(attr)
    assert got == (0 if value is None else value)
    if kind == "float32":
        assert isinstance(got, float)


@pytest.mark.parametrize("key", sorted(_GROUPED_ONLY), ids=lambda k: k[1])
def test_grouped_only_attributes_are_rejected(key):
    attr = getattr(c.MatrixLayoutAttr, key[1])
    kind, enum_type = TYPES["MatrixLayoutAttr"][int(attr)]
    value = list(getattr(c, enum_type))[-1] if enum_type else _WIDE[kind]
    with pytest.raises(hipblaslt.HipblasLtError, match="INVALID_VALUE"):
        _new("MatrixLayoutAttr").set_attribute(attr, value)


@pytest.mark.parametrize("enum_name", sorted(TYPES))
def test_every_header_attribute_has_a_table_entry(enum_name):
    # Sentinels such as ..._MAX aren't bound at all (ALLOWED_MISSING), so every bound value counts.
    missing = [m.name for m in getattr(c, enum_name) if int(m) not in TYPES[enum_name]]
    assert not missing, (
        f"{enum_name} values with no entry in the attribute type table in src/descriptors.cpp: "
        f"{missing}. Add each with its C storage type."
    )


def test_storage_types_the_header_documents():
    # hipblaslt.h's comments, and the implementation where the header says nothing.
    layout = TYPES["MatrixLayoutAttr"]
    A = c.MatrixLayoutAttr
    assert layout[int(A.BATCH_COUNT)][0] == "int32"
    assert layout[int(A.STRIDED_BATCH_OFFSET)][0] == "int64"
    assert layout[int(A.TYPE)][0] == "uint32"
    assert layout[int(A.ORDER)][0] == "int32"
    assert layout[int(A.ROWS)][0] == layout[int(A.COLS)][0] == "uint64"
    assert layout[int(A.LD)][0] == "int64"
    assert layout[int(A.BATCH_MODE)][0] == "int32"
    assert layout[int(A.OFFSET)][0] == "int64"
    assert (
        TYPES["PreferenceAttr"][int(c.PreferenceAttr.MAX_WORKSPACE_BYTES)][0]
        == "uint64"
    )


# --- the mistakes hipBLASLt lets through ---------------------------------------------------------


def test_int64_into_an_int32_attribute_is_refused():
    # hipBLASLt accepts an 8-byte TRANSA and later fails the matmul with INTERNAL_ERROR.
    d = _new("MatmulDescAttr")
    with pytest.raises(ValueError, match="32-bit"):
        d.set_attribute(c.MatmulDescAttr.TRANSA, 2**40)


def test_right_size_wrong_meaning_is_refused():
    # A float's bits as an epilogue: hipBLASLt accepts it and silently runs with no epilogue.
    d = _new("MatmulDescAttr")
    with pytest.raises(ValueError, match="Epilogue"):
        d.set_attribute(c.MatmulDescAttr.EPILOGUE, 0x3F800000)


def test_a_member_of_the_wrong_enum_is_refused():
    d = _new("MatmulDescAttr")
    with pytest.raises(TypeError, match="takes an Operation, not an Epilogue"):
        d.set_attribute(c.MatmulDescAttr.TRANSA, c.Epilogue.RELU)


def test_max_workspace_bytes_is_written_as_uint64():
    # A 4-byte write leaves the setter's 8-byte read with 4 bytes of garbage.
    p = _new("PreferenceAttr")
    p.set_attribute(c.PreferenceAttr.MAX_WORKSPACE_BYTES, 5)
    assert p.get_attribute(c.PreferenceAttr.MAX_WORKSPACE_BYTES) == 5


def test_int64_attribute_keeps_its_high_bits():
    # STRIDED_BATCH_OFFSET is int64_t; hipBLASLt returns INVALID_VALUE for a 4-byte write.
    layout = _new("MatrixLayoutAttr")
    layout.set_attribute(c.MatrixLayoutAttr.STRIDED_BATCH_OFFSET, 2**40 + 3)
    assert layout.get_attribute(c.MatrixLayoutAttr.STRIDED_BATCH_OFFSET) == 2**40 + 3


@pytest.mark.parametrize(
    "attr,value,error",
    [
        (c.MatmulDescAttr.TRANSA, 1.5, TypeError),
        (c.MatmulDescAttr.TRANSA, "T", TypeError),
        (c.MatmulDescAttr.EPILOGUE_ACT_ARG0_EXT, "1.0", TypeError),
        (c.MatmulDescAttr.BIAS_POINTER, -1, ValueError),
        (c.MatmulDescAttr.EPILOGUE_AUX_LD, 2**63, ValueError),
    ],
    ids=[
        "float-for-int",
        "str-for-enum",
        "str-for-float",
        "negative-pointer",
        "int64-overflow",
    ],
)
def test_wrong_python_values_are_refused(attr, value, error):
    with pytest.raises(error):
        _new("MatmulDescAttr").set_attribute(attr, value)


def test_descriptors_work_without_a_gpu():
    # They allocate host memory only; the CPU lanes rely on this.
    d = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    layout = c.MatrixLayout(c.DataType.R_32F, 16, 8, 16)
    p = c.Preference()
    assert d.ptr and layout.ptr and p.ptr
    assert layout.get_attribute(c.MatrixLayoutAttr.ROWS) == 16
    assert layout.get_attribute(c.MatrixLayoutAttr.TYPE) == c.DataType.R_32F
