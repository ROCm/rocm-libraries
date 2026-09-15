# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ substituter's test table, re-run against the Python port.

`codegen/kernel_defines.py` exists only to refuse, at generation time, exactly
what `KernelDefineSubstitution.hpp` refuses at load time. Two implementations
of one grammar drift silently: the divergence does not fail anything here, it
ships a bundle that generates clean and drops its pack on a target machine.

So the table below is the C++ table. Every row names the `TEST(...)` in
`plugin_sdk/tests/ingestor/TestKernelDefineSubstitution.cpp` it came from, and
the schema and metadata fixtures are that file's `testSchema()` and
`completedMetadata()` transcribed. A reader diffs the two files side by side;
a row here with no counterpart there is a rule Python invented.
"""

import pytest

from codegen.kernel_defines import (
    KernelDefineError,
    substitute_kernel_define,
    validate_kernel_define_template,
)

#: `TestKernelDefineSubstitution.cpp`'s `testSchema()`: one field per KMD type,
#: plus a field the kernel omits and the schema defaults.
SCHEMA_FIELD_TYPES = {
    "flag": "bool",
    "block_size": "int",
    "dtype": "string",
    "alpha": "float",
    "tile": "int_list",
    "vector_width": "int",
}
SCHEMA_NAME = "test_schema"

#: `completedMetadata()`: what `completeMetadata` hands `prepare()`, with the
#: defaulted field already filled in.
COMPLETED_METADATA = {
    "flag": True,
    "block_size": 128,
    "dtype": "bfloat16",
    "alpha": 1.0,
    "tile": [64, 32],
    "vector_width": 4,
}

#: `(C++ TEST name, template, expected text)` -- rows the substituter ACCEPTS.
#: Every one of these also validates against the schema, exactly as the C++
#: cases do: a template accepted at prepare() but refused at load would drop a
#: pack the author had already seen work.
ACCEPTED = [
    ("BoolRendersAsOneOrZero", "$kernel.flag", "1"),
    ("IntRendersAsDecimal", "$kernel.block_size", "128"),
    ("StringRendersVerbatim", "$kernel.dtype", "bfloat16"),
    ("DefaultedFieldResolvesFromCompletedMetadata", "$kernel.vector_width", "4"),
    (
        "LiteralTextAroundAndBetweenTokensIsPreserved",
        "hip_$kernel.dtype",
        "hip_bfloat16",
    ),
    (
        "LiteralTextAroundAndBetweenTokensIsPreserved",
        "$kernel.dtype,$kernel.block_size",
        "bfloat16,128",
    ),
    ("LiteralTextAroundAndBetweenTokensIsPreserved", "x$kernel.flag", "x1"),
    (
        "TemplateWithoutTokenPassesThroughByteIdentical",
        "float4(-1) ? a : b",
        "float4(-1) ? a : b",
    ),
    ("ValidationAcceptsEveryRenderableDeclaredField", "", ""),
]

#: `(C++ TEST name, template, fragment both messages must contain)` -- rows
#: `expectRejectedByBoth` rejects through BOTH entry points.
REJECTED_BY_BOTH = [
    ("FloatIsRejected", "$kernel.alpha", "float"),
    ("IntListIsRejected", "$kernel.tile", "int_list"),
    ("NonKernelTokenIsRejected", "$graph.batch", "$kernel."),
    ("NonKernelTokenIsRejected", "$dtype", "$kernel."),
    ("NonKernelTokenIsRejected", "$KERNEL.dtype", "$kernel."),
    ("TokenWithoutFieldNameIsRejected", "$kernel.", "no field name"),
    ("TokenWithoutFieldNameIsRejected", "$kernel.9x", "no field name"),
    (
        "ExpressionSyntaxBesideATokenIsRejected",
        "$kernel.block_size * 2",
        "literal token replacement",
    ),
    (
        "ExpressionSyntaxBesideATokenIsRejected",
        "$kernel.block_size + 1",
        "literal token replacement",
    ),
    (
        "ExpressionSyntaxBesideATokenIsRejected",
        "$kernel.dtype == float16",
        "literal token replacement",
    ),
    (
        "ExpressionSyntaxBesideATokenIsRejected",
        "$kernel.flag ? 1 : 0",
        "literal token replacement",
    ),
    (
        "ExpressionSyntaxBesideATokenIsRejected",
        "max($kernel.block_size)",
        "literal token replacement",
    ),
    (
        "ExpressionSyntaxBesideATokenIsRejected",
        "$kernel.dtype|float16",
        "literal token replacement",
    ),
]


@pytest.mark.parametrize(
    "cpp_test,template_text,expected",
    ACCEPTED,
    ids=[f"{name}:{template!r}" for name, template, _ in ACCEPTED],
)
def test_accepted_row(cpp_test, template_text, expected):
    assert substitute_kernel_define(template_text, COMPLETED_METADATA) == expected
    validate_kernel_define_template(template_text, SCHEMA_FIELD_TYPES, SCHEMA_NAME)


@pytest.mark.parametrize(
    "cpp_test,template_text,fragment",
    REJECTED_BY_BOTH,
    ids=[f"{name}:{template!r}" for name, template, _ in REJECTED_BY_BOTH],
)
def test_rejected_by_both_entry_points(cpp_test, template_text, fragment):
    """The C++ `expectRejectedByBoth` helper, and its reason: a define that
    passes load-time validation and fails at prepare() fails after the engine
    has already advertised itself."""
    with pytest.raises(KernelDefineError) as substitute_error:
        substitute_kernel_define(template_text, COMPLETED_METADATA)
    assert fragment in str(substitute_error.value)

    with pytest.raises(KernelDefineError) as validate_error:
        validate_kernel_define_template(template_text, SCHEMA_FIELD_TYPES, SCHEMA_NAME)
    assert fragment in str(validate_error.value)


def test_bool_false_renders_as_zero():
    """`BoolRendersAsOneOrZero`, second half. Python's bool IS an int, so a port
    that checked int first would render `True` as `1` by luck and `False` as
    `0` -- and any bool-typed field as its int value, which is the same text
    for two different declared types."""
    metadata = {**COMPLETED_METADATA, "flag": False}
    assert substitute_kernel_define("$kernel.flag", metadata) == "0"


def test_negative_int_renders_as_decimal():
    """`IntRendersAsDecimal`, second half: `-7`, not `- 7` and not an error --
    the minus is in the RENDERED value, which is never re-inspected."""
    metadata = {**COMPLETED_METADATA, "block_size": -7}
    assert substitute_kernel_define("$kernel.block_size", metadata) == "-7"


def test_two_variants_of_one_field_render_distinct_text():
    """`TwoVariantsOfOneFieldRenderDistinctText` -- the property the whole
    feature rests on. Two variants that rendered the same flag text would
    compile to the same binary and silently become one kernel."""
    first = substitute_kernel_define("$kernel.dtype", COMPLETED_METADATA)
    second = substitute_kernel_define(
        "$kernel.dtype", {**COMPLETED_METADATA, "dtype": "float16"}
    )
    assert (first, second) == ("bfloat16", "float16")


def test_undeclared_field_is_rejected():
    """`UndeclaredFieldIsRejected`. The two messages differ on purpose: the
    value-level one says the KERNEL does not carry it, the schema-level one
    says the SCHEMA does not declare it."""
    with pytest.raises(KernelDefineError, match="missing"):
        substitute_kernel_define("$kernel.missing", COMPLETED_METADATA)
    with pytest.raises(KernelDefineError, match="does not declare"):
        validate_kernel_define_template(
            "$kernel.missing", SCHEMA_FIELD_TYPES, SCHEMA_NAME
        )


def test_replacement_text_is_not_rescanned():
    """`ReplacementTextIsNotRescanned`. A string field whose value contains a
    `$` is data, not a template: rescanning it would let metadata reach into
    the substituter's own grammar."""
    metadata = {**COMPLETED_METADATA, "dtype": "$kernel.block_size"}
    assert substitute_kernel_define("$kernel.dtype", metadata) == "$kernel.block_size"


def test_token_runs_to_the_first_character_that_cannot_continue_an_identifier():
    """`LiteralTextAroundAndBetweenTokensIsPreserved`'s comment, made
    executable: `$kernel.dtype_t` names the field `dtype_t`, not `dtype` plus a
    `_t` suffix. A port that stopped at the declared field names instead would
    silently accept it and emit `bfloat16_t`."""
    with pytest.raises(KernelDefineError, match="dtype_t"):
        substitute_kernel_define("$kernel.dtype_t", COMPLETED_METADATA)
