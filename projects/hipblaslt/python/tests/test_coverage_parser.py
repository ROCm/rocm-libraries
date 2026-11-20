# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The header-aware enum parser, on a synthetic header with every awkward case it has to handle.

Without this, a parser that silently drops values would make the coverage checks look perfect.
One real case: a doc comment inside hipblasLtMatmulDescAttributes_t contains a brace-enclosed list,
which a simple regex mistakes for the end of the enum.
"""

from hipblaslt._coverage import BoundValue, bound_values, parse_enums

SYNTHETIC = """
/* A file comment with a brace { that never closes */
typedef enum
{
    A_ZERO,                 /**< implicit 0 */
    A_TWO = 2,              /**< A comment with a brace-enclosed list: {0, 1, 2} */
    A_THREE,                // implicit, follows A_TWO
    A_HEX = 0x10,
    A_OR = A_TWO | A_HEX,
    A_SHIFT = (1 << 5),
#if HIPBLASLT_HAS_SOME_FEATURE
    A_GUARDED = 100,
#endif
    A_AFTER_GUARD,
    A_LAST = 200,           /* trailing comma follows */
} enumA_t;

typedef enum enumB_tag {
  B_ONE = 1,
  B_TWO = 2
} enumB_t;

namespace ns
{
    enum class GemmLike : int
    {
        G_ONE = 1,
        G_TWO = 2,
    };
}

#if HIPBLASLT_HAS_OTHER_FEATURE
typedef enum
{
    C_ONLY = 7,
} enumC_t;
#endif
"""


def _values(text, name):
    return [(v.name, v.value) for v in parse_enums(text)[name].values]


def test_implicit_explicit_hex_expressions_and_trailing_comma():
    assert _values(SYNTHETIC, "enumA_t") == [
        ("A_ZERO", 0),
        ("A_TWO", 2),
        ("A_THREE", 3),
        ("A_HEX", 16),
        ("A_OR", 18),
        ("A_SHIFT", 32),
        ("A_GUARDED", 100),
        ("A_AFTER_GUARD", 101),
        ("A_LAST", 200),
    ]


def test_braces_inside_comments_dont_end_the_enum():
    assert set(parse_enums(SYNTHETIC)) == {"enumA_t", "enumB_t", "GemmLike", "enumC_t"}


def test_typedef_enum_with_a_tag():
    assert _values(SYNTHETIC, "enumB_t") == [("B_ONE", 1), ("B_TWO", 2)]


def test_enum_class_in_a_namespace():
    assert _values(SYNTHETIC, "GemmLike") == [("G_ONE", 1), ("G_TWO", 2)]


def test_guards_are_recorded_per_value():
    guards = {v.name: v.guard for v in parse_enums(SYNTHETIC)["enumA_t"].values}
    assert guards["A_GUARDED"] == "if HIPBLASLT_HAS_SOME_FEATURE"
    assert guards["A_AFTER_GUARD"] is None and guards["A_TWO"] is None


def test_a_whole_guarded_enum():
    (value,) = parse_enums(SYNTHETIC)["enumC_t"].values
    assert value.guard == "if HIPBLASLT_HAS_OTHER_FEATURE"


def test_select_skips_enums_it_rejects_even_unreadable_ones():
    # Preprocessed text includes standard-library enums the parser can't evaluate.
    text = SYNTHETIC + "\nenum class _Std { _X = __builtin_thing };\n"
    assert set(parse_enums(text, select=lambda n: n.startswith("enum"))) == {
        "enumA_t",
        "enumB_t",
        "enumC_t",
    }


def test_crlf_line_endings():
    # The installed Windows headers have them.
    assert parse_enums(SYNTHETIC.replace("\n", "\r\n")) == parse_enums(SYNTHETIC)


def test_bound_values_reads_reg_calls_and_their_guards():
    cpp = """
        reg(e, "enumA_t", "ZERO", A_ZERO);
        reg(e,
            "GemmLike",
            "ONE",
            ns::GemmLike::G_ONE);
        // reg(e, "enumA_t", "COMMENTED", A_COMMENTED);
    #if HIPBLASLT_HAS_SOME_FEATURE
        reg(e, "enumA_t", "GUARDED", A_GUARDED);
    #endif
    """
    assert bound_values(cpp) == {
        "enumA_t": {
            "A_ZERO": BoundValue("ZERO"),
            "A_GUARDED": BoundValue("GUARDED", "if HIPBLASLT_HAS_SOME_FEATURE"),
        },
        "GemmLike": {"G_ONE": BoundValue("ONE")},
    }
