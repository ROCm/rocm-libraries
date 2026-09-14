# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Reference resolution and the condition grammar."""
from __future__ import annotations

import pytest

from runner.errors import ConfigError, RefError
from runner.refs import Resolver, parse_condition, render, render_value


def check(condition: str, resolver: Resolver) -> bool:
    return parse_condition(condition).evaluate(resolver)


def make(**kwargs) -> Resolver:
    base = {
        "inputs": {"graph": "g.json", "count": 3},
        "vars": {"root": "/repo"},
        "steps": {"review": {"critical_count": 2, "verdict": "changes_required"}},
    }
    base.update(kwargs)
    return Resolver(**base)


def test_unknown_input_raises_instead_of_rendering_empty():
    # The failure this prevents: a prompt that silently loses its context and an agent
    # that answers the truncated question anyway.
    with pytest.raises(RefError, match="not a declared|unknown input"):
        render("goal: ${inputs.missing}", make())


def test_whole_string_reference_keeps_its_type():
    assert render_value("${inputs.count}", make()) == 3
    assert render_value("count=${inputs.count}", make()) == "count=3"


def test_unknown_namespace_names_the_namespace():
    with pytest.raises(RefError, match="unknown namespace 'stpes'"):
        render("${stpes.review.outputs.verdict}", make())


def test_step_output_before_it_ran_is_an_error_not_empty():
    with pytest.raises(RefError, match="has not run yet"):
        render("${steps.build.outputs.exit_code}", make())


def test_lenient_mode_marks_unresolved_step_outputs_visibly():
    resolver = make(lenient=True)
    assert (
        render("${steps.build.outputs.exit_code}", resolver)
        == "<unresolved:steps.build.outputs.exit_code>"
    )


def test_loop_previous_is_empty_on_first_iteration():
    # Documented contract: prompts reference the previous attempt unconditionally.
    resolver = make(loop={"iteration": 0}, previous=None)
    assert (
        render("prev: ${loop.previous.generate.outputs.kernel_path}", resolver)
        == "prev: "
    )


def test_loop_refs_outside_a_loop_are_rejected():
    with pytest.raises(RefError, match="only available inside a loop"):
        render("${loop.iteration}", make())


@pytest.mark.parametrize(
    "condition, expected",
    [
        ("${steps.review.outputs.critical_count} == 0", False),
        ("${steps.review.outputs.critical_count} > 1", True),
        ("${steps.review.outputs.verdict} == changes_required", True),
        (
            "${steps.review.outputs.critical_count} == 0 or ${steps.review.outputs.verdict} == pass",
            False,
        ),
        (
            "${steps.review.outputs.critical_count} > 0 and ${steps.review.outputs.verdict} == changes_required",
            True,
        ),
        ("${inputs.count} + ${steps.review.outputs.critical_count} > 4", True),
        ("${inputs.graph} contains .json", True),
        ("${steps.review.outputs.verdict} matches ^changes", True),
    ],
)
def test_condition_grammar(condition, expected):
    assert check(condition, make()) is expected


def test_string_numbers_compare_numerically_not_lexically():
    # "10" < "9" as strings; the exit condition of every loop depends on this.
    resolver = make(steps={"review": {"critical_count": "10"}})
    assert check("${steps.review.outputs.critical_count} > 9", resolver) is True


def test_ordering_non_numbers_is_a_config_error_not_a_silent_false():
    with pytest.raises(ConfigError, match="cannot order non-numeric"):
        check("${steps.review.outputs.verdict} > 3", make())


@pytest.mark.parametrize(
    "condition",
    [
        # The one that mattered: a single `=` used to concatenate its operands into the
        # non-empty string "1=0" and report an exit condition as satisfied.
        "${steps.review.outputs.critical_count} = 0",
        "1 ==",
        "== 1",
        "${steps.review.outputs.critical_count} == 0 and",
        "${steps.review.outputs.critical_count} == 0 or",
        # Adjacent terms with no joiner: concatenation has to be spelled with '+'.
        "${steps.review.outputs.verdict} pass",
        "0 1",
        "${steps.review.outputs.critical_count} + ",
        "${steps.review.outputs.critical_count} contains",
        "and",
        "",
    ],
)
def test_malformed_conditions_are_rejected_when_parsed(condition):
    # Parsing happens at flow load, so these never reach a launched agent.
    with pytest.raises(ConfigError):
        parse_condition(condition)


def test_single_equals_does_not_become_truthy():
    # The reproduction from the review: a failing step, an exit condition that should
    # read false, and a parser that used to say true.
    resolver = make(steps={"validate": {"failed": 1}})
    with pytest.raises(ConfigError, match="did you mean '=='"):
        parse_condition("${steps.validate.outputs.failed} = 0").evaluate(resolver)


def test_and_binds_tighter_than_or():
    # `true or (false and false)`, not `(true or false) and false`.
    assert check("true == true or false == true and false == true", make()) is True
    assert check("false == true and false == true or true == true", make()) is True
    assert check("true == true and false == true or false == true", make()) is False


def test_explicit_plus_still_concatenates_and_sums():
    assert check("'a' + 'b' == ab", make()) is True
    assert check("${inputs.count} + 1 == 4", make()) is True
