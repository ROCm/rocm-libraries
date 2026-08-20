"""Tests for the workflow expressions that react to a label-gated cmake flag.

`therock_multiarch_label_flags.py` decides *whether* a flag is active; three
GitHub expressions in `therock-multi-arch-ci.yml` decide what that means for
artifact reuse. Those expressions are the only thing stopping a flag-on run from
inheriting stages that were built flag-off, and they live in YAML, so the
resolver's own tests cannot reach them. These tests extract them from the
workflow and evaluate them for both values of `flags_active`.

The expressions are written in a shape that looks needlessly awkward:

    flags_active != 'true' && (<seed>) || ''

rather than the natural `flags_active == 'true' && '' || (<seed>)`. The natural
form is silently wrong. GitHub's `&&` and `||` yield an *operand*, not a
boolean, and the empty string is falsy -- so `true && '' || (<seed>)` falls
through to `<seed>` and seeds exactly the run that must not be seeded. The
evaluator below reproduces that rule rather than Python's, which is the entire
reason these tests can catch the mistake. The evaluator is itself guarded by
`test_evaluator_reproduces_the_falsy_operand_rule`, since one written with
Python semantics would happily let the broken form pass.
"""

from pathlib import Path
import re
import unittest

import yaml

WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"
RELEASE_WORKFLOW = WORKFLOWS / "therock-multi-arch-ci.yml"
ASAN_WORKFLOW = WORKFLOWS / "therock-multi-arch-ci-asan.yml"

# Stage list the release workflow copies from a baseline run. Asserted against
# rather than re-derived, so that editing the list is a deliberate act.
SEEDED_STAGES = (
    "runtime-tests,comm-libs,cv-libs,storage-libs,debug-tools,"
    "dctools-core,profiler-apps,media-libs,wsl-rocdxg"
)

BASELINE_RUN_ID = "1234567890"


# ---------------------------------------------------------------------------
# A minimal evaluator for the GitHub expression subset these guards use.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<string>'(?:[^']|'')*')
      | (?P<op>==|!=|&&|\|\||\(|\))
      | (?P<path>[A-Za-z_][A-Za-z0-9_.\-]*)
    )
    """,
    re.VERBOSE,
)


def _tokenize(expression):
    tokens = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = _TOKEN_RE.match(expression, position)
        if not match:
            raise ValueError(f"cannot tokenize at offset {position}: {expression!r}")
        kind = match.lastgroup
        text = match.group(kind)
        tokens.append((kind, text))
        position = match.end()
    return tokens


def _truthy(value):
    """GitHub's truthiness: null, false, the empty string and 0 are falsy."""
    return value not in (None, False, "", 0)


def _loose_equal(left, right):
    """GitHub coerces null to the empty string before comparing."""
    return ("" if left is None else left) == ("" if right is None else right)


class _Parser:
    """Recursive descent over `||` > `&&` > equality > primary."""

    def __init__(self, tokens, context):
        self._tokens = tokens
        self._index = 0
        self._context = context

    def parse(self):
        value = self._parse_or()
        if self._index != len(self._tokens):
            raise ValueError(f"trailing tokens: {self._tokens[self._index :]}")
        return value

    def _peek(self):
        if self._index < len(self._tokens):
            return self._tokens[self._index][1]
        return None

    def _parse_or(self):
        value = self._parse_and()
        while self._peek() == "||":
            self._index += 1
            right = self._parse_and()
            # Yields an operand, not a boolean.
            value = value if _truthy(value) else right
        return value

    def _parse_and(self):
        value = self._parse_equality()
        while self._peek() == "&&":
            self._index += 1
            right = self._parse_equality()
            # Yields an operand, not a boolean. This is the rule the guards are
            # shaped around: a falsy left operand wins, and '' is falsy.
            value = right if _truthy(value) else value
        return value

    def _parse_equality(self):
        value = self._parse_primary()
        while self._peek() in ("==", "!="):
            operator = self._tokens[self._index][1]
            self._index += 1
            right = self._parse_primary()
            equal = _loose_equal(value, right)
            value = equal if operator == "==" else not equal
        return value

    def _parse_primary(self):
        if self._index >= len(self._tokens):
            raise ValueError("unexpected end of expression")
        kind, text = self._tokens[self._index]
        if text == "(":
            self._index += 1
            value = self._parse_or()
            if self._peek() != ")":
                raise ValueError("unbalanced parenthesis")
            self._index += 1
            return value
        self._index += 1
        if kind == "string":
            return text[1:-1].replace("''", "'")
        if kind == "path":
            if text in ("true", "false"):
                return text == "true"
            return self._lookup(text)
        raise ValueError(f"unexpected token {text!r}")

    def _lookup(self, path):
        """Resolve a dotted context path; anything missing is null."""
        value = self._context
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value


def evaluate(expression, context):
    return _Parser(_tokenize(expression), context).parse()


# ---------------------------------------------------------------------------
# Workflow extraction.
# ---------------------------------------------------------------------------

_INTERPOLATION_RE = re.compile(r"^\$\{\{(?P<body>.*)\}\}$", re.DOTALL)


def setup_inputs(workflow_path):
    """The `with:` mapping the workflow hands to TheRock's setup_multi_arch."""
    workflow = yaml.safe_load(workflow_path.read_text())
    return workflow["jobs"]["setup"].get("with", {})


def guard_expression(workflow_path, input_name):
    """The bare expression body of a `with:` value that is a single `${{ }}`."""
    raw = setup_inputs(workflow_path)[input_name]
    match = _INTERPOLATION_RE.match(str(raw).strip())
    if not match:
        raise AssertionError(
            f"{workflow_path.name}: `{input_name}` is no longer a single "
            f"interpolation, so this test can no longer evaluate it: {raw!r}"
        )
    return match.group("body")


def build_context(*, flags_active, baseline_run_id=BASELINE_RUN_ID):
    """Contexts as they stand on a pull_request run.

    `inputs` is deliberately empty: the workflow_dispatch inputs are null on a
    pull_request event, which is what makes the `inputs.x || <default>` halves
    of these expressions fall through to their defaults.
    """
    return {
        "inputs": {},
        "needs": {
            "label_flags": {"outputs": {"flags_active": flags_active}},
            "ci-env": {"outputs": {"baseline-run-id": baseline_run_id}},
        },
    }


class EvaluatorTest(unittest.TestCase):
    """The evaluator carries the semantics under test, so test it directly."""

    def test_evaluator_reproduces_the_falsy_operand_rule(self):
        # The mistake the guards avoid: a true condition whose right operand is
        # the empty string falls through to the `||` branch. Python's `and`/`or`
        # on booleans would give a different answer, and an evaluator with
        # Python semantics would let the broken workflow form pass these tests.
        self.assertEqual(
            evaluate("'x' == 'x' && '' || 'fallback'", {}),
            "fallback",
        )
        # The inverted form the workflow actually uses does what it looks like.
        self.assertEqual(evaluate("'x' != 'x' && 'seed' || ''", {}), "")
        self.assertEqual(evaluate("'x' != 'y' && 'seed' || ''", {}), "seed")

    def test_operators_yield_operands_not_booleans(self):
        self.assertEqual(evaluate("'a' && 'b'", {}), "b")
        self.assertEqual(evaluate("'' && 'b'", {}), "")
        self.assertEqual(evaluate("'a' || 'b'", {}), "a")
        self.assertEqual(evaluate("'' || 'b'", {}), "b")

    def test_missing_context_paths_are_null_and_equal_the_empty_string(self):
        self.assertIsNone(evaluate("inputs.nope", {"inputs": {}}))
        self.assertIs(evaluate("inputs.nope == ''", {"inputs": {}}), True)


class ReleaseWorkflowGuardTest(unittest.TestCase):
    """The three expressions that drop artifact reuse when a flag is active."""

    def assert_empty_string(self, value, input_name):
        # Not just falsy: a boolean `false` here would reach the reusable
        # workflow as the literal string "false", which is not an empty input.
        # That is what the trailing `|| ''` on each expression is for.
        self.assertIsInstance(
            value,
            str,
            f"{input_name} evaluated to {value!r}; a non-string renders as its "
            "literal text, not as an empty input",
        )
        self.assertEqual(value, "")

    def test_flags_off_seeds_from_the_baseline_run(self):
        context = build_context(flags_active="false")
        self.assertEqual(
            evaluate(guard_expression(RELEASE_WORKFLOW, "prebuilt_stages"), context),
            SEEDED_STAGES,
        )
        self.assertEqual(
            evaluate(guard_expression(RELEASE_WORKFLOW, "baseline_run_id"), context),
            BASELINE_RUN_ID,
        )
        self.assertEqual(
            evaluate(guard_expression(RELEASE_WORKFLOW, "stage_reuse_mode"), context),
            "reuse-stage",
        )

    def test_flags_on_drops_every_form_of_artifact_reuse(self):
        # The load-bearing case, and the one production has never run: a
        # flag-on build must build every stage itself, because the baseline
        # artifacts were produced under a different cmake configuration.
        context = build_context(flags_active="true")
        self.assert_empty_string(
            evaluate(guard_expression(RELEASE_WORKFLOW, "prebuilt_stages"), context),
            "prebuilt_stages",
        )
        self.assert_empty_string(
            evaluate(guard_expression(RELEASE_WORKFLOW, "baseline_run_id"), context),
            "baseline_run_id",
        )
        self.assertEqual(
            evaluate(guard_expression(RELEASE_WORKFLOW, "stage_reuse_mode"), context),
            "dry-run",
        )

    def test_flags_active_is_compared_as_a_string_not_a_boolean(self):
        # Job outputs are always strings. Anything other than the exact string
        # "true" is flag-off, and must not be mistaken for one.
        for value in ("", "false", "TRUE", "1", "yes"):
            with self.subTest(flags_active=value):
                context = build_context(flags_active=value)
                self.assertEqual(
                    evaluate(
                        guard_expression(RELEASE_WORKFLOW, "stage_reuse_mode"), context
                    ),
                    "reuse-stage",
                )

    def test_flags_off_with_no_baseline_still_clears_prebuilt_stages(self):
        # Pre-existing behavior, unrelated to labels: with no baseline run to
        # copy from there is nothing to seed, and the input must come out empty
        # rather than as the literal "false".
        context = build_context(flags_active="false", baseline_run_id="")
        self.assert_empty_string(
            evaluate(guard_expression(RELEASE_WORKFLOW, "prebuilt_stages"), context),
            "prebuilt_stages",
        )
        self.assert_empty_string(
            evaluate(guard_expression(RELEASE_WORKFLOW, "baseline_run_id"), context),
            "baseline_run_id",
        )


class AsanWorkflowGuardTest(unittest.TestCase):
    """Why the ASAN workflow needs no `flags_active` guard of its own.

    It carries no guard, which is only safe because of three separate facts.
    Each is pinned below, because each could be edited away independently and
    the result would be a flag-on ASAN build quietly reusing flag-off stages.
    """

    def test_stage_reuse_is_left_at_the_dry_run_default(self):
        # Unlike the release workflow, this one never opts in to stage reuse,
        # so there is nothing for a flag-on run to inherit and nothing to force
        # off. Adding this input means adding a guard with it.
        self.assertNotIn(
            "stage_reuse_mode",
            setup_inputs(ASAN_WORKFLOW),
            "the ASAN workflow now sets stage_reuse_mode explicitly; it needs a "
            "flags_active guard like the release workflow's",
        )

    def test_seeding_inputs_are_dispatch_only_and_empty_on_a_pull_request(self):
        # The workflow does forward prebuilt_stages and baseline_run_id, but
        # only from its own workflow_dispatch inputs -- it seeds nothing from a
        # ci-env baseline the way the release workflow does. On a pull_request
        # run, which is the only event where a label can activate a flag, the
        # `inputs` context is null and both come out empty.
        context = build_context(flags_active="true")
        for input_name in ("prebuilt_stages", "baseline_run_id"):
            with self.subTest(input=input_name):
                value = evaluate(guard_expression(ASAN_WORKFLOW, input_name), context)
                self.assertEqual(value, "")

    def test_explicit_dispatch_seeding_is_rejected_by_the_resolver(self):
        # The remaining case -- a dispatch run that passes seeding inputs -- is
        # a hard error in therock_multiarch_label_flags.py rather than a YAML
        # guard, which only works because the workflow hands it those same two
        # values. Pin that wiring; without it the fatal check sees nothing.
        steps = next(
            job
            for name, job in yaml.safe_load(ASAN_WORKFLOW.read_text())["jobs"].items()
            if name == "label_flags"
        )["steps"]
        resolver_env = next(
            step["env"]
            for step in steps
            if "therock_multiarch_label_flags.py" in step.get("run", "")
        )
        self.assertEqual(
            resolver_env["PREBUILT_STAGES"], "${{ inputs.prebuilt_stages || '' }}"
        )
        self.assertEqual(
            resolver_env["BASELINE_RUN_ID"], "${{ inputs.baseline_run_id || '' }}"
        )


if __name__ == "__main__":
    unittest.main()
