# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""${...} resolution and the closed condition grammar.

Two deliberate properties:

* **Strict.** An unknown reference raises, it never renders as an empty string. A
  prompt that silently interpolated nothing is invisible in the output, and the agent
  answers the truncated question without complaining.
* **Closed.** Conditions are comparisons over references and literals -- no `eval`, no
  attribute access, no imports. A flow file is data, including its control flow.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import ConfigError, RefError

REF_RE = re.compile(r"\$\{([^{}]+)\}")

#: Namespaces a reference may name. Anything else is a typo, and is reported as one.
NAMESPACES = ("inputs", "vars", "env", "platform", "run", "step", "loop", "steps")


def platform_facts() -> dict[str, str]:
    if sys.platform.startswith("win"):
        name, suffix = "windows", ".exe"
    elif sys.platform == "darwin":
        name, suffix = "darwin", ""
    else:
        name, suffix = "linux", ""
    return {"os": name, "exe_suffix": suffix, "path_sep": os.pathsep}


def iter_refs(value: Any) -> Iterable[str]:
    """Every reference path inside a string, list or mapping. Used by static validation."""
    if isinstance(value, str):
        yield from (match.group(1).strip() for match in REF_RE.finditer(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_refs(item)


@dataclass
class Resolver:
    """Resolves reference paths against the namespaces available at this point in a run.

    `lenient` is for `--dry-run`: step outputs do not exist yet, so they render as a
    visible placeholder instead of aborting. Every other namespace stays strict even
    then -- a misspelled input is a real error whether or not anything is launched.
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    vars: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    step: dict[str, Any] = field(default_factory=dict)
    loop: dict[str, Any] | None = None
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    previous: dict[str, dict[str, Any]] | None = None
    lenient: bool = False

    def child(self, **overrides: Any) -> "Resolver":
        data = {
            "inputs": self.inputs,
            "vars": self.vars,
            "run": self.run,
            "step": self.step,
            "loop": self.loop,
            "steps": self.steps,
            "previous": self.previous,
            "lenient": self.lenient,
        }
        data.update(overrides)
        return Resolver(**data)

    def lookup(self, path: str) -> Any:
        parts = [part for part in path.strip().split(".") if part]
        if not parts:
            raise RefError("empty reference '${}'")
        head, rest = parts[0], parts[1:]
        if head not in NAMESPACES:
            raise RefError(
                f"unknown namespace '{head}' in '${{{path}}}'; "
                f"expected one of {', '.join(NAMESPACES)}"
            )
        handler = getattr(self, f"_ns_{head}")
        return handler(rest, path)

    # -- namespaces ---------------------------------------------------------

    def _ns_inputs(self, rest: list[str], path: str) -> Any:
        return self._plain(self.inputs, rest, path, "input")

    def _ns_vars(self, rest: list[str], path: str) -> Any:
        return self._plain(self.vars, rest, path, "var")

    def _ns_run(self, rest: list[str], path: str) -> Any:
        return self._plain(self.run, rest, path, "run field")

    def _ns_step(self, rest: list[str], path: str) -> Any:
        return self._plain(self.step, rest, path, "step field")

    def _ns_env(self, rest: list[str], path: str) -> Any:
        if len(rest) != 1:
            raise RefError(f"'${{{path}}}' must be '${{env.NAME}}'")
        try:
            return os.environ[rest[0]]
        except KeyError:
            raise RefError(f"environment variable '{rest[0]}' is not set") from None

    def _ns_platform(self, rest: list[str], path: str) -> Any:
        return self._plain(platform_facts(), rest, path, "platform fact")

    def _ns_loop(self, rest: list[str], path: str) -> Any:
        if self.loop is None:
            raise RefError(f"'${{{path}}}' is only available inside a loop")
        if rest and rest[0] == "previous":
            return self._previous(rest[1:], path)
        return self._plain(self.loop, rest, path, "loop field")

    def _ns_steps(self, rest: list[str], path: str) -> Any:
        return self._step_output(self.steps, rest, path, missing_ok=self.lenient)

    def _previous(self, rest: list[str], path: str) -> Any:
        # Documented contract: on the first iteration there is no previous attempt, so
        # every such reference is the empty string rather than an error. Prompts that
        # say "the kernel you wrote last time: ${loop.previous...}" then render cleanly
        # on iteration 0 without a conditional in the template.
        if not self.previous:
            return ""
        return self._step_output(self.previous, rest, path, missing_ok=True)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _plain(source: Mapping[str, Any], rest: list[str], path: str, what: str) -> Any:
        if len(rest) != 1:
            raise RefError(f"'${{{path}}}' must name exactly one {what}")
        try:
            return source[rest[0]]
        except KeyError:
            known = ", ".join(sorted(source)) or "(none)"
            raise RefError(
                f"unknown {what} '{rest[0]}' in '${{{path}}}'; known: {known}"
            ) from None

    @staticmethod
    def _step_output(
        source: Mapping[str, Mapping[str, Any]],
        rest: list[str],
        path: str,
        missing_ok: bool,
    ) -> Any:
        if len(rest) != 3 or rest[1] != "outputs":
            raise RefError(f"'${{{path}}}' must be '<scope>.<step-id>.outputs.<name>'")
        step_id, _, name = rest
        bucket = source.get(step_id)
        if bucket is None:
            if missing_ok:
                return f"<unresolved:{path}>"
            known = ", ".join(sorted(source)) or "(none)"
            raise RefError(
                f"'${{{path}}}' names step '{step_id}', which has not run yet; "
                f"completed steps: {known}"
            )
        if name not in bucket:
            if missing_ok:
                return f"<unresolved:{path}>"
            known = ", ".join(sorted(bucket)) or "(none)"
            raise RefError(
                f"step '{step_id}' declares no output '{name}'; it has: {known}"
            )
        return bucket[name]


def render(value: str, resolver: Resolver) -> str:
    """Interpolate every reference in `value` and return a string."""
    return REF_RE.sub(lambda m: _stringify(resolver.lookup(m.group(1))), value)


def render_value(value: str, resolver: Resolver) -> Any:
    """Like `render`, but a string that is exactly one reference keeps its type."""
    stripped = value.strip()
    match = REF_RE.fullmatch(stripped)
    if match:
        return resolver.lookup(match.group(1))
    return render(value, resolver)


def render_tree(value: Any, resolver: Resolver) -> Any:
    if isinstance(value, str):
        return render_value(value, resolver)
    if isinstance(value, Mapping):
        return {key: render_tree(item, resolver) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [render_tree(item, resolver) for item in value]
    return value


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return str(value)


# -- condition grammar ------------------------------------------------------
#
#   condition := clause (("and" | "or") clause)*        left to right, no precedence
#   clause    := operand OP operand | operand           bare operand -> truthiness
#   operand   := term ("+" term)*                       numeric sum, or string concat
#   term      := ${ref} | number | 'string' | "string" | bareword
#   OP        := == != < <= > >= contains matches

_COMPARATORS = ("==", "!=", "<=", ">=", "<", ">", "contains", "matches")
#: Order matters: refs and quoted strings first, then operators, then numbers, and a
#: catch-all bareword last. The bareword is deliberately anything non-blank -- operands
#: are values like `.json`, `^changes` or `gfx1151`, and a character-class allowlist
#: rejects them as unparseable instead of comparing them. Tokens must be whitespace
#: separated, which every generated and hand-written condition already is.
_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<ref>\$\{[^{}]+\})
      | (?P<sq>'[^']*')
      | (?P<dq>"[^"]*")
      | (?P<op><=|>=|==|!=|<|>)
      | (?P<plus>\+(?=\s))
      | (?P<num>-?\d+(?:\.\d+)?(?=\s|$))
      | (?P<word>\S+?)(?=\s|$|<=|>=|==|!=|<|>)
    )""",
    re.VERBOSE,
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.match(text, position)
        if not match:
            if text[position:].strip() == "":
                break
            raise ConfigError(f"cannot parse condition at: {text[position:]!r}")
        position = match.end()
        kind = match.lastgroup or ""
        value = match.group(kind)
        tokens.append((kind, value))
    return tokens


def evaluate(condition: str, resolver: Resolver) -> bool:
    """Evaluate a `when` / `until` / `assert` condition. Never uses eval()."""
    tokens = _tokenize(condition)
    if not tokens:
        raise ConfigError("empty condition")
    result: bool | None = None
    joiner: str | None = None
    index = 0
    while index < len(tokens):
        clause_tokens: list[tuple[str, str]] = []
        while index < len(tokens) and not (
            tokens[index][0] == "word" and tokens[index][1] in ("and", "or")
        ):
            clause_tokens.append(tokens[index])
            index += 1
        value = _clause(clause_tokens, condition, resolver)
        if result is None:
            result = value
        elif joiner == "and":
            result = result and value
        else:
            result = result or value
        if index < len(tokens):
            joiner = tokens[index][1]
            index += 1
            if index >= len(tokens):
                raise ConfigError(f"condition ends with '{joiner}': {condition!r}")
    return bool(result)


def _clause(tokens: list[tuple[str, str]], condition: str, resolver: Resolver) -> bool:
    operator_at = next(
        (
            i
            for i, (kind, value) in enumerate(tokens)
            if kind == "op" or (kind == "word" and value in ("contains", "matches"))
        ),
        None,
    )
    if operator_at is None:
        return _truthy(_operand(tokens, condition, resolver))
    operator = tokens[operator_at][1]
    left = _operand(tokens[:operator_at], condition, resolver)
    right = _operand(tokens[operator_at + 1 :], condition, resolver)
    return _compare(left, operator, right, condition)


def _operand(tokens: list[tuple[str, str]], condition: str, resolver: Resolver) -> Any:
    if not tokens:
        raise ConfigError(f"missing operand in condition {condition!r}")
    values = [_term(kind, value, resolver) for kind, value in tokens if kind != "plus"]
    if len(values) == 1:
        return values[0]
    numbers = [_as_number(value) for value in values]
    if all(number is not None for number in numbers):
        return sum(numbers)  # type: ignore[arg-type]
    return "".join(_stringify(value) for value in values)


def _term(kind: str, value: str, resolver: Resolver) -> Any:
    if kind == "ref":
        return resolver.lookup(value[2:-1])
    if kind in ("sq", "dq"):
        return value[1:-1]
    if kind == "num":
        return float(value) if "." in value else int(value)
    if kind == "word":
        if value == "true":
            return True
        if value == "false":
            return False
        return value
    raise ConfigError(f"unexpected token {value!r} in condition")


def _as_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip()
        return float(text) if "." in text else int(text)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no")
    return bool(value)


def _compare(left: Any, operator: str, right: Any, condition: str) -> bool:
    if operator == "contains":
        if isinstance(left, (list, tuple, dict)):
            return right in left
        return _stringify(right) in _stringify(left)
    if operator == "matches":
        return re.search(_stringify(right), _stringify(left)) is not None
    left_number, right_number = _as_number(left), _as_number(right)
    if left_number is not None and right_number is not None:
        left, right = left_number, right_number
    elif operator in ("<", "<=", ">", ">="):
        raise ConfigError(
            f"cannot order non-numeric values ({left!r} {operator} {right!r}) in {condition!r}"
        )
    else:
        left, right = _stringify(left), _stringify(right)
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    return left >= right
