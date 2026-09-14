# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Output extraction, and the agent result-file contract.

An agent's stdout is prose wrapped in whatever its CLI emits. Parsing it for a file
path is a trap, so an agent step instead dictates a result file and validates it: a
missing file, invalid JSON, or a missing required key fails that step instead of
handing an empty string to the next agent.
"""
from __future__ import annotations

import glob as globlib
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import StepError
from .flow import OutputSpec


@dataclass(frozen=True)
class Artifacts:
    stdout_path: Path
    stderr_path: Path
    workdir: Path
    result_path: Path | None = None
    #: One read per log per step. Several extractors over the same log are normal (a
    #: regex, a tail, a JSON envelope); re-reading a multi-megabyte build log once per
    #: extractor is not. The cache dies with the step, so nothing is retained after it.
    _cache: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def path_for(self, source: str) -> Path:
        return self.stdout_path if source == "stdout" else self.stderr_path

    def text(self, source: str) -> str:
        if source not in self._cache:
            self._cache[source] = read_text(self.path_for(source))
        return self._cache[source]


class LazyText:
    """A log's contents, read only when something actually asks for them.

    `stdout` and `stderr` are implicit outputs of every step, but flows overwhelmingly
    pass the *paths* downstream. Materialising a build log into the run's state on the
    chance that some later condition reads it is how a long run ends up holding every
    log it ever produced. Deliberately uncached: retaining the text is the thing being
    avoided, and anything that needs it repeatedly should declare a real extractor.
    """

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path

    def __str__(self) -> str:
        return read_text(self.path)

    def __repr__(self) -> str:
        return f"LazyText({self.path})"


def read_text(path: Path) -> str:
    """Log text for humans and extractors. Undecodable bytes are replaced, not fatal --
    the byte-exact log on disk is the evidence, this is a view of it."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_json(path: Path, what: str) -> Any:
    """Parse a JSON file, turning every way that can fail into a `StepError`."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise StepError(f"{what}: {path} is not valid UTF-8: {error}") from None
    except OSError as error:
        raise StepError(f"{what}: {path} cannot be read: {error}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise StepError(f"{what}: {path} is not valid JSON: {error}") from None


def load_result(step_id: str, path: Path, required: list[str]) -> dict[str, Any]:
    """Read and check a step's result file. A violation fails the step."""
    if not path.is_file():
        raise StepError(
            f"step '{step_id}': result file {path} was not written. The prompt must "
            f"instruct the agent to write its structured result to that exact path."
        )
    data = read_json(path, f"step '{step_id}': result file")
    if not isinstance(data, dict):
        raise StepError(
            f"step '{step_id}': result file {path} must contain a JSON object"
        )
    missing = [key for key in required if key not in data]
    if missing:
        raise StepError(
            f"step '{step_id}': result file {path} is missing required key(s): "
            f"{', '.join(missing)} (has: {', '.join(sorted(data)) or 'nothing'})"
        )
    return data


def extract(
    spec: OutputSpec,
    argument: Any,
    artifacts: Artifacts,
    result: dict[str, Any] | None,
    exit_code: int,
) -> Any:
    value = _raw(spec, argument, artifacts, result, exit_code)
    return _coerce(spec, value)


def _raw(
    spec: OutputSpec,
    argument: Any,
    artifacts: Artifacts,
    result: dict[str, Any] | None,
    exit_code: int,
) -> Any:
    if spec.kind == "regex":
        text = artifacts.text(spec.source)
        try:
            matches = list(re.finditer(str(argument), text, re.MULTILINE))
        except re.error as error:
            raise StepError(
                f"output '{spec.name}': {argument!r} is not a valid regular "
                f"expression: {error}"
            ) from None
        if not matches:
            raise StepError(
                f"output '{spec.name}': pattern {argument!r} matched nothing in {spec.source} "
                f"({text.count(chr(10)) + 1} lines at {artifacts.path_for(spec.source)})"
            )
        # Last match wins: summaries print at the end, and a retry inside one log should
        # not be read as the final answer.
        match = matches[-1]
        try:
            return match.group(spec.group) if match.groups() else match.group(0)
        except IndexError:
            raise StepError(
                f"output '{spec.name}': no capture group {spec.group}"
            ) from None

    if spec.kind == "json":
        text = artifacts.text(spec.source)
        try:
            document: Any = json.loads(text)
        except json.JSONDecodeError as error:
            raise StepError(
                f"output '{spec.name}': {spec.source} is not JSON ({error}). If the tool "
                f"prints prose around its JSON, have it write a result file instead."
            ) from None
        return _json_path(spec, document, str(spec.path))

    if spec.kind == "json_file":
        if argument == "result":
            if result is None:
                raise StepError(f"output '{spec.name}': step has no result file")
            document = result
        else:
            path = Path(str(argument))
            if not path.is_file():
                raise StepError(f"output '{spec.name}': {path} does not exist")
            document = read_json(path, f"output '{spec.name}'")
        return _json_path(spec, document, str(spec.path))

    if spec.kind == "file":
        path = Path(str(argument))
        if not path.exists():
            raise StepError(
                f"output '{spec.name}': expected file {path} does not exist"
            )
        return str(path)

    if spec.kind == "glob":
        return sorted(globlib.glob(str(argument), recursive=True))

    # Content identity of a file the step touched. Pairing this with an assert is how a
    # flow proves a later step did NOT modify an artifact an earlier step produced --
    # tool permissions are a request, a hash is evidence.
    if spec.kind == "sha256":
        path = Path(str(argument))
        if not path.is_file():
            raise StepError(f"output '{spec.name}': {path} does not exist")
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    if spec.kind in ("tail", "lines"):
        count = int(argument)
        lines = artifacts.text(spec.source).splitlines()
        return "\n".join(lines[-count:] if spec.kind == "tail" else lines[:count])

    raise StepError(f"output '{spec.name}': unsupported extractor '{spec.kind}'")


def _json_path(spec: OutputSpec, document: Any, path: str) -> Any:
    current = document
    for token in _tokens(path):
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise StepError(
                    f"output '{spec.name}': index [{token}] is out of range in {path}"
                )
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                available = (
                    ", ".join(sorted(current))
                    if isinstance(current, dict)
                    else type(current).__name__
                )
                raise StepError(
                    f"output '{spec.name}': key '{token}' not found in {path}; available: {available}"
                )
            current = current[token]
    return current


def _tokens(path: str) -> list[str | int]:
    cleaned = path.strip()
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    tokens: list[str | int] = []
    for part in cleaned.split("."):
        if not part:
            continue
        name, *indices = part.replace("]", "").split("[")
        if name:
            tokens.append(name)
        tokens.extend(int(index) for index in indices if index != "")
    return tokens


def _coerce(spec: OutputSpec, value: Any) -> Any:
    if spec.type == "json":
        return value
    if spec.type == "count":
        if isinstance(value, (list, tuple, dict)):
            return len(value)
        raise StepError(
            f"output '{spec.name}': type 'count' needs a list or mapping, got "
            f"{type(value).__name__} ({value!r})"
        )
    # A structure reaching a scalar type means the extractor and the declared type
    # disagree. Passing it through unchanged is how an `int` output ends up holding
    # [1, 2] and every later comparison against it becomes meaningless.
    if isinstance(value, (list, tuple, dict)):
        raise StepError(
            f"output '{spec.name}': extracted a {type(value).__name__} where type "
            f"'{spec.type}' expects a single value ({value!r}). Use 'type: json' to keep "
            f"the structure, or 'type: count' for its length."
        )
    if spec.type == "string":
        return str(value)
    if spec.type == "int":
        try:
            return int(str(value).strip())
        except ValueError:
            raise StepError(
                f"output '{spec.name}': {value!r} is not an integer"
            ) from None
    if spec.type == "float":
        try:
            return float(str(value).strip())
        except ValueError:
            raise StepError(
                f"output '{spec.name}': {value!r} is not a number"
            ) from None
    if spec.type == "bool":
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise StepError(
            f"output '{spec.name}': {path} does not exist. The step reported a path that "
            f"was never written."
        )
    return str(path.resolve())
