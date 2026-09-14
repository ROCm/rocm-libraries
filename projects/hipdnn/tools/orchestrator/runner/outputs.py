# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Output extraction, and the agent result-file contract.

An agent's stdout is prose wrapped in whatever its CLI emits. Parsing it for a file
path is a trap, so an agent step instead dictates a result file and validates it: a
missing file, invalid JSON, or a missing required key fails that step -- with a retry
budget -- instead of handing an empty string to the next agent.
"""
from __future__ import annotations

import glob as globlib
import json
import re
from dataclasses import dataclass
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

    def text(self, source: str) -> str:
        path = self.stdout_path if source == "stdout" else self.stderr_path
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


def load_result(step_id: str, path: Path, required: list[str]) -> dict[str, Any]:
    """Read and check a step's result file. Raises StepError so `retries` can re-ask."""
    if not path.is_file():
        raise StepError(
            f"step '{step_id}': result file {path} was not written. The prompt must "
            f"instruct the agent to write its structured result to that exact path."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StepError(
            f"step '{step_id}': result file {path} is not valid JSON: {error}"
        ) from None
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
        matches = list(re.finditer(str(argument), text, re.MULTILINE))
        if not matches:
            raise StepError(
                f"output '{spec.name}': pattern {argument!r} matched nothing in {spec.source} "
                f"({artifacts.text(spec.source).count(chr(10)) + 1} lines at "
                f"{artifacts.stdout_path if spec.source == 'stdout' else artifacts.stderr_path})"
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
            document = json.loads(text)
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
            document: Any = result
        else:
            path = Path(str(argument))
            if not path.is_file():
                raise StepError(f"output '{spec.name}': {path} does not exist")
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise StepError(
                    f"output '{spec.name}': {path} is not valid JSON: {error}"
                ) from None
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
    if spec.type in ("string", "json") or isinstance(value, list):
        return value if spec.type == "json" or isinstance(value, list) else str(value)
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
