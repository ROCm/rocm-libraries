# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Duplicate-key-safe JSON reading with standard-library imports only.

Tools that read a JSON manifest without needing PyYAML import this module instead
of ``config_loader``, which imports ``yaml`` at module scope.
"""

import json
from typing import Any


class RepeatedJsonKey(Exception):
    """A JSON object declared one key twice."""


def reject_repeated_json_keys(pairs: list) -> dict:
    """``object_pairs_hook`` that refuses an object declaring one key twice."""
    keys = [k for k, _ in pairs]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise RepeatedJsonKey(
            f"found duplicate key {repeated[0]!r}; JSON would silently keep only the "
            f"last value"
        )
    return dict(pairs)


def loads(text: str) -> Any:
    """Parse JSON text. An object declaring one key twice raises `RepeatedJsonKey`;
    text that is not JSON raises ``ValueError``."""
    return json.loads(text, object_pairs_hook=reject_repeated_json_keys)
