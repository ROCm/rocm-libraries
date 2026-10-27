# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a tuned spec and a tunable request look like to the shared machinery."""

from __future__ import annotations

from typing import Mapping, Protocol, Tuple, runtime_checkable

from .axes import Knobs


@runtime_checkable
class TunedSpec(Protocol):
    """A family's concrete tuned spec (a frozen dataclass).

    ``tuning_id`` / ``config_key`` / ``knobs`` / ``variant_id`` name the
    configuration and are the same on every problem. ``identity()`` names the
    compiled kernel, problem and runtime specializations included; it is what
    ``KernelId.spec_hash`` hashes, so wrapper metadata stays out of it. The
    spec owns its launch geometry: candidate ``grid`` / ``block`` and the Torch
    binding all read these methods.
    """

    tuning_id: str
    variant_id: str
    config_key: str
    knobs: Knobs

    def identity(self) -> Mapping: ...

    def launch_block(self) -> Tuple[int, int, int]: ...


@runtime_checkable
class TunableRequest(Protocol):
    """A request that can pin a tuned configuration.

    ``tuning_id`` is ``"auto"`` (the candidate's default spec) or an id;
    ``tuning_knobs`` is the knob dict recorded next to that id, as sorted
    ``(name, value)`` pairs. Requests are frozen dataclasses, so both are
    hashable and survive a JSON round trip through ``normalized()``.
    """

    algorithm: str
    spec_id: str
    tuning_id: str
    tuning_knobs: Knobs
