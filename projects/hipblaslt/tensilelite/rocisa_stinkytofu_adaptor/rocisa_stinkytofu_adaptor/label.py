# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Assembly label management (LabelManager)."""

from __future__ import annotations


class LabelManager:
    """Per-kernel label registry; mirror of ``rocisa::LabelManager``."""

    __slots__ = ("_labels", "_counter")

    def __init__(self) -> None:
        self._labels: dict[str, int] = {}
        self._counter: int = 0

    def addName(self, name: str) -> None:  # noqa: N802
        if name in self._labels:
            self._labels[name] += 1
        else:
            self._labels[name] = 0

    def getName(self, name: str) -> str:  # noqa: N802
        count = self._labels.setdefault(name, 0)
        if count == 0:
            return name
        return f"{name}_{count}"

    def getNameInc(self, name: str) -> str:  # noqa: N802
        self.addName(name)
        count = self._labels[name]
        if count == 0:
            return name
        return f"{name}_{count}"

    def getNameIndex(self, name: str, index: int) -> str:  # noqa: N802
        if name not in self._labels:
            raise RuntimeError(
                "You have to add a label first to get a label name with specific index."
            )
        current = self._labels[name]
        if index > current:
            raise RuntimeError(f"The index {index} exceeded. (> {current})")
        if index == 0:
            return name
        return f"{name}_{index}"

    def getUniqueName(self) -> str:  # noqa: N802
        return self.getUniqueNamePrefix("label")

    def getUniqueNamePrefix(self, prefix: str) -> str:  # noqa: N802
        while True:
            name = f"{prefix}_{self._counter}"
            self._counter += 1
            if name not in self._labels:
                break
        return self.getName(name)

    def __deepcopy__(self, memo) -> "LabelManager":
        clone = LabelManager()
        clone._labels = dict(self._labels)
        clone._counter = self._counter
        memo[id(self)] = clone
        return clone

    def __getstate__(self):
        return (dict(self._labels), self._counter)

    def __setstate__(self, state) -> None:
        data, counter = state
        self._labels = dict(data)
        self._counter = counter

    def __repr__(self) -> str:
        return f"LabelManager({self._labels!r}, counter={self._counter})"
