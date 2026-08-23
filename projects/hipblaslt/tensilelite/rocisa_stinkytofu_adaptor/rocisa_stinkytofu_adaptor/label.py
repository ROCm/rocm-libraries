# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Assembly label management (LabelManager + magicGenerator).

Fully implemented; no stinkytofu dependency.
"""

from __future__ import annotations

import random


# ---------------------------------------------------------------------------
# magicGenerator -- 16-char random label tag.
# ---------------------------------------------------------------------------
#
# 36-char alphabet from rocisa (uppercase letters + digits, no
# punctuation, no lowercase -- assembler-safe in every dialect we
# target).

_MAGIC_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Effective output length.
#
# rocisa names the constant ``LABEL_NAME_LENGTH = 17`` but the
# generator loop iterates ``LABEL_NAME_LENGTH - 1 == 16`` times --
# so the returned string is 16 chars (NOT 17). We mirror the actual
# behavior here, not the misleading constant name.

_MAGIC_LENGTH = 16


def magicGenerator() -> str:  # noqa: N802 (public name; matches rocisa)
    """Return a 16-character random ``[A-Z0-9]`` string.

    Mirror of ``rocisa::magicGenerator``. Used by
    ``LabelManager.getUniqueName`` / ``getUniqueNamePrefix`` to seed
    fresh label names.

    Parity / non-parity:
        * Output character set and length are byte-identical to
          rocisa.
        * Randomness source is Python's ``random.choices`` (Mersenne
          Twister), NOT C++ ``rand()`` -- the bit pattern of any
          given draw will not match. This is intentional: the values
          flow into label names that live only inside a single kernel
          emission, never get persisted in golden asm files, so
          per-byte determinism with a particular seed is not part of
          the contract. Tests that need deterministic output should
          monkeypatch this function.
    """
    return "".join(random.choices(_MAGIC_CHARS, k=_MAGIC_LENGTH))


# ---------------------------------------------------------------------------
# LabelManager -- per-kernel label registry.
# ---------------------------------------------------------------------------
#
# Tracks per-name use counts so labels emitted into the same kernel
# are guaranteed unique. KernelWriter creates one ``LabelManager``
# per kernel and routes EVERY label name through it via
# ``getName`` / ``getNameInc`` / ``getUniqueName*``.
#
# Counter semantics (the only subtle bit):
#   * ``addName`` on a fresh name sets the count to 0; on an
#     existing name it adds 1.
#   * The qualified name returned for count ``c`` is:
#         c == 0 -> name
#         c >  0 -> name_<c>
#   * So the FIRST occurrence emits the bare name, the SECOND
#     emits ``name_1``, the THIRD ``name_2``, and so on. The count
#     is therefore "occurrences minus one", not "occurrences".
#
# Public API parity:
#   * Methods exposed = the methods rocisa's nanobind binding exposes
#     (label.cpp:38-64): ``addName / getName / getNameInc /
#     getNameIndex / getUniqueName / getUniqueNamePrefix``.
#   * ``getData`` is NOT exposed -- it's an internal C++ accessor
#     used only by deepcopy / pickle on the rocisa side. We mirror
#     that by gating dict access behind ``_labels`` (private) plus
#     the dunder protocol methods.


class LabelManager:
    """Per-kernel label registry; mirror of ``rocisa::LabelManager``.

    See the section header above for counter semantics and the
    public-API parity contract.
    """

    __slots__ = ("_labels",)

    def __init__(self) -> None:
        # ``dict[str, int]`` matches the rocisa
        # ``std::map<std::string, int>``. Insertion order is not part
        # of the contract (the C++ side iterates in sorted order via
        # std::map; nothing in KernelWriter or our public API reads
        # the order, so a vanilla dict is fine).
        self._labels: dict[str, int] = {}

    def addName(self, name: str) -> None:  # noqa: N802 (matches rocisa)
        """Bump (or insert at 0) the use-count for ``name``.

        Existing entry → ``+1``; new entry → inserted at ``0``.
        Mirror of ``LabelManager::addName``.
        """
        if name in self._labels:
            self._labels[name] += 1
        else:
            self._labels[name] = 0

    def getName(self, name: str) -> str:  # noqa: N802 (matches rocisa)
        """Return the qualified label string for ``name``.

        Inserts ``name`` at count 0 if absent (matches the C++
        ``m_labels[name] = 0`` side effect when the key is missing,
        triggered by both ``find() == end()`` AND the subsequent
        ``operator[]`` read). Returns ``name`` when the count is 0,
        else ``name_<count>``.

        This method is IDEMPOTENT for an already-tracked name -- it
        does NOT bump the counter. Use ``getNameInc`` for the
        bumping variant.
        """
        # ``setdefault`` mirrors the C++ "insert-on-miss-then-read"
        # pattern in a single call.
        count = self._labels.setdefault(name, 0)
        if count == 0:
            return name
        return f"{name}_{count}"

    def getNameInc(self, name: str) -> str:  # noqa: N802 (matches rocisa)
        """``addName(name)`` then return the qualified name.

        Mirror of ``LabelManager::getNameInc``. KernelWriter calls
        this for every NEW label emission so colliding names get
        suffix-disambiguated automatically.
        """
        self.addName(name)
        count = self._labels[name]
        if count == 0:
            return name
        return f"{name}_{count}"

    def getNameIndex(self, name: str, index: int) -> str:  # noqa: N802 (matches rocisa)
        """Return ``name`` (``index == 0``) or ``name_<index>``.

        Use this when you already know which occurrence you want
        (e.g. KernelWriter referring back to a specific iteration of
        a loop label). Raises:

        - ``RuntimeError("You have to add a label first ...")`` when
          the name has never been added.
        - ``RuntimeError("The index N exceeded. (> M)")`` when the
          requested index is greater than the recorded count.

        Both messages match ``std::runtime_error`` text from rocisa
        byte-for-byte so any consumer that string-matches on the
        exception keeps working.
        """
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

    def getUniqueName(self) -> str:  # noqa: N802 (matches rocisa)
        """Return a freshly-generated label name not yet tracked.

        Loops on ``magicGenerator`` until the candidate is absent
        from the registry, then routes through ``getName`` -- which
        inserts the candidate at count 0 and returns the bare name.
        Subsequent calls with the same candidate (which can't happen
        unless tests monkeypatch ``magicGenerator`` to be
        deterministic) would suffix-disambiguate.
        """
        name = magicGenerator()
        while name in self._labels:
            name = magicGenerator()
        return self.getName(name)

    def getUniqueNamePrefix(self, prefix: str) -> str:  # noqa: N802 (matches rocisa)
        """Like ``getUniqueName`` with a fixed ``<prefix>_`` head.

        KernelWriter uses this for human-readable unique labels such
        as ``loop_top_<magic>`` or ``waitcnt_<magic>``.
        """
        name = f"{prefix}_{magicGenerator()}"
        while name in self._labels:
            name = f"{prefix}_{magicGenerator()}"
        return self.getName(name)

    # ------------------------------------------------------------------
    # Pickle / deepcopy plumbing.
    #
    # rocisa's binding implements ``__deepcopy__`` /
    # ``__getstate__`` / ``__setstate__`` directly on the C++ class
    # (label.cpp:54-64). They use ``getData()`` (return-by-value of
    # the internal map) as the round-trip carrier, so the state
    # shape is a 1-tuple ``(dict,)``.
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo) -> "LabelManager":
        # Mirrors ``new LabelManager(self.getData())``: a fresh
        # manager seeded with a COPY of the underlying map. Counter
        # values are immutable ``int`` so a shallow dict copy
        # suffices for full isolation between original and clone.
        clone = LabelManager()
        clone._labels = dict(self._labels)
        memo[id(self)] = clone
        return clone

    def __getstate__(self):
        # 1-tuple shape matches rocisa (label.cpp:60). The inner
        # dict is COPIED to defend against post-pickle mutation of
        # the live manager bleeding into the pickled bytes (the
        # rocisa side gets this for free via return-by-value).
        return (dict(self._labels),)

    def __setstate__(self, state) -> None:
        # No ``__init__`` was called by pickle -- ``__new__`` gave us
        # an empty shell. Rebuild from scratch, mirroring rocisa's
        # placement-new ``new(&self) LabelManager(get<0>(state))``.
        (data,) = state
        self._labels = dict(data)

    def __repr__(self) -> str:
        return f"LabelManager({self._labels!r})"
