# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Column storage for the rows of a MatchingLibrary.

A matching library maps a problem-size key to a solution (plus a measured
speed for Equality/Range tables). The logic files for one architecture hold
tens of millions of such rows (about 40M for gfx942), and
TensileCreateLibrary's parent keeps all of them from logic loading until the
library files are written. As a list of ``{"key": [...], "index":
IndexSolutionLibrary, "speed": f}`` dicts each row costs roughly 300 to 400
bytes of Python objects (280 measured with shared key ints, about 370 in a
full gfx942 build where every row owns its ints); stored by column (an int64
key matrix, a float64 speed vector, and a list of solution references) it
costs under 60.

Everything the library code does with a table is supported on the column
form: ``len``, indexing (which materializes one legacy row dict), iteration,
``+=`` and ``sort`` for merging, and ``state()`` which produces exactly the
list of dicts the legacy form serializes to. A speed is a float or an int in
the logic files (``0`` marks an unmeasured size), and msgpack encodes the two
differently, so each row remembers which it was. Values that cannot be
stored exactly (a key element that is not a plain int, a speed that is
neither, an int speed beyond float64's exact range) make ``fromColumns``
raise ``NotExactlyRepresentable``; the caller warns and keeps the legacy
list. ``fromColumns`` returns None, silently, for an empty table and for an
interpreter without NumPy: a build environment that imports the source tree
without installing ``requirements.txt`` then loses the memory saving but
still builds (the driver warns about that once, in the parent).
"""

_numpyModule = None


def _numpy():
    """The numpy module, or None when it is not installed."""
    global _numpyModule
    if _numpyModule is None:
        try:
            import numpy

            _numpyModule = numpy
        except ImportError:
            _numpyModule = False
    return _numpyModule or None


def numpyAvailable() -> bool:
    """Whether column storage is possible in this interpreter."""
    return _numpy() is not None


class NotExactlyRepresentable(ValueError):
    """A table value that the column form cannot hold without changing it."""


class MatchingTable:
    __slots__ = ("keys", "solutions", "speeds", "speedIsInt", "_wrappers")

    def __init__(self, keys, solutions, speeds, speedIsInt=None):
        self.keys = keys  # (N, K) int64
        self.solutions = solutions  # list of N Contractions.Solution
        self.speeds = speeds  # None (no speed column), or (N,) float64
        self.speedIsInt = (
            speedIsInt  # None (all float), or (N,) bool: row's speed was an int
        )
        # Legacy rows share one IndexSolutionLibrary per solution; materialized
        # rows keep that identity through this cache. Not pickled.
        self._wrappers = None

    def __getstate__(self):
        return (self.keys, self.solutions, self.speeds, self.speedIsInt)

    def __setstate__(self, state):
        self.keys, self.solutions, self.speeds, self.speedIsInt = state
        self._wrappers = None

    # ----------------------------------------------------------------- build
    @classmethod
    def fromColumns(cls, keys, solutions, speeds):
        """A MatchingTable for parallel lists of key lists, solutions and speeds, or None.

        ``speeds`` is None for a table without a speed column. The result is
        sorted by key, like the legacy table after construction. None is
        returned when NumPy is missing or the table is empty (no key width to
        store); the caller then keeps the legacy row dicts. A value that the
        columns cannot hold exactly raises ``NotExactlyRepresentable`` so the
        caller can say why it fell back to row dicts.
        """
        np = _numpy()
        if np is None or not keys:
            return None
        keyLen = len(keys[0])
        if keyLen == 0:
            return None
        for key in keys:
            if len(key) != keyLen:
                raise NotExactlyRepresentable(
                    f"key {key!r} has {len(key)} elements, the first key has {keyLen}"
                )
            for v in key:
                if type(v) is not int:
                    raise NotExactlyRepresentable(
                        f"key element {v!r} is {type(v).__name__}, not int"
                    )
        try:
            keyArray = np.array(keys, dtype=np.int64)
        except (OverflowError, ValueError, TypeError) as e:
            raise NotExactlyRepresentable(f"a key element does not fit int64: {e}")
        speedArray = speedIsInt = None
        if speeds is not None:
            isInt = [type(v) is int for v in speeds]
            for i, v in enumerate(speeds):
                if not (isInt[i] or type(v) is float):
                    raise NotExactlyRepresentable(
                        f"speed {v!r} is {type(v).__name__}, not int or float"
                    )
            try:
                speedArray = np.array(speeds, dtype=np.float64)
            except (OverflowError, ValueError, TypeError) as e:
                raise NotExactlyRepresentable(f"a speed does not fit float64: {e}")
            if any(isInt):
                speedIsInt = np.array(isInt, dtype=bool)
                # An int must survive the trip through float64 unchanged.
                intValues = [v for v, i in zip(speeds, isInt) if i]
                for v, f in zip(intValues, speedArray[speedIsInt].tolist()):
                    if int(f) != v:
                        raise NotExactlyRepresentable(
                            f"int speed {v} is not exact in float64"
                        )
        table = cls(keyArray, list(solutions), speedArray, speedIsInt)
        table.sort()
        return table

    @classmethod
    def fromRows(cls, rows):
        """A MatchingTable holding ``rows`` (legacy row dicts); see ``fromColumns`` for None and errors."""
        if not rows:
            return None
        hasSpeed = "speed" in rows[0]
        if any(("speed" in row) != hasSpeed for row in rows):
            raise NotExactlyRepresentable("some rows have a speed and some do not")
        return cls.fromColumns(
            [row["key"] for row in rows],
            [row["index"].solution for row in rows],
            [row["speed"] for row in rows] if hasSpeed else None,
        )

    # ------------------------------------------------------------ sequence
    def __len__(self):
        return len(self.solutions)

    def _wrapper(self, solution):
        from .SolutionLibrary import IndexSolutionLibrary

        if self._wrappers is None:
            self._wrappers = {}
        wrapper = self._wrappers.get(id(solution))
        if wrapper is None or wrapper.solution is not solution:
            wrapper = self._wrappers[id(solution)] = IndexSolutionLibrary(solution)
        return wrapper

    def _row(self, i):
        row = {
            "key": self.keys[i].tolist(),
            "index": self._wrapper(self.solutions[i]),
        }
        if self.speeds is not None:
            speed = self.speeds[i].item()
            row["speed"] = (
                int(speed)
                if self.speedIsInt is not None and self.speedIsInt[i]
                else speed
            )
        return row

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._row(j) for j in range(*i.indices(len(self)))]
        return self._row(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self._row(i)

    def toRows(self):
        """The legacy list-of-dicts form."""
        return list(self)

    # --------------------------------------------------------------- merge
    def __iadd__(self, other):
        if not isinstance(other, MatchingTable):
            return NotImplemented
        np = _numpy()
        if len(other) == 0:
            return self
        if len(self) == 0:
            self.keys, self.solutions = other.keys, list(other.solutions)
            self.speeds, self.speedIsInt = other.speeds, other.speedIsInt
            return self
        if (self.speeds is None) != (other.speeds is None) or self.keys.shape[
            1
        ] != other.keys.shape[1]:
            raise ValueError("cannot merge matching tables of different shapes")
        n, m = len(self), len(other)
        self.keys = np.concatenate([self.keys, other.keys])
        self.solutions = self.solutions + other.solutions
        self._wrappers = None
        if self.speeds is not None:
            self.speeds = np.concatenate([self.speeds, other.speeds])
            if self.speedIsInt is not None or other.speedIsInt is not None:
                mine = (
                    self.speedIsInt
                    if self.speedIsInt is not None
                    else np.zeros(n, dtype=bool)
                )
                theirs = (
                    other.speedIsInt
                    if other.speedIsInt is not None
                    else np.zeros(m, dtype=bool)
                )
                self.speedIsInt = np.concatenate([mine, theirs])
        return self

    def sort(self, key=None):
        """Stable sort by key, lexicographic over the key columns (``key`` is ignored)."""
        n = len(self)
        if n < 2:
            return
        np = _numpy()
        order = np.lexsort(
            tuple(self.keys[:, j] for j in reversed(range(self.keys.shape[1])))
        )
        self.keys = self.keys[order]
        if self.speeds is not None:
            self.speeds = self.speeds[order]
        if self.speedIsInt is not None:
            self.speedIsInt = self.speedIsInt[order]
        self.solutions = [self.solutions[i] for i in order.tolist()]

    # --------------------------------------------------------------- state
    def state(self):
        """The serialized rows: the same list of dicts the legacy table yields through Utilities.state."""
        keys = self.keys.tolist()
        indices = [s.index for s in self.solutions]
        if self.speeds is None:
            return [{"key": k, "index": i} for k, i in zip(keys, indices)]
        speeds = self.speeds.tolist()
        if self.speedIsInt is not None:
            speeds = [
                int(s) if isInt else s
                for s, isInt in zip(speeds, self.speedIsInt.tolist())
            ]
        return [
            {"key": k, "index": i, "speed": s} for k, i, s in zip(keys, indices, speeds)
        ]
