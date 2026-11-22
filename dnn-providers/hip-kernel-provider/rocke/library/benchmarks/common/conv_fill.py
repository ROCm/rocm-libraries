"""Input data of the forward conv benchmarks (``--fill``).

``uniform`` (the default): x and w ~ U(-1, 1).

``miopen``: MIOpenDriver's forward data (``driver/conv_driver.hpp``):
- x ~ U(0, 0.01);
- w ~ 0.01 * U(-1/3, 1/2) for fp16 (``conv::RanGenWeights<float16>`` shifts it positive), and
  0.01 * U(-1/2, 1/2) otherwise.

This lets rocke be timed on the same data distribution as MIOpenDriver; the time of a
power-bound loop can depend on the magnitude of its data.

The fills are float32 numpy arrays drawn from a caller-owned
``numpy.random.Generator``; the caller rounds them to its storage dtype.

The verify error is peak-normalised: max|out - ref| / max|ref|. The default floors the
denominator at 1, which is what makes it a relative error on U(-1, 1) data. Small-magnitude data
needs a floor near zero, or every output would pass; ``ref_floor`` gives the floor for the mode.
"""

import numpy as np

# DIRECTIONS: the directions a kernel label starts with (KernelIdentity.label()).
from kernels.conv.kernel_identity import DIRECTIONS

FILL_CHOICES = ("uniform", "miopen")
_MODE = "uniform"


def set_mode(mode: str) -> None:
    global _MODE
    if mode not in FILL_CHOICES:
        raise ValueError(f"--fill {mode!r} not in {FILL_CHOICES}")
    _MODE = mode


def mode() -> str:
    return _MODE


def fill_x(shape, rng: np.random.Generator) -> np.ndarray:
    """New float32 input activations of ``shape``, drawn from ``rng`` per the mode."""
    lo, hi = (0.0, 0.01) if _MODE == "miopen" else (-1.0, 1.0)
    return rng.uniform(lo, hi, shape).astype(np.float32)


def fill_w(shape, dtype: str, rng: np.random.Generator) -> np.ndarray:
    """New float32 weights of ``shape`` (io dtype ``dtype``), drawn from ``rng``
    per the mode."""
    if _MODE == "miopen":
        lo, hi = (-1.0 / 3.0, 0.5) if dtype == "fp16" else (-0.5, 0.5)
        lo, hi = 0.01 * lo, 0.01 * hi
    else:
        lo, hi = -1.0, 1.0
    return rng.uniform(lo, hi, shape).astype(np.float32)


def fwd_operands(
    x_shape, w_shape, dtype: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``(x, w)`` of a forward benchmark: float32 arrays of the given shapes,
    filled per the mode for io dtype ``dtype``, x drawn from ``rng`` first."""
    x = fill_x(x_shape, rng)
    return x, fill_w(w_shape, dtype, rng)


def ref_floor(direction: str = "fwd") -> float:
    """Floor of the reference peak in the peak-normalised verify error.

    ``--fill`` only sets the forward inputs; dgrad / wgrad stay on U(-1, 1),
    so their verify keeps the floor of 1."""
    return 1e-30 if _MODE == "miopen" and direction == "fwd" else 1.0


def add_argument(parser) -> None:
    parser.add_argument(
        "--fill",
        choices=FILL_CHOICES,
        default="uniform",
        help="forward input data: uniform = U(-1,1) (default); miopen = MIOpenDriver's "
        "x ~ U(0,0.01), w ~ 0.01*U(-1/3,1/2) (fp16) / 0.01*U(-1/2,1/2)",
    )


def _labels(label: str) -> tuple:
    """The full labels a ``--kernels-file`` entry names: a direction-prefixed
    label is itself only; a bare one (the label without its direction) is the
    label in every direction, since the runners match ``identity.label()``."""
    if label.startswith(tuple(f"{d}-" for d in DIRECTIONS)):
        return (label,)
    return tuple(f"{d}-{label}" for d in DIRECTIONS)


def load_kernels_file(path):
    """``--kernels-file``: ``None`` (no restriction) or ``{case key or "*": labels}``.
    A line is a kernel label (allowed for every case) or ``case_key<TAB>label``;
    a label may leave out its direction prefix (``fwd-`` ...) to name the
    kernel in every direction."""
    if not path:
        return None
    out: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, label = line.rpartition("\t")
            key = key or "*"
            out.setdefault(key, set()).update(_labels(label))
    return out


def kernels_for(kernel_map, key: str) -> set:
    """The labels ``--kernels-file`` allows for case ``key``."""
    return kernel_map.get(key, set()) | kernel_map.get("*", set())
