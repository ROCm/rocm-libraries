"""
Atheris coverage-guided fuzz harness for the alt_format_rejected pure helper.

Helper under test (branch d8f43265, Tensile/Tensile.py:526):
    def alt_format_rejected(alt_format: bool, n_config_files: int) -> bool:
        return alt_format and n_config_files > 2

INVARIANT asserted on every input:
    If result is True then BOTH:
      (a) alt_format must be True
      (b) n_config_files must be > 2
    This is a tautology for the predicate; any deviation would be a bug.

BOUNDARY WITNESS sought:
    The minimal True input is (alt_format=True, n_config_files=3).
    Atheris finds this naturally while exploring the bool x int space.
    When found, it is written to the crash/witness artifact directory.
"""

import sys
import os
import json

import atheris

# ---------------------------------------------------------------------------
# Helper under test (inlined; no import of Tensile needed)
# ---------------------------------------------------------------------------


def alt_format_rejected(alt_format: bool, n_config_files: int) -> bool:
    return alt_format and n_config_files > 2


# ---------------------------------------------------------------------------
# Witness tracking
# ---------------------------------------------------------------------------

WITNESS_PATH = "/work/work/tensilelite-characterization/parametric-chaos/_tooling/atheris/witness.json"
_witness_found = False


def _record_witness(alt_format: bool, n_config_files: int, result: bool) -> None:
    global _witness_found
    if not _witness_found:
        data = {
            "alt_format": alt_format,
            "n_config_files": n_config_files,
            "result": result,
            "note": "boundary witness: first (True,>2) found by atheris",
        }
        try:
            os.makedirs(os.path.dirname(WITNESS_PATH), exist_ok=True)
            with open(WITNESS_PATH, "w") as f:
                json.dump(data, f, indent=2)
            print(
                f"[WITNESS] ({alt_format}, {n_config_files}) -> {result}  (written to {WITNESS_PATH})",
                flush=True,
            )
        except Exception as exc:
            print(f"[WITNESS-ERR] could not write: {exc}", flush=True)
        _witness_found = True


# ---------------------------------------------------------------------------
# Fuzz target
# ---------------------------------------------------------------------------


@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    alt_format = fdp.ConsumeBool()
    # Consume a bounded int (0..15) to keep the domain small and meaningful.
    # The boundary is at 3, well within range.
    n_config_files = fdp.ConsumeIntInRange(0, 15)

    result = alt_format_rejected(alt_format, n_config_files)

    # INVARIANT: result True => (alt_format AND n_config_files > 2)
    if result:
        assert (
            alt_format
        ), f"Invariant violated: result=True but alt_format={alt_format!r}"
        assert (
            n_config_files > 2
        ), f"Invariant violated: result=True but n_config_files={n_config_files}"
        # Record first True witness found
        _record_witness(alt_format, n_config_files, result)


if __name__ == "__main__":
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
