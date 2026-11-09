#!/usr/bin/env python3
"""wp2a_gen.py (the frozen harness copy, unmodified) with ONE change: HcKnobs.tolerance is True only for ``acc != 1`` (its original meaning), so the
fast-math knobs (invmath=rsq32 / sigm=hw, default-off tolerance-class switches of the patched rocKE tree) are not refused by its "tolerance-class
knobs are not served" guard. Used only by the e2e fast-math arms (arms_e2e.sh); needs the patched rocKE tree (rocke-qk-qm1f) on PYTHONPATH.
Usage: wp2a_gen_fm.py <path of wp2a_gen.py> <its args...>
"""
import runpy
import sys

from kernels.gfx950.qwen38_mono.stages2 import hc as H

assert "invmath" in H.HcKnobs.__dataclass_fields__, "not the fast-math rocKE tree"
H.HcKnobs.tolerance = property(lambda s: s.acc != 1)
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
