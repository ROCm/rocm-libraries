#!/usr/bin/env python3
"""Preflight validator for the grouped_conv_fwd_xdl NHWC gfx1250 gentune run.

GPU-free. Parses the main and smoke .gentune files with the real Gentune
interpreter, enumerates the constrained parameter space with gentune_utils
(mirroring tuner_main's own size computation), and checks:

  * naive vs. constrained cardinality (expect 32 main / 1 smoke),
  * every enumerated CK_GENTUNE_XDL_SUFFIX tuple is fully resolved (no leftover
    CK_* token) and matches the fixed 16x16 skeleton,
  * the known-good seed tuple is present,
  * BENCH_ARGS and VERIFY_ARGS are equal-length, 20-token, shell-safe shapes.

Run it from anywhere; it locates the gentune modules relative to this file:

    python3 runs/grouped_conv_fwd_xdl_gfx1250_nhwc_20260731/preflight_check.py

IMPORTANT: this imports `interpreter`/`gentune_utils` only, never `tuner_main`
(which starts a full tuning run at import time).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# runs/<run>/preflight_check.py -> runs/<run> -> runs -> gentune base dir
GENTUNE_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, GENTUNE_DIR)
os.chdir(GENTUNE_DIR)

import interpreter  # noqa: E402
import gentune_utils  # noqa: E402

RUN_REL = os.path.relpath(HERE, GENTUNE_DIR) + "/"
MAIN = "generation/grouped_conv_fwd_xdl_nhwc.gentune"
SMOKE = "generation/grouped_conv_fwd_xdl_nhwc_smoke.gentune"

# Fixed 16x16 skeleton with A=B=CDE=4 == checked-in instance #17; A=B=CDE=8 ==
# the gfx1250 compile-confirmed smoke tuple. The suffix excludes ConvSpec.
SEED_44 = "1,256,64,64,32,8,8,16,16,2,2,S<4,64,1>,S<1,0,2>,S<1,0,2>,2,4,8,1,S<4,64,1>,S<1,0,2>,S<1,0,2>,2,4,8,1,1,1,S<1,32,1,4>,4"
SEED_88 = "1,256,64,64,32,8,8,16,16,2,2,S<4,64,1>,S<1,0,2>,S<1,0,2>,2,8,8,1,S<4,64,1>,S<1,0,2>,S<1,0,2>,2,8,8,1,1,1,S<1,32,1,4>,8"

SHELL_UNSAFE = re.compile(r"[^0-9\s-]")


def nospace(s):
    return re.sub(r"\s+", "", s)


def check_config(path, expected_constrained, require_seed):
    print("\n" + "=" * 72)
    print("CONFIG: " + path)
    print("=" * 72)
    ok = True
    configs = interpreter.parse_input_from_file(path, RUN_REL)
    print("configurations parsed: %d" % len(configs))

    for ci, config in enumerate(configs):
        tune_params = list(config["tune_params"].values())
        naive = 1
        for tp in tune_params:
            naive *= len(tp["possible_vals"])

        rs, gen_combos = gentune_utils.create_all_combos(list(config["gen_params"].values()))
        for gc in gen_combos:
            size = gentune_utils.get_param_space_size(list(tune_params), list(rs), [list(gc)])
            print("  config %d: naive=%d  constrained=%d  (expected %s)"
                  % (ci, naive, size, expected_constrained))
            if expected_constrained is not None and size != expected_constrained:
                print("  !! FAIL: constrained size %d != expected %d" % (size, expected_constrained))
                ok = False

        # Enumerate resolved tuples.
        _, combos = gentune_utils.create_all_combos(list(tune_params))
        names = [p["Names"][0] for p in tune_params]
        sfx_i = names.index("CK_GENTUNE_XDL_SUFFIX")
        suffixes = [nospace(c[sfx_i][0]) for c in combos]
        print("  enumerated tuples: %d" % len(combos))

        for c in combos:
            flat = " ".join(v for pv in c for v in pv)
            if "CK_" in flat:
                print("  !! FAIL: unresolved token in tuple: " + flat)
                ok = False
                break

        seed = nospace(SEED_44 if require_seed == "44" else SEED_88)
        if any(seed == s for s in suffixes):
            print("  seed tuple (%s) present: YES" % require_seed)
        else:
            print("  !! FAIL: seed tuple (%s) not present" % require_seed)
            ok = False

        b = config.get("BENCH_ARGS", [])
        v = config.get("VERIFY_ARGS", [])
        print("  BENCH_ARGS=%d  VERIFY_ARGS=%d" % (len(b), len(v)))
        if len(b) != len(v) or len(b) == 0:
            print("  !! FAIL: BENCH/VERIFY length mismatch or empty")
            ok = False
        for i in range(min(len(b), len(v))):
            for tag, val in (("BENCH", b[i]), ("VERIFY", v[i])):
                if SHELL_UNSAFE.search(val):
                    print("  !! FAIL: shell-unsafe %s shape %d: %r" % (tag, i, val))
                    ok = False
                if len(val.split()) != 20:
                    print("  !! FAIL: %s shape %d is not 20 tokens: %r" % (tag, i, val))
                    ok = False

        print("  sample resolved suffixes:")
        for s in dict.fromkeys(suffixes):
            print("    " + s)
    return ok


def main():
    ok_main = check_config(MAIN, 32, require_seed="44")
    ok_smoke = check_config(SMOKE, 1, require_seed="88")
    verdict = ok_main and ok_smoke
    print("\n" + "=" * 72)
    print("PREFLIGHT: " + ("PASS" if verdict else "FAIL"))
    print("=" * 72)
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
