#!/usr/bin/env python3
"""Package 5 representative GEMM kernels via okl.py.

Picks five shape classes that exercise different regimes of hipBLASLt's
heuristic dispatch:

  1. Skinny along M (small M, large N+K) -- bandwidth-skewed
  2. Skinny along N (large M+K, small N) -- bandwidth-skewed, other axis
  3. Small square  1024^3
  4. Medium square 4096^3
  5. Large square  8192^3

For each, runs `okl.py --package packages/<name>/`, leaving:
  packages/<name>/kernel.co    (the shipped .co shard, copied)
  packages/<name>/kernel.conf  (slot-list config for okl_run)
  packages/<name>/okl.json     (full okl.py JSON output)
  packages/<name>/okl.stderr   (any warnings/errors from the run)
  packages/<name>/okl.cmd      (verbatim command line for reproducibility)

Plus a top-level packages/summary.json with the chosen-solution metadata
for all five.

Run from this folder:
  python3 package_examples.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE     = Path(__file__).resolve().parent
OKL_PY   = HERE / "okl.py"
OUT_ROOT = HERE / "packages"

# Hardware-specific tooling -- adjust to match your install. These match
# the working ROCm 6.4.3 pairing on this box.
BENCH   = "/opt/rocm-6.4.3/bin/hipblaslt-bench"
LIBPATH = "/opt/rocm-6.4.3/lib/hipblaslt/library"

# All bf16 TN. Change DTYPE_FLAGS if you want a different dtype family.
SHAPES = [
    # (name,             M,    N,    K)
    ("skinny_M",         128,  4096, 4096),
    ("skinny_N",         4096, 128,  4096),
    ("small_square",     1024, 1024, 1024),
    ("medium_square",    4096, 4096, 4096),
    ("large_square",     8192, 8192, 8192),
]

DTYPE_FLAGS = [
    "--a-type", "bf16_r", "--b-type", "bf16_r",
    "--c-type", "bf16_r", "--d-type", "bf16_r",
    "--compute-type", "f32_r",
]


def package_one(name, m, n, k):
    """Invoke okl.py --package for one shape; capture json, stderr, cmd."""
    pkg_dir = OUT_ROOT / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(OKL_PY),
        "-m", str(m), "-n", str(n), "-k", str(k),
        "--transa", "T", "--transb", "N",
        *DTYPE_FLAGS,
        "--bench", BENCH,
        "--libpath", LIBPATH,
        "--package", str(pkg_dir),
    ]
    print(f"== {name}: M={m} N={n} K={k} ==")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    (pkg_dir / "okl.json").write_text(proc.stdout)
    (pkg_dir / "okl.stderr").write_text(proc.stderr)
    (pkg_dir / "okl.cmd").write_text(" ".join(cmd) + "\n")

    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}; see {pkg_dir}/okl.stderr")
        return None

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"  okl.py returned non-JSON; see {pkg_dir}/okl.json")
        return None

    pkg = info.get("package", {})
    if not pkg:
        print(f"  okl.py succeeded but did not emit a package; see {pkg_dir}/okl.json")
        return info

    sym = pkg.get("kernel_symbol", "?")
    print(f"  -> solution {info.get('solution_index')}")
    print(f"     kernel  {sym[:80]}{'...' if len(sym) > 80 else ''}")
    print(f"     MT {pkg.get('macro_tile_0')}x{pkg.get('macro_tile_1')}, "
          f"WG {pkg.get('workgroup_size_threads')}, "
          f"kernarg {pkg.get('kernarg_size')} bytes, "
          f"{pkg.get('num_buffers')} buffers")
    return info


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    results = {}
    for name, m, n, k in SHAPES:
        results[name] = {
            "shape": {"m": m, "n": n, "k": k,
                      "transa": "T", "transb": "N",
                      "a_type": "bf16_r", "compute_type": "f32_r"},
            "okl": package_one(name, m, n, k),
        }
        print()

    (OUT_ROOT / "summary.json").write_text(json.dumps(results, indent=2))

    print("=== Done ===")
    print(f"Packages under: {OUT_ROOT}")
    print(f"Summary:        {OUT_ROOT / 'summary.json'}")
    print()
    print("Run any of them with:")
    print(f"  ./okl_run packages/<name>/kernel.conf")


if __name__ == "__main__":
    main()
