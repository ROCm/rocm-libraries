"""CTest driver: run one architecture's emitted-bundle census in a fresh process.

The census itself is a GTest suite inside the provider's host test binary,
generated beside the descriptors it checks. It reads what actually loaded
through `discoverDescriptorSets()` -- the provider's real typed registration
followed by `loadValidatedDescriptorSets<Handle>()` -- and compares the loaded
pack and kernel identities, the runtime source kind and the SDK version against
the inventory the generator emitted. It proves registration, loading and
inventory; it proves nothing about dispatch.

This script exists for the two things ctest cannot express on its own:

* The architecture is supplied EXPLICITLY, as `HIPDNN_TEST_EXPECTED_ARCH`,
  together with the descriptor shard that architecture was packed into. Neither
  is detected from a device and neither is read back out of the descriptors
  under test.
* `--gtest_filter` selecting nothing is a PASS to gtest -- exit 0, "0 tests
  ran". A census whose suite was renamed, dropped from the binary, or misspelled
  in the registration table would then report green forever. So the filter is
  enumerated with `--gtest_list_tests` first and a zero-case selection fails.

Every invocation is its own process. Native registration and descriptor
discovery are `call_once`/memoized and process-global, so one process cannot
observe two registration states; a per-arch, per-mutation fresh process is the
only arrangement in which a negative case can actually go red.

Absence is a failure here, not a skip. The architectures come from the
configured packaging list, so a missing shard means the build was asked to pack
that architecture and did not.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _selected_case_count(binary: Path, gtest_filter: str, env: dict) -> int:
    """How many cases `gtest_filter` actually selects in `binary`.

    `--gtest_list_tests` prints suite lines flush-left and case lines indented,
    so the indented lines are the count. Parsing the listing rather than
    trusting the run keeps "selected nothing" distinguishable from "ran and
    passed".
    """
    listing = subprocess.run(
        [str(binary), f"--gtest_filter={gtest_filter}", "--gtest_list_tests"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if listing.returncode != 0:
        print(
            f"FAIL: {binary} could not enumerate its tests (exit "
            f"{listing.returncode}):\n{listing.stdout}{listing.stderr}"
        )
        return -1
    return sum(
        1
        for line in listing.stdout.splitlines()
        if line[:1] in (" ", "\t") and line.strip() and not line.strip().startswith("#")
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        required=True,
        help="bare gfx arch this shard was packed for, e.g. gfx942. Passed to the "
        "census as HIPDNN_TEST_EXPECTED_ARCH.",
    )
    parser.add_argument(
        "--descriptor-root",
        required=True,
        type=Path,
        help="the descriptor shard for --arch. Passed as HIPDNN_DESCRIPTOR_DIR.",
    )
    parser.add_argument(
        "--test-binary", required=True, type=Path, help="the provider host test binary"
    )
    parser.add_argument(
        "--gtest-filter",
        required=True,
        help="exact filter selecting this bundle's census suites; must select at "
        "least one case",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if not args.gtest_filter.strip():
        print(
            "FAIL: --gtest-filter is empty, which would select every test in the binary."
        )
        return 1

    if not args.test_binary.is_file():
        print(f"FAIL: test binary '{args.test_binary}' does not exist.")
        return 1

    if not args.descriptor_root.is_dir():
        print(
            f"FAIL: no descriptor shard for '{args.arch}' at "
            f"'{args.descriptor_root}'. That architecture is in the configured "
            f"packaging list, so the build was asked to pack it; an absent shard "
            f"is a packaging failure, not a reason to pass."
        )
        return 1

    env = dict(os.environ)
    env["HIPDNN_TEST_EXPECTED_ARCH"] = args.arch
    env["HIPDNN_DESCRIPTOR_DIR"] = str(args.descriptor_root)

    selected = _selected_case_count(args.test_binary, args.gtest_filter, env)
    if selected < 0:
        return 1
    if selected == 0:
        print(
            f"FAIL: '{args.gtest_filter}' selects no test case in "
            f"{args.test_binary.name}. gtest reports that as a pass; it means the "
            f"census suite is absent from the binary or the registration names it "
            f"wrongly."
        )
        return 1

    print(
        f"census: {selected} case(s) selected by '{args.gtest_filter}' for "
        f"{args.arch} against {args.descriptor_root}"
    )
    run = subprocess.run(
        [str(args.test_binary), f"--gtest_filter={args.gtest_filter}"],
        env=env,
        check=False,
    )
    return run.returncode


if __name__ == "__main__":
    sys.exit(main())
