# gtest conventions

Full test-naming rules live on the wiki: https://github.com/ROCm/MIOpen/wiki/GTest-development#naming
(`Smoke`/`Standard`/`Full`/`Perf`/`Unit` prefixes, `CPU`/`GPU` hardware token, datatype
suffix). `check_names.py` enforces that schema against `miopen_gtest --gtest_list_tests`.

## hipDNN shim surface

A test belongs to the hipDNN backend-swap surface if its full gtest name contains the
token `HipdnnShim`. Parameterized tests get it from the instantiation prefix, e.g.
`INSTANTIATE_TEST_SUITE_P(HipdnnShim, GPU_ConvFwdApi_FP32, ...)`. Non-parameterized tests
carry it directly in the suite name instead, e.g. `TEST_F(GPU_HipdnnShimConvFwdApi_FP32, ...)`,
since `check_names.py`'s prefix check only applies to the `Smoke`/`Standard`/`Full`/`Perf`/
`Unit` token used by parameterized instantiations.

The token is deliberately specific: it is a filter, so a generic word like `Forwarding`
would sooner or later pull in an unrelated test and silently double its runtime.

Select the surface with `--gtest_filter='*HipdnnShim*'`. That filter is what the
`forwarding_parity` ctest entries replay under `MIOPEN_HIPDNN_FORWARDING=disabled` and
`=enabled`; a test outside the surface is never replayed, and a test inside it that is not
reachable through public entry points makes the comparison meaningless.

Two consequences of that being the only entry path, both easy to "fix" in the wrong
direction:

- **Shim-surface tests stay out of `test_categories.yaml` on purpose.** The categorized
  smoke/standard/full entries turn those patterns into a `--gtest_filter`, so a matching
  pattern would register a third run of tests the parity entries already run twice. Absence
  from that file is the design, not a gap.
- **Excluding a shim-surface test means narrowing the parity filter,** not calling
  `add_gtest_negative_filter`. That function feeds the shard filter, which the parity entries
  do not use, so disabling a shim test through it looks like it worked and changes nothing.

To qualify, a shim-surface test must:
1. Reach compute only through a public `miopen.h` entry point — never through
   `miopen::solver::` or `ProblemDescription` directly.
2. Validate against an independent CPU or analytically-known reference, not a
   self-comparison.
3. Use a tolerance appropriate for cross-implementation comparison, not
   bit-reproducibility.

See `hipdnn_shim_conv.cpp` for worked examples.

## Forwarding parity harness

With `MIOPEN_ENABLE_HIPDNN_WRAPPER=ON`, each `*_forwarding_parity` ctest entry runs
`script/run_forwarding_parity.py`. It replays the shim surface twice, once with
`MIOPEN_HIPDNN_FORWARDING=disabled` and once with `=enabled`, and requires the two runs to
agree test for test. The runs can differ only for entry points listed in
`kForwardingEntries` in `src/private/routing.cpp`. That list is empty today, so the entries
cannot fail yet. The harness is in place first so that the first forwarded entry point is
tested by something already known to work.

Why it is set up this way:

- **Only the shim surface is replayed.** The rest of the suite reaches compute through
  internal headers that bypass the wrapper, so replaying it would cost two full runs to prove
  nothing, and any flaky test in it would look like a real divergence.
- **One ctest entry, not three tied together by a fixture.** A sharded ctest run hands out
  whole entries, so the members of a fixture can land in different shards and fail there as
  unsatisfied. The script runs the two replays and the comparison itself, and stops before the
  comparison if a replay fails.
- **The gtest shard variables are pinned to one shard.** A sharded CI run exports
  `GTEST_TOTAL_SHARDS`/`GTEST_SHARD_INDEX`, and a replay that honoured them would compare only
  one shard's slice of the shim surface, with no other entry covering the rest.
- **One bare entry plus one per `ex_gpu_*` label.** CI selects tests by a tier label combined
  with an architecture filter, either `-L ^ex_gpu_<arch>$` on an architecture that
  `test_categories.yaml` declares or `-LE ex_gpu` elsewhere. No single entry survives both, so
  exactly one of these is selected under each. `ForwardingParityGpuLabels.cmake` takes the
  labels from the shared parser's output, so an architecture gets a parity entry exactly when
  the parser registers an enabled test for it, and stops the configure if the parser fails.
  `forwarding_parity` goes on the bare entry only, so `ctest -L forwarding_parity` replays the
  surface once.
- **Tier labels are `quick`, `standard`, `comprehensive` and `full`,** so the harness runs in
  every tier lane. Not `ffm-quick`/`ffm-full`: those run a fixed list of patterns on a tight
  time budget.
- **In a discrete build, only `test_hipdnn_shim_conv` gets an entry.** Any other binary would
  replay a filter that matches nothing, and the comparison rejects two empty runs.
- **No build-tree entry without a GPU.** The shim tests are all `GPU_` tests.

The packaged test list (`bin/MIOpen/CTestTestfile.cmake`) is generated from
`test_categories.yaml` and does not see `add_test()` calls, so the harness entries and
`wrapper_abi_check` are written into it separately. Without that, `ctest -L forwarding_parity`
on a packaged build would select nothing and pass. The packaged entries are not tied to
`MIOPEN_NO_GPU`, which describes the build machine; packages are often built without a GPU and
tested elsewhere. Only the single-binary build (`MIOPEN_TEST_DISCRETE=OFF`) ships them.

`wrapper_abi_check` (`script/check_wrapper_abi.py`) checks the wrapper's exported ABI from the
two built libraries. It loads neither, so it needs no GPU and is registered whenever the flag
is on. The packaged copy leaves out `--public-header`, which compares two source files that
the build-tree entry already checks.

`test_forwarding_parity_scripts` tests the harness scripts themselves. It needs no GPU or
build output, so it carries only the tier labels. It uses `unittest` rather than `pytest`,
because nothing installs `pytest` on a machine that builds MIOpen.

Two cache variables tune the harness:

- `MIOPEN_FORWARDING_PARITY_FILTER` (default `*HipdnnShim*`): the gtest filter replayed.
- `MIOPEN_FORWARDING_PARITY_TIMEOUT` (default `3600`): seconds per entry, covering both
  replays and the comparison.
