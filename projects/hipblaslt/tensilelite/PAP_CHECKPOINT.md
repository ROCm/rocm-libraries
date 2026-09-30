# PAP implementation checkpoint (WIP)

Implementation paused on 2026-09-30 for a local checkpoint. This is unfinished
work, not a validated performance change. The new two-group path is selected
by existing PAP settings for its supported non-multicast geometry; its final
native completion proof is outstanding. GPU correctness and performance of
this draft have not been validated.

## Saved implementation

- Includes the existing PAP multicast correctness work: cluster bank handoff,
  phantom-tile identity preservation, persistent grid behavior, and regressions.
- Includes prepared next-tile addresses, descriptor lifetime fixes, positive
  group1 DepthU increments, two next-tile global-prefetch groups, and retained
  first-PLR operands through epilogue and next-tile setup.
- Group0 uses token0; group1 and surviving current F-bank reads use token1.
  The saved checkpoint adds a typed token0 completion barrier at the common
  next-entry consumer boundary. Multicast remains disabled for this new
  two-group/PLR path; existing legacy multicast behavior is separate.
- Rejects PAP configurations whose bias/vector-scale/local-split-U epilogue
  overwrites prefetched LDS banks. Existing StoreRemap rejection remains.

The saved incremental source/test patch is checkpoint04, SHA256
`0299a736e2fdf8cd8ccf7150dcfd790a6a7362913e1730e86fd5a61d87a1260a`.
It is newer than the latest independently emitted checkpoint03. Local detailed
artifacts remain under `triage/pap_optimization/resume/` and are not committed.

## Existing evidence and unresolved issues

- Before the last completion edit, 93 focused two-group/PLR cases passed; the
  final PLR helper suite passed 15 cases. The adjacent run already in progress
  at pause finished with 414 passed, 9 skipped, and 3 failures. All failures
  are `test_setup_new_tile_releases_waveidx_*` tests whose
  `_SetupNewTilePapTdmWriter` fixture lacks `papPgr2PipelineEnabled`. They are
  recorded without a follow-up fix or rerun.
- Checkpoint03 representative MX native emission/assembly passed at 104 SGPRs
  and 686 VGPRs (previous integrated draft: 104/678). All 102 between-group
  F-bank DS reads carry token1; all 204 retained operand VGPRs survive setup
  and epilogue. Emitted positive descriptor increments passed q2/3/4/6 and
  carry-boundary interpreter checks. This does not prove device correctness.
- Checkpoint03 carried PLR completion is unsafe: its 51-read producer order
  needs first-WMMA `dscnt <= 28`, but both main and q2 routes use `dscnt 41`
  without prior completion. Checkpoint04's typed late completion barrier is
  saved but has not been independently emitted or verified.
- A native `s_wait_tensorcnt 0` still executes before the active group1 bank
  release and drains group0. The intended overlap is therefore unproven.
  Early and legacy fallback group0 sites both use token0. A short-K CFG merge
  is only a hypothesis; the native cause has not been established.
- The in-progress no-MX emission completed at 74 SGPRs/655 VGPRs, but its token,
  lifetime, and wait details were not inspected. Final fallback, bank-parity,
  small-K, native completion, and multicast phase checks remain unfinished.
- No new tests, native builds, or GPU operations were started for this commit.
  Only source preservation, parse, whitespace, and secret checks were run.

## Unmerged native investigation

`triage/pap_checkpoint/rule3-bound-wip.patch` and `rule3_bound_repro.cpp`
preserve a small proposed bound on cluster-barrier Rule3 anchor scans and its
CPU reproducer. The patch is not applied to shared native source and is not
part of the enabled implementation. The patch has zero context; applying it
requires `git apply --unidiff-zero` from the monorepo root. Both original and
patched reproducer
builds failed to link on `stinkytofu::resolveSymbolicOperands`; neither ran.
The multicast extension still needs guarded bank-release and readiness phases,
native phase-balance validation, and resolution of the partial-wait behavior.
