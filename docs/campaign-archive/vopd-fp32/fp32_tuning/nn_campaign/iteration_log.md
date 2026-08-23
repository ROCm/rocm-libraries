# NN campaign + NT 8192³ fix + 3-way roundtrip — Log

Started 2026-06-03. Autonomous via master.sh.

## Goals
1. Fix NT 8192³ gap (was 13.3 TF tuned, because 8192³ never in NT tuned set) — add exact entry.
2. Tune NN orientation (transA=N transB=N, library Cijk_Ailk_Bljk) — seeded narrow, never tuned before.
3. 3-way generic-vs-tuned roundtrip (NT/TN/NN), save to ROUNDTRIP3_RESULTS.md.

## Verified orientation map
- NT = transA=N transB=T = Cijk_Ailk_Bjlk (tuned; +8192³ added here)
- TN = transA=T transB=N = Cijk_Alik_Bljk (tuned; untouched)
- NN = transA=N transB=N = Cijk_Ailk_Bljk (NEW; navi31 generic baseline existed)

## Part A — NT 8192³ fix
Config nt_fix/nt_8192.yaml: NT header, MT128x128-focused fork (TT 8x8/8x16/16x8, WG 16x8/8x16/16x16,
DU 8/16/32, GSU1, WGM 4/8, SU 0/32, LdsPad 0/2), shapes 8192³ + 8192²×{4096,2048} + 4096³ + 4096²×8192.
Early cache-warm result: 8192³ MT128x64/MT64x128 hit ~38-39k GFlops — confirms the fix works
(deployed logic was giving only 13.3 TF). force_merge onto deployed NT logic (backup: NT_pre8192_backup.yaml).

## Part B — NN seeded campaign
seed.yaml: NN header, seeded fork from NT winners (TT [8,16],[16,8],[8,8],[4,8],[8,4],[4,4],[2,2],[1,2],[1,1];
WG 16x8/8x16/8x8; DU 8/16/32/64; GSU1; WGM 4/8; SU 0/32; LdsPad 0/2) = 1728 configs pre-rejection
(vs ~2600 full grid). 866 shapes (860 from NT waves + 6 large squares incl 8192³, 4096³).
Deploy to gfx1100/GridBased/gfx1100_Cijk_Ailk_Bljk_S_B_UserArgs.yaml.

## Part C — 3-way roundtrip
roundtrip3.sh: remove all 3 tuned logics → rebuild → bench generic; restore → rebuild → bench tuned.
30 shapes (shapes_big.txt incl 8192³). Output: ROUNDTRIP3_RESULTS.md + rt3_{nt,tn,nn}_{generic,tuned}.csv.

## Automation
master.sh (PID logged in master.log) chains: wait NT-fix → merge → NN run → NN merge → rebuild →
roundtrip3. Markers: MASTER_COMPLETE in master.log, RESULTS_SAVED in roundtrip3.log.

## Progress
- [~] NT-fix Tensile running (shape1 ~133/323 at 01:05)
- [ ] NT merge | [ ] NN run | [ ] NN merge | [ ] rebuild | [ ] roundtrip3 | [ ] results saved
