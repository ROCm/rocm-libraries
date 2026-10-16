## BISECT — root cause of the complex CC failure (2026-10-16)

Method: full `cgemm_gfx1250.yaml` via test_config.py, GPU execute+validate, **5× each commit**
(fail is NON-deterministic: different (size,solution) fail per run, single run insufficient).
All bisect points share the **PR client** — `DataInitialization.cpp` (d6754) only touches MX-scale
swizzle, NOT complex, so client is ruled out.

Per-commit `cgemm_gfx1250` validFAIL (5 runs):

| commit | what | validFAIL |
|---|---|---|
| baseline 80400e1 | develop | 0 (1 run) |
| **adf32139f1 = 940^** | before feedsWmma | **0,0,0,0,0 — PASS** |
| **940fa1c94a** | **feedsWmma / Phase B WMMA-parent VALU** | **3,3,1,2,3 — FAIL** |
| 573cb6bbf4 | tip minus 7573cc64 | 3,3,4,4,4 — FAIL |
| 7573cc64 (unbound ds_loads) | — | RULED OUT (removing it still fails) |
| tip 249f1019a0 | — | 5,4 — FAIL |

**ROOT CAUSE = `940fa1c94a` (the feeder / feedsWmma).** 940^ passes, 940 fails. feedsWmma promotes
"any VALU feeding a wmma" into Phase B. complex CC has 272 `v_wmma_f32_16x16x4_f32` + deint VALUs
(`v_mov deint A im`, `v_add Ai=-Ai`) that get wrongly promoted like an MX-scale v_perm → reorders
the complex ops → non-deterministic race → wrong answer / hang.

Failing kernels are always **PGR2 + CC (conj A&B) + edge sizes 127-129 / K63-64**. A single
kernel+size in isolation PASSES (10×) — needs the full-run multi-size/solution context to trigger.
Repro kernel cad1b08b = `Cijk_Alik_C_Bjlk_C_C_B MT64x64x16 PGR2`.
940 changed: StinkyDAGSchedulerPass.cpp(+12), CDNA5.hpp(+268 Phase B), ReadyQueue.hpp(+4).

Next: read 940's feedsWmma mark + Phase B promotion, find which complex dep it breaks; fix =
restrict the mark to v_perm (MX-scale splat) only, not all VALU feeding a wmma.


结论：bug 不在 ds_load→v_mov→v_add→wmma 链上，而是 940 让主循环里的两组 split barrier 交错了

1. 日志线索

- cc_940_fullx5 的 5 次 run 里，失败全部落在 Cijk_Alik_C_Bjlk_C_C_B 的 PGR2 kernel 上（MT64x64x8 PGR2 和 MT64x64x16 PGR2），尺寸都是 K=64。
- 错误值和参考值数量级一样，只是部分偏差，像是少累加或累加错了一部分 K-slice。每次 run 的错误元素数都不同（64 到 1489 个），集中在若干 tile 里。
- 这种表现更像跨 wave 的 LDS 读写竞争（多个 wave 抢同一块 LDS，结果取决于时序），不像寄存器值算错。所以之前的 VALU 值检查是 0 违规，也说得通。

2. 汇编对比（cad1b08b，940^ vs 940，只看同步指令）

940^ (PASS) LoopBeginL:                      940 (FAIL) LoopBeginL:
  ds_load X1..X3 <- LDS0                       ds_load X1(部分) <- LDS0
                                               s_wait_tensorcnt 0
                                               s_barrier_signal -1   ← sig(LDS1) 被提前到这里 (L1894)
                                               ds_load X1剩余,X2,X3 <- LDS0
  s_wait_dscnt 0                               s_wait_dscnt 0
  s_barrier_signal -1  ┐ sig/wait(LDS0)        s_barrier_signal -1   ← 第二次 signal，中间没有 wait (L1990)
  s_barrier_wait   -1  ┘                       s_barrier_wait   -1   (L1991)
  tensor_load -> LDS0                          tensor_load -> LDS0
  ...wmma...                                   ...wmma...
  s_wait_tensorcnt 1                           s_barrier_wait   -1   ← sig(LDS1) 对应的 wait (L2198)
  s_barrier_signal -1  ┐ sig/wait(LDS1)
  s_barrier_wait   -1  ┘
  ds_load X0 <- LDS1                           ds_load X0 <- LDS1

940 每个 wave 连续 signal 两次之后才 wait。这样 wa 相位放行，但那时其他 wave 还在发 X1、X2、X3 的ds_load 读 LDS0。紧接着的 tensor_load 就会覆盖正在被读的 LDS0，导致 wrong answer。barrier 相位一旦错开，就可能出现
hang。这一条同时解释了两种症状。

3. 把所有 .co 反汇编扫一遍（脚本 ~/tmp/barrier_pai中间有没有 wait）

┌─────────────────┬────────────────────────────────────────────────────────────────────┐
│      build      │          CC (Alik_C_Bjlk_C)          │                    其他 4 个 problem                     │
├─────────────────┼────────────────────────────────────────────────────────────────────┤
│ baseline / 940^ │ 0 个 kernel                          │ 0                                                        │
├─────────────────┼────────────────────────────────────────────────────────────────────┤
│ 940             │ 6 个：全部 PGR2（MT32/64 × 8/16/32） │ 0                                                        │
├─────────────────┼────────────────────────────────────────────────────────────────────┤
│ 573 / tip       │ 6 个，同上                           │ Alik_Bljk MT32x32x8_PGR2 也有 1 个（目前还没观察到失败） │
└─────────────────┴────────────────────────────────────────────────────────────────────┘

在已有日志里，出问题的 kernel 和这个扫描结果完全对 CC 的 PGR2 kernel 上出现。

4. 机制

- 已确认的部分： DAG 里 barrier 之间的依赖只靠 LDSildImplicitDependencyPass）。sync LDS0 和 sync LDS1的 barrier token 不同，所以两组 barrier 之间没有依赖边，调度器可以合法地把 sig(LDS1) 提到 sig/wait(LDS0) 前面。940
  之前这个漏洞就存在，只是调度顺序恰好没触发。
- 推测，还没用 debug log 验证： 940 把喂给 wmma 的 v_add 移进了 Phase B 的 wmmaParentValuQueue。在 cycle 14 附近，前面几个 phase
  可能都选不出指令，于是落到 CDNA5.hpp:2847 的 bardy 的 barrierQueue.top()，也就是 sig(LDS1)。

修复方向（还没实现）： 在 DAG 构建时，把 region 内gnal/wait -1 按程序顺序串成一条依赖链，保证signal/wait 严格交替。这个修法不依赖 feedsWmma，也能堵住 tip 上 Alik_Bljk 那个还没触发的潜在问题。
需要的话，我可以先用 PASS_DEBUG 确认 sig(LDS1) 是不是从 drain 分支选出来的，或者直接把串链的修复加上，再用 cgemm 全量跑 5 次验证。新的结论已经写进 memory。

## FIX — split-barrier interleave in CDNA5ReadyQueue (2026-10-16, committed c91d4a9d7c, NOT pushed)

Branch `users/briannwu/mx128_rebase` @ tip 249f1019a0. Runs on **GPU1** (`HIP_VISIBLE_DEVICES=1`),
PreloadKernArgs=False local hack applied (MEC fw 0x97e = 2430; do NOT commit).

### 1. Root cause confirmed on today's build (static)

New scanner `~/scripts/barrier_pair_scan.py <cmpasm/st>`: flags any `s_barrier_signal -1` seen while a
previous `-1` signal has no `s_barrier_wait -1` yet.

- tip, cgemm_gfx1250: **7 kernels** double-signal, all in `label_LoopBeginL`:
  `0eb81ae6 6679f2a5 6ac40689 90155ca0 979b3108 cad1b08b fa42b5b7` (= 6 CC PGR2 + 1 Alik_Bljk PGR2,
  matches section 3 above).
- cad1b08b loop (tip): `sig -1` @L1900 (cycle 15) → ds_load X2/X3 **from LDS0** → `s_wait_dscnt 0` →
  `sig -1; wait -1 (sync LDS0)` @L1985 → `tensor_load -> LDS0`. The first wait completes once every
  wave reaches the hoisted L1900 signal, i.e. before slow waves finish reading LDS0 → tensor_load
  overwrites LDS being read. Same mechanism as section 2.

Why the DAG allows it: barriers are ordered only via LDS MemTokenData tokens
(StinkyBuildImplicitDependencyPass). Group LDS0 and group LDS1 share no token → no edge between
them → the queue may swap them (legal) **or interleave them** (illegal). 940 (feedsWmma) only
changed pick timing so the interleave happens; the hole predates it.

### 2. Rejected first attempt: chain all workgroup barriers in program order (DAG edges)

Fixed the 7 kernels (scan 0, 5/5 PASS) but changed **13** kernels: the extra 6 are
`Ailk_Bjlk_C PGR2` (1fa26029 3ae3ecf4 8a7bbe11 a0def882 bfcec394 dbec739f). There Tensile emits
group LDS1 before group LDS0, and the scheduler **legally** hoists the whole LDS0 sig/wait+tensor_load
group above LDS1. Program order kills that: tensor_load LDS0 moves e.g. cycle 350→1022 (3ae3ecf4),
878→2046 (8a7bbe11) = lost prefetch lead. Reverted.

Correct invariant is narrower: whole-group swap OK, **interleave** not OK → signal/wait must strictly
alternate in the *scheduled* order, not program order.

### 3. Fix: `shared/stinkytofu/src/transforms/asm/dag/CDNA5.hpp`

- `splitSignalOpen_`: set in `onScheduled` on an all-wave `s_barrier_signal`, cleared on an all-wave
  `s_barrier_wait`; reset per BB in `onInit`.
- `splitSignalBlocks(node)`: while a signal is open, any workgroup barrier other than an all-wave wait
  (another signal, legacy `s_barrier`) is ineligible.
- Folded into `isBarrierEligibleNow()` → covers decidePromote + `extractForcedBarrier` (threshold
  path) + clusterBarrier drain path.
- Plain drain path (`barrierQueue.top()`, which ignored eligibility when clusterBarrier is off — the
  suspected path in section 4) now picks the lowest-id barrier not blocked by an open signal.
- Last resort before the "all buckets empty" assert: if only blocked barriers remain (input already
  interleaved, wait can't become ready), issue `top()` as before + PASS_DEBUG note. No regression vs
  old behaviour, never hit in cgemm.
- Non-all-wave (cluster/expert) split barriers untouched. gfx1250 always uses CDNA5ReadyQueue.

### 4. Results (cgemm_gfx1250, GPU1)

| build | double-signal kernels | runtime |
|---|---|---|
| tip, no fix | 7 | 6/6 runs PASS (540/540 rows) |
| program-order chain (rejected) | 0 | 5/5 PASS, but 13 kernels changed |
| **CDNA5 open-signal fix** | **0** | **5/5 PASS (540/540 rows)** |

Per-run, final fix (`cgemm_fix2_r1..r5`, `HIP_VISIBLE_DEVICES=1`, same rocisa .so as the commit —
the commit only adds a clang-format comment realignment):

| run | yaml | PASSED rows | FAILED rows | clientExit != 0 |
|---|---|---|---|---|
| r1 | PASS | 540 | 0 | 0 |
| r2 | PASS | 540 | 0 | 0 |
| r3 | PASS | 540 | 0 | 0 |
| r4 | PASS | 540 | 0 | 0 |
| r5 | PASS | 540 | 0 | 0 |

Read with section 5: tip without the fix also passed 6/6 on GPU1, so these runs show **no
regression**, not a runtime fail→pass. The fail→pass evidence is the static scan (7 → 0).

- Fix changes **exactly the 7** broken kernels; other 83 .s byte-identical (incl. the 6 legal swaps).
- Modelled loop length unchanged for all 7 (e.g. cad1b08b 1021 = 1021), wmma count unchanged (272).
- cad1b08b fixed loop: `sig -1` @L1900 (early split signal, legal) … `wait -1 (sync LDS1)` @L2239 →
  `sig/wait (sync LDS0)` → `tensor_load -> LDS0` @L2242.
- Broad regression `-k 'gfx1250 or stinky_sia4'` with fix: stopped by request after 42 yamls —
  42 PASS, 0 FAIL, 0 failed rows.

### 5. Caveats / open items

- **Race did not reproduce at runtime on GPU1**: tip passed 6/6 here (vs 5,4 fails in the earlier
  bisect). Evidence for the fix = static scan + asm, not a GPU fail→pass flip.
- Did not PASS_DEBUG-verify which pick path chose the early sig(LDS1) on tip; fix guards all paths.
- Perf to measure: fixed cad1b08b keeps the early sig(LDS1), so the LDS0 group + tensor_load go to
  loop end. 940^ instead issued sig/wait(LDS0)+tensor_load early and sig/wait(LDS1) late. Which is
  faster needs a FLOPS run.
- No filecheck regression test yet (needs a stinkytofu-opt build; existing barrier tests are in
  `shared/stinkytofu/tests/filecheck/barrier_token_group*_ordering.stir`).
- Local-only changes: PreloadKernArgs=False in `tensilelite/Common/GlobalParameters.py`;
  `~/scripts/tox.sh` test path updated for the `Tensile/` → `tensilelite/` package rename.
- Artifacts: `~/experiments/2026-10-16/{cgemm_base_r1..6, cgemm_fix_r1..5 (rejected chain),
  cgemm_fix2_r1..5, all_gfx1250_fix2 (partial)}`.
