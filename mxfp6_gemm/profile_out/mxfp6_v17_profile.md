# MXFP6 GEMM v17 性能分析报告 (8192³, full mode)

> 被测: `profile_v17` 单 dispatch，dispatcher 在 8192³ 选中的生产配置
> `mxfp6_gemm_pipeline<M_TILE=128, NPW_A=8, NPW_V=0, N_WAVES=4, MIN_OCC=1, WAVES_M=2, WAVES_N=2, SWZ=16>`
> = T512 tile / 2×8 waves / depth-1 软件预取 / L2-aware swizzle(SWZ=16)。
> 采集: 2-agent 并行 (Agent A 动态/GPU, Agent B 静态/CPU)，ROCm 7.0.2.1 + rocprofv3。

---

## 1. 硬件环境

| 项 | 值 |
|---|---|
| GPU | AMD Instinct MI350X (gfx950, CDNA4) |
| CU | 256 (32 SE × 8 CU；8 XCD) |
| SIMD/CU | 4，每 SIMD 16-lane VALU + 独立 Matrix Core |
| 寄存器/SIMD | 256 Arch VGPR + 256 AccVGPR (独立池，AccVGPR 不占 occupancy) |
| LDS | 160 KB/CU, 32 banks |
| 时钟 | ~2200 MHz (gfx950, rocm-smi 不可用，按额定值) |
| ROCm | 7.0.2.1, rocprofv3 |

FP6 MFMA 峰值 (v_mfma_scale_f32_32x32x64, 32 cyc):
`256 CU × 4 × (32·32·64·2) / 32 × 2.2e9 = 9227 TFLOPs`

---

## 2. Benchmark 数据

| M | N | K | 时间 | TFLOPs | % of FP6 峰值 |
|---|---|---|---|---|---|
| 8192 | 8192 | 8192 | **0.712 ms** | **1545** | **16.7%** |

warmup=3, repeat=10 (best of 4×20)。与 memory 记录的 v17+swz 记录 (1557) 一致。

---

## 3. 资源配置 (kernel trace)

| 项 | 值 | 说明 |
|---|---|---|
| Arch VGPR | **212** | occupancy 限制项之一 |
| AccVGPR | **256** (a[0:255]) | 16 个 acc 瓦片 × 16 reg，全部用满 (静态反汇编实测) |
| SGPR | 32 | |
| LDS | **0** | 全程 HBM→VGPR 直读，无 LDS staging |
| Workgroup | 256 线程 = 4 waves | WAVES_M×WAVES_N = 2×2 |
| Grid | 64×16 = 1024 WG (4096 waves) | M/128 × N/512 |
| Kernel 时长 | 726,371 ns ≈ 0.726 ms | |

**Occupancy = 1 wave/SIMD (occ1)**。
- Arch VGPR: floor(256/212) = 1 → occ1
- AccVGPR: 256/256 = 1 → occ1
即使压低 Arch VGPR，**256 AccVGPR 全满**仍把 occupancy 钉死在 1。这是「大寄存器 tile + 低 occ」设计的刻意取舍 (16 acc 瓦片摊薄 A 的非合并加载)，不是 bug。

> ⚠️ kernel trace 的 `Accum_VGPR_Count` 报 0，是 rocprofv3 对 inline-asm AGPR 的已知统计盲区；以静态反汇编 a[0:255]=256 为准。

---

## 4. PMC 指标分析

### 4.1 指令统计 (全 GPU 聚合；SQ_WAVES=4096 = 总 wave 数)

| Counter | 值 | per-wave |
|---|---|---|
| SQ_WAVES | 4,096 | — |
| SQ_INSTS_VALU (含 MFMA) | 73,486,336 | 17,941 |
| SQ_INSTS_MFMA | 8,388,608 | **2,048** |
| pure VALU (VALU−MFMA) | 65,097,728 | 15,893 |
| SQ_INSTS_LDS | **0** | 0 (无 LDS) |
| SQ_INSTS_VALU_MFMA_F6F4 | 8,388,608 | = MFMA 总数 ✓ |

MFMA/wave 实测 2048 = 理论 (128 k_iter × 16 MFMA/iter) **完全吻合**。
pure VALU/wave 15,893 条主要是地址算术 + scale 字节抽取 + next→current 寄存器拷贝 (每 iter ~50 条 v_mov)。

### 4.2 利用率 (关键修正)

| 指标 | 值 | 方法 |
|---|---|---|
| **MFMA duty (墙钟法)** | **16.4%** | 4 WG/CU × 2048 MFMA × 32 cyc ÷ 墙钟 1,598,016 cyc |
| 交叉验证: bench % of peak | **16.7%** | 1545/9227 |
| VALU util | 23.9% | SQ_ACTIVE_INST_VALU / SQ_BUSY_CU_CYCLES |
| ~~SQ 比值法 MFMA util~~ | ~~78.3%~~ | ❌ 聚合口径不一致，弃用 (见下) |

> **重要修正**: `SQ_VALU_MFMA_BUSY_CYCLES = 268,435,456 = MFMA数(8,388,608) × 32`，是对所有 MFMA × 32cyc 流水线的**全 SIMD 求和**；而 `SQ_BUSY_CU_CYCLES` 是 **per-CU** 计数。两者口径差约 4×，直接相除得 78% 是假象。用墙钟法得 **MFMA 真实占空比 ≈ 16.4%**，且与 benchmark 的 16.7%-of-peak 独立吻合。
>
> **结论: 该 kernel 不是 compute-bound。Matrix Core 有 ~84% 时间在空转，被全局加载延迟卡住。**

### 4.3 LDS Bank Conflict

SQ_INSTS_LDS = 0, SQ_LDS_BANK_CONFLICT = 0, SQ_LDS_IDX_ACTIVE = 0。
**零 LDS、零 bank conflict** —— B 在 host 端预 shuffle 成 1536B/瓦片，A 打包成 48B/k-iter/行，全部直读到 VGPR，彻底绕开 LDS。

---

## 5. 瓶颈排名 (基于 PMC)

1. **全局加载延迟 (HBM latency)** —— 主瓶颈。occ1 下只有 1 wave/SIMD，没有第二个 wave 来填 MFMA 之间的加载延迟空窗。
2. **MFMA 占空比仅 16.4%** —— 是 1 的直接结果，不是 Matrix Core 本身不够快。
3. 非 MFMA VALU 开销 (地址算术 + scale 抽取 + 50 movs/iter 双缓冲拷贝) —— 大部分藏在 MFMA 阴影里，但处于 iter 间关键路径。
4. Epilogue 写出延迟 —— 一次性，占比小。

---

## 6. ATT 逐条指令分析 (full mode)

整体 stall% = **71.4%** (total_stall 28,847,876 / total_lat 40,425,892)。
按类别: `s_waitcnt` 52.2% | MFMA 23.6% | global_load 17.5% | global_store 4.7%。

### Top stall 热点 (按总 stall)

| vaddr | 指令 | hit | 总 stall | stall/hit | 语义 |
|---|---|---|---|---|---|
| **0x2e18** | `s_waitcnt vmcnt(5)` | 16,256 | **13,208,456** | **812.5** !!! | 每 iter 等本轮预取的 A/B 加载落地 → **暴露的 HBM 延迟** |
| 0x2e7c | `s_waitcnt vmcnt(0)` | 16,384 | 1,406,132 | 85.8 ! | 等最后的 B[n=0] (最后发射) 到达，MFMA 才能起跑 |
| 0x38dc | `global_store_dwordx4` (首条) | 128 | 775,980 | 6062 !!! | epilogue 首次写 D 的 HBM 写延迟 (一次性) |
| 0x2e90~0x2f70 | 16× `v_mfma_scale` | 16,384 | ~455K each (∑≈7.3M) | ~27.8 check | 背靠背 MFMA 流水线占用 (= 真实计算) |
| 0x2e2c | `s_waitcnt vmcnt(4)` | 16,256 | 426,180 | 26.2 check | 等 scale_B 落地 |

### Pipeline stage 时间分布 (每 K-iter)

```
load-wait  vmcnt(5) 812 + vmcnt(4) 26 + vmcnt(0) 86 ≈ 924 cyc   ← 主导
MFMA       16 × 27.8                              ≈ 445 cyc
其它 (scale抽取/地址/50 movs双缓冲拷贝)            余量, 多藏在阴影
```

**每个 K 迭代 ~924 cyc 在等加载、~445 cyc 在算** → 加载延迟 ≈ 2× 计算时间。这正是 16.4% 占空比的微观来源。

### 关键 ISA 观察 (静态)

- 16 条 MFMA **背靠背、无 s_nop**：写入不相交的 AGPR 区，调度正确，MFMA 之间无浪费的协同执行窗口。
- K-loop 是**真分支循环** (非全展开)，128 次，循环体 ~281 条。
- depth-1 预取生效：ki+1 的 22 条 load 在 ki 的 MFMA 前发射；但 occ1 下 ~445cyc MFMA 盖不住 ~900cyc 加载延迟，残差暴露在 vmcnt(5)。
- **load 顺序 vs 消费顺序错位**: 第一条 MFMA 消费 B[n=0]，而 B[n=0] 是**最后发射**的 load (#21-22)，由 vmcnt(0) 兜底。
- highest VGPR = v161；AGPR 用满 a[0:255] → 实锤 occ1。

---

## 7. 优化建议 (按优先级 + 预期收益)

> 前提: v17 已是当前调优记录 (1545 TFLOPs)。以下是攻 16.4% 占空比天花板的方向，难度递增。

1. **load 顺序对齐消费顺序** (低风险, 预期 +2~5%)
   把首批被 MFMA 消费的 B[n=0]/A[m=0] **最先发射**，并用细粒度 vmcnt 让前几条 MFMA 在其操作数一落地就起跑，缩短 vmcnt(5)/vmcnt(0) 的暴露。当前 B[n=0] 最后发射是反的。

2. **降低每 iter 加载字节 / 提升 L2 命中** (中风险, swizzle 已吃到 +4.4%)
   SWZ=16 已让相邻 WG 共享 B band 留 L2 热。可探索更大的 M 方向 WG 串行复用 A，或对 A 也做 L2-friendly 重排。

3. **occ 与 tile 的取舍前沿** (高风险, 可能回归)
   occ1 是 256 AccVGPR 用满导致。把 16-acc 大瓦片拆成 occ2×8-acc 能多一个 wave 藏延迟，但 memory 记录显示「大 tile 低 occ」是摊薄 A 非合并加载的制胜杠杆——此前实验 occ2 反而更慢。属已探明的回归区，谨慎。

4. **深化预取 (depth-2)** (高风险, 大概率 spill)
   再加一层预取需 ~2× 的 A/B 当前缓冲 VGPR；现已 212/256 Arch + 256/256 AGPR，几无余量，必然 spill。不建议。

5. **削减 iter 间 VALU 关键路径** (低收益)
   50 条 next→current `v_mov` 双缓冲拷贝多藏在 MFMA 阴影里，收益有限；可试 ping-pong 索引消除拷贝，但此前记录显示动态 reg-array 索引会 spill。

**总评**: v17 在 8192³ 已逼近其架构取舍下的上限。真实瓶颈是 occ1 下暴露的 HBM 加载延迟 (每 iter ~924cyc 等待 vs ~445cyc 计算)。

---

## 8. 后续修复实录 (2026-06-02, 已落地)

针对 §6 发现的「depth-1 copy-prefetch 零 overlap」做了深入修复，结论与实测：

**根因确认**: 原 copy-based prefetch 里 MFMA 消费的是 next→current 拷贝结果(VGPR-to-VGPR mov)，而非 load 结果。编译器看不到 load→MFMA 依赖，于是在每个 MFMA cluster 前插 `s_waitcnt vmcnt(0)` 全量 drain → 把刚发射的预取 load 全等完才算 → **零 overlap**。

**证伪的尝试**(均未能让 wait 下沉到 MFMA 之后，逐一实测):
- 源码重排(extract/copy 移到 MFMA 后)、去 `volatile`、换 `__builtin` intrinsic MFMA、`__builtin_amdgcn_sched_barrier(0)`、显式 post-MFMA `s_waitcnt` —— 编译器始终把 vmcnt(0) 钉在 MFMA 前。
- same-iteration(无预取，MFMA 直接吃 load)→ 编译器确实给出 relative vmcnt 重叠，但丢了预取 head-start，反而 **1451 (-6%)**。

**有效修复**: **编译期 2×展开 ping-pong 双缓冲**(静态 buf0/buf1，无动态 reg-array 索引→不 spill)。每个 do_mfma 消费的是「上半迭代就 load 好」的 buffer，编译器把它的 vmcnt 绑到该 buffer(基本已完成)的 load 上，同时另一 buffer 刚发射的 load 留在 in-flight(vmcnt 保持高位 vmcnt(24)/(22))覆盖 MFMA cluster → **真 overlap**。

**实测(同 harness, 2 次均值, 噪声 ~±1.2%)**:
| 形状 | pick | 原 | ping-pong | Δ |
|---|---|---|---|---|
| 8192³ | swz | 1559 | **1632** | **+4.7%** |
| 8192×4096 | N512 | 1626 | 1668 | +2.6% |
| 4096×8192 | swz | 1630 | 1670 | +2.5% |
| 2048×8192 | swz | 1532 | 1554 | +1.5% |

**门控**: 仅用于纯 AGPR occ1 tile (`NPW_V==0`: N512/swz)。V2 混合累加器 tile (NPW_V>0: N576/N640) 因 acc_v 额外占 Arch VGPR，ping-pong 双缓冲加压在 8192×5120 上回归 ~-1.8%，故保留原 copy-prefetch(`else if constexpr`)。correctness 8/8 err=0，dispatch 无回归。

**剩余天花板**: 即便完美 overlap，occ1 单 wave 下 MFMA(~445cyc) < load latency(~880cyc)，残差 ~435cyc 仍暴露。再进一步需改 occ/tile 取舍或减字节(L2)。N576 ping-pong 曾测得 +5.7% 但样本少、且与 N640 回归同源，未启用——留作后续(按 tile 而非按 NPW_V 细化门控)。
