# HANDOFF: MXFP6 高性能 GEMM Kernel (MI350/CDNA4)

## Goal

基于 HIP + 当前 ROCm 在 MI350 (CDNA4, gfx950) 上开发 MXFP6 (FP6 E2M3 + E8M0 block scale) 的高性能 GEMM kernel。以最简单的头文件+源文件形式实现，不依赖 CK 框架。目标问题规模：M=2K~8K, K/N=4K/8K。用户用中文交流。

## TL;DR — 当前状态（先读这段）

- **生产 kernel = `mxfp6_gemm/test_pipeline_v18.cpp`**（v18 = v17 dispatcher + **LDS 深 K 范式**作为新 tile 候选 TLDS）。LDS kernel 在 `mxfp6_gemm/mxfp6_lds.hpp`。v17（`test_pipeline_v17.cpp`）保留作 LDS-free 基线。
- **换范式成功（Step 23, 06-03）**：寄存器直读不再是天花板。**LDS 深 K 暂存**（256×256, KT192 双缓冲, 32×32×64, 深 K 窗口 1536cyc > load 880cyc）在大对齐方阵打赢 V17：**8192³ 1671(+3.1%)**、4096² +3.4%、8192×4096 +1.6%、4096×8192 +1.7%、2048×8192 +3.9%。dispatcher cost model 自动只在填满 CU 的大方阵选 LDS,小矩阵/V2 形状(5120/7680/9216)回退 V17。**12/12 OPT, 10/10 正确, 零回归**。
- v17 部分(仍是非 LDS 形状的最优)：8192³ ~1620 N512[swz];非 2 幂 8192×5120 ≈ 1720 N640[V2]。
- **上轮进展**：full-mode profile 定位到 occ1 下 load/MFMA 零 overlap → 用**编译期 2×展开双缓冲（occ1, NPW_V==0）**取代旧的 copy-based depth-1 预取，8192³ **1557→1628（+4.5%）**。correctness 8/8 err=0，dispatch 12/12 OPT，无 spill。已 push（force-push 到 `zhewan/ck/mxfp6-standalone`）+ 同步 NFS。
- **最新进展（本轮 06-03）= 瓶颈定盘 + 与 CK 对标，无代码改动并入生产**：用户从 RCV 观察到 global_load 发射 stall。彻查结论（Step 22）：**load 发射 stall 不可独立解，是 occ1 latency 天花板的症状**；三法交错全证伪（sched_barrier −8~13%、sched_group_barrier+intrinsic MFMA **−88%**）。轻量 PMC 实测（非 ATT）定盘：**L2 命中 91.3%、HBM 仅 ~10% 带宽（非带宽瓶颈）、等待 1.9× 发射（latency-bound）、occ2 已证伪**；暴露延迟来自 9% L2 miss 尾（每条 880cyc 捅破 512cyc MFMA 窗口）。**与 CK MXFP6 对标：我们 ≈ 0.5×**（8192³ 1632=17.7% vs CK 3200=34.7%）—— 差距是**架构范式**（CK 用 async DMA+LDS 复用 A，我们每个 K 从 global 重读 A），不是调参能补的。
- **⚠️ 已证伪、别重试**：① 教科书「对称 ping-pong」occ1 反而 −1.8%（Step 21）；② 6 种让编译器把 load-wait 下沉到 MFMA 之后的手段全失败（Step 21）；③ **任何 load 发射交错**（sched_barrier/sched_group_barrier，破坏 ping-pong 的 in-flight overlap，vmcnt 从 24/22 塌到 0/1）—— 官方 CK Cliff Notes Lesson 10 也确认 sched_group_barrier 只是 advisory（Step 22）。

**MI350 硬件规格：**
- 256 CUs (8 XCD × 32 CU), 4 SIMDs/CU, max 8 waves/SIMD, wave=64
- LDS: 160 KB/CU, 64 banks；VGPR: 256 Arch + 256 AccVGPR（独立池，AccVGPR 不占 occupancy）
- HBM ~6.5 TB/s；clock ~2.2 GHz
- MFMA 32x32x64 FP6: **32 cycles**；src0/src1/src2/vdst 均可 VGPR 或 AccVGPR (ACC/ACC_CD bit)；gfx950 global_store 可直接从 AccVGPR 读
- FP6 MFMA 峰值 = 256×4×(32·32·64·2)/32×2.2e9 ≈ **9227 TFLOPs**

---

## Current Progress

### Step 1–6 ✅ 基础设施 + 单 MFMA + K 循环 + multi-tile（全部 err=0）
- `mxfp6_types.hpp` FP6/E8M0 编解码 + dense packing；`mxfp6_preprocess.hpp` 量化/转置/scale 重排+合并/B pre-shuffle；`mxfp6_reference.hpp` CPU golden；`test_reference.cpp` 162 checks pass。
- `mxfp6_asm_utils.hpp` inline-asm MFMA（`"v"`/`"a"` 约束控制寄存器文件）、LDS、store。
- **TransposeC**（交换 src0/src1，每 lane 持 1 M-row×16 N-cols）+ **B pre-shuffle**（host 端拆 section0 dwordx4 + section1 dwordx2 → 100% 合并）。**v6i 类型**（MFMA FP6 操作数须精确 6 连续 VGPR）。

### Step 7–9 ✅ pipeline + 编译器 waitcnt + 大 tile 突破
- **编译器管理 waitcnt**：把 load 从 inline asm 拿出来改普通 typed 指针读，SIInsertWaitcnts 自动插相对计数并与 MFMA 重叠（手写 `vmcnt(0)` 做不到）。藏 asm 里编译器看不见→零插入→race。
- **真相①**：A 行主序每 lane 读不同行 → 全局加载天生非合并（cache line 用 37.5%），这是 ~770 的墙；合法范围内 A 加载方式不是杠杆。
- **真相②（突破）**：主导杠杆 = Volkov 式**大寄存器 tile + 低 occupancy**（每 wave 多做 MFMA，把 A 非合并加载摊到更多计算）。v10 N512（16 acc=256 AGPR，occ1，`__launch_bounds__`）8192³ = **1375**。

### Step 10–13 ✅ 三个瓶颈假设连续证伪
- v11 预取（动态索引 spill）/ v12 LDS 合并 / v13 ki-unroll 减地址算术 —— **藏延迟没用→非 latency；砍访存量没用→非 throughput；加指令就更慢→issue/compute bound**。occ1 此 tile 形状 ~1376 近天花板。
- **Tile 自适应（v14）**：纯 host 按 (M,N) 选 N_TILE 填满 256 CU，小矩阵 +32~68%，大矩阵零回归。**Split-K（v15）**：只救连 N128 都填不满 CU 的极瘦矩阵（512×1024 +259%），正常形状零触发。

### Step 14 ✅ 混合累加器 N640 / V2（非 2 幂 N +5~12%）
occ1 合并 VGPR 池（arch+acc=512）在 N512 只用 388/512，闲 124 arch VGPR。让溢出的 acc tile 走 Arch VGPR（MFMA 输出 ACC_CD=0）→ 单 wave 持 >256 AGPR 累加器：N512 的 16 acc + 4 溢出 = **N640 的 20 acc**。⚠️ `N_TILE=WAVES_N×NPW×32` 须整除 N，N640=2^7×5 整除不了 2 幂 N（4096/8192）。**V2 只吃非 2 幂 N**：8192²×**5120**（真实 LLM dim）+11.6%、×7680 +7.6%。N576（2^6×9，18 acc）覆盖另一种可整除性。sweet spot=20 acc，22 掉/24 spill。

### Step 15 ✅ 生产 dispatcher 合一（v17）
`mxfp6_gemm_pipeline<M_TILE,NPW_A,NPW_V,N_WAVES,MIN_OCC,WAVES_M,WAVES_N,SWZ>`：`NPW_A` 列走 AGPR、`NPW_V` 列溢出进 Arch VGPR。统一 cost model `cost(tile)=ceil(WG/256)/(WG×eff)` 扫 5 档（N128/256/512/576/640）取最低。`choose_tile`/`choose_swz` 生产 dispatcher，12/12 OPT。

### Step 16–17 ✅ A 合并最终证伪 + L2 swizzle
- v18 受控实验：occ1 big-tile 下 global→LDS 合并加载 A 反而 **−18~32%**（多一跳 + 每 slab 2 barrier 无第二 wave 顶，全暴露）。**A 合并这条路在 occ1 彻底堵死；876cyc 是 occ1 延迟暴露，非「非合并」本身。**
- **L2-aware WG swizzle**：不碰 compute/访存，只重排 WG→(m,n)，让相邻在飞 WG 共享一段 B 留 L2 热。8192³ N512 1491→**1557（+4.4%）**。门控 `choose_swz`：仅 `nb=N/512≥16`（N≥8192）启用 SWZ=16，只按 N 不看 M。

### Step 18 ✅ 仓库整理：只留生产路径
删光 v2~v15/v18 全部迭代 + 脚手架 + 产物。生产路径 = `test_pipeline_v17.cpp` + 4 个 `mxfp6_*.hpp` + `test_reference.cpp` + `Makefile` + `counters.txt` + `.gitignore`。

### Step 19 ✅ v17 full-mode profile（8192³，gpu-profile skill 2-agent）
交付物在 `mxfp6_gemm/profile_out/`：`mxfp6_v17_profile.md` / `mxfp6_v17_annotated.asm` / `mxfp6_v17_raw.asm` / `mxfp6_v17_rcv_trace*.tar.gz`（RCV 可直接打开的 `ui_output_*`）。专用单 dispatch 驱动 = `profile_v17.cpp`（CLI `warmup repeat`，硬连 8192³/T512/swz16）。

**核心结论：latency-bound，不是 compute-bound。**
- **MFMA 真实占空比 ≈ 16.4%**（墙钟法：4WG/CU × 2048MFMA × 32cyc ÷ 1.598M 墙钟 cyc），与 bench 16.7%-of-peak 独立吻合。
- 整体 instruction-level stall **71.4%**，其中 `s_waitcnt`（等 load）占 **52%**。零 LDS、零 bank conflict。occ1（256 AccVGPR 全满 a[0:255] 钉死，非 Arch VGPR 212）。
- **⚠️ PMC 陷阱**：`SQ_VALU_MFMA_BUSY_CYCLES = MFMA数×32`（全 SIMD 求和），而 `SQ_BUSY_CU_CYCLES` 是 per-CU；直接相除得 78% 是假象。用墙钟法或 bench %-of-peak。

### Step 20 ✅ occ1 双缓冲真 overlap（取代 depth-1 copy-prefetch，8192³ +4.5%）

**根因确认**：旧 depth-1 copy-prefetch **零 compute/load overlap**。MFMA 消费的是 `next→current` 拷贝结果（VGPR mov）而非 load 结果 → 编译器看不见 load→MFMA 依赖 → 在每个 MFMA cluster 前插 `s_waitcnt vmcnt(0)` 全量 drain → 串行 load→wait→MFMA。ATT 实锤：vmcnt 单条 812cyc/iter，占全 stall 45.8%。

**有效修复 = 编译期 2×展开静态双缓冲**（`buf0/buf1` 写死变量，无动态 reg-array 索引→不 spill）。MFMA **直接读 load 进来的 buffer**（非 copy 结果）→ 编译器给**相对 vmcnt**，让无关 load 留 in-flight 覆盖 MFMA。
- 门控 **`NPW_V==0`**（纯 AGPR：N512/swz）。V2 混合累加器（NPW_V>0：N576/N640）保留 copy-prefetch —— ping-pong 双缓冲加压在 8192×5120 回归 ~−1.8%（acc_v 额外占 Arch VGPR）。
- 实测（2 次均值，同 harness）：8192³ swz **1559→1628（+4.5%）**，8192×4096 N512 +2.6%，4096×8192 swz +2.5%，2048×8192 +1.5%。correctness 8/8 err=0，dispatch 无回归，无 spill。

### Step 21 ✅ 「对称 ping-pong」证伪 + 回退（反直觉负结果）

当前 occ1 双缓冲是**非对称**的（每 body：`L(buf1←ki+1); M(buf0); L(buf0←ki+2); M(buf1)` → buf0 提前一整轮 load=warm，buf1 同轮 load 同轮用=cold）。试了教科书**对称版**（prologue 预取 buf0+buf1 两个，消费后才 refill，两个 MFMA 都吃上一轮 load 的）：

- 实测 **1599 vs 非对称 1628 = −1.8%**（生产 harness，各 3 跑，稳定）。
- ATT 原因：occ1 单 wave **喂不动两个 warm buffer**。对称化把非对称的「一个 warm（vmcnt ~4cyc）+ 一个 cold（vmcnt5 ~727cyc）」变成「两个 cold（~900cyc ×2）」。单 wave 的 MFMA（~445cyc）只够藏 ~一个 cluster 的 load 延迟（~880cyc），**集中藏在一个 buffer 比摊到两个好**。
- **结论：当前非对称 2×展开就是 occ1 最优，已回退保留之，别再改对称。**

**让 load-wait 下沉到 MFMA 之后的 6 种尝试全部失败**（编译器始终把 `vmcnt(0)` 钉在 copy-prefetch 的 MFMA 前）：① 源码重排 extract/copy 到 MFMA 后；② 去 `volatile`；③ 换 `__builtin_amdgcn_mfma_scale` intrinsic；④ `__builtin_amdgcn_sched_barrier(0)` 围栏；⑤ 显式 post-MFMA `s_waitcnt`；⑥ same-iteration（MFMA 直吃 load → 能拿相对 vmcnt 但丢预取 head-start，−6% 至 1451）。**唯一有效的就是 Step 20 的双缓冲（让 MFMA 直接消费上一轮 load 进的 buffer）。**

### Step 22 ✅ load 发射 stall 彻查 + 瓶颈定盘 + CK 对标（06-03，无代码并入生产）

用户从 RCV 观察到大量 `global_load` **发射前** stall。三方查证（本地实验 + 本地 CDNA4 ISA PDF + AMD 内网 Confluence + 轻量 PMC），结论：**不可独立解，是 occ1 latency 天花板的症状。**

**A. 交错发射三法全证伪**（baseline 8192³=1627）：
- 手工交错源码 + `__builtin_amdgcn_sched_barrier(0)` 硬栅栏 → 1500（−8%）；mask(1) → 1481。
- builtin MFMA 本身 OK 且中性：`__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`（`USE_BUILTIN_MFMA` 宏门控 `mxfp6_asm_utils.hpp` 的 AccTileA v6i overload；src0=B/scale_b, src1=A/scale_a, v6i 零填 int32x8, cbsz=blgp=2, opsel=0；CK `scale_gfx9.hpp:158` 范式）→ 8/8 正确、AGPR=256 保住、VGPR=165、1616 vs 1627。**scale 传真字节就对，旧 memory「127 归零」坑不触发。**
- builtin + `sched_group_barrier` 细交错（CK mx_flatmm IGLP 配方，0x008=MFMA/0x020=VMEM read）→ **暴跌 ~200（−88%）**。smoking gun（`--save-temps` vmcnt 直方图）：生产 clustered=vmcnt(24)/(22) 大量 in-flight=overlap；交错=11×vmcnt(0)+18×vmcnt(1) 一发即 drain=零 overlap。
- **根因**：occ1 prefetch 命脉=load 扎堆+尽早发射+MFMA 期间在飞；任何穿插都逼编译器每个小 VMEM group 后 drain，摧毁 overlap。发射 stall 是 clustering-prefetch 的必要代价（两害相权轻者）。**全 clustered 即最优。**

**B. 轻量 PMC 定盘**（rocprofv3 非 ATT，profile_v17 单 dispatch 8192³）：
- L2: `TCC_HIT_sum`/`TCC_MISS_sum` = **命中 91.3%**；HBM 读 ≈7.17M miss×64B ≈459MB/0.712ms ≈ **645 GB/s = 仅 10% 峰值 → 非带宽瓶颈**。
- SQ: `SQ_WAIT_ANY`(79.1M)/`SQ_BUSY_CYCLES`(41.8M)= **等待 1.9× 发射（latency-bound）**；`SQ_ACTIVE_INST_VMEM≈0`（发射时间不在 VMEM 上 → 印证瓶颈是等数据非轮不到发，故交错全败）。
- **真相**：latency-bound，暴露延迟主要来自 **9% L2 miss 尾**（occ1 单 wave 下每条 HBM-miss load ~880cyc 捅破 512cyc MFMA 窗口顶住整 wave）。L2-hit 部分(~150cyc)基本藏住。

**C. 与 CK 对标（同 5 shape，本会话实测 vs 用户截图 2026-05-04）**：本 kernel ≈ **CK MXFP6 的 0.5×**。8192³：本 1632(17.7%) vs CK MXFP6 3200(34.7%) vs CK MXFP8 3019。差距是**架构范式**：CK 走 async DMA(`global_load_lds`)→LDS→`ds_read` **复用 A**（不每个 K 从 DRAM 重读），B 直读，IGLP 调度，VMEM 命中 L2 在大 MFMA 窗口藏住（见内网 [1652698858] MX Flatmm gfx950 调度分析）。我们每个 K 从 global 重读 A+B → 算/载比天然低一倍。

**净结论**：1628(17.6%) 已近「occ1 大 tile + 全局直读 VGPR」设计的实用上限。已排除：带宽、occ、调度/交错、L2 带宽。要追上 CK 的 ~2× 必须换范式（A 走 LDS 复用）。

### Step 23 ✅ 换范式成功：LDS 深 K 暂存打赢 V17 (+1.6~3.9% 大方阵，06-03)

用户指出 HANDOFF 旧的「追 CK 2×」前提是错的（那张 2026-05-04 截图很可能是 CK MXFP8/配置不同；CK mxfp6 实测**不如** V17）。并点破 **CK 的 16×16×128 MFMA 密度不对**（FLOPs/指令半、操作数带宽/FLOP 翻倍 vs 32×32×64）——这是 CK mxfp6 慢的部分原因。结论：不照抄 CK,但剥离出 CK 的一个**好点子单独用**:深 K 在 LDS 暂存。

**机制（不是 occupancy! CK 和我们都 occ1）**：深 K 在 LDS 暂存把"一次 global load 对应的 MFMA 计算窗口"做大到 >> load 延迟(880cyc)。寄存器做不到(深 K 操作数 spill),LDS 可以(160KB)。正面解决 Step 22 诊断的"MFMA 窗口 512cyc < load 880cyc"。

**赢的配置**：256×256 tile(16 acc/wave)+ **KT192**(SUBS=3, window=16×3×32=1536cyc)+ 双缓冲 LDS(144KB)+ **32×32×64**(不用 16×16×128)+ swz(门控 N≥M)。data path: `global_load_lds_dwordx4`(async 零 VGPR, lane×16 LDS 布局实测)→ `__syncthreads` → `ds_read`(v6i)→ MFMA。A/B 对称,不用 B pre-shuffle/scale coalesce(LDS gather 自理)。

**踩坑(关键)**：① scale 是头号杀手(naive -19%);typed scale load 和 inline-asm glds 有 **vmcnt 冲突**(编译器为 typed load 插 wait 会 drain 在飞的 glds prefetch)→ 解法 = **asm 手动 vmcnt + tile-grouped 布局 + 单条 dwordx{SUBS} 宽 load**(每 tile scale 6 条→2 条 vmem op,1503→1669)。② 深 K 整除约束:KT192 不整除 K=8192 曾丢 K 尾算假象 → **K padding 补零**到 KT 倍数(多 0.78% 无用算)。③ buf 双缓冲须编译期选(2× ping-pong),动态 saR[kt&1] 会 spill。④ SUBS≤4(scale dwordx4 上限,KT≤256)。

**深 K 已榨到头**：KT192 DB 16-acc(1677)是对称 LDS 天花板。KT256 single(丢重叠)1224 / KT256 DB 只能配 8-acc(962-984) / KT128 DB(1465) 全更差。唯一更深活口 = A-LDS-deep + B-direct 非对称 hybrid(未做,Step 24)。

**集成**:`test_pipeline_v18.cpp` 把 LDS 作为 cost model 新 tile(TLDS, eff=1.05, WG=(M/256)(N/256))。LDS 用 plain B + tiled scale + K pad,V17 用 preshuffled B + coalesced scale,A 共用→prep 分两路。设备代码抽 `mxfp6_lds.hpp` 共用(test_lds.cpp 是 LDS 实验驱动)。glds 布局探针 = `probe_glds.cpp`。见 memory [[mxfp6-lds-paradigm]] [[mxfp6-glds-layout]]。

---

## 🔑 核心技术要点

### occ1 双缓冲 / waitcnt 机制（本项目最关键的一课）
- 编译器对 **copy-prefetch**（MFMA 读 VGPR-copy 结果）→ 看不见 load 依赖 → MFMA 前 `vmcnt(0)` 全 drain → 零 overlap。
- 编译器对 **直接消费**（MFMA 读 load 进的 buffer）→ 插相对 `vmcnt(N)` → 无关 load 留 in-flight → overlap。
- 实现 overlap 的正解：**编译期 2×展开静态双缓冲**（不能用 `buf[ki%2]` 动态索引，会 spill）。
- occ1 单 wave 天花板：MFMA(445cyc) < load latency(880cyc)，单 wave 只够藏一个 cluster；再进需动 occ/tile 或减字节(L2)。

### MFMA 32x32x64 f8f6f4 数据布局 (ISA 7.1.5)
```
src0 = A[M][K]: lane l → A[m=l%32][k_half=l/32], 32 FP6 = 24 bytes (6 DWORD)
src1 = B[K][N]: lane l → B[k_half=l/32][n=l%32], 32 FP6 = 24 bytes (6 DWORD)
```
TransposeC（调用 `mfma(B, A, ...)`）：每 lane 持 1 M-row×16 N-cols，`acc[p]→N=(p%4)+(p/4)*8+m_half*4`，4 组连续 acc→4 连续 N→`global_store_dwordx4`。参数映射 `mfma(src0=B, src1=A, cbsz=B_fmt, blgp=A_fmt, scale_a=B_scale, scale_b=A_scale)`，FP6 cbsz=blgp=2。

### B Pre-shuffle（100% 合并）
原始 B^T[N][K] 每行 48B（stride 48B→33% 合并）。拆两段：section0 `tid×16B`(DWORD0-3)→dwordx4；section1 `tid×8B`(DWORD4-5)→dwordx2。每 32-col tile 共 1536B。

### AccVGPR (gfx950)
MFMA inline asm `"a"` 约束→累加器入 AccVGPR；`v16f{}=0` 自动 `v_accvgpr_write_b32`；`global_store_dwordx4` 直接从 a[..] 写 HBM；A/B 用 `"v"` 保持 Arch VGPR。**`.vgpr_count` metadata = arch+acc 合并值**（算 arch 须减 256）。

### inline asm `"memory"` clobber 原则
只在真读写内存时加：✅ ds_read/ds_write/global_load_lds、s_waitcnt（fence）；❌ v_mov、MFMA（纯寄存器，加了会强制 reload 缓存值）。

### scale 布局 (ISA p65)
`lane 0-15:dim0-15,K0-31 | 16-31:dim16-31,K0-31 | 32-47:dim0-15,K32-63 | 48-63:dim16-31,K32-63`。scale 已 coalesce（v17：每 wave 的 M/N 块按字节连续打包，一次宽 load + 字节抽取，杀掉 per-MFMA 的 vmcnt 瀑布）。

---

## 代码文件

```
mxfp6_gemm/
├── mxfp6_types.hpp          # FP6/E8M0 编解码 + dense packing
├── mxfp6_preprocess.hpp     # 量化/反量化, B 转置, scale 重排+合并, B pre-shuffle
├── mxfp6_reference.hpp      # CPU golden reference GEMM
├── mxfp6_asm_utils.hpp      # GPU ASM: MFMA(inline asm, AccVGPR/Arch VGPR), LDS, store
│                            #   + USE_BUILTIN_MFMA 宏分支(Step 22, inert/生产不定义, 走原 asm)
├── test_reference.cpp       # CPU 单元测试 (162 pass) — ground truth
├── test_pipeline_v17.cpp    # ★生产 kernel + dispatcher: tile-shrink + V2(N640) +
│                            #   occ1 2×双缓冲(NPW_V==0)/copy-prefetch(V2) + 4x4 + L2 swizzle
├── profile_v17.cpp          # 单 dispatch profiling 驱动 (CLI: warmup repeat; 硬连 8192³/swz16)
├── exp_interleave.cpp       # Step22 交错实验(SB_MASK/USE_SGB 宏); 全证伪, 留作参考
├── profile_out/             # full-mode profile 交付物 (md/annotated.asm/raw.asm/rcv tar.gz)
├── Makefile / counters.txt / .gitignore
```

历史迭代 v2~v15/v18、脚手架、实验全部删除（演进见 Step 1–22 与关联 memory；代码可从 git 历史取回）。`*.tar.gz` 与二进制在 `.gitignore`，只在 /tmp + NFS。Step 22 唯一生产文件改动 = `mxfp6_asm_utils.hpp` 的 `USE_BUILTIN_MFMA` 宏分支（12 行，生产不定义→走原 inline asm，行为不变；留作已验证的 intrinsic 等价路径）；exp_interleave.cpp + 实验二进制(exp_sgb/tp_builtin)未提交。

**Git**：分支 `zhewan/ck/mxfp6-standalone`。本轮因本地（06-02，v17-only consolidated）与远端（06-01，含旧探索文件的并行重组）分叉，按用户决定 **force-push 本地覆盖远端**（远端颗粒化 commit 历史被丢弃，内容上本地更新/等价）。NFS 备份 = `/home/AMD/zhewan/rocm-libraries-ck/mxfp6_gemm/`（含 profile_out + 两份 RCV trace）。

**工具**：
- rocprofv3 = 系统版 `/opt/rocm/bin/rocprofv3`（ROCm 7.0.2.1），无需 libatomic workaround。ATT 解码须 `--att-library-path /home/AMD/zhewan/rocm-tools/opt/rocm-7.0.2.1/lib`。
- gpu-profile skill（full 模式 2-agent）一键产 4 件套；RCV = Radeon Compute Viewer，吃 `ui_output_*`。

---

## Next Steps

**当前生产 = v17 + occ1 双缓冲。8192³ ~1628（~17.6% 峰值）；非 2 幂目标形状靠 V2（8192×5120 ~1740）。Step 22 定盘：已近此设计实用上限，调参/调度无空间；要上台阶须换范式。**

1. ~~小矩阵 tile 自适应 / split-K~~ ✅ 已完成（Step 12-13，并入 v17）。
2. ~~更细的 L2/CU 调度~~ / ~~救 cold 半程~~ / ~~occupancy~~：**Step 22 全部排除** —— L2 已 91% 命中、非带宽瓶颈、occ2 已证伪、调度交错证伪。这些不再是活口子。
3. **★ 换范式：A 走 async DMA + LDS 复用**（追 CK 的 ~2× 的唯一路）。当前每个 K 从 global 重读 A → 算/载比天然低一倍。CK MX flatmm 用 `global_load_lds`(async)→`ds_read` 让 A 在 LDS 复用、B 直读、IGLP 调度，做到 ~35% 峰值。⚠️ occ1 下 A-LDS 的一个 VGPR-roundtrip 变体（v18）曾 −18~32%，但那是带 VGPR 中转+双 barrier 的版本，与 async `global_load_lds`（零 VGPR 中转，见 [[mxfp6_a_matrix_shuffle]]）不同。**用户已要求评估此路 + 下载 CK 1.1.0/ROCm7.0 源码对照。**
4. **Epilogue 输出类型**（功能性，未做）：F16/BF16 输出（当前 F32 直写），见 [[mxfp6_output_epilogue]]。不影响峰值。
5. **V2 路径也用双缓冲**（次要，当前门控排除）：N576 曾测 ping-pong +5.7% 但样本少；可按 tile 细化门控再测。

**已证伪、不要重试**：
- 访存：A 合并/LDS(v6-v12,**v18 VGPR-roundtrip 版**)、软件预取藏延迟(v11)、减地址算术(v13)。LDS-VGPR-中转 在 occ1 big-tile 负优化（但 async `global_load_lds` 零中转版未试 = Next Step #3）。
- **让编译器把 copy-prefetch 的 load-wait 下沉到 MFMA 之后**：源码重排/去 volatile/builtin MFMA/sched_barrier/显式 waitcnt/same-iter —— 6 种全失败（Step 21）。唯一解 = 2×双缓冲直接消费。
- **对称 ping-pong**：occ1 喂不动两个 warm buffer，−1.8%（Step 21）。当前非对称 2×展开才是最优。
- **load 发射交错**（Step 22）：sched_barrier(0/1) −8~13%、sched_group_barrier 细交错 −88%。破坏 ping-pong in-flight overlap。官方 Cliff Notes Lesson 10 = sched_group_barrier 只 advisory。**全 clustered 即最优。**
- **occupancy（occ2）**（Step 10 + Step 22 复核）：occ2 N256=1230 « occ1 N512=1628。tile 砍半算/载比变差，得不偿失。

**关键定盘数据（Step 22，8192³）**：L2 命中 91.3%、HBM 645GB/s(10%峰值,非带宽瓶颈)、SQ_WAIT 1.9×SQ_BUSY(latency-bound)、暴露延迟=9% L2 miss 尾(每条 880cyc 捅破 512cyc MFMA 窗口)。vs CK MXFP6 ≈ 0.5×（架构范式差）。见 memory [[mxfp6_l2_sq_measurement]] [[mxfp6_load_issue_stall]] [[ref_confluence_gfx950_gemm]]。

**判定法（屡试不爽的纠错经验）**：凭计数器猜瓶颈屡屡出错（latency/throughput/LDS/barrier/VALU/issue-count 全被证伪过）。**藏延迟没用→非 latency；砍访存量没用→非 throughput；加指令就更慢→issue/compute bound。** 改动前先隔离实验或 profile，改动后必对比**同会话**基线（热漂移，跨会话不可比）。MFMA util 用墙钟法或 bench %-of-peak，别信 SQ 比值（聚合口径不一致）。
