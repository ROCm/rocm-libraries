# HANDOFF: MXFP6 高性能 GEMM Kernel (MI350/CDNA4)

## Goal

基于 HIP 和当前 ROCm 在 MI350 (CDNA4, gfx950) 上开发 MXFP6 (FP6 E2M3 + E8M0 block scale) 的高性能 GEMM kernel。以最简单的头文件+源文件形式实现，不依赖 CK 框架。主要关心的问题规模：M=2K~8K, K/N=4K/8K。

## Context

经过深度设计面试（/grill-me），9 个核心设计决策已全部确认。完整技术文档已生成。所有设计决策保存在 memory 文件中。用户使用中文交流。

**MI350 硬件规格：**
- 256 CUs, 4 SIMDs/CU, max 8 waves/SIMD
- LDS: 160 KB/CU, 64 banks
- VGPR: 256 Arch + 256 AccVGPR
- HBM: ~503 GB, 峰值带宽 ~6.5 TB/s
- MFMA 32x32x64 FP6: 32 cycles, src0/src1 可用 VGPR 或 AccVGPR, src2/vdst 可用 VGPR 或 AccVGPR (ACC/ACC_CD bit)
- gfx950 global_store 可直接从 AccVGPR 读取

## Current Progress

### Step 1 ✅ CPU 参考实现 + 预处理 (162 tests, 0 failures)
- `mxfp6_types.hpp`: FP6 E2M3 ↔ float, E8M0 ↔ float, FP6 密集打包
- `mxfp6_preprocess.hpp`: 量化/反量化, B 转置, scale→MFMA lane layout, B pre-shuffle
- `mxfp6_reference.hpp`: CPU golden reference GEMM
- `test_reference.cpp`: 全部 162 checks pass

### Step 2 ✅ ASM 工具函数
- `mxfp6_asm_utils.hpp`: 完整的 inline asm 基础设施
- 全部 ASM 在 MI350 + ROCm 7.0 上编译+运行通过

### Step 3 ✅ A 矩阵 LDS 数据通路验证
- MI350 实测 64/64 threads correct, 64/64 K positions correct
- rocprofv3 确认 **SQ_LDS_BANK_CONFLICT = 0**

### Step 4 ✅ 单次 MFMA 端到端验证 (1024/1024 exact match)
- **TransposeC**：交换 src0/src1 使每 lane 持有 1 M-row × 16 N-cols
- **B Pre-shuffle**：host 端重排 B 为 section0(dwordx4) + section1(dwordx2) → 100% 合并访存
- 随机数据 1024 元素精确匹配 CPU reference（误差 = 0）

### Step 5 ✅ K 循环 (8/8 tests pass, error = 0)
- A 矩阵不做任何预处理，kernel 直接从行主序 packed data 按 stride 加载
- 加载方式：`global_load_dwordx4` (HBM→VGPR) + `ds_write_b128` (VGPR→LDS)
- LDS padding：每行 52 字节 (48 有效 + 4 padding)，stride=13 DWORDs，gcd(13,64)=1，零 bank conflict
- rocprofv3 确认全部 K=64/128/256/512 的 SQ_LDS_BANK_CONFLICT = 0
- LDS anchor：`asm volatile("" : : "r"(lds) : "memory")` 防止编译器优化掉 `__shared__` 分配

### Step 5.5 ✅ MFMA inline asm + AccVGPR 基础设施
- **MFMA 从 builtin 改为纯 inline asm**，用 `"v"`/`"a"` 约束控制寄存器文件
- **v6i 类型**：MFMA FP6 操作数精确 6 VGPR，v8i 的 8-reg 范围被汇编器拒绝
- **AccTileV / AccTileA**：两种累加器类型，函数重载自动选择 `"+v"` 或 `"+a"` 约束
- AccTileA 路径：Arch VGPR 46→26（省出 20），global_store 直接从 AccVGPR 读，零开销
- **ds_read_fp6x32_complete bug 修复**：
  - `=&v` (early-clobber)：防止编译器将输出 VGPR 和输入共享导致 v_mov 自覆盖
  - 将 `s_waitcnt lgkmcnt(0)` 合并进 complete 的 asm 块，同一块内排序天然保证
  - 不加 `"memory"` clobber（v_mov 是纯寄存器操作，加了会强制编译器 reload 所有缓存的内存值）

### Step 6+8 ✅ Multi-tile 模板化 (128×128, 4 waves, 8/8 tests pass)
- **WG tile 128×128, 4 waves (2×2 layout), 每 wave 2×2 MFMA tiles**
- AccTileA (AccVGPR 累加器): 64 AccVGPR/wave, 47 Arch VGPR, occupancy 4
- 所有 wave 共享 A via LDS (6656 bytes with padding), B 从 VMEM (pre-shuffled)
- Outer-product 内循环：预加载 2 个 B tile 到 VGPR，逐个 A tile 从 LDS 读取，4 次 MFMA
- **store_acc_f32 bug 修复**：m_half 用 `(tid%64)/32` 而非 `tid/32`（多 wave 场景下算错）
- 测试覆盖 M/N=128..512, K=64..512, constant+random, 全部精确匹配

### Step 7 ✅ Double Buffer Pipeline (v1)
- LDS 双缓冲 A tile（2 × 6656 = 13312 bytes），交替读写消除一个 barrier
- Prefetch 下一轮 A 在 MFMA 期间执行
- 每 K 迭代 1 个 barrier（vs 之前 2 个）
- split ds_read 实验：early-clobber 增加 16 VGPR 压力，收益为负，已回退为 blocking ds_read
- `ds_read_fp6x32_complete` 去掉错误的 `"memory"` clobber（纯 v_mov 不碰内存），VGPR 63→60

**性能数据 (MI350, ROCm 7.0)：**

| 问题规模 | 无 pipeline | double buffer | 提升 |
|----------|-----------|-------------|------|
| 2048×4096² | 484.8 TFLOPS | 465.6 TFLOPS | -4.0% |
| 4096³ | 647.4 | 674.3 | +4.2% |
| 8192×4096² | 676.5 | 715.3 | +5.7% |
| 4096×8192×4096 | 663.9 | 692.5 | +4.3% |
| 4096²×8192 | 680.5 | 713.3 | +4.8% |
| **8192³** | **701.4** | **754.5** | **+7.6%** |

FP6 MFMA 理论峰值约 9216 TFLOPS (256 CUs × 4 SIMDs × 131072 FLOPs/MFMA / 32 cycles × 2.2 GHz)。当前利用率约 8%。

### Step 7.5 ✅ rocprofv3 full profile (8192³, MI350X)

完整 PMC + ATT 逐指令分析（交付物在 `mxfp6_gemm/prof_out/`：`pipeline_profile.md` / `pipeline_annotated.asm` / `pipeline_raw.asm` / `pipeline_rcv_trace.tar.gz`）。

**瓶颈类型：latency / stall-bound —— 不是 bandwidth-bound,也不是 compute-bound。**

- MFMA 利用率仅 **8.4%**,但 CU busy **99%**,VALU 发射利用率仅 **14%** → CU 被占满但执行单元空转,waves 在等 load。
- L2 命中 **94.8%**,HBM 读仅 ~964 MB(~10% 时间)→ **HBM 带宽不是瓶颈**。
- LDS bank conflict = 0。occupancy 4 waves/SIMD(arch 64 + acc 64 双双卡 256/64=4)。
- 每个 MFMA 摊 **11.1 条纯 VALU** + 1.38 LDS + 2.38 VMEM_RD。

**ATT 总 stall 的 93% 是内存等待**(MFMA 本身仅 1.6%,被饿着):

| Stall 来源 | 占比 | stall/hit |
|---|---|---|
| 等 B 的 VMEM (`vmcnt`) | 36.7% | 293-408 |
| 等 A 的 LDS (`lgkmcnt`) | 32.9% | **694/696** (critical) |
| ds_read/write | 11.9% | |
| barrier (每 iter __syncthreads) | 11.8% | 609 |
| MFMA | 1.6% | 9-29 |

**关键结构性问题**(见 annotated.asm):① A 读串行阻塞(读完才能 MFMA,零 overlap);② B 无预取,每 iter 现载现等;③ 源码 L111-124 的"预取下一 A"其实排在 MFMA **之前**,VMEM 等待+ds_write 压在关键路径上,完全没和 MFMA 重叠;④ 每 iter 一个 barrier。

> ⚠️ 修正早期判断:瓶颈**不是 "A/B 的 HBM load 延迟"**(HBM 很闲),而是 **load 延迟未被掩盖 + occupancy 过低(4 waves) + 每 iter 计算量太小(仅 4 MFMA / 128 矩阵周期,掩盖不住 ~700cyc 的 LDS 读和 ~400cyc 的 VMEM load)**。

### Step 8 ✅ 让编译器管理 waitcnt（v2，`test_pipeline_v2.cpp`）

原 kernel 手写的 `s_waitcnt` 全是绝对清零 `(0)`、无重叠。**实测确认:把 load 从 inline asm 里拿出来改成普通 typed 指针读,编译器的 SIInsertWaitcnts 会自动插入相对计数(如 `vmcnt(1) lgkmcnt(2)`)并与 MFMA 重叠**——手写 `(0)` 永远做不到。

- 边界(probe_waitcnt.cpp 三变体证实):load **藏在 inline asm 里编译器看不见**,一条 waitcnt 都不插 → race;改普通指针读才行。
- **MFMA 操作数必须保持单个 v6i 向量值**——用标量 `v8i{...,0,0}` 重建会让分配器把两次 b96 读散到不相邻寄存器、漏掉 gather → 操作数错乱(HANDOFF 问题 #2)。新增 `ds_read_fp6x32_plain`(返回 v6i)+ v6i 版 MFMA 重载。
- scale 的 `v_and 0xFF` 是冗余的(ubyte load 自动零扩展),删掉,且它正是逼出每-scale 一次 vmcnt drain 的元凶。
- 结果:8192³ 752→766,小矩阵 2048×4096² +25%,代码大幅简化。

### Step 9 ✅ A 加载方式调查 + 大 tile 突破（v3–v10）

**深挖"A 矩阵该怎么 load",连试 8 个变体,纠正了多个错误诊断:**

| 版本 | 方案 | 8192³ | 合法? |
|---|---|---|---|
| v2 | 行主序非合并 → LDS → ds_read | 766 | ✓ |
| v3 | + B 寄存器双缓冲 | 725 | ✓ 回归(B 非瓶颈) |
| v4 | host pre-shuffle A 成合并 + 直载无 LDS | **1252** | ✗ A 运行时激活不可预处理 |
| v5 | 行主序非合并直载,无 LDS | 769 | ✓ 最简单 |
| v6 | LDS + 2线程/行"合并"加载 | 703 | ✓ |
| v7 | LDS K=256 slab 合并加载 | 792 | ✓ |
| v8 | v7 + 双缓冲软件流水 | 701 | ✓ |
| v9 | v7 + 预计算地址 + ds_read 预取 | 726 | ✓ |
| **v10** | **v5 直载 raw A + 大寄存器 tile** | **1375** | ✓ **最优** |

**关键纠错链(每步都有 PMC/隔离实验支撑):**
1. ~~"LDS 是瓶颈"~~ 错:v5(去 LDS 非合并)=769 ≈ v2(LDS)766 → **LDS 有无是平局**。
2. ~~"barrier 是瓶颈"~~ 错:v2 去 barrier = 733 ≈ 766 → **barrier 不是瓶颈**。
3. ~~"VALU/LDS-wait 是瓶颈"~~ 错:v9 把两者都降了却更慢。
4. **真相①:A 行主序、每 lane 读不同行 → 全局加载天生非合并(cache line 只用 37.5%),这是 ~770 的墙。v4 快只因 pre-shuffle 让访存合并(但对运行时激活非法)。合法范围内 A 加载方式不是杠杆。**
5. **真相②(突破):主导杠杆是 Volkov 式大寄存器 tile + 低 occupancy**(每 wave 做更多 MFMA,把 A 不可避免的非合并加载摊到更多计算上)。

**v10 实现(`test_pipeline_v10.cpp`)**:v5 的直载 raw A(合法)+ 每 wave `M_PER_WAVE×N_PER_WAVE` 的 acc 网格 + `__launch_bounds__`(否则编译器为保 occ4 砍 AGPR 到 64、装不下多 acc → spill):
- **N_TILE=512(2×8=16 acc=256 AGPR,132 VGPR,0 spill,occ1)**:8192³=**1375**,4096³=1368,但 2048×4096²=778(occ1 时 grid 仅 128 WG<256 CU、半数闲)。
- N_TILE=256(2×4=8 acc=128 AGPR,84 VGPR,occ2):8192³=1115,小矩阵均衡(1032)。
- **tile 按规模选**:大矩阵 512,小矩阵 256。

**当前最优:8192³ 合法 1375 TFLOPS(~15% 峰值),从最初 766 接近翻倍,反超需非法预处理的 v4。** B 仍 pre-shuffle(权重,可离线,合法)。

### Step 10 ✅ v10 重新 profile + 瓶颈两次纠错(v11 预取 / v12 LDS 均证伪,不并入)

**用 gpu-profile full(2-agent 并行)重测 v10 最优点(8192³ N_TILE=512),交付物在 `mxfp6_gemm/prof_out_v10/`(profile.md / annotated.asm / raw.asm / rcv.tar.gz)。然后做两个对照实验,把瓶颈判断纠错两次:**

| 实验 | 做法 | 8192³ | 结论 |
|---|---|---|---|
| v10 profile | — | 1376 | 看着 latency-bound(78% stall=vmcnt40%+load38%) |
| v11 预取 | ping-pong 双缓冲跨 k-iter 软件预取 | 1393(+0.9%) | 延迟藏了(vmcnt8→43 in-flight)但没用 → **排除 latency** |
| v12 LDS | big tile+LDS 合并加载 A+ds_read 转置+双缓冲 | 1255(−9%) | L2 流量砍 47%(TCC 73M→39M)反而慢 → **排除 throughput** |

**🔑 最终瓶颈判断(取代 Step 7.5/9 的旧框架):v10 既非 latency 也非 memory-bound,而是 occ1 下的 compute/issue/dependency-bound。**
- MFMA util 仅 16-17%(GRBM 基准),单 wave/SIMD 无法掩盖任何依赖延迟;16 独立 acc 的 ILP 只够到 16%。
- **加任何指令都更慢**(v12 多 ds_write/ds_read/barrier 就降速)= issue/compute bound 的铁证。
- profile 里的 vmcnt stall 是 occ1 症状(无第二 wave 可切),不是访存量问题。
- WAVES 扫描佐证:WAVES_M=4/N=1(A 零冗余)=1171 最差,2×2 最优;且 B(N512)流量是 A(M128)的 4×,A 是少数派。

**判定法(写给后人):藏延迟没用→非 latency;砍访存量没用→非 throughput;加指令就更慢→issue/compute bound。**

v11(`test_pipeline_v11.cpp`)/v12(`test_pipeline_v12.cpp`)保留为"已证伪"记录,**均不并入主线**。v10 的 1376 ≈ occ1 此 tile 形状的天花板。

### Step 11 ✅ 减地址算术 / ki-unroll(v13,证伪,不并入)

**攻 issue-bound 正面**(Step 10 列的剩余杠杆 #1):v10 内循环 ~38 条 `v_lshl_add_u64`/ki(B 16 段指针 + 8 基址重建 + 10 scale + A),编译器不跨 ki 展开,每个基址每迭代从 SGPR base 重建。**v13(`test_pipeline_v13.cpp`)**:ki 方向展开 KU=2,用 `+j*stride` 立即数 offset(global_load 13-bit ±4096:A+48 / B+1536(sec1 max 2560) / scale+64)让同一基址服务相邻 ki。

| 变体 | 8192³ | arch VGPR | 结论 |
|---|---|---|---|
| v10 基线(同会话) | 1379 | 132 | — |
| v13 批量(KU 个 load 全提前发) | 1341(−2.8%) | 192 | 地址砍半(38→19.5/ki)但爆寄存器+拉长启动延迟 |
| v13 交错(load-then-mfma) | 1262(−8.5%) | 168 | 杀掉跨-load vmcnt 瀑布重叠 |

正确性 4/4 error=0。**地址算术机制上确实砍半,性能却回归 → 再次证伪:非 issue-count-bound,而是 occ1 下 load→MFMA 重叠/依赖结构 bound,v10 该结构已近最优。** 这是第三个被证伪的瓶颈假设(latency/throughput/issue-count 全错)。

> 注:gfx950 asm metadata `.vgpr_count` = arch+acc **合并**值(v10=388=132+256, v13=448=192+256);算 arch VGPR 须减 256。

**剩余唯一确定有收益的方向 = 小矩阵 tile 自适应/split-K**(独立于瓶颈,见 Next Steps #3)。"减每-MFMA 指令"已随 v13 排除。

### Step 12 ✅ Tile 自适应 dispatch(v14,小矩阵 +32~68%,大矩阵零回归)

**实现 Next Steps #3a**(唯一确定有收益、独立于峰值瓶颈的方向)。v10 kernel 不变,纯 host 端按 (M,N,K) 选 N_TILE(`test_pipeline_v14.cpp`)。big tile occ1 在小 M/N 时 grid 填不满 256 CU(2048×4096 只 128 WG,半数 CU 闲),换小 tile(occ2/4)做更多 WG 填满。

**heuristic**:选 grid_WG≥256 CU 的**最大** tile(满载时大 tile 每-WG 效率最高);都填不满则选 WG 最多(最小 tile)。三档:N512(occ1)/N256(occ2)/N128(occ4),`__launch_bounds__(256, MIN_OCC)` 模板化(N512 必须 MIN_OCC=1 否则编译器砍 AGPR 装不下 16 acc)。

**实测(min-of-4-rounds 锁高时钟,heuristic vs 全 tile oracle,8/8 命中最优 ✓opt):**

| 尺寸 | N512 | N256 | N128 | 选中 | vs 旧单配置N512 |
|---|---|---|---|---|---|
| 2048×2048×4096 | 407 | 563 | **683** | N128 | **+68%** |
| 2048×4096×4096 | 793 | **1047** | 721 | N256 | **+32%** |
| 2048×8192×4096 | **1241** | 1081 | 763 | N512 | 持平 |
| 4096³ | **1269** | — | — | N512 | 持平 |
| 8192×4096² | **1454** | — | — | N512 | 持平 |
| 8192³ | **1460** | — | — | N512 | 持平 |

正确性 8/8(含强制三档 tile 各跑一遍)。**小矩阵大涨、大矩阵零回归,生产 dispatcher 应内置此 heuristic。** 注:绝对值比 Step 7-11 高因 min-of-rounds 方法学不同(锁高时钟),相对 tile 对比同会话干净。

### Step 13 ✅ Split-K kernel + 统一 dispatcher(v15,仅极瘦矩阵受益 3.6×,正常形状零触发)

**实现 Next Steps #3b**。独立的 split-K kernel(`mxfp6_gemm_splitk`,与 v10 主体同),grid.z=S 沿 K 切分,每 split 算 K/S 的 partial。

**关键设计教训:partial 累加必须用独立 buffer + reduction kernel,不能用 atomicAdd。**
- 首版 atomicAdd 到 D:**崩(2048×4096 N512 S2 = 153 TFLOPS,−86%)**。float4 无法 atomic → 256 scalar atomicAdd/lane;K-slice 变短后 epilogue 占比暴涨 + S 份同时 hammer 同批 cache line 重度争用。
- 改版:每 split 用快速 float4 store 写自己的 `partial[s]`(M×N 平面),再跑 memory-bound 的 `splitk_reduce` 求和 → 无争用。

**实测结论:split-K 打不过 tile-shrink(v14),除非连最小 tile 都填不满 CU。**

| 尺寸 | v14 tile-shrink | split-K | 判定 |
|---|---|---|---|
| 2048×4096×4096 | N256 1062 | N512 S2 897 | **split −17%(reduction+S×写出 > 大tile收益)** |
| 2048×2048×4096 | N128 683 | N512 S4 646 | split −5% |
| 512×1024×8192 | N128 100 | **N128 S8 358** | **split +259%(3.6×)** |
| 512×512×8192 | N128 51 | **N128 S8 197** | **split +285%(3.85×)** |

→ **正常形状 tile-shrink 用换小 tile 免费填满 CU,胜过 split-K(后者有 reduction + S× partial 写出开销)。split-K 只在 M×N 太小、连 N128(最小 tile)都填不满 CU 且 K 够大可切时独占价值(512×1024 仅 32 WG → S8 → 256 WG 满 → 3.6×)。**

**最终 dispatcher(`choose_plan`,生产级):** 先 tile-shrink 填(`choose_tile_nosplit`,v14);grid≥256 WG 就 S=1 不切;只有连最小 tile 都 <256 WG 才 split-K 把它填满。**正常形状零触发→零回归;极瘦矩阵 3-4×。** 正确性 10/10(含每 tile 强制 split2 验 reduction,error=0)。这些极瘦形状超出目标 M=2K~8K,但对 small-batch/decode 真实有用。

### Step 14 ✅ 混合累加器 N640 / V2(v16,非 2 幂 N +5~12%)

**突破"2 幂 N 卡 1379"的第一刀**(2026-05-29)。occ1 合并 VGPR 池(arch+acc=512)在 N512 时只用 388/512,闲 124 个 arch VGPR。让**溢出的 acc tile 走 Arch VGPR**(MFMA 输出可写 Arch VGPR,ACC_CD=0),单 wave 就能持有 >256 AGPR 的累加器:N512 的 16 acc + 4 溢出进 Arch VGPR = **N640 的 20 acc**。

- ⚠️ 限制:`N_TILE = WAVES_N × NPW × 32` 须整除 N。N640=2^7×5,只整除含因子 5 的 N;2 幂 N(4096/8192)整除不了,只能退回 NPW=8(16 acc)→ 目标 2 幂形状仍卡 ~1490。
- **V2(N640,20 acc)只吃非 2 幂 N**:真·同 N 对比 8192²×**5120** +11.6%、×**7680** +7.6%(5120 是真实 LLM hidden dim)。grid 均衡时整体 +5~12%。
- sweet spot = V2(20 acc);V3(22 acc)掉、V4 spill。N576(2^6×9,18 acc)覆盖另一种非 2 幂可整除性。

### Step 15 ✅ 生产 dispatcher 合一(v17 = tile-shrink + V2 + depth-1 预取 + 4x4 wave)

**把 v14/v15 的 tile-shrink、V2 混合累加器、depth-1 软件预取、方 wave-tile 全合进一个 kernel + 一个 cost-model dispatcher。** `mxfp6_gemm_pipeline<M_TILE,NPW_A,NPW_V,N_WAVES,MIN_OCC,WAVES_M,WAVES_N,SWZ>`:`NPW_A` 列走 AGPR、`NPW_V` 列溢出进 Arch VGPR。

- **depth-1 软件预取(仅 occ1)**:ki+1 的 load 在 ki 的 MFMA 之前发,用 MLP 重叠 ~876cyc 的 load 延迟(编译期 double buffer,动态 reg-array index 会 spill)。occ≥2(N128/N256)第二 wave 已藏延迟,预取只占寄存器降 occ → 跳过。**注:这条 +1~15% 已并入,取代了 v11"预取无用"的旧结论——区别在 v11 用动态索引 spill 了。**
- **方 wave-tile(4x4)**:N512 可跑 2x8 或 4x4(同 16 acc/occ1/WG-tile),4x4 每 16 MFMA 只 load 8 vs 10 个 tile(perimeter 小)。见 [[mxfp6_wave_shape]];2 幂方阵 occ1 下 4x4 反输给 2x8+swz(见 Step 17),dispatcher 默认 2x8。
- **统一 cost model**:`cost(tile)=ceil(WG/256)/(WG×eff)`,一个公式覆盖小矩阵 shrink(N128/256 填 CU)+ 大混合-acc tile(N512/576/640 摊 A)+ 非 2 幂可整除性。`choose_tile` 扫 5 档取最低 cost。
- 实测 dispatch **12/12 OPT 0 MISS**(heuristic 选中 == 全 tile oracle 最优),correctness 全 PASS。**生产 kernel = v17。**

### Step 16 ✅ A 合并的最终证伪(v18,LDS-stage 在 occ1 big-tile 负优化)

**受控实验**(`test_pipeline_v18.cpp`,现已删,结论入 [[mxfp6_a_no_lds]]):两 kernel 除 A 路径外全同(同 scale 合并 / 同 B / N512 occ1)。`kern_direct`(非合并直载,= v17)vs `kern_lds`(KSLAB 沿 K 合并 global→LDS→ds_read 转置)。correctness 6/6 err=0,但 **lds 慢 18~32%**(8192³ 1459→990,−32.2%)。根因:occ1 下 global→LDS 多一跳 + 每 slab 2 个 __syncthreads(无第二 wave 顶 barrier)全暴露 > 合并省下的。**∴ A 合并这条路在 occ1/big-tile 彻底堵死,876cyc 是 occ1 延迟暴露非"非合并"本身。**

### Step 17 ✅ L2-aware WG swizzle 打破 2 幂方阵天花板(8192³ 1491→1557)

**不碰 A 加载、不动 occ/tile,只重排 WG→(m,n) 映射就破了 2 幂方阵的"天花板"。** 原 `(blockIdx.x,y)=(m,n)`;swizzle(`SWZ` 模板参 + `if constexpr` 的 host index 重排)让 WG 沿 M 走完一个 `SWZ` 宽的 n-block band 再跳下一 band → 同时在飞的相邻 WG 共享同一段 B,B 留 L2 热着。

- **8192³**:N512=1491 → N512+swz16=**1557(+4.4%)**。这是 prefetch 实验里相对 v17 唯一的新增益(v17 早有 depth-1 预取)。
- **门控(实测定标,`choose_swz`)**:仅当 `nb=N/512≥16`(N≥8192,n-band 填满)才启用 SWZ=16,跨 M(2048~8192)都 +0.8~4.4%;`nb<16` 中性偏负(−0.6~2.1%),`nb=15`(7680)持平 → **只按 N 门控,与 M 无关**(小 M 时重排退化为恒等映射,不伤)。SWZ=8≈16,取 16。
- 已并入 v17,correctness 8/8(含 swz 重映射 err=0),dispatch 12/12 OPT。详见 [[mxfp6_swizzle_breakthrough]]。

### Step 18 ✅ 仓库整理:只留生产路径

删光 v2~v15/v18 全部迭代、prefetch/square/f16/occ 实验、早期脚手架测试(lds_shuffle/k_loop/accvgpr/multitile/bench)、prof/工具/所有产物。**生产路径 = `test_pipeline_v17.cpp` + 4 个 `mxfp6_*.hpp` + `test_reference.cpp` + `Makefile` + `counters.txt` + `.gitignore`。** 已 rsync 同步 NFS(保留 NFS 上 prof_out* 备份)。

## 🔑 核心技术要点

### MFMA 32x32x64 f8f6f4 数据布局 (ISA Section 7.1.5)

A 和 B 都是 **per-lane 分布**，无 broadcast：
```
src0 = A[M][K]: lane l → A[m=l%32][k_half=l/32], 32 FP6 values = 24 bytes (6 DWORDs)
src1 = B[K][N]: lane l → B[k_half=l/32][n=l%32], 32 FP6 values = 24 bytes (6 DWORDs)
```

### TransposeC (交换 src0/src1)

调用 `mfma(B, A, ...)` 使输出布局变为每 lane 持有 1 M-row × 16 N-cols：
```
Lane l: m = l%32, m_half = l/32
acc[p] → N = (p%4) + (p/4)*8 + m_half*4

4 组连续 acc → 4 个连续 N → global_store_dwordx4
```

注意 builtin 参数映射：`mfma(src0=B, src1=A, cbsz=B_fmt, blgp=A_fmt, scale_a=B_scale, scale_b=A_scale)`

### Scale 布局 (ISA Page 65)
```
Lane  0-15:  dim=0..15,   K=0..31
Lane 16-31:  dim=16..31,  K=0..31
Lane 32-47:  dim=0..15,   K=32..63
Lane 48-63:  dim=16..31,  K=32..63
```

### B Pre-shuffle 布局

原始 B^T[N][K] 每行 48 bytes，stride=48B 导致 33% 合并率。Pre-shuffle 拆成两段：
```
Section 0 [0..1023]:    tid × 16B (DWORDs 0-3) → global_load_dwordx4, 100% 合并
Section 1 [1024..1535]: tid × 8B  (DWORDs 4-5) → global_load_dwordx2, 100% 合并
```

### LDS Padding (零 bank conflict)

每行 52 字节 (48 data + 4 padding)，stride = 13 DWORDs，gcd(13, 64) = 1。

### AccVGPR 使用 (gfx950)

- MFMA inline asm 用 `"a"` 约束将累加器放入 AccVGPR
- 编译器自动为 `v16f{} = 0` 生成 `v_accvgpr_write_b32` 初始化
- `global_store_dwordx4` 直接从 `a[0:3]` (AccVGPR) 写 HBM，无需 v_accvgpr_read
- A/B 矩阵数据通过 `"v"` 约束保持在 Arch VGPR（load 天然目标）

### v6i 类型

MFMA FP6 src0/src1 需要精确 6 个连续 VGPR。v8i 的 8-reg 范围被汇编器拒绝。`to_v6i(v8i)` 截取前 6 个元素。

### ds_read_fp6x32 split API 约束

- `ds_read_fp6x32_issue`：`"memory"` clobber（LDS 读操作，正确）
- `ds_read_fp6x32_complete`：`"=&v"` (early-clobber)，**无 `"memory"`**
  - `&`：防止输出和输入共享 VGPR（v_mov 自覆盖 bug）
  - `s_waitcnt lgkmcnt(0)` 合并在同一 asm 块内，排序天然保证
  - 不加 `"memory"`：v_mov 不碰内存，加了会强制 reload 所有缓存值（+3 VGPR）
- 注意：early-clobber 增加 ~6 VGPR 压力（输出不能和输入共享）

### inline asm `"memory"` clobber 原则

只在 asm 确实读写内存时加 `"memory"`：
- ✅ `ds_read_b96`、`ds_write_b128`、`global_load_lds_dwordx4` — 访问 LDS/HBM 内存
- ✅ `s_waitcnt vmcnt/lgkmcnt` — 内存 fence，防止编译器把内存操作移过 wait
- ❌ `v_mov_b32`、MFMA — 纯寄存器操作，加 `"memory"` 会阻止编译器缓存内存值到 VGPR

## 已解决的问题

1. **Intrinsic 参数顺序** ✅
2. **ds_read_b96 编译器 bug** ✅ — v3i 寄存器分配不可靠，用 issue/complete + v_mov_b32 解决
3. **Scale VGPR 脏字节** ✅ — 用 `v_and_b32` 清理高字节
4. **store_acc_f32 M/N 映射** ✅ — TransposeC 后每 lane 1 M-row × 16 N-cols
5. **B 合并访存** ✅ — pre-shuffle 将 stride=48B (33%) 优化为 stride=16B/8B (100%)
6. **LDS 分配被优化掉** ✅ — `asm volatile("" : : "r"(lds) : "memory")` 保留 `__shared__`
7. **ds_write_b128 v4i 约束** ✅ — float4 不能直接用 `"v"` 约束，需要 v4i 向量类型
8. **MFMA builtin 不支持 AccVGPR** ✅ — 改用 inline asm + `"a"` 约束
9. **v8i 范围被 MFMA 汇编器拒绝** ✅ — 添加 v6i 类型，FP6 精确 6 reg
10. **store_acc_f32 多 wave bug** ✅ — m_half 用 `lane/32` 而非 `tid/32`
11. **ds_read_fp6x32_complete 寄存器覆盖** ✅ — 添加 early-clobber `=&v`
12. **ds_read_fp6x32_complete 重排 bug** ✅ — 将 `s_waitcnt` 合并进同一 asm 块（不用 `"memory"` clobber）
13. **ds_read_fp6x32_complete 错误 "memory" clobber** ✅ — v_mov 不碰内存，去掉后 VGPR -3, 性能 +1%

## 代码文件

仓库已整理为只留生产路径(Step 18)。历史迭代 v2~v15/v18 / 脚手架测试 / 实验 / 产物全部删除,演进过程见上面 Step 1–17 与关联 memory(代码可从 git 历史取回)。

```
mxfp6_gemm/
├── mxfp6_types.hpp          # FP6/E8M0 编解码 + dense packing
├── mxfp6_preprocess.hpp     # 量化/反量化, B 转置, scale 重排+合并, B pre-shuffle
├── mxfp6_reference.hpp      # CPU golden reference GEMM
├── mxfp6_asm_utils.hpp      # GPU ASM: MFMA (inline asm, AccVGPR/Arch VGPR), LDS, store
├── test_reference.cpp       # CPU 单元测试 (162 pass) — ground truth
├── test_pipeline_v17.cpp    # ★生产 kernel + dispatcher: tile-shrink + V2 混合acc(N640)
│                            #   + depth-1 预取 + 4x4 wave + L2 swizzle(choose_tile/swz)
├── Makefile                 # make test_pipeline_v17 / test_reference
├── counters.txt             # rocprofv3 计数器配置
└── .gitignore               # build/profiler 产物 + 编辑器配置
```

**工具：**
- rocprofv3: 直接用系统版 `/opt/rocm/bin/rocprofv3` (ROCm 7.0.2.1)，**不再需要** libatomic 的 `LD_LIBRARY_PATH` workaround
  - 仅 ATT 解码追踪时需 `--att-library-path /home/AMD/zhewan/rocm-tools/opt/rocm-7.0.2.1/lib`（trace decoder 库只在该用户态目录）
- ISA PDF: `python3 -c "import fitz; doc = fitz.open('path'); print(doc[page].get_text())"`
- ISA dump: `hipcc -save-temps` 或 `hipcc -S`

## Next Steps

**当前生产 = v17(Step 15)。8192³ 1557 TFLOPS(~17% 峰值,L2 swizzle 后);非 2 幂目标形状靠 V2(N640/576)更高(8192×5120 ~1755)。生产 kernel 采用 v17 的 `choose_tile`/`choose_swz` dispatcher。**

⚠️ **"改 8192³ 峰值是死路"的旧判断已被 Step 17 推翻**:swizzle 不碰 compute/访存、只改 WG 调度顺序就 +4.4%。说明 occ1 之外还有 L2/调度层的杠杆没挖完。剩余方向:

1. ~~**小矩阵 split-K / tile 自适应**~~ ✅ **已完成(Step 12+13,并入 v17)**。tile-shrink 覆盖目标范围;split-K 只救极瘦矩阵。
2. ~~**大矩阵峰值(8192³)是死路**~~ ✅ **部分突破(Step 14 V2 / Step 17 swizzle)**。V2 把非 2 幂 N +5~12%;swizzle 把 2 幂方阵 +4.4%。**下一步同类:更细的 L2/CU 调度(SWZ 自适应 band 宽、grid stride、CU-aware launch)可能还有空间——这是新发现的活口子。**
3. **提 occupancy 但不缩 tile(很难)**:occ1 由 AGPR 256 顶死;部分 acc 走 Arch VGPR(V2 已用此招做 N640)。能否再换 occ2 是 ~17% util 的根因,但 Step 10 已证"加指令就更慢"。
4. **Epilogue/输出类型(功能性,未做)**: F16/BF16 输出(当前 F32 直写),见 [[mxfp6_output_epilogue]]。test_f16.cpp 原型已删,起点在 git 历史。不影响峰值。

**已证伪、不要重试**:
- 访存:A 合并/预取/减冗余(v6/v7/v8/v9/v11/v12)。LDS 路线(v12)L2 砍半反而慢。软件预取(v11)藏延迟无用。
- **减指令/地址算术:v13 ki-unroll 把地址砍半反而 −2.8%(批量爆寄存器,交错杀重叠)。issue-count 不是杠杆。**

**重要提醒**: "凭计数器猜瓶颈"屡屡出错(latency/throughput/LDS/barrier/VALU 都被证伪过)。**判定法:藏延迟没用→非latency;砍访存量没用→非throughput;加指令就更慢→issue/compute bound。改动前先隔离实验或 profile,改动后必对比同会话基线**(热漂移,跨会话不可比)。

**复现 profile**: cp 目标 cpp → 改 main 为单 dispatch(8192³)→ `hipcc -O2 --offload-arch=gfx950 -I.` → `/opt/rocm/bin/rocprofv3 --pmc <counters> -d out --output-format csv -- ./prof`。历史交付物在 `mxfp6_gemm/prof_out_v10/`(NFS 备份保留;/tmp 已清理)。gpu-profile skill(full 模式 2-agent)可一键产出 4 件套。

**关联 memory**: [[mxfp6_swizzle_breakthrough]](swizzle 破 2 幂天花板 + v17 整理,最新)、[[mxfp6_big_tile]](V2 大 tile/N640)、[[mxfp6_a_no_lds]](A 加载诊断 + v18 LDS 证伪)、[[mxfp6_pipeline_bottleneck]](v10 profile 数据)、[[mxfp6_compiler_waitcnt]]、[[mxfp6_wave_shape]](4x4)、[[mxfp6_tile_adaptive]]。
