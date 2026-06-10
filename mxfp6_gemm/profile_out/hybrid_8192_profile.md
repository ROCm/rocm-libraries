# lds_gemm_hybrid 性能分析报告 (full)

**Kernel**: `lds_gemm_hybrid<256,256,192, WM=2,WN=2, occ1, SWZ=0, DB, __half, PFD=6, SHUF=true>`
**范式**: A 留 LDS 深-K 双缓冲；**B 改为从 HBM 直读到寄存器**(coalesced `preshuffle_B` + 6 深寄存器环),跳过 B 的整条 LDS 往返。
**规模**: M=N=K=8192,K padded → Kp=8256(k_iters=129,k_tiles=43),grid 32×32=1024 WG,block 256(4 wave),FP16 输出。
**采集**: gpu-profile full(2-agent),ROCm 7.0.2.1,rocprofv3。日期 2026-06-09。

---

## 1. 硬件环境
| 项 | 值 |
|---|---|
| GPU | AMD Instinct MI350X,**gfx950 (CDNA4)** |
| CU / SIMD | 256 CU,4 SIMD/CU,64 lane/wave |
| 拓扑 | 32 Shader Engine,**8 XCD**(GRBM_GUI_ACTIVE 跨 XCD 聚合,÷8 得真实活跃周期) |
| 时钟 | 峰值 2200 MHz |
| LDS | 160 KB/CU,32 bank |
| Cache | L1(TCP)32 KB/CU,L2(TCC)4 MB/XCD |

## 2. Benchmark
| 配置 | 时间 | TFLOPs | vs 纯-LDS 基线 |
|---|---|---|---|
| **hybrid shuf-B PFD=6**(本核) | 0.512–0.532 ms | **2148(best)/2067(profile driver)** | **+16%** |
| 基线 `lds_gemm_db`(A+B 均 LDS) | 0.593 ms | 1854 | — |

跨形状(best-of-4×20):8192³ +15.8%、4096×8192 +17.8%、2048×8192 +23.7%、8192×4096 +16.7%、4096² +21.7%。小 M 收益更大。

## 3. 资源配置 (kernel trace + code-object 元数据)
| 资源 | hybrid | 基线 |
|---|---|---|
| VGPR(.vgpr_count) | **508**(arch 252 + acc 256) | 500 |
| SGPR | 30 | 36 |
| AGPR | 256 | 256 |
| **LDS** | **73,728 B**(仅 A,×2 双缓冲) | 147,456 B(A+B) |
| spill | **0** | 0 |
| Occupancy | **occ1**(1 wave/SIMD = 12.5%) | occ1 |

> **Occupancy 限制器 = VGPR 合并池**:508/512,acc 计入 gfx950 合并池 → 一个 SIMD 只容 1 wave。LDS(73KB)本可容 2 WG/CU,不是限制器。**hybrid 把 LDS 砍半但 occupancy 不变**(仍被 VGPR 卡死)。

## 4. PMC 指标分析

### 4.1 指令统计 (整核,SE 聚合)
| Counter | 值 |
|---|---|
| SQ_WAVES | 4,096 |
| SQ_INSTS_VALU | 31,932,416 |
| SQ_INSTS_MFMA | 8,454,144(全部 f6f4) |
| pure VALU(VALU−MFMA) | 23,478,272(73.5%) |
| SQ_INSTS_VMEM | **6,426,624** |
| SQ_INSTS_LDS | **3,170,304** |
| SQ_INSTS_SMEM | 24,576 |

> **hybrid 签名**:VMEM(6.43M)> LDS(3.17M)。B 不再进 LDS、改走 VMEM 直读;LDS 流量相对基线大幅下降。

### 4.2 利用率
| 指标 | 值 | 说明 |
|---|---|---|
| **MFMA util** | **26.1%** | SQ_VALU_MFMA_BUSY_CYCLES /(GRBM÷8 × 1024 SIMD) |
| VALU util | SQ_ACTIVE_INST_VALU/SQ_BUSY_CU_CYCLES | pure-VALU 占指令 73.5%,但 busy 周期 MFMA 主导 |
| **L2(TCC)命中** | **86.5%** | miss 6.03M |
| L1(TCP)命中 | 45.7% | |
| SQ_WAIT_INST_LDS | 61,180(busy 的 0.02%) | LDS 等待可忽略 |

### 4.3 LDS Bank Conflict
`SQ_LDS_BANK_CONFLICT = 8,454,144` **恰等于 SQ_INSTS_MFMA** → 已知的 MFMA 驱动伪计数,非真冲突;`SQ_LDS_ADDR_CONFLICT = 0`,`SQ_WAIT_INST_LDS` ≈ 0。**LDS 实质零冲突 / 完全隐藏**(ATT 中 ds_read stall = 0.0% 佐证)。

## 5. 瓶颈排名 (PMC)
1. **HBM 加载延迟暴露(latency-bound)** —— occ1,无并发 wave 隐藏延迟。MFMA util 仅 26% 说明矩阵管线大量时间在等操作数。
2. 计算非瓶颈(MFMA 26%、L2 86.5%、带宽充裕)。
3. 与基线本质相同(都是 occ1 latency-bound),但**慢点从 LDS-read 转移到 HBM-load**(见 §6)。

## 6. ATT 逐条指令分析 (full)
总 stall = 23,514,048 = 延迟的 **79%**(idle 1.76M,exec 4.50M)。

### 6.1 Stall 分类(占总 stall %)
| 类别 | stall | % | 含义 |
|---|---|---|---|
| s_waitcnt | 5.79M | **24.6%** | 主体是 **vmcnt**(等 B 直读),少量 lgkmcnt(等 A ds_read) |
| v_mfma | 4.95M | **21.1%** | 操作数饥饿(被动等 A/B 到位) |
| **global_load(B HBM 直读)** | 4.94M | **21.0%** | hybrid 的新主成本:B 直流的 HBM 延迟 |
| **buffer_load(A HBM→LDS)** | 3.27M | **13.9%** | A 协作加载延迟 |
| global_store(C 输出) | 2.45M | 10.4% | FP16 epilogue 写回 |
| s_barrier | 1.44M | 6.1% | RDB A 双缓冲屏障 |
| v_* VALU / s_* scalar | 0.58M | 2.5% | 地址/索引 |
| **ds_read(A LDS→VGPR)** | **0.0078M** | **0.0%** | ⭐ **LDS-read 墙消失** |

### 6.2 关键发现:墙搬家了
- **基线**:头号 stall 是 `lgkmcnt`(LDS ds_read,~40%),sub 头部 ~258cyc 等 b(ni0)。
- **hybrid**:`ds_read` 降到 **0.0%**。B 移出 LDS → 那堵墙整体消失。新的支配项是**真实的 HBM 加载**(A buffer_load 13.9% + B global_load 21.0% = ~35%)+ 门控它们的 waitcnt(24.6%)。
- 这正是预测的等价交换:**用"第一跳(HBM→片上)延迟"换掉了"第二跳(LDS→VGPR)延迟"**;净赢 +16% 来自(a)B 路径从 3 步(buffer_load+LDS写+ds_read)缩到 1 步(global_load),(b)LDS 流量减半与端口争用消失,(c)那段藏不住的 ds_read 被去掉,而 B 的 global_load 被 6 深环 + MFMA 窗口藏住。

### 6.3 Top stall 热点(stall/hit = 单 wave 平均周期)
| addr | 指令 | stall/hit | sev | 语义 |
|---|---|---|---|---|
| 0x3998 | buffer_load_dwordx4 …lds | 318.9 | !!! | A 协作 HBM→LDS(prefetch buf0 reload) |
| 0x3174 | buffer_load_dwordx4 …lds | 311.9 | !!! | A 协作 HBM→LDS(prefetch buf1) |
| 0x2b50 | **s_waitcnt vmcnt(12) lgkmcnt(3)** | 302.6 | !!! | sub 头部:等 B 环(12 在飞)+ 首个 A ds_read |
| 0x3890 | s_barrier | 263.6 | !!! | RDB buf0 屏障 |
| 0x305c | s_barrier | 261.5 | !!! | RDB buf1 屏障 |
| 0x31ec | global_load_dwordx4 | 200.2 | !!! | B 操作数 lo16B 直读 |
| 0x31c0 | global_load_dwordx4 | 198.4 | !! | B 操作数 lo16B 直读 |

> sub 头部那条 `vmcnt(12) lgkmcnt(3)` 仍是单点最痛(302/次),但性质变了:它现在主要等 **B 的 HBM 直读**(vmcnt),而非基线的 LDS ds_read。`ds_read2st64_b64`/`ds_read_b128`(A 读)在 ATT 里几乎不 stall。

## 7. 优化建议(按预期收益)
1. **A 也是 HBM 加载墙(buffer_load 13.9%,且占据 top-1/2 单条)** —— A 协作加载的延迟在 occ1 下全裸。可试:加深 A 的预取领先(prefetch 提前 ≥2 个 compute 窗口)或更激进的 A double-buffer 调度,把 A buffer_load 从关键路径挪开。预期:再吃几个百分点。
2. **sub 头部 `vmcnt(12) lgkmcnt(3)`(302/次)** —— 把 B 环再加深(PFD 6→7/8)曾在 bench 回落,但配合"A 读再提前"可能松动;或让首个 quartet 的 B 操作数享受更早的 ring 预取(prologue 跨 sub)。
3. **occupancy 仍是天花板(occ1,MFMA 仅 26%)** —— 真正的延迟隐藏需要更多在飞 wave,但 508/512 VGPR 卡死。除非缩 acc tile(伤算术强度,occ2 已证伪),否则只能继续靠 ILP(深-K + 环)。当前 hybrid 已是 ILP 路线的新高点。
4. **global_store(C)10.4%** —— epilogue 写回也暴露;可与最后一个 tile 的 compute 重叠(目前是全 K 后一次性写)。
5. **接入 v18 dispatcher** 当新 tile 候选(投产路径),并扫 swz × hybrid 交互。

---
### 交付物
1. `hybrid_8192_profile.md`(本报告)
2. `hybrid_8192_annotated.asm`(逐条:ATT stall + 源码级语义,含 top-15 + 分类汇总)
3. `hybrid_8192_rcv_trace.tar.gz`(RCV 可视化包)
4. `hybrid_8192_raw.asm`(原始反汇编)
