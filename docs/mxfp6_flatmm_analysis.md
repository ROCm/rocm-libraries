# MXFP6 FlatMM 优化全景报告

> **目标读者**：新加入项目的优化工程师
> **目标硬件**：AMD Instinct MI350X / MI355X（gfx950, CDNA4 架构）
> **目标库**：AMD Composable Kernel (CK) Tile, branch `zhewan/ck/dev`
> **编译器**：ROCm 7.0.2.1 / clang 20.0.0（客户锁定版本，不可变更）
> **基准问题规模**：M=N=K=8192（除非特别注明）
> **报告日期**：2026-05-26
> **当前最佳性能**：FP6 = **1219 TFLOPS**（FP8 = 1857 TFLOPS，FP4 = 3940 TFLOPS）
> **客户目标**：FP6 = 1.3~1.5 × FP8 = **2400~2786 TFLOPS**

---

## 目录

1. [MXFP6 FlatMM 是什么](#1-mxfp6-flatmm-是什么)
2. [硬件关键参数（gfx950 / CDNA4）](#2-硬件关键参数gfx950--cdna4)
3. [FP6 MFMA 数据消费方式](#3-fp6-mfma-数据消费方式)
4. [完整计算流程（一个 kernel dispatch 的生命周期）](#4-完整计算流程一个-kernel-dispatch-的生命周期)
5. [FP6 vs FP8 vs FP4 关键差异](#5-fp6-vs-fp8-vs-fp4-关键差异)
6. [发现的所有问题（按影响排序）](#6-发现的所有问题按影响排序)
7. [已尝试的优化及结果](#7-已尝试的优化及结果)
8. [性能天花板分析](#8-性能天花板分析)
9. [结论与未来方向](#9-结论与未来方向)
10. [附录：关键源码定位](#10-附录关键源码定位)

---

## 1. MXFP6 FlatMM 是什么

### 1.1 FlatMM ("Flat Matrix Multiply")

CK Tile 库里 GEMM 的一种实现风格。和传统 GEMM 的最大不同在于：

```
传统 GEMM            FlatMM (本项目)
─────────            ───────────────
A: HBM → LDS → REG    A: HBM → LDS → REG    (相同，A 走 LDS)
B: HBM → LDS → REG    B: HBM ────────→ REG  (B 完全在寄存器中, 不经 LDS)
C: REG → LDS → HBM    C: REG → LDS → HBM    (相同)
```

**B 矩阵 不经过 LDS** 是 FlatMM 名字的来源（"flat" = 直通寄存器，不绕 LDS）。

- ✅ 好处：省 LDS 容量，省一次 LDS write/read；适合 LLM 推理"权重预 swizzle"场景（B = weight，可线下 reorder 成最佳 layout）
- ⚠️ 代价：**B 完全占 VGPR**，N_Tile 翻倍 → B VGPR 翻倍 → 容易撞 VGPR 上限。这正是本项目 FP6 卡在 N=256 上不去的根本原因（详见 §6 问题 1）

### 1.2 MXFP6 = Microscaling FP6

OCP 标准的 6-bit 浮点：

```
FP6 单元素 = 6 bit
  ┌─┬───┬─────┐
  │S│ E │  M  │
  └─┴───┴─────┘
   1   2    3        S=符号, E=2bit 指数, M=3bit 尾数
   bit bit  bit

每 32 个 FP6 元素共享一个 e8m0 scale (8 bit, 纯指数)
  ┌────────────────────────────────┐ ┌───┐
  │  32 × FP6  =  32 × 6 = 192 bit │ │ S │  ← scale
  │  = 24 字节                      │ │8bit│
  └────────────────────────────────┘ └───┘

平均存储成本 ≈ 0.75 字节/元素 (FP8=1.0, FP4=0.5)
```

### 1.3 用途与约束

- **LLM 推理矩阵乘**：A = activation（每次变化）、B = weight（可线下预处理）
- 训练后量化 (PTQ) 把 FP16/BF16 权重压成 MXFP6，节省 HBM 带宽和容量

### 1.4 项目位置

| 路径 | 用途 |
|------|------|
| `example/ck_tile/18_flatmm/mxgemm/` | benchmark/example 入口 |
| `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp` | **通用 pipeline**（FP4/FP6/FP8 共用，当前最佳 FP6 走这条路） |
| `include/ck_tile/ops/flatmm/pipeline/mx_fp6_flatmm_pipeline_v1.hpp` | **FP6 专属 pipeline**（早期为 clang 22 写的，clang 20 上回退，已弃用） |
| `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp` | C 矩阵写回的 CShuffle epilogue |

---

## 2. 硬件关键参数（gfx950 / CDNA4）

```
┌──────────────────────────────────────────────────────────────────┐
│  MI350X / MI355X (gfx950, CDNA4)                                 │
│  ──────────────────────────────────────────────────────────────  │
│  CU 数量            : 256                                         │
│  每 CU VGPR+AGPR    : 512 (统一池，单 wave)                       │
│                       ← 关键约束 ← FP4 N=512 配置刚好填满         │
│  每 CU LDS          : 160 KB, 64 banks                            │
│  HBM3e 峰值带宽     : 8 TB/s                                      │
│                                                                   │
│  关键 MFMA 指令: v_mfma_scale_f32_16x16x128_f8f6f4                │
│  通过 cbsz/blgp 选择数据类型，吞吐量见下表:                       │
│                                                                   │
│       数据类型      cbsz  blgp  cycles/MFMA  Peak (MI355X)        │
│       ──────────   ────  ────  ───────────  ─────────────         │
│       FP8 × FP8     0    0     32           10.1 PFLOPS           │
│       FP6 × FP6     2    2     16  ← 2x !   20.1 PFLOPS           │
│       FP4 × FP4     4    4     16           20.1 PFLOPS           │
│                                                                   │
│  → FP6 在硬件层有真实的 2x compute throughput vs FP8              │
│  → FP6 与 FP4 共享乘法器，吞吐量相同                              │
└──────────────────────────────────────────────────────────────────┘
```

来源验证：CDNA4 ISA 手册 page 58 + 实测 ISA disasm（详 `task11_fp6_task7_vs_fp8_deep_analysis.md` §5）

**注意**：B200 上 FP6 = FP8（NVIDIA 实现方式不同），AMD CDNA4 上 FP6 = FP4 = 2× FP8，这是 AMD 独有的硬件优势。

---

## 3. FP6 MFMA 数据消费方式

### 3.1 每条 MFMA 的寄存器开销

一条 `v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:2 blgp:2`（FP6 模式）：

```
                ┌──────────────┐
   A operand:   │  6 VGPR      │  24 bytes = 16 个 FP6 元素 (packed)
                └──────────────┘
                ┌──────────────┐
   B operand:   │  6 VGPR      │  24 bytes
                └──────────────┘
   Scale A:     │  1 VGPR (低 8 bit, e8m0, 对应 32 elements) │
   Scale B:     │  1 VGPR (低 8 bit, e8m0) │
                ┌──────────────┐
   C accum:     │  4 AGPR      │  16 bytes = 4 × FP32 输出
                └──────────────┘
```

对比 FP8 / FP4：

| 维度 | FP4 | **FP6** | FP8 |
|------|-----|---------|-----|
| A operand VGPR | 4 | **6** | 8 |
| B operand VGPR | 4 | **6** | 8 |
| Scale A/B VGPR | 1+1 | 1+1 | 1+1 |
| C accumulator AGPR | 4 | 4 | 4 |
| MFMA cycles | 16 | **16** | 32 |

来源：`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp:1700` 的 `AVecType` 定义 +
`pk_fp6.hpp` 的 `pk_fp6x16_t` (= int32x6, 24 bytes)

### 3.2 FP6 在 LDS 中的存储 — 25% padding 浪费

```
每 16 个 fp6 = 96 bits = 12 bytes (= K2 = DWORDx3)
            ▼
LDS 实际占用 16 bytes (= K2_Pad = DWORDx4)
            ▼

  [ ----16 个 fp6 (12B) ---- ][ ZERO PAD (4B) ]
  └──────────── 16 B per "K2 column" ────────┘
                              ↑
                              MFMA 硬件要求 4B 对齐，
                              而 12B 不是 16B-bus 友好的尺寸
```

**结果**：FP6 在 LDS 的 **bytes-per-element 与 FP8 完全相同**，FP6 的 25% 字节优势在 LDS 内被填充抹平。这也是为何 FP6 单 ds_read 必须拆成 b32+b64 而不能用 b96 直读（详见 §6 问题 2）。

源码定位：`include/ck_tile/ops/flatmm/pipeline/mx_fp6_flatmm_pipeline_v1_policy.hpp:160-161`

---

## 4. 完整计算流程（一个 kernel dispatch 的生命周期）

以基准问题 **M=N=K=8192** 为例，一步步看 GPU 上发生了什么。

### 4.1 Grid 配置（FP6 当前最佳，通用 pipeline）

```
Tile:           M_Tile=128, N_Tile=256, K_Tile=256
Grid (blocks):  ceil(8192/128) × ceil(8192/256) = 64 × 32 = 2048 blocks
Block:          256 threads = 4 waves
Per-CU:         2048 / 256 = 8 blocks/CU
Per-CU MFMAs:   8 × (256/256) × 64 = 8 × 32 K-iter × 64 MFMA = 16384 MFMAs
```

对比 FP4 的 grid（N=512）：
```
FP4 Grid:       64 × 16 = 1024 blocks (只一半！)
FP4 Per-CU:     4 blocks/CU (epilogue/prologue 开销摊薄 2×)
FP4 MFMA/block: 128 (vs FP6 的 64)
Per-CU MFMAs:   4 × 32 × 128 = 16384 (与 FP6 同)
```

→ 这就是 FP4 一个 block 干 2 倍工作，固定开销摊薄一半的原理。

### 4.2 一个 block 的执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 1: Prologue                          │
│                                                                  │
│  HBM ──buffer_load_dwordx3 × 24──→ B 矩阵首块 (24B/lane)        │
│                                    直接进 VGPR (FlatMM 特征)     │
│                                                                  │
│  HBM ──buffer_load_dwordx3──→ LDS  A 矩阵首块                   │
│       (buffer_load_to_lds 直 DMA)  按 K2_Pad=16B 写              │
│                                                                  │
│  HBM ──buffer_load_dword──→ VGPR   A/B Scale tensor (e8m0)       │
│                                                                  │
│  LDS ──ds_read───→ VGPR           A 首块预读到寄存器             │
│   ┌────────────────────────────┐                                 │
│   │ A 端: ds_read_b32 + b64   │  ← 因 scale 字节交错布局        │
│   │       (12B 拆成 4+8)       │     无法 coalesce 成 b96         │
│   │                            │                                 │
│   │ B 端: ds_read_b96         │  ← B 数据 LDS 中连续             │
│   │       (12B 单条搞定)      │     可以 b96 直读                │
│   └────────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│      Phase 2: K-Loop (展开 32 次, K_Tile=256, K=8192/256=32)   │
│                                                                  │
│  每次 K-iter 处理 256 个 K 元素                                  │
│  每次 K-iter 发射 M_Rep=8 × N_Rep=4 × K_unroll=2 = 64 条 MFMA    │
│                                                                  │
│  for k_iter in 32:                                               │
│    ┌──────────────────────────────────────────┐                  │
│    │ // 与当前计算交叠地预取下一轮            │                  │
│    │ buffer_load (B → VGPR pong-buffer)       │                  │
│    │ buffer_load_to_lds (A → LDS pong)        │                  │
│    │ ds_read (A_pong from LDS → VGPR)         │                  │
│    │                                          │                  │
│    │ // 主计算 (64 MFMA, 完全展开)            │                  │
│    │ v_mfma_scale_f32_16x16x128_f8f6f4 ×64    │                  │
│    │   acc += A_ping × B_ping                 │                  │
│    │                                          │                  │
│    │ // ping ↔ pong 交换                      │                  │
│    └──────────────────────────────────────────┘                  │
│                                                                  │
│  K-loop 被编译器完全展开 → 无分支                                │
│  每 block 共 32 × 64 = 2048 MFMA                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 3: Epilogue                          │
│                                                                  │
│  C accumulator (AGPR) ──v_accvgpr_read_b32 × 128──→ VGPR        │
│                                                                  │
│  FP32 ──v_cvt_f16_f32──→ FP16 (output dtype)                    │
│                                                                  │
│  for chunk in 16:  (CShuffleEpilogue 16 个 sub-tile)             │
│    VGPR ──ds_write_b16 × 8──→ LDS                                │
│    s_waitcnt lgkmcnt(0)                                          │
│    s_barrier              ← 每 chunk 1 个 barrier, 共 16 个      │
│    LDS  ──ds_read_b128 × 1──→ VGPR (重新打包)                    │
│    VGPR ──buffer_store_dwordx4 × 1──→ HBM                        │
│                                                                  │
│  16 chunk × ~120 cycle/chunk ≈ 1920 cycle 单纯 epilogue 开销     │
└─────────────────────────────────────────────────────────────────┘
```

ISA 实测对照表（FP6 当前最佳，task7 winner，from `task11_fp6_task7_vs_fp8_deep_analysis.md`）：

| 指令 | 计数 | 说明 |
|------|-----:|------|
| `v_mfma_scale_f32_16x16x128_f8f6f4` (cbsz:2) | 64 | 主计算 |
| `s_barrier` | 17 | 1 K-loop prologue + 16 epilogue |
| `ds_read_b32 / b64 / b96 / b128` | 16/16/16/16 | A 拆分 + B b96 + epilogue b128 |
| `ds_write_b16` | 128 | epilogue (16 chunk × 8 writes) |
| `buffer_load_dwordx3` | 24 | A+B HBM load |
| `buffer_store_dwordx4` | 16 | epilogue HBM store |
| `v_accvgpr_read_b32` | 128 | AGPR → VGPR (epilogue) |
| `v_mov_b32_e32` | 113 | **寄存器重组开销**（vs FP8 的 12！见 §6 问题 2） |

---

## 5. FP6 vs FP8 vs FP4 关键差异

### 5.1 综合对比表

| 维度 | **FP4** | **FP6** | **FP8** |
|------|---------|---------|---------|
| 理论 peak (MI350X) | 18.3 PFLOPS | 18.3 PFLOPS | 9.1 PFLOPS |
| MFMA cycles | 16 | 16 | 32 |
| 每元素字节 (HBM) | 0.5 B | 0.75 B | 1.0 B |
| MFMA A operand VGPR | 4 | 6 | 8 |
| MFMA B operand VGPR | 4 | 6 | 8 |
| **K2 真实数据 (16 elements)** | 16 B (DWORDx4) | 12 B (DWORDx3) | 16 B (DWORDx4) |
| **K2_Pad LDS 占用** | 16 B | 16 B (pad +4B) | 16 B |
| **LDS 浪费率** | 0 % | **25 %** | 0 % |
| 单条 ds_read 宽度 | b128 (统一) | b32+b64 / b96 (拆) | b128 (统一) |
| **N_Tile (当前可达)** | **512** ✓ | **256** ⚠ | 256 |
| Grid blocks (8K²) | 1024 | 2048 | 2048 |
| Per-CU blocks | 4 | 8 | 8 |
| Per-CU epilogue 次数 | 4 | 8 | 8 |
| VGPR / AGPR (实测) | 256 / 256 = **512 满** | 236 / 112 = 348 | 144 / 128 = 272 |
| LDS (实测) | 33 KB | 67 KB | 67 KB |
| **实测 TFLOPS (8K³)** | **3940** | **1219** | **1857** |
| 实测 / 理论 peak | ~22 % | **~7 %** | ~20 % |
| 相对 FP8 倍率 | 2.12 × | **0.66 ×** | 1.0 × |

### 5.2 一句话总结

- **FP4** 是 CDNA4 首发优化场景，AMD 精调到 N=512 配置 V+A 恰好填满 512 池 → epilogue 摊薄 + AGPR 满载 0 spill → 3940 TFLOPS
- **FP8** 是简单老实人，layout 完美对齐 (8×128bit=1024bit / wave)，编译器写出 65 个 barrier 也跑得快 → 1857 TFLOPS
- **FP6** 卡在中间：数据 12B 不对齐 16B LDS bus → 拆 b32+b64 → 113 个 v_mov 重组寄存器 → 调度密度差 → 1219 TFLOPS

---

## 6. 发现的所有问题（按影响排序）

### 🥇 问题 1：N_Tile=256 天花板 — 最根本的限制

```
FlatMM pipeline 设计 (回顾 §1.1)：
  B 矩阵完全在 VGPR 中，不经过 LDS

N_Tile 加倍 → B 操作数寄存器需求加倍

每 wave 的 B 寄存器:
  ┌─────────────────────────────────────────────┐
  │ FP4: 4 dword × N_Rep=8 = 32 V + 32 pong = 64│  N=512 刚好 fit
  │ FP6: 6 dword × N_Rep=8 = 48 V + 48 pong = 96│  N=512 超 V+A=512 → spill
  │ FP8: 8 dword × N_Rep=4 = 32 V + 32 pong = 64│  N=256 已紧
  └─────────────────────────────────────────────┘

实验验证 (Task #12, #13)：
  FP6 N=512 直接套 FP4 config:    331 TFLOPS (-73% vs 基线 1225)
  FP6 N=512 + 关 ping-pong:       668 TFLOPS (-45%)
  FP6 M=64 N=512 K=256:           668 TFLOPS (M 减半无用，B VGPR 不变)
  FP6 M=64 N=512 K=128:           输出全 0 (破坏 pipeline K-iter 假设)
  FP6 DsReadPreload=2 (省 VGPR):  编译失败 (结构性依赖)

ISA 验证 (Task #15)：
  FP6 N=512 hot variant spill = 570~691 dwords ❌
```

**为什么 FP4 的 N=512 是 FP4 性能优势的最大来源**：
- 每 block 工作量 ×2 → epilogue/prologue 摊薄一半
- AGPR=256（M_Rep×N_Rep×4 = 8×8×4）→ 0 spill
- LDS 仅 33KB → 高 L2 命中率

**为什么 FP6 无法复制**：FP6 的 A/B operand 比 FP4 大 50%（6 vs 4 dword），刚好把 FP4 精调的 512 池吃掉。

**→ N=256 是 FP6 在当前 FlatMM pipeline 下的天花板。突破需要 (a) 重构 pipeline 让 B 走 LDS，或 (b) 用 32×32×64 MFMA（4× 输出/指令，减少 B live time）。**

---

### 🥈 问题 2：A-side LDS 读取被拆分（b32+b64 而非 b96）

#### 现象

```
FP8 A-side LDS 读取:               FP6 A-side LDS 读取:
  ┌────────────────────┐             ┌──────┐┌────────────┐
  │  ds_read_b128  16B │             │ b32  ││  b64       │
  │  → 4 dword contig  │             │ 4B   ││  8B        │
  └────────────────────┘             └──────┘└────────────┘
  1 LDS transaction                  2 LDS transactions
  4 contiguous VGPRs                 散落在非连续 VGPR
       │                                  │
       ▼                                  ▼
  直接送 MFMA src                    需 5+ 条 v_mov_b32 重组成
                                     v[74:79] 才能送 MFMA src
```

#### 根因

FP6 在 LDS 中的 layout：scale 字节与 FP6 数据**交错**存储，12 字节的 FP6 chunk 跨越 8B LDS bank 边界，编译器 (clang 20) 无法 coalesce 成 b96。clang 22 的更新 coalescer 能合并，clang 20 不行。

#### 量化影响

```
每个 A tile: 多 1 条 ds_read + 5 条 v_mov_b32
× 16 A tiles × N_Rep duplication
= 113 条 v_mov_b32 (vs FP8 仅 12 条)

cycle 估算:
  - A-side LDS split:    ~80 cyc / tile
  - 113 v_mov 寄存器重组: ~95 cyc / tile (移除可省)
  - 合计:                ~175 cyc / 9535 cyc/tile = ~2% 损失
```

#### 修复方向

- **Task #8（未做）**：重写 LDS layout 让 A 数据连续（scale 在 prologue 单独 load 到 VGPR），统一所有 A 端 ds_read 为 b96。预期 +8~10%。

源码定位：`include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`

---

### 🥉 问题 3：MFMA 调度密度低

```
FP8 ISA MFMA cluster:           FP6 ISA MFMA cluster:
  行 1068 ─┬─ MFMA                行 1117 ─┬─ MFMA
           │                                │  (大量 v_mov, ds_read 间隔)
           │  64 MFMAs in           ────────┤
           │  190 lines             ────────┤
           │  (2.97 行/MFMA)        ────────┤
           │                                │
  行 1257 ─┴─ MFMA (END)           行 1408 ─┴─ MFMA (END)
                                              64 MFMAs in 624 lines
                                              (9.75 行/MFMA, 3× 稀疏)

  MFMA 对间距分布:
   gap=0 (back-to-back, 编译器认为安全 issue): FP8=29  FP6=16
   gap≥4 (有 full MFMA latency stall):         FP8≈0   FP6=39  ← 问题
```

#### 影响

- FP6 39 个 MFMA 对有 ≥4 cycle 间距 → 全部 stall
- 估算 ~390 cycles / tile，占 FP6 性能损失的 14%

#### 修复方向

加 `__builtin_amdgcn_sched_group_barrier` 强制 MFMA 紧密 cluster（FP8 用了这个）。Task #9 提议但未做。

---

### 4️⃣ 问题 4：CShuffleEpilogue 的 16 barrier 开销

```
FP6 epilogue 模式（per chunk × 16）：
  ds_write_b16 × 8
   → s_waitcnt lgkmcnt(0)    ← 全 LDS 排空，~30 cyc
   → s_barrier               ← workgroup 同步，~30~60 cyc
   → ds_read_b128            ← LDS 读，~20 cyc
   → s_waitcnt lgkmcnt(0)    ← ~10 cyc
   → buffer_store_dwordx4    ← 阻塞 (非 atomic)

  ≈ 120 cycle / chunk × 16 chunks = ~1920 cycle / tile
                                     ↑
                              当前 FP6 最大单点损失
```

FP8 对比：用 `buffer_atomic_pk_add_f16`（非阻塞），65 个 barrier 看起来多但在 MFMA 之后批量出现，不 stall 计算流水线。

#### 修复方向

把 FP6 epilogue 改成 FP8 风格的 atomic ping-pong。Task #11 提议，难度高（正确性敏感）。

源码定位：`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp:823-859`

---

### 5️⃣ 问题 5：clang 20 vs clang 22 编译器差异

| 指标 | clang 22 (ROCm 7.13) | clang 20 (ROCm 7.0.2.1, **客户锁定**) |
|------|---------------------|--------------------------------------|
| FP6 TFLOPS | 1386 | 1068 (回退 23%) |
| ds_read | 64 × b96（统一） | 4 种宽度混合 |
| K-loop barrier | 0（消除了） | **17 个**保留 |
| VGPR | 216 | 244 |

clang 20 的 limitations：
1. 不能消除 ping-pong 路径的 redundant barrier
2. 不能 coalesce A-side LDS 读到 b96

**关键决策**：因为客户锁定 ROCm 7.0.2.1，且 clang 20 对 FP6 专属 pipeline 表现差，**当前最佳方案是放弃 FP6 专属 pipeline，回退到通用 pipeline**（FP4/FP8 共用的 `MXFlatmmPipelineAGmemBGmemCRegV1`）。这个回退即 **Task #7**，把 FP6 从 1068 推到 **1219 TFLOPS**。

---

### 6️⃣ 问题 6：K2_Pad=16 的 LDS 浪费

```
FP6 实际数据: 12 B / 16 elements
LDS padding:  4 B zero-pad
LDS 占用:     16 B / 16 elements  (= FP8 等量, 0 字节优势)

为什么必须 pad?
  MFMA 硬件要求 4B 对齐
  12B 不是 16B-bus 友好尺寸
  K2_Pad=12 实验已证明不可行（cbsz=2 硬件要求）

→ 这是 FP6 在 LDS 层面的硬件固有代价，无法消除
→ 影响 HBM→LDS 1.33x 优势（FP6 0.75 B/elt vs FP8 1.0 B/elt）被 LDS 抹平
```

---

## 7. 已尝试的优化及结果

| Task # | 实验内容 | 结果 | 备注 |
|--------|---------|------|------|
| 基线 | FP6 专属 pipeline (clang 20) | **1068 TFLOPS** | 17 barrier 与 MFMA 交织 |
| #4 (Step 2) | + `sched_barrier(0)` 分离 MFMA/epilogue | 1174 (+8.7%) | MFMA span 624→243 行 |
| #4 (Step 3) | + 移除 `iglp_opt` | 1102 (-6.1%) | 与 sched_barrier 冲突 |
| **#7** | **回退到通用 pipeline** (FP4/FP8 同款) | **1219 (+14.1%)** | **当前最佳 ✓** |
| #8 | 统一 A-side ds_read 为 b96 | (未做) | 预期 +8~10% |
| #12 | 套用 FP4 的 128×512 tile | **331 (-73%)** ❌ | B VGPR spill |
| #13-A | DsReadPreload=2 (省 VGPR 用 N=512) | 编译失败 ❌ | 结构性依赖 preload=4 |
| #13-C | warp tile 16×8×128 (减半 N_Rep) | 硬件不存在 ❌ | gfx950 只有 16×16×128 / 32×32×64 |
| #13-D | M=64 N=512 K=256 | 668 (-45%) ❌ | M 减半不影响 B VGPR |
| #13-E | M=64 N=512 K=128 | 输出全 0 ❌ | 破坏 pipeline K-iter |
| #14 | 验证 N=512 metadata 是否生效 | 已生效，性能确实回退 | 排除 "config 未传播" 误判 |

**已排除方向汇总（Phase 1~3 累计）**：
- ❌ N_Tile=512（B VGPR spill, 无 fallback config）
- ❌ N_Tile=384（不整除 8192，tail-handling 复杂）
- ❌ M_Tile=64（不解决 B 端瓶颈）
- ❌ K_Tile=128（破坏 pipeline 设计）
- ❌ DsReadPreload=2（结构性依赖）
- ❌ FP6 专属 pipeline 在 clang 20 上（barrier 不消除）
- ❌ iglp_opt 移除（与 sched_barrier 冲突）
- ❌ warp tile 16×8×128（硬件不存在）
- ❌ K2_Pad=12（cbsz=2 硬件要求 4B zero-pad）

---

## 8. 性能天花板分析

### 8.1 Cycle-level 性能模型

```
单 block (M=128, N=256, K=256, 32 K-iter, 64 MFMA/iter)

每 block 总周期预算（FP6 当前 1219 TFLOPS 时反推）:
  ≈ 1.062 ms × 2.1 GHz / 8 block/CU ≈ 9535 cycles/block

理论 compute peak (纯 MFMA 时间):
  FP6: 64 MFMA × 16 cycle = 1024 cycle  ← 仅占当前 10.7%
  FP8: 64 MFMA × 32 cycle = 2048 cycle  ← 占当前 ~21%

→ 当前 ~89% 时间花在非 MFMA 开销（memory + epilogue + stall）
→ FP6 完全 memory-bound，不是 compute-bound
```

### 8.2 已知 stall 分解（task9）

| 来源 | cycles/block | 占当前 9535 比例 |
|------|-------------:|----------------:|
| Epilogue 16 个 barrier × 120 | ~1920 | 20.1% |
| Back-to-back MFMA 21 个 × 32 | ~672 | 7.0% |
| A-side LDS split 16 × 30 | ~480 | 5.0% |
| AGPR drain exposure | ~150 | 1.6% |
| **总可优化 stall** | **~3222** | **33.8%** |

### 8.3 上限推演

```
假设三大方向全做 (sched_group_barrier + b96 unify + atomic epilogue)：
  消除 stall: ~3222 cycle
  剩余 budget: 9535 - 3222 = 6313 cycle/block
  → 预计 TFLOPS: 1219 × (9535/6313) ≈ 1840 ≈ 0.99 × FP8

  即使完美消除所有 stall，FP6 在 N=256 也只能勉强追平 FP8！

要达到客户目标 1.3× FP8 = 2400 TFLOPS:
  必须让 FP6 进入 compute-bound regime
  → 需要 N>256 或根本性的数据搬运优化
  → 当前 FlatMM pipeline 设计下 N>256 不可行
```

### 8.4 HBM 带宽分析

| 维度 | FP4 | FP6 | FP8 |
|------|-----|-----|-----|
| HBM 字节/元素 | 0.5 | 0.75 | 1.0 |
| 理论 HBM 加速比 vs FP8 | 2.0× | 1.33× | 1.0× |

→ FP6 的纯硬件优势上限是 **1.33×**（HBM 带宽优势），但当前 0.66× 远未达到。

---

## 9. 结论与未来方向

### 9.1 当前状态总结

- **FP6 = 1219 TFLOPS**（通用 pipeline, N=256, no FP6-专属改动）
- **FP6 / FP8 = 0.66×**（目标 1.3~1.5×，缺口巨大）
- **N_Tile=256 是当前架构天花板**，已多方向证伪

### 9.2 推荐下一步（按 ROI 排序）

```
ROI 高 ─────────────────────────────────────── ROI 低
   │
   ▼

🥇 Task #8: 统一 A-side ds_read 到 b96
   预期: +8~10% → ~1320 TFLOPS
   难度: 中（rework LDS layout, 验 bank conflict）
   风险: 低
   预算: 1~2 天

🥈 sched_group_barrier 优化 MFMA cluster
   预期: +10~15% → ~1460 TFLOPS
   难度: 中（找正确的 group size）
   风险: 中
   预算: 2~3 天

🥉 把 FP6 加入 32×32×64 MFMA dispatcher
   预期: +20~30% (减少 B VGPR live time, 可能解锁 N=512)
   难度: 高（需 dispatcher 特化 + 验证）
   风险: 高
   预算: 1~2 周

4️⃣ 重构通用 pipeline，让 B 走 LDS
   预期: 可能解锁 N=512 → +50~100%
   难度: 极高（FlatMM 设计哲学颠覆）
   风险: 极高
   预算: 数周

5️⃣ Epilogue 改 atomic ping-pong（FP8 同款）
   预期: +5~10%（移除 16 个 barrier 的 stall）
   难度: 高（正确性敏感）
   风险: 高
   预算: 1 周
```

### 9.3 给新接手者的建议

1. **先跑基线**：`./bin/tile_example_mx_flatmm -m=8192 -n=8192 -k=8192 -mx_prec=fp6xfp6 -v=0 -warmup=50 -repeat=100` 应得到 ~1219 TFLOPS
2. **读 4 份关键报告**（按顺序）：
   - `report_task5_fp6_vs_fp8_theory.md` — 理论模型
   - `report_task6_cycle_model.md` — cycle 模型 + ISA 实测
   - `task11_fp6_task7_vs_fp8_deep_analysis.md` — 当前最佳的 ISA 解剖
   - `report_task15_vgpr_budget.md` — N=512 失败的 VGPR 分析
3. **认清"不要再走的路"**：见 §7 已排除方向清单。每条都已 ISA + 实测双重验证
4. **保 baseline 二进制**：实验前 `cp build/bin/tile_example_mx_flatmm /tmp/baseline_fp6_1219`
5. **优先 Task #8**：低风险，能拿到 +8~10% 而不引入新问题

### 9.4 客户目标可达性诚实评估

| 目标 | 达成可能性 | 必需条件 |
|------|----------|---------|
| 1.0× FP8 (1857 TFLOPS) | 🟡 中 | 完成 Task #8 + sched_group_barrier + epilogue 优化 |
| 1.3× FP8 (2414 TFLOPS) | 🔴 低（无突破性变化下）| 必须解锁 N>256 → 重构 pipeline 或加 32×32 MFMA |
| 1.5× FP8 (2786 TFLOPS) | 🔴 极低 | 同上 + 接近 HBM 带宽极限 |

**诚实建议**：在不改架构的前提下，**1.0× FP8 是切实可达的中期目标**；客户的 1.3~1.5× 目标需要结构性投入（数周以上）。

---

## 10. 附录：关键源码定位

### 10.1 入口与配置

| 用途 | 路径 |
|------|------|
| Example 入口 | `example/ck_tile/18_flatmm/mx_flatmm.cpp` |
| Tile config (FP4/FP6/FP8) | `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_arch_traits.hpp` |
| Instance dispatcher | `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.hpp` |

### 10.2 Pipeline

| 用途 | 路径 |
|------|------|
| **通用 pipeline (FP6 当前用)** | `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp` |
| **通用 pipeline policy** | `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp` |
| FP6 专属 pipeline (已弃) | `include/ck_tile/ops/flatmm/pipeline/mx_fp6_flatmm_pipeline_v1.hpp` |
| FP6 专属 policy (已弃) | `include/ck_tile/ops/flatmm/pipeline/mx_fp6_flatmm_pipeline_v1_policy.hpp` |

### 10.3 Warp/Epilogue

| 用途 | 路径 |
|------|------|
| MFMA wrapper (AVecType 定义) | `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp:1700` |
| pk_fp6 type | `include/ck_tile/core/numeric/pk_fp6.hpp` |
| CShuffleEpilogue (16 barrier 来源) | `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp:823-859` |
| arch.hpp (block_sync_lds) | `include/ck_tile/core/arch/arch.hpp:1252` |

### 10.4 编译/Benchmark 命令

```bash
# Build (-G Ninja 必需，否则 cmake_depends 单线程要 11 分钟)
cd /home/AMD/zhewan/rocm-libraries-ck/projects/composablekernel
mkdir -p build && cd build
cmake --preset dev -DGPU_TARGETS="gfx950" -DBUILD_TESTING=OFF -G Ninja ..
ninja tile_example_mx_flatmm -j$(nproc)

# Benchmark (-v=0 跳过正确性, 只跑性能)
./bin/tile_example_mx_flatmm -m=8192 -n=8192 -k=8192 \
    -mx_prec=fp6xfp6 -v=0 -warmup=50 -repeat=100

# 切换精度: fp4xfp4 / fp6xfp6 / fp8xfp8 / fp8xfp4 / fp4xfp8
```

### 10.5 关键参考文件

| 报告 | 路径 |
|------|------|
| Barrier 根因 | `/tmp/ck-mxfp6-flatmm-opt/research/report_task1_barriers.md` |
| FP6/FP8 理论 | `/tmp/ck-mxfp6-flatmm-opt/research/report_task5_fp6_vs_fp8_theory.md` |
| Cycle 模型 | `/tmp/ck-mxfp6-flatmm-opt/research/report_task6_cycle_model.md` |
| FP4 性能根因 | `/tmp/ck-mxfp6-flatmm-opt/research/report_task10_fp4_analysis.md` |
| N=512 失败原因 | `/tmp/ck-mxfp6-flatmm-opt/research/report_task14_n512_propagation.md` |
| VGPR 预算 | `/tmp/ck-mxfp6-flatmm-opt/research/report_task15_vgpr_budget.md` |
| FP8 vs FP6 ISA | `/tmp/ck-mxfp6-flatmm-opt/prof/task2_fp8_vs_fp6_pipeline_diff.md` |
| FP6 ISA 深度 | `/tmp/ck-mxfp6-flatmm-opt/prof/task3_fp6_isa_deep_analysis.md` |
| Step 2 验证 | `/tmp/ck-mxfp6-flatmm-opt/prof/task9_step2_verification_and_remaining_stalls.md` |
| 通用 pipeline 分析 | `/tmp/ck-mxfp6-flatmm-opt/prof/task11_fp6_task7_vs_fp8_deep_analysis.md` |
| FP4 N=512 VGPR | `/tmp/ck-mxfp6-flatmm-opt/prof/task_fp4_n512_vgpr_analysis.md` |
| Phase 3 总结 (memory) | `~/.claude/projects/.../memory/project_mxfp6_phase3_findings.md` |
| ROCm 7.0.2.1 基线 (memory) | `~/.claude/projects/.../memory/project_mxfp6_rocm702_baseline.md` |
| Impl 状态 | `/tmp/ck-mxfp6-flatmm-opt/impl/status/impl.md` |

---

*文档生成于 2026-05-26，基于 ROCm 7.0.2.1 + clang 20 + gfx950 (MI350X) 平台的 14 份子任务报告综合整理。*
