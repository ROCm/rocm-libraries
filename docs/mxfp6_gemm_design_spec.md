# MXFP6 高性能 GEMM Kernel 技术设计文档

> 目标平台: AMD Instinct MI350 (CDNA4, gfx950)
> 计算: D = A × B + C，A/B 为 MXFP6 (E2M3) + Block Scale (E8M0)
> 问题规模: M=2K~8K, K=4K~8K, N=4K~8K
> 实现: 独立 HIP kernel (头文件+源文件), 不依赖 CK 框架

---

## 1. 硬件规格 (MI350 / CDNA4, 源自 ISA 文档)

| 资源 | 规格 |
|------|------|
| LDS | **160 KB / CU**, 64 banks × 640 DWORDs, 每 bank 32-bit 宽 |
| Arch VGPR | **256 / wave** (V0-V255) |
| AccVGPR | **256 / wave** (AV0-AV255), 与 Arch 分离的独立寄存器文件 |
| SIMD | 4 SIMDs / CU |
| Wave size | 64 threads |
| MFMA AccVGPR 输入 | ACC[0]=1: SRC-A from AccVGPR; ACC[1]=1: SRC-B from AccVGPR (ISA p.596) |
| AccVGPR 直写 HBM | GLOBAL_STORE 支持 ACC=1, DATA 来自 AccVGPR (ISA p.98) |

---

## 2. 核心 MFMA 指令

**`V_MFMA_SCALE_F32_32x32x64_F8F6F4`**

| 属性 | 值 |
|------|-----|
| 输出矩阵 | 32×32 F32 = 16 AccVGPR |
| A/B 输入 | 各 6 VGPR (32 个 FP6 密集打包 = 192 bits) |
| K 维度 | 64 |
| 执行周期 | 32 cycles (FP6×FP6, 非 F8) |
| 格式选择 | CBSZ[2:0]=0b010 (FP6 E2M3 for A), BLGP[2:0]=0b010 (FP6 for B) |
| Scale | ABID[0]=1 启用, E8M0 格式, 每 32 个 K 值共享 1 个 scale |
| Scale OP_SEL | 2-bit 选择 VGPR 的哪个 byte: 00=[7:0], 01=[15:8], 10=[23:16], 11=[31:24] |

**FP6 MFMA 寄存器布局 (32x32x64):**

```
           row0(thr 0-15)   row1(thr 16-31)  row2(thr 32-47)  row3(thr 48-63)
           M/N = [0-15]     M/N = [16-31]    M/N = [0-15]     M/N = [16-31]
v0-v2      k=0-15           k=0-15           k=32-47          k=32-47
v3-v5      k=16-31          k=16-31          k=48-63          k=48-63
```

每个 lane 持有 32 个 FP6 值 = 192 bits = 6 VGPR。16 FP6 = 96 bits = 3 VGPR = 1 次 `DS_READ_B96`。

**Scale 布局 (32x32 输出, K=64):**

```
Lane 0-15:  M=0-15,  K=0..31  (每 lane 1 个 8-bit scale)
Lane 16-31: M=16-31, K=0..31
Lane 32-47: M=0-15,  K=32..63
Lane 48-63: M=16-31, K=32..63
```

与 A 矩阵的线程-M 映射完全一致。每条 MFMA 需 64 个 scale = 1/4 VGPR × 64 lanes。

---

## 3. 数据布局

### 3.1 A 矩阵: `A[M][K]` 行主序

- FP6 沿 K 方向密集打包
- 每行 64 FP6 值 (一次 MFMA 的 K=64) = 384 bits = 48 bytes = 12 DWORDs
- Stride = 48 bytes/行

### 3.2 B 矩阵: `B[N][K]` 列主序 (预处理后)

- **预处理**将 B 从原始格式转成列主序 `B[N][K]`
- 与 A[M][K] **完全对称**: 每"行" = 1 个 N 值的所有 K 值 = 48 bytes
- A 和 B 复用完全相同的 load/shuffle 代码路径

### 3.3 Scale: 预处理为 MFMA lane layout

- Scale_A: 预处理成按 MFMA lane 分布的格式, kernel 内直接 VGPR load
- Scale_B: 同理
- 不走 LDS (数据量太小, 64 bytes/MFMA)
- 加载指令宽度由 K_TILE 推导:
  - `SCALE_BYTES_PER_LANE = K_TILE / 64`
  - K_TILE=64 → `GLOBAL_LOAD_UBYTE`, K_TILE=128 → `GLOBAL_LOAD_USHORT`, K_TILE=256 → `GLOBAL_LOAD_DWORD`
- 内层循环通过 OP_SEL 选择 byte: `OP_SEL = mfma_index % 4, VGPR_index = mfma_index / 4`

### 3.4 输出矩阵 D: 模板参数可选 F32 / F16 / BF16

---

## 4. 数据搬运: A/B 矩阵 LDS Shuffle

A 和 B 共用同一套对称的数据路径:

```
  GLOBAL_LOAD_LDS_DWORDX4     (HBM → LDS 直达, 零 VGPR 消耗)
          ↓ __syncthreads()
  DS_READ_B96 × 2             (LDS → VGPR, shuffle: 每次 12B = 16 FP6)
          ↓
  V_MFMA_SCALE_F32_32x32x64   (6 VGPR 送入 MFMA)
```

### 4.1 Async Load: `GLOBAL_LOAD_LDS_DWORDX4`

- HBM 数据不经过 VGPR, 直达 LDS
- LDS 地址 = `M0[17:0] + TIDinWave × 16` (ISA p.92)
- 每线程 16 bytes, 64 线程 = 1024 bytes/pass
- 每行 48 bytes = 3 个 chunk, 3 线程协作加载 1 行
- 因为 48 = 3 × 16, 线性映射 `LDS[tid × 16]` **天然等于 stride=48 的行主序**
- 32 行 × 48 bytes = 1536 bytes → 2 pass (pass1: 64 线程, pass2: 32 线程)

### 4.2 LDS Read: `DS_READ_B96` × 2

每个线程根据 MFMA 布局从 LDS 读取自己的 M/N 行:

```
线程 i   (0-15):  LDS[i×48]      和 LDS[i×48+12]      → v0-v5 (M=i,   k=0-31)
线程 16+i(0-15):  LDS[(16+i)×48] 和 LDS[(16+i)×48+12]  → v0-v5 (M=16+i,k=0-31)
线程 32+i(0-15):  LDS[i×48+24]   和 LDS[i×48+36]       → v0-v5 (M=i,   k=32-63)
线程 48+i(0-15):  LDS[(16+i)×48+24] 和 LDS[(16+i)×48+36] → v0-v5 (M=16+i,k=32-63)
```

**Shuffle 本质**: 线程 T0 写入 16 bytes (chunk0), 但读取 24 bytes (chunk0 + chunk1 的一部分)。多出的数据来自 T1 写入的字节。

### 4.3 Bank Conflict 分析

**DS_WRITE_B128 (写入):**
- 检查规则: 每 8 个连续线程一组
- 线性写 `LDS[t×16]`: 8 线程 × 4 DWORDs = 32 banks 铺满 → **零冲突 ✓**

**DS_READ_B96 (读取):**
- 检查规则: 与 DS_READ_B128 相同 — {t0-3, t20-23} 同色组检查
- stride = 48 bytes = 12 DWORDs, 每次读 3 DWORDs
- 同组 8 线程起始 bank: {b, b+12, b+24, b+4, b+16, b+28, b+8, b+20} (mod 32)
- 间距 (12) >> 宽度 (3) → **零冲突 ✓**

**数学保证**: 8 线程 × 3 banks = 24/32 banks, stride=12 DW 的均匀分布保证不重叠。

---

## 5. 模板参数

```cpp
template <
    int M_TILE,      // WG tile M 方向, 32 的倍数 (64/128/256)
    int N_TILE,      // WG tile N 方向, 32 的倍数 (64/128/256)
    int K_TILE,      // K 方向每次迭代, 64 的倍数 (64/128), 默认 128
    int N_WAVES,     // WG 内 wave 数 (1/2/4), 需整除 MFMA 总数
    int PIPE_DEPTH,  // LDS buffer 深度 (1=无prefetch, 2=double buffer)
    typename OutputType  // 输出精度 (float/half/__hip_bfloat16)
>
```

### 5.1 硬约束

**AccVGPR ≤ 256 / wave (瓶颈):**
```
累积器 = (M_TILE/32) × (N_TILE/32) / N_WAVES × 16
若 B 放 AccVGPR: + 6 × (N_TILE/32)
总和 ≤ 256
```

**Arch VGPR ≤ 256 / wave (宽裕):**
```
A/B 走 async load 不经过 Arch VGPR
Arch VGPR 仅: DS_READ 数据 (~12) + scale (~4) + 地址 (~8) + 控制 (~6) ≈ 30
```

**LDS ≤ 160 KB / workgroup (宽裕):**
```
(M_TILE + N_TILE) × K_TILE × 0.75 × PIPE_DEPTH + scale
最大配置 256×256, K=128, pipe=3 ≈ 144 KB < 160 KB
```

**MFMA 分配:**
```
(M_TILE/32) × (N_TILE/32) 必须被 N_WAVES 整除
```

### 5.2 典型配置

| 配置 | M×N | K_TILE | Waves | Pipe | AccVGPR | LDS |
|------|-----|--------|-------|------|---------|-----|
| 保守 | 128×128 | 128 | 2 | 2 | 128 | 24 KB |
| 平衡 | 128×256 | 128 | 2 | 2 | 256 | 36 KB |
| 激进 | 256×256 | 128 | 4 | 2 | 256 | 48 KB |

---

## 6. K 循环 Software Pipelining

### 6.1 Double Buffer 主循环

```
// ============ Prologue ============
async_load(A_tile[0], B_tile[0] → LDS_buf[0])
load_scale(scale_A[0], scale_B[0])

// ============ Main Loop ============
for k = K_TILE to K_total step K_TILE:
    s_waitcnt vmcnt(0)                         // 等上一轮 async load 完成

    // Prefetch: 发射下一轮 async load (不消耗 VGPR)
    async_load(A_tile[k], B_tile[k] → LDS_buf[1])
    load_scale(scale_A[k], scale_B[k])

    // Compute: 从当前 buffer 读数据 + MFMA
    for m = 0 to M_TILE/32:
        ds_read_b96 × 2 (A[m] from LDS_buf[0])
        for n = 0 to N_TILE/32:
            ds_read_b96 × 2 (B[n] from LDS_buf[0])
            mfma_scale(acc[m][n], A[m], B[n], scale_A, scale_B)

    swap(LDS_buf[0], LDS_buf[1])

// ============ Epilogue ============
// 最后一轮: 只计算不 prefetch
s_waitcnt vmcnt(0)
[compute last K_TILE]
[store output]
```

### 6.2 MFMA 期间指令交叠

MFMA 发射占 1 cycle, 剩余 31 cycles 可并行发射其他指令:

| 可并行指令 | 条件 |
|-----------|------|
| DS_READ_B96 (下一 MFMA 的 A/B) | 不 overlap VDST |
| GLOBAL_LOAD_LDS (prefetch) | async, 独立 pipeline |
| SALU (地址计算/循环控制) | 完全自由 |
| 下一条 MFMA (SrcC forwarding) | 同 opcode, 同 VDST → 0 wait |

反压安全:
- LGKM_CNT max=15, 任意时刻 DS_READ in-flight ≈ 4-8 < 15 ✓
- VM_CNT max=63, async load in-flight ≈ 12 << 63 ✓

### 6.3 延迟隐藏

HBM latency ≈ 400 cycles。**默认方案: K_TILE=128**。

以 128×128, 2 waves 为例:
```
K_TILE=128 → 每 wave 16 MFMA × 32 cycles = 512 cycles > 400 → 完全藏住 ✓
```

备选方案 (可通过模板参数切换):
1. 增大 M/N tile (256×256): 更多 MFMA → 更多 compute cycles
2. Triple buffer (PIPE_DEPTH=3): prefetch 提前 2 步, 不依赖单步藏延迟
3. K_TILE=128 (**推荐默认**): 最简单, 不挤 AccVGPR

---

## 7. 输出 Epilogue

K 循环结束后, 累积器 (AccVGPR, F32) 写回 HBM:

| 输出精度 | 指令序列 | Arch VGPR 消耗 |
|---------|---------|---------------|
| **F32** | `GLOBAL_STORE_DWORDX4 (ACC=1)` — 直写 | 0 |
| **F16** | `V_ACCVGPR_READ` → `V_CVT_PK_F16_F32` → `GLOBAL_STORE_DWORDX2` | 临时使用 |
| **BF16** | `V_ACCVGPR_READ` → `V_CVT_PK_BF16_F32` → `GLOBAL_STORE_DWORDX2` | 临时使用 |

Epilogue 只执行一次, 不影响主循环性能。可扩展 bias/activation。

---

## 8. 实现方式

### 8.1 Intrinsic + Inline ASM 混用

**Intrinsic** (可读性优先):
- MFMA intrinsic (如 ROCm 提供)
- 数据类型转换 (`__builtin_amdgcn_cvt_*`)
- `__syncthreads()`

**Inline ASM** (精确控制):
- `GLOBAL_LOAD_LDS_DWORDX4` (需设 M0)
- `DS_READ_B96` (非标准宽度)
- `V_MFMA_SCALE_F32_32x32x64_F8F6F4` (4-DWORD 指令, OP_SEL/ACC/CBSZ/BLGP)
- `GLOBAL_STORE (ACC=1)` (AccVGPR 直写)
- `s_waitcnt` (精确 vmcnt/lgkmcnt)
- AccVGPR pinning (`"a"` constraint)

### 8.2 代码组织

```
mxfp6_gemm.hpp          // 主头文件, kernel launch API, 模板参数
mxfp6_gemm_kernel.hpp   // kernel 主体, 主循环 (调用封装函数, 清晰可读)
mxfp6_asm_utils.hpp     // 全部 ASM 封装为 __device__ __forceinline__ 函数
mxfp6_preprocess.hpp    // B 矩阵转置+打包, scale 重排 (host 端)
```

### 8.3 ASM 封装函数列表

| 函数名 | 底层指令 | 用途 |
|--------|---------|------|
| `async_load_lds_b128(gaddr, lds_offset)` | GLOBAL_LOAD_LDS_DWORDX4 | HBM→LDS async |
| `ds_read_b96(v0, v1, v2, lds_addr)` | DS_READ_B96 | LDS→VGPR |
| `mfma_scale_f32_32x32x64_fp6(acc, a, b, sA, sB, opsel)` | V_MFMA_SCALE_F32_32x32x64_F8F6F4 | 矩阵乘累加 |
| `global_store_acc_b128(gaddr, acc_reg)` | GLOBAL_STORE_DWORDX4 (ACC=1) | AccVGPR→HBM |
| `wait_vmcnt(n)` / `wait_lgkmcnt(n)` | s_waitcnt | 同步 |

---

## 9. 预处理 (Host 端)

| 操作 | 输入 | 输出 |
|------|------|------|
| B 转置 | B[K][N] 行主序 | B[N][K] 列主序, FP6 沿 K 打包 |
| Scale_A 重排 | scale_A[M][K/32] 行主序 | MFMA lane layout 格式 |
| Scale_B 重排 | scale_B[N][K/32] | 同上 |

预处理后 B 和 A 的内存布局完全对称, kernel 内可复用同一套模板代码。

---

## 10. 实现步骤 (分步验证)

### Step 1: CPU 参考实现 + 预处理
- 实现 CPU 端 MXFP6 GEMM 参考计算 (naive, 用于验证 GPU 结果)
- 实现 CPU 端预处理:
  - B 矩阵: 原始格式 → 列主序 `B[N][K]`, FP6 沿 K 密集打包
  - Scale_A: `scale_A[M][K/32]` → MFMA lane layout 格式
  - Scale_B: `scale_B[N][K/32]` → 同上
- 验证点: 预处理后的数据能被 CPU 参考实现正确消费, 结果与未预处理版本一致
- 产出: 后续所有 GPU 步骤的 golden reference 数据

### Step 2: ASM 工具函数 + 单元测试
- 实现 `mxfp6_asm_utils.hpp` 中所有封装函数
- 每个函数独立编写 micro-benchmark 验证正确性
- 验证点: `DS_READ_B96` 输出正确, `GLOBAL_LOAD_LDS` 数据到达 LDS, `s_waitcnt` 行为符合预期

### Step 3: A 矩阵 LDS Shuffle 验证
- 实现 async load → LDS → DS_READ_B96 的完整 A 矩阵搬运
- 用 Step 1 的预处理数据作为输入
- 验证 LDS 中的数据排列是否匹配 MFMA 寄存器布局
- 验证 bank conflict (用 profiler 确认零 conflict)

### Step 4: 单次 MFMA 正确性
- 固定 M=N=32, K=64, 单条 V_MFMA_SCALE_F32_32x32x64_F8F6F4 调用
- 输入: Step 1 预处理后的 A/B/scale 数据
- 输出: 与 Step 1 的 CPU reference 对比
- 验证 scale 的 OP_SEL 选择是否正确

### Step 5: K 循环 (无 prefetch)
- 实现完整 K 循环, PIPE_DEPTH=1 (无 double buffer)
- 用 Step 1 的 CPU reference 验证大 K 下的数值正确性
- 这一步不关心性能, 只关心正确性

### Step 6: Double Buffer Pipeline
- 加入 async load prefetch + double buffer
- 验证功能正确性不受 pipeline 影响 (与 Step 5 结果一致)
- 开始关注性能 (profiler 查看 stall/overlap)

### Step 7: 模板化 + Tile Tuning
- 将 M/N/K_TILE, N_WAVES, OutputType 等做成模板参数
- 实现多组配置的 auto-tuning 框架
- 在目标问题规模 (M=2K~8K, K/N=4K/8K) 上找最优配置

### Step 8: 端到端集成 + 性能优化
- 完整 GEMM API: preprocess → kernel launch → output
- Profiling 驱动的微调: 寄存器分配, 指令调度, occupancy
- 可选: triple buffer, B-in-AccVGPR 等高级优化
