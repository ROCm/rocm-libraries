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

```
mxfp6_gemm/
├── mxfp6_types.hpp          # FP6/E8M0 编解码 + dense packing
├── mxfp6_preprocess.hpp     # 量化/反量化, B 转置, scale 重排, B pre-shuffle
├── mxfp6_reference.hpp      # CPU golden reference GEMM
├── mxfp6_asm_utils.hpp      # GPU ASM: MFMA (inline asm, AccVGPR), LDS, store
├── test_reference.cpp       # CPU 单元测试 (162 pass)
├── test_lds_shuffle.cpp     # A 矩阵 LDS 通路验证
├── test_k_loop.cpp          # K 循环测试 (K=64~512, AccTileV)
├── test_accvgpr.cpp         # AccVGPR 路径测试 (AccTileA)
├── test_multitile.cpp       # Multi-tile 128×128 测试
├── test_pipeline.cpp        # Double buffer pipeline 测试 + benchmark
├── bench_gemm.cpp           # 性能 benchmark (无 pipeline 基线)
├── bench_lds_shuffle.cpp    # LDS 带宽基准
├── Makefile
└── counters.txt             # rocprofv3 计数器配置
```

**工具：**
- rocprofv3: 直接用系统版 `/opt/rocm/bin/rocprofv3` (ROCm 7.0.2.1)，**不再需要** libatomic 的 `LD_LIBRARY_PATH` workaround
  - 仅 ATT 解码追踪时需 `--att-library-path /home/AMD/zhewan/rocm-tools/opt/rocm-7.0.2.1/lib`（trace decoder 库只在该用户态目录）
- ISA PDF: `python3 -c "import fitz; doc = fitz.open('path'); print(doc[page].get_text())"`
- ISA dump: `hipcc -save-temps` 或 `hipcc -S`

## Next Steps

**性能优化方向（当前 ~750 TFLOPS, 峰值 ~9200 TFLOPS, 利用率 ~8%）。已由 Step 7.5 的 ATT profile 验证瓶颈,按预期收益排序：**

1. **真正的跨迭代软件流水（最高收益,攻 ~70% stall）**: iter 开始即 issue 下一轮 A 的 ds_read + B 的 global_load,然后立刻对**本轮**已就位数据做 4×MFMA,让 load 在 MFMA 执行期后台完成。当前 A 虽是 LDS 双缓冲但**读仍阻塞**(lgkmcnt 32.9% stall)、B 根本没预取(vmcnt 36.7% stall)。
2. **B 寄存器双缓冲**: 两组 B VGPR 交替,消除每 iter 的 `wait vmcnt(0)` 关键路径。
3. **预取 A 改 async 且移到 MFMA 之后**: 现在源码 L111-124 的预取排在 MFMA 之前、VMEM+ds_write 压在关键路径上;改用 GLOBAL_LOAD_LDS async(见 memory `mxfp6_a_matrix_shuffle`)并置于 MFMA 之后,与之并行。
4. **提高 occupancy**: arch+acc VGPR 同步压到 ≤51 → 5 waves/SIMD,多 wave 互相掩盖延迟(当前 64/64 双双卡 4 waves)。
5. **增大 N 方向寄存器 tile（更多 MFMA/iter）**: 当前每 iter 仅 4 MFMA=128 矩阵周期,太小掩盖不住 load;增大可摊薄 load 延迟与 barrier。
6. **降低 barrier 频率**: ≥3 buffer 让 barrier 不必每 iter(当前 barrier 占 11.8% stall)。
7. **减少每-MFMA 的 VALU 开销**: 11.1 条纯 VALU/MFMA;scale 字节清理(v_and)出内循环、地址增量改加法、精简 ds_read_complete 的 v_mov。

**复现 profile**: `cd mxfp6_gemm && cp test_pipeline.cpp prof_pipeline.cpp`(改 main 为单发,见旧 driver),`hipcc -O3 --offload-arch=gfx950`,然后用 gpu-profile skill(`/opt/rocm/bin/rocprofv3`)。最新结果在 `mxfp6_gemm/prof_out/`。
