# SWMMAC rocWMMA Technical Reference

**版本**: 1.0  
**日期**: 2026-05-15  
**目标硬件**: gfx1200/gfx1201 (RDNA4), gfx1100/gfx1150 (RDNA3 fallback)

---

## 一、架构概览

```
Layer 1: rocwmma_int4.hpp          类型系统 + 寄存器定义 + 架构检测
Layer 2: rocwmma_swmmac.hpp        SWMMAC 后端 (8 个) + WMMA 后端 (2 个)
Layer 3: rocwmma_16chain.hpp       ChainPipeline<N, Backend>
Layer 4: rocwmma_fragment_swmmac.hpp  fragment API bridge
Layer 5: rocwmma_gfx11_fallback.hpp   gfx11 WMMA 替代路径

Core:   internal/types.hpp          int4_t 前向声明
        internal/pack_util_impl.hpp PackTraits<int4_t> (PackRatio=4)
        internal/swmmac_impl.hpp    amdgcn_swmmac 后端 (MFMA 模式)
        internal/swmmac.hpp         Swmmac<> 公共接口
        internal/swmmac_traits.hpp  IO 适配 + 寄存器 traits
```

## 二、API 参考

### 2.1 类型系统

```cpp
#include <rocwmma/rocwmma_int4.hpp>

rocwmma::int4_t           // 4-bit signed integer (-8..7)
rocwmma::SwmmacARegsT     // <2×i32>  A 寄存器 (16 INT4 值)
rocwmma::SwmmacBRegsT     // <4×i32>  B 寄存器 (32 INT4 值)
rocwmma::SwmmacAccumT     // <8×i32>  累加器
rocwmma::SwmmacFpAccumT   // <8×f32>  FP 累加器

// INT4 打包工具
uint32_t rocwmma::pack_int4x8(v0..v7)        // 8 INT4 → 1 i32 (硬件 nibble 顺序)
int4_t   rocwmma::unpack_int4_nibble(u32, n) // 提取第 n 个 nibble
int32_t  rocwmma::broadcast_int4(v)          // 广播填充整个 i32
```

### 2.2 后端 (单次 SWMMAC)

```cpp
#include <rocwmma/rocwmma_swmmac.hpp>

// INT4: 16×16×64, 32768 ops
SwmmacInt4<ASign,BSign,CSign>::exec(a, b, c, sparse_idx)

// INT8: 16×16×32, 16384 ops
SwmmacInt8<ASign,BSign,CSign>::exec(a, b, c, sparse_idx)

// FP8/BF8: 16×16×32, 16384 ops, f32 accum
SwmmacFp8Fp8::exec(a, b, c, sparse_idx)
SwmmacFp8Bf8::exec(a, b, c, sparse_idx)
SwmmacBf8Fp8::exec(a, b, c, sparse_idx)
SwmmacBf8Bf8::exec(a, b, c, sparse_idx)

// FP16/BF16: 16×16×32, 16384 ops, f32 accum, wider registers
SwmmacFp16::exec(a, b, c, sparse_idx)  // A=<8×f16>, B=<16×f16>
SwmmacBf16::exec(a, b, c, sparse_idx)  // A=<8×i16>, B=<16×i16>

// 默认别名 (signed, dense)
SwmmacI4 = SwmmacInt4<true, true, true>
SwmmacI8 = SwmmacInt8<true, true, true>
```

### 2.3 ChainPipeline (多链 XDL 流水线)

```cpp
#include <rocwmma/rocwmma_16chain.hpp>

// INT4 (默认)
ChainPipeline<16> pipe;  // Backend=SwmmacI4
pipe.zero();
pipe.load(C_ptr);
for (int i = 0; i < loops; ++i)
    pipe.step(A, B, sparse_idx);
pipe.step_sparse(A, B);        // 2:4 稀疏
pipe.step_dual(A0,B0,A1,B1);   // 双缓冲
pipe.store(C_out);

// 预定义别名
Chain16           = ChainPipeline<16, SwmmacI4>
Chain14           = ChainPipeline<14, SwmmacI4>
Chain16Int8       = ChainPipeline<16, SwmmacI8>
ChainFp8Fp8       = ChainPipeline<16, SwmmacFp8Fp8>
ChainFp16         = ChainPipeline<16, SwmmacFp16>
ChainBf16         = ChainPipeline<16, SwmmacBf16>

// 架构自动调度
#include <rocwmma/rocwmma_gfx11_fallback.hpp>
AutoChain<16> pipe;  // gfx12→SWMMAC, gfx11→WMMA
```

### 2.4 Fragment API Bridge

```cpp
#include <rocwmma/rocwmma_fragment_swmmac.hpp>

fragment<accumulator, 16, 16, 64, int32_t> d;
fragment<matrix_a, 16, 16, 64, int32_t> a;
fragment<matrix_b, 16, 16, 64, int32_t> b;

load_matrix_sync(a, A_ptr, 16);
load_matrix_sync(b, B_ptr, 16);

swmmac_mma(d, a, b, d);              // 单次 SWMMAC
swmmac_mma_int8(d, a, b, d);         // INT8 变体
swmmac_mma_16chain(accum, a, b, 160, 0);  // 16-chain
swmmac_mma_sparse(d, a, b, d, idx);  // 稀疏模式
auto_mma(d, a, b, c);                // 架构自动选择

store_matrix_sync(C_ptr, d, 16);
```

### 2.5 厂商级 amdgcn_swmmac (MFMA 模式)

```cpp
#include <rocwmma/internal/swmmac.hpp>

// 底层后端 (仿照 amdgcn_mfma)
using SwmmacInt4 = detail::amdgcn_swmmac<int4_t, int4_t, int32_t, 16, 16, 64>;

// 公共接口
using Swmmac = Swmmac<FragM, FragN, FragK, InputTA, InputTB, ComputeT, BlockM, BlockN>;

// Traits 适配
using Traits = SwmmacTraits<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK>;
```

## 三、性能参考 (gfx1200, 32 CUs, 2780 MHz)

| 精度 | K | ops/inst | 峰值 TOPs | 有效 cycle/inst | 推荐场景 |
|------|---|----------|-----------|----------------|---------|
| INT4 | 64 | 32768 | **809** | 8.2 | 最高吞吐, GF(3) 整数计算 |
| INT8 | 32 | 16384 | **401** | 8.2 | 平衡精度/吞吐 |
| FP8  | 32 | 16384 | **403** | 8.2 | 浮点推理, 满血吞吐 |
| FP16 | 32 | 16384 | **202** | 16.4 | 需要 f16 精度时 (硬件限制) |
| BF16 | 32 | 16384 | **200** | 16.4 | 训练兼容 (硬件限制) |

Pipeline 调优: 14-chain 在 256/1024 波时优于 16-chain (双波占用率 +2.7%).

## 四、编译需求

```
编译器: LLVM 23+ (SWMMAC builtins)
目标:   gfx1200, gfx1201 (RDNA4)
定义:   -DROCWMMA_WAVE32_MODE=1
gfx11:  -DROCWMMA_WAVE32_MODE=1 (自动 WMMA fallback)

gfx11 交叉编译: gfx1100, gfx1101, gfx1102, gfx1150, gfx1151
```

## 五、文件清单

```
rocWMMA 头文件 (library/include/rocwmma/):
├── rocwmma_int4.hpp              # 类型 + 寄存器 + 打包 + 架构检测
├── rocwmma_swmmac.hpp            # 8 SWMMAC 后端 + MmaTraits
├── rocwmma_16chain.hpp           # ChainPipeline + 别名
├── rocwmma_fragment_swmmac.hpp   # fragment API bridge
├── rocwmma_gfx11_fallback.hpp    # WMMA fallback + AutoChain
└── internal/
    ├── swmmac_impl.hpp           # amdgcn_swmmac 后端 (MFMA 模式)
    ├── swmmac.hpp                # Swmmac<> 公共接口
    └── swmmac_traits.hpp         # IO 适配 + 寄存器 traits

rocWMMA 核心修改:
└── internal/
    ├── types.hpp                 # +1: int4_t 前向声明
    └── pack_util_impl.hpp        # +14: PackTraits<int4_t>

GPU 基准测试 (21 文件):
    func_test_swmmac.cpp          # 完整功能测试 (0 failures)
    bench_rocwmma.cpp             # 精确吞吐基准
    bench_swmmac_int4_vs_int8.cpp # INT4 vs INT8
    bench_fp_all.cpp              # FP 全族 (8 后端)
    bench_sparse_2x4.cpp          # 2:4 稀疏
    bench_lds_swmmac.cpp          # LDS 优化
    bench_tex_lds.cpp             # LDS+纹理 双路径
    bench_bc4_swmmac.cpp          # BC4 压缩
    bench_async_prefetch.cpp      # 异步预取
    bench_pipeline_tuning.cpp     # 流水线调优
    bench_dma_stream.cpp          # DMA SDR 流
    kernel_swmmac_optimized.cpp   # 生产级参考内核
    kernel_christoffel.cpp        # Christoffel bilinear
    kernel_christoffel_nonlinear.cpp # Christoffel 非线性
    kernel_christoffel.hsaco      # HSACO (117 KB)
    n14_calibration.py            # N14 量子钟校准
    verify_7000.py                # 7000 步验证
    sov_ternary_pfm.py            # GF(3) Ternary-PFM

文档:
    swmmac_rocwmma_integration_spec.md  # 集成规格 (12 真值分析)
    SWMMAC_Technical_Reference.md       # 本文档 (API 参考)
    TODO.md                             # 最终归档
```

## 六、测试矩阵

```
测试                    gfx1200    gfx1100   结果
──────────────────────────────────────────────
func_test_swmmac        ✅ 实测     —        0 failures
bench_fp_all            ✅ 实测     —        8/8 后端
bench_lds_swmmac        ✅ 实测     —        LDS ±0% baseline
bench_tex_lds           ✅ 实测     —        3-path verified
bench_async_prefetch    ✅ 实测     —        +90.6% @8 tiles
bench_pipeline_tuning   ✅ 实测     —        14ch optimal
bench_dma_stream        ✅ 实测     —        93% overlap
kernel_christoffel      ✅ 实测     —        geodesic≠bilinear
gfx11 WMMA fallback     —          ✅ 编译    AutoChain dispatch
PackTraits<int4_t>      ✅ 编译     ✅ 编译    PackRatio=4
amdgcn_swmmac backend   ✅ 编译     ✅ 编译    MFMA pattern
```

## 七、接口标准化

所有后端遵循统一接口:

```
Backend::ARegsT, BRegsT, CRegsT, DRegsT  — 寄存器类型
Backend::BlockM, BlockN, BlockK           — 几何尺寸
Backend::exec(a, b, c, [idx]) → DRegsT    — 执行 (静态内联)

ChainPipeline<CHAINS, Backend>:
  .zero()            — 累加器清零
  .load(ptr)         — 从内存加载
  .step(A, B, [idx]) — N-chain XDL 步进
  .step_sparse(A, B) — 2:4 稀疏步进
  .step_dual(...)    — 双缓冲步进
  .store(ptr)        — 存储到内存
  .theoretical_tops() — 理论峰值
```

> **2026-05-18 更新**: RDNA4 HWXDL Silent Drop 发现。所有 kernel 已升级为 wave 级
> `__builtin_amdgcn_readfirstlane` 任务领取 (Lane 0 代理 + 全波广播, EXEC=0xFFFFFFFF)。
> 反例复现与微架构报告: `/data/rtl-sdr/swmmac/active/silent_drop/`

---

## 六、BF16 SWMMAC 物理数据流映射 (2026-05-18 DOE 逆向)

### 6.1 外积引擎拓扑

BF16 SWMMAC (`v_swmmac_f32_16x16x32_bf16_w32`) 在 gfx1200 上是纯外积并行矩阵引擎：

```
C[row][col] = Σ_k A[col][k] ⊗ B[row][k]
```

- **A 阵列广播**: A[lane=L] 的数据广播到全部 32 个输出 lane 的 elem[L]
- **B 阵行隔离**: B[lane=R] 的数据仅影响输出 lane R
- **K 轴折叠**: lanes 0..15 (K-half 0), lanes 16..31 (K-half 1)

### 6.2 One-Hot DOE 校准数据

| A[lane=L] | 输出映射 | Delta |
|-----------|---------|-------|
| lane 0 | outL0 elem[0] | +16 |
| lane 1 | outL0 elem[1] | +16 |
| ... | ... | +16 |
| lane 7 | outL0 elem[7] | +16 |
| lane 8 | outL16 elem[0] | +16 |
| lane 16 | outL0 elem[0] (2nd K-half) | +16 |
| lane 24 | outL16 elem[0] (2nd K-half) | +16 |

B 活跃位: stride-4 (pos 0,4,8,12 per 16-register group)
2:4 结构化稀疏: s_idx 控制解压模式选择

### 6.3 物理打包公式

```cpp
// A: 列广播 — 外积引擎
target_lane = col_group*8 + col_off + k_half*16;
target_reg  = (k % 16) / 2;

// B: 行隔离
target_lane = row;
target_reg  = k % 16;  // stride-4 active via 2:4 sparsity
```

### 6.4 HW 传递函数

8-chain × 2 K-stride × SWMMAC_intrinsic_factor = 29.265564×
逆常数: 0.034170 (Epilogue 硬编码)
C[0][0] vs reference: 零误差

### 6.5 与 INT4 的关键差异

| 特性 | INT4 SWMMAC | BF16 SWMMAC |
|------|------------|------------|
| 数据流 | 内积 | 外积 |
| K 覆盖 | 16-chain HW 自动分配 | 8-chain 软件 per-chain 分流 |
| K 维度 | K=64 per call | K=32 per call (2:4 sparse → 64 effective) |
| s_idx | sign/sparsity 复合 | 纯 2:4 sparsity 元数据 |
