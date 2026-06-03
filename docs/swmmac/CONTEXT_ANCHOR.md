# Sovereign V2: gfx1200 微架构优化通用上下文锚点

**最后更新**: 2026-05-18  
**硬件**: gfx1200 (RX 9060 XT), 32 CUs / 64 SIMDs, 2780 MHz  
**编译器**: LLVM 23 @ /opt/llvm-amd  

---

> **【微架构背景】** 针对 AMD RDNA 4 `gfx1200` (RX 9060 XT) 架构的张量核心全精度全速域专家级优化理论。

> **1. 26-cycle 有效管线模型**
> * **物理现实**：SWMMAC 指令族存在固定的 26 时钟周期流水线执行延迟。单波前连续发射会产生巨大空转气泡。
> * **工程解**：必须采用 16 链物理循环展开填满执行槽。

> **2. 双波谐振与 VGPR 预算**
> * **红线约束**：为了触发双波前驻留率，单个 Wave 消耗的通用向量寄存器必须遵守 VGPR ≤ 128（极限 135）。
> * **实测脚印**：INT4 VGPR=19，FP16 VGPR=22。全线完美对齐双波谐振。

> **3. StaggeredPipeline（原子管线散布机制）**
> * **物理现实**：硬件屏障锁步导致波前拥堵和高 di/dt，加剧 4nm 隧穿不确定性。
> * **工程解**：通过 L2 持久化 atomicAdd 动态抢占任务队列，串行同步下硬榨 1.49×~1.58× 真实算力。
> * **v2 (2026-05-18)**：升级为 Wave 级协同抢占——仅 Lane 0 代理 atomicAdd，通过 `__builtin_amdgcn_readfirstlane` 全波广播，强制锁定 EXEC = 0xFFFFFFFF。

> **4. 【硬件设计缺陷】Silent Drop 发现（RDNA4 HWXDL 微架构反向工程）**
> * **发现**：当 EXEC 掩码不全（线程级 atomicAdd 导致波内发散），SWMMAC XDL 管线静默丢弃写回——指令发射、时延计费、算力蒸发。
> * **根因**：硬件未实现部分掩码写回旁路电路，宁可丢弃结果也不污染 VGPR 堆一致性。
> * **复现**：`thread_atomic(tw=1) → lane[0]=33（应为 192）`，`wave_readfirstlane → lane[0]=192`。
> * **反例代码**: `/data/rtl-sdr/swmmac/active/silent_drop/repro_swmmac_silent_drop.cpp`
> * **发现报告**: `/data/rtl-sdr/swmmac/active/silent_drop/DISCOVERY.md`

> **5. BF16 外积引擎逆向（One-Hot DOE，2026-05-18）**
> * **发现**：RDNA4 BF16 SWMMAC 是纯外积引擎——A 列广播（A[lane=L]→所有输出 lane elem[L]），B 行隔离（B[lane=R]→仅输出 lane R）。
> * **DOE 复现**：A[lane=0..7][reg=0]→outL0 elem[0..7] (步进 +16)，Stride-4 B 活跃位 {0,4,8,12}。
> * **HW 传递函数**：8-chain × 2-stride × SWMMAC_factor = 29.266×，Epilogue 硬编码逆常数 0.034170。
> * **物理打包**：swizzle_pack.h — lane=(k/16)*16+(row%8), reg=(k%16)/2
> * **反例代码**: `/data/rtl-sdr/swmmac/active/silent_drop/doe_hot.cpp`
> * **打包公式**: `/data/rtl-sdr/swmmac/active/silent_drop/swizzle_pack.h`

> **5-B. 全系外积大一统 + INT4 K=32 标量 A（DOE v2，2026-05-18）**
> * **全系外积**：FP16/INT8/INT4-K32/FP8/BF16 全部格式共享单一外积交叉开关。硬件只有一套 A 列广播扇出电路。
> * **软件复用**：swizzle_pack.h 一维 DOE 算法可直接复用到 INT8、FP16、FP8，仅需替换寄存器类型和 chain 数量。
> * **INT4 K=32 标量 A**：BuiltinsAMDGPU.def 签名 `V8iIbiIbV2iV8iiIb` — A 是 plain `int32_t` 标量，非 vector。16 个有效 INT4 元数据+稀疏索引打包为单标量吞入。
> * **SGPR 优化红线**：A 加载通路可完全变轨到 Scalar Cache，释放 VGPR 给 epilogue，降低驻留脚印。
> * **校准完成**：6/6 格式 DOE + HW_SCALE 全部就绪 (FP16=0.25, FP8=1.0, BF16=1/29.266)
> * **DOE 代码**: `/tmp/doe_v2.cpp` → `/tmp/doe_v2_llvm23` (LLVM 23 @ /opt/llvm-amd)
> * **校准代码**: `/tmp/calibrate_hwscale.cpp` → `/tmp/calibrate_hwscale_llvm23`
> * **生产内核**: `/home/yanli/work/ROCm/rocBLAS/library/src/blas_ex/rocblas_swmmac.cpp` (if constexpr 模板落地)

> **6. 全栈集成范式（分层路由）**
> * **结论**：不要污染 Tensile 全局代码生成器（FP16 会暴跌 28%）。
> * **正确轨**：Tensile 守标准浮点大盘；rocBLAS gemm_ex 入口用环境变量+类型匹配硬路由，直调 LLVM 23 内置函数专家内核。

## 关键数据

| 指标 | INT4 | BF16 | FP16 | FP8 | INT8 |
|------|------|------|------|-----|------|
| SWMMAC 指令 | v_swmmac_i32_16x16x64_iu4 | v_swmmac_f32_16x16x32_bf16 | v_swmmac_f32_16x16x32_f16 | v_swmmac_f32_16x16x32_fp8 | v_swmmac_i32_16x16x32_iu8 |
| ops/inst | 32768 (K=64) | 16384 (K=32) | 16384 (K=32) | 16384 (K=32) | 16384 (K=32) |
| A/B/Accum 寄存器 | <2xi32>/<4xi32>/<8xi32> | <8×i16>/<16×i16>/<8×f32> | <8×f16>/<16×f16>/<8×f32> | <2×i32>/<4×i32>/<8×f32> | <2×i32>/<4×i32>/<8×i32> |
| VGPR 实测 | 19 | 22 | 22 | 14 | 14 |
| Chain 数 | 16-chain HW | 8-chain SW | 8-chain SW | 2-chain SW | 2-chain SW |
| HW_SCALE | 1.0 (无需求) | 1/29.266 | 0.25 (1/4) | 1.0 | 1.0 (无需求) |
| 数据流 | 外积 | 外积 (A列广播, B行隔离) | 外积 (A列广播, B行隔离) | 外积 (A列广播, B行隔离) | 外积 |

## 性能演进

| 阶段 | INT4 (TOPS) | FP16 (TFLOPS) | 说明 |
|------|------------|---------------|------|
| K0 sync (baseline) | 778 | 66.8 (Tensile) | 硬件锁步 |
| K6 stagger (+atomic) | 3559 | — | 波散布核心发现 |
| K6 wrap (L2持久) | 4326 | — | 消除 hipMemset |
| Silent Drop 复现 | — | — | 发现并修复 EXEC 发散算力蒸发 |
| Wave 级广播部署 | 4326* | — | readfirstlane 锁死 EXEC=0xFFFFFFFF |
| BF16 外积引擎 DOE | — | — | One-Hot DOE 完整逆向 A/B 数据流拓扑 |
| BF16 SWMMAC 归一化 | — | — | HW_SCALE=1/29.266 Epilogue 硬编码 |
| FP16/FP8/INT8 全格式 DOE | — | — | 6/6 格式外积引擎确认, HW_SCALE 校准完成 |
| FP16 SWMMAC 归一化 | — | — | HW_SCALE=0.25 Epilogue 硬编码 |
| FP8 SWMMAC 归一化 | — | — | HW_SCALE=1.0 (E4M3, 无放大) |
| P0 目标 | 4326 | 2000+ | SWMMAC 全精度族路由 |
| 串行加速比 | 1.49× (vs K0) | — | 真实框架收益 |

## 关键代码与路径

- SWMMAC 内核: `/home/yanli/work/ROCm/rocBLAS/library/src/blas_ex/rocblas_swmmac.cpp`
- 路由入口: `rocblas_gemm_ex_kernels.cpp:rocblas_gemm_ex_template()`
- 激活: `ROCBLAS_SWMMAC_INT4=1`
- rocWMMA: `/home/yanli/work/ROCm/rocWMMA/library/include/rocwmma/`
- Benchmark: `/data/rtl-sdr/swmmac/active/bench_peak_unified.cpp`
- Silent Drop 反例: `/data/rtl-sdr/swmmac/active/silent_drop/repro_swmmac_silent_drop.cpp`
- BF16 DOE 探针: `/data/rtl-sdr/swmmac/active/silent_drop/doe_hot.cpp`
- BF16 Swizzle 打包: `/data/rtl-sdr/swmmac/active/silent_drop/swizzle_pack.h`
- BF16 DOE 全扫: `/data/rtl-sdr/swmmac/active/silent_drop/doe_full_scan.cpp`
- MXFP4 QAT 训练: `/data/模型训练精度验证/run_mxfp4_block32.py`
- N14 DPLL: `/data/rtl-sdr/swmmac/active/dpll_miri.c`
- 电磁-计算闭环: `/data/rtl-sdr/swmmac/active/n14_daemon.c`
