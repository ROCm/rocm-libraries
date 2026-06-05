# RCV Trace — MXFP6 GEMM 生产 kernel (lds_gemm_db @ 8192³, buffer_load_lds)

ATT trace,可直接用 **RCV (Radeon Compute Viewer)** 打开审阅。
本次抓的是 **Step 30(buffer_load_lds 默认 + M0 s_nop 关闭)之后**的当前最优 kernel (06-05)。

## 抓的是什么

- **Kernel**: `lds_gemm_db<256, 256, 192, 2, 2, MIN_OCC=1, SWZ=0, DB=true, __half>`
  —— v18 dispatcher 在 8192³ 的实际选择(`choose_tile`=TLDS, `choose_swz(8192,8192)=0`)。
- **问题规模**: M=N=K=8192,K padding 到 8256 = 43×192,**单次 dispatch**。
- **配置**: 256×256 tile · KT192 深-K 双缓冲(RDB)· 4 wave(2×2)· 16 累加器/wave(occ1)· swz off。
- **load 路径**: **MUBUF `buffer_load_dwordx4 ... lds`**(Step 29 起默认,base 进描述符);
  **M0 hazard s_nop 关闭**(`MXFP6_M0_NOP=0`,Step 30,KT192 下冗余且免费)。
- **输出类型**: FP16(生产默认)。
- **基线性能**: `./profile_lds 3 20` ≈ **1796 TFLOPs**(0.612 ms, ~19.5% of 9227 峰值)。

## 采集参数 (gpu-profile skill, full ATT)

```
export ROCM_TOOLS_LIB=/home/AMD/zhewan/rocm-tools/opt/rocm-7.0.2.1/lib
rocprofv3 --att --att-target-cu 4 --att-shader-engine-mask 0xff --att-simd-select 0xf \
  --att-buffer-size 0x20000000 --att-activity 8 --att-library-path $ROCM_TOOLS_LIB \
  --kernel-include-regex lds_gemm_db -d /tmp/att_buf -- ./profile_lds 0 1
```

## 文件清单

| 文件 | 说明 |
|---|---|
| `lds_gemm_db_8192_rcv_trace.tar.gz` | **RCV 主包** — 解压得 `ui_output_agent_10217_dispatch_1/`,RCV 直接打开此目录 |
| `lds_gemm_db_8192_raw.asm` | 从 code.json 提取的 gfx950 ISA(2151 instr;160 MFMA / **60 buffer_load_dwordx4** / 92 ds_read_b128 / 9 vmcnt(0) / **仅 1 s_nop** / 768 v_accvgpr),每行尾带 hit/lat/stall/idle |

包内 `ui_output_agent_10217_dispatch_1/`:code.json(反汇编+逐指令 stall/idle/hit)· occupancy.json · 128 个 wave 逐周期轨迹 · se0_perfcounter.json。

## 怎么用 RCV 打开

1. `tar xzf lds_gemm_db_8192_rcv_trace.tar.gz`
2. RCV → Open → 选 `ui_output_agent_10217_dispatch_1/` 目录
3. 关注:**Wave States**(stall/wait)· **Instruction/ISA**(热点指令)· **Occupancy**。

## 本次 stall 热点(sum over sampled waves)

按 opcode 类别(占总 stall %):
- **s_waitcnt 32.9%** — 主要 `lgkmcnt(3)`,等 LDS `ds_read` 返回(occ1 LDS-读延迟全暴露,~250cyc/hit)。
- **buffer_load 26.8%** — 深 K prefetch 的发射 stall(occ1 单 wave 发不出去;与旧 global_load_lds 的 26.1% 同量级)。
- **v_mfma_scale 19.9%** — MFMA 本身,占空比 ~16-20%。
- **s_barrier 7.4%**(仅 5 条)— RDB 后每 2 tile 2 RAW barrier;两条最热 ~460/438 cyc/hit。
- global_store 5.9% / global_load(scale)4.7% / ds_read_b128 1.5%。
- **v_accvgpr_* 768 条 = 0.0% stall** —— acc 初始化 + epilogue(F16 转换强制 AccVGPR→ArchVGPR,ISA:AccVGPR 是 matrix-core 专属 VALU 读不了)**完全不是热点**,一次性被 K 摊薄。

## 结论(与 review 一致)

**latency-bound,occ1**:lgkmcnt(LDS 读)+ load 发射 stall ≈ 60% 总 stall,皆 occ1 latency-hiding 失败;
MFMA 占空比仅 ~20%。换 buffer_load_lds 不改变这个画面(latency-bound),只是 asm 更整洁 + 地址 VALU 减半(+0.8%)。
真正的墙是 **occupancy**(occ1 由 16-acc=256 AccVGPR 强制,gfx950 结构性堵死)。详见 HANDOFF Step 29-30 与
memory `mxfp6_buffer_vs_global_lds` / `mxfp6_lds_occ1_latency_bound`。
