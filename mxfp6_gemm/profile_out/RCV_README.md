# RCV Trace — MXFP6 GEMM 生产 kernel (lds_gemm_db @ 8192³)

ATT trace,可直接用 **RCV (Radeon Compute Viewer)** 打开审阅。

## 抓的是什么

- **Kernel**: `lds_gemm_db<256, 256, 192, 2, 2, MIN_OCC=1, SWZ=16, DB=true, float>`
  —— 即 v18 dispatcher 在 8192³ 的实际选择 (`choose_tile`=TLDS, `choose_swz`=16)。
- **问题规模**: M=N=K=8192 (K padding 到 8256 = 43×192)，**单次 dispatch**。
- **配置**: 256×256 tile · KT192 深-K 双缓冲 · 4 wave (2×2) · 16 累加器/wave (occ1) · L2 swizzle。
- 驱动: `profile_lds.cpp`（`./profile_lds 0 1` = 一次 dispatch；`./profile_lds 3 10` = 计时基线 ≈1655 TFLOPs）。

## 采集参数 (gpu-profile skill, full ATT)

```
rocprofv3 --att --att-target-cu 4 --att-shader-engine-mask 0xff --att-simd-select 0xf \
  --att-buffer-size 0x20000000 --att-activity 8 --att-library-path $ROCM_TOOLS_LIB \
  --kernel-include-regex lds_gemm_db -d /tmp/att_lds -- ./profile_lds 0 1
```

## 文件清单

| 文件 | 说明 |
|---|---|
| `lds_gemm_db_8192_rcv_trace.tar.gz` | **RCV 主包** — 解压得 `ui_output_agent_*_dispatch_1/`，RCV 直接打开此目录 |
| `lds_gemm_db_8192_raw.asm` | 从 `code.json` 提取的 gfx950 ISA (1883 行；160 MFMA / 60 global_load_lds / 92 ds_read_b128 / 10 vmcnt(0)) |

包内 `ui_output_agent_45661_dispatch_1/`：
- `code.json` — 反汇编 + 每条指令的 stall/idle/hit (1883 指令，code 非 null ✓)
- `occupancy.json` — 占用率轨迹
- `se*_sm*_sl*_wv*.json` — 128 个 wave 的逐周期状态轨迹
- `se*_perfcounter.json` — 每 SE 性能计数

## 怎么用 RCV 打开

1. 解压: `tar xzf lds_gemm_db_8192_rcv_trace.tar.gz`
2. RCV → Open → 选 `ui_output_agent_45661_dispatch_1/` 目录
3. 关注视图: **Wave States**(看 stall/wait 占比) · **Instruction/ISA**(看哪条指令热) · **Occupancy**。

## 已知热点 (供对照)

- 主循环 `s_waitcnt vmcnt(0)` 在每个 ds_read 前 (10×) —— 编译器强插的 drain，是 occ1 延迟暴露的体现。
- MFMA 占空比 ~16%，整体延迟受限 (非算力/带宽受限)。
- 详见 HANDOFF Step 19/22 与 memory `mxfp6_v17_8k_profile` / `mxfp6_l2_sq_measurement`。
