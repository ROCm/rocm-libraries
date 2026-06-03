# ROCm LLVM 版本说明 — 优化策略与架构差异

## 背景

AMD ROCm 7.13 捆绑的 LLVM 23（`/opt/rocm/llvm`）与上游 LLVM 23
（`https://github.com/llvm/llvm-project`）**不是同一个代码基底**。

### ROCm LLVM 23 的真实构成

```
ROCm LLVM "23" = LLVM 22 稳定分支 + 紧急指令集补丁（SWMMAC/WMMA/gfx1250）
                + 版本号自动变更为 23

上游 LLVM 23   = LLVM 22 → 23 正常演进（TargetParser 拆分、SISchedule 重写、
                AMDGPUTargetParser 独立、10000+ 文件差异）
```

### 为什么会这样

AMD 的 GPU 指令集自 2013 年 GCN 以来保持稳定约 10 年。
2025 年 RDNA4 (gfx1200) 和 MI350X (gfx1250) 引入了 SWMMAC 全精度族、
WMMA 矩阵指令、2:4 结构化稀疏——这些是指令集层面的断崖式更新。

LLVM 22 后端不识别这些新指令。AMD 不能等上游 LLVM 23 正式发布
（商业交付如 MI350X 服务器计算卡不能依赖开发分支），
因此在 LLVM 22 稳定分支上紧急打补丁，版本号自动变更为 23。

### 差异规模

| 组件 | ROCm LLVM 23 | 上游 LLVM 23 |
|------|-------------|-------------|
| TargetParser | 单文件混合 (AMDGPU 代码在 TargetParser.cpp 中) | 已拆分为 AMDGPUTargetParser.* |
| SISchedule.td | GFX12SpeedModel (LLVM 22 基线) | 重构的调度模型 |
| SWMMAC 支持 | 指令编码可用，调度模型不完整 | 完整支持 |
| 文件差异 | — | ~10000 文件 |

## 我们的优化策略

我们的优化基于**上游 LLVM 23**（`/data/work/compiler/llvm/llvm-gpu` 仓库，
分支 `llvm-23`），包括：

| 优化 | 文件 | 依赖 |
|------|------|------|
| WMMA256bInsts + Wave32 | AMDGPU.td | 上游 LLVM 23 |
| SISchedule GFX12 WMMA overrides | SISchedule.td | 上游 LLVM 23 |
| TargetParser gfx1200 WMMA 传播 | AMDGPUTargetParser.cpp | 上游 LLVM 23 (已拆分) |
| Virtual FP4/MXFP4 支持 | lib/Support + AMDGPU 后端 | 上游 LLVM 23 |

这些优化**不能直接移植到 ROCm 7.13 的 LLVM**，因为：
1. ROCm LLVM 23 是 LLVM 22 打补丁，TargetParser 未拆分
2. SISchedule.td 结构不同
3. 移植工程量 = 在别人的技术债上重建优化链
4. ROCm 编译器团队未来会正式升级到上游 LLVM 23 基底，届时可自然合并

## rocm-libraries 提交策略

我们在 `rocm-libraries` 仓库只提交**库层优化**：

| 模块 | 内容 | 依赖编译器版本 |
|------|------|-------------|
| rocWMMA | SWMMAC 8 后端 + INT4 + 16chain | ROCm LLVM 23 或上游 LLVM 23 |
| rocBLAS | swmmac kernel + StaggeredPipeline | ROCm LLVM 23 或上游 LLVM 23 |
| docs | 物理测量、微架构发现、优化理论 | 无 |

**不在 rocm-libraries 提交编译器层优化**（AMDGPU.td、SISchedule.td、
TargetParser、MXFP4/Virtual FP4）。这些保留在 `llvm-gpu` 仓库，
等 AMD ROCm 编译器团队正式升级 LLVM 基底后提交。

## Tensile YAML 配置说明

`/data/ROCm/rocBLAS` 中有 gfx1200 的 Tensile YAML 配置文件（暂存未提交）。
这些配置是为 ROCm LLVM 23（LLVM 22 补丁版）的后端生成的 GEMM kernel 参数。
在上游 LLVM 23 下，SWMMAC kernel 通过 rocBLAS 直接路由（`rocblas_swmmac.cpp`），
不走 Tensile 代码生成路径，因此不需要这些 YAML 配置。

## 总结

- **rocWMMA + rocBLAS kernel**: 库层代码，与 LLVM 版本无关，已提交到 rocm-libraries
- **LLVM 编译器补丁**: 编译器层代码，保留在 llvm-gpu 仓库，待 ROCm 升级基底后合并
- **Tensile YAML**: ROCm 特定路径的配置，不适用于上游 LLVM 23，不提交
