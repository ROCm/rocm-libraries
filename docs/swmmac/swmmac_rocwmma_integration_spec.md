# SWMMAC rocWMMA Core Integration — AMD Vendor-Level Specification

**日期**: 2026-05-15  
**作者**: Architecture Analysis  
**目标**: 将 SWMMAC 以 amdgcn_mfma 模式集成到 rocWMMA 核心模板体系

---

## 一、12-真值逻辑分析

### T1: SWMMAC 是 wave-level 指令
```
TRUE. 32 个线程协作执行一条 v_swmmac。
与 MFMA 同属 wave-level 后端, 与 WMMA (per-thread) 不同。
```

### T2: rocWMMA 的 Mma<> 驱动器支持 wave-level 后端
```
TRUE. amdgcn_mfma 已经是 wave-level 后端。
Mma<> 通过 BlockM×BlockN×BlockK 分解片段, 对每个块调用 exec()。
```

### T3: PackTraits 必须满足 PackRatio × sizeof(UnpackedT) == sizeof(PackedT)
```
TRUE. 这是 rocWMMA 的核心不变量。
int4_t: sizeof=1, 需要 PackRatio=4 (4×1=4=int32_t).
      逻辑上应是 PackRatio=8 (8个INT4/i32), 但 8×1=8≠4.
      物理解决方案: PackRatio=4, 每个i32只用低4位, 高4位填0.
      硬件SWMMAC格式: PackRatio=8, nibble-packed.
      需要格式转换层。
```

### T4: MFMA 使用 PackRatio 匹配寄存器宽度
```
TRUE. MFMA int8_t: PackRatio=4, BlockM=16, BlockK=32
      A per thread = 16×32/32/4 = 4 i32 = VRegI32x4 ✓
      B per thread = 16×32/32/4 = 4 i32 = VRegI32x4 ✓
```

### T5: SWMMAC 的 A/B 寄存器宽度不对称
```
TRUE. SWMMAC INT4: A=2×i32, B=4×i32.
      这是 rocWMMA fragment 模型的核心冲突:
        A-per-thread ≠ B-per-thread, 但 IO 公式假设对称.
```

### T6: 解决不对称: 分别定义 PackRatio_A 和 PackRatio_B
```
FALSE. rocWMMA 的 PackTraits 是类型级别的, 不是上下文级别.
      同一个 int4_t 不能同时有 PackRatio=4 (for A) 和 PackRatio=8 (for B).
```

### T7: 解决不对称: 使用不同的 DataT 给 A 和 B
```
TRUE (partial). fragment<matrix_a, ..., int32_t> 和 fragment<matrix_b, ..., int32_t>
      可以有不同的 PackTraits, 因为 DataT 不同.
      但如果都用 int32_t, 则 PackRatio 相同.
```

### T8: 使用 int32_t 作为片段数据类型, PackRatio=1
```
TRUE (实用方案). 每个 fragment 元素 = 1 i32 = 8 nibble-packed INT4.
      PackRatio=1: sizeof(int32_t)×1 = sizeof(int32_t) ✓
      SWMMAC A fragment: 2 elements (2 i32) ✓
      SWMMAC B fragment: 4 elements (4 i32) ✓
      SWMMAC accum:    8 elements (8 i32) ✓
      
      需要自定义 IOConfig 来映射 BlockM×BlockK → per-thread elements.
```

### T9: IOConfig 可以自定义 per-thread 元素计数
```
TRUE (需要修改). 当前 IOConfig 从 BlockM×BlockK/WaveSize 推导.
      对于 SWMMAC, 需要覆盖此推导以匹配实际寄存器宽度.
      这是核心模板改动点.
```

### T10: 格式转换层: 片段存储格式 ↔ SWMMAC 硬件格式
```
FALSE (不需要). 如果 PackRatio=1 且 DataT=int32_t,
      片段存储 = 1 i32/元素, 8 INT4 nibble/i32.
      SWMMAC 寄存器 = 同格式.
      不需要转换.
```

### T11: 稀疏参数 sparse_idx 的接口兼容性
```
TRUE (需要扩展). MFMA 的 exec() 无 sparse_idx 参数.
      Mma<> 驱动器调用 exec(ARegsT, BRegsT, CRegsT) → DRegsT.
      SWMMAC 需要 exec(ARegsT, BRegsT, CRegsT, sparse_idx) → DRegsT.
      方案: 模板参数化 sparse_idx, 或默认参数 = 0.
```

### T12: BlockK 分解: FragK=64 → BlockK=16 × 4 iterations
```
TRUE. SWMMAC 硬件 BlockK=64 (INT4).
      对于 FragK=64: 1 次 SwmmacInt4::exec() 完成全部K.
      Mma<> 的 BlocksK = FragK/BlockK = 64/64 = 1 (1次迭代).
      PackRatio=1, BlockK=64:
        A per thread = 16×64/32/1 = 32 i32 ≠ 2 (硬件寄存器) ❌
      
      需要虚化 BlockK: 让 Mma<> 认为 BlockK=16, 硬件实际用 BlockK=64.
      4 次虚迭代 → 每次传相同的 A/B 寄存器, 硬件一次完成.
      但 Mma<> 会修改片段偏移, 导致数据错位.
      
      实际方案: 自定义 Mma 特化, 跳过 BlockK 迭代.
```

---

## 二、解决方案: MFMA-pattern SWMMAC 模板集成

### 方案A: 自定义 IOShape (最小侵入)

在 IOConfig 中添加 SWMMAC 特化, 覆盖 per-thread 元素计算:

```cpp
// io_config.hpp 中添加
template <typename MatrixT, uint32_t BlockM, uint32_t BlockK, typename DataT>
struct SWMMACIOShape {
    // 覆盖: 不使用 BlockM×BlockK/WaveSize 公式
    // 直接返回 SWMMAC 硬件寄存器宽度
    static constexpr uint32_t A_ElementsPerThread = 
        (BlockM == 16 && BlockK == 64) ? 2 : BlockM * BlockK / 32;
    static constexpr uint32_t B_ElementsPerThread = 
        (BlockM == 16 && BlockK == 64) ? 4 : BlockM * BlockK / 32;
    static constexpr uint32_t C_ElementsPerThread = 
        (BlockM == 16 && BlockN == 16) ? 8 : BlockM * BlockN / 32;
};
```

### 方案B: 虚 BlockK 迭代 (MFMA 兼容模式)

MFMA 已经使用此模式: BlockK=16 通过 `concat(regsA, ARegsT{0})` 填充到 BlockK=32.

SWMMAC 等效: 定义 BlockK=16 的虚指令, 填充到 BlockK=64:

```cpp
// SwmmacInt4<BlockK=16>: 虚指令, 填充到 BlockK=64
template <>
struct amdgcn_swmmac<int4_t, int4_t, int32_t, 16u, 16u, 16u> {
    using ARegsT = SwmmacARegsT;  // <2×i32> (硬件原生)
    using BRegsT = SwmmacBRegsT;  // <4×i32> (硬件原生)
    using CRegsT = SwmmacAccumT;
    using DRegsT = SwmmacAccumT;
    
    exec(ARegsT a, BRegsT b, CRegsT c) {
        return SwmmacInt4<true,true,true>::exec(a, b, c, 0);
    }
};
```

然后 Mma<> 驱动会迭代 BlocksK = FragK/16 次, 但每次迭代的寄存器内容相同.
这导致重复计算 — 每次 Mma<> 迭代都执行相同的 SWMMAC.

### 方案C: 自定义 Mma 特化 (推荐)

为 SWMMAC 创建专用的 Mma 特化, 跳过 BlockK 迭代:

```cpp
// 在 mma_impl.hpp 中添加
template <uint32_t FragM, uint32_t FragN, uint32_t FragK,
          bool ASign, bool BSign, bool CSign>
struct Mma<FragM, FragN, FragK, 
           SwmmacInt4<ASign, BSign, CSign>,
           MmaAccumPolicy::ROW_MAJOR>
{
    // 不迭代 BlockK — SWMMAC 一次性完成所有 K
    template <typename VecTA, typename VecTB, typename VecTC>
    ROCWMMA_DEVICE static inline decltype(auto) 
    exec(VecTA&& a, VecTB&& b, VecTC& accum) {
        // 直接从片段存储提取 SWMMAC 寄存器
        auto ra = reinterpret_cast<SwmmacARegsT const&>(a);
        auto rb = reinterpret_cast<SwmmacBRegsT const&>(b);
        auto rc = reinterpret_cast<SwmmacAccumT const&>(accum);
        SwmmacAccumT rd = SwmmacI4::exec(ra, rb, rc, 0);
        accum = reinterpret_cast<VecTC&>(rd);
        return accum;
    }
};
```

---

## 三、原子任务分解

### Phase 1: PackTraits + IO 基础 (4h)

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 1.1 | `pack_util_impl.hpp` | 添加 `PackTraits<int4_t>` | PackRatio=4, PackedT=int32_t (4 INT4 值用 1 i32, 高 4 位填 0) |
| 1.2 | `types.hpp` | 添加 `using int4_t = ...` | 从 rocwmma_int4.hpp 提取到核心类型 |
| 1.3 | `io_shape.hpp` | 添加 SWMMAC IOShape 特化 | 覆盖不对称的 per-thread 元素计数 |
| 1.4 | `io_config.hpp` | 添加 SWMMAC IOConfig 条目 | 绑定 IOShape 到 swmmac 后端 |
| 1.5 | `pack_util_impl.hpp` | 添加 nibble-pack 转换层 | fragment 格式 (4值/i32) ↔ SWMMAC 格式 (8值/i32) |

### Phase 2: 后端注册 (3h)

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 2.1 | `swmmac_impl.hpp` | 新建, 仿照 mfma_impl.hpp | `amdgcn_swmmac` 模板, 按 BlockK 特化 |
| 2.2 | `swmmac.hpp` | 新建, 仿照 mfma.hpp | Swmmac_impl + SwmmacSelector + Swmmac<FragM,FragN,FragK> |
| 2.3 | `mma_traits_impl.hpp` | 添加 `swmmac_traits` | MmaTraits 特化, 注册到 is_swmmac |
| 2.4 | `mma_selector.hpp` | 添加 Swmmac 条目 | 自动选择最佳 BlockK |

### Phase 3: Mma 驱动器适配 (5h)

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 3.1 | `mma_impl.hpp` | 添加 Swmmac Mma 特化 | BlockK 迭代跳过 (SWMMAC 一次完成) |
| 3.2 | `mma_impl.hpp` | sparse_idx 传递 | 从模板参数或默认值传递到 exec() |
| 3.3 | `io_bearer_impl.hpp` | SWMMAC 数据加载路径 | fragment ↔ SWMMAC 寄存器格式 |
| 3.4 | `fragment_traits_impl.hpp` | SWMMAC 片段 Traits | VGPR 计数, 打包尺寸 |

### Phase 4: 公共 API + 测试 (4h)

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 4.1 | `rocwmma.hpp` | 添加 `#include "internal/swmmac.hpp"` | 公开 SWMMAC 后端 |
| 4.2 | `rocwmma_impl.hpp` | 添加 swmmac include | 实现层集成 |
| 4.3 | test | 新建 SWMMAC 测试 | `fragment + load_matrix_sync + mma_sync + store_matrix_sync` |
| 4.4 | test | 兼容性测试 | 与现有 WMMA/MFMA 测试不冲突 |

### Phase 5: gfx11 fallback 集成 (2h)

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 5.1 | `swmmac_impl.hpp` | gfx11 WMMA 退化路径 | `enable_gfx11_t` → WMMA 替代 SWMMAC |
| 5.2 | `mma_selector.hpp` | 架构自动调度 | gfx12→SWMMAC, gfx11→WMMA |

---

## 四、关键不变量

```
1. PackRatio(int4_t) = 4  (sizeof(int4_t)=1, 4×1=4=int32_t)
2. 片段存储: 4 INT4 值/i32 (高4位填0) — 与 rocWMMA IO 系统兼容
3. SWMMAC 硬件: 8 INT4 值/i32 (nibble-packed) — 后端内部转换
4. A 寄存器: 2 i32/线程 (16 INT4, 片段中=4 i32, 需要合并)
5. B 寄存器: 4 i32/线程 (32 INT4, 片段中=8 i32, 需要合并)
6. C/D 寄存器: 8 i32/线程 (片段中=8 i32, 直接对应)
7. BlockK=64 (硬件原生), FragK 须为 64 倍数
8. 不支持 BlockK 迭代 (硬件一次完成 K=64)
```

---

## 五、格式转换数学

```
片段格式 (PackRatio=4):        SWMMAC 硬件格式 (PackRatio=8):
  i32[0] = [v3,v2,v1,v0,0,0,0,0]    i32[0] = [v7,v6,v5,v4,v3,v2,v1,v0]
  i32[1] = [v7,v6,v5,v4,0,0,0,0]    i32[1] = [f,e,d,c,b,a,9,8]
  i32[2] = [b,a,9,8,0,0,0,0]
  i32[3] = [f,e,d,c,0,0,0,0]

转换 (A 寄存器, 16 INT4 值):
  // 硬件期望 A[0] = v[7:0], A[1] = v[15:8]
  SwmmacARegsT hw_A;
  hw_A[0] = (frag_A[0] & 0x0F0F0F0F) | ((frag_A[2] & 0x0F0F0F0F) << 4);
  hw_A[1] = (frag_A[1] & 0x0F0F0F0F) | ((frag_A[3] & 0x0F0F0F0F) << 4);

转换 (B 寄存器, 32 INT4 值):
  类似合并

此转换在 SwmmacInt4::exec() 内部完成, 对 Mma<> 驱动器透明.
```

---

## 六、总工作量: ~18 小时 (约 3 天)

```
Phase 1: PackTraits + IO   ████████  4h
Phase 2: 后端注册          ██████    3h
Phase 3: Mma 驱动器适配    ██████████ 5h
Phase 4: 公共 API + 测试   ████████  4h
Phase 5: gfx11 fallback    ████      2h
                           ─────────
                           18h (~3d)
```

> **2026-05-18 更新**: RDNA4 HWXDL Silent Drop 发现。所有 kernel 已升级为 wave 级
> `__builtin_amdgcn_readfirstlane` 任务领取 (Lane 0 代理 + 全波广播, EXEC=0xFFFFFFFF)。
> 反例复现与微架构报告: `/data/rtl-sdr/swmmac/active/silent_drop/`
