# 任务：MXFP6 LDS GEMM —— occupancy-2 redesign（新文件实现）

> 给新 hip-kernel-team 的启动 handoff。可作 `/hip-kernel-team` 设置时的 Goal，或团队起来后作为给 team lead 的第一条指令。memory 已更新，团队能读到。
> 生成于 2026-06-05，基线 commit 5b1c3a53。**本任务专攻 occ2，不做 ping-pong**（ping-pong 是未验证的规定解法，先排除；occ2 的 overlap 先靠 occupancy 本身的硬件轮询拿）。

## 一句话目标
**在一个全新文件里**实现一个 occupancy-2 的 MXFP6 LDS GEMM kernel，把 occ2 做到足够好（五个轴联合优化，见下），目标是打穿 occ1 唯一没翻过的墙：buffer_load 发射背压（~9.3M stall / 占 ~30%，单 wave 发不出去）。**不要改生产 mxfp6_lds.hpp**（它已 commit 的 +5.86% 要保住）。

## 现状（已落 git + NFS + memory，先读 memory 再动手）
- 分支 `zhewan/ck/mxfp6-standalone`，HEAD=`5b1c3a53`。生产 kernel =
  `lds_gemm_db<256,256,192, 2,2, MIN_OCC=1, SWZ0, DB=true, __half>`（mxfp6_lds.hpp）
  256×256 tile, KT192 双缓冲(144KB LDS), 32×32×64 MFMA, 16 acc/wave(4×4 blk,2×2 wave), 256 AGPR, occ1。
  生产 SWZ0 @8192³ FP16 = **1869**（干净 5×median；真实 base 是 1765 非旧记的 1673）。
- 上一轮成果（纯指令重排，全三重验证）：ds_read 软流水(懒读b+跨sub,+4.11%) + scale load 前移(+1.7%)。
  这些是 occ1 范式的实用极限，**别再在 occ1 重排上花时间**。

## 为什么是 occ2（核心认知，memory 里有完整记录）
- gfx950 occ1 = 512 的 **arch VGPR + AGPR 合并池**（不是独立）。256 个 AGPR 累加器**同时钉死 occ=1 + 吃掉池子一半** —— 这是整个 kernel 的核心死结。
- 后果：单 wave 把 18 条 buffer_load 怼进内存系统、发射队列背压、**没有第二个 wave 在它卡住时顶上** → buffer_load 发射 stall 在 occ1 下结构性打不穿（occ1 内 fused-drip 交错已实测 spill159 撞墙）。
- occ2 对发射背压有**两条独立攻击线**（都要拿）：
  - **inter-wave（轮询）**：2 拨驻留 wave，硬件在一拨 stall(等 buffer_load)时切另一拨算 MFMA → 背压被自然 overlap。
  - **intra-wave（交错发射）**：occ2 砍半 acc 腾出寄存器后，可把 mfma 和 buffer_load 在指令流里**物理交替**——每条 buffer_load 之间的 mfma 执行期正好让内存队列 drain 出发射槽，下一条 load 少等。**这条 occ1 试过(fused-drip)但 spill159 撞墙——撞的是寄存器墙,不是 occupancy；occ2 寄存器减半就活了。这条甚至不强依赖第二拨 wave 真在算,只要寄存器够。**

## ⚠️ 诚实警告（这是个 bet，不是稳赢）
- **裸 occ2（只缩 tile、不管其它轴）历史上实测输给 occ1 大 tile**（缩 tile→算术密度降，没补回来）。本任务赌的是：**五个轴一起调到位**（尤其用矩形 tile 保住算术密度 + 多 wave 藏掉 buffer_load 背压），让 occ2 净赢。**阶段 1 的裸 occ2 baseline 就是 go/no-go 闸——若那时已掉太狠且看不到补回的路，就如实收手。**
- **不做 ping-pong**：显式 ping-pong 编排（setprio/condBarrier/sched_group）是未在我们 kernel 验证过的复杂手法，且 memory 记着 sched 控制只 advisory、barrier 数量不决定性能。**先看 occ2 本身的硬件轮询 overlap 能拿多少**；若确有明确剩余且论证清楚，再另开议题，不在本任务范围。
- AMD Confluence(MLSE 744193432) 报过 occ2 路线 660-730 tflops，但那是 **AMD 自己的 kernel + 他们的 ping-pong 调度**，不是我们的、没验证过——**当上限参照，不当目标**。

---

## 设计空间：五个轴联合优化（核心，researcher + implementer 一起权衡，不是孤立调）
occ2 的成败是这五个轴的**联合最优**，互相牵制，必须一起算：

**轴1 · 算术密度（tile 形状）**：MFMA数=面积(MPW×NPW)、load数=周长(MPW+NPW)，密度∝面积/周长。occ2 逼 tile 变小→密度降。要找"小到能 occ2、又不至于密度损失盖过 occ2 收益"的甜点——可能不是 128×128，而是**偏宽矩形（如 128×256，A 复用更多）**。和轴2 绑死。

**轴2 · 寄存器用量**：occ2 要求每 wave ≤256 combined（才能 2 拨塞进 512 池）。acc 数定 AGPR（8 acc=128 AGPR）。还要给 prefetch 状态留余量。**这是最硬的约束**，轴1/3/4/5 全受它制约。改完盯 `.vgpr_spill_count`。

**轴3 · mfma↔buffer_load 交错（occ2 解锁的核心收益，直攻那 9.3M 发射 stall，成败手）**：occ1 做不到、occ2 专门要拿的，**两条线都是第一类杠杆，独立评估**：
   - ① **inter-wave 轮询 overlap**：2 拨 wave，硬件在一拨等 buffer_load 时切另一拨算 MFMA。这是 occupancy 本分，先确认能拿多少。
   - ② **intra-wave 交错发射（物理交替 mfma/buffer_load 指令流）**：把 18 条 buffer_load 从"背靠背一坨"散开、每条之间塞 mfma → mfma 执行期让内存队列 drain 出发射槽,下一条 load 少等。**机制是发射成本散进 compute 窗口,不是靠第二拨 wave;occ1 的 fused-drip 正是这条,spill159 撞的是寄存器墙——occ2 砍半 acc 腾出寄存器就活。** 注意保持 mfma quartet 完整(上一轮证实拆 quartet 伤吞吐),交错放在 quartet 边界。
   - 两条可叠加,也可能其一就够;团队分别量化各自贡献。

**轴4 · 合适深的 K 窗口**：深 K 让"一次 load 的 MFMA 窗口 >> load 延迟(880cyc)"。occ2 tile 小→每 K-step 的 mfma 少→窗口浅→可能要**更深的 K** 补。好消息：小 tile 省 LDS（128×128 KT192 DB ≈72KB « 144KB），有空间做**更深 K 或多级缓冲(multistage，>2 buffer)**。和 LDS 容量 + 轴3 绑。

**轴5 · MFMA 指令本身**：当前 32×32×64（16 acc/wave）。**16×16×128**（4 acc/指令，输出更小）在 occ1 被实测证伪 8 次（FLOPs/指令半、操作数带宽/FLOP 翻倍=密度错）。**但 occ2 下绑定约束从"算术密度"变成"寄存器压力"**，更小输出的 MFMA 可能让 tile/acc 打包更容易塞进 occ2 预算。**指令 = researcher 必查的一个轴**：若从寄存器/调度角度论证 16×16×128 在 occ2 语境确有理（不是 occ1 那套），**可以试**，但带 occ1 已证伪的诚实先验、用 occ2 的新算账论证清楚再试，别盲试。其它 MFMA 变体一并评估。

## 执行计划：分阶段，每阶段 go/no-go，不许一把梭
**阶段 0（researcher，纸面，不写 kernel）**：五轴联合可行性评估 —— 给候选设计点（tile 形状 × acc 数 × MFMA 指令 × K 深）的寄存器/LDS 账 + 预估，读 memory + Confluence(744193432 当背景，不抄 ping-pong)。产出 1-2 个最优候选 + go/no-go。

**阶段 1（implementer，新文件 occ2 骨架）**：按选定候选实现到**新文件**（建议 `mxfp6_lds_occ2.hpp` + `test_lds_occ2.cpp`；不碰 mxfp6_lds.hpp/test_lds.cpp）。先求：correctness 全过 + LDS 装得下 + 编译确认真 occ2（VGPR 减半、occ=2）。预期 perf ~平/略降（裸 occ2 baseline）。**go/no-go 闸**。

**阶段 2（在 occ2 骨架上把五轴调到位）**：矩形 tile 保算术密度、更深 K 用掉省下的 LDS、轴3 的 buffer_load 交错（先吃 occupancy 轮询 overlap，再看 wave 内 drip）、必要时换 MFMA 指令。目标：occ2 净赢 occ1 的 1869。每步要能干净量出某一轴的增量。

## 方法论（铁律，违反则结论不可信）
1. **验收节奏（别每步都抓 trace，那很慢很浪费）**：
   - **每次改动的必跑闸（便宜，KEEP 前提）**：严谨 A/B（区间不重叠，见 (a)）+ ISA 自检 + correctness（全过，F32 err=0）。**counter 不在每次必跑里。**
   - **counter 深挖（只在三种情况触发，非每迭代）**：① 一个要 KEEP 的赢点 → 确认机制对了再落袋；② 任何意外（超预期增益 / 回归）→ 按 profiler protocol 自动调查；③ 写最终机制结论时。
   - **RCV(ATT) 与 PMC 按问题择一，不是两个都跑**：要逐 opcode/逐 wave 的 stall 拆解（总 stall、lgkmcnt 分桶、buffer_load stall）→ **ATT/RCV**（重，是"总 stall 自洽"检查的来源）；只要快速看聚合比值（L2 命中、SQ_WAIT、MFMA duty）→ **PMC**（轻）。
   - 参考上一轮实际节奏：每个实验都跑 A/B+ISA+correctness；counter(ATT) 只在确认赢点机制 / 查 wash / 查意外增益时跑过几次，不是每迭代。

   两条判定标准的精确定义（别用松标准 KEEP 错东西）：

   **(a) A/B "区间不重叠才算数"**（A=已 KEEP 的基准 / 旧版本，B=本次新改动）：
   - GPU 跑分有 run-to-run 噪声（本 kernel ±0.5~0.9%，个别 driver bimodal ±9%）。光看"中位数 B>A"可能纯噪声。
   - 做法：各跑 5 次（丢首跑 warmup，取 run2-5），分别记 A 的取值**范围(min–max)** 和 B 的范围。
   - **判定：B 的范围和 A 的范围不重叠（B 最差一跑都 > A 最好一跑）→ 真加速，算数；范围重叠 → 当噪声，不算赢（wash），不 KEEP。**
   - 实例（上一轮）：某改动 ★1 范围 1796-1821 / 改动 #4 范围 1830-1845，min(1830)>max(1821) 不重叠 → 真 +1.5% KEEP；而 a-operand JIT 实验全配置 ±0.9% 内、区间重叠、无一致方向 → 判 WASH 回退。这条规则正是挡掉假赢的闸。

   **(b) counter "自洽" = counter 机制故事必须和 perf Δ 对得上；信总 stall 不信单桶**：
   - 对不上 = 要么测错、要么机制理解错，必须先和解再信结果。
   - ⚠️ **别 cherry-pick 单个 counter 桶**——上一轮 lgkmcnt 这个桶骗了两次：① ds_read 软流水后 lgkmcnt 总量反而 +9%（与"降 lgkmcnt"预期矛盾），真相是 lgkmcnt(3) 砍 72% 但被重分配到别的桶，只有**总 stall −7.27%** 才和 perf +4.11% 对上；② a-JIT 后 lgkmcnt −52% 看着超棒但 perf 是 wash，因为 stall 只是搬家到 ds_read(+830%)，**总 stall +0.46%** 才和 wash 对上。
   - **判定：以"总 stall Δ 是否和 perf Δ 同向同量级"为自洽锚点，不被单个桶的涨跌带偏。**

2. **"跑 N 次没错 ≠ 正确"**：WAR/RAW race 只在部分填充方阵(N≤M, WG 48-480, K≥3072)暴露，8192³ lockstep 会掩盖。正确性靠结构性保证不赌时序。
3. 双缓冲/多级缓冲的 buffer **必须编译期选**（named current/next + unrolled），动态寄存器数组下标必 spill（历史上多次"预取证伪"的真因）。
4. 改完盯 `.vgpr_spill_count`（编译器对扰动极敏感，易悬崖式 spill）。
5. **新文件实现，生产 mxfp6_lds.hpp 不动**。失败干净回退别污染生产。NFS 同步只在 KEEP 后由 lead 统一做（中间实验别往 NFS 推，曾覆盖过备份）。memory 新建文件在本环境会被清，要沉淀就折进已有文件（如 `mxfp6_lds_occ1_latency_bound.md`）。

## 已证伪的死路（memory 记录，除非有 occ2 语境的新论证否则别试）
bank-conflict 修复（red herring，1.16%）/ A-LDS+B-direct hybrid（编译器 vmcnt(0) drain）/ K-tail（慢于 padding）/ sched_barrier 当热循环调度控制（advisory）。
**注**：16×16×128 和 buffer_load 交错此前在 **occ1** 证伪/撞墙，但二者正是 occ2 要重新评估的轴（见轴5/轴3）——occ2 语境下允许带新论证重试。

## 关键文件 + 参考
- 现生产 kernel（**只读参考，不改**）: `mxfp6_gemm/mxfp6_lds.hpp`、`mxfp6_asm_utils.hpp`（inline asm 封装可复用）
- 现 driver（参考）: `test_lds.cpp`、`profile_lds.cpp`（单dispatch ATT/RCV）、`bench_ck_match.cpp`
- 新建（本任务产物）: `mxfp6_lds_occ2.hpp` + `test_lds_occ2.cpp`
- memory（先读）: `mxfp6_lds_occ1_latency_bound`（寄存器池真相+occ2框架+CK对比基线，本任务最重要的一条）、`mxfp6_lds_paradigm`（深K范式）、`mxfp6_wave_shape`（tile形状/算术密度）、`mxfp6_war_barrier_cost`（race测法）、`mxfp6_lds_raw_sync`（同步真相）、`reference_gfx950_mfma_f8f6f4_cycles`（MFMA cycle，轴5用）、`mxfp6_16x16x128_builtin`（16×16×128 builtin 用法+occ1证伪记录，轴5用）
- Confluence（MCP，researcher 读，当背景不抄 ping-pong）: 744193432（occ2 调度分析）、1652698858（MX flatmm）、1661646564（CK Cliff Notes）
- 硬件: `docs/amd-instinct-cdna4-instruction-set-architecture.pdf`（父目录）

## 第一步
spawn researcher 做阶段 0 的五轴联合可行性评估，profiler 待命建 occ2 baseline 量化，implementer 待命。评估回来后由 lead + 用户选定候选设计点，再进阶段 1。**不要直接写 kernel。**
