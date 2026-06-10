# 报告：`s_waitcnt lgkmcnt` 真在只等它该等的数据吗？

**对象**：`lds_gemm_db<256,256,192,2,2,occ1,swz0,DB,__half>` @8192³
**证据**：源码 `mxfp6_lds.hpp` 的 `compute()` + RCV 实测汇编 `lds_gemm_db_8192_raw.asm` 主循环（行 512–1026）
**日期**：2026-06-09

---

## 裁决（TL;DR）

**是 —— 在「单个 in-order L/G/K/M 计数器」所能表达的精度内，每个 `lgkmcnt(N)` 只等它该等的数据，不漏等、不错等、不过等。** 唯一的不精确来自 `ds_read2` 把两个 operand 的 8B 尾融合成一个计数事件，但这是**省指令的优化、且无害**。主循环里那些巨大的 stall（620k–694k cyc）**不是**计数错误，而是 occ1 下真实的 LDS-read 延迟暴露。

---

## 0. 前提：`lgkmcnt` 的硬件语义

`lgkmcnt` 是**单一**计数器，合计四类在飞操作：**L**DS + **G**DS + **K**onstant(`s_load`) + **M**essage。
- `s_waitcnt lgkmcnt(N)` = 阻塞到「未决的 L/G/K/M 操作 ≤ N」。
- **同类型 in-order 完成**：同一 wave 的 DS read 按发射顺序返回数据。
- 推论：发射了 K 个 DS read 后，`lgkmcnt(N)` 保证「最老的 K−N 个已完成」，因为它们按序回。

这意味着两件事会破坏「只等该等的」：
1. **污染**：若计数器里混入 `s_load`/message，DS 的进度被无关操作干扰。
2. **耦合**：单计数器无法说「只要寄存器 X」，只能说「排空到剩 N 个」。

下面逐一用证据检验。

---

## 1. 源码意图（compute()）

`mxfp6_lds.hpp:232-275`：b 是一条贯穿整个 tile 的 just-in-time 流，只有 `b_cur`+`b_next` 活着；**下一个 b 的 `ds_read` 在本 ni 的 MFMA 四连发之前发出**，让 LDS-read 延迟与 4 条 MFMA 重叠。a[mi] 在每个 sub 开头一次性读齐（被 4 个 ni 复用）。

设计期望的等待形状：
```
读 a[0..3]（被所有 ni 复用，最老）
读 b(ni0); 读 b(ni1); 读 b(ni2); 读 b(ni3)（按消费顺序）
lgkmcnt(3) → 放行 ni0 四连发    （只剩 b1,b2,b3 在飞）
lgkmcnt(2) → 放行 ni1 四连发
lgkmcnt(1) → 放行 ni2 四连发
lgkmcnt(0) → 放行 ni3 四连发
```

---

## 2. 汇编证据

### 2.1 计数器是「纯 DS」—— 零污染 ✓
主循环（512–1026）静态扫描：
```
s_waitcnt vmcnt        : 0 条
s_load / s_buffer_load : 0 条
s_sendmsg / ds_gws     : 0 条
ds_write               : 0 条
```
→ 稳态热循环里 `lgkmcnt` **只数 `ds_read`**。没有 `s_load`/message 把计数器搅浑（kernarg 的 `s_load` 在入口，紧跟一条 `lgkmcnt(0)` 排空，行 12）。**前提 1（污染）排除。**

> 旁注：`buffer_load_lds`（HBM→LDS）走 **vmcnt**，但它是 M0-隐式 inline asm，编译器看不见 load→ds_read 依赖，故循环里 `vmcnt` = 0 条。RAW 不靠计数器、靠深-K 时间余量兜底（见 `mxfp6_lds.hpp:278` 注释，已 bit-exact 验证）。这与本报告的 `lgkmcnt` 问题正交。

### 2.2 阶梯与深度
```
lgkmcnt 取值：0(×6)  1(×12)  2(×18)  3(×12)
两次 wait 之间最多发射 DS：4
```
计数器恒在 0–3，远低于硬件上限，**无饱和、无精度丢失**。深度 ~4 与「保持 ~3 个 b-read 在飞」的流水设计吻合。

### 2.3 一个四连发块的依赖追踪（asm 行号见原始文件）

被喂的 MFMA 四连发块（一个 sub）：
| ni | MFMA 的 b 操作数 (src0) | 由谁放行 |
|---|---|---|
| 0 | `v[32:37]` | `lgkmcnt(3)` |
| 1 | `v[26:31]` | `lgkmcnt(2)` |
| 2 | `v[126:131]` | `lgkmcnt(1)` |
| 3 | `v[120:125]` | `lgkmcnt(0)` |

喂它的 `ds_read`（**按 ni 消费顺序发射**）：
```
ds_read_b128 v[32:35], v118              ; b(ni0)
ds_read_b128 v[26:29], v118 offset:4608  ; b(ni1)
ds_read_b128 v[126:129],v118 offset:9216 ; b(ni2)
ds_read_b128 v[120:123],v118 offset:13824; b(ni3)
lgkmcnt(3) → 排空到剩 3 = {b1,b2,b3} 在飞，b0 已就位 → 放行 ni0
```
a 操作数 `v[2:7] v[8:13] v[14:19] v[20:25]` 在更早就已发射（行 119/120/125/126 等），**比 b-read 老**。

---

## 3. 三项判定

### 3.1 正确性：不漏等、不错等 ✓
编译器 `SIInsertWaitcnts` 对每个被消费的寄存器按其生产 `ds_read` 的「计分」插入覆盖所有消费者的最小 N。DS in-order ⇒ `lgkmcnt(3)` 放行 ni0 时，**所有比 b(ni0) 老的读（即 4 个 a）必然已完成**。所以一条 `lgkmcnt(3)` 同时担保了 `a[0..3] + b(ni0)` 全部就位 —— 该等的全等到了，bit-exact 验证佐证。

### 3.2 紧致性：阶梯是单计数器的最优解 ✓
这里依赖结构恰好「a 被所有 ni 复用（最老）、b 一个一个用」，而 a 又先于 b 读。于是「排空到剩 N 个 b 在飞」**同时**满足两件事：(i) a 全到、(ii) 当前 ni 的 b 到、(iii) 未来 3 个 b 继续在后台跑。**没有一拍是白等的** —— 单 in-order 计数器在这种结构下不是将就，而是理想匹配。

也不会「过等」：因为 in-order，任何还在飞的更老的读，本就是更新的读完成的前提；不存在「为一个用不到的老读而多等」。

### 3.3 唯一的结构性不精确：`ds_read2` 融合（无害）
6-VGPR operand = `ds_read_b128`(16B) + `ds_read_b64`(8B)。编译器把**相邻两个 operand 的 8B 尾**两两融合成一条 `ds_read2st64_b64`（如行 111 `v[4:7]`、行 4 `v[10:13] offset0:18 offset1:27`）。后果：两个 operand 的尾共享**一个**计数事件 —— 一个四连发理论上会被一条同时携带「下一个还用不到的 operand 尾」的融合读 gate 住。

但这**无害且划算**：
- 它把 DS op 数砍半（实测 92 b128 + 36 read2 + 20 b64，本应是 ~184 条），**计数器更浅、发射压力更低**；
- 两个尾在同一条指令里同时完成，被「提前捎带」就绪的那个 operand 几条 MFMA 后正好要用；
- 不影响正确性。

---

## 4. 大 stall ≠ 错等

| wait 位置 | 典型 stall (cyc) | 含义 |
|---|---|---|
| 每个 sub 头部 `lgkmcnt(3)` | **620k – 694k** | 真实 LDS-read 延迟暴露：occ1 下 wave 手里没别的活可发，只能干等首批读回来 |
| 四连发内 `lgkmcnt(2/1/0)` | 10k – 16k | 很小：下一个 b 早一个四连发发出，回来时基本已就位 → **流水在起作用** |

头部那一下不是计数器等错了东西，而是**第一笔 LDS 数据本来就要 ~880cyc，而 occ1 没有第二个 wave 来填这段空窗**。证据：同一条 `lgkmcnt` 指令，在四连发*内部*（数据已预取）stall 掉到 ~1/50。把它「修掉」的办法是 ILP/occupancy（更多在飞工作），**不是**改 waitcnt —— waitcnt 已经是最紧的。

---

## 5. 结论与可行项

**结论**：`lgkmcnt` 确实只等它该等的数据。计数器纯净（仅 DS）、阶梯紧致（单 in-order 计数器的理想形态）、无漏/错/过等；唯一的「两尾融合耦合」是省指令的优化且无害。主循环的大停顿是 occ1 的延迟暴露，与 `lgkmcnt` 的正确性/紧致性无关。

**可行项**（都不是改 waitcnt）：
1. 想压头部那 ~650k stall，唯一杠杆是制造更多 ILP/在飞工作来填空窗 —— 这正是「深-K + 大寄存器 tile + prefetch 提前」已在做的；进一步只能动 occupancy（occ2 已证伪净亏 -14%）或更深的跨-sub 预取。
2. 若哪天 K_TILE 缩到接近加载延迟，深-K 时间余量会失效 —— 那时才需要给 RAW 加显式 `wait_vmcnt(0)`（~-0.3%），但那是 vmcnt 的事，与本报告的 lgkmcnt 无关。

---
*RCV-instrumented · gfx950 CDNA4 · 源码与汇编逐条核对*
