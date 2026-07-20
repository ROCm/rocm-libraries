# Pseudo Cluster Barrier — 實作計畫 / 記錄

## 背景

先前的 `InsertClusterBarrierPass` 直接插入實體的 `s_barrier_signal/wait -3`
與 branch，會發生 **SCC overwritten** 的問題。該 pass 已於
commit `5e4dea4c36`（`refactor(stinkytofu): remove InsertClusterBarrierPass to
prepare reimplementation`）整組移除。

本計畫以「**自訂 pseudo instruction + collapse/expand**」的方式重寫，語意上參考
既有的 `EXEC_GROUP`（`ExecMaskGrouping.cpp`）：在進 DAG scheduler 前插入一個
不產生組語的 placeholder，讓 scheduler 依 operand 依賴一起排程，出 DAG 後再由
另一個 pass 展開成實際指令序列。

## 核心機制

- **依賴表達**：DAG scheduler 依每道指令的 `getSrcRegs()` / `getDestRegs()`
  建立 RAW/WAW/WAR edge（`StinkyDAGSchedulerPass.cpp`）。因此讓 placeholder
  跟某道 anchor 指令有依賴、被一起排程的方法，是給它掛上對應的
  src/dest operand（或 memtoken）。
- **不要 side effect**：`hasSideEffect(inst)` 為真的指令會被 DAG 當成 region
  邊界（不自由排程）。placeholder 因此**不設** `IF_HasSideEffect`，比照
  `EXEC_GROUP`。
- **emit 安全網**：emitter 有 `if (isPseudoInst(&inst)) return;`，故 placeholder
  併入 `isPseudoInst()` 後即使漏展開也不會 emit 出組語。DAG 節點建立不參考
  `isPseudoInst`，故併入不影響排程。

## 實作步驟

### Step A — 定義自訂指令（opcode + modifier）
- **A-1** `tools/tablegen/GenInstructions.cpp`：unified 清單新增
  `PSEUDO_CLUSTER_BARRIER`（與 `FENCE`/`EXEC_GROUP` 同區，`INVALID` 之前）。
- **A-2** `include/stinkytofu/ir/asm/StinkyModifiers.hpp`：新增
  `Modifier::Type::PSEUDO_CLUSTER_BARRIER` 與 `PseudoClusterBarrierData`
  modifier（承載展開所需資料，例如 signal/wait 種類）。
- **A-3** `include/stinkytofu/ir/asm/StinkyAsmIR.hpp`：新增
  `createPseudoClusterBarrier(...)` builder + inline `HwInstDesc`（無 side
  effect）、`isPseudoClusterBarrier()` helper，並將其併入 `isPseudoInst()`。

### Step B — 插入 pass（DAG 前）
- 新 pass：找到 anchor 指令，插入 placeholder，掛上 src/dest operand（或
  memtoken）建立與 anchor 的依賴，並把展開資料寫入 `PseudoClusterBarrierData`。
- 放在 `createStinkyDAGSchedulerPass()` 之前、`createStinkyBuildImplicitDependencyPass()`
  之後（若採 memtoken 依賴）。

### Step C — DAG 排程
- 通常不需改 DAG：operand 依賴正確、`issueCycles`/`latencyCycles` 有設即可。

### Step D — 展開 pass（DAG 後）
- 新 pass：掃描 placeholder，依 modifier 展開成實際 `s_barrier_signal/wait -3`
  序列（比照 `expandExecMaskedGroups`）。放在 kernel scope、region adaptor 之後。

### Step E — emitter 安全網 + wiring
- emitter：placeholder 併入 `isPseudoInst()`（不 emit）。
- 兩個新 pass 的 `.hpp/.cpp` + `CMakeLists.txt` / `Gfx1250Backend.cpp` /
  `stinkytofu-opt.hpp` / `ApiExportTest.cpp` wiring。

## SIA / OptLevel gating 注意事項

- `ScheduleIterAlg == 4` → `_StinkyTofuOptLevel = 3` → `runScheduler=true` → 跑 DAG。
- `SIA=0/1/2/3` → `_StinkyTofuOptLevel = 0` → `runScheduler=false` → **不跑 DAG**。
- `runScheduler` 只 gate region block（含 DAG）與少數 kernel-scope pass；
  region adaptor 之後的 kernel-scope pass 無條件執行（O0 也跑）。

**建議擺法（同時支援 SIA=4 與 SIA0）**：插入 pass 放 kernel scope、region
adaptor 之前（無條件）；展開 pass 放 kernel scope 最後（無條件）。
- SIA=4：插入 → DAG 一起排 → 展開。
- SIA0：插入 → DAG 跳過（原位）→ 展開。
- 注意：SIA0 無 DAG，operand/memtoken 依賴不影響正確性，最終位置由插入點決定，
  需確保插入點在 SIA0 也正確。

## 命名決定

- 使用者指定假指令名 `psuedo_cluster_barrier`；因為它會成為 `enum GFX` 的
  C++ identifier 且需與 `FENCE`/`EXEC_GROUP` 同慣例，採用大寫
  `PSEUDO_CLUSTER_BARRIER`（builder：`createPseudoClusterBarrier`）。

## 進度

- [x] 移除舊 `InsertClusterBarrierPass`（commit `5e4dea4c36`）
- [x] Step A：定義 `PSEUDO_CLUSTER_BARRIER` opcode + modifier + builder
- [ ] Step B：插入 pass
- [ ] Step C：DAG 排程驗證
- [ ] Step D：展開 pass
- [ ] Step E：emitter 安全網 + wiring
