# ReuseAcrossPersistent (RAP) v0 實作計畫

> 目標：讓 A / MXSA 的 K-tile 常駐 VGPR，跨 persistent loop iteration 重用，
> 減少 `tensor_load_to_lds` 與 `ds_load` 的資料搬移。
>
> v0 範圍：Equality tuning（特定 problem size）能成功產生 RAP kernel 即可。
> 只支援 `ceil(M/MT0)==1`、K 鎖定為特定值，A/MXSA 全部放進 VGPR，放不下就不產生 kernel。

---

## 進度

| 步驟 | 狀態 | 結果 |
|---|---|---|
| 0 上限實驗與 config 選定 | 完成 | MT64x512 天花板 5.7%、MT64x256 天花板 10.1%。兩個 config 並存：MT64x256 當開發／回歸（訊號乾淨），MT64x512 當出貨 |
| 1 參數接線 | 完成 | `RAP0` / `RAP1` 一律進 kernel 名稱、無重複 kernel |
| 2 predicate 與 reject | 完成 | 不新增調參參數：`_RAPNumResidentKTiles = PGR+1`，並發 `SizeEqual(K)=768` 與 `SizeLessThan(M)=MT0+1` 兩個 predicate |
| 3 常駐 VGPR 配置 | 完成 | ValuA 64→192、MXSA 4→12、跨 store 保留，PASSED 且時間不變 |
| 4a k-tile 常駐索引 | 完成 | 三個既有區段各自定址自己的 buffer，PASSED，指令數與時間都不變 |
| 4b-1 抽出 `_persistentComputeSection` | 完成 | 純重構，組語逐字相同 |
| 4b-2 iter0/iterN 剝離 | 完成 | 五個 bug 全修，預設開啟 |
| 5 iterN 丟掉 A/MXSA 的搬移 | 完成 | A 與 MX scale 都常駐，預設開啟 |
| 6 驗證與量測 | 完成 | 全量驗證（`NumElementsToValidate: -1`）PASSED、unit 6223 passed、`RegisterPool` 零 warning |

### 最終效能

三次配對量測（同一輪內比較；跨輪絕對值有雜訊，只有輪內差值可信）：

| config | RAP=0 (µs) | RAP=1 (µs) | 改善 | 步驟 0 天花板 |
|---|---|---|---|---|
| MT 64x512 | 56.16 / 56.04 / 56.91 | **54.77 / 55.26 / 54.73** | **−2.5%** | 5.7% |
| MT 64x256 | 64.41 / 64.08 / 64.33 | **59.44 / 59.72 / 59.68** | **−7.2%** | 10.1% |

MT64x256 拿到天花板的七成。天花板是「A 的搬移完全免費」的理想值，而常駐本身要佔用暫存器
（ValuA 64→192、MXSA 4→12），所以拿不滿是合理的。

MT64x512 的比例較低，與它 store 餘裕很薄（`elementsPerBatch` 134 對需要的 128）一致——
它本來就處在暫存器壓力的臨界點。

### 4b-2 剝離：五個 bug 與定位方式

修完這五個之後，兩份副本的指令本體完全一致（ds_load 189=189、wmma 112=112、
dscnt 52=52、barrier 12=12）。

分階段驗證是解開這題的關鍵。`TENSILE_RAP_PEEL=2` 讓兩份副本純接續（無分支、無入口標籤，
結果必錯但計時與結構有效），一次就證明**複製本身沒問題**——後端對兩份都正常插入
waitcnt——把範圍鎖死在控制流。

1. **`rapKTileIdx` 未在每次發射開頭歸零**（在 writer 上、不在狀態快照範圍內）。
2. **`CloneSpec` 的 `startLabel` 寫死**，iterN 的標籤有後綴所以拿不到 clone。
3. **無條件分支讓 iterN 在 CFG 裡沒有前驅**。iter0 用 `s_branch` 跳過它，唯一入口是
   `s_setpc` 的間接 back-edge；CFG builder 只對**條件**分支建 fall-through 邊，
   所以 iterN 被排除在後端資料流外，**一個 waitcnt 都沒插**。改用永遠成立的條件分支即可。
4. **barrier 重建的起始狀態錯誤**。`postMainLoopBarrierCheckAndReset` 線性走訪 token，
   但 iterN 是 back-edge 目標。在 `label_RAP_IterN` 還原成 persistent loop 入口的 token 狀態。
5. **`CloneSpec` 的轉換靠名稱識別**。第二個 clone 命名成 `InitCIterWmmaRAPIterN` 時區域有被
   複製，但把 WMMA 的 src C 改寫成 0 的轉換沒套用（實測 iter0 有 16 個歸零 WMMA、iterN 有 0），
   **C 從第二個 tile 起不再歸零**。名稱必須同為 `InitCIterWmma`，只有 startLabel 不同。

### 常駐暫存器一共有三處會被借走，全部都要排除

這是整個功能最容易漏的地方——同一件事分散在三個站點，漏掉任何一個就會被覆蓋：

| 站點 | 何時執行 | 借走的範圍 |
|---|---|---|
| `KernelWriterAssembly.py:2881`（`setupNewTile`） | **每個 persistent iteration** | ValuAB + MXSAB 當 scratch |
| `KernelWriterAssembly.py:6280`（`initC`） | 每個 iteration | 上者的對應 remove，必須對稱 |
| `KernelWriter.py:6431` / `:6986`（主迴圈後） | 每個 iteration | 還給 store 用 |

`setupNewTile` 那段的註解是 "C regs are not used during initialization"，但它在
persistent loop 內、每個 tile 都跑一次。add 與 remove 不對稱時會出現大量
`RegisterPool::remove ... already unavailable` warning——那個 warning 是有意義的訊號。

### 步驟 5：MXSA 常駐被 store 覆蓋（第六個 bug）

只丟 A 的 ds_load → PASSED；只丟 MXSA 的 → FAILED。真正的原因是**最單純的暫存器被覆蓋**：

```python
# KernelWriter.py:6986（步驟 3 修過的 ValuA 回收的姊妹站點，當時漏掉）
if self.states.lastValuMXSAB:
    self.vgprPool.add(0, self.states.lastValuMXSAB, "ValuMXSAB")   # [0,30) 還給 store
```

主迴圈結束後這行把整個 MXS 區塊還給 store 當暫存，而 **MXSA 常駐在 v0–v11**——
正是 store 位址計算最愛用的低位暫存器（`v_add_nc_u32 v2` 與 `v_add_co_u32 v1` 各出現 252 次）。
A 之所以沒事，只是因為它住在 v176 以上。

修法與 ValuA 相同：RAP 開啟時只把非常駐的部分（MXSB）還回 pool。

**為什麼繞了很久**：兩次覆蓋掃描都只比對 `v[數字]` 的括號形式，而 store 用的是
`v0`、`v1` 這種**裸寫法**，整個被漏掉。掃描器的盲點讓「沒有東西覆蓋常駐範圍」
這個錯誤結論撐了好幾輪，還讓我編出一套 TileSpan 的錯誤解釋。

**已排除的假設（避免後人重走）**：

| 假設 | 排除方式 |
|---|---|
| TileSpan 讓 scale 的索引語意與 A 不同 | 規格確認 `matrix_a_scale:1` 只是選 lane 16–31，與 `ds_load` 無依賴 |
| iter0 沒填滿所有 MXSA buffer | iter0 填 X0–X5、iterN 讀 X0–X5，完全一致 |
| 兩次 ds_load 互相覆蓋 | 那是 `InitCIterWmma` clone 與原本迴圈的**替代路徑**，執行期只走一條 |
| `s_wait_dscnt` 低估 | 52 個 wait 中只有 7 個來自 Python，其餘由 StinkyTofu 後端依真實相依性插入 |
| `numReadsPerIter` 歸零造成排程問題 | 保留不歸零仍然失敗 |
| WMMA 指令不同 | 9 處差異在 PASSED 與 FAILED 的 build 中**完全相同**，故非原因 |

### 除錯管道（開發期使用，已於收尾時移除）

開發過程中曾在產生器裡掛上一組環境變數開關來做二分法定位，功能完成後已全部刪除，
以下僅作為方法紀錄。若日後要擴充 v0.1，同樣的切法仍然適用。

| 環境變數 | 用途 |
|---|---|
| `TENSILE_RAP_PEEL=2` | 兩份副本純接續（無分支／無入口標籤）。結果必錯但可判斷後端是否處理第二份 |
| `TENSILE_RAP_PEEL=3` | 跳過 iter0，只執行 iterN，用來分辨「iterN 本身壞」與「轉換壞」 |
| `TENSILE_RAP_DROP=ds\|tdm\|both` | 只丟 ds_load／只丟 TDM／全丟 |
| `TENSILE_RAP_DROP_TC=A\|MXSA` | 只丟單一 operand，分辨是 A 還是 scale 的問題 |
| `TENSILE_RAP_KEEP_COUNTS=1` | 丟讀取但不歸零 `numReadsPerIter*`，分辨資料問題與計數問題 |
| `TENSILE_RAP_FULL_DRAIN=1` | 把 iterN 的 wait 強制設 0（僅能改到 Python 發出的那 7 個） |
| `TENSILE_RAP_DEBUG_STATE=1` | 印出一次發射改變了哪些發射器狀態 |
| `TENSILE_RAP_DEBUG_WAIT=1` | 統計 `_wait` 呼叫次數 |

### 除錯方法上的教訓

1. **分階段隔離比逐項猜測有效。** `TENSILE_RAP_PEEL=2` 一次就把「複製本身有問題」
   從候選中刪掉，把範圍鎖到控制流。
2. **掃描器的盲點會製造假結論。** 只比對 `v[N]` 而漏掉 `vN`，讓「沒有東西覆蓋常駐範圍」
   這個錯誤結論撐了好幾輪。定位覆蓋問題時，兩種寫法都要涵蓋，`s_set_vgpr_msb` 的高位也要解析。
3. **先驗證前提再解釋現象。** 我一度用 TileSpan 編了一套聽起來合理但錯誤的解釋；
   真正有用的是回頭確認「iter0 到底填了哪些 buffer、iterN 讀了哪些」。
4. **編譯期檢查值得寫。** `_rapCheckALocalReadsDropped` 在開發過程中擋下兩次不一致的
   組合，把靜默的 `-nan` 變成當場失敗。

### waitcnt 是後端插的，不是 Python

值得單獨記下來，因為它推翻了計畫最初的一個假設。`ScheduleIterAlg=4` 會被改寫成
`_ScheduleIterAlg=0` + `_StinkyTofuOptLevel=3`，於是 **waitcnt 由 StinkyTofu 後端插入**
（`StinkyWaitCntInsertionPass`，並由 `RemoveDscntPass` / `StinkyRemoveWaitCntPass` 修剪）。
iterN 的 52 個 `s_wait_dscnt` 只有 7 個存在於 Python IR。

因此 Q9 花很多篇幅擔心的「`numReadsPerIter*` 低估導致 wait 變 no-op」在這個 config 下
影響有限——後端會依真實相依性重算。`numReadsPerIter*` 仍要維持正確（它同時餵
`SIA.py` 的排程延遲預算），但它不是唯一的安全網。

`rapDropALoads` 是步驟 5 三個機制（跳過 A/MXSA ds_load、cselect 清 TDM descriptor、
`numReadsPerIter` 歸零）的總開關，見 `KernelWriter.isRapDropALoadsActive`，預設開啟。

---

## 決定計畫形狀的關鍵事實

| 事實 | 後果 |
|---|---|
| `ReuseAcrossPersistent` 在 codebase 完全不存在 | yaml 現在連 parse 都不會過，第一件事是接線 |
| 暫存器上限 1024，現有 kernel 用 1023 | 但 1023 是 **epilogue** 高水位，主迴圈只到 613 |
| store 批次 = `numVgprAvailable // numVgprsPerElement`，元素數 128 | store 成本是**階梯**不是斜坡，K≤768 完全不受影響 |
| K 迴圈 trip count 永遠是 runtime SGPR，無完全展開機制 | 必須新增機制，這是最大的一塊 |
| C++ runtime 已支援 `SizeEqual` predicate | 不需要新增調參參數，K 由 `(PGR+1)*DepthU` 推導後直接發 predicate |
| MX scale 運算元在 `s_set_vgpr_msb` 沒有欄位 | MXSA/MXSB 必須在 v0–v255，這是編碼限制不是 bank 偏好 |
| `s_wait_dscnt` 立即數由 `numReadsPerIter*` 常數解析算出 | 刪 ds_load 必須同步改常數，算多了是靜默的 `-nan` |
| StreamK tile 空間跨 batch 線性化 | batch offset 每個 tile 在 `setupNewTile` 重算，RAP 未動此架構，任意 batch 皆已驗證 PASSED |
| A 只有 48 KB 且 1280 個 tile 共用 | RAP 省的是 LDS/L2 流量，不是 DRAM |

### 基準環境

- 測試 yaml：`mxf8mxf4_gfx1250_rap.yaml`
- Problem size：`Exact: [64, 655360, 1, 768]`（1280 tiles，5 tiles/WG）
- Config：MT 64x512、DepthU 256、MI 16x16x128、MIWaveTile [2,16]、MIWaveGroup [2,2]、
  WavefrontSize 32、NumWaves 4、NumThreads 128、PGR2、PLR1、SIA4、TDMInst 3
- 既有 RAP=0 kernel：`.vgpr_count 1023`、`CUOccupancy 1`、LDS 219648 bytes、
  K=1536 時實測 1.5551e6 GFlops（約 83 µs）
- 驗證指令：`./build_tmp/Tensile.sh mxf8mxf4_gfx1250_rap.yaml mxf8mxf4_gfx1250_rap`
  （GPU 卡住時前面加 `HIP_VISIBLE_DEVICES=x` 換一顆）
>
---

## 步驟 0：上限實驗與 config 選定

**目標**：在寫任何真正的 RAP 程式碼之前，知道天花板在哪、以及該把 RAP 落在哪個 config 上。

**做法**：改 codegen，無條件刪掉 A/MXSA 的 ds_load、並無條件對偶數 wave 把 A/MXSA 的
TDM descriptor count 清 0。不做常駐、不做展開、不加參數。設 `NumElementsToValidate: 0`，
驗證失敗是預期的。算出來的結果是錯的，但**執行時間是有效的**。

這個數字的意思是：「如果 A 的搬移完全免費，這個 kernel 能跑多快」——那就是 RAP 的**效益天花板**。

- wait 一律用保守的 `s_wait_dscnt 0` full drain，否則留著原本偏大的立即數會讓 wait 變 no-op，
  量到過度樂觀的數字
- 這個 hack 用到的機制正是步驟 5 要的，不是白工

**兩個 config 各量一次**：

1. 現況 `MIWaveTile [2,16]` / MT 64x512 — A 占 LDS 流量 20%
2. `MIWaveTile [2,8]` / MT 64x256 — A 占 33%，且 ValuC 256→128、ValuB 256→128，
   等於多釋出約 256 個暫存器給常駐用

codebase 裡**沒有任何**把 MIWaveTile 綁到 StreamK 或 PAP 的 reject 規則，
yaml 註解「StreamK+PAP needs MIWT 2x16」不是程式碼層面的硬限制，換 config 這條路是通的。

### 實測結果（已完成）

同一輪內配對比較（跨輪的絕對值有雜訊，只有輪內差值可信）：

| config | RAP=0 | oracle | 天花板 | 三次量測散布 |
|---|---|---|---|---|
| MT 64x512（MIWT 2x16） | 56.58 µs | 53.35 µs | **5.7%** | 3.47 / 5.72 / 7.91% |
| MT 64x256（MIWT 2x8） | 64.52 µs | 57.97 µs | **10.1%** | 10.04 / 10.17 / 10.21% |

結論：**兩個 config 並存**。MT64x256 的天花板是兩倍，而且量測散布只有 0.09%
（三次基準 64.53 / 64.48 / 64.54），是唯一能讓後面每一步「±2% 才算通過」
真正判定得了的 config，所以拿它當開發／回歸標的。
MT64x512 的絕對效能好 14%，是出貨標的，但它的量測雜訊 1.8%、
store 餘裕又薄，不適合當每一步的判定依據。

**輔助量測（選配）**：`rocprof-compute` 跑一次拿 roofline 與記憶體層級分解。
若要自己挑計數器，先跑 `rocprofv3-avail` 確認 gfx1250 支援哪些，重點是：

- `TCC_EA*_RDREQ` / `TCC_EA*_WRREQ` — 是否真的打 DRAM（K=768 的 DRAM 流量估計約 603 MB，
  其中 B 240 MB、C 讀 160 MB、D 寫 160 MB、MXSB 15 MB、**A 只有 48 KB**）
- `TCC_HIT` / `TCC_MISS` — A 是否已在 L2（若是，RAP 省不到 DRAM）
- `SQ_WAIT_INST_LDS` / `SQ_BUSY_CYCLES` — LDS 是否為瓶頸，這才是 RAP 真正攻擊的東西

---

## 步驟 1：參數接線

**目標**：讓 yaml 跑得起來，並產出 RAP=0 的基準。

要動的四個地方，抄 `PrefetchAcrossPersistent` 的既有慣例：

1. `Tensile/Common/ValidParameters.py` — 加 `"ReuseAcrossPersistent": [0, 1]`
2. `Tensile/Common/GlobalParameters.py` 的 `defaultBenchmarkCommonParameters` —
   加 `{"ReuseAcrossPersistent": [0]}`（`defaultSolution` 會自動衍生，不用另外改）
3. `Tensile/Common/RequiredParameters.py` 的 `getRequiredParametersMin()` — **必加**。
   否則 RAP=0 與 RAP=1 會 hash 出同一個 kernel 名字，其中一個被當重複靜默丟掉，
   `[0,1]` fork 會變成兩筆同一個 kernel 的數據。若不想讓所有既有 kernel 改名，
   用 `Naming.py:217` 那個「只在啟用時才加入命名」的 opt-in 慣例
4. `Tensile/SolutionStructs/Solution.py:1870` — 在 StreamK 關閉時把 `ReuseAcrossPersistent`
   歸零，避免產生兩個完全一樣的 kernel

`SizeMapping.StateKeys`（`Contractions.py`）**不用改**，RAP 純粹是 codegen 時期的開關，
host 端不需要看到。

還要重新產生兩個 syrupy snapshot（`ValidParameters` 與 `SolutionClass` 的 `.ambr`），
並比照 `Tensile/Tests/unit/test_PrefetchAcrossPersistent.py:670` 加一個對應的註冊測試。

**驗收**：`Tensile.sh` 跑得完，產出兩個名字不同的 kernel，RAP=0 的時間記錄為基準。

---

## 步驟 2：predicate 與 reject 條件

**目標**：把 K 鎖死、把所有「A 跨 tile 不變」的前提變成明確的 reject。

### 2a. 新增 `SizeEqual` emission

在 `Tensile/Contractions.py:478` 的 `ProblemPredicate.FromOriginalKeyPair` 補上 `SizeEqual`
（順手補 `SizeLessThan`），並在 `validParameters` / `defaultBenchmarkCommonParameters`
加對應條目。C++ runtime 端零改動——`SizeEqual` 已註冊且已實作。

- index 慣例：0=M、1=N、2=batch、3=K
- standalone client 也會執行 predicate，不符時報 `DID_NOT_SATISFY_ASSERTS`

### 2b. reject 條件區塊

放在 `Solution.py:1777` 的 PAP guard 旁邊，每條要有獨立訊息：

- `PrefetchAcrossPersistent == 1`
- `StreamK == 3` 且 `StreamKForceDPOnly == 1`
  — 允許 K-split 的話不同 persistent iteration 只涵蓋部分 K，常駐前提直接崩掉
- `InnerUnroll == 1`、`NoTailLoop == True`
- `DirectToVgprA == False`、`ExpandPointerSwap == False`、`GlobalSplitU == 0`
  （StreamK 下 GSU 被強制為 0 不是 1）
- `enableTDMA and enableTDMB and NumWaves > 1`（wave 分工的前提）
- `TDMSplit == False`、`UseSubtileImpl == False`
- K 不再由調參參數釘死：`_RAPNumResidentKTiles` 直接取 `PrefetchGlobalRead + 1`，
  對應的 `K == (PGR+1)*DepthU` 由 `SizeEqual` predicate 在執行期擋掉不合的尺寸
- 常駐暫存器算出來放不下 → reject，**絕不 silently 退回 RAP=0**

### 2c. problem-size 條件只能是 predicate，不能是 reject

原本這份計畫把 `ceil(M/MacroTile0)==1` 列在 reject 條件裡，**那是錯的**：
solution 是在不知道 problem size 的情況下推導的，`assignDerivedParameters` 執行時
M 根本不存在。凡是 problem-size 條件都必須走 runtime predicate。

`Contractions.CompoundPredicates` 在 `ReuseAcrossPersistent` 為真時發三個 predicate：

- M：`SizeEqual` index 0，值為 `MacroTile0`。一條同時涵蓋兩件事——只有一個 M-tile
  （常駐的 A 才會對所有 tile 都相同），以及 M 方向無邊界（partial tile 會走 masked store）
- N：`SizeMultiple` index 1，值為 `MacroTile1`。N 方向無邊界
- K：`SizeEqual` index `NumIndicesC`，值為 `(PrefetchGlobalRead + 1) * DepthU`

M 和 N 這兩條把 problem 限制在 **edge=0** 的 store 路徑，這是 v0 唯一驗證過批次行為的路徑。
負向測試：N=655488（不整除 256）與 M=32 都會得到 `DID_NOT_SATISFY_ASSERTS`，RAP0 則照常執行。

K 那條和 `NoTailLoop` 是互補而非重複：NoTailLoop 保證 K 是 DepthU 的整數倍（不會有半個
k-tile），`SizeEqual` 則把 k-tile 的**個數**釘在 `PGR+1`。unroll loop 執行次數是
`ItersPerTile − PGR`，而它每一輪都用同一個產生期常數 `rapKTileIdx = 0`，所以必須剛好跑一次。

batch **不需要 predicate**。RAP 沒有動 SK3 的 batch 位址架構，A 的 batch offset
仍由每個 tile 的 `setupNewTile` 重算，任意 batchCount 都已實測 PASSED。

### 2d. store 中立性守門

RAP 把 A 的搬移省下來，但常駐區塊不還給 store 的暫存器池，store 因此看到比較少的暫存器、
每批塞得下的元素變少，有可能多出一批。多一批 store 的代價可能吃掉省下來的 A 流量，
所以這件事必須擋，不能靠人工檢查。

`KernelWriterAssembly.rapCheckStoreNeutrality` 在 `refineOccupancy` 算完批次數後執行：

```
W          = 常駐區塊大小（見下）
availBase  = numVgprAvailable + W          # 還原成沒有常駐時的可用量
batchesRef = ceil(E / (availBase // numVgprsPerElement))
若 numBatches > batchesRef → overflowedResources = 9
```

`W` 不是估算，是兩個 `vgprPool.add` 的兩個分支相減——RAP0 從 `a.startVgprValu` /
MXSA 的 0 開始還，RAP1 從各自常駐區塊的尾端開始還，差額就是 RAP1 扣住的量：

```python
W = (b.startVgprValu - a.startVgprValu) + (mxsa.startVgprValu + mxsa.numVgprValu)
```

以 MT64x256 為例 W = ValuA 192 + MXSA 12 = **204**，跟從組語反推的數字一致
（六個 store 變體的 `elementsPerBatch` 差值 110/73/110/45/43/74，用 204 去除得到各變體的
`numVgprsPerElement` ≈ 2/3/2/5/5/3）。

只擋 **beta=1 且 edge=0** 那一個變體：M/N 的 predicate 已經把 problem 限制在無邊界，
而 beta=1 每元素成本高於 beta=0，是兩者中較緊的，守住它就涵蓋 beta=0。

reject 訊息會把上限一起算出來，讓調參知道往哪個方向動 DepthU：

```
ReuseAcrossPersistent holds 204 vgpr resident, splitting the store into 2 batches
instead of 1; largest store-neutral K is 256 but this kernel needs 768
```

實測 MT64x256 與 MT64x512 兩個 config 都沒觸發（`Batch #0` 是唯一的批次，
beta=1/edge=0 那檔是 218 → 163 對上 E=128，餘裕 35 個元素）。因為真實 config 走不到
拒絕分支，這段邏輯改由 `test_ReuseAcrossPersistent.py` 直接測：中立通過、退化拒絕並檢查
訊息裡的兩個 K 值、以及 beta=0 / edge=1 不會被誤擋。

**已知限制**：`numVgprAvailable + W` 還原出來的是「同一顆 kernel 但把常駐區塊還回去」，
等同真正的 RAP0 baseline 的前提是 occupancy 沒變（`numVgprAvailable` 的上限 `maxVgprs`
來自 `setOccupancy`）。目前的 config 沒跨 occupancy 邊界，但若常駐區塊大到掉一個 wave slot，
這個還原會失準，而且掉 occupancy 的損失遠大於多一批 store。

### 2e. codegen 端的 enable 判斷

在 `KernelWriter.py` 加 `isReuseAcrossPersistentEnabled(kernel)`，
比照 `:10817` 的 `isPrefetchAcrossPersistentEnabled` 自己重算 enable 條件，不信任 state flag。

### 已知的既有問題，記錄但不修

yaml 寫 `AssertSummationElementMultiple: 256`，但產出的 solution 是 32、
發到 library 的 predicate 也是 `BoundSizeMultiple: 32`，同時 `NoTailLoop: true`。
也就是這個 kernel 現在會接受 K=1568 這種非 256 倍數的 size，然後用一個沒有 tail loop 的
kernel 去算它。

這是既有的鬆動，混進 RAP 的 commit 會讓歸因變模糊——**開一張獨立 issue**。
但要知道它存在，因為新加的 `SizeEqual` 會是唯一真正鎖住 K 的東西。

---

## 步驟 3：常駐 VGPR 配置

**目標**：讓 A/MXSA 的暫存器涵蓋整條 K，並活過 store。

### 為什麼必須活過 store

`store 在 persistent loop 裡面`，時間軸是：

```
label_PersistentLoopStart:
  tile n   : 載入 A → 主迴圈 → WMMA 算完
             store tile n 的結果        ← KernelWriter.py:6081 現在就是在這裡回收 ValuA
  tile n+1 : 載入 A → 主迴圈 → ...     ← RAP 要消滅的就是這次載入
  ...(重複 5 次)
```

「WMMA 算完就可以放掉」在單一個 tile 之內是對的，但 RAP 的價值主張就是
「tile n+1 不要再載一次 A」。如果 store 把那些暫存器拿去用，tile n+1 就沒有東西可以重用。
常駐 A 的生命週期是「從 iter0 第一次 ds_read 到最後一個 persistent iteration 結束」，
**中間跨越每一次 store**。

### 核心改動

`KernelWriter.py:7182` 的 `numVgprBuffer`。RAP 開啟時，A 與 MXSA 的 buffer 數
從 `LoopIters`（2）變成 `LoopIters × (K_fit / DepthU)`。K=768 就是 6。
既有的 sizing 公式 `numVgprValuA = numVgprValuAPerBlock × numVgprBuffer × InnerUnroll`
和 `vgprValuA_X{n}_I0` 的命名都會自動跟著長大。

K=768 的具體數字：

| | 現況 | RAP |
|---|---|---|
| ValuA | 64（2 buffer × 32） | **192**（6 × 32） |
| ValuMXSA | 4（2 × 2） | **12**（6 × 2） |
| 靜態配置總計 | 613 | **749** |
| store 可用 | ~740 | **~531** |
| beta 路徑 `elementsPerBatch` | 185 | **~132** |

132 對需要的 128，餘裕只有 3%。
**改完第一件事就是去看新組語裡 `elementsPerBatch=` 那行印出來的實際值**，
它會直接告訴你有沒有跳成兩批。

參考公式（n = K/DepthU 個常駐 k-tile）：`available(n) ≈ 740 − 68n`，
beta 無 edge 路徑每元素 4 個暫存器、共 128 個元素，所以需要 `available ≥ 512`。

| K | n | store 可用 | 批次 |
|---|---|---|---|
| 512 | 2 | 604 | 1 |
| **768** | **3** | **536** | **1** |
| 1024 | 4 | 468 | 2 |
| 1536 | 6 | 332 | 2 |

### 三個實作要點

1. **常駐區塊不能被回收**。`KernelWriter.py:6081` 現在無條件把
   `[a.startVgprValu, lastValuAB)` 還回 pool 給 store 用。RAP 時要把 A 那半邊排除，
   只還 ValuB。正下方 `if not isPrefetchAcrossPersistentEnabled` 那個
   「刻意保留 G2L/位址」的分支就是現成的寫法範本。
   注意 `valuC` 是疊在 AB tile 上面配置的（`KernelWriter.py:8567-8575`），
   所以保留 A 也要把 `c.startVgprValu` 往後推。

2. **MXSA 必須留在 v0–v255**。這已經是現有 layout（`vgprMXSBase = 0`，
   `vgprAllocationImplClassic` 一開始就 `vgprIdx = 0` 把 MX scale 放最前面）。
   原因是 `s_set_vgpr_msb` 只有 src0/src1/src2/dst 四個 2-bit 欄位，
   而 scaled WMMA 有六個暫存器運算元，`setMsb(kStr, {a, b, acc2}, acc)`
   沒有餵 mxsa/mxsb——它們**沒有欄位可以表達高位索引**。
   RAP 只是把這個 block 從 4 撐大到 12，ValuC 起點往後推。12 遠在 256 以內，沒有風險。

3. **常駐 A 區塊要對齊運算元寬度**。`s_set_vgpr_msb` 的高位是從**運算元起點**算的，
   所以任何一個運算元都不能跨過 256 的倍數。對齊值取「一個 A 運算元的寬度向上取 2 的冪，
   最小 16」——本 config 是 16。對齊 32 也能work，但在 MT64x512 上會多浪費 16 個暫存器，
   而那個 config 的 store 餘裕本來就很薄。
   - `KernelWriterAssembly.py:6744` 有個 `valuVgprAlignment = 8 if HasVgprMSB else 2`
     **算了但整棵樹沒人用**，底下還是傳字面的 2。
     **只在 RAP 的常駐區塊局部對齊，不動這個死變數**——
     接上它會改變所有 gfx1250 kernel 的配置，讓 A/B 比較失去意義。另開一張 issue。

### 實測結果（已完成）

四個 kernel 全部 **PASSED**，時間與 RAP=0 沒有差別
（MT64x512 56.06 / 56.41 µs，MT64x256 64.73 / 64.53 µs）。
也就是說**單純配置 194 個常駐暫存器並跨越 store 保留，成本是零**。

| | MT64x512 RAP0 | MT64x512 RAP1 | MT64x256 RAP0 | MT64x256 RAP1 |
|---|---|---|---|---|
| `vgprBase` | 282 | 304 | 154 | 176 |
| ValuA | 64 (X0–X1) | **192 (X0–X5)** | 64 | **192** |
| ValuMXSA | 4 | **12** | 4 | **12** |
| `vgprValuC` | 22 | 30 | 22 | 30 |
| edge=0 beta `elementsPerBatch` | 185 | **134** | 218 | **166** |
| 需要的元素數 | 128 | 128 | 64 | 64 |
| store 批次 | 1 | **1** | 1 | **1** |

**MT64x512 的餘裕只剩 134 對 128（4.7%）**，任何進一步的暫存器成長都會把它推成兩批；
MT64x256 是 166 對 64，餘裕 2.6 倍。這正是把 MT64x256 當開發 config 的理由。

---

## 步驟 4：K 迴圈完全展開與 iter0/iterN 剝離

這是最大的一塊。**這一步不刪任何指令，只改結構。**

### 4a 的實際結論：K ≤ (PGR+1)×DepthU 時不需要任何展開機制（已完成）

原本以為要新增「把 unroll loop 展開成固定份數」的機制。實際檢查發現不必要：

`LoopCounterL` 從 `ItersPerTile` 開始，進入條件 `> PGR`、退出條件 `<= PGR`，
所以 unroll loop 執行 `ItersPerTile - PGR` 次。K=768、DepthU=256、PGR=2 時
`ItersPerTile = 3`，**迴圈本體剛好只執行一次**。三個 k-tile 因此天然對應到
三個已經分離的程式碼區段：unroll loop 本體、NGLL、NLL。

所以 v0 只需要：

1. `Solution.py` reject `K/DepthU > PGR + 1`
   （更大的 K 才需要真正的展開，明確標記為未實作）
2. 每個區段指定一個 k-tile 索引：loop body=0、NGLL=n-2、NLL=n-1
3. A/MXSA 的 buffer 索引改用**絕對迭代索引**而非 PLR 輪替窗

索引公式（`KernelWriter.rapBufferIdx`）：

- MFMA 運算元：`kTileIdx * LoopIters + u`
- local read：`kTileIdx * LoopIters + u + numItersPLR`（讀取跑在前面一格）

超出最後一個 k-tile 的索引回傳 None 並抑制該次讀取。那次讀取的語意是
「預取下一個 tile 的第一個 k-tile」，正是 RAP 要消除的搬移；
若不抑制它會繞回 X0，**覆蓋掉常駐的 k-tile 0**。
（實測顯示既有的 `doReadA` 條件本來就不會發出那次讀取，所以這道防護目前沒有觸發，
但保留它，因為它是唯一擋住那個覆蓋的東西。）

**實測**：四個 kernel 全部 PASSED，WMMA 讀遍 X0–X5、總數 224 與基準一致，
A 的 ds_load 仍是 56 次、MXSA 7 次、TDM 8 次——**指令完全沒變，只是換了目標暫存器**，
時間也沒變。這是最理想的檢查點：常駐定址已證明正確，且尚未引入任何行為差異。

---

### 4b. K 迴圈完全展開（K > (PGR+1)×DepthU 才需要，v0 不做）

`K_fit` 被 predicate 鎖死之後 `ItersPerTile` 成為編譯期常數，展開才合法。
展開是必要的，因為組合語言沒有暫存器動態索引（gfx12 沒有 `s_set_gpr_idx`），
第 j 個 k-tile 要讀第 j 組常駐暫存器就必須是編譯期決定的靜態編號。

**只展開中間那段 loop，NGLL 與 NLL 原封不動保留在後面**：

```
K=768, ItersPerTile=3, PGR=2
  展開的 loop body ×1   ← 原本的 unroll loop，發 global load
  NGLL                  ← 第 2 趟，不發 load
  NLL (+PAP)            ← 第 3 趟，不發 load，掛 PAP 交接
```

NGLL/NLL 承載 PGR2 管線收尾與 PAP 的狀態借用／還原
（`setupPrefetchAcrossPersistentLoads` 的 contract 註解列了四類 borrowed state），
重建它們的風險遠大於收益。

**每一段展開出來的 body 都要加註解標明它屬於哪一類**，例如：

```
/* RAP unrolled k-tile 0/3 (loop body, issues global loads) */
/* RAP k-tile 1/3 (NGLL) */
/* RAP k-tile 2/3 (NLL + PAP handoff) */
```

### 4b. iter0/iterN 剝離

**只複製計算段，store 共用一份。**

store 從第 3604 行 `/* Global Write Elements */` 到第 23095 行，佔全檔 84%。
照字面複製整個 persistent loop body 會讓 kernel 從 23,000 行變成 42,000 行，
`.amdhsa_inst_pref_size 255` 與那 33 道 `s_prefetch_inst_pc_rel` 的預取策略會整個失效。

結構改成：

```
label_PersistentLoopStart:        ← 只有 iter0 從這裡進入
  iter0 計算段（ds_read 寫入常駐暫存器）
  s_branch label_RAP_StoreJoin
label_RAP_IterN:                  ← 後續 iteration 的迴圈頭
  iterN 計算段（步驟 5 才會在這裡刪東西）
label_RAP_StoreJoin:
  store（共用）
  close persistent loop → 回跳 label_RAP_IterN
```

要處理的邊界情況：

- **WG 只分到一個 tile 時 iterN 一次都不會執行**
  （`PersistentLoop.py:177` 的 `StreamKIter >= StreamKIterEnd` 判斷），
  回跳目標必須是 `label_RAP_IterN` 而不是 `PersistentLoopStart`
- `SkPrefetchPrimed` 的分支要保留在兩段裡——它會在某些 slice 跳過 NLL 時被防禦性清 0
  （`KernelWriterAssembly.py:6226` 與 `:9999`），
  所以「iterN 進來時一定是 1」並不成立，不能靜態化掉

### 常駐資料怎麼進到常駐暫存器

**iter0 的 ds_read 直接以常駐暫存器為目的地**，不做 `v_mov` 複製。
等於把 ValuA 從「PLR 雙緩衝的小視窗」改成「涵蓋整條 K 的完整陣列」，
MFMA 在 iter0 和 iterN 都直接讀常駐區，零搬移。

LDS 不需要變大——iter0 的第 j 個 k-iteration 仍然從 LDS 讀同一塊，
只是寫進不同的常駐暫存器。

### ⚠ 第一個效能檢查點

這一步只是展開，指令總數不變，但有兩個效能變因：
kernel 變大（估計 23k → 28k 行）帶來的 I-cache 壓力，以及排程可能改變。

**驗收**：RAP=1 結果 PASSED，且執行時間與 RAP=0 落在 ±2% 內。
**若明顯退化，先分析並解決再往下走**——把 `s_prefetch_inst_pc_rel` 的位置和
`.amdhsa_inst_pref_size` 列為第一嫌疑。
這一步的數字乾淨與否，直接決定步驟 5 的數字能不能歸因。

---

## 步驟 5：刪除 ds_load 與停用 TDM

### 5a. iterN 刪掉 A/MXSA 的 ds_load

單純從發出去的指令流移除。

### 5b. iterN 讓偶數 wave 的 A/MXSA TDM 失效

抄 `KernelWriterAssembly.py:11429-11446` 的 HalfPLR 前例：

```
s_bitcmp1_b32 s[sgprWaveIdx], 0     // check wave parity
s_cmov_b32 s[sgprtdmAGroup0+0], 0   // even wave: NULL A descriptor
s_cmov_b32 s[sgprtdmMXSAGroup0+0], 0
```

比較指令與 cmov 要相鄰出現以利閱讀。四個要點：

- **不能只設一次**。descriptor 每個 persistent iteration 都會被重建
  （`s_mov_b32 s[sgprtdmAGroup0+0], 1` 出現在 `PersistentLoopStart` 之後，
  還有一處標著 `restore PAP LDS bank after descriptor refresh`）。
  cmov 必須跟在**每一次 descriptor 重建之後**。
- B/MXSB 的 descriptor 是**別名**到 A 的 SGPR
  （`RegSet("s", "sgprtdmBGroup0", "sgprtdmAGroup0")`），
  奇數 wave 拿到的是 B 的內容，所以只對偶數 wave 清零，B 完全不受影響。
- SGPR 名字是 **`sgprWaveIdx`**，而且它會被 UNDEF 回收；StaggerU=0 的情況下不保證還活著，
  **必須透過 `_emitTdmWaveParitySCC`（`KernelWriterAssembly.py:19192`）取得 parity**，
  不要直接讀 `sgpr("WaveIdx")`。
- TDM 的 wait 是獨立的 `s_wait_tensorcnt`，非 subtile 路徑上每個 call site 傳的都是 0
  （full drain），所以清零 descriptor 不影響 wait 正確性。
  （清零後指令仍會發出並遞增 counter，跟這個 full drain 的設計一致。）

### 5c. `numReadsPerIter*` 改成依區段查表

`s_wait_dscnt` 的立即數在 `KernelWriter._makeSubIterSchedule` 由
`states.numReadsPerIterA` / `numReadsPerIterMXSA` 等常數算出
（`:2537-2547`、`:2642-2645`），`KernelWriterModules.wait()` 也用同一組。
要讓 iter0 與 iterN 帶不同的值。

號誌方向：`s_wait_dscnt N` 是「等到未完成的 DS 操作不超過 N 個」。

- **算多了** → wait 變 no-op → WMMA 讀到未到達的資料 → **結果錯，`-nan`**
- **算少了** → 等超過需要 → **慢，但正確**

**同時必須加一道編譯期檢查**：產完每個區段之後走一遍它的 module，
用 `isinstance(inst, DSLoadInstruction)` 數出實際發出的 ds_load 數量，跟查表值比對，
不一致就讓 codegen 失敗。`postMainLoopBarrierCheckAndReset`（`KernelWriter.py:10919`）
已經在做同樣的走訪，抄它就行。

理由：這個立即數跟指令流完全脫鉤；codebase 裡 TileSpan 那段註解就是他們踩過這個坑
留下的紀錄。**這道檢查不是選配**，它是把「希望對」變成「錯了會當場知道」的唯一手段。

若查表法造成 FAILED，退回 `s_wait_dscnt 0` full drain 確認正確性，再逐步收緊。

### 注意歸因

`numReadsPerIter*` 也餵 `SIA.py` 的排程延遲預算與 SIA3 的 interleave 修正，
所以 iterN 的**排程整個會變**，不會是「iter0 減掉幾道 ds_load」。
這一步同時改了指令數和排程兩個變因。若要拆開，先用 full drain 量一次、再換查表法量一次。

---

## 步驟 6：最終驗證與量測

```bash
./build_tmp/Tensile.sh mxf8mxf4_gfx1250_rap.yaml mxf8mxf4_gfx1250_rap
```

GPU 卡住時前面加 `HIP_VISIBLE_DEVICES=x` 換一顆。

- **正確性簽核**：把 `NumElementsToValidate` 改成 `-1` 跑一次全量驗證。
  日常迭代再調回 1000。這個改動讓「某些 wave 不搬資料」，1000 個元素的覆蓋率不足以信任。
- **效能**：與步驟 0 的天花板、步驟 1 的 RAP=0 基準、步驟 4 的展開後基準三者對照。
- **檢查 `elementsPerBatch=`** 那行，確認 store 仍是單一批次。
- K=768 通過之後，把 problem size 換成 K=1536 再跑一次。
  此時預期會跳成兩批 store，兩者的差就是「多一批 store」的獨立代價。

### 效益預期

A+MXSA 占每個 DepthU 搬移量的 19.5%（A 16384 + MXSA 512 + B 65536 + MXSB 4096 bytes），
RAP 拿掉其中 4/5，**理論上省下 15.6% 的資料搬移，這個比例跟 K 無關**——
重用倍數是「一個 WG 跑幾個 tile」（5），不是 k-tile 有幾個。

但這是 LDS/L2 層級的節省，不是 DRAM。A 整個只有 48 KB 且 1280 個 tile 共用，
幾乎確定已在 L2。**若 kernel 是 DRAM bandwidth bound，RAP 的效益接近零**——
這就是步驟 0 要先量的原因。

---

## 風險清單

| 風險 | 徵兆 | 對策 |
|---|---|---|
| `numReadsPerIter*` 與實際指令數不一致 | 偶發 `-nan` | 步驟 5c 的編譯期檢查（強制） |
| store 跳成兩批 | 步驟 6 效能不如預期 | 看 `elementsPerBatch=` 註解，降 K 或換 config |
| kernel 變大導致 I-cache miss | 步驟 4 就退化 | 步驟 4 是獨立檢查點，先解決再往下 |
| 常駐區塊跨 256 邊界 | 隨機錯誤結果 | 對齊到一個 A 運算元的寬度（本 config 16） |
| cmov 只設一次被 descriptor 重建蓋掉 | A 仍在搬、效能沒改善但結果正確 | 檢查組語中 cmov 的出現次數與位置 |
| 上限本來就很低 | 全部做完只快 2% | 步驟 0 先量，不到 5% 就換 config 再量一次 |
| **常駐暫存器被借去當 scratch** | 結果錯誤但指令看起來完全正確 | 三處借用站點都要排除（見上表）；掃描時 `v[N]` 與裸 `vN` 都要涵蓋 |

---

## 明確排除在 v0 之外

- 可調的常駐 k-tile 數（`j < K/DepthU`）——v0 是「全放或不產生 kernel」
- `K/DepthU > PrefetchGlobalRead + 1`（本 config 上限 K=768）——更大的 K 需要把 unroll loop
  展開成一個 k-tile 一個區段，`Solution.py` 已明確 reject
- `ceil(M/MT0) > 1`、batch > 1、K-split（StreamK 非 DP-only）
- 修 ASEM 被降成 32 的既有問題（獨立 issue）
- 接上 `valuVgprAlignment` 死變數（獨立 issue）
- subtile 路徑（`UseSubtileImpl=1`）——`InstructionEmitter.py:280` 那裡的 tensorcnt
  是數實際發出的指令，需要另外處理
- A 繞過 LDS 直接 global→VGPR（DirectToVgpr 風格）

### 出貨前必補

`ceil(M/MacroTile0)==1` 與 `batchCount==1` 這兩個 runtime predicate（見 2c 節）**尚未實作**。
v0 的測試 problem size 天然滿足，但沒有它們，library 會把 RAP kernel 配給 M>MT0
或多 batch 的 problem，結果會是錯的。

---

## 主要程式碼位置速查

| 用途 | 位置 |
|---|---|
| 參數宣告 | `Tensile/Common/ValidParameters.py` |
| 參數預設值 | `Tensile/Common/GlobalParameters.py`（`defaultBenchmarkCommonParameters`） |
| kernel 命名 roster | `Tensile/Common/RequiredParameters.py`（`getRequiredParametersMin`） |
| reject 機制 | `Tensile/SolutionStructs/Utilities.py`（`reject`） |
| PAP guard block（抄這個） | `Tensile/SolutionStructs/Solution.py:1777` |
| 父功能關閉時歸零 | `Tensile/SolutionStructs/Solution.py:1870` |
| predicate emission | `Tensile/Contractions.py:478`（`FromOriginalKeyPair`） |
| codegen enable 判斷（抄這個） | `Tensile/KernelWriter.py:10817` |
| VGPR 靜態配置 | `Tensile/KernelWriter.py:8464`（`vgprAllocationImplClassic`） |
| `numVgprBuffer` | `Tensile/KernelWriter.py:7182` |
| ValuAB 回收（要改這裡） | `Tensile/KernelWriter.py:6081` |
| `numReadsPerIter*` 定義 | `Tensile/KernelWriter.py:9892-10015`（`_initKernel`） |
| dscnt 計算 | `Tensile/KernelWriter.py:2537-2547`、`:2642-2645` |
| persistent loop 開/關 | `Tensile/Components/PersistentLoop.py:82` / `:177` |
| PAP 交接 | `Tensile/KernelWriterAssembly.py:18398` |
| HalfPLR cmov 前例（抄這個） | `Tensile/KernelWriterAssembly.py:11429-11446` |
| wave parity helper | `Tensile/KernelWriterAssembly.py:19192`（`_emitTdmWaveParitySCC`） |
| TDM descriptor 定義 | `Tensile/KernelWriterAssembly.py:942`（`defineTdmSgprs`） |
| store 批次計算 | `Tensile/KernelWriterAssembly.py:16225`（`refineOccupancy`） |
| module 走訪前例 | `Tensile/KernelWriter.py:10919`（`postMainLoopBarrierCheckAndReset`） |
