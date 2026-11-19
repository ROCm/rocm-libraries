# Plan: wariant `addr64` kerneli conv fwd dla tensorów > 2 GiB

Status: faza 0 zaimplementowana i scalona. Z fazy 1 (sekcja 7) zaimplementowane
i **zweryfikowane na realnym GPU (gfx950)** są prymitywy z sekcji 3/5:
- `tile_origin_decode_i64`, `carry_decode_i32` (+ mirror C++), transform
  `UnmergeAddr64`/`unmerge_addr64` — i64 M-decode `m0 -> (n0,ho0,wo0)` raz na
  WG + i32 per-lane carry, bez magic division.
- `make_tile_buffer_resource` (Faza 0) — origin i64 scalaryzowany do SGPR,
  `num_records` clamp do `2^31-1`.
- `WarpGrid.bind(..., m_origin_i64=True)` (`platform/python/rocke/helpers/geometry.py`)
  — i64-owy odpowiednik `block_m_off` (`block_id * tile_m`), bo to mnożenie
  dzieje się dziś w i32 we współdzielonym helperze geometrii (każda rodzina
  kerneli MMA-tile, nie tylko conv), więc i64 origin musi powstać tam, nie
  dopiero w pochodnej `block_m_off_v`.
- ABI (sekcja 4): `conv_arg_names(direction="fwd", addr64=True)` + mirror C++,
  2D/non-split-K/non-3D only.

Test GPU (`platform/tests/instances/test_addr64_gpu_numeric.py`, 10/10 PASS na
gfx950): `make_tile_buffer_resource` z prawdziwą alokacją > 4 GiB (odczyt
poprawnej wartości) i z originem > 4 GiB względem małej alokacji (clamp do
zera, bez faulta); `tile_origin_decode_i64`+`carry_decode_i32` kontra
referencja w czystym Pythonie dla obu scenariuszy (ogromne N, ogromne Ho*Wo).

Żadna rodzina kerneli jeszcze nie buduje ścieżki `addr64`
(`validate()` odmawia `addr64=True`) — prymitywy są gotowe i zweryfikowane,
ale **nie podłączone** do `build_implicit_gemm_conv`. Zakres: wszystkie
rodziny **forward** convolution w `library/kernels/common/` i ich mirrory C++
w `library/cpp/`.

**Co blokuje podłączenie do rzeczywistego kernela (dwa osobne, współdzielone
miejsca, oba wymagają namysłu przed zmianą):**

1. **Łańcuch deskryptora A/D.** `make_a_descriptor_dynamic`/
   `make_d_descriptor_dynamic` (`_conv_implicit_gemm_common.py`) liczą
   *bezwzględny* offset jako `sum(coord_i * stride_i)` po całym łańcuchu
   transformacji — łącznie z dekodowanym `n`, mnożonym przez `stride_n`.
   Podmiana samego `spatial_unmerge_dynamic` na `UnmergeAddr64` (który też
   produkuje bezwzględne `(n, ho, wo)`) nie usuwa więc przepełnienia i32:
   `n * stride_n` to dokładnie ten iloczyn, który `addr64` ma ominąć dla
   dużego `N`. Ten iloczyn musi wejść w skład **originu** (i64, przez
   `make_tile_buffer_resource`, już gotowe), a per-lane arytmetyka i32 może
   liczyć wyłącznie ograniczoną deltę w obrębie kafla — czyli deskryptor dla
   `addr64` powinien w ogóle nie nosić współrzędnej `n` (ani jej stride'u) w
   swoim łańcuchu `coord_names`/`strides`, bo ten wkład jest już wliczony w
   origin. Najbliższy gotowy budulec: `DynamicTensorDescriptor.offset_i64_split`/
   `offset_i64` (`platform/python/rocke/helpers/transforms.py`), wzorzec
   „jeden duży wskaźnik bazowy + reszta i32” już używany gdzie indziej.
2. **Epilog store'u D (odkryte przy czytaniu `d_addr`'s wywołujących).**
   `d_addr` nie jest wołane bezpośrednio w `conv_implicit_gemm.py` — jest
   przekazywane jako `addr_fn=d_addr` do `CShuffleEpilogue.store`/
   `DirectEpilogue.store` (`platform/python/rocke/helpers/epilogues.py`),
   które **same** liczą `m_val` z `block_m_off_v` + offset per-warp/lane
   (ten sam i32-mnożenie problem co w geometry.py) i maskują go przeciw
   `bounds=(p_M, c_N_bound)` (`m_val < p_M`, i32 compare). Te klasy epilogu są
   współdzielone z innymi rodzinami kerneli (GEMM, prawdopodobnie attention),
   więc `addr64` wymaga **dwóch** osobnych zmian we współdzielonym kodzie
   (geometrii — już zrobione — i epilogu — jeszcze nie), nie tylko w
   `conv_implicit_gemm.py` samym.

Dopiero po (1) i (2) którykolwiek builder implicit-GEMM może bezpiecznie
ustawić `addr64=True`. Oba punkty mają teraz solidne, zweryfikowane na GPU
building blocks (origin i64, M-decode) do wykorzystania — brakuje
przeprojektowania samego łańcucha offsetu i epilogu, co jest osobnym,
większym zadaniem niż dodanie kolejnego prymitywu.

## 1. Cel i zasady

Cel: każdy kernel conv fwd dostaje opcjonalny wariant `addr64`, który obsługuje
tensory, batch i liczbę pikseli wyjścia poza zakresem i32, bez dzielenia batcha
na hoście.

Zasady, które wynikają z wymagania „bez regresji, nie ruszać ścieżki i32”:

1. **Osobne IR.** `addr64` to pole specu (compile-time knob), domyślnie `False`.
   Przy `addr64=False` builder emituje **bajt w bajt** to samo co dziś. Ścieżka
   i64 jest osobną gałęzią w builderze wybieraną w Pythonie / C++ w czasie
   budowania (`if spec.addr64:`), nigdy runtime'owym `scf_if` w kernelu i32.
2. **Dowód braku regresji = gate.** `check_byte_identity.py` jest GREEN na
   wszystkich flavorach **bez re-blessu żadnego istniejącego golden**. Nowe
   golden powstają tylko dla nowych przypadków `addr64`.
3. **Stabilne nazwy i hashe.**
   - `kernel_name()`: flaga `"a64"` w `flags={...}` dopisywana tylko gdy
     `addr64=True` (konwencja jak `"tf32"` / `"splitk"` w
     `conv_implicit_gemm.py:434`).
   - `KernelIdentity.addr64: bool = False` usuwane z payloadu w
     `stable_hash()` przy wartości domyślnej (jak `prefetch_addr`,
     `kernel_identity.py:249`) i pomijane w `label()` (`kernel_label.py`) przy
     `False`. Istniejące identity, cache AOT i katalog nie zmieniają się.
4. **Dispatch wybiera `addr64` tylko tam, gdzie i32 nie działa.** Problem, który
   dziś obsługuje kernel i32, nadal trafia do kernela i32. `addr64` jest
   dopuszczany wyłącznie gdy któryś limit i32 jest przekroczony (sekcja 6).
5. **Oba silniki w tej samej zmianie** (reguła byte-identity), helpery z testem
   (reguła `platform/AGENTS.md#helpers-placement`).

## 2. Model adresowania w wariancie `addr64`

Dwupoziomowy adres: **64-bitowy origin kafla w SGPR + 32-bitowa delta per lane.**

```
origin_i64  = adres bajtowy pierwszego elementu, który czyta/pisze kafel   (skalar, raz na WG)
base_ptr    = global_ptr_offset(ptr, origin_i64)
num_records = clamp(total_bytes_i64 - origin_i64, 0, 0x7FFFFFFF)          (skalar, i32)
rsrc        = buffer_rsrc(base_ptr, num_records)
voffset_i32 = delta współrzędnych kafla * strides (i32)                    (per lane)
```

Własności:
- Zachowane: OOB-zero bufora, `async_buffer_load_lds`, liczba VGPR, wartownik
  `INT32_MAX` dla zamaskowanych lane'ów (bo `num_records <= 0x7FFFFFFF`).
- `num_records` jest zawsze i32, więc działa także na llvm20 i w lowering HIP
  (które nie przyjmują i64 `num_bytes`).
- Mnożenia per lane zostają w i32. Host gwarantuje (sekcja 6.2), że
  **footprint kafla** (maksymalna delta + szerokość wektora) < 2^31 B, więc
  strides można obciąć do i32 po stronie kernela tylko w części „delta”.
- Delta musi być nieujemna dla każdego **ważnego** dostępu. Padding i ogony
  są maskowane przez `select(valid, off, SENTINEL)` przed użyciem (tak jak dziś).

### 2.1 Wybór originu

| Operand | Origin | Uwagi |
|---|---|---|
| A (wejście, NHWC) | `n0*Sn + hi_first*Sh` gdzie `hi_first = max(ho0*sH - pH, 0)` dla pierwszego piksela kafla `m0 = (n0, ho0, wo0)` | Każdy ważny dostęp lane'a z `m >= m0` ma adres >= origin (ten sam obraz: `hi >= ho0*sH - pH`, `hi >= 0`; następny obraz: dalej w pamięci przy upakowanym NHWC). |
| B (filtr, KYXC) | `k0 * Sk` (+ `group * cpg` dla grouped) | footprint `block_n * Sk` |
| D (wyjście) | `m0 * K` dla upakowanego NHWK, ogólnie `n0*Sn + ho0*Sh + wo0*Sw` | |
| workspace split-K | `slice_id * M * K * 4` w i64 | dotyczy `conv_implicit_gemm.py:2296` i reduce |

### 2.2 Kafle przecinające granicę obrazu

Przy upakowanym NHWC delta między ostatnim pikselem obrazu `n` a pierwszym
obrazu `n+1` jest mała (pamięć ciągła), więc zwykłe kafle są poprawne. Problem
pojawia się tylko gdy **pojedynczy obraz** jest tak duży, że `(n - n0) * Sn`
(lub równoważna delta) przekracza footprint 2^31 B, albo przy strides z
przerwami między obrazami.

Rozwiązanie: kernarg `p_image_aligned` (i32, 0/1) w ABI `addr64`. Przy 1 grid
dekoduje kafel jako `(n, kafel w obrębie obrazu)`, więc kafel nigdy nie
przecina obrazu. To jednorodny (scalar) wybór w prologu, nie w pętli K.
Host ustawia 1 tylko gdy footprint kafla zwykłego przekroczyłby limit (dla
małych obrazów image-aligned marnowałby większość kafla, więc nie jest domyślny).

## 3. Indeksowanie bez magic division (dekodowanie M)

Źródło limitu `N*Ho*Wo < 2^31` to magic division w dekodowaniu `m -> (n, ho, wo)`
(`do_magic_division` działa dla dzielnej < 2^31). **Dotyczy wyłącznie wariantu
`addr64`** — kernele i32 zachowują magic division, kernargi magic i swoje IR
bez zmian. W wariancie `addr64`:

1. **Origin kafla (skalar, raz na WG):** `m0 = tile_m_idx * block_m` w i64,
   `(n0, ho0, wo0)` przez **dokładne dzielenie i64** (`udiv`/`urem` i64 na
   wartościach jednorodnych; sekwencja programowa, ale tylko w prologu WG).
2. **Lane (per lane):** `d = m - m0 < block_m`. Współrzędne przez propagację
   przeniesienia w i32: `wo = wo0 + d`, `ho += wo / Wo`, `wo %= Wo`,
   `n += ho / Ho`, `ho %= Ho` — zwykłe `udiv` i32 na małych wartościach
   (`wo0 + d < Wo + block_m`), bez magic numbers i bez kernargów magic.
3. Liczba kafli: `tile_m_idx` w i32 (siatka), więc `M < 2^31 * block_m` —
   praktycznie bez limitu. Dekodowanie `wgid -> (tile_m, tile_n)` bez zmian
   co do algorytmu, ale **bez** limitu 65535 na oś (patrz 3.1).
4. **Dekodowanie K (`k -> (y, x, c)`) zostaje na magic i32 + `mul_u24`.**
   Te wartości są ograniczone rozmiarem filtra (`Y*X*cpg < 2^23`,
   `MUL24_REDUCTION_LIMIT`), nie rozmiarem tensora, a dekodowanie leży w pętli K.
   Usunięcie go nie poszerza wsparcia dużych tensorów, a kosztuje w gorącej pętli.
   Osobny, opcjonalny krok (poza tym planem): wariant bez `mul_u24` dla
   `Y*X*cpg >= 2^23`.

### 3.1 Siatka

- Implicit GEMM: dziś „every grid axis fits 65535”. W `addr64` liczba kafli M
  idzie na oś x (limit 2^31-1), oś y/z dla N-kafli / grup / split-K.
- Direct: dekodowanie `blk -> (k_tile, w_tile, h_tile, n)` przechodzi na
  i64 dla składowej `n` lub na oś z dla obrazu.
- Provider (`ConvNative.cpp:706`, `Gfx950ConvFwdGeometry.hpp:997`) już liczy
  siatkę w i64 i odrzuca > `UINT32_MAX` wątków; ta reguła zostaje.

## 4. ABI `addr64`

Osobne listy w `kernels/common/conv_abi.py` i `include/rocke/instance_conv_abi.h`
(nowe funkcje lub parametr `addr64=True`; listy i32 bez zmian):

- `A_bytes`, `B_bytes`, `D_bytes`, `ws_bytes`: **i64**.
- Strides zewnętrzne (`p_A_stride_n`, `p_A_stride_di`, `p_A_stride_hi`,
  `p_D_stride_n`, `p_D_stride_do`, `p_D_stride_ho`, `p_B_stride_k`): **i64**.
  Strides wewnętrzne (`*_stride_wi`, `*_stride_x`, …) zostają i32.
- `p_M`: **i64**. Magic pary dla `m_` (`_magic("m_", ...)`) **usunięte**.
  Magic pary dla `k_` zostają.
- Nowy `p_image_aligned` (i32).
- Kolejność zgodna z regułą pliku: `[pointers, byte sizes, problem block, extras]`.
- `runtime/packing.py` już obsługuje `"i64"` (`'q'`). `conv_args.py` dostaje
  producenta wartości dla `addr64`; `test_conv_abi.py` dostaje asercje kolejności
  dla nowych list.
- Direct (`conv_direct_arg_names`, `emit_direct_params` w
  `conv_direct_grouped.py:110`): analogicznie bytes i64 + strides zewnętrzne i64.

## 5. Platforma: helpery (faza 0)

W `platform/python/rocke/helpers/tensor_view.py` (+ mirror C++ w
`platform/cpp/`), każdy z testem:

1. `make_tile_buffer_resource(b, ptr, origin_i64, total_bytes_i64)` ->
   `BufferResource` z `num_records = clamp(total - origin, 0, 0x7FFFFFFF)`,
   origin przypięty do SGPR (`to_sgpr` / `readfirstlane` na obu połówkach).
   Wewnątrz `global_ptr_offset` (nie `global_ptr_add`, bo origin może wskazywać
   poza alokację przy pustym kaflu).
2. `tile_origin_decode_i64(b, m0_i64, dims_i64...)` -> `(n0, ho0, wo0)` dokładnym
   dzieleniem i64.
3. `carry_decode_i32(b, d, origin_coords, dims)` -> współrzędne lane'a.
4. Ewentualne brakujące op-y IR: sprawdzić, czy `udiv`/`urem` i64, `smin`/`umin`
   i64 i `trunc` i64->i32 istnieją w obu silnikach i w lowering HIP; dopisać
   brakujące (op + oba engine'y + `dsl_docs/reference/op_vocabulary.md`).

Testy CPU: golden-IR helperów, test clampu (`origin > total` -> 0 rekordów;
`total - origin > 2^31` -> `0x7FFFFFFF`). Test GPU: mikrokernel czytający
z originem > 4 GiB.

## 6. Host: footprint, admisja, dispatch

### 6.1 Kiedy wybierać `addr64`

Kernel `addr64` jest kandydatem tylko gdy wariant i32 **nie** obsłuży problemu:
dowolny tensor (A, B, D, workspace) > 2^31-1 B, albo `N*[Do*]Ho*Wo >= 2^31`, albo
siatka przekracza limit osi wariantu i32. W pozostałych przypadkach dispatch
nie widzi kandydatów `addr64` (sweep: `opt_in=True`). Gwarantuje to, że żaden
dziś obsługiwany problem nie zmieni kernela.

Relacja z istniejącym batch-splitem (`nongroupedBatchChunks`,
`_batch_split_plan`, winograd `batch_chunks`): zostaje bez zmian. `addr64` jest
dodatkową opcją; ranking między „batch-split i32” a „addr64” rozstrzyga
benchmark / heurystyka po pomiarach (faza 6).

### 6.2 Sprawdzenie footprintu (host, w obu: Python dispatch i provider C++)

Dla każdego operandu: maksymalna delta bajtowa w obrębie kafla (z uwzględnieniem
okna filtra, strides, dylatacji, wektora) musi być < 2^31. Jeśli nie, próba
z `p_image_aligned=1`; jeśli dalej nie, **odmowa** (przypadki patologiczne, np.
pojedynczy wiersz wejścia × wysokość okna > 2 GiB). Jedna funkcja w Pythonie
(`kernels/conv/` lub `dispatch/`) i jej mirror w providerze, z testem
równoważności na tablicy kształtów.

### 6.3 Python dispatch / launch

- `dispatch/grouped_convolution.py`: `launch_plan_for` / budowanie kernargów
  (`kind == "bytes"` -> i64 dla `addr64`), `_implicit_gemm_serves` i
  odpowiedniki direct uwzględniają `addr64`.
- `kernels/conv/aot_jobs.py` (`_fwd_jobs`, `_fwd_split_jobs`,
  `_fwd_merged_jobs`, `job_for_identity`, `spec_for_job`) i
  `kernels/conv/direct_aot_jobs.py` (`_knob_grid`, `_identity`,
  `direct_job_for_identity`), `direct_variants.py` (`make_direct_spec`): oś
  `addr64` w sweepie, ograniczona do podzbioru konfiguracji (nie podwajać
  całego katalogu).
- `engine_bridge.py` (winograd `:1353`, deep-fused `:616`): przekazanie knob'a.

### 6.4 Provider (`src/engines/kernel_ingestor_engine/packs/`)

- `Gfx950ConvFwdGeometry.hpp`: `tensorByteSize` zwraca dziś `optional<int32_t>`;
  dodać ścieżkę i64 dla `addr64`, nie zmieniając zachowania dla i32.
  `ImplicitGemmKernargs = std::array<int32_t, N>` (`:216`) -> nowy typ kernargów
  `addr64` (struct z polami i64) i nowa alternatywa w `ConvKernargs`
  (`Gfx950ConvFwdNative.cpp:82`).
- `matchedOf` / `aotInstanceLaunch` / `aotKernelFits`: instancja `addr64` tylko
  gdy matched problem przekracza i32 (6.1) i footprint przechodzi (6.2).
- Overlap check już działa na i64.
- `ConvNative.cpp` (exact-shape engine): ta sama reguła admisji.
- Regeneracja katalogu: `enumerate_conv_aot_instances.py`,
  `make_conv_aot_config.py` (procedura z `docs/gfx950_conv_fwd/aot_instances.md`
  §„Regenerating the bundle”); zaktualizować §„Current limits”.

## 7. Kolejność rodzin (fazy)

Każda faza = osobny PR: Python + C++ + parity emittery + golden dla nowych
przypadków + testy GPU + dispatch/provider dla tej rodziny.

| Faza | Rodziny | Specy / kluczowe miejsca | Uwagi |
|---|---|---|---|
| 0 | helpery platformy, `KernelIdentity.addr64`, ABI | sekcje 4–5 | bez zmian w kernelach |
| 1 | **implicit GEMM fwd** (MFMA gfx942/950, WMMA gfx1151/1201/1250), split-K, `conv_fwd_split_k_reduce` | `ImplicitGemmConvSpec` (`conv_implicit_gemm.py:165`); offsety `a_descriptor :1733`, `b_descriptor[_merged] :1759/:1797`, `a/b_offset_split :1941/:1979`, `d_addr :2267-2299`; `make_*_descriptor_dynamic` w `_conv_implicit_gemm_common.py :1497/:1600/:1652`; m-decode `:1943-1950`, `:1282`, common `:506`; reduce: `FwdSplitKReduceSpec` (`global_load_vN` z indeksem i32 -> rebase wskaźnika per blok) | największy zasięg; przy okazji usuwa limit 65535 osi |
| 2 | **direct buffer-based**: grouped (16c/4c/8c/32c/generic/depthwise std/tiled/col/spatial), nongrouped (+`DirectSplitReduceSpec`), `fwd_small_mfma` | `conv_direct_grouped.py` (rsrc `:1157,:2045,:2257,:2709`; wartownik `(1<<31)-1` w ~12 miejscach; m-decode `unmerge_magic :1268,:2759,:3242,:4434`); `conv_direct_nongrouped.py` (`_OOB_BASE=0x7F000000 :183`, guard `:567-596` — w `addr64` OOB_BASE liczony względem originu kafla); `conv_fwd_small_mfma.py` (`_OOB_OFFSET :76`, store'y `global_store*` z indeksem i32 `:501,:842-851`) | wartownik/`_OOB_BASE` muszą pozostać > `num_records` po rebase — przez clamp do `0x7FFFFFFF` i footprint < `_OOB_BASE` |
| 3 | **pp/flex MFMA** (gfx950): `conv_direct_pp_mfma`, `conv_direct_fwd_pp_mfma`, `conv_direct_fwd_flex_mfma` | już rebase per obraz (`conv_direct_pp_mfma.py:403-411`, flex `:503`); brakuje: ABI i64, `num_records` per obraz z clampem, rebase per kafel H gdy obraz > 2 GiB | najmniej pracy |
| 4 | **gfx1250 TDM**: `conv_direct_fwd_wmma`, `fwd_gstream_wmma`, `fwd_tapouter_wmma`, `fwd_small_wmma`, `dw_fwd_wmma`, helper `conv_fwd_wmma_tdm.cpp`, gate `conv_fwd_wmma_gate.py:153` | deskryptor TDM bierze 64-bit adres (`_tdm_descriptor_3d`, `conv_direct_fwd_wmma.py:607`) -> origin przez `global_ptr_offset` z i64 zamiast `global_ptr_add(ptr, off_i32*2)`; `global_load_vN`/`global_store_vN` z indeksem i32 -> rebase wskaźnika per kafel; `dw_fwd_wmma` dziedziczy z `build_direct_depthwise_dgrad_windowed` (wartowniki `1<<30`) | **otwarte:** szerokości pól wymiarów/strides w deskryptorze TDM — zweryfikować z dokumentacją ISA gfx1250 przed implementacją; byte-identity tylko w lane llvm23 (`TARGET_OP_PROFILE_FAMILIES`) |
| 5 | **winograd** (`conv_winograd_fwd`, `_fwd_rt`, `_gemm_wmma`, `_fused_1d_wmma`) i **deep_fused_conv_pool** | winograd: wiele kerneli + workspace U/V/M, `_OOB=0x7FFF0000`, limity `conv_winograd_select.py:87`, `conv_winograd_gen.py:250`, magic w `_fwd_rt` na dekodowaniu kafli/pikseli; deep-fused dziedziczy z implicit GEMM (`_build_implicit_gemm_conv_impl`) | winograd ma już batch-split; `addr64` potrzebny tylko gdy jeden obraz / workspace jednego obrazu > 2 GiB — możliwe odłożenie |
| 6 | pomiar i heurystyka | — | sekcja 8.3 |

## 8. Testy i weryfikacja

### 8.1 Bez GPU
- `check_byte_identity.py` na llvm20/22/23/24: GREEN, **zero zmian w istniejących
  golden** (dowód, że ścieżka i32 nietknięta).
- Emittery `library/tests/parity/<family>_emit.{py,c}`: nowe przypadki `addr64`
  **dopisane na końcu** listy indeksów (`_spec(idx)`), żeby indeksy istniejących
  przypadków się nie przesunęły. Sprawdzić limit 256/512 przypadków na rodzinę
  (`run_diff.py:193-201`).
- `test_conv_abi.py`: kolejność list `addr64`.
- `KernelIdentity`: `stable_hash()` i `label()` identyczne dla `addr64=False`;
  round-trip `from_label(label())` dla `addr64=True`.
- Test równoważności footprintu Python ↔ provider C++.
- `tests/run_all.py`, `library/tests/test_library_layering.py`,
  `test_library_reach.py` zielone.

### 8.2 GPU (numeryka)
1. **Równoważność na małych tensorach:** `addr64` vs i32 na tych samych małych
   problemach — wynik bit-exact (ta sama kolejność akumulacji). Tanie, pokrywa
   wszystkie konfiguracje sweepa.
2. **Duże tensory (> 2 GiB, > 4 GiB):** porównanie z wynikiem i32 batch-split
   (gdzie istnieje) albo z referencją CPU/torch na próbce pikseli (początek,
   granice obrazów, ostatni kafel, padding). Przypadki: duże N, jeden obraz
   > 2 GiB (`p_image_aligned=1`), filtr > 2 GiB, workspace split-K > 2 GiB,
   `N*Ho*Wo >= 2^31`.
3. Parity harness `library/builders/**/parity_*.py` w tolerancjach runbooka §1.4.
4. Test providera (`src/tests/.../TestGfx950ConvFwd*`, integration
   `IntegrationGpuGfx950ConvFwd.cpp`): admisja `addr64` tylko powyżej limitu.

### 8.3 Wydajność
- Nie wpływa na ścieżkę domyślną (osobne IR, wybór tylko powyżej limitu i32),
  więc pomiar służy do rankingu, nie do akceptacji regresji.
- Zmierzyć: `addr64` vs i32 na kształtach < 2 GiB (koszt wariantu) oraz `addr64`
  vs batch-split i32 na > 2 GiB. Wyniki wyłącznie w PR lub w pliku wyników
  zgodnym z polityką (`platform/AGENTS.md` §Compliance), nie w tym planie.

## 9. Dokumentacja (Definition of Done, `KERNEL_AUTHORING.md` §B/§C)
- Knob `addr64` w knob reference i Knob Catalog runbooka.
- `SUPPORT_MATRIX.md` / operation-support: wsparcie > 2 GiB per rodzina.
- `library/kernels/conv/docs/convolution.md`: sekcje „every tensor below 2 GiB”
  uzupełnić o wariant `addr64`.
- `docs/gfx950_conv_fwd/aot_instances.md` (provider): §„Which graphs an instance
  serves”, §„Current limits”.

## 10. Ryzyka i otwarte pytania
1. **TDM gfx1250:** limity pól deskryptora (wymiary, strides) — do sprawdzenia;
   może wymagać rebase per kafel także dla TDM.
2. **SGPR:** dodatkowe rsrc i origin i64 per operand (+4–6 SGPR/operand) —
   sprawdzić occupancy w ISA (`isa-inspection-rocke.md`) dla największych
   kafli.
3. **Lowering HIP:** czy wszystkie użyte op-y i64 mają lowering HIP; jeśli nie,
   `addr64` jest tylko LLVM (udokumentować).
4. **Rozmiar katalogu AOT:** oś `addr64` ograniczyć do podzbioru konfiguracji.
5. **Przypadki patologiczne** (footprint kafla > 2 GiB) są odrzucane; ewentualny
   ogólny kernel na `global_*` z adresem 64-bit w VGPR to osobna praca.
6. **Pozostające limity nie związane z rozmiarem tensora:** `Y*X*cpg < 2^23`
   (`mul_u24`), wymiary pojedyncze < 2^31 — bez zmian w tym planie.
