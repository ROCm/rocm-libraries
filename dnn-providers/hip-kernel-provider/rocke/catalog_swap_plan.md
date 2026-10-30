# Podmiana katalogu AOT gfx950 conv fwd na `instance_fwd_gfx950_new.txt`

Stan na 2026-10-29, branch `users/aledudek/gfx950-conv-aot-uhd` (repo `/home/barkocot/rocm3`).

## Commit

Scalono i wypchnięto na `users/aledudek/gfx950-conv-aot-uhd`: `18318f32980`
"feat(hip-kernel-provider): swap gfx950 conv fwd AOT catalog to 207 kernels/dtype,
add windowed direct win variant" (23 pliki, +3274/-3681).

## Postęp wykonania — sesja 2026-10-30 (Krok 3.3 fix + Krok 4 UHD)

### Naprawa `artifacts.dvc` (blokowało Krok 4 od commita `18318f32980`)

- **Problem**: `artifacts.dvc` dla `gfx950_conv_fwd_aot` wskazywał na niekompletny/przestarzały
  katalog `artifacts/*.kdp.json` (sprzed podmiany katalogu, bez wariantu "win", z
  niezgodnymi ID-kami dokumentów względem 40-polowego KMD z Kroku 2). Zdalny pipeline
  UHD, budując obraz silnika z czystego commita `18318f32980`, łapał dokładnie to:
  `hkp_pack: error: ... [gfx950: specialization fields must exhaustively partition the KMD]`
  (zweryfikowane w logu `sbatch_austin.out` zawiedzionego joba `f1ac40a9-...`).
- **Naprawa**: `dvc add artifacts` (nowy hash `394467292ba21e27fc55cf4ec09557cb.dir`, 16
  plików, 733106 B — katalog zgodny z nowym KMD, zremapowane ID-ki, przygotowany
  wcześniej lokalnie w working tree), `dvc push -r ingestor` (wymagało jednorazowego
  `dvc remote modify --local ingestor allow_anonymous_login false` w git-ignorowanym
  `.dvc/config.local` — committed `.dvc/config` ma `allow_anonymous_login = true`, więc
  bez override każdy zapis dostawał `AccessDenied` niezależnie od poprawności klucza AWS;
  opisane w `src/engines/kernel_ingestor_engine/descriptors/README.md:145-149`), commit
  `b6cdc991fd64c5eb421187097459fc0479b88265` (tylko wskaźnik `.dvc`), push na
  `users/aledudek/gfx950-conv-aot-uhd`.
- **Potwierdzone działa**: `build-hipdnn-image` na nowym SHA `b6cdc991fd64...` zakończył
  się `COMPLETED` (silnik `hipkernel:Gfx950ConvFwdAot` poprawnie się zarejestrował, bez
  błędu KMD) — zarówno w pierwszej próbie zgłoszenia, jak i w cache przy drugiej.

### Zgłoszenia treningu UHD (Krok 4) na nowym SHA

Parametry (stałe dla obu prób): `engine=hipkernel:Gfx950ConvFwdAot`,
`sku=gfx950-mi355x`, `ref=b6cdc991fd64c5eb421187097459fc0479b88265`, `tier=exhaustive`,
`corpus_operation=conv_fwd`.

1. **Próba 1** — `request_id 4fa999feae416003`, job `ae7fd642-...`. `build-hipdnn-image`
   OK. Padło na `uhd-corpus`: `model_shapes /home/barkocot/work/uhd_train/cases_conv_fwd_model_shapes.csv
   is not a file the pipeline can read` — ścieżka lokalna, pipeline czyta tylko pliki
   wgrane przez dashboard (`/projects/uhd_pipeline_cerb/uploads/...`).
2. **Próba 2** — plik wgrany przez usera na
   `/projects/uhd_pipeline_cerb/uploads/f5eafd7bbd31d2bf/graphs/cases_conv_fwd_model_shapes.csv`.
   `request_id 480d067b90665433`, job `c14bdff9-f609-43f3-ae2c-a2ba27045c78`.
   `build-hipdnn-image` cache hit (OK). Padło na `uhd-corpus` (GPU job, slurm job
   `68217131`): `Slurm accounting (sacct) unreachable for 1810s` —
   **czysto infrastrukturalne** (ta sama awaria dotknęła tego dnia inne joby w kolejce
   UHD pipeline'u, nie nasz kod/dane). `outcome: failed_infra`, retryowalne.
   Przypadkowo wywołany retry (job `4956a82d-...`) został **anulowany na żądanie
   użytkownika** — decyzja o ponowieniu zostaje do niego.

**Status**: Krok 4 zatrzymany tuż po starcie etapu `uhd-corpus` (pierwszy GPU etap tego
pipeline'u — wymaga realnego GPU gfx950, bo `hipdnn_corpus_gen` oferuje każdy
proponowany problem silnikowi na prawdziwym sprzęcie). Reszta pipeline'u: `uhd-collect`
(GPU), `uhd-generate` (CPU), `uhd-validate` (GPU).

**Próba awaryjna na innym SKU/klastrze (mi350x, Markham)**: `gfx950-mi355x` jest tylko
na klastrze Austin, `gfx950-mi350x` tylko na Markham (`list_clusters`) — różne klastry,
różny Slurm.

**Ważna korekta (wcześniej błędnie uogólnione w tej sesji)**: podczas *treningu* dataset/
corpus/model są trzymane ściśle per dokładny SKU (stąd osobne joby) — ale w **runtime**
hipDNN wybiera zainstalowany model po `uhd_id` zadeklarowanym **per arch**
(`declared_ids_by: "arch"`, potwierdzone dla `Gfx950ConvFwdAot` przez `get_uhd_engine`),
nie per SKU — sterownik zgłasza tylko arch (`gfx950`), nie dokładny model karty. Czyli
**jeden zainstalowany model faktycznie obsługuje oba SKU** w produkcji, niezależnie na
którym był trenowany; potwierdzone wprost w metadanych realnego packetu
(`get_uhd_packet` dla `Gfx950ConvDgradAot`): *"trained on gfx950-mi355x; the runtime
keys models by gfx950, so installed it also serves gfx950-mi350x"*. Konsekwencja: "działa"
≠ "dobrze wybiera" — model nigdy nie widział pomiarów z SKU, na którym go nie trenowano,
więc jego jakość tam jest nieprzetestowaną ekstrapolacją. Zwalidowany model z mi355x
*już teraz* obsłużyłby też mi350x bez osobnego treningu — drugi job (mi350x) nie jest
więc "wymagany do pokrycia obu SKU", tylko: (a) obejściem awarii Austin, żeby dostać
jakikolwiek model, (b) weryfikacją realnej jakości na mi350x zamiast ekstrapolacji. Jeśli
oba joby dojdą do końca, trzeba będzie **wybrać jeden do promocji** pod wspólny `uhd_id`
(instaluje się tylko jeden model na arch) — ten wybór wpłynie na jakość na obu SKU.
Złożone jako osobne zgłoszenie:
`request_id 396e4a41c337fe8e`, job `922ea0fe-5988-41be-981b-869cae7a0088`,
`sku=gfx950-mi350x`. `build-hipdnn-image` cache hit, `uhd-corpus` ruszył i faktycznie
działa (nie czeka w kolejce) — awaria `sacct` była specyficzna dla Austin, Markham nie
jest nią dotknięty.

**Decyzja użytkownika (po korekcie powyżej): porzucić mi355x, dokończyć tylko mi350x.**
Skoro model trenowany na jednym SKU i tak obsługuje oba w runtime (per-arch `uhd_id`),
nie ma sensu ciągnąć dwóch równoległych treningów — `c14bdff9-f609-43f3-ae2c-a2ba27045c78`
(mi355x, Austin, `FAILED`/`failed_infra`) **porzucony bez retry**. Po dokończeniu
(`uhd-collect` → `uhd-generate` → `uhd-validate` → packet) ten model, po walidacji, ma
zostać **promowany jako jedyny model dla `gfx950_conv_fwd_aot`** — obsłuży wtedy zarówno
mi350x, jak i mi355x.

**Ale mi350x też padł** (job `922ea0fe-5988-41be-981b-869cae7a0088`), i to szybko
(12:31→12:36), na etapie `uhd-corpus`, zanim korpus zdążył się realnie policzyć —
podczas kopiowania obrazu silnika na Markham przez cross-site broker:
```
RemoteJobError: sacct failed 15 times in a row ...
sacct: error: Unable to connect to database (ctr2-alola-ctrl-01:6819)
```
To **Slurm accounting na Markham**, inna baza niż ta, która zawiodła na Austin dla
mi355x — czyli to już nie jest problem specyficzny dla jednego klastra, wygląda na
szerszą awarię Slurm accounting obejmującą oba klastry w tym samym oknie czasowym
(2026-10-30, ok. 09:50–12:40 UTC). Retryowalne (`actions.retry: true`), ale **nie
wywołane automatycznie** — czeka na decyzję użytkownika, czy retry'ować teraz (ryzyko
złapania tej samej awarii ponownie) czy poczekać, aż accounting wróci.

Monitorowane cronem co 10 min (tylko ten job, `922ea0fe-...`).

### Próba lokalnego builda hipDNN (niezależny tor, dla Kroku 3.5 — testy C++ "win")

- Cel: skompilować lokalnie `Gfx950ConvFwdGeometry.hpp`/`Gfx950ConvFwdNative.cpp` (Krok 2,
  dotąd tylko ręcznie przejrzane + `g++ -fsyntax-only` na samym nagłówku), żeby dopisać i
  zweryfikować testy C++ geometrii dla wariantu "win"
  (`TestGfx950ConvFwdGeometry.cpp`/`TestGfx950ConvFwdAotPacks.cpp`) bez zgadywania
  wartości grid/block. **Niezależne od UHD** — UHD trenuje na obrazie zbudowanym przez
  zdalny pipeline (który już poprawnie kompiluje ten kod), nie na lokalnym buildzie.
- Standalone build `projects/hipdnn` (`cmake --preset release`, instalacja w drzewie,
  `build/release/installdir`, zero ryzyka dla `/opt`): konfiguracja przeszła po
  doinstalowaniu lokalnie `clang-format 18.1.4` (`/tmp/cf_bin`) i dodaniu
  `/opt/lib/llvm/bin` do PATH (clang-tidy 24 → tylko warning o niezgodności wersji,
  jak w zdalnym pipeline) oraz `-DALLOW_FETCH_DEPS=ON` (brakujący GTest, sieć dostępna).
- **Build padł** (nie w naszym kodzie) na `TestRuntimeLoadHeaderCpp20.cpp.o`:
  `-Werror` łapie błędy (`-Wshadow`, `-Wold-style-cast`) w **systemowych nagłówkach HIP**
  pod `/opt/include/hip/amd_detail/...` — bo lokalny ROCm to **7.1.1**
  (`ROCM_CMAKE_PATH /opt/rocm-7.1.1`), a branch oczekuje **7.14.0** (dokładnie to, czego
  używa kontener zdalnego pipeline'u `rocm/dev-ubuntu-24.04:7.14.0-full`, gdzie build
  przechodzi czysto). To ta sama rodzina niezgodności co blokada Kroku 3.5 z 2026-10-29
  (tam: clang-tidy 20 oczekiwane vs 24 znalezione w `/opt`'owym SDK).
- **Decyzja usera**: odpuścić lokalny build na razie — testy C++ dla "win" zostają jako
  jawnie oznaczony dług do zrobienia później (albo z poprawnym lokalnym ROCm 7.14.0,
  albo przez kogoś z dostępem do środowiska zgodnego z pipeline'owym obrazem).

## Postęp wykonania (ta sesja, zakres: Kroki 1–3.5)

Szczegółowy plan wykonawczy: `/home/barkocot/.claude/plans/distributed-munching-peacock.md`.

- [x] **Krok 1 — przeniesienie `direct_depthwise_win` (forward) z `fwd-impr-v4`**
  — zastosowano ręcznie zweryfikowany diff (~513 linii łącznie) na:
  - `library/kernels/common/conv_direct_grouped.py` (+453: `fwd`/`out_stride` na
    `DirectDepthwiseDgradWindowedSpec`, `dw_win_fwd_problem()`, zmiany w
    walidatorach i w `_build_dw_dgrad_win_mfma`/`_dw_win_byte_strides`),
  - `library/kernels/conv/direct_variants.py` (+18: stała, `make_direct_spec`,
    `validate_direct_spec`, `select_direct_kernel_builder`),
  - `library/kernels/conv/direct_aot_jobs.py` (+44: import, stałe gridu,
    `DIRECT_CAPABILITIES`, `_knob_grid`).
  - Weryfikacja na sucho (CPU, bez GPU): `make_direct_spec` +
    `validate_direct_spec` + `select_direct_kernel_builder` (czyli realne
    zbudowanie IR kernela) przechodzą dla depthwise (`cpg=1`), grupowego
    (`cpg=8`) i stride=2 — wszystkie trzy nowe ścieżki kodu w
    `_build_dw_dgrad_win_mfma` się kompilują do `KernelDef` bez błędu.
  - Dodatkowo przeniesiono (z tego samego diffu na `fwd-impr-v4`) nowy test
    `TestForwardNumerics` do `library/tests/test_conv_dgrad_depthwise_windowed.py`
    (porównanie z referencją NumPy przez forward ABI; brakowało go na naszym
    branchu, a bez niego nie było jak zweryfikować Kroku 1 end-to-end).
  - **Weryfikacja GPU wykonana (gfx950, MI350X)** — wszystko zielone:
    - `test_conv_dgrad_depthwise_windowed.py` — **19/19 testów, 75/75 subtestów**
      (w tym nowy `TestForwardNumerics`: 7/7 subtestów — depthwise i grupowy
      cpg∈{8,16,24,32}, stride 1 i 2, fp16 i bf16; zero regresji w istniejących
      testach dgrad/VALU/MFMA/non-finite).
    - `test_library_layering.py` — 4/4 (warstwy `kernels -> dispatch ->
      builders` nienaruszone).
    - `test_direct_conv_correctness.py` — 9 passed / 42 skipped (skip
      pre-istniejący, niezwiązany z tą zmianą; nic nowego nie failuje).
  - Env do odtworzenia: `ROCKE_LLVM_FLAVOR=llvm20` (loaded COMGR na tej
    maszynie to LLVM 20.0.0 — `llvm22` daje `ComgrError: datalayout mismatch`),
    `PYTHONPATH=$ROCKE/platform/python:$ROCKE/library`, pytest z `/tmp/rk_venv/bin/python`.
- [x] **Krok 2 — adapter hipDNN AOT dla "win"** (Python zrobiony i zweryfikowany
  na CPU; C++ napisany, nie skompilowany — brak w tej sesji pełnego
  środowiska budowania hipDNN):
  - **Python, `projects/hipdnn/tools/IngestorGenerator/tools/make_conv_aot_config.py`**:
    `DIRECT_KNOB_FIELDS` +`prefetch_rows`/`w_fold`, `DIRECT_VARIANT_FIELDS["win"]
    = ("block_h", "prefetch_rows", "w_fold", "cpg", "kpg")`, `ENGINE_KNOBS` +oba
    knoby, `PACKAGED_DIRECT_ALGORITHMS["direct_depthwise_win"] = "win"`,
    `_DIRECT_VARIANT_NAMES["win"]`, grupa `("win",)` w `build_config`.
  - **Python, `library/builders/common/convolution_forward.py`**:
    `DIRECT_VARIANT_WIN = "win"`, wpis w `AOT_DIRECT_ALGORITHMS`, nowe pola
    dataclassu `Gfx950ConvFwdAotSpec.prefetch_rows`/`.w_fold` (brakowało ich w
    ogóle — bez nich konstrukcja spec rzucała `TypeError`), wpis w
    `AOT_DIRECT_KNOBS`, `_AOT_DIRECT_KNOB_DEFAULTS` +oba knoby,
    `_AOT_DIRECT_EXTRA_FIELDS[WIN] = ("cpg", "kpg")` (win pakuje **oba**
    warianty — depthwise cpg=1 i grupowy cpg∈{8..64} — więc cpg/kpg muszą być w
    deskryptorze zawsze, nie tylko dla grupowego, inaczej dwie różne skompilowane
    binarki o różnym cpg byłyby nierozróżnialne), `aot_direct_instance_fields`
    +gałąź `elif variant == DIRECT_VARIANT_WIN`, i **poprawka wykryta w trakcie**:
    `_direct_caps()` zwracał domyślnie `(1, 1)` dla każdego wariantu poza
    `GROUPED`/halo — bez dopisania `DIRECT_VARIANT_WIN` tam, grupowy `win`
    (cpg>1) nigdy nie zostałby poprawnie odróżniony od depthwise `win` przy
    `gfx950_conv_fwd_aot_serves`.
  - **Weryfikacja Python (CPU, bez GPU)**: ręczny round-trip
    `identity → spec → identity` dla depthwise, grupowego (cpg=8) i stride=2 —
    wszystkie 3 się zgadzają; `gfx950_conv_fwd_aot_serves` poprawnie odrzuca
    instancję o złym cpg (depthwise nie serwuje grupowego problemu i odwrotnie).
    `test_convolution_forward_aot_packaging.py`: **92/92 passed** (87 istniejących
    + 5 nowych `test_win_serves_depthwise_and_grouped_stride_1_and_2` i 4 nowe
    parametryzacje w `test_out_of_contract_variant_knobs_are_declined`), po
    dodaniu brakującej gałęzi `DIRECT_VARIANT_WIN` w teście `_instance_problems`
    (bez niej 3 generyczne testy failowały — czysto fixture'owy brak, nie bug
    produkcyjny). `test_make_conv_aot_config.py` (hipDNN): w trakcie.
  - **C++, `Gfx950ConvFwdGeometry.hpp`/`Gfx950ConvFwdNative.cpp`**: nowa wartość
    enuma `DirectVariant::WIN`, parser `"win"`, nowe pola `DirectInstance.wFold`/
    `.prefetchRows`, nowa funkcja `windowedDirectLaunch` (replikuje siatkę
    `DirectDepthwiseDgradWindowedSpec.grid()` dla `fwd=True`: transponowany
    problem stride-1, `tile_w = 32/w_fold`, `block_ch = 8*block_waves`),
    wpis w switchu `directInstanceLaunch`, `AOT_DIRECT_KNOB_DEFAULTS` (18→20),
    `aotDirectVariantUses[WIN]`, `aotDirectInstance` wypełnia nowe pola. **Jedna
    poprawka wykryta w trakcie**: `isDepthwiseDirectVariant`/`aotScore` zakładały,
    że każdy "pure depthwise" wariant (ranking ponad implicit-GEMM) jest
    jednoznacznie określony przez `DirectVariant` — nieprawda dla `win`, które
    pakuje zarówno depthwise (cpg=1, powinien rankować jak tiled/col) jak i
    grupowy (cpg>1, powinien rankować jak grouped/nongrouped, neutralnie).
    Naprawione przez sprawdzenie `direct->cpg` w `aotScore` zamiast rozszerzać
    `isDepthwiseDirectVariant` o samo `WIN`.
    **Nie skompilowane** — plik wymaga pełnego środowiska SDK hipDNN
    (flatbuffers, hipdnn_plugin_sdk), czego nie ma w tej sesji; tylko
    `Gfx950ConvFwdGeometry.hpp` (samodzielny nagłówek) przeszedł
    `g++ -std=c++20 -fsyntax-only` czysto. Realna kompilacja i testy
    (`TestGfx950ConvFwdGeometry.cpp`, `TestGfx950ConvFwdAotPacks.cpp`) to
    Krok 3.5, na GPU.
  - **Testy C++** — dodano `parseDirectVariant("win")`/`("grouped_halo")` do
    `FamilyAndVariantValueSetsAreClosed` w `TestGfx950ConvFwdGeometry.cpp`
    (`grouped_halo` było pominięte już wcześniej — nie dodawałem pełnego zestawu
    testów geometrii dla "win" analogicznego do istniejących
    `DirectInstanceLaunchAppliesTheGuardAndTheVariantRules` — zbyt duży nakład
    bez możliwości kompilacji/weryfikacji w tej sesji; **do zrobienia przed
    merge'em**).
  - **Testy hipDNN (`test_make_conv_aot_config.py`)**: `_DIRECT_GRID["direct_depthwise_win"]
    = 759` (policzone z `enumerate_direct_jobs`), wpis `("win",)` w
    `test_direct_instances_are_variants_stating_their_own_fields`. **Nie
    dodano** "win" do `test_the_seed_runs_every_direct_variant_knob` ani do
    CSV-fixture'owego testu `AotDirectVariantsLaunchWhereAndHowTheAdapterDoes`-
    odpowiednika (linia ~1121, już dziś pomija `grouped_halo`) — oba wymagają
    realnych, przypiętych na GPU instancji seeda / złotego CSV, czego nie da się
    wygenerować bez GPU i pełnego builda; **do zrobienia osobno, nie blokuje
    Kroku 3**.
  - **Regeneracja artefaktu wykryta w trakcie**: dodanie `prefetch_rows`/`w_fold`
    do `ENGINE_KNOBS`/`DIRECT_KNOB_FIELDS` zmieniło globalną listę knobów
    silnika, co złamało `test_seed_config_is_what_the_tool_generates` (porównanie
    z zakomitowanym, wygenerowanym `configs/gfx950_conv_fwd_aot_seed.yaml`).
    Naprawione przez faktyczne odpalenie
    `tools/make_conv_aot_config.py --instances configs/gfx950_conv_fwd_aot_seed.instances.json
    --output configs/gfx950_conv_fwd_aot_seed.yaml --engine-name hipkernel:Gfx950ConvFwdAotSeed
    --authored-subpath rocKE/gfx950_conv_fwd_aot_seed --sharding dtype` i podmianę
    pliku (regenerowany, nie edytowany ręcznie, zgodnie z nagłówkiem "do not
    edit by hand") — diff ograniczony dokładnie do dwóch nowych wpisów knobów w
    3 miejscach (`engine.knobs`, `kmd_fields`, `specialization.metadata_fields`),
    zero instancji "win" w seedzie (bo `*.instances.json` go nie zawiera).
  - **Wynik końcowy Kroku 2 (Python)**: `test_convolution_forward_aot_packaging.py`
    92/92 passed; `test_make_conv_aot_config.py` (hipDNN) 63 passed, 2 skipped
    (pre-istniejący skip, niezwiązany).
- [x] **Krok 3.1 — podmiana `labels.txt` i regeneracja instancji**:
  podmieniono `gfx950_conv_fwd_aot.labels.txt` (stary zbackupowany pod
  `/tmp/gfx950_conv_fwd_aot.labels.txt.old`) na `instance_fwd_gfx950_new.txt`
  (414 linii). `enumerate_conv_aot_instances.py` — **kluczowa pułapka wykryta w
  trakcie**: domyślny `current_llvm_flavor()` w tym środowisku to `llvm20`
  (lokalny COMGR), ale etykiety w katalogu są spisane pod `llvm22` i
  `KernelIdentity.stable_hash()` **wlicza `llvm_flavor` do hasza** — więc
  walidacja pod `ROCKE_LLVM_FLAVOR=llvm20` failowała na WSZYSTKICH 414 liniach
  ("not a gfx950 forward sweep identity"), mimo że same identity są poprawne.
  Naprawione uruchomieniem pod `ROCKE_LLVM_FLAVOR=llvm22` (ta walidacja nic nie
  kompiluje, tylko liczy haszy w Pythonie, więc nie potrzebuje realnego COMGR
  llvm22) — **414/414 zwalidowane, 0 błędów**: `fp16 79 implicit + 128 direct`,
  `bf16` tak samo. Rozkład rodzin direct: `nongrouped 73, win 38, grouped_halo 9,
  tiled 4, col 3, spatial 1` (per dtype) — **potwierdza end-to-end, że port
  Kroku 1 + adapter Kroku 2 działają poprawnie na pełnym nowym katalogu**, nie
  tylko na syntetycznych przypadkach testowych.
- [x] **Krok 3.2 — `make_conv_aot_config.py`**: wygenerowano config z obu
  `*.instances.json` (`--engine-name hipkernel:Gfx950ConvFwdAot`,
  `--authored-subpath rocKE/gfx950_conv_fwd_aot`, `--sharding k-loop`, zgodnie z
  `aot_instances.md`) — **414/414 instancji zaakceptowane przez adapter, 0
  unsupported**.
- [x] **Krok 3.3 — `generate.py` + merge dokumentów** (częściowo, patrz niżej):
  wygenerowano bundle do `/tmp/gfx950_aot_gen` (30 plików: kmd/ued/udd×2/uhd/umd
  + 16 kdp.json + C++ scaffolding + fragmenty cmake). Odkryto, że
  `packs/Gfx950ConvFwdAotNative.cpp`/`tests/TestGfx950ConvFwdAotMatchers.cpp`
  to **świeże szkielety z placeholderami** (generator nie zna ręcznie
  dopisanej logiki C++) — **nie kopiować ich**, zgodnie z
  `aot_instances.md`: kopiuje się tylko 6 małych deskryptorów JSON + wygenerowany
  `tests/TestGfx950ConvFwdAotPacks.cpp`.
  - **Zachowanie ID dokumentów** (`aot_instances.md`: "Keep the document ids...
    so the diff shows only the kernels"): napisano mały skrypt scalający, który
    bierze nowo wygenerowaną treść, ale nadpisuje `"id"` starą wartością z
    zakomitowanego pliku, dla: kmd, udd, split_c.udd, uhd, umd. Dla **ued**
    dodatkowo zachowano `metadata` (referencja do KMD) i `sort_kernel_catalog`
    (wskaźniki na wytrenowany model UHD pod `heuristics/`) ze starego pliku —
    zaktualizowano **tylko** `knobs` (38→40, +`prefetch_rows`/`w_fold`), zgodnie
    z dokumentacją. Skopiowano `tests/TestGfx950ConvFwdAotPacks.cpp` (wygenerowane
    762 linie) i przepuszczono przez `clang-format 18.1.4` (doinstalowany lokalnie
    do `/tmp/cf_tool`, bo nie było w PATH) — diff ograniczony do realnej treści
    (nowy katalog), zero szumu formatowania.
  - **Czego NIE zrobiono, i dlaczego — blokuje to dopiero Krok 3.5, nie
    wcześniej**: katalog `artifacts/` (prawdziwe `*.kdp.json` z realnymi
    skompilowanymi kernelami, DVC-trackowany) **nie jest wyewidencjonowany w tym
    środowisku** (`dvc` w ogóle nie jest zainstalowane) — nie dało się pobrać
    starych `artifacts/*.kdp.json`, więc nie dało się zachować ich ID-ków
    analogicznie do powyższego merge'a 6 plików. To wymaga realnego środowiska
    z dostępem do zdalnego DVC (`dvc pull`, potem `dvc add`/`dvc push -r
    ingestor` po zbudowaniu). **Nie commitować / nie pushować niczego pod
    `artifacts/` bez tego** — ryzyko rozjechania ID-ków między dokumentami.
  - Przepuszczono `clang-format` też przez ręcznie edytowane pliki C++ z Kroku
    2: `Gfx950ConvFwdGeometry.hpp` wymagał reformatu (diff ograniczony
    dokładnie do mojego dodanego kodu), `Gfx950ConvFwdNative.cpp` i
    `TestGfx950ConvFwdGeometry.cpp` już były zgodne ze stylem.
- [x] **Krok 3.4 — zaktualizować testy/dokumenty ze sztywnymi liczbami**:
  - `test_enumerate_conv_aot_instances.py`: jedyne miejsce z faktycznie
    zahardkodowaną liczbą w kodzie — `test_the_catalog_lists_537_kernels_per_dtype_the_same_in_both`
    → `..._207_...`, `{True: 227, False: 310}` → `{True: 79, False: 128}`.
    **Pułapka wykryta w trakcie (niezwiązana z produktowym kodem)**:
    `test_main_writes_one_list_per_dtype_ordered_by_family_then_hash` failowało,
    bo bierze pierwsze 5 nie-implicit-gemm wpisów z PRAWDZIWEGO katalogu bez
    filtrowania po dtype, po czym `_both_dtypes()` nadpisuje dtype obu kopii —
    z nowym katalogiem dwa wpisy (ten sam kernel, różne oryginalne dtype)
    kolidowały w jeden identyczny label po nadpisaniu. Naprawione filtrowaniem
    `direct` do jednego dtype przed wywołaniem `_both_dtypes`.
  - `docs/gfx950_conv_fwd/aot_instances.md`: przeliczono i zaktualizowano z
    rzeczywistych danych (nie z szacunków) — tabelę rodzin (537/537 → 207/207,
    nowy wiersz `win` 38/38, zniknęły `std`/`grouped`), "That is 10,027 → 10,786
    identities per dtype" (tabela w sekcji "Direct instances", sweep-wide, +759
    za `win`), tabelę shardów KDP (nowe realne rozmiary z wygenerowanego configu:
    mem_plain 19, mem_unroll 13, compv3_plain 22, compv3_unroll 1, compv4_plain
    18, compv4_unroll 6, direct 99, direct_split 29), "Build and package cost"
    (1,074→414 wariantów, 182→58 split-C), tabelę szerokości load/store
    implicit-GEMM (przeliczoną z realnych instances.json, z wyłączeniem
    group-merged), "Current limits" (537→207, dopisano że model UHD nie zna
    `win`/`prefetch_rows`/`w_fold`). **Oznaczono jawnie jako nieaktualne (zamiast
    zgadywać)** dwie liczby wymagające realnego GPU: `aotGraphs()`
    candidate-selection (140 / 1-38 / 6-11) i `aotSeedDirectGraphs()`-podobne
    liczby (146/110 direct) — do zmierzenia dopiero w Kroku 3.5.
  - `test_convolution_forward_packaging.py` (exact-shape engine, NIE AOT) —
    potwierdzone: 0 odniesień do liczb katalogu AOT, nic do zmiany (ma tylko
    std/spatial, niezależnie od katalogu AOT).
  - `.pre-commit-config.yaml`/`artifacts.dvc`/`heuristics.dvc` — bez zmian
    ręcznych (potwierdzone wcześniej: regexy count-agnostic); `.dvc` pliki
    wymagają realnego `dvc add`/`dvc push` po zbudowaniu w Kroku 3.5 (DVC nie
    jest nawet zainstalowane w tej sesji).
  - **Druga pułapka wykryta w trakcie, poważniejsza**: `test_gpu_suite_selected_counts_are_the_python_runtime_candidates`
    failowało (`(11, 4, 10, 53) == (11, 4, 24, 140)` dla grafu "Smoke") —
    okazało się, że `IntegrationGpuGfx950ConvFwd.cpp` (realny plik C++ testu
    integracyjnego GPU) ma **wbudowaną w kod tabelę `aotGraphs()`** z 26
    ręcznie wyliczonymi wierszami (nazwa grafu, problem, oczekiwana liczba
    wybranych instancji direct/implicit-gemm dla seeda i dla pełnego katalogu,
    per dtype) — Python test parsuje ten plik C++ regexem i porównuje z tym,
    co faktycznie liczy adapter. To nie jest "dokumentacja do poprawienia
    później" — to aktywny fixture testowy, który musiał być poprawny od razu.
    Napisano skrypt (`tool.instance_specs` + `gfx950_conv_fwd_aot_runtime_candidates`
    na realnych `configs/gfx950_conv_fwd_aot_{fp16,bf16}.instances.json`),
    który przeliczył wszystkich 26 wierszy i podmienił tylko 3 końcowe pola
    liczbowe (`direct`, `implicitFp16`, `implicitBf16`) w `IntegrationGpuGfx950ConvFwd.cpp`,
    zostawiając `name`/`problem`/`seedDirect`/`seedImplicit` bez zmian (seed
    niezmieniony). Diff ograniczony dokładnie do 26 linii po `clang-format`.
    **Test teraz zielony** (`test_gpu_suite_selected_counts_are_the_python_runtime_candidates`
    PASSED). To **jednocześnie odpowiada** na wcześniej oznaczone w dokumentacji
    "stale" liczby (140/1-38/6-11, 146/110) — realne nowe wartości są teraz w
    kodzie C++ (np. Smoke: direct 10, implicit 53/53 zamiast 24/140/140); treść
    prozy w `aot_instances.md` nadal mówi "stale, do zmierzenia" w dwóch miejscach
    bo te konkretne zdania opisują AGREGATY (suma po wszystkich grafach), nie
    pojedyncze wiersze — warto to zaktualizować przy realnym teście GPU w Kroku
    3.5, skoro i tak trzeba będzie go uruchomić.
  - Sekcja "The seed engine and the GPU tests" — **celowo nietknięta**: opisuje
    `gfx950_conv_fwd_aot_seed` (220 instancji), którego zawartości nie
    zmieniliśmy (tylko listę knobów w jego wygenerowanym yaml, patrz Krok 2) —
    nie zawiera "win" i nie musi, dopóki nie dojdzie realny seed dla "win" (praca
    poza zakresem tej sesji, odnotowana wyżej przy Kroku 2).
- [x] **Krok 3.5 — próba buildu C++ na GPU: zablokowana infrastrukturalnie,
  nie przez kod** (GPU samo w sobie jest wolne i dostępne — sprawdzone
  `rocm-smi`, MI350X, 0% użycia poza niegroźnym procesem z wcześniejszego
  testu).
  - Znaleziono zainstalowany system-wide hipDNN SDK pod `/opt` (`hipdnn_plugin_sdk`,
    `hipdnn_data_sdk`, `/opt/bin/hip_kernel_provider_tests` już istnieje — ktoś
    kiedyś to zbudował). Skonfigurowano fresh out-of-tree build
    (`cmake -GNinja -DCMAKE_PREFIX_PATH=/opt -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON`
    w `/tmp/hkp_build`), doinstalowano lokalnie `clang-format 18.1.4` (wymagany
    przy konfiguracji, brak w PATH) i obszedł się brak `GTest::gmock` przez
    oficjalną flagę `-DALLOW_FETCH_DEPS=ON` (sieć dostępna, pobrało się OK).
  - **Twardy bloker**: `HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON` — plik,
    który musi dostarczyć `hipdnn_plugin_sdkConfig.cmake` zainstalowanego SDK —
    jest pusty w instalacji pod `/opt`. Plik źródłowy istnieje w repo
    (`projects/hipdnn/plugin_sdk/data/gpu_generic_targets.json`), ale
    `CMakeLists.txt:264` **celowo** robi `unset(... CACHE)` tuż przed
    `find_package(hipdnn_plugin_sdk)`, żeby nie dało się podsunąć wartości z
    linii poleceń (`-D`) ani starej wartości z cache — to świadome
    zabezpieczenie przed dokładnie tym obejściem, które próbowałem. Oznacza to,
    że **zainstalowany pod `/opt` SDK jest starszy niż to, czego oczekuje
    obecne źródło `hip-kernel-provider`** (brakuje mu funkcji, którą ten branch
    już zakłada) — niespójność wersji, nie coś do obejścia bezpiecznie bez
    edycji współdzielonego systemowego pliku pod `/opt` (czego celowo nie
    zrobiłem — ryzyko dla innych użytkowników maszyny).
  - **Właściwa naprawa** (poza zakresem tej sesji): zbudować hipDNN od zera z
    `projects/hipdnn` (`ninja install` do świeżego prefiksu) zgodnie z
    `dnn-providers/hip-kernel-provider/README.md`, i budować providera
    przeciw TEJ instalacji, nie przeciw `/opt`. To osobny, duży build
    (cały hipDNN, nie tylko provider), nie mieści się rozsądnie w budżecie tej
    sesji.
  - **Co to NIE blokuje**: cały Krok 1 (rocKE kernel), Krok 2 (adapter Python +
    kod C++ napisany i przejrzany ręcznie, `Gfx950ConvFwdGeometry.hpp`
    zweryfikowany samodzielnym `g++ -fsyntax-only`), Krok 3.1–3.4 (podmiana
    katalogu, regeneracja, wszystkie testy Python zielone, w tym live-fixture
    `IntegrationGpuGfx950ConvFwd.cpp`'s `aotGraphs()` przeliczone i
    zweryfikowane przez `test_make_conv_aot_config.py`) — to wszystko jest
    zrobione i zweryfikowane tak dokładnie, jak dało się bez pełnego C++
    builda. Nieskompilowane pozostają: `Gfx950ConvFwdNative.cpp`/
    `Gfx950ConvFwdGeometry.hpp` jako część realnego `.so`, i stąd też
    `TestGfx950ConvFwdAotPacks.cpp`/`TestGfx950ConvFwdGeometry.cpp` na GPU,
    `artifacts/` DVC (patrz Krok 3.3), realny `IntegrationGpuGfx950ConvFwd`
    na GPU.

## Sytuacja

Branch integruje silnik `hipkernel:Gfx950ConvFwdAot` (katalog AOT kerneli conv fwd na
gfx950) z hipDNN, razem z heurystyką UHD. Obecny, wysłany katalog to 537 kerneli na
dtype (fp16/bf16), wyselekcjonowany jako top-1 zwycięzców sweepa rocKE
(`e6806ca7d98`, `8c91c8ee005`).

Źródło prawdy katalogu to lista etykiet (`KernelIdentity.label()`, jedna na linię):

```
/home/barkocot/rocm3/projects/hipdnn/tools/IngestorGenerator/configs/gfx950_conv_fwd_aot.labels.txt
```

z niej `tools/enumerate_conv_aot_instances.py` generuje
`configs/gfx950_conv_fwd_aot_{bf16,fp16}.instances.json`, a z tych z kolei
`tools/make_conv_aot_config.py` generuje config dla `generate.py`, który emituje
KDP-y (`gfx950_conv_fwd_aot_seed_*.kdp.json` i pełne, w DVC pod `artifacts/`).

Plik `instance_fwd_gfx950_new.txt` (w `rocke/`, nieśledzony) jest **w dokładnie tym
samym formacie** (`KernelIdentity.label()`) — to kandydat na nową wersję
`gfx950_conv_fwd_aot.labels.txt`, prawdopodobnie wynik nowszego/innego sweepa albo
selekcji.

## Porównanie: stary katalog vs `instance_fwd_gfx950_new.txt`

| | stary (`labels.txt`) | nowy (`instance_fwd_gfx950_new.txt`) |
|---|---|---|
| razem (oba dtype) | 1074 (537 × 2) | 414 (207 × 2) |
| wspólne z obydwoma | 262 | 262 |
| tylko w starym (do usunięcia) | 812 | — |
| tylko w nowym (do dodania) | — | 152 |

Rozkład rodzin tylko-w-starym (do usunięcia):

```
 10 fwd-direct_depthwise          (std)
  4 fwd-direct_depthwise_spatial
150 fwd-direct_depthwise_tiled
  6 fwd-direct_grouped
 78 fwd-direct_grouped_halo
242 fwd-direct_nongrouped
322 fwd-implicit_gemm
```

Rozkład rodzin tylko-w-nowym (do dodania):

```
  6 fwd-direct_depthwise_col
 76 fwd-direct_depthwise_win   <-- BLOKER, patrz niżej
  6 fwd-direct_grouped_halo
 38 fwd-direct_nongrouped
 26 fwd-implicit_gemm
```

## WYJAŚNIONE: `direct_depthwise_win` istnieje — na branchu `users/barkocot/fwd-impr-v4`

76 linii nowego pliku (38 na dtype) ma rodzinę `fwd-direct_depthwise_win-...`. Na
branchu `users/aledudek/gfx950-conv-aot-uhd` (ten, na którym pracujemy) ten algorytm
faktycznie nie istnieje dla forward — jedyny wpis to `direct_depthwise_dgrad_win`
(backward-data). Ale sprawdzone na `users/barkocot/fwd-impr-v4`
(`9fb189d660d`, zweryfikowane przez `git grep` na `FETCH_HEAD` bez checkoutu):

- `library/kernels/conv/direct_variants.py`: stała
  `DIRECT_DEPTHWISE_WIN_FWD = "direct_depthwise_win"`, wpis w builder-mapie
  (`dc.build_direct_depthwise_dgrad_windowed` — ten sam kernel co dgrad-owy
  windowed depthwise, forward pass na obróconym filtrze) i w walidatorze
  (`direct_spec_for_problem` akceptuje `("direct_depthwise_dgrad_win",
  DIRECT_DEPTHWISE_WIN_FWD)` razem).
- `library/kernels/conv/direct_aot_jobs.py`: stałe gridu (`DW_WIN_FWD_FILTERS = (3,
  5, 7, 9)`, `DW_WIN_FWD_G8_FILTERS`, `DW_WIN_FWD_GROUP_CH`) i wpięcie w
  `enumerate_direct_jobs` (branch `elif variant == _DW_WIN_FWD`) — sweep faktycznie
  go enumeruje, więc `enumerate_conv_aot_instances.py` przyjmie te etykiety.
- Test `test_conv_dgrad_depthwise_windowed.py` ćwiczy `DIRECT_DEPTHWISE_WIN_FWD`
  przez `direct_spec_for_problem`.

Czyli `instance_fwd_gfx950_new.txt` nie jest pomyłką — pochodzi z/dla
`fwd-impr-v4`, gdzie ten kernel jest po stronie rocKE kompletny. Blokerem jest
**tylko** to, że integracja hipDNN na branchu `gfx950-conv-aot-uhd` (gdzie pracujemy)
go jeszcze nie zna.

**Dobra wiadomość o konfliktach:** `fwd-impr-v4` rozszedł się z naszym branchem na
wspólnym przodku `52cfb30a72f`. Od tego punktu:
- `direct_variants.py` / `direct_aot_jobs.py` — **zero zmian na naszym branchu**
  (diff `fwd-impr-v4` vs wspólny przodek: +61/-1 linii, tylko tam). Cherry-pick albo
  ręczne przeniesienie tych dwóch plików (a raczej tylko fragmentu dot.
  `direct_depthwise_win`) będzie bezkonfliktowe.
- Za to `fwd-impr-v4` to duży, osobny branch refaktoryzujący generator
  IngestorGeneratora (`codegen/generator.py` +830 linii, nowy `codegen/prior_ids.py`
  +498 linii, `generate.py` +300 linii — rework stabilizacji ID i inne rzeczy
  niezwiązane z katalogiem gfx950). Nasz branch też dotyka `generator.py` (+43/-16).
  **Nie merge'ować/rebase'ować całego `fwd-impr-v4`** — to osobna, duża zmiana z
  własnym ryzykiem; trzeba **cherry-pickować / ręcznie przenieść tylko** zmiany w
  `direct_variants.py` i `direct_aot_jobs.py` dot. `direct_depthwise_win`.
- `fwd-impr-v4` w ogóle nie dotyka plików hipDNN specyficznych dla tego katalogu
  (`make_conv_aot_config.py`, `convolution_forward.py`, `gfx950_conv_fwd_aot*`
  configi) — ta strona integracji (Ścieżka B krok 3 niżej) to w każdym razie praca
  do zrobienia na nowo, niezależnie od merge'a.

**Plan spłaszcza się do jednej ścieżki (dawnej Ścieżki B, bez kroku "napisać kernel
od zera" — tylko "przenieść go")** — patrz zaktualizowane kroki niżej.

## Plan działania

### Krok 1 — przenieść `direct_depthwise_win` z `fwd-impr-v4` na nasz branch
Nie merge'ować/rebase'ować całego `fwd-impr-v4` (niesie niezwiązany, duży rework
generatora). Zamiast tego ręcznie przenieść (cherry-pick fragmentu albo
diff/patch) tylko zmiany dot. `direct_depthwise_win` w:
- `library/kernels/conv/direct_variants.py` (+18 linii na `fwd-impr-v4`: stała
  `DIRECT_DEPTHWISE_WIN_FWD`, wpis w builder-mapie, w walidatorze),
- `library/kernels/conv/direct_aot_jobs.py` (+44 linie: stałe gridu, wpięcie w
  `enumerate_direct_jobs`).

Oba pliki są nietknięte na naszym branchu od wspólnego przodka (`52cfb30a72f`) —
przeniesienie powinno być bezkonfliktowe (`git diff 52cfb30a72f fwd-impr-v4 -- <te
pliki> | git apply` albo ręczne skopiowanie fragmentów).

Do zweryfikowania przy przenoszeniu: czy `build_direct_depthwise_dgrad_windowed`
(współdzielony builder dgrad/fwd) wymaga też zmian poza tymi dwoma plikami — sam
`git grep DIRECT_DEPTHWISE_WIN_FWD` na `fwd-impr-v4` dał tylko te dwa pliki + test,
ale builder sam w sobie (`dc.build_direct_depthwise_dgrad_windowed`) już istnieje
na naszym branchu (to ten sam co dla dgrad), więc nie powinno być tu dodatkowej
roboty — ale warto to potwierdzić, uruchamiając testy korektności dla fwd po
przeniesieniu (`test_conv_dgrad_depthwise_windowed.py` i ewentualny
`test_direct_conv_correctness.py`).

### Krok 2 — dopiąć stronę hipDNN AOT (nowa rodzina w integracji)
`fwd-impr-v4` nie dotyka plików hipDNN, więc to trzeba zrobić od zera, analogicznie
do istniejącego wariantu `col`/`tiled`:
- `tools/make_conv_aot_config.py`: dodać `"direct_depthwise_win": "win"` (czy pod
  inną nazwą wariantu) do `PACKAGED_DIRECT_ALGORITHMS`, wpis w
  `DIRECT_VARIANT_FIELDS` (jakie `block_*`/`prefetch_rows`/`w_fold` knoby niesie —
  widać je w etykietach: `block_h`, `block_waves`, `prefetch_rows`, `w_fold`) i w
  `_DIRECT_VARIANT_NAMES`.
- `builders/common/convolution_forward.py`: wpis w
  `gfx950_conv_fwd_aot_spec_for_identity` (i ewentualnie `..._serves` /
  `..._batch_chunks`, jeśli windowed depthwise ma inne reguły batch-split niż
  pozostałe depthwise).
- C++: `Gfx950ConvFwdNative.cpp` / `IngestorPacks.*` — nowa ścieżka launchu, jeśli
  kernel ma inny układ argumentów niż istniejące direct warianty (prawdopodobnie
  podobny do `tiled`/`col`, bo to też depthwise, ale do potwierdzenia przy kodzie).
- Testy: `test_make_conv_aot_config.py`, `test_convolution_forward_aot_packaging.py`,
  `TestGfx950ConvFwdAotPacks.cpp`, `TestGfx950ConvFwdGeometry.cpp` — nowe przypadki
  dla `win`.

### Krok 3 — podmiana katalogu, rozbita na podkroki

#### 3.1 — podmienić źródło prawdy i wygenerować nowe listy instancji
1. Podmienić `configs/gfx950_conv_fwd_aot.labels.txt` na
   `instance_fwd_gfx950_new.txt` (414 linii).
2. `python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" --output-dir configs`
   — zweryfikuje każdą etykietę wobec gridu sweepa (teraz już zna `direct_depthwise_win`
   dzięki Krokowi 1) i zapisze nowe `*.instances.json`.

#### 3.2 — wygenerować config generatora i sprawdzić zgodność z adapterem
3. `python tools/make_conv_aot_config.py --rocke-root "$ROCKE" --instances configs/gfx950_conv_fwd_aot_{fp16,bf16}.instances.json --output <config>`
   — sprawdzi, że adapter (`gfx950_conv_fwd_aot_spec_for_identity`, zaktualizowany
   w Kroku 2) akceptuje każdą z 414 instancji.

#### 3.3 — wygenerować i zestabilizować KDP-y
4. `generate.py --config <config> --output-dir <dir> --force`.
5. `make_conv_aot_config.py --stabilize-ids` na wygenerowanych KDP-ach, żeby
   kernel UUID-y zostały wyprowadzone z `instance_id` (stabilne między
   regeneracjami, bo autotune-owy zwycięzca jest kluczowany po UUID kernela).

#### 3.4 — zaktualizować testy i dokumenty ze sztywnymi liczbami starego katalogu
6. `git grep -n "537\b"` (i podobne: listy konkretnych etykiet/rodzin) w:
   `test_make_conv_aot_config.py`, `test_convolution_forward_packaging.py`,
   `TestGfx950ConvFwdAotPacks.cpp`, `TestGfx950ConvFwdAotSeedPacks.cpp`,
   `aot_instances.md`, `gfx950_conv_fwd_aot.profile.yaml`.
7. Sprawdzić, czy usunięcie całych rodzin `direct_depthwise` (std) i
   `direct_grouped` (non-halo) z katalogu psuje któryś test, który explicite
   zakłada ich obecność (obsługa w adapterze jest generyczna, więc same się nie
   wywalą — ale testy z twardo wpisanymi etykietami tych rodzin tak).

#### 3.5 — przebudować i przepuścić testy integracyjne C++ na GPU
8. Build + `IntegrationGpuGfx950ConvFwd.cpp`, `TestGfx950ConvFwdAotPacks.cpp` na
   gfx950 — realnie kompilują i uruchamiają kernele z nowego katalogu (w tym nowy
   wariant `win` z Kroku 2).

#### 3.6 — checkpoint: autotune `cases.txt` na nowym katalogu, tylko rocKE — ZATRZYMAĆ SIĘ TU
Zanim ruszy trening UHD (najdroższy i najwolniejszy krok), zmierzyć nowy katalog
na realnych kształtach z `cases.txt` (corpus conv-kpi, fwd+dgrad fp16/bf16, NHWC —
patrz `[[project-conv-kpi-hipdnn-bench]]`), **bez MIOpen**, żeby dostać szybki,
samodzielny sygnał "czy nowy katalog w ogóle dobrze kryje te kształty", zanim
zainwestuje się czas w pełny UHD retrening:

1. Zbudować/zaktualizować plugin hipDNN z nowego katalogu (3.1–3.5) i zainstalować
   go w środowisku `dnn-benchmark` (ten sam setup co poprzednio:
   `~/work/dnnb-uhd` / `~/work/reval`, wzorem
   `conv_bench/new_catalog/run.sh` z 2026-10-22 — ale na nowym, 414-kernelowym
   katalogu, nie na starym 886/537).
2. Jeśli `fwd_graphs.txt` z poprzedniego runu (ten sam `cases.txt`→graphs przez
   `dnn-convert-shapes`) jest nadal aktualny, użyć go; inaczej przekonwertować
   `cases.txt` na graphy na nowo.
3. Odpalić **tylko**:
   ```
   dnn-benchmark -g $(cat fwd_graphs.txt) -e hipkernel:Gfx950ConvFwdAot --autotune \
     --hipdnn-cache-dir <fresh-cache> -o results_cases_new_catalog.json
   ```
   (bez `-e MIOPEN_ENGINE,...` i bez `--validate pytorch` heurystycznego runu —
   zgodnie z prośbą: tylko rocKE, bez MIOpen).
4. Podsumować wynik (podobnie do `conv_bench/new_catalog/compare_new.py`, ale bez
   kolumny MIOpen): ile grafów rocKE w ogóle serwuje (identity/applicability nowego
   katalogu), ile przechodzi walidację PyTorch/autotune, rozkład czasu po
   rodzinach kernela (szczególnie nowy `win`), ile grafów nie ma żadnego
   serwującego kernela (dziura w pokryciu, wymagałaby dodania etykiet do
   `instance_fwd_gfx950_new.txt`, nie czekania na UHD).
5. **Zatrzymać się tutaj** — pokazać te wyniki i czekać na decyzję, zanim ruszy
   krok 4 (UHD). Jeśli pokrycie/autotune pokaże dziury, taniej je załatać na
   poziomie katalogu (dodać/poprawić etykiety) niż odkryć je dopiero po
   kosztownym treningu UHD.

#### 3.6 — wyniki (sesja 2026-10-30, katalog `instancesv3.txt`, 466 kerneli/dtype)

Odpalono dokładnie zgodnie z recepturą wyżej, bez MIOpen i bez `--validate pytorch`:

```
source ~/work/dnn-benchmarking/.workspace/.venv/bin/activate
export LD_LIBRARY_PATH=/home/barkocot/work/aot-prefix-rocm3/lib:$LD_LIBRARY_PATH
P=/home/barkocot/work/aot-prefix-rocm3/lib/hipdnn_plugins/engines
dnn-benchmark -g $(cat ~/work/reval/fwd_graphs.txt) -e hipkernel:Gfx950ConvFwdAot \
  --autotune --plugin-path $P \
  --hipdnn-cache-dir ~/work/aot_cases/fwd_cache_v3 \
  -o ~/work/aot_cases/results_cases_new_catalog_v3.json
```

- **2013/2013 grafów zmierzone, 0 błędów, 0 skipped, `no_engine_graphs: 0`** — **pełne
  pokrycie**: każdy graf z `cases.txt`→`fwd_graphs.txt` (corpus conv-kpi) ma co
  najmniej jedną serwującą instancję w nowym 466-kernelowym katalogu. Zero dziur
  do łatania na poziomie katalogu przed Krokiem 4.
- TFLOPS (z `ootb.tflops`, 2013 punktów): mediana **252.4**, p10 **7.8**, p90
  **856.2**, max **1170.4** — szeroki rozkział bo `cases.txt` miesza bardzo różne
  kształty (depthwise niskiej intensywności arytmetycznej vs duże dense conv).
- **Rozkład zwycięzców autotune po rodzinie kernela** (z cache'u autotune
  `fwd_cache_v3/ingestor-winners/.../winners.jsonl`, per-graf najszybszy z
  realnie zmierzonych kandydatów, średnio ~60 kandydatów/graf, zakres 1–108):

  | rodzina | grafów | udział |
  |---|---:|---:|
  | `implicit_gemm` | 946 | 47.0% |
  | `direct_nongrouped` | 708 | 35.2% |
  | `direct_win` (nowa, z Kroku 1) | 249 | 12.4% |
  | `direct_col` | 45 | 2.2% |
  | `direct_grouped_halo` | 34 | 1.7% |
  | `direct_tiled` | 30 | 1.5% |
  | `direct_spatial` | 1 | 0.0% |

  Nowa rodzina `direct_win` realnie wygrywa na **12.4%** grafów `cases.txt` — nie
  jest martwym dodatkiem, konkretnie bierze udział w autotune i czasem wygrywa.
  `direct_nongrouped`+`implicit_gemm` razem to >80% zwycięstw, zgodnie z
  oczekiwaniem (te dwie rodziny mają najszerszy sweep w katalogu).
- **Checkpoint zamknięty — czekam na decyzję przed Krokiem 4** (trening UHD).
  Brak dziur w pokryciu oznacza, że nie trzeba nic dokładać do
  `instancesv3.txt`/`labels.txt` przed treningiem; `direct_win` jest już realnie
  użyteczna, więc trening UHD dla tego katalogu ma sens, jeśli user da zielone
  światło.

### Krok 4 — trening UHD (dopiero po akceptacji wyników Kroku 3.6)

UHD retrening to osobna, wielostopniowa praca na serwerze UHD pipeline
(`mcp__uhd__*` / skille `uhd-generate`, `uhd-data`, `uhd-evaluate`,
`uhd-promote-validate`), nie jednorazowa komenda:

#### 4.1 — przygotować corpus: tier `exhaustive` (5000/operację) + `cases.txt` jako dodatkowe kształty
- `cases.txt` jest w formacie MIOpenDriver (`convbfp16 -F 1 -n ... -c ... -k ...`),
  nie w formacie CSV `op,q.<parametr>`, którego oczekuje `corpus_gen`
  (`model_shapes` w `trigger_uhd_corpus`/`preview_heuristics_request`). Potrzebny
  mały konwerter MIOpenDriver-args → CSV (per operację: kolumny `q.n`, `q.c`,
  `q.k`, `q.filter_h/w`, `q.pad_h/w`, `q.stride_h/w`, `q.dilation_h/w`, `q.groups`,
  `q.dtype`/`q.layout` jeśli są deklarowanymi parametrami engine'u) — **do
  napisania, nic takiego nie ma dziś w repo** (sprawdzone: brak
  `model_shapes`/`q\.` w `rocke`/`IngestorGenerator`). To osobna, mała działka
  roboty przed samym `trigger_uhd_corpus`.
- `trigger_uhd_corpus(engine="hipkernel:Gfx950ConvFwdAot", sku=<gfx950 SKU>,
  tier="exhaustive", model_shapes=[<cases.txt.csv>], container_image=<obraz z
  nowym katalogiem>)` — `exhaustive` daje 5000 grafów/operację, floor 250/regime,
  served extremes; `model_shapes` dokłada kształty z `cases.txt` jako własną grupę
  (osobno mierzoną i ocenianą, nie tylko trenowaną na niej).
- Sprawdzić `corpus.plan.notes` — czy obraz silnika ma wszystkie opcje `corpus_gen`
  tier-a (jeśli nie, część ustawień tier-a jest pomijana i to widać w planie).

#### 4.2 — zmierzyć corpus na silniku (collect) i przekonwertować do datasetu
- `trigger_uhd_collect(engine=..., corpus_id=<z 4.1>)` — GPU job, mierzy każdy graf
  na silniku, zapisuje surowe kolekcje i od razu konwertuje do datasetu
  (`list_uhd_datasets`). To najdłuższy krok w GPU-wall-clock (zależnie od liczby
  shardów; 5000+ grafów/operację × kilka operacji to wielogodzinny bieg na
  klastrze Slurm pipeline'u UHD, nie liczony tu w dniach-osobach).

#### 4.3 — wytrenować model i ocenić na held-out
- `trigger_uhd_generate(engine=..., sku=...)` — trenuje `sort_kernel_catalog` (L2,
  jeśli silnik jest deskryptorowy) i/lub predyktor silnika (L1) z zebranego
  datasetu, od razu buduje packet modelu i liczy held-out score na fixed eval
  slice (w tym osobno na grupie `cases.txt`, dzięki 4.1).
- `get_uhd_coverage` / `get_uhd_eval_report` — sprawdzić pokrycie per-regime i czy
  kształty z `cases.txt` (real-workload) są w ogóle wytrenowane/ocenione, nie
  tylko ogólny wynik.

#### 4.4 — walidacja promocyjna (Stage V) na GPU
- `trigger_uhd_validate(model_id=...)` — instaluje kandydata jak do wysyłki,
  mierzy go na próbce held-out, porównuje z aktualnie wysłanym modelem
  (`max_regression`). To kolejny GPU job.

#### 4.5 — (opcjonalnie, zależnie od wyniku 4.3/4.4) rundy wzrostu (`growth_rounds`)
- Jeśli `tier="exhaustive"` (2 rundy wzrostu) albo ręcznie
  `trigger_uhd_size` → `trigger_uhd_corpus(sizing_id=...)` →
  `trigger_uhd_collect` → `trigger_uhd_generate` → `trigger_uhd_validate`
  (`trigger_uhd_grow`) — każda runda to kolejny cykl GPU + retrening, w regimach,
  które model słabiej pokrywa.

#### 4.6 — pobranie i aplikacja packetu
- `get_uhd_packet(model_id)` → `download_script` (pokazać userowi + ostrzeżenia)
  → dopiero po akceptacji `apply_script` (`git am -3` patchy) w checkoucie.
- Zaktualizować DVC (`artifacts.dvc`/`heuristics.dvc`: nowe KDP-y + nowy
  `model.bin`) i `.pre-commit-config.yaml`, jeśli wskazuje stare liczby.

## Szacowany koszt

| Krok | Szacunek (osobo-dni, 1 inżynier + agent) | Główne ryzyko |
|---|---|---|
| **Krok 1** — przenieść `direct_depthwise_win` z `fwd-impr-v4` | **0.5 dnia** | niskie — pliki nietknięte na naszym branchu, bezkonfliktowe; trzeba tylko przepuścić testy korektności fwd |
| **Krok 2** — dopiąć nową rodzinę po stronie hipDNN (adapter + generator config + C++ + testy) | **1–2 dni** | główna niewiadoma kosztu: czy C++ launch dla `win` różni się istotnie od `tiled`/`col`, czy wystarczy nowy wariant w istniejącym wzorcu |
| **Krok 3.1–3.5** — podmiana katalogu, regeneracja, aktualizacja testów ze sztywnymi liczbami | **1 dzień** | aktualizacja wielu miejsc ze sztywną liczbą 537; GPU do przebudowy/testów integracyjnych |
| **Krok 3.6** — autotune `cases.txt` na nowym katalogu (rocKE only) + podsumowanie | **0.5 dnia** (głównie GPU-wall-clock autotune na ~2000 grafów, kolejność godzin, nie dni-osoby) | **checkpoint — zatrzymać się i czekać na decyzję** przed Krokiem 4 |
| **Krok 4** — trening UHD pełny (konwerter cases.txt→CSV, corpus exhaustive 5000+cases.txt, collect, generate, validate, ew. growth rounds, packet) | **2–4 dni pracy inżyniera** (głównie pisanie konwertera, czytanie coverage/eval raportów, iteracja) **+ osobno kilkanaście godzin GPU-wall-clock** w kolejce Slurm UHD pipeline'u (collect na 5000+/operację to realnie najdłuższy pojedynczy krok) | brak gotowego konwertera MIOpenDriver→CSV `model_shapes`; długość kolejki Slurm; możliwe rundy wzrostu, jeśli held-out nie przejdzie progu za pierwszym razem |
| **Razem do Kroku 3.6 (checkpoint)** | **~2–3.5 dnia** | |
| **Razem łącznie z Krokiem 4** | **~4.5–7.5 dnia pracy + GPU-wall-clock w tle** | |

W porównaniu z poprzednim szacunkiem (4–8 dni dla "nowego kernela od zera") koszt
samej integracji (Kroki 1–3) spada mniej więcej o połowę, bo kernel po stronie
rocKE już istnieje. Trening UHD (Krok 4) to jednak realnie osobny, kosztowny
kawałek pracy — stąd checkpoint w 3.6, żeby nie zacząć go, zanim nie będzie wiadomo,
że nowy katalog w ogóle dobrze kryje `cases.txt`.

## Rekomendacja

Krok 1 zrobić jako pierwszy i osobno zweryfikować (testy korektności) zanim ruszy
się dalej — to tani, niskoryzykowny krok, który odblokowuje resztę. Krok 2 to
właściwa nowa praca (nowa rodzina w integracji) i warto go przejrzeć wg
`KERNEL_AUTHORING.md` (Definition of Done dla "new knob"/"new feature" w
integracji, skoro sam kernel rocKE nie jest nowy tu, tylko jego hipDNN-owe
opakowanie). Kroki 3.1–3.5 (sama podmiana) mają sens dopiero po Kroku 2.

**Nie przechodzić do Kroku 4 (trening UHD) bez jawnej zgody po zobaczeniu wyników
Kroku 3.6** — to zgodnie z poleceniem: policzyć autotune na `cases.txt` na nowym
katalogu, tylko rocKE (bez MIOpen), pokazać wyniki i zatrzymać plan w tym miejscu.

## Postęp wykonania — sesja 2026-10-30 (druga podmiana: `instancesv3.txt`, 466 instancji)

Po commicie `18318f32980` (414 instancji z `instance_fwd_gfx950_new.txt`) user poprosił
o kolejną podmianę katalogu na instancje z `/home/barkocot/rocm3/instancesv3.txt`
(466 linii, nadzbiór poprzedniego — głównie dokłada warianty `direct_depthwise_win`).
**UHD pipeline nadal nie działa** (zob. sekcja wyżej, `uhd-corpus` padał na
infrastrukturę) — to zadanie dotyczy wyłącznie Kroków 3.1–3.5 i rozgrzewki do
lokalnego autotune (Krok 3.6), **nie** Kroku 4.

### Katalog dispatchera rocKE (`library/dispatch/grouped_convolution_instances/gfx950_fwd.json`)

- Podmieniono na 466 instancji z `instancesv3.txt` (`instance_list.py new`), co
  zepsuło 12/113 testów `library/tests/dispatch/test_grouped_conv_dispatch.py` —
  `instancesv3.txt` nie ma żadnej `direct_depthwise_tiled` ze stride 1x1 (tylko
  stride 2x2), więc stride-1 depthwise (3x3/7x7/11x11) traciło kandydata z rodziny
  `tiled` i spadało na `implicit_gemm`/`direct_depthwise_spatial`.
- **Decyzja usera**: dołożyć z powrotem 132 instancje `direct_depthwise_tiled`
  stride 1x1 ze starego katalogu (1136 instancji, backup `/tmp/gfx950_fwd.json.bak`)
  przez `instance_list.py add --from`. Katalog rocKE ma teraz **598 instancji**
  (466 + 132), kanonicznie sformatowany, komentarz nagłówkowy zachowany.
  **Uwaga**: to rozjeżdża katalog rocKE (598) z katalogiem wysyłanym do hipDNN
  (466, patrz niżej, zgodnie z dosłowną prośbą o `instancesv3.txt`) — świadomie,
  tylko po to, by nie zostawić czerwonych testów dispatchera; do rozważenia przy
  następnej synchronizacji obu katalogów.
- Po dołożeniu 132 instancji zostało **11/113 testów czerwonych** z dwóch
  nieoczywistych luk w harness rocke, **nigdy wcześniej nie ćwiczonych** (nie z
  braku treści katalogu):
  1. `KeyError: 'ws_split'` (`dense3x3`, `dense_stride2`) — nowy katalog dopuszcza
     dla tych C/K instancje `direct_nongrouped` ze `split_c=2`, czego stary katalog
     (1136) nigdy nie dopuszczał dla tych kształtów; fixture testu (`_PTRS`/`_SIZES`
     w `test_grouped_conv_dispatch.py`) nigdy nie definiował `'ws_split'`.
  2. Niezgodność launch-planu (`depthwise11x11`, `11x11_g32`, `7x7`, `s2`) — wariant
     `implicit_gemm` z `group_merge=8` (dopuszczony teraz po raz pierwszy dla tych
     kształtów) daje inny plan z prawdziwego dispatchera niż referencyjny
     `_cache_side_plan` w samym teście.
  - **Decyzja usera**: zostawić te 11 testów czerwone jako osobny, znany temat
    (luka w harness, nie w kernelach) i iść dalej do właściwego celu (katalog
    hipDNN + autotune). **Do zrobienia osobno**: zdiagnozować/naprawić `ws_split`
    i `group_merge=8` w `direct_kernel_sweep.py` / `test_grouped_conv_dispatch.py`.

### Katalog hipDNN (`tools/IngestorGenerator/configs/gfx950_conv_fwd_aot.labels.txt`)

Podmieniono dokładnie na `instancesv3.txt` (466 linii, bez dodatkowych 132
stride-1 `tiled` — to rozmyślnie inny zestaw niż katalog rocKE, patrz wyżej).
Powtórzono Kroki 3.1–3.4 z poprzedniej sesji, analogicznie:

- **Krok 3.1**: `enumerate_conv_aot_instances.py --rocke-root "$ROCKE"` pod
  `ROCKE_LLVM_FLAVOR=llvm22` (ta sama pułapka co poprzednio — domyślny
  `current_llvm_flavor()` to `llvm20` lokalnie) — **466/466 zwalidowane, 0
  błędów**: `fp16 89 implicit + 144 direct`, `bf16` tak samo.
- **Krok 3.2**: `make_conv_aot_config.py --instances <fp16,bf16>.instances.json
  --output /tmp/gfx950_conv_fwd_aot_v3.yaml --engine-name hipkernel:Gfx950ConvFwdAot
  --authored-subpath rocKE/gfx950_conv_fwd_aot --sharding k-loop` — **466/466
  zaakceptowane przez adapter, 0 unsupported**.
- **Krok 3.3**: `generate.py --config ... --output-dir /tmp/gfx950_aot_gen_v3 --force`
  (30 plików jak poprzednio, 16 kdp.json). Tym razem **knob vocabulary identyczny**
  jak w już zakomitowanym 40-polowym KMD z poprzedniej sesji (`instancesv3.txt` nie
  wprowadza nowych knobów) — zweryfikowane programowo: `kmd`/`udd`/`split_c.udd`/
  `uhd`/`umd`/`ued` wygenerowane na nowo są bajtowo identyczne z zakomitowanymi,
  **poza `id`** (i poza `ued.metadata`/`ued.sort_kernel_catalog`, referencjami do
  wytrenowanego modelu UHD). Scalono zachowując stare ID-ki (mały skrypt
  rekurencyjnego remapu `{new_id: old_id}` na kmd/udd/split_c.udd/uhd/umd/ued) —
  **zero zmian w tych 6 deskryptorach względem HEAD**.
  - `make_conv_aot_config.py --stabilize-ids` na 16 nowych kdp.json (UUID-y
    kerneli z `instance_id`), następnie **ten sam remap ID-ków** zastosowany do
    pól `engine`/`dispatch`/`matchers`/`provenance.*` wewnątrz kdp.json (żeby
    wskazywały na zachowane stare ID-ki kmd/udd/uhd/umd zamiast na świeżo
    wygenerowane) — bez tego testy `test_shipped_bundle_is_its_catalog_with_instance_derived_ids`
    i `..._is_sharded_below_the_hosting_size_limit` failowały na niezgodności ID.
  - **DVC**: `dvc` jest zainstalowany w tym środowisku (3.67.1, w odróżnieniu od
    poprzedniej sesji) i `artifacts/` był już lokalnie wyewidencjonowany (z
    poprzedniej podmiany, 414-katalog) — podmieniono 16 plików `*.kdp.json` pod
    `artifacts/` i przeliczono wskaźnik `dvc add` **lokalnie**
    (`b9fe46e388cd03bc8c5ac074db167ad7.dir`, 1275446 B). **`dvc pull`/`dvc push`
    nie działa w tym środowisku** (`Unable to locate credentials` do S3) — nowy
    hash NIE jest wypchnięty na `ingestor`, tylko lokalny. Trzeba `dvc push -r
    ingestor` z maszyny/sesji z dostępem do AWS, zanim commit z nowym
    `artifacts.dvc` będzie użyteczny dla kogokolwiek poza tym workdir.
- **Krok 3.4**: zaktualizowano dwa miejsca ze sztywnymi liczbami:
  - `test_enumerate_conv_aot_instances.py`:
    `test_the_catalog_lists_207_kernels_per_dtype_the_same_in_both` →
    `..._233_...`, `{True: 79, False: 128}` → `{True: 89, False: 144}`.
  - `IntegrationGpuGfx950ConvFwd.cpp` (`aotGraphs()`, 26 wierszy) — przeliczono
    programowo (`test_make_conv_aot_config.py`'s `_selected_counts` na realnym
    466-instancyjnym katalogu) i podmieniono tylko 3 końcowe pola (`direct`,
    `implicitFp16`, `implicitBf16`) na linię, `name`/`problem`/`seedDirect`/
    `seedImplicit` bez zmian (seed niezmieniony) — diff ograniczony do 26 linii,
    bez zmiany indentacji.
  - **Wynik**: `test_enumerate_conv_aot_instances.py` (19/19) +
    `test_make_conv_aot_config.py` (65/65) — **84/84 passed**.
  - `test_convolution_forward_aot_packaging.py` (rocKE) — **92/92 passed**
    (307s, długi test).

### Krok 3.5 — build zrealizowany (osobny build tree wskazany na `/home/barkocot/rocm3`)

`howtobuild.md` opisywał build tree `/home/barkocot/rocm-libraries/build-aot` —
**to inny checkout** (branch `users/barkocot/gfx950-dgrad-aot-catalog-prune-150`,
z własnymi niezacommitowanymi zmianami dgrad), nie ten, w którym robiono dzisiejszą
podmianę katalogu (`/home/barkocot/rocm3`, branch `users/aledudek/gfx950-conv-aot-uhd`).
Zamiast mieszać te dwa drzewa, skonfigurowano **nowy, osobny build tree**:

- Źródło: `/home/barkocot/rocm3` (ten sam layout monorepo co `rocm-libraries`).
- Build dir: `/home/barkocot/work/build-aot-rocm3`, install prefix:
  `/home/barkocot/work/aot-prefix-rocm3`.
- Flagi z `howtobuild.md` (`HIPKERNELPROVIDER_ENABLE_ROCKE=ON`, `..._LLVM_FLAVOR=llvm22`,
  `..._COMGR_LIB=...`) **nie wystarczyły same** — konfiguracja monorepo wymagała
  dodatkowo:
  - `-DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;miopen-provider;hipblaslt-provider;hip-kernel-provider"`
    (bez tego configure ciągnie cały monorepo, w tym rocRoller → brak `msgpackc-cxx`).
  - `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON` (domyślnie off na tej ścieżce configure).
  - `-DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON` (brak lokalnego `shared/kpack/python`
    w `rocm3`, w przeciwieństwie do `rocm-libraries`).
  - `-DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++` / `CMAKE_C_COMPILER=...clang`
    (AMD clang 20) — bez tego configure bierze systemowy `/usr/bin/c++` (GCC 11),
    co dalej psuje build warningami-jako-błędami na kodzie generowanym pod clanga.
  - `PATH` musiał zawierać **oba** `/tmp/cf_tool/bin` (clang-format 18.1.4,
    doinstalowany lokalnie, `PYTHONPATH=/tmp/cf_tool`) i `/opt/10.1.0a20260826/lib/llvm/bin`
    (clang-tidy, tylko jako WARNING o niezgodnej wersji, nie blocker) —
    `/opt/10.1.0a20260826` samo nie ma clang-format, `/opt/rocm` samo nie ma clang-tidy.
  - `-DCMAKE_CXX_FLAGS="-Wno-error=deprecated-declarations"` — **twardy bloker bez
    tego**: nagłówki systemowego libstdc++ GCC 12 (`bits/stl_tempbuf.h`) oznaczają
    `std::get_temporary_buffer` jako deprecated, co pod `-Werror` wywala kompilację
    `std::stable_sort` wszędzie, gdzie jest użyty (w tym w samym
    `hip_kernel_provider_impl` — `ResampleFwdPlan.cpp`, `LayernormFwdPlan.cpp`, oraz
    w `hipdnn_plugin_sdk`/`corpus_gen`/`miopen-provider` testach). Dokładnie ta sama
    flaga jest już w działającym `rocm-libraries/build-aot`'s `CMakeCache.txt`.
  - Jedna przejściowa race condition przy `-j64` (`rocke-pyenv.dir`: "Text file busy"
    na `rocke-pyenv/bin/python3.12`, dwa cele edytowalnie instalujące jednocześnie) —
    zniknęła po prostym retry builda.
  - `-k` (keep-going) żeby nie przerywać się na niepowiązanym, pre-istniejącym
    `miopen_plugin_tests`/`hipdnn_corpus_gen_tests` (ten sam deprecated-declarations
    problem, ale w testach spoza mojej ścieżki — po dodaniu `-Wno-error=...` globalnie
    te też przestały failować, `-k` ostatecznie nie był już potrzebny do sukcesu).
- **Wynik: `cmake --build . --target install -- -j64` zakończony sukcesem, 0 błędów.**
  `libhip_kernel_provider.so` zbudowane z nowym (466-instancyjnym) katalogiem
  (mtime 2026-10-30 11:07), `hip_kernel_provider_tests`/`hip_kernel_provider_integration_tests`
  skompilowane (w tym `IntegrationGpuGfx950ConvFwd.cpp` z przeliczonym `aotGraphs()`).

Poprzednia notatka o blockerze `/opt` SDK (poniżej) dotyczyła prób budowania
*przeciw zainstalowanemu system-wide SDK pod `/opt`* — ten nowy build tree
**nie korzysta z tej ścieżki** (buduje cały stack hipDNN+providers z monorepo od
zera), więc blocker jej nie dotyczy.

### Historyczna notatka (nieaktualna dla tego build tree) — blocker `/opt` SDK

`HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON` nadal nie istnieje w zainstalowanym
pod `/opt` `hipdnn_plugin_sdkConfig.cmake` (sprawdzone programowo, `grep` nie
znajduje tej zmiennej nigdzie w configu) — **zainstalowany SDK pod `/opt` jest
wciąż starszy niż to, czego oczekuje obecne źródło `hip-kernel-provider`**. Bez
tego nie da się zbudować providera przeciw `/opt`, więc **nie da się przebudować
`IntegrationGpuGfx950ConvFwd.cpp`/`TestGfx950ConvFwdAotPacks.cpp` ani uruchomić
realnego lokalnego autotune** (`dnn-benchmark --autotune` wymaga działającego
pluginu z nowym katalogiem). Jedyna właściwa naprawa to pełny build hipDNN z
`projects/hipdnn` do świeżego prefiksu (duży, wielogodzinny build, poza
budżetem automatycznej sesji) — **zatrzymano się tutaj i czeka na decyzję usera**,
zgodnie z poleceniem "jak będziesz gotowy do autotune to zatrzymaj się i daj znać".
