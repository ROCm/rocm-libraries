# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Sourced by generate.sbatch and bakeoff.sbatch: the UHD ids an engine with no UED declares
# in provider code, one per ranking metric. Such an engine reads only these ids, so a model
# trained or installed under any other one is never read. Mirrors MiopenContainer.cpp
# (MIOPEN_ENGINE_L1_MODELS, per metric under `default`) and AsmSdpaEngine.hpp
# (L1_MODEL_IDS, per arch); `uhd_gen generate` cross-checks the id against the one the
# engine's own description reports, so a drifted row fails the run instead of training
# under a stale id.

# declared_uhd_ids ENGINE ARCH -> `+`-separated METRIC=UUID pairs (empty for a UED engine).
declared_uhd_ids() {
    case "$1" in
        MIOPEN_ENGINE) echo "tflops=c47e1b3a-8f60-4a92-b5d4-1e08c9a27f63+time=30284ebe-6e15-4f8e-968d-09f92d8a9480" ;;
        MIOPEN_ENGINE_DETERMINISTIC) echo "tflops=2d95f8e7-16c4-4b03-a8f1-7be25390c4da+time=8ae8a348-8d47-4ad8-992f-2114d722f7cb" ;;
        ASM_SDPA_ENGINE)
            case "$2" in
                gfx942) echo "tflops=5f2a7c14-9d3b-4e86-b0a1-6c4f21d8e370" ;;
                gfx950) echo "tflops=8b61d0c9-24af-4d17-9e52-3a7c06b8f145" ;;
            esac ;;
    esac
}

# uhd_id_args PAIRS METRICS -> one `--uhd-id METRIC=UUID` argument pair per line, for the
# pairs whose metric is in the `+`-separated METRICS (generate refuses an id for a metric it
# was not asked to train).
uhd_id_args() {
    local pair
    local -a pairs
    IFS='+' read -ra pairs <<< "$1"
    for pair in "${pairs[@]}"; do
        if [[ "+$2+" == *"+${pair%%=*}+"* ]]; then
            printf '%s\n' --uhd-id "$pair"
        fi
    done
}
