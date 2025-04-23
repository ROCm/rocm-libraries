if [ ! -f model_separable_rnnt.py ]; then
    wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/model_separable_rnnt.py
fi

if [ ! -f rnn.py ]; then
    wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/rnn.py
fi

if [ ! -f preprocessing.py ]; then
    wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/preprocessing.py
fi

if [ ! -f helpers.py ]; then
    wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/helpers.py
fi

if [ ! -f metrics.py ]; then
    wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/metrics.py
fi


mkdir -p parts

if [ ! -f parts/features.py ]; then
    wget -P parts https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/parts/features.py
fi

if [ ! -f parts/segment.py ]; then
    wget -P parts https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/parts/segment.py
fi


mkdir -p configs

if [ ! -f configs/rnnt.toml ]; then
    wget -P configs https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/configs/rnnt.toml
fi
