# RNN-T
Speech Recognition using MIGraphX optimizations on ROCm platform.
This version was taken from [MLCommons](https://github.com/mlcommons/inference/tree/master/retired_benchmarks/speech_recognition/rnnt)

## Steps

To run the RNN-T, follow these steps below.

1) Install MIGraphX to your environment. Please follow the steps to build MIGraphX given at https://github.com/ROCmSoftwarePlatform/AMDMIGraphX

2) Require the python venv to installed

```bash
python3 -m venv rnnt_venv
. rnnt_venv/bin/activate
```

3) Install dependencies

```bash
pip install -r torch_requirements.txt -r requirements.txt
```

4) Download helper files 

```bash
bash download_helper_files.sh
```

4) Download RNN-T Model Weights

```bash
wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O rnnt.pt
```

5) Run the inference, it will compile and run the model on the first sentence in [hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)

```bash
python3 rnnt.py 
```

This will do four things:
- Create the PyTorch model 
- Export the PyTorch model to ONNX
- Use MIGraphX to read the ONNX model 
- Run the MIGraphX model 

