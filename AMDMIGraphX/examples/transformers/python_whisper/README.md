# Whisper

This version was tested with [rocm 6.0](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/rocm-6.0.0) revision.

## Jupyter notebook

There is a dedicated step-by-step notebook. See [whisper.ipynb](./whisper.ipynb)

## Console application

To run the console application, follow these steps below.

Setup python environment

```bash
# this will require the python venv to installed (e.g. apt install python3.8-venv)
python3 -m venv w_venv
. w_venv/bin/activate
```

Install dependencies

`ffmpeg` needed to handle audio files.

```bash
apt install ffmpeg
```

```bash
pip install -r requirements.txt
```

Use MIGraphX Python Module

```bash
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

Use the helper script to download with optimum.
The attention_mask for decoder is not exposed by default, but required to work with MIGraphX.

```bash
python download_whisper.py
```

*Note: `models/whisper-tiny.en_modified` will be used in the scripts*

There are *optional* samples which can be downloaded. But the example can be tested without them.

```bash
./download_samples.sh
```

Run the automatic-speech-recognition script with the following example input:

```bash
python asr.py --audio audio/sample1.flac --log-process
```

Or without any audio input to run the [Hugging Face dummy dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy) samples.


## Gradio application

Note: requires `Console application` to work

Install gradio dependencies

```bash
pip install -r gradio_requirements.txt
```

Usage

```bash
python gradio_app.py
```

This will load the models (which can take several minutes), and when the setup is ready, starts a server on `http://127.0.0.1:7860`.
