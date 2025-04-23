# Llama-2

This version was tested with [rocm 6.0](https://github.com/ROCm/AMDMIGraphX/tree/rocm-6.0.0) revision.

## Jupyter notebook

There is a dedicated step-by-step notebook. See [llama2.ipynb](./llama2.ipynb)

## Console application

To run the console application, follow these steps below.

Setup python environment

```bash
# this will require the python venv to installed (e.g. apt install python3.8-venv)
python3 -m venv ll2_venv
. ll2_venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Use MIGraphX Python Module

```bash
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

Llama2 requires logging to access the models

```bash
huggingface-cli login
```

Get models with optimum

```bash
optimum-cli export onnx --model meta-llama/Llama-2-7b-chat-hf models/llama-2-7b-chat-hf --task text-generation --framework pt --library transformers --no-post-process
```
*Note: `models/llama-2-7b-chat-hf` will be used in the scripts.*

Run the text-generation script with the following example prompt:

```bash
python txtgen.py --prompt "Where is Szeged?" --log-process
```

*Note: The first run will compile the models and cache them to make subsequent runs faster.*


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
