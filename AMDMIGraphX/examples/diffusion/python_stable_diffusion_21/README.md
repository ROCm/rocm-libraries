# Stable Diffusion 2.1

This version was tested with [rocm 6.0](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/rocm-6.0.0) revision.

## Jupyter notebook

There is a dedicated step-by-step notebook. See [sd21.ipynb](./sd21.ipynb)

## Console application

To run the console application, follow these steps below.

Setup python environment

```bash
# this will require the python venv to installed (e.g. apt install python3.8-venv)
python3 -m venv sd_venv
. sd_venv/bin/activate
```

Install dependencies

```bash
pip install -r torch_requirements.txt -r requirements.txt
```

Use MIGraphX Python Module

```bash
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

Get models with optimum

```bash
optimum-cli export onnx --model stabilityai/stable-diffusion-2-1 models/sd21-onnx --task stable-diffusion
```
*Note: `models/sd21-onnx` will be used in the scripts.*

Run the text-to-image script with the following example prompt and seed (optionally, you can change the batch size / number of images generated for that prompt)

```bash
MIGRAPHX_MLIR_USE_SPECIFIC_OPS="attention,dot,fused,convolution" python txt2img.py --prompt "a photograph of an astronaut riding a horse" --seed 13 --output astro_horse.jpg --batch 1
```
*Note: The first run will compile the models and cache them to make subsequent runs faster. New batch sizes will result in the models re-compiling.*

The result should look like this:

![example_output.jpg](./example_output.jpg)

## Gradio application

Note: requires `Console application` to work

Install gradio dependencies

```bash
pip install -r gradio_requirements.txt
```

Usage

```bash
python gradio_app.py -p "a photograph of an astronaut riding a horse" --seed 13
```

This will load the models (which can take several minutes), and when the setup is ready, starts a server on `http://127.0.0.1:7860`.
