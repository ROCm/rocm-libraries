# Test Generator with Datasets

Helper module to generate real samples from datasets for specific models.

## Prerequisites

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

To use audio based datasets, install sndfile
```bash
apt install libsndfile1
```

## Usage

```bash
usage: generate.py [-h]
                   [--image {all,none,...}]
                   [--text {all,none,...}]
                   [--audio {all,none,...}]
                   [--output-folder-prefix OUTPUT_FOLDER_PREFIX]
                   [--sample-limit SAMPLE_LIMIT]
                   [--decode-limit DECODE_LIMIT]

optional arguments:
  -h, --help            show this help message and exit
  --image {all,none,...}
                        Image models to test with imagenet-2012-val dataset samples
  --text {all,none,...}
                        Text models to test with squad-hf dataset samples
  --audio {all,none,...}
                        Audio models to test with librispeech-asr dataset samples
  --output-folder-prefix OUTPUT_FOLDER_PREFIX
                        Output path will be "<this-prefix>/<dataset-name>/<model-name>"
  --sample-limit SAMPLE_LIMIT
                        Max number of samples generated. Use 0 to ignore it.
  --decode-limit DECODE_LIMIT
                        Max number of sum-samples generated for decoder models. Use 0 to ignore it. (Only for decoder models)
```

> [!NOTE]
> Some models require permission to access, use `huggingface-cli login`.

To generate everything:
```bash
python generate.py
```

To generate a subset of the supported models:
- `none` to skip it
- `all` for every models
- <name> list supported model names

```bash
python generate.py --image resnet50_v1.5 clip-vit-large-patch14 --text none --audio none
```

## Test models

`test_models.sh` will run all downloaded models on the `generated` samples. The result will be in `logs`.

```bash
./test_models.sh generated/
```

> [!NOTE]
> `generated` is the default output folder, make sure to match `--output-folder-prefix` name.

## Adding more models

To add mode models, first choose the proper place:
- [image](./sample_generator/model/image.py)
- [text](./sample_generator/model/text.py)
- [audio](./sample_generator/model/audio.py)
- [hybrid](./sample_generator/model/hybrid.py)

For example, adding basic would be this (e.g. ResNet):

```python
class ResNet50_v1_5(OptimumHFModelDownloadMixin,
                    AutoImageProcessorHFMixin, BaseModel):
    @property
    def model_id(self):
        return "microsoft/resnet-50"

    @staticmethod
    def name():
        return "resnet50_v1.5"
```

Define the class with the proper `Mixin`s:
- `OptimumHFModelDownloadMixin`: Download model from Hugging Face and export it to onnx with Optimum
- `AutoImageProcessorHFMixin`: Define the processor from Hugging Face (This depends on the model type)
- `BaseModel`: Default model type, other choice is `DecoderModel`

Provide 2 mandatory fields:
- `model_id`: Hugging Face url
- `name`: unique name for model

To add a more complex model (e.g. Decoder), check [text](./sample_generator/model/text.py).

The [generate](./generate.py) part will need further updating to include the model.

## Adding more datasets

The 3 most common use cases are handled:
- `Image`:  with [imagenet](./sample_generator/dataset/imagenet.py)
- `Text`:  with [squad](./sample_generator/dataset/squad.py)
- `Audio`:  with [librispeech](./sample_generator/dataset/librispeech.py)

To add a new use case, e.g. Video, create a new python file in dataset, and inherit a new class from Base.

The [generate](./generate.py) part will need further updating to include the dataset.
