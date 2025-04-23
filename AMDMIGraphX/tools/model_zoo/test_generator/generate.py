#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#####################################################################################
from argparse import ArgumentParser
from collections import namedtuple
import warnings

warnings.filterwarnings('ignore')

# datasets
from sample_generator.dataset.imagenet import ImageNet2012Val
from sample_generator.dataset.coco import COCO2017Val
from sample_generator.dataset.librispeech import LibriSpeechASR
from sample_generator.dataset.squad import SQuAD_HF
from sample_generator.dataset.prompts import StylePrompts

# models
from sample_generator.model.image import ResNet50_v1, ResNet50_v1_5, VitBasePatch16_224, TIMM_MobileNetv3_large
from sample_generator.model.text import BERT_large_uncased, DistilBERT_base_cased_distilled_SQuAD, RobertaBaseSquad2
from sample_generator.model.text import GPTJ, Llama2_7b_chat_hf, Llama3_8b_instruct, T5_base, Gemma_2b_it
from sample_generator.model.audio import Wav2Vec2_base_960h, WhisperSmallEn
from sample_generator.model.hybrid import ClipVitLargePatch14
from sample_generator.model.diffusion import StableDiffusion21, StableDiffusionXL

# generator
from sample_generator.generator import generate_test_dataset, generate_diffusion_data

DatasetModelsPair = namedtuple('DatasetModelsPair', ['dataset', 'models'])

imagenet_models = (
    ResNet50_v1,
    ResNet50_v1_5,
    VitBasePatch16_224,
    TIMM_MobileNetv3_large,
    ClipVitLargePatch14,
)

squad_models = (
    BERT_large_uncased,
    DistilBERT_base_cased_distilled_SQuAD,
    RobertaBaseSquad2,
    GPTJ,
    T5_base,
    Gemma_2b_it,
    Llama2_7b_chat_hf,
    Llama3_8b_instruct,
)

librispeech_models = (Wav2Vec2_base_960h, WhisperSmallEn)

diffusion_models = (StableDiffusion21, StableDiffusionXL)

default_dataset_model_mapping = {
    "image": DatasetModelsPair(ImageNet2012Val, imagenet_models),
    "text": DatasetModelsPair(SQuAD_HF, squad_models),
    "audio": DatasetModelsPair(LibriSpeechASR, librispeech_models),
    "diffusion": DatasetModelsPair((COCO2017Val, StylePrompts),
                                   diffusion_models)
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--image',
        choices=['all', 'none'] + [model.name() for model in imagenet_models],
        nargs='+',
        default='all',
        dest='image_model_names',
        type=str,
        help=
        f'Image models to test with {ImageNet2012Val.name()} dataset samples')
    parser.add_argument(
        '--text',
        choices=['all', 'none'] + [model.name() for model in squad_models],
        nargs='+',
        default='all',
        dest='text_model_names',
        type=str,
        help=f'Text models to test with {SQuAD_HF.name()} dataset samples')
    parser.add_argument(
        '--audio',
        choices=['all', 'none'] +
        [model.name() for model in librispeech_models],
        nargs='+',
        default='all',
        dest='audio_model_names',
        type=str,
        help=
        f'Audio models to test with {LibriSpeechASR.name()} dataset samples')
    parser.add_argument(
        '--diffusion',
        choices=['all', 'none'] + [model.name() for model in diffusion_models],
        nargs='+',
        default='all',
        dest='diffusion_model_names',
        type=str,
        help=
        f'DiffusionModel models to test with {COCO2017Val.name()} and {StylePrompts} dataset samples'
    )
    parser.add_argument(
        '--output-folder-prefix',
        default='generated',
        help='Output path will be "<this-prefix>/<dataset-name>/<model-name>"')
    parser.add_argument(
        '--sample-limit',
        default=5,
        type=int,
        help="Max number of samples generated. Use 0 to ignore it.")
    parser.add_argument(
        '--decode-limit',
        default=5,
        type=int,
        help=
        "Max number of sum-samples generated for decoder models. Use 0 to ignore it. (Only for decoder models)"
    )
    return parser.parse_args()


def get_dataset_models_pairs(dataset_type, model_names):
    if 'none' in model_names:
        return None
    ds_ms_mapping = default_dataset_model_mapping[dataset_type]
    if 'all' in model_names:
        return ds_ms_mapping

    return DatasetModelsPair(dataset=ds_ms_mapping.dataset,
                             models=(model for model in ds_ms_mapping.models
                                     if model.name() in model_names))


def main(image_model_names='all',
         text_model_names='all',
         audio_model_names='all',
         diffusion_model_names='all',
         output_folder_prefix='generated',
         sample_limit=5,
         decode_limit=5):
    for dataset_type, model_names in zip(
        ('image', 'text', 'audio'),
        (image_model_names, text_model_names, audio_model_names)):
        dataset_model_pair = get_dataset_models_pairs(dataset_type,
                                                      model_names)
        if dataset_model_pair is None:
            print(f"Skip {dataset_type}...")
            continue

        dataset = dataset_model_pair.dataset
        for model in dataset_model_pair.models:
            try:
                generate_test_dataset(
                    model(),
                    dataset(),
                    output_folder_prefix=output_folder_prefix,
                    sample_limit=sample_limit,
                    decode_limit=decode_limit)
            except Exception as e:
                print(f"Something went wrong:\n{e}\nSkipping model...")
                continue

    for dataset_type, model_names in zip(('diffusion', ),
                                         (diffusion_model_names)):
        dataset_model_pair = get_dataset_models_pairs(dataset_type,
                                                      model_names)
        if dataset_model_pair is None:
            print(f"Skip {dataset_type}...")
            continue

        image_dataset, prompt_dataset = dataset_model_pair.dataset
        for model in dataset_model_pair.models:
            try:
                generate_diffusion_data(
                    model(),
                    image_dataset(),
                    prompt_dataset(),
                    output_folder_prefix=output_folder_prefix,
                    sample_limit=sample_limit,
                    decode_limit=decode_limit)
            except Exception as e:
                print(f"Something went wrong:\n{e}\nSkipping model...")
                continue

    print(
        f'Use this to test MIGraphX with the result:\n./test_models.sh {output_folder_prefix}'
    )


if '__main__' == __name__:
    args = get_args()
    main(**vars(args))
