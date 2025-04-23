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
from .base import DiffusionModel
from transformers import AutoTokenizer
from diffusers import EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from optimum.exporters.onnx import main_export
import numpy as np
import os
from PIL import ImageOps


class OptimumHFDiffusionModelDownloadMixin(object):
    def get_models(self, folder, models, force_download=False):
        filepaths = [f"{folder}/{model}/model.onnx" for model in models]
        if force_download or not all(
                os.path.isfile(filepath) for filepath in filepaths):
            print(f"Download model from {self.model_id} to {filepaths}")
            main_export(
                self.model_id,
                output=folder,
                monolith=True,  # forces export into one model
                task=self.task)
        return filepaths


class AutoTokenizerHFMixin(object):

    _tokenizer = None
    _tokenizer_2 = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer")
        return self._tokenizer

    @property
    def tokenizer_2(self):
        if self._tokenizer_2 is None:
            self._tokenizer_2 = AutoTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer_2")
        return self._tokenizer_2

    def text_preprocess(self, prompt, max_length=77, version_2=False):
        tokenizer = self.tokenizer_2 if version_2 else self.tokenizer
        return tokenizer(prompt,
                         padding='max_length',
                         max_length=max_length,
                         return_tensors="np")


class VAEImageProcessorHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = VaeImageProcessor()
        return self._processor

    def image_preprocess(self, image_data):
        return self.processor.preprocess(image_data).numpy()


class EulerDiscreteSchedulerHFMixin(object):

    _scheduler = None

    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = EulerDiscreteScheduler.from_pretrained(
                self.model_id, subfolder="scheduler")
        return self._scheduler


class StableDiffusion21(OptimumHFDiffusionModelDownloadMixin,
                        AutoTokenizerHFMixin, VAEImageProcessorHFMixin,
                        EulerDiscreteSchedulerHFMixin, DiffusionModel):
    def __init__(self):
        # it should be in pipeline exec order
        self.models = ('text_encoder', 'vae_encoder', 'unet', 'vae_decoder')

    @property
    def model_id(self):
        return "stabilityai/stable-diffusion-2-1"

    @staticmethod
    def name():
        return "stable-diffusion-2-1"

    def get_model(self, folder, force_download=False):
        return super().get_models(folder, self.models, force_download)

    def preprocess(self, *args, **kwargs):
        raise RuntimeError("Call text_preprocess or image_preprocess directly")

    def text_preprocess(self, *args, **kwargs):
        prompt, negative_prompt = args
        prompt_result = super().text_preprocess(prompt, **kwargs)
        negative_prompt_result = super().text_preprocess(
            negative_prompt, **kwargs)
        result = {
            'input_ids':
            np.concatenate(
                (prompt_result['input_ids'],
                 negative_prompt_result['input_ids'])).astype(np.int32)
        }
        return result

    def text_preprocess_2(self, *args, **kwargs):
        raise RuntimeError("No tokenizer_2 for SD21 model")

    def image_preprocess(self, *args, **kwargs):
        resized_image = ImageOps.fit(args[0], (512, 512))
        result = super().image_preprocess(resized_image, **kwargs)
        return {"sample": result}


class StableDiffusionXL(OptimumHFDiffusionModelDownloadMixin,
                        AutoTokenizerHFMixin, VAEImageProcessorHFMixin,
                        EulerDiscreteSchedulerHFMixin, DiffusionModel):
    def __init__(self):
        # it should be in pipeline exec order
        self.models = ('text_encoder', 'text_encoder_2', 'vae_encoder', 'unet',
                       'vae_decoder')

    @property
    def model_id(self):
        return "stabilityai/stable-diffusion-xl-base-1.0"

    @staticmethod
    def name():
        return "stable-diffusion-xl"

    def get_model(self, folder, force_download=False):
        return super().get_models(folder, self.models, force_download)

    def preprocess(self, *args, **kwargs):
        raise RuntimeError("Call text_preprocess or image_preprocess directly")

    def text_preprocess(self, *args, **kwargs):
        prompt, negative_prompt = args
        dtype = np.int64 if kwargs.get('version_2', False) else np.int32
        prompt_result = super().text_preprocess(prompt, **kwargs)
        negative_prompt_result = super().text_preprocess(
            negative_prompt, **kwargs)
        result = {
            'input_ids':
            np.concatenate((prompt_result['input_ids'],
                            negative_prompt_result['input_ids'])).astype(dtype)
        }
        return result

    def text_preprocess_2(self, *args, **kwargs):
        new_kwargs = dict(kwargs)
        new_kwargs['version_2'] = True
        return self.text_preprocess(*args, **new_kwargs)

    def image_preprocess(self, *args, **kwargs):
        resized_image = ImageOps.fit(args[0], (1024, 1024))
        result = super().image_preprocess(resized_image, **kwargs)
        return {"sample": result}
