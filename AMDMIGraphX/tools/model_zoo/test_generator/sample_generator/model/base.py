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
import abc
import os
from optimum.exporters.onnx import main_export
from ..utils import download


class BaseModel(abc.ABC):
    @property
    @abc.abstractmethod
    def model_id(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def name():
        pass

    @property
    def task(self):
        return "auto"

    @abc.abstractmethod
    def get_model(self, folder, force_download):
        pass

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    def is_decoder(self):
        return False

    def is_diffuser(self):
        return False


class DecoderModel(BaseModel):
    def is_decoder(self):
        return True

    @abc.abstractmethod
    def decode_step(self, *args, **kwargs):
        pass


class DiffusionModel(BaseModel):
    def is_diffuser(self):
        return True

    @abc.abstractmethod
    def get_models(self, folder, models, force_download):
        pass

    @abc.abstractmethod
    def text_preprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def text_preprocess_2(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def image_preprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def scheduler(self, *args, **kwargs):
        pass


class SingleModelDownloadMixin(object):
    def get_model(self, output_folder, force_download=False):
        filepath = f"{output_folder}/model.onnx"
        if force_download or not os.path.isfile(filepath):
            print(f"Download model from {self.model_id} to {filepath}")
            download(self.model_id, filepath)
        return filepath


class OptimumHFModelDownloadMixin(object):
    def get_model(self, folder, force_download=False):
        filepath = f"{folder}/model.onnx"
        if force_download or not os.path.isfile(filepath):
            print(f"Download model from {self.model_id} to {filepath}")
            main_export(
                self.model_id,
                output=folder,
                monolith=True,  # forces export into one model
                task=self.task)
        return filepath
