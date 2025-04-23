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
from .base import BaseModel, DecoderModel, OptimumHFModelDownloadMixin
import numpy as np
from transformers import AutoTokenizer


class AutoTokenizerHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoTokenizer.from_pretrained(self.model_id)
        return self._processor

    def preprocess(self,
                   question,
                   context,
                   max_length,
                   truncation='only_second'):
        return self.processor(question,
                              context,
                              padding='max_length',
                              max_length=max_length,
                              truncation=truncation,
                              return_tensors="np")


class TextGenerationDecoderOnlyMixin(object):
    def __init__(self):
        # no pad token by default
        self.processor.pad_token = self.processor.eos_token

    def preprocess(self, *args, **kwargs):
        # swap squad's default "question - answer" order
        new_args, new_kwargs = list(args), kwargs
        new_args[0], new_args[1] = new_args[1], new_args[0]
        new_kwargs["truncation"] = "only_first"
        result = super().preprocess(*new_args, **new_kwargs)

        # result only contains "input_ids" and "attention_mask", extend it with "position_ids"
        result["position_ids"] = np.arange(0,
                                           len(result["input_ids"][0]),
                                           dtype=np.int64)
        result["position_ids"] = result["position_ids"][np.newaxis]
        return result

    def decode_step(self, input_map, output_map):
        timestep = np.argmax(
            input_map["input_ids"][0] == self.processor.eos_token_id)
        new_token = np.argmax(output_map["logits"][0][timestep - 1])
        input_map["input_ids"][0][timestep] = new_token
        input_map["attention_mask"][0][timestep] = 1
        return new_token == self.processor.eos_token_id


class BERT_large_uncased(OptimumHFModelDownloadMixin, AutoTokenizerHFMixin,
                         BaseModel):
    @property
    def model_id(self):
        return "google-bert/bert-large-uncased"

    @staticmethod
    def name():
        return "bert-large-uncased"

    def preprocess(self, *args, **kwargs):
        # use squad's question for masking
        question, context = args
        # replace first word with [MASK]
        masked_question = question.replace(question[:question.index(' ')],
                                           '[MASK]', 1)
        # swap question-answer order
        new_args = [context, masked_question]
        result = super().preprocess(*new_args, **kwargs)
        return result


class DistilBERT_base_cased_distilled_SQuAD(OptimumHFModelDownloadMixin,
                                            AutoTokenizerHFMixin, BaseModel):
    @property
    def model_id(self):
        return "distilbert/distilbert-base-cased-distilled-squad"

    @staticmethod
    def name():
        return "distilbert-base-cased-distilled-squad"


class RobertaBaseSquad2(OptimumHFModelDownloadMixin, AutoTokenizerHFMixin,
                        BaseModel):
    @property
    def model_id(self):
        return "deepset/roberta-base-squad2"

    @staticmethod
    def name():
        return "roberta-base-squad2"


# Note: the inheritance order here important
class GPTJ(OptimumHFModelDownloadMixin, TextGenerationDecoderOnlyMixin,
           AutoTokenizerHFMixin, DecoderModel):
    @property
    def model_id(self):
        return "EleutherAI/gpt-j-6b"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text-generation"

    @staticmethod
    def name():
        return "gpt-j"


# Note: the inheritance order here important
class Llama2_7b_chat_hf(OptimumHFModelDownloadMixin,
                        TextGenerationDecoderOnlyMixin, AutoTokenizerHFMixin,
                        DecoderModel):
    @property
    def model_id(self):
        return "meta-llama/Llama-2-7b-chat-hf"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text-generation"

    @staticmethod
    def name():
        return "llama2-7b-chat-hf"


# Note: the inheritance order here important
class Llama3_8b_instruct(OptimumHFModelDownloadMixin,
                         TextGenerationDecoderOnlyMixin, AutoTokenizerHFMixin,
                         DecoderModel):
    @property
    def model_id(self):
        return "meta-llama/Meta-Llama-3-8B-Instruct"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text-generation"

    @staticmethod
    def name():
        return "llama3-8b-instruct"


class T5_base(OptimumHFModelDownloadMixin, AutoTokenizerHFMixin, DecoderModel):
    def __init__(self):
        max_length = 384  # default for squad
        # Note: no real start token, it requires the pad token
        self.initial_input_ids = np.array(
            [[self.processor.pad_token_id] + [self.processor.eos_token_id] *
             (max_length - 1)])

    @property
    def model_id(self):
        return "google-t5/t5-base"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text2text-generation"

    @staticmethod
    def name():
        return "t5-base"

    def preprocess(self, *args, **kwargs):
        # only use squad's question
        new_args = ["translate English to French:", args[0]]
        result = super().preprocess(*new_args, **kwargs)
        result["decoder_input_ids"] = np.copy(self.initial_input_ids)
        return result

    def decode_step(self, input_map, output_map):
        timestep = np.argmax(
            input_map["decoder_input_ids"][0] == self.processor.eos_token_id)
        new_token = np.argmax(output_map["logits"][0][timestep - 1])
        input_map["decoder_input_ids"][0][timestep] = new_token
        input_map["attention_mask"][0][timestep] = 1
        return new_token == self.processor.eos_token_id


class Gemma_2b_it(OptimumHFModelDownloadMixin, AutoTokenizerHFMixin,
                  DecoderModel):
    @property
    def model_id(self):
        return "google/gemma-2b-it"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text-generation"

    @staticmethod
    def name():
        return "gemma-2b-it"

    def preprocess(self, *args, **kwargs):
        # swap squad's default "question - answer" order
        new_args, new_kwargs = list(args), kwargs
        new_args[0], new_args[1] = new_args[1], new_args[0]
        new_kwargs["truncation"] = "only_first"
        result = super().preprocess(*new_args, **new_kwargs)
        # Note: padding-side is left, pos ids will be 1,1,1,...,0,1,2,3,...
        result["position_ids"] = np.cumsum(result["attention_mask"][0], -1) - 1
        result["position_ids"][:np.argmax(result["position_ids"] == 0)] = 1
        result["position_ids"] = result["position_ids"][np.newaxis]
        return result

    def decode_step(self, input_map, output_map):
        # The result is in the last logits
        new_token = np.argmax(output_map["logits"][0][-1])
        # Move everything left 1 step
        input_map["input_ids"][0] = np.roll(input_map["input_ids"][0], -1)
        input_map["input_ids"][0][-1] = new_token
        input_map["attention_mask"][0] = np.roll(
            input_map["attention_mask"][0], -1)
        input_map["attention_mask"][0][-1] = 1
        input_map["position_ids"][0] = np.roll(input_map["position_ids"][0],
                                               -1)
        input_map["position_ids"][0][-1] += input_map["position_ids"][0][-2]
        return new_token == self.processor.eos_token_id
