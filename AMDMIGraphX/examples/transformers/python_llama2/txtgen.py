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
#####################################################################################

from argparse import ArgumentParser
from transformers import LlamaTokenizer
import numpy as np
import migraphx as mgx
import os
import sys
import time
from functools import wraps


# measurement helper
def measure(fn):
    @wraps(fn)
    def measure_ms(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter_ns()
        print(
            f"Elapsed time for {fn.__name__}: {(end_time - start_time) * 1e-6:.4f} ms\n"
        )
        return result

    return measure_ms


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Input prompt",
    )

    parser.add_argument(
        "-l",
        "--log-process",
        action="store_true",
        help="Print the current state of transcribing.",
    )

    parser.add_argument("-s",
                        "--max-seq-len",
                        type=int,
                        choices=[256, 512, 1024, 2048, 4096],
                        default=1024,
                        help="Max sequence length the model can handle")

    return parser.parse_args()


class Llama2MGX():
    def __init__(self, max_seq_len=1024):
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        self.max_seq_len = max_seq_len

        print("Load mgx model")
        self.model = Llama2MGX.load_mgx_model(
            max_seq_len, {
                "input_ids": [1, max_seq_len],
                "attention_mask": [1, max_seq_len],
                "position_ids": [1, max_seq_len]
            })
        print(f"Load AutoTokenizer model from {model_id}")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_id)

    @staticmethod
    @measure
    def load_mgx_model(max_seq_len, shapes):
        file = "models/llama-2-7b-chat-hf/model"
        print(f"Loading {max_seq_len} seq-len version model from {file}")
        if os.path.isfile(f"{file}-{max_seq_len}.mxr"):
            print("Found mxr, loading it...")
            model = mgx.load(f"{file}-{max_seq_len}.mxr", format="msgpack")
        elif os.path.isfile(f"{file}.onnx"):
            print("Parsing from onnx file...")
            model = mgx.parse_onnx(f"{file}.onnx", map_input_dims=shapes)
            model.compile(mgx.get_target("gpu"))
            print("Saving model to mxr file...")
            mgx.save(model, f"{file}-{max_seq_len}.mxr", format="msgpack")
        else:
            print("No model found. Please download it and re-try.")
            sys.exit(1)
        return model

    @measure
    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors="np").input_ids

    @measure
    def get_input_features_for_input_ids(self, input_ids):
        input_ids_len = len(input_ids[0])
        padding_len = self.max_seq_len - input_ids_len
        input_ids = np.hstack([input_ids,
                               np.zeros((1, padding_len))]).astype(np.int64)
        # 0 masked | 1 un-masked
        attention_mask = np.array([1] * input_ids_len +
                                  [0] * padding_len).astype(np.int64)
        attention_mask = attention_mask[np.newaxis]
        position_ids = np.arange(0, self.max_seq_len, dtype=np.int64)
        position_ids = position_ids[np.newaxis]

        return (input_ids, attention_mask, position_ids)

    @measure
    def decode_step(self, input_ids, attention_mask, position_ids):
        return np.array(
            self.model.run({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids
            })[0])

    @measure
    def decode_tokens(self, generated_tokens, skip_special_tokens=True):
        return ''.join(
            self.tokenizer.decode(generated_tokens,
                                  skip_special_tokens=skip_special_tokens))

    @measure
    def generate(self, input_ids, log_process=False):
        start_timestep = len(input_ids[0]) - 1
        end_timestep = self.max_seq_len

        input_ids, attention_mask, position_ids = self.get_input_features_for_input_ids(
            input_ids)
        print("Generating response...")
        for timestep in range(start_timestep, self.max_seq_len):
            # get logits for the current timestep
            logits = self.decode_step(input_ids, attention_mask, position_ids)
            # greedily get the highest probable token
            new_token = np.argmax(logits[0][timestep])

            # add it to the tokens and unmask it
            input_ids[0][timestep + 1] = new_token
            attention_mask[0][timestep + 1] = 1

            if log_process:
                print(self.decode_tokens(input_ids[0][:timestep + 2]))

            if new_token == self.tokenizer.eos_token_id:
                end_timestep = timestep + 1
                break

        return self.decode_tokens(input_ids[0][:end_timestep + 1])


if __name__ == "__main__":
    args = get_args()
    llama = Llama2MGX(args.max_seq_len)

    print(f"Call tokenizer with \"{args.prompt}\"")

    input_ids = llama.tokenize(args.prompt)
    result = llama.generate(input_ids, log_process=args.log_process)

    print(f"Result text: {result}")
