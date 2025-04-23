#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import torch
from transformers import (CLIPTokenizer, T5TokenizerFast, CLIPTextModel,
                          T5EncoderModel)
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
import migraphx as mgx


class MGXModel:
    def __init__(self,
                 model,
                 input_shapes=None,
                 fp16=False,
                 bf16=True,
                 exhaustive_tune=False,
                 config=None):

        if isinstance(model, mgx.program):
            self.model = model

        elif isinstance(model, str) and os.path.isfile(model):

            if model.endswith(".mxr"):
                self.model = mgx.load(model, format="msgpack")

            elif model.endswith(".onnx"):
                if not input_shapes:
                    raise ValueError(
                        "input_shapes need to be specified for loading a .onnx file"
                    )
                self.model = mgx.parse_onnx(model, map_input_dims=input_shapes)
                if fp16:
                    mgx.quantize_fp16(self.model)
                elif bf16:
                    mgx.quantize_bf16(self.model)
                self.model.compile(mgx.get_target("gpu"),
                                   exhaustive_tune=exhaustive_tune,
                                   offload_copy=False)
            else:
                raise ValueError(
                    f"File type not recognized (should eend with .mxr or .onnx): {model}"
                )
        else:
            raise ValueError(
                "model should be a migraphx.program object or path to .mxr/.onnx file"
            )
        self.config = config

        self.mgx_to_torch_dtype_dict = {
            "bool_type": torch.bool,
            "uint8_type": torch.uint8,
            "int8_type": torch.int8,
            "int16_type": torch.int16,
            "int32_type": torch.int32,
            "int64_type": torch.int64,
            "float_type": torch.float32,
            "double_type": torch.float64,
            "half_type": torch.float16,
        }
        self.torch_to_mgx_dtype_dict = {
            v: k
            for k, v in self.mgx_to_torch_dtype_dict.items()
        }

        self.input_names = []
        self.output_names = []

        for n in self.model.get_parameter_names():
            if "main:#output_" in n:
                self.output_names.append(n)
            else:
                self.input_names.append(n)

        self.torch_buffers = {}
        self.mgx_args = {}

        self.start_events = []
        self.end_events = []

        self.prealloc_buffers(self.output_names)

    def run_async(self, stream=None, **inputs):
        if stream is None:
            stream = torch.cuda.current_stream()

        for name, tensor in inputs.items():
            self.mgx_args[name] = self.tensor_to_arg(tensor)

        self.start_events.append(torch.cuda.Event(enable_timing=True))
        self.end_events.append(torch.cuda.Event(enable_timing=True))

        self.start_events[-1].record()
        self.model.run_async(self.mgx_args, stream.cuda_stream, "ihipStream_t")
        self.end_events[-1].record()

        return {p: self.torch_buffers[p] for p in self.output_names}

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mgx.save(self.model, path, format="msgpack")

    def tensor_to_arg(self, tensor):
        mgx_shape = mgx.shape(type=self.torch_to_mgx_dtype_dict[tensor.dtype],
                              lens=list(tensor.size()),
                              strides=list(tensor.stride()))
        return mgx.argument_from_pointer(mgx_shape, tensor.data_ptr())

    def prealloc_buffers(self, param_names):
        for param_name in param_names:
            param_shape = self.model.get_parameter_shapes()[param_name]

            type_str, lens = param_shape.type_string(), param_shape.lens()
            strides = param_shape.strides()
            torch_dtype = self.mgx_to_torch_dtype_dict[type_str]
            tensor = torch.empty_strided(lens,
                                         strides,
                                         dtype=torch_dtype,
                                         device=torch.cuda.current_device())
            self.torch_buffers[param_name] = tensor
            self.mgx_args[param_name] = self.tensor_to_arg(tensor)

    def get_run_times(self):
        return [
            s.elapsed_time(e)
            for s, e in zip(self.start_events, self.end_events)
        ]

    def clear_events(self):
        self.start_events = []
        self.end_events = []


def get_scheduler(local_dir, hf_model_path, scheduler_dir="scheduler"):
    scheduler_local_dir = os.path.join(local_dir, scheduler_dir)
    scheduler_cls = FlowMatchEulerDiscreteScheduler

    if not os.path.exists(scheduler_local_dir):
        model = scheduler_cls.from_pretrained(hf_model_path,
                                              subfolder=scheduler_dir)
        model.save_pretrained(scheduler_local_dir)
    else:
        print(f"Loading {scheduler_cls} scheduler from {scheduler_local_dir}")
        model = scheduler_cls.from_pretrained(scheduler_local_dir)

    return model


def get_tokenizer(local_dir,
                  hf_model_path,
                  tokenizer_type="clip",
                  tokenizer_dir="tokenizer"):
    tokenizer_local_dir = os.path.join(local_dir, tokenizer_dir)
    if tokenizer_type == "clip":
        tokenizer_class = CLIPTokenizer
    elif tokenizer_type == "t5":
        tokenizer_class = T5TokenizerFast
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")

    if not os.path.exists(tokenizer_local_dir):
        model = tokenizer_class.from_pretrained(hf_model_path,
                                                subfolder=tokenizer_dir)
        model.save_pretrained(tokenizer_local_dir)
    else:
        print(f"Loading {tokenizer_type} tokenizer from {tokenizer_local_dir}")
        model = tokenizer_class.from_pretrained(tokenizer_local_dir)

    return model


def get_local_path(local_dir, model_dir):
    model_local_dir = os.path.join(local_dir, model_dir)
    if not os.path.exists(model_local_dir):
        os.makedirs(model_local_dir)
    return model_local_dir


def get_clip_model(local_dir,
                   hf_model_path,
                   compiled_dir,
                   model_dir="text_encoder",
                   torch_dtype=torch.float32,
                   bs=1,
                   exhaustive_tune=False,
                   fp16=False,
                   bf16=True):
    clip_local_dir = get_local_path(local_dir, model_dir)
    onnx_file = "model.onnx"
    onnx_path = os.path.join(clip_local_dir, onnx_file)

    def get_compiled_file_name():
        name = f"model_b{bs}"
        if fp16:
            name += "_fp16"
        elif bf16:
            name += "_bf16"

        if exhaustive_tune: name += "_exh"
        return name + ".mxr"

    clip_compiled_dir = get_local_path(compiled_dir, model_dir)
    mxr_file = get_compiled_file_name()
    mxr_path = os.path.join(clip_compiled_dir, mxr_file)

    if os.path.isfile(mxr_path):
        print(f"found compiled model.. loading CLIP encoder from {mxr_path}")
        model = MGXModel(mxr_path)
        return model

    sample_inputs = (torch.zeros(bs, 77, dtype=torch.int32), )
    input_names = ["input_ids"]
    if not os.path.isfile(onnx_path):
        print("ONNX file not found.. exporting CLIP encoder to ONNX")
        model = CLIPTextModel.from_pretrained(hf_model_path,
                                              subfolder=model_dir,
                                              torch_dtype=torch_dtype)

        output_names = ["text_embeddings"]
        dynamic_axes = {"input_ids": {0: 'B'}, "text_embeddings": {0: 'B'}}

        # CLIP export requires nightly pytorch due to bug in onnx parser
        with torch.inference_mode():
            torch.onnx.export(model,
                              sample_inputs,
                              onnx_path,
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes)

    assert os.path.isfile(onnx_path)
    print(f"Generating MXR from ONNX file: {onnx_path}")
    input_shapes = {
        n: list(t.size())
        for n, t in zip(input_names, sample_inputs)
    }
    model = MGXModel(onnx_path,
                     input_shapes=input_shapes,
                     exhaustive_tune=exhaustive_tune,
                     fp16=fp16,
                     bf16=bf16)
    model.save_model(os.path.join(clip_compiled_dir, get_compiled_file_name()))

    return model

    # migraphx-driver perf FLUX.1-schnell/text_encoder/model.onnx --input-dim @input_ids 1 77 --fill1 input_ids --fp16


def get_t5_model(local_dir,
                 hf_model_path,
                 compiled_dir,
                 max_len=512,
                 model_dir="text_encoder_2",
                 torch_dtype=torch.float32,
                 bs=1,
                 exhaustive_tune=False,
                 fp16=False,
                 bf16=True):
    t5_local_dir = get_local_path(local_dir, model_dir)
    onnx_file = "model.onnx"
    onnx_path = os.path.join(t5_local_dir, onnx_file)

    def get_compiled_file_name():
        name = f"model_b{bs}"
        name += f"_l{max_len}"
        if fp16:
            name += "_fp16"
        elif bf16:
            name += "_bf16"
        if exhaustive_tune: name += "_exh"
        return name + ".mxr"

    t5_compiled_dir = get_local_path(compiled_dir, model_dir)
    mxr_file = get_compiled_file_name()
    mxr_path = os.path.join(t5_compiled_dir, mxr_file)

    if os.path.isfile(mxr_path):
        print(f"found compiled model.. loading T5 encoder from {mxr_path}")
        model = MGXModel(mxr_path)
        return model

    sample_inputs = (torch.zeros(bs, max_len, dtype=torch.int32), )
    input_names = ["input_ids"]
    if not os.path.isfile(onnx_path):
        model = T5EncoderModel.from_pretrained(hf_model_path,
                                               subfolder=model_dir,
                                               torch_dtype=torch_dtype)
        output_names = ["text_embeddings"]
        dynamic_axes = {"input_ids": {0: 'B'}, "text_embeddings": {0: 'B'}}

        with torch.inference_mode():
            torch.onnx.export(model,
                              sample_inputs,
                              onnx_path,
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes)

    assert os.path.isfile(onnx_path)
    print(f"Generating MXR from ONNX file: {onnx_path}")
    input_shapes = {
        n: list(t.size())
        for n, t in zip(input_names, sample_inputs)
    }
    model = MGXModel(onnx_path,
                     input_shapes=input_shapes,
                     exhaustive_tune=exhaustive_tune,
                     fp16=fp16,
                     bf16=bf16)
    model.save_model(os.path.join(t5_compiled_dir, get_compiled_file_name()))

    return model

    # migraphx-driver perf FLUX.1-schnell/text_encoder_2/model.onnx --input-dim @input_ids 1 512 --fill1 input_ids --fp16


## Following decorators required to apply fp16 inference patch to the transformer blocks
## Note that we do not export fp16 weights directly to ONNX to allow migraphx to
##  perform optimizations before quantizing down to fp16. This results in better
##  accuracy compared to exporting fp16 directly to onnx
def transformer_block_clip_wrapper(fn):
    def new_forward(*args, **kwargs):
        encoder_hidden_states, hidden_states = fn(*args, **kwargs)
        return encoder_hidden_states.clip(-65504, 65504), hidden_states

    return new_forward


def single_transformer_block_clip_wrapper(fn):
    def new_forward(*args, **kwargs):
        hidden_states = fn(*args, **kwargs)
        return hidden_states.clip(-65504, 65504)

    return new_forward


def add_output_clippings_for_fp16(model):
    for b in model.transformer_blocks:
        b.forward = transformer_block_clip_wrapper(b.forward)

    for b in model.single_transformer_blocks:
        b.forward = single_transformer_block_clip_wrapper(b.forward)


def get_flux_transformer_model(local_dir,
                               hf_model_path,
                               compiled_dir,
                               img_height=1024,
                               img_width=1024,
                               compression_factor=8,
                               max_len=512,
                               model_dir="transformer",
                               torch_dtype=torch.float32,
                               bs=1,
                               exhaustive_tune=False,
                               fp16=False,
                               bf16=True):

    transformer_local_dir = get_local_path(local_dir, model_dir)
    onnx_file = "model_clip_workaround.onnx" if fp16 else "model.onnx"
    onnx_path = os.path.join(transformer_local_dir, onnx_file)
    latent_h, latent_w = img_height // compression_factor, img_width // compression_factor

    def get_compiled_file_name():
        name = f"model_b{bs}"
        name += f"_h{latent_h}_w{latent_w}_l{max_len}"
        if fp16:
            name += "_fp16"
        elif bf16:
            name += "_bf16"
        if exhaustive_tune: name += "_exh"
        return name + ".mxr"

    transformer_compiled_dir = get_local_path(compiled_dir, model_dir)
    mxr_file = get_compiled_file_name()
    mxr_path = os.path.join(transformer_compiled_dir, mxr_file)

    config = FluxTransformer2DModel.load_config(hf_model_path,
                                                subfolder=model_dir)

    if os.path.isfile(mxr_path):
        print(
            f"found compiled model.. loading flux transformer from {mxr_path}")
        model = MGXModel(mxr_path, config=config)
        return model

    sample_inputs = (
        torch.randn(bs, (latent_h // 2) * (latent_w // 2),
                    config["in_channels"],
                    dtype=torch_dtype),
        torch.randn(bs,
                    max_len,
                    config['joint_attention_dim'],
                    dtype=torch_dtype),
        torch.randn(bs, config['pooled_projection_dim'], dtype=torch_dtype),
        torch.tensor([1.] * bs, dtype=torch_dtype),
        torch.randn((latent_h // 2) * (latent_w // 2), 3, dtype=torch_dtype),
        torch.randn(max_len, 3, dtype=torch_dtype),
        torch.tensor([1.] * bs, dtype=torch_dtype),
    )

    input_names = [
        'hidden_states', 'encoder_hidden_states', 'pooled_projections',
        'timestep', 'img_ids', 'txt_ids', 'guidance'
    ]
    if not os.path.isfile(onnx_path):
        model = FluxTransformer2DModel.from_pretrained(hf_model_path,
                                                       subfolder=model_dir,
                                                       torch_dtype=torch_dtype)

        if fp16:
            print("applying fp16 clip workarounds to transformer")
            add_output_clippings_for_fp16(model)

        output_names = ["latent"]
        dynamic_axes = {
            'hidden_states': {
                0: 'B',
                1: 'latent_dim'
            },
            'encoder_hidden_states': {
                0: 'B',
                1: 'L'
            },
            'pooled_projections': {
                0: 'B'
            },
            'timestep': {
                0: 'B'
            },
            'img_ids': {
                0: 'latent_dim'
            },
            'txt_ids': {
                0: 'L'
            },
            'guidance': {
                0: 'B'
            },
        }

        with torch.inference_mode():
            torch.onnx.export(model,
                              sample_inputs,
                              onnx_path,
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes)

    assert os.path.isfile(onnx_path)
    print(f"Generating MXR from ONNX file: {onnx_path}")
    input_shapes = {
        n: list(t.size())
        for n, t in zip(input_names, sample_inputs)
    }
    model = MGXModel(onnx_path,
                     input_shapes=input_shapes,
                     exhaustive_tune=exhaustive_tune,
                     fp16=fp16,
                     bf16=bf16,
                     config=config)
    model.save_model(
        os.path.join(transformer_compiled_dir, get_compiled_file_name()))

    return model
    # migraphx-driver perf FLUX.1-schnell/transformer/model.onnx --input-dim @hidden_states 1 4096 64 @encoder_hidden_states 1 512 4096 @pooled_projections 1 768 @timestep 1 @img_ids 4096 3 @txt_ids 512 3 --fp16
    # migraphx-driver perf FLUX.1-dev/transformer/model.onnx --input-dim @hidden_states 1 4096 64 @encoder_hidden_states 1 512 4096 @pooled_projections 1 768 @timestep 1 @img_ids 4096 3 @txt_ids 512 3 @guidance 1 --fp16


def get_vae_model(local_dir,
                  hf_model_path,
                  compiled_dir,
                  img_height=1024,
                  img_width=1024,
                  compression_factor=8,
                  model_dir="vae",
                  torch_dtype=torch.float32,
                  bs=1,
                  exhaustive_tune=False,
                  fp16=False,
                  bf16=True):

    vae_local_dir = get_local_path(local_dir, model_dir)
    onnx_file = "model.onnx"
    onnx_path = os.path.join(vae_local_dir, onnx_file)
    latent_h, latent_w = img_height // compression_factor, img_width // compression_factor

    def get_compiled_file_name():
        name = f"model_b{bs}"
        name += f"_h{latent_h}_w{latent_w}"
        if fp16:
            name += "_fp16"
        elif bf16:
            name += "_bf16"
        if exhaustive_tune: name += "_exh"
        return name + ".mxr"

    vae_compiled_dir = get_local_path(compiled_dir, model_dir)
    mxr_file = get_compiled_file_name()
    mxr_path = os.path.join(vae_compiled_dir, mxr_file)

    config = AutoencoderKL.load_config(hf_model_path, subfolder=model_dir)

    if os.path.isfile(mxr_path):
        print(f"found compiled model.. loading VAE decoder from {mxr_path}")
        model = MGXModel(mxr_path, config=config)
        return model

    sample_inputs = (torch.randn(bs,
                                 config['latent_channels'],
                                 latent_h,
                                 latent_w,
                                 dtype=torch_dtype), )
    input_names = ["latent"]
    if not os.path.isfile(onnx_path):
        model = AutoencoderKL.from_pretrained(hf_model_path,
                                              subfolder=model_dir,
                                              torch_dtype=torch_dtype)
        model.forward = model.decode

        output_names = ["images"]
        dynamic_axes = {
            'latent': {
                0: 'B',
                2: 'H',
                3: 'W'
            },
            'images': {
                0: 'B',
                2: '8H',
                3: '8W'
            }
        }

        with torch.inference_mode():
            torch.onnx.export(model,
                              sample_inputs,
                              onnx_path,
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes)

    assert os.path.isfile(onnx_path)
    print(f"Generating MXR from ONNX file: {onnx_path}")
    input_shapes = {
        n: list(t.size())
        for n, t in zip(input_names, sample_inputs)
    }
    model = MGXModel(onnx_path,
                     input_shapes=input_shapes,
                     exhaustive_tune=exhaustive_tune,
                     fp16=fp16,
                     bf16=bf16,
                     config=config)
    model.save_model(os.path.join(vae_compiled_dir, get_compiled_file_name()))

    return model
    # migraphx-driver perf FLUX.1-schnell/vae/model.onnx --input-dim @latent 1 16 128 128 --fp16
