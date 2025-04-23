#  The MIT License (MIT)
#
#  Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the 'Software'), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

# Model Download Instructions from diffusers/scripts/
# python3 convert_stable_diffusion_controlnet_to_onnx.py \
#     --model_path runwayml/stable-diffusion-v1-5 \
#     --controlnet_path lllyasviel/sd-controlnet-canny \
#     --output_path sd15-onnx \
#     --fp16

from argparse import ArgumentParser
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer
from PIL import Image
import torchvision.transforms as transforms

import migraphx as mgx
import os
import sys
import torch
import time
from functools import wraps

from hip import hip
from collections import namedtuple
HipEventPair = namedtuple('HipEventPair', ['start', 'end'])


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
    # Model compile
    parser.add_argument(
        "--onnx-model-path",
        type=str,
        default="models/sd15-onnx/",
        help="Path to onnx model files.",
    )

    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default=None,
        help=
        "Path to compiled mxr model files. If not set, it will be saved next to the onnx model.",
    )

    parser.add_argument(
        "--fp16",
        choices=["all", "vae_decoder", "clip", "unet"],
        nargs="+",
        help="Quantize models with fp16 precision.",
    )

    parser.add_argument(
        "--force-compile",
        action="store_true",
        default=False,
        help="Ignore existing .mxr files and override them",
    )

    parser.add_argument(
        "--exhaustive-tune",
        action="store_true",
        default=False,
        help="Perform exhaustive tuning when compiling onnx models",
    )

    # Runtime
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "-t",
        "--steps",
        type=int,
        default=20,
        help="Number of steps",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Prompt",
    )

    parser.add_argument(
        "-i",
        "--control_image",
        type=str,
        required=True,
        help="Control Image",
    )

    parser.add_argument(
        "-n",
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=7.0,
        help="Guidance scale",
    )

    parser.add_argument(
        "--conditioning_scale",
        type=float,
        default=1.0,
        help="Conditioning scale",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output name",
    )
    return parser.parse_args()


mgx_to_torch_dtype_dict = {
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

torch_to_mgx_dtype_dict = {
    value: key
    for (key, value) in mgx_to_torch_dtype_dict.items()
}


def tensor_to_arg(tensor):
    return mgx.argument_from_pointer(
        mgx.shape(
            **{
                "type": torch_to_mgx_dtype_dict[tensor.dtype],
                "lens": list(tensor.size()),
                "strides": list(tensor.stride())
            }), tensor.data_ptr())


def tensors_to_args(tensors):
    return {name: tensor_to_arg(tensor) for name, tensor in tensors.items()}


def get_output_name(idx):
    return f"main:#output_{idx}"


def copy_tensor_sync(tensor, data):
    tensor.copy_(data)
    torch.cuda.synchronize()


def run_model_sync(model, args):
    model.run(args)
    mgx.gpu_sync()


def allocate_torch_tensors(model):
    input_shapes = model.get_parameter_shapes()
    data_mapping = {
        name: torch.zeros(shape.lens()).to(
            mgx_to_torch_dtype_dict[shape.type_string()]).to(device="cuda")
        for name, shape in input_shapes.items()
    }
    return data_mapping


def image_to_tensor(image_path):
    image = Image.open(image_path)
    image = image.resize((512, 512))

    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image)
    return tensor


class StableDiffusionMGX():
    def __init__(self, onnx_model_path, compiled_model_path, fp16,
                 force_compile, exhaustive_tune):
        model_id = "runwayml/stable-diffusion-v1-5"
        print(f"Using {model_id}")

        print("Creating EulerDiscreteScheduler scheduler")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler")

        print("Creating CLIPTokenizer tokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id,
                                                       subfolder="tokenizer")
        if fp16 is None:
            fp16 = []
        elif "all" in fp16:
            fp16 = ["vae_decoder", "clip", "unet"]

        print("Load models...")
        self.models = {
            "vae_decoder":
            StableDiffusionMGX.load_mgx_model(
                "vae_decoder", {"latent_sample": [1, 4, 64, 64]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="vae_decoder" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "clip":
            StableDiffusionMGX.load_mgx_model(
                "text_encoder", {"input_ids": [2, 77]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="clip" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "unet":
            StableDiffusionMGX.load_mgx_model(
                "unet", {
                    "sample": [2, 4, 64, 64],
                    "encoder_hidden_states": [2, 77, 768],
                    "timestep": [1],
                    "controlnet_conds": [1, 1, 3, 512, 512],
                    "conditioning_scales": [1]
                },
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="unet" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False)
        }

        self.tensors = {
            "clip": allocate_torch_tensors(self.models["clip"]),
            "unet": allocate_torch_tensors(self.models["unet"]),
            "vae_decoder": allocate_torch_tensors(self.models["vae_decoder"]),
        }

        self.model_args = {
            "clip": tensors_to_args(self.tensors['clip']),
            "unet": tensors_to_args(self.tensors['unet']),
            "vae_decoder": tensors_to_args(self.tensors['vae_decoder']),
        }

        self.events = {
            "warmup":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
            "run":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
            "clip":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
            "denoise":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
            "decode":
            HipEventPair(start=hip.hipEventCreate()[1],
                         end=hip.hipEventCreate()[1]),
        }

        self.stream = hip.hipStreamCreate()[1]

    def cleanup(self):
        for event in self.events.values():
            hip.hipEventDestroy(event.start)
            hip.hipEventDestroy(event.end)
        hip.hipStreamDestroy(self.stream)

    def profile_start(self, name):
        if name in self.events:
            hip.hipEventRecord(self.events[name].start, None)

    def profile_end(self, name):
        if name in self.events:
            hip.hipEventRecord(self.events[name].end, None)

    @measure
    @torch.no_grad()
    def run(self, prompt, negative_prompt, control_image, steps, seed, scale,
            conditioning_scale):
        torch.cuda.synchronize()
        self.profile_start("run")

        # need to set this for each run
        self.scheduler.set_timesteps(steps, device="cuda")

        print("Tokenizing prompts...")
        prompt_tokens = self.tokenize(prompt, negative_prompt)

        print("Creating text embeddings...")
        self.profile_start("clip")
        text_embeddings = self.get_embeddings(prompt_tokens)
        self.profile_end("clip")

        print(
            f"Creating random input data ({1}x{4}x{64}x{64}) (latents) with seed={seed}..."
        )
        latents = torch.randn(
            (1, 4, 64, 64),
            generator=torch.manual_seed(seed)).to(device="cuda")

        print("Apply initial noise sigma\n")
        latents = latents * self.scheduler.init_noise_sigma

        print("Running denoising loop...")
        self.profile_start("denoise")
        for step, t in enumerate(self.scheduler.timesteps):
            print(f"#{step}/{len(self.scheduler.timesteps)} step")
            latents = self.denoise_step(text_embeddings, latents,
                                        control_image, t, scale,
                                        conditioning_scale)
        self.profile_end("denoise")

        print("Scale denoised result...")
        latents = 1 / 0.18215 * latents

        self.profile_start("decode")
        print("Decode denoised result...")
        image = self.decode(latents)
        self.profile_end("decode")

        torch.cuda.synchronize()
        self.profile_end("run")
        return image

    def print_summary(self, denoise_steps):
        print('WARMUP\t{:>9.2f} ms'.format(
            hip.hipEventElapsedTime(self.events['warmup'].start,
                                    self.events['warmup'].end)[1]))
        print('CLIP\t{:>9.2f} ms'.format(
            hip.hipEventElapsedTime(self.events['clip'].start,
                                    self.events['clip'].end)[1]))
        print('UNetx{}\t{:>9.2f} ms'.format(
            str(denoise_steps),
            hip.hipEventElapsedTime(self.events['denoise'].start,
                                    self.events['denoise'].end)[1]))
        print('VAE-Dec\t{:>9.2f} ms'.format(
            hip.hipEventElapsedTime(self.events['decode'].start,
                                    self.events['decode'].end)[1]))
        print('RUN\t{:>9.2f} ms'.format(
            hip.hipEventElapsedTime(self.events['run'].start,
                                    self.events['run'].end)[1]))

    @staticmethod
    @measure
    def load_mgx_model(name,
                       shapes,
                       onnx_model_path,
                       compiled_model_path=None,
                       use_fp16=False,
                       force_compile=False,
                       exhaustive_tune=False,
                       offload_copy=True):
        print(f"Loading {name} model...")
        if compiled_model_path is None:
            compiled_model_path = onnx_model_path
        onnx_file = f"{onnx_model_path}/{name}/model.onnx"
        mxr_file = f"{compiled_model_path}/{name}/model_{'fp16' if use_fp16 else 'fp32'}_{'gpu' if not offload_copy else 'oc'}.mxr"
        if not force_compile and os.path.isfile(mxr_file):
            print(f"Found mxr, loading it from {mxr_file}")
            model = mgx.load(mxr_file, format="msgpack")
        elif os.path.isfile(onnx_file):
            print(f"No mxr found at {mxr_file}")
            print(f"Parsing from {onnx_file}")
            model = mgx.parse_onnx(onnx_file, map_input_dims=shapes)
            if use_fp16:
                mgx.quantize_fp16(model)
            model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=exhaustive_tune,
                          offload_copy=offload_copy)
            print(f"Saving {name} model to {mxr_file}")
            os.makedirs(os.path.dirname(mxr_file), exist_ok=True)
            mgx.save(model, mxr_file, format="msgpack")
        else:
            print(
                f"No {name} model found at {onnx_file} or {mxr_file}. Please download it and re-try."
            )
            sys.exit(1)
        return model

    @measure
    def tokenize(self, prompt, negative_prompt):
        return self.tokenizer([prompt, negative_prompt],
                              padding="max_length",
                              max_length=self.tokenizer.model_max_length,
                              truncation=True,
                              return_tensors="pt")

    @measure
    def get_embeddings(self, prompt_tokens):
        copy_tensor_sync(self.tensors["clip"]["input_ids"],
                         prompt_tokens.input_ids.to(torch.int32))
        run_model_sync(self.models["clip"], self.model_args["clip"])
        return self.tensors["clip"][get_output_name(0)]

    @staticmethod
    def convert_to_rgb_image(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return Image.fromarray(images[0])

    @staticmethod
    def save_image(pil_image, filename="output.png"):
        pil_image.save(filename)

    @measure
    def denoise_step(self, text_embeddings, latents, control_image, t, scale,
                     conditioning_scale):
        latents_model_input = torch.cat([latents] * 2)
        latents_model_input = self.scheduler.scale_model_input(
            latents_model_input, t).to(torch.float32).to(device="cuda")
        timestep = torch.atleast_1d(t.to(torch.int64)).to(
            device="cuda")  # convert 0D -> 1D

        copy_tensor_sync(self.tensors["unet"]["sample"], latents_model_input)
        copy_tensor_sync(self.tensors["unet"]["encoder_hidden_states"],
                         text_embeddings)
        copy_tensor_sync(self.tensors["unet"]["timestep"], timestep)
        copy_tensor_sync(self.tensors["unet"]["controlnet_conds"],
                         control_image)
        copy_tensor_sync(self.tensors["unet"]["conditioning_scales"],
                         torch.tensor(conditioning_scale))

        run_model_sync(self.models["unet"], self.model_args['unet'])

        noise_pred_text, noise_pred_uncond = torch.tensor_split(
            self.tensors["unet"][get_output_name(0)], 2)

        # perform guidance
        noise_pred = noise_pred_uncond + scale * (noise_pred_text -
                                                  noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        return self.scheduler.step(noise_pred, t, latents).prev_sample

    @measure
    def decode(self, latents):
        copy_tensor_sync(self.tensors["vae_decoder"]["latent_sample"], latents)
        run_model_sync(self.models["vae_decoder"],
                       self.model_args["vae_decoder"])
        return self.tensors["vae_decoder"][get_output_name(0)]

    @measure
    def warmup(self, num_runs):
        self.profile_start("warmup")
        copy_tensor_sync(self.tensors["clip"]["input_ids"],
                         torch.ones((2, 77)).to(torch.int32))
        copy_tensor_sync(self.tensors["unet"]["sample"],
                         torch.randn((2, 4, 64, 64)).to(torch.float32))
        copy_tensor_sync(self.tensors["unet"]["encoder_hidden_states"],
                         torch.randn((2, 77, 768)).to(torch.float32))
        copy_tensor_sync(self.tensors["unet"]["timestep"],
                         torch.atleast_1d(torch.randn(1).to(torch.int64)))
        copy_tensor_sync(self.tensors["vae_decoder"]["latent_sample"],
                         torch.randn((1, 4, 64, 64)).to(torch.float32))

        for _ in range(num_runs):
            run_model_sync(self.models["clip"], self.model_args["clip"])
            run_model_sync(self.models["unet"], self.model_args["unet"])
            run_model_sync(self.models["vae_decoder"],
                           self.model_args["vae_decoder"])
        self.profile_end("warmup")


if __name__ == "__main__":
    args = get_args()

    sd = StableDiffusionMGX(args.onnx_model_path, args.compiled_model_path,
                            args.fp16, args.force_compile,
                            args.exhaustive_tune)
    print("Warmup")
    sd.warmup(5)
    print("Run")
    result = sd.run(args.prompt, args.negative_prompt,
                    image_to_tensor(args.control_image), args.steps, args.seed,
                    args.scale, args.conditioning_scale)

    print("Summary")
    sd.print_summary(args.steps)
    print("Cleanup")
    sd.cleanup()

    print("Convert result to rgb image...")
    image = StableDiffusionMGX.convert_to_rgb_image(result)
    filename = args.output if args.output else f"output_s{args.seed}_t{args.steps}.png"
    StableDiffusionMGX.save_image(image, filename)
    print(f"Image saved to {filename}")
