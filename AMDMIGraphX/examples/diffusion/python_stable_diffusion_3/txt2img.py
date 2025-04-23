#  The MIT License (MIT)
#
#  Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#  Copyright (c) 2024 Stability AI
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

from argparse import ArgumentParser
from diffusers import FlowMatchEulerDiscreteScheduler

from other_impls import SD3Tokenizer

from PIL import Image

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
        default="models/sd3",
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
        choices=["all", "vae", "clip", "mmdit"],
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

    parser.add_argument(
        "--skip-t5",
        action="store_true",
        default=False,
        help=
        "Skip the third text encoder. Small accuracy penalty but large memory savings."
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
        default=50,
        help="Number of steps",
    )

    parser.add_argument("-b",
                        "--batch",
                        type=int,
                        default=1,
                        help="Batch count or number of images to produce")

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Prompt",
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
        default=5.0,
        help="Guidance scale",
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


class StableDiffusionMGX():
    def __init__(self,
                 onnx_model_path,
                 compiled_model_path,
                 fp16,
                 batch,
                 force_compile=False,
                 exhaustive_tune=False,
                 skip_t5=False):

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="scheduler")

        self.tokenizer = SD3Tokenizer()
        self.device = "cuda"
        self.skip_t5 = skip_t5

        if fp16 is None:
            fp16 = []
        elif "all" in fp16:
            fp16 = ["vae", "clip", "mmdit"]

        self.batch = batch

        print("Load models...")
        self.models = {
            "vae":
            StableDiffusionMGX.load_mgx_model(
                "vae_decoder", {"latent_sample": [self.batch, 16, 128, 128]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="vae" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False,
                batch=self.batch),
            "clip-l":
            StableDiffusionMGX.load_mgx_model(
                "text_encoder", {"input_ids": [1, 77]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="clip" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "clip-g":
            StableDiffusionMGX.load_mgx_model(
                "text_encoder_2", {"input_ids": [1, 77]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="clip" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "mmdit":
            StableDiffusionMGX.load_mgx_model(
                "transformer", {
                    "hidden_states": [2 * self.batch, 16, 128, 128],
                    "timestep": [2 * self.batch],
                    "encoder_hidden_states": [2 * self.batch, 154, 4096],
                    "pooled_projections": [2 * self.batch, 2048],
                },
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="mmdit" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False,
                batch=self.batch)
        }

        self.tensors = {
            "clip-g": allocate_torch_tensors(self.models["clip-g"]),
            "clip-l": allocate_torch_tensors(self.models["clip-l"]),
            # "t5xxl": allocate_torch_tensors(self.models["t5xxl"]),
            "mmdit": allocate_torch_tensors(self.models["mmdit"]),
            "vae": allocate_torch_tensors(self.models["vae"]),
        }

        self.model_args = {
            "clip-g": tensors_to_args(self.tensors['clip-g']),
            "clip-l": tensors_to_args(self.tensors['clip-l']),
            # "t5xxl": tensors_to_args(self.tensors['t5xxl']),
            "mmdit": tensors_to_args(self.tensors['mmdit']),
            "vae": tensors_to_args(self.tensors['vae']),
        }

        if not self.skip_t5:
            self.models["t5xxl"] = StableDiffusionMGX.load_mgx_model(
                "text_encoder_3", {"input_ids": [1, 77]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="clip" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False)
            self.tensors["t5xxl"] = allocate_torch_tensors(
                self.models["t5xxl"])
            self.model_args["t5xxl"] = tensors_to_args(self.tensors['t5xxl'])

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
    def run(self, prompt, negative_prompt, steps, seed, scale):
        torch.cuda.synchronize()
        self.profile_start("run")

        print("Tokenizing prompts...")
        prompt_tokens = self.tokenize(prompt)
        neg_prompt_tokens = self.tokenize(negative_prompt)

        print("Creating text embeddings...")
        self.profile_start("clip")
        prompt_embeddings = self.get_embeddings(prompt_tokens)
        neg_prompt_embeddings = self.get_embeddings(neg_prompt_tokens)
        self.profile_end("clip")

        # fix height and width for now
        # TODO: check for valid height/width combinations
        # and make them member variables
        height = 1024
        width = 1024
        latent = torch.empty(1, 16, height // 8, width // 8, device="cpu")

        generator = torch.manual_seed(seed)
        latent = torch.randn(latent.size(),
                             dtype=torch.float32,
                             layout=latent.layout,
                             generator=generator).to(latent.dtype)

        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps

        print("Running denoising loop...")
        self.profile_start("denoise")
        for step in timesteps:
            latent = self.denoise(latent, prompt_embeddings,
                                  neg_prompt_embeddings, step, scale)

        self.profile_end("denoise")

        latent = (latent / 1.5305) + 0.0609

        self.profile_start("decode")
        print("Decode denoised result...")
        image = self.decode(latent)
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
        print('mmditx{}\t{:>9.2f} ms'.format(
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
                       offload_copy=True,
                       batch=1):
        print(f"Loading {name} model...")
        if compiled_model_path is None:
            compiled_model_path = onnx_model_path
        onnx_file = f"{onnx_model_path}/{name}/model.onnx"
        mxr_file = f"{compiled_model_path}/{name}/model_{'fp16' if use_fp16 else 'fp32'}_b{batch}_{'gpu' if not offload_copy else 'oc'}.mxr"
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
    def tokenize(self, prompt):
        return self.tokenizer.tokenize_with_weights(prompt)

    def encode_token_weights(self, model_name, token_weight_pairs):
        tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        tokens = torch.tensor([tokens], dtype=torch.int64, device=self.device)
        copy_tensor_sync(self.tensors[model_name]["input_ids"],
                         tokens.to(torch.int32))
        run_model_sync(self.models[model_name], self.model_args[model_name])
        encoder_out = self.tensors[model_name][get_output_name(0)]
        encoder_out2 = None
        if model_name != 't5xxl':
            # flipped outputs for clip text encoders...
            encoder_out2 = encoder_out
            encoder_out = self.tensors[model_name][get_output_name(1)]

        if encoder_out2 is not None:
            first_pooled = encoder_out2[0:1]
        else:
            first_pooled = encoder_out2
        output = [encoder_out[0:1]]

        return torch.cat(output, dim=-2), first_pooled

    @measure
    def get_embeddings(self, prompt_tokens):
        l_out, l_pooled = self.encode_token_weights("clip-l",
                                                    prompt_tokens["l"])
        # stable-diffusion-3-lite-onnx has swapped outputs for clip-l text encoder
        if l_out.shape != (1, 77, 768):
            l_out, l_pooled = l_pooled, l_out

        g_out, g_pooled = self.encode_token_weights("clip-g",
                                                    prompt_tokens["g"])
        if not self.skip_t5:
            t5_out, _ = self.encode_token_weights("t5xxl",
                                                  prompt_tokens["t5xxl"])
        else:
            t5_out = torch.zeros((1, 77, 4096)).cuda()
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))

        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1)

    @staticmethod
    def convert_to_rgb_image(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return [Image.fromarray(images[i]) for i in range(images.shape[0])]

    @staticmethod
    def save_image(pil_image, filename="output.png"):
        pil_image.save(filename)

    def CFGDenoiser(self, x, timestep, cond, uncond, cond_scale):
        # Run cond and uncond in a batch together
        x_concat = torch.cat([x, x])
        timestep_concat = timestep.expand([2])
        c_crossattn = torch.cat([cond["c_crossattn"], uncond["c_crossattn"]])
        y = torch.cat([cond["y"], uncond["y"]])

        copy_tensor_sync(self.tensors["mmdit"]["hidden_states"], x_concat)
        copy_tensor_sync(self.tensors["mmdit"]["timestep"], timestep_concat)
        copy_tensor_sync(self.tensors["mmdit"]["encoder_hidden_states"],
                         c_crossattn)
        copy_tensor_sync(self.tensors["mmdit"]["pooled_projections"], y)

        run_model_sync(self.models["mmdit"], self.model_args['mmdit'])

        mmdit_out = self.tensors["mmdit"][get_output_name(0)]

        # Then split and apply CFG Scaling
        pos_out, neg_out = torch.tensor_split(mmdit_out, 2)

        scaled = neg_out + (pos_out - neg_out) * cond_scale

        # scheduler step function requies all tensors be on the CPU
        scaled = scaled.detach().clone().cpu()
        scheduler_out = self.scheduler.step(model_output=scaled,
                                            timestep=timestep,
                                            sample=x,
                                            return_dict=False)[0]
        return scheduler_out

    def fix_cond(self, cond):
        cond, pooled = (cond[0].cuda(), cond[1].cuda())
        return {"c_crossattn": cond, "y": pooled}

    def denoise(self, latent, conditioning, neg_cond, step, cfg_scale):
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        return self.CFGDenoiser(latent, step, conditioning, neg_cond,
                                cfg_scale)

    @measure
    def decode(self, latents):
        copy_tensor_sync(self.tensors["vae"]["latent_sample"], latents)
        run_model_sync(self.models["vae"], self.model_args["vae"])
        return self.tensors["vae"][get_output_name(0)]

    @measure
    def warmup(self, num_runs):
        self.profile_start("warmup")
        copy_tensor_sync(self.tensors["clip-l"]["input_ids"],
                         torch.ones((1, 77)).to(torch.int32))
        copy_tensor_sync(self.tensors["clip-g"]["input_ids"],
                         torch.ones((1, 77)).to(torch.int32))
        if not self.skip_t5:
            copy_tensor_sync(self.tensors["t5xxl"]["input_ids"],
                             torch.ones((1, 77)).to(torch.int32))
        copy_tensor_sync(
            self.tensors["mmdit"]["hidden_states"],
            torch.randn((2 * self.batch, 16, 128, 128)).to(torch.float))
        copy_tensor_sync(self.tensors["mmdit"]["timestep"],
                         torch.randn((2 * self.batch)).to(torch.float))
        copy_tensor_sync(
            self.tensors["mmdit"]["encoder_hidden_states"],
            torch.randn((2 * self.batch, 154, 4096)).to(torch.float))
        copy_tensor_sync(self.tensors["mmdit"]["pooled_projections"],
                         torch.randn((2 * self.batch, 2048)).to(torch.float))
        copy_tensor_sync(
            self.tensors["vae"]["latent_sample"],
            torch.randn((self.batch, 16, 128, 128)).to(torch.float))

        for _ in range(num_runs):
            run_model_sync(self.models["clip-l"], self.model_args["clip-l"])
            run_model_sync(self.models["clip-g"], self.model_args["clip-g"])
            if not self.skip_t5:
                run_model_sync(self.models["t5xxl"], self.model_args["t5xxl"])
            run_model_sync(self.models["mmdit"], self.model_args["mmdit"])
            run_model_sync(self.models["vae"], self.model_args["vae"])
        self.profile_end("warmup")


if __name__ == "__main__":
    args = get_args()

    sd = StableDiffusionMGX(args.onnx_model_path, args.compiled_model_path,
                            args.fp16, args.batch, args.force_compile,
                            args.exhaustive_tune, args.skip_t5)
    print("Warmup")
    sd.warmup(5)
    print("Run")
    result = sd.run(args.prompt, args.negative_prompt, args.steps, args.seed,
                    args.scale)

    print("Summary")
    sd.print_summary(args.steps)
    print("Cleanup")
    sd.cleanup()

    print("Convert result to rgb image...")
    images = StableDiffusionMGX.convert_to_rgb_image(result)
    for i, image in enumerate(images):
        filename = f"{args.batch}_{args.output}" if args.output else f"output_s{args.seed}_t{args.steps}_{i}.png"
        StableDiffusionMGX.save_image(image, filename)
        print(f"Image saved to {filename}")
