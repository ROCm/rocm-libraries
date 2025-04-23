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
import warnings
import time
from tabulate import tabulate
import numpy as np
import torch
from models import (get_tokenizer, get_clip_model, get_t5_model,
                    get_flux_transformer_model, get_vae_model, get_scheduler)
from PIL import Image
# import migraphx as mgx


def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FluxPipeline:
    def __init__(self,
                 hf_model_path="black-forest-labs/FLUX.1-dev",
                 local_dir=None,
                 compile_dir=None,
                 pipeline_type="txt2img",
                 img_height=1024,
                 img_width=1024,
                 guidance_scale=3.5,
                 max_sequence_length=512,
                 batch_size=1,
                 denoising_steps=50,
                 fp16=False,
                 bf16=True,
                 exhaustive_tune=False,
                 manual_seed=None):

        self.hf_model_path = hf_model_path
        self.height = img_height
        self.width = img_width
        self.guidance_scale = guidance_scale
        self.max_sequence_length = max_sequence_length
        self.pipeline_type = pipeline_type
        self.bs = batch_size
        self.steps = denoising_steps
        self.fp16 = fp16
        self.bf16 = bf16
        self.exhaustive_tune = exhaustive_tune

        if not local_dir:
            self.local_dir = self.hf_model_path.split("/")[-1]
        if not compile_dir:
            self.compile_dir = self.hf_model_path.split("/")[-1] + "_compiled"

        self.models = {}

        # self.stages = ["clip", "t5", "transformer", "vae"]

        self.generator = torch.Generator(device="cuda")
        if manual_seed:
            self.generator.manual_seed(manual_seed)
        self.device = torch.cuda.current_device()

        self.times = []

    def load_models(self):
        self.scheduler = get_scheduler(self.local_dir, self.hf_model_path)
        self.tokenizer = get_tokenizer(self.local_dir, self.hf_model_path)
        self.tokenizer2 = get_tokenizer(self.local_dir, self.hf_model_path,
                                        "t5", "tokenizer_2")

        self.clip = get_clip_model(self.local_dir,
                                   self.hf_model_path,
                                   self.compile_dir,
                                   fp16=self.fp16,
                                   bf16=self.bf16,
                                   bs=self.bs,
                                   exhaustive_tune=self.exhaustive_tune)

        self.t5 = get_t5_model(self.local_dir,
                               self.hf_model_path,
                               self.compile_dir,
                               self.max_sequence_length,
                               bs=self.bs,
                               bf16=self.bf16,
                               exhaustive_tune=self.exhaustive_tune)

        self.flux_transformer = get_flux_transformer_model(
            self.local_dir,
            self.hf_model_path,
            self.compile_dir,
            img_height=self.height,
            img_width=self.width,
            max_len=self.max_sequence_length,
            fp16=self.fp16,
            bf16=self.bf16,
            bs=self.bs,
            exhaustive_tune=self.exhaustive_tune)

        self.vae = get_vae_model(self.local_dir,
                                 self.hf_model_path,
                                 self.compile_dir,
                                 img_height=self.height,
                                 img_width=self.width,
                                 bs=self.bs,
                                 bf16=self.bf16,
                                 exhaustive_tune=self.exhaustive_tune)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height,
                      width):
        """
        Reshapes latents from (B, C, H, W) to (B, H/2, W/2, C*4) as expected by the denoiser
        """
        latents = latents.view(batch_size, num_channels_latents, height // 2,
                               2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2),
                                  num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """
        Reshapes denoised latents to the format (B, C, H, W)
        """
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2,
                                  width * 2)

        return latents

    @staticmethod
    def _prepare_latent_image_ids(height, width, dtype, device):
        """
        Prepares latent image indices
        """
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = (latent_image_ids[..., 1] +
                                    torch.arange(height // 2)[:, None])
        latent_image_ids[..., 2] = (latent_image_ids[..., 2] +
                                    torch.arange(width // 2)[None, :])

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape)

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width,
            latent_image_id_channels)

        return latent_image_ids.to(device=device, dtype=dtype)

    def initialize_latents(
            self,
            batch_size,
            num_channels_latents,
            latent_height,
            latent_width,
            latents_dtype=torch.float32,
    ):
        latents_dtype = latents_dtype  # text_embeddings.dtype
        latents_shape = (batch_size, num_channels_latents, latent_height,
                         latent_width)
        latents = torch.randn(
            latents_shape,
            device=torch.cuda.current_device(),
            dtype=latents_dtype,
            generator=self.generator,
        )

        latents = self._pack_latents(latents, batch_size, num_channels_latents,
                                     latent_height, latent_width)

        latent_image_ids = self._prepare_latent_image_ids(
            latent_height, latent_width, latents_dtype, self.device)

        return latents, latent_image_ids

    def encode_prompt(self,
                      prompt,
                      encoder="clip",
                      max_sequence_length=None,
                      pooled_output=False):
        tokenizer = self.tokenizer2 if encoder == "t5" else self.tokenizer
        encoder = self.t5 if encoder == "t5" else self.clip
        max_sequence_length = (tokenizer.model_max_length
                               if max_sequence_length is None else
                               max_sequence_length)

        def tokenize(prompt, max_sequence_length):
            text_input_ids = (tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            ).input_ids.type(torch.int32).to(self.device))

            untruncated_ids = tokenizer(prompt,
                                        padding="longest",
                                        return_tensors="pt").input_ids.type(
                                            torch.int32).to(self.device)
            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, max_sequence_length - 1:-1])
                warnings.warn(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f"{max_sequence_length} tokens: {removed_text}")

            # NOTE: output tensor for the encoder must be cloned because it will be overwritten when called again for prompt2
            outputs = encoder.run_async(input_ids=text_input_ids)
            output_name = ("main:#output_0"
                           if not pooled_output else "main:#output_1")
            text_encoder_output = outputs[output_name].clone()

            return text_encoder_output

        # Tokenize prompt
        text_encoder_output = tokenize(prompt, max_sequence_length)

        # return (text_encoder_output.to(torch.float16)
        #         if self.fp16 else text_encoder_output)
        return text_encoder_output

    def denoise_latent(
            self,
            latents,
            timesteps,
            text_embeddings,
            pooled_embeddings,
            text_ids,
            latent_image_ids,
            denoiser="transformer",
            guidance=None,
    ):

        # handle guidance
        if self.flux_transformer.config["guidance_embeds"] and guidance is None:
            guidance = torch.full([latents.shape[0]],
                                  self.guidance_scale,
                                  device=self.device,
                                  dtype=torch.float32)

        for step_index, timestep in enumerate(timesteps):
            # prepare inputs
            timestep_inp = timestep.expand(latents.shape[0]).to(latents.dtype)
            params = {
                "hidden_states": latents,
                "timestep": timestep_inp / 1000,
                "pooled_projections": pooled_embeddings,
                "encoder_hidden_states": text_embeddings,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
            }
            if guidance is not None:
                params.update({"guidance": guidance})

            noise_pred = self.flux_transformer.run_async(
                **params)["main:#output_0"]

            latents = self.scheduler.step(noise_pred,
                                          timestep,
                                          latents,
                                          return_dict=False)[0]

        return latents.to(dtype=torch.float32)

    def decode_latent(self, latents):
        images = self.vae.run_async(latent=latents)["main:#output_0"]
        return images

    def infer(self, prompt, prompt2, warmup=False):
        assert len(prompt) == len(prompt2)
        batch_size = len(prompt)

        self.vae_scale_factor = 2**(len(self.vae.config["block_out_channels"]))
        latent_height = 2 * (int(self.height) // self.vae_scale_factor)
        latent_width = 2 * (int(self.width) // self.vae_scale_factor)

        num_inference_steps = self.steps

        with torch.inference_mode():
            torch.cuda.synchronize()

            self.e2e_tic = time.perf_counter()

            latents, latent_image_ids = self.initialize_latents(
                batch_size=batch_size,
                num_channels_latents=self.flux_transformer.
                config["in_channels"] // 4,
                # num_channels_latents=16,
                latent_height=latent_height,
                latent_width=latent_width,
                # latents_dtype=torch.float16 if self.fp16 else torch.float32,
                latents_dtype=torch.float32)

            pooled_embeddings = self.encode_prompt(prompt, pooled_output=True)
            text_embeddings = self.encode_prompt(
                prompt2,
                encoder="t5",
                max_sequence_length=self.max_sequence_length)
            text_ids = torch.zeros(text_embeddings.shape[1],
                                   3).to(device=self.device,
                                         dtype=text_embeddings.dtype)

            # Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps,
                                 num_inference_steps)
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            self.scheduler.set_timesteps(sigmas=sigmas,
                                         mu=mu,
                                         device=self.device)
            timesteps = self.scheduler.timesteps.to(self.device)
            num_inference_steps = len(timesteps)

            latents = self.denoise_latent(
                latents,
                timesteps,
                text_embeddings,
                pooled_embeddings,
                text_ids,
                latent_image_ids,
            )

            latents = self._unpack_latents(latents, self.height, self.width,
                                           self.vae_scale_factor)
            latents = (latents / self.vae.config["scaling_factor"]
                       ) + self.vae.config["shift_factor"]

            images = self.decode_latent(latents)
            torch.cuda.synchronize()
            self.e2e_toc = time.perf_counter()
            if not warmup:
                self.record_times()

        return images

    def record_times(self):
        self.times.append(self.e2e_toc - self.e2e_tic)

    def save_image(self, images, prefix, output_dir="./"):
        images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(
            0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
        for i in range(images.shape[0]):
            path = os.path.join(output_dir, f"{prefix}_{i}.png")
            Image.fromarray(images[i]).save(path)

    def print_summary(self):
        headers = ["Model", "Latency(ms)"]
        rows = []
        for mod in ("clip", "t5", "flux_transformer", "vae"):
            name = f"{mod} (x{self.steps})" if mod == "flux_transformer" else mod
            rows.append([name, np.average(getattr(self, mod).get_run_times())])

        rows.append(["e2e", np.average(self.times) * 1000])
        print(tabulate(rows, headers=headers))

    def clear_run_data(self):
        self.times = []
        for mod in (self.clip, self.t5, self.flux_transformer, self.vae):
            mod.clear_events()
