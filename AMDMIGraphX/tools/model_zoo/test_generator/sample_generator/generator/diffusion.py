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
import os
import onnxruntime as ort
import numpy as np
import torch

from .generic import _inference
from ..utils import get_model_io


class DiffusionStage(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self._initilize()

    def _initilize(self):
        self.inputs, self.outputs = get_model_io(self.model_path)
        self.session = ort.InferenceSession(self.model_path)
        self.folder_name_prefix = f"{os.path.dirname(self.model_path)}/test_data_set"

    def inference(self, input_data_map, test_idx):
        folder_name = f"{self.folder_name_prefix}_{test_idx}"
        output_data_map = _inference(
            self.session,
            input_data_map,
            self.outputs,
            folder_name,
        )
        return output_data_map


def generate_diffusion_data(model,
                            image_dataset,
                            prompt_dataset,
                            output_folder_prefix=None,
                            sample_limit=None,
                            decode_limit=None):
    assert model.is_diffuser() == True, "Only diffuser supported"
    output_path = f"{output_folder_prefix or 'generated'}/diffusion/{model.name()}"

    try:
        model_paths = model.get_model(output_path)
    except Exception as e:
        print(f"Something went wrong:\n{e}\nSkipping model...")
        return

    is_xl = 'xl' in model.name()
    assert len(model_paths) == (4 + is_xl)
    text_encoder = DiffusionStage(model_paths[0])
    text_encoder_2 = DiffusionStage(model_paths[1]) if is_xl else None
    vae_encoder = DiffusionStage(model_paths[1 + is_xl])
    unet = DiffusionStage(model_paths[2 + is_xl])
    vae_decoder = DiffusionStage(model_paths[3 + is_xl])

    print('\n'.join([
        f"Creating {stage.folder_name_prefix}s..." for stage in
        [text_encoder, text_encoder_2, vae_encoder, unet, vae_decoder]
        if stage is not None
    ]))

    test_idx = 0
    scale = 7.0
    seed = 42
    # TODO: get it from latent size
    size = 128 if is_xl else 64
    sample_shape = (1, 4, size, size)
    # TODO from config?
    vae_scaling_factor = 0.18215
    noise = torch.randn(sample_shape, generator=torch.manual_seed(seed))
    time_ids = np.array([[size * 8, size * 8, 0, 0, size * 8, size * 8]] * 2,
                        dtype=np.float32)

    for idx, (image, prompt) in enumerate(zip(image_dataset, prompt_dataset)):
        text_encoder_input_data_map = prompt_dataset.transform(
            text_encoder.inputs, prompt, model.text_preprocess)
        text_encoder_output_data_map = text_encoder.inference(
            text_encoder_input_data_map, test_idx)

        if is_xl:
            text_encoder_2_input_data_map = prompt_dataset.transform(
                text_encoder_2.inputs, prompt, model.text_preprocess_2)
            text_encoder_2_output_data_map = text_encoder_2.inference(
                text_encoder_2_input_data_map, test_idx)
            text_encoder_output_data_map['last_hidden_state'] = np.concatenate(
                (text_encoder_output_data_map['last_hidden_state'],
                 text_encoder_2_output_data_map['last_hidden_state']),
                axis=2)

        vae_encoder_input_data_map = image_dataset.transform(
            vae_encoder.inputs, image, model.image_preprocess)
        vae_encoder_output_data_map = vae_encoder.inference(
            vae_encoder_input_data_map, test_idx)

        model.scheduler.set_timesteps(decode_limit)
        latents = torch.from_numpy(
            vae_encoder_output_data_map["latent_sample"]) * vae_scaling_factor
        latents = model.scheduler.add_noise(latents, noise,
                                            model.scheduler.timesteps[:1])

        for step, t in enumerate(model.scheduler.timesteps):
            latents_model_input = torch.concatenate([latents] * 2)
            latents_model_input = model.scheduler.scale_model_input(
                latents_model_input, t).numpy().astype(np.float32)
            timestep = np.atleast_1d(t.numpy().astype(np.int64))

            unet_input_data_map = {
                'sample':
                latents_model_input,
                'encoder_hidden_states':
                text_encoder_output_data_map['last_hidden_state'],
                'timestep':
                timestep
            }
            if is_xl:
                unet_input_data_map[
                    'text_embeds'] = text_encoder_2_output_data_map[
                        'text_embeds']
                unet_input_data_map['time_ids'] = time_ids

            unet_output_data_map = unet.inference(
                unet_input_data_map, test_idx * decode_limit + step)

            noise_pred_text, noise_pred_uncond = np.array_split(
                unet_output_data_map['out_sample'], 2)

            noise_pred = noise_pred_uncond + scale * (noise_pred_text -
                                                      noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = model.scheduler.step(torch.from_numpy(noise_pred), t,
                                           latents).prev_sample

        latents = 1 / vae_scaling_factor * latents
        vae_decoder_input_data_map = {
            "latent_sample": latents.numpy().astype(np.float32)
        }
        _ = vae_decoder.inference(vae_decoder_input_data_map, test_idx)

        test_idx += 1
        if sample_limit and sample_limit - 1 <= idx:
            break
