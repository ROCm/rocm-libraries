# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
from tkinter import W
from amd.rocal.pipeline import pipeline_def
import numpy as np
import rocal_pybind as b
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


seed = 1549361629
image_dir = "../../../data/images/AMD-tinyDataSet/"
batch_size = 4
gpu_id = 0

def show_images(image_batch, device):
    columns = 4
    rows = (batch_size + 1) // (columns)
    #fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        #print('\n Display image: ', j)
        plt.subplot(gs[j])
        img = image_batch[j]
        plt.axis("off")
        if device == "cpu":
            plt.imshow(img)
        else:
            plt.imshow(img.cpu())
    plt.show()

def show_images(image_batch, image_batch1, device):
    columns = 4
    rows = (batch_size + 1) // (columns)
    #fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns*2)
    for j in range(rows*columns):
        #print('\n Display image: ', j)
        k = j*2
        plt.subplot(gs[k])
        img = image_batch[j]
        plt.axis("off")
        if device == "cpu":
            plt.imshow(img)
        else:
            plt.imshow(img.cpu())

        plt.subplot(gs[k+1])
        img = image_batch1[j]
        plt.axis("off")
        if device == "cpu":
            plt.imshow(img)
        else:
            plt.imshow(img.cpu())
    plt.show()



def show_pipeline_output(pipe, device):
    pipe.build()
    data_loader = ROCALClassificationIterator(pipe, device)
    images = next(iter(data_loader))
    show_images(images[0], device)

def show_pipeline_outputs(pipe0, pipe1, device):
    pipe0.build()
    pipe1.build()
    data_loader = ROCALClassificationIterator(pipe0, device)
    data_loader1 = ROCALClassificationIterator(pipe1, device)
    images = next(iter(data_loader))
    images1 = next(iter(data_loader1))
    show_images(images[0], images1[0], device)

@pipeline_def(seed=seed)
def inference_pipeline(device="cpu", path=image_dir):
    jpegs, labels = fn.readers.file(file_root=path, shard_id=0, num_shards=1, random_shuffle=False)
    images = fn.decoders.image(jpegs, file_root=path, max_decoded_width=1024, max_decoded_height=1024, device="cpu", output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    images_res = fn.resize(images, scaling_mode=types.SCALING_MODE_NOT_SMALLER, interpolation_type=types.TRIANGULAR_INTERPOLATION, resize_shorter=256)
    return fn.centre_crop(images_res, crop=(224, 224))

@pipeline_def(seed=seed)
def inference_pipeline_cmn(device="cpu", path=image_dir):
    jpegs, labels = fn.readers.file(file_root=path, shard_id=0, num_shards=1, random_shuffle=False)
    images = fn.decoders.image(jpegs, file_root=path, max_decoded_width=1024, max_decoded_height=1024, device="cpu", output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    images_res = fn.resize(images, scaling_mode=types.SCALING_MODE_NOT_SMALLER, interpolation_type=types.TRIANGULAR_INTERPOLATION, resize_shorter=256)
    return fn.crop_mirror_normalize(images_res , device="cpu",
                                        output_dtype=types.FLOAT,
                                        output_layout=types.NHWC,
                                        crop=(224, 224),
                                        mirror=0,
                                        image_type=types.RGB,
                                        mean=[0,0,0],
                                        std=[255.0,255.0,255.0])

def main():
    print ('Optional arguments: <cpu/gpu image_folder>')
    bs = batch_size
    rocal_device = "cpu"
    img_folder = image_dir
    if  len(sys.argv) > 1:
      if(sys.argv[1] == "gpu"):
          rocal_device = "gpu"
    if  len(sys.argv) > 2:
      img_folder = sys.argv[2]

    pipe = inference_pipeline(batch_size=bs, num_threads=1, device_id=gpu_id, rocal_cpu=True, tensor_layout=types.NHWC,
                                  reverse_channels=True, multiplier = [0.00392,0.00392,0.00392], device=rocal_device, path=img_folder)
    pipe1 = inference_pipeline_cmn(batch_size=bs, num_threads=1, device_id=gpu_id, rocal_cpu=True, tensor_layout=types.NHWC,
                                  reverse_channels=True, multiplier = [0.00392,0.00392,0.00392], device=rocal_device, path=img_folder)
    show_pipeline_outputs(pipe, pipe1, device=rocal_device)

if __name__ == '__main__':
    main()
