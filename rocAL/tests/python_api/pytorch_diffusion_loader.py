# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import random
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np
import sys
import os
import timeit


def draw_patches(img, idx, layout="nchw", dtype="fp32", device="cpu"):
    # image is expected as a tensor
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    if layout == "nchw":
        image = image.transpose([1, 2, 0])
    if dtype == "fp16":
        image = image.astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/" + str(idx) + ".png", image*255)


def main():
    if len(sys.argv) < 3:
        print("Please pass image_folder cpu/gpu batch_size")
        exit(0)
    try:
        path = "output_folder/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    rocal_cpu = True if sys.argv[2] == "cpu" else False
    batch_size = int(sys.argv[3])

    local_rank = 0
    world_size = 1

    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=local_rank, seed=local_rank+10, rocal_cpu=rocal_cpu, tensor_dtype=types.FLOAT,
                    tensor_layout=types.NCHW, prefetch_queue_depth=6, output_memory_type=types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    # Diffusion pytorch dataloader converted to rocAL
    # Reference: CelebA Dataset loader - https://github.com/tqch/ddpm-torch/blob/master/ddpm_torch/datasets.py#L73
    with pipe:
        jpegs, labels = fn.readers.file(file_root=data_path)
        decode = fn.decoders.image(jpegs, output_type=types.RGB,
                                   file_root=data_path, shard_id=local_rank, num_shards=world_size, max_decoded_width=178, max_decoded_height=218)
        roi_resize = fn.roi_resize(decode, resize_width=64, resize_height=64, output_layout=types.NHWC, output_dtype=types.UINT8,
                                           interpolation_type=types.TRIANGULAR_INTERPOLATION, roi_w=148, roi_h=148, roi_pos_x=15, roi_pos_y=40)

        flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(roi_resize,
                                        output_layout=types.NCHW,
                                        output_dtype=types.FLOAT,
                                        crop=(64, 64),
                                        mirror=flip_coin,
                                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255])
        pipe.set_outputs(cmnp)

    pipe.build()
    imageIteratorPipeline = ROCALClassificationIterator(
        pipe, device='cpu' if rocal_cpu else 'gpu', device_id=local_rank)
    cnt = 0
    start = timeit.default_timer()

    for epoch in range(3):
        print(
            "+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++", epoch)
        for i, it in enumerate(imageIteratorPipeline):
            print(
                "************************************** i *************************************", i)
            batch = it[0][0]
            for img in batch:
                draw_patches(img, cnt, layout="nchw",
                             dtype="fp32", device=rocal_cpu)
                cnt += 1
        imageIteratorPipeline.reset()
    print("*********************************************************************")
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)


if __name__ == "__main__":
    main()
