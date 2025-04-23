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

from math import sqrt
import torch
import itertools
import os
import ctypes
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np
from parse_config import parse_args


class ROCALCOCOIterator(object):
    """
    COCO ROCAL iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.output_list = self.dimensions = self.torch_dtype = None
        self.display = display
        # Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        # Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.output_memory_type = self.loader._output_memory_type

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocal_run() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.get_output_tensors()

        if self.output_list is None:
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                self.torch_dtype = self.output_tensor_list[i].dtype()
                if self.device == "cpu":
                    self.output = torch.empty(
                        self.dimensions, dtype=getattr(torch, self.torch_dtype))
                else:
                    torch_gpu_device = torch.device('cuda', self.device_id)
                    self.output = torch.empty(self.dimensions, dtype=getattr(
                        torch, self.torch_dtype), device=torch_gpu_device)
                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    self.output.data_ptr()), self.output_memory_type)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    self.output_list[i].data_ptr()), self.output_memory_type)

        self.labels = self.loader.get_bounding_box_labels()
        # 1D bboxes array in a batch
        self.bboxes = self.loader.get_bounding_box_cords()
        self.loader.get_image_id(self.image_id)
        image_id_tensor = torch.tensor(self.image_id)
        image_size_tensor = torch.tensor(self.img_size).view(-1, self.bs, 2)

        for i in range(self.bs):
            if self.display:
                img = self.output
                draw_patches(img[i], self.image_id[i],
                             self.bboxes[i], self.device, self.tensor_dtype, self.tensor_format)
        return (self.output), self.bboxes, self.labels, image_id_tensor, image_size_tensor

    def reset(self):
        self.loader.rocal_reset_loaders()

    def __iter__(self):
        return self


def draw_patches(img, idx, bboxes, device, dtype, layout):
    # image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    if dtype == types.FLOAT16:
        image = (image).astype('uint8')
    if layout == types.NCHW:
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bboxes = np.reshape(bboxes, (-1, 4))

    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
            (loc_[2])), int((loc_[3]))), color, thickness)
        cv2.imwrite("output_folder/coco_reader/" +
                    str(idx)+"_"+"train"+".png", image)


def main():
    args = parse_args()
    # Args
    image_path = args.image_dataset_path
    annotation_path = args.json_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    display = args.display
    num_threads = args.num_threads
    local_rank = args.local_rank
    world_size = args.world_size
    random_seed = args.seed
    tensor_format = types.NHWC if args.NHWC else types.NCHW
    tensor_dtype = types.FLOAT16 if args.fp16 else types.FLOAT
    try:
        path = "output_folder/coco_reader/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)

    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=args.local_rank,
                    seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=tensor_format, tensor_dtype=tensor_dtype)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        jpegs, bboxes, labels = fn.readers.coco(
            annotations_file=annotation_path)
        images_decoded = fn.decoders.image(jpegs, output_type=types.RGB, file_root=image_path,
                                           annotations_file=annotation_path, random_shuffle=False, shard_id=local_rank, num_shards=world_size)
        res_images = fn.resize(
            images_decoded, resize_width=300, resize_height=300)
        saturation = fn.uniform(range=[0.1, 0.4])
        contrast = fn.uniform(range=[0.1, 25.0])
        brightness = fn.uniform(range=[0.875, 1.125])
        hue = fn.uniform(range=[5.0, 170.0])
        ct_images = fn.color_twist(
            res_images, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmn_images = fn.crop_mirror_normalize(ct_images,
                                              crop=(224, 224),
                                              crop_pos_x=0.0,
                                              crop_pos_y=0.0,
                                              mean=[0, 0, 0],
                                              std=[1, 1, 1],
                                              mirror=flip_coin,
                                              output_layout=tensor_format,
                                              output_dtype=tensor_dtype)
        pipe.set_outputs(cmn_images)
    # Build the pipeline
    pipe.build()
    # Dataloader
    if (args.rocal_gpu):
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, display=display, tensor_layout=tensor_format, tensor_dtype=tensor_dtype, device="cuda")
    else:
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, display=display, tensor_layout=tensor_format, tensor_dtype=tensor_dtype, device="cpu")

    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(data_loader, 0):
            if args.print_tensor:
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\nIMAGES : \n", it[0])
                print("\nBBOXES:\n", it[1])
                print("\nLABELS:\n", it[2])
                print("\nIMAGE ID:\n", it[3])
                print("\nIMAGE SIZE:\n", it[4])
                print("**************ends*******************")
                print("**************", i, "*******************")
        data_loader.reset()
    # Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)

    print("##############################  COCO READER  SUCCESS  ############################")


if __name__ == '__main__':
    main()
