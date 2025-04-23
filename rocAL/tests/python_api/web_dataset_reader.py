# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types
import amd.rocal.fn as fn
import os
import cv2
from parse_config import parse_args


def draw_patches(img, idx, args=None):
    # image is expected as a tensor
    if args.rocal_gpu:
        image = img.cpu().numpy()
    else:
        image = img.detach().numpy()
    if not args.NHWC:
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/web_dataset_reader/" +
                str(idx)+"_"+"train"+".png", image)


def main():
    args = parse_args()
    # Args
    image_path = args.image_dataset_path
    index_file = args.index_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    num_threads = args.num_threads
    random_seed = args.seed
    device = "gpu" if args.rocal_gpu else "cpu"
    try:
        path = "output_folder/web_dataset_reader/"
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as error:
        print(error)
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,
                    device_id=args.local_rank, seed=random_seed, rocal_cpu=rocal_cpu)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        img_raw = fn.readers.webdataset(path=image_path, ext=[{'JPEG', 'cls'}], index_paths=index_file, missing_components_behavior=types.MISSING_COMPONENT_ERROR)
        img = fn.decoders.image(img_raw, file_root=image_path,
                                max_decoded_width=500, max_decoded_height=500, index_path=index_file)
        pipe.set_outputs(img)
    # Build the pipeline
    pipe.build()
    # Dataloader
    data_loader = ROCALClassificationIterator(pipe, display=0, device=device, device_id=args.local_rank)
    # Training loop
    cnt = 0
    # Enumerate over the Dataloader
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        print("epoch:: ", epoch)
        for i, ([image_batch], meta_data) in enumerate(data_loader, 0):
            if i == 0 and args.print_tensor:
                print("\r Mini-batch " + str(i))
                print("Images", image_batch)
                print("Meta Data - ASCII", meta_data)
            for element in list(range(batch_size)):
                cnt += 1
                draw_patches(image_batch[element],
                             cnt, args=args)
        data_loader.reset()
    print("##############################  WEBDATASET READER SUCCESS  ############################")


if __name__ == "__main__":
    main()
