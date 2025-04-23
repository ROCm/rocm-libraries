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

from amd.rocal.plugin.generic import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
from parse_config import parse_args
import os
import sys
import cv2

INTERPOLATION_TYPES = {
    0: types.NEAREST_NEIGHBOR_INTERPOLATION,
    1: types.LINEAR_INTERPOLATION,
    2: types.CUBIC_INTERPOLATION,
    3: types.LANCZOS_INTERPOLATION,
    4: types.GAUSSIAN_INTERPOLATION,
    5: types.TRIANGULAR_INTERPOLATION
}

SCALING_MODES = {
    0: types.SCALING_MODE_DEFAULT,
    1: types.SCALING_MODE_STRETCH,
    2: types.SCALING_MODE_NOT_SMALLER,
    3: types.SCALING_MODE_NOT_LARGER
}


def draw_patches(img, idx, args=None):
    # image is expected as a tensor, bboxes as numpy
    if args.fp16:
        img = (img).astype('uint8')
    if not args.color_format:
        img = img.transpose([0, 2, 3, 1])
    images_list = []
    for im in img:
        images_list.append(im)
    img = cv2.vconcat(images_list)
    if args.color_format:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(args.output_file_name + ".png", img,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

def dump_meta_data(labels, args=None):
    labels_list = labels.tolist()
    with open(args.output_file_name, 'w') as file:
        for label in labels_list:
            file.write(str(label) + '\n')

def main():
    args = parse_args()
    # Args
    data_path = args.image_dataset_path
    reader_type = args.reader_type
    augmentation_name = args.augmentation_name
    print("\n AUGMENTATION NAME: ", augmentation_name)
    rocal_cpu = False if args.rocal_gpu else True
    device = "cpu" if rocal_cpu else "cuda"
    batch_size = args.batch_size
    max_height = args.max_height
    max_width = args.max_width
    color_format = types.RGB if args.color_format else types.GRAY
    tensor_layout = types.NHWC if args.color_format else types.NCHW
    tensor_dtype = types.UINT8
    num_threads = args.num_threads
    random_seed = args.seed
    local_rank = args.local_rank
    world_size = args.world_size
    interpolation_type = INTERPOLATION_TYPES[args.interpolation_type]
    scaling_mode = SCALING_MODES[args.scaling_mode]
    if (scaling_mode != types.SCALING_MODE_DEFAULT and interpolation_type !=
            types.LINEAR_INTERPOLATION):
        interpolation_type = types.LINEAR_INTERPOLATION
    if augmentation_name in ["hue", "saturation", "color_twist"] and color_format == types.GRAY:
        print("Not a valid option! Exiting!")
        sys.exit(0)

    try:
        path = "output_folder/file_reader/" + args.augmentation_name
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=local_rank,
                    seed=random_seed,
                    rocal_cpu=rocal_cpu,
                    tensor_layout=tensor_layout,
                    tensor_dtype=tensor_dtype,
                    output_memory_type=types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    # Set Params
    output_set = 0
    rocal_device = 'cpu' if rocal_cpu else 'gpu'
    # hardcoding decoder_device to cpu to compare against golden outputs taken with turbojpeg decoder
    decoder_device = 'cpu'
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        if reader_type == "file":
            jpegs, labels = fn.readers.file(file_root=data_path)
            images = fn.decoders.image(jpegs,
                                       file_root=data_path,
                                       device=decoder_device,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       output_type=color_format,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)

        elif reader_type == "coco":
            annotation_path = args.json_path
            jpegs, _, _ = fn.readers.coco(annotations_file=annotation_path)
            images = fn.decoders.image(jpegs,
                                       file_root=data_path,
                                       annotations_file=annotation_path,
                                       device=decoder_device,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       output_type=color_format,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)

        elif reader_type == "tf_classification":
            try:
                import tensorflow as tf
            except ImportError:
                print('Install tensorflow to run tf_classification tests')
                exit()
            featureKeyMap = {
                'image/encoded': 'image/encoded',
                'image/class/label': 'image/class/label',
                'image/filename': 'image/filename'
            }
            features = {
                'image/encoded': tf.io.FixedLenFeature((), tf.string, ""),
                'image/class/label': tf.io.FixedLenFeature([1], tf.int64,  -1),
                'image/filename': tf.io.FixedLenFeature((), tf.string, "")
            }
            inputs = fn.readers.tfrecord(
                data_path, featureKeyMap, features, reader_type=0)
            jpegs = inputs["image/encoded"]
            images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap,
                                       output_type=color_format, path=data_path,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)

        elif reader_type == "tf_detection":
            try:
                import tensorflow as tf
            except ImportError:
                print('Install tensorflow to run tf_detection tests')
                exit()
            featureKeyMap = {
                'image/encoded': 'image/encoded',
                'image/class/label': 'image/object/class/label',
                'image/class/text': 'image/object/class/text',
                'image/object/bbox/xmin': 'image/object/bbox/xmin',
                'image/object/bbox/ymin': 'image/object/bbox/ymin',
                'image/object/bbox/xmax': 'image/object/bbox/xmax',
                'image/object/bbox/ymax': 'image/object/bbox/ymax',
                'image/filename': 'image/filename'
            }
            features = {
                'image/encoded': tf.io.FixedLenFeature((), tf.string, ""),
                'image/class/label': tf.io.FixedLenFeature([1], tf.int64,  -1),
                'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/filename': tf.io.FixedLenFeature((), tf.string, "")
            }
            inputs = fn.readers.tfrecord(
                path=data_path, reader_type=1, features=features, user_feature_key_map=featureKeyMap)
            jpegs = inputs["image/encoded"]
            _ = inputs["image/class/label"]
            images = fn.decoders.image_random_crop(jpegs, user_feature_key_map=featureKeyMap,
                                                   max_decoded_width=max_width,
                                                   max_decoded_height=max_height,
                                                   output_type=color_format,
                                                   shard_id=local_rank,
                                                   num_shards=world_size,
                                                   random_shuffle=False, path=data_path)

        elif reader_type == "caffe_classification":
            jpegs, _ = fn.readers.caffe(path=data_path, bbox=False)
            images = fn.decoders.image(jpegs,
                                       path=data_path,
                                       device=decoder_device,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       output_type=color_format,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)

        elif reader_type == "caffe_detection":
            jpegs, _, _ = fn.readers.caffe(path=data_path, bbox=True)
            images = fn.decoders.image(jpegs,
                                       path=data_path,
                                       device=decoder_device,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       output_type=color_format,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)

        elif reader_type == "caffe2_classification":
            jpegs, _ = fn.readers.caffe2(path=data_path)
            images = fn.decoders.image(jpegs,
                                       path=data_path,
                                       device=decoder_device,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       output_type=color_format,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)

        elif reader_type == "caffe2_detection":
            jpegs, _, _ = fn.readers.caffe2(path=data_path, bbox=True)
            images = fn.decoders.image(jpegs,
                                       path=data_path,
                                       device=decoder_device,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       output_type=color_format,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)

        elif reader_type == "mxnet":
            jpegs = fn.readers.mxnet(path=data_path)
            images = fn.decoders.image(jpegs,
                                       path=data_path,
                                       device=decoder_device,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       output_type=color_format,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)
        elif reader_type == "web_dataset":
            jpegs = fn.readers.webdataset(path=data_path, ext=[{'jpg', 'cls'}])
            images = fn.decoders.image(jpegs,
                                       path=data_path,
                                       device=decoder_device,
                                       max_decoded_width=max_width,
                                       max_decoded_height=max_height,
                                       output_type=color_format,
                                       shard_id=local_rank,
                                       num_shards=world_size,
                                       random_shuffle=False)
        if augmentation_name == "resize":
            resize_w = 400
            resize_h = 400
            if (scaling_mode == types.SCALING_MODE_STRETCH):
                resize_h = 416
            output = fn.resize(images,
                               resize_width=resize_w,
                               resize_height=resize_h,
                               output_layout=tensor_layout,
                               output_dtype=tensor_dtype,
                               scaling_mode=scaling_mode,
                               interpolation_type=interpolation_type)
        elif augmentation_name == "rotate":
            output = fn.rotate(images,
                               angle=45.0,
                               dest_width=416,
                               dest_height=416,
                               output_layout=tensor_layout,
                               output_dtype=tensor_dtype,
                               interpolation_type=interpolation_type)
        elif augmentation_name == "brightness":
            output = fn.brightness(images,
                                   brightness=1.9,
                                   brightness_shift=20.0,
                                   output_layout=tensor_layout,
                                   output_dtype=tensor_dtype)
        elif augmentation_name == "gamma_correction":
            output = fn.gamma_correction(images,
                                         output_layout=tensor_layout,
                                         output_dtype=tensor_dtype)
        elif augmentation_name == "contrast":
            output = fn.contrast(images,
                                 contrast=30.0,
                                 contrast_center=80.0,
                                 output_layout=tensor_layout,
                                 output_dtype=tensor_dtype)
        elif augmentation_name == "flip":
            output = fn.flip(images,
                             horizontal=1,
                             output_layout=tensor_layout,
                             output_dtype=tensor_dtype)
        elif augmentation_name == "blur":
            output = fn.blur(images,
                             window_size=5,
                             output_layout=tensor_layout,
                             output_dtype=tensor_dtype)
        elif augmentation_name == "warp_affine":
            output = fn.warp_affine(images, dest_height=416, dest_width=416, matrix=[1.0, 1.0, 0.5, 0.5, 7.0, 7.0],
                                    output_layout=tensor_layout, output_dtype=tensor_dtype, interpolation_type=types.LINEAR_INTERPOLATION)
        elif augmentation_name == "fish_eye":
            output = fn.fish_eye(images,
                                 output_layout=tensor_layout,
                                 output_dtype=tensor_dtype)
        elif augmentation_name == "vignette":
            output = fn.vignette(images,
                                 output_layout=tensor_layout,
                                 output_dtype=tensor_dtype)
        elif augmentation_name == "jitter":
            output = fn.jitter(images,
                               kernel_size=3,
                               output_layout=tensor_layout,
                               output_dtype=tensor_dtype)
        elif augmentation_name == "snp_noise":
            output = fn.snp_noise(images,
                                  p_noise=0.2,
                                  p_salt=0.2,
                                  noise_val=0.2,
                                  salt_val=0.5,
                                  seed=0,
                                  output_layout=tensor_layout,
                                  output_dtype=tensor_dtype)
        elif augmentation_name == "snow":
            output = fn.snow(images,
                             snow=0.2,
                             output_layout=tensor_layout,
                             output_dtype=tensor_dtype)
        elif augmentation_name == "rain":
            output = fn.rain(images,
                             rain=0.5,
                             rain_width=2,
                             rain_height=16,
                             rain_transparency=0.25,
                             output_layout=tensor_layout,
                             output_dtype=tensor_dtype)
        elif augmentation_name == "fog":
            output = fn.fog(images,
                            output_layout=tensor_layout,
                            output_dtype=tensor_dtype)
        elif augmentation_name == "pixelate":
            output = fn.pixelate(images,
                                 output_layout=tensor_layout,
                                 output_dtype=tensor_dtype)
        elif augmentation_name == "exposure":
            output = fn.exposure(images,
                                 exposure=1.0,
                                 output_layout=tensor_layout,
                                 output_dtype=tensor_dtype)
        elif augmentation_name == "hue":
            output = fn.hue(images,
                            hue=150.0,
                            output_layout=tensor_layout,
                            output_dtype=tensor_dtype)
        elif augmentation_name == "saturation":
            output = fn.saturation(images,
                                   saturation=0.3,
                                   output_layout=tensor_layout,
                                   output_dtype=tensor_dtype)
        elif augmentation_name == "color_twist":
            output = fn.color_twist(images,
                                    brightness=0.2,
                                    contrast=10.0,
                                    hue=100.0,
                                    saturation=0.25,
                                    output_layout=tensor_layout,
                                    output_dtype=tensor_dtype)
        elif augmentation_name == "crop":
            output = fn.crop(images,
                             crop=(3, 224, 224),
                             crop_pos_x=0.0,
                             crop_pos_y=0.0,
                             crop_pos_z=0.0,
                             output_layout=tensor_layout,
                             output_dtype=tensor_dtype)
        elif augmentation_name == "crop_mirror_normalize":
            output = fn.crop_mirror_normalize(images,
                                              output_layout=tensor_layout,
                                              output_dtype=tensor_dtype,
                                              crop=(224, 224),
                                              crop_pos_x=0.0,
                                              crop_pos_y=0.0,
                                              mean=[128, 128, 128],
                                              std=[1.2, 1.2, 1.2])
        elif augmentation_name == "resize_mirror_normalize":
            resize_w = 400
            resize_h = 400
            if (scaling_mode == types.SCALING_MODE_STRETCH):
                resize_h = 416
            output = fn.resize_mirror_normalize(images,
                                                resize_width=resize_w,
                                                resize_height=resize_h,
                                                output_layout=tensor_layout,
                                                output_dtype=tensor_dtype,
                                                scaling_mode=scaling_mode,
                                                interpolation_type=interpolation_type,
                                                mean=[128, 128, 128],
                                                std=[1.2, 1.2, 1.2])
        elif augmentation_name == "nop":
            output = fn.nop(images,
                            output_layout=tensor_layout,
                            output_dtype=tensor_dtype)
        elif augmentation_name == "centre_crop":
            output = fn.centre_crop(images,
                                    output_layout=tensor_layout,
                                    output_dtype=tensor_dtype)
        elif augmentation_name == "color_temp":
            output = fn.color_temp(images,
                                   adjustment_value=70,
                                   output_layout=tensor_layout,
                                   output_dtype=tensor_dtype)
        elif augmentation_name == "copy":
            output = fn.copy(images,
                             output_layout=tensor_layout,
                             output_dtype=tensor_dtype)
        elif augmentation_name == "resize_crop_mirror":
            output = fn.resize_crop_mirror(images,
                                           resize_height=400,
                                           resize_width=400,
                                           crop_h=200,
                                           crop_w=200,
                                           output_layout=tensor_layout,
                                           output_dtype=tensor_dtype)
        elif augmentation_name == "lens_correction":
            output = fn.lens_correction(images,
                                        strength=2.9,
                                        zoom=1.2,
                                        output_layout=tensor_layout,
                                        output_dtype=tensor_dtype)
        elif augmentation_name == "blend":
            output1 = fn.rotate(images,
                                angle=45.0,
                                dest_width=416,
                                dest_height=416,
                                output_layout=tensor_layout,
                                output_dtype=tensor_dtype)
            output = fn.blend(images,
                              output1,
                              ratio=0.5,
                              output_layout=tensor_layout,
                              output_dtype=tensor_dtype)
        elif augmentation_name == "resize_crop":
            output = fn.resize_crop(images,
                                    resize_width=416,
                                    resize_height=416,
                                    crop_area_factor=0.25,
                                    crop_aspect_ratio=1.2,
                                    x_drift=0.6,
                                    y_drift=0.4,
                                    output_layout=tensor_layout,
                                    output_dtype=tensor_dtype)
        elif augmentation_name == "center_crop":
            output = fn.center_crop(images,
                                    crop=[2, 224, 224],
                                    output_layout=tensor_layout,
                                    output_dtype=tensor_dtype)
        elif augmentation_name == "one_hot":
            output = fn.crop(images,
                             crop=(3, 224, 224),
                             crop_pos_x=0.0,
                             crop_pos_y=0.0,
                             crop_pos_z=0.0,
                             output_layout=tensor_layout,
                             output_dtype=tensor_dtype)
            num_classes = len(next(os.walk(data_path))[1])
            labels_onehot = fn.one_hot(labels, num_classes=num_classes)

        if output_set == 0:
            pipe.set_outputs(output)
    # build the pipeline
    pipe.build()
    # Dataloader
    data_loader = ROCALClassificationIterator(
        pipe, device=device, device_id=local_rank)
    cnt = 0
    import timeit
    start = timeit.default_timer()

    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::", epoch)
        for i, (output_list, labels) in enumerate(data_loader, 0):
            for j in range(len(output_list)):
                if args.print_tensor:
                    print("**************", i, "*******************")
                    print("**************starts*******************")
                    print("\nImages:\n", output_list[j])
                    print("\nLABELS:\n", labels)
                    print("**************ends*******************")
                    print("**************", i, "*******************")
                if args.augmentation_name == "one_hot":
                    dump_meta_data(labels, args=args)
                else:
                    draw_patches(output_list[j], cnt, args=args)
                    cnt += len(output_list[j])

        data_loader.reset()

    stop = timeit.default_timer()

    print('\n Time: ', stop - start)
    print('Number of times loop iterates is:', cnt)

    print(
        f"##############################  {augmentation_name.upper()}  SUCCESS  ############################")


if __name__ == '__main__':
    main()
