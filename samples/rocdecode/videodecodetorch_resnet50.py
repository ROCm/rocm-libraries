# Copyright © Advanced Micro Devices, Inc., or its affiliates.
 
# SPDX-License-Identifier:  MIT License

import datetime
import sys
import argparse
import os.path
import torch
import torchvision
import pyRocVideoDecode.decoder as dec
import pyRocVideoDecode.demuxer as dmx


def Decoder(
        input_file_path,
        output_file_path,
        device_id,
        rgb_format):

    # Init resnet
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    model.to("cuda")

    # get labels as list
    labels_file = open("data/labels.txt", "r")
    data = labels_file.read()
    categories = data.split("\n")
    labels_file.close()

    # resnet expects images to be 3 channel planar RGB of 224x244 size at
    # least.
    target_w, target_h = 224, 224

    # demuxer instance
    demuxer = dmx.demuxer(input_file_path)

    # get the used coded id
    codec_id = dec.GetRocDecCodecID(demuxer.GetCodecId())

    # decoder instance
    viddec = dec.decoder(codec_id)

    # Get GPU device information
    cfg = viddec.GetGpuInfo()

    # check if codec is supported
    if (viddec.IsCodecSupported(device_id, codec_id, demuxer.GetBitDepth()) == False):
        print("ERROR: Codec is not supported on this GPU " + cfg.device_name)
        exit()

    #  print some GPU info out
    print("\ninfo: Input file: " +
          input_file_path +
          '\n' +
          "info: Using GPU device " +
          str(device_id) +
          " - " +
          cfg.device_name +
          "[" +
          cfg.gcn_arch_name +
          "] on PCI bus " +
          str(cfg.pci_bus_id) +
          ":" +
          str(cfg.pci_domain_id) +
          "." +
          str(cfg.pci_device_id))
    print("info: decoding started, please wait! \n")

    # -----------------
    # The decoding loop
    # -----------------
    n_frame = 0
    total_dec_time = 0.0
    output_format = dec.GetOutputFormat(rgb_format)

    while True:
        start_time = datetime.datetime.now()
        packet = demuxer.DemuxFrame()
        n_frame_returned = viddec.DecodeFrame(packet)

        for i in range(n_frame_returned):
            pts = viddec.GetFrameRgb(packet, rgb_format)

            if(pts == -1):
                print("Error: GetFrameRgb returned failure.\n")
                continue

            # using torch tensor
            rgb_tensor = torch.from_dlpack(packet.ext_buf[0].__dlpack__(packet))

            # save tensors to file, with original decoded Size
            if (output_file_path is not None):
                surface_info = viddec.GetOutputSurfaceInfo()
                viddec.SaveFrameToFile(
                    output_file_path,
                    rgb_tensor.data_ptr(),
                    surface_info,
                    output_format)

            # for inference
            rgb_tensor.resize_(3, target_h, target_w)
            rgb_tensor = rgb_tensor.type(dtype=torch.cuda.FloatTensor)
            rgb_tensor = torch.divide(rgb_tensor, 255.0)
            data_transforms = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            surface_tensor = data_transforms(rgb_tensor)
            input_batch = surface_tensor.unsqueeze(0).to("cuda")

            # Run inference.
            with torch.no_grad():
                output = model(input_batch)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for i in range(top5_prob.size(0)):
                print(categories[top5_catid[i]], top5_prob[i].item())
            print()

            # release frame
            viddec.ReleaseFrame(packet)

        # measure after completing a whole frame
        end_time = datetime.datetime.now()
        time_per_frame = end_time - start_time
        total_dec_time = total_dec_time + time_per_frame.total_seconds()

        # increament frames counter
        n_frame += n_frame_returned

        if (packet.bitstream_size <= 0):  # no more to decode?
            break

    # beyond the decoding loop
    n_frame += viddec.GetNumOfFlushedFrames()

    print("info: Total frame decoded: " + str(n_frame))
    print("info: frame count= ", n_frame)
    print()


if __name__ == "__main__":

    # get passed arguments
    parser = argparse.ArgumentParser(
        description='PyRocDecode Video Decode Arguments')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Input File Path - required',
        required=True)
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output File Path - optional',
        required=False)
    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
        help='GPU device ID - optional, default 0',
        required=False)
    parser.add_argument(
        '-of',
        '--rgb_format',
        type=int,
        default=3,
        help="Rgb Format to use as tensor - 1:bgr, 2:bgr48, 3:rgb, 4:rgb48, 5:bgra, 6:bgra64, 7:rgba, 8:rgba64, converts decoded YUV frame to Tensor in RGB format, optional, default: 3",
        required=False)

    try:
        args = parser.parse_args()
    except BaseException:
        sys.exit()

    # get params
    input_file_path = args.input
    output_file_path = args.output
    device_id = args.device
    rgb_format = args.rgb_format

    # handel params
    rgb_format = 3 if (rgb_format < 1 or rgb_format > 8) else rgb_format
    if not os.path.exists(input_file_path):  # Input file (must exist)
        print("ERROR: input file doesn't exist.")
        exit()

    # torch GPU
    print("\nPyTorch Using: ", torch.cuda.get_device_name(0))

    Decoder(
        input_file_path,
        output_file_path,
        device_id,
        rgb_format)
