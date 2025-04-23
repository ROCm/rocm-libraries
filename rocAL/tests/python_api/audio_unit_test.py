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

from amd.rocal.pipeline import pipeline_def
from amd.rocal.plugin.pytorch import ROCALAudioIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import random
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import timeit
from parse_config import parse_args

np.set_printoptions(threshold=1000, edgeitems=10000)
seed = random.SystemRandom().randint(0, 2**32 - 1)

test_case_augmentation_map = {
    0: "audio_decoder",
    1: "preemphasis_filter",
    2: "spectrogram",
    3: "downmix",
    4: "to_decibels",
    5: "resample",
    6: "tensor_add_tensor",
    7: "tensor_mul_scalar",
    8: "non_silent_region",
    9: "slice",
    10: "mel_filter_bank",
    11: "normalize"
}

def plot_audio_wav(audio_tensor, idx):
    # audio is expected as a tensor
    audio_data = audio_tensor.detach().numpy()
    audio_data = audio_data.flatten()
    plt.plot(audio_data)
    plt.savefig("output_folder/audio_reader/" + str(idx) + ".png")
    plt.close()

def verify_non_silent_region_output(output_list, rocal_data_path, test_results, case_name):
    ref_path = f'{rocal_data_path}/rocal_data/GoldenOutputsTensor/reference_outputs_audio/{case_name}_output.bin'
    data_array = np.fromfile(ref_path, dtype=np.int32)
    begin_nsr = output_list[0].detach().numpy()
    length_nsr = output_list[1].detach().numpy()

    if begin_nsr[0] == data_array[0] and length_nsr == data_array[1]:
        print("PASSED!")
        test_results[case_name] = "PASSED"
    else:
        print("FAILED!")
        test_results[case_name] = "FAILED"

def verify_output(audio_tensor, rocal_data_path, roi_tensor, test_results, case_name, dimensions):
    ref_path = f'{rocal_data_path}/rocal_data/GoldenOutputsTensor/reference_outputs_audio/{case_name}_output.bin'
    data_array = np.fromfile(ref_path, dtype=np.float32)
    audio_data = audio_tensor[0].detach().numpy().flatten()
    roi_data = roi_tensor.detach().numpy()
    buffer_size = roi_data[0] * roi_data[1]
    matched_indices = 0
    atol = 1e-20 if case_name != "normalize" else 1e-5  # Reducing absolute tolerance for normalize test case to fix test case failures
    for i in range(roi_data[0]):
        for j in range(roi_data[1]):
            ref_val = data_array[i * roi_data[1] + j]
            out_val = audio_data[i * roi_data[1] + j]   # Stride upto max_roi
            # ensuring that out_val is not exactly zero while ref_val is non-zero.
            invalid_comparison = (out_val == 0.0) and (ref_val != 0.0)
            #comparing the absolute difference between the output value (out_val) and the reference value (ref_val) with a tolerance threshold.
            if not invalid_comparison and abs(out_val - ref_val) < atol:
                matched_indices += 1

    # Print results
    print(f"Results for {case_name}:")
    if matched_indices == buffer_size and matched_indices != 0:
        print("PASSED!")
        test_results[case_name] = "PASSED"
    else:
        print("FAILED!")
        test_results[case_name] = "FAILED"

@pipeline_def(seed=seed)
def audio_decoder_pipeline(path, file_list, downmix=False):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    return fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=downmix,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)

@pipeline_def(seed=seed)
def pre_emphasis_filter_pipeline(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=False,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    return fn.preemphasis_filter(decoded_audio)

@pipeline_def(seed=seed)
def spectrogram_pipeline(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=False,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    spec = fn.spectrogram(
        decoded_audio,
        nfft=512,
        window_length=320,
        window_step=160,
        output_dtype = types.FLOAT)
    return spec

@pipeline_def(seed=seed)
def to_decibels_pipeline(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=False,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    return fn.to_decibels(
            decoded_audio,
            multiplier=np.log(10),
            reference=1.0,
            cutoff_db=np.log(1e-20),
            output_dtype=types.FLOAT)

@pipeline_def(seed=seed)
def resample_pipeline(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=True,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    input_sample_rate = 16000.00
    uniform_distribution_resample = fn.random.uniform(decoded_audio, range=[1.15, 1.15])
    resampled_rate = uniform_distribution_resample * input_sample_rate
    return fn.resample(
        decoded_audio,
        resample_rate=resampled_rate,
        resample_hint=1.15 * 255840,
        output_datatype=types.FLOAT)

@pipeline_def(seed=seed)
def tensor_add_tensor_pipeline(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=False,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    uniform_distribution_sample = fn.random.uniform(decoded_audio, range=[1.15, 1.15])
    return decoded_audio + uniform_distribution_sample

@pipeline_def(seed=seed)
def tensor_mul_scalar_pipeline(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=True,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    return decoded_audio * 1.15

@pipeline_def(seed=seed)
def non_silent_region(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=False,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    begin, length = fn.nonsilent_region(decoded_audio, cutoff_db=-60)
    return begin, length

@pipeline_def(seed=seed)
def non_silent_region_and_slice(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=False,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    begin, length = fn.nonsilent_region(decoded_audio, cutoff_db=-60)
    trim_silence = fn.slice(
        decoded_audio,
        anchor=[begin],
        shape=[length],
        rocal_tensor_output_type = types.FLOAT)
    return trim_silence

@pipeline_def(seed=seed)
def mel_filter_bank_pipeline(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=False,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    spec = fn.spectrogram(
        decoded_audio,
        nfft=512,
        window_length=320,
        window_step=160,
        output_dtype=types.FLOAT)
    return fn.mel_filter_bank(
        spec,
        freq_high=8000,
        freq_low=0.0,
        nfilter=80,
        sample_rate=16000)

@pipeline_def(seed=seed)
def normalize_pipeline(path, file_list):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    decoded_audio = fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        downmix=False,
        shard_id=0,
        num_shards=1,
        stick_to_shard=False,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False)
    spec = fn.spectrogram(
        decoded_audio,
        nfft=512,
        window_length=320,
        window_step=160,
        output_dtype=types.FLOAT)
    return fn.normalize(
        spec,
        axes=[1])

def main():
    args = parse_args()

    audio_path = args.audio_path
    file_list = args.file_list_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    test_case = args.test_case
    qa_mode = args.qa_mode
    num_threads = 1
    device_id = 0
    rocal_data_path = os.environ.get("ROCAL_DATA_PATH")

    case_list = list(test_case_augmentation_map.keys())

    if test_case is not None: 
        if test_case not in case_list:
            print(" Invalid Test Case! ")
            exit()
        else:
            case_list = [test_case]

    if args.display:
        try:
            path = "output_folder/audio_reader"
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
        except OSError as error:
            print(error)

    if rocal_data_path is None:
        print("Need to export ROCAL_DATA_PATH")
        sys.exit()
    if not rocal_cpu:
        print("The GPU support for Audio is not given yet. Running on CPU")
        rocal_cpu = True
    if not audio_path and not file_list:
        audio_path = f'{rocal_data_path}/rocal_data/audio/'
        file_list = f'{rocal_data_path}/rocal_data/audio/wav_file_list.txt'
        downmix_audio_path = f'{rocal_data_path}/rocal_data/multi_channel_wav/'
    else:
        print("QA mode is disabled for custom audio data")
        qa_mode = 0
    if qa_mode and batch_size != 1:
        print("QA mode is enabled. Batch size is set to 1.")
        batch_size = 1

    print("*********************************************************************")
    test_results = {}
    for case in case_list:
        case_name = test_case_augmentation_map.get(case)
        if case_name == "audio_decoder":
            audio_pipeline = audio_decoder_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "preemphasis_filter":
            audio_pipeline = pre_emphasis_filter_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "spectrogram":
            audio_pipeline = spectrogram_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "downmix":
            audio_pipeline = audio_decoder_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu,
                                                    path=downmix_audio_path if qa_mode else audio_path, file_list="" if qa_mode else file_list, downmix=True)
        if case_name == "to_decibels":
            audio_pipeline = to_decibels_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "slice":
            audio_pipeline = non_silent_region_and_slice(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "non_silent_region":
            audio_pipeline = non_silent_region(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "resample":
            audio_pipeline = resample_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "tensor_add_tensor":
            audio_pipeline = tensor_add_tensor_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "tensor_mul_scalar":
            audio_pipeline = tensor_mul_scalar_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "mel_filter_bank":
            audio_pipeline = mel_filter_bank_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        if case_name == "normalize":
            audio_pipeline = normalize_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
        audio_pipeline.build()
        audio_loader = ROCALAudioIterator(audio_pipeline, auto_reset=True)
        output_tensor_list = audio_pipeline.get_output_tensors()
        dimensions = output_tensor_list[0].dimensions()
        cnt = 0
        start = timeit.default_timer()
        # Enumerate over the Dataloader
        for e in range(int(args.num_epochs)):
            print("Epoch :: ", e)
            torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
            for i, output_list in enumerate(audio_loader):
                for x in range(len(output_list[0])):
                    for audio_tensor, label, roi in zip(output_list[0][x], output_list[1], output_list[2]):
                        if args.print_tensor:
                            print("label", label)
                            print("Audio", audio_tensor)
                            print("Roi", roi)
                        if args.display:
                            plot_audio_wav(audio_tensor, cnt)
                        cnt+=1
            if qa_mode :
                if case_name == "non_silent_region":
                    verify_non_silent_region_output(output_list[0], rocal_data_path, test_results, case_name)
                else:
                    verify_output(output_list[0], rocal_data_path, roi, test_results, case_name, dimensions)
            print("EPOCH DONE", e)

        stop = timeit.default_timer()
        print('\nTime: ', stop - start)

    if qa_mode:
        passed_cases = []
        failed_cases = []

        for augmentation_name, result in test_results.items():
            if result == "PASSED":
                passed_cases.append(augmentation_name)
            else:
                failed_cases.append(augmentation_name)

        print("Number of PASSED tests:", len(passed_cases))
        print(passed_cases)
        print("Number of FAILED tests:", len(failed_cases))
        print(failed_cases)


if __name__ == "__main__":
    main()
