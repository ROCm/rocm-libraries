import toml
import torch
import argparse
from datasets import load_dataset
import numpy as np
from preprocessing import AudioPreprocessing


def audio_processing(waveform, config_toml='configs/rnnt.toml'):
    config = toml.load(config_toml)

    featurizer_config = config['input_eval']
    audio_preprocessor = AudioPreprocessing(**featurizer_config)
    audio_preprocessor.eval()

    assert waveform.ndim == 1
    waveform_length = np.array(waveform.shape[0], dtype=np.int64)
    waveform = np.expand_dims(waveform, 0)
    waveform_length = np.expand_dims(waveform_length, 0)
    with torch.no_grad():
        waveform = torch.from_numpy(waveform)
        waveform_length = torch.from_numpy(waveform_length)
        feature, feature_length = audio_preprocessor.forward(
            (waveform, waveform_length))
        assert feature.ndim == 3
        assert feature_length.ndim == 1
        feature = feature.permute(2, 0, 1)

    return feature, feature_length


def librespeech_huggingface(config_toml='configs/rnnt.toml'):
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                      "clean",
                      split="validation")

    waveform = ds[0]["audio"]["array"]
    transcript = ds[0]["text"]
    feature, feature_length = audio_processing(waveform, config_toml)
    return feature, feature_length, transcript.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_toml", default="configs/rnnt.toml")
    args = parser.parse_args()

    return librespeech_huggingface(args.config_toml)


if __name__ == "__main__":
    main()
