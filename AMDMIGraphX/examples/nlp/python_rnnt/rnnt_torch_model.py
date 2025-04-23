import toml
import torch
import argparse
from model_separable_rnnt import RNNT


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


def pytorch_rnnt_model(config_toml='configs/rnnt.toml',
                       checkpoint_path='rnnt.pt'):
    config = toml.load(config_toml)

    def add_blank_label(labels):
        if not isinstance(labels, list):
            raise ValueError("labels must be a list of symbols")
        labels.append("<BLANK>")
        return labels

    dataset_vocab = config['labels']['labels']
    rnnt_vocab = add_blank_label(dataset_vocab)
    featurizer_config = config['input_eval']
    model = RNNT(feature_config=featurizer_config,
                 rnnt=config['rnnt'],
                 num_classes=len(rnnt_vocab))

    model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path))
    model.to('cuda')
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_toml", default="configs/rnnt.toml")
    parser.add_argument("--checkpoint_path", default="rnnt.pt")
    args = parser.parse_args()
    pytorch_rnnt_model(args.mlcommons_inference_path, args.checkpoint_path)


if __name__ == "__main__":
    main()
