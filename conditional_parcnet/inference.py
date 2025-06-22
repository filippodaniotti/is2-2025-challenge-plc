import argparse
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import yaml
from parcnet import PARCnet
from torch import Tensor
from tqdm import tqdm


def _to_numpy(y: Union[Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy()
    return y.squeeze()


def mse(
    y_pred: Union[Tensor, np.ndarray], y_true: Union[Tensor, np.ndarray]
) -> np.ndarray:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    return 10 * np.log10(np.mean(np.square(y_true - y_pred)))


def sdr(y_pred: Union[Tensor, np.ndarray], y_true: Union[Tensor, np.ndarray]):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    num = np.linalg.norm(y_true) ** 2 + 1e-7
    den = np.linalg.norm(y_true - y_pred) ** 2 + 1e-7

    return 10 * np.log10(num / den)


def main():
    # Parse command-line options
    parser = argparse.ArgumentParser(description="Run PARCnet inference.")
    parser.add_argument(
        "-c",
        "--config",
        default="inference_config_tfilm.yaml",
        help="baseline config file",
    )
    parser.add_argument(
        "--simulate_packet_loss",
        action="store_true",
        help="whether to use the trace for simulating packet loss via zero-filling; useful when testing the baseline with clean audio files.",
    )
    args = parser.parse_args()

    # Open config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Read paths from config file
    meta = config["inference"]["meta"]
    lossy_audio_dir = Path(config["inference"]["lossy_audio_dir"])
    traces_dir = Path(config["inference"]["traces_dir"])
    model_checkpoint = Path(config["inference"]["model_checkpoint"])
    enhanced_audio_dir = Path(config["inference"]["enhanced_audio_dir"])

    # Read global params from config file
    sr = int(config["global"]["sr"])
    packet_dim = int(config["global"]["packet_dim"])
    extra_pred_dim = int(config["global"]["extra_pred_dim"])

    # Read AR params from config file
    ar_order = int(config["AR"]["ar_order"])
    diagonal_load = float(config["AR"]["diagonal_load"])
    ar_context_dim = int(config["AR"]["ar_context_dim"])

    # Read NN params from config file
    nn_context_len = int(config["neural_net"]["nn_context_dim"])
    nn_fade_dim = int(config["neural_net"]["fade_dim"])
    lite = bool(config["neural_net"]["lite"])

    # Create enhanced audio directory
    enhanced_audio_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate PARCnet
    parcnet = PARCnet(
        model_checkpoint=model_checkpoint,
        packet_dim=packet_dim,
        extra_pred_dim=extra_pred_dim,
        ar_order=ar_order,
        ar_diagonal_load=diagonal_load,
        ar_context_dim=ar_context_dim,
        nn_context_dim=nn_context_len,
        nn_fade_dim=nn_fade_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lite=lite,
    )

    # Iterate over test files
    with open(meta) as f:
        for i, file_id in enumerate(tqdm(f.readlines())):

            file_id = file_id.rstrip("\n")

            # Load the lossy audio clip
            lossy, __ = librosa.load(
                lossy_audio_dir.joinpath(f"{file_id}.wav"), sr=sr, mono=True
            )

            # Load packet trace
            trace = np.loadtxt(traces_dir.joinpath(f"{file_id}.txt"), dtype=int)

            # Simulate packet loss from trace
            if args.simulate_packet_loss:
                loss_trace = 1 - np.repeat(trace, packet_dim)
                loss_trace = np.pad(
                    loss_trace,
                    (0, lossy.shape[0] - loss_trace.shape[0]),
                    mode="constant",
                )
                lossy *= loss_trace

            # Predict missing packets using PARCnet
            enhanced = parcnet(lossy, trace)

            # Save enhanced wav file
            sf.write(enhanced_audio_dir.joinpath(f"{file_id}.wav"), enhanced.T, sr)


if __name__ == "__main__":
    main()
