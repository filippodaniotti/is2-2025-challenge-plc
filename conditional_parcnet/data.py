from __future__ import annotations

import random
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from ar_model import ARModel
from nn_model import HybridModel
from torch import Tensor, no_grad
from torch.utils.data import DataLoader, Dataset


class NMPDataset(Dataset):

    def __init__(
        self,
        fold: str,
        audio_dir: Union[str, Path],
        meta_path: Union[str, Path],
        sample_rate: int,
        packet_dim: int,
        extra_pred_dim: int,
        nn_context_dim: int,
        ar_context_dim: int,
        ar_order: int,
        ar_fade_dim: int,
        diagonal_load: float,
        device: str,
        surrogate_model_checkpoint: Union[str, Path],
        nn_fade_dim: int,
        n_corrupted_packets: int,
        steps_per_epoch: Union[int, None] = None,
        batch_size: Union[int, None] = None,
    ):

        self.fold = fold
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.ar_order = ar_order
        self.ar_fade_dim = ar_fade_dim
        self.diagonal_load = diagonal_load
        self.packet_dim = packet_dim
        self.extra_pred_dim = extra_pred_dim
        self.pred_dim = packet_dim + extra_pred_dim
        self.nn_context_dim = nn_context_dim
        self.ar_context_dim = ar_context_dim
        self.nn_context_dim_spls = nn_context_dim * packet_dim
        self.ar_context_dim_spls = ar_context_dim * packet_dim
        self.output_dim = self.nn_context_dim_spls + packet_dim + extra_pred_dim
        self.chunk_dim = self.output_dim + self.ar_context_dim_spls
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.device = device

        self.ar_model = ARModel(ar_order, diagonal_load)
        self.up_ramp = np.linspace(0, 1, self.ar_fade_dim)
        self.down_ramp = np.linspace(1, 0, self.ar_fade_dim)

        meta = pd.read_csv(meta_path)
        self.meta = meta[meta["subset"] == fold]
        self.num_audio_files = len(self.meta)

        self.n_corrupted_packets = n_corrupted_packets

        self.surrogate_model = HybridModel.load_from_checkpoint(
            surrogate_model_checkpoint,
            packet_dim=packet_dim,
            extra_pred_dim=extra_pred_dim,
            channels=1,
            lite=True,
        ).to(self.device)

        # Define fade-in modulation vector (neural network contribution only)
        self.nn_fade_dim = nn_fade_dim
        self.nn_fade = np.linspace(0.0, 1.0, nn_fade_dim)

        # Define fade-in and fade-out modulation vectors
        self.fade_in = np.linspace(0.0, 1.0, extra_pred_dim)
        self.fade_out = np.linspace(1.0, 0.0, extra_pred_dim)

    def __len__(self):
        if self.fold == "training":
            return self.steps_per_epoch * self.batch_size
        else:
            return self.num_audio_files

    def __getitem__(self, index):
        # Randomize index at training time
        if self.fold == "training":
            index = random.randint(0, self.num_audio_files - 1)

        # Read metadata
        row = self.meta.iloc[index]

        # Medley-solos-DB file path; modify this line of code if another dataset is used
        filepath = Path(
            self.audio_dir,
            f"Medley-solos-DB_{self.fold}-{row['instrument_id']}_{row['uuid4']}.wav",
        )

        # Load audio file
        wav, __ = librosa.load(filepath, sr=self.sample_rate, mono=True)
        wav = wav.astype(np.float32)

        if self.fold == "training":
            # Data augmentation is not implemented yet
            wav = self._augment(wav)
            # Randomize the training chunk within the audio file
            idx = random.randint(0, len(wav) - self.chunk_dim - 1)
            chunk = wav[idx : idx + self.chunk_dim]
        else:
            # Use the very first chunk in every audio file for validation and test
            chunk = wav[: self.chunk_dim]

        # AR model contribution
        ar_data = self._get_ar_data(chunk)

        # Ground-truth audio data
        true = chunk[-self.output_dim :]

        # Get corrupted packets mask
        corrupted_packets = np.random.permutation(
            [1] * self.n_corrupted_packets
            + [0] * (self.nn_context_dim - self.n_corrupted_packets)
        )
        corrupted_packets_chunk = np.concat(
            (np.zeros(self.ar_context_dim), corrupted_packets), axis=0
        ).astype(np.float32)
        corrupted_samples_mask = np.repeat(corrupted_packets_chunk, self.packet_dim)
        corrupted_samples_mask = np.concat(
            (corrupted_samples_mask, np.zeros(self.pred_dim)), axis=0
        ).astype(np.float32)

        # Get lossy past
        chunk = chunk * (1 - corrupted_samples_mask)

        # Use surrogate model to get predictions for lossy past
        is_burst = False
        for i, lost in enumerate(corrupted_packets_chunk):
            if i < self.ar_context_dim:
                continue

            if lost:
                # Start index of the ith packet
                idx = i * self.packet_dim

                # AR model prediction
                ar_context = chunk[max(0, idx - self.ar_context_dim_spls) : idx]
                ar_context = np.pad(
                    ar_context, (self.ar_context_dim_spls - len(ar_context), 0)
                )
                ar_pred = self.ar_model.predict(valid=ar_context, steps=self.pred_dim)

                # NN model context
                nn_context = chunk[max(0, idx - self.nn_context_dim_spls) : idx]
                nn_context = np.pad(
                    nn_context,
                    (self.nn_context_dim_spls - len(nn_context), self.pred_dim),
                )
                nn_context = Tensor(nn_context[None, None, ...]).to(self.device)
                corrupted_packets_context = Tensor(
                    corrupted_packets_chunk[i - self.nn_context_dim : i]
                ).unsqueeze(0)

                # NN model inference
                with no_grad():
                    nn_pred = self.surrogate_model(
                        (nn_context, corrupted_packets_context)
                    )
                    nn_pred = nn_pred[..., -self.pred_dim :]
                    nn_pred = nn_pred.squeeze().cpu().numpy()

                # Apply fade-in to the neural network contribution (inbound fade-in)
                nn_pred[: self.nn_fade_dim] *= self.nn_fade

                # Combine the two predictions
                prediction = ar_pred + nn_pred

                # Cross-fade the compound prediction (outbound fade-out)
                prediction[-self.extra_pred_dim :] *= self.fade_out

                if is_burst:
                    # Cross-fade the prediction in case of consecutive packet losses (inbound fade-in)
                    prediction[: self.extra_pred_dim] *= self.fade_in

                # Cross-fade the output signal (outbound fade-in)
                chunk[idx + self.packet_dim : idx + self.pred_dim] *= self.fade_in

                # Conceal lost packet
                chunk[idx : idx + self.pred_dim] += prediction

                # Keep track of consecutive packet losses
                is_burst = True

            else:
                # Reset burst loss indicator
                is_burst = False

        # Valid neural network input, obtained by zeroing out the samples to be predicted
        past = chunk[-self.output_dim :].copy()
        past[-self.pred_dim :] = 0.0

        return (
            Tensor(true[None, :]),
            Tensor(past[None, :]),
            Tensor(ar_data[None, :]),
            Tensor(corrupted_packets[None, :]),
        )

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        # wav = self.augmentation(wav, sample_rate=self.sample_rate)
        return wav

    def _get_ar_data(self, chunk: np.ndarray) -> np.ndarray:
        ar_data = np.zeros(self.output_dim)

        for i in range(self.nn_context_dim + 1):
            idx = i * self.packet_dim
            valid = chunk[idx : idx + self.ar_context_dim * self.packet_dim]
            steps = (
                self.pred_dim
                if i == self.nn_context_dim
                else self.packet_dim + self.ar_fade_dim
            )
            ar_pred = self.ar_model.predict(valid=valid, steps=steps)
            ar_pred = self._apply_ar_fade(frame=ar_pred, index=i)
            ar_data[idx : idx + len(ar_pred)] += ar_pred

        return ar_data

    def _apply_ar_fade(self, frame: np.ndarray, index: int) -> np.ndarray:
        if self.ar_fade_dim:
            # Fade-out
            if index == 0:
                frame[-self.ar_fade_dim :] *= self.down_ramp
            # Fade-in
            elif index == self.nn_context_dim:
                frame[: self.ar_fade_dim] *= self.up_ramp
            # Cross-fade
            else:
                frame[: self.ar_fade_dim] *= self.up_ramp
                frame[-self.ar_fade_dim :] *= self.down_ramp

        return frame


class NMPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        audio_dir: str,
        meta_path: str,
        sample_rate: int,
        packet_dim: int,
        extra_pred_dim: int,
        nn_context_dim: int,
        ar_context_dim: int,
        ar_order: int,
        ar_fade_dim: int,
        diagonal_load: float,
        nn_fade_dim: int,
        device: str,
        surrogate_model_checkpoint: str,
        steps_per_epoch: int,
        batch_size: int,
        n_corrupted_packets: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.kwargs = {"num_workers": 8, "persistent_workers": True}

        self.train_dataset = NMPDataset(
            fold="training",
            audio_dir=audio_dir,
            meta_path=meta_path,
            sample_rate=sample_rate,
            packet_dim=packet_dim,
            extra_pred_dim=extra_pred_dim,
            nn_context_dim=nn_context_dim,
            ar_context_dim=ar_context_dim,
            ar_order=ar_order,
            ar_fade_dim=ar_fade_dim,
            diagonal_load=diagonal_load,
            nn_fade_dim=nn_fade_dim,
            device=device,
            surrogate_model_checkpoint=surrogate_model_checkpoint,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            n_corrupted_packets=n_corrupted_packets,
        )

        self.val_dataset = NMPDataset(
            fold="validation",
            audio_dir=audio_dir,
            meta_path=meta_path,
            sample_rate=sample_rate,
            packet_dim=packet_dim,
            extra_pred_dim=extra_pred_dim,
            nn_context_dim=nn_context_dim,
            ar_context_dim=ar_context_dim,
            ar_order=ar_order,
            ar_fade_dim=ar_fade_dim,
            diagonal_load=diagonal_load,
            nn_fade_dim=nn_fade_dim,
            device=device,
            surrogate_model_checkpoint=surrogate_model_checkpoint,
            batch_size=batch_size,
            n_corrupted_packets=n_corrupted_packets,
        )

        self.test_dataset = NMPDataset(
            fold="test",
            audio_dir=audio_dir,
            meta_path=meta_path,
            sample_rate=sample_rate,
            packet_dim=packet_dim,
            extra_pred_dim=extra_pred_dim,
            nn_context_dim=nn_context_dim,
            ar_context_dim=ar_context_dim,
            ar_order=ar_order,
            ar_fade_dim=ar_fade_dim,
            diagonal_load=diagonal_load,
            nn_fade_dim=nn_fade_dim,
            device=device,
            surrogate_model_checkpoint=surrogate_model_checkpoint,
            batch_size=batch_size,
            n_corrupted_packets=n_corrupted_packets,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True, **self.kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, self.batch_size, shuffle=False, **self.kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, self.batch_size, shuffle=False, **self.kwargs
        )
