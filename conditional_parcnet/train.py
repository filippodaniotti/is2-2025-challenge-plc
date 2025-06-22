import argparse

import pytorch_lightning as pl
import torch
import yaml
from data import NMPDataModule
from nn_model import HybridModel
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision("medium")


def main():
    # Parse command-line options
    parser = argparse.ArgumentParser(description="Train PARCnet.")
    parser.add_argument(
        "-c", "--config", default="config_tfilm.yaml", help="baseline config file"
    )
    args = parser.parse_args()

    # Set manual seed
    pl.seed_everything(42, workers=True)

    # Open config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Instantiate the neural network model
    model = HybridModel(
        packet_dim=config["global"]["packet_dim"],
        extra_pred_dim=config["global"]["extra_pred_dim"],
        lite=config["neural_net"]["lite"],
        channels=1,
    )

    # Instantiate TensorBoardLogger
    logger = pl_loggers.WandbLogger(
        project="is2-2025-plc-challenge",
        name=f"PARCNet with TFiLM and surrogate model {config["global"]["n_corrupted_packets"]} corrupted packets",
    )

    # Set up the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.save_dir
        + f"/checkpoints/parcnet_tfilm_{config["global"]["n_corrupted_packets"]}final",
        monitor="packet_val_mse",
        filename="parcnet-is2_baseline_{epoch:03d}_{val_mse:.2f}_{packet_val_mse:.2f}_{val_sdr:.3f}_{packet_val_sdr:.3f}",
        save_on_train_epoch_end=True,
        save_weights_only=True,
        save_top_k=10,
        verbose=True,
    )

    # Set up the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=config["neural_net"]["max_epochs"],
        gradient_clip_val=config["neural_net"]["gradient_clip"],
    )

    # Instantiate the Lightning datamodule
    datamodule = NMPDataModule(
        audio_dir=config["path"]["source_audio_dir"],
        meta_path=config["path"]["meta"],
        sample_rate=config["global"]["sr"],
        packet_dim=config["global"]["packet_dim"],
        extra_pred_dim=config["global"]["extra_pred_dim"],
        nn_context_dim=config["neural_net"]["nn_context_dim"],
        ar_context_dim=config["AR"]["ar_context_dim"],
        ar_order=config["AR"]["ar_order"],
        ar_fade_dim=config["AR"]["fade_dim"],
        diagonal_load=config["AR"]["diagonal_load"],
        steps_per_epoch=config["neural_net"]["steps_per_epoch"],
        nn_fade_dim=config["neural_net"]["fade_dim"],
        device="cpu",
        surrogate_model_checkpoint=config["path"]["surrogate_model_checkpoint"],
        batch_size=config["neural_net"]["batch_size"],
        n_corrupted_packets=config["global"]["n_corrupted_packets"],
    )

    # Train the neural network

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()