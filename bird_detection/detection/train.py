"""Hydra entrypoint for training detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from bird_detection.detection.augmentations import build_transform_pipeline
from bird_detection.detection.dataloaders import build_dataloaders
from bird_detection.detection.models import build_model
from bird_detection.detection.trainer import DetectionTrainer
from bird_detection.detection.utils import save_pretrained, set_seed


def _prepare_data_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = dict(cfg)
    data_cfg["birds_dir"] = to_absolute_path(data_cfg["birds_dir"])
    data_cfg["birds_mapping_csv"] = to_absolute_path(data_cfg["birds_mapping_csv"])
    if data_cfg.get("squirrels_dir"):
        data_cfg["squirrels_dir"] = to_absolute_path(data_cfg["squirrels_dir"])
    return data_cfg


def _build_optimizer(model: torch.nn.Module, training_cfg: Dict[str, Any]):
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = training_cfg.get("optimizer", "sgd").lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=training_cfg["lr"],
            momentum=training_cfg.get("momentum", 0.9),
            weight_decay=training_cfg.get("weight_decay", 0.0005),
            nesterov=training_cfg.get("nesterov", False),
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=training_cfg["lr"],
            weight_decay=training_cfg.get("weight_decay", 0.0001),
        )
    elif opt_name == "adam":
        betas = training_cfg.get("betas", (0.9, 0.999))
        if isinstance(betas, list):
            betas = tuple(betas)
        optimizer = torch.optim.Adam(
            params,
            lr=training_cfg["lr"],
            betas=betas,
            eps=training_cfg.get("eps", 1e-8),
            weight_decay=training_cfg.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'")
    return optimizer


def _build_scheduler(optimizer, training_cfg: Dict[str, Any]):
    sched_cfg = training_cfg.get("lr_scheduler")
    if not sched_cfg:
        return None
    milestones = sched_cfg.get("milestones", [])
    gamma = sched_cfg.get("gamma", 0.1)
    if not milestones:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(to_absolute_path(path_str))
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log_level = getattr(logging, str(cfg.logging.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    logger = logging.getLogger("train")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_aug_cfg = OmegaConf.to_container(cfg.augmentations.train, resolve=True)
    val_aug_cfg = OmegaConf.to_container(cfg.augmentations.val, resolve=True)
    resize_cfg = None
    if cfg.augmentations.get("resize_if_needed"):
        resize_cfg = OmegaConf.to_container(
            cfg.augmentations.resize_if_needed, resolve=True
        )
    data_cfg = _prepare_data_config(OmegaConf.to_container(cfg.data, resolve=True))
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)

    train_transform = build_transform_pipeline(
        train_aug_cfg.get("standard"),
        cfg.data.image_mean,
        cfg.data.image_std,
        resize_cfg=resize_cfg,
    )
    eval_transform = build_transform_pipeline(
        val_aug_cfg.get("standard"),
        cfg.data.image_mean,
        cfg.data.image_std,
        resize_cfg=resize_cfg,
    )

    train_loader, val_loader, test_loader, metadata = build_dataloaders(
        data_cfg=data_cfg,
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=cfg.seed,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg.get("pin_memory", True),
        advanced_train_aug=train_aug_cfg.get("advanced"),
    )

    model = build_model(model_cfg, metadata.num_classes).to(device)
    optimizer = _build_optimizer(model, training_cfg)
    scheduler = _build_scheduler(optimizer, training_cfg)

    tensorboard_dir = _resolve_path(str(cfg.logging.tensorboard_dir))
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    trainer = DetectionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        writer=writer,
        use_amp=training_cfg.get("amp", False),
        grad_clip=training_cfg.get("grad_clip", 0.0),
        log_every=training_cfg.get("log_every", 25),
    )

    resume_path = training_cfg.get("resume_from")
    start_epoch = 1
    if resume_path:
        resume_file = _resolve_path(str(resume_path))
        if resume_file.exists():
            checkpoint = torch.load(resume_file, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and checkpoint.get("scheduler_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if trainer.use_amp and checkpoint.get("scaler_state_dict"):
                trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            trainer.global_step = checkpoint.get("global_step", 0)
            start_epoch = checkpoint.get("epoch", 0) + 1
            logger.info(
                "Resuming training from %s at epoch %d (global_step=%d)",
                resume_file,
                start_epoch,
                trainer.global_step,
            )
        else:
            logger.warning(
                "Resume checkpoint %s not found. Starting from scratch.", resume_file
            )

    checkpoint_dir = _resolve_path(str(training_cfg["checkpoint_dir"]))
    max_epochs = training_cfg["epochs"]
    metrics: Dict[str, float | str] = {"train_loss": float("nan")}
    if start_epoch > max_epochs:
        logger.info(
            "Checkpoint epoch %d exceeds configured max epochs %d. Skipping additional training.",
            start_epoch - 1,
            max_epochs,
        )
    else:
        metrics = trainer.fit(
            train_loader,
            val_loader if len(val_loader.dataset) > 0 else None,
            max_epochs=max_epochs,
            checkpoint_dir=checkpoint_dir,
            start_epoch=start_epoch,
        )

    best_checkpoint = metrics.get("best_checkpoint")
    if best_checkpoint:
        state_dict = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    if len(test_loader.dataset) > 0:
        trainer.evaluate(test_loader, split="test")

    save_dir = _resolve_path(str(training_cfg["model_dir"]))
    model.eval()
    config_to_save = OmegaConf.to_container(cfg, resolve=True)
    save_pretrained(model, save_dir, metadata, config_to_save)
    writer.close()
    logger.info("Training completed. Model saved to %s", save_dir)


if __name__ == "__main__":
    main()
