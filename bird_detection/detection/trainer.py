"""Simple training loop for torchvision detection models."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


Batch = Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]


class DetectionTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        device: torch.device,
        logger: logging.Logger,
        writer: Optional["SummaryWriter"] = None,
        use_amp: bool = False,
        grad_clip: float = 0.0,
        log_every: int = 25,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.writer = writer
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.scaler = GradScaler(enabled=self.use_amp)
        self.global_step = 0

    def fit(
        self,
        train_loader: Iterable[Batch],
        val_loader: Optional[Iterable[Batch]],
        max_epochs: int,
        checkpoint_dir: Path,
        start_epoch: int = 1,
    ) -> Dict[str, float]:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        best_path: Optional[Path] = None
        best_state_path: Optional[Path] = None

        # Epoch progress bar
        epoch_pbar = tqdm(
            range(start_epoch, max_epochs + 1),
            desc="Training",
            initial=start_epoch - 1,
            total=max_epochs,
        )

        for epoch in epoch_pbar:
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss = (
                self._evaluate(val_loader, split="val")
                if val_loader is not None
                else None
            )

            if self.scheduler is not None:
                self.scheduler.step()

            self._save_checkpoint(checkpoint_dir, epoch, filename="last_checkpoint.pt")

            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_path = checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                best_state_path = checkpoint_dir / "best_checkpoint.pt"
                self._save_checkpoint(
                    checkpoint_dir, epoch, filename="best_checkpoint.pt"
                )
                self.logger.info(
                    "New best model saved at %s (val_loss=%.4f)", best_path, best_val
                )

            # Update epoch progress bar
            if val_loss is not None:
                epoch_pbar.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    best=f"{best_val:.4f}",
                )
            else:
                epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}")

            if self.writer:
                self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("epoch/val_loss", val_loss, epoch)

        epoch_pbar.close()

        metrics: Dict[str, float | str] = {
            "train_loss": train_loss,
            "last_checkpoint": str(checkpoint_dir / "last_checkpoint.pt"),
        }
        if val_loader is not None and best_val < float("inf"):
            metrics["val_loss"] = best_val
        if best_path is not None:
            metrics["best_checkpoint"] = str(best_path)
        if best_state_path is not None:
            metrics["best_state"] = str(best_state_path)
        return metrics

    def evaluate(self, data_loader: Iterable[Batch], split: str = "test") -> float:
        return self._evaluate(data_loader, split=split)

    def _prepare_batch(
        self, batch: Batch
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        images, targets = batch
        images = [img.to(self.device) for img in images]
        processed_targets: List[Dict[str, torch.Tensor]] = []
        for target in targets:
            processed_targets.append(
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in target.items()
                }
            )
        return images, processed_targets

    def _train_one_epoch(self, data_loader: Iterable[Batch], epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        step_count = 0
        last_log_time = time.perf_counter()

        # Training progress bar
        train_pbar = tqdm(
            enumerate(data_loader, start=1),
            total=len(data_loader),
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for step, batch in train_pbar:
            images, targets = self._prepare_batch(batch)
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if self.use_amp:
                self.scaler.scale(losses).backward()
                if self.grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            running_loss += losses.item()
            self.global_step += 1
            step_count = step

            # Update progress bar
            avg_loss = running_loss / step
            lr = self.optimizer.param_groups[0]["lr"]
            train_pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.6f}")

            if step % self.log_every == 0:
                now = time.perf_counter()
                iter_time = (now - last_log_time) / self.log_every
                last_log_time = now
                self.logger.info(
                    "Epoch %d Step %d - loss: %.4f lr: %.6f (%.3fs/step)",
                    epoch,
                    step,
                    avg_loss,
                    lr,
                    iter_time,
                )
                if self.writer:
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

        train_pbar.close()
        return running_loss / max(1, step_count)

    def _evaluate(self, data_loader: Iterable[Batch], split: str) -> float:
        if data_loader is None:
            return float("nan")

        was_training = self.model.training
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        # Evaluation progress bar
        eval_pbar = tqdm(
            data_loader,
            desc=f"Eval {split}",
            leave=False,
        )

        with torch.no_grad():
            for batch in eval_pbar:
                images, targets = self._prepare_batch(batch)
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                total_steps += 1
                
                # Update progress bar
                avg_loss = total_loss / total_steps
                eval_pbar.set_postfix(loss=f"{avg_loss:.4f}")

        eval_pbar.close()

        if not was_training:
            self.model.eval()

        avg_loss = total_loss / max(1, total_steps)
        self.logger.info("%s loss: %.4f", split.capitalize(), avg_loss)
        if self.writer:
            self.writer.add_scalar(f"{split}/loss", avg_loss, self.global_step)
        return avg_loss

    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int, filename: str) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "global_step": self.global_step,
        }
        torch.save(state, checkpoint_dir / filename)
