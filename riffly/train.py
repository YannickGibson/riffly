"""This module contains functions for training, inference, validation. It is meant to be imported."""

from __future__ import annotations

# Standard libraries
import copy
import gc
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

# Third-party libraries
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from riffly.plots import show_val_score_and_loss

# Local libraries
from riffly.utils.general import get_threshold

if TYPE_CHECKING:
    from neptune.metadata_containers import Run


def inference_loop(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pbar_text: str = "",
    loss_fn: nn.Module | None = None,
    vae: bool = False,
    verbose: bool = True,
    **predict_kwargs,
) -> tuple[list, list, float]:

    _preds, _labels, _losses, files = [], [], [], []
    model.eval()
    with torch.no_grad():
        iterable = tqdm(dataloader) if verbose else dataloader
        for midi_ids, inputs in iterable:
            inputs = inputs.to(device)
            preds = model(inputs, **predict_kwargs)
            if loss_fn is not None:
                if vae:
                    preds, mu, logvar = preds
                    loss = loss_fn(preds, inputs, mu, logvar).item()
                else:
                    loss = loss_fn(preds, inputs).item()
                _losses.append(loss)
            # Save predictions and labels
            preds = preds.detach().cpu().numpy()
            threshold = get_threshold(preds)
            preds = preds > threshold
            for pred in preds:
                _preds.append(pred)
            # Labels
            inputs = inputs.detach().cpu().numpy()
            for label in inputs:
                _labels.append(label)
            # Files
            files.extend(midi_ids)

            if verbose:
                iterable.set_description(pbar_text)

    for i in range(len(_labels)):
        _labels[i] = (_labels[i] > 0.5).astype(bool)
        _preds[i] = (_preds[i] > 0.5).astype(bool)
    _labels = np.array(_labels)
    _preds = np.array(_preds)

    return _preds, _labels, _losses, files


def validation_loop(
    model: nn.Module,
    val_dl: DataLoader,
    val_metrics: list[Callable],
    loss_fn: nn.Module,
    device: torch.device,
    vae: bool = False,
    verbose=True,
):
    """Validation loop."""
    predictions, labels, val_loss, files = inference_loop(
        model=model,
        dataloader=val_dl,
        pbar_text="Running validation loop",
        loss_fn=loss_fn,
        device=device,
        vae=vae,
        verbose=verbose,
    )
    scores = {}

    for metric in val_metrics:
        scores[metric.__name__] = metric(labels.flatten(), predictions.flatten())

    scores["val_loss"] = val_loss
    return scores, predictions, labels, val_loss, files


def train_loop(
    model: nn.Module,
    train_dl: DataLoader,
    loss_fn: nn.Module,
    epoch: int,
    epochs: int,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    vae: bool = False,
    verbose: bool = False,
    grad_clip: float = 0.0,
    ema=None,
) -> None:
    """Train one epoch."""
    model.train()
    epoch_postfix = "st" if epoch == 0 else "nd" if epoch == 1 else "rd" if epoch == 2 else "th"
    best_loss = float("inf")
    losses = []
    iterable = tqdm(train_dl) if verbose else train_dl
    for _file, inputs in iterable:
        inputs = inputs.to(device)
        # Step
        optimizer.zero_grad()
        if scaler is None:
            outputs = model(inputs)
        else:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                inputs = inputs.half()
                outputs = model(inputs)

        if vae:
            recon, mu, logvar = outputs
            loss = loss_fn(recon, inputs, mu, logvar)
        else:
            loss = loss_fn(outputs, inputs)

        if scaler is None:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            best_loss = min(best_loss, loss.item())
        if ema is not None:
            ema.update(model)
        lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
        if verbose:
            iterable.set_description(
                f"Running {epoch + 1}{epoch_postfix}/{epochs} epoch, "
                f"lr: {lr:.1e}, Best Loss: {best_loss:.3f}, "
                f"Loss: {loss.item():.3f}",
            )
        losses.append(loss.item())
        # Step
    return losses


class EMA:
    """Exponential moving average of model parameters, kept as a shadow state_dict.

    Call ``update(model)`` after every optimizer step, ``copy_to(model)`` to
    swap EMA weights in, and ``restore(model)`` to swap the live weights back.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup: dict | None = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            return
        model.load_state_dict(self._backup, strict=False)
        self._backup = None


def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    loss_fn: nn.Module,
    optimizer_name: str,
    lr: float,
    epochs: int,
    device: torch.device,
    val_metrics: dict[str, Callable] | None = None,
    val_score_metric: Callable | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    half: bool = False,
    vae: bool = False,
    use_neptune: bool = False,
    verbose: bool = True,
    save_folder: str | None = None,
    weight_decay: float = 0.0,
    grad_clip: float = 0.0,
    early_stop_patience: int = 0,
    ema_decay: float = 0.0,
):
    """Use this for training a model from scratch."""
    if val_metrics is not None and val_score_metric not in val_metrics:
        msg = f"val_metrics must contain a '{val_score_metric}' key with a callable metric function."
        raise ValueError(
            msg,
        )

    # Init optimizer
    optimizer_name = optimizer_name.upper()
    if optimizer_name == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        msg = "Invalid optimizer name. Must be 'adam', 'sgd' or 'adamw' or others."
        raise ValueError(msg)

    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    if use_neptune:
        import neptune

        neptune_run: Run = neptune.init_run(
            project=os.environ["NEPTUNE_PROJECT"],
            api_token=os.environ["NEPTUNE_API_TOKEN"],
        )
        hidden_layers = model.hidden_layers if hasattr(model, "hidden_layers") else None
        if hidden_layers is not None:
            ", ".join([str(n_features) for n_features in model.hidden_layers])
        neptune_run["parameters"] = {
            "cols": model.columns,
            "rows": model.rows,
            "lr0": lr,
            "train_size": len(train_dl.dataset),
            "val_size": len(val_dl.dataset),
            "batch_size": train_dl.batch_size,
            "epochs": epochs,
            "optimizer": optimizer_name,
            "scheduler": str(scheduler),
            "loss_fn": str(loss_fn),
            "hidden_layers": str(hidden_layers),
            # "dropout": model.dropout_rate,
            "half": half,
            "crossval": False,
        }
    else:
        neptune_run = None

    # Create save directories if save_folder is provided
    weights_folder = None
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        weights_folder = os.path.join(save_folder, "weights")
        os.makedirs(weights_folder, exist_ok=True)

    val_losses = []
    val_scores = []
    val_scores_dict = {metric.__name__: [] for metric in val_metrics}
    losses = []
    if verbose:
        pass
    best_score = 0
    best_val_loss = float("inf")
    best_model = None
    best_epoch = 0
    last_score = None
    time.time()
    scaler = torch.amp.GradScaler() if half else None
    ema = EMA(model, decay=ema_decay) if ema_decay and ema_decay > 0 else None
    epochs_since_improve = 0

    iterable = range(epochs) if verbose else tqdm(range(epochs))

    for epoch in iterable:
        # Per-epoch loss hook (e.g. KL annealing schedules)
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)
        # Train loop
        gc.collect()
        # print(f"Epoch {epoch + 1}/{epochs}\n-------------------------------")
        _losses = train_loop(
            train_dl=train_dl,
            model=model,
            loss_fn=loss_fn,
            vae=vae,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            epochs=epochs,
            device=device,
            verbose=verbose,
            grad_clip=grad_clip,
            ema=ema,
        )
        losses.extend(_losses)
        if scheduler:
            scheduler.step()

        # Validation loop — use EMA weights if enabled
        if ema is not None:
            ema.copy_to(model)
        metric_scores = validation_loop(
            model,
            val_dl=val_dl,
            val_metrics=val_metrics,
            loss_fn=loss_fn,
            device=device,
            vae=vae,
            verbose=verbose,
        )[0]
        val_score = metric_scores[val_score_metric.__name__]
        _val_losses = metric_scores["val_loss"]
        avg_val_loss = np.average(_val_losses)
        val_scores.append(val_score)
        val_losses.append(avg_val_loss)
        for key, value in metric_scores.items():
            if key in val_scores_dict:
                val_scores_dict[key].append(value)
        if verbose:
            pass
        best_score = max(best_score, val_score)
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            # Snapshot the weights that produced this val score — EMA if active, else live.
            best_model = copy.deepcopy(model).to("cpu")
            epochs_since_improve = 0
            if best_score > 0.8 or val_score > 0.8:
                save_name = f"best_model_{best_val_loss:.3f}.pth"
                if weights_folder is not None:
                    save_path = os.path.join(weights_folder, save_name)
                else:
                    save_path = save_name
                print(f"Saving best model with val_loss {best_val_loss:.3f} (val_score {best_score:.3f})")
                torch.save(best_model.state_dict(), save_path)
        else:
            epochs_since_improve += 1
        last_score = val_score

        # Restore live weights for next training epoch
        if ema is not None:
            ema.restore(model)

        if not verbose:
            iterable.set_description(
                f"Results: val_score: {val_score:.3f}, val_loss: {avg_val_loss:.3f}, train_loss: {losses[-1]:.3f}",
            )

        if use_neptune:
            neptune_run["train/loss"].append(_losses)
            for key, value in metric_scores.items():
                neptune_run[f"val/{key}"].append(value)
            neptune_run["val/loss"].append(float(avg_val_loss))
            neptune_run["lr"].append(scheduler.get_last_lr()[0])
            neptune_run["epoch"].append(epoch + 1)

        if early_stop_patience > 0 and epochs_since_improve >= early_stop_patience:
            print(f"[early-stop] no val-loss improvement for {early_stop_patience} epochs; stopping at epoch {epoch + 1}")
            break
        ## Epoch

    time.time()

    show_val_score_and_loss(
        val_scores_dict=val_scores_dict,
        val_score_metric_name=val_score_metric.__name__,
        val_losses=val_losses,
        losses=losses,
        batch_count=len(train_dl),
        batch_size=train_dl.batch_size,
        epochs=epochs,
        best_epoch=best_epoch,
        neptune_run=neptune_run,
        neptune_only=False,
        save_folder=save_folder,
    )

    if neptune_run:
        neptune_run.stop()

    return losses, val_losses, val_scores, best_score, best_model, best_epoch, last_score


def single_train(model: nn.Module, columns: int, rows: int, verbose: bool = False) -> None:
    """Example train for prototyping."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="Tempo, Key or Time signature change events found on "
        "non-zero tracks.  This is not a valid type 0 or type 1 "
        "MIDI file.  Tempo, Key or Time Signature may be wrong.",
    )
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from riffly.datasets import MIDIDataset

    MIDIS_PATH = "../datasets"
    dataset = MIDIDataset(path=MIDIS_PATH, columns=columns, rows=rows)
    # files = dataset.files
    # train_files, val_files = train_test_split(files, test_size=0.2, random_state=33)
    # train_dataset = MIDIDataset(files=train_files, columns=COLUMNS, rows=ROWS)
    # val_dataset = MIDIDataset(files=val_files, columns=COLUMNS, rows=ROWS)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def focal_loss(pred, target, alpha=0.25, gamma=2.0):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction="none",
        )
        p = torch.sigmoid(pred)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        p_t = target * p + (1 - target) * (1 - p)
        fl = alpha_t * (1 - p_t) ** gamma * BCE_loss
        return fl.mean()

    def pixel_iou(y_true, y_pred, average="macro"):
        # Flatten the 2D arrays to 1D
        y_true_flat = y_true
        y_pred_flat = y_pred

        # True Positives, False Positives, False Negatives
        intersection = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
        union = np.sum((y_true_flat == 1) | (y_pred_flat == 1))

        # IoU calculation
        return intersection / union if union != 0 else 0

    # torch.nn.BCEWithLogitsLoss(), torch.nn.MSELoss()
    # bce with logits usually converges to all zeros
    epochs = 10
    batch_size = 16

    model = model.to(device)

    loss_fn = focal_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    half = False  # only for gpus

    from sklearn.metrics import roc_auc_score

    val_score_metric = f1_score
    val_metrics = [val_score_metric, pixel_iou, roc_auc_score]

    train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    train(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        val_metrics=val_metrics,
        val_score_metric=val_score_metric,
        half=half,
        verbose=verbose,
    )


if __name__ == "__main__":
    COLUMNS = 16
    ROWS = 7
    from models.autoencoder import AE

    model = AE(columns=COLUMNS, rows=ROWS)
    single_train(model, columns=COLUMNS, rows=ROWS, verbose=False)
