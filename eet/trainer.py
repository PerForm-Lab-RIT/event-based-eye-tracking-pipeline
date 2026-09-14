import pathlib
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig as Config

from .agents import BaseAgent

from .utils import Agg, print, Counter, timer
from .utils import when
from .utils.logger import Logger


class Trainer:
  def __init__(self, config: Config, train_loader: DataLoader,
        val_loader: DataLoader, agent: BaseAgent, logger: Logger):
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.agent: BaseAgent = agent
    self.config = config
    self.logger = logger
    self.device = next(self.agent.parameters()).device

    self.start_epoch = 1
    self.best_eval_score = float("-inf")  #
    self.best_eval_metric = "eval/p5"    #
    self.best_eval_p5 = float("-inf")
    self.latest_checkpoint_path = pathlib.Path(config.logdir) / "checkpoint_latest.pt"
    self.best_checkpoint_path = pathlib.Path(config.logdir) / "checkpoint_best.pt"

    self._epoch = 0
    self.step = Counter()
    self.should_log = when.Clock(float(self.config.log_every), first=True)
    print(f"Agent FLOPS: {agent.flops / 1e9:.4f} GFLOPS", color="green")


  def run(self):
    """Run the main training pipeline."""
    self._maybe_load_pretrain()

    # trainable_params = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
    # total_params = sum(p.numel() for p in self.agent.parameters())
    # print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / max(total_params, 1):.1f}%)", color="green")
    print(f"Trainable params: {self.agent.trainable_n_params:,} / {self.agent.n_params:,} ({100 * self.agent.trainable_n_params / max(self.agent.n_params, 1):.1f}%)", color="green")

    max_epochs = int(self.config.max_epochs)
    eval_every = max(1, int(self.config.eval_every))
    self.should_write = when.Clock(float(self.config.log_every), first=False)

    print(f"Training from epoch {self.start_epoch} to {max_epochs}")

    for epoch in range(self.start_epoch, max_epochs + 1):
      self._epoch = epoch

      with timer.section('train_epoch'):
        train_metrics = self.train()
      epoch_metrics = {**train_metrics}

      should_eval = (epoch % eval_every == 0) or (epoch == max_epochs)
      if should_eval:
        with timer.section('eval_epoch'):
          eval_metrics = self.eval()
        epoch_metrics.update(eval_metrics)

        # current_eval_loss = self._extract_eval_loss(eval_metrics)
        # current_eval_p5 = eval_metrics['eval/p5']
        # if current_eval_p5 > self.best_eval_p5:
        #   self.best_eval_p5 = current_eval_p5
        #   self._save_checkpoint(self.best_checkpoint_path, epoch)
        current_metric_name, current_eval_score = self._select_eval_score(eval_metrics)

        if current_eval_score > self.best_eval_score:
          self.best_eval_score = current_eval_score
          self.best_eval_metric = current_metric_name

          # Backward compatibility: if this is a localization run, keep best_eval_p5 updated.
          if current_metric_name == "eval/p5":
            self.best_eval_p5 = current_eval_score

          self._save_checkpoint(self.best_checkpoint_path, epoch)
      self._save_checkpoint(self.latest_checkpoint_path, epoch)

      # Add metrics to logger
      self._log_metrics(epoch_metrics)
      timer_stats = timer.stats()
      # Legacy key kept for backward compatibility with existing analysis scripts.
      # NOTE: This is actually forward time per batch call, not per frame.
      train_forward_key = "train_epoch/forward/avg"
      eval_forward_key = "eval_epoch/forward/avg"
      data_fetch_key = "train_epoch/data_fetch/avg"
      timer_stats["inference_time_per_frame"] = timer_stats[train_forward_key]
      timer_stats["inference_time_train_forward_per_batch"] = timer_stats[train_forward_key]
      # Eval forward timing is closer to deployment-style inference (no backward).
      if eval_forward_key in timer_stats:
        timer_stats["inference_time_eval_forward_per_batch"] = timer_stats[eval_forward_key]
      # Includes dataloader + preprocessing before tensor-to-device.
      if data_fetch_key in timer_stats:
        timer_stats["data_fetch_time_per_batch"] = timer_stats[data_fetch_key]
      self._log_metrics(timer_stats, prefix="timer")
      self.logger.write(int(self.step), fps=True)

    #print(f"Best eval p5: {self.best_eval_p5:.6f}")
    print(f"Best eval metric: {self.best_eval_metric} = {self.best_eval_score:.6f}")  #
    print(f"Latest checkpoint: {self.latest_checkpoint_path}")
    print(f"Best checkpoint: {self.best_checkpoint_path}")

    # Final flush
    self.logger.write(int(self.step), fps=True)

  def eval(self):
    """Evaluate one validation epoch and return aggregated metrics."""
    self.agent.eval()
    agg = Agg()

    with torch.no_grad():
      for step, batch in enumerate(self.val_loader, start=1):
        inputs, labels = self._prepare_batch(batch)
        metrics = self.agent.report(inputs, labels)
        scalar_metrics, media_metrics = self._split_scalar_and_media_metrics(metrics)
        agg.add(self._to_numpy_metrics(scalar_metrics))
        if media_metrics:
          self._log_metrics(media_metrics, prefix='openloop')
        if self.should_log(step):
          running = agg.result(reset=False)
          # print(
          #   f"[Eval][Epoch {self._epoch}][Step {step}/{len(self.val_loader)}] "
          #   f"loss={float(scalar_metrics['loss']):.6f} "
          #   f"p5={float(scalar_metrics['p5']):.6f}",
          #   # f" FLOPs:%s   MACs:%s   Params:%s \n" %scalar_metrics['FLOPs'], #Mobina
          #   color="blue"
          # )
          print(
            f"[Eval][Epoch {self._epoch}][Step {step}/{len(self.val_loader)}] "
            f"{self._format_metrics(scalar_metrics)}",
            color="blue"
          )
          # Log intermediate eval metrics
          self._log_metrics(self._to_float_metrics(running), prefix='eval_step')
          self.logger.write(int(self.step), fps=True)

    return self._to_float_metrics(agg.result(prefix="eval"))

  def train(self):
    """Train for one epoch and return aggregated metrics."""
    self.agent.train()
    agg = Agg()
    loader_iter = iter(self.train_loader)
    n_steps = len(self.train_loader)
    step = 0
    while True:
      with timer.section("data_fetch"):
        try:
          batch = next(loader_iter)
        except StopIteration:
          break
      step += 1

      inputs, labels = self._prepare_batch(batch)
      metrics = self.agent.update(inputs, labels)
      agg.add(self._to_numpy_metrics(metrics))

      # batch_size = int(inputs.shape[0])
      # batch_length = int(inputs.shape[1]) if inputs.ndim >= 2 else 1
      # self.step.increment(batch_size * batch_length)
      self.step.increment() # increase by 1 instead

      if self.should_log(step):
        running = agg.result(reset=False)
        print(
          f"[Train][Epoch {self._epoch}][Step {step}/{n_steps}] "  
          f"{self._format_metrics(metrics)}",
          color="cyan"
        )
        # Log intermediate train metrics
        self._log_metrics(self._to_float_metrics(running), prefix='train_step')
        self.logger.write(int(self.step), fps=True)

    return self._to_float_metrics(agg.result(prefix="train"))

  def _format_metrics(self, metrics: Mapping[str, Any]) -> str:
    """Format whatever scalar metrics are available.

    Works for both localization agents and segmentation agents.
    """
    parts = []
    preferred_keys = [
      "loss",
      "p1", "p2", "p3", "p5", "p10",
      "rmse", "mae",
      "pixel_acc", "miou",
    ]

    for key in preferred_keys:
      if key not in metrics:
        continue

      value = metrics[key]
      try:
        if isinstance(value, torch.Tensor):
          value = value.detach().cpu().item()
        elif hasattr(value, "item"):
          value = value.item()
        value = float(value)
        parts.append(f"{key}={value:.6f}")
      except Exception:
        parts.append(f"{key}={value}")

    if len(parts) == 0:
      parts.append(f"metrics={list(metrics.keys())}")

    return " ".join(parts)

  def _select_eval_score(self, eval_metrics: Mapping[str, Any]) -> tuple[str, float]:
    """Choose the metric used for best-checkpoint selection.

    Higher score is always better. For losses, we negate the value.
    """
    higher_is_better = [
      "eval/p5",          # localization
      "eval/p10",         # localization fallback
      "eval/miou",        # segmentation
      "eval/pixel_acc",   # segmentation fallback
    ]

    for key in higher_is_better:
      if key in eval_metrics:
        return key, float(eval_metrics[key])

    lower_is_better = [
      "eval/loss",
      "eval/loss/total",
      "eval/loss/huber",
    ]

    for key in lower_is_better:
      if key in eval_metrics:
        return key, -float(eval_metrics[key])

    raise KeyError(
      f"Could not select eval score. Available eval metric keys: {list(eval_metrics.keys())}"
    )

  def _prepare_batch(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if "input" in batch:
      inputs = batch["input"]
    elif "data" in batch:
      inputs = batch["data"]
    else:
      raise KeyError(f"Batch must contain 'input' or 'data'. Got keys: {list(batch.keys())}")

    if "label" in batch:
      labels = batch["label"]
    elif "labels" in batch:
      labels = batch["labels"]
    else:
      raise KeyError(f"Batch must contain 'label' or 'labels'. Got keys: {list(batch.keys())}")


    if not isinstance(inputs, torch.Tensor):
      inputs = torch.as_tensor(inputs)
    if not isinstance(labels, torch.Tensor):
      labels = torch.as_tensor(labels)

    inputs = inputs.to(self.device, non_blocking=True).float()
    labels = labels.to(self.device, non_blocking=True).float()

    return inputs, labels

  def _pixel_threshold_metrics(self, diff: torch.Tensor) -> Dict[str, torch.Tensor]:
    if diff.shape[-1] < 2:
      return {}

    error = torch.linalg.vector_norm(diff[..., :2], ord=2, dim=-1)
    return {
      "metric/p3": (error <= 3.0).float().mean(),
      "metric/p5": (error <= 5.0).float().mean(),
      "metric/p10": (error <= 10.0).float().mean(),
    }

  def _to_numpy_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    output = {}
    for key, value in metrics.items():
      if isinstance(value, torch.Tensor):
        if value.ndim == 0:
          output[key] = value.detach().cpu().item()
        else:
          output[key] = value.detach().cpu().numpy()
      else:
        output[key] = value
    return output

  def _to_float_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
    output = {}
    for key, value in metrics.items():
      if isinstance(value, torch.Tensor):
        if value.ndim == 0:
          output[key] = float(value.detach().cpu().item())
      elif hasattr(value, "item"):
        try:
          output[key] = float(value.item())
        except (TypeError, ValueError):
          continue
      else:
        try:
          output[key] = float(value)
        except (TypeError, ValueError):
          continue
    return output

  def _split_scalar_and_media_metrics(self, metrics: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    scalar_metrics: Dict[str, Any] = {}
    media_metrics: Dict[str, Any] = {}
    for key, value in metrics.items():
      if isinstance(value, torch.Tensor):
        is_media = value.ndim >= 3
      else:
        arr = np.asarray(value)
        is_media = arr.ndim >= 3
      if is_media:
        media_metrics[key] = value
      else:
        scalar_metrics[key] = value
    return scalar_metrics, media_metrics

  def _log_metrics(self, metrics: Mapping[str, Any], prefix: str | None = None) -> None:
    for key, value in metrics.items():
      if key == "summary":
        continue
      name = f"{prefix}/{key}" if prefix else key

      if isinstance(value, torch.Tensor):
        if value.ndim == 5:
          self.logger.video(name, value.detach().cpu().numpy())
          continue
        if value.ndim == 0:
          self.logger.scalar(name, float(value.detach().cpu().item()))
          continue

      array_value = np.asarray(value)
      if array_value.ndim == 5:
        self.logger.video(name, array_value)
        continue

      try:
        self.logger.scalar(name, float(value))
      except (TypeError, ValueError):
        continue

  def _extract_eval_loss(self, eval_metrics: Mapping[str, float]) -> float:
    for key in ("eval/loss/huber", "eval/loss", "eval/loss/total"):
      if key in eval_metrics:
        return float(eval_metrics[key])
    return float("inf")

  def _save_checkpoint(self, path: pathlib.Path, epoch: int) -> None:
    payload = {
      "epoch": int(epoch),
      "step": self.step.save(),
      "best_eval_p5": float(self.best_eval_p5),
      "agent": self.agent.save()
    }
    torch.save(payload, path)

  def _maybe_load_pretrain(self) -> None:
    pretrain = pathlib.Path(self.config.from_pretrained) if (self.config.from_pretrained and str(self.config.from_pretrained) != "0") else None
    ckpt_path: pathlib.Path = None
    best_ckpt_path: pathlib.Path = None
    if pretrain is not None:
      ckpt_path = pretrain
      best_ckpt_path = pretrain
    else:
      ckpt_path = self.latest_checkpoint_path
      best_ckpt_path = self.best_checkpoint_path

    if not ckpt_path.exists():
      return

    # Load the best agent and best stats if available, otherwise load the latest checkpoint
    checkpoint = torch.load(ckpt_path, map_location=self.device)
    if best_ckpt_path.exists() and best_ckpt_path != ckpt_path:
      best_checkpoint = torch.load(best_ckpt_path, map_location=self.device)
      self.best_eval_p5 = float(best_checkpoint['best_eval_p5'])
      self.agent.load(best_checkpoint["agent"])
    else:
      self.agent.load(checkpoint["agent"])
      self.best_eval_p5 = float(checkpoint['best_eval_p5'])

    # For step and epoch, always load from the latest checkpoint to ensure the training progress is correctly
    #   reflected, even if the agent weights are loaded from a different checkpoint
    self.start_epoch = checkpoint['epoch'] + 1
    self.step.load(checkpoint['step'])

    print(f"Loaded pretrain checkpoint: {ckpt_path}")


