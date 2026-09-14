
# """
# File: main.py
# Author: Viet Nguyen
# Date: 2025-12-08

# Description: Main file
# """

"""
Standalone evaluation script.

Runs validation on checkpoint_best.pt and exports:
  - predictions.npz: flattened predictions and labels in pixel space
  - eval_summary.npz: mean scalar metrics from agent.loss/report path
"""

import atexit
import pathlib
import sys
from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as Config

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from eet.agents import BaseAgent, build_agent
from eet.data import build_dataloader, build_dataset
from eet.utils import preprocess_config, print, setup_console_log


def _prepare_batch(batch: Dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  inputs = batch["input"]
  labels = batch["label"]

  if not isinstance(inputs, torch.Tensor):
    inputs = torch.as_tensor(inputs)
  if not isinstance(labels, torch.Tensor):
    labels = torch.as_tensor(labels)

  return (
    inputs.to(device, non_blocking=True).float(),
    labels.to(device, non_blocking=True).float(),
  )


def _extract_predictions(agent: BaseAgent, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
  """Return predictions in pixel space for any agent.

  Preferred path:
    agent.loss(...) -> third return item is predictions_px in this codebase.
  Fallback path:
    agent.decode_outputs(agent.forward(...))
  """
  # loss_out = agent.loss(inputs, labels)
  # if isinstance(loss_out, tuple) and len(loss_out) >= 3:
  #   predictions_px = loss_out[2]
  #   if isinstance(predictions_px, torch.Tensor):
  #     return predictions_px
  return agent.decode_outputs(agent(inputs))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: Config):
  config = preprocess_config(config)
  assert config.logdir is not None, "logdir must be specified in the config for evaluation."
  ckpt_path = config.logdir / "checkpoint_best.pt"
  assert ckpt_path.exists(), f"checkpoint_best.pt not found in {config.logdir}"

  setup_console_log(config.logdir, filename="console_eval.log")
  atexit.register(setup_console_log, config.logdir, filename="console_eval.log")

  val_dataset, input_space, label_space = build_dataset(config, split="val")
  val_loader = build_dataloader(
    val_dataset,
    batch_size=int(config.batch_size),
    num_workers=config.data.num_workers,
    prefetch_factor=config.data.prefetch_factor,
    persistent_workers=config.data.persistent_workers,
  )

  device = torch.device(config.device)
  agent: BaseAgent = build_agent(config, input_space, label_space)
  checkpoint = torch.load(ckpt_path, map_location="cpu")
  agent.load(checkpoint["agent"])
  agent.to(device)
  agent.eval()

  prediction_list: list[np.ndarray] = []
  label_list: list[np.ndarray] = []
  scalar_metrics: Dict[str, list[float]] = {}

  with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader, start=1):
      print(f"Evaluating batch {batch_idx}/{len(val_loader)}", end="\r", color="green")

      inputs, labels_seq = _prepare_batch(batch, device)
      predictions_px = _extract_predictions(agent, inputs, labels_seq)

      # Keep sequence-shaped tensors for agent.report/loss compatibility.
      predictions_to_save = predictions_px.flatten(0, 1) if predictions_px.ndim > 2 else predictions_px
      labels_to_save = labels_seq.flatten(0, 1) if labels_seq.ndim > 2 else labels_seq

      prediction_list.append(predictions_to_save.detach().cpu().numpy())
      label_list.append(labels_to_save.detach().cpu().numpy())

      # Optional scalar metrics via report() for comparable eval summary.
      report_metrics = agent.report(inputs, labels_seq)
      for key, value in report_metrics.items():
        if isinstance(value, torch.Tensor) and value.ndim == 0:
          scalar_metrics.setdefault(key, []).append(float(value.detach().cpu().item()))
        elif np.isscalar(value):
          scalar_metrics.setdefault(key, []).append(float(value))

  predictions = np.concatenate(prediction_list, axis=0)
  labels = np.concatenate(label_list, axis=0)
  np.savez(config.logdir / "predictions.npz", predictions=predictions, labels=labels)

  if scalar_metrics:
    summary = {key: float(np.mean(values)) for key, values in scalar_metrics.items() if values}
    np.savez(config.logdir / "eval_summary.npz", **summary)
    print(f"Saved eval summary with {len(summary)} scalar metrics to {config.logdir / 'eval_summary.npz'}", color="blue")

  print(f"Saved predictions to {config.logdir / 'predictions.npz'}", color="blue")


if __name__ == "__main__":
  main()
