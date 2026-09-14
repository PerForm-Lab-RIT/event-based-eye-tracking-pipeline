import contextlib
import io
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init
from torch.utils.tensorboard import SummaryWriter
import torch

from . import print

class CudaBenchmark:
  def __init__(self, comment):
    self._comment = comment

  def __enter__(self):
    self._st = torch.cuda.Event(enable_timing=True)
    self._nd = torch.cuda.Event(enable_timing=True)
    self._st.record()

  def __exit__(self, *args):
    self._nd.record()
    torch.cuda.synchronize()
    print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
  def __init__(self, logdir, filename="metrics.jsonl"):
    self._logdir = logdir
    self._filename = filename
    self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
    self._last_step = None
    self._last_time = None
    self._scalars = {}
    self._images = {}
    self._videos = {}
    self._histograms = {}

  def scalar(self, name, value):
    self._scalars[name] = float(value)

  def image(self, name, value):
    self._images[name] = np.array(value)

  def video(self, name, value):
    self._videos[name] = np.array(value)

  def histogram(self, name, value):
    self._histograms[name] = np.array(value)

  def write(self, step, fps=False):
    scalars = list(self._scalars.items())
    if fps:
      scalars.append(("fps/fps", self._compute_fps(step)))
    print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars), color='green')
    with (self._logdir / self._filename).open("a") as f:
      f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
    for name, value in scalars:
      if "/" not in name:
        self._writer.add_scalar("scalars/" + name, value, step)
      else:
        self._writer.add_scalar(name, value, step)
    for name, value in self._images.items():
      self._writer.add_image(name, value, step)
    for name, value in self._videos.items():
      name = name if isinstance(name, str) else name.decode("utf-8")
      if np.issubdtype(value.dtype, np.floating):
        value = np.clip(255 * value, 0, 255).astype(np.uint8)
      B, T, H, W, C = value.shape
      value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
      self._writer.add_video(name, value, step, 16)
    for name, value in self._histograms.items():
      self._writer.add_histogram(name, value, step)

    self._writer.flush()
    self._scalars = {}
    self._images = {}
    self._videos = {}

  def _compute_fps(self, step):
    if self._last_step is None:
      self._last_time = time.time()
      self._last_step = step
      return 0
    steps = step - self._last_step
    duration = time.time() - self._last_time
    self._last_time += duration
    self._last_step = step
    return steps / duration

  def log_hydra_config(self, config, name="config", step=0, log_hparams=False, hparams_run_name="."):
    """
    Log a Hydra/OmegaConf config to TensorBoard:
      - as YAML text under "{name}/yaml"
      - as flattened hparams to the HParams plugin
    """
    # 1) Log YAML to Text plugin
    yaml_str = None
    try:
      from omegaconf import (
        OmegaConf,  # local import to avoid hard dependency at module import
      )

      yaml_str = OmegaConf.to_yaml(config, resolve=True)
    except ImportError:
      # Fallback to string representation
      yaml_str = str(config)
    self._writer.add_text(f"{name}/yaml", f"```yaml\n{yaml_str}\n```", step)

    # 2) Log flattened hparams to HParams plugin
    flat = {}
    container = None
    try:
      from omegaconf import OmegaConf  # local import again

      container = OmegaConf.to_container(config, resolve=True)
    except Exception:
      container = None

    if log_hparams and container is not None:

      def _flatten(prefix, obj):
        if isinstance(obj, dict):
          for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (list, tuple)):
          flat[prefix] = str(obj)
        elif isinstance(obj, (int, float, bool, str)) or obj is None:
          flat[prefix] = obj if obj is not None else "null"
        else:
          flat[prefix] = str(obj)

      _flatten("", container)
      # add_hparams requires a non-empty metrics dict
      with contextlib.suppress(TypeError):
        # Avoid creating a timestamped subdirectory by specifying run_name (PyTorch >= 1.14)
        self._writer.add_hparams(flat, {"_": 0}, run_name=hparams_run_name)

