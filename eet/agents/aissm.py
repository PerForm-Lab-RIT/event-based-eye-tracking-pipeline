import functools
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from calflops import calculate_flops
from omegaconf import DictConfig as Config

from ..data.processors.event_transform import build_event_transform
from ..networks import ConvEncoder2, MLP
from ..networks.utils import functional as F2
from ..utils import Space, print, timer
from .base import BaseAgent
from ..networks.ssm import AISSM


class AISSMAgent(BaseAgent):
  def __init__(self, config: Config, input_space: Space, label_space: Space):
    super().__init__(config, input_space, label_space)

    c, h, w = input_space.shape[-3], input_space.shape[-2], input_space.shape[-1]
    self._height = int(h)
    self._width = int(w)
    self.register_buffer("x_coords", torch.arange(self._width, dtype=torch.float32))
    self.register_buffer("y_coords", torch.arange(self._height, dtype=torch.float32))
    # scale to convert normalized predictions to pixel space for loss calculation and visualization
    label_scale = torch.as_tensor([float(self._height - 1), float(self._width - 1)])
    self.register_buffer("label_scale", label_scale)

    self.encoder = ConvEncoder2(input_shape=(c, h, w), **config.aissm.encoder).to(self.device)
    self.dynamic = AISSM(config.aissm.dynamic, embed_size=self.encoder.out_dim).to(self.device)
    self.mlp = MLP(inp_dim=self.dynamic.out_dim, **config.aissm.mlp).to(self.device)
    self.readout = nn.Sequential(
      nn.Linear(self.mlp.out_dim, config.aissm.head.units),
      nn.ReLU(),
      nn.Dropout(config.aissm.head.dropout),
      nn.Linear(config.aissm.head.units, 2)
    ).to(self.device)

    self.all_modules.update({
      "encoder": self.encoder,
      "dynamic": self.dynamic,
      "mlp": self.mlp,
      "readout": self.readout,
    })

  @functools.cached_property
  def flops(self) -> float:
    sample_inputs = torch.zeros(1, self.config.batch_length, *self.input_space.shape, device=self.device)
    try:
      flop_count, _, _ = calculate_flops(
        model=self,
        args=[sample_inputs],
        output_as_string=False,
        output_precision=6,
        print_results=False,
        print_detailed=False,
      )
    except Exception as error:
      print(f"FLOPs calculation failed with error: {error}", color="red")
      flop_count = 0.0
    return flop_count

  @property
  def flops_per_frame(self) -> float:
    """Already amortised: the encoder ran T/S times to produce T outputs, so dividing the clip total by T is the honest per-output-frame figure. There is no separate 'deployed' number..."""
    return self.flops / max(int(self.config.batch_length), 1)

  def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # print(f"input: {inputs.mean()}, {inputs.std()}, {inputs.min()}, {inputs.max()}")

    # inputs: (B, T, C, H, W)
    embed = self.encoder(inputs) # (B, T, D)
    initial = self.dynamic.initial(embed.shape[0]) # (B, D), (B, S, K)
    deters, post_stochs, post_logits, prior_stochs, prior_logits = self.dynamic.observe(embed, initial) # (B, T, S, K), (B, T, D), (B, T, S, K)

    # alpha
    alphalogit = self.dynamic.alpha(embed) # (B, T)
    alpha = torch.sigmoid(alphalogit) # (B, T)
    _alpha = alpha.unsqueeze(-1).unsqueeze(-1) # (B, T, 1, 1)
    _alpha = _alpha.expand_as(post_stochs) # (B, T, S, K)
    post_stochs = _alpha * post_stochs + (1 - _alpha) * prior_stochs # (B, T, S, K)

    # forward
    features = self.dynamic.get_feat(post_stochs, deters) # (B, T, F)
    mlp_out = self.mlp(features) # (B, T, MLP_out_dim)
    logits = self.readout(mlp_out) # (B, T, 2)
    pred = torch.sigmoid(logits) # (B, T, 2)
    # (B, T, width), (B, T, height), ((B, T, D), (B, T, S, K))
    # return x_logits, y_logits, alphalogit
    return pred, alphalogit

  def decode_outputs(self, outputs):
    return F2.to_pixel_space(outputs[0], self.label_scale)

  def report(self, inputs: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
      _, metrics, predictions_px = self.loss(inputs, label)
      video = F2.build_pred_video(inputs, label, predictions_px)
      metrics["video"] = video
      return metrics

  def loss(self, inputs: torch.Tensor, label: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any]:
    # inputs: (B, T, C, H, W), label: (B, T, 2)
    label = label.long()
    with timer.section("forward"):
      # x_logits, y_logits, alphalogit = self.forward(inputs)
      # x_logits, y_logits = self.forward(inputs)
      pred, alphalogit = self.forward(inputs)

    # loss alpha
    target_alpha = self._compute_confidence(inputs, label) # (B, T)
    alpha_loss = F.smooth_l1_loss(torch.sigmoid(alphalogit), target_alpha, reduction="none").reshape(-1) # (BT,)

    # (BT, width), (BT, height)

    # predictions_px = self._decode_predictions(x_logits, y_logits) # (B, T, 2)
    # label is [height, width] for the last dimension
    # predictions_px = pred * torch.tensor([self._height - 1, self._width -1], device=pred.device) # (BT, 2)
    predictions_px = F2.to_pixel_space(pred, self.label_scale)
    coord_loss = F.smooth_l1_loss(predictions_px, label.float(), reduction="none").reshape(-1, 2).sum(-1) # (BT,)
    loss = (coord_loss + alpha_loss).mean()

    diff = predictions_px - label.float()
    mse = torch.mean(diff * diff)
    mae = diff.abs().mean()
    rmse = torch.sqrt(mse)

    metrics: Dict[str, torch.Tensor] = {
      "loss": loss,
      "loss/coord": coord_loss.mean(),
      "loss/alpha": alpha_loss.mean(),  # ADD
      "mse": mse,
      "mae": mae,
      "rmse": rmse,
    }

    return loss, {key: value.detach() for key, value in metrics.items()}, predictions_px.detach()

  def _compute_confidence(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Compute signal quality metric for RAW (un-normalized) event frames.

    For event-based eye tracking, this function computes a signal quality score that
    serves as the alpha parameter for the AIS2M dynamics model. This function operates
    on RAW event data BEFORE z-score normalization, allowing us to use actual event
    magnitudes and traditional signal-to-noise concepts.

    Args:
      inputs (torch.Tensor): (B, T, C, H, W) RAW (un-normalized) event frames
      label (torch.Tensor): (B, T, 2) pupil center coordinates in [y, x] format

    Returns:
      torch.Tensor: Signal quality scores with shape (B, T)
        - Values in range [0, 1] where:
          * 0 = poor signal quality (low event density ratio)
          * 1 = excellent signal quality (high event density ratio)
          * default_alpha used for frames with invalid labels
    """
    # Config extraction
    conf = self.config.aissm.confidence
    roi_w, roi_h = conf.roi_size
    event_threshold = conf.event_threshold
    coverage_threshold = conf.event_coverage_ratio_threshold
    quality_w = conf.weights.quality
    coverage_w = conf.weights.coverage
    weight_sum = max(quality_w + coverage_w, 1e-6)

    B, T, C, H, W = inputs.shape
    frame = inputs.abs()
    label = label.float()

    # ROI bounds and mask generation
    # (B, T)
    y: torch.Tensor = label[..., 0].clamp(0, H - 1).long()
    x: torch.Tensor = label[..., 1].clamp(0, W - 1).long()
    half_h: int = roi_h // 2
    half_w: int = roi_w // 2
    y_min: torch.Tensor = (y - half_h).clamp(0, H - 1)
    y_max: torch.Tensor = (y + half_h).clamp(0, H - 1)
    x_min: torch.Tensor = (x - half_w).clamp(0, W - 1)
    x_max: torch.Tensor = (x + half_w).clamp(0, W - 1)
    # (1, 1, H, 1)  (1, 1, 1, W)
    y_grid: torch.Tensor = torch.arange(H, device=inputs.device).view(1, 1, H, 1)
    x_grid: torch.Tensor = torch.arange(W, device=inputs.device).view(1, 1, 1, W)
    signal_mask = (
      (y_grid >= y_min.unsqueeze(-1).unsqueeze(-1)) &
      (y_grid <= y_max.unsqueeze(-1).unsqueeze(-1)) &
      (x_grid >= x_min.unsqueeze(-1).unsqueeze(-1)) &
      (x_grid <= x_max.unsqueeze(-1).unsqueeze(-1))
    ).unsqueeze(2).expand(B, T, C, H, W) # (B, T, C, H, W)

    # counting
    event_mask = frame > event_threshold
    signal_event_count = (event_mask & signal_mask).float().sum(dim=(2, 3, 4)) # (B, T)
    noise_event_count = (event_mask & (~signal_mask)).float().sum(dim=(2, 3, 4)) # (B, T)

    # SNR (B, T)
    signal_quality = torch.sigmoid(signal_event_count / (noise_event_count + 1e-6))
    # event density (B, T)
    signal_area = signal_mask.float().sum(dim=(2, 3, 4)).clamp_min(1.0)
    signal_coverage = (signal_event_count / signal_area / coverage_threshold).clamp(0.0, 1.0)

    confidence = (quality_w * signal_quality + coverage_w * signal_coverage) / weight_sum
    confidence = confidence.clamp(0.0, 1.0)
    return confidence





