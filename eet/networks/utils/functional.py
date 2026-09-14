"""
File: functional.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Include some functional operations
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import math
import torch
from torch.nn import functional as F

from ..utils import tree

def to_pixel_space(predictions: torch.Tensor, label_scale: torch.Tensor) -> torch.Tensor:
  # Scale the prediction from normalized space [0, 1] to pixel space [0, height-1] and [0, width-1]
  # predictions: (..., 2)
  # label scale: (..., 2)
  scale = label_scale.to(device=predictions.device, dtype=predictions.dtype)
  return predictions * scale

# convert tensor (..., 2) with [height, width] in the last dimension to to 320x320 space
#to_320x320 = lambda x, h, w: x * torch.tensor([319.0 / (h - 1), 319.0 / (w - 1)], device=x.device)
# Convert tensor (..., 2) with [y, x] coordinates to true 320x240 space.
# y-axis target range: 0 to 239
# x-axis target range: 0 to 319
to_320x240 = lambda x, h, w: x * torch.tensor(
  [239.0 / (h - 1), 319.0 / (w - 1)],
  device=x.device,
  dtype=x.dtype,
)

def build_pred_video(
  inputs: torch.Tensor,
  label: torch.Tensor,
  predictions_px: torch.Tensor,
  max_batch: int = 6,
) -> torch.Tensor | None:
  """Build a visualization video with GT and prediction crosshairs.

  Args:
      inputs (torch.Tensor): Input tensor of shape (B, T, C, H, W)
      label (torch.Tensor): Label tensor of shape (B, T, 2) in the pixel space (not normalized space)
      predictions_px (torch.Tensor): Predicted tensor in pixel space of shape (B, T, 2) in the pixel space (not normalized space)
      max_batch (int, optional): Maximum number of batches to visualize. Defaults to 6.

  Returns:
      torch.Tensor | None: Visualization video tensor of shape (RB, T, H, W, 3) or None if invalid input.
  """
  if inputs.ndim != 5 or label.ndim != 3 or predictions_px.ndim != 3:
    return None
  if inputs.shape[-3] <= 0 or label.shape[-1] < 2 or predictions_px.shape[-1] < 2:
    return None

  rb = min(int(max_batch), int(inputs.shape[0]))
  if rb <= 0:
    return None

  frames = inputs[:rb]  # (RB, T, C, H, W)
  label_px = torch.round(label[:rb, :, :2]).to(torch.int64)
  pred_px = torch.round(predictions_px[:rb, :, :2]).to(torch.int64)

  # Use channel-sum for robust grayscale rendering across event modalities.
  render = frames.to(torch.float32).sum(dim=2, keepdim=True)  # (RB, T, 1, H, W)
  frame_min = render.amin(dim=(-2, -1), keepdim=True)
  frame_max = render.amax(dim=(-2, -1), keepdim=True)
  render = (render - frame_min) / (frame_max - frame_min).clamp_min(1e-8)
  render = (255.0 * render).clamp(0, 255).to(torch.uint8)
  render = render.repeat(1, 1, 3, 1, 1).permute(0, 1, 3, 4, 2).contiguous()  # (RB, T, H, W, 3)

  video = draw_cross_hair_with_dot(render, label_px[..., 0], label_px[..., 1], color=ORANGE)
  video = draw_cross_hair_with_dot(video, pred_px[..., 0], pred_px[..., 1], color=BLUE)

  video = F.pad(video.permute(0, 1, 4, 2, 3), (2, 2, 2, 2)).permute(0, 1, 3, 4, 2).contiguous()
  green = torch.tensor(GREEN, dtype=video.dtype, device=video.device).view(1, 1, 1, 1, 3)
  video[:, :, :2, :, :] = green
  video[:, :, -2:, :, :] = green
  video[:, :, :, :2, :] = green
  video[:, :, :, -2:, :] = green

  blank_tail = torch.zeros((video.shape[0], 10, video.shape[2], video.shape[3], 3), dtype=video.dtype, device=video.device)
  video = torch.cat([video, blank_tail], dim=1)
  return video


def symlog(x):
  return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
  return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def get_act(name: str):
  if callable(name):
    return name
  elif name == 'none' or name == 'identity' or name == None:
    return lambda x: x
  elif name == 'relu':
    return F.relu
  elif name == 'gelu_tanh':
    return gelu_tanh
  elif name == 'gelu_quick':
    return lambda x: x * torch.sigmoid(1.702 * x)
  elif name == 'relu2':
    return lambda x: torch.square(F.relu(x))
  elif name == 'swiglu':
    def fn(x):
      x, y = torch.split(x, 2, -1)
      return F.silu(x) * y
    return fn
  elif name == 'mish':
    return lambda x: x * torch.tanh(F.softplus(x))
  elif hasattr(F, name):
    return getattr(F, name)
  else:
    raise NotImplementedError(name)

def get_act_module_class(name: str) -> callable:
  if callable(name):
    return name
  elif name == 'none' or name == 'identity' or name == None:
    return torch.nn.Identity
  elif name == 'relu':
    return torch.nn.ReLU
  elif name == 'silu':
    return torch.nn.SiLU
  elif name == 'gelu':
    return torch.nn.GELU
  elif name == 'leaky_relu':
    return torch.nn.LeakyReLU
  elif hasattr(torch.nn, name):
    return getattr(torch.nn, name)
  else:
    raise NotImplementedError(name)


def gelu_tanh(x):
  # Constants used in the approximation
  sqrt_2_over_pi = (2.0 / torch.pi)**0.2
  coeff = 0.044715
  # GELU approximation formula
  return 0.5 * x * (1 + torch.tanh(sqrt_2_over_pi * (x + coeff * torch.pow(x, 3))))


def where(condition: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor):
  """

  Args:
      condition (Tree): any tree with shape (*some_shape,)
      xs (Tree): any tree with shape (*some_shape, *dim)
      ys (Tree): any tree with shape (*some_shape, *dim)

  Returns:
      Tree: same shape as xs and ys. Resulted masked value with mask broadcasted to the shape of xs and ys
  """
  assert condition.dtype == torch.bool, condition.dtype
  def fn(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape, (x.shape, y.shape)
    expanded = condition.view(*condition.shape, *([1] * (x.ndim - condition.ndim)))
    return torch.where(expanded, x, y)
  return tree.map(fn, xs, ys)


def mask(xs, mask):
  """Resulted masked value with mask broadcasted to the shape of xs
    negative mask values will become zeros.

  Args:
      xs (Tree): any tree with shape (*some_shape, *dim)
      mask (torch.Tensor): (*some_shape,)

  Returns:
      Tree: same shape as xs. Resulted masked value with mask broadcasted to the shape of xs
        negative mask values will become zeros.
  """
  return where(mask, xs, tree.map(torch.zeros_like, xs))



def sinusoidal(d_model: int, shape: Tuple[int, ...]):
  """
  Computes sinusoidal positional embeddings in PyTorch.

  Args:
    d_model (int): The embedding dimension.
    shape (Tuple[int, ...]): The event shape, which multiplies to the sequence length.

  Returns:
    torch.Tensor: A tensor of shape (*shape, d_model) containing the positional encodings.
  """
  assert len(shape) > 0, "Shape must be non-empty"
  assert d_model % 2 == 0, f"Hidden dim must be divisible by 2, got d_model = {d_model}"
  T = math.prod(shape)  # Compute total sequence length
  position = torch.arange(T, dtype=torch.float32).unsqueeze(1)  # Shape: (T, 1)
  div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
  # Compute sin and cos values separately
  sin_values = torch.sin(position * div_term)  # Shape: (T, d_model/2)
  cos_values = torch.cos(position * div_term)  # Shape: (T, d_model/2)
  # Interleave sin and cos values correctly
  pe = torch.cat([sin_values, cos_values], dim=-1).reshape(T, d_model)
  return pe.reshape((*shape, d_model))  # Shape: (*shape, d_model)


BLUE = (0, 114, 178)
ORANGE = (230, 159, 0)
GREEN = (0, 255, 0)

def draw_cross_hair_with_dot(
	image: torch.Tensor,
	cy: torch.Tensor,
	cx: torch.Tensor,
	color: tuple[int, int, int],
	length: int = 5,
	thickness: int = 2,
	gap: int = 2,
	dot_radius: int = 1,
) -> torch.Tensor:
	"""Draw an X-shaped crosshair with a center dot on batched video frames."""
	assert image.ndim == 5 and image.shape[-1] == 3, image.shape
	assert image.dtype == torch.uint8, image.dtype

	_, _, h, w, _ = image.shape
	device = image.device

	yy = torch.arange(h, device=device).view(1, 1, h, 1)
	xx = torch.arange(w, device=device).view(1, 1, 1, w)
	cy = cy.clamp(0, h - 1).to(torch.int64).unsqueeze(-1).unsqueeze(-1)
	cx = cx.clamp(0, w - 1).to(torch.int64).unsqueeze(-1).unsqueeze(-1)

	dy = yy - cy
	dx = xx - cx
	diag1 = (
		(torch.abs(dy - dx) <= thickness // 2)
		& (torch.abs(dx) <= length)
		& (torch.abs(dy) <= length)
		& ((torch.abs(dx) >= gap) | (torch.abs(dy) >= gap))
	)
	diag2 = (
		(torch.abs(dy + dx) <= thickness // 2)
		& (torch.abs(dx) <= length)
		& (torch.abs(dy) <= length)
		& ((torch.abs(dx) >= gap) | (torch.abs(dy) >= gap))
	)
	dot = (dy * dy + dx * dx) <= int(dot_radius) ** 2
	mask = (diag1 | diag2 | dot).unsqueeze(-1)

	color_t = torch.tensor(color, dtype=image.dtype, device=device).view(1, 1, 1, 1, 3)
	return torch.where(mask, color_t, image)

