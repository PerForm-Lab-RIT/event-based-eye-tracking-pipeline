
from typing import List, Union, Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F

from .utils import functional as T


class Conv2dSamePad(nn.Conv2d):
  """A Conv2d layer that emulates TensorFlow's 'SAME' padding."""

  def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
    i_div_s_ceil = (i + s - 1) // s
    return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    ih, iw = x.size()[-2:]
    pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
    pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])

    if pad_h > 0 or pad_w > 0:
      x = F.pad(
        x,
        [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
      )

    return F.conv2d(
      x,
      self.weight,
      self.bias,
      self.stride,
      self.padding,
      self.dilation,
      self.groups,
    )

# https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
  def __init__(self, units: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(units))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    return output


class RMSNorm2D(nn.RMSNorm):
  """RMSNorm over channel-last format applied to 4D tensors."""

  def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
    super().__init__(ch, eps=eps, dtype=dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Apply RMSNorm over the channel dimension.
    return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvEncoder(nn.Module):
  def __init__(self, input_shape: tuple, act: str = 'SiLU', depth: int = 32,
        kernel_size: int = 5, mults: list = [2, 3, 4, 4], norm: bool = True):
    super().__init__()
    act = getattr(torch.nn, act)
    input_ch, h, w = input_shape
    self.depths = tuple(int(depth) * int(mult) for mult in list(mults))
    self.kernel_size = int(kernel_size)
    in_dim = input_ch
    layers = []
    for i, depth in enumerate(self.depths):
      layers.append(
        Conv2dSamePad(
          in_channels=in_dim,
          out_channels=depth,
          kernel_size=self.kernel_size,
          stride=1,
          bias=True,
        )
      )
      layers.append(nn.MaxPool2d(2, 2))
      if norm:
        layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
      layers.append(act())
      in_dim = depth
      h, w = h // 2, w // 2

    self.out_dim = self.depths[-1] * h * w
    self.layers = nn.Sequential(*layers)

  def forward(self, obs):
    # Input shape: (B, T, C, H, W)
    """Encode image-like observations with a CNN."""
    # (B*T, C, H, W)
    x = obs.reshape(-1, *obs.shape[-3:])
    # (B*T, C_feat, H_feat, W_feat)
    x = self.layers(x)
    # (B*T, C_feat*H_feat*W_feat)
    x = x.reshape(x.shape[0], -1)
    # (B, T, C_feat*H_feat*W_feat)
    return x.reshape(*obs.shape[:-3], x.shape[-1])


class ConvEncoder2(nn.Module):
  def __init__(self, input_shape: tuple, pool_size: int, kernels: List[int],
      channels: List[int], temporal_kernel: int, dropout: float):
    super().__init__()
    if len(kernels) != len(channels):
      raise ValueError("`kernels` and `channels` must have the same length.")

    self.spatial_blocks = nn.ModuleList()
    self.temporal_blocks = nn.ModuleList()
    self.pool_size = pool_size
    in_ch, pooled_height, pooled_width = input_shape
    last_idx = len(channels) - 1

    for i, (kernel, out_ch) in enumerate(zip(kernels, channels)):
      self.spatial_blocks.append(nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=(1, kernel, kernel), padding=(0, kernel // 2, kernel // 2)),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(),
      ))

      temporal_layers = [
        nn.Conv3d(out_ch, out_ch, kernel_size=(temporal_kernel, 3, 3), padding=(temporal_kernel // 2, 1, 1), bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(),
      ]
      if i < last_idx:
        temporal_layers.append(nn.AvgPool3d((1, 3, 3), stride=(1, 3, 3)))
        pooled_width = max(1, (pooled_width - 3) // 3 + 1)
        pooled_height = max(1, (pooled_height - 3) // 3 + 1)
      else:
        temporal_layers.append(nn.Dropout3d(dropout))

      self.temporal_blocks.append(nn.Sequential(*temporal_layers))
      in_ch = out_ch

    self.out_dim = channels[-1] * self.pool_size * self.pool_size
    self.layer_norm = nn.LayerNorm(self.out_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    batch_size = x.shape[0]
    # x: (B, T, C, H, W) -> (B, C, T, H, W)
    x = x.permute(0, 2, 1, 3, 4)
    for i, (spatial_block, temporal_block) in enumerate(zip(self.spatial_blocks, self.temporal_blocks)):
      x = spatial_block(x)
      x = temporal_block(x)
      # print(f"After block {i}: {x.shape}")
    x = F.adaptive_avg_pool3d(x, (seq_len, self.pool_size, self.pool_size))
    # print(f"After pooling: {x.shape}")
    # (B, C_feat, T_feat, H_feat, W_feat) -> (B, T_feat, C_feat*H_feat*W_feat)
    flatten = x.permute(0, 2, 1, 3, 4).reshape(batch_size, seq_len, -1)
    # print(f"After flattening: {flatten.shape}")
    # dropout
    if self.training:
      keep_prob = 0.5
      mask = flatten.new_empty(flatten.shape[0], 1, flatten.shape[-1]).bernoulli_(keep_prob)
      embed = flatten * mask.div_(keep_prob)
    else:
      embed = flatten
    # print(f"After dropout: {embed.shape}")
    embed = self.layer_norm(embed)
    # print(f"After layer norm: {embed.shape}")
    return embed


class MLP(nn.Module):
  def __init__(
    self,
    inp_dim: int,
    act: str = 'SiLU',
    symlog_inputs: bool = False,
    units: int = 256,
    layers: int = 1
  ):
    super().__init__()
    act = getattr(torch.nn, act)
    self._symlog_inputs = bool(symlog_inputs)
    self.layers = nn.Sequential()
    for i in range(layers):
      self.layers.add_module(f"linear{i}", nn.Linear(inp_dim, units, bias=True))
      # self.layers.add_module(f"norm{i}", nn.RMSNorm(units, eps=1e-04, dtype=torch.float32))
      self.layers.add_module(f"act{i}", act())
      inp_dim = units
    self.out_dim = units

  def forward(self, x):
    # (B, T, I)
    if self._symlog_inputs:
      x = T.symlog(x)
    # (B, T, U)
    return self.layers(x)

class GRU(nn.Module):
  def __init__(self, in_dim: int, out_dim: int) -> None:
    super().__init__()
    self._in_unit = in_dim
    self._out_unit = out_dim
    self.out_dim = out_dim
    # self.act = getattr(torch.nn.functional, act)
    # self.norm_layer = nn.LayerNorm(self._out_unit + self._in_unit, eps=1e-03) if norm else None
    self.norm_layer = nn.RMSNorm(self._out_unit + self._in_unit, eps=1e-04, dtype=torch.float32)

    # Create linear layer with custom initialization
    self.linear = nn.Linear(self._in_unit + self._out_unit, 3 * self._out_unit, bias=True)

    # a trick to have a device-aware module without passing device around
    self.register_buffer('_dummy', torch.zeros(0))

  def initial(self, batch_size: int) -> torch.Tensor:
    return torch.zeros(batch_size, self._out_unit, device=self._dummy.device)

  def forward(self, carry: torch.Tensor, inputs: torch.Tensor,
      resets: torch.Tensor = None, single: bool = True):
    """
    Args:
      carry (torch.Tensor): (B, U) - hidden state
      inputs (torch.Tensor): (B, I) or (B, T, I) - inputs
      resets (torch.Tensor): (B,) or (B, T) - boolean reset mask
      single (bool, optional): If True, process single timestep. Defaults to True.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: (carry, outputs)
        - carry: (B, U) - final hidden state
        - outputs: (B, U) if single=True, else (B, T, U) - all outputs
    """
    assert carry.dtype in (torch.float32, torch.float16, torch.bfloat16), carry.dtype
    assert inputs.dtype in (torch.float32, torch.float16, torch.bfloat16), inputs.dtype
    if resets is not None:
      assert resets.dtype == torch.bool, resets.dtype

    if single:
      carry = self.step(carry, inputs, resets)
      return carry

    # Scan over time dimension (similar to jax.lax.scan)
    outputs = []
    T = inputs.shape[1]
    for t in range(T):
      carry = self.step(carry, inputs[:, t], resets[:, t] if resets is not None else None)
      outputs.append(carry)

    outputs = torch.stack(outputs, dim=1)  # (B, T, U)
    return outputs

  def step(self, carry: torch.Tensor, inp: torch.Tensor, reset: torch.Tensor = None):
    """Single timestep update.

    Args:
        carry (torch.Tensor): (B, U)
        inp (torch.Tensor): (B, I)
        reset (torch.Tensor): (B,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (new_carry, output)
    """
    # Mask carry based on reset (zero out when reset is True)
    if reset is not None:
      carry = T.mask(carry, ~reset)

    # Concatenate carry and input
    x = torch.cat([carry, inp], dim=-1)  # (B, U+I)

    # Apply norm if exists
    if self.norm_layer is not None:
      x = self.norm_layer(x)

    # Linear projection
    x = self.linear(x)  # (B, 3*U)

    # Split into three parts
    res, cand, update = torch.split(x, self._out_unit, dim=-1)

    # GRU equations
    cand = torch.tanh(torch.sigmoid(res) * cand)
    update = torch.sigmoid(update - 1) # Bias towards carry
    carry = update * cand + (1 - update) * carry

    return carry


class LambdaLayer(nn.Module):
  """Wrap an arbitrary callable into an ``nn.Module``."""

  def __init__(self, lambd):
    super().__init__()
    self.lambd = lambd

  def forward(self, x):
    return self.lambd(x)


class BlockLinear(nn.Module):
  """Block-wise linear layer.

  Weight layout is chosen to cooperate with PyTorch's fan-in/fan-out
  calculation used by initializers.
  """

  def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0):
    super().__init__()
    self.in_ch = int(in_ch)
    self.out_ch = int(out_ch)
    self.blocks = int(blocks)
    self.outscale = float(outscale)

    # Store weight in a layout that works with torch's fan calculation.
    # (O/G, I/G, G)
    self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
    self.bias = nn.Parameter(torch.empty(self.out_ch))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # (..., I)
    batch_shape = x.shape[:-1]
    # Reshape to expose block dimension.
    # (..., I) -> (..., G, I/G)
    x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)

    # Block-wise multiplication
    # (..., G, I/G), (O/G, I/G, G) -> (..., G, O/G)
    x = torch.einsum("...gi,oig->...go", x, self.weight)
    # Merge block dimension back.
    # (..., G, O/G) -> (..., O)
    x = x.reshape(*batch_shape, self.out_ch)
    return x + self.bias



