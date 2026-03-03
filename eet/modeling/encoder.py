"""
File: encoder.py
Author: Viet Nguyen
Date: 2025-06-26

Description: Contain all types of encoder
"""

import jax
import jax.numpy as jnp
import numpy as np
from lib import nn
import functools

from lib import Space, print
from .cbconvlstm import CBConvLSTM

class Encoder(nn.Module):
  """
  A flexible neural network encoder for processing multi-modal observations.

  This encoder can handle both vector and image-based inputs, applying different
  processing strategies based on input type. It supports:
  - Vector inputs processed through multi-layer perceptron (MLP)
  - Image inputs processed through convolutional neural network (CNN)
  - Optional symlog transformation for continuous inputs
  - Configurable network depth, units, normalization, and activation

  Example config:
    # for 480x640 input
    **{depth: 16, mults: [2, 3, 3, 4, 4, 4, 4], layers: 1, units: 16, act: silu, norm: layer, winit: trunc_normal_in, symlog: True, outer: False, kernel: 5, strided: True}
  => 4x5 output
  """

  units: int = 64  # Number of units in the MLP layers
  norm: str = 'layer'  # Normalization type (Root Mean Square)
  act: str = 'silu'  # Activation function (Gaussian Error Linear Unit)
  depth: int = 16  # Base depth for convolutional layers
  mults: tuple = (2, 3, 3, 4, 4, 4, 4)  # Multipliers for increasing depth in CNN layers
  layers: int = 1  # Number of MLP layers
  kernel: int = 5  # Kernel size for convolutional layers
  symlog: bool = True  # Whether to apply symlog transformation to continuous inputs
  outer: bool = False  # Special handling for first CNN layer
  strided: bool = True  # Whether to use strided convolutions

  def __init__(self, obs_space, **kw):
    """
    Initialize the encoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining input characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  def __call__(self, obs, reset, training, single=False):
    """
    Process multi-modal observations through vector and image encoders.

    Args:
      obs (dict): Input observations dictionary
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag
      single (bool, optional): Whether processing a single timestep. Defaults to False.

    Returns:
      embed (jax.Array): Encoded representation of input observations
    """
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.functional.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      # NOTE: TODO: This is weird, need to be fixed.
      #   For now, we just dividing by some number for it to be lowered down.
      #   In the future, we have to look deeper into this.
      # assert all(x.dtype == jnp.uint8 for x in imgs)
      # x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = nn.cast(jnp.concatenate(imgs, -1), force=True)
      # print(f"[encoder] x: {x}")
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.get_act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    embed = x.reshape((*bshape, *x.shape[1:]))
    return embed

  def macs(self, obs, reset):
    # Estimate MACs for the encoder's forward pass.
    total = 0

    # Determine batch/time dims from reset
    rshape = getattr(reset, 'shape', None)
    if rshape is None:
      return 0
    if len(rshape) == 2:
      B, T = int(rshape[0]), int(rshape[1])
    else:
      B, T = int(rshape[0]), 1

    bdims = 1 if T == 1 else 2

    # --- Vector (MLP) branch ---
    if self.veckeys:
      # compute input dim
      input_dim = 0
      for k in self.veckeys:
        a = obs[k]
        input_dim += int(a.shape[-1])
      in_ch = input_dim
      batch_for_linear = B * T if T > 1 else B
      for i in range(self.layers):
        # Linear
        try:
          linear = nn.Linear(int(self.units))
          total += int(linear.macs(jnp.zeros((batch_for_linear, in_ch))))
        except Exception:
          total += int(batch_for_linear * in_ch * int(self.units))
        # Norm
        try:
          total += int(nn.Norm(self.norm).macs(jnp.zeros((batch_for_linear, int(self.units)))))
        except Exception:
          total += int(batch_for_linear * int(self.units) * 6)
        # Activation (~4 ops per element)
        total += int(batch_for_linear * int(self.units) * 4)
        in_ch = int(self.units)

    # --- Image (CNN) branch ---
    if self.imgkeys:
      img0 = obs[sorted(self.imgkeys)[0]]
      if img0.ndim == 5:
        _, maybe_T, H, W, C0 = img0.shape
      else:
        maybe_T = 1
        _, H, W, C0 = img0.shape
      # concatenated channels across imgkeys
      in_ch = 0
      for k in sorted(self.imgkeys):
        a = obs[k]
        in_ch += int(a.shape[-1])

      # kernel area
      if isinstance(self.kernel, (tuple, list)):
        k_h, k_w = self.kernel
      else:
        k_h = k_w = int(self.kernel)
      k_area = int(k_h) * int(k_w)

      cur_H, cur_W = int(H), int(W)
      for i, depth in enumerate(self.depths):
        out_ch = int(depth)
        # stride as in __call__
        if self.outer and i == 0:
          stride = 1
        elif self.strided:
          stride = 2
        else:
          stride = 1

        batch_for_conv = B * T if T > 1 else B
        dummy_inp = jnp.zeros((batch_for_conv, cur_H, cur_W, in_ch))
        try:
          conv = nn.Conv2D(out_ch, (k_h, k_w), stride)
          total += int(conv.macs(dummy_inp))
        except Exception:
          out_H = max(1, cur_H // stride) if stride > 1 else cur_H
          out_W = max(1, cur_W // stride) if stride > 1 else cur_W
          total += int(B * T * out_H * out_W * out_ch * k_area * in_ch)

        # compute output spatial dims
        if self.outer and i == 0:
          out_H, out_W = cur_H, cur_W
        else:
          out_H, out_W = max(1, cur_H // 2), max(1, cur_W // 2)

        # norm + activation
        total += int(B * T * out_H * out_W * out_ch * 4)
        total += int(B * T * out_H * out_W * out_ch * 4)

        in_ch = out_ch
        cur_H, cur_W = out_H, out_W

    return int(total)


class CBConvLSTMEncoder(nn.Module):
  """
  A flexible neural network encoder for processing multi-modal observations.

  This encoder can handle both vector and image-based inputs, applying different
  processing strategies based on input type. It supports:
  - Vector inputs processed through multi-layer perceptron (MLP)
  - Image inputs processed through convolutional neural network (CNN)
  - Optional symlog transformation for continuous inputs
  - Configurable network depth, units, normalization, and activation

  Example config:
    # for 480x640 input
    **{depth: 16, mults: [2, 3, 3, 4, 4, 4, 4], layers: 1, units: 16, act: silu, norm: layer, winit: trunc_normal_in, symlog: True, outer: False, kernel: 5, strided: True}
  => 4x5 output
  """

  units: int = 64  # Number of units in the MLP layers
  norm: str = 'layer'  # Normalization type (Root Mean Square)
  act: str = 'silu'  # Activation function (Gaussian Error Linear Unit)
  depth: int = 16  # Base depth for convolutional layers
  mults: tuple = (2, 3, 3, 4, 4, 4, 4)  # Multipliers for increasing depth in CNN layers
  layers: int = 1  # Number of MLP layers
  kernel: int = 5  # Kernel size for convolutional layers
  symlog: bool = True  # Whether to apply symlog transformation to continuous inputs
  outer: bool = False  # Special handling for first CNN layer
  strided: bool = True  # Whether to use strided convolutions

  def __init__(self, obs_space, **kw):
    """
    Initialize the encoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining input characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw
    self.corekw = {k: v for k, v in kw.items() if k in ('bias', 'winit', 'binit', 'norm', 'update_bias')}
    self.cores = [CBConvLSTM(depth, self.kernel, **self.corekw, name=f"core{i}") for i, depth in enumerate(self.depths) if i != 0]

  def initial(self, obs: dict, single=True):
    carries = []
    if self.imgkeys:
      if single:
        B, H, W, C = obs[self.imgkeys[0]].shape
        T = 1
      else:
        B, T, H, W, C = obs[self.imgkeys[0]].shape
      bdims = 1 if single else 2
      bshape = (B,) if single else (B, T)
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      x = nn.cast(jnp.concatenate(imgs, -1), force=True)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if i != 0:
          _carry = self.cores[i-1].initial(B, (x.shape[-2], x.shape[-3]), input_size=x.shape[-1])
          # print(f"[CBConvLSTMEncoder.initial] _carry: {jax.tree.map(lambda x: x.shape, _carry)}", color="yellow")
          carries.append(_carry)
          x = x.reshape((*bshape, *x.shape[1:]))
          _, x = self.cores[i-1](_carry, x, jnp.zeros(bshape).astype(jnp.bool), single=single)
          x = x.reshape((-1, *x.shape[bdims:]))
          # print(f"[CBConvLSTMEncoder.initial] currinp.shape: {currinp.shape}", color="red")
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.get_act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
    # print(f"[CBConvLSTMEncoder.initial] carries: {carries}")
    return tuple(carries)

  def __call__(self, obs, reset, carry, training, single=False):
    """
    Process multi-modal observations through vector and image encoders.

    Args:
      obs (dict): Input observations dictionary
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag
      single (bool, optional): Whether processing a single timestep. Defaults to False.

    Returns:
      embed (jax.Array): Encoded representation of input observations
    """
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape
    new_carries = []

    # print(f"[CBConvLSTMEncoder] started! carries: {carries}", color="green")

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.functional.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      # NOTE: TODO: This is weird, need to be fixed.
      #   For now, we just dividing by some number for it to be lowered down.
      #   In the future, we have to look deeper into this.
      # assert all(x.dtype == jnp.uint8 for x in imgs)
      # x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = nn.cast(jnp.concatenate(imgs, -1), force=True)
      # print(f"[encoder] x: {x}")
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if i != 0:
          _carry = carry[i-1]
          # print(f"[CBConvLSTMEncoder.__call__] inside the loop: {_carry}", color="yellow")
          x = x.reshape((*bshape, *x.shape[1:]))
          new_carry, x = self.cores[i-1](_carry, x, reset, single=single)
          x = x.reshape((-1, *x.shape[bdims:]))
          new_carries.append(new_carry)
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.get_act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    embed = x.reshape((*bshape, *x.shape[1:]))
    return embed, tuple(new_carries)

  def macs(self, obs, reset, carry):
    # Estimate MACs for the encoder across time steps.
    total = 0

    # Infer batch and time dims from reset
    rshape = getattr(reset, 'shape', None)
    if rshape is None:
      return 0
    if len(rshape) == 2:
      B, T = int(rshape[0]), int(rshape[1])
    else:
      B, T = int(rshape[0]), 1

    # --- Vector (MLP) part ---
    if self.veckeys:
      # input dim is sum of vector feature dims
      input_dim = 0
      for k in self.veckeys:
        a = obs[k]
        # obs[k] may be (B,T,D) or (B,D)
        input_dim += int(a.shape[-1])
      in_ch = input_dim
      # Use the Linear.macs implementation and Norm.macs heuristic
      for i in range(self.layers):
        # Prepare dummy input shape: (B*T, in_ch) or (B, in_ch)
        batch_for_linear = B * T if T > 1 else B
        dummy_x = jnp.zeros((batch_for_linear, in_ch))
        try:
          linear_layer = nn.Linear(int(self.units))
          linear_m = linear_layer.macs(dummy_x)
        except Exception:
          linear_m = int(batch_for_linear * in_ch * int(self.units))
        total += int(linear_m)
        # Norm cost on output
        try:
          norm_layer = nn.Norm(self.norm)
          norm_m = norm_layer.macs(jnp.zeros((batch_for_linear, int(self.units))))
        except Exception:
          norm_m = int(batch_for_linear * int(self.units) * 6)
        # Activation cost estimate (approx 4 ops per element)
        act_m = int(batch_for_linear * int(self.units) * 4)
        total += int(norm_m + act_m)
        in_ch = int(self.units)

    # --- Image (CNN + cores) part ---
    if self.imgkeys:
      # Determine spatial dims and initial channels
      img0 = obs[sorted(self.imgkeys)[0]]
      if img0.ndim == 5:
        _, maybe_T, H, W, C0 = img0.shape
      else:
        maybe_T = 1
        _, H, W, C0 = img0.shape
      # concatenated channels across imgkeys
      in_ch = 0
      for k in sorted(self.imgkeys):
        a = obs[k]
        in_ch += int(a.shape[-1])

      # kernel area
      if isinstance(self.kernel, (tuple, list)):
        k_h, k_w = self.kernel
      else:
        k_h = k_w = int(self.kernel)
      k_area = int(k_h) * int(k_w)

      cur_H, cur_W = int(H), int(W)
      for i, depth in enumerate(self.depths):
        depth = int(depth)
        # If there's a core before this conv (i != 0), include its MACs.
        if i != 0:
          core = self.cores[i-1]
          # prepare dummy inputs for core.macs: shape (B,T,H,W,in_ch) or (B,H,W,in_ch)
          if T == 1:
            inp_shape = (B, cur_H, cur_W, in_ch)
          else:
            inp_shape = (B, T, cur_H, cur_W, in_ch)
          dummy_inp = jnp.zeros(inp_shape)
          core_carry = carry[i-1] if carry is not None and len(carry) >= i else None
          try:
            core_m = core.macs(core_carry, dummy_inp, reset)
          except Exception:
            core_m = 0
          total += int(core_m)
          # after core, the channel count becomes core.units (the core's output)
          in_ch = int(core.units)

        # Conv2D for this layer: use the layer's own macs() implementation.
        out_ch = depth
        # Determine stride as in __call__ (outer -> stride=1, strided -> stride=2, else stride=1)
        if self.outer and i == 0:
          stride = 1
        elif self.strided:
          stride = 2
        else:
          stride = 1

        # Prepare dummy input for conv.macs: flatten time into batch if needed
        batch_for_conv = B * T if T > 1 else B
        dummy_inp = jnp.zeros((batch_for_conv, cur_H, cur_W, in_ch))
        try:
          conv_layer = nn.Conv2D(out_ch, (k_h, k_w), stride)
          conv_m = conv_layer.macs(dummy_inp)
        except Exception:
          # Fallback to heuristic if conv.macs fails
          out_H = max(1, cur_H // stride) if stride > 1 else cur_H
          out_W = max(1, cur_W // stride) if stride > 1 else cur_W
          conv_m = B * T * out_H * out_W * out_ch * k_area * in_ch
        total += int(conv_m)

        # compute output spatial dims after layer (matching runtime: pooling or strided conv)
        if self.outer and i == 0:
          out_H, out_W = cur_H, cur_W
        else:
          out_H, out_W = max(1, cur_H // 2), max(1, cur_W // 2)

        # norm + activation after conv
        norm_m = B * T * out_H * out_W * out_ch * 4
        act_m = B * T * out_H * out_W * out_ch * 4
        total += int(norm_m + act_m)

        # update for next layer
        in_ch = out_ch
        cur_H, cur_W = out_H, out_W

    return int(total)
