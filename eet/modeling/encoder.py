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

