
import math
import jax
import jax.numpy as jnp
from typing import Dict, Callable
import numpy as np

from lib.common import Space
from lib import nn
from lib.nn import functional as F

class Decoder(nn.Module):
  """
  A flexible neural network decoder for reconstructing multi-modal observations.

  This decoder can handle both vector and image-based outputs, applying different
  reconstruction strategies based on output type. It supports:
  - Vector outputs processed through multi-layer perceptron (MLP)
  - Image outputs processed through transposed convolutional neural network (CNN)
  - Optional symlog transformation for continuous outputs
  - Configurable network depth, units, normalization, and activation

  Attributes:
    units (int): Number of units in the MLP layers. Defaults to 1024.
    norm (str): Normalization type. Defaults to 'rms'.
    act (str): Activation function. Defaults to 'gelu'.
    outscale (float): Output scaling factor. Defaults to 1.0.
    depth (int): Base depth for convolutional layers. Defaults to 64.
    mults (tuple): Multipliers for increasing depth in CNN layers. Defaults to (2, 3, 4, 4).
    layers (int): Number of MLP layers. Defaults to 3.
    kernel (int): Kernel size for convolutional layers. Defaults to 5.
    symlog (bool): Whether to apply symlog transformation to continuous outputs. Defaults to True.
    bspace (int): Block space size for spatial transformations. Defaults to 8.
    outer (bool): Special handling for first CNN layer. Defaults to False.
    strided (bool): Whether to use strided convolutions. Defaults to False.
  """

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    """
    Initialize the decoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining output characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  def __call__(self, feat: Dict[str, jax.Array] | jax.Array, reset: jax.Array, training: bool):
    """
    Reconstruct multi-modal observations from latent features.

    Args:
      feat (dict): Dictionary of latent features, typically containing 'stoch' and 'deter' keys or a simple feature vector
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag

    Returns:
      recons (dict): Dictionary of reconstructed observations for each output type
    """
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    if isinstance(feat, jax.Array):
      # This is for working with a state space with stochastic and deterministic states
      inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
      inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
      inp = jnp.concatenate(inp, -1)
    elif isinstance(feat, jax.Array):
      # this is for working with a single feature vector
      inp = feat.reshape((math.prod(bshape), -1))
    else:
      raise TypeError(f"Unsupported feature type: {type(feat)}")

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', nn.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0) # (..., dim)
        x0 = x0.reshape((*x0.shape[:-1], g, minres[0], minres[1], -1)) # (..., dim) -> (..., g, h, w, c)
        # # (..., g, h, w, c) -> (..., h, w, g, c) -> (..., h, w, gc)
        s = len(x0.shape) - 4
        x0 = x0.transpose((*range(s), s + 1, s + 2, s, s + 3)).reshape((*x0.shape[:-4], minres[0], minres[1], -1))
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.get_act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.get_act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.get_act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = nn.distributions.Agg(nn.distributions.MSE(out), 3, jnp.sum)
        recons[k] = out
    return recons

  def macs(self, feat, reset):
    """Estimate MACs for the decoder given latent feat and reset.

    Mirrors the forward pass: MLP/BlockLinear/Linear + conv transpose stack.
    """
    total = 0
    rshape = getattr(reset, 'shape', None)
    if rshape is None:
      return 0
    if len(rshape) == 2:
      B, T = int(rshape[0]), int(rshape[1])
    else:
      B, T = int(rshape[0]), 1

    # Prepare inp shapes similar to forward
    bprod = B * T

    # Vector outputs (MLP + DictHead)
    if self.veckeys:
      # MLP: use nn.MLP.macs
      try:
        mlp_layer = nn.MLP(self.layers, self.units)
        dummy_inp = jnp.zeros((bprod, feat['deter'].shape[-1] + feat['stoch'].shape[-1]))
        total += int(mlp_layer.macs(dummy_inp))
      except Exception:
        total += int(bprod * (feat['deter'].shape[-1] + feat['stoch'].shape[-1]) * self.units * self.layers)
      # DictHead and output heads are harder to account exactly; approximate as linear from mlp output
      try:
        dicthead = nn.DictHead({k: self.obs_space[k] for k in self.veckeys}, {})
        dummy_x = jnp.zeros((bprod, self.units))
        total += int(dicthead.macs(dummy_x))
      except Exception:
        total += int(bprod * self.units * sum(self.obs_space[k].shape[-1] for k in self.veckeys))

    # Image outputs
    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      shape = (*minres, self.depths[-1])
      # BlockLinear / Linear space projection
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        # sp0: BlockLinear
        try:
          bl = nn.BlockLinear(u, g)
          dummy_x0 = jnp.zeros((bprod, feat['deter'].shape[-1]))
          total += int(bl.macs(dummy_x0))
        except Exception:
          total += int(bprod * feat['deter'].shape[-1] * u)
        # sp1 and sp2: two linears
        try:
          l1 = nn.Linear(2 * self.units)
          dummy_x1 = jnp.zeros((bprod, feat['stoch'].shape[-1]))
          total += int(l1.macs(dummy_x1))
        except Exception:
          total += int(bprod * feat['stoch'].shape[-1] * 2 * self.units)
        try:
          l2 = nn.Linear(shape)
          dummy_x1b = jnp.zeros((bprod, 2 * self.units))
          total += int(l2.macs(dummy_x1b))
        except Exception:
          total += int(bprod * 2 * self.units * math.prod(shape))
        # spnorm
        try:
          total += int(nn.Norm(self.norm).macs(jnp.zeros((bprod, math.prod(shape)))))
        except Exception:
          total += int(bprod * math.prod(shape) * 6)
      else:
        # space linear
        try:
          ls = nn.Linear(shape)
          dummy_inp = jnp.zeros((bprod, feat['deter'].shape[-1] + feat['stoch'].shape[-1]))
          total += int(ls.macs(dummy_inp))
        except Exception:
          total += int(bprod * (feat['deter'].shape[-1] + feat['stoch'].shape[-1]) * math.prod(shape))
        try:
          total += int(nn.Norm(self.norm).macs(jnp.zeros((bprod, math.prod(shape)))))
        except Exception:
          total += int(bprod * math.prod(shape) * 6)

      # Now the upsampling conv stack (reversed depths[:-1])
      cur_H, cur_W = minres[0], minres[1]
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        out_ch = int(depth)
        if self.strided:
          stride = 2
          try:
            conv = nn.Conv2D(out_ch, self.kernel, stride)
            dummy_x = jnp.zeros((bprod, cur_H, cur_W, int(self.depths[-1]) if i == len(self.depths[:-1]) - 1 else out_ch))
            total += int(conv.macs(dummy_x))
          except Exception:
            total += int(bprod * max(1, cur_H * 2) * max(1, cur_W * 2) * out_ch * self.kernel * self.kernel * out_ch)
          cur_H, cur_W = max(1, cur_H * 2), max(1, cur_W * 2)
        else:
          # upsample by repeat
          try:
            conv = nn.Conv2D(out_ch, self.kernel)
            dummy_x = jnp.zeros((bprod, cur_H * 2, cur_W * 2, int(self.depths[-1]) if i == len(self.depths[:-1]) - 1 else out_ch))
            total += int(conv.macs(dummy_x))
          except Exception:
            total += int(bprod * max(1, cur_H * 2) * max(1, cur_W * 2) * out_ch * self.kernel * self.kernel * out_ch)
          cur_H, cur_W = max(1, cur_H * 2), max(1, cur_W * 2)
        # norm + act
        total += int(bprod * cur_H * cur_W * out_ch * 6)

      # final imgout conv
      try:
        out_ch = int(self.imgdep)
        if self.outer:
          convf = nn.Conv2D(out_ch, self.kernel)
        elif self.strided:
          convf = nn.Conv2D(out_ch, self.kernel, 2)
        else:
          convf = nn.Conv2D(out_ch, self.kernel)
        dummy_xf = jnp.zeros((bprod, cur_H, cur_W, int(self.depths[-1])))
        total += int(convf.macs(dummy_xf))
      except Exception:
        total += int(bprod * cur_H * cur_W * int(self.depths[-1]) * self.kernel * self.kernel * int(self.imgdep))

    return int(total)