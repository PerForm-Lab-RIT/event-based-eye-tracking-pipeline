"""
File: heads.py
Author: Viet Nguyen
Date: 2025-03-21

Description: Heavily adapted some modules from Danijar Hafner's implementation
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
from .. import nn
from . import distributions
from ..common.space import Space

i32 = jnp.int32
f32 = jnp.float32


class MLPHead(nn.Module):

  units: int = 1024
  layers: int = 5
  act: str = 'silu'
  norm: str = 'rms'
  bias: bool = True
  winit: str | Callable = nn.Initializer('trunc_normal')
  binit: str | Callable = nn.Initializer('zeros')

  def __init__(self, space, dist, **hkw):
    shared = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    mkw = dict(**shared, act=self.act, norm=self.norm)
    hkw = dict(**shared, **hkw)
    self.mlp = nn.MLP(self.layers, self.units, **mkw, name='mlp')
    if isinstance(space, dict):
      self.head = DictHead(space, dist, **hkw, name='head')
    else:
      print(f"space: {space}, dist: {dist}, hkw: {hkw}")
      self.head = Head(space, dist, **hkw, name='head')

  def __call__(self, x, bdims):
    """
    Process input through MLP and head layers.

    Args:
        x (jax.Array or dict): Input tensor or dictionary of tensors
        bdims (int): Number of batch dimensions to preserve

    Returns:
        jax.Array or dict: Processed output from head layer
    """
    bshape = jax.tree.leaves(x)[0].shape[:bdims]
    x = x.reshape((*bshape, -1))
    x = self.mlp(x)
    x = self.head(x)
    return x


class DictHead(nn.Module):

  def __init__(self, spaces, dists, **kw):
    assert spaces, spaces
    if not isinstance(spaces, dict):
      spaces = {'output': spaces}
    if not isinstance(dists, dict):
      dists = {'output': dists}
    assert spaces.keys() == dists.keys(), (spaces, dists)
    self.spaces = spaces
    self.dists = dists
    self.kw = kw

  def __call__(self, x):
    outputs = {}
    for key, dist in self.dists.items():
      space = self.spaces[key]
      outputs[key] = self.sub(key, Head, space, dist, **self.kw)(x)
    return outputs


class Head(nn.Module):

  minstd: float = 1.0
  maxstd: float = 1.0
  unimix: float = 0.0
  bins: int = 255
  outscale: float = 1.0

  def __init__(self, space: Union[int, tuple, Space], dist: str, **kw):
    if isinstance(space, int):
      space = (space,)
    if isinstance(space, tuple):
      space = Space(np.float32, space)
    if dist == 'onehot':
      classes = np.asarray(space.classes).flatten()
      assert (classes == classes[0]).all(), classes
      shape = (*space.shape, classes[0].item())
      space = Space(f32, shape, 0.0, 1.0)
    self.space = space
    self.impl = dist
    self.kw = {**kw, 'outscale': self.outscale}

  def __call__(self, x):
    if not hasattr(self, self.impl):
      raise NotImplementedError(self.impl)
    x = nn.ensure_dtypes(x)
    dist = getattr(self, self.impl)(x)
    if self.space.shape:
      dist = distributions.AggregatedDistribution(dist, len(self.space.shape), jnp.sum)
    assert dist.pred().shape[x.ndim - 1:] == self.space.shape, (
        self.space, self.impl, x.shape, dist.pred().shape)
    return dist

  def binary(self, x):
    assert np.all(self.space.classes == 2), f"space: {self.space}; classes: {self.space.classes}"
    logit = self.sub('logit', nn.Linear, self.space.shape, **self.kw)(x)
    return distributions.Binary(logit)

  def categorical(self, x):
    assert self.space.discrete
    classes = np.asarray(self.space.classes).flatten()
    assert (classes == classes[0]).all(), classes
    shape = (*self.space.shape, classes[0].item())
    logits = self.sub('logits', nn.Linear, shape, **self.kw)(x)
    output = distributions.Categorical(logits)
    output.minent = 0
    output.maxent = np.log(logits.shape[-1])
    return output

  def onehot(self, x):
    assert not self.space.discrete
    logits = self.sub('logits', nn.Linear, self.space.shape, **self.kw)(x)
    return distributions.OneHot(logits, self.unimix)

  def identity(self, x):
    assert not self.space.discrete
    pred = self.sub('pred', nn.Linear, self.space.shape, **self.kw)(x)
    return distributions.Identity(pred)

  def symlog_mse(self, x):
    assert not self.space.discrete
    pred = self.sub('pred', nn.Linear, self.space.shape, **self.kw)(x)
    return distributions.MSE(pred, nn.functional.symlog)

  def symexp_twohot(self, x):
    assert not self.space.discrete
    shape = (*self.space.shape, self.bins)
    logits = self.sub('logits', nn.Linear, shape, **self.kw)(x)
    if self.bins % 2 == 1:
      half = jnp.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=f32)
      half = nn.functional.symexp(half)
      bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
    else:
      half = jnp.linspace(-20, 0, self.bins // 2, dtype=f32)
      half = nn.functional.symexp(half)
      bins = jnp.concatenate([half, -half[::-1]], 0)
    return distributions.TwoHot(logits, bins)

  def bounded_normal(self, x):
    assert not self.space.discrete
    mean = self.sub('mean', nn.Linear, self.space.shape, **self.kw)(x)
    stddev = self.sub('stddev', nn.Linear, self.space.shape, **self.kw)(x)
    lo, hi = self.minstd, self.maxstd
    stddev = (hi - lo) * jax.nn.sigmoid(stddev + 2.0) + lo
    output = distributions.Normal(jnp.tanh(mean), stddev)
    output.minent = distributions.Normal(jnp.zeros_like(mean), self.minstd).entropy()
    output.maxent = distributions.Normal(jnp.zeros_like(mean), self.maxstd).entropy()
    return output

  def normal_logstd(self, x):
    assert not self.space.discrete
    mean = self.sub('mean', nn.Linear, self.space.shape, **self.kw)(x)
    logstd = self.sub('stddev', nn.Linear, self.space.shape, **self.kw)(x)
    # output = distributions.Normal(mean, jnp.exp(stddev))
    output = distributions.Normal(mean, jnp.exp(jnp.clip(logstd, -10.0, 2.0)))
    return output
