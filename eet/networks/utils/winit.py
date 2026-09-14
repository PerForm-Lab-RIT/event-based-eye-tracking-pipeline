
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F

from .. import utils as U

def get_initializer(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return Initializer(dist, fan, 1.0)


class Initializer:

  def __init__(self, dist='trunc_normal', fan='in', scale=1.0):
    self.dist = dist
    self.fan = fan
    self.scale = scale

  def __call__(self, shape, dtype=torch.float32, fshape=None, device=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    assert all(isinstance(x, int) for x in shape), (
      shape, [type(x) for x in shape])
    assert all(x > 0 for x in shape), shape
    fanin, fanout = self.compute_fans(shape if fshape is None else fshape)
    fan = {
      'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout, 'none': 1,
    }[self.fan]
    if self.dist == 'zeros':
      x = torch.zeros(shape, dtype=dtype, device=device)
    elif self.dist == 'uniform':
      limit = np.sqrt(1 / fan)
      x = torch.empty(shape, dtype=dtype, device=device).uniform_(-limit, limit)
    elif self.dist == 'normal':
      x = torch.randn(shape, dtype=dtype, device=device)
      x *= np.sqrt(1 / fan)
    elif self.dist == 'trunc_normal':
      x = torch.empty(shape, dtype=dtype, device=device)
      # Truncated normal: sample from [-2, 2] and scale by 1.1368 * sqrt(1/fan)
      nn.init.trunc_normal_(x, mean=0.0, std=1.0, a=-2.0, b=2.0)
      x *= 1.1368 * np.sqrt(1 / fan)
    elif self.dist == 'normed':
      x = torch.empty(shape, dtype=dtype, device=device).uniform_(-1, 1)
      x *= (1 / torch.linalg.norm(x.reshape((-1, shape[-1])), ord=2, dim=0))
    else:
      raise NotImplementedError(self.dist)
    x *= self.scale
    return x

  def __repr__(self):
    return f'Initializer({self.dist}, {self.fan}, {self.scale})'

  def __eq__(self, other):
    attributes = ('dist', 'fan', 'scale')
    return all(getattr(self, k) == getattr(other, k) for k in attributes)

  @staticmethod
  def compute_fans(shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    else:
      space = math.prod(shape[:-2])
      return (shape[-2] * space, shape[-1] * space)


def initialize_module(tensor: torch.Tensor, init: str | None = None, outscale: float = 1.0):
  # Apply custom weight initialization
  if init:
    weight_init = U.get_initializer(init)
    with torch.no_grad():
      tensor.copy_(weight_init(tensor.shape) * outscale)