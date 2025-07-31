"""
File: utils.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Heavily adapted some modules from Danijar Hafner's implementation
"""

import collections
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import checkify
from tensorflow_probability.substrates import jax as tfp
import functools

from . import ninjax as nj

tfd = tfp.distributions
tfb = tfp.bijectors
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
f32 = jnp.float32
i32 = jnp.int32

COMPUTE_DTYPE = jnp.bfloat16
PARAM_DTYPE = jnp.float32
ENABLE_CHECKS = False
LAYER_CALLBACK = lambda tensor, name: tensor


@functools.partial(jax.custom_vjp, nondiff_argnums=[1, 2])
def ensure_dtypes(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert x.dtype == fwd, (x.dtype, fwd)
  return x
def ensure_dtypes_fwd(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  return ensure_dtypes(x, fwd, bwd), ()
def ensure_dtypes_bwd(fwd, bwd, cache, dx):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert dx.dtype == bwd, (dx.dtype, bwd)
  return (dx,)
ensure_dtypes.defvjp(ensure_dtypes_fwd, ensure_dtypes_bwd)


def cast_to_compute(values):
  return treemap(
    lambda x: x if x.dtype == COMPUTE_DTYPE else x.astype(COMPUTE_DTYPE),
    values)

def cast_to_compute(xs, force=True):
  if force:
    should = lambda x: True
  else:
    should = lambda x: jnp.issubdtype(x.dtype, jnp.floating)
  return jax.tree.map(lambda x: COMPUTE_DTYPE(x) if should(x) else x, xs)


def cast_to_param(xs, force=True):
  if force:
    should = lambda x: True
  else:
    should = lambda x: jnp.issubdtype(x.dtype, jnp.floating)
  return jax.tree.map(lambda x: PARAM_DTYPE(x) if should(x) else x, xs)


def get_compute_dtype():
  return COMPUTE_DTYPE


def get_param_dtype():
  return PARAM_DTYPE


def check(predicate, message, **kwargs):
  if ENABLE_CHECKS:
    checkify.check(predicate, message, **kwargs)


def parallel():
  try:
    jax.lax.axis_index('i')
    return True
  except NameError:
    return False


# NOTE: This is a custom scan that does not use ninjax.scan,
#   removing because it conflicts with ninjax.scan
# def scan(fun, carry, xs, unroll=False, axis=0):
#   unroll = jax.tree_util.tree_leaves(xs)[0].shape[axis] if unroll else 1
#   return nj.scan(fun, carry, xs, False, unroll, axis)


def tensorstats(tensor, prefix=None):
  assert not prefix.startswith("_"), "Prefix cannot start with `_` because this tensorstats method is solely for logging, not intermediate output, we reserve `_` for intermediate output that requires lambda computation outside of GPU"
  assert tensor.size > 0, tensor.shape
  assert jnp.issubdtype(tensor.dtype, jnp.floating), tensor.dtype
  tensor = tensor.astype(f32)  # To avoid overflows.
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).mean(),
      'min': tensor.min(),
      'max': tensor.max(),
      'dist': subsample(tensor),
  }
  if prefix:
    metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
  return metrics


def subsample(values, amount=1024):
  values = values.flatten()
  if len(values) > amount:
    values = jax.random.permutation(nj.seed(), values)[:amount]
  return values


def switch(pred, lhs, rhs):
  def fn(lhs, rhs):
    assert lhs.shape == rhs.shape, (pred.shape, lhs.shape, rhs.shape)
    mask = pred
    while len(mask.shape) < len(lhs.shape):
      mask = mask[..., None]
    return jnp.where(mask, lhs, rhs)
  return treemap(fn, lhs, rhs)


def reset(xs, reset):
  def fn(x):
    mask = reset
    while len(mask.shape) < len(x.shape):
      mask = mask[..., None]
    return x * (1 - mask.astype(x.dtype))
  return treemap(fn, xs)


def video_grid(video, separator: int = 0):
  """
  Convert a batched video tensor of shape (B, T, H, W, C) to a grid of frames along the width axis: (T, H, B * W, C)
  If separator is not 0, a separator of shape (T, H, separator, C) will be added between each frame in the hieght and width axis.
  """
  B, T, H, W, C = video.shape
  separator_color = jnp.ones((1, 1, 1, 1, C))
  separator_img = jnp.tile(separator_color, (B, T, H, separator, C))
  result = jnp.concatenate([video, separator_img], axis=3)
  result = result.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * (W + separator), C))
  return result

def balance_stats(dist, target, thres):
  # Values are NaN when there are no positives or negatives in the current
  # batch, which means they will be ignored when aggregating metrics via
  # np.nanmean() later, as they should.
  pos = (target.astype(f32) > thres).astype(f32)
  neg = (target.astype(f32) <= thres).astype(f32)
  pred = (dist.mean().astype(f32) > thres).astype(f32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      rate=pos.mean(),
      avg=target.astype(f32).mean(),
      pred=dist.mean().astype(f32).mean(),
  )


class DictConcat:
  """
  A utility class for concatenating dictionary inputs with different spaces.

  This class takes a dictionary of input spaces and performs the following operations:
  1. Sorts and validates input keys
  2. Handles different input types (discrete and continuous)
  3. Applies masking to handle missing or invalid inputs
  4. Converts discrete inputs to one-hot encoding
  5. Applies an optional squish function to continuous inputs
  6. Reshapes and concatenates inputs along the last dimension

  Args:
    spaces (dict): A dictionary of input spaces with their specifications
    fdims (int): Number of batch dimensions to preserve during reshaping
    squish (callable, optional): A function to transform continuous inputs. Defaults to identity.
  """

  def __init__(self, spaces, fdims, squish=lambda x: x):
    """
    Initialize the DictConcat utility for concatenating dictionary inputs.

    Args:
      spaces (Dict[str, Space]): A dictionary of input spaces with their specifications.
        Each space defines the shape, dtype, and other characteristics of an input.

      fdims (int): Number of batch dimensions to preserve during reshaping.
        Must be at least 1. This determines how many leading dimensions are kept
        when reshaping inputs before concatenation.

      squish (callable, optional): A function to transform continuous inputs.
        By default, it's the identity function (returns input unchanged).
        Useful for applying transformations like symlog to continuous inputs.
        Defaults to lambda x: x.
    """
    assert 1 <= fdims, fdims
    self.keys = sorted(spaces.keys())
    self.spaces = spaces
    self.fdims = fdims
    self.squish = squish

  def __call__(self, xs):
    """
    Concatenate inputs from different spaces.

    Args:
      xs (dict): A dictionary of input tensors corresponding to the spaces

    Returns:
      jnp.ndarray: A concatenated tensor with inputs from all spaces
    """
    assert all(k in xs for k in self.spaces), (self.spaces, xs.keys())
    bdims = xs[self.keys[0]].ndim - len(self.spaces[self.keys[0]].shape)
    ys = []
    for key in self.keys:
      space = self.spaces[key]
      x = xs[key]
      assert x.shape[bdims:] == space.shape, (key, bdims, space.shape, x.shape)
      if space.dtype == jnp.uint8 and len(space.shape) in (2, 3):
        raise NotImplementedError('Images are not supported.')
      elif space.discrete:
        classes = np.asarray(space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        classes = classes[0].item()
        x = x.astype(jnp.int32)
        x = jax.nn.one_hot(x, classes, dtype=COMPUTE_DTYPE)
      else:
        x = self.squish(x)
        x = x.astype(COMPUTE_DTYPE)
      x = x.reshape((*x.shape[:bdims + self.fdims - 1], -1))
      ys.append(x)
    return jnp.concatenate(ys, -1)



def concat_dict(mapping, batch_shape=None):
  tensors = [v for _, v in sorted(mapping.items(), key=lambda x: x[0])]
  if batch_shape is not None:
    tensors = [x.reshape((*batch_shape, -1)) for x in tensors]
  return jnp.concatenate(tensors, -1)


def onehot_dict(mapping, spaces, filter=False, limit=256):
  result = {}
  for key, value in mapping.items():
    if key not in spaces and filter:
      continue
    space = spaces[key]
    if space.need_onehot and space.dtype != jnp.uint8:
      if limit:
        assert space.classes <= limit, (key, space, limit)
      value = jax.nn.one_hot(value, space.classes)
    result[key] = value
  return result

