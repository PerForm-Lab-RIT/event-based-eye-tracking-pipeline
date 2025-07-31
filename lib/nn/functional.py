"""
File: functional.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Include some functional operations
"""

from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
import math
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from . import ninjax as nj
from .utils import cast_to_compute, sg
f32 = jnp.float32


def min_max_norm(arr: jax.Array | np.ndarray, axis=-1, eps=1e-5):
  min_arr = arr.min(axis=axis, keepdims=True)
  max_arr = arr.max(axis=axis, keepdims=True)
  out = (arr - min_arr) / (max_arr - min_arr).clip(eps)
  return out


def masked_fill_other(x: jax.Array, mask: jax.Array, other=0) -> jax.Array:
  """Return an output with masked condition, with non-masked value
    be the other value

  Args:
      x (jax.Array): _description_
      mask (jax.Array): _description_
      other (int, optional): _description_. Defaults to 0.

  Returns:
      jax.Array: _description_
  """
  return jnp.where(mask, x, jnp.broadcast_to(other, x.shape))


def masked_fill(x: jax.Array, mask: jax.Array, value=0) -> jax.Array:
  """Return an output with masked condition, with non-masked value
    be the other value

  Args:
      x (jax.Array): _description_
      mask (jax.Array): _description_
      other (int, optional): _description_. Defaults to 0.

  Returns:
      jax.Array: _description_
  """
  return jnp.where(mask, jnp.broadcast_to(value, x.shape), x)


def resize_2d(x: jax.Array, size: Tuple[int, int], method: str = 'bilinear'):
  *B, H, W, C = x.shape
  return jax.image.resize(x, (*B, *size, C), method=method)


def reflection_pad_2d(x: jax.Array, pad: int):
  *B, H, W, C = x.shape
  pad_width = [(0, 0) for _ in range(len(B))] + [(pad, pad), (pad, pad), (0, 0)]
  return jnp.pad(x, pad_width, mode='reflect') # equals to reflection pad 2D


def pad_2d(x: jax.Array, pad: int):
  *B, H, W, C = x.shape
  pad_width = [(0, 0) for _ in range(len(B))] + [(pad, pad), (pad, pad), (0, 0)]
  return jnp.pad(x, pad_width, mode='constant', constant_values=0) # equals to reflection pad 2D


def bce(inputs: jax.Array, target: jax.Array, eps=1e-8) -> jax.Array:
  """Binary cross entropy loss with inputs as probabilities

  Args:
      inputs (jax.Array): _description_
      target (jax.Array): _description_
      eps (_type_, optional): _description_. Defaults to 1e-8.

  Returns:
      jax.Array: _description_
  """
  return - (target * (jnp.log(inputs.clip(eps))) + (1 - target) * jnp.log((1 - inputs).clip(eps)))


def bce_with_logits(logits: jax.Array, target: jax.Array, eps=1e-8) -> jax.Array:
  """Binary cross entropy loss with inputs as logits

  Args:
      logits (jax.Array): (*B, dim)
      target (jax.Array): (*B, dim)
      eps (_type_, optional): _description_. Defaults to 1e-8.

  Returns:
      jax.Array: _description_
  """
  # return - (target * (jax.nn.log_sigmoid(logits)) + (1 - target) * (jax.nn.log_sigmoid(-logits)))
  return jax.nn.softplus(-logits) + (logits - jax.nn.softplus(logits)) * target # more stable version


def focal_loss_with_logits(logits: jax.Array, targets: jax.Array,
    gamma: float = 2.0, eps: float = 1e-8) -> jax.Array:
  """Compute Focal Loss from BCE.

  Args:
    logits (jax.Array): Logits before sigmoid activation. (...) or (..., classes)
    targets (jax.Array): Ground truth labels (0 or 1). (...) or (..., classes)
    gamma (float): Focusing parameter gamma (default: 2.0).
    eps (float): Small value for numerical stability.

  Returns:
    jax.Array: Focal loss for each sample.
  """
  probs = jax.nn.sigmoid(logits)  # Convert logits to probabilities
  probs = jnp.clip(probs, eps, 1 - eps)  # Prevent log(0) issues
  # Standard BCE Loss
  bce_loss = - (targets * jnp.log(probs) + (1 - targets) * jnp.log(1 - probs))
  # Focal Loss scaling factor
  focal_weight = (1 - probs) ** gamma * targets + probs ** gamma * (1 - targets)
  # Apply focal weighting
  return focal_weight * bce_loss


def focal_loss_multiclass_with_logits(
    logits: jax.Array,
    targets: jax.Array,
    gamma: float = 2.0,
    eps: float = 1e-8,
    class_weights: jax.Array | None = None  # shape: (num_classes,)
) -> jax.Array:
    """
    Multi-class Focal Loss with logits.
    This is the focal loss for multi-class classification tasks that have
      probabilities summed to 1.0

    Args:
      logits: (batch_size, num_classes) — raw model outputs
      targets: (batch_size,) — integer class labels (0 to num_classes-1)
      gamma: focusing parameter
      eps: small value to prevent log(0)
      class_weights: optional array of shape (num_classes,)

    Returns:
      loss: (batch_size,) — focal loss per sample
    """
    log_probs = jax.nn.log_softmax(logits)  # (B, C)
    probs = jnp.exp(log_probs)              # (B, C)

    # Gather the predicted prob and log-prob for the target class
    targets_one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
    pt = jnp.sum(probs * targets_one_hot, axis=-1)           # (B,)
    pt = jnp.clip(pt, eps, 1.0)                              # prevent log(0)
    log_pt = jnp.sum(log_probs * targets_one_hot, axis=-1)   # (B,)

    # Apply class weights (gather per-sample)
    if class_weights is not None:
        weight_per_sample = class_weights[jnp.int32(targets)]     # (B,)
    else:
        weight_per_sample = 1.0

    loss = -weight_per_sample * ((1 - pt) ** gamma) * log_pt  # (B,)

    return loss  # You can .mean() outside if needed


def nll_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
  """Negative log likelihood loss

  Args:
      logits (jax.Array): (B, T, C)
      labels (jax.Array): (B, T, C)

  Returns:
      jax.Array: (B, T)
  """
  assert logits.shape == labels.shape, (logits.shape, labels.shape)
  # return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.sum(labels * (jnp.log(labels + 1e-8) - jax.nn.log_softmax(logits)), axis=-1)


def symlog(x):
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def l2_norm(x: jax.Array):
  dtype = x.dtype
  L = x.shape[-1]
  epsilon = 1e-6
  x = jnp.sqrt((x**2).sum(-1, keepdims=True) + epsilon).repeat(L, -1) # (*B, 1) -> (*B, L)
  return x.astype(dtype)


def gelu_tanh(x):
  # Constants used in the approximation
  sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
  coeff = 0.044715
  # GELU approximation formula
  return 0.5 * x * (1 + jnp.tanh(sqrt_2_over_pi * (x + coeff * jnp.power(x, 3))))


def categorical_cross_entropy(logits, labels, onehot=True):
  # logits: (*B, E), label: (*B)
  log_probs = jax.nn.log_softmax(logits)
  if onehot:
    _labels = jax.nn.one_hot(labels, logits.shape[-1]) # (*B, E)
  else:
    _labels = labels # (*B, E)
  loss = -jnp.sum(log_probs * _labels, axis=-1) # (*B)
  return loss


def where(condition, xs, ys):
  """

  Args:
      condition (Tree): any tree with shape (*some_shape,)
      xs (Tree): any tree with shape (*some_shape, *dim)
      ys (Tree): any tree with shape (*some_shape, *dim)

  Returns:
      Tree: same shape as xs and ys. Resulted masked value with mask broadcasted to the shape of xs and ys
  """
  assert condition.dtype == bool, condition.dtype
  def fn(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    expanded = jnp.expand_dims(condition, list(range(condition.ndim, x.ndim)))
    return jnp.where(expanded, x, y)
  return jax.tree.map(fn, xs, ys)


def mask(xs, mask):
  """Resulted masked value with mask broadcasted to the shape of xs
    negative mask values will become zeros.

  Args:
      xs (Tree): any tree with shape (*some_shape, *dim)
      mask (jax.Array): (*some_shape,)

  Returns:
      Tree: same shape as xs. Resulted masked value with mask broadcasted to the shape of xs
        negative mask values will become zeros.
  """
  return where(mask, xs, jax.tree.map(jnp.zeros_like, xs))


def dropout(x, prob, training):
  if not prob or not training:
    return x
  keep = jax.random.bernoulli(nj.seed(), 1.0 - prob, x.shape)
  return x * keep / (1.0 - prob)


def rms(xs):
  """Compute root mean square for the whole tree

  Args:
      xs (_type_): _description_

  Returns:
      _type_: _description_
  """
  xs = jax.tree.leaves(xs)
  count = sum(x.size for x in xs)
  sumsq = jnp.stack([f32(jnp.square(x).sum()) for x in xs]).sum()
  return jnp.sqrt(sumsq / f32(count))


def normalize(x, p=2, axis=-1, eps=1e-12):
  norm = jnp.linalg.norm(x, ord=p, axis=axis, keepdims=True)
  return x / (norm + eps)

