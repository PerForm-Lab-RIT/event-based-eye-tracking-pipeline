"""
File: utils.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Heavily adapted some modules from Danijar Hafner's implementation
"""

import datetime
import collections
import io
import os
import json
from typing import Dict, Tuple, List
import pathlib
import re
import time
import random

import numpy as np
import collections
import re
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter

from ...utils import tree, Space


f32 = torch.float32
i32 = torch.int32

COMPUTE_DTYPE = torch.float32
PARAM_DTYPE = torch.float32


def to_f32(x):
  return x.to(dtype=torch.float32)


def to_i32(x):
  return x.to(dtype=torch.int32)



"""Convert tensor tree to numpy array tree."""
to_np = lambda x, dtype=None: tree.map(lambda x2: x2.detach().cpu().numpy().astype(dtype) if dtype is not None else x2.detach().cpu().numpy(), x)

"""Stop gradient - detach tensor from computation graph."""
sg = lambda x: tree.map(lambda x2: x2.detach(), x)

to_torch = lambda x, device=None: tree.map(lambda x2: torch.as_tensor(x2, device=device), x)

to_device = lambda x, device: tree.map(lambda x2: x2.to(device), x)

TensorTree = Dict[str, torch.Tensor] | torch.Tensor | Tuple[torch.Tensor, ...] | List[torch.Tensor]

cast_to_compute = lambda x: tree.map(lambda x2: x2.to(COMPUTE_DTYPE), x)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)


def tensorstats(tensor: torch.Tensor, prefix=None):
  assert not prefix.startswith("_"), "Prefix cannot start with `_` because this tensorstats method is solely for logging, not intermediate output, we reserve `_` for intermediate output that requires lambda computation outside of GPU"
  tensor = tensor.to(f32)  # To avoid overflows.
  metrics = {
      'mean': to_np(tensor.mean()),
      'std': to_np(tensor.std()),
      'mag': to_np(torch.abs(tensor).mean()),
      'min': to_np(tensor.min()),
      'max': to_np(tensor.max()),
      'dist': to_np(subsample(tensor)),
  }
  if prefix:
    metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
  return metrics


def subsample(values: torch.Tensor, amount=1024):
  values = values.flatten()
  if len(values) > amount:
    values = values[torch.randperm(len(values))[:amount]]
  return values


def switch(pred, lhs, rhs):
  def fn(lhs, rhs):
    assert lhs.shape == rhs.shape, (pred.shape, lhs.shape, rhs.shape)
    mask = pred
    while len(mask.shape) < len(lhs.shape):
      mask = mask[..., None]
    return torch.where(mask, lhs, rhs)
  return tree.map(fn, lhs, rhs)


def reset(xs: torch.Tensor, reset: torch.Tensor):
  def fn(x: torch.Tensor):
    mask = reset
    while len(mask.shape) < len(x.shape):
      mask = mask[..., None]
    return x * (1 - mask.to(x.dtype))
  return tree.map(fn, xs)


def video_grid(video, separator: int = 0):
  """
  Convert a batched video tensor of shape (B, T, H, W, C) to a grid of frames along the width axis: (T, H, B * W, C)
  If separator is not 0, a separator of shape (T, H, separator, C) will be added between each frame in the hieght and width axis.
  """
  B, T, H, W, C = video.shape
  separator_color = torch.ones((1, 1, 1, 1, C), device=video.device, dtype=video.dtype)
  separator_img = separator_color.repeat(B, T, H, separator, 1)
  result = torch.cat([video, separator_img], dim=3)
  result = result.permute((1, 2, 0, 3, 4)).reshape((T, H, B * (W + separator), C))
  return result


def balance_stats(dist, target, thres):
  # Values are NaN when there are no positives or negatives in the current
  # batch, which means they will be ignored when aggregating metrics via
  # np.nanmean() later, as they should.
  pos = (target.to(f32) > thres).to(f32)
  neg = (target.to(f32) <= thres).to(f32)
  pred = (dist.mean().to(f32) > thres).to(f32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      rate=pos.mean(),
      avg=target.to(f32).mean(),
      pred=dist.mean().to(f32).mean(),
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

      fdims (int): target feature dims. Number of feature dimensions to preserve after batch dims during
        reshaping. Must be at least 1. This determines how many leading dimensions
        after the batch dimension are kept when reshaping inputs before concatenation.

      squish (callable, optional): A function to transform continuous inputs.
        By default, it's the identity function (returns input unchanged).
        Useful for applying transformations like symlog to continuous inputs.
        Defaults to lambda x: x.
    """
    assert 1 <= fdims, fdims
    if fdims > 1:
      raise ValueError("Although the class is implemented for general case, there is no point in using fdims > 1. If you have concrete purpose, you can comment this line out!")
    self.keys = sorted(spaces.keys())
    self.spaces: Dict[str, Space] = spaces
    self.fdims: int = fdims
    self.squish: callable = squish

  def __call__(self, xs: Dict[str, torch.Tensor]):
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
      x: torch.Tensor = xs[key]
      assert x.shape[bdims:] == space.shape, (key, bdims, space.shape, x.shape)
      if space.dtype == torch.uint8 and len(space.shape) in (2, 3):
        raise NotImplementedError('Images are not supported.')
      elif space.discrete:
        classes = np.asarray(space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        classes = classes[0].item()
        x = x.to(dtype=torch.int32)
        x = torch.nn.functional.one_hot(x.long(), classes).to(dtype=COMPUTE_DTYPE)
      else:
        x = self.squish(x)
        x = x.to(dtype=COMPUTE_DTYPE)
      x = x.reshape((*x.shape[:bdims + self.fdims - 1], -1))
      ys.append(x)
    return torch.concatenate(ys, -1)

  @staticmethod
  def units(spaces: Dict[str, Space], fdims: int):
    """Compute the feature units of the output

    Args:
        spaces (Dict[str, Space]): dictionary of spaces
        fdims (int): target feature dims. Number of feature dimensions to preserve after batch dims during
        reshaping. Must be at least 1. This determines how many leading dimensions
        after the batch dimension are kept when reshaping inputs before concatenation.
    """
    assert fdims == 1, "Computing dimension for fdims > 1 is not supported"
    total = 0
    for key in spaces.keys():
      space = spaces[key]
      if space.dtype == np.uint8 and len(space.shape) in (2, 3):
        raise NotImplementedError('Images are not supported.')
      elif space.discrete:
        classes = np.asarray(space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        inputs = np.prod(space.shape)
        classes = classes[0].item()
        total += inputs * classes
      else:
        total += np.prod(space.shape)
    return int(total)


def concat_dict(mapping: Dict[str, torch.Tensor], batch_shape=None):
  tensors = [v for _, v in sorted(mapping.items(), key=lambda x: x[0])]
  if batch_shape is not None:
    tensors = [x.reshape((*batch_shape, -1)) for x in tensors]
  return torch.cat(tensors, -1)


def onehot_dict(mapping: Dict[str, torch.Tensor], spaces: Dict[str, Space], filter=False, limit=256):
  result = {}
  for key, value in mapping.items():
    if key not in spaces and filter:
      continue
    space = spaces[key]
    if space.need_onehot and space.dtype != torch.uint8:
      if limit:
        assert space.classes <= limit, (key, space, limit)
      value = torch.nn.functional.one_hot(value, space.classes)
    result[key] = value
  return result


