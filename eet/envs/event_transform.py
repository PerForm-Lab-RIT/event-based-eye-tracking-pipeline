"""
File: event_transform.py
Author: Viet Nguyen
Date: 2025-06-07

Description: Event representation classes
"""

import functools
from typing import Dict, List, Tuple, Iterator
import numpy as np
from lib import print, Space


EVENT_DTYPE = np.dtype([
  ('t', np.int64),  # Use int64 for nanosecond timestamps
  ('x', np.int32),  # 32-bit ints sufficient for coordinates
  ('y', np.int32),
  ('p', np.int8)    # Single byte for binary polarity
])


def resize_positions(x: np.ndarray, y: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int):
  """Resize event positions from source to destination dimensions."""
  scale_x = dst_width / src_width
  scale_y = dst_height / src_height
  x_resized = (x * scale_x).astype(np.int32).clip(0, dst_width - 1)
  y_resized = (y * scale_y).astype(np.int32).clip(0, dst_height - 1)
  return x_resized, y_resized


class EventTransform:

  @property
  def transform_space(self):
    raise NotImplementedError("Returns: Space")

  def __call__(self, events: np.ndarray, *args, **kwargs) -> np.ndarray:
    raise NotImplementedError("Subclass must implement this method")


class Binary(EventTransform):
  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height

  @functools.cached_property
  def transform_space(self):
    return Space(dtype=np.uint32, shape=(self.height, self.width, 2))

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    """Convert events to binary representation.

    Args:
        events: numpy array of shape (N, 4) or structured array with fields (t, x, y, p)

    Returns:
        numpy array of shape (H, W, 2) containing (positive, negative) event counts
    """
    frame = np.zeros((self.height, self.width, 2), dtype=np.uint32)
    # Support both structured and regular arrays
    x, y = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    p = events['p']
    # Polarity: assume p in {0, 1}
    # Map p to 0 (positive) and 1 (negative)
    # p_bin = ((p < 0).astype(int) if np.any((p == -1) | (p == 1)) else p)
    for xi, yi, pi in zip(x, y, p):
      if 0 <= xi < self.width and 0 <= yi < self.height and pi in (0, 1):
        frame[yi, xi, pi] += 1
    return frame


class BinaryRepresentation(EventTransform):
  def __init__(self, width: int, height: int, microbins: int = 4):
    self.width = width
    self.height = height
    self.microbins = microbins
    self.mask = np.concatenate([
      np.array([1.0]),
      np.cumprod([2.0 for _ in range(microbins - 1)])
    ], axis = 0)
    self.mask /= self.mask[-1] # normalize to 1 as max range
    self.mask = self.mask[None, None]

  @functools.cached_property
  def transform_space(self):
    return Space(dtype=np.float32, shape=(self.height, self.width, 1))

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    if len(events) == 0:
      return np.zeros((self.height, self.width, 1), dtype=np.float32)

    ts = events['t']
    xs, ys = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    ps = events['p'] * 2 - 1 # [0, 1] -> [-1, 1]

    # Normalize timestamps to [0, num_bins - 1]
    t_norms = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8) * (self.microbins - 1)
    t_bins = t_norms.astype(int)

    # Create a 3D array to store the voxel grid
    binarep = np.full((self.height, self.width, self.microbins), 0.0, dtype=np.float32)

    # Count events in each voxel
    for i in range(len(ts)):
      xi = xs[i]
      yi = ys[i]
      pi = ps[i]
      bin_idx = t_bins[i]
      if 0 <= xi < self.width and 0 <= yi < self.height and 0 <= bin_idx < self.microbins:
        binarep[yi, xi, bin_idx] += pi
    return (binarep * self.mask).sum(axis=-1, keepdims=True)


class VoxelGrid(EventTransform):
  def __init__(self, width: int, height: int, microbins: int = 4):
    self.width = width
    self.height = height
    self.microbins = microbins

  @functools.cached_property
  def transform_space(self):
    return Space(dtype=np.float32, shape=(self.height, self.width, self.microbins))

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    if len(events) == 0:
      return np.zeros((self.height, self.width, 1), dtype=np.float32)

    ts = events['t']
    xs, ys = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    ps = events['p'] * 2 - 1 # [0, 1] -> [-1, 1]

    # Normalize timestamps to [0, num_bins - 1]
    t_norms = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8) * (self.microbins - 1)
    t_bins = t_norms.astype(int)

    # Create a 3D array to store the voxel grid
    voxel_grid = np.full((self.height, self.width, self.microbins), 0.0, dtype=np.float32)

    # Count events in each voxel
    for i in range(len(ts)):
      xi = xs[i]
      yi = ys[i]
      pi = ps[i]
      bin_idx = t_bins[i]
      if 0 <= xi < self.width and 0 <= yi < self.height and 0 <= bin_idx < self.microbins:
        voxel_grid[yi, xi, bin_idx] += pi
    return voxel_grid


class Histogram(EventTransform):
  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height

  @functools.cached_property
  def transform_space(self):
    return Space(dtype=np.float32, shape=(self.height, self.width, 2))

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    img = np.full((self.height, self.width, 2), 0, dtype=np.float32)
    x, y = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    for i, event in enumerate(events):
      img[y[i], x[i], event["p"]] += 1.0
    return img


class EventFrame(EventTransform):
  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height

  @functools.cached_property
  def transform_space(self):
    return Space(dtype=np.float32, shape=(self.height, self.width, 1))

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    if len(events) == 0:
      return np.full((self.height, self.width, 1), 0.5, dtype=np.float32)
    img = np.full((self.height, self.width, 1), 0.5, dtype=np.float32) # mid value is 128
    x, y = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    img[y, x, 0] = events['p']
    return img


def build_event_transform(eventrepr: str, *args, **kwargs):
  return {
    "binary": Binary,
    "binarep": BinaryRepresentation,
    "histogram": Histogram,
    "voxelgrid": VoxelGrid,
    "eventframe": EventFrame,
  }[eventrepr](*args, **kwargs)


