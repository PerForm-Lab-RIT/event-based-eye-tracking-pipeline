"""
File: event_transform.py
Author: Viet Nguyen
Date: 2025-06-07

Description: Event representation classes
"""
from abc import ABC, abstractmethod
import functools
from typing import Dict, List, Tuple, Iterator
import numpy as np
import tonic.transforms as tonic_transforms

from ..types import EVENT_DTYPE


def resize_positions(x: np.ndarray, y: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int):
  """Resize event positions from source to destination dimensions."""
  scale_x = dst_width / src_width
  scale_y = dst_height / src_height
  x_resized = (x * scale_x).astype(np.int32).clip(0, dst_width - 1)
  y_resized = (y * scale_y).astype(np.int32).clip(0, dst_height - 1)
  return x_resized, y_resized


class EventTransform(ABC):

  def __call__(self, events: np.ndarray, *args, **kwargs) -> np.ndarray:
    raise NotImplementedError("Subclass must implement this method")

  @property
  @abstractmethod
  def output_channels(self) -> int:
    """Return the number of output channels for the transformed event representation."""
    pass


class Binary(EventTransform):
  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height

  @property
  def output_channels(self):
    return 2

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    """Convert events to binary representation.

    Args:
        events: numpy array of shape (N, 4) or structured array with fields (t, x, y, p)

    Returns:
        numpy array of shape (H, W, 2) containing (positive, negative) event counts
    """
    frame = np.zeros((self.height, self.width, 2), dtype=np.uint32)
    
    if len(events) == 0:
      return frame
    
    # Support both structured and regular arrays
    x, y = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    p = events['p']
    
    # Vectorized: filter valid coordinates and polarities
    valid_mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height) & ((p == 0) | (p == 1))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    p_valid = p[valid_mask]
    
    # Vectorized accumulation using np.add.at
    np.add.at(frame, (y_valid, x_valid, p_valid), 1)
    
    return frame


class BinaryRepresentation(EventTransform):
  def __init__(self, width: int, height: int, microbins: int = 4):
    self.width = width
    self.height = height
    self.microbins = microbins
    # Match Bina-Rep weighting: [2^(N-1), ..., 2^0], normalized by (2^N - 1).
    self.mask = (2.0 ** np.arange(microbins - 1, -1, -1, dtype=np.float32))[None, None]
    self.norm = float((2 ** microbins) - 1)

  @property
  def output_channels(self):
    return 2

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:

    if len(events) == 0:
      return np.zeros((self.height, self.width, 2), dtype=np.float32)

    ts = events['t']
    xs, ys = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    ps = (events['p'] > 0).astype(np.int32)

    # Normalize timestamps to [0, num_bins - 1]
    t_norms = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8) * (self.microbins - 1)
    t_bins = t_norms.astype(int)

    # Store bins per polarity channel and microbin.
    binarep = np.zeros((self.height, self.width, 2, self.microbins), dtype=np.float32)

    # Vectorized: filter valid coordinates
    valid_mask = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height) & (t_bins >= 0) & (t_bins < self.microbins) & ((ps == 0) | (ps == 1))
    xs_valid = xs[valid_mask]
    ys_valid = ys[valid_mask]
    ps_valid = ps[valid_mask]
    t_bins_valid = t_bins[valid_mask]

    # Vectorized accumulation
    np.add.at(binarep, (ys_valid, xs_valid, ps_valid, t_bins_valid), 1.0)

    # Match reference pipeline where each bit-plane is binary before weighting.
    binarep = (binarep > 0).astype(np.float32)

    return (binarep * self.mask[:, :, None, :]).sum(axis=-1) / self.norm


class VoxelGrid(EventTransform):
  def __init__(self, width: int, height: int, microbins: int = 4):
    self.width = width
    self.height = height
    self.microbins = microbins

  @property
  def output_channels(self):
    return self.microbins


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

    # Vectorized: filter valid coordinates
    valid_mask = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height) & (t_bins >= 0) & (t_bins < self.microbins)
    xs_valid = xs[valid_mask]
    ys_valid = ys[valid_mask]
    ps_valid = ps[valid_mask]
    t_bins_valid = t_bins[valid_mask]

    # Vectorized accumulation
    np.add.at(voxel_grid, (ys_valid, xs_valid, t_bins_valid), ps_valid)

    return voxel_grid


class Histogram(EventTransform):
  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height

  @property
  def output_channels(self):
    return 2

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    img = np.full((self.height, self.width, 2), 0, dtype=np.float32)
    
    if len(events) == 0:
      return img
    
    x, y = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    p = events["p"]
    
    # Vectorized: filter valid coordinates
    valid_mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height) & ((p == 0) | (p == 1))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    p_valid = p[valid_mask]
    
    # Vectorized accumulation
    np.add.at(img, (y_valid, x_valid, p_valid), 1.0)
    
    return img


class EventFrame(EventTransform):
  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height

  @property
  def output_channels(self):
    return 1

  def __call__(self, events: np.ndarray, original_width: int, original_height: int) -> np.ndarray:

    if len(events) == 0:
      return np.full((self.height, self.width, 1), 0, dtype=np.float32)
    
    img = np.full((self.height, self.width, 1), 0, dtype=np.float32) # mid value is 128
    x, y = resize_positions(events['x'], events['y'], original_width, original_height, self.width, self.height)
    
    # Vectorized: filter valid coordinates
    valid_mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # Vectorized accumulation (all events contribute +1)
    np.add.at(img, (y_valid, x_valid, 0), 1.0)
    
    return img


def build_event_transform(eventrepr: str, *args, **kwargs):
  return {
    "binary": Binary,
    "binarep": BinaryRepresentation,
    "histogram": Histogram,
    "voxelgrid": VoxelGrid,
    "eventframe": EventFrame,
  }[eventrepr](*args, **kwargs)


