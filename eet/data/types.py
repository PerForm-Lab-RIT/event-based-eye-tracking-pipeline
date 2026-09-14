
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import torch
import numpy as np
import functools


EVENT_DTYPE = np.dtype([
  ('t', np.int64),  # Use int64 for nanosecond timestamps
  ('x', np.int32),  # 32-bit ints sufficient for coordinates
  ('y', np.int32),
  ('p', np.int8)    # Single byte for binary polarity
])


@dataclass
class BatchItemDataType:
  """
  Data type for a single batch item, containing input data and labels.
  """
  data: List[np.ndarray] # [T,] stream bins or RGB frames
  labels: List[np.ndarray] # [T,] of label
  width: int # the original width
  height: int # the original height
  timebin: int # The amount of time in microseconds that each label is recorded
  batch_length: int
  modality: str = "event" # 'event' or 'rgb'
  temporal: bool = True
  source_path: Optional[str] = None
  # When the batch item is processed, the data will appear in this variable
  #  In fact, this is the VERY IMPORTANT variable and only this variable will be used in dataloading
  #  This is because events are in raw format and cannot be made easily into tensor,
  #  so we need to process the data into tensors, i.e., using event representation, scaling, augmentation,
  #  and so on
  processed: Optional[Dict[str, Any]] = None

  def __post_init__(self):
    """Validate the BatchItemDataType structure and data types."""
    # Validate data structure
    if not isinstance(self.data, list):
      raise TypeError(f"data must be a list, got {type(self.data)}")

    # Check batch_length matches the length of data list
    if len(self.data) != self.batch_length:
      raise ValueError(
        f"data length ({len(self.data)}) must match batch_length ({self.batch_length})"
      )

    # Validate each stream/frame array
    for i, data_array in enumerate(self.data):
      if not isinstance(data_array, np.ndarray):
        raise TypeError(
          f"data[{i}] must be a numpy array, got {type(data_array)}"
        )

      # Event modality requires structured dtype.
      if self.modality == "event" and data_array.dtype != EVENT_DTYPE:
        if data_array.dtype.names is not None:
          self.data[i] = data_array.astype(EVENT_DTYPE)
        else:
          raise ValueError(
            f"data[{i}] must have dtype {EVENT_DTYPE} for modality='event', got {data_array.dtype}."
          )

    # Validate labels structure
    if not isinstance(self.labels, list):
      raise TypeError(f"labels must be a list, got {type(self.labels)}")

    # Check labels length matches batch_length
    if len(self.labels) != self.batch_length:
      raise ValueError(
        f"labels length ({len(self.labels)}) must match batch_length ({self.batch_length})"
      )

    # Validate each label is a numpy array
    for i, label_array in enumerate(self.labels):
      if not isinstance(label_array, np.ndarray):
        raise TypeError(
          f"labels[{i}] must be a numpy array, got {type(label_array)}"
        )



@dataclass
class EyeEpisodeDataType:
  """
  Data type for a single eye episode, containing raw stream data and labels.

  Attributes:
    events: Structured numpy array with dtype EVENT_DTYPE containing event data
      (t: int64, x: int32, y: int32, p: int8)
    labels: Numpy array containing label data
  """
  data: np.ndarray # dtype: EVENT_DTYPE with a lot of stream entries
  labels: np.ndarray # label (N, ...) # N is the number of labels
  width: int # the original width
  height: int # the original height
  timebin: int # The amount of time in microseconds that each label is recorded, with respect to THIS EPISODE
  source_path: Optional[str] = None

  @property
  def n_labels(self) -> int:
    """Return the number of labels in the episode."""
    return self.labels.shape[0]

  def __post_init__(self):
    """Validate and ensure data has the correct dtype."""
    if not isinstance(self.data, np.ndarray):
      raise TypeError(f"data must be a numpy array, got {type(self.data)}")

    # Convert to EVENT_DTYPE if not already
    if self.data.dtype != EVENT_DTYPE:
      if self.data.dtype.names is not None:
        self.data = self.data.astype(EVENT_DTYPE)
      else:
        raise ValueError(
          f"data must have dtype {EVENT_DTYPE}, got {self.data.dtype}. "
          "Cannot convert unstructured array to structured array automatically."
        )

    if not isinstance(self.labels, np.ndarray):
      raise TypeError(f"labels must be a numpy array, got {type(self.labels)}")

    # Also validate labels shape, it should have more than 1 dims
    if self.labels.ndim < 2:
      raise ValueError(f"labels must have at least 2 dimensions (N, ...), got {self.labels.ndim}")


@dataclass
class RGBEpisodeDataType:
  """
  Data type for a single RGB episode.

  Attributes:
    frames: Numpy array containing frame data with shape (T, ...)
    labels: Numpy array containing labels with shape (T, ...)
  """
  frames: np.ndarray
  labels: np.ndarray
  width: int
  height: int
  timebin: int = 1
  source_path: Optional[str] = None

  @property
  def n_labels(self) -> int:
    return self.labels.shape[0]

  def __repr__(self):
    return (
      f"RGBEpisodeDataType(frames shape={self.frames.shape}, "
      f"labels shape={self.labels.shape}, width={self.width}, "
      f"height={self.height}, timebin={self.timebin}, "
      f"source_path={self.source_path})"
    )

  def __str__(self):
    return self.__repr__()

  def __post_init__(self):
    if not isinstance(self.frames, np.ndarray):
      raise TypeError(f"frames must be a numpy array, got {type(self.frames)}")
    if self.frames.ndim < 3:
      raise ValueError(f"frames must have at least 3 dimensions (T, H, W, ...), got {self.frames.ndim}")
    if not isinstance(self.labels, np.ndarray):
      raise TypeError(f"labels must be a numpy array, got {type(self.labels)}")
    if self.labels.shape[0] != self.frames.shape[0]:
      raise ValueError(
        f"labels first dimension ({self.labels.shape[0]}) must match frames T ({self.frames.shape[0]})"
      )
