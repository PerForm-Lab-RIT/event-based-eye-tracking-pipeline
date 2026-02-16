
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Optional
import torch
import numpy as np

# class BatchItemDataType(dict):
#   """This contain a sequence of data points that has fixed sequence length. {(T, ...)}
#     NOTE: The BatchItemDataType is expected to be a FLAT DICTIONARY, not a nested one.
#     Expected equal data type to this class: Dict[str, np.ndarray]

#     Note: Validation is lenient to allow PyTorch DataLoader to convert arrays to tensors.
#   """

#   def __setitem__(self, key: str, value: Any):
#     """
#     Override __setitem__ to validate that values are array-like.

#     Args:
#       key: Dictionary key
#       value: Value to set (should be np.ndarray or torch.Tensor)

#     Raises:
#       TypeError: If value is not array-like
#     """
#     # Allow both numpy arrays and torch tensors (DataLoader converts numpy to torch)
#     if not isinstance(value, (np.ndarray, torch.Tensor)):
#       raise TypeError(
#         f"BatchItemDataType values must be numpy arrays or torch tensors, "
#         f"got {type(value)} for key '{key}'"
#       )
#     super().__setitem__(key, value)


EVENT_DTYPE = np.dtype([
  ('t', np.int64),  # Use int64 for nanosecond timestamps
  ('x', np.int32),  # 32-bit ints sufficient for coordinates
  ('y', np.int32),
  ('p', np.int8)    # Single byte for binary polarity
])


@dataclass
class BatchItemDataType:
  """
  Data type for a single batch item, containing event data and labels.
  """
  events: List[np.ndarray] # [T,] of EVENT_DTYPE with some events
  labels: List[np.ndarray] # [T,] of label
  width: int # the original width
  height: int # the original height
  timebin: int # The amount of time in microseconds that each label is recorded
  batch_length: int
  # When the batch item is processed, the data will appear in this variable
  #  In fact, this is the VERY IMPORTANT variable and only this variable will be used in dataloading
  #  This is because events are in raw format and cannot be made easily into tensor,
  #  so we need to process the data into tensors, i.e., using event representation, scaling, augmentation,
  #  and so on
  processed: Optional[Dict[str, Any]] = None

  def __post_init__(self):
    """Validate the BatchItemDataType structure and data types."""
    # Validate events structure
    if not isinstance(self.events, list):
      raise TypeError(f"events must be a list, got {type(self.events)}")

    # Check batch_length matches the length of events list
    if len(self.events) != self.batch_length:
      raise ValueError(
        f"events length ({len(self.events)}) must match batch_length ({self.batch_length})"
      )

    # Validate each event array has EVENT_DTYPE
    for i, event_array in enumerate(self.events):
      if not isinstance(event_array, np.ndarray):
        raise TypeError(
          f"events[{i}] must be a numpy array, got {type(event_array)}"
        )

      # Check dtype
      if event_array.dtype != EVENT_DTYPE:
        # If it's a structured array with compatible fields, convert it
        if event_array.dtype.names is not None:
          self.events[i] = event_array.astype(EVENT_DTYPE)
        else:
          raise ValueError(
            f"events[{i}] must have dtype {EVENT_DTYPE}, got {event_array.dtype}. "
            "Cannot convert unstructured array to structured array automatically."
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
  Data type for a single eye episode, containing raw event data and labels.

  Attributes:
    events: Structured numpy array with dtype EVENT_DTYPE containing event data
      (t: int64, x: int32, y: int32, p: int8)
    labels: Numpy array containing label data
  """
  events: np.ndarray # dtype: EVENT_DTYPE with a lot of events
  labels: np.ndarray # label (N, ...) # N is the number of labels
  width: int # the original width
  height: int # the original height
  timebin: int # The amount of time in microseconds that each label is recorded, with respect to THIS EPISODE

  @property
  def n_labels(self) -> int:
    """Return the number of labels in the episode."""
    return self.labels.shape[0]

  def __post_init__(self):
    """Validate and ensure events has the correct dtype."""
    if not isinstance(self.events, np.ndarray):
      raise TypeError(f"events must be a numpy array, got {type(self.events)}")

    # Convert to EVENT_DTYPE if not already
    if self.events.dtype != EVENT_DTYPE:
      # If it's a structured array with compatible fields, convert it
      if self.events.dtype.names is not None:
        self.events = self.events.astype(EVENT_DTYPE)
      else:
        raise ValueError(
          f"events must have dtype {EVENT_DTYPE}, got {self.events.dtype}. "
          "Cannot convert unstructured array to structured array automatically."
        )

    if not isinstance(self.labels, np.ndarray):
      raise TypeError(f"labels must be a numpy array, got {type(self.labels)}")

    # Also validate labels shape, it should have more than 1 dims
    if self.labels.ndim < 2:
      raise ValueError(f"labels must have at least 2 dimensions (N, ...), got {self.labels.ndim}")
