"""
File: loader.py
Author: Viet Nguyen
Date: 2026-02-02

Loader base class

"""

import pathlib
from abc import ABC, abstractmethod

from .types import EyeEpisodeDataType, RGBEpisodeDataType
from ..utils import Space

class Loader(ABC):

  """
  Abstract base class for episode-level data loading.

  Loaders handle format-specific operations:
  - Parse dataset metadata
  - Load individual eye episodes
  - Return standardized data format
  - The episode loader only loads metadata at initialization, not episode data.
    It also only loads one episode at a time via __getitem__(). This is to ensure
    that memory usage is kept low and only the necessary data is loaded into memory.

  Loaders should NOT handle:
  - Batching (that's PyTorch DataLoader's responsibility)
  - Sequence extraction (that's Dataset's responsibility)
  """

  # default timebin size in microseconds
  #   the default timebin for recording each label OF THIS DATASET
  TIMEBIN: int

  @abstractmethod
  def __init__(self, dataset_path: str | pathlib.Path, **kwargs):
    """
    Initialize loader with dataset path and format-specific arguments.

    Args:
      dataset_path: Path to dataset root directory
      **kwargs: Format-specific arguments (e.g., video_backend, modality_configs)
    """
    self.dataset_path = pathlib.Path(dataset_path)

  @abstractmethod
  def __len__(self) -> int:
    """
    Return number of episodes in dataset (list-like interface).

    Returns:
      Number of episodes
    """
    pass

  @abstractmethod
  def __getitem__(self, idx: int) -> EyeEpisodeDataType | RGBEpisodeDataType:
    """
    Load episode data using list-like indexing (loader[idx]).

    Args:
      idx: Episode index to load

    Returns:
      Episode data dictionary (see load_episode for format)
    """
    pass
